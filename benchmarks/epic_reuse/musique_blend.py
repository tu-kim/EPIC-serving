# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC PIC/reuse end-to-end probe on CacheBlend's *musique* RAG data.

Standalone GPU script (NOT a pytest module -- no ``test_*`` functions, runs from
``__main__``). It answers the question "does EPIC non-prefix reuse PRESERVE the
answer and SAVE prefill time on real multi-hop RAG prompts?" using the exact
dataset CacheBlend ships (``inputs/musique_s.json``). Each musique sample is a
multi-hop question with 10 retrieved Wikipedia contexts and a gold answer list.

WHAT IT DOES (per the task spec)
--------------------------------
For each of N samples we assemble a prompt ``[ctx0][ctx1]...[ctxK][question]``
at the TOKEN level and run it through four engine modes, each in a FRESH
subprocess (one in-process engine per mode, processing all N samples
sequentially so the per-engine EPIC store accumulates harmlessly -- content
hashes differ across samples):

  * ``full``        -- no connector. Whole prompt full-prefill. == ground truth.
  * ``reuse-only``  -- epic@0. Warm each ctx individually (connector stores its
                       chunks), then issue the whole prompt as a reuse request:
                       the ctx chunks are matched NON-PREFIX and loaded, ZERO
                       recompute (M = question + last token only) -- pure stale
                       reused KV + PIC re-rotary.
  * ``epic@k``      -- same as reuse-only but LegoLink recomputes the leading
                       ``k`` tokens of EACH non-prefix chunk
                       (``epic_link_tokens=k``).

``full`` runs the connector OFF (sparse off). ``reuse-only`` / ``epic@k`` run
sparse ON + fusion mask + FLEX_ATTENTION + enforce_eager (the S7 safety gate
requires exactly this) and assert in-band engagement
(``epic_debug_counters``: sparse_match>=1, chunks_loaded>=1).

CHUNK ALIGNMENT (why we pad each ctx -- and how this DIFFERS from CacheBlend)
----------------------------------------------------------------------------
The EPIC chunk store hashes WHOLE ``chunk_size`` chunks counted from prompt
position 0; a chunk is only reusable if it is byte-identical AND lands on a
chunk boundary in both the warm prompt (where it was saved) and the reuse
prompt (where it is matched). Re-tokenizing "ctx_a + ctx_b" does NOT keep
ctx_b's ids stable or chunk-aligned (subword boundaries shift with neighbours).
So we assemble at the TOKEN level and PAD EACH ctx up to a multiple of
``chunk_size`` with rotating REAL-WORD filler (never blank space -- blank
collapses output to a trivial constant). Each ctx then occupies an integer
number of whole chunks and its id-slice is byte-identical between the per-ctx
warm prefill and the full reuse prompt -> the content hash collides -> a real
non-prefix hit.

  *** DIFFERENCE vs CacheBlend ***: CacheBlend concatenates the raw contexts
  with NO padding and reuses arbitrary token spans. Our current EPIC
  implementation reuses BLOCK-ALIGNED whole chunks only, so we must pad each ctx
  to a chunk multiple. The padding is wasted compute on the warm side and a
  (small) length overhead on the reuse side; we LOG the waste. Removing the
  alignment constraint (arbitrary-span reuse like CacheBlend) is future work in
  the connector, not this script.

  ``link=k`` is applied PER non-prefix CHUNK. If a ctx spans C chunks, the
  leading k tokens of EACH of its C chunks are recomputed (so the effective
  recompute for that ctx is ~C*k tokens, NOT k). With a LARGE ``--chunk-size``
  (e.g. 512/768) most ctxs fit in ~1 chunk, so "link=k per chunk" ~= "k per
  ctx"; we log chunks-per-ctx so the reader can interpret k correctly.

TIMING
------
Only the MEASURED (reuse / full) request is timed; warm prefills are excluded.
We time the prefill via a ``max_tokens=1`` greedy request's wall-clock around
``llm.generate`` (TTFT-equivalent for a single short request), then a SEPARATE
``max_tokens``-token greedy request produces the answer text scored for answer
containment. All modes use FLEX_ATTENTION + enforce_eager so the only thing
that differs is the algorithm (fairness). ``--warmup-discard`` drops the first
sample's numbers (cold-start compile/autotune) from the aggregate.

OUTPUT
------
A per-sample table (mode | answer_hit | prefill_ms | generated-text head) and an
aggregate table (mode | answer_hit_rate | mean_prefill_ms | speedup vs full).

CPU / no-GPU behaviour: prints guidance and exits 0 (so a structure-only import
check is not penalised). ``--help`` and ``py_compile`` work with no GPU.

Usage (CUDA box, after ``VLLM_USE_PRECOMPILED=1 uv pip install -e .``):

    .venv/bin/python benchmarks/epic_reuse/musique_blend.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --num-samples 5 --ctx-per-sample 6 --chunk-size 256 --link 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass

# fork-safety: probe CUDA via NVML (no driver context in the parent) and force
# spawn for any vLLM child. Identical rationale to gpu_smoke.py -- the parent
# tokenizes + dispatches per-mode workers and must NOT create a CUDA context
# (a forked EngineCore could not re-init it). Set before torch/vllm import.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Make the repo root importable so ``tests.*`` and ``benchmarks.*`` resolve when
# this script is run directly (python benchmarks/epic_reuse/musique_blend.py)
# from any cwd. The repo root is three levels up from this file
# (benchmarks/epic_reuse/musique_blend.py -> repo root).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the gpu_smoke subprocess-isolation + result plumbing verbatim (no
# duplicate implementations): the RESULT_JSON sentinel/parse, the spec
# serialise/parse, and the fork-safe CUDA probe. We only ADD a richer worker
# (per-sample warm groups + timing + block_size) here because the smoke worker
# spec is fixed to a single shared warm list.
from tests.v1.kv_connector.unit.epic.gpu_smoke import (  # noqa: E402
    _RESULT_PREFIX,
    parse_result_json,
)

# Pure helpers reused from the bench common module (tokenizer loader, chunk hash
# mirror, answer normalisation/containment). No re-implementation.
from benchmarks.epic_reuse.common import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    answer_containment,
    effective_chunk_size,
    epic_chunk_hash,
    load_tokenizer,
)

# CacheBlend musique sample file (list[150]; each has ctxs/question/answers).
_MUSIQUE_URL = (
    "https://raw.githubusercontent.com/YaoJiayi/CacheBlend/"
    "refs/heads/main/inputs/musique_s.json"
)
_DEFAULT_DATA_PATH = "/tmp/musique_s.json"

# Rotating REAL-WORD filler pool (NOT blank space). Mirrors common._WORD_POOL /
# gpu_smoke._FILLER_WORDS: distinct words so a ctx padded up to a chunk boundary
# is ordinary text, not a run of identical blank tokens (blank padding collapses
# dense+sparse to the same trivial output -> a vacuous 1.000 answer match).
_FILLER_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu amber basil cedar dawn ember frost grove harbor ivory "
    "jade kite lotus maple nectar opal pearl quartz river slate timber umbra "
    "violet willow xenon yarrow zephyr anchor beacon canyon dune eagle fjord"
).split()


def _log(msg: str) -> None:
    print(f"[epic-musique] {msg}", flush=True)


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    _log(f"FAIL: {msg}")
    sys.exit(1)


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception as e:  # noqa: BLE001
        _log(f"torch/CUDA probe failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Data loading (CacheBlend musique_s.json)
# ---------------------------------------------------------------------------
@dataclass
class MusiqueSample:
    """One musique example: contexts + question + accepted answers."""

    ctxs: list[str]
    question: str
    answers: list[str]


def parse_musique_records(raw: list) -> list[MusiqueSample]:
    """Parse the decoded musique JSON (list of dicts) into MusiqueSamples (pure).

    Each record is ``{"ctxs": [{"title","text"}...], "question", "answers"}``.
    The ctx string used downstream is ``title + "\\n" + text`` when a title is
    present (matches CacheBlend's passage formatting), else just the text.
    Records missing required keys are skipped (defensive against a truncated
    download). Raises ValueError if NOTHING parses.
    """
    out: list[MusiqueSample] = []
    for rec in raw or []:
        if not isinstance(rec, dict):
            continue
        ctxs_raw = rec.get("ctxs")
        question = rec.get("question")
        answers = rec.get("answers")
        if not ctxs_raw or not question or not answers:
            continue
        ctx_strs: list[str] = []
        for c in ctxs_raw:
            if not isinstance(c, dict):
                continue
            title = (c.get("title") or "").strip()
            text = (c.get("text") or "").strip()
            if not text:
                continue
            ctx_strs.append(f"{title}\n{text}" if title else text)
        if not ctx_strs:
            continue
        out.append(
            MusiqueSample(
                ctxs=ctx_strs,
                question=str(question),
                answers=[str(a) for a in answers if str(a)],
            )
        )
    if not out:
        raise ValueError(
            "no usable musique records parsed (file empty / wrong schema?)"
        )
    return out


def load_musique(path: str, *, download: bool = True) -> list[MusiqueSample]:
    """Load musique samples from ``path``; download from CacheBlend if absent.

    Pure file/JSON parsing on top of ``parse_musique_records``; the network
    fetch is the only side effect and is skipped when the file already exists or
    ``download`` is False (in which case a missing file is a clear error).
    """
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(
                f"musique data not found at {path} and download disabled"
            )
        _log(f"musique data absent at {path}; downloading from {_MUSIQUE_URL}")
        try:
            urllib.request.urlretrieve(_MUSIQUE_URL, path)  # noqa: S310
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"failed to download musique data to {path}: {e}. Fetch it "
                f"manually:\n  curl -L -o {path} {_MUSIQUE_URL}"
            ) from e
    with open(path) as f:
        raw = json.load(f)
    return parse_musique_records(raw)


# ---------------------------------------------------------------------------
# Token-level, chunk-aligned prompt assembly
# ---------------------------------------------------------------------------
def _filler_token_pool(encode) -> list[int]:
    """Encode the real-word filler pool into a flat id list for cycling (pure
    apart from the injected ``encode``). Falls back to ``[0]`` if encoding
    yields nothing (degenerate tokenizer)."""
    ids: list[int] = []
    for w in _FILLER_WORDS:
        ids.extend(encode(" " + w))
    return ids or [0]


@dataclass
class AssembledPrompt:
    """A token-level chunk-aligned musique prompt + the metadata the run/test
    asserts on. ``warm_ctx_ids`` are the per-ctx warm prefill prompts (each is a
    chunk-aligned ctx, optionally with a tiny lead so it is a valid prompt); the
    reuse/full prompt is ``prompt_ids``.
    """

    prompt_ids: list[int]            # [ctx0..ctxK][question]  (reuse / full)
    warm_ctx_ids: list[list[int]]    # one warm prefill per ctx (seeds store)
    ctx_offsets: list[int]           # start offset of each ctx in prompt_ids
    ctx_token_lens: list[int]        # padded token length of each ctx
    ctx_chunk_counts: list[int]      # whole chunks per ctx
    ctx_chunk_hashes: list[list[str]]  # hashes of each ctx's chunks (in prompt)
    question_len: int
    chunk_size: int
    real_tokens: int                 # ctx tokens BEFORE padding (content)
    pad_tokens: int                  # filler tokens added for alignment


def assemble_musique_prompt(
    *,
    ctx_token_lists: list[list[int]],
    question_ids: list[int],
    chunk_size: int,
    filler_ids: list[int],
    warm_lead_ids: list[int] | None = None,
) -> AssembledPrompt:
    """Assemble ``[ctx0..ctxK][question]`` with EACH ctx padded to a chunk
    multiple, returning the reuse/full prompt + per-ctx warm prefills (pure).

    Alignment invariants (the whole reason this exists):
      * each ctx is padded UP to a multiple of ``chunk_size`` with cycled
        real-word filler, so it occupies whole hashable chunks and the NEXT ctx
        starts on a chunk boundary;
      * the SAME padded ctx id-slice is used in the warm prefill and in the
        reuse prompt -> byte-identical -> EPIC content hash collides -> a real
        non-prefix hit for every ctx chunk;
      * the question is appended AFTER all ctxs as a trailing (partial) chunk --
        never hashed/saved, matching the connector's whole-chunks-only rule.

    The warm prefill for a ctx is ``warm_lead_ids + padded_ctx``; ``warm_lead``
    is padded to a chunk multiple too so the ctx still starts on a chunk
    boundary in the warm prompt (default lead is empty -> ctx starts at 0).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    pool = list(filler_ids) if filler_ids else [0]

    def _fill(n: int) -> list[int]:
        return [pool[i % len(pool)] for i in range(n)]

    def _pad_to_chunk(ids: list[int]) -> tuple[list[int], int, int]:
        """Pad ids up to a chunk multiple. Returns (padded, real, pad)."""
        real = len(ids)
        rem = real % chunk_size
        pad = (chunk_size - rem) % chunk_size
        if real == 0:
            pad = chunk_size  # never emit a zero-length ctx
        return list(ids) + _fill(pad), real, pad

    # Warm lead: pad to a chunk multiple so the ctx begins chunk-aligned in the
    # warm prompt (empty lead -> begins at 0, also aligned).
    lead = list(warm_lead_ids or [])
    if lead:
        lead, _, _ = _pad_to_chunk(lead)

    prompt_ids: list[int] = []
    warm_ctx_ids: list[list[int]] = []
    ctx_offsets: list[int] = []
    ctx_token_lens: list[int] = []
    ctx_chunk_counts: list[int] = []
    ctx_chunk_hashes: list[list[str]] = []
    total_real = 0
    total_pad = 0

    for ctx_ids in ctx_token_lists:
        padded, real, pad = _pad_to_chunk(ctx_ids)
        total_real += real
        total_pad += pad
        n_chunks = len(padded) // chunk_size
        # Hash each chunk of the padded ctx (these are the hashes EPIC matches).
        hashes = [
            epic_chunk_hash(padded[c * chunk_size:(c + 1) * chunk_size])
            for c in range(n_chunks)
        ]
        ctx_offsets.append(len(prompt_ids))
        ctx_token_lens.append(len(padded))
        ctx_chunk_counts.append(n_chunks)
        ctx_chunk_hashes.append(hashes)
        prompt_ids.extend(padded)
        # Warm prefill: lead (chunk-aligned) + the SAME padded ctx ids.
        warm_ctx_ids.append(list(lead) + list(padded))

    prompt_ids.extend(question_ids)

    return AssembledPrompt(
        prompt_ids=prompt_ids,
        warm_ctx_ids=warm_ctx_ids,
        ctx_offsets=ctx_offsets,
        ctx_token_lens=ctx_token_lens,
        ctx_chunk_counts=ctx_chunk_counts,
        ctx_chunk_hashes=ctx_chunk_hashes,
        question_len=len(question_ids),
        chunk_size=chunk_size,
        real_tokens=total_real,
        pad_tokens=total_pad,
    )


# ---------------------------------------------------------------------------
# Mode -> kv-transfer config + engine kwargs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mode:
    """A measured engine mode. ``link`` is None for ``full`` (no connector)."""

    label: str
    sparse: bool
    link: int | None  # epic_link_tokens (None for full)

    @property
    def is_reuse(self) -> bool:
        return self.sparse


def mode_full() -> Mode:
    return Mode(label="full", sparse=False, link=None)


def mode_reuse_only() -> Mode:
    return Mode(label="reuse-only", sparse=True, link=0)


def mode_epic(k: int) -> Mode:
    if k < 0:
        raise ValueError(f"link k must be >= 0, got {k}")
    return Mode(label=f"epic@{k}", sparse=True, link=k)


def kv_config_for_mode(mode: Mode, *, chunk_size: int) -> dict | None:
    """KV-transfer config dict for a mode.

    full       -> None (no connector; sparse OFF, prefix caching OFF -> a true
                  full prefill ground truth).
    reuse/epic -> EpicConnector, sparse-forward + fusion mask ON, the requested
                  link, and the SAME chunk_size used for assembly (so the
                  connector's chunk boundaries match ours).
    """
    if not mode.sparse:
        return None
    extra: dict = {
        "epic_chunk_size": int(chunk_size),
        "epic_link_tokens": int(mode.link or 0),
        "epic_sparse_forward": True,
        "epic_fusion_mask": True,
        # In-band engagement counters (assert sparse actually fired) -- requires
        # the in-process engine so scheduler + worker share the class state.
        "epic_debug_counters": True,
    }
    return {
        "kv_connector": "EpicConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra,
    }


# ---------------------------------------------------------------------------
# Worker spec (one engine = one mode, processes ALL samples sequentially)
# ---------------------------------------------------------------------------
def build_musique_worker_spec(
    *,
    mode: Mode,
    model: str,
    chunk_size: int,
    samples: list[dict],
    max_tokens: int,
    block_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> dict:
    """JSON-serialisable spec for ONE mode's engine run over N samples (pure).

    Each entry of ``samples`` is ``{"warm": [[ids]...], "prompt": [ids],
    "answers": [str]}``. For ``reuse-only`` / ``epic@k`` the worker first runs
    every warm prefill (seeds the store) for a sample, then times that sample's
    reuse prompt; for ``full`` the warm lists are ignored (no connector).

    sparse modes set ``in_process`` so the connector's class-level engagement
    counters are visible to the worker (scheduler + worker share state); ``full``
    does not need it but it is harmless.
    """
    return {
        "kind": "musique",
        "mode_label": mode.label,
        "sparse": bool(mode.sparse),
        "model": model,
        "chunk_size": int(chunk_size),
        "kv_config": kv_config_for_mode(mode, chunk_size=chunk_size),
        "samples": [
            {
                "warm": [[int(t) for t in w] for w in s["warm"]],
                "prompt": [int(t) for t in s["prompt"]],
                "answers": list(s["answers"]),
            }
            for s in samples
        ],
        "max_tokens": int(max_tokens),
        "block_size": int(block_size),
        "max_model_len": int(max_model_len),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        # sparse needs the in-process engine for the in-band counters.
        "in_process": bool(mode.sparse),
        "read_counters": bool(mode.sparse),
    }


def serialize_spec(spec: dict) -> str:
    return json.dumps(spec, separators=(",", ":"))


def parse_spec(s: str) -> dict:
    return json.loads(s)


def _emit_result(result: dict) -> None:
    print(f"{_RESULT_PREFIX} {json.dumps(result)}", flush=True)


# ---------------------------------------------------------------------------
# Aggregation (pure, CPU-testable)
# ---------------------------------------------------------------------------
def speedup(full_ms: float, mode_ms: float) -> float:
    """``full_ms / mode_ms`` -- prefill speedup of a mode vs the full baseline.

    >1 means the mode is FASTER than full. Guards a zero/non-positive mode time
    (returns inf, "infinitely faster") and a non-positive full time (returns
    0.0, undefined baseline). Pure."""
    if mode_ms <= 0.0:
        return float("inf")
    if full_ms <= 0.0:
        return 0.0
    return full_ms / mode_ms


def mean(xs: list[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


@dataclass
class ModeAggregate:
    label: str
    n: int
    answer_hits: int
    mean_prefill_ms: float
    speedup_vs_full: float = 0.0

    @property
    def answer_hit_rate(self) -> float:
        return (self.answer_hits / self.n) if self.n else 0.0


def aggregate_mode(
    label: str, per_sample: list[dict], *, warmup_discard: bool
) -> ModeAggregate:
    """Aggregate one mode's per-sample results into hit-rate + mean prefill.

    ``per_sample`` rows are ``{"answer_hit": bool, "prefill_ms": float, ...}``.
    ``warmup_discard`` drops the FIRST row (cold compile/autotune) from BOTH the
    hit count and the timing mean. Pure."""
    rows = per_sample[1:] if (warmup_discard and len(per_sample) > 1) else per_sample
    n = len(rows)
    hits = sum(1 for r in rows if r.get("answer_hit"))
    ms = mean([float(r["prefill_ms"]) for r in rows if r.get("prefill_ms") is not None])
    return ModeAggregate(label=label, n=n, answer_hits=hits, mean_prefill_ms=ms)


def fill_speedups(aggs: list[ModeAggregate]) -> list[ModeAggregate]:
    """Set ``speedup_vs_full`` on each aggregate using the ``full`` mode's mean
    prefill as the baseline (full speedup == 1.0). If there is no ``full`` row,
    speedups stay 0.0. Pure (mutates + returns the list)."""
    full = next((a for a in aggs if a.label == "full"), None)
    base = full.mean_prefill_ms if full else 0.0
    for a in aggs:
        a.speedup_vs_full = speedup(base, a.mean_prefill_ms) if base > 0 else 0.0
    return aggs


# ---------------------------------------------------------------------------
# Worker (runs inside the per-mode subprocess; needs a GPU)
# ---------------------------------------------------------------------------
def run_musique_worker(spec: dict) -> dict:
    """Build ONE engine for ``spec``'s mode, process all samples, return result.

    Per sample (sparse modes): run every warm prefill (discarded; seeds store),
    then time a ``max_tokens=1`` reuse request (prefill wall-clock) and run a
    ``max_tokens`` request for the answer text. ``full`` skips the warm step. On
    any error returns ``{"ok": False, "error": ...}`` so the parent gets a clean
    structured failure. Device memory is freed by PROCESS EXIT.
    """
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    label = spec.get("mode_label", "?")
    try:
        if spec.get("in_process"):
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        kwargs: dict = dict(
            model=spec["model"],
            enforce_eager=True,
            block_size=int(spec["block_size"]),
            max_model_len=int(spec["max_model_len"]),
            gpu_memory_utilization=float(spec["gpu_memory_utilization"]),
            # FLEX_ATTENTION for ALL modes (fairness + EPIC hard requirement).
            attention_backend="FLEX_ATTENTION",
        )
        sparse = bool(spec.get("sparse"))
        if sparse:
            kwargs["kv_transfer_config"] = spec["kv_config"]
            kwargs["enable_prefix_caching"] = True
        else:
            # full = true ground truth: no reuse of anything.
            kwargs["enable_prefix_caching"] = False
        llm = LLM(**kwargs)

        time_params = SamplingParams(temperature=0.0, max_tokens=1)
        gen_params = SamplingParams(
            temperature=0.0, max_tokens=int(spec["max_tokens"])
        )

        per_sample: list[dict] = []
        for s in spec["samples"]:
            # Warm: seed each ctx's chunks into the per-engine store. Discarded
            # outputs; excluded from timing. Same engine/process as the reuse
            # request so the store is visible.
            if sparse:
                for w in s["warm"]:
                    llm.generate(
                        [TokensPrompt(prompt_token_ids=list(w))], time_params
                    )

            prompt = list(s["prompt"])
            # Timing: a single-token greedy request's wall-clock == prefill TTFT
            # for one short request. Only the MEASURED request is timed.
            t0 = time.perf_counter()
            llm.generate([TokensPrompt(prompt_token_ids=prompt)], time_params)
            prefill_ms = (time.perf_counter() - t0) * 1000.0

            # Answer text: a separate greedy generation (max_tokens tokens).
            outs = llm.generate(
                [TokensPrompt(prompt_token_ids=prompt)], gen_params
            )
            text = outs[0].outputs[0].text
            hit = answer_containment(text, s["answers"])
            per_sample.append(
                {
                    "answer_hit": bool(hit),
                    "prefill_ms": float(prefill_ms),
                    "text": text,
                    "answers": s["answers"],
                }
            )

        result: dict = {
            "ok": True,
            "mode_label": label,
            "per_sample": per_sample,
        }
        if spec.get("read_counters"):
            from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (  # noqa: E501
                EpicConnector,
            )

            result["counters"] = dict(EpicConnector.debug_counters)
        return result
    except Exception as e:  # noqa: BLE001
        import traceback

        return {
            "ok": False,
            "mode_label": label,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def run_mode_subprocess(spec: dict) -> dict:
    """Parent side: run one mode's engine in a FRESH subprocess, parse result.

    Reuses gpu_smoke's RESULT_JSON contract (``parse_result_json``). Subprocess
    exit returns ALL device memory (no reliance on Python cleanup), so the four
    modes never accumulate VRAM. stderr is inherited (engine + EPIC logs stream
    live); only stdout is parsed.
    """
    import subprocess

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--_worker-json",
        serialize_spec(spec),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    try:
        return parse_result_json(proc.stdout)
    except ValueError as e:
        raise RuntimeError(
            f"worker (mode={spec.get('mode_label')}) produced no RESULT_JSON "
            f"(exit code {proc.returncode}): {e}"
        ) from None


# ---------------------------------------------------------------------------
# Parent orchestration
# ---------------------------------------------------------------------------
def _encode(tok):
    def enc(text: str) -> list[int]:
        # add_special_tokens=False: we control boundaries; a stray BOS inside a
        # ctx would break byte-identity vs. the same ctx in the warm prompt.
        return list(tok.encode(text, add_special_tokens=False))

    return enc


@dataclass
class PreparedSample:
    """A musique sample assembled into chunk-aligned token prompts + the gold
    answers and the assembly metadata (for logging)."""

    assembled: AssembledPrompt
    answers: list[str]
    question: str


def prepare_samples(
    samples: list[MusiqueSample],
    *,
    tok,
    chunk_size: int,
    ctx_per_sample: int,
) -> list[PreparedSample]:
    """Tokenize + chunk-align N musique samples (parent side, CPU-only).

    Each ctx is encoded independently (so its ids are stable) and the prompt is
    assembled at the token level with per-ctx chunk padding. Returns the
    prepared prompts + answers; logging of padding waste is the caller's job.
    """
    enc = _encode(tok)
    prepared: list[PreparedSample] = []
    for s in samples:
        ctxs = s.ctxs[:ctx_per_sample]
        ctx_token_lists = [enc(c) for c in ctxs]
        # Question with a small instruction tail so an instruct model answers.
        q_text = (
            f"\n\nQuestion: {s.question}\nAnswer the question using ONLY the "
            "passages above. Answer:"
        )
        q_ids = enc(q_text)
        assembled = assemble_musique_prompt(
            ctx_token_lists=ctx_token_lists,
            question_ids=q_ids,
            chunk_size=chunk_size,
            filler_ids=_filler_token_pool(enc),
        )
        prepared.append(
            PreparedSample(
                assembled=assembled, answers=s.answers, question=s.question
            )
        )
    return prepared


def _samples_for_spec(prepared: list[PreparedSample]) -> list[dict]:
    return [
        {
            "warm": p.assembled.warm_ctx_ids,
            "prompt": p.assembled.prompt_ids,
            "answers": p.answers,
        }
        for p in prepared
    ]


def _print_padding_report(prepared: list[PreparedSample], chunk_size: int) -> None:
    total_real = sum(p.assembled.real_tokens for p in prepared)
    total_pad = sum(p.assembled.pad_tokens for p in prepared)
    total = total_real + total_pad
    waste = (total_pad / total * 100.0) if total else 0.0
    chunk_counts = [c for p in prepared for c in p.assembled.ctx_chunk_counts]
    avg_chunks = mean([float(c) for c in chunk_counts])
    _log(
        f"prompt assembly: {len(prepared)} samples, chunk_size={chunk_size} "
        f"(block-aligned), ctx content tokens={total_real}, padding tokens="
        f"{total_pad} ({waste:.1f}% wasted on chunk alignment), "
        f"avg chunks/ctx={avg_chunks:.2f}"
    )
    if avg_chunks > 1.05:
        _log(
            f"NOTE: avg {avg_chunks:.2f} chunks/ctx -> link=k recomputes k "
            "tokens PER CHUNK, i.e. ~"
            f"{avg_chunks:.1f}*k tokens per ctx. Raise --chunk-size so ctx~=1 "
            "chunk for a clean 'k tokens per ctx' interpretation."
        )
    # CacheBlend-difference reminder (also in the module docstring).
    _log(
        "NOTE: CacheBlend concatenates contexts WITHOUT padding (arbitrary-span "
        "reuse); our EPIC reuses BLOCK-ALIGNED whole chunks, hence the padding "
        "above. This is a current-implementation constraint, not the algorithm."
    )


def _print_sample_table(mode_results: dict[str, dict]) -> None:
    """Per-sample table: one block per mode."""
    _log("PER-SAMPLE RESULTS")
    for label, res in mode_results.items():
        _log(f"  mode={label}")
        _log("    sample | answer_hit | prefill_ms | generated_text_head")
        rows = res.get("per_sample", [])
        for i, r in enumerate(rows):
            head = (r.get("text") or "").replace("\n", " ")[:60]
            ms = r.get("prefill_ms")
            ms_str = f"{ms:9.1f}" if ms is not None else "    n/a"
            _log(
                f"    {i:6d} | {('HIT' if r.get('answer_hit') else 'miss'):>10s}"
                f" | {ms_str} | {head!r}"
            )


def _print_aggregate_table(aggs: list[ModeAggregate]) -> None:
    _log("AGGREGATE (mode | answer_hit_rate | mean_prefill_ms | speedup_vs_full)")
    _log("    mode        | hit_rate (n)   | mean_prefill_ms | speedup_vs_full")
    for a in aggs:
        sp = (
            "  n/a" if a.label == "full"
            else (f"{a.speedup_vs_full:6.2f}x"
                  if a.speedup_vs_full not in (float("inf"),) else "   inf")
        )
        _log(
            f"    {a.label:11s} | {a.answer_hit_rate:5.2f} ({a.answer_hits}/"
            f"{a.n}) | {a.mean_prefill_ms:15.1f} | {sp}"
        )


def _check_sparse_engagement(label: str, res: dict) -> None:
    """Assert the connector actually engaged for a sparse mode (in-band counter
    check, same gate as gpu_smoke step4). Reports loudly; does not crash a run
    that simply had no reusable chunk (e.g. chunk_size > every ctx)."""
    counters = res.get("counters", {})
    if not (counters.get("sparse_match", 0) >= 1
            and counters.get("chunks_loaded", 0) >= 1):
        _log(
            f"WARNING: sparse mode {label!r} engagement counters {counters} "
            "do NOT show sparse_match>=1 AND chunks_loaded>=1. The reuse path "
            "may not have fired (check chunk alignment / byte-identity / "
            "in-process engine). Numbers for this mode are SUSPECT."
        )
    else:
        _log(f"  mode={label}: sparse engaged in-band, counters={counters}")


def run_all(args) -> int:
    samples_all = load_musique(args.data, download=not args.no_download)
    samples = samples_all[: args.num_samples]
    if not samples:
        _fail("no samples after slicing --num-samples")

    block_size = DEFAULT_BLOCK_SIZE
    # Force chunk_size to a block_size multiple (the connector does this anyway;
    # we mirror it so our assembly hashes match the connector's chunk grid).
    chunk_size = effective_chunk_size(args.chunk_size, block_size)
    if chunk_size != args.chunk_size:
        _log(
            f"--chunk-size {args.chunk_size} rounded UP to {chunk_size} "
            f"(must be a multiple of block_size {block_size})"
        )

    tok, is_real = load_tokenizer(args.model, allow_fallback=False)
    if not is_real:
        _fail("could not load a real tokenizer; cannot build model prompts")

    prepared = prepare_samples(
        samples, tok=tok, chunk_size=chunk_size,
        ctx_per_sample=args.ctx_per_sample,
    )
    _print_padding_report(prepared, chunk_size)

    # Length sanity vs max_model_len.
    max_prompt = max(len(p.assembled.prompt_ids) for p in prepared)
    if max_prompt + args.max_tokens + 8 > args.max_model_len:
        _fail(
            f"longest assembled prompt is {max_prompt} tokens + {args.max_tokens}"
            f" gen > --max-model-len {args.max_model_len}. Raise --max-model-len "
            "or lower --ctx-per-sample / --chunk-size."
        )

    # Build the mode list: full, reuse-only, epic@<link or sweep>.
    modes: list[Mode] = [mode_full(), mode_reuse_only()]
    link_ks = (
        sorted(set(args.link_sweep)) if args.link_sweep else [args.link]
    )
    for k in link_ks:
        if k > 0:  # epic@0 == reuse-only, already added.
            modes.append(mode_epic(k))

    _log(
        f"modes={[m.label for m in modes]} num_samples={len(prepared)} "
        f"ctx_per_sample={args.ctx_per_sample} chunk_size={chunk_size} "
        f"max_tokens={args.max_tokens} warmup_discard={args.warmup_discard}"
    )

    sample_specs = _samples_for_spec(prepared)
    mode_results: dict[str, dict] = {}
    aggs: list[ModeAggregate] = []
    for mode in modes:
        _log(f"=== running mode={mode.label} (fresh subprocess engine) ===")
        spec = build_musique_worker_spec(
            mode=mode,
            model=args.model,
            chunk_size=chunk_size,
            samples=sample_specs,
            max_tokens=args.max_tokens,
            block_size=block_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
        )
        res = run_mode_subprocess(spec)
        if not res.get("ok"):
            _fail(
                f"mode {mode.label} engine failed: {res.get('error')}\n"
                + res.get("traceback", "")
            )
        mode_results[mode.label] = res
        if mode.sparse:
            _check_sparse_engagement(mode.label, res)
        aggs.append(
            aggregate_mode(
                mode.label, res["per_sample"],
                warmup_discard=args.warmup_discard,
            )
        )

    fill_speedups(aggs)
    _print_sample_table(mode_results)
    _print_aggregate_table(aggs)
    _log("DONE")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _worker_main(spec_json: str) -> int:
    try:
        spec = parse_spec(spec_json)
    except Exception as e:  # noqa: BLE001
        _emit_result({"ok": False, "mode_label": "?",
                      "error": f"bad spec: {type(e).__name__}: {e}"})
        return 0
    _emit_result(run_musique_worker(spec))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="EPIC PIC/reuse probe on CacheBlend musique data"
    )
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct",
                    help="HF model id (small Llama-family recommended).")
    ap.add_argument("--data", default=_DEFAULT_DATA_PATH,
                    help=f"musique_s.json path (default {_DEFAULT_DATA_PATH}; "
                         "downloaded from CacheBlend if absent).")
    ap.add_argument("--no-download", action="store_true",
                    help="do NOT download musique data if the file is missing.")
    ap.add_argument("--num-samples", type=int, default=5,
                    help="number of musique samples to run (default 5).")
    ap.add_argument("--ctx-per-sample", type=int, default=6,
                    help="contexts per sample, capped from the 10 available "
                         "(default 6; controls prompt length).")
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="EPIC chunk size; rounded UP to a block_size (16) "
                         "multiple (default 256). Larger -> ctx~=1 chunk.")
    ap.add_argument("--link", type=int, default=8,
                    help="epic@k LegoLink leading-recompute tokens per "
                         "non-prefix chunk (default 8).")
    ap.add_argument("--link-sweep", type=str, default=None,
                    help="comma-separated link values to run epic@k for each "
                         "(overrides --link), e.g. '0,8,64'.")
    ap.add_argument("--max-tokens", type=int, default=32,
                    help="greedy generation length for the answer (default 32).")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="max_model_len per engine (default 8192).")
    ap.add_argument("--gpu-mem-util", type=float, default=0.45,
                    help="gpu_memory_utilization per engine (default 0.45).")
    ap.add_argument("--warmup-discard", action="store_true",
                    help="drop the FIRST sample from the aggregate (cold "
                         "compile/autotune warmup).")
    ap.add_argument("--_worker-json", dest="worker_json", default=None,
                    help=argparse.SUPPRESS)
    return ap


def _parse_link_sweep(arg: str | None) -> list[int] | None:
    if arg is None:
        return None
    out: list[int] = []
    for tok in str(arg).split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v < 0:
            raise ValueError(f"link value must be >= 0, got {v}")
        out.append(v)
    return out or None


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.worker_json is not None:
        return _worker_main(args.worker_json)

    try:
        args.link_sweep = _parse_link_sweep(args.link_sweep)
    except ValueError as e:
        _fail(f"bad --link-sweep: {e}")

    if not _has_cuda():
        _log(
            "No CUDA GPU detected. This script requires a GPU + a built EPIC "
            "vLLM. See vllm/.../epic/PHASE2.md for the build command. Exiting "
            "without running (CPU import/structure check is a no-op here)."
        )
        return 0

    return run_all(args)


if __name__ == "__main__":
    sys.exit(main())
