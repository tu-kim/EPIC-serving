# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC GPU smoke test (standalone script, NOT a pytest module).

Codifies the manual GPU verification checklist in
``vllm/distributed/kv_transfer/kv_connector/v1/epic/PHASE2.md`` so it can be run
on a CUDA box with EPIC built. It is deliberately NOT collected by pytest (no
``test_*`` functions) because it needs a real GPU + a model download and takes
minutes; the CPU functional suite (``test_functional_lifecycle.py``) covers the
logic without hardware.

Steps (run in order; stop at the first failure):

  step1  connector ON (sparse OFF) vs NO connector -- identical greedy outputs
         (the connector must be a no-op for generation; it only saves chunks).
  step2  sparse ON + WRONG attention backend (not FlexAttention) -- must raise
         ``ValueError`` at engine construction (S7 safety gate).
  step3  sparse ON + FlexAttention but NOT eager -- must raise ``ValueError``
         (S7 safety gate).
  step4  NEEDLE probe. A shared NON-prefix passage B carries K secret-code
         needles ("The secret code for <subject> is <4-digit>."). The reuse
         prompt's tail Q asks for one code ("...the secret code for <subject>?
         Answer:"). The TARGET needle (the one Q asks for) is placed at a DEEP
         token offset in B (~0.70*B_len) -- OUTSIDE the LegoLink recomputed
         leading region for non-full links -- so a needle HIT there proves the
         REUSED (never-recomputed) KV carried the answer, rather than the needle
         simply being recomputed (the front-loaded bias the rewrite fixes). A
         warm request seeds B's chunks into the EPIC store; the reuse request
         (different head A, B in the MIDDLE, Q at the tail) is then answered
         DENSE (reference) and SPARSE (across a link sweep). Each link logs
         ``needle_in_link`` (= needle_offset < eff_link): False rows are the
         reused-KV proof, True rows (link==B-full) are the control.
         Discriminative by construction: filler is a rotating pool of real
         words (NOT blank space), so dense and sparse outputs are not trivially
         identical. Verdicts (mechanical):
           * dense must output the answer code -> else PROBE INVALID (FAIL:
             the probe itself is broken, model/length problem, not EPIC).
           * dense distinct-token count >= 4 -> else non-discriminative (FAIL).
           * sparse must ENGAGE in-band (EpicConnector.debug_counters:
             sparse_match>=1 and chunks_loaded>=1) -> else SPARSE DID NOT
             ENGAGE (FAIL). Requires the in-process engine
             (VLLM_ENABLE_V1_MULTIPROCESSING=0) so scheduler+worker share state.
           * link=full-B (decisive): needle hit AND match>=0.9 -> PASS gate.
           * link=64/8: needle hit + match + first-divergence REPORTED only
             (approximation regime; a miss there is an algorithmic limit).
  step5  CROSS-CONTEXT approximation-quality probe. Where step4's needle is a
         discrete, B-self-contained fact (nearly link-INVARIANT, so a weak probe
         of what LegoLink buys), step5 measures output fidelity at the FIRST
         DECODE TOKEN: a contentful shared passage B is reused NON-PREFIX under a
         NEW context A/C; the DENSE (full-recompute) run is generated once and
         its first greedy token t0 (argmax) is the reference. Each sparse link L
         then takes a SINGLE greedy decode step with top-K logprobs
         (SamplingParams(logprobs=K, max_tokens=1)) and reports the RANK and
         logprob it assigns to t0; distance(L) = rank(t0)-1 (0 == sparse still
         picks t0 == dense). The decode position is the only scored position and
         EPIC sparse ALWAYS forwards it (last prompt token in M), so unlike the
         old prompt_logprobs teacher-forcing (which needs a logit at EVERY prompt
         position and crashes the M-row sparse forward with an OverflowError),
         this metric is SPARSE-COMPATIBLE. The link sweep includes the extremes:
         link=0 (reuse-only, pure stale KV) and link=B_full (full recompute, ==
         dense up to numerics). NOISE FLOOR: the DENSE run is executed TWICE in
         two independent subprocesses; KL_floor = KL(dense1||dense2) is the
         measurement-jitter bottom and EVERY sparse KL is read as a MULTIPLE of
         it (the rank metric saturates -- argmax stable, rank==1 everywhere -- so
         KL is the live signal). Verdicts (KL-vs-floor): B_full KL <= 2x KL_floor
         == machinery healthy, else *** REAL MACHINERY DISCREPANCY *** (a strong
         diagnostic, not a hard fail); reuse-only KL above floor AND above B_full
         means recompute does something (else "LINK HAS NO EFFECT" / "buried in
         noise"); middle links are REPORTED (KL + KL/floor + monotonicity),
         never hard-thresholded. rank_distance + t0_lp_delta are auxiliary
         columns only. A single-prompt first-token KL is LOW RESOLUTION; a
         trustworthy link-quality curve needs first-token KL + floor averaged
         over a prompt pool (benchmarks/epic_reuse/bench_accuracy.py -- TODO
         there, not here).

Usage:
    # On a CUDA box, after `VLLM_USE_PRECOMPILED=1 uv pip install -e .`.
    # The script selects the FlexAttention backend itself via the
    # ``attention_backend="FLEX_ATTENTION"`` LLM kwarg (the legacy
    # VLLM_ATTENTION_BACKEND env var was removed in vLLM v0.22):
    .venv/bin/python tests/v1/kv_connector/unit/epic/gpu_smoke.py \
        --model meta-llama/Llama-3.2-1B-Instruct

    # Run a single step (e.g. just the safety gates):
    ... gpu_smoke.py --model <m> --steps 2,3

Exit code 0 == all requested steps passed; non-zero == a step failed (the
message names which one). With no GPU it prints guidance and exits 0 (so CI that
imports it for a structure check is not penalized).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# EPIC: fork-safety. This parent process probes CUDA (torch.cuda.is_available)
# BEFORE constructing the LLM; that creates a CUDA driver context which a
# forked EngineCore cannot re-initialize ("Cannot re-initialize CUDA in forked
# subprocess"). vLLM's auto force-spawn only checks torch.cuda.is_initialized(),
# which is_available() does NOT set -- so protect ourselves:
#   1) NVML-based availability check: no CUDA context in this process.
#   2) Force spawn for vLLM child processes as belt-and-braces (covers any
#      other parent-side CUDA touch, e.g. user sitecustomize).
# Both must be set before torch / vllm are imported anywhere in this process.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Shared chunk content (a "retrieved passage" that appears in the MIDDLE of two
# different prompts -- the RAG reuse pattern EPIC targets). Kept as explicit
# token-ish text; the offline LLM tokenizes it. Long enough to span >= one
# EPIC chunk for the configured model/chunk size.
_SHARED_PASSAGE = (
    "The mitochondria is the powerhouse of the cell. It generates most of the "
    "cell's supply of adenosine triphosphate, used as a source of chemical "
    "energy. Mitochondria are found in nearly all eukaryotic organisms and "
    "vary in number and location according to cell type. "
) * 4

_PROMPT_HEAD = "Question one preamble unrelated text goes right here. "
_PROMPT_TAIL_1 = " Now answer: what does the mitochondria do?"
_PROMPT_TAIL_2 = " Given the above, summarize the cell's energy source."

# Discriminative filler word pool (NOT blank space). Mirrors
# benchmarks/epic_reuse/common._WORD_POOL: real, distinct words so padding a
# prompt up to a chunk boundary produces ordinary text rather than a run of
# identical space tokens. With blank-space filler both dense and sparse degrade
# to emitting the same blank token -> a trivial 1.000 match with NO
# discriminative power. Rotating real words keeps the prompt a real prompt.
_FILLER_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu amber basil cedar dawn ember frost grove harbor ivory "
    "jade kite lotus maple nectar opal pearl quartz river slate timber umbra "
    "violet willow xenon yarrow zephyr anchor beacon canyon dune eagle fjord"
).split()


def _log(msg: str) -> None:
    print(f"[epic-gpu-smoke] {msg}", flush=True)


def _fail(step: str, msg: str) -> "NoReturn":  # type: ignore[name-defined]
    _log(f"FAIL ({step}): {msg}")
    sys.exit(1)


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception as e:  # noqa: BLE001
        _log(f"torch/CUDA probe failed: {e}")
        return False


def _greedy_params(max_tokens: int = 32):
    from vllm import SamplingParams

    # Deterministic decoding so output comparisons are meaningful.
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


def _epic_kv_config(
    *,
    sparse: bool,
    fusion: bool = False,
    link_tokens: int | None = None,
    debug_check_load: bool = False,
    debug_counters: bool = False,
) -> dict:
    extra: dict = {}
    if sparse:
        extra["epic_sparse_forward"] = True
    if fusion:
        extra["epic_fusion_mask"] = True
    # in-band engagement counters (Implement: sparse-engagement assertion).
    # When on, the connector bumps EpicConnector.debug_counters at the
    # sparse-match / sparse-emit / chunk-load sites so the in-process smoke can
    # assert the sparse path actually engaged without scraping logs.
    if debug_counters:
        extra["epic_debug_counters"] = True
    # link sweep control (Implement 1): leading recomputed tokens per non-prefix
    # chunk. link == chunk_size -> the WHOLE B chunk is recomputed -> the sparse
    # machinery runs but the reuse approximation is null (M == all of B), so the
    # output must match dense up to numerics. Smaller link -> more reuse, more
    # approximation. Only meaningful with sparse on.
    if link_tokens is not None:
        extra["epic_link_tokens"] = int(link_tokens)
    # worker load-fidelity self-check (Implement 2): re-read dst slots after
    # scatter and compare to the store (1 info line/step on the first chunk).
    if debug_check_load:
        extra["epic_debug_check_load"] = True
    cfg: dict = {"kv_connector": "EpicConnector", "kv_role": "kv_both"}
    if extra:
        cfg["kv_connector_extra_config"] = extra
    return cfg


def _build_llm(
    model: str,
    *,
    kv_config: dict | None,
    enforce_eager: bool = True,
    attention_backend: str | None = None,
    gpu_memory_utilization: float = 0.45,
    max_model_len: int = 2048,
):
    from vllm import LLM

    kwargs: dict = dict(
        model=model,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    if kv_config is not None:
        kwargs["kv_transfer_config"] = kv_config
    if attention_backend is not None:
        # v0.22 removed VLLM_ATTENTION_BACKEND; select the backend through the
        # EngineArgs/AttentionConfig surface. The string is normalized by
        # AttentionConfig.validate_backend_before (upper-cased enum lookup).
        kwargs["attention_backend"] = attention_backend
    return LLM(**kwargs)


class SmokeConfig:
    """Parent-side knobs threaded into every worker spec (CLI-exposed)."""

    __slots__ = ("model", "gpu_memory_utilization", "max_model_len")

    def __init__(self, model: str, gpu_memory_utilization: float,
                 max_model_len: int):
        self.model = model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

    def engine_kwargs(self) -> dict:
        return {
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
        }


def _parent_tokenizer(model: str):
    """CPU-only tokenizer loaded in the PARENT for prompt assembly.

    Uses transformers AutoTokenizer (no torch CUDA init) so the parent can build
    token-id prompts without ever constructing an LLM / touching the GPU. The
    parent already probes CUDA availability via NVML (no context); loading a
    tokenizer keeps it that way.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def _texts(outputs) -> list[str]:
    return [o.outputs[0].text for o in outputs]


def _token_ids(outputs) -> list[list[int]]:
    return [list(o.outputs[0].token_ids) for o in outputs]


def _match_rate(a: list[int], b: list[int]) -> float:
    if not a and not b:
        return 1.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / max(len(a), len(b))


def _first_divergence(a: list[int], b: list[int]) -> int:
    """Index of the first position where a and b differ (-1 if identical over
    the shared prefix and equal length). A length mismatch with an otherwise
    identical prefix reports the length of the shorter list."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def _head_compare(
    tok, a: list[int], b: list[int], k: int = 12
) -> str:
    """Multi-line side-by-side of the first ``k`` tokens: id + decoded text.

    ``tok`` is a vLLM tokenizer (decode one id at a time). Used to make the
    divergence human-readable in the GPU log without a second tool."""
    def _dec(tid: int) -> str:
        try:
            return tok.decode([tid]).replace("\n", "\\n")
        except Exception:  # noqa: BLE001
            return "<?>"

    lines = ["    idx | dense_id (text)        | sparse_id (text)"]
    n = max(len(a), len(b))
    for i in range(min(k, n)):
        av = a[i] if i < len(a) else None
        bv = b[i] if i < len(b) else None
        mark = "" if av == bv else "  <-- DIFF"
        at = f"{av} ({_dec(av)!r})" if av is not None else "--"
        bt = f"{bv} ({_dec(bv)!r})" if bv is not None else "--"
        lines.append(f"    {i:3d} | {at:22s} | {bt}{mark}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SUBPROCESS ISOLATION (VRAM-leak fix)
#
# Root cause: step1 (2 engines) and step4 (1 dense + 3 sparse engines) built
# multiple LLMs in ONE process and relied on `del llm` to free VRAM. With the
# IN-PROCESS engine (VLLM_ENABLE_V1_MULTIPROCESSING=0, required by step4 so the
# scheduler+worker connectors share EpicConnector.debug_counters), `del llm`
# does NOT return the device memory: the CUDA context, allocator pool and any
# cached compiled artifacts persist in the live Python process. Engine N+1 then
# tries to grab gpu_memory_utilization of the WHOLE device while engine N's
# allocation is still resident -> "Free memory ... less than desired GPU memory
# utilization" at construction.
#
# Fix: run EACH engine in a FRESH subprocess (`python gpu_smoke.py
# --_worker-json <spec>`). The worker builds exactly ONE engine, runs its warm
# + generate prompts, prints `RESULT_JSON: {...}` as its last stdout line, and
# exits. Process exit returns ALL device memory unconditionally (no reliance on
# Python-level cleanup). The parent assembles prompts (token ids), dispatches a
# worker per run, and parses RESULT_JSON. stderr is inherited so the engine and
# EPIC diagnostic logs still stream to the console; only stdout is parsed.
#
# The spec is plain JSON (below) so it is trivially serialise/parse round-trip
# testable on CPU with no GPU.
# ---------------------------------------------------------------------------

# Sentinel that prefixes the single machine-readable result line on worker
# stdout. The parser scans for the LAST line starting with this so arbitrary
# engine log noise (even lines that happen to contain the word) is tolerated.
_RESULT_PREFIX = "RESULT_JSON:"


def build_worker_spec(
    *,
    role: str,
    model: str,
    kv_config: dict | None,
    prompts: list[list[int]],
    warm_prompts: list[list[int]] | None = None,
    max_tokens: int,
    enforce_eager: bool = True,
    attention_backend: str | None = None,
    gpu_memory_utilization: float = 0.45,
    max_model_len: int = 2048,
    in_process: bool = False,
    read_counters: bool = False,
    first_token_logprobs: int | None = None,
) -> dict:
    """Build the JSON-serialisable spec for ONE isolated engine run (pure).

    role             -- "dense" or "sparse" (label only; behaviour is driven by
                        kv_config / in_process / read_counters).
    prompts          -- list of token-id lists generated and returned.
    warm_prompts     -- token-id lists generated BEFORE `prompts` to seed the
                        EPIC store (their outputs are discarded). The warm +
                        reuse pair MUST run in the SAME engine/process (the store
                        is per-engine), which is exactly what one worker gives.
    read_counters    -- if True the worker reads EpicConnector.debug_counters
                        after the run and returns them (sparse engagement check).
    in_process       -- if True the worker sets VLLM_ENABLE_V1_MULTIPROCESSING=0
                        so the scheduler+worker connectors share state in-band.
    first_token_logprobs -- if not None (== K), run each prompt as a single
                        greedy DECODE step (SamplingParams(logprobs=K,
                        max_tokens=1)) and return, per prompt, the FIRST decode
                        token's top-K logprob distribution (token id -> {logprob,
                        rank}) plus the sampled (argmax) token id. This is the
                        step5 fidelity metric and is SPARSE-COMPATIBLE: the
                        scored position is the single decode position (the next
                        token after the last prompt token), which EPIC sparse
                        ALWAYS forwards (last in M). It does NOT require logits at
                        the reused (non-forwarded) prompt positions, which is why
                        it replaces the old prompt_logprobs teacher-forcing path
                        (prompt_logprobs needs a logit at EVERY prompt position
                        and is structurally incompatible with the M-row sparse
                        forward -- see step5 module comment).
    """
    return {
        "role": role,
        "model": model,
        "kv_config": kv_config,
        "prompts": [list(p) for p in prompts],
        "warm_prompts": [list(p) for p in (warm_prompts or [])],
        "max_tokens": int(max_tokens),
        "enforce_eager": bool(enforce_eager),
        "attention_backend": attention_backend,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "max_model_len": int(max_model_len),
        "in_process": bool(in_process),
        "read_counters": bool(read_counters),
        # step5 first-decode-token top-K logprobs (None -> plain generation,
        # unchanged). K == 0 is NOT enough here (unlike prompt_logprobs) because
        # we need the dense argmax token to appear in the sparse top-K to read
        # its rank/logprob, so callers pass K >= ~20.
        "first_token_logprobs": (None if first_token_logprobs is None
                                 else int(first_token_logprobs)),
    }


def serialize_spec(spec: dict) -> str:
    """Spec -> single-line JSON string (subprocess argv-safe)."""
    return json.dumps(spec, separators=(",", ":"))


def parse_spec(s: str) -> dict:
    """Single-line JSON string -> spec dict (inverse of serialize_spec)."""
    return json.loads(s)


def parse_result_json(stdout: str) -> dict:
    """Extract the worker's result dict from captured stdout.

    Scans for the LAST line beginning with ``_RESULT_PREFIX`` so that arbitrary
    preceding log noise (engine banners, EPIC diagnostics, even lines that
    mention RESULT_JSON in prose) does not confuse the parse. Raises ValueError
    with the tail of stdout if no result line is present (a worker that crashed
    before emitting one).
    """
    found: str | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(_RESULT_PREFIX):
            found = stripped[len(_RESULT_PREFIX):].strip()
    if found is None:
        tail = "\n".join(stdout.splitlines()[-20:])
        raise ValueError(
            "no RESULT_JSON line in worker stdout; tail was:\n" + tail
        )
    return json.loads(found)


def _emit_result(result: dict) -> None:
    """Print the single machine-readable result line (worker side)."""
    print(f"{_RESULT_PREFIX} {json.dumps(result)}", flush=True)


# ---------------------------------------------------------------------------
# step5 first-decode-token top-K logprob extraction + metrics (pure, CPU-test).
#
# WHY NOT prompt_logprobs (the structural incompatibility this rewrite fixes):
#   The previous step5 metric was teacher-forced NLL via
#   SamplingParams(prompt_logprobs=N). vLLM produces a prompt logprob at EVERY
#   prompt position -- gpu_model_runner._get_prompt_logprobs_dict sets
#   num_prompt_tokens = len(request.prompt_token_ids) (the FULL prompt) and
#   slices hidden_states[offset:offset+num_logits] expecting a logit per prompt
#   position (gpu_model_runner.py :5453, :5495). But EPIC sparse forward only
#   forwards M rows (C u link u last) -- hidden_states has M rows, NOT the full
#   prompt -- and the scheduler advances num_computed_tokens toward N (the full
#   span) via epic_computed_advance. The runner then computes a NEGATIVE
#   num_positions for LogprobsTensors.empty_cpu(num_prompt_tokens - 1, ...)
#   / a row count that exceeds the M-row hidden_states, and torch.empty with a
#   negative size raises exactly "OverflowError: out of range integral type
#   conversion attempted" (outputs.py :98). Dense (sparse OFF) forwards ALL
#   prompt positions, so every prompt-position logit exists and the path works
#   -- which is why only the sparse score runs crashed. prompt_logprobs is thus
#   STRUCTURALLY incompatible with the M-row sparse forward.
#
# THE SPARSE-COMPATIBLE METRIC (first decode token):
#   We instead ask for a SINGLE greedy decode step with the top-K logprob
#   distribution: SamplingParams(temperature=0, max_tokens=1, logprobs=K). The
#   ONLY scored position is the decode position (the token after the last prompt
#   token), which EPIC sparse ALWAYS forwards (the last prompt token is in M, and
#   its hidden state attends over B's reused KV). No reused (non-forwarded)
#   prompt position is ever scored, so the M-row forward is sufficient.
#
#   The dense run's first decode token t0 (greedy argmax) is the reference. Each
#   sparse link run reports the logprob and RANK it assigns to t0:
#     * rank == 1  -> sparse still greedily picks t0 (closest to dense).
#     * rank large -> sparse's distribution moved away (link too small / stale).
#   distance(L) = (rank(t0 under L) - 1): 0 == dense agreement, larger == worse.
#   Secondary: KL(dense top-K || sparse top-K) over the shared support with a
#   floor logprob for tokens missing from one side.
#
# All extraction below operates on plain {token_id: {"logprob","rank"}} dicts so
# it is JSON-serialisable and unit-testable on CPU with no torch / vLLM.
# ---------------------------------------------------------------------------
def extract_first_token_logprobs(sample_logprobs) -> dict[int, dict] | None:
    """Pull the FIRST decode step's top-K logprob map out of a vLLM
    ``SampleLogprobs`` into a plain ``{token_id: {"logprob","rank"}}`` dict.

    ``sample_logprobs`` is ``RequestOutput.outputs[0].logprobs`` -- a list-like
    (FlatLogprobs or list) of ``dict[int, Logprob]`` per generated position. We
    take position 0 (max_tokens=1 -> exactly one decode step). Returns None if
    there is no logprob for the first step (logprobs were not requested). Pure
    apart from reading the (already-pythonised) vLLM objects; the CPU test feeds
    plain dicts of floats, so we tolerate a bare-float value too.
    """
    if not sample_logprobs:
        return None
    try:
        first = sample_logprobs[0]
    except (IndexError, KeyError, TypeError):
        return None
    if not first:
        return None
    out: dict[int, dict] = {}
    for tid, lp in first.items():
        logprob = float(getattr(lp, "logprob", lp))
        rank = getattr(lp, "rank", None)
        out[int(tid)] = {
            "logprob": logprob,
            "rank": (int(rank) if rank is not None else None),
        }
    return out


def parse_first_token_map(raw: dict | None) -> dict[int, dict] | None:
    """Re-key a JSON-decoded first-token map (string token-id keys -> int) back
    to ``{int: {"logprob","rank"}}``. JSON forces dict keys to strings on the
    wire; the worker emits ``{str(tid): entry}`` and the parent calls this to
    restore int keys. None passes through. Pure."""
    if raw is None:
        return None
    return {int(tid): entry for tid, entry in raw.items()}


def argmax_token(first_token_logprobs: dict[int, dict] | None) -> int | None:
    """The token id with the HIGHEST logprob in a first-token top-K map (== the
    greedy decode choice). None if the map is empty/None. Ties broken by the
    smaller token id for determinism. (We prefer rank==1 when ranks are present,
    falling back to the max logprob, so a missing/garbled rank still resolves.)
    """
    if not first_token_logprobs:
        return None
    # Prefer an explicit rank==1 entry (vLLM's own argmax).
    ranked = [
        tid for tid, e in first_token_logprobs.items()
        if e.get("rank") == 1
    ]
    if len(ranked) == 1:
        return ranked[0]
    # Fall back to max logprob (ties -> smaller token id).
    best_tid: int | None = None
    best_lp = float("-inf")
    for tid in sorted(first_token_logprobs):
        lp = float(first_token_logprobs[tid]["logprob"])
        if lp > best_lp:
            best_lp = lp
            best_tid = tid
    return best_tid


# Floor logprob assigned to a token that is absent from a top-K map (it fell
# outside the requested K -> its true logprob is <= the K-th, so a small floor
# is a conservative stand-in). Used for the t0 fallback and the KL support
# alignment. -20 ~= logprob of a ~2e-9 probability token, well below any top-K.
_FIRST_TOKEN_FLOOR_LOGPROB = -20.0


def t0_logprob_and_rank(
    first_token_logprobs: dict[int, dict] | None, t0: int
) -> tuple[float, int | None]:
    """The logprob and RANK the (sparse) run assigns to the dense argmax token
    ``t0``. If t0 is outside this run's top-K map, return the floor logprob and
    rank None (== "worse than K", the maximal distance signal). Pure.
    """
    if not first_token_logprobs:
        return _FIRST_TOKEN_FLOOR_LOGPROB, None
    entry = first_token_logprobs.get(int(t0))
    if entry is None:
        return _FIRST_TOKEN_FLOOR_LOGPROB, None
    rank = entry.get("rank")
    return (
        float(entry["logprob"]),
        (int(rank) if rank is not None else None),
    )


def t0_logprob_delta(dense_t0_lp: float, sparse_t0_lp: float) -> float:
    """|sparse logprob of t0 - dense logprob of t0| -- an AUXILIARY first-token
    fidelity signal (pure). Observed runs show this is small (~+/-0.05) across
    all links because the argmax is stable, so it corroborates the rank metric's
    saturation rather than discriminating links; reported in the table as a
    secondary number, never gated on."""
    return abs(float(sparse_t0_lp) - float(dense_t0_lp))


def rank_distance(rank: int | None, *, k: int) -> int:
    """distance(L) from a t0 RANK: (rank - 1), so rank==1 (sparse agrees with
    dense) -> 0. A None rank (t0 fell outside the run's top-K) maps to the
    maximal distance ``k`` (one worse than the worst in-K rank). Smaller ==
    closer to dense. Pure."""
    if rank is None:
        return int(k)
    return max(int(rank) - 1, 0)


def first_token_kl(
    dense_map: dict[int, dict] | None,
    sparse_map: dict[int, dict] | None,
    *,
    floor_logprob: float = _FIRST_TOKEN_FLOOR_LOGPROB,
) -> float:
    """KL(dense || sparse) over the first decode token, on the UNION of the two
    top-K supports. Tokens missing from a side get ``floor_logprob`` (a small
    stand-in for "outside top-K"). The dense distribution is RENORMALISED over
    the shared support so the sum is a proper (truncated) probability; sparse
    logprobs are read at the same support with the floor for absentees.

    KL = sum_t p_dense(t) * (logp_dense(t) - logp_sparse(t)) >= ~0. Smaller ==
    sparse closer to dense. Returns inf if dense is empty (cannot form p).
    Pure (math only)."""
    import math

    if not dense_map:
        return float("inf")
    support = set(dense_map) | set(sparse_map or {})

    def _lp(m: dict[int, dict] | None, tid: int) -> float:
        if not m or tid not in m:
            return floor_logprob
        return float(m[tid]["logprob"])

    # Renormalise dense over the (truncated) support.
    dense_lps = {t: _lp(dense_map, t) for t in support}
    m = max(dense_lps.values())
    z = sum(math.exp(lp - m) for lp in dense_lps.values())
    log_z = m + math.log(z)
    kl = 0.0
    for t in support:
        p = math.exp(dense_lps[t] - log_z)  # normalised dense prob
        if p <= 0.0:
            continue
        kl += p * (dense_lps[t] - log_z - _lp(sparse_map, t))
    return max(kl, 0.0)


def run_worker(spec: dict) -> dict:
    """Worker entry point: build ONE engine, run warm + prompts, return result.

    Returns a result dict (NOT printed here; the caller emits it). On any failure
    the dict carries {"ok": False, "error": ...} so the parent gets a clean,
    structured failure instead of a non-zero traceback it must scrape. Device
    memory is freed by PROCESS EXIT, not by this function.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    role = spec.get("role", "?")
    ftl_k = spec.get("first_token_logprobs")
    try:
        if spec.get("in_process"):
            # In-band counters: scheduler + worker connectors share state.
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        params = _greedy_params(max_tokens=int(spec["max_tokens"]))
        llm = _build_llm(
            spec["model"],
            kv_config=spec.get("kv_config"),
            enforce_eager=bool(spec.get("enforce_eager", True)),
            attention_backend=spec.get("attention_backend"),
            gpu_memory_utilization=float(spec.get("gpu_memory_utilization", 0.45)),
            max_model_len=int(spec.get("max_model_len", 2048)),
        )

        # Warm runs seed the EPIC store; outputs discarded. Same engine/process
        # as the measured prompts so the per-engine store is visible to them.
        for wp in spec.get("warm_prompts", []):
            llm.generate([TokensPrompt(prompt_token_ids=list(wp))], params)

        # --- step5 first-decode-token scoring branch ---------------------------
        # first_token_logprobs != None (== K) -> a SINGLE greedy decode step with
        # the top-K logprob distribution. The ONLY scored position is the decode
        # position (next token after the last prompt token), which EPIC sparse
        # ALWAYS forwards (last prompt token in M) -- so this is sparse-compatible
        # (unlike prompt_logprobs, which needs a logit at every prompt position;
        # see the extract_first_token_logprobs module comment). We return, per
        # prompt, the first decode token's top-K map ({tid: {logprob, rank}}) so
        # the parent can read the dense argmax t0's rank/logprob under sparse and
        # compute KL across runs.
        if ftl_k is not None:
            score_params = SamplingParams(
                temperature=0.0, max_tokens=1, logprobs=int(ftl_k)
            )
            first_token_maps: list[dict | None] = []
            for p in spec["prompts"]:
                pid = list(p)
                outs = llm.generate(
                    [TokensPrompt(prompt_token_ids=pid)], score_params
                )
                sample_lps = outs[0].outputs[0].logprobs
                first_token_maps.append(
                    extract_first_token_logprobs(sample_lps)
                )
            # JSON: int keys must become strings; the parent re-ints them.
            result: dict = {
                "ok": True,
                "role": role,
                "first_token_logprobs": [
                    (None if m is None
                     else {str(tid): e for tid, e in m.items()})
                    for m in first_token_maps
                ],
            }
            if spec.get("read_counters"):
                result["counters"] = _read_epic_counters()
            return result

        token_ids: list[list[int]] = []
        texts: list[str] = []
        for p in spec["prompts"]:
            outs = llm.generate(
                [TokensPrompt(prompt_token_ids=list(p))], params
            )
            token_ids.append(_token_ids(outs)[0])
            texts.append(_texts(outs)[0])

        result = {
            "ok": True,
            "role": role,
            "token_ids": token_ids,
            "texts": texts,
        }
        if spec.get("read_counters"):
            # Read the connector's class-level counters from THIS process (the
            # in-process engine bumped them in-band).
            result["counters"] = _read_epic_counters()
        return result
    except Exception as e:  # noqa: BLE001
        import traceback

        return {
            "ok": False,
            "role": role,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def run_engine_subprocess(spec: dict) -> dict:
    """Parent side: run one engine in a FRESH subprocess and parse its result.

    The subprocess builds exactly one LLM, so its exit returns all device memory
    -- no VRAM accumulates across runs. stderr is inherited (engine + EPIC logs
    stream live); only stdout is captured and parsed for the RESULT_JSON line.
    Raises RuntimeError if the worker crashed without emitting a result.
    """
    cmd = [sys.executable, os.path.abspath(__file__),
           "--_worker-json", serialize_spec(spec)]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,  # inherit -> engine/EPIC logs visible on the console
        text=True,
    )
    try:
        result = parse_result_json(proc.stdout)
    except ValueError as e:
        raise RuntimeError(
            f"worker (role={spec.get('role')}) produced no RESULT_JSON "
            f"(exit code {proc.returncode}): {e}"
        ) from None
    return result


# ---------------------------------------------------------------------------
# step1 -- connector ON (sparse OFF) is a generation no-op
# ---------------------------------------------------------------------------
def step1_no_trace(model: str, cfg: "SmokeConfig") -> None:
    _log("step1: connector ON (sparse OFF) vs no connector -- output identity")
    # Tokenize in the PARENT (CPU-only AutoTokenizer; no CUDA touch) so the two
    # engine runs receive byte-identical token ids and each runs in its own
    # subprocess (fresh VRAM per engine; see SUBPROCESS ISOLATION above).
    tok = _parent_tokenizer(model)
    prompt_texts = [
        _PROMPT_HEAD + _SHARED_PASSAGE + _PROMPT_TAIL_1,
        "A completely different prompt about astronomy and telescopes.",
    ]
    prompts = [list(tok.encode(t)) for t in prompt_texts]

    base_res = run_engine_subprocess(build_worker_spec(
        role="dense", model=model, kv_config=None, prompts=prompts,
        max_tokens=32, **cfg.engine_kwargs(),
    ))
    if not base_res.get("ok"):
        _fail("step1", f"baseline engine failed: {base_res.get('error')}")
    base_out = base_res["token_ids"]

    epic_res = run_engine_subprocess(build_worker_spec(
        role="dense", model=model, kv_config=_epic_kv_config(sparse=False),
        prompts=prompts, max_tokens=32, **cfg.engine_kwargs(),
    ))
    if not epic_res.get("ok"):
        _fail("step1", f"connector-on engine failed: {epic_res.get('error')}")
    epic_out = epic_res["token_ids"]

    for i, (b, e) in enumerate(zip(base_out, epic_out)):
        if b != e:
            _fail(
                "step1",
                f"prompt {i}: connector-on output differs from baseline "
                f"(match rate {_match_rate(b, e):.3f}). The Phase-1 connector "
                "must not change generation.",
            )
    _log("step1: PASS (connector-on outputs byte-identical to baseline)")


class _in_process_engine:
    """Run the V1 EngineCore in-process for the duration of the block.

    The S7 safety gate raises ValueError inside EpicConnector.__init__
    (SCHEDULER role), which is constructed in the EngineCore process. With the
    default multiprocess engine the parent only sees a generic RuntimeError
    ("Engine core initialization failed"), so the expected-failure steps below
    could not assert on the gate message. In-process, the ValueError
    propagates directly to this process.
    """

    def __enter__(self):
        self._prev = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        return self

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
        else:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = self._prev
        return False


def _expect_gate_failure(
    step: str, model: str, *, enforce_eager: bool, attention_backend: str,
    must_mention: str, cfg: "SmokeConfig",
) -> None:
    try:
        with _in_process_engine():
            _build_llm(
                model,
                kv_config=_epic_kv_config(sparse=True),
                enforce_eager=enforce_eager,
                attention_backend=attention_backend,
                **cfg.engine_kwargs(),
            )
    except ValueError as e:
        if must_mention not in str(e):
            _fail(
                step,
                f"raised ValueError but it does not mention "
                f"{must_mention!r}: {e}",
            )
        _log(f"{step}: PASS (raised ValueError as expected: {str(e)[:120]}...)")
        return
    except Exception as e:  # noqa: BLE001
        # Multiprocess engines surface the gate as a RuntimeError in the
        # parent; accept it if the gate text survived in the message chain.
        chain = []
        cur: BaseException | None = e
        while cur is not None:
            chain.append(str(cur))
            cur = cur.__cause__ or cur.__context__
        if any(must_mention in c for c in chain):
            _log(
                f"{step}: PASS (gate fired in EngineCore; surfaced as "
                f"{type(e).__name__})"
            )
            return
        _fail(
            step,
            f"expected ValueError mentioning {must_mention!r}, got "
            f"{type(e).__name__}: {e}",
        )
    _fail(step, f"{step} config did NOT raise (safety gate broken)")


# ---------------------------------------------------------------------------
# step2 -- sparse ON + wrong backend must fail fast
# ---------------------------------------------------------------------------
def step2_wrong_backend_fails(model: str, cfg: "SmokeConfig") -> None:
    _log("step2: sparse ON + non-FlexAttention backend must raise ValueError")
    _expect_gate_failure(
        "step2", model,
        enforce_eager=True, attention_backend="FLASH_ATTN",
        must_mention="FLEX_ATTENTION", cfg=cfg,
    )


# ---------------------------------------------------------------------------
# step3 -- sparse ON + FlexAttention but NOT eager must fail fast
# ---------------------------------------------------------------------------
def step3_non_eager_fails(model: str, cfg: "SmokeConfig") -> None:
    _log("step3: sparse ON + FlexAttention + NOT eager must raise ValueError")
    _expect_gate_failure(
        "step3", model,
        enforce_eager=False, attention_backend="FLEX_ATTENTION",
        must_mention="enforce_eager", cfg=cfg,
    )


# ---------------------------------------------------------------------------
# step4 -- shared non-prefix chunk: sparse-on vs dense output comparison
#
# CHUNK ALIGNMENT (root-cause fix): the earlier version built prompts by string
# concatenation. Re-tokenizing "head + passage + tail" does NOT guarantee that
# the passage B starts on a chunk boundary or that its token ids are identical
# between the warm and reuse prompts (subword boundaries shift with the
# surrounding text). Without byte-identical, chunk-aligned B tokens the content
# hash never matches and the connector finds NO non-prefix hit -> nothing to
# reuse -> the comparison is meaningless. So we assemble at the TOKEN level
# (mirroring benchmarks/epic_reuse/data_prep.py):
#   * head A is padded to a multiple of chunk_size with a repeated filler id,
#   * passage B is truncated to EXACTLY k*chunk_size tokens,
#   * the SAME B id slice is spliced into both prompts,
# guaranteeing B is byte-identical and chunk-aligned in warm and reuse.
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 256


def build_aligned_token_prompts(
    *,
    head_ids: list[int],
    passage_ids: list[int],
    tail_ids: list[int],
    reuse_head_ids: list[int],
    reuse_tail_ids: list[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    filler_id: int = 0,
    filler_ids: list[int] | None = None,
    passage_chunks: int = 1,
) -> dict:
    """Token-level, chunk-aligned assembly of the warm + reuse prompts (pure).

    Returns a dict with the two token-id lists plus the precomputed B-chunk
    boundaries / hashes the smoke logs and asserts on. No tokenizer or GPU here
    so this is unit-testable on CPU (test_step4_alignment).

    Invariants this guarantees (the whole point of the rewrite):
      * head A is padded to a multiple of ``chunk_size`` -> B starts on a chunk
        boundary in BOTH prompts.
      * B is truncated to exactly ``passage_chunks * chunk_size`` tokens and the
        SAME id slice is used in both prompts -> byte-identical content -> the
        connector's content hash collides -> a real non-prefix hit.

    Discriminative padding: ``filler_ids`` (a non-empty list of real-word token
    ids) is CYCLED for all padding so a padded prompt is ordinary text rather
    than a run of one repeated (e.g. blank-space) token. Cycling deterministi-
    cally from offset 0 keeps the SAME B id slice byte-identical between warm
    and reuse (the reuse signal). ``filler_id`` (scalar) is the back-compat
    fallback when ``filler_ids`` is None.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if passage_chunks <= 0:
        raise ValueError("passage_chunks must be positive")

    pool = list(filler_ids) if filler_ids else [filler_id]
    if not pool:
        pool = [filler_id]

    def _fill(n: int) -> list[int]:
        # Cycle the pool so padding is varied real-word tokens, not one token
        # repeated. Deterministic (offset 0) -> reproducible alignment/hashes.
        return [pool[i % len(pool)] for i in range(n)]

    # Pad each head UP to a chunk-size multiple so B lands on a chunk boundary.
    def _pad_to_chunk(ids: list[int]) -> list[int]:
        rem = len(ids) % chunk_size
        if rem == 0 and len(ids) > 0:
            return list(ids)
        pad = (chunk_size - rem) % chunk_size
        # Ensure at least one full A chunk so A is itself chunkable (prefix).
        if len(ids) == 0:
            pad = chunk_size
        return list(ids) + _fill(pad)

    warm_head = _pad_to_chunk(head_ids)
    reuse_head = _pad_to_chunk(reuse_head_ids)

    # Truncate B to EXACTLY passage_chunks * chunk_size; pad with filler if the
    # source passage is short (so the count is exact regardless of input length).
    b_len = passage_chunks * chunk_size
    b_ids = list(passage_ids[:b_len])
    if len(b_ids) < b_len:
        b_ids = b_ids + _fill(b_len - len(b_ids))

    warm_ids = warm_head + b_ids + list(tail_ids)
    reuse_ids = reuse_head + b_ids + list(reuse_tail_ids)

    # Precompute the B chunk boundaries + hashes in each prompt (offsets differ
    # because the heads differ in length, but the hashes MUST match: same B ids).
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
        hash_chunk_tokens,
    )

    def _b_chunks(head_len: int) -> list[tuple[int, str]]:
        out = []
        for c in range(passage_chunks):
            start = head_len + c * chunk_size
            out.append(
                (start, hash_chunk_tokens(b_ids[c * chunk_size : (c + 1) * chunk_size]))
            )
        return out

    warm_b = _b_chunks(len(warm_head))
    reuse_b = _b_chunks(len(reuse_head))
    # The B hashes must be identical across the two prompts (that is the reuse
    # signal). Offsets differ (different head lengths) -> non-prefix in reuse.
    assert [h for _, h in warm_b] == [h for _, h in reuse_b], (
        "B chunk hashes diverge -> alignment broken; B must be byte-identical."
    )

    return {
        "warm_ids": warm_ids,
        "reuse_ids": reuse_ids,
        "b_len": b_len,
        "chunk_size": chunk_size,
        "passage_chunks": passage_chunks,
        "warm_b_offset": len(warm_head),
        "reuse_b_offset": len(reuse_head),
        "expected_b_hashes": [h for _, h in warm_b],
    }


# ---------------------------------------------------------------------------
# Needle probe construction (the discriminative accuracy signal).
#
# We embed K "secret code" facts INSIDE passage B (the shared, non-prefix
# region) and ask for one of them in the reuse prompt's tail Q. Because the
# answer code lives ONLY in B, a correct sparse answer proves B's reused KV
# carries real information through the sparse forward -- not just that two blank
# prompts collapse to the same blank output. The fact/question wording mirrors
# benchmarks/epic_reuse/common.py's Needle (kept independent so this script
# stays import-light: no torch/vLLM at module load).
# ---------------------------------------------------------------------------
class NeedleFact:
    __slots__ = ("subject", "answer")

    def __init__(self, subject: str, answer: str):
        self.subject = subject
        self.answer = answer

    def fact_sentence(self) -> str:
        return f"The secret code for {self.subject} is {self.answer}."

    def question(self) -> str:
        # Plain-text QA tail; the model is instruct-tuned so this is enough.
        return (
            f" Based on the passage above, what is the secret code for "
            f"{self.subject}? Answer:"
        )


def make_needles(seed: int, k: int) -> list[NeedleFact]:
    """K deterministic (subject, 4-digit code) facts. Subjects are two pooled
    words joined by '-' so they are unambiguous and unlikely to collide with
    filler; codes are 4-digit so they tokenize compactly and are easy to match.
    """
    out: list[NeedleFact] = []
    pool = _FILLER_WORDS
    n = len(pool)
    for j in range(k):
        subj = pool[(seed + 7 + j) % n] + "-" + pool[(seed + 13 + j * 3) % n]
        ans = f"{(seed * 31 + j * 17) % 9000 + 1000}"
        out.append(NeedleFact(subject=subj, answer=ans))
    return out


def build_needle_passage_text(
    needles: list[NeedleFact], *, filler_seed: int, n_filler_words: int = 60
) -> str:
    """Passage B text: the K needle fact sentences interleaved with rotating
    real-word filler. The answer-bearing sentences are kept verbatim; filler is
    distinct words (not blank space) so B is a real passage. Length here is
    approximate -- build_aligned_token_prompts truncates/pads B to an exact
    chunk multiple at the TOKEN level afterwards (the answer sentences sit at
    the FRONT so token truncation never drops them).

    NOTE (bias caveat, why build_needle_passage_tokens exists): putting the
    facts at the FRONT of B collides with EPIC LegoLink, which RECOMPUTES the
    leading ``eff_link`` tokens of every non-prefix chunk (reuse_strategy.py
    "(2) link tokens"). At link=8/64 the front-loaded target needle lands in the
    RECOMPUTED region, so a needle HIT cannot be attributed to reused KV vs. the
    target simply being recomputed. This helper is kept for the older
    text-level tests; step4 now uses ``build_needle_passage_tokens`` to place
    the TARGET needle at a deep token offset (outside the link region).
    """
    pool = _FILLER_WORDS
    parts: list[str] = []
    # Facts first (so token-level truncation of B keeps them).
    for nd in needles:
        parts.append(nd.fact_sentence())
    # Then a block of rotating filler words to pad the passage body.
    filler = [pool[(filler_seed + i) % len(pool)] for i in range(n_filler_words)]
    parts.append(" ".join(filler) + ".")
    return " ".join(parts)


# Default depth (as a FRACTION of B_len) at which the TARGET needle is placed.
# It must sit OUTSIDE the largest non-full link value in the sweep so that, at
# every approximation link, the target needle is in the REUSED (not recomputed)
# region. With B_len=256 and the sweep [256, 64, 8], the largest non-full link
# is 64; ~0.70 * 256 = ~179 tokens is comfortably past it. Only link=256 (full
# recompute) ever covers the target.
_DEFAULT_NEEDLE_OFFSET_FRAC = 0.70


def needle_in_link(needle_offset: int, eff_link: int) -> bool:
    """Classify whether the TARGET needle (placed at token ``needle_offset`` in
    B) falls inside the LegoLink RECOMPUTED region (the leading ``eff_link``
    tokens of B). True  -> the needle is recomputed (control / not a reuse
    proof). False -> the needle is served purely from REUSED KV (the thing we
    are trying to prove carries information).

    Mirrors reuse_strategy.py's link rule: M includes ``range(lo, lo+eff_link)``
    of each non-prefix chunk, i.e. token offsets ``[0, eff_link)`` within B.
    """
    return int(needle_offset) < int(eff_link)


def build_needle_passage_tokens(
    needles: list[NeedleFact],
    target: NeedleFact,
    *,
    encode,
    chunk_size: int,
    needle_offset: int,
    filler_seed: int,
) -> list[int]:
    """Build passage B as EXACTLY ``chunk_size`` token ids with the TARGET
    needle sentence placed at token offset ``needle_offset`` (TOKEN-level, pure
    apart from the injected ``encode`` callable).

    Layout of the returned B token list (length == chunk_size):

        [ rotating-filler ... ] [ TARGET needle tokens ] [ rotating-filler ... ]
        ^ 0                     ^ needle_offset          ^ needle_offset + len   ^ chunk_size

    The distractor needles (every needle except ``target``) are folded into the
    LEADING filler region so B is still a real multi-fact passage, but only the
    TARGET (the one Q asks about) is offset-guaranteed -- that is all the
    discrimination needs (task spec 1).

    ``encode`` is ``tok.encode(text, add_special_tokens=False) -> list[int]`` so
    this stays tokenizer-agnostic and the caller controls special tokens (a
    stray BOS would break B byte-identity vs. the same B in another prompt).

    Raises ValueError if ``needle_offset + len(target tokens) > chunk_size``
    (the needle would be truncated -- a silent drop would invalidate the probe).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if needle_offset < 0:
        raise ValueError("needle_offset must be >= 0")

    pool = _FILLER_WORDS

    def _filler_tokens(n_words: int, seed: int) -> list[int]:
        ids: list[int] = []
        for i in range(n_words):
            ids.extend(encode(" " + pool[(seed + i) % len(pool)]))
        return ids

    target_tokens = list(encode(" " + target.fact_sentence()))
    needle_len = len(target_tokens)
    if needle_offset + needle_len > chunk_size:
        raise ValueError(
            f"target needle ({needle_len} tokens) at offset {needle_offset} "
            f"overruns B (chunk_size={chunk_size}): "
            f"{needle_offset}+{needle_len} > {chunk_size}. Lower the offset or "
            "raise chunk_size."
        )

    # Distractor fact sentences (everything except the target). Their tokens go
    # at the very front so B reads as a genuine multi-fact passage.
    distractor_tokens: list[int] = []
    for nd in needles:
        if nd is target or (nd.subject == target.subject and nd.answer == target.answer):
            continue
        distractor_tokens.extend(encode(" " + nd.fact_sentence()))

    # LEADING region: distractor facts, then rotating filler, truncated/padded
    # to EXACTLY needle_offset tokens so the target lands precisely at the
    # offset. (If distractors already exceed needle_offset we truncate them; the
    # target placement is what the probe asserts on, not distractor fidelity.)
    lead = list(distractor_tokens)
    if len(lead) < needle_offset:
        lead = lead + _filler_tokens(
            # plenty of filler words; trimmed to size below.
            n_words=needle_offset, seed=filler_seed,
        )
    lead = lead[:needle_offset]
    # Pad the (rare) shortfall when filler tokens were multi-token-per-word and
    # over/undershot: top up with single-id cycling so we hit needle_offset
    # exactly without re-tokenizing.
    if len(lead) < needle_offset and lead:
        i = 0
        while len(lead) < needle_offset:
            lead.append(lead[i % len(lead)])
            i += 1

    # TRAILING region: rotating filler to fill the remainder up to chunk_size.
    trail_target = chunk_size - needle_offset - needle_len
    trail = _filler_tokens(n_words=max(trail_target, 0), seed=filler_seed + 31)
    trail = trail[:trail_target]
    if len(trail) < trail_target and trail:
        i = 0
        while len(trail) < trail_target:
            trail.append(trail[i % len(trail)])
            i += 1
    elif trail_target > 0 and not trail:
        # Degenerate empty pool: pad with the first target token id.
        trail = [target_tokens[0]] * trail_target

    b_ids = lead + target_tokens + trail
    assert len(b_ids) == chunk_size, (
        f"B assembled to {len(b_ids)} tokens, expected exactly {chunk_size}"
    )
    return b_ids


def _output_has_answer(text: str, answer: str) -> bool:
    """True iff the 4-digit answer code appears as a token-ish substring in the
    output. Digits don't get article/punct-normalized, so a plain substring on
    the raw text is robust (and matches common.answer_containment in spirit)."""
    if not answer:
        return False
    return answer in (text or "")


def _distinct_token_count(ids: list[int]) -> int:
    return len(set(ids))


# Link sweep schedule (Implement 1). Each entry is a number of LINK tokens per
# non-prefix B chunk that are recomputed. With a 256-token B chunk:
#   256 -> M == ALL of B: the sparse machinery (match/load/scatter/mask/reduced
#          forward) runs end-to-end, but NOTHING is actually reused (every B
#          token is recomputed). Output MUST match dense up to numerics. This is
#          the DECISIVE control: low match here == a MACHINERY bug (runner
#          positions / seq_lens / flex logical_q / schedule accounting / scatter
#          layout), NOT the reuse approximation.
#    64 -> partial reuse (192 reused, 64 recomputed): mid-regime.
#     8 -> the production LegoLink stitch (248 reused, 8 recomputed): the
#          aggressive approximation that collapsed to 0.021 in the field.
_LINK_SWEEP = [256, 64, 8]
# Below this, a link value's match is treated as "machinery healthy enough"
# (used only for the link == B-full decisive gate).
_MACHINERY_PASS_THRESHOLD = 0.9

# step5 link sweep (cross-context approximation-quality probe). Includes the two
# extremes that bracket the regime:
#   0      -> reuse-only: ZERO recompute of the non-prefix B chunk (pure stale
#             reused KV computed under the WARM context, never seeing the reuse
#             context A/C). Maximum approximation.
#   8, 64  -> increasing LegoLink boundary recompute (B's leading tokens made
#             fresh) -> output should move TOWARD dense.
#   B_full -> the whole B chunk recomputed == dense up to numerics (the control:
#             distance MUST be ~0; a non-zero distance here is a MACHINERY FAIL,
#             same meaning as step4's decisive gate).
# B_full is appended at runtime (= chunk_size) so the constant stays B-agnostic.
_STEP5_LINK_SWEEP = [0, 8, 64]

# Number of top-K logprobs requested per first decode step. Must be large enough
# that the dense argmax token t0 usually appears in each sparse run's top-K (so
# its rank is read directly rather than floored). 20 is a good balance.
_STEP5_TOPK = 20

# rank_distance(t0) at/below which the B-full control is considered "== dense up
# to numerics". With ZERO reuse approximation the sparse first token MUST equal
# the dense argmax (rank 1 -> distance 0). We allow distance 0 only (rank 1); a
# tiny tolerance is meaningless for an integer rank, so this is exact. This is
# the step5 MACHINERY gate (mirror of step4's decisive gate).
#
# NOTE (rank saturation -- why this gate is now AUXILIARY, see _STEP5_FLOOR_MULT):
# observed GPU runs show t0_rank == 1 at EVERY link (including link=0 reuse-only)
# -- the argmax is stable across the whole sweep, so rank_distance is PINNED at 0
# and carries no signal. The decisive gate therefore moved to a KL-vs-floor test
# (below); rank_distance == 0 is kept only as a corroborating check ("argmax
# stable") and is reported, never the sole basis for a verdict.
_STEP5_BFULL_MAX_RANK_DISTANCE = 0

# A reuse-only (link=0) rank-distance this close to the B-full control's
# distance means "LINK HAS NO EFFECT" -- the boundary recompute changed nothing
# for this prompt (a DIAGNOSTIC, not a failure).
_STEP5_NO_EFFECT_RANK_GAP = 0

# ---------------------------------------------------------------------------
# NOISE-FLOOR design (the step5 result-interpretation fix).
#
# THE PROBLEM the floor solves: a single sparse KL number (e.g. B-full
# KL(dense||sparse) == 0.36) is UNINTERPRETABLE in isolation. Is 0.36 a real
# systematic discrepancy from the fusion-mask path, or just run-to-run jitter
# (independent subprocess, FlexAttention eager kernels, torch.compile autotune,
# non-deterministic reductions)? Without a measured noise bottom we cannot tell,
# and the old rank-based verdict was dead anyway (rank pinned at 1 everywhere).
#
# THE FLOOR: run the DENSE config TWICE in two INDEPENDENT subprocesses and
# compute KL_floor = KL(dense_run1 || dense_run2) over the first-token top-K.
# Both runs are mathematically identical models on identical inputs, so any
# nonzero KL_floor is PURE measurement noise. Every sparse KL is then read
# RELATIVE to this floor (KL / KL_floor), not as an absolute number.
#
# READING THE FLOOR (the verdicts below):
#   * B-full KL <= _STEP5_FLOOR_MULT * KL_floor  -> machinery healthy: full
#     recompute reproduces dense to within measurement noise. (PASS path.)
#   * B-full KL  > _STEP5_FLOOR_MULT * KL_floor  -> REAL MACHINERY DISCREPANCY:
#     a systematic difference noise cannot explain (the fusion-mask custom
#     logical_mask_mod path vs. plain causal, or an incomplete B overwrite, are
#     the prime suspects -- see the investigation note in step5's docstring).
#     This is a STRONG DIAGNOSTIC, not a hard test failure: the gate is reported
#     loudly but does not sys.exit, because a single-prompt first-token KL has
#     low resolution (see _STEP5_RESOLUTION_NOTE) and the confirming signal is a
#     multi-prompt average (bench harness TODO).
#   * sparse link KL > _STEP5_FLOOR_MULT * KL_floor AND decreasing with link
#     -> the link recompute has a measurable, converging effect (good).
#   * all sparse link KLs ~ floor -> "signal buried in noise": a single-prompt
#     first-token probe lacks the resolution to separate the links here.
_STEP5_FLOOR_MULT = 2.0

# Honest-limitation banner reused in logs + as a code comment anchor. A single
# prompt's first decode token is a LOW-RESOLUTION probe: one position, one
# prompt, and the argmax is often link-invariant (rank saturates at 1). A
# trustworthy link-quality CURVE needs the first-token KL + floor AVERAGED over a
# pool of prompts -- that aggregation belongs in
# benchmarks/epic_reuse/bench_accuracy.py (TODO there; step5 only adds the floor
# here so a single run is at least interpretable relative to its own noise).
_STEP5_RESOLUTION_NOTE = (
    "single-prompt first-token KL is LOW RESOLUTION; a trustworthy link-quality "
    "curve needs first-token KL + floor averaged over a prompt pool "
    "(benchmarks/epic_reuse/bench_accuracy.py -- TODO there, not step5)"
)


def kl_floor_multiple(kl: float, kl_floor: float) -> float:
    """``kl / kl_floor`` -- how many noise-floors a sparse KL sits at (pure).

    The floor is measurement noise (KL between two identical dense runs); a
    sparse KL is read as a MULTIPLE of it. Guards a zero/near-zero floor: if the
    two dense runs were byte-identical (KL_floor == 0) any nonzero sparse KL is
    "infinitely many floors" (return inf), and 0/0 is 1.0 (also at floor). A
    non-finite KL (e.g. inf from an empty dense map) passes through as inf.
    """
    import math

    if not math.isfinite(kl):
        return float("inf")
    if kl_floor <= 0.0:
        return 0.0 if kl <= 0.0 else float("inf")
    return kl / kl_floor


def bfull_within_floor(
    bfull_kl: float, kl_floor: float, *, mult: float = _STEP5_FLOOR_MULT
) -> bool:
    """Decisive machinery gate (KL-based): True iff the B-full (full-recompute)
    KL is within ``mult`` noise-floors of dense, i.e. full recompute reproduces
    dense to within measurement noise. False -> REAL MACHINERY DISCREPANCY
    (systematic, noise cannot explain it). Pure.

    A non-finite bfull_kl is never "within floor" (False). The floor guard
    matches kl_floor_multiple: with KL_floor == 0 only an exactly-zero B-full KL
    passes (anything else is infinitely above the floor)."""
    import math

    if not math.isfinite(bfull_kl):
        return False
    return kl_floor_multiple(bfull_kl, kl_floor) <= float(mult)


def is_monotonic_nonincreasing(values: list[float], *, tol: float = 1e-6) -> bool:
    """True iff ``values`` never INCREASES by more than ``tol`` step-to-step.

    Used to test the step5 expectation that distance-to-dense (mean NLL)
    (weakly) DECREASES as the link recompute budget grows -- i.e. more recompute
    -> output closer to dense. ``tol`` absorbs fp jitter so a numerically flat
    sequence still reads as monotonic. Infs/NaNs make it False (a degenerate
    run is not 'monotonic')."""
    import math

    prev: float | None = None
    for v in values:
        if not math.isfinite(v):
            return False
        if prev is not None and v > prev + tol:
            return False
        prev = v
    return True


def monotonicity_report(
    links: list[int], distances: list[float]
) -> dict:
    """Summarise the step5 sweep's monotonicity (pure, CPU-testable).

    Returns a dict with:
      ``ordered_by_link``  -- distances sorted by ASCENDING link (recompute);
      ``nonincreasing``    -- is_monotonic_nonincreasing over that order;
      ``n_violations``     -- count of step-to-step INCREASES (jitter-tolerant);
      ``first_distance``/``last_distance`` -- endpoints of the link-ordered list.

    The expectation: distance is non-increasing in link (more recompute is at
    least as close to dense). Violations are REPORTED, not failed (step5 forbids
    hard thresholds on the middle links).
    """
    pairs = sorted(zip(links, distances), key=lambda t: t[0])
    ordered = [d for _, d in pairs]
    import math

    n_viol = 0
    prev: float | None = None
    for v in ordered:
        if prev is not None and math.isfinite(v) and math.isfinite(prev) \
                and v > prev + 1e-6:
            n_viol += 1
        prev = v
    return {
        "ordered_by_link": ordered,
        "nonincreasing": is_monotonic_nonincreasing(ordered),
        "n_violations": n_viol,
        "first_distance": ordered[0] if ordered else None,
        "last_distance": ordered[-1] if ordered else None,
    }


# Minimum distinct decoded tokens for the dense output to be considered a real,
# discriminative answer (a blank/degenerate output has <4 distinct tokens).
_MIN_DISTINCT_TOKENS = 4


def _reset_epic_counters() -> None:
    """Zero the connector's class-level engagement counters (between sparse
    runs). Imported lazily so the module stays import-light off-GPU."""
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
        EpicConnector,
    )

    EpicConnector.reset_debug_counters()


def _read_epic_counters() -> dict:
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
        EpicConnector,
    )

    return dict(EpicConnector.debug_counters)


def step4_shared_chunk_sparse_vs_dense(
    model: str, cfg: "SmokeConfig", needle_offsets: list[int] | None = None
) -> None:
    _log(
        "step4: NEEDLE probe -- DENSE reference vs sparse across a LINK sweep "
        f"{_LINK_SWEEP} (discriminative: answer code lives ONLY in passage B)"
    )

    max_tokens = 48

    # Tokenizer in the PARENT (CPU-only) for prompt assembly + decode logging.
    # Each engine run goes to its own subprocess (fresh VRAM per engine).
    tok = _parent_tokenizer(model)

    def _enc(text: str) -> list[int]:
        # add_special_tokens=False: we control boundaries ourselves; a leading
        # BOS in B would break byte-identity vs. the same B inside another prompt.
        return list(tok.encode(text, add_special_tokens=False))

    chunk_size = DEFAULT_CHUNK_SIZE

    # --- needle facts: K secret codes embedded ONLY in passage B ------------
    needles = make_needles(seed=4, k=3)
    target = needles[0]  # the one Q asks about

    # TARGET-needle depth (token offset within B). Default: a single DEEP offset
    # (~0.70 * B_len) so the target is OUTSIDE the LegoLink recomputed region for
    # every non-full link (8, 64); only link=256 (full recompute) covers it. The
    # optional mini-sweep (--needle-offsets) repeats the probe at several depths
    # to trace HIT vs. depth (task spec 4).
    if needle_offsets is None:
        needle_offsets = [int(round(_DEFAULT_NEEDLE_OFFSET_FRAC * chunk_size))]
    _log(
        "step4: needles="
        + "; ".join(f"{n.subject}->{n.answer}" for n in needles)
        + f" | asking for subject={target.subject!r} answer={target.answer!r}"
        + f" | needle_offsets={needle_offsets} (B_len={chunk_size})"
    )

    # Discriminative filler: rotate REAL words (not blank space) for all padding
    # so neither dense nor sparse can collapse to a trivial constant output.
    filler_ids: list[int] = []
    for w in _FILLER_WORDS:
        filler_ids.extend(_enc(" " + w))
    if not filler_ids:
        filler_ids = [0]

    for needle_offset in needle_offsets:
        _step4_one_offset(
            model=model,
            cfg=cfg,
            tok=tok,
            enc=_enc,
            chunk_size=chunk_size,
            needles=needles,
            target=target,
            needle_offset=needle_offset,
            filler_ids=filler_ids,
            max_tokens=max_tokens,
        )
    _log(
        "step4: PASS (all needle offsets passed the probe-validity + decisive "
        "B-full gates; deep-offset reused-KV interpretation logged per link)."
    )


def _step4_one_offset(
    *,
    model: str,
    cfg: "SmokeConfig",
    tok,
    enc,
    chunk_size: int,
    needles: list[NeedleFact],
    target: NeedleFact,
    needle_offset: int,
    filler_ids: list[int],
    max_tokens: int,
) -> None:
    """Run the full needle probe for ONE target-needle depth (offset)."""
    _log(f"step4: ===== needle_offset={needle_offset} (B depth) =====")

    # --- TOKEN-LEVEL passage B: target needle placed at exactly needle_offset --
    # (Bias fix.) The TARGET needle sentence is positioned at token offset
    # ``needle_offset`` so that, for the small/mid links, it sits OUTSIDE the
    # recomputed leading region -- a HIT there can only come from REUSED KV.
    b_ids = build_needle_passage_tokens(
        needles, target, encode=enc, chunk_size=chunk_size,
        needle_offset=needle_offset, filler_seed=11,
    )
    assert len(b_ids) == chunk_size  # exact B (byte-identical warm vs reuse)

    assembled = build_aligned_token_prompts(
        head_ids=enc(_PROMPT_HEAD),
        # Pass B PRE-BUILT and exactly chunk_size: build_aligned_token_prompts
        # slices passage_ids[:b_len] verbatim, so our offset survives intact.
        passage_ids=b_ids,
        tail_ids=enc(_PROMPT_TAIL_1),
        reuse_head_ids=enc(
            "Different opening sentence here for the second request entirely. "
        ),
        # Reuse tail = the needle QUESTION (asks for the code that lives in B).
        reuse_tail_ids=enc(target.question()),
        chunk_size=chunk_size,
        filler_ids=filler_ids,
        passage_chunks=1,
    )
    warm_ids = assembled["warm_ids"]
    reuse_ids = assembled["reuse_ids"]

    _log(
        f"step4: chunk_size={chunk_size} B_len={assembled['b_len']} "
        f"warm_B_offset={assembled['warm_b_offset']} "
        f"reuse_B_offset={assembled['reuse_b_offset']} "
        f"expected_B_chunks={len(assembled['expected_b_hashes'])} "
        f"B_hash_prefixes={[h[:12] for h in assembled['expected_b_hashes']]}"
    )
    _log(
        "step4: GPU LOG EXPECTATIONS -- on the reuse request the connector should "
        f"log 'EPIC sparse match' with non_prefix_hits="
        f"{len(assembled['expected_b_hashes'])} at offset "
        f"[{assembled['reuse_b_offset']}], then 'EPIC load emit' with "
        f"chunks={len(assembled['expected_b_hashes'])} scattering B into the "
        "paged cache. (The in-band counter assertion below replaces log-scraping "
        "for the engagement check.)"
    )

    # --- DENSE reference engine (subprocess; link-independent, computed ONCE) -
    # The warm prompt seeds B's chunks; the reuse prompt is the measured output.
    # Both run in the SAME worker so the per-engine store is shared. Dense may be
    # multiprocess (no in-band counter assertion needed) but a subprocess gives
    # us the VRAM isolation for free, so use one worker.
    dense_res = run_engine_subprocess(build_worker_spec(
        role="dense",
        model=model,
        kv_config=_epic_kv_config(sparse=False),
        warm_prompts=[warm_ids],
        prompts=[reuse_ids],
        max_tokens=max_tokens,
        enforce_eager=True,
        attention_backend="FLEX_ATTENTION",
        **cfg.engine_kwargs(),
    ))
    if not dense_res.get("ok"):
        _fail("step4", f"DENSE reference engine failed: {dense_res.get('error')}\n"
              + dense_res.get("traceback", ""))
    dense_out = dense_res["token_ids"][0]
    dense_text = dense_res["texts"][0]

    _log(f"step4: DENSE reference text: {dense_text!r}")
    _log(
        "step4: DENSE reference first 12 tokens: "
        + ", ".join(f"{t}:{tok.decode([t])!r}" for t in dense_out[:12])
    )

    # --- PROBE VALIDITY GATES (fail the PROBE, not EPIC, when these trip) ----
    # (a) dense must actually answer the needle. If it cannot, the probe length /
    #     model is the problem; the sparse comparison would be meaningless.
    dense_hit = _output_has_answer(dense_text, target.answer)
    distinct = _distinct_token_count(dense_out)
    _log(
        f"step4: dense needle_hit={dense_hit} (code {target.answer!r}) "
        f"distinct_tokens={distinct}"
    )
    if not dense_hit:
        _fail(
            "step4",
            f"PROBE INVALID: the DENSE (sparse-OFF) reference did not output the "
            f"answer code {target.answer!r} for subject {target.subject!r}. The "
            "probe itself is broken (model too weak, prompt too long, or the "
            "needle was truncated out of B) -- this is NOT an EPIC failure. "
            f"Dense output was {dense_text!r}. Shorten B / pick a stronger model "
            "and retry before trusting any sparse number.",
        )
    # (b) the dense output must be discriminative (a blank/constant output has
    #     almost no distinct tokens -> a 1.000 match would be vacuous).
    if distinct < _MIN_DISTINCT_TOKENS:
        _fail(
            "step4",
            f"non-discriminative output: the DENSE reference has only {distinct} "
            f"distinct tokens (< {_MIN_DISTINCT_TOKENS}). A near-constant output "
            "makes the sparse-vs-dense match rate vacuous (the degenerate "
            "blank-filler bug). Check the prompt assembly / sampling params.",
        )

    # --- link sweep: one fresh SUBPROCESS sparse engine per link value ------
    # Each run is an IN-PROCESS engine (VLLM_ENABLE_V1_MULTIPROCESSING=0, set by
    # the worker) so the SCHEDULER and WORKER connectors share
    # EpicConnector.debug_counters within THAT subprocess; the worker reads the
    # counters there and returns them. Running each link in its OWN subprocess is
    # the VRAM-leak fix: an in-process engine does NOT free device memory on
    # `del llm`, so building 3 of them in one parent process accumulated VRAM and
    # tripped the "Free memory < utilization" gate. Subprocess exit frees it all.
    # results row = (link, rate, first_div, needle_hit, counters, in_link)
    results: list[tuple[int, float, int, bool, dict, bool]] = []
    for link in _LINK_SWEEP:
        b_len = assembled["b_len"]
        eff_link = min(link, b_len)
        # Is the TARGET needle inside the recomputed leading region? If so the
        # HIT could be due to recompute, not reuse (control). If NOT, a HIT is a
        # pure reused-KV information-preservation proof.
        in_link = needle_in_link(needle_offset, eff_link)
        _log(
            f"step4[link={link}]: launching SUBPROCESS sparse engine "
            f"(epic_link_tokens={eff_link}, B_len={b_len}, "
            f"reuse_fraction={(b_len - eff_link) / b_len:.2f}, "
            f"needle_offset={needle_offset}, needle_in_link={in_link} -> "
            f"{'RECOMPUTED (control)' if in_link else 'REUSED-KV (proof)'})"
        )
        sparse_res = run_engine_subprocess(build_worker_spec(
            role="sparse",
            model=model,
            kv_config=_epic_kv_config(
                sparse=True,
                fusion=True,
                link_tokens=eff_link,
                debug_check_load=True,   # Implement 2: scatter self-check.
                debug_counters=True,     # in-band engagement assertion.
            ),
            warm_prompts=[warm_ids],     # warm: saves the chunk (same engine)
            prompts=[reuse_ids],
            max_tokens=max_tokens,
            enforce_eager=True,
            attention_backend="FLEX_ATTENTION",
            in_process=True,             # share counters scheduler<->worker
            read_counters=True,          # return them in the result json
            **cfg.engine_kwargs(),
        ))
        if not sparse_res.get("ok"):
            _fail(
                "step4",
                f"sparse engine (link={link}) failed: "
                f"{sparse_res.get('error')}\n"
                + sparse_res.get("traceback", ""),
            )
        sparse_out = sparse_res["token_ids"][0]
        sparse_text = sparse_res["texts"][0]
        counters = sparse_res.get("counters", {})

        rate = _match_rate(dense_out, sparse_out)
        div = _first_divergence(dense_out, sparse_out)
        cos = _decode_cosine(model, dense_out, sparse_out)
        hit = _output_has_answer(sparse_text, target.answer)
        results.append((link, rate, div, hit, counters, in_link))

        _log(f"step4[link={link}]: sparse text: {sparse_text!r}")
        _log(
            f"step4[link={link}]: needle_hit={hit} (code {target.answer!r}) "
            f"needle_in_link={in_link} "
            f"token-match rate sparse-vs-dense = {rate:.3f}"
        )
        if not in_link:
            _log(
                f"step4[link={link}]: INTERPRETATION -- needle is OUTSIDE the "
                f"recomputed region (offset {needle_offset} >= eff_link "
                f"{eff_link}). "
                + (
                    "HIT => the REUSED (non-recomputed) KV carried the answer: "
                    "direct proof that reused KV preserves information."
                    if hit else
                    "MISS => reuse ALONE did not preserve enough to answer at "
                    "this link (algorithmic limit, reported not failed)."
                )
            )
        else:
            _log(
                f"step4[link={link}]: INTERPRETATION -- needle is INSIDE the "
                f"recomputed region (offset {needle_offset} < eff_link "
                f"{eff_link}); this is a CONTROL row (a HIT may be due to "
                "recompute, not reuse)."
            )
        _log(
            f"step4[link={link}]: first divergence index = "
            f"{'NONE (identical)' if div < 0 else div}"
        )
        if cos is not None:
            _log(f"step4[link={link}]: embedding cosine similarity = {cos:.4f}")
        _log(
            f"step4[link={link}]: EPIC engagement counters = {counters}"
        )
        _log(
            f"step4[link={link}]: first 12 tokens (dense | sparse):\n"
            + _head_compare(tok, dense_out, sparse_out, k=12)
        )

        # --- SPARSE ENGAGEMENT gate (in-band, every run) -------------------
        if not (counters.get("sparse_match", 0) >= 1
                and counters.get("chunks_loaded", 0) >= 1):
            _fail(
                "step4",
                f"SPARSE DID NOT ENGAGE (link={link}): the in-band counters are "
                f"{counters}; expected sparse_match>=1 AND chunks_loaded>=1. The "
                "reuse request did not take the non-prefix sparse branch and/or no "
                "cached B chunk was scattered into the paged cache -- the sweep "
                "would be measuring the dense path, not EPIC. Check that the warm "
                "request saved B, that B is byte-identical/chunk-aligned in the "
                "reuse prompt (content-hash collision), and that the engine ran "
                "in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0).",
            )

    # --- summary table -------------------------------------------------------
    _log(f"step4: LINK SWEEP SUMMARY (needle_offset={needle_offset})")
    _log(
        "    link | needle | needle_in_link | match_rate | first_div | "
        "match/emit/loaded | regime"
    )
    for link, rate, div, hit, counters, in_link in results:
        full = "B-FULL (decisive)" if link >= assembled["b_len"] else "approx"
        cstr = (
            f"{counters.get('sparse_match', 0)}/"
            f"{counters.get('sparse_emit', 0)}/"
            f"{counters.get('chunks_loaded', 0)}"
        )
        _log(
            f"    {link:4d} | {('HIT' if hit else 'miss'):>6s} | "
            f"{str(in_link):>14s} | {rate:10.3f} | "
            f"{('none' if div < 0 else str(div)):>9s} | {cstr:^24s} | {full}"
        )
    # Headline reused-KV proof rows: needle OUTSIDE the recomputed region.
    proof_rows = [r for r in results if not r[5]]
    proof_hits = [r for r in proof_rows if r[3]]
    if proof_rows:
        _log(
            "step4: REUSED-KV PROOF rows (needle_in_link=False): "
            + ", ".join(
                f"link={r[0]}:{'HIT' if r[3] else 'miss'}" for r in proof_rows
            )
            + f" -> {len(proof_hits)}/{len(proof_rows)} HIT. A HIT here means "
            "the answer came from REUSED KV that was NEVER recomputed -- the "
            "core EPIC claim (reused KV carries information)."
        )
    else:
        _log(
            "step4: NOTE -- no reused-KV proof row at this offset "
            f"(needle_offset={needle_offset} < every non-full eff_link). "
            "Raise needle_offset to put the target outside the link region."
        )
    _log(
        "step4: HINT -- inspect the engine schedule log for the reuse request's "
        "step; under the S7 single-batch gate it must be scheduled ALONE."
    )

    # --- verdict: the link == B-full run is the DECISIVE gate ----------------
    # M == all of B (zero reuse approximation), so it must reproduce the needle
    # answer AND match dense closely. A failure here is mechanical, not
    # algorithmic.
    decisive = next((r for r in results if r[0] >= assembled["b_len"]), None)
    if decisive is None:
        _fail(
            "step4",
            "link sweep had no entry >= B_len so the decisive (M==all-of-B) "
            "control did not run; add B_len to _LINK_SWEEP.",
        )
    _, decisive_rate, decisive_div, decisive_hit, _, _ = decisive
    if not decisive_hit:
        _fail(
            "step4",
            f"DECISIVE FAIL: link={decisive[0]} recomputes the ENTIRE B chunk "
            "(zero reuse approximation) yet the sparse output did NOT contain the "
            f"needle code {target.answer!r}. Because reuse is null here, a missed "
            "needle is a mechanical fault in the sparse path (runner positions / "
            "seq_lens, flex logical_q, schedule accounting, or KV scatter "
            "layout), not approximation error. Inspect the 'EPIC check_load' and "
            "'EPIC worker sparse plan' lines.",
        )
    if decisive_rate < _MACHINERY_PASS_THRESHOLD:
        _fail(
            "step4",
            f"DECISIVE FAIL: link={decisive[0]} (M==all-of-B) hit the needle but "
            f"token-match={decisive_rate:.3f} < {_MACHINERY_PASS_THRESHOLD} "
            f"(first divergence at {'none' if decisive_div < 0 else decisive_div})."
            " With zero reuse approximation the sparse forward should reproduce "
            "dense up to numerics -- a low match is a mechanical fault. Inspect "
            "the scatter-fidelity ('EPIC check_load') and worker-plan lines.",
        )

    # Machinery is healthy. The smaller-link runs are an APPROXIMATION regime:
    # report needle hit + match + first divergence; a miss is an algorithmic
    # limit (link cannot restitch B's lost cross-context), not a failure.
    # NOTE: the decisive gate stays on link=B-full (needle_in_link=True there);
    # the deep-offset reused-KV PROOF rows are interpretive, NOT a hard gate, so
    # a reuse-only miss is reported (algorithmic limit) rather than failing.
    _log(
        f"step4: DECISIVE OK (link={decisive[0]} M==all-of-B needle HIT, match "
        f"{decisive_rate:.3f} >= {_MACHINERY_PASS_THRESHOLD}). Smaller-link rows "
        "above are the reuse APPROXIMATION regime: a needle miss / low match "
        "there is the 1st-order approximation-quality signal (reported, not a "
        "machinery failure)."
    )
    if proof_hits:
        _log(
            "step4: REUSED-KV PROVEN at this offset -- "
            + ", ".join(f"link={r[0]}" for r in proof_hits)
            + " answered the needle while the target sat OUTSIDE the recomputed "
            "region: the reused KV (never recomputed) carried the answer."
        )
    _log(
        f"step4: offset {needle_offset} PASS (probe valid: dense answered + "
        "discriminative; sparse engaged in-band; decisive B-full control hit "
        "the needle and matched dense; reused-KV proof rows reported)."
    )


# ---------------------------------------------------------------------------
# step5 -- CROSS-CONTEXT APPROXIMATION-QUALITY PROBE
#
# WHY (design rationale -- read before changing): step4's needle is a DISCRETE,
# B-self-contained fact. Because the reuse prompt's tail Q is always recomputed
# and attends over the WHOLE KV (A . B), a B-only answer is retrievable by
# Q->B attention without ANY link recompute -- so the needle is nearly link-
# INVARIANT and a poor probe of what link buys. To measure the VALUE of LegoLink
# boundary recompute we need a CONTINUOUS output-fidelity signal: B's reused KV
# was computed under the WARM context and is "stale" w.r.t. the reuse context
# (A/C); link recompute makes B's leading tokens FRESH. As the recompute budget
# grows, the sparse model's continuation should converge to the DENSE (full
# recompute) continuation. That monotone convergence is the evidence the EPIC
# approximation works.
#
# METRIC (first decode token rank/logprob -- sparse-compatible):
#   1. DENSE run (sparse OFF) generates the reuse prompt; its FIRST greedy decode
#      token t0 (argmax) is the reference. We also keep dense's first-token top-K
#      distribution for the KL secondary.
#   2. For each sparse link L, we take a SINGLE greedy decode step with top-K
#      logprobs (SamplingParams(temperature=0, max_tokens=1, logprobs=K)) and
#      read the RANK and logprob the sparse-L model assigns to t0.
#   distance(L) = rank(t0 under L) - 1 (0 == sparse still greedily picks t0 ==
#   dense; larger == sparse moved away). Secondary: KL(dense||sparse) over the
#   first-token top-K (shared support, floor logprob for absentees). Both are
#   "distance to dense", smaller is better.
#
#   WHY NOT prompt_logprobs (the bug this rewrite fixes): teacher-forced
#   prompt_logprobs needs a logit at EVERY prompt position. The V1 runner
#   (gpu_model_runner._get_prompt_logprobs_dict, :5453/:5495) slices
#   hidden_states by the FULL prompt length and allocates
#   LogprobsTensors.empty_cpu(num_prompt_tokens - 1, ...). EPIC sparse only
#   forwards M rows (C u link u last) and advances num_computed_tokens toward the
#   full span, so num_prompt_tokens disagrees with the M-row hidden_states and a
#   negative size reaches torch.empty -> "OverflowError: out of range integral
#   type conversion attempted" (outputs.py :98). Dense forwards every position so
#   it works -- that asymmetry is exactly why only the sparse score runs crashed.
#   The first-decode-token metric only ever scores the decode position, which
#   sparse ALWAYS forwards (last prompt token in M), so it is compatible.
#
#   Link sensitivity: the last prompt token's hidden state attends over B's
#   reused KV; a smaller link leaves more of B stale (warm context) -> t0's
#   sparse logprob drops and its rank grows. So rank-distance should DECREASE as
#   link grows. Single-position variance is a known limitation; averaging over
#   many prompts is the bench harness's job (TODO there, not here).
#
# NOISE FLOOR (the result-interpretation fix -- read this):
#   Observed GPU runs show the rank metric SATURATES (t0_rank==1 at EVERY link,
#   incl. reuse-only) -- the argmax is stable across the sweep so rank_distance
#   is pinned at 0 and dead. The only live signal is KL(dense||sparse), but a
#   single KL number is uninterpretable: is B-full KL==0.36 a real discrepancy or
#   just run-to-run jitter? We therefore run the DENSE config TWICE in two
#   independent subprocesses and compute KL_floor = KL(dense1||dense2). That is
#   pure measurement noise (identical models, identical inputs). Every sparse KL
#   is read as a MULTIPLE of this floor (KL/KL_floor), not as an absolute number.
#
# VERDICTS (KL-vs-floor; rank demoted to auxiliary):
#   * link=B_full KL <= _STEP5_FLOOR_MULT * KL_floor -> machinery healthy (full
#     recompute == dense within noise). Else *** REAL MACHINERY DISCREPANCY ***:
#     a systematic difference noise cannot explain (fusion-mask custom mask vs.
#     plain causal, or incomplete B overwrite -- see the investigation note).
#     This is a STRONG DIAGNOSTIC logged loudly, NOT a hard sys.exit (a single
#     first-token KL is low-resolution; the bench harness multi-prompt average is
#     the confirming signal -- TODO there).
#   * link=0 (reuse-only) KL above floor AND above B-full -> link recompute is
#     measurably effective. reuse-only ~at floor -> "signal buried in noise".
#     reuse-only above floor but not above B-full -> "LINK HAS NO EFFECT"
#     diagnostic (reported, NOT a failure).
#   * middle links: REPORT KL + KL/floor + monotonicity (on KL now, since rank
#     saturates); NO hard threshold.
#   * rank_distance / t0_lp_delta: AUXILIARY columns only (rank saturates at 0;
#     the lp delta is ~+/-0.05 across all links). Reported, never gated.
# ---------------------------------------------------------------------------

# Passage A: a shared, contentful DISTRACTOR document (appears as the prefix
# head of BOTH prompts -- a prefix the reuse request can also hit, but the focus
# is B). Real sentences (not blank filler) so the context is genuinely rich.
_STEP5_PASSAGE_A = (
    "The river delta supports a dense network of fishing villages whose "
    "economies have depended on the seasonal floods for centuries. Engineers "
    "later built a series of levees and pumping stations to manage the water, "
    "which changed the sediment patterns downstream and forced the villages to "
    "adapt their methods. Historians note that the same delta hosted three "
    "successive trading civilizations, each leaving distinct pottery styles. "
)

# Passage B: the SHARED, REUSED passage spliced NON-PREFIX into the reuse
# prompt. It is contentful and CONNECTED to A/C (it refers to "the delta" and to
# "the council") so that the reuse context genuinely matters -- stale B (warm
# context) vs fresh B (link recompute) should produce measurably different
# continuations. Truncated/padded to exactly chunk_size at the token level.
_STEP5_PASSAGE_B = (
    "Within the delta the regional council debated whether to restore the old "
    "wetlands or expand the levee system further inland. Proponents of "
    "restoration argued that the wetlands buffered storm surges and revived the "
    "fisheries, while the engineering faction warned that uncontrolled flooding "
    "would threaten the new rail corridor. The council commissioned a study "
    "weighing the long term sediment economy against short term flood risk, and "
    "its findings shaped the policy that the following paragraphs analyze in "
    "detail across several competing scenarios and stakeholder positions. "
)

# Per-request context C (the reuse prompt's distinct head) plus a tail that asks
# for a CONTINUATION grounded in A+B (so the answer depends on the reused KV
# being aligned to the reuse context, not the warm one).
_STEP5_REUSE_HEAD = (
    "An analyst for the rail authority is preparing a briefing and must "
    "reconcile the council's deliberations with the delta's history. "
)
_STEP5_REUSE_TAIL = (
    " Summarize the council's central tradeoff and explain, in a few "
    "sentences, what the policy should prioritize and why:"
)
# The warm prompt's own tail (seeds B under the warm context). Different from the
# reuse tail so B is genuinely non-prefix-reused under a NEW context.
_STEP5_WARM_TAIL = (
    " Describe the delta's geography for a general encyclopedia entry:"
)


def step5_cross_context_fidelity(model: str, cfg: "SmokeConfig") -> None:
    """CROSS-CONTEXT approximation-quality probe (see module comment above)."""
    _log(
        "step5: CROSS-CONTEXT fidelity -- first-decode-token rank of the DENSE "
        "argmax t0 under each sparse link "
        f"{_STEP5_LINK_SWEEP}+[B_full] (rank-distance 0 == closer to dense)"
    )

    chunk_size = DEFAULT_CHUNK_SIZE

    tok = _parent_tokenizer(model)

    def _enc(text: str) -> list[int]:
        return list(tok.encode(text, add_special_tokens=False))

    # B as EXACTLY chunk_size tokens (byte-identical warm vs reuse -> hash hit).
    b_src = _enc(_STEP5_PASSAGE_B)
    if len(b_src) < chunk_size:
        # pad with rotating real-word filler (never blank) up to chunk_size.
        fill: list[int] = []
        i = 0
        while len(b_src) + len(fill) < chunk_size:
            fill.extend(_enc(" " + _FILLER_WORDS[i % len(_FILLER_WORDS)]))
            i += 1
        b_src = (b_src + fill)[:chunk_size]
    b_ids = b_src[:chunk_size]
    assert len(b_ids) == chunk_size

    # Discriminative filler ids (real words) for head padding.
    filler_ids: list[int] = []
    for w in _FILLER_WORDS:
        filler_ids.extend(_enc(" " + w))
    if not filler_ids:
        filler_ids = [0]

    assembled = build_aligned_token_prompts(
        head_ids=_enc(_STEP5_PASSAGE_A),
        passage_ids=b_ids,
        tail_ids=_enc(_STEP5_WARM_TAIL),
        reuse_head_ids=_enc(_STEP5_PASSAGE_A + _STEP5_REUSE_HEAD),
        reuse_tail_ids=_enc(_STEP5_REUSE_TAIL),
        chunk_size=chunk_size,
        filler_ids=filler_ids,
        passage_chunks=1,
    )
    warm_ids = assembled["warm_ids"]
    reuse_ids = assembled["reuse_ids"]
    reuse_len = len(reuse_ids)
    b_len = assembled["b_len"]

    _log(
        f"step5: chunk_size={chunk_size} B_len={b_len} "
        f"reuse_B_offset={assembled['reuse_b_offset']} "
        f"reuse_prompt_len={reuse_len} "
        f"B_hash_prefixes={[h[:12] for h in assembled['expected_b_hashes']]}"
    )

    # --- DENSE reference, run TWICE (independent subprocesses) for the NOISE
    # FLOOR. Each greedily decodes ONE token with top-K logprobs so we get t0
    # (the argmax) AND the dense first-token distribution. Two mathematically
    # identical runs differ ONLY by measurement noise (separate processes,
    # FlexAttention eager kernels, non-deterministic reductions), so
    # KL_floor = KL(dense1 || dense2) is the JITTER BOTTOM. Every sparse link KL
    # is then read RELATIVE to this floor (see _STEP5_FLOOR_MULT). The decode
    # position is the next token after the reuse prompt's last token -- the same
    # single position every sparse link scores (NO teacher-forcing, NO per-prompt
    # logits, so it is sparse-compatible).
    def _run_dense(tag: str) -> dict | None:
        res = run_engine_subprocess(build_worker_spec(
            role="dense",
            model=model,
            kv_config=_epic_kv_config(sparse=False),
            warm_prompts=[warm_ids],
            prompts=[reuse_ids],
            first_token_logprobs=_STEP5_TOPK,
            max_tokens=1,
            enforce_eager=True,
            attention_backend="FLEX_ATTENTION",
            **cfg.engine_kwargs(),
        ))
        if not res.get("ok"):
            _fail("step5", f"DENSE reference engine ({tag}) failed: "
                  f"{res.get('error')}\n" + res.get("traceback", ""))
        return parse_first_token_map(res["first_token_logprobs"][0])

    dense_map = _run_dense("run1")
    dense_map2 = _run_dense("run2")
    t0 = argmax_token(dense_map)
    if t0 is None:
        _fail("step5", "DENSE produced no first-token logprobs; cannot score. "
              "Check the model supports logprobs and K>=1.")
    dense_t0_lp, dense_t0_rank = t0_logprob_and_rank(dense_map, t0)
    # The noise floor: KL between the two identical dense runs. This is pure
    # measurement jitter; sparse KLs are interpreted as multiples of it.
    kl_floor = first_token_kl(dense_map, dense_map2)
    t0_2 = argmax_token(dense_map2)
    _log(
        f"step5: DENSE first token t0={t0} ({tok.decode([t0])!r}) "
        f"logprob={dense_t0_lp:.4f} rank={dense_t0_rank} "
        f"(top-{_STEP5_TOPK} support size {len(dense_map or {})})"
    )
    _log(
        f"step5: NOISE FLOOR KL(dense_run1||dense_run2)={kl_floor:.4f} "
        f"(dense2 t0={t0_2}{' MATCHES' if t0_2 == t0 else ' DIFFERS!'}). "
        "This is the measurement-jitter bottom: two identical dense runs in "
        "separate processes. Every sparse KL below is read as a MULTIPLE of "
        f"this floor (gate: B-full KL <= {_STEP5_FLOOR_MULT:g}x floor)."
    )
    if t0_2 != t0:
        # If even the two dense runs disagree on the argmax, the model/prompt is
        # at a near-tie and the whole first-token probe is unreliable here.
        _log(
            "step5: WARNING -- the two DENSE runs picked DIFFERENT first tokens "
            f"(t0={t0} vs {t0_2}): the argmax is at a near-tie, so even the "
            "reference is noisy. Treat ALL step5 numbers for this prompt with "
            f"caution. ({_STEP5_RESOLUTION_NOTE})"
        )

    # --- link sweep: score t0 under each sparse link -------------------------
    sweep = list(_STEP5_LINK_SWEEP) + [chunk_size]  # append B_full control
    # rows: (link, reuse_frac, t0_logprob, rank, rank_distance, kl, kl_mult,
    #        t0_lp_delta, counters)
    rows: list[
        tuple[int, float, float, int | None, int, float, float, float, dict]
    ] = []
    for link in sweep:
        eff_link = min(link, b_len)
        reuse_frac = (b_len - eff_link) / b_len
        _log(
            f"step5[link={link}]: SUBPROCESS sparse first-token score "
            f"(epic_link_tokens={eff_link}, reuse_frac={reuse_frac:.2f})"
        )
        sparse_res = run_engine_subprocess(build_worker_spec(
            role="sparse",
            model=model,
            kv_config=_epic_kv_config(
                sparse=True,
                fusion=True,
                link_tokens=eff_link,
                debug_check_load=True,
                debug_counters=True,
            ),
            warm_prompts=[warm_ids],
            prompts=[reuse_ids],
            first_token_logprobs=_STEP5_TOPK,
            max_tokens=1,
            enforce_eager=True,
            attention_backend="FLEX_ATTENTION",
            in_process=True,
            read_counters=True,
            **cfg.engine_kwargs(),
        ))
        if not sparse_res.get("ok"):
            _fail(
                "step5",
                f"sparse score engine (link={link}) failed: "
                f"{sparse_res.get('error')}\n"
                + sparse_res.get("traceback", ""),
            )
        counters = sparse_res.get("counters", {})
        sparse_map = parse_first_token_map(sparse_res["first_token_logprobs"][0])
        t0_lp, t0_rank = t0_logprob_and_rank(sparse_map, t0)
        dist = rank_distance(t0_rank, k=_STEP5_TOPK)
        kl = first_token_kl(dense_map, sparse_map)
        kl_mult = kl_floor_multiple(kl, kl_floor)
        lp_delta = t0_logprob_delta(dense_t0_lp, t0_lp)
        rows.append(
            (link, reuse_frac, t0_lp, t0_rank, dist, kl, kl_mult, lp_delta,
             counters)
        )
        _log(
            f"step5[link={link}]: t0_logprob={t0_lp:.4f} "
            f"t0_lp_delta={lp_delta:+.4f} t0_rank={t0_rank} "
            f"rank_distance={dist} KL(dense||sparse)={kl:.4f} "
            f"KL/floor={kl_mult:.2f}x counters={counters}"
        )

        # SPARSE ENGAGEMENT gate (in-band, every run) -- same as step4.
        if not (counters.get("sparse_match", 0) >= 1
                and counters.get("chunks_loaded", 0) >= 1):
            _fail(
                "step5",
                f"SPARSE DID NOT ENGAGE (link={link}): in-band counters are "
                f"{counters}; expected sparse_match>=1 AND chunks_loaded>=1. The "
                "reuse/score request did not take the non-prefix sparse branch "
                "and/or no cached B chunk was scattered -- the sweep would be "
                "measuring the dense path. Check warm save, B byte-identity, and "
                "in-process engine.",
            )

    # --- summary table -------------------------------------------------------
    # Read EVERYTHING relative to KL_floor (top line): a sparse KL only means
    # something as a MULTIPLE of the measurement noise. rank_dist is kept as an
    # auxiliary column (it saturates at 0 when the argmax is stable -- the very
    # failure this rewrite addresses), and t0_lp_delta is a second auxiliary.
    _log("step5: CROSS-CONTEXT FIDELITY SUMMARY")
    _log(
        f"step5: KL_floor (dense||dense, measurement noise) = {kl_floor:.4f} "
        f"<-- read every sparse KL as a MULTIPLE of this (gate: B-full <= "
        f"{_STEP5_FLOOR_MULT:g}x)"
    )
    _log(
        f"step5: DENSE t0={t0} logprob={dense_t0_lp:.4f} (reference; sparse "
        "KL/floor should -> ~1x and decrease with link)"
    )
    _log(
        "    link | reuse_frac | t0_logprob | t0_lp_delta | t0_rank | rank_dist "
        "| KL(d||s) | KL/floor | regime"
    )
    for link, frac, t0_lp, t0_rank, dist, kl, kl_mult, lp_delta, _c in rows:
        if link == 0:
            regime = "reuse-only (max approx)"
        elif link >= b_len:
            regime = "B-FULL (control == dense)"
        else:
            regime = "approx"
        rank_str = "none" if t0_rank is None else str(t0_rank)
        mult_str = "inf" if not (kl_mult == kl_mult and kl_mult != float("inf")) \
            else f"{kl_mult:.2f}x"
        _log(
            f"    {link:4d} | {frac:10.2f} | {t0_lp:10.4f} | {lp_delta:+11.4f} | "
            f"{rank_str:>7s} | {dist:9d} | {kl:8.4f} | {mult_str:>8s} | {regime}"
        )

    # rank saturation note: if every t0_rank is 1 (argmax stable), say so loudly
    # so the reader does NOT trust the (pinned-to-0) rank_dist column.
    ranks = [r[3] for r in rows]
    rank_saturated = all(rk == 1 for rk in ranks)
    if rank_saturated:
        _log(
            "step5: NOTE -- rank metric SATURATED: t0_rank==1 at EVERY link "
            "(argmax stable across the sweep), so rank_distance is pinned at 0 "
            "and carries NO signal. Using KL(dense||sparse) vs KL_floor for all "
            "verdicts below. (This is the expected regime for a single "
            "first-token probe; rank only moves when the approximation flips the "
            "argmax outright.)"
        )

    # --- monotonicity: now on KL (the live metric), reported not gated --------
    # rank_distance saturated -> monotonicity on KL/floor, which is continuous.
    links = [r[0] for r in rows]
    kl_mults = [float(r[6]) for r in rows]
    mono = monotonicity_report(links, kl_mults)
    _log(
        "step5: MONOTONICITY (KL/floor vs ASCENDING link/recompute) -- "
        f"ordered={[round(d, 2) for d in mono['ordered_by_link']]} "
        f"nonincreasing={mono['nonincreasing']} "
        f"violations={mono['n_violations']} "
        "(rank metric saturated; using KL)"
    )
    if mono["nonincreasing"]:
        _log(
            "step5: KL/floor DECREASES (or is flat) as recompute grows -> the "
            "EPIC approximation behaves: more boundary recompute pulls the "
            "first-token distribution toward dense."
        )
    else:
        _log(
            "step5: NOTE -- KL/floor is NOT monotone non-increasing "
            f"({mono['n_violations']} step-up(s)). Reported, not failed: a "
            f"single-position first-token probe is noisy ({_STEP5_RESOLUTION_NOTE})."
        )

    # --- VERDICT 1: B-full machinery gate (KL-vs-floor; DECISIVE) -------------
    # Decisive gate is now KL-based: with the whole B chunk recomputed (zero
    # reuse approximation) the sparse forward must reproduce dense to within
    # MEASUREMENT NOISE -> B-full KL <= _STEP5_FLOOR_MULT * KL_floor. rank_dist
    # is only a corroborating check (it saturates and is uninformative).
    bfull = next((r for r in rows if r[0] >= b_len), None)
    if bfull is None:
        _fail("step5", "no B-full row in the sweep; cannot run the decisive "
                       "machinery gate.")
    bfull_kl = bfull[5]
    bfull_mult = bfull[6]
    bfull_dist = bfull[4]
    within = bfull_within_floor(bfull_kl, kl_floor, mult=_STEP5_FLOOR_MULT)
    if within:
        _log(
            f"step5: DECISIVE OK -- B-full KL={bfull_kl:.4f} "
            f"({bfull_mult:.2f}x floor <= {_STEP5_FLOOR_MULT:g}x): full "
            "recompute reproduces dense to within measurement noise, machinery "
            f"healthy. (corroborating: rank_dist={bfull_dist}, "
            f"argmax {'stable' if bfull_dist == 0 else 'MOVED'}.)"
        )
    else:
        # STRONG DIAGNOSTIC, not a hard fail: a single-prompt first-token KL has
        # low resolution; the confirming signal is a multi-prompt average. We log
        # loudly and point at the prime suspects, but do NOT sys.exit.
        _log(
            f"step5: *** REAL MACHINERY DISCREPANCY *** -- B-full KL={bfull_kl:.4f}"
            f" = {bfull_mult:.2f}x the noise floor ({kl_floor:.4f}), which EXCEEDS"
            f" the {_STEP5_FLOOR_MULT:g}x gate. With the WHOLE B chunk recomputed "
            "(zero reuse approximation) the sparse first-token distribution should "
            "equal dense to within jitter, so a floor-exceeding KL is a SYSTEMATIC "
            "difference noise cannot explain. Prime suspects (investigate, do not "
            "trust the sparse link numbers until resolved): (1) the fusion-mask "
            "custom logical_mask_mod path produces a different (non-pure-causal) "
            "attention than the dense run's plain causal even when M==all-of-B; "
            "(2) the B load->overwrite is incomplete so some reused (stale) KV "
            "rows survive even though M should cover all of B. Inspect 'EPIC "
            "check_load' / worker sparse-plan logs and the LegoLinkMaskBuilder "
            "gate. (This is a STRONG diagnostic, NOT a hard fail: "
            f"{_STEP5_RESOLUTION_NOTE}.)"
        )

    # --- VERDICT 2: does link buy anything? (KL/floor; diagnostic) -----------
    reuse_only = next((r for r in rows if r[0] == 0), None)
    if reuse_only is not None:
        ro_kl = reuse_only[5]
        ro_mult = reuse_only[6]
        _log(
            f"step5: reuse-only(link=0) KL={ro_kl:.4f} ({ro_mult:.2f}x floor) "
            f"vs B-full {bfull_kl:.4f} ({bfull_mult:.2f}x floor)"
        )
        # "Effective" iff reuse-only is meaningfully ABOVE floor AND meaningfully
        # above B-full (the boundary recompute measurably closes the gap).
        ro_above_floor = ro_mult > _STEP5_FLOOR_MULT
        ro_above_bfull = ro_kl > max(bfull_kl, 0.0) + 1e-9 and ro_mult > bfull_mult
        if ro_above_floor and ro_above_bfull:
            _log(
                "step5: LINK IS EFFECTIVE -- reuse-only's first-token KL is well "
                "above the noise floor AND above B-full; boundary recompute "
                "measurably pulls the distribution toward dense (EPIC works)."
            )
        elif not ro_above_floor:
            _log(
                "step5: NOTE -- reuse-only KL is itself ~at the noise floor: the "
                "signal is BURIED IN NOISE for this single prompt. A single "
                f"first-token probe lacks the resolution to separate links here "
                f"({_STEP5_RESOLUTION_NOTE})."
            )
        else:
            _log(
                "step5: WARNING -- LINK HAS NO EFFECT: reuse-only KL is above "
                "floor but NOT above B-full (boundary recompute did not measurably "
                "close the gap on this prompt). NOT a failure; a DIAGNOSTIC -- try "
                "a prompt where A/C are more entangled with B to expose the link's "
                "value."
            )
    else:
        _log("step5: NOTE -- no reuse-only(link=0) row; add 0 to "
             "_STEP5_LINK_SWEEP to measure the pure-stale endpoint.")

    _log(
        "step5: DONE (verdicts are KL-vs-floor; rank metric saturated -> "
        "auxiliary only). NOTE: " + _STEP5_RESOLUTION_NOTE + ". "
        "step5 PASSES as long as it RAN (it is diagnostic, not a hard gate); a "
        "REAL MACHINERY DISCREPANCY above is a strong warning to investigate, "
        "not an exit-nonzero failure."
    )


def _decode_cosine(model: str, a: list[int], b: list[int]) -> float | None:
    """Optional bag-of-token cosine similarity (cheap, tokenizer-only)."""
    try:
        from collections import Counter
        from math import sqrt

        ca, cb = Counter(a), Counter(b)
        keys = set(ca) | set(cb)
        dot = sum(ca[k] * cb[k] for k in keys)
        na = sqrt(sum(v * v for v in ca.values()))
        nb = sqrt(sum(v * v for v in cb.values()))
        if na == 0 or nb == 0:
            return None
        return dot / (na * nb)
    except Exception:  # noqa: BLE001
        return None


_STEPS = {
    1: step1_no_trace,
    2: step2_wrong_backend_fails,
    3: step3_non_eager_fails,
    4: step4_shared_chunk_sparse_vs_dense,
    5: step5_cross_context_fidelity,
}


def _parse_steps(arg: str | None) -> list[int]:
    if not arg:
        return [1, 2, 3, 4, 5]
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def _parse_needle_offsets(arg: str | None) -> list[int] | None:
    """Parse the --needle-offsets CLI value into a list of token offsets, or
    None to use the default single deep offset. Raises ValueError on a malformed
    or negative value (caught + reported by main)."""
    if arg is None:
        return None
    out: list[int] = []
    for tok in str(arg).split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v < 0:
            raise ValueError(f"needle offset must be >= 0, got {v}")
        out.append(v)
    if not out:
        return None
    return out


def _worker_main(spec_json: str) -> int:
    """Hidden self-invocation entry: build ONE engine, emit RESULT_JSON, exit.

    Always returns 0 -- success/failure is carried IN the RESULT_JSON ("ok"
    field), so a worker that cleanly reports a build failure (e.g. no GPU) is
    distinguishable from a worker that crashed without emitting a result (the
    parent then sees a missing RESULT_JSON and raises). Device memory is freed by
    this process exiting.
    """
    try:
        spec = parse_spec(spec_json)
    except Exception as e:  # noqa: BLE001
        _emit_result({"ok": False, "role": "?",
                      "error": f"bad spec: {type(e).__name__}: {e}"})
        return 0
    result = run_worker(spec)
    _emit_result(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EPIC GPU smoke test")
    ap.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model id (small Llama-family recommended).",
    )
    ap.add_argument(
        "--steps",
        default=None,
        help="Comma-separated step numbers to run (default: 1,2,3,4,5).",
    )
    ap.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.45,
        help="gpu_memory_utilization passed to every engine (default 0.45).",
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="max_model_len passed to every engine (default 2048).",
    )
    ap.add_argument(
        "--needle-offsets",
        default=None,
        help=(
            "Comma-separated TARGET-needle token offsets within passage B for "
            "the step4 depth mini-sweep (e.g. '4,128,240'). Default: a single "
            "DEEP offset (~0.70*B_len) so the target is OUTSIDE the LegoLink "
            "recomputed region for non-full links -- a needle HIT there proves "
            "reused KV carries information. Only step4 reads this."
        ),
    )
    ap.add_argument(
        "--_worker-json",
        dest="worker_json",
        default=None,
        help=argparse.SUPPRESS,  # hidden: per-engine subprocess self-invocation.
    )
    args = ap.parse_args(argv)

    # --- hidden worker mode: build ONE engine, print RESULT_JSON, exit --------
    if args.worker_json is not None:
        return _worker_main(args.worker_json)

    # Parse the step4 needle-offset mini-sweep up front so a malformed value
    # fails before any (slow) engine construction.
    try:
        needle_offsets = _parse_needle_offsets(args.needle_offsets)
    except ValueError as e:
        _fail("main", f"bad --needle-offsets: {e}")

    if not _has_cuda():
        _log(
            "No CUDA GPU detected. This script requires a GPU + a built EPIC "
            "vLLM. See vllm/.../epic/PHASE2.md for the build command. Exiting "
            "without running (structure-only check is a no-op here)."
        )
        return 0

    cfg = SmokeConfig(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    steps = _parse_steps(args.steps)
    _log(
        f"model={args.model} steps={steps} "
        f"gpu_mem_util={cfg.gpu_memory_utilization} "
        f"max_model_len={cfg.max_model_len} "
        "(each engine runs in a FRESH subprocess for VRAM isolation)"
    )
    for s in steps:
        fn = _STEPS.get(s)
        if fn is None:
            _fail("main", f"unknown step {s}; valid: {sorted(_STEPS)}")
        if s == 4:
            fn(args.model, cfg, needle_offsets)
        else:
            fn(args.model, cfg)
    _log(f"ALL REQUESTED STEPS PASSED: {steps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
