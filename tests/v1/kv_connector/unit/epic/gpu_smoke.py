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
         Answer:"). A warm request seeds B's chunks into the EPIC store; the
         reuse request (different head A, B in the MIDDLE, Q at the tail) is
         then answered DENSE (reference) and SPARSE (across a link sweep).
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
import os
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
):
    from vllm import LLM

    kwargs: dict = dict(
        model=model,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.45,
        max_model_len=2048,
    )
    if kv_config is not None:
        kwargs["kv_transfer_config"] = kv_config
    if attention_backend is not None:
        # v0.22 removed VLLM_ATTENTION_BACKEND; select the backend through the
        # EngineArgs/AttentionConfig surface. The string is normalized by
        # AttentionConfig.validate_backend_before (upper-cased enum lookup).
        kwargs["attention_backend"] = attention_backend
    return LLM(**kwargs)


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
# step1 -- connector ON (sparse OFF) is a generation no-op
# ---------------------------------------------------------------------------
def step1_no_trace(model: str) -> None:
    _log("step1: connector ON (sparse OFF) vs no connector -- output identity")
    prompts = [
        _PROMPT_HEAD + _SHARED_PASSAGE + _PROMPT_TAIL_1,
        "A completely different prompt about astronomy and telescopes.",
    ]
    params = _greedy_params()

    llm_base = _build_llm(model, kv_config=None)
    base_out = _token_ids(llm_base.generate(prompts, params))
    del llm_base

    llm_epic = _build_llm(model, kv_config=_epic_kv_config(sparse=False))
    epic_out = _token_ids(llm_epic.generate(prompts, params))
    del llm_epic

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
    must_mention: str,
) -> None:
    try:
        with _in_process_engine():
            _build_llm(
                model,
                kv_config=_epic_kv_config(sparse=True),
                enforce_eager=enforce_eager,
                attention_backend=attention_backend,
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
def step2_wrong_backend_fails(model: str) -> None:
    _log("step2: sparse ON + non-FlexAttention backend must raise ValueError")
    _expect_gate_failure(
        "step2", model,
        enforce_eager=True, attention_backend="FLASH_ATTN",
        must_mention="FLEX_ATTENTION",
    )


# ---------------------------------------------------------------------------
# step3 -- sparse ON + FlexAttention but NOT eager must fail fast
# ---------------------------------------------------------------------------
def step3_non_eager_fails(model: str) -> None:
    _log("step3: sparse ON + FlexAttention + NOT eager must raise ValueError")
    _expect_gate_failure(
        "step3", model,
        enforce_eager=False, attention_backend="FLEX_ATTENTION",
        must_mention="enforce_eager",
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


def step4_shared_chunk_sparse_vs_dense(model: str) -> None:
    _log(
        "step4: NEEDLE probe -- DENSE reference vs sparse across a LINK sweep "
        f"{_LINK_SWEEP} (discriminative: answer code lives ONLY in passage B)"
    )

    from vllm.inputs import TokensPrompt

    params = _greedy_params(max_tokens=48)

    # --- dense reference engine (also gives us the tokenizer for assembly) ---
    # The dense reference is independent of the link value, so compute it ONCE.
    # Dense reference may run multiprocess (no in-band counter assertion needed).
    llm_dense = _build_llm(
        model,
        kv_config=_epic_kv_config(sparse=False),
        attention_backend="FLEX_ATTENTION",
    )
    tok = llm_dense.get_tokenizer()

    def _enc(text: str) -> list[int]:
        # add_special_tokens=False: we control boundaries ourselves; a leading
        # BOS in B would break byte-identity vs. the same B inside another prompt.
        return list(tok.encode(text, add_special_tokens=False))

    chunk_size = DEFAULT_CHUNK_SIZE

    # --- needle facts: K secret codes embedded ONLY in passage B ------------
    needles = make_needles(seed=4, k=3)
    target = needles[0]  # the one Q asks about
    passage_text = build_needle_passage_text(needles, filler_seed=11)
    _log(
        "step4: needles="
        + "; ".join(f"{n.subject}->{n.answer}" for n in needles)
        + f" | asking for subject={target.subject!r} answer={target.answer!r}"
    )

    # Discriminative filler: rotate REAL words (not blank space) for all padding
    # so neither dense nor sparse can collapse to a trivial constant output.
    filler_ids: list[int] = []
    for w in _FILLER_WORDS:
        filler_ids.extend(_enc(" " + w))
    if not filler_ids:
        filler_ids = [0]

    assembled = build_aligned_token_prompts(
        head_ids=_enc(_PROMPT_HEAD),
        passage_ids=_enc(passage_text),
        tail_ids=_enc(_PROMPT_TAIL_1),
        reuse_head_ids=_enc(
            "Different opening sentence here for the second request entirely. "
        ),
        # Reuse tail = the needle QUESTION (asks for the code that lives in B).
        reuse_tail_ids=_enc(target.question()),
        chunk_size=chunk_size,
        filler_ids=filler_ids,
        passage_chunks=1,
    )
    warm_prompt = TokensPrompt(prompt_token_ids=assembled["warm_ids"])
    reuse_prompt = TokensPrompt(prompt_token_ids=assembled["reuse_ids"])

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

    # warm: saves the shared B chunk(s). dense_out is the link-independent ref.
    llm_dense.generate([warm_prompt], params)
    dense_outs = llm_dense.generate([reuse_prompt], params)
    dense_out = _token_ids(dense_outs)[0]
    dense_text = _texts(dense_outs)[0]
    del llm_dense

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

    # --- link sweep: one fresh IN-PROCESS sparse engine per link value ------
    # In-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the SCHEDULER and WORKER
    # connectors share EpicConnector.debug_counters in THIS process -> we can
    # assert engagement in-band (no log scraping). Reset the counters per run.
    results: list[tuple[int, float, int, bool, dict]] = []
    for link in _LINK_SWEEP:
        b_len = assembled["b_len"]
        eff_link = min(link, b_len)
        _log(
            f"step4[link={link}]: building IN-PROCESS sparse engine "
            f"(epic_link_tokens={eff_link}, B_len={b_len}, "
            f"reuse_fraction={(b_len - eff_link) / b_len:.2f})"
        )
        _reset_epic_counters()
        with _in_process_engine():
            llm_sparse = _build_llm(
                model,
                kv_config=_epic_kv_config(
                    sparse=True,
                    fusion=True,
                    link_tokens=eff_link,
                    debug_check_load=True,   # Implement 2: scatter self-check.
                    debug_counters=True,     # in-band engagement assertion.
                ),
                enforce_eager=True,
                attention_backend="FLEX_ATTENTION",
            )
            llm_sparse.generate([warm_prompt], params)  # warm: saves the chunk
            sparse_outs = llm_sparse.generate([reuse_prompt], params)
            sparse_out = _token_ids(sparse_outs)[0]
            sparse_text = _texts(sparse_outs)[0]
            counters = _read_epic_counters()
            del llm_sparse

        rate = _match_rate(dense_out, sparse_out)
        div = _first_divergence(dense_out, sparse_out)
        cos = _decode_cosine(model, dense_out, sparse_out)
        hit = _output_has_answer(sparse_text, target.answer)
        results.append((link, rate, div, hit, counters))

        _log(f"step4[link={link}]: sparse text: {sparse_text!r}")
        _log(
            f"step4[link={link}]: needle_hit={hit} (code {target.answer!r}) "
            f"token-match rate sparse-vs-dense = {rate:.3f}"
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
    _log("step4: LINK SWEEP SUMMARY")
    _log(
        "    link | needle | match_rate | first_div | "
        "match/emit/loaded | regime"
    )
    for link, rate, div, hit, counters in results:
        full = "B-FULL (decisive)" if link >= assembled["b_len"] else "approx"
        cstr = (
            f"{counters.get('sparse_match', 0)}/"
            f"{counters.get('sparse_emit', 0)}/"
            f"{counters.get('chunks_loaded', 0)}"
        )
        _log(
            f"    {link:4d} | {('HIT' if hit else 'miss'):>6s} | {rate:10.3f} | "
            f"{('none' if div < 0 else str(div)):>9s} | {cstr:^24s} | {full}"
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
    _, decisive_rate, decisive_div, decisive_hit, _ = decisive
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
    _log(
        f"step4: DECISIVE OK (link={decisive[0]} M==all-of-B needle HIT, match "
        f"{decisive_rate:.3f} >= {_MACHINERY_PASS_THRESHOLD}). Smaller-link rows "
        "above are the reuse APPROXIMATION regime: a needle miss / low match "
        "there is the 1st-order approximation-quality signal (reported, not a "
        "machinery failure)."
    )
    _log(
        "step4: PASS (probe valid: dense answered + discriminative; sparse "
        "engaged in-band; decisive B-full control hit the needle and matched "
        "dense; approximation sweep reported)."
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
}


def _parse_steps(arg: str | None) -> list[int]:
    if not arg:
        return [1, 2, 3, 4]
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="EPIC GPU smoke test")
    ap.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model id (small Llama-family recommended).",
    )
    ap.add_argument(
        "--steps",
        default=None,
        help="Comma-separated step numbers to run (default: 1,2,3,4).",
    )
    args = ap.parse_args()

    if not _has_cuda():
        _log(
            "No CUDA GPU detected. This script requires a GPU + a built EPIC "
            "vLLM. See vllm/.../epic/PHASE2.md for the build command. Exiting "
            "without running (structure-only check is a no-op here)."
        )
        return 0

    steps = _parse_steps(args.steps)
    _log(f"model={args.model} steps={steps}")
    for s in steps:
        fn = _STEPS.get(s)
        if fn is None:
            _fail("main", f"unknown step {s}; valid: {sorted(_STEPS)}")
        fn(args.model)
    _log(f"ALL REQUESTED STEPS PASSED: {steps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
