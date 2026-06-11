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
  step4  two requests sharing a NON-prefix chunk: request 1 prefills + saves the
         chunk, request 2 (A+C+B layout, shared chunk in the MIDDLE) runs with
         sparse ON; compare its output to the dense (sparse OFF) run and print
         token-match rate + a hint to inspect the schedule log for single-batch
         isolation.

Usage:
    # On a CUDA box, after `VLLM_USE_PRECOMPILED=1 uv pip install -e .`:
    VLLM_ATTENTION_BACKEND=FLEX_ATTENTION \
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


def _epic_kv_config(*, sparse: bool, fusion: bool = False) -> dict:
    extra: dict = {}
    if sparse:
        extra["epic_sparse_forward"] = True
    if fusion:
        extra["epic_fusion_mask"] = True
    cfg: dict = {"kv_connector": "EpicConnector", "kv_role": "kv_both"}
    if extra:
        cfg["kv_connector_extra_config"] = extra
    return cfg


def _build_llm(model: str, *, kv_config: dict | None, enforce_eager: bool = True):
    from vllm import LLM

    kwargs: dict = dict(
        model=model,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.45,
        max_model_len=2048,
    )
    if kv_config is not None:
        kwargs["kv_transfer_config"] = kv_config
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


# ---------------------------------------------------------------------------
# step2 -- sparse ON + wrong backend must fail fast
# ---------------------------------------------------------------------------
def step2_wrong_backend_fails(model: str) -> None:
    _log("step2: sparse ON + non-FlexAttention backend must raise ValueError")
    prev = os.environ.get("VLLM_ATTENTION_BACKEND")
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    try:
        _build_llm(
            model, kv_config=_epic_kv_config(sparse=True), enforce_eager=True
        )
    except ValueError as e:
        if "FLEX_ATTENTION" not in str(e):
            _fail(
                "step2",
                f"raised ValueError but it does not name FLEX_ATTENTION: {e}",
            )
        _log(f"step2: PASS (raised ValueError as expected: {str(e)[:120]}...)")
        return
    except Exception as e:  # noqa: BLE001
        _fail("step2", f"expected ValueError, got {type(e).__name__}: {e}")
    finally:
        if prev is None:
            os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        else:
            os.environ["VLLM_ATTENTION_BACKEND"] = prev
    _fail("step2", "sparse ON with FLASH_ATTN did NOT raise (safety gate broken)")


# ---------------------------------------------------------------------------
# step3 -- sparse ON + FlexAttention but NOT eager must fail fast
# ---------------------------------------------------------------------------
def step3_non_eager_fails(model: str) -> None:
    _log("step3: sparse ON + FlexAttention + NOT eager must raise ValueError")
    prev = os.environ.get("VLLM_ATTENTION_BACKEND")
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"
    try:
        _build_llm(
            model, kv_config=_epic_kv_config(sparse=True), enforce_eager=False
        )
    except ValueError as e:
        if "enforce_eager" not in str(e):
            _fail(
                "step3",
                f"raised ValueError but it does not mention enforce_eager: {e}",
            )
        _log(f"step3: PASS (raised ValueError as expected: {str(e)[:120]}...)")
        return
    except Exception as e:  # noqa: BLE001
        _fail("step3", f"expected ValueError, got {type(e).__name__}: {e}")
    finally:
        if prev is None:
            os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        else:
            os.environ["VLLM_ATTENTION_BACKEND"] = prev
    _fail(
        "step3",
        "sparse ON + FlexAttention but non-eager did NOT raise (safety gate broken)",
    )


# ---------------------------------------------------------------------------
# step4 -- shared non-prefix chunk: sparse-on vs dense output comparison
# ---------------------------------------------------------------------------
def step4_shared_chunk_sparse_vs_dense(model: str) -> None:
    _log("step4: shared NON-prefix chunk, sparse ON output vs dense output")
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"

    # Request 1: a prompt whose MIDDLE is the shared passage -> EPIC saves the
    # passage chunk(s). Request 2: a DIFFERENT prompt whose middle is the SAME
    # passage (A=head, C=tail differs, B=shared passage in the middle) -> the
    # passage is a non-prefix content match for request 2.
    warm_prompt = _PROMPT_HEAD + _SHARED_PASSAGE + _PROMPT_TAIL_1
    reuse_prompt = (
        "Different opening sentence here for the second request entirely. "
        + _SHARED_PASSAGE
        + _PROMPT_TAIL_2
    )
    params = _greedy_params(max_tokens=48)

    # --- dense reference: sparse OFF (connector still saves, but dense forward) ---
    llm_dense = _build_llm(model, kv_config=_epic_kv_config(sparse=False))
    llm_dense.generate([warm_prompt], params)  # warm: saves the shared chunk
    dense_out = _token_ids(llm_dense.generate([reuse_prompt], params))[0]
    del llm_dense

    # --- sparse run: sparse ON + fusion mask, FlexAttention + eager ---
    llm_sparse = _build_llm(
        model,
        kv_config=_epic_kv_config(sparse=True, fusion=True),
        enforce_eager=True,
    )
    llm_sparse.generate([warm_prompt], params)  # warm: saves the shared chunk
    sparse_out = _token_ids(llm_sparse.generate([reuse_prompt], params))[0]
    del llm_sparse

    rate = _match_rate(dense_out, sparse_out)
    cos = _decode_cosine(model, dense_out, sparse_out)
    _log(f"step4: token-match rate sparse-vs-dense = {rate:.3f}")
    if cos is not None:
        _log(f"step4: embedding cosine similarity = {cos:.4f}")
    _log(
        "step4: HINT -- inspect the engine schedule log for the reuse request's "
        "step; under the S7 single-batch gate it must be scheduled ALONE "
        "(no other request in the same step)."
    )
    # We do NOT hard-assert exact equality (sparse reuse is an approximation);
    # a very low match rate signals a real numeric break.
    if rate < 0.5:
        _fail(
            "step4",
            f"sparse output diverges badly from dense (match {rate:.3f} < 0.5); "
            "likely a mask / PIC / position bug, not just approximation error.",
        )
    _log("step4: PASS (sparse output within tolerance of dense)")


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
