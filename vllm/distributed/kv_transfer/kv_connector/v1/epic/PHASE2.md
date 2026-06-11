# EPIC Phase 2 — Non-prefix (sparse) reuse hook points

Phase 1 (this directory) only reuses content-matched chunks that fall in the
**contiguous prefix** of a new request, because V1's scheduler and runner both
assume a single contiguous computed prefix. Phase 2 lifts that restriction to
reuse chunks scattered anywhere in the prompt (true EPIC / CacheBlend behavior).
It is **invasive** and is intentionally out of scope for Phase 1.

The lower half of this file (§1-§4) records the exact code anchors a Phase 2
implementation must touch, plus the EPIC original behavior each one replaces.
The upper half (this **Status** section) tracks what is now implemented.

---

## Status (Phase 2b) — CPU-validated, GPU-unverified

Phase 2b (sparse / non-contiguous forward) is wired end-to-end and CPU-validated.
It is **gated OFF by default** behind the `epic_sparse_forward` connector
`kv_connector_extra_config` flag. With the flag off, every Phase 2b code path is
inert and Phase 1/2a behavior (dense forward, prefix-only reuse) is byte-for-byte
unchanged.

Implemented in batches (all in `vllm/`):

| Batch | Scope | Where |
|---|---|---|
| B1 | connector sparse plan: derive M (LegoLink), emit `EpicReqSparse` on metadata; sparse match path reports external=\|A\|+\|B\| | `epic_connector.py` (`_emit_sparse`, `get_num_new_matched_tokens`), `metadata.py` (`EpicReqSparse`) |
| B2 | scheduler accounting: rewrite `num_scheduled_tokens`→\|M\|, advance `num_computed` by N−external, `delay_cache_blocks` for sparse reqs | `scheduler.py` (`_apply_epic_sparse_overrides`, `_update_after_schedule`, waiting-loop `delay_cache_blocks`) |
| B3 | runner: explicit per-token sparse RoPE positions + seq_len override | `gpu_model_runner.py` (sparse positions / seq_lens patches) |
| B4 | **S7 safety gating** (this batch): single-batch isolation + FlexAttention/eager validation | `scheduler.py` (waiting-loop gate), `epic_connector.py` (`_validate_sparse_safety`) |

### S7 — safety gating (Batch 4)

Three guards, all active only when `epic_sparse_forward` is on:

1. **FlexAttention + eager config validation** — `EpicConnector._validate_sparse_safety`
   (called from `__init__` only when sparse is on). Sparse forward installs a
   LegoLink `logical_mask_mod` that only the **FlexAttention** V1 backend
   consumes; FlashAttention (the V1 default) would silently ignore it and return
   wrong results. So the connector *validates* (does not silently mutate) that
   `vllm_config.attention_config.backend.name == "FLEX_ATTENTION"`; otherwise it
   raises a `ValueError` instructing the user to pass
   `--attention-backend FLEX_ATTENTION` (serving) or
   `attention_backend="FLEX_ATTENTION"` (offline LLM/EngineArgs). The legacy
   `VLLM_ATTENTION_BACKEND` env var was removed in vLLM v0.22 and is silently
   ignored. The connector deliberately does **not** override
   backend auto-selection — flipping vLLM's platform-level backend choice from a
   KV connector would be far more invasive and fragile than a config assertion.
   The same method also requires `enforce_eager` (first-pass safety; see #3).

2. **Single-batch isolation** — `scheduler.schedule()` waiting loop. A sparse
   request must be the only request in its step: the per-step LegoLink fusion-mask
   is a single shared tensor and the reduced (\|M\|) forward-row layout is only
   validated for a singleton batch. The gate (a) **defers** a sparse request that
   would otherwise join a non-empty batch (running reqs, encoder prefills, or
   earlier waiting reqs) by re-queuing it to a later step, and (b) **breaks** the
   waiting loop right after a sparse request is scheduled so nothing joins behind
   it. Both halves read the connector's duck-typed `is_sparse_request(req_id)`
   hook and are inert (`epic_sparse` stays False) on every non-sparse step.

3. **enforce_eager required** — validated in #1. The \|M\| sparse forward and the
   per-step mask tensor have not been validated under CUDA graph / piecewise
   compile capture, so eager is the first-pass safe mode. Relaxing to PIECEWISE is
   a post-GPU-verification TODO.

### CPU tests (regression-clean)

```
/tmp/epic-test-venv/bin/python -m pytest \
    tests/v1/kv_connector/unit/epic/ \
    tests/v1/core/test_scheduler.py
```

* `tests/v1/kv_connector/unit/epic/` — 75 passed (65 prior + 10 S7:
  `test_sparse_batch_gate.py` ×5, `test_sparse_safety_validation.py` ×5).
* `tests/v1/core/test_scheduler.py` — 97 passed (no regression).

### GPU verification procedure (REQUIRED before claiming Phase 2b works)

Everything above is CPU logic/plumbing validation. The actual sparse forward
(numerics, FlexAttention mask correctness, PIC re-rotary alignment, recompile
behavior) has **not** run on a GPU. To verify, on a CUDA box with EPIC installed:

```bash
# 0. Build (heavy; CUDA compile). Do NOT run on the CPU dev box.
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# NOTE: VLLM_ATTENTION_BACKEND was removed in vLLM v0.22 (silently ignored).
#       Select the backend with the --attention-backend CLI arg instead.

# 1. Sanity: sparse OFF must be identical to baseline (no-trace check).
python -m vllm.entrypoints.openai.api_server \
    --model <model> --enforce-eager --attention-backend FLEX_ATTENTION \
    --kv-transfer-config '{"kv_connector":"EpicConnector","kv_role":"kv_both"}'
#    -> behaves exactly like no connector for outputs (only saves chunks).

# 2. Safety gate: sparse ON but WRONG backend must fail fast.
python -m vllm.entrypoints.openai.api_server \
    --model <model> --enforce-eager --attention-backend FLASH_ATTN \
    --kv-transfer-config '{"kv_connector":"EpicConnector","kv_role":"kv_both","kv_connector_extra_config":{"epic_sparse_forward":true}}'
#    -> EXPECT a ValueError naming --attention-backend FLEX_ATTENTION.

# 3. Safety gate: sparse ON, FlexAttention, but NOT eager must fail fast.
python -m vllm.entrypoints.openai.api_server \
    --model <model> --attention-backend FLEX_ATTENTION \
    --kv-transfer-config '{"kv_connector":"EpicConnector","kv_role":"kv_both","kv_connector_extra_config":{"epic_sparse_forward":true}}'
#    -> EXPECT a ValueError about enforce_eager.

# 4. Real sparse run: FlexAttention + eager + sparse ON.
python -m vllm.entrypoints.openai.api_server --attention-backend FLEX_ATTENTION \
    --model <model> --enforce-eager \
    --kv-transfer-config '{"kv_connector":"EpicConnector","kv_role":"kv_both","kv_connector_extra_config":{"epic_sparse_forward":true,"epic_fusion_mask":true}}'
#    Drive two requests that share a NON-prefix chunk (e.g. RAG: same retrieved
#    passage in the MIDDLE of two different prompts). Verify:
#      a. the 2nd request's step schedules ONLY that request (single-batch gate),
#      b. output quality matches a dense (sparse-off) run within tolerance,
#      c. no FlexAttention recompile storms across steps (stable mask_mod identity).
```

Run order matters: 1 (no-trace) → 2,3 (gate fails closed) → 4 (numerics). Stop and
report at the first failure rather than proceeding.

### Remaining TODOs / risks (post-GPU)

* **PIECEWISE relaxation.** enforce_eager is forced; validate the \|M\| forward +
  mask tensor under `cudagraph_mode=PIECEWISE` and relax the eager requirement in
  `_validate_sparse_safety` once stable.
* **Mixed batches.** The single-batch gate is conservative (one sparse req per
  step). A validated multi-request sparse layout (per-request mask offsets, no
  shared-tensor collision) would lift the isolation cost.
* **2-D mask.** The current LegoLink mask is the dense/causal-equivalent shape in
  Phase 2a wiring; the genuine sparse 2-D recompute mask must be exercised and its
  recompile behavior confirmed on GPU (DESIGN §6.1).
* **CacheBlend prepass.** The dynamic importance pre-pass (DESIGN §4.3) is
  interface-only; the second strategy implementation (CacheBlend) and its check-
  layer scoring are not yet built.

---

## Code anchors (reference)

This section records the exact code anchors a Phase 2 implementation must touch,
plus the EPIC original behavior each one replaces.

## 1. Scheduler: contiguous `num_computed_tokens` is a scalar

`vllm/v1/request.py` (`Request.__init__`):

```python
self.num_computed_tokens = 0   # scalar prefix length
```

`get_num_new_matched_tokens(request, num_computed_tokens)` in
`KVConnectorBase_V1` can only return a single prefix length (`base.py` docstring:
"largest prefix of prompt-tokens"). There is no way to tell the scheduler "tokens
[0:256) and [512:768) are precomputed but [256:512) is not."

**Phase 2 needs:** a way to express a *mask* of computed positions (not a scalar)
through the scheduler so it does not schedule full-forward over already-reusable
non-prefix tokens. Either:
  - extend the connector return / `update_state_after_alloc` to carry a computed
    bitmap per request, and have the scheduler subtract those token counts, or
  - keep scheduling the full forward but mark non-prefix-reused tokens so the
    model runner skips/short-circuits their QKV projection + attention writes.

EPIC original equivalent: it bypassed the scheduler entirely and drove fusion
from `cache_fuse_metadata["kvlink"]` inside the model forward
(`vllm_epic/.../models/llama.py`), slicing `positions` / `hidden_states` down to
`imp_indices`. That is the "global dict 관통" we are avoiding.

## 2. Runner: positions are computed as a contiguous range

`vllm/v1/worker/gpu_model_runner.py` (~line 1896):

```python
positions_np = (
    self.input_batch.num_computed_tokens_cpu[req_indices]
    + self.query_pos.np[: cu_num_tokens[-1]]
)
```

Positions are `computed_prefix + arange(num_scheduled)`. A sparse-reuse request
whose reused tokens are non-contiguous cannot get correct RoPE positions from
this. EPIC handled it by re-deriving `org_pos` and re-rotating in the model.

**Phase 2 needs:** the runner to accept an explicit per-token position vector for
requests flagged by the EPIC connector (carry it via `EpicConnectorMetadata` and
read it in the runner's position-builder), rather than assuming a contiguous
range. The `PICRotator` here already produces correctly re-rotated K for any
target positions, so the load path is reusable as-is; only the *query/forward*
positions need the runner change.

## 3. Attention: partial / custom-mask fusion attention

EPIC's selective recompute (`recomp_ratio` ~0.16, `imp_indices`) and partial
attention bias lived in `vllm_epic/vllm/attention/backends/xformers.py`. V1's
default backend is FlashAttention (`vllm/v1/attention/backends/flash_attn.py`),
which does **not** accept an arbitrary additive bias / custom mask.

**Phase 2 needs:** a backend that supports a custom attention mask so the small
set of "important" recomputed tokens can attend over the reused KV while the
reused tokens themselves are not recomputed. Candidates in this tree:
  - `vllm/v1/attention/backends/flex_attention.py` (PyTorch FlexAttention —
    supports `mask_mod` / `score_mod`, the natural fit for partial masks).
  - `vllm/v1/attention/backends/flashinfer.py` (custom mask support).
Select the backend at config time when EPIC sparse mode is on, and pass the
importance mask / recompute indices through `EpicConnectorMetadata` →
`forward_context`.

## 4. Selective recompute (importance scoring)

EPIC measured token importance from attention probabilities at a "check layer"
(`status == 1`, `check_layers=[1]`) to pick `imp_indices`. V1 FlashAttention does
not expose attention probabilities. Phase 2 must either:
  - run a cheap importance pass on the check layer with a prob-returning backend
    (FlexAttention can emit scores), or
  - approximate importance with a KV-norm / position heuristic.

The `recomp_ratio` and check-layer selection should be carried as structured
fields on `EpicConnectorMetadata` (Phase 1 already records `NonPrefixHit`s, which
are the candidate set for recompute scoring).

## Summary of what Phase 1 already provides for Phase 2

- `PICRotator.rotate_keys` works for arbitrary (non-prefix, non-contiguous,
  negative-delta) target positions — directly reusable for sparse loads.
- `EpicChunkStore` is position-independent (content hash, not prefix chain), so
  middle-of-prompt matches are already discoverable.
- `get_num_new_matched_tokens` already records non-prefix hits
  (`metadata.NonPrefixHit`) without loading them.
