# Non-contiguous KV reuse — common strategy interface

Status: design + skeleton. EPIC is the first concrete implementation; CacheBlend
is the validation target for the abstraction. Heavy algorithm bodies are left to
the migrator; this document fixes the **interface seams** and where each seam is
called in the V1 scheduler/worker lifecycle.

## 0. Why a strategy layer (and why a thin one)

`KVConnectorBase_V1` already gives us the *transport + lifecycle* contract
(match → alloc → build meta → load → forward → save → finish). It does **not**
give us the *algorithmic* decisions that distinguish EPIC from CacheBlend.
Profiling the two algorithms against the migration brief, only **four** decisions
actually diverge:

| # | Decision | EPIC | CacheBlend |
|---|----------|------|------------|
| 1 | selection — which content chunks to reuse | content-hash match | content-hash match |
| 2 | alignment — fix position mismatch of reused K | PIC delta re-rotary | none (recompute absorbs it) |
| 3 | recompute policy — which M tokens to forward | LegoLink: new + link tokens, static | check-layer V-deviation top-k, dynamic |
| 4 | fusion attention — mask for M queries over full KV | partial causal+link mask | partial mask over recomputed set |

Selection (1) is effectively **common** (both are content-hash chunk matching —
already implemented in `EpicChunkStore` + `get_num_new_matched_tokens`). The
genuine variability is in 2/3/4. So the strategy layer is three small interfaces
plus a thin selection hook, all sitting on top of the *unchanged*
`KVConnectorBase_V1` contract. The connector owns transport and lifecycle; the
strategies own algorithm.

The base connector (`ReuseConnectorBase`, future refactor target) holds:
the chunk store, the scheduler-side match bookkeeping, the worker-side
scatter/gather of KV into paged blocks, and the lifecycle plumbing. EPIC and
CacheBlend subclasses differ **only** by which strategy objects they construct.

## 1. The "M is fixed before forward entry" first-class constraint

This is the load-bearing invariant of the whole design and it is dictated by V1:

* V1 builds `positions`, `slot_mapping`, `query_start_loc`, and the attention
  metadata **once**, before the model forward, from a per-request scalar
  `num_scheduled_tokens`. Nothing downstream of `gpu_model_runner.execute_model`
  can change the number of query rows mid-forward without invalidating CUDA
  graphs and every attention backend's metadata.
* EPIC's original V0 code *did* change shapes mid-forward (it sliced
  `hidden_states`/`positions` down to `imp_indices` at the check layer — the
  `status==1 → status==2` transition). That is exactly the "global dict 관통"
  the migration forbids and it is structurally incompatible with V1.

Therefore the interface makes **M (the set of token rows actually forwarded) a
quantity that is fully determined before the model runs.** `RecomputePolicy`
returns M as scheduler-time (or pre-pass-time) output, never as a mid-forward
mutation.

* **Static policies (EPIC LegoLink)** compute M directly at schedule time from
  chunk boundaries + new tokens. One pass, no surprises. This is the supported
  path for Phase 2.
* **Dynamic policies (CacheBlend)** need attention statistics from a "check
  layer" to score importance. We do *not* let them resize the forward. Instead
  the interface exposes a **two-phase contract**: an optional cheap *importance
  pre-pass* produces a score vector, the policy converts scores → a fixed M, and
  only then the real forward runs with M frozen. The pre-pass is modeled as its
  own (small) forward whose M is itself fixed (e.g. one check layer over the
  reuse candidate set). See §4.3. This keeps the "M fixed before forward" rule
  true for both, at the cost of one extra short pass for dynamic policies.

## 2. Interface definitions and call sites

All four live in `reuse_strategy.py`. Signatures are typed against existing V1
objects (`Request`, `SchedulerOutput`, `ForwardContext`, `FlexAttentionMetadata`)
so they can only be filled with real hooks.

### 2.1 `SelectionStrategy` — scheduler side

```
select(request, num_computed_tokens, store) -> ReuseSelection
```
* **Called from:** `connector.get_num_new_matched_tokens(...)` (scheduler).
* **Returns:** prefix-extent chunks (loadable now) + non-prefix hits (Phase 2
  candidates). For Phase 1 the base connector already does this inline; the
  strategy just formalizes it so a future selection variant (e.g. fuzzy/n-gram
  match for CacheBlend) can replace the exact-hash walk without touching the
  connector.
* **Must be side-effect free** (the base docstring says `get_num_new_matched_tokens`
  may be called multiple times per request).

### 2.2 `AlignmentStrategy` — worker side, load time

```
align_keys(key, old_positions, new_positions, layer_name) -> key
```
* **Called from:** the base connector's chunk-load path (`_load_chunk`, inside
  `start_load_kv`), once per layer per chunk, *before* the K is scattered into
  the paged cache.
* **EPIC:** wraps `PICRotator.rotate_keys` (delta re-rotary). **CacheBlend:**
  identity (it does not re-rotate; mismatch is absorbed by recompute). Making
  this a strategy means the load path is shared and only the rotation differs.
* This is exactly where Phase 1 already calls `self._pic.rotate_keys(...)`; the
  refactor only moves that call behind the interface.

### 2.3 `RecomputePolicy` — scheduler side (static) + optional pre-pass (dynamic)

```
plan_recompute(request, selection, block_size) -> RecomputePlan   # static, schedule time
needs_importance_prepass() -> bool
score_to_plan(request, selection, scores) -> RecomputePlan        # dynamic, after pre-pass
```
* **`plan_recompute` called from:** `connector.build_connector_meta(...)`
  (scheduler), where the runner-facing metadata is assembled. Produces the **M
  set**: the token offsets that will actually be forwarded, plus per-token target
  positions for those M tokens (so the runner can build sparse positions — see
  §5). This is the field Phase 2 carries on `EpicConnectorMetadata`.
* **`needs_importance_prepass`** lets a dynamic policy opt into the two-phase
  contract. EPIC returns `False`; CacheBlend returns `True`.
* **`score_to_plan`** is the dynamic completion: given importance scores from the
  pre-pass, freeze M. Called by the runner-side pre-pass driver (Phase 2 runner
  patch), still **before** the main forward — preserving §1.

### 2.4 `FusionMaskBuilder` — worker side, attention time

```
build_logical_mask_mod(plan, attn_meta) -> Callable[[b,h,q_idx,kv_idx], BoolTensor]
build_block_sparsity_hint(plan, attn_meta) -> BlockSparsityHint | None
```
* **Called from:** Phase 2 attention wiring, where `FlexAttentionMetadata.
  logical_mask_mod` is set (default `causal_mask_mod`). The builder returns a
  `mask_mod` matching FlexAttention's real `(b, h, q_idx, kv_idx) -> bool`
  signature.
* **Critical constraint (brief decision e):** the returned closure must **not**
  capture Python scalars/lists that vary per step (that re-triggers
  `torch.compile`). It must read all per-request reuse state from **metadata
  tensors** registered on the attention metadata and indexed by `q_idx`/`kv_idx`.
  The interface enforces this by passing `plan` as a struct of tensors and
  requiring the mask to be expressed as tensor lookups (`recompute_flag[q_idx]`,
  `reused_flag[kv_idx]`, `link_window[...]`), never as captured constants.
* **EPIC:** "M queries attend causally over all preceding KV; reused (non-M) KV
  rows are visible but never produce queries; link tokens get the LegoLink
  window." **CacheBlend:** "recomputed tokens attend over the full reused KV;
  non-recomputed reused rows are read-only." Both are the same tensor-lookup
  shape; only the predicate differs → a clean strategy boundary.

## 3. Lifecycle map (where each seam fires)

```
SCHEDULER process
  on_new_request
  get_num_new_matched_tokens   ── SelectionStrategy.select          (§2.1)
  update_state_after_alloc
  build_connector_meta         ── RecomputePolicy.plan_recompute    (§2.3 static)
                                  (emits M set + sparse positions + mask plan
                                   into EpicConnectorMetadata)
        │  meta pickled →
WORKER process
  bind_connector_metadata
  [Phase 2] importance pre-pass ─ RecomputePolicy.score_to_plan     (§2.3 dynamic)
  start_load_kv                ── AlignmentStrategy.align_keys       (§2.2)
                                  then scatter into paged blocks
  (model forward)
     attention layer           ── FusionMaskBuilder.build_*         (§2.4)
     save_kv_layer                (harvest chunks → store)
  wait_for_save
  clear_connector_metadata
  request_finished / get_finished
```

Selection and recompute(static) are **scheduler-side**; alignment, fusion mask,
and recompute(dynamic, via pre-pass) are **worker-side**. This split matters: the
scheduler must know M *before* it allocates and schedules, so static M is
schedule-time; dynamic scoring needs the GPU so it is worker-time but still
pre-forward.

## 4. EPIC / CacheBlend mapping table

| Strategy point | Base (shared) | EPIC fills | CacheBlend fills |
|---|---|---|---|
| Selection | content-hash chunk store + prefix/non-prefix walk | exact-hash, chunk=block-multiple | exact-hash (could add fuzzy later) |
| Alignment | per-layer load + scatter into paged blocks | `PICRotator` delta re-rotary | identity (no re-rotary) |
| Recompute policy | M frozen before forward; carries M + sparse positions | LegoLink: M = new tokens ∪ chunk-boundary link tokens; **static**, `needs_prepass=False` | check-layer V-deviation top-k (`recomp_ratio`); **dynamic**, `needs_prepass=True`, importance pre-pass then `score_to_plan` |
| Fusion attention | sets `FlexAttentionMetadata.logical_mask_mod` from a tensor-lookup mask_mod | LegoLink causal + per-chunk link window, all via metadata tensors | recomputed-token mask over reused KV, via metadata tensors |

### 4.1 What EPIC LegoLink makes M
`M = {all genuinely new tokens (no matched chunk)} ∪ {link tokens at each reused
chunk boundary}`. Fully determined from the selection result and chunk size at
schedule time. Number of forwarded rows = `len(M)` ≪ prompt length → real compute
savings (the reason full-forward+token_mask was rejected, decision d).

### 4.2 What CacheBlend makes M
M depends on V-deviation measured at a check layer, unknown at schedule time.
Handled by the two-phase contract (§1): pre-pass over the candidate set →
scores → `score_to_plan` freezes top-k by `recomp_ratio` → main forward with M
fixed. The connector never resizes the main forward.

### 4.3 The importance pre-pass (dynamic only)
Modeled as a *separate, small forward* whose own M is fixed (the candidate set
restricted to one check layer, emitting scores via FlexAttention `score_mod`
capture or a KV-norm heuristic — see PHASE2 §4). Output: a score tensor. This is
the only place a dynamic policy touches the GPU before M is frozen, and it is
itself shape-stable. EPIC skips it entirely.

## 5. Interface boundary vs. Phase 2 runner/scheduler patches

Hard line between *connector/strategy* (this directory) and *core V1 patches*
(invasive, tracked in PHASE2.md):

| Concern | Owner | Mechanism |
|---|---|---|
| Which chunks, which M, target positions, mask plan | **connector + strategies** | computed here, emitted on `EpicConnectorMetadata` |
| Loading/aligning/scattering reused KV | **connector + AlignmentStrategy** | `start_load_kv` |
| Telling the scheduler "fewer than prompt-len tokens to forward" | **core patch** | scheduler must accept a non-scalar computed set OR a reduced `num_scheduled_tokens` (PHASE2 §1). Connector *provides* the number; scheduler *consumes* it. |
| Per-token sparse `positions` for the forward | **core patch** | runner reads explicit positions from `EpicConnectorMetadata` instead of `computed_prefix + arange` (PHASE2 §2, `gpu_model_runner.py:~1896`). Connector *provides* the vector; runner *consumes* it. |
| Selecting + wiring a custom-mask backend | **core/config patch** | force FlexAttention when reuse-sparse is on; set `logical_mask_mod` from the builder (PHASE2 §3). Connector *provides* the mask_mod; runner/backend *installs* it. |

Rule of thumb: **the connector and strategies are pure producers of metadata.**
Every place that *consumes* that metadata to change V1's forward shape/positions/
mask is a core patch enumerated in PHASE2.md. The strategy interface is designed
so those consumer patches are as small and mechanical as possible (read a tensor
off metadata; no algorithm in the runner).

## 6. Open risks / must-verify

Legend: **[OPEN]** unresolved · **[MITIGATED]** addressed in code, GPU-unverified ·
**[RESOLVED]** addressed + CPU-validated. Phase 2b batches B1-B4 are CPU-validated
only; numerics remain GPU-unverified (PHASE2.md "GPU verification procedure").

1. **[MITIGATED] mask_mod recompile.** The "tensor-lookup, no closure capture"
   rule is implemented: `LegoLinkMaskBuilder` hands back a single stable mask_mod
   object with fixed-capacity backing tensors (`_install_fusion_mask`,
   `_mask_capacity = max_model_len`), so FlexAttention's identity check fires once.
   Still must be confirmed empirically on GPU (no recompile storms) — PHASE2.md
   step 4c. The S7 single-batch gate + enforce_eager requirement keep this
   tractable (one request, no graph capture) until verified.
2. **[OPEN] Pre-pass cost for CacheBlend.** Unchanged: the two-phase contract adds
   a forward. Interface-only; no CacheBlend implementation yet. Needs measurement;
   may push toward a KV-norm heuristic that avoids the pre-pass.
3. **[RESOLVED] Scheduler scalar `num_computed_tokens`.** Addressed by B1+B2.
   `get_num_new_matched_tokens` reports external=\|A\|+\|B\| and the scheduler-core
   patch (`_apply_epic_sparse_overrides`) rewrites `num_scheduled_tokens`→\|M\| and
   advances `num_computed` by N−external. The connector stays a pure metadata
   producer; the scheduler consumes \|M\| and the advance. CPU-validated in
   `test_scheduler_core_patch.py`. (Inert when `epic_sparse_forward` is off.)
3b. **[MITIGATED] positions int32/int64 + CUDA-graph capture.** B3 writes explicit
   sparse RoPE positions into the runner's positions buffer. The dtype/buffer match
   is CPU-validated (`test_runner_sparse.py`); CUDA-graph capture is sidestepped for
   now by the S7 **enforce_eager** requirement (no capture under eager). Relaxing to
   PIECEWISE and re-validating capture is the remaining TODO.
4. **[OPEN] Alignment identity for CacheBlend correctness.** Unchanged: EPIC uses
   PIC re-rotary; CacheBlend's identity-alignment assumption must be confirmed
   against the CacheBlend algorithm. No CacheBlend implementation yet.
5. **[RESOLVED] Save/harvest under sparse forward.** Addressed by the B1 S2 save
   guard in `build_connector_meta`: a chunk is harvested only when *every* token in
   it is in M (genuinely recomputed); aligned-but-not-recomputed (rotated) KV is
   never re-saved as canonical. CPU-validated in the sparse connector tests. (No-op
   when `m_set is None`, i.e. non-sparse.)
6. **[OPEN] Mixed batches (S7).** The single-batch gate isolates each sparse
   request into its own step (shared fusion-mask tensor + \|M\| layout not validated
   for mixed batches). Conservative but correct; a validated multi-request sparse
   layout is a future optimization (PHASE2.md TODOs).
</content>
