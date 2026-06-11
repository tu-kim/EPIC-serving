# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC KV connector (Phase 1, non-invasive).

Implements position-independent chunk reuse on top of ``KVConnectorBase_V1``,
restricted to the *prefix extent* of new requests (so it stays compatible with
V1's contiguous ``num_computed_tokens`` scheduler and contiguous runner
positions). Non-prefix content matches are tracked only (Phase 2).

Mapping from EPIC original (vllm_epic, vLLM 0.7 / V0) to here (V1):

  * EPIC `cache_fuse_metadata` dict on the model  -> `EpicConnectorMetadata`
    (scheduler-built, bound via forward_context).
  * EPIC `self.hack_kv` dense KV on the model      -> `EpicChunkStore` (CPU).
  * EPIC `rotary_emb(org_pos, fake_q, old_kv[0])`   -> `PICRotator.rotate_keys`
    (delta rotation on already-rotated K; no fake_q hack).
  * EPIC out-of-band chunk match in benchmark       -> content-hash lookup in
    `get_num_new_matched_tokens` reported to the V1 scheduler as a prefix length.

NOT done in Phase 1 (see epic/PHASE2.md): selective recompute (recomp_ratio /
imp_indices), partial-mask fusion attention, sparse (non-prefix) forward.
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    EpicChunkStore,
    StoredChunk,
    hash_chunk_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
    ChunkLoadSpec,
    EpicConnectorMetadata,
    EpicReqLoad,
    EpicReqSave,
    EpicReqSparse,
    FusionMaskPlan,
    NonPrefixHit,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.pic import PICRotator
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    AlignmentStrategy,
    EpicSelection,
    LegoLinkMaskBuilder,
    LegoLinkRecompute,
    PicAlignment,
    RecomputePlan,
    RecomputePolicy,
    ReuseSelection,
    SelectionStrategy,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# 8 GB default CPU budget for cached chunks.
DEFAULT_CAPACITY_BYTES = 8 * (1024**3)
DEFAULT_CHUNK_SIZE = 256


class _PromptLenShim:
    """Minimal stand-in exposing ``prompt_token_ids`` so a RecomputePolicy can
    read the exact prompt length N without the V1 ``Request`` (unavailable in
    ``build_connector_meta``; only the new-req descriptor's token ids are)."""

    __slots__ = ("prompt_token_ids",)

    def __init__(self, token_ids: list[int]):
        self.prompt_token_ids = token_ids


def _slot_ids_from_blocks(
    block_ids: list[int], block_size: int, start_token: int, num_tokens: int
) -> list[int]:
    """Compute paged slot ids for tokens [start_token, start_token+num_tokens)."""
    slots: list[int] = []
    for i in range(start_token, start_token + num_tokens):
        block_idx = i // block_size
        offset = i % block_size
        slots.append(block_ids[block_idx] * block_size + offset)
    return slots


class EpicConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        extra = self._kv_transfer_config.kv_connector_extra_config or {}

        chunk_size = int(extra.get("epic_chunk_size", DEFAULT_CHUNK_SIZE))
        # Chunk size must be a multiple of block_size so chunk boundaries align
        # with paged blocks (gather/scatter is block-granular friendly and slot
        # math is exact). Round up to the next block multiple.
        if chunk_size % self._block_size != 0:
            chunk_size = (
                (chunk_size + self._block_size - 1)
                // self._block_size
                * self._block_size
            )
        self._chunk_size = max(chunk_size, self._block_size)

        capacity = int(extra.get("epic_cpu_bytes", DEFAULT_CAPACITY_BYTES))
        world_size = vllm_config.parallel_config.world_size
        self._store = EpicChunkStore(
            capacity_bytes=capacity // max(world_size, 1),
            pin_memory=bool(extra.get("epic_pin_memory", True)),
        )

        # Scheduler-side bookkeeping: req_id -> matched prefix chunk hashes.
        self._matched_prefix: dict[str, list[tuple[str, int]]] = {}
        # req_id -> non-prefix hits (Phase 2 record only).
        self._non_prefix: dict[str, list[NonPrefixHit]] = {}
        # Requests that asked to load this step (after alloc).
        self._loads_pending: dict[str, "Request"] = {}

        # --- Sparse (non-prefix) forward flag (Phase 2b, S0) ---------------
        # OFF by default: every new code path below is inert and Phase 1/2a
        # behavior (dense forward, prefix-only reuse) is byte-for-byte unchanged.
        # ON: the connector derives a sparse recompute set M (LegoLink) per
        # request and emits it on EpicConnectorMetadata.sparse for the scheduler
        # / runner core patches (next batch) to consume.
        self._sparse_forward: bool = bool(
            extra.get("epic_sparse_forward", False)
        )
        # LegoLink link-token count: leading tokens of each non-prefix chunk
        # that are always recomputed (the boundary stitch). DESIGN §4.1.
        self._link_tokens: int = int(extra.get("epic_link_tokens", 8))

        # --- Strategy seams (see epic/reuse_strategy.py, epic/DESIGN.md) ---
        # EPIC = exact-hash selection + PIC alignment + static LegoLink recompute.
        # CacheBlend would swap alignment->Identity and recompute->dynamic.
        # Phase 1 routes selection through EpicSelection and alignment through
        # PicAlignment but is otherwise behaviorally identical to before.
        self._selection: SelectionStrategy = EpicSelection()
        # phase1_dense is the inverse of sparse_forward: when sparse is OFF the
        # policy short-circuits to an empty (dense) plan == Phase 1/2a.
        self._recompute: RecomputePolicy = LegoLinkRecompute(
            num_link_tokens=self._link_tokens,
            phase1_dense=not self._sparse_forward,
        )
        # Scheduler-side: req_id -> ReuseSelection from get_num_new_matched_tokens
        # (kept only when sparse_forward is on; needed to derive M at build time).
        self._selections: dict[str, "ReuseSelection"] = {}
        # Scheduler-side: req_id -> (N, external=|A|+|B|) for requests routed
        # through the sparse path this step. Used by the scheduler core hooks
        # (is_sparse_request / get_sparse_computed_advance) and cleared per step
        # in build_connector_meta. Empty when sparse forward is off.
        self._sparse_reqs: dict[str, tuple[int, int]] = {}

        # Phase 2a fusion-mask: enable installing the LegoLink FlexAttention
        # mask_mod. OFF by default so Phase 1 behavior (backend default causal)
        # is unchanged unless explicitly opted in via extra config.
        self._fusion_enabled: bool = bool(extra.get("epic_fusion_mask", False))
        # Fixed mask-tensor capacity (== max logical position) so the worker's
        # mask_mod object identity is stable across requests (recompile
        # avoidance). Sized to the longest possible sequence.
        try:
            self._mask_capacity = int(vllm_config.model_config.max_model_len)
        except Exception:  # noqa: BLE001
            self._mask_capacity = 0

        # Worker-side: lazily built once kv caches are registered.
        self._pic: PICRotator | None = None
        self._alignment: AlignmentStrategy | None = None
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        # Worker-side fusion-mask builder (owns fixed-size mask tensors + the
        # single stable mask_mod object). Built lazily on first install.
        self._mask_builder: LegoLinkMaskBuilder | None = None
        # FlexAttention layers we have installed our mask_mod onto (so we can
        # leave non-fusion layers untouched and restore on disable).
        self._flex_layers_patched: set[str] = set()

        # --- Sparse-mode safety gating (Phase 2b, S7) ----------------------
        # When epic_sparse_forward is ON, the worker forward depends on two
        # config invariants that the connector cannot enforce by itself
        # (changing them would be far more invasive than a config check):
        #   1. the FlexAttention backend (the only V1 backend that accepts the
        #      LegoLink logical_mask_mod the connector installs), and
        #   2. enforce_eager (the |M| sparse forward + per-step mask tensor are
        #      not yet validated under CUDA graph / piecewise compile capture).
        # Validate (do NOT silently mutate) so the failure is loud and actionable
        # rather than a wrong-numbers run. Fully inert when sparse is OFF.
        if self._sparse_forward:
            self._validate_sparse_safety(vllm_config)

        if role == KVConnectorRole.WORKER:
            try:
                self._pic = PICRotator.from_vllm_config(vllm_config)
                self._alignment = PicAlignment(self._pic)
            except Exception as e:  # noqa: BLE001
                # Non-Llama / unsupported rope_scaling: disable reuse loads but
                # keep the connector functional (saves still work).
                logger.warning("EPIC PICRotator unavailable, loads disabled: %s", e)
                self._pic = None
                self._alignment = None

    # ===================== sparse-mode safety gating (S7) =====================

    def _validate_sparse_safety(self, vllm_config: VllmConfig) -> None:
        """Fail fast if sparse-forward prerequisites are not met (Phase 2b, S7).

        Only called when ``epic_sparse_forward`` is on, so it never runs on the
        Phase 1/2a default path. Two checks:

          1. FlexAttention backend. Sparse forward installs a LegoLink
             ``logical_mask_mod`` (see ``_install_fusion_mask``); only the
             FlexAttention V1 backend consumes it. FlashAttention (the V1
             default) silently ignores the mask -> wrong results. We *validate*
             rather than force the backend: vLLM auto-selects the backend
             deep in platform code and overriding it from a connector would be
             far more invasive and fragile than a config assertion.

          2. enforce_eager. The reduced (|M|) sparse forward and the per-step
             fusion-mask tensor have not been validated under CUDA graph /
             piecewise-compile capture; eager is the first-pass safe mode.
             PIECEWISE relaxation is a post-GPU-verification TODO (PHASE2 §S7).
        """
        # --- (1) backend must be FlexAttention -----------------------------
        # vllm_config.attention_config.backend is the *configured* backend
        # (AttentionBackendEnum | None). None == "auto", which on the default
        # platform path resolves to FlashAttention, NOT FlexAttention -> reject
        # so the user explicitly pins FLEX_ATTENTION.
        backend = None
        try:
            backend = vllm_config.attention_config.backend
        except Exception:  # noqa: BLE001
            backend = None

        backend_name = getattr(backend, "name", None)
        if backend_name != "FLEX_ATTENTION":
            chosen = backend_name if backend_name is not None else "auto (default)"
            raise ValueError(
                "EPIC sparse forward (epic_sparse_forward=true) requires the "
                "FlexAttention backend, which is the only V1 backend that "
                "consumes the LegoLink logical_mask_mod the connector installs. "
                f"The configured attention backend is {chosen!r}. "
                "Select it explicitly via the engine argument "
                "--attention-backend FLEX_ATTENTION (serving) or the "
                "attention_backend='FLEX_ATTENTION' kwarg / "
                "attention_config.backend (offline LLM/EngineArgs), then retry. "
                "Note: the legacy VLLM_ATTENTION_BACKEND environment variable "
                "was removed in vLLM v0.22 and is silently ignored. "
                "The connector does not override backend auto-selection on "
                "purpose: doing so from a KV connector would be far more "
                "invasive than this configuration check."
            )

        # --- (2) eager required (first-pass safety) ------------------------
        enforce_eager = False
        try:
            enforce_eager = bool(vllm_config.model_config.enforce_eager)
        except Exception:  # noqa: BLE001
            enforce_eager = False

        if not enforce_eager:
            raise ValueError(
                "EPIC sparse forward (epic_sparse_forward=true) currently "
                "requires enforce_eager=true. The reduced (|M|) sparse forward "
                "and the per-step FlexAttention fusion-mask tensor have not been "
                "validated under CUDA graph / piecewise-compile capture. "
                "Pass --enforce-eager (or enforce_eager=True) and retry. "
                "Relaxing this to PIECEWISE is a post-GPU-verification TODO "
                "(see epic/PHASE2.md S7)."
            )

    # ===================== chunking helpers =====================

    def _split_prompt_into_chunks(
        self, token_ids: list[int]
    ) -> list[tuple[int, int, str]]:
        """Return [(start_token, length, chunk_hash), ...] for full chunks only.

        Only emits whole ``chunk_size`` chunks (a trailing partial chunk is left
        to normal prefill); this keeps hashes position-independent and aligned.
        """
        out: list[tuple[int, int, str]] = []
        n = len(token_ids)
        start = 0
        while start + self._chunk_size <= n:
            chunk = token_ids[start : start + self._chunk_size]
            out.append((start, self._chunk_size, hash_chunk_tokens(chunk)))
            start += self._chunk_size
        return out

    # ===================== scheduler-side =====================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        token_ids = list(request.prompt_token_ids or [])
        if not token_ids:
            return 0, False

        chunks = self._split_prompt_into_chunks(token_ids)

        # SelectionStrategy (scheduler side, side-effect free): which content
        # chunks to reuse. EPIC = exact-hash prefix walk; the contiguous prefix
        # is loadable now, non-prefix hits are Phase 2 candidates (recorded).
        sel = self._selection.select(
            request, num_computed_tokens, self._store, chunks
        )

        self._non_prefix[request.request_id] = sel.non_prefix_hits
        # Stash the full selection for sparse-M derivation in build_connector_meta
        # (only used when epic_sparse_forward is on). Side-effect-free w.r.t. the
        # return value, so multiple calls per request remain safe.
        if self._sparse_forward:
            self._selections[request.request_id] = sel

        # ---- sparse (non-prefix) reuse path (Phase 2b, S3) ----------------
        # When sparse forward is ON *and* this prompt has non-prefix content
        # matches, report external = |A| + |B| (prefix extent + sum of
        # non-prefix hit lengths). The kv_cache_manager.allocate_slots is
        # count-based (total_computed = local + external -> block count), so
        # reporting |A|+|B| makes it allocate the blocks that cover all N
        # logical positions (the genuinely-new |C| tokens plus the reused
        # A/B regions). The scheduler core patch (S3) then overrides the
        # forward-row count to |M| and the num_computed advance to N-external.
        #
        # Invariants the scheduler asserts (:624 external<=N, :670 num_new>0):
        #   * external = |A| + Σ|B| <= N  -- A and the B chunks are disjoint
        #     subranges of [0, N) by construction (prefix walk + chunk hits),
        #     so their total length cannot exceed N.
        #   * num_new = N - external >= 1 -- guaranteed because we only take
        #     this branch when external < N (see `if external < n` below); the
        #     degenerate external == N (whole prompt cached) falls through to
        #     the prefix-only path which itself returns num_new only when > 0.
        if self._sparse_forward and sel.non_prefix_hits:
            n = len(token_ids)
            non_prefix_len = sum(max(0, int(h.length)) for h in sel.non_prefix_hits)
            external = sel.prefix_extent + non_prefix_len
            # Clamp: hits are disjoint subranges of [0, N), but defend against a
            # store-length overestimate so external never exceeds N (assert :624).
            external = min(external, n)
            num_new = max(0, external - num_computed_tokens)
            if 0 < external < n and num_new > 0:
                # Mark this request sparse for the scheduler hooks; remember N
                # and external so the advance (N-external) can be derived later.
                self._matched_prefix[request.request_id] = sel.prefix_chunks
                self._sparse_reqs[request.request_id] = (n, external)
                # Phase 1 loads (prefix chunks) still happen synchronously in
                # start_load_kv; non-prefix B loading is the Batch-3 runner's job.
                return num_new, False
            # external == n (fully cached) or no genuinely-new tokens: fall
            # through to the prefix-only path (dense), which handles num_new==0.

        # Report only the part of the prefix extent beyond what vLLM computed.
        # Align to block_size (chunk_size already is a block multiple, so the
        # extent is too).
        num_new = max(0, sel.prefix_extent - num_computed_tokens)
        if num_new == 0:
            self._matched_prefix.pop(request.request_id, None)
            return 0, False

        self._matched_prefix[request.request_id] = sel.prefix_chunks
        # Phase 1 loads synchronously inside start_load_kv -> async = False.
        return num_new, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if num_external_tokens > 0 and request.request_id in self._matched_prefix:
            self._loads_pending[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = EpicConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            token_ids = list(new_req.prompt_token_ids or [])
            block_ids = new_req.block_ids[0]

            # ---- sparse recompute set M (Phase 2b, S0/S1) ----
            # OFF by default -> m_set is None and every guard below is inert
            # (Phase 1/2a dense behavior). ON -> derive M via the LegoLink
            # RecomputePolicy and emit it for the scheduler/runner core patches.
            m_set: set[int] | None = None
            if self._sparse_forward:
                m_set = self._emit_sparse(meta, req_id, token_ids)

            # ---- loads ----
            if req_id in self._loads_pending:
                prefix_chunks = self._matched_prefix.get(req_id, [])
                load = EpicReqLoad(req_id=req_id)
                for chunk_hash, start_token in prefix_chunks:
                    length = self._chunk_size
                    dst_slots = _slot_ids_from_blocks(
                        block_ids, self._block_size, start_token, length
                    )
                    load.chunks.append(
                        ChunkLoadSpec(
                            chunk_hash=chunk_hash,
                            dst_slot_ids=dst_slots,
                            old_pos_start=-1,  # resolved on worker from store
                            new_pos_start=start_token,
                            length=length,
                        )
                    )
                for h in self._non_prefix.get(req_id, []):
                    load.non_prefix_hits.append(h)
                meta.add_load(load)

            # ---- saves ----
            # Harvest full content chunks of this prompt that are NOT already
            # cached, so future requests can reuse them position-independently.
            save = EpicReqSave(req_id=req_id)
            already_loaded = {
                h for h, _ in self._matched_prefix.get(req_id, [])
            }
            for start, length, h in self._split_prompt_into_chunks(token_ids):
                if h in already_loaded or self._store.contains(h):
                    continue
                # ---- S2 save guard (sparse mode only) ----
                # In sparse forward, only the M tokens are freshly computed; the
                # rest of a chunk's KV in the paged cache is reused/rotated KV
                # (aligned for THIS request's positions), not canonical. Saving
                # such a chunk would poison the store. Emit a save only when the
                # whole chunk is in M (every token genuinely recomputed). Non-
                # sparse mode (m_set is None) is unchanged: always save.
                if m_set is not None and not all(
                    pos in m_set for pos in range(start, start + length)
                ):
                    continue
                slots = _slot_ids_from_blocks(
                    block_ids, self._block_size, start, length
                )
                positions = list(range(start, start + length))
                save.chunk_hashes.append(h)
                save.chunk_slot_ids.append(slots)
                save.chunk_positions.append(positions)
            if save.chunk_hashes:
                meta.add_save(save)

        # ---- fusion mask plan (Phase 2a) ----
        # Build a LegoLink mask plan covering this step. In Phase 2a the runner
        # still forwards every token, so the plan is a *dense* causal-equivalent
        # mask (gate OFF, reused_offsets empty). It is correct (== standard
        # causal) yet exercises the full mask-install path; Phase 2b will fill
        # recompute/reused offsets and set gate using the SAME wiring.
        if self._fusion_enabled:
            meta.fusion_mask = self._build_fusion_mask_plan(scheduler_output)

        # Reset per-step scheduler state. (_sparse_reqs is consumed by
        # _emit_sparse above, which copies external into EpicReqSparse, so the
        # scheduler core hooks read everything they need off ``meta`` -- the
        # connector stays a pure metadata producer.)
        self._loads_pending.clear()
        self._matched_prefix.clear()
        self._non_prefix.clear()
        self._selections.clear()
        self._sparse_reqs.clear()
        return meta

    def _emit_sparse(
        self,
        meta: EpicConnectorMetadata,
        req_id: str,
        token_ids: list[int],
    ) -> set[int] | None:
        """Derive M for one request and emit it on the metadata (Phase 2b, S1).

        Returns the M set (logical positions) for the S2 save guard, or ``None``
        if there is no sparse plan (degenerate / dense -> caller skips the
        guard, i.e. behaves like non-sparse). Only called when
        ``epic_sparse_forward`` is on.
        """
        n = len(token_ids)
        if n == 0:
            return None

        # Use the selection recorded at match time if available; otherwise fall
        # back to a fresh side-effect-free selection so M can still be derived.
        sel = self._selections.get(req_id)
        if sel is None:
            chunks = self._split_prompt_into_chunks(token_ids)
            sel = self._selection.select(None, 0, self._store, chunks)  # type: ignore[arg-type]

        # The policy's _seq_len() reads request.prompt_token_ids first; pass a
        # minimal shim carrying the exact N (the V1 Request isn't in scope in
        # build_connector_meta, only the new-req descriptor's token ids).
        plan: RecomputePlan = self._recompute.plan_recompute(
            request=_PromptLenShim(token_ids),
            selection=sel,
            block_size=self._block_size,
        )
        if not plan.recompute_offsets:
            return None  # dense / degenerate -> no sparse emission, no guard.

        seq_len = plan.seq_len or n
        # computed_advance = N - external. ``external`` is what the scheduler
        # already counted via num_external_computed_tokens (= |A|+|B|), recorded
        # in _sparse_reqs by get_num_new_matched_tokens. If this request did not
        # go through the sparse match path (e.g. _emit_sparse fired but external
        # was never recorded), fall back to seq_len so num_computed still lands
        # at N when external==0. Advancing by this (not len(M)) avoids double-
        # counting the link/last M tokens that overlap reused B positions.
        rec = getattr(self, "_sparse_reqs", {}).get(req_id)
        external = rec[1] if rec is not None else 0
        computed_advance = max(0, seq_len - external)

        meta.add_sparse(
            EpicReqSparse(
                req_id=req_id,
                sparse_positions=list(plan.recompute_offsets),
                full_seq_len=seq_len,
                computed_advance=computed_advance,
            )
        )
        return set(plan.recompute_offsets)

    # ----- core-hook interface (S0): scheduler calls these in the NEXT batch -----
    # Defined here so the scheduler-core patch (S3) has a stable connector API to
    # bind against. They read ONLY the per-step metadata the connector already
    # produced (EpicConnectorMetadata.sparse); they do NOT touch core V1 files.

    def get_sparse_num_scheduled_tokens(
        self, meta: "EpicConnectorMetadata", req_id: str
    ) -> int | None:
        """Number of token rows to forward for ``req_id`` under sparse reuse.

        Returns ``len(M)`` if a sparse plan exists for the request this step,
        else ``None`` (scheduler keeps its normal contiguous count). Pure lookup;
        no side effects.
        """
        for sp in meta.sparse:
            if sp.req_id == req_id and sp.is_sparse:
                return len(sp.sparse_positions)
        return None

    def get_sparse_positions(
        self, meta: "EpicConnectorMetadata", req_id: str
    ) -> list[int] | None:
        """Explicit per-token RoPE positions (M's logical positions) for ``req_id``.

        The runner core patch (next batch) will write these into the captured
        ``positions`` buffer instead of ``computed_prefix + arange`` (PHASE2 §2).
        Returns ``None`` when there is no sparse plan for the request.
        """
        for sp in meta.sparse:
            if sp.req_id == req_id and sp.is_sparse:
                return list(sp.sparse_positions)
        return None

    def has_sparse_requests(self, meta: "EpicConnectorMetadata") -> bool:
        """True iff any request this step has a sparse (non-dense) plan."""
        return any(sp.is_sparse for sp in meta.sparse)

    def get_sparse_computed_advance(
        self, meta: "EpicConnectorMetadata", req_id: str
    ) -> int | None:
        """How much to advance ``num_computed_tokens`` for ``req_id`` this step.

        Returns ``N - external`` (so num_computed converges to N at prefill end)
        if a sparse plan exists, else ``None`` (scheduler keeps the default
        ``+= num_scheduled_tokens`` advance). NOT ``len(M)``: M overlaps reused
        KV positions (link/last tokens inside B), so advancing by len(M) would
        double-count. Pure lookup; no side effects.
        """
        for sp in meta.sparse:
            if sp.req_id == req_id and sp.is_sparse and sp.computed_advance > 0:
                return sp.computed_advance
        return None

    # ----- schedule-time hook (S3): called DURING the scheduling loop, BEFORE
    # build_connector_meta, so it reads live scheduler-side state (_sparse_reqs)
    # rather than the per-step metadata (which isn't built yet at that point).

    def is_sparse_request(self, req_id: str) -> bool:
        """True iff ``req_id`` is being scheduled through the sparse reuse path
        this step (recorded by ``get_num_new_matched_tokens``).

        Used by the scheduler to set ``delay_cache_blocks=True`` so the approx
        (reused/rotated) KV blocks of a sparse request are NOT registered into
        the native prefix cache (avoids poisoning it; DESIGN / S3 brief).
        """
        return req_id in self._sparse_reqs

    def _build_fusion_mask_plan(
        self, scheduler_output: SchedulerOutput
    ) -> FusionMaskPlan | None:
        """Derive this step's LegoLink fusion-mask plan (Phase 2a: dense).

        Uses ``RecomputePolicy.plan_recompute`` so the EPIC strategy seam owns
        what M is. Phase 1/2a's ``LegoLinkRecompute(phase1_dense=True)`` returns
        an empty plan -> dense causal-equivalent mask. The max logical position
        the mask must address is the longest scheduled sequence this step.
        """
        # Upper bound on the logical positions the mask must address this step.
        # The worker further widens capacity from attn_meta.max_seq_len, so an
        # under-estimate here is corrected on the worker; we only need a sane
        # nonzero hint.
        max_seq_len = 0
        for n in scheduler_output.num_scheduled_tokens.values():
            max_seq_len = max(max_seq_len, int(n))
        for new_req in scheduler_output.scheduled_new_reqs:
            max_seq_len = max(max_seq_len, len(new_req.prompt_token_ids or []))
        if max_seq_len <= 0:
            return None

        # Strategy-owned M (Phase 2a: empty -> dense). RecomputePlan.is_sparse
        # is False, so gate stays OFF and the mask == standard causal.
        plan: RecomputePlan = self._recompute.plan_recompute(
            request=None,  # type: ignore[arg-type]
            selection=None,  # type: ignore[arg-type]
            block_size=self._block_size,
        )
        return FusionMaskPlan(
            enabled=True,
            seq_len=max_seq_len,
            recompute_offsets=list(plan.recompute_offsets),
            reused_offsets=list(plan.reused_offsets),
            gate=plan.is_sparse,
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        # Phase 1 saves are harvested during the prefill forward (save_kv_layer)
        # for new requests; nothing to defer here. Return synchronous-free.
        return False, None

    # ===================== worker-side =====================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv_caches = kv_caches
        self._layer_names = list(kv_caches.keys())

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Scatter cached chunks (PIC-rotated) and install the fusion mask.

        Phase 1 is an eager, one-shot all-layer load. Layer-wise async pipelining
        (overlapping load with compute via wait_for_layer_load) is a TODO.

        Phase 2a additionally installs the LegoLink FlexAttention mask_mod onto
        the attention layers (via forward_context) before the model forward.
        """
        meta = self._get_connector_metadata()
        if not isinstance(meta, EpicConnectorMetadata):
            return

        # Install / refresh the fusion mask BEFORE chunk loads so it is ready by
        # the time the forward runs. Independent of whether there are loads.
        self._install_fusion_mask(forward_context, meta)

        if not meta.loads or self._alignment is None:
            return

        for load in meta.loads:
            for spec in load.chunks:
                stored = self._store.get(spec.chunk_hash)
                if stored is None:
                    # Evicted between scheduler match and worker load.
                    continue
                self._load_chunk(stored, spec)

    def _install_fusion_mask(
        self,
        forward_context: "ForwardContext",
        meta: EpicConnectorMetadata,
    ) -> None:
        """Install the LegoLink mask_mod onto FlexAttention layers (Phase 2a).

        Non-invasive: FlexAttention's ``forward`` already auto-picks up a
        ``logical_mask_mod`` attribute set on the attention layer object (see
        ``vllm/v1/attention/backends/flex_attention.py`` ~L1122). We set that
        attribute on each attention layer reachable via
        ``forward_context.no_compile_layers`` (== ``static_forward_context``,
        the ``Attention`` modules that are passed as ``layer`` to
        ``impl.forward``).

        Recompile avoidance: the mask builder hands back the SAME mask_mod
        object every step (only the backing tensor *contents* change), so
        FlexAttention's identity check (``logical_mask_mod is not layer_mask_mod``)
        fires once and the compiled kernel is reused thereafter.
        """
        plan = meta.fusion_mask
        layers = getattr(forward_context, "no_compile_layers", None)
        if not layers:
            return

        if plan is None or not plan.enabled:
            # Fusion disabled this step: remove any mask we previously installed
            # so the backend reverts to its default causal mask. Leave layers we
            # never touched alone.
            for name in self._flex_layers_patched:
                layer = layers.get(name)
                if layer is not None and hasattr(layer, "logical_mask_mod"):
                    try:
                        delattr(layer, "logical_mask_mod")
                    except AttributeError:
                        layer.logical_mask_mod = None
            self._flex_layers_patched.clear()
            return

        # Build/refresh the single stable mask_mod object from the plan tensors.
        # Fixed capacity -> the mask_mod object identity never changes across
        # requests, so FlexAttention reuses its compiled kernel. Tensors live on
        # the KV-cache device so the on-device mask_mod can index them.
        if self._mask_builder is None:
            cap = self._mask_capacity or max(plan.seq_len, 1)
            device = "cpu"
            for kv in self._kv_caches.values():
                device = str(kv.device)
                break
            self._mask_builder = LegoLinkMaskBuilder(capacity=cap, device=device)
        # Translate the picklable plan into a RecomputePlan the builder fills.
        recompute_plan = RecomputePlan(
            recompute_offsets=list(plan.recompute_offsets),
            reused_offsets=list(plan.reused_offsets),
            seq_len=plan.seq_len,
        )
        # No FlexAttentionMetadata here (we run before the backend builds its
        # per-step metadata); pass None and let the builder size from the plan.
        mask_mod = self._mask_builder.build_logical_mask_mod(
            recompute_plan, attn_meta=None  # type: ignore[arg-type]
        )

        for name in self._flex_install_targets(layers):
            layer = layers.get(name)
            if layer is None:
                continue
            # Setting the attribute is the documented FlexAttention hook.
            layer.logical_mask_mod = mask_mod
            self._flex_layers_patched.add(name)

    def _flex_install_targets(self, layers: dict[str, Any]) -> list[str]:
        """Names of attention layers to install the mask on.

        Phase 2a installs on every attention layer (full-forward, mask ==
        causal). Phase 2b can restrict this to check/fusion layers via the
        recompute plan.
        """
        return list(layers.keys())

    def _load_chunk(self, stored: StoredChunk, spec: ChunkLoadSpec) -> None:
        length = min(stored.length, spec.length)
        if length == 0:
            return
        # Destination slots and positions.
        new_positions = torch.arange(
            spec.new_pos_start, spec.new_pos_start + length, dtype=torch.int64
        )
        old_positions = stored.old_positions[:length].to(torch.int64)

        for layer_name in self._layer_names:
            kv_cache = self._kv_caches.get(layer_name)
            if kv_cache is None or layer_name not in stored.k_per_layer:
                continue
            device = kv_cache.device
            k_cpu = stored.k_per_layer[layer_name][:length]
            v_cpu = stored.v_per_layer[layer_name][:length]
            k = k_cpu.to(device, non_blocking=True)
            v = v_cpu.to(device, non_blocking=True)

            # AlignmentStrategy: EPIC = PIC delta re-rotary (no fake_q hack;
            # see pic.py). CacheBlend would plug IdentityAlignment here.
            assert self._alignment is not None
            k = self._alignment.align_keys(
                k,
                old_positions.to(device),
                new_positions.to(device),
                layer_name,
            )

            self._scatter_kv(kv_cache, k, v, spec.dst_slot_ids[:length])

    def _scatter_kv(
        self,
        kv_cache: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_ids: list[int],
    ) -> None:
        """Scatter [num_tokens, heads, head_size] K/V into the paged cache.

        FlashAttention V1 layout: (2, num_blocks, block_size, num_kv_heads,
        head_size). slot = block_id*block_size + offset.
        """
        slots = torch.as_tensor(slot_ids, device=kv_cache.device, dtype=torch.long)
        # kv_cache[0] = K bank, kv_cache[1] = V bank.
        shape = kv_cache.shape
        if shape[0] == 2 and kv_cache.dim() == 5:
            num_blocks, block_size = shape[1], shape[2]
            k_bank = kv_cache[0].reshape(num_blocks * block_size, *shape[3:])
            v_bank = kv_cache[1].reshape(num_blocks * block_size, *shape[3:])
            k_bank[slots] = k.to(k_bank.dtype)
            v_bank[slots] = v.to(v_bank.dtype)
        else:
            # Unknown layout: skip (Phase 1 supports default FlashAttention only).
            logger.warning_once(
                "EPIC: unsupported kv cache layout %s; skipping load.", tuple(shape)
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Phase 1 loads synchronously in start_load_kv -> nothing to wait for.
        # TODO(Phase 2): layer-wise async load pipelining.
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Harvest this layer's KV for to-be-saved chunks into the CPU store."""
        meta = self._connector_metadata
        if not isinstance(meta, EpicConnectorMetadata) or not meta.saves:
            return
        shape = kv_layer.shape
        if not (shape[0] == 2 and kv_layer.dim() == 5):
            return
        num_blocks, block_size = shape[1], shape[2]
        k_bank = kv_layer[0].reshape(num_blocks * block_size, *shape[3:])
        v_bank = kv_layer[1].reshape(num_blocks * block_size, *shape[3:])

        for save in meta.saves:
            for ci, chunk_hash in enumerate(save.chunk_hashes):
                slot_ids = save.chunk_slot_ids[ci]
                positions = save.chunk_positions[ci]
                slots = torch.as_tensor(
                    slot_ids, device=kv_layer.device, dtype=torch.long
                )
                k = k_bank[slots].detach().to("cpu")
                v = v_bank[slots].detach().to("cpu")
                k = self._store.maybe_pin(k)
                v = self._store.maybe_pin(v)

                stored = self._store.get(chunk_hash)
                if stored is None:
                    stored = StoredChunk(
                        chunk_hash=chunk_hash,
                        length=len(slot_ids),
                        old_positions=torch.as_tensor(positions, dtype=torch.int64),
                    )
                stored.k_per_layer[layer_name] = k
                stored.v_per_layer[layer_name] = v
                # put() refreshes bytes / LRU; safe to call repeatedly per layer.
                self._store.put(stored)

    def wait_for_save(self):
        # All saves are synchronous CPU copies in save_kv_layer.
        # TODO(Phase 2): async D2H copy with a stream + wait here.
        return
