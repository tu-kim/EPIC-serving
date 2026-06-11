# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strategy interfaces for non-contiguous KV cache reuse.

This module factors the *algorithmic* decisions that distinguish reuse
algorithms (EPIC vs CacheBlend) out of the transport/lifecycle plumbing provided
by ``KVConnectorBase_V1``. See ``epic/DESIGN.md`` for the full rationale, call
sites, and the EPIC/CacheBlend mapping table.

Four seams, each an ABC:

  * ``SelectionStrategy``   -- scheduler side; which content chunks to reuse.
  * ``AlignmentStrategy``   -- worker/load side; fix reused-K position mismatch.
  * ``RecomputePolicy``     -- which M tokens are actually forwarded (M is frozen
                               BEFORE the forward; see DESIGN §1).
  * ``FusionMaskBuilder``   -- worker/attention side; FlexAttention ``mask_mod``
                               for M queries over the full (reused) KV.

The base connector owns the chunk store, scatter/gather, and lifecycle; concrete
reuse algorithms differ only by which strategy objects they construct.

NOTE: This is a skeleton. The EPIC implementations here adapt the *already
working* Phase 1 code (``PICRotator``, the prefix/non-prefix walk) so the
existing connector keeps passing its tests. Heavy Phase 2 bodies
(LegoLink M derivation, FlexAttention mask wiring, CacheBlend) are intentionally
left as ``NotImplementedError`` for the migrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
    NonPrefixHit,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.pic import PICRotator

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
        EpicChunkStore,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.fusion_mask import (
        FusionMaskTensors,
    )
    from vllm.v1.attention.backends.flex_attention import (
        BlockSparsityHint,
        FlexAttentionMetadata,
    )
    from vllm.v1.request import Request

# FlexAttention's real mask_mod signature: (b, h, q_idx, kv_idx) -> bool tensor.
MaskMod = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


@runtime_checkable
class SupportsChunkMembership(Protocol):
    """The minimal membership surface SelectionStrategy needs.

    Both the worker ``EpicChunkStore`` (real tensors) and the scheduler
    ``EpicSchedulerIndex`` (metadata-only mirror) implement this, so selection is
    agnostic to which side it runs on. SCHEDULER role passes the index here; the
    WORKER role never runs selection. This is the seam that fixes the role-split
    bug: scheduler selection queries the mirror, not its own (always-empty)
    store.
    """

    def contains(self, chunk_hash: str) -> bool: ...

    def get_length(self, chunk_hash: str) -> int | None: ...


# ============================================================================
# Shared data structures (strategy outputs)
# ============================================================================


@dataclass
class ReuseSelection:
    """Output of ``SelectionStrategy.select``.

    Separates chunks that lie in the contiguous prefix (loadable today, Phase 1)
    from content matches scattered elsewhere in the prompt (Phase 2 candidates).
    """

    # (chunk_hash, prompt_start_token) for chunks covering the contiguous prefix.
    prefix_chunks: list[tuple[str, int]] = field(default_factory=list)
    # Token count covered by the contiguous prefix hits from position 0.
    prefix_extent: int = 0
    # Content matches not in the contiguous prefix (recorded, not loaded in P1).
    non_prefix_hits: list[NonPrefixHit] = field(default_factory=list)


@dataclass
class RecomputePlan:
    """Output of ``RecomputePolicy``. The frozen description of M.

    ``M`` = the set of token rows that will actually be forwarded. Everything
    else is served from reused (aligned) KV in the paged cache. This object is
    produced BEFORE the model forward and is immutable thereafter (DESIGN §1).

    Phase 1 leaves this empty (full dense forward, no sparse M). Phase 2 fills
    ``recompute_offsets`` and ``target_positions``; the runner reads
    ``target_positions`` to build sparse RoPE positions (PHASE2 §2), and the
    fusion mask reads the tensor fields by ``q_idx``/``kv_idx`` (DESIGN §2.4).
    """

    # Prompt-relative token offsets that are forwarded (the M set). Sorted.
    recompute_offsets: list[int] = field(default_factory=list)
    # Per-M-token absolute target position (for sparse RoPE). len == len(M).
    target_positions: list[int] = field(default_factory=list)
    # Sequence length this plan describes (logical positions [0, seq_len)).
    # Used to size the fusion-mask fill; 0 means "infer from attn_meta".
    seq_len: int = 0
    # Logical positions whose KV participates in attention (reused chunks +
    # recomputed M tokens). Empty list with ``seq_len == 0`` (Phase 1/2a dense)
    # means "all positions live" -> the mask reduces to standard causal.
    reused_offsets: list[int] = field(default_factory=list)
    # Optional dense tensors the FlexAttention mask_mod indexes WITHOUT closure
    # capture (DESIGN §2.4 / risk 1). Built lazily on the worker.
    #   recompute_flag[token_idx] == 1  -> this row is an M (recomputed) query
    #   reused_flag[kv_idx]       == 1  -> this KV row is reused (read-only)
    recompute_flag: torch.Tensor | None = None
    reused_flag: torch.Tensor | None = None

    @property
    def is_sparse(self) -> bool:
        """True iff fewer than all prompt tokens are forwarded."""
        return bool(self.recompute_offsets)


# ============================================================================
# Strategy ABCs
# ============================================================================


class SelectionStrategy(ABC):
    """Decide which content-matched chunks to reuse (scheduler side).

    Called from ``connector.get_num_new_matched_tokens`` and MUST be
    side-effect free (the base contract allows multiple calls per request).
    """

    @abstractmethod
    def select(
        self,
        request: "Request",
        num_computed_tokens: int,
        store: "SupportsChunkMembership",
        chunks: list[tuple[int, int, str]],
    ) -> ReuseSelection:
        """Return prefix + non-prefix matches.

        Args:
            store: any object exposing ``contains`` / ``get_length`` -- on the
                scheduler this is the ``EpicSchedulerIndex`` mirror, never the
                (empty) per-role ``EpicChunkStore``.
            chunks: ``[(start_token, length, chunk_hash), ...]`` full chunks of
                the prompt (already block-aligned by the connector).
        """
        raise NotImplementedError


class AlignmentStrategy(ABC):
    """Correct the position mismatch of reused K (worker side, load time).

    Called once per layer per chunk inside the connector's load path, before the
    K is scattered into the paged cache.
    """

    @abstractmethod
    def align_keys(
        self,
        key: torch.Tensor,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """Return K aligned to ``new_positions``.

        Args:
            key: [num_tokens, num_kv_heads, head_size], already RoPE'd at
                ``old_positions``.
        """
        raise NotImplementedError


class RecomputePolicy(ABC):
    """Decide which tokens (M) are forwarded. M is frozen before forward.

    Static policies (EPIC) fill the plan at schedule time in ``plan_recompute``.
    Dynamic policies (CacheBlend) request an importance pre-pass and complete the
    plan in ``score_to_plan`` -- still before the main forward (DESIGN §1, §4.3).
    """

    def needs_importance_prepass(self) -> bool:
        """True if M depends on attention/V statistics from a check layer."""
        return False

    @abstractmethod
    def plan_recompute(
        self,
        request: "Request",
        selection: ReuseSelection,
        block_size: int,
    ) -> RecomputePlan:
        """Schedule-time M. For dynamic policies, return a *candidate* plan that
        the pre-pass refines via ``score_to_plan``."""
        raise NotImplementedError

    def score_to_plan(
        self,
        request: "Request",
        selection: ReuseSelection,
        scores: torch.Tensor,
    ) -> RecomputePlan:
        """Dynamic completion: freeze M from importance ``scores``.

        Only called when ``needs_importance_prepass()`` is True. Default raises
        so static policies need not implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is static; score_to_plan not applicable."
        )


class FusionMaskBuilder(ABC):
    """Build the FlexAttention mask for M queries over reused KV (worker side).

    Returns a ``mask_mod`` with FlexAttention's real
    ``(b, h, q_idx, kv_idx) -> bool`` signature. CRITICAL (DESIGN §2.4 / risk 1):
    the returned closure must read per-request reuse state from metadata TENSORS
    indexed by ``q_idx``/``kv_idx`` -- never capture per-step Python scalars/lists
    -- so FlexAttention does not recompile every step.
    """

    @abstractmethod
    def build_logical_mask_mod(
        self,
        plan: RecomputePlan,
        attn_meta: "FlexAttentionMetadata",
    ) -> MaskMod:
        raise NotImplementedError

    def build_block_sparsity_hint(
        self,
        plan: RecomputePlan,
        attn_meta: "FlexAttentionMetadata",
    ) -> "BlockSparsityHint | None":
        """Optional KV-block pruning hint (skip fully-masked blocks)."""
        return None


# ============================================================================
# EPIC concrete strategies
# ============================================================================


class EpicSelection(SelectionStrategy):
    """Exact content-hash prefix walk (the Phase 1 behavior, formalized).

    Walks chunks from the prompt start; contiguous hits form the prefix extent,
    everything else with a hit is recorded as a non-prefix candidate.
    """

    def select(
        self,
        request: "Request",
        num_computed_tokens: int,
        store: "SupportsChunkMembership",
        chunks: list[tuple[int, int, str]],
    ) -> ReuseSelection:
        sel = ReuseSelection()
        contiguous = True
        for start, length, h in chunks:
            hit = store.contains(h)
            if contiguous and start == sel.prefix_extent and hit:
                sel.prefix_chunks.append((h, start))
                sel.prefix_extent += length
            else:
                contiguous = False
                if hit:
                    stored_len = store.get_length(h) or length
                    sel.non_prefix_hits.append(
                        NonPrefixHit(
                            chunk_hash=h,
                            prompt_offset=start,
                            old_pos_start=0,  # resolved at load from store
                            length=stored_len,
                        )
                    )
        return sel


class PicAlignment(AlignmentStrategy):
    """EPIC alignment: PIC delta re-rotary (wraps the Phase 1 ``PICRotator``)."""

    def __init__(self, rotator: PICRotator):
        self._rotator = rotator

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "PicAlignment":
        return cls(PICRotator.from_vllm_config(vllm_config))

    def align_keys(
        self,
        key: torch.Tensor,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        return self._rotator.rotate_keys(key, old_positions, new_positions)


class IdentityAlignment(AlignmentStrategy):
    """CacheBlend alignment: no re-rotary (mismatch absorbed by recompute)."""

    def align_keys(
        self,
        key: torch.Tensor,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        return key


class LegoLinkRecompute(RecomputePolicy):
    """EPIC LegoLink: M = new tokens + per-chunk-boundary link tokens. Static.

    Derives the recompute set M at schedule time from the selection result and
    chunk boundaries (DESIGN §4.1). M is the union of:

      * **C** -- every genuinely new token (no matched chunk covers it),
      * **link tokens** -- the leading ``num_link_tokens`` of every non-prefix
        reused chunk (the LegoLink boundary stitch),
      * **{N-1}** -- the final prompt token (always recomputed so the first
        decode step sees a freshly-computed last position).

    The contiguous prefix A (``selection.prefix_extent``) is reused natively and
    is **never** part of M.

    Phase 1/2a (``phase1_dense=True``) short-circuits to an empty (dense forward)
    plan, because the V1 scheduler/runner cannot yet consume a sparse M
    (PHASE2 §1, §2). Phase 2b flips ``phase1_dense=False`` and the body below
    runs, producing the sorted, deduplicated M with its invariants asserted.
    """

    def __init__(self, num_link_tokens: int = 0, phase1_dense: bool = True):
        self._num_link_tokens = max(0, int(num_link_tokens))
        self._phase1_dense = phase1_dense

    def needs_importance_prepass(self) -> bool:
        return False  # LegoLink is static.

    def _seq_len(
        self, request: "Request | None", selection: ReuseSelection
    ) -> int:
        """Resolve N (full prompt length) for this plan.

        Prefers the request's prompt length; falls back to the furthest extent
        implied by the selection (so the policy is testable without a Request).
        """
        if request is not None:
            ids = getattr(request, "prompt_token_ids", None)
            if ids is not None:
                return len(ids)
        n = selection.prefix_extent
        for hit in selection.non_prefix_hits:
            n = max(n, hit.prompt_offset + hit.length)
        return n

    def plan_recompute(
        self,
        request: "Request | None",
        selection: ReuseSelection | None,
        block_size: int,
    ) -> RecomputePlan:
        if self._phase1_dense:
            # Phase 1/2a: no sparse forward -> empty plan == full dense forward.
            return RecomputePlan()

        # Phase 2b sparse M derivation.
        if selection is None:
            # Legitimate dense fall-through: the per-step fusion-mask plan
            # (_build_fusion_mask_plan) calls without a selection even when
            # sparse mode is on. In sparse mode the fusion mask intentionally
            # stays DENSE (standard causal): all logical KV positions are live
            # (reused A/B loaded + M computed), and the per-request sparse M is
            # carried separately via EpicReqSparse. Empty plan == gate OFF.
            return RecomputePlan()
        n = self._seq_len(request, selection)
        if n <= 0:
            return RecomputePlan()

        prefix_extent = max(0, int(selection.prefix_extent))

        # Degenerate: the whole prompt is the contiguous prefix A (pure Phase-1
        # prefix reuse, no non-prefix content). There is no sparse forward to
        # express and forcing N-1 into M would put a prefix-A position in M,
        # violating the "A never in M" invariant. Defer to the dense/native
        # path with an empty plan.
        if prefix_extent >= n:
            return RecomputePlan()

        # ``reused``: logical positions served from reused (aligned) KV -- the
        # prefix A plus every non-prefix chunk B. M tokens that fall on a reused
        # position are *recomputed over* that position, but the position is
        # still "live" KV from attention's perspective.
        reused: set[int] = set(range(min(prefix_extent, n)))
        for hit in selection.non_prefix_hits:
            lo = max(0, int(hit.prompt_offset))
            hi = min(n, lo + max(0, int(hit.length)))
            reused.update(range(lo, hi))

        # ``M`` set.
        m: set[int] = set()

        # (1) C -- genuinely new tokens: everything not covered by any reused
        # chunk (prefix A or non-prefix B). Prefix A is reused natively and is
        # excluded from M by construction.
        for pos in range(n):
            if pos not in reused:
                m.add(pos)

        # (2) link tokens -- the leading ``num_link_tokens`` of each non-prefix
        # chunk B. ``min`` clamps the boundary case where link > chunk length.
        # Prefix-A chunks contribute NO link tokens (A is fully native).
        link = self._num_link_tokens
        if link > 0:
            for hit in selection.non_prefix_hits:
                lo = max(0, int(hit.prompt_offset))
                hi = min(n, lo + min(link, max(0, int(hit.length))))
                m.update(range(lo, hi))

        # (3) last prompt token -- always recomputed. Skip-add if already in M
        # (C or link) so there is no duplicate.
        m.add(n - 1)

        offsets = sorted(m)

        # ---- invariants (DESIGN §4.1, brief S1) ----
        assert offsets == sorted(set(offsets)), "M must be sorted + unique"
        assert all(0 <= o < n for o in offsets), "M offsets must be in [0, N)"
        assert offsets[-1] == n - 1, "last M offset must be N-1"
        # Prefix A must never appear in M.
        assert all(o >= prefix_extent for o in offsets), (
            "prefix A positions must not be in M"
        )

        # Positions that participate in attention: reused KV ∪ M (sorted).
        reused.update(offsets)
        reused_offsets = sorted(reused)

        return RecomputePlan(
            recompute_offsets=offsets,
            target_positions=list(offsets),  # logical == target position
            seq_len=n,
            reused_offsets=reused_offsets,
        )


class LegoLinkMaskBuilder(FusionMaskBuilder):
    """EPIC fusion mask (Phase 2a).

    Emits a FlexAttention ``mask_mod`` for the LegoLink reuse pattern: each M
    (recompute) query attends causally over the full reused+recomputed KV. In
    Phase 2a every row is forwarded, so the mask is *equivalent to standard
    causal* (full-forward back-compatibility). Phase 2b flips the recompute
    gate to forward only M rows with the SAME mask_mod object.

    Recompile avoidance (DESIGN §2.4 / risk 1): all per-request reuse state
    lives in fixed-size, pre-allocated ``FusionMaskTensors`` (see
    ``fusion_mask.py``). The ``mask_mod`` closure is built ONCE over those
    tensors and reads them by ``q_idx``/``kv_idx``; it captures no per-step
    Python scalar/list. Across requests only the tensor *contents* change
    (``fill_request``), so FlexAttention reuses the same compiled kernel.

    The builder is stateful by design: it owns the backing tensors and the
    single compiled-graph-stable mask_mod object. Call ``build_logical_mask_mod``
    once per request -- it refills the tensors in place and returns the SAME
    function object every time (verifiable: object identity is stable).
    """

    def __init__(self, capacity: int | None = None, device: str = "cpu"):
        # Backing tensors + the single mask_mod object. Allocated on first use
        # (so capacity can be inferred from the first attn_meta) or eagerly if
        # ``capacity`` is provided.
        self._tensors: "FusionMaskTensors | None" = None
        self._mask_mod: MaskMod | None = None
        self._eager_capacity = capacity
        self._device = device
        if capacity is not None:
            self._ensure_built(capacity, device)

    def _ensure_built(self, capacity: int, device: str) -> None:
        from vllm.distributed.kv_transfer.kv_connector.v1.epic.fusion_mask import (
            FusionMaskTensors,
            build_legolink_mask_mod,
        )

        if self._tensors is not None and self._tensors.capacity >= capacity:
            return
        # (Re)allocate to >= requested capacity and (re)build the mask object.
        # This only happens when capacity grows; steady state reuses the same
        # object so the compiled graph is preserved.
        self._tensors = FusionMaskTensors.allocate(capacity, device)
        self._mask_mod = build_legolink_mask_mod(self._tensors)

    @property
    def tensors(self) -> "FusionMaskTensors":
        assert self._tensors is not None, "mask builder not built yet"
        return self._tensors

    def _infer_capacity(
        self, plan: RecomputePlan, attn_meta: "FlexAttentionMetadata | None"
    ) -> int:
        if self._eager_capacity is not None:
            return self._eager_capacity
        cap = plan.seq_len
        if attn_meta is not None:
            cap = max(cap, getattr(attn_meta, "max_seq_len", 0))
            cap = max(cap, getattr(attn_meta, "num_actual_tokens", 0))
        return max(cap, 1)

    def build_logical_mask_mod(
        self,
        plan: RecomputePlan,
        attn_meta: "FlexAttentionMetadata",
    ) -> MaskMod:
        capacity = self._infer_capacity(plan, attn_meta)
        self._ensure_built(capacity, self._device)

        seq_len = plan.seq_len
        if seq_len <= 0:
            # Dense Phase-2a default: the whole sequence is live -> causal.
            if attn_meta is not None:
                seq_len = int(getattr(attn_meta, "max_seq_len", 0)) or capacity
            else:
                seq_len = capacity

        # Phase 2a: gate OFF -> every row forwarded; reduces to causal. Phase 2b
        # sets recompute_offsets / reused_offsets and gate=True with the SAME
        # mask_mod object.
        gate_on = plan.is_sparse
        reused = plan.reused_offsets or None
        recompute = plan.recompute_offsets or None

        self.tensors.fill_request(
            seq_len=seq_len,
            recompute_offsets=recompute,
            reused_offsets=reused,
            gate=gate_on,
        )

        assert self._mask_mod is not None
        return self._mask_mod


__all__ = [
    "MaskMod",
    "SupportsChunkMembership",
    "ReuseSelection",
    "RecomputePlan",
    "SelectionStrategy",
    "AlignmentStrategy",
    "RecomputePolicy",
    "FusionMaskBuilder",
    "EpicSelection",
    "PicAlignment",
    "IdentityAlignment",
    "LegoLinkRecompute",
    "LegoLinkMaskBuilder",
]
