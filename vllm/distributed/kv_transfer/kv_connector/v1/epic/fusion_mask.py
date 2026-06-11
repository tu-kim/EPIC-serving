# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC LegoLink fusion mask for FlexAttention (Phase 2a).

This module materializes the *tensor-lookup* scheme that lets the EPIC LegoLink
``mask_mod`` be installed on a FlexAttention layer WITHOUT triggering a
``torch.compile`` recompile on every step (DESIGN.md §2.4 / risk 1).

Recompile avoidance — the load-bearing property
-----------------------------------------------
FlexAttention compiles its kernel keyed on (among other things) the *identity*
of the ``mask_mod`` callable and the *shapes* of any Tensors it closes over. If
the mask closure captured per-request Python scalars/lists (``imp_indices``,
chunk boundaries, ...) the closure object would be rebuilt every step and the
captured shapes would vary, forcing a recompile each step.

We avoid that by:

  * Allocating ONE set of fixed-size metadata tensors per worker
    (``FusionMaskTensors.allocate``), sized to the maximum batched token count.
  * Building the ``mask_mod`` closure ONCE over those fixed tensors
    (``build_legolink_mask_mod``). Its captured Tensor identities/shapes never
    change.
  * Per request we only *overwrite the contents in place* (``fill_request``);
    the closure object and captured shapes are unchanged, so the same compiled
    graph is reused.

The mask reads everything by ``q_idx`` / ``kv_idx`` (the *logical* indices that
FlexAttention's paged ``final_mask_mod`` hands the logical mask — see
``flex_attention.FlexAttentionMetadata._convert_physical_to_logical``).

Semantics (LegoLink, Phase 2a == full-forward compatible)
---------------------------------------------------------
Every M (recompute) query row attends causally over the full KV (reused +
recomputed). In Phase 2a *all* rows are forwarded, so non-M rows must reduce to
standard causal too. The mask is therefore expressed as:

    allow = causal(q_idx, kv_idx)
            AND kv_live[kv_idx]
            AND ( gate == 0  OR  recompute_flag[q_idx] )

where, indexed by logical position within the request:

  * ``kv_live[kv_idx] == 1``       -> this KV row participates (reused or
                                      recomputed). Phase 2a fills it all-ones
                                      over the valid range, so it is inert.
  * ``recompute_flag[q_idx] == 1`` -> this query row is an M (recompute) row.
  * ``gate``                       -> a 1-element int tensor. ``0`` (Phase 2a:
                                      every row forwarded, recompute gating
                                      OFF) makes the whole predicate reduce to
                                      ``causal AND kv_live`` == standard causal.
                                      ``1`` (Phase 2b: only M rows forwarded)
                                      turns on per-row recompute gating with no
                                      change to the closure / compiled graph.

Phase 2b inherits this module unchanged: it flips ``gate`` to 1 and fills
``recompute_flag`` for the M rows; the same ``mask_mod`` object and the same
compiled kernel are reused.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import MaskMod


@dataclass
class FusionMaskTensors:
    """Pre-allocated, fixed-size metadata tensors backing the LegoLink mask.

    Indexed by *logical* position within a request (the index space the
    FlexAttention logical mask_mod receives). All tensors are allocated once at
    worker init time to a fixed capacity and reused across steps; only their
    contents are overwritten per request (``fill_request``). This is what makes
    the captured-tensor shapes constant and avoids FlexAttention recompiles.

    Attributes:
        recompute_flag: int32 [capacity]. ``1`` at logical positions that are M
            (recompute) query rows, else ``0``.
        kv_live: int32 [capacity]. ``1`` at logical positions whose KV row
            participates in attention (reused or recomputed), else ``0``.
        gate: int32 [1]. ``0`` disables recompute-row gating (Phase 2a, every
            row forwarded == standard causal). ``1`` enables it (Phase 2b,
            M-only forward).
    """

    recompute_flag: torch.Tensor
    kv_live: torch.Tensor
    gate: torch.Tensor

    @classmethod
    def allocate(
        cls,
        capacity: int,
        device: torch.device | str = "cpu",
    ) -> "FusionMaskTensors":
        """Allocate fixed-size backing tensors.

        Args:
            capacity: max logical index addressable (>= max batched tokens, or
                max sequence length for a paged single-request mask).
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        dev = torch.device(device)
        return cls(
            recompute_flag=torch.zeros(capacity, dtype=torch.int32, device=dev),
            kv_live=torch.zeros(capacity, dtype=torch.int32, device=dev),
            gate=torch.zeros(1, dtype=torch.int32, device=dev),
        )

    @property
    def capacity(self) -> int:
        return int(self.recompute_flag.shape[0])

    def reset(self) -> None:
        """Zero all contents (no reallocation -> no shape change)."""
        self.recompute_flag.zero_()
        self.kv_live.zero_()
        self.gate.zero_()

    def fill_request(
        self,
        *,
        seq_len: int,
        recompute_offsets: list[int] | torch.Tensor | None = None,
        reused_offsets: list[int] | torch.Tensor | None = None,
        gate: bool = False,
    ) -> None:
        """Overwrite contents in place for one request's logical positions.

        Args:
            seq_len: number of logical positions of this request. ``kv_live`` is
                set to 1 over ``[0, seq_len)`` (every token participates) unless
                ``reused_offsets`` narrows it.
            recompute_offsets: logical positions that are M (recompute) rows. If
                ``None`` *and* ``gate`` is False (Phase 2a), recompute gating is
                inert so this may be omitted.
            reused_offsets: if given, restrict ``kv_live`` to exactly these
                positions plus the recompute positions (Phase 2b sparse KV). If
                ``None`` every position in ``[0, seq_len)`` is live (Phase 2a).
            gate: whether to enable recompute-row gating (Phase 2b). Phase 2a
                passes False so the predicate reduces to standard causal.

        Note: fills positions ``[0, seq_len)``. Callers wiring a multi-request
        batch must offset positions into the shared tensors themselves; the
        single-request paged path (one FlexAttention mask over one sequence)
        uses position 0..seq_len directly.
        """
        if seq_len > self.capacity:
            raise ValueError(
                f"seq_len {seq_len} exceeds mask capacity {self.capacity}"
            )
        self.reset()

        if reused_offsets is None:
            # Phase 2a: every token participates -> all-ones causal-equivalent.
            self.kv_live[:seq_len] = 1
        else:
            idx = torch.as_tensor(
                reused_offsets, dtype=torch.long, device=self.kv_live.device
            )
            if idx.numel():
                self.kv_live[idx] = 1

        if recompute_offsets is not None:
            ridx = torch.as_tensor(
                recompute_offsets,
                dtype=torch.long,
                device=self.recompute_flag.device,
            )
            if ridx.numel():
                self.recompute_flag[ridx] = 1
                # Recomputed rows are themselves live KV.
                self.kv_live[ridx] = 1

        self.gate[0] = 1 if gate else 0


def build_legolink_mask_mod(tensors: FusionMaskTensors) -> MaskMod:
    """Build the LegoLink ``mask_mod`` over fixed metadata tensors, ONCE.

    The returned closure captures only ``tensors``' Tensors (fixed identity,
    fixed shape). It performs pure tensor lookups by ``q_idx`` / ``kv_idx`` and
    captures NO per-step Python scalar/list -- so FlexAttention reuses the same
    compiled kernel across requests/steps (DESIGN.md risk 1).

    The same returned object may be reused indefinitely; only the *contents* of
    ``tensors`` change between requests (via ``fill_request``).
    """
    recompute_flag = tensors.recompute_flag
    kv_live = tensors.kv_live
    gate = tensors.gate
    capacity = tensors.capacity

    def legolink_mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        # Standard causal over logical positions.
        causal = q_idx >= kv_idx

        # KV row participates? Clamp the index defensively: the paged
        # final_mask_mod already gates out-of-range kv via `is_valid`, but
        # clamping keeps this function safe to call standalone (tests/vmap).
        kv_in_range = (kv_idx >= 0) & (kv_idx < capacity)
        kv_ok = kv_in_range & (kv_live[kv_idx.clamp(0, capacity - 1)] != 0)

        # Recompute-row gating: inert when gate == 0 (Phase 2a, every row
        # forwarded == causal); active when gate == 1 (Phase 2b, M-only).
        q_in_range = (q_idx >= 0) & (q_idx < capacity)
        is_recompute = q_in_range & (
            recompute_flag[q_idx.clamp(0, capacity - 1)] != 0
        )
        gate_on = gate[0] != 0
        row_ok = (~gate_on) | is_recompute

        return causal & kv_ok & row_ok

    return legolink_mask_mod
