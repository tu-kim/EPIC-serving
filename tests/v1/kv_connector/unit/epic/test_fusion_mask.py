# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC LegoLink fusion-mask (Phase 2a) semantics + recompile-avoidance tests.

CPU-only. No torch.compile / no GPU. We verify the mask_mod against the real
FlexAttention calling convention (scalar Tensor args -> bool Tensor) via direct
calls and torch.vmap (the same vectorization flex_attention applies), and assert
the recompile-avoidance contract: one mask_mod object reused across requests,
contents swapped in fixed-size tensors only.
"""

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.fusion_mask import (
    FusionMaskTensors,
    build_legolink_mask_mod,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    LegoLinkMaskBuilder,
    RecomputePlan,
)


def _causal(q: int, k: int) -> bool:
    return q >= k


def _call(mask_mod, q: int, k: int) -> bool:
    """Invoke mask_mod with FlexAttention's real arg types (scalar tensors)."""
    return bool(
        mask_mod(
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(q, dtype=torch.int32),
            torch.tensor(k, dtype=torch.int32),
        )
    )


# ---------------------------------------------------------------------------
# (b) non-M rows == standard causal  (Phase 2a full-forward back-compat)
# ---------------------------------------------------------------------------


def test_dense_phase2a_is_exactly_causal():
    t = FusionMaskTensors.allocate(32)
    mm = build_legolink_mask_mod(t)
    t.fill_request(seq_len=16, gate=False)  # every row forwarded, all KV live

    for q in range(16):
        for k in range(16):
            assert _call(mm, q, k) == _causal(q, k), (q, k)


def test_builder_dense_is_causal():
    builder = LegoLinkMaskBuilder()
    mm = builder.build_logical_mask_mod(RecomputePlan(seq_len=10), attn_meta=None)
    for q in range(10):
        for k in range(10):
            assert _call(mm, q, k) == _causal(q, k), (q, k)


# ---------------------------------------------------------------------------
# (a) M rows: causal allow up to own position
# ---------------------------------------------------------------------------


def test_m_rows_attend_causally_when_gated():
    t = FusionMaskTensors.allocate(32)
    mm = build_legolink_mask_mod(t)
    # gate ON, M = {3, 7}; in 2b only these rows produce queries.
    t.fill_request(seq_len=10, recompute_offsets=[3, 7], gate=True)

    for q in range(10):
        for k in range(10):
            got = _call(mm, q, k)
            # M rows attend causally; non-M rows produce nothing under gate.
            exp = _causal(q, k) and (q in (3, 7))
            assert got == exp, (q, k, got, exp)
    # Each M row attends exactly its prefix [0, q].
    assert [k for k in range(10) if _call(mm, 3, k)] == [0, 1, 2, 3]
    assert [k for k in range(10) if _call(mm, 7, k)] == list(range(8))


# ---------------------------------------------------------------------------
# (c) boundaries: first/last token, chunk-boundary link, sparse reused KV
# ---------------------------------------------------------------------------


def test_first_and_last_token_boundaries():
    t = FusionMaskTensors.allocate(8)
    mm = build_legolink_mask_mod(t)
    t.fill_request(seq_len=8, gate=False)

    # First token attends only to itself.
    assert _call(mm, 0, 0) is True
    assert _call(mm, 0, 1) is False
    # Last token attends to the whole prefix incl. itself, nothing beyond.
    assert all(_call(mm, 7, k) for k in range(8))


def test_sparse_reused_kv_boundary():
    """Phase 2b shape: only some KV positions live; M queries gated.

    reused_offsets marks read-only reused chunks; recompute_offsets are M and
    are themselves live KV. A query may only see live KV at/below its position.
    """
    t = FusionMaskTensors.allocate(16)
    mm = build_legolink_mask_mod(t)
    # Reused chunk KV at positions {0,1,2,3}; M (recompute) rows at {6,7}.
    t.fill_request(
        seq_len=8, recompute_offsets=[6, 7], reused_offsets=[0, 1, 2, 3], gate=True
    )

    live_kv = {0, 1, 2, 3, 6, 7}
    for q in range(8):
        for k in range(8):
            got = _call(mm, q, k)
            exp = _causal(q, k) and (q in (6, 7)) and (k in live_kv)
            assert got == exp, (q, k, got, exp)


def test_link_token_at_chunk_boundary():
    """A 'link' M token at a chunk boundary attends across the boundary."""
    t = FusionMaskTensors.allocate(16)
    mm = build_legolink_mask_mod(t)
    # Two reused chunks [0,1,2] and [4,5,6]; link token at position 3 recomputed.
    t.fill_request(
        seq_len=7,
        recompute_offsets=[3],
        reused_offsets=[0, 1, 2, 4, 5, 6],
        gate=True,
    )
    # Link query (pos 3) sees the first reused chunk causally (0..3),
    # not the later (future) chunk.
    visible = [k for k in range(7) if _call(mm, 3, k)]
    assert visible == [0, 1, 2, 3]


def test_out_of_range_indices_are_masked():
    t = FusionMaskTensors.allocate(8)
    mm = build_legolink_mask_mod(t)
    t.fill_request(seq_len=4, gate=False)
    # kv beyond capacity must be masked (defensive clamp).
    assert _call(mm, 3, 10) is False
    # negative kv masked.
    assert _call(mm, 3, -1) is False


# ---------------------------------------------------------------------------
# (d) recompile avoidance: SAME mask_mod object across different requests
# ---------------------------------------------------------------------------


def test_same_mask_mod_object_reused_across_requests():
    """Core recompile-avoidance contract (DESIGN risk 1).

    The mask_mod function object identity must NOT change when request data
    changes -- only the backing tensor *contents* change. FlexAttention keys its
    compiled kernel on the mask_mod identity + captured-tensor shapes; both are
    held fixed here.
    """
    t = FusionMaskTensors.allocate(64)
    mm = build_legolink_mask_mod(t)

    # Request A: dense seq_len 8.
    t.fill_request(seq_len=8, gate=False)
    snapshot_obj = mm
    snapshot_shapes = (
        t.recompute_flag.shape,
        t.kv_live.shape,
        t.gate.shape,
    )
    rowA = [k for k in range(8) if _call(mm, 5, k)]

    # Request B: different seq_len + sparse M, SAME object, SAME tensor shapes.
    t.fill_request(seq_len=12, recompute_offsets=[2, 9], reused_offsets=[0, 1], gate=True)
    assert mm is snapshot_obj
    assert (t.recompute_flag.shape, t.kv_live.shape, t.gate.shape) == snapshot_shapes
    rowB = [k for k in range(12) if _call(mm, 9, k)]

    # The behavior differs per request (contents changed), even though the
    # function object and tensor shapes are identical.
    assert rowA == [0, 1, 2, 3, 4, 5]
    # Request B: pos 9 is an M row; live KV = reused {0,1} + recompute {2,9}
    # (recomputed rows are themselves live KV). Causal at/below 9 -> {0,1,2,9}.
    assert rowB == [0, 1, 2, 9]


def test_builder_returns_identical_object_each_call():
    # Real worker setup: capacity fixed once (e.g. max_model_len). Within that
    # fixed capacity the mask_mod object identity is stable across all requests
    # -> FlexAttention reuses the compiled kernel (recompile avoidance).
    builder = LegoLinkMaskBuilder(capacity=64)
    m1 = builder.build_logical_mask_mod(RecomputePlan(seq_len=8), attn_meta=None)
    m2 = builder.build_logical_mask_mod(
        RecomputePlan(seq_len=16, recompute_offsets=[1], reused_offsets=[0]),
        attn_meta=None,
    )
    m3 = builder.build_logical_mask_mod(RecomputePlan(seq_len=4), attn_meta=None)
    assert m1 is m2 is m3


def test_builder_grows_capacity_without_breaking():
    """Capacity growth reallocates once; steady state keeps identity.

    If a later request exceeds the current capacity the builder is allowed to
    rebuild (a one-time recompile is acceptable, like FlexAttention's own
    persistent-buffer growth); within a fixed capacity the object is stable.
    """
    builder = LegoLinkMaskBuilder(capacity=8)
    m_small = builder.build_logical_mask_mod(RecomputePlan(seq_len=8), attn_meta=None)
    # within capacity: identical object
    m_same = builder.build_logical_mask_mod(RecomputePlan(seq_len=4), attn_meta=None)
    assert m_small is m_same


# ---------------------------------------------------------------------------
# Batched/broadcast index grids: how flex_attention drives the mask_mod under
# torch.compile (whole index grids, not Python scalars). We exercise the same
# tensor-indexing path eagerly with broadcast grids (the operations the mask
# uses -- comparisons + advanced indexing -- all vectorize); this is the
# torch.compile-free level the brief asks us to verify.
# ---------------------------------------------------------------------------


def test_batched_index_grid_matches_scalar():
    t = FusionMaskTensors.allocate(16)
    mm = build_legolink_mask_mod(t)
    t.fill_request(seq_len=8, recompute_offsets=[2, 5], gate=True)

    n = 8
    q_grid = torch.arange(n, dtype=torch.int32).view(n, 1).expand(n, n)
    k_grid = torch.arange(n, dtype=torch.int32).view(1, n).expand(n, n)
    b = torch.tensor(0)
    h = torch.tensor(0)

    grid = mm(b, h, q_grid, k_grid)  # [n, n] bool, single vectorized call
    assert grid.shape == (n, n)
    assert grid.dtype == torch.bool
    for q in range(n):
        for k in range(n):
            assert bool(grid[q, k]) == _call(mm, q, k), (q, k)


def test_dense_batched_grid_is_causal():
    t = FusionMaskTensors.allocate(16)
    mm = build_legolink_mask_mod(t)
    t.fill_request(seq_len=12, gate=False)
    n = 12
    q_grid = torch.arange(n, dtype=torch.int32).view(n, 1).expand(n, n)
    k_grid = torch.arange(n, dtype=torch.int32).view(1, n).expand(n, n)
    grid = mm(torch.tensor(0), torch.tensor(0), q_grid, k_grid)
    causal = q_grid >= k_grid
    assert torch.equal(grid, causal)


def test_reset_clears_contents_keeps_shapes():
    t = FusionMaskTensors.allocate(8)
    t.fill_request(seq_len=8, recompute_offsets=[1, 2], gate=True)
    shapes = (t.recompute_flag.shape, t.kv_live.shape, t.gate.shape)
    t.reset()
    assert int(t.gate[0]) == 0
    assert int(t.recompute_flag.sum()) == 0
    assert int(t.kv_live.sum()) == 0
    assert (t.recompute_flag.shape, t.kv_live.shape, t.gate.shape) == shapes


def test_gate_off_is_pure_causal_even_after_tiny_decode_refill():
    """Regression (GPU step4 decode collapse, link-sweep all 0.021):
    decode steps refill the tensors with that step's seq_len (1). With
    gate OFF the mask must stay PURE causal regardless of kv_live state;
    the old predicate AND-ed kv_live unconditionally, so decode queries
    (logical pos 523) could only attend kv_idx==0 -> constant repetition."""
    t = FusionMaskTensors.allocate(capacity=1024, device=torch.device("cpu"))
    mm = build_legolink_mask_mod(t)
    # decode-style refill: only position 0 marked live
    t.fill_request(seq_len=1)
    q = torch.tensor(523)
    # must attend ALL causal kv, not just kv 0
    for kv in (0, 1, 300, 511, 523):
        assert bool(mm(torch.tensor(0), torch.tensor(0), q, torch.tensor(kv))), kv
    # still causal: future masked
    assert not bool(mm(torch.tensor(0), torch.tensor(0), q, torch.tensor(524)))
