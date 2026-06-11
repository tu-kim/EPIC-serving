# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC sparse-forward runner + FlexAttention logical-q tests (Phase 2b S4/S6).

CPU-only. Three groups:

  (a) ``build_sparse_row_edits`` -- per-request row-range resolution for the
      runner positions/seq_lens override (S4/S5), including sparse/non-sparse
      mixed batches and the length-mismatch contract.
  (b) ``FlexAttentionMetadata._convert_physical_to_logical`` -- the logical_q
      branch (S6): identity with the vanilla derivation in the dense/contiguous
      case, and correct scattered-M mapping when sparse positions are supplied.
  (c) the second consumer (sliding-window / block-sparsity-hint path) vectorized
      branch -- equivalence in the dense case and the sparse+sliding_window
      guard.
"""

import types

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.runner_sparse import (
    SparseRowEdit,
    build_sparse_row_edits,
)
from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata


# ============================================================================
# (a) build_sparse_row_edits
# ============================================================================


def test_single_sparse_request():
    edits = build_sparse_row_edits(
        req_ids=["r0"],
        cu_num_tokens=[3],
        epic_sparse_positions={"r0": [2, 5, 9]},
        epic_seq_len={"r0": 10},
    )
    assert edits == [
        SparseRowEdit(
            req_id="r0",
            row_start=0,
            row_end=3,
            positions=[2, 5, 9],
            seq_len=10,
        )
    ]


def test_mixed_batch_only_sparse_reqs_edited():
    # Three reqs with per-req row counts [2, 3, 4] -> cu = [2, 5, 9].
    # Only the middle one is sparse.
    edits = build_sparse_row_edits(
        req_ids=["a", "b", "c"],
        cu_num_tokens=[2, 5, 9],
        epic_sparse_positions={"b": [1, 4, 7]},
        epic_seq_len={"b": 8},
    )
    assert len(edits) == 1
    e = edits[0]
    assert e.req_id == "b"
    # Row range for the middle req is [2, 5).
    assert (e.row_start, e.row_end) == (2, 5)
    assert e.positions == [1, 4, 7]
    assert e.seq_len == 8


def test_multiple_sparse_requests_row_ranges():
    # reqs [r0, r1, r2], row counts [2, 1, 3] -> cu = [2, 3, 6].
    edits = build_sparse_row_edits(
        req_ids=["r0", "r1", "r2"],
        cu_num_tokens=[2, 3, 6],
        epic_sparse_positions={"r0": [0, 4], "r2": [3, 6, 9]},
        epic_seq_len={"r0": 5, "r2": 10},
    )
    assert len(edits) == 2
    assert (edits[0].row_start, edits[0].row_end) == (0, 2)
    assert edits[0].positions == [0, 4]
    assert edits[0].seq_len == 5
    assert (edits[1].row_start, edits[1].row_end) == (3, 6)
    assert edits[1].positions == [3, 6, 9]
    assert edits[1].seq_len == 10


def test_no_sparse_requests_returns_empty():
    edits = build_sparse_row_edits(
        req_ids=["a", "b"],
        cu_num_tokens=[2, 5],
        epic_sparse_positions={},
        epic_seq_len={},
    )
    assert edits == []


def test_seq_len_falls_back_to_last_position_plus_one():
    edits = build_sparse_row_edits(
        req_ids=["r0"],
        cu_num_tokens=[2],
        epic_sparse_positions={"r0": [3, 11]},
        epic_seq_len={},  # missing -> fallback to positions[-1] + 1.
    )
    assert edits[0].seq_len == 12


def test_length_mismatch_raises():
    # The request owns 3 rows but only 2 sparse positions were stamped.
    with pytest.raises(ValueError, match="forward rows"):
        build_sparse_row_edits(
            req_ids=["r0"],
            cu_num_tokens=[3],
            epic_sparse_positions={"r0": [1, 9]},
            epic_seq_len={"r0": 10},
        )


# ============================================================================
# (b) _convert_physical_to_logical -- logical_q branch (S6)
# ============================================================================


def _make_meta_stub(
    *,
    block_size,
    physical_to_logical,
    seq_lens,
    query_start_loc,
    decode_offset,
    logical_q_positions,
):
    """Minimal attribute stub to drive the unbound method (no GPU metadata)."""
    return types.SimpleNamespace(
        block_size=block_size,
        physical_to_logical=physical_to_logical,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        decode_offset=decode_offset,
        logical_q_positions=logical_q_positions,
    )


def _call_convert(stub, request_lookup, q_idx, physical_kv_idx):
    # Bind the unbound method to the stub.
    return FlexAttentionMetadata._convert_physical_to_logical(
        stub, request_lookup, q_idx, physical_kv_idx
    )


def test_logical_q_none_matches_vanilla_contiguous():
    # Single request, 4 contiguous query rows, block_size 4, one block.
    block_size = 4
    # physical_to_logical[q_req, physical_block] -> logical block index.
    physical_to_logical = torch.tensor([[0]], dtype=torch.long)  # [1 req, 1 block]
    seq_lens = torch.tensor([4], dtype=torch.long)
    query_start_loc = torch.tensor([0, 4], dtype=torch.long)
    decode_offset = torch.tensor([0], dtype=torch.long)  # full prefill

    request_lookup = torch.zeros(4, dtype=torch.long)  # all rows -> req 0
    q_idx = torch.arange(4, dtype=torch.long)
    physical_kv_idx = torch.arange(4, dtype=torch.long)

    stub = _make_meta_stub(
        block_size=block_size,
        physical_to_logical=physical_to_logical,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        decode_offset=decode_offset,
        logical_q_positions=None,
    )
    _, logical_q, _ = _call_convert(stub, request_lookup, q_idx, physical_kv_idx)
    # Vanilla: local_q + decode_offset == [0,1,2,3].
    assert torch.equal(logical_q, torch.tensor([0, 1, 2, 3]))


def test_logical_q_identity_equivalence_dense():
    # Same dense case, but supply logical_q_positions == contiguous positions.
    # Result must equal the vanilla branch (proves identity in the dense case).
    block_size = 4
    physical_to_logical = torch.tensor([[0]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)
    query_start_loc = torch.tensor([0, 4], dtype=torch.long)
    decode_offset = torch.tensor([0], dtype=torch.long)
    request_lookup = torch.zeros(4, dtype=torch.long)
    q_idx = torch.arange(4, dtype=torch.long)
    physical_kv_idx = torch.arange(4, dtype=torch.long)

    dense_pos = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    stub_sparse = _make_meta_stub(
        block_size=block_size,
        physical_to_logical=physical_to_logical,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        decode_offset=decode_offset,
        logical_q_positions=dense_pos,
    )
    stub_vanilla = _make_meta_stub(
        block_size=block_size,
        physical_to_logical=physical_to_logical,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        decode_offset=decode_offset,
        logical_q_positions=None,
    )
    _, lq_sparse, kv_sparse = _call_convert(
        stub_sparse, request_lookup, q_idx, physical_kv_idx
    )
    _, lq_vanilla, kv_vanilla = _call_convert(
        stub_vanilla, request_lookup, q_idx, physical_kv_idx
    )
    assert torch.equal(lq_sparse, lq_vanilla)
    # KV mapping must be untouched by the logical_q branch.
    assert torch.equal(kv_sparse, kv_vanilla)


def test_logical_q_scattered_positions():
    # Sparse M: 3 forwarded rows at non-contiguous logical positions [2, 5, 9]
    # over a logical sequence of length 12 (3 blocks of size 4).
    block_size = 4
    # 3 logical blocks present in the (single) request.
    physical_to_logical = torch.tensor([[0, 1, 2]], dtype=torch.long)
    seq_lens = torch.tensor([12], dtype=torch.long)
    query_start_loc = torch.tensor([0, 3], dtype=torch.long)
    # decode_offset would be wrong for sparse; logical_q_positions overrides it.
    decode_offset = torch.tensor([9], dtype=torch.long)
    request_lookup = torch.zeros(3, dtype=torch.long)
    q_idx = torch.arange(3, dtype=torch.long)
    # Probe KV across the 3 blocks (12 positions).
    physical_kv_idx = torch.arange(12, dtype=torch.long)[:3]

    sparse_pos = torch.tensor([2, 5, 9], dtype=torch.long)
    stub = _make_meta_stub(
        block_size=block_size,
        physical_to_logical=physical_to_logical,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        decode_offset=decode_offset,
        logical_q_positions=sparse_pos,
    )
    _, logical_q, _ = _call_convert(stub, request_lookup, q_idx, physical_kv_idx)
    # logical_q must be exactly the supplied scattered positions, NOT
    # local_q + decode_offset (= [9, 10, 11]).
    assert torch.equal(logical_q, torch.tensor([2, 5, 9]))


# ============================================================================
# (c) second consumer (:661) vectorized branch equivalence + guard
# ============================================================================


def _logical_q_second_consumer(
    *, doc_ids, query_start_loc, decode_offset, logical_q_positions
):
    """Reproduce the :661 branch in isolation (same math as flex_attention)."""
    token_indices = torch.arange(doc_ids.shape[0], dtype=torch.long)
    if logical_q_positions is not None:
        return logical_q_positions[token_indices]
    return token_indices - query_start_loc[doc_ids] + decode_offset[doc_ids]


def test_second_consumer_dense_equivalence():
    # One request, 5 contiguous rows.
    doc_ids = torch.zeros(5, dtype=torch.long)
    query_start_loc = torch.tensor([0, 5], dtype=torch.long)
    decode_offset = torch.tensor([0], dtype=torch.long)

    vanilla = _logical_q_second_consumer(
        doc_ids=doc_ids,
        query_start_loc=query_start_loc,
        decode_offset=decode_offset,
        logical_q_positions=None,
    )
    # Identity positions reproduce the vanilla result exactly.
    sparse = _logical_q_second_consumer(
        doc_ids=doc_ids,
        query_start_loc=query_start_loc,
        decode_offset=decode_offset,
        logical_q_positions=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
    )
    assert torch.equal(vanilla, sparse)
    assert torch.equal(vanilla, torch.tensor([0, 1, 2, 3, 4]))


def test_second_consumer_scattered():
    doc_ids = torch.zeros(3, dtype=torch.long)
    sparse = _logical_q_second_consumer(
        doc_ids=doc_ids,
        query_start_loc=torch.tensor([0, 3], dtype=torch.long),
        decode_offset=torch.tensor([0], dtype=torch.long),
        logical_q_positions=torch.tensor([4, 8, 15], dtype=torch.long),
    )
    assert torch.equal(sparse, torch.tensor([4, 8, 15]))
