# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC sparse-forward STRICT-SUBSET (partial-M) accounting invariants (CPU).

The previously-existing sparse tests only exercised either ``link == 0`` /
``link == 8`` on small prompts or a *contiguous* M (``list(range(...))`` in
``test_scheduler_core_patch``). The musique GPU collapse showed up only for
``link < chunk_size`` with *multiple* non-prefix chunks -- i.e. a genuinely
NON-CONTIGUOUS M that is a strict subset of ``[0, N)``. That path's accounting
(external = |A| + |B| counted as computed, ``num_scheduled`` overridden to |M|,
``num_computed`` advanced by ``N - external``) is exactly where a double-count
breaks a runner-derived value.

This module drives the REAL connector -> scheduler-override -> runner
row-edit/positions/slot-mapping derivation on CPU and asserts, for several
strict-subset shapes, the invariants the runner contract depends on:

  I1  every logical position 0..N-1 is covered by EXACTLY ONE writer
      (M-forward OR connector B-load); no holes, no unintended double writes.
  I2  ``num_scheduled_tokens == len(M)``                (runner_sparse contract).
  I3  after the step ``num_computed == N`` EXACTLY      (external + advance == N).
  I4  the M-forward slot_mapping covers exactly the M positions' slots, and the
      B-load slots cover exactly the reused positions' slots; their union is the
      whole sequence and M overwrites the B/M overlap (intended).
  I5  the input-id gather row i picks ``token_ids[M[i]]``
      (positions_np + req*width path).
  I6  ``query_start_loc`` / ``logits_indices = qsl[1:]-1`` point at M's last row,
      whose logical position is N-1.
  I7  the derived per-request optimistic seq-len sizing equals N (NOT
      ``external + |M|``): this is the value that feeds ``max_seq_len`` /
      ``seq_lens_cpu_upper_bound`` into the attention backend, and the
      double-count here was the latent sizing bug fixed alongside this test.

All math is the SAME math the runner runs (numpy positions, slot ids from the
block table, ``build_sparse_row_edits``), so a break here is a real runner break.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    EpicChunkStore,
    EpicSchedulerIndex,
    StoredChunk,
    hash_chunk_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
    EpicConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    EpicSelection,
    LegoLinkRecompute,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.runner_sparse import (
    build_sparse_row_edits,
)

BLOCK = 16


# --------------------------------------------------------------------------
# Minimal SCHEDULER-role connector + fake scheduler output (no GPU / VllmConfig).
# --------------------------------------------------------------------------


class _Req:
    def __init__(self, rid: str, toks: list[int]):
        self.request_id = rid
        self.prompt_token_ids = toks


@dataclass
class _NewReq:
    req_id: str
    prompt_token_ids: list[int]
    block_ids: list[list[int]]


@dataclass
class _SchedOut:
    scheduled_new_reqs: list = field(default_factory=list)
    num_scheduled_tokens: dict = field(default_factory=dict)


class _LiveStoreIndex(EpicSchedulerIndex):
    """Scheduler index that reads membership through a live worker store, so
    chunks pre-stored before build are visible (models "a worker already saved
    these chunks")."""

    def __init__(self, store: EpicChunkStore):
        super().__init__(
            capacity_bytes=10**8,
            num_layers=1,
            num_kv_heads=1,
            head_size=1,
            cache_dtype_size=4,
        )
        self._backing = store

    def contains(self, h: str) -> bool:
        return self._backing.contains(h) or super().contains(h)

    def get_length(self, h: str):
        ln = self._backing.get_length(h)
        return ln if ln is not None else super().get_length(h)


def _make_connector(*, link: int, chunk_size: int, store: EpicChunkStore):
    c = object.__new__(EpicConnector)
    c._block_size = BLOCK
    c._chunk_size = chunk_size
    c._store = None
    c._index = _LiveStoreIndex(store)
    c._matched_prefix = {}
    c._non_prefix = {}
    c._loads_pending = {}
    c._selections = {}
    c._sparse_reqs = {}
    c._sparse_forward = True
    c._link_tokens = link
    c._selection = EpicSelection()
    c._recompute = LegoLinkRecompute(num_link_tokens=link, phase1_dense=False)
    c._fusion_enabled = False
    c._debug_counters = False
    return c


def _store_chunk(store: EpicChunkStore, tokens: list[int], old_positions: list[int]):
    h = hash_chunk_tokens(tokens)
    sc = StoredChunk(
        chunk_hash=h,
        length=len(tokens),
        old_positions=torch.tensor(old_positions, dtype=torch.int64),
    )
    sc.k_per_layer["l0"] = torch.zeros(len(tokens), 1, 1)
    sc.v_per_layer["l0"] = torch.zeros(len(tokens), 1, 1)
    store.put(sc)
    return h


def _slots_for(block_ids: list[int], positions) -> list[int]:
    return [block_ids[p // BLOCK] * BLOCK + p % BLOCK for p in positions]


# --------------------------------------------------------------------------
# Core invariant harness: build a prompt of `n_chunks`, with the
# `nonprefix_idx` chunks cached at NON-prefix offsets, run the whole
# connector->scheduler-override->runner derivation, return everything needed.
# --------------------------------------------------------------------------


def _run_case(
    *,
    chunk_size: int,
    link: int,
    n_chunks: int,
    nonprefix_idx: list[int],
    same_offset_reuse: bool = True,
):
    """Returns a dict of the derived runner values for the single sparse req."""
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _make_connector(link=link, chunk_size=chunk_size, store=store)

    # Build prompt chunks with distinct content per chunk index.
    chunks = [
        list(range(1000 * (i + 1), 1000 * (i + 1) + chunk_size))
        for i in range(n_chunks)
    ]
    # Cache exactly the non-prefix chunks (at a DIFFERENT stored offset to
    # exercise PIC, but PIC fidelity is a worker concern; here we only check
    # accounting, so old_positions value is irrelevant to these invariants).
    cached_hashes = set()
    for i in nonprefix_idx:
        old_off = 0 if same_offset_reuse else (i + 5) * chunk_size
        h = _store_chunk(
            store, chunks[i], list(range(old_off, old_off + chunk_size))
        )
        cached_hashes.add(h)

    tokens: list[int] = []
    for ch in chunks:
        tokens += ch
    n = len(tokens)
    nblocks = (n + BLOCK - 1) // BLOCK
    block_ids = list(range(nblocks))

    # --- scheduler: external + num_new ---
    ext, _ = c.get_num_new_matched_tokens(_Req("r0", tokens), 0)
    sched_num_new = max(0, n - ext)  # scheduler.py:665 num_new = N - num_computed

    # --- build_connector_meta: emits sparse plan + B loads ---
    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, [block_ids])],
        num_scheduled_tokens={"r0": sched_num_new},
    )
    meta = c.build_connector_meta(sout)
    assert len(meta.sparse) == 1, "strict-subset prompt must emit a sparse plan"
    sp = meta.sparse[0]
    M = list(sp.sparse_positions)
    advance = sp.computed_advance

    # --- scheduler override (mirror _apply_epic_sparse_overrides) ---
    num_scheduled = len(M)  # overridden to |M|

    # --- runner row edits ---
    cu_num_tokens = [num_scheduled]
    edits = build_sparse_row_edits(
        ["r0"], cu_num_tokens, {"r0": M}, {"r0": sp.full_seq_len}
    )
    assert len(edits) == 1
    edit = edits[0]

    # --- runner positions_np (after EPIC overwrite) ---
    num_computed_runner = ext  # NewRequestData snapshot == external
    req_indices = np.repeat(np.arange(1), [num_scheduled])
    positions_np = (
        np.array([num_computed_runner])[req_indices]
        + np.arange(num_scheduled)
    )
    positions_np[edit.row_start : edit.row_end] = edit.positions

    # --- token-id gather: row i -> token_ids[M[i]] ---
    width = 1 << 20  # >> n, mirrors token_ids_cpu width
    token_indices = positions_np + req_indices * width
    gathered = [tokens[ti] for ti in token_indices]  # req 0 -> ti == position

    # --- slot mapping (M rows) + connector B-load slots ---
    m_slots = _slots_for(block_ids, positions_np)
    load_writes: dict[int, int] = {}  # logical pos -> 'load' marker (slot)
    for ld in meta.loads:
        for ch in ld.chunks:
            for p in range(ch.new_pos_start, ch.new_pos_start + ch.length):
                load_writes[p] = block_ids[p // BLOCK] * BLOCK + p % BLOCK

    # --- coverage: final writer per position (load first, M overwrites) ---
    final_writer: dict[int, str] = {p: "load" for p in load_writes}
    for p in positions_np:
        final_writer[int(p)] = "M"

    # --- query_start_loc / logits_indices ---
    qsl = [0, num_scheduled]
    logits_row = qsl[-1] - 1
    logits_pos = int(positions_np[logits_row])

    # --- optimistic seq len (post-fix should equal N) ---
    optimistic_raw = num_computed_runner + num_scheduled  # the pre-fix value
    optimistic_fixed = edit.seq_len  # the value the fix writes

    return {
        "n": n,
        "ext": ext,
        "sched_num_new": sched_num_new,
        "M": M,
        "num_scheduled": num_scheduled,
        "advance": advance,
        "positions": list(positions_np),
        "gathered": gathered,
        "tokens": tokens,
        "m_slots": m_slots,
        "load_writes": load_writes,
        "final_writer": final_writer,
        "logits_row": logits_row,
        "logits_pos": logits_pos,
        "optimistic_raw": optimistic_raw,
        "optimistic_fixed": optimistic_fixed,
        "full_seq_len": sp.full_seq_len,
        "block_ids": block_ids,
    }


# The shapes that previously went UNTESTED: strict-subset, MULTI-chunk M.
# (chunk_size, link, n_chunks, nonprefix_idx)
STRICT_SUBSET_CASES = [
    # 2 non-prefix chunks, link one short of chunk_size (musique scaled down).
    (32, 31, 3, [1, 2]),
    # smaller chunk to exercise more blocks-per-chunk granularity.
    (32, 8, 3, [1, 2]),
    # many non-prefix chunks (the musique 18-chunk shape, scaled).
    (16, 15, 6, [1, 2, 3, 4, 5]),
    # link in the middle.
    (32, 16, 4, [1, 2, 3]),
    # leading chunk also cached but non-prefix (gap before it forces non-prefix).
    (32, 24, 5, [2, 3, 4]),
]


@pytest.mark.parametrize("chunk_size,link,n_chunks,nonprefix_idx", STRICT_SUBSET_CASES)
def test_strict_subset_invariants(chunk_size, link, n_chunks, nonprefix_idx):
    r = _run_case(
        chunk_size=chunk_size,
        link=link,
        n_chunks=n_chunks,
        nonprefix_idx=nonprefix_idx,
    )
    n = r["n"]

    # Sanity: this really is a STRICT subset (the whole point of the test).
    assert len(r["M"]) < n, "expected a partial-M (strict subset) case"
    assert r["M"] == sorted(set(r["M"])), "M must be sorted + unique"

    # I1 + I4: coverage -- every position has exactly one final writer; union==N.
    assert set(r["final_writer"]) == set(range(n)), "coverage holes / overflow"
    # M overwrites the B/M overlap (intended), B-only positions stay 'load'.
    m_set = set(r["M"])
    for p in range(n):
        expect = "M" if p in m_set else "load"
        assert r["final_writer"][p] == expect, f"pos {p} wrong final writer"
    # Pure-load positions exist (otherwise it would degenerate to dense).
    pure_load = [p for p in range(n) if r["final_writer"][p] == "load"]
    assert pure_load, "strict subset must leave >=1 pure-load B position"

    # I2: runner contract num_scheduled == len(M).
    assert r["num_scheduled"] == len(r["M"])

    # I3: num_computed converges to N exactly (external + advance == N).
    assert r["ext"] + r["advance"] == n
    # advance is N - external, NOT len(M).
    assert r["advance"] == n - r["ext"]
    assert r["advance"] != len(r["M"]) or len(r["M"]) == n - r["ext"]

    # I5: input-id gather row i == token_ids[M[i]].
    assert r["gathered"] == [r["tokens"][p] for p in r["M"]]

    # I6: logits row is M's last row, at logical position N-1.
    assert r["logits_row"] == len(r["M"]) - 1
    assert r["logits_pos"] == n - 1
    assert r["M"][-1] == n - 1
    assert r["full_seq_len"] == n

    # I4 (slots): M slots are exactly the slots of M positions.
    assert r["m_slots"] == _slots_for(r["block_ids"], r["M"])

    # I7: the optimistic seq-len used for attention sizing must be N, not the
    # double-counted ``external + |M|``. The pre-fix raw value is strictly larger
    # whenever M overlaps B (it always does in a strict subset), which is exactly
    # the latent sizing bug; the fix writes ``edit.seq_len == N``.
    assert r["optimistic_fixed"] == n
    assert r["optimistic_raw"] > n  # demonstrates the double-count being fixed.


def test_link_equals_chunk_size_is_dense_and_safe():
    """Control: link == chunk_size makes M == every token (contiguous, == N).

    This is the GPU "link=256" baseline that already worked. The override still
    fires (M is emitted) but M == [0, N) so positions are contiguous and every B
    position is recomputed (overwritten) -- the loaded B KV is never read. The
    invariants still hold and optimistic seq len already equals N here because
    advance == num_new in the degenerate case is NOT true (advance == N-external),
    but |M| == N so the runner forwards the whole sequence.
    """
    r = _run_case(chunk_size=32, link=32, n_chunks=3, nonprefix_idx=[1, 2])
    n = r["n"]
    assert r["M"] == list(range(n)), "link==chunk_size -> M is the whole sequence"
    assert r["num_scheduled"] == n
    assert r["ext"] + r["advance"] == n
    # Coverage trivially complete; every position is an M (recompute) row.
    assert all(v == "M" for v in r["final_writer"].values())
    # optimistic fixed is N; raw double-counts (external + N).
    assert r["optimistic_fixed"] == n
    assert r["optimistic_raw"] == r["ext"] + n


def test_advance_is_n_minus_external_not_len_m():
    """Pin the load-bearing distinction: advancing num_computed by |M| (instead
    of N - external) would overshoot N because M overlaps the B positions that
    external already counted as computed."""
    r = _run_case(chunk_size=32, link=31, n_chunks=3, nonprefix_idx=[1, 2])
    n, ext = r["n"], r["ext"]
    # |M| overlaps B -> advancing by |M| from external overshoots N.
    assert ext + len(r["M"]) > n
    # the connector's advance lands exactly on N.
    assert ext + r["advance"] == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
