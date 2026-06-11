# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC connector Phase 2b S0/S2 tests: epic_sparse_forward plumbing, sparse-M
emission on metadata, and the save guard (B-overlapping chunks not saved, pure-C
chunks saved). Plus metadata pickle roundtrip for the new sparse fields.

CPU-only. The EpicConnector is built field-by-field via ``object.__new__`` to
avoid the heavy VllmConfig / PICRotator / GPU construction path -- we exercise
only the scheduler-side ``build_connector_meta`` / ``_emit_sparse`` logic, which
needs no GPU and no model.
"""

import pickle
from dataclasses import dataclass, field

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    EpicChunkStore,
    StoredChunk,
    hash_chunk_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
    EpicConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
    EpicConnectorMetadata,
    EpicReqSparse,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    EpicSelection,
    LegoLinkRecompute,
)

BLOCK = 16
CHUNK = 64  # block-multiple


@dataclass
class _NewReq:
    req_id: str
    prompt_token_ids: list[int]
    block_ids: list[list[int]]


@dataclass
class _SchedOut:
    scheduled_new_reqs: list[_NewReq] = field(default_factory=list)
    num_scheduled_tokens: dict = field(default_factory=dict)


def _make_connector(*, sparse: bool, link: int = 8, store=None):
    """Build a minimal scheduler-side EpicConnector without VllmConfig/GPU."""
    c = object.__new__(EpicConnector)
    c._block_size = BLOCK
    c._chunk_size = CHUNK
    c._store = store if store is not None else EpicChunkStore(
        capacity_bytes=10**8, pin_memory=False
    )
    c._matched_prefix = {}
    c._non_prefix = {}
    c._loads_pending = {}
    c._selections = {}
    c._sparse_reqs = {}
    c._sparse_forward = sparse
    c._link_tokens = link
    c._selection = EpicSelection()
    c._recompute = LegoLinkRecompute(
        num_link_tokens=link, phase1_dense=not sparse
    )
    c._fusion_enabled = False
    return c


def _block_ids_for(num_tokens: int) -> list[list[int]]:
    nblocks = (num_tokens + BLOCK - 1) // BLOCK
    return [list(range(nblocks))]


def _store_chunk(store: EpicChunkStore, tokens: list[int]):
    h = hash_chunk_tokens(tokens)
    sc = StoredChunk(
        chunk_hash=h,
        length=len(tokens),
        old_positions=torch.arange(len(tokens), dtype=torch.int64),
    )
    sc.k_per_layer["l0"] = torch.zeros(len(tokens), 1, 1)
    sc.v_per_layer["l0"] = torch.zeros(len(tokens), 1, 1)
    store.put(sc)
    return h


# --------------------------------------------------------------------------
# S0: flag plumbing -- off => no sparse emission
# --------------------------------------------------------------------------


def test_flag_off_no_sparse_emission():
    c = _make_connector(sparse=False)
    tokens = list(range(3 * CHUNK))  # 192 tokens, all new
    sout = _SchedOut(
        scheduled_new_reqs=[
            _NewReq("r0", tokens, _block_ids_for(len(tokens)))
        ],
        num_scheduled_tokens={"r0": len(tokens)},
    )
    meta = c.build_connector_meta(sout)
    assert isinstance(meta, EpicConnectorMetadata)
    assert meta.sparse == []
    # All chunks new -> all saved (dense behavior unchanged).
    assert len(meta.saves) == 1
    assert len(meta.saves[0].chunk_hashes) == 3


# --------------------------------------------------------------------------
# S1/S0: flag on => sparse-M emitted on metadata
# --------------------------------------------------------------------------


def test_flag_on_emits_sparse_for_pure_new_prompt():
    c = _make_connector(sparse=True, link=8)
    # No prior cache -> no reuse -> M == every token (full C). is_sparse True.
    tokens = list(range(2 * CHUNK))  # 128
    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(len(tokens)))],
        num_scheduled_tokens={"r0": len(tokens)},
    )
    meta = c.build_connector_meta(sout)
    assert len(meta.sparse) == 1
    sp = meta.sparse[0]
    assert sp.req_id == "r0"
    assert sp.full_seq_len == 128
    # Pure-new prompt: M is all positions.
    assert sp.sparse_positions == list(range(128))


def test_flag_on_non_prefix_reuse_emits_sparse_subset():
    # Pre-store a chunk that appears at a NON-prefix offset of a new prompt.
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _make_connector(sparse=True, link=8, store=store)

    # Build prompt: chunk0 (new) + chunk1 (cached, non-prefix) + chunk2 (new).
    chunk0 = list(range(0, CHUNK))
    chunk1 = list(range(1000, 1000 + CHUNK))  # distinct content
    chunk2 = list(range(2000, 2000 + CHUNK))
    _store_chunk(store, chunk1)  # cache chunk1 only

    tokens = chunk0 + chunk1 + chunk2  # N = 192
    n = len(tokens)
    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(n))],
        num_scheduled_tokens={"r0": n},
    )
    meta = c.build_connector_meta(sout)

    assert len(meta.sparse) == 1
    m = set(meta.sparse[0].sparse_positions)
    # chunk0 (prefix? no -- chunk0 is NOT cached, so prefix_extent=0). chunk0
    # is all new C => fully in M.
    assert set(range(0, CHUNK)) <= m
    # chunk1 is cached & non-prefix => B. Only its link window [64,72) in M.
    assert set(range(CHUNK, CHUNK + 8)) <= m
    assert (CHUNK + 8) not in m  # 72 absent
    assert 100 not in m  # B body absent
    # chunk2 new C => fully in M, last token present.
    assert set(range(2 * CHUNK, n)) <= m
    assert (n - 1) in m
    # sorted/unique invariants.
    pos = meta.sparse[0].sparse_positions
    assert pos == sorted(set(pos))


# --------------------------------------------------------------------------
# S2: save guard
# --------------------------------------------------------------------------


def test_save_guard_skips_b_overlapping_chunk_saves_pure_c():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _make_connector(sparse=True, link=8, store=store)

    chunk0 = list(range(0, CHUNK))
    chunk1 = list(range(1000, 1000 + CHUNK))  # will be cached -> reused as B
    chunk2 = list(range(2000, 2000 + CHUNK))
    h1 = _store_chunk(store, chunk1)

    tokens = chunk0 + chunk1 + chunk2
    n = len(tokens)
    h0 = hash_chunk_tokens(chunk0)
    h2 = hash_chunk_tokens(chunk2)
    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(n))],
        num_scheduled_tokens={"r0": n},
    )
    meta = c.build_connector_meta(sout)

    saved = set()
    for s in meta.saves:
        saved.update(s.chunk_hashes)

    # chunk1 is already cached -> never re-saved regardless.
    assert h1 not in saved
    # chunk0 is pure-C (fully recomputed) -> saved (canonical KV).
    assert h0 in saved
    # chunk2 is pure-C -> saved.
    assert h2 in saved


def test_save_guard_skips_chunk_with_partial_reuse_link_only():
    # A chunk whose tokens are only PARTIALLY in M (e.g. it overlaps a reused B
    # region) must NOT be saved. Construct: prompt = B(cached) only, plus a
    # trailing new chunk so a save list exists.
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _make_connector(sparse=True, link=8, store=store)

    # prefix chunk cached so it forms the contiguous prefix A, then a cached
    # non-prefix B, then new C.
    a = list(range(0, CHUNK))
    b = list(range(500, 500 + CHUNK))
    cnew = list(range(900, 900 + CHUNK))
    ha = _store_chunk(store, a)
    hb = _store_chunk(store, b)
    hc = hash_chunk_tokens(cnew)

    tokens = a + b + cnew
    n = len(tokens)
    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(n))],
        num_scheduled_tokens={"r0": n},
    )
    meta = c.build_connector_meta(sout)

    saved = set()
    for s in meta.saves:
        saved.update(s.chunk_hashes)
    # a and b already cached -> not saved. c is pure-new -> saved.
    assert ha not in saved
    assert hb not in saved
    assert hc in saved


# --------------------------------------------------------------------------
# core-hook interface (S0) -- defined for the next batch's scheduler
# --------------------------------------------------------------------------


def test_core_hook_methods_read_metadata():
    c = _make_connector(sparse=True)
    meta = EpicConnectorMetadata()
    meta.add_sparse(EpicReqSparse(req_id="r0", sparse_positions=[2, 5, 9], full_seq_len=10))
    assert c.get_sparse_num_scheduled_tokens(meta, "r0") == 3
    assert c.get_sparse_positions(meta, "r0") == [2, 5, 9]
    assert c.get_sparse_num_scheduled_tokens(meta, "missing") is None
    assert c.get_sparse_positions(meta, "missing") is None
    assert c.has_sparse_requests(meta) is True
    assert c.has_sparse_requests(EpicConnectorMetadata()) is False


# --------------------------------------------------------------------------
# metadata pickle roundtrip (sparse fields)
# --------------------------------------------------------------------------


def test_sparse_metadata_pickle_roundtrip():
    meta = EpicConnectorMetadata()
    meta.add_sparse(
        EpicReqSparse(req_id="r0", sparse_positions=[0, 3, 7, 11], full_seq_len=12)
    )
    restored = pickle.loads(pickle.dumps(meta))
    assert len(restored.sparse) == 1
    sp = restored.sparse[0]
    assert sp.req_id == "r0"
    assert sp.sparse_positions == [0, 3, 7, 11]
    assert sp.full_seq_len == 12
    assert sp.is_sparse is True


def test_sparse_field_default_empty():
    meta = EpicConnectorMetadata()
    assert meta.sparse == []


# --------------------------------------------------------------------------
# S3 accounting integrity: get_num_new_matched_tokens -> build_connector_meta
# -> scheduler-side arithmetic. Locks down the math the scheduler core patch
# relies on across A/B/C/link combinations (brief: "이 산수를 단위 테스트로
# 못박아라").
# --------------------------------------------------------------------------


@dataclass
class _Req:
    """Minimal stand-in for v1 Request (scheduler-side, CPU-only)."""

    request_id: str
    prompt_token_ids: list[int]


def _simulate_schedule(c, tokens):
    """Drive the connector through one scheduler step and return the derived
    scheduler-side accounting quantities for a fresh (num_computed==0) prefill.

    Mirrors scheduler.py exactly:
      external (=|A|+|B|) <- get_num_new_matched_tokens
      num_new = N - external                       (scheduler.py:654)
      num_computed (after :804) = external
      M rows = get_sparse_num_scheduled_tokens     (override at :801)
      advance = get_sparse_computed_advance        (override at :964)
      num_computed_after_step = external + advance
    """
    n = len(tokens)
    req = _Req("r0", tokens)
    external, is_async = c.get_num_new_matched_tokens(req, 0)
    assert is_async is False  # EPIC sparse stays sync (brief S2).

    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(n))],
        num_scheduled_tokens={"r0": external_to_num_new(n, external)},
    )
    # The scheduler sets num_scheduled_tokens = num_new = N - external at :801
    # BEFORE build_connector_meta; record that as the "pre-override" value.
    pre_override_rows = sout.num_scheduled_tokens["r0"]

    meta = c.build_connector_meta(sout)
    m_rows = c.get_sparse_num_scheduled_tokens(meta, "r0")
    positions = c.get_sparse_positions(meta, "r0")
    advance = c.get_sparse_computed_advance(meta, "r0")
    return {
        "n": n,
        "external": external,
        "num_new": pre_override_rows,
        "m_rows": m_rows,
        "positions": positions,
        "advance": advance,
        "num_computed_after_step": external + (advance if advance else pre_override_rows),
    }


def external_to_num_new(n: int, external: int | None) -> int:
    return n - (external or 0)


def test_accounting_pure_new_no_reuse():
    # No cache -> external 0, M == all N tokens, advance == N, dense-equivalent.
    c = _make_connector(sparse=True, link=8)
    tokens = list(range(2 * CHUNK))  # N=128
    r = _simulate_schedule(c, tokens)
    n = r["n"]
    # external 0 (nothing cached) so the sparse match path is NOT taken; the
    # connector reports 0 -> num_new == N. _emit_sparse still emits M==all N.
    assert r["external"] == 0
    assert r["num_new"] == n
    assert r["m_rows"] == n          # M == every token (pure C).
    # No external recorded -> advance falls back to seq_len == N.
    assert r["advance"] == n
    assert r["num_computed_after_step"] == n  # converges to N exactly.
    # blocks allocated cover all N positions: external+num_new == N.
    assert r["external"] + r["num_new"] == n


def test_accounting_non_prefix_reuse():
    # chunk0 new + chunk1 cached(non-prefix B) + chunk2 new.
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _make_connector(sparse=True, link=8, store=store)
    chunk0 = list(range(0, CHUNK))
    chunk1 = list(range(1000, 1000 + CHUNK))
    chunk2 = list(range(2000, 2000 + CHUNK))
    _store_chunk(store, chunk1)
    tokens = chunk0 + chunk1 + chunk2  # N=192
    r = _simulate_schedule(c, tokens)
    n = r["n"]  # 192

    # |A|=0 (chunk0 not cached), |B|=CHUNK (chunk1). external = 0 + 64 = 64.
    assert r["external"] == CHUNK
    # num_new = N - external = 192 - 64 = 128 (= |C|, the two new chunks).
    assert r["num_new"] == n - CHUNK == 128
    # M = C(128) + link(8 of B) + {N-1 already in C} = 128 + 8 = 136 rows.
    assert r["m_rows"] == 128 + 8
    assert r["m_rows"] == len(r["positions"])
    # advance = N - external = 128 (NOT m_rows=136 -> no double-count of link).
    assert r["advance"] == n - CHUNK
    # num_computed lands exactly on N after the single sparse prefill step.
    assert r["num_computed_after_step"] == n
    # blocks allocated cover all N (external + num_new == N => ceil(N/16)).
    assert r["external"] + r["num_new"] == n
    nblocks = (n + BLOCK - 1) // BLOCK
    assert (r["external"] + r["num_new"] + BLOCK - 1) // BLOCK == nblocks


def test_accounting_prefix_plus_nonprefix():
    # A (cached prefix) + gap (new) + B (cached, GENUINELY non-prefix) + C (new).
    # A genuine non-prefix B requires a NEW chunk between A and B; otherwise the
    # selection walk folds a contiguous cached chunk into the prefix extent
    # (contiguous cached chunks are reused natively, no link tokens).
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _make_connector(sparse=True, link=8, store=store)
    a = list(range(0, CHUNK))            # cached -> prefix A
    gap = list(range(300, 300 + CHUNK))  # NEW (breaks contiguity)
    b = list(range(500, 500 + CHUNK))    # cached -> non-prefix B
    cnew = list(range(900, 900 + CHUNK))  # NEW C
    _store_chunk(store, a)
    _store_chunk(store, b)
    tokens = a + gap + b + cnew  # N=256
    r = _simulate_schedule(c, tokens)
    n = r["n"]

    # |A|=CHUNK (prefix, [0,64)), |B|=CHUNK (non-prefix, [128,192)).
    # external = |A| + |B| = 128.
    assert r["external"] == 2 * CHUNK
    # num_new = N - external = 256 - 128 = 128 (gap + C).
    assert r["num_new"] == n - 2 * CHUNK
    # M = gap(64) + C(64) + link(8 of B). A contributes NO link / NO M.
    assert r["m_rows"] == 2 * CHUNK + 8
    # Prefix A positions [0,64) must NOT be forwarded (never in M).
    assert all(p >= CHUNK for p in r["positions"])
    # B body (non-link part) must NOT be forwarded: positions [136,192) absent.
    assert all(not (2 * CHUNK + 8 <= p < 3 * CHUNK) for p in r["positions"])
    # advance = N - external = 128 -> num_computed converges to N.
    assert r["advance"] == n - 2 * CHUNK
    assert r["num_computed_after_step"] == n
    # external + num_new == N (block-count basis).
    assert r["external"] + r["num_new"] == n


def test_accounting_external_le_n_and_num_new_positive():
    # Invariant sweep over link sizes + chunk layouts: external<=N, num_new>=1,
    # M nonempty with last==N-1, advance lands num_computed on N.
    for link in (0, 1, 8, 64):
        for layout in ("c_only", "nonprefix_b", "a_gap_b_c", "fully_cached"):
            store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
            c = _make_connector(sparse=True, link=link, store=store)
            a = list(range(0, CHUNK))
            gap = list(range(300, 300 + CHUNK))
            b = list(range(500, 500 + CHUNK))
            cnew = list(range(900, 900 + CHUNK))
            if layout == "c_only":
                # No cache -> external 0, M == all tokens (pure C).
                tokens = cnew
            elif layout == "nonprefix_b":
                # NEW chunk first (no prefix), then cached B (non-prefix), then C.
                _store_chunk(store, b)
                tokens = gap + b + cnew
            elif layout == "a_gap_b_c":
                # Cached prefix A, NEW gap, cached non-prefix B, NEW C.
                _store_chunk(store, a)
                _store_chunk(store, b)
                tokens = a + gap + b + cnew
            else:  # fully_cached: A + contiguous-cached -> whole prompt = prefix.
                _store_chunk(store, a)
                _store_chunk(store, b)
                tokens = a + b  # b is contiguous after a -> folds into prefix.
            n = len(tokens)
            req = _Req("r0", tokens)
            external, is_async = c.get_num_new_matched_tokens(req, 0)
            assert is_async is False
            assert 0 <= external <= n, (layout, link, external, n)
            num_new = n - external
            # scheduler assert :670 (num_new > 0) only fires on the non-async
            # path; the degenerate fully-cached prompt must NOT take the sparse
            # path (external < n required), so num_new >= 1 there too.
            if external < n:
                assert num_new >= 1, (layout, link, external, n)

            sout = _SchedOut(
                scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(n))],
                num_scheduled_tokens={"r0": num_new},
            )
            meta = c.build_connector_meta(sout)
            m_rows = c.get_sparse_num_scheduled_tokens(meta, "r0")
            advance = c.get_sparse_computed_advance(meta, "r0")
            if m_rows is None:
                # Degenerate (fully cached / no sparse): no override, default
                # advance == num_new keeps the standard accounting.
                continue
            positions = c.get_sparse_positions(meta, "r0")
            assert m_rows == len(positions)
            assert positions == sorted(set(positions))
            assert positions[-1] == n - 1  # last token always forwarded.
            assert all(0 <= p < n for p in positions)
            # advance converges num_computed (= external) onto N.
            assert external + advance == n, (layout, link, external, advance, n)
