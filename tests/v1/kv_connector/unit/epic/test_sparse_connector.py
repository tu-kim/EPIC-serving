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
    EpicSchedulerIndex,
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


class _LiveStoreIndex(EpicSchedulerIndex):
    """Test-only scheduler index that reads membership THROUGH a live worker
    store. Many scheduler-logic tests populate the store AFTER building the
    connector; a live proxy keeps those tests one-store while still routing
    selection through the index seam (the production code path). ``register`` /
    ``touch`` are tracked but membership defers to the backing store so post-build
    ``_store_chunk`` calls are visible. (The dedicated 2-instance / LRU mirror
    tests use the real ``EpicSchedulerIndex`` instead.)
    """

    def __init__(self, store: EpicChunkStore):
        super().__init__(
            capacity_bytes=10**8,
            num_layers=1,
            num_kv_heads=1,
            head_size=1,
            cache_dtype_size=4,
        )
        self._backing = store

    def contains(self, chunk_hash: str) -> bool:
        return self._backing.contains(chunk_hash) or super().contains(chunk_hash)

    def get_length(self, chunk_hash: str):
        ln = self._backing.get_length(chunk_hash)
        return ln if ln is not None else super().get_length(chunk_hash)


def _make_connector(*, sparse: bool, link: int = 8, store=None):
    """Build a minimal SCHEDULER-role EpicConnector without VllmConfig/GPU.

    Role-split: this is the SCHEDULER instance, so it carries the mirror INDEX
    (not the worker store). The index proxies a live worker store so chunks
    seeded before OR after build are visible -- modeling "a worker already saved
    these chunks" (the 2-instance reality) without per-test ordering fuss.
    """
    c = object.__new__(EpicConnector)
    c._block_size = BLOCK
    c._chunk_size = CHUNK
    worker_store = store if store is not None else EpicChunkStore(
        capacity_bytes=10**8, pin_memory=False
    )
    # SCHEDULER role: store is None, index mirrors the worker store.
    c._store = None
    c._index = _LiveStoreIndex(worker_store)
    c._matched_prefix = {}
    c._non_prefix = {}
    c._loads_pending = {}
    c._selections = {}
    c._sparse_reqs = {}
    c._native_computed = {}
    c._max_sparse_rows = 0  # 0 == no budget limit (unit tests)
    c._long_prefill_threshold = 0
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


def test_flag_on_pure_new_prompt_is_dense_no_sparse():
    # EPIC fix (consistency): a pure-new prompt with NO non-prefix hits must NOT
    # emit a sparse plan -- it is a dense forward. Emitting M == every token here
    # would diverge from the scheduler-side condition (which only marks a request
    # sparse when there is a non-prefix B hit) and needlessly trip the
    # single-batch gate. So: no sparse metadata at all.
    c = _make_connector(sparse=True, link=8)
    tokens = list(range(2 * CHUNK))  # 128, all genuinely new
    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(len(tokens)))],
        num_scheduled_tokens={"r0": len(tokens)},
    )
    meta = c.build_connector_meta(sout)
    assert meta.sparse == []
    assert c.has_sparse_requests(meta) is False
    # All chunks new -> all saved (dense behavior unchanged).
    assert len(meta.saves) == 1
    assert len(meta.saves[0].chunk_hashes) == 2


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
    # Real flow: the scheduler asks for matches FIRST; only a request the match
    # step actually REGISTERED sparse may emit a plan (consistency guard --
    # _emit_sparse refuses requests without a _sparse_reqs record).
    ext, _ = c.get_num_new_matched_tokens(_Req("r0", tokens), 0)
    assert ext == CHUNK  # external == |B| (chunk1)
    assert "r0" in c._sparse_reqs
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


def test_zero_non_prefix_hits_no_sparse_emission():
    """EPIC fix 2 (consistency): a sparse-enabled request with ZERO non-prefix
    hits must NOT emit any sparse metadata and must NOT register as sparse.

    Covers two no-hit shapes:
      * pure-new prompt (nothing cached),
      * pure-prefix prompt (the whole prompt is a contiguous cached prefix A).
    Both are dense; only a genuine non-prefix B turns a request sparse.
    """
    # (a) pure-new: nothing cached.
    c = _make_connector(sparse=True, link=8)
    tokens = list(range(2 * CHUNK))
    ext, _ = c.get_num_new_matched_tokens(_Req("r0", tokens), 0)
    assert ext == 0
    assert "r0" not in c._sparse_reqs  # not registered sparse.
    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(len(tokens)))],
        num_scheduled_tokens={"r0": len(tokens)},
    )
    meta = c.build_connector_meta(sout)
    assert meta.sparse == []
    assert c.has_sparse_requests(meta) is False

    # (b) pure-prefix: the whole prompt is a contiguous cached prefix.
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c2 = _make_connector(sparse=True, link=8, store=store)
    a = list(range(0, CHUNK))
    b = list(range(64, 64 + CHUNK))  # contiguous after a -> folds into prefix.
    _store_chunk(store, a)
    _store_chunk(store, b)
    ptoks = a + b
    ext2, _ = c2.get_num_new_matched_tokens(_Req("r1", ptoks), 0)
    # Whole prompt is the contiguous prefix -> no non-prefix hit, not sparse.
    assert "r1" not in c2._sparse_reqs
    sout2 = _SchedOut(
        scheduled_new_reqs=[_NewReq("r1", ptoks, _block_ids_for(len(ptoks)))],
        num_scheduled_tokens={"r1": max(0, len(ptoks) - ext2)},
    )
    meta2 = c2.build_connector_meta(sout2)
    assert meta2.sparse == []


def test_b_only_request_registers_load_pending():
    """EPIC fix 1 (root cause): a B-only sparse request (no prefix hit) must be
    registered as a pending load by get_num_new_matched_tokens so that
    build_connector_meta emits the B chunk load. Previously only prefix hits set
    _loads_pending, so a pure-B request's KV was never scattered."""
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _make_connector(sparse=True, link=8, store=store)
    gap = list(range(300, 300 + CHUNK))  # new -> no prefix.
    b = list(range(700, 700 + CHUNK))    # cached, non-prefix.
    hb = _store_chunk(store, b)
    tokens = gap + b
    ext, _ = c.get_num_new_matched_tokens(_Req("r0", tokens), 0)
    assert ext == CHUNK
    assert "r0" in c._loads_pending  # B-only still pending a load.
    assert "r0" in c._sparse_reqs

    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(len(tokens)))],
        num_scheduled_tokens={"r0": len(tokens) - ext},
    )
    meta = c.build_connector_meta(sout)
    assert len(meta.loads) == 1
    chunks = meta.loads[0].chunks
    assert len(chunks) == 1
    assert chunks[0].chunk_hash == hb
    assert chunks[0].new_pos_start == CHUNK


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
    # No cache -> external 0, no non-prefix hits -> DENSE (no sparse emission).
    # EPIC fix (consistency): the pure-new prompt is not sparse; the connector
    # reports external 0 -> num_new == N and emits NO sparse plan, so the
    # scheduler keeps its default contiguous accounting.
    c = _make_connector(sparse=True, link=8)
    tokens = list(range(2 * CHUNK))  # N=128
    r = _simulate_schedule(c, tokens)
    n = r["n"]
    assert r["external"] == 0
    assert r["num_new"] == n
    # No non-prefix hit -> no sparse plan -> the override hooks return None and
    # the runner keeps the dense contiguous schedule.
    assert r["m_rows"] is None
    assert r["positions"] is None
    assert r["advance"] is None
    # Default advance (== num_new) lands num_computed on N: external + num_new.
    assert r["num_computed_after_step"] == n
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


# --------------------------------------------------------------------------
# Infix-scenario fixes: emit/registration consistency, budget guard, and the
# native-computed-region protection (effective prefix + load trimming).
# --------------------------------------------------------------------------


def test_emit_sparse_requires_match_registration():
    """_emit_sparse must emit ONLY for requests the match step registered
    sparse (_sparse_reqs). A stashed selection alone (e.g. a declined sparse
    branch) must NOT produce a plan -- emitting one stamps computed_advance
    against an external the scheduler never counted, which trips the
    post-schedule length invariant and kills the engine."""
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _make_connector(sparse=True, link=8, store=store)

    chunk_b = list(range(1000, 1000 + CHUNK))
    _store_chunk(store, chunk_b)
    tokens = list(range(0, CHUNK)) + chunk_b + list(range(2000, 2000 + CHUNK))

    # Simulate "match ran but the sparse branch was declined": selection is
    # stashed, but no _sparse_reqs registration.
    ext, _ = c.get_num_new_matched_tokens(_Req("r0", tokens), 0)
    assert "r0" in c._sparse_reqs
    del c._sparse_reqs["r0"]  # decline happened (budget / external==N / etc.)
    assert "r0" in c._selections  # stash still present

    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(len(tokens)))],
        num_scheduled_tokens={"r0": len(tokens) - ext},
    )
    meta = c.build_connector_meta(sout)
    assert meta.sparse == []  # no registration -> no emission, dense step.


def test_budget_guard_declines_sparse_when_prefill_would_be_chunked():
    """When the one-step sparse prefill cannot fit the scheduler token budget
    (N - external > budget or |M| > budget), the connector must decline the
    sparse branch entirely: no external report from the hits, no _sparse_reqs
    registration, no sparse emission -- a plain dense prefill."""
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _make_connector(sparse=True, link=8, store=store)
    c._max_sparse_rows = 32  # << N - external (=128) -> must decline.

    chunk_b = list(range(1000, 1000 + CHUNK))
    _store_chunk(store, chunk_b)
    tokens = list(range(0, CHUNK)) + chunk_b + list(range(2000, 2000 + CHUNK))
    n = len(tokens)

    ext, _ = c.get_num_new_matched_tokens(_Req("r0", tokens), 0)
    assert ext == 0  # no prefix hit; sparse declined -> nothing external.
    assert "r0" not in c._sparse_reqs

    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(n))],
        num_scheduled_tokens={"r0": n},
    )
    meta = c.build_connector_meta(sout)
    assert meta.sparse == []
    assert meta.loads == []  # nothing counted external -> nothing to load.


def test_budget_guard_respects_long_prefill_threshold():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _make_connector(sparse=True, link=8, store=store)
    c._long_prefill_threshold = 32  # < N - external -> scheduler would chunk.

    chunk_b = list(range(1000, 1000 + CHUNK))
    _store_chunk(store, chunk_b)
    tokens = list(range(0, CHUNK)) + chunk_b + list(range(2000, 2000 + CHUNK))

    ext, _ = c.get_num_new_matched_tokens(_Req("r0", tokens), 0)
    assert ext == 0
    assert "r0" not in c._sparse_reqs


def test_budget_guard_allows_fitting_sparse():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _make_connector(sparse=True, link=8, store=store)
    c._max_sparse_rows = 4096  # plenty -> sparse engages normally.

    chunk_b = list(range(1000, 1000 + CHUNK))
    _store_chunk(store, chunk_b)
    tokens = list(range(0, CHUNK)) + chunk_b + list(range(2000, 2000 + CHUNK))

    ext, _ = c.get_num_new_matched_tokens(_Req("r0", tokens), 0)
    assert ext == CHUNK
    assert "r0" in c._sparse_reqs


def test_native_prefix_extends_effective_prefix_and_keeps_native_out_of_m():
    """User scenario: A is cached ONLY in the native prefix cache (chunk store
    may or may not hold it). With num_computed_tokens = |A| the effective
    prefix must cover A, so (a) the sparse branch engages off the native
    extent, (b) A's positions never enter M, and (c) accounting still lands
    num_computed on N."""
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _make_connector(sparse=True, link=8, store=store)

    # Prompt: A (1 chunk, NOT in the store -- native only) + X (new) +
    # B (file chunk, in store) + tail (new).
    a = list(range(0, CHUNK))
    x = list(range(5000, 5000 + CHUNK))
    b = list(range(1000, 1000 + CHUNK))
    tail = list(range(2000, 2000 + CHUNK))
    _store_chunk(store, b)
    tokens = a + x + b + tail
    n = len(tokens)
    nc = CHUNK  # native prefix cache covers exactly A.

    num_new, _ = c.get_num_new_matched_tokens(_Req("r0", tokens), nc)
    # effective prefix = nc, external = nc + |B|, num_new = external - nc = |B|.
    assert "r0" in c._sparse_reqs
    assert c._sparse_reqs["r0"] == (n, nc + CHUNK)
    assert num_new == CHUNK

    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(n))],
        num_scheduled_tokens={"r0": n - (nc + CHUNK)},
    )
    meta = c.build_connector_meta(sout)
    assert len(meta.sparse) == 1
    m = set(meta.sparse[0].sparse_positions)
    # A (native) never in M; X fully in M; B only its link head; tail fully.
    assert m.isdisjoint(range(0, CHUNK))
    assert set(range(CHUNK, 2 * CHUNK)) <= m
    assert set(range(2 * CHUNK, 2 * CHUNK + 8)) <= m
    assert (2 * CHUNK + 8) not in m
    assert set(range(3 * CHUNK, n)) <= m
    # advance converges onto N.
    assert meta.sparse[0].computed_advance == n - (nc + CHUNK)


def test_native_computed_trims_prefix_chunk_loads():
    """Chunk loads must never scatter into positions below the native computed
    extent: those blocks hold EXACT KV and may be SHARED with other requests
    (poisoning risk). A prefix chunk fully below nc is skipped; one straddling
    nc is head-trimmed with src_offset."""
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _make_connector(sparse=True, link=8, store=store)

    # Store TWO contiguous prefix chunks + one non-prefix file chunk.
    p0 = list(range(0, CHUNK))
    p1 = list(range(64, 64 + CHUNK))
    b = list(range(1000, 1000 + CHUNK))
    for t in (p0, p1, b):
        _store_chunk(store, t)
    x = list(range(5000, 5000 + CHUNK))  # new segment between prefix and B
    tokens = p0 + p1 + x + b + list(range(2000, 2000 + CHUNK))
    n = len(tokens)

    # Native cache covers p0 entirely plus HALF of p1 (block-aligned 96).
    nc = CHUNK + CHUNK // 2
    num_new, _ = c.get_num_new_matched_tokens(_Req("r0", tokens), nc)
    # store prefix extent (128) > nc (96) -> effective prefix stays 128.
    assert c._sparse_reqs["r0"] == (n, 2 * CHUNK + CHUNK)

    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(n))],
        num_scheduled_tokens={"r0": n - (2 * CHUNK + CHUNK)},
    )
    meta = c.build_connector_meta(sout)
    assert len(meta.loads) == 1
    specs = {s.new_pos_start: s for s in meta.loads[0].chunks}
    # p0 fully below nc=96 -> NO load emitted for position 0.
    assert 0 not in specs
    # p1 straddles nc: trimmed to [96, 128), src_offset = 96-64 = 32.
    assert 96 in specs
    assert specs[96].src_offset == CHUNK // 2
    assert specs[96].length == CHUNK // 2
    assert len(specs[96].dst_slot_ids) == CHUNK // 2
    # B (non-prefix) is fully above nc -> untrimmed.
    assert 3 * CHUNK in specs
    assert specs[3 * CHUNK].src_offset == 0
    assert specs[3 * CHUNK].length == CHUNK


def test_load_chunk_applies_src_offset():
    """Worker side: _load_chunk must read stored K/V/old_positions from
    src_offset onward, so a head-trimmed spec lands the chunk TAIL at the
    trimmed destination slots."""
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
        check_scatter_fidelity,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
        ChunkLoadSpec,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
        IdentityAlignment,
    )

    heads, hd = 1, 4
    kv_cache = torch.zeros(2, 8, BLOCK, heads, hd)

    length, src_off = 32, 8
    stored = StoredChunk(
        chunk_hash="h",
        length=length,
        old_positions=torch.arange(length, dtype=torch.int64),
    )
    k_src = torch.randn(length, heads, hd)
    v_src = torch.randn(length, heads, hd)
    stored.k_per_layer["l0"] = k_src
    stored.v_per_layer["l0"] = v_src

    w = object.__new__(EpicConnector)
    w._layer_names = ["l0"]
    w._kv_caches = {"l0": kv_cache}
    w._alignment = IdentityAlignment()  # isolate the src trim from PIC math.
    w._store = None

    dst_slots = list(range(40, 40 + (length - src_off)))
    spec = ChunkLoadSpec(
        chunk_hash="h",
        dst_slot_ids=dst_slots,
        old_pos_start=-1,
        new_pos_start=src_off,
        length=length - src_off,
        src_offset=src_off,
    )
    w._load_chunk(stored, spec)

    res = check_scatter_fidelity(
        kv_cache, k_src[src_off:], v_src[src_off:], dst_slots
    )
    assert res is not None
    k_ok, _, v_ok, _ = res
    assert k_ok and v_ok
