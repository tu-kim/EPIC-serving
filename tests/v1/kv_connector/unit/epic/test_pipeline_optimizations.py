# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline optimization pass: every optimization must be observably applied
AND behavior-identical to the unoptimized path.

  A. vectorized token hashing (byte-identical to the per-token loop)
  B. per-step chunk-split cache (prompt hashed once, not twice)
  C. budget-guard plan reused by _emit_sparse (M derived once)
  D. PIC cos/sin memo (trig once per chunk, not once per layer)
  E. identity-delta rotation skip (prefix reload at original positions)
  F. save-path harvest (CPU path byte-for-byte unchanged)
"""

import hashlib
import threading

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    ChainHasher,
    EpicChunkStore,
    EpicSchedulerIndex,
    StoredChunk,
    chain_hash_tokens,
    hash_chunk_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
    EpicConnector,
    check_scatter_fidelity,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
    ChunkLoadSpec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.pic import PICRotator
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    AlignmentStrategy,
    EpicSelection,
    LegoLinkRecompute,
    PicAlignment,
)

BLOCK = 16
CHUNK = 64
HEADS, HD = 1, 4


# ---------------------------------------------------------------------------
# A. hashing equivalence.
# ---------------------------------------------------------------------------


def _reference_chunk_hash(token_ids: list[int]) -> str:
    """The original per-token loop, kept as the compatibility oracle."""
    h = hashlib.sha256()
    h.update(b"epic-chunk-v1")
    h.update(len(token_ids).to_bytes(4, "little"))
    for t in token_ids:
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def _reference_chain_hash(token_ids: list[int]) -> str:
    h = hashlib.sha256(b"epic-chain-v1")
    for t in token_ids:
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def test_vectorized_chunk_hash_is_byte_identical():
    cases = [
        [],
        [0],
        [2**32 - 1],
        list(range(256)),
        [7, 0, 2**31, 123456, 2**32 - 1] * 51,
        torch.randint(0, 2**31, (1024,)).tolist(),
    ]
    for tokens in cases:
        assert hash_chunk_tokens(tokens) == _reference_chunk_hash(tokens)


def test_vectorized_chain_hash_is_byte_identical():
    tokens = torch.randint(0, 2**31, (777,)).tolist()
    assert chain_hash_tokens(tokens) == _reference_chain_hash(tokens)
    # Incremental updates == one-shot (the running-boundary property the
    # connector's split relies on).
    ch = ChainHasher()
    ch.update(tokens[:100])
    ch.update(tokens[100:333])
    mid = ch.digest()
    assert mid == _reference_chain_hash(tokens[:333])
    ch.update(tokens[333:])
    assert ch.digest() == _reference_chain_hash(tokens)


def test_out_of_range_token_still_raises():
    with pytest.raises(OverflowError):
        hash_chunk_tokens([-1])
    with pytest.raises(OverflowError):
        hash_chunk_tokens([2**32])


# ---------------------------------------------------------------------------
# Scheduler-side helpers (match + build driving, as the scheduler would).
# ---------------------------------------------------------------------------


class _LiveIndex(EpicSchedulerIndex):
    def __init__(self, store: EpicChunkStore):
        super().__init__(
            capacity_bytes=10**8,
            num_layers=1,
            num_kv_heads=HEADS,
            head_size=HD,
            cache_dtype_size=4,
        )
        self._backing = store

    def contains(self, chunk_hash: str) -> bool:
        return self._backing.contains(chunk_hash) or super().contains(chunk_hash)

    def get_length(self, chunk_hash: str):
        ln = self._backing.get_length(chunk_hash)
        return ln if ln is not None else super().get_length(chunk_hash)

    def get_old_pos_start(self, chunk_hash: str):
        old = self._backing.get_old_pos_start(chunk_hash)
        return old if old is not None else super().get_old_pos_start(chunk_hash)

    def get_chain(self, chunk_hash: str):
        chain = self._backing.get_chain(chunk_hash)
        if chain is not None and chain != (None, None):
            return chain
        return super().get_chain(chunk_hash)


class _Req:
    def __init__(self, req_id: str, tokens: list[int]):
        self.request_id = req_id
        self.prompt_token_ids = tokens


class _NewReq:
    def __init__(self, req_id: str, tokens: list[int]):
        self.req_id = req_id
        self.prompt_token_ids = tokens
        nblocks = (len(tokens) + BLOCK - 1) // BLOCK
        self.block_ids = [list(range(nblocks))]


class _SchedOut:
    def __init__(self, new_reqs, counts):
        self.scheduled_new_reqs = new_reqs
        self.num_scheduled_tokens = counts


def _connector(store, *, budget: int = 0):
    c = object.__new__(EpicConnector)
    c._block_size = BLOCK
    c._chunk_size = CHUNK
    c._store = None
    c._index = _LiveIndex(store)
    c._matched_prefix = {}
    c._non_prefix = {}
    c._loads_pending = {}
    c._selections = {}
    c._sparse_reqs = {}
    c._native_computed = {}
    c._max_sparse_rows = budget
    c._long_prefill_threshold = 0
    c._sparse_forward = True
    c._link_tokens = 8
    c._selection = EpicSelection(strict_prefix_chain=True)
    c._recompute = LegoLinkRecompute(num_link_tokens=8, phase1_dense=False)
    c._fusion_enabled = False
    return c


def _store_chunk(store, tokens):
    h = hash_chunk_tokens(tokens)
    sc = StoredChunk(
        chunk_hash=h,
        length=len(tokens),
        old_positions=torch.arange(len(tokens), dtype=torch.int64),
    )
    sc.k_per_layer["l0"] = torch.zeros(len(tokens), HEADS, HD)
    sc.v_per_layer["l0"] = torch.zeros(len(tokens), HEADS, HD)
    store.put(sc)
    return h


def _drive(c, tokens):
    req = _Req("r0", tokens)
    ext, _ = c.get_num_new_matched_tokens(req, 0)
    if ext:
        c.update_state_after_alloc(req, None, ext)
    sout = _SchedOut(
        [_NewReq("r0", tokens)], {"r0": max(1, len(tokens) - ext)}
    )
    return ext, c.build_connector_meta(sout)


def _count_calls(obj, attr):
    """Wrap obj.attr with a call counter; returns the counter holder."""
    fn = getattr(obj, attr)
    holder = {"n": 0}

    def wrapped(*a, **k):
        holder["n"] += 1
        return fn(*a, **k)

    setattr(obj, attr, wrapped)
    return holder


# ---------------------------------------------------------------------------
# B. split cache: the prompt is split+hashed once per request per step.
# ---------------------------------------------------------------------------


def test_prompt_split_and_hashed_once_per_step():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    file_chunk = list(range(1000, 1000 + CHUNK))
    _store_chunk(store, file_chunk)
    c = _connector(store)
    calls = _count_calls(c, "_split_prompt_into_chunks")

    tokens = list(range(CHUNK)) + file_chunk + list(range(2000, 2000 + CHUNK))
    _, meta = _drive(c, tokens)  # match + build (save walk) in one step.
    assert calls["n"] == 1  # previously 2: match AND save walk both split.
    assert len(meta.sparse) == 1  # behavior unchanged.
    assert c.__dict__["_chunk_split_cache"] == {}  # cleared per step.

    # Next step recomputes (no stale reuse across steps).
    _, _ = _drive(c, tokens)
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# C. plan reuse: the budget guard's plan feeds _emit_sparse.
# ---------------------------------------------------------------------------


def test_budget_guard_plan_reused_by_emit():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    file_chunk = list(range(1000, 1000 + CHUNK))
    _store_chunk(store, file_chunk)
    tokens = list(range(CHUNK)) + file_chunk + list(range(2000, 2000 + CHUNK))

    # Budget configured -> guard derives the plan; emit must NOT re-derive.
    c1 = _connector(store, budget=4096)
    n1 = _count_calls(c1._recompute, "plan_recompute")
    _, meta1 = _drive(c1, tokens)
    assert n1["n"] == 1

    # Budget disabled -> guard early-returns; emit derives exactly once.
    c2 = _connector(store, budget=0)
    n2 = _count_calls(c2._recompute, "plan_recompute")
    _, meta2 = _drive(c2, tokens)
    assert n2["n"] == 1

    # Identical sparse output either way.
    assert (
        meta1.sparse[0].sparse_positions == meta2.sparse[0].sparse_positions
    )
    assert meta1.sparse[0].computed_advance == meta2.sparse[0].computed_advance


# ---------------------------------------------------------------------------
# D. PIC cos/sin memo: trig once per chunk across layers.
# ---------------------------------------------------------------------------


def _rotator():
    return PICRotator(
        head_size=HD, rotary_dim=HD, base=10000.0, is_neox_style=True,
        dtype=torch.float64,
    )


def test_pic_memo_computes_trig_once_for_shared_positions():
    rot = _rotator()
    calls = _count_calls(rot, "_cos_sin_for_delta")
    old = torch.arange(CHUNK, dtype=torch.int64)
    new = torch.arange(100, 100 + CHUNK, dtype=torch.int64)
    keys = [torch.randn(CHUNK, HEADS, HD, dtype=torch.float64) for _ in range(3)]

    fresh = _rotator()  # oracle without memo interference (new instance).
    outs = [rot.rotate_keys(k, old, new) for k in keys]
    assert calls["n"] == 1  # one trig computation for three "layers".
    for k, out in zip(keys, outs):
        torch.testing.assert_close(out, fresh.rotate_keys(k, old, new))

    # Different position tensors -> memo miss -> recompute (still correct).
    old2 = torch.arange(5, 5 + CHUNK, dtype=torch.int64)
    out2 = rot.rotate_keys(keys[0], old2, new)
    assert calls["n"] == 2
    torch.testing.assert_close(out2, _rotator().rotate_keys(keys[0], old2, new))


# ---------------------------------------------------------------------------
# D+E. worker load path: positions shared across layers, identity skipped.
# ---------------------------------------------------------------------------


class _SpyAlignment(AlignmentStrategy):
    def __init__(self, fail_if_called: bool = False):
        self.calls: list[tuple[int, int]] = []
        self.fail = fail_if_called

    def align_keys(self, key, old_positions, new_positions, layer_name):
        if self.fail:
            raise AssertionError("alignment must be skipped for delta==0")
        self.calls.append((id(old_positions), id(new_positions)))
        return key


def _worker(alignment, num_layers: int = 2):
    w = object.__new__(EpicConnector)
    w._layer_names = [f"l{i}" for i in range(num_layers)]
    w._kv_caches = {
        name: torch.zeros(2, 16, BLOCK, HEADS, HD) for name in w._layer_names
    }
    w._alignment = alignment
    w._store = None
    return w


def _stored_multi(num_layers: int, old_start: int) -> StoredChunk:
    sc = StoredChunk(
        chunk_hash="h",
        length=CHUNK,
        old_positions=torch.arange(
            old_start, old_start + CHUNK, dtype=torch.int64
        ),
    )
    for i in range(num_layers):
        g = torch.Generator().manual_seed(i)
        sc.k_per_layer[f"l{i}"] = torch.randn(CHUNK, HEADS, HD, generator=g)
        sc.v_per_layer[f"l{i}"] = torch.randn(CHUNK, HEADS, HD, generator=g)
    return sc


def _spec(new_start: int) -> ChunkLoadSpec:
    return ChunkLoadSpec(
        chunk_hash="h",
        dst_slot_ids=list(range(new_start, new_start + CHUNK)),
        old_pos_start=-1,
        new_pos_start=new_start,
        length=CHUNK,
    )


def test_identity_delta_skips_alignment_and_scatters_verbatim():
    spy = _SpyAlignment(fail_if_called=True)
    w = _worker(spy)
    sc = _stored_multi(2, old_start=0)
    w._load_chunk(sc, _spec(new_start=0))  # old == new -> no alignment.
    for i in range(2):
        res = check_scatter_fidelity(
            w._kv_caches[f"l{i}"],
            sc.k_per_layer[f"l{i}"],
            sc.v_per_layer[f"l{i}"],
            list(range(CHUNK)),
        )
        assert res is not None and res[0] and res[2]


def test_nonzero_delta_aligns_with_shared_position_tensors():
    spy = _SpyAlignment()
    w = _worker(spy)
    sc = _stored_multi(2, old_start=0)
    w._load_chunk(sc, _spec(new_start=128))  # delta != 0 -> align per layer.
    assert len(spy.calls) == 2
    # The SAME position tensor objects across layers (what the memo keys on).
    assert spy.calls[0] == spy.calls[1]


def test_pic_end_to_end_load_matches_unmemoized_reference():
    """Full _load_chunk with the REAL PicAlignment: scattered K must equal a
    fresh, per-layer rotation with no memo in play."""
    rot = _rotator()
    w = _worker(PicAlignment(rot))
    # float64 caches so the comparison is exact.
    w._kv_caches = {
        name: torch.zeros(2, 16, BLOCK, HEADS, HD, dtype=torch.float64)
        for name in w._layer_names
    }
    sc = _stored_multi(2, old_start=0)
    for name in list(sc.k_per_layer):
        sc.k_per_layer[name] = sc.k_per_layer[name].to(torch.float64)
        sc.v_per_layer[name] = sc.v_per_layer[name].to(torch.float64)
    new_start = 96
    w._load_chunk(sc, _spec(new_start=new_start))

    old = torch.arange(CHUNK, dtype=torch.int64)
    new = torch.arange(new_start, new_start + CHUNK, dtype=torch.int64)
    for i in range(2):
        expected = _rotator().rotate_keys(
            sc.k_per_layer[f"l{i}"], old, new
        )
        res = check_scatter_fidelity(
            w._kv_caches[f"l{i}"],
            expected,
            sc.v_per_layer[f"l{i}"],
            list(range(new_start, new_start + CHUNK)),
        )
        assert res is not None and res[0] and res[2]


# ---------------------------------------------------------------------------
# F. save-path harvest: CPU path byte-for-byte unchanged.
# ---------------------------------------------------------------------------


def test_harvest_to_cpu_on_cpu_is_identity_copy():
    w = object.__new__(EpicConnector)
    w._store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    src = torch.randn(CHUNK, HEADS, HD)
    out = w._harvest_to_cpu(src)
    assert out.device.type == "cpu"
    torch.testing.assert_close(out, src)
    # No pending fence on the pure-CPU path.
    assert getattr(w, "_pending_save_sync", False) is False
    w.wait_for_save()  # must be a no-op, not a crash, without CUDA.


# ---------------------------------------------------------------------------
# G-pass (2nd round): single-buffer split, vectorized plan/slots, slot-tensor
# hoists -- each locked against a reference implementation.
# ---------------------------------------------------------------------------


def test_single_buffer_split_matches_per_chunk_hashing():
    """_split_prompt_into_chunks (single whole-prompt buffer + memoryview
    windows) must produce the SAME hashes/chains as hashing each chunk's
    token list independently."""
    c = _connector(EpicChunkStore(capacity_bytes=10**8, pin_memory=False))
    tokens = torch.randint(0, 2**31, (5 * CHUNK + 17,)).tolist()  # ragged tail
    out = c._split_prompt_into_chunks(tokens)
    assert len(out) == 5  # tail below chunk size never emitted.
    for start, length, h, cb, ca in out:
        assert length == CHUNK
        assert h == hash_chunk_tokens(tokens[start : start + CHUNK])
        assert cb == chain_hash_tokens(tokens[:start])
        assert ca == chain_hash_tokens(tokens[: start + CHUNK])
    # Shorter-than-chunk prompt: empty split (early return path).
    assert c._split_prompt_into_chunks(tokens[: CHUNK - 1]) == []


def test_vectorized_slot_ids_match_loop_reference():
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
        _slot_ids_from_blocks,
    )

    def reference(block_ids, block_size, start, num):
        return [
            block_ids[i // block_size] * block_size + i % block_size
            for i in range(start, start + num)
        ]

    block_ids = [7, 3, 11, 0, 42, 5]
    for start, num in [(0, 16), (5, 40), (16, 64), (37, 3), (0, 96), (95, 1)]:
        assert _slot_ids_from_blocks(block_ids, 16, start, num) == reference(
            block_ids, 16, start, num
        )
    with pytest.raises(IndexError):
        _slot_ids_from_blocks(block_ids, 16, 90, 16)  # past the last block.


def _reference_plan(n, prefix_extent, hits, link, link_per_run, policy):
    """The original set-based M derivation, kept as the oracle."""
    if prefix_extent >= n or n <= 0:
        return None
    reused = set(range(min(prefix_extent, n)))
    for hit in hits:
        lo = max(0, int(hit.prompt_offset))
        hi = min(n, lo + max(0, int(hit.length)))
        reused.update(range(lo, hi))
    m = {p for p in range(n) if p not in reused}
    if link > 0:
        prev = None
        for hit in sorted(hits, key=lambda h: int(h.prompt_offset)):
            run_internal = (
                link_per_run
                and prev is not None
                and policy._run_continuous(prev, hit)
            )
            prev = hit
            if run_internal:
                continue
            lo = max(0, int(hit.prompt_offset))
            hi = min(n, lo + min(link, max(0, int(hit.length))))
            m.update(range(lo, hi))
    m.add(n - 1)
    offsets = sorted(m)
    reused.update(offsets)
    return offsets, sorted(reused)


def test_vectorized_plan_matches_set_reference_fuzz():
    """Randomized layouts: numpy-mask M derivation == the original set-based
    derivation, including per-run link mode."""
    import random

    from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
        NonPrefixHit,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
        ReuseSelection,
    )

    rng = random.Random(1234)

    class _R:
        def __init__(self, n):
            self.prompt_token_ids = list(range(n))

    for trial in range(60):
        n = rng.randrange(2, 700)
        prefix = rng.choice([0, 0, CHUNK, 2 * CHUNK, n // 2])
        prefix = min(prefix, n - 1)
        hits = []
        pos = prefix
        while pos + 8 < n and len(hits) < 4 and rng.random() < 0.75:
            pos += rng.randrange(0, 64)
            length = rng.randrange(1, 128)
            if pos + length > n or pos < prefix:
                break
            old = rng.choice([-1, 0, pos, 4096])
            hits.append(
                NonPrefixHit(
                    chunk_hash=f"h{len(hits)}",
                    prompt_offset=pos,
                    old_pos_start=old,
                    length=length,
                )
            )
            pos += length
        link = rng.choice([0, 1, 8, 64, 1000])
        per_run = rng.random() < 0.5
        policy = LegoLinkRecompute(
            num_link_tokens=link, phase1_dense=False, link_per_run=per_run
        )
        sel = ReuseSelection(prefix_extent=prefix, non_prefix_hits=hits)
        plan = policy.plan_recompute(_R(n), sel, block_size=16)
        ref = _reference_plan(n, prefix, hits, link, per_run, policy)
        if ref is None:
            assert plan.recompute_offsets == [], (trial, n, prefix)
            continue
        ref_offsets, ref_reused = ref
        assert plan.recompute_offsets == ref_offsets, (trial, n, prefix, link)
        assert plan.reused_offsets == ref_reused, (trial, n, prefix, link)


def test_load_chunk_converts_slots_once_per_chunk():
    """_scatter_kv must receive the SAME slots tensor object for every layer
    of a chunk (one list->tensor conversion per chunk, not per layer)."""
    spy_slots: list[int] = []
    w = _worker(_SpyAlignment(), num_layers=3)
    sc = _stored_multi(3, old_start=0)
    orig = w._scatter_kv

    def spying_scatter(kv_cache, k, v, slots):
        spy_slots.append(id(slots))
        return EpicConnector._scatter_kv(w, kv_cache, k, v, slots)

    w._scatter_kv = spying_scatter
    w._load_chunk(sc, _spec(new_start=16))
    assert len(spy_slots) == 3
    assert len(set(spy_slots)) == 1  # one tensor object shared by all layers.


def test_save_slots_tensor_memoized_per_list_identity():
    w = object.__new__(EpicConnector)
    slot_ids = list(range(32, 32 + CHUNK))
    t1 = w._save_slots_tensor(slot_ids, torch.device("cpu"))
    t2 = w._save_slots_tensor(slot_ids, torch.device("cpu"))
    assert t1 is t2  # memo hit for the same list object.
    other = list(slot_ids)
    t3 = w._save_slots_tensor(other, torch.device("cpu"))
    assert t3 is not t1
    torch.testing.assert_close(t3, t1)
    # wait_for_save clears the memo (metadata lifetime boundary).
    w._store = None
    w.wait_for_save()
    assert w.__dict__["_save_slots_cache"] == {}
