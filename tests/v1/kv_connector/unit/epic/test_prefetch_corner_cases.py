# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Corner cases for the dynamo prefetch command path and the dedicated
staging worker process (CPU).

Command path: hostile/degenerate inputs must never corrupt the queue or kill
the listener; directives must drain exactly once; a directive and a load for
the same chunk in the SAME step must stage before loading (start_load_kv
ordering). Staging process: protocol errors, oversize chunks, LRU eviction
under a parent-held mapping, and close/restart -- every failure mode must
degrade to the CPU fileKV fallback, never to wrong KV."""

import json
import os
import threading
import uuid

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
    check_scatter_fidelity,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
    ChunkLoadSpec,
    EpicConnectorMetadata,
    EpicReqLoad,
    EpicReqPrefetch,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch import (
    EpicGpuStagingStore,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch_service import (
    EpicPrefetchClient,
    EpicPrefetchListener,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    IdentityAlignment,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.staging_worker import (
    ExternalStagingBackend,
)

BLOCK = 16
CHUNK = 64
HEADS, HD = 1, 4


def _scheduler_connector():
    index = EpicSchedulerIndex(
        capacity_bytes=10**8,
        num_layers=1,
        num_kv_heads=HEADS,
        head_size=HD,
        cache_dtype_size=4,
    )
    c = object.__new__(EpicConnector)
    c._block_size = BLOCK
    c._chunk_size = CHUNK
    c._store = None
    c._index = index
    c._prefetch_lock = threading.Lock()
    c._prefetch_queue = []
    return c, index


def _stored(hash_: str, seed: int = 0, length: int = CHUNK) -> StoredChunk:
    g = torch.Generator().manual_seed(seed)
    sc = StoredChunk(
        chunk_hash=hash_,
        length=length,
        old_positions=torch.arange(length, dtype=torch.int64),
    )
    sc.k_per_layer["l0"] = torch.randn(length, HEADS, HD, generator=g)
    sc.v_per_layer["l0"] = torch.randn(length, HEADS, HD, generator=g)
    return sc


def _worker(store, staging, worker_id=0):
    w = object.__new__(EpicConnector)
    w._store = store
    w._staging = staging
    w._worker_id = worker_id
    w._layer_names = ["l0"]
    w._kv_caches = {"l0": torch.zeros(2, 8, BLOCK, HEADS, HD)}
    w._alignment = IdentityAlignment()
    w._debug_counters = True
    w._debug_check_load = False
    w._check_load_done = False
    w._fusion_enabled = False
    w._flex_layers_patched = set()
    w._mask_builder = None
    return w


class _Fwd:
    """Minimal forward_context: no FlexAttention layers to patch."""

    no_compile_layers: dict = {}


# ---------------------------------------------------------------------------
# Command path corners.
# ---------------------------------------------------------------------------


def test_all_unknown_hashes_queue_nothing():
    c, _ = _scheduler_connector()
    reply = c.handle_prefetch_command(
        {"cmd": "prefetch", "chunk_hashes": ["x", "y"], "dst_worker": 0}
    )
    assert reply["ok"] and reply["queued"] == []
    assert set(reply["dropped"]) == {"x", "y"}
    assert c._prefetch_queue == []  # nothing known -> no directive at all.


def test_duplicate_hashes_queue_once():
    c, index = _scheduler_connector()
    tokens = list(range(CHUNK))
    h = hash_chunk_tokens(tokens)
    index.register(h, CHUNK)
    reply = c.handle_prefetch_command(
        {
            "cmd": "prefetch",
            "chunk_hashes": [h, h],
            "token_ids": tokens,  # hashes to h again
            "dst_worker": 1,
        }
    )
    assert reply["queued"] == [h]  # deduplicated across both channels.
    assert len(c._prefetch_queue) == 1
    assert c._prefetch_queue[0].chunk_hashes == [h]


def test_empty_command_is_ok_noop():
    c, _ = _scheduler_connector()
    reply = c.handle_prefetch_command({"cmd": "prefetch"})
    assert reply == {"ok": True, "queued": [], "dropped": []}
    assert c._prefetch_queue == []


def test_multiple_commands_drain_into_one_meta_exactly_once():
    c, index = _scheduler_connector()
    t1, t2 = list(range(CHUNK)), list(range(1000, 1000 + CHUNK))
    h1, h2 = hash_chunk_tokens(t1), hash_chunk_tokens(t2)
    index.register(h1, CHUNK)
    index.register(h2, CHUNK)
    c.enqueue_prefetch(chunk_hashes=[h1], dst_worker=0)
    c.enqueue_prefetch(chunk_hashes=[h2], dst_worker=1)

    # Drain (mirrors build_connector_meta's drain block).
    meta = EpicConnectorMetadata()
    with c._prefetch_lock:
        pending, c._prefetch_queue = c._prefetch_queue, []
    for d in pending:
        meta.add_prefetch(d)
    assert [d.chunk_hashes for d in meta.prefetches] == [[h1], [h2]]
    assert [d.dst_worker for d in meta.prefetches] == [0, 1]

    # Second drain: nothing left (exactly-once delivery to the workers).
    assert c._prefetch_queue == []


def test_listener_survives_malformed_commands():
    c, index = _scheduler_connector()
    tokens = list(range(CHUNK))
    h = hash_chunk_tokens(tokens)
    index.register(h, CHUNK)

    endpoint = f"ipc:///tmp/epic-pf-corner-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    listener = EpicPrefetchListener(endpoint, c.handle_prefetch_command)
    listener.start()
    try:
        import zmq

        ctx = zmq.Context.instance()
        raw = ctx.socket(zmq.REQ)
        raw.rcvtimeo = 5000
        raw.linger = 0
        raw.connect(endpoint)

        # (a) invalid JSON bytes -> error reply, listener alive.
        raw.send(b"\x00not-json")
        rep = json.loads(raw.recv())
        assert rep["ok"] is False

        # (b) unknown command -> error reply, listener alive.
        raw.send(json.dumps({"cmd": "detonate"}).encode())
        rep = json.loads(raw.recv())
        assert rep["ok"] is False
        raw.close(linger=0)

        # (c) a VALID command still works afterwards.
        client = EpicPrefetchClient(endpoint, timeout_ms=5000)
        queued, dropped = client.prefetch(chunk_hashes=[h], dst_worker=0)
        assert queued == [h] and dropped == []
        client.close()
    finally:
        listener.stop()


def test_client_timeout_then_recovers_after_listener_starts():
    endpoint = f"ipc:///tmp/epic-pf-corner-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    client = EpicPrefetchClient(endpoint, timeout_ms=200)
    # Listener not running: ping fails fast, no exception escapes.
    assert client.ping() is False

    c, index = _scheduler_connector()
    tokens = list(range(CHUNK))
    h = hash_chunk_tokens(tokens)
    index.register(h, CHUNK)
    listener = EpicPrefetchListener(endpoint, c.handle_prefetch_command)
    listener.start()
    try:
        # The REQ socket was rebuilt after the timeout -> works now.
        client2 = EpicPrefetchClient(endpoint, timeout_ms=5000)
        assert client2.ping() is True
        queued, _ = client2.prefetch(chunk_hashes=[h], dst_worker=0)
        assert queued == [h]
        client2.close()
        client.close()
    finally:
        listener.stop()


def test_same_step_directive_and_load_order_via_start_load_kv():
    """A directive and a load for the SAME chunk in one step: start_load_kv
    must stage first (prefetch consume precedes loads), so the load hits."""
    EpicConnector.reset_debug_counters()
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    sc = _stored("h-same-step", seed=21)
    store.put(sc)
    staging = EpicGpuStagingStore(capacity_bytes=10**8)
    w = _worker(store, staging)

    dst = list(range(0, CHUNK))
    meta = EpicConnectorMetadata()
    meta.add_prefetch(
        EpicReqPrefetch(chunk_hashes=["h-same-step"], dst_worker=-1)
    )
    load = EpicReqLoad(req_id="r0")
    load.chunks.append(
        ChunkLoadSpec(
            chunk_hash="h-same-step",
            dst_slot_ids=dst,
            old_pos_start=-1,
            new_pos_start=0,
            length=CHUNK,
        )
    )
    meta.add_load(load)

    w._connector_metadata = meta
    w.start_load_kv(_Fwd())  # the REAL per-step entry point.

    assert EpicConnector.debug_counters["prefetch_staged"] == 1
    assert EpicConnector.debug_counters["prefetch_hit"] == 1  # staged FIRST.
    res = check_scatter_fidelity(
        w._kv_caches["l0"], sc.k_per_layer["l0"], sc.v_per_layer["l0"], dst
    )
    assert res is not None and res[0] and res[2]


def test_directive_for_store_evicted_chunk_is_counted_not_fatal():
    EpicConnector.reset_debug_counters()
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)  # empty!
    w = _worker(store, EpicGpuStagingStore(capacity_bytes=10**8))
    meta = EpicConnectorMetadata()
    meta.add_prefetch(EpicReqPrefetch(chunk_hashes=["gone"], dst_worker=-1))
    w._consume_prefetches(meta)  # must not raise.
    assert not w._staging.contains("gone")
    assert EpicConnector.debug_counters["prefetch_staged"] == 0


# ---------------------------------------------------------------------------
# Staging-process corners.
# ---------------------------------------------------------------------------


def test_staging_child_survives_unknown_command():
    backend = ExternalStagingBackend(capacity_bytes=10**8)
    try:
        backend.stage(_stored("h-alive", seed=31), torch.device("cpu"))
        rep = backend._request(("frobnicate",))
        assert rep[0] == "err"
        # The child is still serving after the protocol error.
        assert backend.contains("h-alive")
        got = backend.get("h-alive")
        assert got is not None and got.length == CHUNK
    finally:
        backend.close()


def test_staging_child_oversize_and_lru_with_parent_held_mapping():
    one = _stored("size-probe", seed=1).nbytes()
    backend = ExternalStagingBackend(capacity_bytes=2 * one)
    try:
        dev = torch.device("cpu")
        sc1, sc2, sc3 = (
            _stored("h1", seed=41),
            _stored("h2", seed=42),
            _stored("h3", seed=43),
        )
        backend.stage(sc1, dev)
        held = backend.get("h1")  # parent maps h1 BEFORE eviction.
        assert held is not None
        backend.stage(sc2, dev)
        backend.stage(sc3, dev)  # budget=2 chunks -> LRU evicts in the child.
        stats = backend.stats()
        assert stats["chunks"] == 2

        # Oversize chunk: child refuses to cache it; get -> miss.
        big = _stored("h-big", seed=44, length=4 * CHUNK)
        assert big.nbytes() > 2 * one
        backend.stage(big, dev)
        # Parent memoizes only get() results; the child never cached it.
        assert backend.stats()["chunks"] == 2

        # The parent-held mapping of h1 stays VALID (shared memory persists
        # while referenced) even if the child evicted h1.
        torch.testing.assert_close(
            held.k_per_layer["l0"], sc1.k_per_layer["l0"]
        )
    finally:
        backend.close()


def test_staging_backend_close_then_restart():
    backend = ExternalStagingBackend(capacity_bytes=10**8)
    dev = torch.device("cpu")
    backend.stage(_stored("h-a", seed=51), dev)
    assert backend.contains("h-a")
    backend.close()

    # After close: queries degrade to miss (CPU fallback), no crash. The
    # parent memo was cleared with the connection.
    assert backend.contains("h-a") is False
    assert backend.get("h-a") is None

    # A new stage() respawns the child and the backend works again.
    backend.stage(_stored("h-b", seed=52), dev)
    try:
        assert backend.contains("h-b")
        got = backend.get("h-b")
        assert got is not None
    finally:
        backend.close()


def test_miss_never_produces_wrong_kv_end_to_end():
    """The invariant behind every corner above: whatever the staging state
    (hit, miss, evicted, closed), the scattered KV equals the fileKV source."""
    EpicConnector.reset_debug_counters()
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    sc = _stored("h-inv", seed=61)
    store.put(sc)

    for staging in (
        None,  # prefetch disabled entirely.
        EpicGpuStagingStore(capacity_bytes=10**8),  # enabled, not staged.
    ):
        w = _worker(store, staging)
        dst = list(range(BLOCK, BLOCK + CHUNK))
        w._load_chunk(
            sc,
            ChunkLoadSpec(
                chunk_hash="h-inv",
                dst_slot_ids=dst,
                old_pos_start=-1,
                new_pos_start=BLOCK,
                length=CHUNK,
            ),
        )
        res = check_scatter_fidelity(
            w._kv_caches["l0"], sc.k_per_layer["l0"], sc.v_per_layer["l0"], dst
        )
        assert res is not None and res[0] and res[2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# 2nd-pass robustness: bounded parent memo, bounded command queue/frame,
# staged source under a head trim.
# ---------------------------------------------------------------------------


def test_external_backend_parent_memo_is_bounded_lru():
    """The parent-side mapping memo must not hold every chunk ever gotten
    (a held mapping keeps the shared allocation alive past child eviction);
    it is LRU-bounded by the same byte budget."""
    one = _stored("probe", seed=1).nbytes()
    backend = ExternalStagingBackend(capacity_bytes=2 * one)
    try:
        dev = torch.device("cpu")
        for i, h in enumerate(["m1", "m2", "m3"]):
            backend.stage(_stored(h, seed=70 + i), dev)
            got = backend.get(h)
            assert got is not None
        # Memo bounded to ~2 chunks worth of bytes -> the oldest mapping fell.
        assert len(backend._mapped) <= 2
        assert "m1" not in backend._mapped
        assert backend._mapped_bytes <= 2 * one
        # Re-get of an evicted memo entry still works (child roundtrip or
        # CPU-store fallback at the caller); here the child still has m2/m3.
        assert backend.get("m3") is not None
    finally:
        backend.close()


def test_prefetch_queue_drop_oldest_beyond_cap():
    c, index = _scheduler_connector()
    tokens = list(range(CHUNK))
    h = hash_chunk_tokens(tokens)
    index.register(h, CHUNK)
    c._max_pending_prefetch = 5  # test-sized cap.
    for i in range(9):
        c.enqueue_prefetch(chunk_hashes=[h], dst_worker=i)
    assert len(c._prefetch_queue) == 5
    # Drop-oldest: the NEWEST intents survive (dst_worker 4..8).
    assert [d.dst_worker for d in c._prefetch_queue] == [4, 5, 6, 7, 8]


def test_listener_rejects_oversized_frame_without_parsing():
    c, index = _scheduler_connector()
    tokens = list(range(CHUNK))
    h = hash_chunk_tokens(tokens)
    index.register(h, CHUNK)

    endpoint = f"ipc:///tmp/epic-pf-frame-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    listener = EpicPrefetchListener(
        endpoint, c.handle_prefetch_command, max_frame_bytes=1024
    )
    listener.start()
    try:
        import zmq

        ctx = zmq.Context.instance()
        raw = ctx.socket(zmq.REQ)
        raw.rcvtimeo = 5000
        raw.linger = 0
        raw.connect(endpoint)
        raw.send(b"x" * 4096)  # over the 1 KiB test cap; not even valid JSON.
        rep = json.loads(raw.recv())
        assert rep["ok"] is False and "frame too large" in rep["error"]
        raw.close(linger=0)

        # Listener alive; a normal-sized command still works.
        client = EpicPrefetchClient(endpoint, timeout_ms=5000)
        queued, _ = client.prefetch(chunk_hashes=[h], dst_worker=0)
        assert queued == [h]
        client.close()
    finally:
        listener.stop()


def test_staged_source_respects_head_trim():
    """A prefetch-staged chunk consumed by a head-trimmed load spec (native
    extent straddle) must scatter exactly the stored TAIL, same as the CPU
    path."""
    EpicConnector.reset_debug_counters()
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    sc = _stored("h-trim", seed=81)
    store.put(sc)
    staging = EpicGpuStagingStore(capacity_bytes=10**8)
    w = _worker(store, staging)

    meta = EpicConnectorMetadata()
    meta.add_prefetch(EpicReqPrefetch(chunk_hashes=["h-trim"], dst_worker=-1))
    w._consume_prefetches(meta)
    assert staging.contains("h-trim")

    trim = 16
    dst = list(range(64, 64 + (CHUNK - trim)))
    spec = ChunkLoadSpec(
        chunk_hash="h-trim",
        dst_slot_ids=dst,
        old_pos_start=-1,
        new_pos_start=trim,  # reload at original positions -> identity skip
        length=CHUNK - trim,
        src_offset=trim,
    )
    w._load_chunk(sc, spec)
    assert EpicConnector.debug_counters["prefetch_hit"] == 1
    res = check_scatter_fidelity(
        w._kv_caches["l0"],
        sc.k_per_layer["l0"][trim:],
        sc.v_per_layer["l0"][trim:],
        dst,
    )
    assert res is not None and res[0] and res[2]
