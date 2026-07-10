# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""feature/prefetch CPU tests: tool-call parsing, the FileKVPrefetcher
pipeline, the GPU staging store (exercised on CPU), and the connector
integration (enqueue -> metadata -> worker staging -> load fast path with
fallback to the CPU fileKV store)."""

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
    EpicReqPrefetch,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch import (
    EpicGpuStagingStore,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch_parser import (
    FileKVPrefetcher,
    parse_tool_call_reads,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    IdentityAlignment,
)

BLOCK = 16
CHUNK = 64
HEADS, HD = 1, 4


# ---------------------------------------------------------------------------
# Parser.
# ---------------------------------------------------------------------------


def test_parse_xmlish_tool_call_with_line_range():
    text = (
        "I'll read the file first.\n"
        "<tool_call><function=read>\n"
        "<parameter=filePath>src/foo.py</parameter>\n"
        "<parameter=startLine>10</parameter>\n"
        "<parameter=endLine>80</parameter>\n"
        "</function></tool_call>"
    )
    reads = parse_tool_call_reads(text)
    assert len(reads) == 1
    assert reads[0].file_path == "src/foo.py"
    assert reads[0].start_line == 10
    assert reads[0].end_line == 80


def test_parse_multiple_calls_skips_non_read_functions():
    text = (
        "<tool_call><function=read>"
        "<parameter=filePath>a.py</parameter>"
        "</function></tool_call>"
        "<tool_call><function=write>"
        "<parameter=filePath>b.py</parameter>"
        "</function></tool_call>"
        "<tool_call><function=read_file>"
        "<parameter=path>c.py</parameter>"
        "</function></tool_call>"
    )
    reads = parse_tool_call_reads(text)
    assert [r.file_path for r in reads] == ["a.py", "c.py"]
    assert reads[0].start_line is None and reads[0].end_line is None


def test_parse_openai_json_tool_call():
    text = (
        'thinking... {"name": "read", "arguments": '
        '{"filePath": "pkg/mod.py", "startLine": 5}} done'
    )
    reads = parse_tool_call_reads(text)
    assert len(reads) == 1
    assert reads[0].file_path == "pkg/mod.py"
    assert reads[0].start_line == 5


def test_parse_malformed_returns_empty():
    assert parse_tool_call_reads("no tool calls here") == []
    assert parse_tool_call_reads(
        "<tool_call><function=read></function></tool_call>"  # no path param
    ) == []
    assert parse_tool_call_reads(
        '{"name": "read", "arguments": "{broken json"}'
    ) == []


# ---------------------------------------------------------------------------
# FileKVPrefetcher pipeline.
# ---------------------------------------------------------------------------


def test_prefetcher_hashes_match_connector_grid_and_dedup():
    # Two reads of the SAME file -> rendered/tokenized once each, hashes
    # deduplicated; padding to the chunk grid must match hash_chunk_tokens.
    shipped: list[tuple[list[str], int]] = []
    file_tokens = list(range(100))  # 100 tokens -> pads to 128 = 2 chunks.

    prefetcher = FileKVPrefetcher(
        render_fn=lambda read: read.file_path,
        tokenize_fn=lambda text: list(file_tokens),
        transport_fn=lambda hashes, dst: shipped.append((hashes, dst)),
        chunk_size=CHUNK,
        pad_token_id=0,
    )
    out = prefetcher.prefetch_for_output(
        "<tool_call><function=read>"
        "<parameter=filePath>x.py</parameter></function></tool_call>"
        "<tool_call><function=read>"
        "<parameter=filePath>x.py</parameter></function></tool_call>",
        dst_worker=3,
    )
    padded = file_tokens + [0] * (2 * CHUNK - len(file_tokens))
    expected = [
        hash_chunk_tokens(padded[:CHUNK]),
        hash_chunk_tokens(padded[CHUNK:]),
    ]
    assert out == expected  # deduplicated across the duplicate read.
    assert shipped == [(expected, 3)]


def test_prefetcher_no_reads_ships_nothing():
    shipped = []
    prefetcher = FileKVPrefetcher(
        render_fn=lambda r: "",
        tokenize_fn=lambda t: [],
        transport_fn=lambda h, d: shipped.append((h, d)),
        chunk_size=CHUNK,
    )
    assert prefetcher.prefetch_for_output("plain text output", 0) == []
    assert shipped == []


def test_prefetcher_render_failure_skips_that_file_only():
    shipped = []

    def render(read):
        if read.file_path == "bad.py":
            raise FileNotFoundError(read.file_path)
        return read.file_path

    prefetcher = FileKVPrefetcher(
        render_fn=render,
        tokenize_fn=lambda t: list(range(CHUNK)),
        transport_fn=lambda h, d: shipped.append((h, d)),
        chunk_size=CHUNK,
    )
    out = prefetcher.prefetch_for_output(
        "<tool_call><function=read>"
        "<parameter=filePath>bad.py</parameter></function></tool_call>"
        "<tool_call><function=read>"
        "<parameter=filePath>good.py</parameter></function></tool_call>",
        dst_worker=1,
    )
    assert out == [hash_chunk_tokens(list(range(CHUNK)))]
    assert shipped == [(out, 1)]


# ---------------------------------------------------------------------------
# GPU staging store (CPU device).
# ---------------------------------------------------------------------------


def _stored(hash_: str, length: int = CHUNK, seed: int = 0) -> StoredChunk:
    g = torch.Generator().manual_seed(seed)
    sc = StoredChunk(
        chunk_hash=hash_,
        length=length,
        old_positions=torch.arange(length, dtype=torch.int64),
    )
    sc.k_per_layer["l0"] = torch.randn(length, HEADS, HD, generator=g)
    sc.v_per_layer["l0"] = torch.randn(length, HEADS, HD, generator=g)
    return sc


def test_staging_store_stage_get_roundtrip():
    store = EpicGpuStagingStore(capacity_bytes=10**8)
    sc = _stored("h1")
    staged = store.stage(sc, torch.device("cpu"))
    assert store.contains("h1")
    got = store.get("h1")
    assert got is staged
    torch.testing.assert_close(got.k_per_layer["l0"], sc.k_per_layer["l0"])
    torch.testing.assert_close(got.v_per_layer["l0"], sc.v_per_layer["l0"])
    assert torch.equal(got.old_positions, sc.old_positions)
    assert got.ready_event is None  # CPU staging is synchronous.
    # Idempotent re-stage: same object, no duplicate bytes.
    bytes_before = store.current_bytes
    assert store.stage(sc, torch.device("cpu")) is staged
    assert store.current_bytes == bytes_before


def test_staging_store_lru_eviction_by_budget():
    one_chunk_bytes = _stored("x").nbytes()
    store = EpicGpuStagingStore(capacity_bytes=2 * one_chunk_bytes)
    store.stage(_stored("h1", seed=1), torch.device("cpu"))
    store.stage(_stored("h2", seed=2), torch.device("cpu"))
    assert store.get("h1") is not None  # touch h1 -> h2 becomes LRU.
    store.stage(_stored("h3", seed=3), torch.device("cpu"))
    assert store.contains("h1")
    assert not store.contains("h2")  # evicted (LRU).
    assert store.contains("h3")


def test_staging_store_oversize_chunk_not_cached():
    small = EpicGpuStagingStore(capacity_bytes=8)
    staged = small.stage(_stored("big"), torch.device("cpu"))
    assert staged is not None  # returned for one-shot use...
    assert not small.contains("big")  # ...but not cached.
    assert len(small) == 0


# ---------------------------------------------------------------------------
# Connector integration (scheduler enqueue -> meta; worker staging -> load).
# ---------------------------------------------------------------------------


def _scheduler_connector(index: EpicSchedulerIndex):
    import threading

    c = object.__new__(EpicConnector)
    c._block_size = BLOCK
    c._chunk_size = CHUNK
    c._store = None
    c._index = index
    c._prefetch_lock = threading.Lock()
    c._prefetch_queue = []
    return c


def test_enqueue_prefetch_filters_unknown_and_drains_once():
    index = EpicSchedulerIndex(
        capacity_bytes=10**8,
        num_layers=1,
        num_kv_heads=HEADS,
        head_size=HD,
        cache_dtype_size=4,
    )
    known_tokens = list(range(CHUNK))
    known_hash = hash_chunk_tokens(known_tokens)
    index.register(known_hash, CHUNK)

    c = _scheduler_connector(index)
    queued = c.enqueue_prefetch(
        chunk_hashes=["deadbeef"],  # unknown -> dropped
        token_ids=known_tokens,  # known via the token path
        dst_worker=2,
    )
    assert queued == [known_hash]

    # Drain into metadata exactly once.
    meta = EpicConnectorMetadata()
    if getattr(c, "_prefetch_queue", None):
        with c._prefetch_lock:
            pending, c._prefetch_queue = c._prefetch_queue, []
        for d in pending:
            meta.add_prefetch(d)
    assert len(meta.prefetches) == 1
    assert meta.prefetches[0].chunk_hashes == [known_hash]
    assert meta.prefetches[0].dst_worker == 2
    assert c._prefetch_queue == []  # drained.

    # Nothing known -> nothing queued.
    assert c.enqueue_prefetch(chunk_hashes=["unknown"]) == []


def _worker_connector(store, staging, worker_id: int):
    w = object.__new__(EpicConnector)
    w._store = store
    w._staging = staging
    w._worker_id = worker_id
    w._layer_names = ["l0"]
    w._kv_caches = {"l0": torch.zeros(2, 8, BLOCK, HEADS, HD)}
    w._alignment = IdentityAlignment()
    w._debug_counters = True
    return w


def test_consume_prefetches_respects_worker_targeting():
    EpicConnector.reset_debug_counters()
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    sc = _stored("h1", seed=7)
    store.put(sc)
    staging = EpicGpuStagingStore(capacity_bytes=10**8)

    meta = EpicConnectorMetadata()
    meta.add_prefetch(EpicReqPrefetch(chunk_hashes=["h1"], dst_worker=1))

    # Worker 0: directive addressed to worker 1 -> must NOT stage.
    w0 = _worker_connector(store, EpicGpuStagingStore(10**8), worker_id=0)
    w0._consume_prefetches(meta)
    assert not w0._staging.contains("h1")

    # Worker 1: stages.
    w1 = _worker_connector(store, staging, worker_id=1)
    w1._consume_prefetches(meta)
    assert staging.contains("h1")
    assert EpicConnector.debug_counters["prefetch_staged"] == 1

    # Broadcast directive (-1) reaches everyone.
    meta2 = EpicConnectorMetadata()
    meta2.add_prefetch(EpicReqPrefetch(chunk_hashes=["h1"], dst_worker=-1))
    w0._consume_prefetches(meta2)
    assert w0._staging.contains("h1")


def test_load_chunk_uses_staged_copy_and_counts_hit():
    EpicConnector.reset_debug_counters()
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    sc = _stored("h1", seed=11)
    store.put(sc)
    staging = EpicGpuStagingStore(capacity_bytes=10**8)
    w = _worker_connector(store, staging, worker_id=0)

    meta = EpicConnectorMetadata()
    meta.add_prefetch(EpicReqPrefetch(chunk_hashes=["h1"], dst_worker=-1))
    w._consume_prefetches(meta)
    assert staging.contains("h1")

    dst_slots = list(range(32, 32 + CHUNK))
    spec = ChunkLoadSpec(
        chunk_hash="h1",
        dst_slot_ids=dst_slots,
        old_pos_start=-1,
        new_pos_start=32,
        length=CHUNK,
    )
    w._load_chunk(sc, spec)
    assert EpicConnector.debug_counters["prefetch_hit"] == 1
    assert EpicConnector.debug_counters["prefetch_miss"] == 0
    # Scattered content == the staged (== stored, IdentityAlignment) K/V.
    res = check_scatter_fidelity(
        w._kv_caches["l0"], sc.k_per_layer["l0"], sc.v_per_layer["l0"], dst_slots
    )
    assert res is not None and res[0] and res[2]


def test_load_chunk_miss_falls_back_to_cpu_store():
    EpicConnector.reset_debug_counters()
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    sc = _stored("h1", seed=13)
    store.put(sc)
    # Staging enabled but EMPTY -> miss counted, CPU fileKV path used.
    w = _worker_connector(store, EpicGpuStagingStore(10**8), worker_id=0)

    dst_slots = list(range(0, CHUNK))
    spec = ChunkLoadSpec(
        chunk_hash="h1",
        dst_slot_ids=dst_slots,
        old_pos_start=-1,
        new_pos_start=0,
        length=CHUNK,
    )
    w._load_chunk(sc, spec)
    assert EpicConnector.debug_counters["prefetch_hit"] == 0
    assert EpicConnector.debug_counters["prefetch_miss"] == 1
    res = check_scatter_fidelity(
        w._kv_caches["l0"], sc.k_per_layer["l0"], sc.v_per_layer["l0"], dst_slots
    )
    assert res is not None and res[0] and res[2]


def test_staged_copy_survives_cpu_store_eviction():
    """The fallback chain in start_load_kv: CPU store evicted the chunk but the
    GPU staging copy survives -> StagedChunk serves the load (duck-compatible
    with StoredChunk for the fields _load_chunk reads)."""
    EpicConnector.reset_debug_counters()
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    sc = _stored("h1", seed=17)
    store.put(sc)
    staging = EpicGpuStagingStore(capacity_bytes=10**8)
    w = _worker_connector(store, staging, worker_id=0)

    meta = EpicConnectorMetadata()
    meta.add_prefetch(EpicReqPrefetch(chunk_hashes=["h1"], dst_worker=-1))
    w._consume_prefetches(meta)

    # Evict from the CPU store; staging still holds the copy.
    store.put(
        StoredChunk(
            chunk_hash="h1",
            length=0,
            old_positions=torch.empty(0, dtype=torch.int64),
        )
    )
    staged = staging.get("h1")
    assert staged is not None

    dst_slots = list(range(16, 16 + CHUNK))
    spec = ChunkLoadSpec(
        chunk_hash="h1",
        dst_slot_ids=dst_slots,
        old_pos_start=-1,
        new_pos_start=16,
        length=CHUNK,
    )
    # start_load_kv would pass the staged chunk as the source; do the same.
    w._load_chunk(staged, spec)
    res = check_scatter_fidelity(
        w._kv_caches["l0"], sc.k_per_layer["l0"], sc.v_per_layer["l0"], dst_slots
    )
    assert res is not None and res[0] and res[2]
