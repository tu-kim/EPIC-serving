# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mid-turn file-modification consistency + partial (line-range) reads.

Three layers under test:
  1. ENGINE invariant (needs no new code -- content addressing): a modified
     file's chunks can never serve the new prompt, and BOTH versions coexist
     correctly for concurrently running agents.
  2. Eviction lifecycle: on_file_modified -> evict_hashes directive -> the
     workers drop stale STAGED copies (in-process and external backend).
  3. Frontend catalog/bridge: version bookkeeping shared across agents,
     torn-render detection, re-warm of the new version, and range
     canonicalization / sub-chunk skip for partial reads.
"""

import threading

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
from vllm.distributed.kv_transfer.kv_connector.v1.epic.filekv_catalog import (
    FileKVCatalog,
    RangeKey,
    canonicalize_range,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
    EpicConnectorMetadata,
    EpicReqPrefetch,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch import (
    EpicGpuStagingStore,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch_parser import (
    ToolCallRead,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch_service import (
    DynamoPrefetchBridge,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    EpicSelection,
    IdentityAlignment,
    LegoLinkRecompute,
)

BLOCK = 16
CHUNK = 64


# ---------------------------------------------------------------------------
# Layer 1: engine content-addressing invariant (v1/v2 coexistence).
# ---------------------------------------------------------------------------


class _LiveIndex(EpicSchedulerIndex):
    def __init__(self, store):
        super().__init__(
            capacity_bytes=10**8,
            num_layers=1,
            num_kv_heads=1,
            head_size=1,
            cache_dtype_size=4,
        )
        self._backing = store

    def contains(self, h):
        return self._backing.contains(h) or super().contains(h)

    def get_length(self, h):
        ln = self._backing.get_length(h)
        return ln if ln is not None else super().get_length(h)

    def get_old_pos_start(self, h):
        old = self._backing.get_old_pos_start(h)
        return old if old is not None else super().get_old_pos_start(h)

    def get_chain(self, h):
        chain = self._backing.get_chain(h)
        if chain is not None and chain != (None, None):
            return chain
        return super().get_chain(h)


def _connector(store):
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
    c._max_sparse_rows = 0
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
    sc.k_per_layer["l0"] = torch.zeros(len(tokens), 1, 1)
    sc.v_per_layer["l0"] = torch.zeros(len(tokens), 1, 1)
    store.put(sc)
    return h


class _Req:
    def __init__(self, rid, tokens):
        self.request_id = rid
        self.prompt_token_ids = tokens


def test_modified_file_never_serves_new_prompt_and_versions_coexist():
    """Agent edits file C mid-turn (v1 -> v2). The engine store still holds
    v1's chunks. A NEW prompt embedding v2 must get ZERO reuse for C (no
    stale KV can leak); a concurrent agent whose prompt still embeds v1
    keeps its (correct) reuse. This is the content-addressing invariant that
    makes mid-turn modification safe without engine-side coordination."""
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    v1 = list(range(2000, 2000 + CHUNK))  # file C, version 1
    v2 = list(range(2000, 2000 + CHUNK))
    v2[10] = 999999  # one edited token -> different bytes.
    _store_chunk(store, v1)

    a = list(range(CHUNK))  # head segment (uncached)
    g = list(range(4000, 4000 + CHUNK))

    # Agent B (new turn, post-edit): prompt embeds v2 -> NO hit for C.
    c1 = _connector(store)
    ext_v2, _ = c1.get_num_new_matched_tokens(_Req("r-v2", a + v2 + g), 0)
    assert ext_v2 == 0  # v2 bytes never saw a save -> nothing external.
    assert "r-v2" not in c1._sparse_reqs

    # Agent A (in-flight context still embeds v1): reuse still works and is
    # CORRECT -- its prompt literally contains the old bytes.
    c2 = _connector(store)
    ext_v1, _ = c2.get_num_new_matched_tokens(_Req("r-v1", a + v1 + g), 0)
    assert ext_v1 == CHUNK
    assert "r-v1" in c2._sparse_reqs

    # After v2 is warmed (saved), BOTH versions serve their own prompts.
    _store_chunk(store, v2)
    c3 = _connector(store)
    ext_v2b, _ = c3.get_num_new_matched_tokens(_Req("r-v2b", a + v2 + g), 0)
    assert ext_v2b == CHUNK


# ---------------------------------------------------------------------------
# Layer 2: eviction lifecycle (engine side).
# ---------------------------------------------------------------------------


def _scheduler_connector():
    index = EpicSchedulerIndex(
        capacity_bytes=10**8,
        num_layers=1,
        num_kv_heads=1,
        head_size=4,
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


def test_evict_command_reaches_directive_even_without_new_hashes():
    c, _ = _scheduler_connector()
    reply = c.handle_prefetch_command(
        {"cmd": "prefetch", "evict_hashes": ["old1", "old2"], "dst_worker": 1}
    )
    assert reply["ok"] and reply["queued"] == []
    assert reply["evict_queued"] == 2
    assert len(c._prefetch_queue) == 1
    assert c._prefetch_queue[0].evict_hashes == ["old1", "old2"]
    assert c._prefetch_queue[0].chunk_hashes == []


def _worker(store, staging):
    w = object.__new__(EpicConnector)
    w._store = store
    w._staging = staging
    w._worker_id = 0
    w._layer_names = ["l0"]
    w._kv_caches = {"l0": torch.zeros(2, 8, BLOCK, 1, 4)}
    w._alignment = IdentityAlignment()
    w._debug_counters = True
    return w


def test_worker_evicts_stale_staging_before_staging_new_version():
    EpicConnector.reset_debug_counters()
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    v1_tokens = list(range(2000, 2000 + CHUNK))
    v2_tokens = list(range(3000, 3000 + CHUNK))
    h1 = _store_chunk(store, v1_tokens)
    h2 = _store_chunk(store, v2_tokens)
    staging = EpicGpuStagingStore(capacity_bytes=10**8)
    w = _worker(store, staging)

    # Turn t: v1 staged.
    meta1 = EpicConnectorMetadata()
    meta1.add_prefetch(EpicReqPrefetch(chunk_hashes=[h1], dst_worker=-1))
    w._consume_prefetches(meta1)
    assert staging.contains(h1)

    # File modified: one directive evicts v1 AND stages v2.
    meta2 = EpicConnectorMetadata()
    meta2.add_prefetch(
        EpicReqPrefetch(
            chunk_hashes=[h2], dst_worker=-1, evict_hashes=[h1]
        )
    )
    w._consume_prefetches(meta2)
    assert not staging.contains(h1)  # stale version reclaimed.
    assert staging.contains(h2)
    assert EpicConnector.debug_counters["prefetch_evicted"] == 1


# ---------------------------------------------------------------------------
# Layer 3: catalog + bridge (frontend side).
# ---------------------------------------------------------------------------


def test_canonicalize_range():
    assert canonicalize_range(None, None, 100) == (None, None)
    assert canonicalize_range(50, 100, 0) == (50, 100)  # snapping off.
    assert canonicalize_range(50, 100, 100) == (1, 100)
    assert canonicalize_range(60, 110, 100) == (1, 200)
    assert canonicalize_range(101, 150, 100) == (101, 200)
    assert canonicalize_range(None, 80, 100) == (1, 100)
    assert canonicalize_range(120, None, 100) == (101, None)


def test_catalog_versioning_and_invalidation():
    cat = FileKVCatalog()
    key_full = RangeKey("f.py")
    key_range = RangeKey("f.py", 1, 100)
    cat.record(key_full, "fp1", ["h1", "h2"])
    cat.record(key_range, "fp1", ["h2", "h3"])
    cat.record(RangeKey("other.py"), "fpX", ["hx"])
    assert cat.is_current(key_full, "fp1")
    assert cat.current_version("f.py") == 0

    records = cat.on_file_modified("f.py")
    assert cat.current_version("f.py") == 1
    stale = {h for r in records for h in r.chunk_hashes}
    assert stale == {"h1", "h2", "h3"}  # union over units, other.py untouched.
    assert cat.lookup(key_full) is None
    assert cat.lookup(RangeKey("other.py")) is not None

    # stale_check: fingerprint drift on a recorded unit yields its hashes.
    cat.record(key_full, "fp2", ["h4"])
    assert cat.stale_check(key_full, "fp2") == []
    assert cat.stale_check(key_full, "fp3") == ["h4"]
    assert cat.lookup(key_full) is None  # dropped by the check.


def test_catalog_thread_safety_smoke():
    cat = FileKVCatalog()
    errors: list[Exception] = []

    def worker(tid):
        try:
            for i in range(200):
                key = RangeKey(f"f{i % 5}.py")
                cat.record(key, f"fp{tid}-{i}", [f"h{tid}-{i}"])
                if i % 7 == 0:
                    cat.on_file_modified(key.path)
                cat.is_current(key, f"fp{tid}-{i}")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


class _FakeClient:
    """Engine double: knows `known` hashes; records every command."""

    def __init__(self, known):
        self.known = set(known)
        self.calls: list[dict] = []

    def prefetch(self, chunk_hashes=None, token_ids=None, dst_worker=-1,
                 evict_hashes=None):
        call = {
            "hashes": list(chunk_hashes or []),
            "evict": list(evict_hashes or []),
            "dst": dst_worker,
        }
        self.calls.append(call)
        queued = [h for h in call["hashes"] if h in self.known]
        return queued, [h for h in call["hashes"] if h not in self.known]


def _file_tokens(version: int, lines: "tuple[int, int] | None" = None):
    base = 5000 * version
    if lines is None:
        return list(range(base, base + 2 * CHUNK))  # whole file: 2 chunks.
    return list(range(base + lines[0], base + lines[0] + CHUNK))


def _bridge(client, catalog, fingerprints: dict, snap: int = 0, warm=None):
    """fingerprints: mutable {path: fp} the test flips to simulate edits."""

    def render(read):
        return f"{read.file_path}|{read.start_line}|{read.end_line}|" + (
            fingerprints[read.file_path]
        )

    tokens_by_render: dict[str, list[int]] = {}

    def tokenize(text):
        # Distinct deterministic tokens per distinct render text.
        if text not in tokens_by_render:
            seed = abs(hash(text)) % 10**6
            tokens_by_render[text] = list(range(seed, seed + CHUNK))
        return tokens_by_render[text]

    return DynamoPrefetchBridge(
        client=client,
        render_fn=render,
        tokenize_fn=tokenize,
        chunk_size=CHUNK,
        warm_fn=warm,
        catalog=catalog,
        fingerprint_fn=lambda path: fingerprints[path],
        snap_lines=snap,
    )


def test_bridge_on_file_modified_evicts_and_rewarns():
    cat = FileKVCatalog()
    fps = {"f.py": "v1"}
    warmed: list[str] = []
    client = _FakeClient(known=[])  # engine knows nothing -> drops -> warm.
    bridge = _bridge(
        client, cat, fps, warm=lambda read, text: warmed.append(text)
    )

    # Turn t: file read -> recorded + prefetched (dropped -> warmed).
    out1 = bridge.on_turn_response(
        "<tool_call><function=read>"
        "<parameter=filePath>f.py</parameter></function></tool_call>",
        dst_worker=0,
    )
    v1_hashes = out1["queued"] + out1["dropped"]
    assert len(cat.units_for("f.py")) == 1

    # Mid-turn edit: fingerprint flips; tool executor notifies the bridge.
    fps["f.py"] = "v2"
    out2 = bridge.on_file_modified("f.py", dst_worker=0)
    # Old hashes evicted on the workers; new version re-rendered + re-warmed.
    assert set(out2["evicted"]) == set(v1_hashes)
    assert client.calls[-1]["evict"] == out2["evicted"]
    assert out2["dropped"]  # new hashes unknown to the engine yet...
    assert len(warmed) == 2  # ...warm fired for turn t AND the re-warm.
    assert "v2" in warmed[-1]  # the re-warm rendered the NEW content.
    # Catalog now tracks the new version's unit.
    rec = cat.lookup(RangeKey("f.py"))
    assert rec is not None and rec.fingerprint == "v2" and rec.version == 1
    assert set(rec.chunk_hashes).isdisjoint(v1_hashes)


def test_bridge_torn_render_is_skipped():
    cat = FileKVCatalog()
    flips = iter(["v1", "v2", "v2", "v2"])  # changes DURING the first render.
    client = _FakeClient(known=[])
    bridge = DynamoPrefetchBridge(
        client=client,
        render_fn=lambda read: "text",
        tokenize_fn=lambda text: list(range(CHUNK)),
        chunk_size=CHUNK,
        catalog=cat,
        fingerprint_fn=lambda path: next(flips),
    )
    out = bridge.on_turn_response(
        "<tool_call><function=read>"
        "<parameter=filePath>f.py</parameter></function></tool_call>",
        dst_worker=0,
    )
    assert out == {"queued": [], "dropped": [], "warmed": []}
    assert client.calls == []  # torn render -> nothing sent, nothing recorded.
    assert len(cat.units_for("f.py")) == 0


def test_bridge_snaps_overlapping_ranges_to_one_unit():
    cat = FileKVCatalog()
    fps = {"f.py": "v1"}
    client = _FakeClient(known=[])
    bridge = _bridge(client, cat, fps, snap=100)

    reads = (
        "<tool_call><function=read><parameter=filePath>f.py</parameter>"
        "<parameter=startLine>50</parameter>"
        "<parameter=endLine>100</parameter></function></tool_call>"
    )
    out1 = bridge.on_turn_response(reads, dst_worker=0)
    reads2 = (
        "<tool_call><function=read><parameter=filePath>f.py</parameter>"
        "<parameter=startLine>1</parameter>"
        "<parameter=endLine>90</parameter></function></tool_call>"
    )
    out2 = bridge.on_turn_response(reads2, dst_worker=0)

    # Both requests snapped to canonical (1, 100) -> SAME render -> same
    # hashes -> one catalog unit; the second call is a pure repeat.
    assert out1["queued"] + out1["dropped"] == out2["queued"] + out2["dropped"]
    assert len(cat.units_for("f.py")) == 1
    assert cat.lookup(RangeKey("f.py", 1, 100)) is not None


def test_bridge_skips_subchunk_range_render():
    cat = FileKVCatalog()
    client = _FakeClient(known=[])
    bridge = DynamoPrefetchBridge(
        client=client,
        render_fn=lambda read: "tiny",
        tokenize_fn=lambda text: list(range(10)),  # 10 tokens << chunk.
        chunk_size=CHUNK,
        catalog=cat,
        fingerprint_fn=lambda path: "fp",
    )
    out = bridge.on_turn_response(
        "<tool_call><function=read><parameter=filePath>f.py</parameter>"
        "<parameter=startLine>50</parameter>"
        "<parameter=endLine>52</parameter></function></tool_call>",
        dst_worker=0,
    )
    assert out == {"queued": [], "dropped": [], "warmed": []}
    assert client.calls == []  # sub-chunk: not cacheable, recompute is cheap.
    assert len(cat.units_for("f.py")) == 0


def test_external_staging_evict_roundtrip():
    """The dedicated staging-worker process honors evict: both the parent
    mapping and the child copy are dropped; a later get is a clean miss."""
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.staging_worker import (
        ExternalStagingBackend,
    )

    backend = ExternalStagingBackend(capacity_bytes=10**8)
    try:
        sc = StoredChunk(
            chunk_hash="ev-h1",
            length=CHUNK,
            old_positions=torch.arange(CHUNK, dtype=torch.int64),
        )
        sc.k_per_layer["l0"] = torch.randn(CHUNK, 1, 4)
        sc.v_per_layer["l0"] = torch.randn(CHUNK, 1, 4)
        backend.stage(sc, torch.device("cpu"))
        assert backend.get("ev-h1") is not None  # parent mapping exists.

        backend.evict("ev-h1")
        assert not backend.contains("ev-h1")
        assert backend.get("ev-h1") is None  # child copy gone too.
        assert backend.stats()["chunks"] == 0
        assert backend._mapped_bytes == 0
    finally:
        backend.close()
