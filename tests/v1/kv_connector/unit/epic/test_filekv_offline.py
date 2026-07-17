# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline fileKV build (directory scan -> host-DRAM KVBM store) tests.

CPU-only. Covers: deterministic scanning with filters, the hash/chain plan
oracle, the KvbmChunkStore adapter (round-trip through serialized host-DRAM
blobs, membership-without-deserialization, LRU parity with EpicChunkStore,
external pool loss), the builder (direct + engine-backed warm shapes,
incremental re-scan, modification invalidation, manifest), and duck
compatibility with the selection layer and the worker load path.
"""

import json

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    EpicChunkStore,
    StoredChunk,
    chain_hash_tokens,
    hash_chunk_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.filekv_catalog import (
    FileKVCatalog,
    RangeKey,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.filekv_offline import (
    OfflineFileKVBuilder,
    default_render,
    file_fingerprint,
    plan_chunks,
    scan_directory,
    write_manifest,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.kvbm_store import (
    DynamoKvbmHostPool,
    KvbmChunkStore,
    PinnedHostPool,
)

CHUNK = 8  # small grid so tests stay tiny
HEADS, HD = 1, 4


def _tokenize(text: str) -> list[int]:
    # Deterministic toy tokenizer: one id per whitespace token.
    return [(hash(w) & 0x7FFFFFFF) for w in text.split()]


def _chunk(chunk_hash: str, length: int = CHUNK, seed: int = 0,
           old_start: int = 0) -> StoredChunk:
    g = torch.Generator().manual_seed(seed)
    return StoredChunk(
        chunk_hash=chunk_hash,
        length=length,
        old_positions=torch.arange(old_start, old_start + length),
        k_per_layer={"l0": torch.randn(length, HEADS, HD, generator=g)},
        v_per_layer={"l0": torch.randn(length, HEADS, HD, generator=g)},
        chain_start="cs",
        chain_end="ce",
    )


def _direct_warm(padded, plans):
    """Direct-shape warm: fabricate KV tensors for each planned chunk."""
    return [
        _chunk("placeholder", length=p.length, seed=i, old_start=p.start)
        for i, p in enumerate(plans)
    ]


def _builder(store=None, catalog=None, warm=_direct_warm):
    if store is None:  # NOTE: `store or ...` would drop an EMPTY store
        store = KvbmChunkStore(pool=PinnedHostPool(10**8, pin_memory=False),
                               capacity_bytes=10**8, pin_memory=False)
    return OfflineFileKVBuilder(
        store=store, tokenize_fn=_tokenize, warm_fn=warm,
        catalog=catalog or FileKVCatalog(), chunk_size=CHUNK, pad_token_id=0)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def _tree(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "b.py").write_text("def b(): pass\n" * 20)
    (tmp_path / "src" / "a.py").write_text("def a(): pass\n" * 20)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("noise")
    (tmp_path / "img.bin").write_bytes(b"\x00\x01\x02" * 100)
    (tmp_path / "README.md").write_text("hello world " * 50)
    return tmp_path


def test_scan_is_deterministic_and_filters(tmp_path):
    root = _tree(tmp_path)
    files = scan_directory(root)
    rels = [str(p.relative_to(root)) for p in files]
    assert rels == ["README.md", "src/a.py", "src/b.py"]  # sorted, no .git/bin
    assert scan_directory(root) == files  # deterministic


def test_scan_include_exclude_and_size_cap(tmp_path):
    root = _tree(tmp_path)
    only_py = scan_directory(root, include=("**/*.py",))
    assert [p.name for p in only_py] == ["a.py", "b.py"]
    no_b = scan_directory(root, include=("**/*.py",), exclude=("**/b.py",))
    assert [p.name for p in no_b] == ["a.py"]
    assert scan_directory(root, max_file_bytes=10) == []


# ---------------------------------------------------------------------------
# Hash plan oracle
# ---------------------------------------------------------------------------


def test_plan_chunks_matches_store_hash_and_chain_oracles():
    ids = list(range(1, 2 * CHUNK + 3))  # forces padding on the tail chunk
    padded, plans = plan_chunks(ids, CHUNK, pad_token_id=0)
    assert len(padded) == 3 * CHUNK and padded[len(ids):] == [0] * (CHUNK - 2)
    assert [p.chunk_hash for p in plans] == [
        hash_chunk_tokens(padded[i:i + CHUNK])
        for i in range(0, len(padded), CHUNK)
    ]
    for p in plans:  # chains == digests of the padded prefix (save-time oracle)
        assert p.chain_start == chain_hash_tokens(padded[:p.start])
        assert p.chain_end == chain_hash_tokens(padded[:p.start + p.length])
    assert plan_chunks([], CHUNK, 0) == ([], [])


# ---------------------------------------------------------------------------
# KvbmChunkStore adapter
# ---------------------------------------------------------------------------


def test_kvbm_store_round_trip_preserves_tensors():
    pool = PinnedHostPool(10**8, pin_memory=False)
    store = KvbmChunkStore(pool=pool, capacity_bytes=10**8, pin_memory=False)
    src = _chunk("h1", seed=3, old_start=42)
    store.put(src)
    assert pool.used_bytes > 0  # payload actually lives in the host pool
    out = store.get("h1")
    assert out is not None and out.chunk_hash == "h1"
    assert torch.equal(out.old_positions, src.old_positions)
    assert torch.equal(out.k_per_layer["l0"], src.k_per_layer["l0"])
    assert torch.equal(out.v_per_layer["l0"], src.v_per_layer["l0"])
    assert (out.chain_start, out.chain_end) == ("cs", "ce")


def test_kvbm_store_membership_needs_no_blob():
    """Scheduler-side queries must work off the local index even when the
    pool's payload is gone (contains/get_length/get_chain/old_pos)."""
    pool = PinnedHostPool(10**8, pin_memory=False)
    store = KvbmChunkStore(pool=pool, capacity_bytes=10**8, pin_memory=False)
    store.put(_chunk("h1", old_start=7))
    # Sabotage the pool directly; index-only queries still answer.
    assert pool.evict("h1")
    assert store.contains("h1") and store.get_length("h1") == CHUNK
    assert store.get_old_pos_start("h1") == 7
    assert store.get_chain("h1") == ("cs", "ce")
    # A tensor read discovers the loss and self-heals the index.
    assert store.get("h1") is None
    assert not store.contains("h1") and store.current_bytes == 0


def test_kvbm_store_lru_parity_with_epic_store():
    """Same put sequence + tight budget -> same surviving membership as
    EpicChunkStore (the scheduler mirror's determinism depends on this)."""
    chunks = [_chunk(f"h{i}", seed=i) for i in range(6)]
    budget = sum(len(__import__(
        "vllm.distributed.kv_transfer.kv_connector.v1.epic.kvbm_store",
        fromlist=["_serialize"])._serialize(c)) for c in chunks[:3])
    kvbm = KvbmChunkStore(pool=PinnedHostPool(10**9, pin_memory=False),
                          capacity_bytes=budget, pin_memory=False)
    for c in chunks:
        kvbm.put(_chunk(c.chunk_hash, seed=int(c.chunk_hash[1:])))
    # Only a suffix survives, in insertion order (LRU evicts oldest).
    surviving = [h for h, *_ in kvbm.iter_membership()]
    assert surviving == [c.chunk_hash for c in chunks[-len(surviving):]]
    assert 1 <= len(surviving) <= 5
    # Byte accounting matches the pool exactly.
    assert kvbm.current_bytes == kvbm._pool.used_bytes


def test_kvbm_store_oversize_chunk_refused():
    store = KvbmChunkStore(pool=PinnedHostPool(10**8, pin_memory=False),
                           capacity_bytes=100, pin_memory=False)
    store.put(_chunk("big"))
    assert not store.contains("big") and len(store) == 0


def test_dynamo_adapter_validates_and_delegates():
    class GoodPool:
        def __init__(self):
            self.d = {}

        def put(self, k, p):
            self.d[k] = p
            return True

        def get(self, k):
            return self.d.get(k)

        def contains(self, k):
            return k in self.d

        def evict(self, k):
            return self.d.pop(k, None) is not None

    adapter = DynamoKvbmHostPool(GoodPool())
    assert adapter.put("k", b"v") and adapter.get("k") == b"v"
    assert adapter.contains("k") and adapter.evict("k")

    with pytest.raises(TypeError, match="lacks required methods"):
        DynamoKvbmHostPool(object())


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_build_directory_lands_chunks_and_manifest(tmp_path):
    root = _tree(tmp_path)
    b = _builder()
    results = b.build_directory(root, include=("**/*.py",))
    assert [r.path for r in results] == ["src/a.py", "src/b.py"]
    for r in results:
        assert r.error is None and not r.skipped and r.chunk_hashes
        # Every planned chunk is resident in the host pool via the store.
        assert all(b._store.contains(h) for h in r.chunk_hashes)
        # Hashes/chains come from the token oracle, not from the warm.
        text = (root / r.path).read_text()
        padded, plans = plan_chunks(
            _tokenize(default_render(r.path, text)), CHUNK, 0)
        assert r.chunk_hashes == [p.chunk_hash for p in plans]
        assert r.chains == [(p.chain_start, p.chain_end) for p in plans]
        # Catalog now serves this unit.
        rec = b.catalog.lookup(RangeKey(path=r.path))
        assert rec is not None and rec.chunk_hashes == r.chunk_hashes

    out = tmp_path / "manifest.json"
    write_manifest(results, out, root=str(root), chunk_size=CHUNK)
    doc = json.loads(out.read_text())
    assert doc["chunk_size"] == CHUNK
    assert [f["path"] for f in doc["files"]] == ["src/a.py", "src/b.py"]
    assert all(f["chunk_hashes"] for f in doc["files"])


def test_rescan_skips_unchanged_and_rebuilds_modified(tmp_path):
    root = _tree(tmp_path)
    b = _builder()
    first = b.build_directory(root, include=("**/*.py",))
    old_hashes = {r.path: r.chunk_hashes for r in first}

    second = b.build_directory(root, include=("**/*.py",))
    assert all(r.skipped for r in second)  # fingerprint + residency current
    assert {r.path: r.chunk_hashes for r in second} == old_hashes

    (root / "src" / "a.py").write_text("def a2(): return 1\n" * 25)
    third = b.build_directory(root, include=("**/*.py",))
    by_path = {r.path: r for r in third}
    assert not by_path["src/a.py"].skipped
    assert by_path["src/a.py"].chunk_hashes != old_hashes["src/a.py"]
    assert by_path["src/b.py"].skipped


def test_rescan_rebuilds_when_chunks_evicted(tmp_path):
    """Catalog current but store lost the chunks (LRU churn) -> re-warm."""
    root = _tree(tmp_path)
    store = KvbmChunkStore(pool=PinnedHostPool(10**8, pin_memory=False),
                           capacity_bytes=10**8, pin_memory=False)
    b = _builder(store=store)
    first = b.build_directory(root, include=("**/*.py",))
    victim = first[0].chunk_hashes[0]
    store._pool.evict(victim)
    assert store.get(victim) is None  # self-heals index
    second = b.build_directory(root, include=("**/*.py",))
    by_path = {r.path: r for r in second}
    assert not by_path[first[0].path].skipped
    assert store.contains(victim)


def test_engine_backed_warm_verifies_membership(tmp_path):
    """warm_fn returning None == engine's save path stored the chunks itself;
    the builder must verify residency and report a miss as an error."""
    root = _tree(tmp_path)
    store = KvbmChunkStore(pool=PinnedHostPool(10**8, pin_memory=False),
                           capacity_bytes=10**8, pin_memory=False)

    def engine_warm(padded, plans):
        for i, p in enumerate(plans):  # simulate the connector's save
            c = _chunk(p.chunk_hash, length=p.length, seed=i,
                       old_start=p.start)
            c.chain_start, c.chain_end = p.chain_start, p.chain_end
            store.put(c)
        return None

    ok = _builder(store=store, warm=engine_warm).build_directory(
        root, include=("**/a.py",))
    assert ok[0].error is None and ok[0].chunk_hashes

    lossy_store = KvbmChunkStore(pool=PinnedHostPool(10**8, pin_memory=False),
                                 capacity_bytes=10**8, pin_memory=False)
    missed = _builder(store=lossy_store, warm=lambda p, pl: None)
    bad = missed.build_directory(root, include=("**/a.py",))
    assert bad[0].error is not None and "not resident" in bad[0].error
    # A failed build must NOT be recorded as servable.
    assert missed.catalog.lookup(RangeKey(path="src/a.py")) is None


def test_warm_length_mismatch_is_an_error_not_a_partial_store(tmp_path):
    root = _tree(tmp_path)
    b = _builder(warm=lambda padded, plans: [_chunk("x")])  # wrong count
    res = b.build_directory(root, include=("**/a.py",))
    assert res[0].error is not None and "plan expects" in res[0].error
    assert len(b._store) == 0


# ---------------------------------------------------------------------------
# Duck compatibility with the online layers
# ---------------------------------------------------------------------------


def test_selection_layer_accepts_kvbm_store(tmp_path):
    """EpicSelection consumes SupportsChunkMembership; a KVBM-backed store
    must produce the same fold decisions as the CPU store."""
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (  # noqa: E501
        EpicSelection,
    )
    ids = list(range(100, 100 + 2 * CHUNK))
    padded, plans = plan_chunks(ids, CHUNK, 0)

    def fill(store):
        for i, p in enumerate(plans):
            c = _chunk(p.chunk_hash, length=p.length, seed=i,
                       old_start=p.start)
            c.chain_start, c.chain_end = p.chain_start, p.chain_end
            store.put(c)
        return store

    kvbm = fill(KvbmChunkStore(pool=PinnedHostPool(10**8, pin_memory=False),
                               capacity_bytes=10**8, pin_memory=False))
    cpu = fill(EpicChunkStore(capacity_bytes=10**8, pin_memory=False))
    sel = EpicSelection(strict_prefix_chain=True)
    chunk_entries = [(p.start, p.length, p.chunk_hash, p.chain_start,
                      p.chain_end) for p in plans]
    a = sel.select(None, 0, kvbm, chunk_entries)
    b = sel.select(None, 0, cpu, chunk_entries)
    assert a.prefix_extent == b.prefix_extent > 0
    assert [(h.chunk_hash, h.prompt_offset) for h in a.non_prefix_hits] == [
        (h.chunk_hash, h.prompt_offset) for h in b.non_prefix_hits
    ]


def test_worker_load_path_reads_from_kvbm_store():
    """The connector's staging consume path (store.get -> GPU/CPU staging)
    works against the KVBM-backed store unchanged."""
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (  # noqa: E501
        EpicConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
        EpicConnectorMetadata,
        EpicReqPrefetch,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch import (
        EpicGpuStagingStore,
    )

    store = KvbmChunkStore(pool=PinnedHostPool(10**8, pin_memory=False),
                           capacity_bytes=10**8, pin_memory=False)
    store.put(_chunk("h1", seed=5))

    w = object.__new__(EpicConnector)
    w._store = store
    w._staging = EpicGpuStagingStore(capacity_bytes=10**8)
    w._worker_id = 0
    w._debug_counters = False
    w._kv_caches = {"l0": torch.zeros(2, 8, 16, HEADS, HD)}

    meta = EpicConnectorMetadata()
    meta.add_prefetch(EpicReqPrefetch(chunk_hashes=["h1"], dst_worker=-1))
    w._consume_prefetches(meta)
    assert w._staging.contains("h1")
