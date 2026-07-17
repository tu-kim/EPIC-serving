# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-pipeline E2E (CPU): offline fileKV build -> dynamo prefetch -> online
non-contiguous reuse, with numerical KV fidelity at the end.

One continuous story, every leg through the REAL production code path:

  1. OFFLINE  -- scan a source directory, warm each file (real RoPE'd KV at
                 file-local positions), land the chunks in host DRAM through
                 the KVBM interface (KvbmChunkStore over PinnedHostPool),
                 write the prefetch manifest + FileKVCatalog.
  2. FRONTEND -- the previous turn's decode output names the files in tool
                 calls; DynamoPrefetchBridge parses/renders/hashes them and
                 injects a prefetch command over a REAL ZMQ round trip
                 (EpicPrefetchClient -> EpicPrefetchListener ->
                 handle_prefetch_command on the scheduler connector).
  3. SCHEDULER-- the next turn's prompt A+C+D+F+G arrives (C, F == the file
                 renders at NON-prefix offsets); get_num_new_matched_tokens
                 counts the external reuse, build_connector_meta emits the
                 chunk loads, the sparse plan M, AND drains the prefetch
                 directive into the same step's metadata.
  4. BOUNDARY -- the metadata crosses the scheduler->worker process boundary
                 (pickle round trip).
  5. WORKER   -- start_load_kv consumes the directive (stages the chunks out
                 of host DRAM), then loads them via the staged fast path and
                 scatters into the paged cache with PIC re-rotation.
  6. FIDELITY -- every scattered K equals the ground truth (the SAME raw K
                 RoPE'd directly at its NEW prompt position); V is identical;
                 untouched blocks stay zero; new segments are saved with
                 context chains while reused file chunks are NOT re-saved.

Dimensions and helpers are shared with test_functional_lifecycle so the
numerics (float64, atol 1e-6 through PIC) match the proven baselines.
"""

import os
import uuid

import pytest
import torch

from tests.v1.kv_connector.unit.epic.test_functional_lifecycle import (
    BASE,
    BLOCK,
    CHUNK,
    HEAD_SIZE,
    LAYER,
    NUM_KV_HEADS,
    ROTARY_DIM,
    _block_ids_for,
    _make_paged_cache,
    _make_scheduler,
    _make_worker,
    _neox_rope_reference,
    _NewReq,
    _pickle_roundtrip,
    _Req,
    _SchedOut,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    StoredChunk,
    hash_chunk_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
    EpicConnector,
    _slot_ids_from_blocks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.filekv_catalog import (
    FileKVCatalog,
    RangeKey,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.filekv_offline import (
    OfflineFileKVBuilder,
    write_manifest,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.kvbm_store import (
    KvbmChunkStore,
    PinnedHostPool,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch import (
    EpicGpuStagingStore,
)

zmq = pytest.importorskip("zmq")

from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch_service import (  # noqa: E402,E501
    DynamoPrefetchBridge,
    EpicPrefetchClient,
    EpicPrefetchListener,
)

LINK = 8
FILE_CHUNKS = 2  # each code file renders to exactly 2 grid chunks


def _tokenize(text: str) -> list[int]:
    return [int(w) for w in text.split()]


def _file_text(base: int) -> str:
    return " ".join(str(base + i) for i in range(FILE_CHUNKS * CHUNK))


class _E2EWorld:
    """The full offline+frontend+engine wiring for one E2E run."""

    def __init__(self, tmp_path):
        # --- source directory: two "code files" C.py / F.py ---------------
        self.root = tmp_path / "repo"
        self.root.mkdir()
        (self.root / "C.py").write_text(_file_text(1000))
        (self.root / "F.py").write_text(_file_text(5000))

        # --- host-DRAM store behind the KVBM interface ---------------------
        self.store = KvbmChunkStore(
            pool=PinnedHostPool(10**8, pin_memory=False),
            capacity_bytes=10**8,
            pin_memory=False,
        )
        self.catalog = FileKVCatalog()
        # Ground truth: chunk_hash -> (k_raw, v_raw, file-local start pos).
        self.truth: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}

        def warm(padded: list[int], plans) -> list[StoredChunk]:
            """Real warm semantics: one contiguous forward over the render at
            positions [0, N) -- K is RoPE'd at the chunk's file-local span."""
            out = []
            for p in plans:
                g = torch.Generator().manual_seed(hash(p.chunk_hash) & 0xFFFF)
                k_raw = torch.randn(p.length, NUM_KV_HEADS, HEAD_SIZE,
                                    generator=g, dtype=torch.float64)
                v_raw = torch.randn(p.length, NUM_KV_HEADS, HEAD_SIZE,
                                    generator=g, dtype=torch.float64)
                old = torch.arange(p.start, p.start + p.length,
                                   dtype=torch.int64)
                sc = StoredChunk(chunk_hash="", length=p.length,
                                 old_positions=old)
                sc.k_per_layer[LAYER] = _neox_rope_reference(
                    k_raw, old, BASE, ROTARY_DIM)
                sc.v_per_layer[LAYER] = v_raw
                self.truth[p.chunk_hash] = (k_raw, v_raw, p.start)
                out.append(sc)
            return out

        self.builder = OfflineFileKVBuilder(
            store=self.store, tokenize_fn=_tokenize, warm_fn=warm,
            catalog=self.catalog, render_fn=lambda path, text: text,
            chunk_size=CHUNK, pad_token_id=0)

        # --- engine connectors (scheduler + worker roles) -------------------
        self.sched = _make_scheduler(sparse=True, link=LINK,
                                     live_store=self.store)
        import threading
        self.sched._prefetch_lock = threading.Lock()
        self.sched._prefetch_queue = []
        self.worker = _make_worker(sparse=True, link=LINK, store=self.store)
        self.worker._staging = EpicGpuStagingStore(capacity_bytes=10**8)
        self.worker._worker_id = 3  # the frontend targets this replica
        self.worker._debug_counters = True

    def file_tokens(self, name: str) -> list[int]:
        return _tokenize((self.root / name).read_text())

    def file_hashes(self, name: str) -> list[str]:
        rec = self.catalog.lookup(RangeKey(path=name))
        assert rec is not None
        return list(rec.chunk_hashes)


@pytest.fixture
def world(tmp_path):
    return _E2EWorld(tmp_path)


def test_e2e_offline_build_to_online_reuse(world, tmp_path):
    EpicConnector.reset_debug_counters()

    # ================= 1) OFFLINE: scan -> host DRAM =======================
    results = world.builder.build_directory(world.root)
    assert [r.path for r in results] == ["C.py", "F.py"]
    assert all(r.error is None and len(r.chunk_hashes) == FILE_CHUNKS
               for r in results)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(results, manifest_path, root=str(world.root),
                   chunk_size=CHUNK)
    # The KV physically resides in the (pinned-capable) host pool.
    assert world.store._pool.used_bytes > 0
    c_hashes = world.file_hashes("C.py")
    f_hashes = world.file_hashes("F.py")
    assert len(set(c_hashes + f_hashes)) == 2 * FILE_CHUNKS

    # ============ 2) FRONTEND: tool calls -> ZMQ prefetch injection ========
    endpoint = f"ipc:///tmp/epic-e2e-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    listener = EpicPrefetchListener(endpoint,
                                    world.sched.handle_prefetch_command)
    listener.start()
    try:
        client = EpicPrefetchClient(endpoint, timeout_ms=5000)
        bridge = DynamoPrefetchBridge(
            client=client,
            render_fn=lambda read: (world.root / read.file_path).read_text(),
            tokenize_fn=_tokenize,
            chunk_size=CHUNK,
            pad_token_id=0,
        )
        prev_turn_output = (
            "I need to inspect both files.\n"
            "<tool_call><function=read>"
            "<parameter=filePath>C.py</parameter></function></tool_call>\n"
            "<tool_call><function=read>"
            "<parameter=filePath>F.py</parameter></function></tool_call>"
        )
        reply = bridge.on_turn_response(prev_turn_output, dst_worker=3)
    finally:
        listener.stop()
    # The bridge's independently-computed hashes agree with the offline
    # manifest, and every chunk was known to the store index (none dropped).
    assert reply["queued"] == c_hashes + f_hashes
    assert reply["dropped"] == [] and reply["warmed"] == []

    # ================= 3) SCHEDULER: match + plan + drain ==================
    a = list(range(100, 100 + CHUNK))  # genuinely new segments
    d = list(range(200, 200 + CHUNK))
    g = list(range(300, 300 + CHUNK))
    c_tok = world.file_tokens("C.py")
    f_tok = world.file_tokens("F.py")
    tokens = a + c_tok + d + f_tok + g
    n = len(tokens)
    assert n == 7 * CHUNK

    external, is_async = world.sched.get_num_new_matched_tokens(
        _Req("r0", tokens), 0)
    assert is_async is False
    assert external == 2 * FILE_CHUNKS * CHUNK  # |C| + |F| reused
    num_new = n - external
    assert num_new == 3 * CHUNK  # A + D + G

    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(n))],
        num_scheduled_tokens={"r0": num_new},
    )
    meta = world.sched.build_connector_meta(sout)

    # Loads: all four file chunks, at their NON-prefix prompt offsets.
    assert len(meta.loads) == 1
    specs = {s.chunk_hash: s for s in meta.loads[0].chunks}
    assert set(specs) == set(c_hashes + f_hashes)
    c_off, f_off = CHUNK, 4 * CHUNK  # A | C C | D | F F | G
    for i, h in enumerate(c_hashes):
        assert specs[h].new_pos_start == c_off + i * CHUNK
    for i, h in enumerate(f_hashes):
        assert specs[h].new_pos_start == f_off + i * CHUNK

    # Sparse plan M = new segments ∪ per-chunk link heads ∪ {N-1}.
    assert len(meta.sparse) == 1
    m = meta.sparse[0].sparse_positions
    new_positions = (set(range(0, CHUNK))  # A
                     | set(range(3 * CHUNK, 4 * CHUNK))  # D
                     | set(range(6 * CHUNK, 7 * CHUNK)))  # G
    link_heads = set()
    for cidx in (1, 2, 4, 5):  # the four reused file chunks
        link_heads |= set(range(cidx * CHUNK, cidx * CHUNK + LINK))
    assert m == sorted(new_positions | link_heads | {n - 1})
    assert meta.sparse[0].computed_advance == n - external

    # The frontend's ZMQ-injected directive rides THIS step's metadata.
    assert len(meta.prefetches) == 1
    assert meta.prefetches[0].dst_worker == 3
    assert set(meta.prefetches[0].chunk_hashes) == set(c_hashes + f_hashes)

    # ================= 4) BOUNDARY: scheduler -> worker ====================
    meta = _pickle_roundtrip(meta)

    # ================= 5) WORKER: stage from host DRAM, then load ==========
    cache = _make_paged_cache(num_blocks=n // BLOCK)
    world.worker.register_kv_caches({LAYER: cache})
    world.worker._connector_metadata = meta

    class _NoLayerCtx:
        no_compile_layers = None

    world.worker.start_load_kv(_NoLayerCtx())

    # The directive staged all four chunks out of the KVBM host pool, and the
    # loads hit the staged copies (no fallback misses).
    for h in c_hashes + f_hashes:
        assert world.worker._staging.contains(h)
    counters = EpicConnector.debug_counters
    assert counters["prefetch_staged"] == 4
    assert counters["prefetch_hit"] == 4
    assert counters.get("prefetch_miss", 0) == 0

    # ================= 6) FIDELITY: scattered KV == ground truth ===========
    k_bank = cache[0].reshape(n, NUM_KV_HEADS, HEAD_SIZE)
    v_bank = cache[1].reshape(n, NUM_KV_HEADS, HEAD_SIZE)
    for h in c_hashes + f_hashes:
        spec = specs[h]
        k_raw, v_raw, _old_start = world.truth[h]
        dst = _slot_ids_from_blocks(
            [spec.new_pos_start // BLOCK], BLOCK, 0, CHUNK)
        new_pos = torch.arange(spec.new_pos_start,
                               spec.new_pos_start + CHUNK, dtype=torch.int64)
        # PIC re-rotation from the file-local warm positions must equal a
        # direct RoPE at the chunk's NEW prompt positions...
        assert torch.allclose(
            k_bank[dst],
            _neox_rope_reference(k_raw, new_pos, BASE, ROTARY_DIM),
            atol=1e-6, rtol=1e-5)
        # ...and V survives host-DRAM serialization byte-exactly.
        assert torch.equal(v_bank[dst], v_raw)
    # New-segment blocks (A, D, G) were never written by the connector.
    for blk in (0, 3, 6):
        s = _slot_ids_from_blocks([blk], BLOCK, 0, CHUNK)
        assert torch.count_nonzero(k_bank[s]) == 0
        assert torch.count_nonzero(v_bank[s]) == 0

    # Save guard: file chunks are NOT re-saved; the new segments are, with
    # save-time context chains for future fold-soundness.
    saved = {h for s in meta.saves for h in s.chunk_hashes}
    assert saved.isdisjoint(c_hashes + f_hashes)
    assert hash_chunk_tokens(a) in saved
    for s in meta.saves:
        assert len(s.chunk_chains) == len(s.chunk_hashes)
        assert all(cs and ce for cs, ce in s.chunk_chains)


def test_e2e_prefetch_miss_falls_back_to_host_pool(world):
    """Same pipeline without the frontend leg: no staging directive -> the
    worker load path reads straight from the KVBM host pool (miss counters),
    and the scattered KV is STILL exact -- prefetch is a latency hint, never
    a correctness dependency."""
    EpicConnector.reset_debug_counters()
    world.builder.build_directory(world.root)
    c_hashes = world.file_hashes("C.py")

    new = list(range(400, 400 + CHUNK))
    tokens = new + world.file_tokens("C.py")
    n = len(tokens)
    external, _ = world.sched.get_num_new_matched_tokens(_Req("r1", tokens), 0)
    assert external == FILE_CHUNKS * CHUNK
    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r1", tokens, _block_ids_for(n))],
        num_scheduled_tokens={"r1": n - external},
    )
    meta = _pickle_roundtrip(world.sched.build_connector_meta(sout))
    assert meta.prefetches == []

    cache = _make_paged_cache(num_blocks=n // BLOCK)
    world.worker.register_kv_caches({LAYER: cache})
    world.worker._connector_metadata = meta

    class _NoLayerCtx:
        no_compile_layers = None

    world.worker.start_load_kv(_NoLayerCtx())

    counters = EpicConnector.debug_counters
    assert counters.get("prefetch_hit", 0) == 0
    assert counters["prefetch_miss"] == FILE_CHUNKS

    k_bank = cache[0].reshape(n, NUM_KV_HEADS, HEAD_SIZE)
    for i, h in enumerate(c_hashes):
        k_raw, _v_raw, _ = world.truth[h]
        start = CHUNK + i * CHUNK
        dst = _slot_ids_from_blocks([start // BLOCK], BLOCK, 0, CHUNK)
        new_pos = torch.arange(start, start + CHUNK, dtype=torch.int64)
        assert torch.allclose(
            k_bank[dst],
            _neox_rope_reference(k_raw, new_pos, BASE, ROTARY_DIM),
            atol=1e-6, rtol=1e-5)
