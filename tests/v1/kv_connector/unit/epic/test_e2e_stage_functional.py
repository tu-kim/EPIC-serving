# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-stage functional verification of the offline->online pipeline.

test_e2e_offline_to_online.py proves the six stages compose into one working
story; this file verifies EACH stage in isolation so a failure names its
stage instead of surfacing as a downstream symptom. Stage numbering matches
the E2E docstring:

  1. OFFLINE   -- scan -> warm -> host-DRAM residency -> manifest/catalog
  2. FRONTEND  -- tool calls -> render/hash -> ZMQ command -> scheduler queue
  3. SCHEDULER -- match accounting, load specs, sparse plan, directive drain
  4. BOUNDARY  -- metadata pickle fidelity (field-by-field)
  5. WORKER    -- directive consume (targeting/evict), staged-hit vs fallback
  6. FIDELITY  -- PIC scatter numerics, V exactness, non-touch, save guard

Each stage builds ONLY the state it needs (via the shared _E2EWorld wiring),
and every stage includes at least one negative/boundary case.
"""

import os
import pickle
import uuid

import pytest
import torch

from tests.v1.kv_connector.unit.epic.test_e2e_offline_to_online import (
    FILE_CHUNKS,
    LINK,
    _E2EWorld,
    _file_text,
    _tokenize,
)
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
    _neox_rope_reference,
    _NewReq,
    _Req,
    _SchedOut,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    chain_hash_tokens,
    hash_chunk_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
    EpicConnector,
    _slot_ids_from_blocks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.filekv_catalog import (
    RangeKey,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.kvbm_store import (
    _deserialize,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
    EpicConnectorMetadata,
    EpicReqPrefetch,
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


@pytest.fixture
def world(tmp_path):
    return _E2EWorld(tmp_path)


@pytest.fixture
def built(world):
    """World with the offline build already done (stages 2+ start here)."""
    results = world.builder.build_directory(world.root)
    assert all(r.error is None for r in results)
    return world


def _listener(world):
    endpoint = f"ipc:///tmp/epic-stage-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    lst = EpicPrefetchListener(endpoint, world.sched.handle_prefetch_command)
    lst.start()
    return endpoint, lst


def _bridge(world, endpoint, **kw):
    return DynamoPrefetchBridge(
        client=EpicPrefetchClient(endpoint, timeout_ms=5000),
        render_fn=lambda read: (world.root / read.file_path).read_text(),
        tokenize_fn=_tokenize,
        chunk_size=CHUNK,
        pad_token_id=0,
        **kw,
    )


def _tool_call(name: str) -> str:
    return (f"<tool_call><function=read>"
            f"<parameter=filePath>{name}</parameter></function></tool_call>")


def _prompt(world, *, prefix_new=True):
    """A(new) + C(file) + D(new) + F(file) + G(new); returns (tokens, n)."""
    a = list(range(100, 100 + CHUNK)) if prefix_new else []
    d = list(range(200, 200 + CHUNK))
    g = list(range(300, 300 + CHUNK))
    tokens = (a + world.file_tokens("C.py") + d + world.file_tokens("F.py")
              + g)
    return tokens, len(tokens)


def _meta_for(world, tokens, req_id="r0"):
    n = len(tokens)
    external, _ = world.sched.get_num_new_matched_tokens(
        _Req(req_id, tokens), 0)
    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq(req_id, tokens, _block_ids_for(n))],
        num_scheduled_tokens={req_id: n - external},
    )
    return world.sched.build_connector_meta(sout), external


class _NoLayerCtx:
    no_compile_layers = None


# ===========================================================================
# Stage 1 -- OFFLINE: scan -> warm -> host DRAM -> manifest/catalog
# ===========================================================================


class TestStage1Offline:
    def test_warm_kv_is_resident_in_host_pool_bytes(self, world):
        assert world.store._pool.used_bytes == 0
        world.builder.build_directory(world.root)
        # 4 chunks resident; pool bytes == store accounting exactly.
        assert len(world.store) == 2 * FILE_CHUNKS
        assert world.store._pool.used_bytes == world.store.current_bytes > 0

    def test_stored_chunk_round_trips_from_pool_blob(self, built):
        """What sits in host DRAM is the REAL serialized chunk: deserialize
        the raw pool payload directly and compare against ground truth."""
        h = built.file_hashes("C.py")[0]
        blob = built.store._pool.get(h)
        assert blob is not None
        chunk = _deserialize(blob, pin=False)
        k_raw, v_raw, start = built.truth[h]
        old = torch.arange(start, start + CHUNK, dtype=torch.int64)
        assert torch.equal(chunk.old_positions, old)
        assert torch.equal(chunk.v_per_layer[LAYER], v_raw)
        assert torch.allclose(
            chunk.k_per_layer[LAYER],
            _neox_rope_reference(k_raw, old, BASE, ROTARY_DIM), atol=1e-12)

    def test_chains_prove_whole_file_warm(self, built):
        """chain_start of chunk i must equal the digest of the render's first
        i*CHUNK tokens -- the run-coherence proof per-run link relies on."""
        toks = built.file_tokens("C.py")
        for i, h in enumerate(built.file_hashes("C.py")):
            cs, ce = built.store.get_chain(h)
            assert cs == chain_hash_tokens(toks[:i * CHUNK])
            assert ce == chain_hash_tokens(toks[:(i + 1) * CHUNK])

    def test_catalog_and_hash_oracle_agree(self, built):
        toks = built.file_tokens("F.py")
        expect = [hash_chunk_tokens(toks[i:i + CHUNK])
                  for i in range(0, len(toks), CHUNK)]
        assert built.file_hashes("F.py") == expect
        rec = built.catalog.lookup(RangeKey(path="F.py"))
        assert rec is not None and rec.version == 0

    def test_negative_modified_file_invalidates_fingerprint(self, built):
        (built.root / "C.py").write_text(_file_text(9999))
        rec = built.catalog.lookup(RangeKey(path="C.py"))
        import hashlib
        new_fp = hashlib.sha256(
            (built.root / "C.py").read_bytes()).hexdigest()
        assert rec.fingerprint != new_fp  # stale record detectable
        # Re-build picks up the new content with NEW hashes.
        res = {r.path: r for r in built.builder.build_directory(built.root)}
        assert not res["C.py"].skipped
        assert res["C.py"].chunk_hashes != rec.chunk_hashes


# ===========================================================================
# Stage 2 -- FRONTEND: tool calls -> ZMQ -> scheduler queue
# ===========================================================================


class TestStage2Frontend:
    def test_command_lands_in_scheduler_queue_with_target(self, built):
        endpoint, lst = _listener(built)
        try:
            reply = _bridge(built, endpoint).on_turn_response(
                _tool_call("C.py"), dst_worker=7)
        finally:
            lst.stop()
        assert reply["queued"] == built.file_hashes("C.py")
        assert len(built.sched._prefetch_queue) == 1
        directive = built.sched._prefetch_queue[0]
        assert directive.dst_worker == 7
        assert directive.chunk_hashes == built.file_hashes("C.py")

    def test_unknown_file_drops_and_triggers_warm(self, built):
        (built.root / "new.py").write_text(_file_text(7000))  # never warmed
        warmed = []
        endpoint, lst = _listener(built)
        try:
            b = _bridge(built, endpoint,
                        warm_fn=lambda read, text: warmed.append(
                            read.file_path))
            reply = b.on_turn_response(_tool_call("new.py"), dst_worker=0)
        finally:
            lst.stop()
        # Store never held these chunks -> every hash dropped, warm_fn fired,
        # and nothing was queued.
        assert reply["queued"] == [] and len(reply["dropped"]) == FILE_CHUNKS
        assert warmed == ["new.py"] and built.sched._prefetch_queue == []

    def test_no_tool_calls_is_a_noop(self, built):
        endpoint, lst = _listener(built)
        try:
            reply = _bridge(built, endpoint).on_turn_response(
                "plain prose, no tool calls", dst_worker=0)
        finally:
            lst.stop()
        assert reply == {"queued": [], "dropped": [], "warmed": []}
        assert built.sched._prefetch_queue == []

    def test_frontend_hashes_equal_offline_manifest_hashes(self, built):
        """The two independent hash computations (offline plan vs frontend
        render) MUST agree or prefetch can never hit."""
        from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch_parser import (  # noqa: E501
            FileKVPrefetcher,
        )
        pf = FileKVPrefetcher(
            render_fn=lambda r: (built.root / r.file_path).read_text(),
            tokenize_fn=_tokenize, transport_fn=lambda *a, **k: None,
            chunk_size=CHUNK, pad_token_id=0)
        assert pf.chunk_hashes_for_tokens(
            built.file_tokens("C.py")) == built.file_hashes("C.py")


# ===========================================================================
# Stage 3 -- SCHEDULER: match, load specs, sparse plan, drain
# ===========================================================================


class TestStage3Scheduler:
    def test_match_counts_exactly_the_file_tokens(self, built):
        tokens, n = _prompt(built)
        external, is_async = built.sched.get_num_new_matched_tokens(
            _Req("r0", tokens), 0)
        assert (external, is_async) == (2 * FILE_CHUNKS * CHUNK, False)

    def test_load_specs_carry_correct_offsets_and_positions(self, built):
        tokens, n = _prompt(built)
        meta, _ = _meta_for(built, tokens)
        specs = {s.chunk_hash: s for s in meta.loads[0].chunks}
        for i, h in enumerate(built.file_hashes("C.py")):
            s = specs[h]
            assert s.new_pos_start == CHUNK + i * CHUNK
            assert s.length == CHUNK and s.src_offset == 0
            assert len(s.dst_slot_ids) == CHUNK
            # The spec's old_pos_start is DIAGNOSTIC (the scheduler index may
            # not know it -> -1); PIC correctness comes from the WORKER
            # store's stored positions, which must be the file-local span.
            assert s.old_pos_start in (-1, i * CHUNK)
            assert built.store.get_old_pos_start(h) == i * CHUNK

    def test_sparse_plan_m_and_advance(self, built):
        tokens, n = _prompt(built)
        meta, external = _meta_for(built, tokens)
        sp = meta.sparse[0]
        m = sp.sparse_positions
        assert m == sorted(set(m)) and m[-1] == n - 1
        assert sp.full_seq_len == n
        assert sp.computed_advance == n - external
        # Link heads only for reused chunks; file BODIES excluded from M.
        for cidx in (1, 2, 4, 5):
            body = range(cidx * CHUNK + LINK, (cidx + 1) * CHUNK)
            assert all(p not in m for p in body if p != n - 1)

    def test_drain_moves_queue_into_metadata_exactly_once(self, built):
        built.sched.enqueue_prefetch(
            chunk_hashes=built.file_hashes("C.py"), dst_worker=3)
        tokens, n = _prompt(built)
        meta1, _ = _meta_for(built, tokens, req_id="r0")
        assert len(meta1.prefetches) == 1
        meta2, _ = _meta_for(built, tokens, req_id="r1")
        assert meta2.prefetches == []  # drained, not duplicated

    def test_negative_no_hits_no_sparse_no_loads(self, built):
        tokens = list(range(900, 900 + 2 * CHUNK))  # nothing cached
        external, _ = built.sched.get_num_new_matched_tokens(
            _Req("rx", tokens), 0)
        assert external == 0
        sout = _SchedOut(
            scheduled_new_reqs=[_NewReq("rx", tokens,
                                        _block_ids_for(len(tokens)))],
            num_scheduled_tokens={"rx": len(tokens)},
        )
        meta = built.sched.build_connector_meta(sout)
        assert meta.loads == [] and meta.sparse == []
        assert len(meta.saves) == 1  # new content still harvested


# ===========================================================================
# Stage 4 -- BOUNDARY: pickle fidelity, field by field
# ===========================================================================


class TestStage4Boundary:
    def test_full_metadata_survives_pickle_exactly(self, built):
        built.sched.enqueue_prefetch(
            chunk_hashes=built.file_hashes("C.py"), dst_worker=3,
            evict_hashes=["stale"])
        tokens, _ = _prompt(built)
        meta, _ = _meta_for(built, tokens)
        rt = pickle.loads(pickle.dumps(meta))

        assert [s.chunk_hash for ld in rt.loads for s in ld.chunks] == \
               [s.chunk_hash for ld in meta.loads for s in ld.chunks]
        for a, b in zip(rt.loads[0].chunks, meta.loads[0].chunks):
            assert (a.dst_slot_ids, a.old_pos_start, a.new_pos_start,
                    a.length, a.src_offset) == \
                   (b.dst_slot_ids, b.old_pos_start, b.new_pos_start,
                    b.length, b.src_offset)
        assert rt.sparse[0].sparse_positions == meta.sparse[0].sparse_positions
        assert rt.sparse[0].computed_advance == meta.sparse[0].computed_advance
        assert rt.prefetches[0].chunk_hashes == meta.prefetches[0].chunk_hashes
        assert rt.prefetches[0].dst_worker == 3
        assert rt.prefetches[0].evict_hashes == ["stale"]
        assert [s.chunk_chains for s in rt.saves] == \
               [s.chunk_chains for s in meta.saves]

    def test_metadata_is_self_contained_no_live_refs(self, built):
        """Nothing in the pickled payload may reference live engine objects
        (store/pool/tensors) -- that is what makes the process split safe."""
        tokens, _ = _prompt(built)
        meta, _ = _meta_for(built, tokens)
        payload = pickle.dumps(meta)
        # A payload carrying the store or pool would be megabytes.
        assert len(payload) < 64 * 1024
        rt = pickle.loads(payload)
        assert isinstance(rt, EpicConnectorMetadata)


# ===========================================================================
# Stage 5 -- WORKER: directive consume + staged-hit vs fallback
# ===========================================================================


class TestStage5Worker:
    def _armed_worker(self, built, n):
        cache = _make_paged_cache(num_blocks=n // BLOCK)
        built.worker.register_kv_caches({LAYER: cache})
        return cache

    def test_directive_targeting_and_broadcast(self, built):
        h = built.file_hashes("C.py")[0]
        self._armed_worker(built, 2 * CHUNK)
        # Addressed elsewhere -> our worker (id 3) must not stage.
        meta = EpicConnectorMetadata()
        meta.add_prefetch(EpicReqPrefetch(chunk_hashes=[h], dst_worker=1))
        built.worker._consume_prefetches(meta)
        assert not built.worker._staging.contains(h)
        # Addressed to us -> staged.
        meta2 = EpicConnectorMetadata()
        meta2.add_prefetch(EpicReqPrefetch(chunk_hashes=[h], dst_worker=3))
        built.worker._consume_prefetches(meta2)
        assert built.worker._staging.contains(h)

    def test_evict_directive_reclaims_staging(self, built):
        h = built.file_hashes("C.py")[0]
        self._armed_worker(built, 2 * CHUNK)
        meta = EpicConnectorMetadata()
        meta.add_prefetch(EpicReqPrefetch(chunk_hashes=[h], dst_worker=-1))
        built.worker._consume_prefetches(meta)
        assert built.worker._staging.contains(h)
        meta2 = EpicConnectorMetadata()
        meta2.add_prefetch(EpicReqPrefetch(evict_hashes=[h], dst_worker=-1))
        built.worker._consume_prefetches(meta2)
        assert not built.worker._staging.contains(h)

    def test_staged_hit_skips_store_read(self, built):
        """After staging, the load path must NOT touch the host pool: sever
        the pool payloads and the staged load still succeeds."""
        EpicConnector.reset_debug_counters()
        tokens = list(range(400, 400 + CHUNK)) + built.file_tokens("C.py")
        n = len(tokens)
        built.sched.enqueue_prefetch(
            chunk_hashes=built.file_hashes("C.py"), dst_worker=3)
        meta, _ = _meta_for(built, tokens)
        cache = self._armed_worker(built, n)
        built.worker._connector_metadata = pickle.loads(pickle.dumps(meta))
        # Stage first (directives run before loads inside start_load_kv)...
        # then sabotage the host pool to prove the loads never read it.
        built.worker._consume_prefetches(built.worker._connector_metadata)
        built.worker._connector_metadata.prefetches = []
        for h in built.file_hashes("C.py"):
            built.store._pool.evict(h)
        built.worker.start_load_kv(_NoLayerCtx())
        c = EpicConnector.debug_counters
        assert c["prefetch_hit"] == FILE_CHUNKS
        assert c.get("prefetch_miss", 0) == 0
        assert torch.count_nonzero(cache) > 0  # KV actually landed

    def test_fallback_reads_host_pool_when_not_staged(self, built):
        EpicConnector.reset_debug_counters()
        tokens = list(range(400, 400 + CHUNK)) + built.file_tokens("C.py")
        meta, _ = _meta_for(built, tokens)
        self._armed_worker(built, len(tokens))
        built.worker._connector_metadata = pickle.loads(pickle.dumps(meta))
        built.worker.start_load_kv(_NoLayerCtx())
        c = EpicConnector.debug_counters
        assert c.get("prefetch_hit", 0) == 0
        assert c["prefetch_miss"] == FILE_CHUNKS


# ===========================================================================
# Stage 6 -- FIDELITY: numerics, non-touch, save guard
# ===========================================================================


class TestStage6Fidelity:
    def _load(self, built, tokens):
        meta, external = _meta_for(built, tokens)
        n = len(tokens)
        cache = _make_paged_cache(num_blocks=n // BLOCK)
        built.worker.register_kv_caches({LAYER: cache})
        built.worker._connector_metadata = pickle.loads(pickle.dumps(meta))
        built.worker.start_load_kv(_NoLayerCtx())
        return meta, cache, n

    def test_pic_scatter_equals_direct_rope_at_new_positions(self, built):
        tokens, _ = _prompt(built)
        meta, cache, n = self._load(built, tokens)
        k_bank = cache[0].reshape(n, NUM_KV_HEADS, HEAD_SIZE)
        v_bank = cache[1].reshape(n, NUM_KV_HEADS, HEAD_SIZE)
        for s in meta.loads[0].chunks:
            k_raw, v_raw, _ = built.truth[s.chunk_hash]
            dst = _slot_ids_from_blocks(
                [s.new_pos_start // BLOCK], BLOCK, 0, CHUNK)
            pos = torch.arange(s.new_pos_start, s.new_pos_start + CHUNK,
                               dtype=torch.int64)
            assert torch.allclose(
                k_bank[dst], _neox_rope_reference(k_raw, pos, BASE,
                                                  ROTARY_DIM),
                atol=1e-6, rtol=1e-5)
            assert torch.equal(v_bank[dst], v_raw)

    def test_same_chunk_two_prompts_two_correct_rotations(self, built):
        """Position independence: C loaded at offset CHUNK in one prompt and
        offset 2*CHUNK in another must each match their OWN position."""
        h0 = built.file_hashes("C.py")[0]
        k_raw, _, _ = built.truth[h0]
        for lead_chunks in (1, 2):
            lead = list(range(400, 400 + lead_chunks * CHUNK))
            tokens = lead + built.file_tokens("C.py")
            _, cache, n = self._load(built, tokens)
            k_bank = cache[0].reshape(n, NUM_KV_HEADS, HEAD_SIZE)
            start = lead_chunks * CHUNK
            dst = _slot_ids_from_blocks([start // BLOCK], BLOCK, 0, CHUNK)
            pos = torch.arange(start, start + CHUNK, dtype=torch.int64)
            assert torch.allclose(
                k_bank[dst],
                _neox_rope_reference(k_raw, pos, BASE, ROTARY_DIM),
                atol=1e-6, rtol=1e-5)

    def test_new_segment_blocks_untouched(self, built):
        tokens, _ = _prompt(built)
        _, cache, n = self._load(built, tokens)
        k_bank = cache[0].reshape(n, NUM_KV_HEADS, HEAD_SIZE)
        v_bank = cache[1].reshape(n, NUM_KV_HEADS, HEAD_SIZE)
        for blk in (0, 3, 6):  # A, D, G blocks: forward's job, not ours
            s = _slot_ids_from_blocks([blk], BLOCK, 0, CHUNK)
            assert torch.count_nonzero(k_bank[s]) == 0
            assert torch.count_nonzero(v_bank[s]) == 0

    def test_save_guard_and_chains(self, built):
        tokens, _ = _prompt(built)
        meta, _ = _meta_for(built, tokens)
        saved = {h for s in meta.saves for h in s.chunk_hashes}
        reused = set(built.file_hashes("C.py") + built.file_hashes("F.py"))
        assert saved.isdisjoint(reused)
        a_hash = hash_chunk_tokens(list(range(100, 100 + CHUNK)))
        assert a_hash in saved
        for s in meta.saves:
            assert len(s.chunk_chains) == len(s.chunk_hashes)
            assert all(cs and ce for cs, ce in s.chunk_chains)
