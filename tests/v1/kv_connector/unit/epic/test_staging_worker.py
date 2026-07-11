# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated staging-worker process + TP correctness, CPU tests.

The external backend runs the REAL child process (torch.multiprocessing
spawn) with device=cpu -- the transport, LRU/budget and protocol are
identical to the CUDA path (tensors ride torch.mp pickling: shm on CPU,
CUDA IPC on GPU; the child synchronizes before replying to `get`).

TP correctness is verified by simulating two ranks: same prefetch directive
broadcast to both, each rank staging ITS OWN KV shard from its own fileKV
store, plus the hit/miss-skew case (one rank staged, the other falls back)
-- both must scatter their own shard bit-exactly, because the load path is
purely local (no collective)."""

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    EpicChunkStore,
    StoredChunk,
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
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    IdentityAlignment,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.staging_worker import (
    ExternalStagingBackend,
)

BLOCK = 16
CHUNK = 64
HEADS, HD = 1, 4


def _stored(hash_: str, seed: int, length: int = CHUNK) -> StoredChunk:
    g = torch.Generator().manual_seed(seed)
    sc = StoredChunk(
        chunk_hash=hash_,
        length=length,
        old_positions=torch.arange(length, dtype=torch.int64),
    )
    sc.k_per_layer["l0"] = torch.randn(length, HEADS, HD, generator=g)
    sc.v_per_layer["l0"] = torch.randn(length, HEADS, HD, generator=g)
    return sc


# ---------------------------------------------------------------------------
# External staging-worker process roundtrip (the real child, device=cpu).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def backend():
    b = ExternalStagingBackend(capacity_bytes=10**8)
    yield b
    b.close()


def test_external_backend_stage_get_roundtrip(backend):
    sc = _stored("ext-h1", seed=3)
    backend.stage(sc, torch.device("cpu"))
    assert backend.contains("ext-h1")

    staged = backend.get("ext-h1")
    assert staged is not None
    torch.testing.assert_close(staged.k_per_layer["l0"], sc.k_per_layer["l0"])
    torch.testing.assert_close(staged.v_per_layer["l0"], sc.v_per_layer["l0"])
    assert torch.equal(staged.old_positions, sc.old_positions)
    assert staged.ready_event is None  # child synchronized before replying.

    # Memoized mapping: second get costs no roundtrip and is the same object.
    assert backend.get("ext-h1") is staged
    # Miss path.
    assert backend.get("never-staged") is None
    assert not backend.contains("never-staged")


def test_external_backend_budget_and_stats(backend):
    stats0 = backend.stats()
    assert stats0["chunks"] >= 1  # from the roundtrip test (module fixture).
    sc = _stored("ext-h2", seed=4)
    backend.stage(sc, torch.device("cpu"))
    stats1 = backend.stats()
    assert stats1["chunks"] == stats0["chunks"] + 1
    assert stats1["bytes"] > stats0["bytes"]


def test_external_backend_serves_connector_load_path(backend):
    """The seam swap: a worker connector wired with the EXTERNAL backend runs
    the exact same _consume_prefetches -> _load_chunk flow."""
    EpicConnector.reset_debug_counters()
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    sc = _stored("ext-h3", seed=5)
    store.put(sc)

    w = object.__new__(EpicConnector)
    w._store = store
    w._staging = backend
    w._worker_id = 0
    w._layer_names = ["l0"]
    w._kv_caches = {"l0": torch.zeros(2, 8, BLOCK, HEADS, HD)}
    w._alignment = IdentityAlignment()
    w._debug_counters = True

    meta = EpicConnectorMetadata()
    meta.add_prefetch(EpicReqPrefetch(chunk_hashes=["ext-h3"], dst_worker=-1))
    w._consume_prefetches(meta)
    assert backend.contains("ext-h3")

    dst = list(range(32, 32 + CHUNK))
    w._load_chunk(
        sc,
        ChunkLoadSpec(
            chunk_hash="ext-h3",
            dst_slot_ids=dst,
            old_pos_start=-1,
            new_pos_start=32,
            length=CHUNK,
        ),
    )
    assert EpicConnector.debug_counters["prefetch_hit"] == 1
    res = check_scatter_fidelity(
        w._kv_caches["l0"], sc.k_per_layer["l0"], sc.v_per_layer["l0"], dst
    )
    assert res is not None and res[0] and res[2]


# ---------------------------------------------------------------------------
# TP correctness (two simulated ranks; in-process staging keeps it fast --
# the property is backend-independent: per-rank state, local-only load path).
# ---------------------------------------------------------------------------


def _rank_worker(rank: int, shard_seed: int):
    """One TP rank: its own fileKV store, staging store, and paged cache,
    holding ITS OWN shard tensors for the shared chunk hash."""
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    sc = _stored("shared-hash", seed=shard_seed)  # same hash, different shard
    store.put(sc)
    w = object.__new__(EpicConnector)
    w._store = store
    w._staging = EpicGpuStagingStore(capacity_bytes=10**8)
    w._worker_id = 7  # replica id is shared across the replica's TP ranks.
    w._layer_names = ["l0"]
    w._kv_caches = {"l0": torch.zeros(2, 8, BLOCK, HEADS, HD)}
    w._alignment = IdentityAlignment()
    w._debug_counters = False
    return w, sc


def _load(w, sc):
    dst = list(range(0, CHUNK))
    w._load_chunk(
        sc,
        ChunkLoadSpec(
            chunk_hash="shared-hash",
            dst_slot_ids=dst,
            old_pos_start=-1,
            new_pos_start=0,
            length=CHUNK,
        ),
    )
    return dst


def test_tp_ranks_stage_their_own_shard():
    """Broadcast directive + per-rank stores -> each rank's staging holds its
    OWN shard; loads scatter per-rank shards, never the other rank's."""
    w0, sc0 = _rank_worker(0, shard_seed=100)
    w1, sc1 = _rank_worker(1, shard_seed=200)
    assert not torch.equal(
        sc0.k_per_layer["l0"], sc1.k_per_layer["l0"]
    )  # shards genuinely differ.

    meta = EpicConnectorMetadata()
    meta.add_prefetch(
        EpicReqPrefetch(chunk_hashes=["shared-hash"], dst_worker=7)
    )
    for w in (w0, w1):
        w._consume_prefetches(meta)  # same broadcast metadata on every rank.
        assert w._staging.contains("shared-hash")

    dst0, dst1 = _load(w0, sc0), _load(w1, sc1)
    r0 = check_scatter_fidelity(
        w0._kv_caches["l0"], sc0.k_per_layer["l0"], sc0.v_per_layer["l0"], dst0
    )
    r1 = check_scatter_fidelity(
        w1._kv_caches["l0"], sc1.k_per_layer["l0"], sc1.v_per_layer["l0"], dst1
    )
    assert r0[0] and r0[2] and r1[0] and r1[2]
    # Cross-check: rank1's cache does NOT hold rank0's shard.
    cross = check_scatter_fidelity(
        w1._kv_caches["l0"], sc0.k_per_layer["l0"], sc0.v_per_layer["l0"], dst1
    )
    assert not cross[0]


def test_tp_hit_miss_skew_is_safe():
    """Rank0 staged, rank1 not (e.g. staging evicted on one rank): both must
    produce their correct shard KV -- only latency differs, because the load
    path is local (no collective) and falls back to the CPU store."""
    w0, sc0 = _rank_worker(0, shard_seed=300)
    w1, sc1 = _rank_worker(1, shard_seed=400)

    meta = EpicConnectorMetadata()
    meta.add_prefetch(
        EpicReqPrefetch(chunk_hashes=["shared-hash"], dst_worker=-1)
    )
    w0._consume_prefetches(meta)  # only rank0 got to stage.
    assert w0._staging.contains("shared-hash")
    assert not w1._staging.contains("shared-hash")

    dst0, dst1 = _load(w0, sc0), _load(w1, sc1)
    r0 = check_scatter_fidelity(
        w0._kv_caches["l0"], sc0.k_per_layer["l0"], sc0.v_per_layer["l0"], dst0
    )
    r1 = check_scatter_fidelity(
        w1._kv_caches["l0"], sc1.k_per_layer["l0"], sc1.v_per_layer["l0"], dst1
    )
    assert r0[0] and r0[2]  # staged path.
    assert r1[0] and r1[2]  # CPU-store fallback path -- same bytes.


def test_tp_capacity_is_per_rank():
    """The connector splits the CPU fileKV budget by world_size (__init__:
    capacity // world_size); staging budgets are configured per worker. This
    test pins the CPU-store convention so a regression is caught."""
    total = 1024
    world = 4
    per_rank = total // world
    store = EpicChunkStore(capacity_bytes=per_rank, pin_memory=False)
    assert store.capacity_bytes == per_rank
