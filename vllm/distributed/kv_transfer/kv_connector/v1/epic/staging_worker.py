# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated per-GPU staging worker process (feature/prefetch, advanced).

Motivation (user requirement): staging fileKV directly inside the vLLM worker
process can contend with in-flight generation -- allocator pressure on the
same CUDA context, host-thread time on the worker loop, and copy traffic
scheduled from the same process. Isolating staging into a DEDICATED worker
per GPU removes that coupling: the vLLM worker only *checks* whether a chunk
is staged and maps it; all H2D work happens in the staging process.

Why NOT NVIDIA MIG (reviewed, rejected)
---------------------------------------
MIG partitions a GPU into fully isolated instances -- isolation is exactly
what breaks the design: **MIG instances cannot share memory** (no P2P, no
CUDA IPC across instances). A staging buffer allocated on one MIG slice is
unreachable from a vLLM worker on another slice; the only path would be
staging-GPU -> host -> vLLM-GPU, which is strictly worse than the CPU-pinned
-> GPU copy it was meant to replace. MIG also shrinks the vLLM slice's
SMs/HBM permanently, whether or not staging is busy.

Chosen architecture: **same-GPU separate process + CUDA IPC** (optionally
under MPS with ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`` to cap the staging
process's SM share -- copies are DMA-engine work and barely use SMs anyway):

  * The staging process owns its own CUDA context and memory pool, capped by
    the staging budget -- it cannot fragment or OOM the vLLM worker's
    allocator, and its Python/host work never runs on the vLLM worker loop.
  * Staged tensors are exported to the vLLM worker via torch's CUDA IPC
    (``torch.multiprocessing`` pickling of CUDA tensors sends memory HANDLES,
    not bytes). The vLLM worker maps them zero-copy and reads them directly
    in its PIC-rotate + scatter -- a same-device read, no extra copy.
  * torch's CUDA-IPC ref-counting keeps the underlying allocation alive until
    every consumer releases its mapping, so an LRU eviction in the staging
    process cannot pull memory out from under an in-flight load.
  * On CPU (unit tests / no CUDA) the same pipe transport moves tensors into
    POSIX shared memory instead -- the protocol is identical, which is what
    makes the whole path CPU-testable.

Tensor-parallel (TP) correctness
--------------------------------
Each TP rank holds a DIFFERENT KV shard (num_kv_heads / tp per rank), so
staging is strictly per-rank state:

  * one staging worker per vLLM worker rank, pinned to that rank's device;
    each rank ships chunks from ITS OWN CPU fileKV store (already per-rank --
    the connector divides the CPU budget by world_size), so shards can never
    mix by construction.
  * prefetch directives arrive identically on every rank (connector metadata
    is broadcast), and each rank stages its own shard -- the same chunk hash
    maps to different (correct) tensors per rank.
  * divergent hit/miss across ranks is SAFE: the load path is purely local
    (gather -> PIC rotate -> scatter; no collective), so rank i serving a
    load from staging while rank j falls back to its CPU store produces
    identical logical KV, only with per-rank latency skew. Nothing in the
    sparse forward reads "was this staged" -- shapes, positions and masks
    come from scheduler metadata that is identical on all ranks.

Verified on CPU by tests/v1/kv_connector/unit/epic/test_staging_worker.py
(process roundtrip, LRU/budget, per-rank shard isolation, hit/miss skew).
"""

from __future__ import annotations

import threading
from typing import Any

import torch
import torch.multiprocessing as tmp

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    StoredChunk,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch import (
    StagedChunk,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Child process body.
# ---------------------------------------------------------------------------


def _staging_child_main(conn, device_str: str, capacity_bytes: int) -> None:
    """Staging worker loop (runs in its own process, its own CUDA context).

    Commands over the duplex pipe (tuples, torch tensors ride torch.mp
    pickling: CUDA -> IPC handles, CPU -> shared memory):

      ("stage", hash, length, old_positions, {layer: k}, {layer: v})
          -> ("ok", staged: bool)     # synchronous H2D inside THIS process
      ("get", hash)      -> ("hit", length, old_positions, k_dict, v_dict)
                          | ("miss",)
      ("contains", hash) -> ("ok", bool)
      ("stats",)         -> ("ok", {"chunks": n, "bytes": b})
      ("stop",)          -> ("ok",) and exit
    """
    device = torch.device(device_str)
    # LRU store, byte-budgeted -- mirrors EpicGpuStagingStore but lives here.
    from collections import OrderedDict

    store: "OrderedDict[str, StagedChunk]" = OrderedDict()
    cur_bytes = 0

    def evict_if_needed() -> None:
        nonlocal cur_bytes
        while cur_bytes > capacity_bytes and store:
            _, evicted = store.popitem(last=False)
            cur_bytes -= evicted.nbytes()
            # torch CUDA-IPC refcounting keeps the allocation alive for any
            # consumer that already mapped it; we only drop OUR reference.

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        cmd = msg[0]
        try:
            if cmd == "stage":
                _, h, length, old_positions, k_dict, v_dict = msg
                if h in store:
                    store.move_to_end(h)
                    conn.send(("ok", True))
                    continue
                staged = StagedChunk(
                    chunk_hash=h,
                    length=int(length),
                    old_positions=old_positions.to(device),
                )
                for name, k in k_dict.items():
                    staged.k_per_layer[name] = k.to(device)
                for name, v in v_dict.items():
                    staged.v_per_layer[name] = v.to(device)
                if device.type == "cuda":
                    # Block THIS process (never the vLLM worker) until the
                    # copies land, so a later "get" hands out ready tensors.
                    torch.cuda.synchronize(device)
                nbytes = staged.nbytes()
                ok = nbytes <= capacity_bytes
                if ok:
                    store[h] = staged
                    cur_bytes += nbytes
                    evict_if_needed()
                conn.send(("ok", ok))
            elif cmd == "get":
                _, h = msg
                staged = store.get(h)
                if staged is None:
                    conn.send(("miss",))
                else:
                    store.move_to_end(h)
                    conn.send(
                        (
                            "hit",
                            staged.length,
                            staged.old_positions,
                            staged.k_per_layer,
                            staged.v_per_layer,
                        )
                    )
            elif cmd == "contains":
                conn.send(("ok", msg[1] in store))
            elif cmd == "stats":
                conn.send(("ok", {"chunks": len(store), "bytes": cur_bytes}))
            elif cmd == "stop":
                conn.send(("ok",))
                break
            else:
                conn.send(("err", f"unknown cmd {cmd!r}"))
        except Exception as e:  # noqa: BLE001 -- keep the daemon alive.
            try:
                conn.send(("err", repr(e)))
            except (BrokenPipeError, OSError):
                break
    conn.close()


# ---------------------------------------------------------------------------
# Parent-side backend (drop-in for EpicGpuStagingStore in the connector).
# ---------------------------------------------------------------------------


class ExternalStagingBackend:
    """vLLM-worker-side handle to a dedicated staging worker process.

    Implements the same informal interface the connector uses on the
    in-process ``EpicGpuStagingStore`` -- ``contains`` / ``get`` / ``stage``
    -- so ``epic_staging_mode: "external"`` is a pure seam swap. All GPU
    work (H2D, allocation) happens in the child; this side only ships CPU
    shm handles out and maps device tensors back (CUDA IPC / CPU shm).

    ``get`` results are memoized per hash so repeated loads of a staged chunk
    cost one pipe roundtrip total. The memo is a BOUNDED LRU (same byte budget
    as the staging worker): a parent-held mapping keeps the underlying shared
    allocation alive even after a child-side eviction (that is the safety
    property), so an unbounded memo would defeat the child's budget entirely
    -- evicting the memo entry releases the mapping and lets the memory
    actually return once the child has evicted too.
    """

    def __init__(self, capacity_bytes: int, spawn_method: str = "spawn"):
        self.capacity_bytes = int(capacity_bytes)
        self._ctx = tmp.get_context(spawn_method)
        self._proc: Any = None
        self._conn: Any = None
        self._lock = threading.Lock()
        from collections import OrderedDict

        self._mapped: "OrderedDict[str, StagedChunk]" = OrderedDict()
        self._mapped_bytes = 0

    # -- lifecycle --

    def _ensure_started(self, device: torch.device) -> None:
        if self._proc is not None:
            return
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        self._proc = self._ctx.Process(
            target=_staging_child_main,
            args=(child_conn, str(device), self.capacity_bytes),
            daemon=True,
            name="epic-staging-worker",
        )
        self._proc.start()
        child_conn.close()
        self._conn = parent_conn
        logger.info(
            "EPIC staging worker started (pid=%s device=%s budget=%d bytes)",
            self._proc.pid,
            device,
            self.capacity_bytes,
        )

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.send(("stop",))
                    self._conn.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self._conn.close()
                self._conn = None
            if self._proc is not None:
                self._proc.join(timeout=5)
                if self._proc.is_alive():
                    self._proc.terminate()
                self._proc = None
        self._mapped.clear()
        self._mapped_bytes = 0

    def _request(self, msg: tuple) -> tuple:
        assert self._conn is not None, "staging worker not started"
        with self._lock:
            self._conn.send(msg)
            return self._conn.recv()

    # -- the connector-facing interface --

    def _memo_put(self, chunk_hash: str, staged: StagedChunk) -> None:
        self._mapped[chunk_hash] = staged
        self._mapped_bytes += staged.nbytes()
        while self._mapped_bytes > self.capacity_bytes and len(self._mapped) > 1:
            _, evicted = self._mapped.popitem(last=False)  # LRU mapping.
            self._mapped_bytes -= evicted.nbytes()

    def contains(self, chunk_hash: str) -> bool:
        if chunk_hash in self._mapped:
            return True
        if self._conn is None:
            return False
        rep = self._request(("contains", chunk_hash))
        return bool(rep[0] == "ok" and rep[1])

    def stage(self, stored: StoredChunk, device: torch.device) -> None:
        """Ship a CPU chunk to the staging worker (which does the H2D).

        The pipe pickling moves CPU tensors into shared memory once; the
        child copies shm -> GPU on ITS context and synchronizes there. This
        call returns after the child acked receipt -- the vLLM worker never
        blocks on the H2D itself beyond the host-side handoff.
        """
        self._ensure_started(device)
        rep = self._request(
            (
                "stage",
                stored.chunk_hash,
                stored.length,
                stored.old_positions,
                dict(stored.k_per_layer),
                dict(stored.v_per_layer),
            )
        )
        if rep[0] != "ok":
            logger.warning("EPIC staging worker stage failed: %s", rep)

    def get(self, chunk_hash: str) -> StagedChunk | None:
        cached = self._mapped.get(chunk_hash)
        if cached is not None:
            self._mapped.move_to_end(chunk_hash)
            return cached
        if self._conn is None:
            return None
        rep = self._request(("get", chunk_hash))
        if rep[0] != "hit":
            return None
        _, length, old_positions, k_dict, v_dict = rep
        staged = StagedChunk(
            chunk_hash=chunk_hash,
            length=int(length),
            old_positions=old_positions,
            k_per_layer=dict(k_dict),
            v_per_layer=dict(v_dict),
            ready_event=None,  # child synchronized before replying.
        )
        self._memo_put(chunk_hash, staged)
        return staged

    # -- introspection --

    def stats(self) -> dict:
        if self._conn is None:
            return {"chunks": 0, "bytes": 0}
        rep = self._request(("stats",))
        return rep[1] if rep[0] == "ok" else {}

    @property
    def current_bytes(self) -> int:
        return int(self.stats().get("bytes", 0))

    def __len__(self) -> int:
        return int(self.stats().get("chunks", 0))
