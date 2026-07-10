# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident fileKV staging for EPIC prefetch (feature/prefetch).

Motivation (agentic serving): the previous turn's LLM output names the files
the next turn will read (structured tool calls). Between turns the frontend
(e.g. a dynamo scheduler) already knows WHICH worker/replica will serve the
next turn, so the needed fileKV chunks can be copied CPU -> GPU *ahead of
time*. When the next turn's prefill then loads those chunks, the H2D copy --
the dominant latency of ``EpicConnector._load_chunk`` -- is already done and
the load degrades to a GPU->GPU scatter (plus the PIC rotation, which always
ran on-device anyway).

Design:

  * ``EpicGpuStagingStore`` mirrors the CPU ``EpicChunkStore`` shape (per-layer
    K/V + old_positions per chunk) but holds tensors on the KV-cache device,
    byte-budgeted with LRU eviction.
  * Staging copies run on a DEDICATED CUDA side stream and record a
    ``torch.cuda.Event`` per chunk, so they overlap the ongoing decode steps of
    the *current* turn instead of blocking any forward. Consumers order their
    stream on the event (``current_stream().wait_event``) -- a device-side
    dependency, not a host sync.
  * On CPU (unit tests / no CUDA) staging is a plain synchronous copy and the
    event is ``None``; the store is fully functional.

The store is deliberately *lossy* (budgeted cache): a prefetch is a hint. A
miss at load time simply falls back to the CPU fileKV store -- the caller's
required behavior ("있으면 바로 읽고 없으면 fileKV에서 찾는다").
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    StoredChunk,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class StagedChunk:
    """One chunk's KV staged on the target device (usually the GPU).

    Field-compatible with ``StoredChunk`` where it matters (length,
    old_positions, k_per_layer, v_per_layer) so the load path can treat either
    as the K/V source; ``ready_event`` additionally carries the async-copy
    completion marker (None == already synchronous/complete).
    """

    chunk_hash: str
    length: int
    old_positions: torch.Tensor  # int64 [length], on the staging device
    k_per_layer: dict[str, torch.Tensor] = field(default_factory=dict)
    v_per_layer: dict[str, torch.Tensor] = field(default_factory=dict)
    ready_event: "torch.cuda.Event | None" = None

    def nbytes(self) -> int:
        total = self.old_positions.element_size() * self.old_positions.nelement()
        for t in self.k_per_layer.values():
            total += t.element_size() * t.nelement()
        for t in self.v_per_layer.values():
            total += t.element_size() * t.nelement()
        return total


class EpicGpuStagingStore:
    """LRU, byte-budgeted device-side staging cache for prefetched chunks."""

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = int(capacity_bytes)
        self._store: "OrderedDict[str, StagedChunk]" = OrderedDict()
        self._cur_bytes = 0
        # Dedicated copy stream (CUDA only). Created lazily on the first stage
        # so the store can be constructed before the device is known.
        self._copy_stream: "torch.cuda.Stream | None" = None

    # ----- membership -----

    def contains(self, chunk_hash: str) -> bool:
        return chunk_hash in self._store

    def get(self, chunk_hash: str) -> StagedChunk | None:
        """Return the staged chunk, ordering the CURRENT stream on its copy.

        The event wait is a device-side dependency (no host block); after it,
        tensors read on the current stream see the completed H2D copy. Marks
        the chunk most-recently-used.
        """
        staged = self._store.get(chunk_hash)
        if staged is None:
            return None
        self._store.move_to_end(chunk_hash)
        ev = staged.ready_event
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
        return staged

    # ----- staging (async H2D on the side stream) -----

    def stage(self, stored: StoredChunk, device: torch.device) -> StagedChunk:
        """Copy a CPU StoredChunk to ``device`` ahead of use.

        Idempotent: an already-staged hash is refreshed in LRU order and
        returned as-is (no re-copy). On CUDA the copies are issued on the
        dedicated side stream (overlapping compute on the default stream) and
        a ready event is recorded; elsewhere the copy is synchronous.
        """
        existing = self._store.get(stored.chunk_hash)
        if existing is not None:
            self._store.move_to_end(stored.chunk_hash)
            return existing

        use_cuda = device.type == "cuda" and torch.cuda.is_available()
        if use_cuda and self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=device)

        def _copy_all() -> StagedChunk:
            staged = StagedChunk(
                chunk_hash=stored.chunk_hash,
                length=stored.length,
                old_positions=stored.old_positions.to(device, non_blocking=True),
            )
            for name, k in stored.k_per_layer.items():
                staged.k_per_layer[name] = k.to(device, non_blocking=True)
            for name, v in stored.v_per_layer.items():
                staged.v_per_layer[name] = v.to(device, non_blocking=True)
            return staged

        if use_cuda:
            assert self._copy_stream is not None
            with torch.cuda.stream(self._copy_stream):
                staged = _copy_all()
                ev = torch.cuda.Event()
                ev.record(self._copy_stream)
                staged.ready_event = ev
        else:
            staged = _copy_all()

        nbytes = staged.nbytes()
        if nbytes > self.capacity_bytes:
            logger.warning(
                "EPIC prefetch: chunk %s (%d bytes) exceeds the GPU staging "
                "budget (%d bytes); not staged.",
                stored.chunk_hash,
                nbytes,
                self.capacity_bytes,
            )
            return staged  # usable by the caller this once, but not cached.
        self._store[stored.chunk_hash] = staged
        self._cur_bytes += nbytes
        self._evict_if_needed()
        return staged

    def evict(self, chunk_hash: str) -> None:
        staged = self._store.pop(chunk_hash, None)
        if staged is not None:
            self._cur_bytes -= staged.nbytes()

    def _evict_if_needed(self) -> None:
        while self._cur_bytes > self.capacity_bytes and self._store:
            _, evicted = self._store.popitem(last=False)  # LRU = oldest
            self._cur_bytes -= evicted.nbytes()

    # ----- introspection -----

    @property
    def current_bytes(self) -> int:
        return self._cur_bytes

    def __len__(self) -> int:
        return len(self._store)
