# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-side content-addressed KV chunk store for EPIC.

EPIC original (vllm_epic) kept reusable KV as a dense tensor on the model object
(``self.hack_kv``) and matched chunks out-of-band in its benchmark harness; it
never integrated with vLLM's prefix cache or block hashing.

Here we build a small content-addressed store:

  * A *chunk* is ``chunk_size`` tokens (a multiple of ``block_size``).
  * Its key is a hash of the chunk's **token ids alone** -- NOT a prefix chain.
    This is the position-independence핵심: the same text yields the same hash no
    matter where it appeared in a previous prompt. (vLLM's native prefix-cache
    block hash, by contrast, chains each block hash with its parent, so an
    identical chunk at a different offset hashes differently and cannot be
    reused out-of-prefix.)
  * For every stored chunk we keep per-layer K and V on (pinned) CPU plus the
    *original absolute positions* ``p_old`` the K was RoPE'd at, so the worker
    can PIC-delta-rotate it to its new position on reuse.

LRU eviction is byte-budgeted (configurable via ``extra_config``).
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

import torch


def hash_chunk_tokens(token_ids: list[int]) -> str:
    """Position-independent content hash of a chunk's token ids.

    EPIC change vs. vLLM prefix-cache: no parent-hash chaining, no block offset,
    no cache_salt -- purely the chunk's own bytes. Identical content at any
    position collides intentionally (that is the reuse signal).
    """
    h = hashlib.sha256()
    # Fixed-width little-endian ints for a stable, language-agnostic encoding.
    h.update(b"epic-chunk-v1")
    h.update(len(token_ids).to_bytes(4, "little"))
    for t in token_ids:
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.hexdigest()


@dataclass
class StoredChunk:
    """One cached chunk's KV across all layers."""

    chunk_hash: str
    length: int
    # Absolute positions this chunk's K was rotated to when first computed.
    old_positions: torch.Tensor  # int64 [length], CPU
    # Per-layer tensors. Each entry: K [length, num_kv_heads, head_size] and
    # V [length, num_kv_heads, head_size], CPU (pinned if available).
    k_per_layer: dict[str, torch.Tensor] = field(default_factory=dict)
    v_per_layer: dict[str, torch.Tensor] = field(default_factory=dict)

    def nbytes(self) -> int:
        total = self.old_positions.element_size() * self.old_positions.nelement()
        for t in self.k_per_layer.values():
            total += t.element_size() * t.nelement()
        for t in self.v_per_layer.values():
            total += t.element_size() * t.nelement()
        return total


class EpicChunkStore:
    """LRU, byte-budgeted CPU store of content-addressed KV chunks."""

    def __init__(self, capacity_bytes: int, pin_memory: bool = False):
        self.capacity_bytes = capacity_bytes
        self.pin_memory = pin_memory
        self._store: "OrderedDict[str, StoredChunk]" = OrderedDict()
        self._cur_bytes = 0

    # ----- lookup (scheduler side; existence only, no tensor touch) -----

    def contains(self, chunk_hash: str) -> bool:
        return chunk_hash in self._store

    def get_length(self, chunk_hash: str) -> int | None:
        chunk = self._store.get(chunk_hash)
        return None if chunk is None else chunk.length

    # ----- read (worker side; marks as recently used) -----

    def get(self, chunk_hash: str) -> StoredChunk | None:
        chunk = self._store.get(chunk_hash)
        if chunk is not None:
            self._store.move_to_end(chunk_hash)
        return chunk

    # ----- write (worker side) -----

    def put(self, chunk: StoredChunk) -> None:
        if chunk.chunk_hash in self._store:
            # Refresh existing entry.
            old = self._store.pop(chunk.chunk_hash)
            self._cur_bytes -= old.nbytes()
        chunk_bytes = chunk.nbytes()
        # A single chunk larger than the whole budget is simply not stored.
        if chunk_bytes > self.capacity_bytes:
            return
        self._store[chunk.chunk_hash] = chunk
        self._cur_bytes += chunk_bytes
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while self._cur_bytes > self.capacity_bytes and self._store:
            _, evicted = self._store.popitem(last=False)  # LRU = oldest
            self._cur_bytes -= evicted.nbytes()

    def maybe_pin(self, t: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor to pinned memory when enabled and possible."""
        if self.pin_memory and t.device.type == "cpu" and not t.is_pinned():
            try:
                return t.pin_memory()
            except (RuntimeError, NotImplementedError):
                return t
        return t

    # ----- introspection (tests / metrics) -----

    @property
    def current_bytes(self) -> int:
        return self._cur_bytes

    def __len__(self) -> int:
        return len(self._store)
