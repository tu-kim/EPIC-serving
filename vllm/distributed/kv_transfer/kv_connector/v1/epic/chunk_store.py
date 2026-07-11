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

import numpy as np
import torch


def _tokens_to_bytes(token_ids: "list[int] | np.ndarray") -> bytes:
    """Fixed-width little-endian uint32 encoding of a token sequence.

    Byte-identical to the original per-token
    ``int(t).to_bytes(4, "little", signed=False)`` loop, but one vectorized
    numpy conversion instead of len(token_ids) Python calls -- hashing is on
    the scheduler hot path (every prompt is content- AND chain-hashed at
    match and save time). Raises OverflowError for out-of-range ids exactly
    like ``int.to_bytes`` did (numpy would silently wrap otherwise).
    """
    arr = np.asarray(token_ids, dtype=np.int64)
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= 1 << 32):
        raise OverflowError("token id out of uint32 range")
    return arr.astype("<u4").tobytes()


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
    h.update(_tokens_to_bytes(token_ids))
    return h.hexdigest()


class ChainHasher:
    """Incremental digest of a prompt's token sequence from position 0.

    ``digest()`` after feeding tokens [0, i) identifies the SAVE-TIME LEFT
    CONTEXT of a chunk starting at i. Two chunks agree on their chain digest
    iff their preceding token sequences are identical (collision-negligible).

    This is what makes the EXACT-prefix fold sound: a content-hash hit says
    "same bytes", but only a matching chain says "computed with the same
    context" -- a fileKV chunk warmed in isolation has chain == digest(empty)
    and must NOT be folded into the exact prefix of a prompt where other
    content precedes it (it needs the EPIC link stitch instead).
    """

    def __init__(self) -> None:
        self._h = hashlib.sha256(b"epic-chain-v1")

    def update(self, token_ids: list[int]) -> None:
        self._h.update(_tokens_to_bytes(token_ids))

    def digest(self) -> str:
        return self._h.copy().hexdigest()


def chain_hash_tokens(token_ids: list[int]) -> str:
    """Chain digest of a full token prefix (convenience for tests/clients)."""
    ch = ChainHasher()
    ch.update(list(token_ids))
    return ch.digest()


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
    # Save-time context chain (see ChainHasher): digests of the save prompt's
    # tokens [0, start) and [0, start+length). None == unknown (legacy /
    # hand-built test chunks) -> consumers treat context as unverifiable.
    chain_start: str | None = None
    chain_end: str | None = None

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

    def get_old_pos_start(self, chunk_hash: str) -> int | None:
        """First absolute position the stored chunk's K was rotated at.

        Membership-style query (no LRU touch). Used by selection to detect
        SAME-FILE contiguity between adjacent hits: two chunks saved from one
        contiguous warm request have contiguous old positions, which is what
        makes a run-internal chunk boundary coherent (per-run link tokens).
        """
        chunk = self._store.get(chunk_hash)
        if chunk is None or chunk.old_positions.numel() == 0:
            return None
        return int(chunk.old_positions[0].item())

    def get_chain(self, chunk_hash: str) -> tuple[str | None, str | None] | None:
        """Save-time context chain (chain_start, chain_end); None == not stored.

        Membership-style query (no LRU touch). ``(None, None)`` entries are
        legacy/hand-built chunks whose save context is unverifiable.
        """
        chunk = self._store.get(chunk_hash)
        if chunk is None:
            return None
        return (chunk.chain_start, chunk.chain_end)

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

    def iter_membership(self):
        """Yield ``(chunk_hash, length, old_pos_start, chain_start, chain_end)``
        in LRU order.

        Lets a scheduler index be seeded from a (test-built or pre-warmed) store
        without reaching into private state.
        """
        for h, chunk in self._store.items():
            old_pos = (
                int(chunk.old_positions[0].item())
                if chunk.old_positions.numel() > 0
                else -1
            )
            yield h, chunk.length, old_pos, chunk.chain_start, chunk.chain_end


# ---------------------------------------------------------------------------
# Scheduler-side mirror index (role-split fix)
# ---------------------------------------------------------------------------
#
# Root cause this exists to fix: the EpicConnector is instantiated TWICE -- once
# with role==SCHEDULER inside the EngineCore process (where get_num_new_matched_
# tokens / build_connector_meta run) and once with role==WORKER inside each
# worker process (where save_kv_layer / start_load_kv run). They are SEPARATE
# Python objects in SEPARATE processes, so the worker's ``EpicChunkStore`` (which
# actually holds the harvested chunk tensors) is invisible to the scheduler. The
# scheduler's selection (EpicSelection.select -> store.contains/get_length) would
# therefore always miss -> 0 hits -> "EPIC sparse skip: no non-prefix hits".
#
# Fix: every save is *decided by the scheduler* in build_connector_meta (the
# worker only executes the emitted EpicReqSave). So the scheduler can mirror
# worker-store membership DETERMINISTICALLY by registering each chunk it emits a
# save for into this lightweight, byte-budgeted LRU index -- using the SAME
# eviction policy and the SAME per-chunk byte accounting as the worker's
# EpicChunkStore. The index holds NO tensors (only ``length`` + ``old_pos_start``
# + the deterministic byte size); it exists purely so that
# ``contains()`` / ``get_length()`` on the scheduler side return the same answer
# the worker store would.
#
# Membership protocol (what EpicSelection.select consumes): ``contains(hash)``
# and ``get_length(hash)``. Both EpicChunkStore and EpicSchedulerIndex implement
# them, so selection is store/index agnostic (see SupportsChunkMembership).


@dataclass
class _IndexEntry:
    """One mirrored chunk's metadata (no tensors)."""

    length: int
    nbytes: int
    old_pos_start: int = -1  # diagnostics only; worker resolves real old pos.
    # Save-time context chain (mirrors StoredChunk.chain_start/chain_end).
    chain_start: str | None = None
    chain_end: str | None = None


def stored_chunk_nbytes(
    length: int,
    num_layers: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_size: int,
) -> int:
    """Deterministic byte size of a StoredChunk, identical on both sides.

    Mirrors ``StoredChunk.nbytes()`` exactly:

      * per layer: K and V are each ``[length, num_kv_heads, head_size]`` of the
        cache dtype -> ``length * num_kv_heads * head_size * cache_dtype_size``
        bytes each, so ``2 *`` that per layer, ``* num_layers`` layers.
      * plus ``old_positions``: an int64 tensor of ``length`` -> ``8 * length``.

    Pinning (``pin_memory``) does NOT change ``element_size() * nelement()``, so
    the worker's actual ``StoredChunk.nbytes()`` equals this for the same dims.
    The connector computes the dims once from ``kv_cache_config`` / ``VllmConfig``
    so both roles agree without ever materializing tensors on the scheduler.
    """
    per_token_kv = 2 * num_layers * num_kv_heads * head_size * cache_dtype_size
    old_positions_bytes = 8 * length  # int64
    return length * per_token_kv + old_positions_bytes


class EpicSchedulerIndex:
    """Scheduler-side deterministic mirror of the worker ``EpicChunkStore``.

    Same byte budget + same LRU policy + same per-chunk byte accounting as the
    worker store, but holds only metadata (no KV tensors). Registering a chunk
    here when the scheduler emits its save reproduces the worker store's
    membership and eviction order deterministically (every save is scheduler-
    decided), so scheduler-side selection sees real hits.

    Exposes the SAME membership surface as EpicChunkStore
    (``contains`` / ``get_length``) so EpicSelection.select is agnostic to which
    one it is handed.
    """

    def __init__(
        self,
        capacity_bytes: int,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_size: int,
    ):
        self.capacity_bytes = capacity_bytes
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size
        self._cache_dtype_size = cache_dtype_size
        self._store: "OrderedDict[str, _IndexEntry]" = OrderedDict()
        self._cur_bytes = 0

    def nbytes_for(self, length: int) -> int:
        return stored_chunk_nbytes(
            length,
            self._num_layers,
            self._num_kv_heads,
            self._head_size,
            self._cache_dtype_size,
        )

    # ----- membership (matches EpicChunkStore; consumed by EpicSelection) -----

    def contains(self, chunk_hash: str) -> bool:
        return chunk_hash in self._store

    def get_length(self, chunk_hash: str) -> int | None:
        entry = self._store.get(chunk_hash)
        return None if entry is None else entry.length

    def get_old_pos_start(self, chunk_hash: str) -> int | None:
        entry = self._store.get(chunk_hash)
        return None if entry is None else entry.old_pos_start

    # ----- mirror write (scheduler side, at save-emit time) -----

    def register(
        self,
        chunk_hash: str,
        length: int,
        old_pos_start: int = -1,
        chain_start: str | None = None,
        chain_end: str | None = None,
    ) -> None:
        """Mirror a save the scheduler just emitted.

        Same effect as EpicChunkStore.put: refresh-or-insert, byte-accounted,
        LRU-evict. A chunk larger than the whole budget is not registered (the
        worker store drops it too), keeping the mirror consistent.
        """
        nbytes = self.nbytes_for(length)
        if chunk_hash in self._store:
            old = self._store.pop(chunk_hash)
            self._cur_bytes -= old.nbytes
        if nbytes > self.capacity_bytes:
            return
        self._store[chunk_hash] = _IndexEntry(
            length=length,
            nbytes=nbytes,
            old_pos_start=old_pos_start,
            chain_start=chain_start,
            chain_end=chain_end,
        )
        self._cur_bytes += nbytes
        self._evict_if_needed()

    def get_chain(self, chunk_hash: str) -> tuple[str | None, str | None] | None:
        """Save-time context chain (matches EpicChunkStore.get_chain)."""
        entry = self._store.get(chunk_hash)
        if entry is None:
            return None
        return (entry.chain_start, entry.chain_end)

    def seed_from_store(self, store: "EpicChunkStore") -> None:
        """Mirror an existing store's membership (LRU order preserved).

        Used in tests / warm-start to make the scheduler index reflect a store
        that was populated out of band. In production the index is kept in sync
        incrementally via ``register`` at save-emit time.
        """
        for chunk_hash, length, old_pos, c_start, c_end in (
            store.iter_membership()
        ):
            self.register(
                chunk_hash,
                length,
                old_pos_start=old_pos,
                chain_start=c_start,
                chain_end=c_end,
            )

    def touch(self, chunk_hash: str) -> None:
        """Mark a chunk most-recently-used (mirrors EpicChunkStore.get's
        move_to_end on a worker read). Call when the scheduler emits a LOAD for
        an already-cached chunk so the mirror's LRU ordering tracks the worker's.
        """
        if chunk_hash in self._store:
            self._store.move_to_end(chunk_hash)

    def _evict_if_needed(self) -> None:
        while self._cur_bytes > self.capacity_bytes and self._store:
            _, evicted = self._store.popitem(last=False)  # LRU = oldest
            self._cur_bytes -= evicted.nbytes

    # ----- introspection -----

    @property
    def current_bytes(self) -> int:
        return self._cur_bytes

    def __len__(self) -> int:
        return len(self._store)
