# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EPIC chunk store and content hashing (CPU-only)."""

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    EpicChunkStore,
    StoredChunk,
    hash_chunk_tokens,
)


def test_hash_is_position_independent():
    # Same content at different prompt positions -> same hash.
    content = [10, 20, 30, 40]
    h_a = hash_chunk_tokens(content)
    h_b = hash_chunk_tokens(content)
    assert h_a == h_b

    # Prefixing different tokens before the chunk must NOT change its hash:
    # the hash is over the chunk's own tokens only, not a prefix chain.
    assert hash_chunk_tokens(content) == hash_chunk_tokens(list(content))

    # Different content -> different hash.
    assert hash_chunk_tokens([10, 20, 30, 41]) != h_a
    # Different length -> different hash (length is mixed in).
    assert hash_chunk_tokens([10, 20, 30]) != h_a


def test_hash_not_prefix_chained():
    # Two distinct chunks; the hash of the second does not depend on the first.
    c1 = [1, 2, 3, 4]
    c2 = [5, 6, 7, 8]
    # Concatenation prefix should not leak into c2's hash.
    assert hash_chunk_tokens(c2) == hash_chunk_tokens([5, 6, 7, 8])
    assert hash_chunk_tokens(c1) != hash_chunk_tokens(c2)


def _make_chunk(name: str, length: int, layers: int) -> StoredChunk:
    chunk = StoredChunk(
        chunk_hash=name,
        length=length,
        old_positions=torch.arange(length, dtype=torch.int64),
    )
    for li in range(layers):
        chunk.k_per_layer[f"l{li}"] = torch.zeros(length, 2, 8)
        chunk.v_per_layer[f"l{li}"] = torch.zeros(length, 2, 8)
    return chunk


def test_save_lookup_roundtrip():
    store = EpicChunkStore(capacity_bytes=10**9)
    c = _make_chunk("h1", length=16, layers=3)
    store.put(c)

    assert store.contains("h1")
    assert store.get_length("h1") == 16
    got = store.get("h1")
    assert got is not None
    assert set(got.k_per_layer.keys()) == {"l0", "l1", "l2"}
    assert not store.contains("missing")
    assert store.get("missing") is None


def test_lru_eviction_by_bytes():
    # Each chunk is identical in size; budget fits ~2 of them.
    c0 = _make_chunk("h0", length=16, layers=2)
    one = c0.nbytes()
    store = EpicChunkStore(capacity_bytes=int(one * 2.5))

    store.put(_make_chunk("h0", 16, 2))
    store.put(_make_chunk("h1", 16, 2))
    assert store.contains("h0") and store.contains("h1")

    # Touch h0 so h1 is the LRU victim.
    store.get("h0")
    store.put(_make_chunk("h2", 16, 2))

    assert store.contains("h0")
    assert store.contains("h2")
    assert not store.contains("h1")  # evicted as LRU
    assert store.current_bytes <= store.capacity_bytes


def test_oversized_chunk_not_stored():
    store = EpicChunkStore(capacity_bytes=10)
    store.put(_make_chunk("big", 16, 2))
    assert not store.contains("big")
    assert len(store) == 0
