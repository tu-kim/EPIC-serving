# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the EPIC worker load-fidelity self-check (Implement 2).

``check_scatter_fidelity`` is the pure read-back oracle the worker uses (when
``epic_debug_check_load`` is on) to confirm that a scatter actually landed in the
slots it was supposed to, in the supported FlashAttention paged layout
``(2, num_blocks, block_size, H, D)``. It is the in-situ detector for
layout/stride/aliasing bugs -- the failure class that produces a correct-looking
reuse plan but a corrupted forward.

These tests run on CPU with a hand-built fake paged cache: a correct scatter must
read back as allclose with ~0 diff, and a DELIBERATE slot-offset error (the
canonical "off-by-one block / wrong stride" mistake) must be caught (allclose
False, non-zero max-abs-diff).
"""

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
    check_scatter_fidelity,
)

NUM_BLOCKS = 4
BLOCK_SIZE = 8
H = 2
D = 3


def _fake_cache(dtype=torch.float32) -> torch.Tensor:
    # (2, num_blocks, block_size, H, D): bank 0 = K, bank 1 = V. Start at zero so
    # any un-written slot is distinguishable from scattered content.
    return torch.zeros(2, NUM_BLOCKS, BLOCK_SIZE, H, D, dtype=dtype)


def _scatter(cache, k, v, slots):
    """Mirror EpicConnector._scatter_kv exactly (the code under test wraps the
    read-back, not the write; we drive the write directly here)."""
    num_blocks, block_size = cache.shape[1], cache.shape[2]
    k_bank = cache[0].reshape(num_blocks * block_size, H, D)
    v_bank = cache[1].reshape(num_blocks * block_size, H, D)
    s = torch.as_tensor(slots, dtype=torch.long)
    k_bank[s] = k
    v_bank[s] = v


def test_correct_scatter_reads_back_clean():
    cache = _fake_cache()
    n = 5
    k = torch.randn(n, H, D)
    v = torch.randn(n, H, D)
    # Non-prefix-ish slots spanning two blocks (offset into the cache).
    slots = [10, 11, 12, 13, 14]
    _scatter(cache, k, v, slots)

    res = check_scatter_fidelity(cache, k, v, slots)
    assert res is not None
    k_ok, k_diff, v_ok, v_diff = res
    assert k_ok and v_ok
    assert k_diff == 0.0 and v_diff == 0.0


def test_wrong_slot_offset_is_detected():
    # Inject the canonical stride/offset bug: we WRITE to the right slots but
    # ask the checker to verify a DIFFERENT (off-by-block) slot range. That is
    # exactly what an aliasing/stride mistake looks like from the oracle's view:
    # the read-back slots do not hold the reference content.
    cache = _fake_cache()
    n = 5
    k = torch.randn(n, H, D)
    v = torch.randn(n, H, D)
    written_slots = [10, 11, 12, 13, 14]
    _scatter(cache, k, v, written_slots)

    # Verify against slots shifted by one full block (8) -> all zeros there.
    wrong_slots = [s + BLOCK_SIZE for s in written_slots]
    res = check_scatter_fidelity(cache, k, v, wrong_slots)
    assert res is not None
    k_ok, k_diff, v_ok, v_diff = res
    assert not k_ok and not v_ok
    assert k_diff > 0.0 and v_diff > 0.0


def test_partial_corruption_is_detected():
    # A single corrupted row (one slot overwritten after scatter) must trip the
    # allclose check even though most rows match.
    cache = _fake_cache()
    n = 6
    k = torch.randn(n, H, D)
    v = torch.randn(n, H, D)
    slots = list(range(20, 26))
    _scatter(cache, k, v, slots)
    # Corrupt one K row in place (simulating a stride that aliases one token).
    k_bank = cache[0].reshape(NUM_BLOCKS * BLOCK_SIZE, H, D)
    k_bank[slots[3]] += 1.0

    res = check_scatter_fidelity(cache, k, v, slots)
    assert res is not None
    k_ok, k_diff, v_ok, v_diff = res
    assert not k_ok  # K corrupted
    assert k_diff >= 1.0
    assert v_ok and v_diff == 0.0  # V untouched


def test_unsupported_layout_returns_none():
    # Not the (2, ...5-D) FlashAttention layout -> oracle returns None (caller
    # logs SKIP). A 3-D tensor stands in for an unsupported backend layout.
    cache = torch.zeros(2, 16, 4)
    res = check_scatter_fidelity(cache, torch.zeros(1, 1), torch.zeros(1, 1), [0])
    assert res is None


def test_dtype_downcast_not_flagged():
    # The store may hold a wider dtype than the cache; check compares against the
    # cast-to-cache-dtype reference, so a benign downcast is NOT a mismatch.
    cache = _fake_cache(dtype=torch.float16)
    n = 4
    k = torch.randn(n, H, D, dtype=torch.float32)
    v = torch.randn(n, H, D, dtype=torch.float32)
    slots = [0, 1, 2, 3]
    # Scatter casts to cache dtype (fp16), mirroring _scatter_kv's .to(dtype).
    _scatter(cache, k.to(torch.float16), v.to(torch.float16), slots)

    res = check_scatter_fidelity(cache, k, v, slots)
    assert res is not None
    k_ok, _, v_ok, _ = res
    # Placement is correct; only difference is the fp32->fp16 round, which the
    # checker's .to(cache.dtype) reference absorbs -> allclose holds.
    assert k_ok and v_ok
