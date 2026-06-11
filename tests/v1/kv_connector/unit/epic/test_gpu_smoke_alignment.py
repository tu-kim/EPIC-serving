# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit test for the gpu_smoke step4 token-alignment precompute.

The GPU smoke itself needs a CUDA box + a model, so it is not collected by
pytest. But its NEW chunk-alignment logic (build_aligned_token_prompts) is pure
list math and is the crux of fix 3: it must guarantee that the shared passage B
is byte-identical AND chunk-aligned across the warm and reuse prompts, so the
connector's content hash collides and a real non-prefix hit is produced. We unit
test that here, on CPU, with no tokenizer / GPU.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    hash_chunk_tokens,
)

# gpu_smoke is a standalone script (no test_* functions) but importing it is
# safe: the heavy work is gated behind __main__ / CUDA checks.
from tests.v1.kv_connector.unit.epic.gpu_smoke import (
    build_aligned_token_prompts,
)

CHUNK = 64  # small block-multiple for fast CPU math.


def _assemble(head_len, reuse_head_len, passage_len, passage_chunks=1):
    return build_aligned_token_prompts(
        head_ids=list(range(10, 10 + head_len)),
        passage_ids=list(range(1000, 1000 + passage_len)),
        tail_ids=list(range(5000, 5010)),
        reuse_head_ids=list(range(20, 20 + reuse_head_len)),
        reuse_tail_ids=list(range(6000, 6007)),
        chunk_size=CHUNK,
        filler_id=0,
        passage_chunks=passage_chunks,
    )


def test_b_is_chunk_aligned_in_both_prompts():
    # Heads of DIFFERENT, non-multiple lengths -> padding must still land B on a
    # chunk boundary in both prompts.
    out = _assemble(head_len=70, reuse_head_len=30, passage_len=200)
    assert out["warm_b_offset"] % CHUNK == 0
    assert out["reuse_b_offset"] % CHUNK == 0
    # Heads differ -> B is at DIFFERENT offsets (so it is genuinely non-prefix in
    # the reuse prompt, not a shared prefix).
    assert out["warm_b_offset"] != out["reuse_b_offset"]


def test_b_tokens_byte_identical_and_hashes_match():
    out = _assemble(head_len=70, reuse_head_len=30, passage_len=200)
    warm, reuse = out["warm_ids"], out["reuse_ids"]
    bo_w, bo_r = out["warm_b_offset"], out["reuse_b_offset"]
    b_len = out["b_len"]
    # The exact B id slices are byte-identical across the two prompts.
    assert warm[bo_w : bo_w + b_len] == reuse[bo_r : bo_r + b_len]
    # And their per-chunk hashes equal the precomputed expectation.
    expected = out["expected_b_hashes"]
    assert len(expected) == out["passage_chunks"]
    for c, h in enumerate(expected):
        chunk = warm[bo_w + c * CHUNK : bo_w + (c + 1) * CHUNK]
        assert hash_chunk_tokens(chunk) == h
        rchunk = reuse[bo_r + c * CHUNK : bo_r + (c + 1) * CHUNK]
        assert hash_chunk_tokens(rchunk) == h


def test_b_truncated_to_exact_chunk_multiple():
    # Passage longer than one chunk but we only take passage_chunks=1.
    out = _assemble(head_len=64, reuse_head_len=64, passage_len=500)
    assert out["b_len"] == CHUNK
    # Two-chunk B.
    out2 = _assemble(head_len=64, reuse_head_len=64, passage_len=500, passage_chunks=2)
    assert out2["b_len"] == 2 * CHUNK
    assert len(out2["expected_b_hashes"]) == 2


def test_short_passage_is_padded_to_exact_length():
    # Passage shorter than the requested chunk count -> filler-padded to exact.
    out = _assemble(head_len=64, reuse_head_len=64, passage_len=10)
    assert out["b_len"] == CHUNK
    bo = out["warm_b_offset"]
    b_slice = out["warm_ids"][bo : bo + CHUNK]
    assert len(b_slice) == CHUNK
    # Same exact slice in reuse -> hashes still match.
    assert out["expected_b_hashes"] == [hash_chunk_tokens(b_slice)]


def test_head_already_multiple_not_overpadded():
    # A head that is already a chunk multiple must not gain an extra chunk.
    out = _assemble(head_len=CHUNK, reuse_head_len=2 * CHUNK, passage_len=CHUNK)
    assert out["warm_b_offset"] == CHUNK
    assert out["reuse_b_offset"] == 2 * CHUNK
