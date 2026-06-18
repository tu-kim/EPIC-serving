# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the step4 NEEDLE probe rewrite of gpu_smoke.py.

The needle probe needs a GPU + a model to RUN, but every piece of logic it
relies on -- needle/question construction, chunk-aligned assembly with
non-blank rotating filler, the degeneracy guards, the answer detector, and the
in-band engagement counters on EpicConnector -- is pure and must be correct
before any GPU time is spent. We verify it all here, on CPU, with no
tokenizer / GPU. (A blank-space filler producing a vacuous 1.000 match was the
exact bug this rewrite fixes, so these tests pin the discriminative invariants.)
"""

import pytest

from tests.v1.kv_connector.unit.epic.gpu_smoke import (
    DEFAULT_CHUNK_SIZE,
    NeedleFact,
    _DEFAULT_NEEDLE_OFFSET_FRAC,
    _distinct_token_count,
    _epic_kv_config,
    _LINK_SWEEP,
    _output_has_answer,
    _parse_needle_offsets,
    build_aligned_token_prompts,
    build_needle_passage_text,
    build_needle_passage_tokens,
    make_needles,
    needle_in_link,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    hash_chunk_tokens,
)


def _char_enc(text: str) -> list[int]:
    """Deterministic, tokenizer-free encoder for the token-level placement
    tests: one id per character (often >1 token per word, like a real BPE), so
    the offset arithmetic is exercised without loading a tokenizer/torch."""
    return [ord(c) % 1000 for c in text]

# ---------------------------------------------------------------------------
# Needle facts + question
# ---------------------------------------------------------------------------


def test_make_needles_deterministic_and_distinct():
    a = make_needles(seed=4, k=3)
    b = make_needles(seed=4, k=3)
    assert [(n.subject, n.answer) for n in a] == [(n.subject, n.answer) for n in b]
    # K distinct subjects and K distinct codes (so the question is unambiguous).
    assert len({n.subject for n in a}) == 3
    assert len({n.answer for n in a}) == 3
    # Codes are 4-digit strings.
    for n in a:
        assert n.answer.isdigit() and len(n.answer) == 4


def test_fact_sentence_and_question_carry_subject_and_answer():
    nd = NeedleFact(subject="alpha-bravo", answer="1234")
    fact = nd.fact_sentence()
    q = nd.question()
    assert "alpha-bravo" in fact and "1234" in fact
    # The question asks for the subject but must NOT leak the answer.
    assert "alpha-bravo" in q
    assert "1234" not in q
    assert q.strip().endswith("Answer:")


def test_passage_contains_every_needle_fact():
    needles = make_needles(seed=4, k=3)
    passage = build_needle_passage_text(needles, filler_seed=11)
    for n in needles:
        assert n.fact_sentence() in passage
        assert n.answer in passage


def test_passage_has_filler_words_not_blank_padding():
    needles = make_needles(seed=4, k=1)
    passage = build_needle_passage_text(needles, filler_seed=11, n_filler_words=20)
    # The filler block is real words separated by spaces, so the passage has
    # many distinct word tokens (not a run of one repeated blank).
    words = passage.split()
    assert len(set(words)) >= 10


# ---------------------------------------------------------------------------
# Answer detection + degeneracy helpers
# ---------------------------------------------------------------------------


def test_output_has_answer_substring():
    assert _output_has_answer("The secret code is 1234 indeed.", "1234") is True
    assert _output_has_answer("no code here", "1234") is False
    assert _output_has_answer("1234", "") is False
    assert _output_has_answer("", "1234") is False


def test_distinct_token_count():
    assert _distinct_token_count([1, 1, 1]) == 1
    assert _distinct_token_count([1, 2, 3, 3]) == 3
    assert _distinct_token_count([]) == 0


# ---------------------------------------------------------------------------
# Chunk-aligned assembly with rotating (non-blank) filler
# ---------------------------------------------------------------------------


def test_alignment_holds_with_rotating_filler():
    # Heads of awkward (non-chunk-multiple) length; filler is a multi-id pool.
    cs = 16
    pool = [101, 102, 103, 104, 105]
    out = build_aligned_token_prompts(
        head_ids=[1, 2, 3],
        passage_ids=list(range(200, 200 + cs)),  # exactly one chunk of B
        tail_ids=[9, 9],
        reuse_head_ids=[4, 5, 6, 7, 8, 9, 10],  # different length -> non-prefix
        reuse_tail_ids=[7],
        chunk_size=cs,
        filler_ids=pool,
        passage_chunks=1,
    )
    # Both heads padded to a chunk multiple -> B starts on a chunk boundary.
    assert out["warm_b_offset"] % cs == 0
    assert out["reuse_b_offset"] % cs == 0
    # B byte-identical across prompts -> the B-chunk hash collides (reuse signal).
    assert out["expected_b_hashes"] == [hash_chunk_tokens(list(range(200, 200 + cs)))]
    # The reuse signal is differing HEAD CONTENT (the first A chunk's hash
    # differs), NOT differing B offsets: both heads pad to the same chunk
    # multiple, so B sits at the same offset, but the prefix-hash walk stops at
    # the differing chunk-0 -> B is a NON-prefix hit. Confirm the first chunks
    # of the two prompts hash differently.
    warm_chunk0 = hash_chunk_tokens(out["warm_ids"][:cs])
    reuse_chunk0 = hash_chunk_tokens(out["reuse_ids"][:cs])
    assert warm_chunk0 != reuse_chunk0


def test_rotating_filler_is_not_a_single_repeated_token():
    cs = 16
    pool = [101, 102, 103, 104, 105]
    out = build_aligned_token_prompts(
        head_ids=[1, 2, 3],
        passage_ids=list(range(200, 200 + cs)),
        tail_ids=[],
        reuse_head_ids=[4, 5],
        reuse_tail_ids=[],
        chunk_size=cs,
        filler_ids=pool,
        passage_chunks=1,
    )
    # The padded region of warm_head (positions after the 3 real head ids, up to
    # the chunk boundary) must use MULTIPLE distinct filler ids, not one.
    pad_region = out["warm_ids"][3 : out["warm_b_offset"]]
    assert len(pad_region) > 1
    assert len(set(pad_region)) >= 2  # rotation, not a constant


def test_scalar_filler_back_compat():
    # filler_ids=None falls back to the scalar filler_id (old behavior).
    cs = 8
    out = build_aligned_token_prompts(
        head_ids=[1, 2, 3],
        passage_ids=list(range(50, 50 + cs)),
        tail_ids=[],
        reuse_head_ids=[4, 5],
        reuse_tail_ids=[],
        chunk_size=cs,
        filler_id=0,
        passage_chunks=1,
    )
    pad_region = out["warm_ids"][3 : out["warm_b_offset"]]
    assert set(pad_region) == {0}


def test_short_passage_padded_with_rotating_filler_keeps_hash_stable():
    # A passage SHORTER than one chunk is padded with the rotating pool; the same
    # padding must apply to both prompts so the hash still collides.
    cs = 16
    pool = [7, 8, 9]
    out = build_aligned_token_prompts(
        head_ids=[1],
        passage_ids=[200, 201],  # only 2 tokens -> padded to 16
        tail_ids=[],
        reuse_head_ids=[2, 3, 4],
        reuse_tail_ids=[],
        chunk_size=cs,
        filler_ids=pool,
        passage_chunks=1,
    )
    assert out["b_len"] == cs
    # Reconstruct the padded B and confirm the recorded hash matches it.
    b = [200, 201] + [pool[i % len(pool)] for i in range(cs - 2)]
    assert out["expected_b_hashes"] == [hash_chunk_tokens(b)]


# ---------------------------------------------------------------------------
# Token-level TARGET-needle placement (the link-overlap bias fix)
#
# Bias being fixed: build_needle_passage_text put the facts at the FRONT of B,
# but EPIC LegoLink RECOMPUTES the leading eff_link tokens of each non-prefix
# chunk (reuse_strategy.py "(2) link tokens"). At link=8/64 a front-loaded
# target needle lands in the recomputed region, so a HIT cannot be attributed
# to reused KV vs. recompute. build_needle_passage_tokens places the TARGET at
# a deep token offset so it sits OUTSIDE the recomputed region for non-full
# links -- a HIT there is a pure reused-KV proof.
# ---------------------------------------------------------------------------


def test_needle_tokens_target_lands_exactly_at_offset_and_B_is_exact():
    cs = 256
    needles = make_needles(seed=4, k=3)
    target = needles[0]
    off = 179
    b = build_needle_passage_tokens(
        needles, target, encode=_char_enc, chunk_size=cs,
        needle_offset=off, filler_seed=11,
    )
    # B is EXACTLY chunk_size tokens.
    assert len(b) == cs
    # The TARGET needle tokens occupy exactly [off, off+len).
    tt = _char_enc(" " + target.fact_sentence())
    assert b[off : off + len(tt)] == tt


def test_needle_tokens_warm_reuse_byte_identical_hash_collision():
    # The same prebuilt B spliced into warm + reuse prompts must stay
    # byte-identical (the hash-collision reuse signal) and remain exactly
    # chunk_size tokens after assembly, with the target still at its offset.
    cs = 256
    needles = make_needles(seed=4, k=3)
    target = needles[0]
    off = 179
    b = build_needle_passage_tokens(
        needles, target, encode=_char_enc, chunk_size=cs,
        needle_offset=off, filler_seed=11,
    )
    out = build_aligned_token_prompts(
        head_ids=_char_enc("Question head here. "),
        passage_ids=b,                       # prebuilt, exactly cs tokens
        tail_ids=_char_enc(" tail"),
        reuse_head_ids=_char_enc("Different opening sentence entirely. "),
        reuse_tail_ids=_char_enc(target.question()),
        chunk_size=cs,
        filler_ids=[11, 12, 13],
        passage_chunks=1,
    )
    assert out["b_len"] == cs
    warm_b = out["warm_ids"][out["warm_b_offset"] : out["warm_b_offset"] + cs]
    reuse_b = out["reuse_ids"][out["reuse_b_offset"] : out["reuse_b_offset"] + cs]
    # Byte-identical B in both prompts -> the content hash collides (reuse).
    assert warm_b == reuse_b == b
    # The recorded B-chunk hash equals the hash of the prebuilt B.
    assert out["expected_b_hashes"] == [hash_chunk_tokens(b)]
    # Target survived the splice at its offset in BOTH prompts.
    tt = _char_enc(" " + target.fact_sentence())
    assert warm_b[off : off + len(tt)] == tt
    assert reuse_b[off : off + len(tt)] == tt


def test_needle_in_link_classification_for_default_sweep():
    # With the default sweep [256, 64, 8] and a deep offset (>= 64), the target
    # is OUTSIDE the recomputed region for the non-full links and INSIDE only at
    # link == B-full (the control).
    cs = 256
    off = round(_DEFAULT_NEEDLE_OFFSET_FRAC * cs)  # ~179
    assert off >= 64
    eff = {link: min(link, cs) for link in _LINK_SWEEP}
    assert needle_in_link(off, eff[256]) is True    # full recompute: control
    assert needle_in_link(off, eff[64]) is False     # reused-KV proof row
    assert needle_in_link(off, eff[8]) is False      # reused-KV proof row


def test_needle_in_link_boundary_is_strict_less_than():
    # offset == eff_link is OUTSIDE the recomputed [0, eff_link) region.
    assert needle_in_link(64, 64) is False
    assert needle_in_link(63, 64) is True
    assert needle_in_link(0, 8) is True
    assert needle_in_link(8, 8) is False


def test_needle_tokens_overrun_raises():
    # offset + needle_len > chunk_size must error (a silent truncation would
    # drop the answer and invalidate the probe).
    cs = 64
    needles = make_needles(seed=4, k=3)
    target = needles[0]
    needle_len = len(_char_enc(" " + target.fact_sentence()))
    bad_off = cs - needle_len + 1  # one past the last fitting offset
    with pytest.raises(ValueError, match="overruns B"):
        build_needle_passage_tokens(
            needles, target, encode=_char_enc, chunk_size=cs,
            needle_offset=bad_off, filler_seed=11,
        )


def test_needle_tokens_default_offset_is_outside_largest_nonfull_link():
    # The default offset must be strictly past the largest NON-full link in the
    # sweep so the headline rows are reused-KV proofs, not controls.
    cs = DEFAULT_CHUNK_SIZE
    off = round(_DEFAULT_NEEDLE_OFFSET_FRAC * cs)
    nonfull = [link for link in _LINK_SWEEP if link < cs]
    assert nonfull, "sweep must contain a non-full link for the proof rows"
    assert off >= max(nonfull)


def test_parse_needle_offsets():
    assert _parse_needle_offsets(None) is None
    assert _parse_needle_offsets("") is None
    assert _parse_needle_offsets("4,128,240") == [4, 128, 240]
    assert _parse_needle_offsets(" 179 ") == [179]
    with pytest.raises(ValueError):
        _parse_needle_offsets("-1")
    with pytest.raises(ValueError):
        _parse_needle_offsets("abc")


# ---------------------------------------------------------------------------
# step4 config: debug_counters flag plumbing
# ---------------------------------------------------------------------------


def test_epic_kv_config_passes_debug_counters():
    cfg = _epic_kv_config(sparse=True, link_tokens=8, debug_counters=True)
    assert cfg["kv_connector_extra_config"]["epic_debug_counters"] is True


def test_epic_kv_config_omits_debug_counters_when_off():
    cfg = _epic_kv_config(sparse=True, link_tokens=8)
    assert "epic_debug_counters" not in cfg["kv_connector_extra_config"]


# ---------------------------------------------------------------------------
# EpicConnector in-band engagement counters (increment / reset)
# ---------------------------------------------------------------------------


def test_connector_counters_reset_and_increment():
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
        EpicConnector,
    )

    EpicConnector.reset_debug_counters()
    # All counters zeroed; the key set may grow (load-fidelity diagnostics) so
    # assert zeroed-ness + the load-bearing keys rather than an exact key set.
    assert all(v == 0 for v in EpicConnector.debug_counters.values())
    for k in ("sparse_match", "sparse_emit", "chunks_loaded"):
        assert k in EpicConnector.debug_counters
    EpicConnector._bump_counter("sparse_match")
    EpicConnector._bump_counter("sparse_match")
    EpicConnector._bump_counter("chunks_loaded", 3)
    assert EpicConnector.debug_counters["sparse_match"] == 2
    assert EpicConnector.debug_counters["chunks_loaded"] == 3
    assert EpicConnector.debug_counters["sparse_emit"] == 0
    # Reset zeroes everything again (so runs do not bleed into each other).
    EpicConnector.reset_debug_counters()
    assert all(v == 0 for v in EpicConnector.debug_counters.values())


def test_counters_shared_across_instances_same_class():
    # The counters are class-level so the SCHEDULER-role and WORKER-role
    # instances (same process, in-band smoke) bump the SAME dict.
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
        EpicConnector,
    )

    EpicConnector.reset_debug_counters()
    a = object.__new__(EpicConnector)
    b = object.__new__(EpicConnector)
    a._debug_counters = True
    b._debug_counters = True
    a._bump_counter("sparse_match")
    b._bump_counter("chunks_loaded")
    # Both reads see both bumps (shared class dict).
    assert EpicConnector.debug_counters["sparse_match"] == 1
    assert EpicConnector.debug_counters["chunks_loaded"] == 1
    EpicConnector.reset_debug_counters()


def test_load_chunk_bumps_chunks_loaded_only_when_enabled():
    # _load_chunk should bump chunks_loaded when _debug_counters is on, and stay
    # inert (no AttributeError, no bump) when the flag is absent/off -- the
    # partially-built (object.__new__) instance shape used by other unit tests.
    import torch

    from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
        StoredChunk,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
        EpicConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
        ChunkLoadSpec,
    )

    EpicConnector.reset_debug_counters()

    conn = object.__new__(EpicConnector)
    conn._layer_names = []          # no layers -> scatter loop is a no-op
    conn._kv_caches = {}
    conn._alignment = None          # not reached (no layers)
    conn._debug_check_load = False
    conn._check_load_done = True

    stored = StoredChunk(
        chunk_hash="h",
        length=4,
        old_positions=torch.arange(4, dtype=torch.int64),
    )
    spec = ChunkLoadSpec(
        chunk_hash="h",
        dst_slot_ids=[0, 1, 2, 3],
        old_pos_start=-1,
        new_pos_start=0,
        length=4,
    )

    # Flag absent -> no bump, no crash (getattr default False).
    conn._load_chunk(stored, spec)
    assert EpicConnector.debug_counters["chunks_loaded"] == 0

    # Flag on -> one bump per loaded chunk.
    conn._debug_counters = True
    conn._load_chunk(stored, spec)
    assert EpicConnector.debug_counters["chunks_loaded"] == 1

    # Zero-length chunk -> no bump (nothing scattered).
    EpicConnector.reset_debug_counters()
    empty = StoredChunk(chunk_hash="h", length=0,
                        old_positions=torch.zeros(0, dtype=torch.int64))
    spec0 = ChunkLoadSpec(chunk_hash="h", dst_slot_ids=[], old_pos_start=-1,
                          new_pos_start=0, length=0)
    conn._load_chunk(empty, spec0)
    assert EpicConnector.debug_counters["chunks_loaded"] == 0
    EpicConnector.reset_debug_counters()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
