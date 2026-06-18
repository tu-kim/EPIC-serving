# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the musique EPIC-reuse probe's PURE helpers.

The real LLM path needs a GPU (benchmarks/epic_reuse/musique_blend.py runs from
__main__ on CUDA). Everything around the engine -- data loading, token-level
chunk-aligned prompt assembly (byte-identity + hash-collision prediction +
padding accounting), answer normalisation/containment, the mode -> kv-config
mapping, and the aggregation/speedup math -- is pure and is verified here on
CPU. These tests must NOT import torch/vLLM at collection time.
"""

import json
import os

import pytest

from benchmarks.epic_reuse.common import effective_chunk_size, epic_chunk_hash
from benchmarks.epic_reuse.musique_blend import (
    SCENARIOS,
    MusiqueSample,
    ModeAggregate,
    aggregate_mode,
    assemble_musique_prompt,
    assemble_scenario_prompt,
    build_musique_worker_spec,
    fill_speedups,
    kv_config_for_mode,
    load_musique,
    mode_epic,
    mode_full,
    mode_reuse_only,
    parse_musique_records,
    parse_spec,
    serialize_spec,
    speedup,
)
from benchmarks.epic_reuse.common import answer_containment, normalize_answer


# ---------------------------------------------------------------------------
# data loading / parsing
# ---------------------------------------------------------------------------
def _rec(title="T", text="some passage text", q="who?", answers=("a",)):
    return {
        "ctxs": [{"title": title, "text": text}],
        "question": q,
        "answers": list(answers),
    }


def test_parse_musique_basic_structure():
    recs = parse_musique_records([_rec(text="hello world", answers=["x"])])
    assert len(recs) == 1
    s = recs[0]
    assert isinstance(s, MusiqueSample)
    assert s.question == "who?"
    assert s.answers == ["x"]
    # title present -> "title\ntext"
    assert s.ctxs == ["T\nhello world"]


def test_parse_musique_titleless_ctx_uses_text_only():
    recs = parse_musique_records([_rec(title="", text="bare text")])
    assert recs[0].ctxs == ["bare text"]


def test_parse_musique_skips_malformed_and_empty_ctx():
    recs = parse_musique_records([
        {"ctxs": [], "question": "q", "answers": ["a"]},          # no ctx
        {"question": "q", "answers": ["a"]},                       # no ctxs key
        {"ctxs": [{"title": "t", "text": ""}], "question": "q",    # empty text
         "answers": ["a"]},
        _rec(text="real"),                                          # good
        "not a dict",                                               # junk
    ])
    assert len(recs) == 1
    assert recs[0].ctxs == ["T\nreal"]


def test_parse_musique_all_bad_raises():
    with pytest.raises(ValueError):
        parse_musique_records([{"foo": 1}, "junk"])


def test_load_musique_missing_file_no_download_raises(tmp_path):
    p = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        load_musique(str(p), download=False)


def test_load_musique_reads_existing_file(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps([_rec(text="abc", answers=["z"])]))
    out = load_musique(str(p), download=False)
    assert len(out) == 1
    assert out[0].answers == ["z"]


def test_load_musique_real_sample_if_present():
    # If the real CacheBlend file is staged at /tmp, sanity-check its schema.
    path = "/tmp/musique_s.json"
    if not os.path.exists(path):
        pytest.skip("real musique_s.json not staged")
    out = load_musique(path, download=False)
    assert len(out) >= 1
    s = out[0]
    assert s.question and s.answers and s.ctxs


# ---------------------------------------------------------------------------
# prompt assembly: chunk alignment, byte-identity, hash-collision prediction
# ---------------------------------------------------------------------------
def _ids(*xs):
    return list(xs)


def test_assemble_pads_each_ctx_to_chunk_multiple():
    cs = 8
    # ctx0 = 5 tokens -> padded to 8; ctx1 = 8 tokens -> stays 8.
    a = assemble_musique_prompt(
        ctx_token_lists=[_ids(1, 2, 3, 4, 5), _ids(6, 7, 8, 9, 10, 11, 12, 13)],
        question_ids=_ids(99, 100),
        chunk_size=cs,
        filler_ids=[7, 8, 9],
    )
    assert a.ctx_token_lens == [8, 8]
    assert a.ctx_chunk_counts == [1, 1]
    # ctx offsets are chunk-aligned (multiples of chunk_size).
    assert a.ctx_offsets == [0, 8]
    assert all(off % cs == 0 for off in a.ctx_offsets)
    # prompt = 8 + 8 + 2 (question) = 18
    assert len(a.prompt_ids) == 18
    # padding accounting: 5+8=13 real, 8+8=16 padded -> 3 pad tokens.
    assert a.real_tokens == 13
    assert a.pad_tokens == 3


def test_assemble_padding_uses_cycled_filler_not_blank():
    a = assemble_musique_prompt(
        ctx_token_lists=[_ids(1, 2, 3)],
        question_ids=_ids(50),
        chunk_size=8,
        filler_ids=[101, 102, 103],
    )
    # 3 real + 5 pad. Pad ids must be the cycled filler pool (not a single id).
    pad = a.prompt_ids[3:8]
    assert pad == [101, 102, 103, 101, 102]


def test_assemble_warm_ctx_is_byte_identical_to_prompt_slice():
    cs = 8
    a = assemble_musique_prompt(
        ctx_token_lists=[_ids(1, 2, 3, 4, 5), _ids(20, 21, 22)],
        question_ids=_ids(77),
        chunk_size=cs,
        filler_ids=[5, 6],
    )
    for i, off in enumerate(a.ctx_offsets):
        ln = a.ctx_token_lens[i]
        in_prompt = a.prompt_ids[off:off + ln]
        # warm prompt has no lead -> it IS the padded ctx.
        assert a.warm_ctx_ids[i] == in_prompt, (
            "warm ctx must be byte-identical to the ctx slice in the reuse "
            "prompt (the EPIC reuse signal)"
        )


def test_assemble_hashes_predict_collision_with_connector_hash():
    cs = 8
    a = assemble_musique_prompt(
        ctx_token_lists=[_ids(1, 2, 3, 4, 5)],
        question_ids=_ids(9),
        chunk_size=cs,
        filler_ids=[0],
    )
    # The stored ctx_chunk_hashes must equal epic_chunk_hash of the actual chunk
    # ids in the assembled prompt (so the bench can predict connector collisions).
    chunk0 = a.prompt_ids[0:cs]
    assert a.ctx_chunk_hashes == [[epic_chunk_hash(chunk0)]]
    # And the warm ctx hashes to the same value -> guaranteed hit.
    assert epic_chunk_hash(a.warm_ctx_ids[0][0:cs]) == a.ctx_chunk_hashes[0][0]


def test_assemble_multichunk_ctx_counts_and_hashes():
    cs = 4
    # 9-token ctx -> padded to 12 -> 3 chunks.
    a = assemble_musique_prompt(
        ctx_token_lists=[_ids(1, 2, 3, 4, 5, 6, 7, 8, 9)],
        question_ids=_ids(0),
        chunk_size=cs,
        filler_ids=[1],
    )
    assert a.ctx_chunk_counts == [3]
    assert len(a.ctx_chunk_hashes[0]) == 3
    # each predicted hash matches the corresponding chunk in the prompt.
    for c in range(3):
        chunk = a.prompt_ids[c * cs:(c + 1) * cs]
        assert a.ctx_chunk_hashes[0][c] == epic_chunk_hash(chunk)


def test_assemble_zero_length_ctx_gets_full_chunk():
    a = assemble_musique_prompt(
        ctx_token_lists=[[]],
        question_ids=_ids(1),
        chunk_size=8,
        filler_ids=[3],
    )
    # empty ctx -> a full chunk of filler so we never emit a zero-length ctx.
    assert a.ctx_token_lens == [8]
    assert a.ctx_chunk_counts == [1]


def test_assemble_bad_chunk_size_raises():
    with pytest.raises(ValueError):
        assemble_musique_prompt(
            ctx_token_lists=[[1]], question_ids=[2], chunk_size=0,
            filler_ids=[0],
        )


def test_two_samples_same_ctx_same_hash_diff_ctx_diff_hash():
    cs = 8
    common_ctx = _ids(1, 2, 3, 4, 5, 6, 7, 8)
    a1 = assemble_musique_prompt(
        ctx_token_lists=[common_ctx], question_ids=_ids(11),
        chunk_size=cs, filler_ids=[0],
    )
    a2 = assemble_musique_prompt(
        ctx_token_lists=[common_ctx], question_ids=_ids(99),  # diff question
        chunk_size=cs, filler_ids=[0],
    )
    # Same ctx ids -> identical chunk hash (reuse across samples is harmless).
    assert a1.ctx_chunk_hashes == a2.ctx_chunk_hashes
    a3 = assemble_musique_prompt(
        ctx_token_lists=[_ids(9, 9, 9, 9, 9, 9, 9, 9)], question_ids=_ids(11),
        chunk_size=cs, filler_ids=[0],
    )
    assert a3.ctx_chunk_hashes != a1.ctx_chunk_hashes


# ---------------------------------------------------------------------------
# system segment (LegoLink-engagement fix): prepended NON-warmed instruction
# chunk breaks position-0 contiguity so EVERY ctx becomes a non-prefix hit.
# ---------------------------------------------------------------------------
def _mirror_select(chunks, warm_hashes):
    """Pure mirror of EpicSelection.select's prefix/non-prefix walk (no torch).

    ``chunks`` = [(start, length, hash), ...]; ``warm_hashes`` = set of hashes
    present in the (warm-seeded) store. Returns (prefix_extent, non_prefix_count,
    non_prefix_offsets) -- the same classification the connector performs.
    """
    prefix_extent = 0
    contiguous = True
    non_prefix = []
    for start, length, h in chunks:
        hit = h in warm_hashes
        if contiguous and start == prefix_extent and hit:
            prefix_extent += length
        else:
            contiguous = False
            if hit:
                non_prefix.append(start)
    return prefix_extent, len(non_prefix), non_prefix


def _chunks_of(a):
    """[(start, length, hash)] for every whole ctx chunk in an AssembledPrompt
    (sys-padded leading region produces extra leading chunks that are NOT in the
    ctx hash list -> they are non-hits unless warmed)."""
    chunks = []
    # leading sys region: one (start, len, hash) per chunk, hash = sentinel not in
    # any warm set (the sys segment is never warmed).
    cs = a.chunk_size
    for i in range(a.sys_len // cs):
        chunks.append((i * cs, cs, f"__sys__{i}"))
    for off, ln, hashes in zip(a.ctx_offsets, a.ctx_token_lens, a.ctx_chunk_hashes):
        for c, h in enumerate(hashes):
            chunks.append((off + c * cs, cs, h))
    return chunks


def test_sys_segment_prepended_and_padded_to_chunk_multiple():
    cs = 8
    a = assemble_musique_prompt(
        ctx_token_lists=[_ids(1, 2, 3, 4, 5), _ids(6, 7, 8)],
        question_ids=_ids(99),
        chunk_size=cs,
        filler_ids=[200, 201],
        sys_ids=_ids(50, 51, 52),  # 3 sys tokens -> padded to 8
    )
    assert a.sys_len == cs  # 3 -> padded up to one whole chunk
    # ctx offsets are shifted by sys_len yet still chunk-aligned.
    assert a.ctx_offsets == [cs, 2 * cs]
    assert all(off % cs == 0 for off in a.ctx_offsets)
    # prompt = sys(8) + ctx0(8) + ctx1(8) + question(1) = 25
    assert len(a.prompt_ids) == 25
    # sys ids are the literal prefix (then filler pad) -- NOT in any warm prompt.
    assert a.prompt_ids[:3] == [50, 51, 52]


def test_sys_segment_makes_every_ctx_non_prefix():
    cs = 8
    a = assemble_musique_prompt(
        ctx_token_lists=[_ids(1, 2, 3, 4, 5, 6, 7, 8),
                         _ids(9, 10, 11, 12, 13, 14, 15, 16),
                         _ids(17, 18, 19)],
        question_ids=_ids(99),
        chunk_size=cs,
        filler_ids=[0],
        sys_ids=_ids(50, 51),  # leading non-warmed sys -> one chunk
    )
    # WARM each ctx (its chunk hashes are in the store); sys is NEVER warmed.
    warm = {h for hs in a.ctx_chunk_hashes for h in hs}
    prefix_extent, n_non_prefix, offsets = _mirror_select(_chunks_of(a), warm)
    # sys chunk at pos 0 is a non-hit -> contiguity broken at chunk 0 -> NOTHING
    # is in the contiguous prefix and every ctx chunk is a non-prefix hit.
    assert prefix_extent == 0
    n_ctx_chunks = sum(a.ctx_chunk_counts)
    assert n_non_prefix == n_ctx_chunks
    # offsets are the ctx chunk starts (all >= sys_len, never 0).
    assert all(o >= a.sys_len for o in offsets)


def test_no_sys_segment_reverts_to_all_prefix_control():
    cs = 8
    a = assemble_musique_prompt(
        ctx_token_lists=[_ids(1, 2, 3, 4, 5, 6, 7, 8),
                         _ids(9, 10, 11, 12, 13, 14, 15, 16)],
        question_ids=_ids(99),
        chunk_size=cs,
        filler_ids=[0],
        sys_ids=None,  # --no-system control
    )
    assert a.sys_len == 0
    assert a.ctx_offsets == [0, cs]  # ctx0 at position 0
    warm = {h for hs in a.ctx_chunk_hashes for h in hs}
    prefix_extent, n_non_prefix, _ = _mirror_select(_chunks_of(a), warm)
    # No sys -> ctx0 hits at pos 0, ctx1 hits at pos 8 contiguously -> ALL fold
    # into the contiguous prefix -> ZERO non-prefix hits -> LegoLink INERT.
    assert prefix_extent == 2 * cs
    assert n_non_prefix == 0


def test_sys_segment_keeps_ctx_byte_identical_and_hashes_stable():
    cs = 8
    ctxs = [_ids(1, 2, 3, 4, 5), _ids(20, 21, 22)]
    q = _ids(77)
    with_sys = assemble_musique_prompt(
        ctx_token_lists=ctxs, question_ids=q, chunk_size=cs,
        filler_ids=[5, 6], sys_ids=_ids(50, 51, 52),
    )
    no_sys = assemble_musique_prompt(
        ctx_token_lists=ctxs, question_ids=q, chunk_size=cs,
        filler_ids=[5, 6], sys_ids=None,
    )
    # ctx chunk hashes depend only on the (padded) ctx ids, not absolute offset,
    # so adding the sys segment must NOT change them -> warm-side hashes still
    # collide with reuse-side hashes.
    assert with_sys.ctx_chunk_hashes == no_sys.ctx_chunk_hashes
    # warm ctx prefills are identical too (sys is excluded from warming).
    assert with_sys.warm_ctx_ids == no_sys.warm_ctx_ids
    # and each warm ctx is byte-identical to its slice in the sys-shifted prompt.
    for i, off in enumerate(with_sys.ctx_offsets):
        ln = with_sys.ctx_token_lens[i]
        assert with_sys.warm_ctx_ids[i] == with_sys.prompt_ids[off:off + ln]


def test_sys_nonce_makes_first_sys_chunk_hash_differ_per_sample():
    # (a) Two samples that share the SAME sys text but get a different per-sample
    # nonce must have a DIFFERENT first sys chunk hash -> the chunk an earlier
    # sample's reuse request saved can never be a hit for a later sample. This is
    # the residual-bug fix (connector saves the sys chunk on the first reuse).
    cs = 8
    sys_ids = _ids(50, 51)
    a0 = assemble_musique_prompt(
        ctx_token_lists=[_ids(1, 2, 3, 4)], question_ids=_ids(9),
        chunk_size=cs, filler_ids=[0],
        sys_ids=sys_ids, sys_nonce_ids=_ids(900),  # sample 0 nonce
    )
    a1 = assemble_musique_prompt(
        ctx_token_lists=[_ids(1, 2, 3, 4)], question_ids=_ids(9),
        chunk_size=cs, filler_ids=[0],
        sys_ids=sys_ids, sys_nonce_ids=_ids(901),  # sample 1 nonce
    )
    h0 = epic_chunk_hash(a0.prompt_ids[0:cs])
    h1 = epic_chunk_hash(a1.prompt_ids[0:cs])
    assert h0 != h1, "per-sample sys nonce must change the first sys chunk hash"
    # the nonce token is literally in the first chunk (so it is hashed).
    assert a0.prompt_ids[0] == 900
    assert a1.prompt_ids[0] == 901


def test_sys_nonce_same_sample_full_and_reuse_share_sys():
    # (b) The SAME sample (same nonce) must produce the SAME prompt_ids -- the
    # bench uses one AssembledPrompt.prompt_ids for both full and reuse, so they
    # are trivially identical; this asserts the nonce is deterministic per nonce
    # value (no hidden randomness) so re-assembling sample i in another mode's
    # subprocess yields the byte-identical sys.
    cs = 8
    kw = dict(
        ctx_token_lists=[_ids(1, 2, 3, 4)], question_ids=_ids(9),
        chunk_size=cs, filler_ids=[0], sys_ids=_ids(50, 51),
        sys_nonce_ids=_ids(900),
    )
    a = assemble_musique_prompt(**kw)
    b = assemble_musique_prompt(**kw)
    assert a.prompt_ids == b.prompt_ids
    assert a.sys_len == b.sys_len


def test_sys_nonce_does_not_change_ctx_hashes_or_alignment():
    # (c) The nonce is confined to the sys region: ctx chunk hashes, ctx
    # byte-identity vs. warm prefills, and chunk alignment must be invariant to
    # the nonce (only the sys offset shifts, but the sys stays a chunk multiple).
    cs = 8
    ctxs = [_ids(1, 2, 3, 4, 5), _ids(20, 21, 22)]
    q = _ids(77)
    base = assemble_musique_prompt(
        ctx_token_lists=ctxs, question_ids=q, chunk_size=cs,
        filler_ids=[5, 6], sys_ids=_ids(50, 51, 52), sys_nonce_ids=None,
    )
    nonced = assemble_musique_prompt(
        ctx_token_lists=ctxs, question_ids=q, chunk_size=cs,
        filler_ids=[5, 6], sys_ids=_ids(50, 51, 52), sys_nonce_ids=_ids(900, 901),
    )
    # ctx hashes + warm prefills unchanged by the nonce.
    assert base.ctx_chunk_hashes == nonced.ctx_chunk_hashes
    assert base.warm_ctx_ids == nonced.warm_ctx_ids
    # offsets still chunk-aligned in both (sys length may differ but stays a
    # chunk multiple, so ctx0 still lands on a boundary).
    assert all(off % cs == 0 for off in nonced.ctx_offsets)
    assert nonced.sys_len % cs == 0
    # each warm ctx is still byte-identical to its slice in the nonced prompt.
    for i, off in enumerate(nonced.ctx_offsets):
        ln = nonced.ctx_token_lens[i]
        assert nonced.warm_ctx_ids[i] == nonced.prompt_ids[off:off + ln]


def test_sys_nonce_all_ctx_non_prefix_regardless_of_sample_index():
    # (d) Mirror-select check: with a (nonced) sys chunk at position 0 that is
    # NOT in the warm/saved set, EVERY ctx is a non-prefix hit, independent of the
    # sample index. We also model the residual bug it fixes: a PRIOR sample saved
    # its OWN nonced sys chunk into the store; because this sample's nonce differs
    # the prior sys chunk is NOT a hit here, so position 0 stays a non-hit.
    cs = 8
    ctxs = [_ids(1, 2, 3, 4, 5, 6, 7, 8), _ids(9, 10, 11, 12, 13, 14, 15, 16)]
    for idx in (0, 1, 5):
        a = assemble_musique_prompt(
            ctx_token_lists=ctxs, question_ids=_ids(99), chunk_size=cs,
            filler_ids=[0], sys_ids=_ids(50, 51),
            sys_nonce_ids=_ids(900 + idx),
        )
        # store holds: every ctx chunk (warmed) PLUS every PRIOR sample's nonced
        # sys chunk (saved by their reuse request). This sample's sys nonce is
        # unique so its own sys chunk is NOT in the store -> non-hit at pos 0.
        warm = {h for hs in a.ctx_chunk_hashes for h in hs}
        for prior in range(idx):
            prior_a = assemble_musique_prompt(
                ctx_token_lists=ctxs, question_ids=_ids(99), chunk_size=cs,
                filler_ids=[0], sys_ids=_ids(50, 51),
                sys_nonce_ids=_ids(900 + prior),
            )
            warm.add(epic_chunk_hash(prior_a.prompt_ids[0:cs]))
        chunks = _chunks_of_with_real_sys_hash(a)
        prefix_extent, n_non_prefix, offsets = _mirror_select(chunks, warm)
        assert prefix_extent == 0, f"sample {idx}: sys must not be a prefix hit"
        assert n_non_prefix == sum(a.ctx_chunk_counts)
        assert all(o >= a.sys_len for o in offsets)


def _chunks_of_with_real_sys_hash(a):
    """Like ``_chunks_of`` but uses the REAL content hash for the leading sys
    chunks (so a store that contains a saved sys chunk can match it). This models
    the residual bug: the connector saves the sys chunk on the first reuse, so a
    later sample could see it as a hit -- unless the nonce makes the hash unique.
    """
    chunks = []
    cs = a.chunk_size
    for i in range(a.sys_len // cs):
        chunks.append(
            (i * cs, cs, epic_chunk_hash(a.prompt_ids[i * cs:(i + 1) * cs]))
        )
    for off, ln, hashes in zip(a.ctx_offsets, a.ctx_token_lens, a.ctx_chunk_hashes):
        for c, h in enumerate(hashes):
            chunks.append((off + c * cs, cs, h))
    return chunks


def test_full_and_reuse_prompts_both_include_sys_identically():
    # The bench uses the SAME AssembledPrompt.prompt_ids for full and reuse, so
    # both modes share the leading sys + identical ctx bytes (fair comparison).
    cs = 8
    a = assemble_musique_prompt(
        ctx_token_lists=[_ids(1, 2, 3, 4)], question_ids=_ids(9),
        chunk_size=cs, filler_ids=[0], sys_ids=_ids(50, 51),
    )
    # prompt begins with the sys ids; warm prompts (reuse warm side) do NOT.
    assert a.prompt_ids[:2] == [50, 51]
    assert all(w[:2] != [50, 51] for w in a.warm_ctx_ids)


# ---------------------------------------------------------------------------
# answer scoring (reused from common) -- spot check via the bench's import
# ---------------------------------------------------------------------------
def test_answer_normalisation_and_containment():
    assert normalize_answer("Exeter College.") == "exeter college"
    assert answer_containment(
        "The author studied at Exeter College, Oxford.", ["Exeter College"]
    ) is True
    assert answer_containment("studied at Oxford", ["Exeter College"]) is False
    assert answer_containment("", ["x"]) is False


# ---------------------------------------------------------------------------
# mode -> kv-config mapping
# ---------------------------------------------------------------------------
def test_mode_full_has_no_connector():
    assert kv_config_for_mode(mode_full(), chunk_size=256) is None


def test_mode_reuse_only_is_epic_link0():
    m = mode_reuse_only()
    assert m.label == "reuse-only"
    assert m.sparse is True
    assert m.link == 0
    cfg = kv_config_for_mode(m, chunk_size=256)
    extra = cfg["kv_connector_extra_config"]
    assert cfg["kv_connector"] == "EpicConnector"
    assert extra["epic_link_tokens"] == 0
    assert extra["epic_sparse_forward"] is True
    assert extra["epic_fusion_mask"] is True
    assert extra["epic_debug_counters"] is True
    assert extra["epic_chunk_size"] == 256


def test_mode_epic_k_sets_link():
    cfg = kv_config_for_mode(mode_epic(8), chunk_size=512)
    assert cfg["kv_connector_extra_config"]["epic_link_tokens"] == 8
    assert cfg["kv_connector_extra_config"]["epic_chunk_size"] == 512


def test_mode_epic_negative_link_raises():
    with pytest.raises(ValueError):
        mode_epic(-1)


# ---------------------------------------------------------------------------
# worker spec serialise/parse round-trip
# ---------------------------------------------------------------------------
def test_worker_spec_roundtrip():
    spec = build_musique_worker_spec(
        mode=mode_epic(8),
        model="some/model",
        chunk_size=256,
        samples=[{"warm": [[1, 2, 3, 4], [5, 6, 7, 8]],
                  "prompt": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                  "answers": ["x"]}],
        max_tokens=32,
        block_size=16,
        max_model_len=8192,
        gpu_memory_utilization=0.45,
    )
    back = parse_spec(serialize_spec(spec))
    assert back == spec
    assert back["mode_label"] == "epic@8"
    assert back["sparse"] is True
    assert back["in_process"] is True
    assert back["read_counters"] is True
    assert back["block_size"] == 16
    assert back["samples"][0]["warm"] == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_worker_spec_full_mode_no_connector_no_inprocess():
    spec = build_musique_worker_spec(
        mode=mode_full(),
        model="m",
        chunk_size=256,
        samples=[{"warm": [], "prompt": [1, 2], "answers": ["a"]}],
        max_tokens=8,
        block_size=16,
        max_model_len=2048,
        gpu_memory_utilization=0.4,
    )
    assert spec["kv_config"] is None
    assert spec["sparse"] is False
    assert spec["in_process"] is False
    assert spec["read_counters"] is False


def test_serialize_spec_single_line():
    spec = build_musique_worker_spec(
        mode=mode_full(), model="m", chunk_size=256,
        samples=[{"warm": [], "prompt": [1], "answers": ["a"]}],
        max_tokens=4, block_size=16, max_model_len=2048,
        gpu_memory_utilization=0.4,
    )
    assert "\n" not in serialize_spec(spec)


# ---------------------------------------------------------------------------
# aggregation + speedup math
# ---------------------------------------------------------------------------
def test_speedup_basic_and_guards():
    assert speedup(100.0, 50.0) == pytest.approx(2.0)
    assert speedup(50.0, 100.0) == pytest.approx(0.5)
    assert speedup(100.0, 0.0) == float("inf")   # mode infinitely fast
    assert speedup(0.0, 50.0) == 0.0             # undefined baseline


def test_aggregate_mode_hit_rate_and_mean():
    rows = [
        {"answer_hit": True, "prefill_ms": 100.0},
        {"answer_hit": False, "prefill_ms": 200.0},
        {"answer_hit": True, "prefill_ms": 300.0},
    ]
    agg = aggregate_mode("epic@8", rows, warmup_discard=False)
    assert agg.n == 3
    assert agg.answer_hits == 2
    assert agg.answer_hit_rate == pytest.approx(2 / 3)
    assert agg.mean_prefill_ms == pytest.approx(200.0)


def test_aggregate_mode_warmup_discard_drops_first():
    rows = [
        {"answer_hit": False, "prefill_ms": 999.0},  # cold warmup (dropped)
        {"answer_hit": True, "prefill_ms": 100.0},
        {"answer_hit": True, "prefill_ms": 200.0},
    ]
    agg = aggregate_mode("full", rows, warmup_discard=True)
    assert agg.n == 2
    assert agg.answer_hits == 2
    assert agg.mean_prefill_ms == pytest.approx(150.0)


def test_aggregate_mode_warmup_discard_keeps_single_row():
    rows = [{"answer_hit": True, "prefill_ms": 100.0}]
    agg = aggregate_mode("full", rows, warmup_discard=True)
    # never drop the only row.
    assert agg.n == 1
    assert agg.mean_prefill_ms == pytest.approx(100.0)


def test_fill_speedups_uses_full_as_baseline():
    aggs = [
        ModeAggregate("full", n=3, answer_hits=3, mean_prefill_ms=300.0),
        ModeAggregate("reuse-only", n=3, answer_hits=2, mean_prefill_ms=100.0),
        ModeAggregate("epic@8", n=3, answer_hits=3, mean_prefill_ms=150.0),
    ]
    fill_speedups(aggs)
    by = {a.label: a for a in aggs}
    assert by["full"].speedup_vs_full == pytest.approx(1.0)
    assert by["reuse-only"].speedup_vs_full == pytest.approx(3.0)
    assert by["epic@8"].speedup_vs_full == pytest.approx(2.0)


def test_fill_speedups_no_full_baseline_is_zero():
    aggs = [ModeAggregate("reuse-only", n=1, answer_hits=1, mean_prefill_ms=10.0)]
    fill_speedups(aggs)
    assert aggs[0].speedup_vs_full == 0.0


# ---------------------------------------------------------------------------
# chunk-size block alignment mirror (connector rounds up to block_size)
# ---------------------------------------------------------------------------
def test_effective_chunk_size_rounds_to_block_multiple():
    # 250 -> 256 (next multiple of 16); 256 stays.
    assert effective_chunk_size(250, 16) == 256
    assert effective_chunk_size(256, 16) == 256
    assert effective_chunk_size(7, 16) == 16


# ---------------------------------------------------------------------------
# realistic RAG reuse scenarios (reorder / insert / same): warm [sys][a][b][c]
# full prefill then a scenario-specific measured prompt. These verify the
# prompt structure, byte-identity of warmed ctxs, held-out d, sys-at-front,
# and the prefix/non-prefix split (PIC-only reorder vs sparse insert).
# ---------------------------------------------------------------------------
CS = 8  # chunk size for the scenario tests (each ctx -> whole chunks)
_SYS = _ids(50, 51, 52)         # leading RAG system text (-> 1 padded chunk)
_A = _ids(1, 2, 3, 4, 5, 6, 7, 8)
_B = _ids(11, 12, 13, 14, 15, 16, 17, 18)
_C = _ids(21, 22, 23, 24, 25, 26, 27, 28)
_D = _ids(31, 32, 33, 34, 35, 36, 37, 38)
_Q = _ids(99)


def _scn(scenario, *, held_out=None, nonce=None):
    return assemble_scenario_prompt(
        scenario=scenario,
        ctx_token_lists=[list(_A), list(_B), list(_C)],
        held_out_ctx_ids=held_out,
        question_ids=list(_Q),
        chunk_size=CS,
        filler_ids=[0],
        sys_ids=list(_SYS),
        sys_nonce_ids=nonce,
    )


def _slice(prompt_ids, offset, length):
    return prompt_ids[offset:offset + length]


def test_scenarios_constant_lists_three():
    assert SCENARIOS == ("same", "reorder", "insert")


def test_scenario_warm_prompt_is_sys_a_b_c():
    # The warm prompt is [sys][a][b][c]: the leading sys (nonce-free) followed by
    # the three padded ctxs in order. It seeds sys,a,b,c into the store.
    p = _scn("reorder")
    assert p.warm_ids[:3] == [50, 51, 52]               # sys text at front
    assert p.warm_ids[CS:2 * CS] == list(_A)            # a
    assert p.warm_ids[2 * CS:3 * CS] == list(_B)        # b
    assert p.warm_ids[3 * CS:4 * CS] == list(_C)        # c
    assert len(p.warm_ids) == 4 * CS                    # sys+a+b+c, no query


def test_scenario_system_prompt_first_in_measured_prompt():
    for scenario in SCENARIOS:
        held = list(_D) if scenario == "insert" else None
        nonce = _ids(900) if scenario == "same" else None
        p = _scn(scenario, held_out=held, nonce=nonce)
        assert p.seg_labels[0] == "sys"
        assert p.seg_offsets[0] == 0
        # nonce-free scenarios start with the literal sys text; 'same' starts
        # with its nonce token (still a sys segment at the front).
        if scenario == "same":
            assert p.prompt_ids[0] == 900
        else:
            assert p.prompt_ids[:3] == [50, 51, 52]


def test_scenario_reorder_layout_is_sys_b_a_c_query():
    p = _scn("reorder")
    assert p.seg_labels == ["sys", "b", "a", "c", "query"]
    # b sits where a was warmed -> different position -> non-zero PIC delta.
    assert _slice(p.prompt_ids, p.seg_offsets[1], CS) == list(_B)  # b first now
    assert _slice(p.prompt_ids, p.seg_offsets[2], CS) == list(_A)  # a second
    assert _slice(p.prompt_ids, p.seg_offsets[3], CS) == list(_C)  # c third
    assert p.held_out_label is None


def test_scenario_reorder_is_pure_prefix_no_sparse():
    # sys is warmed (nonce-free, HIT) and b,a,c are all hits contiguous from the
    # sys boundary -> EVERYTHING folds into the prefix -> ZERO non-prefix hits ->
    # the sparse branch is NOT expected. This is the PIC-only test.
    p = _scn("reorder")
    assert p.expect_non_prefix_offsets == []
    assert p.expect_sparse is False
    # the whole measured prompt (minus the trailing query) is the prefix extent.
    assert p.expect_prefix_extent == p.sys_len + 3 * CS


def test_scenario_reorder_ctxs_byte_identical_to_warm():
    # a,b,c are byte-identical between warm and measured (only their POSITIONS
    # change) -> their content hashes collide -> the warmed chunks are reused.
    p = _scn("reorder")
    warm = p.warm_ids
    a_warm = warm[CS:2 * CS]
    b_warm = warm[2 * CS:3 * CS]
    c_warm = warm[3 * CS:4 * CS]
    assert _slice(p.prompt_ids, p.seg_offsets[1], CS) == b_warm
    assert _slice(p.prompt_ids, p.seg_offsets[2], CS) == a_warm
    assert _slice(p.prompt_ids, p.seg_offsets[3], CS) == c_warm


def test_scenario_insert_layout_is_sys_d_a_c_query():
    p = _scn("insert", held_out=list(_D))
    assert p.seg_labels == ["sys", "d", "a", "c", "query"]
    assert p.held_out_label == "d"
    assert _slice(p.prompt_ids, p.seg_offsets[1], CS) == list(_D)  # d spliced in
    assert _slice(p.prompt_ids, p.seg_offsets[2], CS) == list(_A)
    assert _slice(p.prompt_ids, p.seg_offsets[3], CS) == list(_C)


def test_scenario_insert_d_is_held_out_never_warmed():
    # d's chunk hash must NOT be in the warm-stored set (it is held out), while
    # a and c hashes ARE stored.
    p = _scn("insert", held_out=list(_D))
    d_hash = epic_chunk_hash(_slice(p.prompt_ids, p.seg_offsets[1], CS))
    a_hash = epic_chunk_hash(_slice(p.prompt_ids, p.seg_offsets[2], CS))
    c_hash = epic_chunk_hash(_slice(p.prompt_ids, p.seg_offsets[3], CS))
    assert d_hash not in p.warm_stored_hashes
    assert a_hash in p.warm_stored_hashes
    assert c_hash in p.warm_stored_hashes
    # d is also not in the warm prompt at all.
    assert list(_D) not in [p.warm_ids[i:i + CS] for i in range(0, len(p.warm_ids), CS)]


def test_scenario_insert_breaks_contiguity_a_c_non_prefix():
    # sys HITS (warmed, nonce-free) -> prefix_extent == sys_len. d at the next
    # slot is a NON-hit -> contiguity breaks -> a and c (both warmed hits) become
    # NON-PREFIX hits -> the sparse-forward path is expected to engage.
    p = _scn("insert", held_out=list(_D))
    assert p.expect_prefix_extent == p.sys_len  # only sys folds in
    # a and c offsets are the non-prefix hits (d is a non-hit so not listed).
    assert p.expect_non_prefix_offsets == [p.seg_offsets[2], p.seg_offsets[3]]
    assert p.expect_sparse is True


def test_scenario_same_nonce_makes_sys_non_hit_all_ctx_non_prefix():
    # 'same' keeps the per-sample nonce -> the measured sys chunk hash differs
    # from the warmed sys -> sys is a NON-hit at position 0 -> a,b,c (all warmed)
    # become non-prefix hits at their ORIGINAL positions (PIC delta zero).
    p = _scn("same", nonce=_ids(900))
    assert p.seg_labels == ["sys", "a", "b", "c", "query"]
    # the nonced sys chunk is NOT in the warm-stored set.
    sys_hash = epic_chunk_hash(p.prompt_ids[0:CS])
    assert sys_hash not in p.warm_stored_hashes
    assert p.expect_prefix_extent == 0
    assert p.expect_non_prefix_offsets == [
        p.seg_offsets[1], p.seg_offsets[2], p.seg_offsets[3]
    ]
    assert p.expect_sparse is True


def test_scenario_reorder_sys_is_warm_hit():
    # In reorder the measured sys is nonce-free, so its chunk hash IS in the
    # warm-stored set -> sys is a prefix HIT (the precondition for b,a,c folding
    # into the PIC-rotated prefix).
    p = _scn("reorder")
    sys_hash = epic_chunk_hash(p.prompt_ids[0:CS])
    assert sys_hash in p.warm_stored_hashes


def test_scenario_insert_requires_held_out_ctx():
    with pytest.raises(ValueError):
        assemble_scenario_prompt(
            scenario="insert",
            ctx_token_lists=[list(_A), list(_B), list(_C)],
            held_out_ctx_ids=None,
            question_ids=list(_Q),
            chunk_size=CS,
            filler_ids=[0],
            sys_ids=list(_SYS),
        )


def test_scenario_needs_exactly_three_ctxs():
    with pytest.raises(ValueError):
        assemble_scenario_prompt(
            scenario="reorder",
            ctx_token_lists=[list(_A), list(_B)],  # only 2
            held_out_ctx_ids=None,
            question_ids=list(_Q),
            chunk_size=CS,
            filler_ids=[0],
            sys_ids=list(_SYS),
        )


def test_scenario_unknown_raises():
    with pytest.raises(ValueError):
        _scn("shuffle")


def test_scenario_measured_ends_with_query():
    for scenario in SCENARIOS:
        held = list(_D) if scenario == "insert" else None
        nonce = _ids(900) if scenario == "same" else None
        p = _scn(scenario, held_out=held, nonce=nonce)
        assert p.seg_labels[-1] == "query"
        assert p.prompt_ids[-1] == _Q[-1]
        # query is a trailing partial chunk (not hashed).
        assert p.seg_chunk_hashes[-1] == []
