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
    MusiqueSample,
    ModeAggregate,
    aggregate_mode,
    assemble_musique_prompt,
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
