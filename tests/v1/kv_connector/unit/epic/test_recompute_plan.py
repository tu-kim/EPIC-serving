# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC LegoLink ``plan_recompute`` (Phase 2b, S1) M-derivation tests.

CPU-only, pure logic. Covers A+C+B layouts (B one/two, empty C, last token in
B-tail / C-tail, link > chunk-length boundary), the flag-off dense path, and the
RecomputePlan invariants (sorted, unique, in [0,N), last == N-1, prefix A
excluded).
"""

from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
    NonPrefixHit,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    LegoLinkRecompute,
    RecomputePlan,
    ReuseSelection,
)


class _PromptLenShim:
    __slots__ = ("prompt_token_ids",)

    def __init__(self, n: int):
        self.prompt_token_ids = list(range(n))


def _plan(prefix_extent, hits, n, link=8):
    pol = LegoLinkRecompute(num_link_tokens=link, phase1_dense=False)
    sel = ReuseSelection(prefix_extent=prefix_extent, non_prefix_hits=list(hits))
    return pol.plan_recompute(_PromptLenShim(n), sel, block_size=16)


def _assert_invariants(plan: RecomputePlan, n: int, prefix_extent: int):
    off = plan.recompute_offsets
    assert off == sorted(off), "M must be sorted ascending"
    assert len(off) == len(set(off)), "M must be unique"
    assert all(0 <= o < n for o in off), "M offsets must be in [0, N)"
    assert off[-1] == n - 1, "last M offset must be N-1"
    assert all(o >= prefix_extent for o in off), "prefix A must not be in M"
    # target positions == logical positions.
    assert plan.target_positions == off
    assert plan.seq_len == n


# --------------------------------------------------------------------------
# Layout coverage
# --------------------------------------------------------------------------


def test_single_b_chunk():
    # A=[0,256) C=[256,512) B=[512,768) C-tail=[768,800), N=800, link=8.
    n = 800
    plan = _plan(256, [NonPrefixHit("h", 512, 0, 256)], n)
    _assert_invariants(plan, n, 256)
    m = set(plan.recompute_offsets)
    # C body present.
    assert {256, 300, 511} <= m
    # link window of B: 512..519 present, 520 absent.
    assert set(range(512, 520)) <= m
    assert 520 not in m
    # B body (not link, not C) absent.
    assert 600 not in m and 767 not in m
    # C-tail after B present.
    assert {768, 799} <= m


def test_two_b_chunks():
    # A=[0,128) B1=[128,256) C=[256,384) B2=[384,512), N=520, link=4.
    n = 520
    hits = [NonPrefixHit("b1", 128, 0, 128), NonPrefixHit("b2", 384, 0, 128)]
    plan = _plan(128, hits, n, link=4)
    _assert_invariants(plan, n, 128)
    m = set(plan.recompute_offsets)
    # B1 link 128..131, B2 link 384..387.
    assert set(range(128, 132)) <= m
    assert set(range(384, 388)) <= m
    assert 132 not in m and 388 not in m
    # C body [256,384).
    assert {256, 383} <= m
    # B bodies absent.
    assert 200 not in m and 450 not in m
    # C tail [512,520) and last token.
    assert {512, 519} <= m


def test_empty_c_last_token_in_b_tail():
    # A=[0,256) B=[256,512), N=512, no C. last token 511 is the B tail.
    n = 512
    plan = _plan(256, [NonPrefixHit("h", 256, 0, 256)], n)
    _assert_invariants(plan, n, 256)
    m = set(plan.recompute_offsets)
    # link of B: 256..263.
    assert set(range(256, 264)) <= m
    # last token (B tail) must be in M even though it is mid-chunk reuse.
    assert 511 in m
    # B body between link and last is NOT recomputed.
    assert 300 not in m and 510 not in m


def test_last_token_in_c_tail():
    # A=[0,128) B=[128,256) C=[256,300), N=300. last token 299 is C tail.
    n = 300
    plan = _plan(128, [NonPrefixHit("h", 128, 0, 128)], n)
    _assert_invariants(plan, n, 128)
    m = set(plan.recompute_offsets)
    assert 299 in m
    # last token already in C; not double-counted (uniqueness asserted above).
    assert plan.recompute_offsets.count(299) == 1


def test_link_exceeds_chunk_length():
    # link=64 > chunk length 32. Whole B chunk becomes link tokens (clamped).
    # A=[0,32) B=[32,64) C=[64,100), N=100.
    n = 100
    plan = _plan(32, [NonPrefixHit("h", 32, 0, 32)], n, link=64)
    _assert_invariants(plan, n, 32)
    m = set(plan.recompute_offsets)
    # entire B [32,64) is link (clamped to chunk length), no out-of-chunk spill.
    assert set(range(32, 64)) <= m
    assert 64 in m  # first C token, not a B-link spill
    # C body + last.
    assert {64, 99} <= m


def test_no_b_pure_c():
    # A=[0,256) then all new C=[256,400), N=400. No non-prefix chunk.
    n = 400
    plan = _plan(256, [], n)
    _assert_invariants(plan, n, 256)
    m = set(plan.recompute_offsets)
    # C is fully recomputed.
    assert set(range(256, 400)) == m


def test_no_reuse_at_all():
    # prefix_extent 0, no hits -> every token is C.
    n = 50
    plan = _plan(0, [], n)
    _assert_invariants(plan, n, 0)
    assert plan.recompute_offsets == list(range(50))


def test_link_zero_only_c_and_last():
    # link=0: M = C only (+ last token). B contributes no link tokens.
    # A=[0,128) B=[128,256) C=[256,300), N=300.
    n = 300
    plan = _plan(128, [NonPrefixHit("h", 128, 0, 128)], n, link=0)
    _assert_invariants(plan, n, 128)
    m = set(plan.recompute_offsets)
    # No B link tokens.
    assert 128 not in m and 130 not in m
    # C body + last present.
    assert {256, 299} <= m


# --------------------------------------------------------------------------
# reused_offsets (KV participation) correctness
# --------------------------------------------------------------------------


def test_reused_offsets_cover_prefix_and_b_and_m():
    n = 800
    plan = _plan(256, [NonPrefixHit("h", 512, 0, 256)], n)
    reused = set(plan.reused_offsets)
    # prefix A fully present.
    assert set(range(0, 256)) <= reused
    # B fully present (whole reused chunk's KV participates).
    assert set(range(512, 768)) <= reused
    # M (recompute) positions also live KV.
    assert set(plan.recompute_offsets) <= reused
    # union covers the whole sequence here (A ∪ B ∪ C == [0,N)).
    assert reused == set(range(n))


# --------------------------------------------------------------------------
# Flag-off dense path (Phase 1/2a unchanged behavior)
# --------------------------------------------------------------------------


def test_flag_off_returns_dense_empty_plan():
    pol = LegoLinkRecompute(num_link_tokens=8, phase1_dense=True)
    sel = ReuseSelection(
        prefix_extent=256, non_prefix_hits=[NonPrefixHit("h", 512, 0, 256)]
    )
    plan = pol.plan_recompute(_PromptLenShim(800), sel, block_size=16)
    assert plan.recompute_offsets == []
    assert plan.target_positions == []
    assert plan.is_sparse is False


def test_full_prefix_degenerate_is_dense():
    # Whole prompt is the contiguous prefix -> no sparse forward, empty plan,
    # and the "prefix A never in M" invariant is preserved (N-1 not forced in).
    n = 256
    plan = _plan(256, [], n)
    assert plan.recompute_offsets == []
    assert plan.is_sparse is False
