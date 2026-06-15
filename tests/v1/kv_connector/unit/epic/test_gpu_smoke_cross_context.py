# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the step5 CROSS-CONTEXT fidelity probe of gpu_smoke.py.

step5 measures, as a CONTINUOUS distance-to-dense, how the EPIC sparse output
converges to the full-recompute (dense) output as the LegoLink boundary
recompute budget grows. The actual measurement needs a GPU + a model, but every
piece of logic it relies on is pure and must be correct before any GPU time is
spent:

  * the teacher-forced NLL extraction (vLLM prompt_logprobs -> per-position
    actual-token logprob) and its serialise/parse round-trip,
  * the NLL / mean-NLL / perplexity arithmetic and the scored-region masking,
  * the monotonicity judgement (distance should be non-increasing in link),
  * the worker spec gaining the prompt_logprobs / logprob_prefix_lens fields
    (round-trip).

These pin the metric direction (smaller == closer to dense) and the structural
contracts so the GPU run only exercises the model, not the plumbing.
"""

import math

import pytest

from tests.v1.kv_connector.unit.epic.gpu_smoke import (
    _STEP5_LINK_SWEEP,
    build_worker_spec,
    extract_actual_token_logprobs,
    is_monotonic_nonincreasing,
    mean_nll_to_perplexity,
    monotonicity_report,
    parse_spec,
    serialize_spec,
    step5_cross_context_fidelity,
    teacher_forced_nll,
)


# ---------------------------------------------------------------------------
# A minimal stand-in for vLLM's Logprob dataclass (has a `.logprob` attribute),
# plus a plain-float variant: extract_actual_token_logprobs must tolerate both
# (the GPU path yields Logprob objects; the CPU test feeds plain floats).
# ---------------------------------------------------------------------------
class _FakeLogprob:
    def __init__(self, logprob: float):
        self.logprob = logprob


# ---------------------------------------------------------------------------
# extract_actual_token_logprobs: per-position ACTUAL-token logprob
# ---------------------------------------------------------------------------


def test_extract_actual_token_logprobs_picks_actual_token():
    ids = [10, 11, 12]
    # vLLM puts None at position 0; each later position's dict includes the real
    # token id (plus optionally others). We must pull the REAL token's logprob.
    plp = [
        None,
        {11: _FakeLogprob(-0.5), 99: _FakeLogprob(-3.0)},
        {12: _FakeLogprob(-1.25), 7: _FakeLogprob(-9.0)},
    ]
    out = extract_actual_token_logprobs(ids, plp)
    assert out == [None, -0.5, -1.25]


def test_extract_actual_token_logprobs_tolerates_plain_floats():
    ids = [1, 2]
    plp = [None, {2: -0.75}]  # plain float entry, not a Logprob object
    assert extract_actual_token_logprobs(ids, plp) == [None, -0.75]


def test_extract_actual_token_logprobs_missing_token_is_none():
    # If the actual token id is not in the position dict (shouldn't happen with
    # vLLM, but be defensive), that position is None (unscored), not a crash.
    ids = [1, 2]
    plp = [None, {999: _FakeLogprob(-0.1)}]
    assert extract_actual_token_logprobs(ids, plp) == [None, None]


def test_extract_actual_token_logprobs_none_container():
    ids = [1, 2, 3]
    assert extract_actual_token_logprobs(ids, None) == [None, None, None]


def test_extract_actual_token_logprobs_shorter_container():
    # prompt_logprobs may be shorter than the prompt (defensive): missing tail
    # positions are None.
    ids = [1, 2, 3, 4]
    plp = [None, {2: _FakeLogprob(-0.2)}]
    assert extract_actual_token_logprobs(ids, plp) == [None, -0.2, None, None]


# ---------------------------------------------------------------------------
# teacher_forced_nll: sum -logprob over the SCORED region only
# ---------------------------------------------------------------------------


def test_teacher_forced_nll_scores_only_continuation():
    # prefix_len=2 -> positions 0,1 (the reuse prompt) are NOT scored.
    lps = [None, -1.0, -0.5, -2.0]  # positions 2,3 are the continuation
    s, n = teacher_forced_nll(lps, prefix_len=2)
    assert n == 2
    assert s == pytest.approx(0.5 + 2.0)


def test_teacher_forced_nll_skips_none_positions():
    lps = [None, -1.0, None, -2.0]
    s, n = teacher_forced_nll(lps, prefix_len=1)
    # positions 1 and 3 scored (-(-1)=1, -(-2)=2); position 2 None skipped.
    assert n == 2
    assert s == pytest.approx(3.0)


def test_teacher_forced_nll_empty_scored_region():
    lps = [None, -1.0]
    s, n = teacher_forced_nll(lps, prefix_len=5)  # prefix past end
    assert (s, n) == (0.0, 0)


def test_teacher_forced_nll_negative_prefix_clamped():
    lps = [-1.0, -2.0]
    s, n = teacher_forced_nll(lps, prefix_len=-3)
    assert n == 2
    assert s == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# mean_nll_to_perplexity: arithmetic + degenerate handling, direction
# ---------------------------------------------------------------------------


def test_mean_nll_to_perplexity_basic():
    mean, ppl = mean_nll_to_perplexity(sum_nll=3.0, n_scored=2)
    assert mean == pytest.approx(1.5)
    assert ppl == pytest.approx(math.exp(1.5))


def test_mean_nll_to_perplexity_zero_scored_is_inf():
    mean, ppl = mean_nll_to_perplexity(sum_nll=0.0, n_scored=0)
    assert mean == float("inf")
    assert ppl == float("inf")


def test_mean_nll_smaller_is_closer_to_dense():
    # Direction convention: a closer-to-dense run has SMALLER mean NLL. Two runs
    # scoring the SAME continuation: the one assigning higher logprob (less
    # negative) has lower NLL -> smaller distance.
    close_mean, _ = mean_nll_to_perplexity(sum_nll=0.4, n_scored=2)  # lps ~ -0.2
    far_mean, _ = mean_nll_to_perplexity(sum_nll=4.0, n_scored=2)    # lps ~ -2.0
    assert close_mean < far_mean


# ---------------------------------------------------------------------------
# monotonicity: distance should be non-increasing as link (recompute) grows
# ---------------------------------------------------------------------------


def test_is_monotonic_nonincreasing_true():
    assert is_monotonic_nonincreasing([3.0, 2.0, 2.0, 1.0]) is True


def test_is_monotonic_nonincreasing_flat_within_tol():
    assert is_monotonic_nonincreasing([1.0, 1.0 + 1e-9, 1.0]) is True


def test_is_monotonic_nonincreasing_false_on_increase():
    assert is_monotonic_nonincreasing([1.0, 2.0]) is False


def test_is_monotonic_nonincreasing_false_on_inf():
    assert is_monotonic_nonincreasing([1.0, float("inf")]) is False


def test_monotonicity_report_orders_by_link():
    # links given out of order; the report must sort by ASCENDING link first.
    links = [64, 0, 8, 256]
    # distance keyed to link: more recompute -> smaller distance (good case).
    dist = {0: 3.0, 8: 2.0, 64: 1.5, 256: 1.0}
    distances = [dist[link] for link in links]
    rep = monotonicity_report(links, distances)
    assert rep["ordered_by_link"] == [3.0, 2.0, 1.5, 1.0]
    assert rep["nonincreasing"] is True
    assert rep["n_violations"] == 0
    assert rep["first_distance"] == 3.0
    assert rep["last_distance"] == 1.0


def test_monotonicity_report_counts_violations():
    links = [0, 8, 64, 256]
    # link=8 distance jumps UP (a violation) then resumes decreasing.
    distances = [3.0, 3.5, 1.5, 1.0]
    rep = monotonicity_report(links, distances)
    assert rep["nonincreasing"] is False
    assert rep["n_violations"] == 1


# ---------------------------------------------------------------------------
# worker spec gains the step5 scoring fields (round-trip)
# ---------------------------------------------------------------------------


def test_worker_spec_logprob_fields_roundtrip():
    spec = build_worker_spec(
        role="sparse",
        model="some/model",
        kv_config={"kv_connector": "EpicConnector", "kv_role": "kv_both"},
        prompts=[[1, 2, 3, 4, 5]],
        warm_prompts=[[9, 8, 7]],
        max_tokens=1,
        in_process=True,
        read_counters=True,
        prompt_logprobs=0,
        logprob_prefix_lens=[3],
    )
    back = parse_spec(serialize_spec(spec))
    assert back == spec
    assert back["prompt_logprobs"] == 0
    assert back["logprob_prefix_lens"] == [3]


def test_worker_spec_logprob_fields_default_none():
    # Plain generation specs (steps 1-4) must keep the new fields as None so the
    # worker's scoring branch is NOT taken.
    spec = build_worker_spec(
        role="dense", model="m", kv_config=None, prompts=[[1, 2]],
        max_tokens=8,
    )
    assert spec["prompt_logprobs"] is None
    assert spec["logprob_prefix_lens"] is None


def test_worker_spec_logprob_prefix_lens_coerced_to_int():
    spec = build_worker_spec(
        role="dense", model="m", kv_config=None, prompts=[[1, 2]],
        max_tokens=1, prompt_logprobs=0, logprob_prefix_lens=[1.0, 2.0],
    )
    assert spec["logprob_prefix_lens"] == [1, 2]
    assert all(isinstance(x, int) for x in spec["logprob_prefix_lens"])


# ---------------------------------------------------------------------------
# step5 is registered + sweep includes the reuse-only extreme
# ---------------------------------------------------------------------------


def test_step5_in_registry():
    from tests.v1.kv_connector.unit.epic.gpu_smoke import _STEPS

    assert _STEPS.get(5) is step5_cross_context_fidelity


def test_step5_default_steps_include_5():
    from tests.v1.kv_connector.unit.epic.gpu_smoke import _parse_steps

    assert _parse_steps(None) == [1, 2, 3, 4, 5]
    assert _parse_steps("5") == [5]


def test_step5_link_sweep_has_reuse_only_extreme():
    # link=0 (pure stale reuse, zero recompute) MUST be in the sweep: it is the
    # endpoint that exposes the link's value (B_full is appended at runtime).
    assert 0 in _STEP5_LINK_SWEEP


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
