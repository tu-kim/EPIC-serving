# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the step5 CROSS-CONTEXT fidelity probe of gpu_smoke.py.

step5 measures how the EPIC sparse output converges to the full-recompute
(dense) output as the LegoLink boundary recompute budget grows. The actual
measurement needs a GPU + a model, but every piece of logic it relies on is pure
and must be correct before any GPU time is spent.

The metric was rewritten from teacher-forced ``prompt_logprobs`` NLL (which is
STRUCTURALLY incompatible with the M-row EPIC sparse forward -- the V1 runner
slices ``hidden_states`` by the FULL prompt length and allocates
``LogprobsTensors.empty_cpu(num_prompt_tokens - 1, ...)``, so the M-row sparse
forward yields a negative/oversized row count that reaches ``torch.empty`` and
raises ``OverflowError: out of range integral type conversion attempted``) to a
SPARSE-COMPATIBLE first-decode-token metric: the dense run's first greedy token
t0 is the reference, and each sparse link reports the RANK and logprob it assigns
to t0 (rank-distance = rank-1, smaller == closer to dense), plus a KL of the two
first-token top-K distributions. The decode position is the ONLY scored position
and EPIC sparse always forwards it (last prompt token in M), so no reused
(non-forwarded) prompt position is ever needed.

These tests pin:
  * first-token top-K extraction (vLLM SampleLogprobs -> plain map) + the
    JSON string-key round-trip the subprocess uses,
  * argmax / t0 logprob+rank lookup (incl. the floor when t0 falls outside K),
  * rank_distance direction (rank 1 -> 0; missing -> max),
  * the KL over shared support with a floor logprob,
  * the monotonicity judgement (distance non-increasing in link),
  * the worker spec gaining the first_token_logprobs field (round-trip).
"""

import math

import pytest

from tests.v1.kv_connector.unit.epic.gpu_smoke import (
    _FIRST_TOKEN_FLOOR_LOGPROB,
    _STEP5_LINK_SWEEP,
    _STEP5_TOPK,
    argmax_token,
    build_worker_spec,
    extract_first_token_logprobs,
    first_token_kl,
    is_monotonic_nonincreasing,
    monotonicity_report,
    parse_first_token_map,
    parse_spec,
    rank_distance,
    serialize_spec,
    step5_cross_context_fidelity,
    t0_logprob_and_rank,
)


# ---------------------------------------------------------------------------
# A minimal stand-in for vLLM's Logprob dataclass (has .logprob / .rank), plus
# a plain-float variant: extract_first_token_logprobs tolerates both (the GPU
# path yields Logprob objects; the CPU test feeds plain floats).
# ---------------------------------------------------------------------------
class _FakeLogprob:
    def __init__(self, logprob: float, rank: int | None = None):
        self.logprob = logprob
        self.rank = rank


# ---------------------------------------------------------------------------
# extract_first_token_logprobs: first decode step's top-K map
# ---------------------------------------------------------------------------


def test_extract_first_token_logprobs_basic():
    # SampleLogprobs is a list of dict[int, Logprob] per generated position.
    # With max_tokens=1 there is exactly one position; we take position 0.
    sample = [
        {
            10: _FakeLogprob(-0.1, rank=1),
            11: _FakeLogprob(-2.0, rank=2),
            12: _FakeLogprob(-3.5, rank=3),
        }
    ]
    out = extract_first_token_logprobs(sample)
    assert out == {
        10: {"logprob": -0.1, "rank": 1},
        11: {"logprob": -2.0, "rank": 2},
        12: {"logprob": -3.5, "rank": 3},
    }


def test_extract_first_token_logprobs_tolerates_plain_floats():
    # The CPU test may feed plain floats (no .logprob); rank is then None.
    sample = [{5: -0.25, 6: -1.0}]
    out = extract_first_token_logprobs(sample)
    assert out == {5: {"logprob": -0.25, "rank": None},
                   6: {"logprob": -1.0, "rank": None}}


def test_extract_first_token_logprobs_empty_or_none():
    assert extract_first_token_logprobs(None) is None
    assert extract_first_token_logprobs([]) is None
    assert extract_first_token_logprobs([{}]) is None


# ---------------------------------------------------------------------------
# JSON string-key round-trip (the subprocess emits {str(tid): entry})
# ---------------------------------------------------------------------------


def test_parse_first_token_map_reints_keys():
    raw = {"10": {"logprob": -0.1, "rank": 1},
           "11": {"logprob": -2.0, "rank": 2}}
    out = parse_first_token_map(raw)
    assert out == {10: {"logprob": -0.1, "rank": 1},
                   11: {"logprob": -2.0, "rank": 2}}
    assert all(isinstance(k, int) for k in out)


def test_parse_first_token_map_none_passthrough():
    assert parse_first_token_map(None) is None


# ---------------------------------------------------------------------------
# argmax_token: the dense reference t0
# ---------------------------------------------------------------------------


def test_argmax_token_prefers_rank_one():
    m = {10: {"logprob": -2.0, "rank": 2}, 11: {"logprob": -0.1, "rank": 1}}
    assert argmax_token(m) == 11


def test_argmax_token_falls_back_to_max_logprob_when_no_rank():
    m = {10: {"logprob": -2.0, "rank": None}, 11: {"logprob": -0.1, "rank": None}}
    assert argmax_token(m) == 11


def test_argmax_token_tie_breaks_to_smaller_id():
    m = {7: {"logprob": -1.0, "rank": None}, 3: {"logprob": -1.0, "rank": None}}
    assert argmax_token(m) == 3


def test_argmax_token_empty_is_none():
    assert argmax_token(None) is None
    assert argmax_token({}) is None


# ---------------------------------------------------------------------------
# t0_logprob_and_rank: read the dense argmax under a (sparse) run
# ---------------------------------------------------------------------------


def test_t0_logprob_and_rank_present():
    m = {10: {"logprob": -0.1, "rank": 1}, 11: {"logprob": -2.0, "rank": 2}}
    lp, rank = t0_logprob_and_rank(m, 11)
    assert lp == pytest.approx(-2.0)
    assert rank == 2


def test_t0_logprob_and_rank_missing_floors():
    # t0 outside this run's top-K -> floor logprob + rank None (maximal distance).
    m = {10: {"logprob": -0.1, "rank": 1}}
    lp, rank = t0_logprob_and_rank(m, 999)
    assert lp == _FIRST_TOKEN_FLOOR_LOGPROB
    assert rank is None


def test_t0_logprob_and_rank_empty_map_floors():
    lp, rank = t0_logprob_and_rank(None, 5)
    assert lp == _FIRST_TOKEN_FLOOR_LOGPROB
    assert rank is None


# ---------------------------------------------------------------------------
# rank_distance: direction (rank 1 -> 0; None -> max k)
# ---------------------------------------------------------------------------


def test_rank_distance_rank_one_is_zero():
    assert rank_distance(1, k=20) == 0


def test_rank_distance_increases_with_rank():
    assert rank_distance(2, k=20) == 1
    assert rank_distance(5, k=20) == 4


def test_rank_distance_none_is_max():
    assert rank_distance(None, k=20) == 20


def test_rank_distance_smaller_is_closer_to_dense():
    # Direction convention: a run that still picks t0 (rank 1) has the SMALLEST
    # distance; a run where t0 dropped to rank 5 is farther.
    assert rank_distance(1, k=20) < rank_distance(5, k=20)


# ---------------------------------------------------------------------------
# first_token_kl: KL(dense || sparse) over shared support with a floor
# ---------------------------------------------------------------------------


def test_first_token_kl_identical_is_zero():
    m = {10: {"logprob": -0.1, "rank": 1}, 11: {"logprob": -2.0, "rank": 2}}
    kl = first_token_kl(m, dict(m))
    assert kl == pytest.approx(0.0, abs=1e-9)


def test_first_token_kl_nonnegative_and_increases_with_divergence():
    dense = {10: {"logprob": -0.1}, 11: {"logprob": -2.3}}
    close = {10: {"logprob": -0.2}, 11: {"logprob": -1.8}}
    far = {10: {"logprob": -3.0}, 11: {"logprob": -0.05}}
    kl_close = first_token_kl(dense, close)
    kl_far = first_token_kl(dense, far)
    assert kl_close >= 0.0
    assert kl_far >= 0.0
    assert kl_far > kl_close


def test_first_token_kl_uses_floor_for_absent_sparse_token():
    # t0 present in dense but ABSENT from sparse -> sparse gets the floor logprob,
    # producing a large (finite) KL rather than a crash.
    dense = {10: {"logprob": -0.1}, 11: {"logprob": -2.3}}
    sparse = {11: {"logprob": -2.3}}  # token 10 (dense argmax) missing
    kl = first_token_kl(dense, sparse)
    assert math.isfinite(kl)
    assert kl > 0.0


def test_first_token_kl_empty_dense_is_inf():
    assert first_token_kl(None, {10: {"logprob": -0.1}}) == float("inf")
    assert first_token_kl({}, {10: {"logprob": -0.1}}) == float("inf")


# ---------------------------------------------------------------------------
# monotonicity: rank-distance should be non-increasing as link grows
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
    # rank-distance keyed to link: more recompute -> smaller distance (good case).
    dist = {0: 5.0, 8: 3.0, 64: 1.0, 256: 0.0}
    distances = [dist[link] for link in links]
    rep = monotonicity_report(links, distances)
    assert rep["ordered_by_link"] == [5.0, 3.0, 1.0, 0.0]
    assert rep["nonincreasing"] is True
    assert rep["n_violations"] == 0
    assert rep["first_distance"] == 5.0
    assert rep["last_distance"] == 0.0


def test_monotonicity_report_counts_violations():
    links = [0, 8, 64, 256]
    # link=8 distance jumps UP (a violation) then resumes decreasing.
    distances = [3.0, 4.0, 1.0, 0.0]
    rep = monotonicity_report(links, distances)
    assert rep["nonincreasing"] is False
    assert rep["n_violations"] == 1


# ---------------------------------------------------------------------------
# worker spec gains the step5 first_token_logprobs field (round-trip)
# ---------------------------------------------------------------------------


def test_worker_spec_first_token_logprobs_roundtrip():
    spec = build_worker_spec(
        role="sparse",
        model="some/model",
        kv_config={"kv_connector": "EpicConnector", "kv_role": "kv_both"},
        prompts=[[1, 2, 3, 4, 5]],
        warm_prompts=[[9, 8, 7]],
        max_tokens=1,
        in_process=True,
        read_counters=True,
        first_token_logprobs=20,
    )
    back = parse_spec(serialize_spec(spec))
    assert back == spec
    assert back["first_token_logprobs"] == 20


def test_worker_spec_first_token_logprobs_default_none():
    # Plain generation specs (steps 1-4) must keep the new field None so the
    # worker's first-token scoring branch is NOT taken (generation unchanged).
    spec = build_worker_spec(
        role="dense", model="m", kv_config=None, prompts=[[1, 2]],
        max_tokens=8,
    )
    assert spec["first_token_logprobs"] is None


def test_worker_spec_first_token_logprobs_coerced_to_int():
    spec = build_worker_spec(
        role="dense", model="m", kv_config=None, prompts=[[1, 2]],
        max_tokens=1, first_token_logprobs=20.0,
    )
    assert spec["first_token_logprobs"] == 20
    assert isinstance(spec["first_token_logprobs"], int)


def test_worker_spec_no_legacy_prompt_logprob_fields():
    # The old teacher-forcing fields must be GONE (the source of the GPU
    # OverflowError); their presence would re-enable the incompatible path.
    spec = build_worker_spec(
        role="sparse", model="m", kv_config=None, prompts=[[1, 2]],
        max_tokens=1, first_token_logprobs=20,
    )
    assert "prompt_logprobs" not in spec
    assert "logprob_prefix_lens" not in spec


# ---------------------------------------------------------------------------
# step5 is registered + sweep includes the reuse-only extreme + K sane
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


def test_step5_topk_large_enough_for_rank_lookup():
    # K must be > 1 so the dense argmax t0 usually appears in each sparse run's
    # top-K (read its rank directly rather than flooring to "outside K").
    assert _STEP5_TOPK >= 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
