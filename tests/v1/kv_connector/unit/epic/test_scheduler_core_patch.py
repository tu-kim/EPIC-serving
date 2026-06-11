# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC Phase 2b S3 scheduler-core patch tests (CPU-only).

Validates the three invasive scheduler.py seams in isolation, with a real
``Scheduler`` instance and a duck-typed mock connector, so no GPU / PICRotator /
real EpicConnector worker path is needed:

  * ``_apply_epic_sparse_overrides`` rewrites ``num_scheduled_tokens`` to |M|,
    fixes ``total_num_scheduled_tokens``, and stamps the three ``epic_*`` dicts.
  * ``_update_after_schedule`` advances ``num_computed_tokens`` by
    ``epic_computed_advance`` (not ``num_scheduled_tokens``) so it converges to N.
  * the VANILLA path (no Epic hooks / empty dicts) is byte-for-byte unchanged.
"""

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.output import SchedulerOutput


# --------------------------------------------------------------------------
# Mock connector exposing only the duck-typed S3 hooks.
# --------------------------------------------------------------------------


class _MockSparseConnector:
    """Stand-in for EpicConnector's scheduler-core hook surface.

    ``plan[req_id] = (m_rows, positions, advance)``. A req_id absent from the
    plan is treated as non-sparse (every hook returns None) -> default path.
    """

    def __init__(self, plan):
        self.plan = plan

    def get_sparse_num_scheduled_tokens(self, meta, req_id):
        entry = self.plan.get(req_id)
        return entry[0] if entry is not None else None

    def get_sparse_positions(self, meta, req_id):
        entry = self.plan.get(req_id)
        return entry[1] if entry is not None else None

    def get_sparse_computed_advance(self, meta, req_id):
        entry = self.plan.get(req_id)
        return entry[2] if entry is not None else None


def _empty_output(num_scheduled):
    so = SchedulerOutput.make_empty()
    so.num_scheduled_tokens = dict(num_scheduled)
    so.total_num_scheduled_tokens = sum(num_scheduled.values())
    return so


# --------------------------------------------------------------------------
# override application
# --------------------------------------------------------------------------


def test_overrides_rewrite_rows_and_stamp_dicts():
    sched = create_scheduler()
    # req "0": sparse N=192, external=128 -> num_new(=|C|)=64 originally.
    #   M = 72 rows, positions last == N-1, advance = N-external = 64.
    conn = _MockSparseConnector(
        {"0": (72, list(range(120, 192)), 64)}  # 72 positions
    )
    so = _empty_output({"0": 64, "1": 10})  # "1" is a non-sparse decode-ish req.

    sched._apply_epic_sparse_overrides(conn, so, meta=None)

    # row count overridden to |M|.
    assert so.num_scheduled_tokens["0"] == 72
    # non-sparse req untouched.
    assert so.num_scheduled_tokens["1"] == 10
    # aggregate kept consistent: +(72-64) = +8.
    assert so.total_num_scheduled_tokens == 64 + 10 + 8
    # the three epic dicts stamped for the sparse req only.
    assert so.epic_sparse_positions["0"] == list(range(120, 192))
    assert so.epic_seq_len["0"] == 192  # positions[-1]+1
    assert so.epic_computed_advance["0"] == 64
    assert "1" not in so.epic_sparse_positions
    assert "1" not in so.epic_computed_advance


def test_overrides_noop_for_non_epic_connector():
    sched = create_scheduler()

    class _Plain:
        pass  # no get_sparse_* hooks.

    so = _empty_output({"0": 64, "1": 10})
    before_rows = dict(so.num_scheduled_tokens)
    before_total = so.total_num_scheduled_tokens

    sched._apply_epic_sparse_overrides(_Plain(), so, meta=None)

    assert so.num_scheduled_tokens == before_rows
    assert so.total_num_scheduled_tokens == before_total
    assert so.epic_sparse_positions == {}
    assert so.epic_seq_len == {}
    assert so.epic_computed_advance == {}


def test_overrides_noop_when_no_sparse_plan():
    sched = create_scheduler()
    conn = _MockSparseConnector({})  # Epic connector but nothing sparse this step.
    so = _empty_output({"0": 64})
    before = dict(so.num_scheduled_tokens)

    sched._apply_epic_sparse_overrides(conn, so, meta=None)

    assert so.num_scheduled_tokens == before
    assert so.total_num_scheduled_tokens == 64
    assert so.epic_computed_advance == {}


# --------------------------------------------------------------------------
# _update_after_schedule advance override
# --------------------------------------------------------------------------


def test_update_after_schedule_uses_epic_advance():
    sched = create_scheduler()
    reqs = create_requests(num_requests=1, num_tokens=192)
    req = reqs[0]
    sched.requests[req.request_id] = req
    # Simulate the scheduler having set num_computed_tokens to external=128
    # (local+external) at :804, then the override.
    req.num_computed_tokens = 128

    so = _empty_output({req.request_id: 72})  # |M|=72 rows forwarded.
    so.epic_computed_advance[req.request_id] = 64  # N - external = 192 - 128.

    sched._update_after_schedule(so)

    # num_computed advanced by 64 (NOT 72) -> lands exactly on N=192.
    assert req.num_computed_tokens == 192
    # prefill complete -> next step is decode.
    assert req.is_prefill_chunk is False


def test_update_after_schedule_default_advance_unchanged():
    sched = create_scheduler()
    reqs = create_requests(num_requests=1, num_tokens=192)
    req = reqs[0]
    sched.requests[req.request_id] = req
    req.num_computed_tokens = 0

    # No epic_computed_advance entry -> default += num_scheduled_tokens.
    so = _empty_output({req.request_id: 100})

    sched._update_after_schedule(so)

    assert req.num_computed_tokens == 100
    assert req.is_prefill_chunk is True  # 100 < 192.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
