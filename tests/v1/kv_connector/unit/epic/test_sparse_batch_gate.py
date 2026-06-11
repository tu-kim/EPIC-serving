# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC Phase 2b S7 single-batch gate tests (CPU-only).

Validates the scheduler-core gate that isolates an EPIC *sparse* (non-contiguous
reuse) request into its own step, with a real ``Scheduler`` instance and a
duck-typed mock connector (no GPU / PICRotator / real EpicConnector worker path).

The gate has two halves in ``scheduler.schedule()``:

  * a sparse request is *deferred* (not co-scheduled) when the step already has
    other scheduled requests (running or earlier waiting), and
  * once a sparse request is scheduled, the waiting loop *breaks* so no later
    request joins the batch.

The complementary invariant under test: when nothing is sparse this step, the
default batching is unchanged (several requests scheduled together).
"""

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.request import RequestStatus


# --------------------------------------------------------------------------
# Minimal mock connector exposing only what scheduler.schedule() touches.
# --------------------------------------------------------------------------


class _MockGateConnector:
    """Stand-in for EpicConnector's scheduler-facing surface for the gate.

    ``sparse_ids`` is the set of request ids that should be reported as sparse
    (non-contiguous reuse) this step. ``is_sparse_request`` returns the exact
    ``True`` the EPIC hook contracts for those ids and ``False`` otherwise so the
    gate's strict ``is True`` check fires only for genuine sparse requests.
    """

    def __init__(self, sparse_ids=()):
        self.sparse_ids = set(sparse_ids)

    # --- lifecycle no-ops the scheduler calls ---
    def on_new_request(self, request):
        pass

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        return KVConnectorMetadata()

    def request_finished(self, request, block_ids):
        return False, None

    # --- match: report no external tokens; mark some reqs sparse ---
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # external == 0 -> the request schedules its full prompt as new tokens.
        # Sparseness is reported separately via is_sparse_request (the S7 gate
        # reads that, not the matched-token count).
        return 0, False

    def is_sparse_request(self, req_id):
        return req_id in self.sparse_ids


def _ids(reqs):
    return [r.request_id for r in reqs]


def _scheduled_ids(output):
    return set(output.num_scheduled_tokens.keys())


# --------------------------------------------------------------------------
# Gate behavior
# --------------------------------------------------------------------------


def test_sparse_request_scheduled_alone_when_first():
    """A sparse request that is first in the waiting queue schedules alone and
    closes the batch (later waiting requests are deferred to the next step)."""
    sched = create_scheduler()
    sched.connector = _MockGateConnector(sparse_ids={"0"})

    reqs = create_requests(num_requests=3, num_tokens=32)
    for r in reqs:
        sched.add_request(r)

    output = sched.schedule()

    # Only the sparse request "0" was scheduled this step.
    assert _scheduled_ids(output) == {"0"}
    # The other two are still waiting (deferred), not lost.
    assert sched.get_num_unfinished_requests() == 3


def test_sparse_request_deferred_when_batch_nonempty():
    """When a non-sparse request is scheduled first, a following sparse request
    is deferred (not co-scheduled) so the sparse request never shares a batch."""
    sched = create_scheduler()
    # Second request in queue order is sparse.
    sched.connector = _MockGateConnector(sparse_ids={"1"})

    reqs = create_requests(num_requests=3, num_tokens=32)
    for r in reqs:
        sched.add_request(r)

    output = sched.schedule()

    scheduled = _scheduled_ids(output)
    # The sparse request "1" must NOT be in a batch with others.
    assert "1" not in scheduled
    # The non-sparse "0" (and "2", which precedes "1" once "1" is deferred) are
    # scheduled normally; the key invariant is "1" is isolated.
    assert "0" in scheduled


def test_two_sparse_requests_scheduled_in_separate_steps():
    """Two sparse requests each get their own step (never batched together)."""
    sched = create_scheduler()
    sched.connector = _MockGateConnector(sparse_ids={"0", "1"})

    reqs = create_requests(num_requests=2, num_tokens=32)
    for r in reqs:
        sched.add_request(r)

    out1 = sched.schedule()
    assert _scheduled_ids(out1) == {"0"}

    # Step the first sparse request to completion so it leaves the running set,
    # then the second sparse request schedules alone in a later step. We only
    # need to confirm they are never co-scheduled, so assert step 1 was a
    # singleton and step 2 (after finishing "0") is a singleton too.
    sched.finish_requests("0", RequestStatus.FINISHED_ABORTED)
    out2 = sched.schedule()
    assert _scheduled_ids(out2) == {"1"}


# --------------------------------------------------------------------------
# Non-sparse path is unchanged (no-trace when nothing is sparse).
# --------------------------------------------------------------------------


def test_non_sparse_batch_unchanged():
    """With no sparse requests, the default multi-request batching is intact."""
    sched = create_scheduler()
    sched.connector = _MockGateConnector(sparse_ids=set())  # nothing sparse.

    reqs = create_requests(num_requests=3, num_tokens=32)
    for r in reqs:
        sched.add_request(r)

    output = sched.schedule()

    # All three batched together (token budget is large) -> gate is inert.
    assert _scheduled_ids(output) == {"0", "1", "2"}


def test_no_connector_batch_unchanged():
    """Without any connector the gate is unreachable; batching is the default."""
    sched = create_scheduler()  # connector is None.

    reqs = create_requests(num_requests=3, num_tokens=32)
    for r in reqs:
        sched.add_request(r)

    output = sched.schedule()
    assert _scheduled_ids(output) == {"0", "1", "2"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
