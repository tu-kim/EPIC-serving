# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External prefetch command path (dynamo frontend -> engine), CPU tests.

Covers handle_prefetch_command (transport-agnostic), the full ZMQ
listener/client roundtrip over an ipc:// socket, and the DynamoPrefetchBridge
warm-on-dropped flow."""

import os
import threading
import uuid

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    EpicSchedulerIndex,
    hash_chunk_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
    EpicConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch_service import (
    DynamoPrefetchBridge,
    EpicPrefetchClient,
    EpicPrefetchListener,
)

BLOCK = 16
CHUNK = 64


def _scheduler_connector():
    index = EpicSchedulerIndex(
        capacity_bytes=10**8,
        num_layers=1,
        num_kv_heads=1,
        head_size=4,
        cache_dtype_size=4,
    )
    c = object.__new__(EpicConnector)
    c._block_size = BLOCK
    c._chunk_size = CHUNK
    c._store = None
    c._index = index
    c._prefetch_lock = threading.Lock()
    c._prefetch_queue = []
    return c, index


def test_handle_prefetch_command_reports_queued_and_dropped():
    c, index = _scheduler_connector()
    known_tokens = list(range(CHUNK))
    known = hash_chunk_tokens(known_tokens)
    index.register(known, CHUNK)

    reply = c.handle_prefetch_command(
        {
            "cmd": "prefetch",
            "chunk_hashes": ["unknown-hash"],
            "token_ids": known_tokens,
            "dst_worker": 3,
        }
    )
    assert reply["ok"] is True
    assert reply["queued"] == [known]
    assert reply["dropped"] == ["unknown-hash"]
    assert len(c._prefetch_queue) == 1
    assert c._prefetch_queue[0].dst_worker == 3

    assert c.handle_prefetch_command({"cmd": "ping"}) == {"ok": True}
    assert c.handle_prefetch_command({"cmd": "nope"})["ok"] is False


def test_zmq_listener_client_roundtrip():
    c, index = _scheduler_connector()
    tokens = list(range(CHUNK))
    known = hash_chunk_tokens(tokens)
    index.register(known, CHUNK)

    endpoint = f"ipc:///tmp/epic-prefetch-test-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    listener = EpicPrefetchListener(endpoint, c.handle_prefetch_command)
    listener.start()
    try:
        client = EpicPrefetchClient(endpoint, timeout_ms=5000)
        assert client.ping() is True
        queued, dropped = client.prefetch(
            chunk_hashes=["bogus"], token_ids=tokens, dst_worker=1
        )
        assert queued == [known]
        assert dropped == ["bogus"]
        # The command landed in the scheduler-side queue (next step's meta).
        assert len(c._prefetch_queue) == 1
        assert c._prefetch_queue[0].chunk_hashes == [known]
        assert c._prefetch_queue[0].dst_worker == 1
        client.close()
    finally:
        listener.stop()


class _FakeClient:
    """Bridge unit-test double: engine knows only `known` hashes."""

    def __init__(self, known: set[str]):
        self.known = known
        self.calls: list[tuple[list[str], int]] = []

    def prefetch(self, chunk_hashes=None, token_ids=None, dst_worker=-1):
        hashes = list(chunk_hashes or [])
        self.calls.append((hashes, dst_worker))
        queued = [h for h in hashes if h in self.known]
        return queued, [h for h in hashes if h not in self.known]


def test_bridge_warms_files_with_dropped_chunks():
    good = "good.py"
    cold = "cold.py"
    good_tokens = list(range(CHUNK))
    cold_tokens = list(range(1000, 1000 + CHUNK))
    tok = {good: good_tokens, cold: cold_tokens}
    good_hash = hash_chunk_tokens(good_tokens)

    warmed: list[str] = []
    bridge = DynamoPrefetchBridge(
        client=_FakeClient(known={good_hash}),
        render_fn=lambda read: read.file_path,
        tokenize_fn=lambda text: list(tok[text]),
        chunk_size=CHUNK,
        warm_fn=lambda read, text: warmed.append(read.file_path),
    )
    out = bridge.on_turn_response(
        f"<tool_call><function=read>"
        f"<parameter=filePath>{good}</parameter></function></tool_call>"
        f"<tool_call><function=read>"
        f"<parameter=filePath>{cold}</parameter></function></tool_call>",
        dst_worker=2,
    )
    assert out["queued"] == [good_hash]
    assert len(out["dropped"]) == 1  # cold.py's chunk unknown to the engine.
    assert out["warmed"] == [cold]  # -> warm_fn fired for the cold file only.
    assert warmed == [cold]


def test_bridge_no_tool_calls_is_noop():
    client = _FakeClient(known=set())
    bridge = DynamoPrefetchBridge(
        client=client,
        render_fn=lambda r: "",
        tokenize_fn=lambda t: [],
        chunk_size=CHUNK,
    )
    out = bridge.on_turn_response("no reads here", dst_worker=0)
    assert out == {"queued": [], "dropped": [], "warmed": []}
    assert client.calls == []
