# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External prefetch command path: dynamo frontend -> vLLM engine.

The agentic loop this wires up (user's architecture):

  1. a decode worker's LLM output contains the tool-call schema;
  2. the **dynamo frontend** receives the response and parses it
     (``prefetch_parser.parse_tool_call_reads``);
  3. the frontend already does worker scheduling, so IT decides the
     prefetch placement (which engine/replica = ``dst_worker``) and
     **injects the command** into that engine;
  4. the engine's workers pull the named fileKV chunks and stage them into
     their (staging) cache;
  5. the next turn's prefill on that engine does non-contiguous KV reuse
     with the H2D latency already hidden.

This module provides step 3's transport plus the frontend-side glue:

  * ``EpicPrefetchListener`` -- a ZMQ REP server thread living in the
    ENGINE-CORE process next to the SCHEDULER-role connector. Enabled by the
    ``epic_prefetch_endpoint`` extra config (e.g. ``tcp://0.0.0.0:5557`` or
    ``ipc:///tmp/epic-prefetch.sock``). It only forwards decoded JSON to
    ``EpicConnector.handle_prefetch_command`` -- no policy here.
  * ``EpicPrefetchClient`` -- what the dynamo frontend calls.
  * ``DynamoPrefetchBridge`` -- parse + render + tokenize + send, and
    (optionally) WARM files whose chunks the engine does not know yet: a
    prefetch can only stage what the fileKV store holds, so the reply's
    ``dropped`` hashes trigger the caller-supplied ``warm_fn`` (typically a
    max_tokens=1 generation over the rendered file text, which populates the
    store via the normal save path; the next prefetch then hits).

Protocol (JSON over ZMQ REQ/REP):

  -> {"cmd": "prefetch", "chunk_hashes": [...], "token_ids": [...],
      "dst_worker": 2}
  <- {"ok": true, "queued": [...], "dropped": [...]}
  -> {"cmd": "ping"}
  <- {"ok": true}

The endpoint is a control-plane socket: bind it on a trusted interface.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable

from vllm.distributed.kv_transfer.kv_connector.v1.epic.prefetch_parser import (
    RenderFn,
    TokenizeFn,
    ToolCallRead,
    parse_tool_call_reads,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Handles one decoded command dict, returns the reply dict
# (== EpicConnector.handle_prefetch_command).
CommandHandler = Callable[[dict], dict]


class EpicPrefetchListener:
    """ZMQ REP server thread that feeds commands to the connector.

    ``max_frame_bytes`` bounds a single command frame (default 16 MiB --
    generous for hash lists, tiny for abuse): oversized frames are answered
    with an error WITHOUT JSON-decoding them, keeping a misbehaving frontend
    from ballooning the engine-core process.
    """

    def __init__(
        self,
        endpoint: str,
        handler: CommandHandler,
        max_frame_bytes: int = 16 * 1024 * 1024,
    ):
        self._endpoint = endpoint
        self._handler = handler
        self._max_frame_bytes = int(max_frame_bytes)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ctx = None

    def start(self) -> None:
        import zmq

        self._ctx = zmq.Context.instance()
        sock = self._ctx.socket(zmq.REP)
        sock.bind(self._endpoint)
        # Poll so the thread can observe the stop flag between requests.
        sock.rcvtimeo = 200  # ms

        def _serve() -> None:
            while not self._stop.is_set():
                try:
                    raw = sock.recv()
                except zmq.Again:
                    continue
                except zmq.ZMQError:
                    break
                try:
                    if len(raw) > self._max_frame_bytes:
                        reply = {
                            "ok": False,
                            "error": (
                                f"frame too large ({len(raw)} bytes > "
                                f"{self._max_frame_bytes})"
                            ),
                        }
                    else:
                        msg = json.loads(raw.decode("utf-8"))
                        reply = self._handler(msg)
                except Exception as e:  # noqa: BLE001 -- reply, don't die.
                    reply = {"ok": False, "error": repr(e)}
                try:
                    sock.send(json.dumps(reply).encode("utf-8"))
                except zmq.ZMQError:
                    break
            sock.close(linger=0)

        self._thread = threading.Thread(
            target=_serve, name="epic-prefetch-listener", daemon=True
        )
        self._thread.start()
        logger.info("EPIC prefetch listener bound at %s", self._endpoint)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


class EpicPrefetchClient:
    """Frontend-side client (what dynamo calls to inject a prefetch)."""

    def __init__(self, endpoint: str, timeout_ms: int = 2000):
        import zmq

        self._ctx = zmq.Context.instance()
        self._endpoint = endpoint
        self._timeout_ms = int(timeout_ms)
        self._lock = threading.Lock()
        self._sock = None
        self._zmq = zmq

    def _connect(self):
        sock = self._ctx.socket(self._zmq.REQ)
        sock.rcvtimeo = self._timeout_ms
        sock.sndtimeo = self._timeout_ms
        sock.linger = 0
        sock.connect(self._endpoint)
        return sock

    def _roundtrip(self, msg: dict) -> dict:
        with self._lock:
            if self._sock is None:
                self._sock = self._connect()
            try:
                self._sock.send(json.dumps(msg).encode("utf-8"))
                return json.loads(self._sock.recv().decode("utf-8"))
            except self._zmq.ZMQError as e:
                # REQ sockets wedge after a timeout; rebuild on any error.
                self._sock.close(linger=0)
                self._sock = None
                raise TimeoutError(
                    f"EPIC prefetch endpoint {self._endpoint}: {e}"
                ) from e

    def ping(self) -> bool:
        try:
            return bool(self._roundtrip({"cmd": "ping"}).get("ok"))
        except TimeoutError:
            return False

    def prefetch(
        self,
        chunk_hashes: list[str] | None = None,
        token_ids: list[int] | None = None,
        dst_worker: int = -1,
    ) -> tuple[list[str], list[str]]:
        """Inject a prefetch command. Returns (queued, dropped) hashes."""
        reply = self._roundtrip(
            {
                "cmd": "prefetch",
                "chunk_hashes": list(chunk_hashes or []),
                "token_ids": list(token_ids or []),
                "dst_worker": int(dst_worker),
            }
        )
        if not reply.get("ok"):
            raise RuntimeError(f"prefetch rejected: {reply}")
        return list(reply.get("queued", [])), list(reply.get("dropped", []))

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                self._sock.close(linger=0)
                self._sock = None


class DynamoPrefetchBridge:
    """Frontend-side orchestration: LLM response -> prefetch (+ warm) commands.

    The dynamo frontend owns worker scheduling, so it passes the placement
    decision (``dst_worker``) per turn; everything else -- parsing the tool
    calls out of the decode worker's response, byte-faithful rendering,
    tokenization, chunk hashing -- happens here, engine-independent.
    """

    def __init__(
        self,
        *,
        client: EpicPrefetchClient,
        render_fn: RenderFn,
        tokenize_fn: TokenizeFn,
        chunk_size: int = 256,
        pad_token_id: int = 0,
        warm_fn: Callable[[ToolCallRead, str], None] | None = None,
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (  # noqa: E501
            hash_chunk_tokens,
        )

        self._client = client
        self._render = render_fn
        self._tokenize = tokenize_fn
        self._chunk_size = int(chunk_size)
        self._pad = int(pad_token_id)
        self._warm = warm_fn
        self._hash = hash_chunk_tokens

    def _hashes(self, token_ids: list[int]) -> list[str]:
        ids = list(token_ids)
        if not ids:
            return []
        rem = len(ids) % self._chunk_size
        if rem:
            ids.extend([self._pad] * (self._chunk_size - rem))
        return [
            self._hash(ids[i : i + self._chunk_size])
            for i in range(0, len(ids), self._chunk_size)
        ]

    def on_turn_response(
        self, llm_output: str, dst_worker: int
    ) -> dict[str, list[str]]:
        """Parse a decode worker's response and inject prefetch command(s).

        Returns {"queued": [...], "dropped": [...], "warmed": [file, ...]}.
        Files whose chunks the engine dropped (not yet in its fileKV store)
        are handed to ``warm_fn`` so the store gets populated for next time.
        """
        per_read: list[tuple[ToolCallRead, str, list[str]]] = []
        all_hashes: list[str] = []
        seen: set[str] = set()
        for read in parse_tool_call_reads(llm_output):
            try:
                text = self._render(read)
                hashes = self._hashes(self._tokenize(text))
            except Exception as e:  # noqa: BLE001 -- best-effort per file.
                logger.warning(
                    "EPIC prefetch bridge: render/tokenize failed for %s: %s",
                    read.file_path,
                    e,
                )
                continue
            per_read.append((read, text, hashes))
            for h in hashes:
                if h not in seen:
                    seen.add(h)
                    all_hashes.append(h)

        if not all_hashes:
            return {"queued": [], "dropped": [], "warmed": []}

        queued, dropped = self._client.prefetch(
            chunk_hashes=all_hashes, dst_worker=dst_worker
        )
        warmed: list[str] = []
        if dropped and self._warm is not None:
            dropped_set = set(dropped)
            for read, text, hashes in per_read:
                if any(h in dropped_set for h in hashes):
                    try:
                        self._warm(read, text)
                        warmed.append(read.file_path)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "EPIC prefetch bridge: warm failed for %s: %s",
                            read.file_path,
                            e,
                        )
        return {"queued": queued, "dropped": dropped, "warmed": warmed}
