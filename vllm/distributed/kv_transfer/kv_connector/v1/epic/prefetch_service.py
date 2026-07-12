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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.filekv_catalog import (  # noqa: E501
        FileKVCatalog,
    )

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
        evict_hashes: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Inject a prefetch command. Returns (queued, dropped) hashes.

        ``evict_hashes`` invalidates stale STAGED chunks on the workers
        (file-modification lifecycle); may be sent with or without new
        hashes to stage.
        """
        reply = self._roundtrip(
            {
                "cmd": "prefetch",
                "chunk_hashes": list(chunk_hashes or []),
                "token_ids": list(token_ids or []),
                "dst_worker": int(dst_worker),
                "evict_hashes": list(evict_hashes or []),
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

    Consistency (mid-turn file modifications; see filekv_catalog.py):
      * ``catalog`` + ``fingerprint_fn``: every processed read is recorded as
        (path, canonical range) -> (fingerprint, hashes). The file is
        fingerprinted BEFORE and AFTER rendering; a mismatch means the render
        may be TORN (an agent was writing concurrently) -> that file is
        skipped this round (a prefetch is a hint).
      * ``on_file_modified(path, dst_worker)``: the tool executor calls this
        when an agent edits a file. Stale staged chunks are EVICTED on the
        workers, and every previously-recorded unit of the path is
        re-rendered at the new version and re-prefetched (its ``dropped``
        reply fires ``warm_fn``, which repopulates fileKV via the engine's
        normal save path -- the "re-save" flow).
    Partial reads: ranges are canonicalized with ``snap_lines`` (see
    ``canonicalize_range``); renders below ``min_cache_tokens`` are skipped
    (sub-chunk text cannot fill a grid chunk and recompute is cheap).
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
        catalog: "FileKVCatalog | None" = None,
        fingerprint_fn: Callable[[str], str] | None = None,
        snap_lines: int = 0,
        min_cache_tokens: int | None = None,
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
        self._catalog = catalog
        self._fingerprint = fingerprint_fn
        self._snap_lines = int(snap_lines)
        # Below this many raw tokens a render cannot fill one grid chunk;
        # caching it would be pure overhead (recompute is microseconds).
        self._min_cache_tokens = (
            int(min_cache_tokens)
            if min_cache_tokens is not None
            else self._chunk_size
        )

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

    def _canonical_read(self, read: ToolCallRead) -> ToolCallRead:
        from vllm.distributed.kv_transfer.kv_connector.v1.epic.filekv_catalog import (  # noqa: E501
            canonicalize_range,
        )

        start, end = canonicalize_range(
            read.start_line, read.end_line, self._snap_lines
        )
        if (start, end) == (read.start_line, read.end_line):
            return read
        return ToolCallRead(
            file_path=read.file_path, start_line=start, end_line=end
        )

    def _process_read(
        self, read: ToolCallRead
    ) -> "tuple[ToolCallRead, str, list[str], list[str]] | None":
        """Canonicalize, render (torn-safe), hash and record one read.

        Returns (canonical_read, text, hashes, stale_evictions) or None when
        the read is skipped (torn render / sub-chunk / render error).
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.epic.filekv_catalog import (  # noqa: E501
            RangeKey,
        )

        read = self._canonical_read(read)
        try:
            fp_before = (
                self._fingerprint(read.file_path) if self._fingerprint else ""
            )
            text = self._render(read)
            fp_after = (
                self._fingerprint(read.file_path) if self._fingerprint else ""
            )
            if fp_before != fp_after:
                logger.warning(
                    "EPIC prefetch bridge: %s changed WHILE rendering (torn "
                    "render); skipping this round.",
                    read.file_path,
                )
                return None
            ids = self._tokenize(text)
        except Exception as e:  # noqa: BLE001 -- best-effort per file.
            logger.warning(
                "EPIC prefetch bridge: render/tokenize failed for %s: %s",
                read.file_path,
                e,
            )
            return None
        if len(ids) < self._min_cache_tokens:
            logger.debug(
                "EPIC prefetch bridge: %s render below %d tokens; "
                "not cacheable (recompute is cheap), skipping.",
                read.file_path,
                self._min_cache_tokens,
            )
            return None
        hashes = self._hashes(ids)
        stale: list[str] = []
        if self._catalog is not None:
            key = RangeKey(read.file_path, read.start_line, read.end_line)
            # Missed-modification defense: if the recorded fingerprint
            # differs, the old unit's hashes are stale -> evict them along
            # with this command.
            stale = self._catalog.stale_check(key, fp_after)
            self._catalog.record(key, fp_after, hashes)
        return read, text, hashes, stale

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
        evictions: list[str] = []
        seen: set[str] = set()
        for read in parse_tool_call_reads(llm_output):
            processed = self._process_read(read)
            if processed is None:
                continue
            c_read, text, hashes, stale = processed
            per_read.append((c_read, text, hashes))
            evictions.extend(h for h in stale if h not in set(evictions))
            for h in hashes:
                if h not in seen:
                    seen.add(h)
                    all_hashes.append(h)

        if not all_hashes and not evictions:
            return {"queued": [], "dropped": [], "warmed": []}

        kwargs: dict = {"chunk_hashes": all_hashes, "dst_worker": dst_worker}
        if evictions:
            kwargs["evict_hashes"] = evictions
        queued, dropped = self._client.prefetch(**kwargs)
        warmed = self._warm_dropped(per_read, dropped)
        return {"queued": queued, "dropped": dropped, "warmed": warmed}

    def _warm_dropped(
        self,
        per_read: list[tuple[ToolCallRead, str, list[str]]],
        dropped: list[str],
    ) -> list[str]:
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
        return warmed

    def on_file_modified(self, path: str, dst_worker: int) -> dict:
        """File-modification lifecycle: evict stale staging, re-warm the new
        version ("다시 fileKV로 저장").

        Call from the tool executor whenever an agent writes ``path``
        mid-turn. Effects, in order:
          1. the catalog bumps the path's version and yields every recorded
             unit -> their hashes are sent as ``evict_hashes`` (workers drop
             the stale STAGED copies; the CPU store is untouched -- content
             addressing keeps old chunks valid for in-flight prompts that
             still embed the old bytes, and they age out via LRU);
          2. every previously-known unit (whole file / canonical ranges) is
             re-rendered at the NEW content and re-prefetched; hashes the
             engine does not know yet come back as ``dropped`` and fire
             ``warm_fn`` -- the engine's normal save path then re-saves the
             new version's fileKV.
        Without a catalog only step 2 for the whole file runs (no eviction
        bookkeeping is possible).
        """
        stale: list[str] = []
        reads: list[ToolCallRead] = []
        if self._catalog is not None:
            records = self._catalog.on_file_modified(path)
            seen: set[str] = set()
            for rec in records:
                for h in rec.chunk_hashes:
                    if h not in seen:
                        seen.add(h)
                        stale.append(h)
                reads.append(
                    ToolCallRead(
                        file_path=path,
                        start_line=rec.key.start_line,
                        end_line=rec.key.end_line,
                    )
                )
        if not reads:
            reads = [ToolCallRead(file_path=path)]  # whole file fallback.

        per_read: list[tuple[ToolCallRead, str, list[str]]] = []
        new_hashes: list[str] = []
        seen_h: set[str] = set()
        for read in reads:
            processed = self._process_read(read)
            if processed is None:
                continue
            c_read, text, hashes, more_stale = processed
            per_read.append((c_read, text, hashes))
            stale.extend(h for h in more_stale if h not in set(stale))
            for h in hashes:
                if h not in seen_h:
                    seen_h.add(h)
                    new_hashes.append(h)

        if not new_hashes and not stale:
            return {"evicted": [], "queued": [], "dropped": [], "warmed": []}
        kwargs: dict = {"chunk_hashes": new_hashes, "dst_worker": dst_worker}
        if stale:
            kwargs["evict_hashes"] = stale
        queued, dropped = self._client.prefetch(**kwargs)
        warmed = self._warm_dropped(per_read, dropped)
        return {
            "evicted": stale,
            "queued": queued,
            "dropped": dropped,
            "warmed": warmed,
        }
