# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool-call parsing + fileKV prefetch orchestration (feature/prefetch).

Agentic serving loop this targets:

  1. Turn t's LLM output contains structured tool calls naming the files the
     agent will read, e.g.::

         <tool_call><function=read>
         <parameter=filePath>src/foo.py</parameter>
         <parameter=startLine>10</parameter>
         <parameter=endLine>80</parameter>
         </function></tool_call>

     (or OpenAI-style JSON tool calls). Parse them AT EMIT TIME -- before the
     tool even executes -- to learn which fileKV the next turn needs.
  2. The frontend scheduler (dynamo) knows which worker/replica will serve
     turn t+1; it hands us that worker id.
  3. ``FileKVPrefetcher`` renders each file exactly as the tool result will
     render it (BYTE fidelity is the caller's responsibility -- same line
     numbering, wrappers and line ranges -- otherwise the content hashes miss),
     tokenizes, pads to the chunk grid, hashes, and ships the hashes to the
     designated worker via the transport hook (in-process: connector
     ``enqueue_prefetch``; out-of-process: whatever RPC the deployment has).

The parser and prefetcher deliberately do NO file IO and NO tokenization
themselves: rendering fidelity and tokenizer choice are deployment-specific
(see the chunk-grid alignment convention in epic/DESIGN.md), so both are
injected callables.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    hash_chunk_tokens,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Renders a parsed read into the EXACT text the tool result will contain.
RenderFn = Callable[["ToolCallRead"], str]
# Tokenizes rendered text into ids (the serving tokenizer, no special tokens).
TokenizeFn = Callable[[str], list[int]]
# Ships chunk hashes to the designated worker (connector.enqueue_prefetch or a
# deployment RPC). Signature: (chunk_hashes, dst_worker) -> None.
TransportFn = Callable[[list[str], int], None]


@dataclass(frozen=True)
class ToolCallRead:
    """One parsed file-read tool call from LLM output."""

    file_path: str
    start_line: int | None = None  # 1-based, inclusive; None = from start
    end_line: int | None = None  # 1-based, inclusive; None = to EOF


# --- format 1: <tool_call><function=read> ... XML-ish parameters ------------

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<fn>[\w.\-]+)>(?P<body>.*?)"
    r"(?:</function>\s*)?</tool_call>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=(?P<name>[\w.\-]+)>(?P<value>.*?)</parameter>", re.DOTALL
)

# Tool names that mean "read a file". Extend per deployment.
_READ_FUNCTIONS = frozenset({"read", "read_file", "readfile", "view", "open"})
_PATH_KEYS = ("filePath", "file_path", "path", "file", "filename")
_START_KEYS = ("startLine", "start_line", "offset", "start")
_END_KEYS = ("endLine", "end_line", "end")


def _to_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _read_from_params(params: dict[str, str]) -> ToolCallRead | None:
    path = next(
        (params[k].strip() for k in _PATH_KEYS if params.get(k, "").strip()),
        None,
    )
    if not path:
        return None
    start = next((_to_int(params[k]) for k in _START_KEYS if k in params), None)
    end = next((_to_int(params[k]) for k in _END_KEYS if k in params), None)
    return ToolCallRead(file_path=path, start_line=start, end_line=end)


def _parse_xmlish(text: str) -> list[ToolCallRead]:
    reads: list[ToolCallRead] = []
    for m in _TOOL_CALL_RE.finditer(text):
        if m.group("fn").lower() not in _READ_FUNCTIONS:
            continue
        params = {
            p.group("name"): p.group("value")
            for p in _PARAM_RE.finditer(m.group("body"))
        }
        read = _read_from_params(params)
        if read is not None:
            reads.append(read)
    return reads


# --- format 2: OpenAI-style JSON tool calls ---------------------------------


def _parse_json_tool_calls(text: str) -> list[ToolCallRead]:
    """Best-effort scan for JSON objects shaped like
    ``{"name": "read", "arguments": {...}}`` (arguments may be a JSON string,
    as the OpenAI API encodes it)."""
    reads: list[ToolCallRead] = []
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"(?P<fn>[\w.\-]+)"[^{}]*'
                         r'"arguments"\s*:\s*(?P<args>\{.*?\}|".*?")\s*\}',
                         text, re.DOTALL):
        if m.group("fn").lower() not in _READ_FUNCTIONS:
            continue
        raw = m.group("args")
        try:
            args = json.loads(raw)
            if isinstance(args, str):  # doubly-encoded arguments
                args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(args, dict):
            continue
        read = _read_from_params({k: str(v) for k, v in args.items()})
        if read is not None:
            reads.append(read)
    return reads


def parse_tool_call_reads(text: str) -> list[ToolCallRead]:
    """Extract every file-read tool call from a turn's LLM output text.

    Supports the XML-ish ``<tool_call><function=read>`` convention and
    OpenAI-style JSON tool calls. Unknown functions and malformed calls are
    skipped silently (a prefetch is a hint; never fail the serving path).
    Order of appearance is preserved; duplicates are kept (dedup happens at
    the hash level in the prefetcher).
    """
    return _parse_xmlish(text) + _parse_json_tool_calls(text)


# --- orchestration -----------------------------------------------------------


class FileKVPrefetcher:
    """Turns parsed tool calls into chunk-hash prefetch directives.

    Pipeline per LLM output: parse -> render (caller-supplied, byte-faithful)
    -> tokenize -> pad to the chunk grid -> hash full chunks -> transport to
    the designated worker. The pad token MUST be the same one the prompt
    assembly uses (chunk-grid alignment convention), or the hashes miss.
    """

    def __init__(
        self,
        *,
        render_fn: RenderFn,
        tokenize_fn: TokenizeFn,
        transport_fn: TransportFn,
        chunk_size: int = 256,
        pad_token_id: int = 0,
    ):
        self._render = render_fn
        self._tokenize = tokenize_fn
        self._transport = transport_fn
        self._chunk_size = int(chunk_size)
        self._pad_token_id = int(pad_token_id)

    def chunk_hashes_for_tokens(self, token_ids: Sequence[int]) -> list[str]:
        """Chunk-grid pad + hash, mirroring the connector's grid exactly."""
        ids = list(token_ids)
        if not ids:
            return []
        rem = len(ids) % self._chunk_size
        if rem:
            ids.extend([self._pad_token_id] * (self._chunk_size - rem))
        return [
            hash_chunk_tokens(ids[i : i + self._chunk_size])
            for i in range(0, len(ids), self._chunk_size)
        ]

    def prefetch_for_output(self, llm_output: str, dst_worker: int) -> list[str]:
        """Parse a turn's output and ship prefetch directives.

        Returns the (deduplicated, order-preserving) chunk hashes shipped --
        empty when no read tool calls were found. Render/tokenize errors on
        one file skip that file only.
        """
        hashes: list[str] = []
        seen: set[str] = set()
        for read in parse_tool_call_reads(llm_output):
            try:
                text = self._render(read)
                ids = self._tokenize(text)
            except Exception as e:  # noqa: BLE001 -- prefetch is best-effort.
                logger.warning(
                    "EPIC prefetch: render/tokenize failed for %s: %s",
                    read.file_path,
                    e,
                )
                continue
            for h in self.chunk_hashes_for_tokens(ids):
                if h not in seen:
                    seen.add(h)
                    hashes.append(h)
        if hashes:
            self._transport(hashes, dst_worker)
        return hashes
