# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FileKV version catalog: consistency for mid-turn file modifications and
partial (line-range) reads.

Why this layer exists -- and what it does NOT need to solve
-----------------------------------------------------------
The engine-side store is CONTENT-ADDRESSED: a chunk's key is a hash of its
token bytes, and (in sparse/infix mode) the exact-prefix fold additionally
requires a save-time context-chain match. Two consequences do the heavy
lifting for consistency:

  1. **No stale reuse, ever.** A modified file renders to different bytes ->
     different chunk hashes -> the old version's chunks simply cannot match a
     prompt containing the new version. There is no correctness race to fix
     at the engine: reuse can only serve bytes that are literally in the
     prompt.
  2. **Versions coexist safely.** Another agent whose in-flight context still
     embeds the OLD bytes keeps hitting the old chunks -- which is
     semantically CORRECT, because that is what its prompt says. Old chunks
     age out via LRU; a miss just means recompute.

What content addressing does NOT give us is *lifecycle efficiency*, which is
this catalog's job (frontend-side, shared by every agent the frontend
serves):

  * **Invalidation**: a mid-turn edit makes the old version's GPU-staged
    chunks dead weight (they can never match again). The catalog remembers
    which hashes belong to which (path, range, version) so the bridge can
    send an ``evict_hashes`` directive and reclaim staging budget now, not
    at LRU age-out.
  * **Re-warm ("다시 fileKV로 저장")**: saving happens through the engine's
    normal save path -- a warm request over the NEW rendered text. The
    catalog tells the bridge that a version bump happened so it re-renders,
    re-prefetches (whose ``dropped`` reply triggers ``warm_fn``), and the
    next turn hits.
  * **Torn-render detection**: the renderer reads the file while an agent
    may still be writing it. The bridge fingerprints the file BEFORE and
    AFTER rendering (caller-supplied ``fingerprint_fn``); a mismatch means
    the render may be torn -> skip that file this round (prefetch is a hint;
    the post-edit ``on_file_modified`` round covers it).
  * **Multi-agent sharing**: all methods are lock-guarded; agents (threads)
    on one frontend share a single catalog so version bumps from one agent
    invalidate the others' pending intents for the same path.

Partial (line-range) reads
--------------------------
A range read's tool-result text is its OWN byte content -> its own chunks;
the engine needs no special handling. The catalog adds two policies:

  * **Sub-chunk renders are not cached** (< chunk_size tokens): they cannot
    produce a single full chunk on the 256-token grid, and recomputing a few
    hundred tokens costs microseconds -- caching would be pure overhead.
    They are counted and skipped.
  * **Canonical range snapping** (``snap_lines`` > 0): nearby range requests
    (50-100, then 60-110) normally render different bytes and share nothing.
    Snapping the requested range OUTWARD to ``snap_lines`` boundaries (e.g.
    1-100 -> 1-100, 60-110 -> 1-200 with snap=100) makes both map to the
    SAME canonical render -> same chunks -> the second call reuses the
    first's KV. Requires the tool result to actually serve the snapped
    superset (deployment choice); with snapping disabled, exact ranges are
    cached as independent units when they reach chunk size.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RangeKey:
    """Canonical identity of one cached render unit of a file."""

    path: str
    start_line: int | None = None  # None == whole file
    end_line: int | None = None


@dataclass
class FileUnitRecord:
    """One (path, canonical range) unit at one content version."""

    key: RangeKey
    fingerprint: str  # content fingerprint of the FILE at record time
    chunk_hashes: list[str] = field(default_factory=list)
    version: int = 0


def canonicalize_range(
    start_line: int | None,
    end_line: int | None,
    snap_lines: int,
) -> tuple[int | None, int | None]:
    """Snap a 1-based inclusive line range OUTWARD to snap_lines boundaries.

    snap<=0 -> unchanged (exact-range policy). Whole-file (None, None) is
    already canonical. Examples with snap=100: (50,100)->(1,100),
    (60,110)->(1,200), (101,150)->(101,200), (None,80)->(1,100),
    (120,None)->(101,None) (open end stays open).
    """
    if snap_lines <= 0 or (start_line is None and end_line is None):
        return (start_line, end_line)
    s = 1 if start_line is None else max(1, int(start_line))
    snapped_start = ((s - 1) // snap_lines) * snap_lines + 1
    if end_line is None:
        return (snapped_start, None)
    e = max(s, int(end_line))
    snapped_end = ((e + snap_lines - 1) // snap_lines) * snap_lines
    return (snapped_start, snapped_end)


class FileKVCatalog:
    """Thread-safe frontend-side registry: which chunk hashes represent which
    (file, canonical range) at which content version.

    Shared by every agent the frontend serves -- a version bump from one
    agent's edit invalidates the intents of all of them for that path.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # path -> current version counter (bumped by on_file_modified).
        self._versions: dict[str, int] = {}
        # RangeKey -> current FileUnitRecord.
        self._units: dict[RangeKey, FileUnitRecord] = {}

    # ----- versioning -----

    def current_version(self, path: str) -> int:
        with self._lock:
            return self._versions.get(path, 0)

    def on_file_modified(self, path: str) -> list[FileUnitRecord]:
        """Bump the path's version; pop and return every now-stale unit.

        The returned records carry (a) the chunk hashes the bridge should
        send as ``evict_hashes`` (union over records, dedup at the caller)
        and (b) the range keys to re-render at the new version. The unit
        records are removed: the next render re-records under the new
        version.
        """
        with self._lock:
            self._versions[path] = self._versions.get(path, 0) + 1
            popped: list[FileUnitRecord] = []
            for key in [k for k in self._units if k.path == path]:
                popped.append(self._units.pop(key))
            return popped

    # ----- unit records -----

    def record(
        self,
        key: RangeKey,
        fingerprint: str,
        chunk_hashes: list[str],
    ) -> FileUnitRecord:
        """Register (or refresh) a unit's hashes at the CURRENT version.

        Returns the record. If a record for the key exists with a DIFFERENT
        fingerprint, it is replaced (the caller should already have sent
        evictions via on_file_modified; replacing here is the last line of
        defense against a missed modification signal).
        """
        with self._lock:
            rec = FileUnitRecord(
                key=key,
                fingerprint=fingerprint,
                chunk_hashes=list(chunk_hashes),
                version=self._versions.get(key.path, 0),
            )
            self._units[key] = rec
            return rec

    def lookup(self, key: RangeKey) -> FileUnitRecord | None:
        with self._lock:
            return self._units.get(key)

    def is_current(self, key: RangeKey, fingerprint: str) -> bool:
        """Whether the recorded unit matches the given content fingerprint
        (i.e., its chunk hashes are still worth prefetching)."""
        with self._lock:
            rec = self._units.get(key)
            return rec is not None and rec.fingerprint == fingerprint

    def stale_check(self, key: RangeKey, fingerprint: str) -> list[str]:
        """If the recorded unit's fingerprint differs from ``fingerprint``,
        drop it and return its hashes for eviction; else []. Per-unit variant
        of on_file_modified for callers that detect drift at render time."""
        with self._lock:
            rec = self._units.get(key)
            if rec is None or rec.fingerprint == fingerprint:
                return []
            del self._units[key]
            return list(rec.chunk_hashes)

    # ----- introspection -----

    def units_for(self, path: str) -> list[FileUnitRecord]:
        with self._lock:
            return [r for k, r in self._units.items() if k.path == path]

    def __len__(self) -> int:
        with self._lock:
            return len(self._units)
