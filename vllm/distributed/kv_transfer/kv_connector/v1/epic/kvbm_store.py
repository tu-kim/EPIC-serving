# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVBM-backed fileKV storage: EPIC chunks resident in host DRAM.

Offline fileKV builds (filekv_offline.py) must land the chunk KV in **host
DRAM** so any worker on the node can onboard it later without touching a GPU.
The Dynamo stack manages exactly that tier -- KVBM's G2 pool (pinned CPU
memory, sized by ``DYN_KVBM_CPU_CACHE_GB``). This module bridges EPIC's
content-addressed chunk store onto such a pool:

  * ``HostPool`` -- the narrow byte-blob protocol this module needs from any
    host-DRAM block manager: content-key -> payload put/get/contains/evict.
  * ``PinnedHostPool`` -- a self-contained implementation holding payloads in
    *pinned* host-DRAM buffers (``torch.empty(pin_memory=True)``). This is the
    default and the CPU-test backend; pinned residency means a later onboard
    to any GPU is a straight DMA.
  * ``DynamoKvbmHostPool`` -- adapter over the ``kvbm`` package's host pool.
    KVBM's *public* Python surface today is the vLLM connector
    (``kvbm.vllm_integration.connector``) plus env-var sizing; a raw
    put/get-bytes API is not part of that contract, so this adapter is
    dependency-injected: the deployment hands it any object exposing the four
    ``HostPool`` methods (a thin wrapper over KVBM's block-pool bindings).
    ``from_env()`` attempts the known binding names and fails loudly with
    guidance instead of silently degrading.
  * ``KvbmChunkStore`` -- duck-compatible with ``EpicChunkStore`` (the
    ``SupportsChunkMembership`` protocol plus get/put/iter_membership), but
    tensor payloads live in the pool as serialized blobs while a local
    metadata index answers scheduler-side membership queries with ZERO
    deserialization. Select the backend with the connector extra-config
    ``epic_store_backend: "kvbm"``.

Serialization: one ``torch.save`` blob per chunk (tensors + str/None chain
fields only -> ``weights_only=True`` round-trip). K/V tensors are re-pinned on
deserialize when ``pin_memory`` is set so the H2D path keeps its DMA speed.
"""

import io
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    StoredChunk,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Host-DRAM pool protocol + implementations
# ---------------------------------------------------------------------------


class HostPool(Protocol):
    """Byte-blob storage in host DRAM, content-addressed by chunk hash."""

    def put(self, key: str, payload: bytes) -> bool:
        """Store payload; False == refused (over capacity)."""
        ...

    def get(self, key: str) -> bytes | None: ...

    def contains(self, key: str) -> bool: ...

    def evict(self, key: str) -> bool:
        """Drop the payload; True if it was present."""
        ...


class PinnedHostPool:
    """Host-DRAM pool holding each payload in a pinned uint8 buffer.

    Pinned residency is the point: fileKV built offline sits DMA-ready for
    whichever GPU worker onboards it later. Byte-budgeted but NOT an LRU --
    eviction ordering is the owning ``KvbmChunkStore``'s job (it mirrors the
    exact ``EpicChunkStore`` policy); the pool only enforces the hard cap.
    """

    def __init__(self, capacity_bytes: int, pin_memory: bool = True):
        self.capacity_bytes = int(capacity_bytes)
        self._pin = pin_memory
        self._blobs: dict[str, torch.Tensor] = {}
        self._used = 0

    def put(self, key: str, payload: bytes) -> bool:
        old = self._blobs.pop(key, None)
        if old is not None:
            self._used -= old.numel()
        if self._used + len(payload) > self.capacity_bytes:
            if old is not None:  # refuse, but don't lose accounting
                self._blobs[key] = old
                self._used += old.numel()
            return False
        try:
            buf = torch.empty(len(payload), dtype=torch.uint8,
                              pin_memory=self._pin)
        except (RuntimeError, NotImplementedError):
            buf = torch.empty(len(payload), dtype=torch.uint8)
        buf.copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
        self._blobs[key] = buf
        self._used += buf.numel()
        return True

    def get(self, key: str) -> bytes | None:
        blob = self._blobs.get(key)
        return None if blob is None else blob.numpy().tobytes()

    def contains(self, key: str) -> bool:
        return key in self._blobs

    def evict(self, key: str) -> bool:
        blob = self._blobs.pop(key, None)
        if blob is None:
            return False
        self._used -= blob.numel()
        return True

    @property
    def used_bytes(self) -> int:
        return self._used


class DynamoKvbmHostPool:
    """Adapter binding ``HostPool`` onto a KVBM host-tier object.

    Dependency-injected: ``pool_obj`` is any object exposing put/get/contains/
    evict for byte payloads (the deployment's thin wrapper over KVBM's G2
    block-pool bindings). Method-name indirection keeps this adapter honest
    about KVBM's API surface instead of hard-coding calls that may not exist
    in the installed version.
    """

    _METHODS = ("put", "get", "contains", "evict")

    def __init__(self, pool_obj: Any):
        missing = [m for m in self._METHODS if not callable(
            getattr(pool_obj, m, None))]
        if missing:
            raise TypeError(
                f"KVBM pool object lacks required methods {missing}; wrap the "
                "kvbm host-pool bindings in an object exposing "
                "put(key, bytes)->bool / get(key)->bytes|None / "
                "contains(key)->bool / evict(key)->bool")
        self._pool = pool_obj

    def put(self, key: str, payload: bytes) -> bool:
        return bool(self._pool.put(key, payload))

    def get(self, key: str) -> bytes | None:
        return self._pool.get(key)

    def contains(self, key: str) -> bool:
        return bool(self._pool.contains(key))

    def evict(self, key: str) -> bool:
        return bool(self._pool.evict(key))

    @classmethod
    def from_env(cls) -> "DynamoKvbmHostPool":
        """Best-effort construction from an installed ``kvbm`` package.

        KVBM's supported integration is the vLLM connector + env sizing
        (``DYN_KVBM_CPU_CACHE_GB``); a standalone host-pool handle is not a
        stable public API. We probe the known binding names and raise with
        actionable guidance otherwise -- callers then either inject their own
        wrapper (``DynamoKvbmHostPool(my_wrapper)``) or fall back to
        ``PinnedHostPool`` (same host-DRAM semantics, process-local).
        """
        try:
            import kvbm  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "kvbm is not installed (pip install kvbm). For a "
                "process-local host-DRAM pool use PinnedHostPool instead."
            ) from e
        for name in ("HostBlockPool", "host_pool", "BlockPool"):
            obj = getattr(kvbm, name, None)
            if obj is not None:
                pool = obj() if callable(obj) else obj
                return cls(pool)
        raise NotImplementedError(
            "installed kvbm exposes no standalone host-pool binding; wrap "
            "its block-pool API in an object with put/get/contains/evict "
            "and pass it to DynamoKvbmHostPool(...) directly")


def build_host_pool(extra: dict) -> HostPool:
    """Pool factory for the connector / offline builder extra-config.

    ``epic_kvbm_pool``: "pinned" (default) -> PinnedHostPool sized by
    ``epic_cpu_bytes``; "dynamo" -> DynamoKvbmHostPool.from_env(); or an
    already-constructed pool object (tests / deployment wrappers).
    """
    spec = extra.get("epic_kvbm_pool", "pinned")
    if not isinstance(spec, str):
        return DynamoKvbmHostPool(spec)
    if spec == "dynamo":
        return DynamoKvbmHostPool.from_env()
    if spec == "pinned":
        from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (  # noqa: E501
            DEFAULT_CAPACITY_BYTES,
        )
        return PinnedHostPool(
            capacity_bytes=int(extra.get("epic_cpu_bytes",
                                         DEFAULT_CAPACITY_BYTES)),
            pin_memory=bool(extra.get("epic_pin_memory", True)),
        )
    raise ValueError(f"unknown epic_kvbm_pool spec: {spec!r}")


# ---------------------------------------------------------------------------
# EpicChunkStore-compatible adapter
# ---------------------------------------------------------------------------


@dataclass
class _Meta:
    """Local per-chunk metadata: answers membership without touching blobs."""

    length: int
    nbytes: int
    old_pos_start: int
    chain_start: str | None
    chain_end: str | None


def _serialize(chunk: StoredChunk) -> bytes:
    buf = io.BytesIO()
    torch.save(
        {
            "chunk_hash": chunk.chunk_hash,
            "length": chunk.length,
            "old_positions": chunk.old_positions,
            "k_per_layer": chunk.k_per_layer,
            "v_per_layer": chunk.v_per_layer,
            "chain_start": chunk.chain_start,
            "chain_end": chunk.chain_end,
        },
        buf,
    )
    return buf.getvalue()


def _deserialize(payload: bytes, pin: bool) -> StoredChunk:
    d = torch.load(io.BytesIO(payload), weights_only=True)

    def _maybe_pin(t: torch.Tensor) -> torch.Tensor:
        if pin and t.device.type == "cpu" and not t.is_pinned():
            try:
                return t.pin_memory()
            except (RuntimeError, NotImplementedError):
                return t
        return t

    return StoredChunk(
        chunk_hash=d["chunk_hash"],
        length=d["length"],
        old_positions=d["old_positions"],
        k_per_layer={n: _maybe_pin(t) for n, t in d["k_per_layer"].items()},
        v_per_layer={n: _maybe_pin(t) for n, t in d["v_per_layer"].items()},
        chain_start=d["chain_start"],
        chain_end=d["chain_end"],
    )


class KvbmChunkStore:
    """EPIC chunk store whose tensor payloads live in a host-DRAM pool.

    Duck-compatible with ``EpicChunkStore`` everywhere the connector and the
    selection strategies touch it: SupportsChunkMembership (contains /
    get_length) plus get_old_pos_start / get_chain / get / put /
    iter_membership / current_bytes / __len__ / maybe_pin. Same LRU + byte
    budget + oversize-refusal semantics, so the scheduler-side
    ``EpicSchedulerIndex`` mirror stays deterministic regardless of backend.
    """

    def __init__(self, pool: HostPool, capacity_bytes: int,
                 pin_memory: bool = True):
        self.capacity_bytes = int(capacity_bytes)
        self.pin_memory = pin_memory
        self._pool = pool
        self._meta: "OrderedDict[str, _Meta]" = OrderedDict()
        self._cur_bytes = 0

    # ----- lookup (scheduler side; metadata only, no blob touch) -----

    def contains(self, chunk_hash: str) -> bool:
        return chunk_hash in self._meta

    def get_length(self, chunk_hash: str) -> int | None:
        m = self._meta.get(chunk_hash)
        return None if m is None else m.length

    def get_old_pos_start(self, chunk_hash: str) -> int | None:
        m = self._meta.get(chunk_hash)
        if m is None or m.old_pos_start < 0:
            return None
        return m.old_pos_start

    def get_chain(self, chunk_hash: str) -> tuple[str | None, str | None] | None:
        m = self._meta.get(chunk_hash)
        if m is None:
            return None
        return (m.chain_start, m.chain_end)

    # ----- read (worker side; marks as recently used) -----

    def get(self, chunk_hash: str) -> StoredChunk | None:
        m = self._meta.get(chunk_hash)
        if m is None:
            return None
        payload = self._pool.get(chunk_hash)
        if payload is None:
            # Pool lost the blob behind our back (external eviction, e.g. a
            # shared KVBM pool under pressure) -> drop the stale index entry.
            logger.warning(
                "KvbmChunkStore: pool dropped %s behind the index; evicting",
                chunk_hash)
            self._meta.pop(chunk_hash)
            self._cur_bytes -= m.nbytes
            return None
        self._meta.move_to_end(chunk_hash)
        return _deserialize(payload, pin=self.pin_memory)

    # ----- write (offline builder / worker save path) -----

    def put(self, chunk: StoredChunk) -> None:
        payload = _serialize(chunk)
        nbytes = len(payload)
        if nbytes > self.capacity_bytes:
            return  # oversize: same refusal as EpicChunkStore
        old = self._meta.pop(chunk.chunk_hash, None)
        if old is not None:
            self._pool.evict(chunk.chunk_hash)
            self._cur_bytes -= old.nbytes
        # Make room FIRST (LRU), so the pool put cannot fail on capacity.
        while self._cur_bytes + nbytes > self.capacity_bytes and self._meta:
            evicted_hash, evicted = self._meta.popitem(last=False)
            self._pool.evict(evicted_hash)
            self._cur_bytes -= evicted.nbytes
        if not self._pool.put(chunk.chunk_hash, payload):
            logger.warning(
                "KvbmChunkStore: pool refused %s (%d bytes); chunk not stored",
                chunk.chunk_hash, nbytes)
            return
        old_pos = (int(chunk.old_positions[0].item())
                   if chunk.old_positions.numel() > 0 else -1)
        self._meta[chunk.chunk_hash] = _Meta(
            length=chunk.length,
            nbytes=nbytes,
            old_pos_start=old_pos,
            chain_start=chunk.chain_start,
            chain_end=chunk.chain_end,
        )
        self._cur_bytes += nbytes

    # ----- parity helpers (mirror EpicChunkStore) -----

    def maybe_pin(self, t: torch.Tensor) -> torch.Tensor:
        if self.pin_memory and t.device.type == "cpu" and not t.is_pinned():
            try:
                return t.pin_memory()
            except (RuntimeError, NotImplementedError):
                return t
        return t

    @property
    def current_bytes(self) -> int:
        return self._cur_bytes

    def __len__(self) -> int:
        return len(self._meta)

    def iter_membership(self):
        for h, m in self._meta.items():
            yield h, m.length, m.old_pos_start, m.chain_start, m.chain_end
