# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline fileKV builder: scan a directory, warm every file, land the KV in
host DRAM through the KVBM interface.

Pipeline (per file)
-------------------
  scan -> render (byte-faithful, same convention as serving) -> tokenize ->
  chunk-grid pad -> hash plan (content hashes + save-time context chains) ->
  warm (one contiguous forward over the whole render; positions 0..N) ->
  ``store.put`` each full chunk into a host-DRAM ``KvbmChunkStore`` ->
  register in the ``FileKVCatalog`` -> manifest entry.

The warm is injected (``warm_fn``): the offline job does not construct an
engine itself. Two deployment shapes:

  * **Engine-backed** (GPU box): ``warm_fn`` wraps a vLLM ``LLM`` built with
    the EpicConnector and ``epic_store_backend: "kvbm"`` -- a 1-token
    ``generate`` per render makes the connector's own save path harvest the
    chunks straight into the KVBM host pool; the builder then only verifies
    membership against its hash plan. Recipe in the CLI epilog / README.
  * **Direct** (tests, or precomputed KV import): ``warm_fn`` returns the
    per-chunk ``StoredChunk`` tensors and the builder puts them itself.

Context invariant (why whole-file warm): each file is ONE contiguous forward
starting at position 0, so intra-file long-range attention is fully captured
and every chunk carries a provable context chain (chunk i's ``chain_start`` ==
digest of tokens[0:start]) -- exactly what the online selection needs for
sound per-run folds. Splitting a file into independently-computed pieces would
destroy that (see the deferred challenge-1 notes).

Re-scan is incremental: a file whose content fingerprint already has a
current catalog record AND whose chunks are still resident is skipped.
"""

import argparse
import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    ChainHasher,
    StoredChunk,
    hash_chunk_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.filekv_catalog import (
    FileKVCatalog,
    RangeKey,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.kvbm_store import (
    KvbmChunkStore,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Directory names never worth warming (VCS internals, caches, envs).
DEFAULT_EXCLUDE_DIRS = frozenset(
    {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
     ".mypy_cache", ".ruff_cache", ".pytest_cache", ".idea", ".vscode"})
DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024


def default_render(path: str, text: str) -> str:
    """Path-header render. MUST byte-match how the serving side embeds file
    reads into prompts (same convention FileKVPrefetcher documents: rendering
    fidelity is the caller's responsibility) -- override ``render_fn`` when
    the deployment's tool-result format differs."""
    return f"# file: {path}\n{text}"


def file_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def scan_directory(
    root: str | Path,
    include: Sequence[str] = ("**/*",),
    exclude: Sequence[str] = (),
    exclude_dirs: Iterable[str] = DEFAULT_EXCLUDE_DIRS,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> list[Path]:
    """Deterministic (sorted, relative-path order) list of warmable files.

    Skips excluded directories, non-files, oversized files, and binary
    content (null-byte sniff). ``include``/``exclude`` are glob patterns
    matched against the path relative to ``root``.
    """
    root = Path(root).resolve()
    exclude_dirs = set(exclude_dirs)
    picked: dict[Path, None] = {}
    for pattern in include:
        for p in sorted(root.glob(pattern)):
            rel = p.relative_to(root)
            if not p.is_file() or p.is_symlink():
                continue
            if any(part in exclude_dirs for part in rel.parts):
                continue
            if any(rel.match(pat) for pat in exclude):
                continue
            try:
                if p.stat().st_size > max_file_bytes:
                    logger.warning("filekv scan: %s exceeds %d bytes; skipped",
                                   rel, max_file_bytes)
                    continue
                head = p.read_bytes()
            except OSError as e:
                logger.warning("filekv scan: cannot read %s: %s", rel, e)
                continue
            if _is_binary(head):
                continue
            picked[p] = None
    return sorted(picked, key=lambda p: str(p.relative_to(root)))


# ---------------------------------------------------------------------------
# Hash plan (CPU-only prediction of what the warm must produce)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkPlan:
    start: int  # token offset of the chunk in the padded render
    length: int
    chunk_hash: str
    chain_start: str  # digest of tokens[0:start]  (save-time context chain)
    chain_end: str  # digest of tokens[0:start+length]


def plan_chunks(
    token_ids: Sequence[int], chunk_size: int, pad_token_id: int
) -> tuple[list[int], list[ChunkPlan]]:
    """Pad to the chunk grid and compute the content-hash + chain plan.

    Mirrors the connector's grid exactly (same pad convention as
    ``FileKVPrefetcher.chunk_hashes_for_tokens``); the chains are what the
    connector's save path would stamp for a prompt == this render.
    """
    ids = list(token_ids)
    if not ids:
        return [], []
    rem = len(ids) % chunk_size
    if rem:
        ids.extend([pad_token_id] * (chunk_size - rem))
    plans: list[ChunkPlan] = []
    hasher = ChainHasher()
    for start in range(0, len(ids), chunk_size):
        chain_before = hasher.digest()
        window = ids[start:start + chunk_size]
        hasher.update(window)
        plans.append(
            ChunkPlan(
                start=start,
                length=chunk_size,
                chunk_hash=hash_chunk_tokens(window),
                chain_start=chain_before,
                chain_end=hasher.digest(),
            ))
    return ids, plans


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

# Returns the warmed per-chunk KV for the (padded) token ids of one render, in
# chunk order. None == "the warm landed the chunks in the store itself"
# (engine-backed shape) -- the builder then verifies membership only.
WarmFn = Callable[[list[int], list[ChunkPlan]], list[StoredChunk] | None]


@dataclass
class FileBuildResult:
    path: str
    fingerprint: str
    token_count: int  # pre-pad token count
    chunk_hashes: list[str] = field(default_factory=list)
    chains: list[tuple[str, str]] = field(default_factory=list)
    skipped: bool = False  # already current + resident
    error: str | None = None


class OfflineFileKVBuilder:
    """Scans files into host-DRAM fileKV chunks via the KVBM store."""

    def __init__(
        self,
        *,
        store: KvbmChunkStore,
        tokenize_fn: Callable[[str], list[int]],
        warm_fn: WarmFn,
        catalog: FileKVCatalog | None = None,
        render_fn: Callable[[str, str], str] = default_render,
        chunk_size: int = 256,
        pad_token_id: int = 0,
    ):
        self._store = store
        self._tokenize = tokenize_fn
        self._warm = warm_fn
        self._catalog = catalog or FileKVCatalog()
        self._render = render_fn
        self._chunk_size = int(chunk_size)
        self._pad_token_id = int(pad_token_id)

    @property
    def catalog(self) -> FileKVCatalog:
        return self._catalog

    def build_file(self, root: Path, path: Path) -> FileBuildResult:
        rel = str(path.relative_to(root))
        data = path.read_bytes()
        fp = file_fingerprint(data)
        key = RangeKey(path=rel)  # whole-file unit

        # Incremental skip: current fingerprint AND every chunk still resident.
        rec = self._catalog.lookup(key)
        if (rec is not None and rec.fingerprint == fp
                and all(self._store.contains(h) for h in rec.chunk_hashes)):
            return FileBuildResult(path=rel, fingerprint=fp,
                                   token_count=0,
                                   chunk_hashes=list(rec.chunk_hashes),
                                   skipped=True)

        text = data.decode("utf-8", errors="replace")
        ids = self._tokenize(self._render(rel, text))
        padded, plans = plan_chunks(ids, self._chunk_size, self._pad_token_id)
        if not plans:
            return FileBuildResult(path=rel, fingerprint=fp, token_count=0)

        chunks = self._warm(padded, plans)
        if chunks is not None:
            if len(chunks) != len(plans):
                return FileBuildResult(
                    path=rel, fingerprint=fp, token_count=len(ids),
                    error=(f"warm returned {len(chunks)} chunks, "
                           f"plan expects {len(plans)}"))
            for chunk, plan in zip(chunks, plans):
                # The plan is the oracle: hashes/chains are computed from the
                # tokens, never trusted from the warm side.
                chunk.chunk_hash = plan.chunk_hash
                chunk.chain_start = plan.chain_start
                chunk.chain_end = plan.chain_end
                self._store.put(chunk)

        missing = [p.chunk_hash for p in plans
                   if not self._store.contains(p.chunk_hash)]
        if missing:
            return FileBuildResult(
                path=rel, fingerprint=fp, token_count=len(ids),
                error=f"{len(missing)}/{len(plans)} chunks not resident "
                      "after warm (store budget too small, or the engine-side "
                      "save missed the render)")

        self._catalog.record(key, fp, [p.chunk_hash for p in plans])
        return FileBuildResult(
            path=rel, fingerprint=fp, token_count=len(ids),
            chunk_hashes=[p.chunk_hash for p in plans],
            chains=[(p.chain_start, p.chain_end) for p in plans])

    def build_directory(
        self,
        root: str | Path,
        include: Sequence[str] = ("**/*",),
        exclude: Sequence[str] = (),
    ) -> list[FileBuildResult]:
        root = Path(root).resolve()
        results = []
        for path in scan_directory(root, include=include, exclude=exclude):
            try:
                results.append(self.build_file(root, path))
            except Exception as e:  # noqa: BLE001 -- one bad file != abort scan
                logger.warning("filekv build failed for %s: %s", path, e)
                results.append(
                    FileBuildResult(path=str(path.relative_to(root)),
                                    fingerprint="", token_count=0,
                                    error=str(e)))
        return results


def write_manifest(results: list[FileBuildResult], out_path: str | Path,
                   *, root: str, chunk_size: int) -> None:
    """Manifest the dynamo frontend consumes to prefetch without re-rendering:
    path -> fingerprint + chunk hashes (+ chains for run-coherence checks)."""
    doc = {
        "root": root,
        "chunk_size": chunk_size,
        "files": [
            {
                "path": r.path,
                "fingerprint": r.fingerprint,
                "token_count": r.token_count,
                "chunk_hashes": r.chunk_hashes,
                "chains": r.chains,
                "skipped": r.skipped,
                "error": r.error,
            }
            for r in results
        ],
    }
    Path(out_path).write_text(json.dumps(doc, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "GPU engine recipe (real KV): build a vLLM LLM with\n"
            "  kv_transfer_config = KVTransferConfig(\n"
            "      kv_connector='EpicConnector', kv_role='kv_both',\n"
            "      kv_connector_extra_config={'epic_store_backend': 'kvbm',\n"
            "                                 'epic_kvbm_pool': 'pinned'})\n"
            "then warm_fn = one llm.generate(render, max_tokens=1) per file\n"
            "returning None (the connector's save path lands the chunks).\n"
            "With the full Dynamo stack, size the host tier via\n"
            "DYN_KVBM_CPU_CACHE_GB and pass 'epic_kvbm_pool': 'dynamo'."))
    ap.add_argument("--dir", required=True, help="directory to scan")
    ap.add_argument("--include", default="**/*",
                    help="comma-separated glob patterns (relative to --dir)")
    ap.add_argument("--exclude", default="",
                    help="comma-separated glob patterns to skip")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--pad-token-id", type=int, default=0)
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer name/path; MUST be the serving "
                         "tokenizer or the hashes will never match online")
    ap.add_argument("--manifest-out", default="filekv_manifest.json")
    ap.add_argument("--host-pool-bytes", type=int, default=8 * 1024**3)
    ap.add_argument("--dry-run", action="store_true",
                    help="scan + hash plan + manifest only; no KV, no store")
    args = ap.parse_args()

    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenize = lambda text: tok(text, add_special_tokens=False)["input_ids"]  # noqa: E731
    else:
        logger.warning(
            "no --tokenizer: whitespace fallback in use; hashes will only "
            "match a serving stack using the same fallback (dry-run/dev only)")
        tokenize = lambda text: [  # noqa: E731
            int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "little")
            for w in text.split()
        ]

    from vllm.distributed.kv_transfer.kv_connector.v1.epic.kvbm_store import (
        PinnedHostPool,
    )
    pool = PinnedHostPool(capacity_bytes=args.host_pool_bytes)
    store = KvbmChunkStore(pool=pool, capacity_bytes=args.host_pool_bytes)

    if args.dry_run:
        # Plan-only warm: nothing stored, manifest still exact.
        def warm(padded: list[int], plans: list[ChunkPlan]):
            return []

        # Bypass residency verification in dry-run by planning directly.
        results = []
        root = Path(args.dir).resolve()
        for path in scan_directory(
                root,
                include=args.include.split(","),
                exclude=[p for p in args.exclude.split(",") if p]):
            rel = str(path.relative_to(root))
            data = path.read_bytes()
            ids = tokenize(default_render(rel,
                                          data.decode("utf-8",
                                                      errors="replace")))
            _, plans = plan_chunks(ids, args.chunk_size, args.pad_token_id)
            results.append(
                FileBuildResult(
                    path=rel, fingerprint=file_fingerprint(data),
                    token_count=len(ids),
                    chunk_hashes=[p.chunk_hash for p in plans],
                    chains=[(p.chain_start, p.chain_end) for p in plans]))
    else:
        raise SystemExit(
            "non-dry-run offline builds need an engine-backed warm_fn (GPU); "
            "use the recipe in --help, or drive OfflineFileKVBuilder from "
            "Python with your warm_fn. --dry-run works everywhere.")

    write_manifest(results, args.manifest_out, root=str(Path(args.dir).resolve()),
                   chunk_size=args.chunk_size)
    built = sum(1 for r in results if r.chunk_hashes and not r.skipped)
    print(f"planned {built} files, {sum(len(r.chunk_hashes) for r in results)} "
          f"chunks -> {args.manifest_out}")


if __name__ == "__main__":
    main()
