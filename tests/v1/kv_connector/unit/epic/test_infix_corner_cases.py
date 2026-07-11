# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Infix corner cases, connector-level (CPU).

Segment naming follows the canonical scenario: prompt = A|B|C|D|F|G with
A = cached prefix, B/D/G = new text, C/F = code files in the fileKV store.
Each test drives the REAL scheduler-side connector path
(get_num_new_matched_tokens -> build_connector_meta) and asserts what must
be loaded, what must be recomputed (M), and what must be left alone.
"""

from dataclasses import dataclass, field

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    chain_hash_tokens,
    EpicChunkStore,
    EpicSchedulerIndex,
    StoredChunk,
    hash_chunk_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
    EpicConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    EpicSelection,
    LegoLinkRecompute,
)

BLOCK = 16
CHUNK = 64
LINK = 8


@dataclass
class _NewReq:
    req_id: str
    prompt_token_ids: list[int]
    block_ids: list[list[int]]


@dataclass
class _SchedOut:
    scheduled_new_reqs: list[_NewReq] = field(default_factory=list)
    num_scheduled_tokens: dict = field(default_factory=dict)


@dataclass
class _Req:
    request_id: str
    prompt_token_ids: list[int]


class _LiveIndex(EpicSchedulerIndex):
    """Index proxy over a live worker store (incl. old-pos passthrough, which
    per-run link detection consumes)."""

    def __init__(self, store: EpicChunkStore):
        super().__init__(
            capacity_bytes=10**8,
            num_layers=1,
            num_kv_heads=1,
            head_size=1,
            cache_dtype_size=4,
        )
        self._backing = store

    def contains(self, chunk_hash: str) -> bool:
        return self._backing.contains(chunk_hash) or super().contains(chunk_hash)

    def get_length(self, chunk_hash: str):
        ln = self._backing.get_length(chunk_hash)
        return ln if ln is not None else super().get_length(chunk_hash)

    def get_old_pos_start(self, chunk_hash: str):
        old = self._backing.get_old_pos_start(chunk_hash)
        return old if old is not None else super().get_old_pos_start(chunk_hash)

    def get_chain(self, chunk_hash: str):
        chain = self._backing.get_chain(chunk_hash)
        if chain is not None and chain != (None, None):
            return chain
        return super().get_chain(chunk_hash)


def _connector(store: EpicChunkStore, *, link: int = LINK,
               link_per_run: bool = False):
    c = object.__new__(EpicConnector)
    c._block_size = BLOCK
    c._chunk_size = CHUNK
    c._store = None
    c._index = _LiveIndex(store)
    c._matched_prefix = {}
    c._non_prefix = {}
    c._loads_pending = {}
    c._selections = {}
    c._sparse_reqs = {}
    c._native_computed = {}
    c._max_sparse_rows = 0
    c._long_prefill_threshold = 0
    c._sparse_forward = True
    c._link_tokens = link
    c._selection = EpicSelection(strict_prefix_chain=True)
    c._recompute = LegoLinkRecompute(
        num_link_tokens=link, phase1_dense=False, link_per_run=link_per_run
    )
    c._fusion_enabled = False
    return c


def _store_chunk(
    store: EpicChunkStore,
    tokens: list[int],
    old_start: int = 0,
    save_context: list[int] | None = None,
):
    """Store a chunk; ``save_context`` = the tokens that preceded it in its
    SAVE-TIME prompt (None = legacy chunk with unknown context chain;
    [] = warmed isolated at the prompt start)."""
    h = hash_chunk_tokens(tokens)
    chain_start = chain_end = None
    if save_context is not None:
        chain_start = chain_hash_tokens(save_context)
        chain_end = chain_hash_tokens(save_context + tokens)
    sc = StoredChunk(
        chunk_hash=h,
        length=len(tokens),
        old_positions=torch.arange(
            old_start, old_start + len(tokens), dtype=torch.int64
        ),
        chain_start=chain_start,
        chain_end=chain_end,
    )
    sc.k_per_layer["l0"] = torch.zeros(len(tokens), 1, 1)
    sc.v_per_layer["l0"] = torch.zeros(len(tokens), 1, 1)
    store.put(sc)
    return h


def _blocks(n: int) -> list[list[int]]:
    return [list(range((n + BLOCK - 1) // BLOCK))]


def _seg(base: int) -> list[int]:
    """A distinct CHUNK-token segment."""
    return list(range(base, base + CHUNK))


def _schedule(c, tokens: list[int], nc: int = 0):
    """Run match (+ build) exactly as the scheduler would; return (ext, meta)."""
    n = len(tokens)
    req = _Req("r0", tokens)
    ext, is_async = c.get_num_new_matched_tokens(req, nc)
    assert is_async is False
    if ext:
        # The scheduler registers the pending load after block allocation.
        c.update_state_after_alloc(req, None, ext)
    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _blocks(n))],
        num_scheduled_tokens={"r0": max(1, n - nc - ext)},
    )
    return ext, c.build_connector_meta(sout)


def _m(meta) -> set[int]:
    assert len(meta.sparse) == 1
    return set(meta.sparse[0].sparse_positions)


def _load_positions(meta) -> set[int]:
    out: set[int] = set()
    for load in meta.loads:
        for s in load.chunks:
            out.update(range(s.new_pos_start, s.new_pos_start + s.length))
    return out


# ---------------------------------------------------------------------------
# Corner 1 (the user's example): the native prefix cache already holds A+B+C.
# C must NOT be loaded from fileKV -- and never recomputed either.
# ---------------------------------------------------------------------------


def test_native_prefix_already_covers_c_no_filekv_load():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    a, b, csg, d, f, g = (_seg(i * 1000) for i in range(6))
    _store_chunk(store, csg)  # C is (also) in the fileKV store
    _store_chunk(store, f)
    c = _connector(store)

    tokens = a + b + csg + d + f + g  # N = 384
    nc = 3 * CHUNK  # native prefix cache covers A+B+C (a previous request).
    ext, meta = _schedule(c, tokens, nc=nc)

    # external = native(A+B+C) + F; only F is genuinely new external KV.
    assert c.debug_selection == []  # (counters off; just documenting)
    assert ext == CHUNK  # F only.
    loaded = _load_positions(meta)
    # C region [128, 192) must NOT be loaded from fileKV (native KV is exact).
    assert loaded.isdisjoint(range(2 * CHUNK, 3 * CHUNK))
    # F region [256, 320) IS loaded from fileKV.
    assert loaded == set(range(4 * CHUNK, 5 * CHUNK))
    # M = D + link(F) + G; nothing below the native extent.
    m = _m(meta)
    assert m.isdisjoint(range(0, nc))
    assert set(range(3 * CHUNK, 4 * CHUNK)) <= m  # D
    assert set(range(4 * CHUNK, 4 * CHUNK + LINK)) <= m  # link(F)
    assert (4 * CHUNK + LINK) not in m  # F body reused
    assert set(range(5 * CHUNK, 6 * CHUNK)) <= m  # G
    # Accounting: advance converges num_computed onto N.
    assert meta.sparse[0].computed_advance == len(tokens) - (nc + CHUNK)


# ---------------------------------------------------------------------------
# Corner 2: C sits immediately after the cached prefix (no B). The prefix walk
# folds C into the contiguous prefix -> loaded as prefix, NO link tokens.
# ---------------------------------------------------------------------------


def test_file_adjacent_to_cached_prefix_folds_into_prefix():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    a, csg, d, f, g = (_seg(i * 1000) for i in range(5))
    _store_chunk(store, a)
    _store_chunk(store, csg)
    _store_chunk(store, f)
    c = _connector(store)

    tokens = a + csg + d + f + g  # A|C|D|F|G, N = 320
    ext, meta = _schedule(c, tokens, nc=0)

    # A+C form the contiguous prefix (128) + non-prefix F (64) = 192 external.
    assert ext == 3 * CHUNK
    m = _m(meta)
    # C is prefix now: NO link tokens at C's head, nothing of A/C in M.
    assert m.isdisjoint(range(0, 2 * CHUNK))
    assert set(range(2 * CHUNK, 3 * CHUNK)) <= m  # D
    assert set(range(3 * CHUNK, 3 * CHUNK + LINK)) <= m  # link(F)
    # A and C are loaded as prefix chunks; F as a non-prefix chunk.
    assert _load_positions(meta) == set(range(0, 2 * CHUNK)) | set(
        range(3 * CHUNK, 4 * CHUNK)
    )


# ---------------------------------------------------------------------------
# Corner 3: duplicate file content (C == F). Two hits on the SAME stored
# chunk, loaded at both offsets, each with its own PIC target position.
# ---------------------------------------------------------------------------


def test_duplicate_file_content_loads_at_both_offsets():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    a, b, csg, d, g = (_seg(i * 1000) for i in range(5))
    _store_chunk(store, csg)
    c = _connector(store)

    tokens = a + b + csg + d + csg + g  # C appears twice, N = 384
    ext, meta = _schedule(c, tokens, nc=0)

    assert ext == 2 * CHUNK  # both occurrences count as external.
    loaded = _load_positions(meta)
    assert set(range(2 * CHUNK, 3 * CHUNK)) <= loaded  # first C
    assert set(range(4 * CHUNK, 5 * CHUNK)) <= loaded  # second C
    specs = {s.new_pos_start for load in meta.loads for s in load.chunks}
    assert {2 * CHUNK, 4 * CHUNK} <= specs  # distinct PIC targets.
    m = _m(meta)
    assert set(range(2 * CHUNK, 2 * CHUNK + LINK)) <= m  # link at BOTH heads
    assert set(range(4 * CHUNK, 4 * CHUNK + LINK)) <= m


# ---------------------------------------------------------------------------
# Corner 4: file not aligned to the chunk grid -> no hash hit -> plain dense
# recompute of that file (correct, just no reuse), no sparse plan.
# ---------------------------------------------------------------------------


def test_unaligned_file_degrades_to_dense_recompute():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    csg = _seg(2000)
    _store_chunk(store, csg)  # stored on the grid...
    c = _connector(store)

    a = _seg(0)
    b_half = list(range(1000, 1000 + CHUNK // 2))  # 32 tokens -> C misaligned.
    tokens = a + b_half + csg + _seg(3000)
    ext, meta = _schedule(c, tokens, nc=0)

    assert ext == 0  # C's grid chunks hash differently -> no hit at all.
    assert meta.sparse == []  # dense forward; nothing loaded.
    assert meta.loads == []


# ---------------------------------------------------------------------------
# Corner 5: the native extent cuts INTO C. The load must skip the natively-
# covered head (src_offset) and the link window anchors at the cut point.
# ---------------------------------------------------------------------------


def test_native_extent_straddles_file_c():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    a, b, csg, g = (_seg(i * 1000) for i in range(4))
    _store_chunk(store, csg)
    c = _connector(store)

    tokens = a + b + csg + g  # N = 256, C at [128, 192)
    cut = 2 * CHUNK + BLOCK  # native covers A+B+ first 16 tokens of C.
    ext, meta = _schedule(c, tokens, nc=cut)

    assert ext == CHUNK - BLOCK  # only C's uncovered tail is external.
    specs = [s for load in meta.loads for s in load.chunks]
    assert len(specs) == 1
    assert specs[0].new_pos_start == cut
    assert specs[0].src_offset == BLOCK  # head trim into the stored chunk.
    assert specs[0].length == CHUNK - BLOCK
    m = _m(meta)
    assert m.isdisjoint(range(0, cut))  # native region untouched.
    assert set(range(cut, cut + LINK)) <= m  # link anchored at the cut.


# ---------------------------------------------------------------------------
# Corner 6: the prompt ENDS inside file F (no G). N-1 must still be
# recomputed (first decode step needs a fresh last position).
# ---------------------------------------------------------------------------


def test_prompt_ending_inside_file_recomputes_last_token():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    a, b, f = _seg(0), _seg(1000), _seg(2000)
    _store_chunk(store, f)
    c = _connector(store)

    tokens = a + b + f  # F is last; N-1 sits inside reused F.
    n = len(tokens)
    ext, meta = _schedule(c, tokens, nc=0)

    m = _m(meta)
    assert (n - 1) in m
    pos = meta.sparse[0].sparse_positions
    assert pos == sorted(set(pos)) and pos[-1] == n - 1
    # F body between link head and last token stays reused.
    assert (2 * CHUNK + LINK) not in m


# ---------------------------------------------------------------------------
# Corner 7: exact repeat of a fully-cached prompt -> pure prefix path, no
# sparse plan, no crash (the consistency guard's degenerate case).
# ---------------------------------------------------------------------------


def test_fully_cached_repeat_prompt_is_prefix_only():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    segs = [_seg(i * 1000) for i in range(4)]
    for s in segs:
        _store_chunk(store, s)
    c = _connector(store)

    tokens = sum(segs, []) + [42, 43]  # + trailing partial (never chunked)
    ext, meta = _schedule(c, tokens, nc=0)

    assert ext == 4 * CHUNK  # whole grid-covered prompt is contiguous prefix.
    assert meta.sparse == []  # no non-prefix hit -> dense, guard holds.
    assert _load_positions(meta) == set(range(0, 4 * CHUNK))


# ---------------------------------------------------------------------------
# Corner 8: per-run ("per-file") link tokens. A 2-chunk file saved from ONE
# contiguous warm gets a link window only at its head chunk; broken old-pos
# contiguity (chunks saved separately) falls back to per-chunk links.
# ---------------------------------------------------------------------------


def _two_chunk_file_prompt(store, contiguous_save: bool):
    c1, c2 = _seg(2000), _seg(3000)
    if contiguous_save:
        _store_chunk(store, c1, old_start=0)
        _store_chunk(store, c2, old_start=CHUNK)  # same warm, positions chain.
    else:
        _store_chunk(store, c1, old_start=0)
        _store_chunk(store, c2, old_start=0)  # separate warms, both at 0.
    return _seg(0) + _seg(1000) + c1 + c2 + _seg(4000)  # A|B|C1|C2|G


def test_link_per_run_single_head_for_contiguous_file():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    tokens = _two_chunk_file_prompt(store, contiguous_save=True)
    c = _connector(store, link_per_run=True)
    _, meta = _schedule(c, tokens, nc=0)

    m = _m(meta)
    assert set(range(2 * CHUNK, 2 * CHUNK + LINK)) <= m  # head chunk: link.
    # Run-internal boundary (C2's head) needs NO stitch: saved contiguously.
    assert m.isdisjoint(range(3 * CHUNK, 3 * CHUNK + LINK))


def test_link_per_run_falls_back_when_old_positions_break():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    tokens = _two_chunk_file_prompt(store, contiguous_save=False)
    c = _connector(store, link_per_run=True)
    _, meta = _schedule(c, tokens, nc=0)

    m = _m(meta)
    # Old positions do NOT chain -> boundary coherence unprovable -> stitch.
    assert set(range(2 * CHUNK, 2 * CHUNK + LINK)) <= m
    assert set(range(3 * CHUNK, 3 * CHUNK + LINK)) <= m


def test_per_chunk_default_links_every_chunk_head():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    tokens = _two_chunk_file_prompt(store, contiguous_save=True)
    c = _connector(store, link_per_run=False)  # EPIC original behavior.
    _, meta = _schedule(c, tokens, nc=0)

    m = _m(meta)
    assert set(range(2 * CHUNK, 2 * CHUNK + LINK)) <= m
    assert set(range(3 * CHUNK, 3 * CHUNK + LINK)) <= m


# ---------------------------------------------------------------------------
# Corner 9 (context soundness): A cached, C warmed INDEPENDENTLY (isolated
# fileKV). A+C is contiguous in the prompt, but C's stored KV was computed
# WITHOUT A in context -- it must NOT fold into the exact prefix; it needs
# the EPIC link stitch (non-prefix reuse) instead.
# ---------------------------------------------------------------------------


def test_independent_file_adjacent_to_prefix_is_not_folded():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    a, csg, g = _seg(0), _seg(2000), _seg(4000)
    # A saved as a prompt prefix (context = nothing before it).
    _store_chunk(store, a, save_context=[])
    # C warmed ISOLATED as fileKV: its save-time context is ALSO empty --
    # i.e. computed without A. Content-adjacent != context-continuous.
    _store_chunk(store, csg, save_context=[])
    c = _connector(store)

    tokens = a + csg + g  # A|C|G, C immediately after A.
    ext, meta = _schedule(c, tokens, nc=0)

    # A folds (its save context == this prompt's context at position 0).
    # C must NOT fold: it becomes a non-prefix hit -> loaded + link-stitched.
    assert ext == 2 * CHUNK  # A (prefix) + C (non-prefix) both external.
    m = _m(meta)
    assert m.isdisjoint(range(0, CHUNK))  # A exact, not in M.
    assert set(range(CHUNK, CHUNK + LINK)) <= m  # C head: LINK STITCH.
    assert (CHUNK + LINK) not in m  # C body reused.
    # C is loaded as a NON-prefix chunk (still reused, PIC delta 0 here).
    loaded = _load_positions(meta)
    assert set(range(CHUNK, 2 * CHUNK)) <= loaded


def test_file_saved_with_matching_context_still_folds():
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    a, csg, g = _seg(0), _seg(2000), _seg(4000)
    _store_chunk(store, a, save_context=[])
    # C saved from a previous DENSE run of A+C...: context == A -> its KV is
    # exactly what this prompt's positions [64,128) would compute.
    _store_chunk(store, csg, save_context=list(a))
    c = _connector(store)

    tokens = a + csg + g
    ext, meta = _schedule(c, tokens, nc=0)

    assert ext == 2 * CHUNK  # A+C fold into one exact contiguous prefix.
    assert meta.sparse == []  # no non-prefix hit at all -> dense remainder.
    m_loads = _load_positions(meta)
    assert m_loads == set(range(0, 2 * CHUNK))


def test_context_mismatched_prefix_head_never_folds():
    """Even the FIRST chunk must satisfy the chain: content that matches the
    prompt head but was saved mid-prompt elsewhere is not exact prefix."""
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    a, g = _seg(0), _seg(4000)
    # A's content saved from a DIFFERENT context (it followed 64 other tokens
    # in its save prompt) -> chain_start != digest(empty).
    _store_chunk(store, a, save_context=_seg(9000))
    c = _connector(store)

    tokens = a + g
    ext, meta = _schedule(c, tokens, nc=0)

    # Not foldable; it becomes a non-prefix hit at offset 0 -> link stitch.
    assert ext == CHUNK
    m = _m(meta)
    assert set(range(0, LINK)) <= m  # stitch at the very head.
    assert LINK not in m


def test_legacy_chunks_without_chains_keep_folding():
    """Chunks stored without chain info (legacy/hand-built) keep the lenient
    Phase-1 fold so existing stores stay usable."""
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    a, csg = _seg(0), _seg(2000)
    _store_chunk(store, a)  # save_context=None -> no chains recorded.
    _store_chunk(store, csg)
    c = _connector(store)

    tokens = a + csg + _seg(4000)
    ext, meta = _schedule(c, tokens, nc=0)
    assert ext == 2 * CHUNK
    assert meta.sparse == []  # folded as before (lenient on unknown chains).


def test_save_path_records_chains_end_to_end():
    """The REAL save path (build_connector_meta) must register chains so a
    SECOND request with a different left context does not fold the chunk."""
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c = _connector(store)

    # Request 1: prompt = C1+C2 (a standalone fileKV warm). Saves register
    # chains into the scheduler index (and the EpicReqSave carries them).
    file_tokens = _seg(2000) + _seg(3000)
    ext1, meta1 = _schedule(c, file_tokens, nc=0)
    assert len(meta1.saves) == 1
    assert len(meta1.saves[0].chunk_chains) == 2
    # Mirror what the worker store would hold (chains included).
    for ci, h in enumerate(meta1.saves[0].chunk_hashes):
        cs, ce = meta1.saves[0].chunk_chains[ci]
        sc = StoredChunk(
            chunk_hash=h,
            length=CHUNK,
            old_positions=torch.arange(
                ci * CHUNK, (ci + 1) * CHUNK, dtype=torch.int64
            ),
            chain_start=cs,
            chain_end=ce,
        )
        sc.k_per_layer["l0"] = torch.zeros(CHUNK, 1, 1)
        sc.v_per_layer["l0"] = torch.zeros(CHUNK, 1, 1)
        store.put(sc)

    # Request 2: prompt = A + C1 + C2 + G. C1's save context was EMPTY, the
    # new context is A -> C1/C2 must NOT fold; they are non-prefix hits and
    # (being one contiguous warm) form ONE run for per-run links.
    tokens = _seg(0) + file_tokens + _seg(4000)
    ext2, meta2 = _schedule(c, tokens, nc=0)
    m = _m(meta2)
    assert set(range(CHUNK, CHUNK + LINK)) <= m  # stitch at file head.
    # Per-chunk default: second chunk head stitched too.
    assert set(range(2 * CHUNK, 2 * CHUNK + LINK)) <= m


def test_link_per_run_uses_chain_continuity():
    """Per-run links accept chain proof: two chunks saved from ONE warm
    (prev.chain_end == cur.chain_start) get a single head stitch even when
    judged by chains alone."""
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c1, c2 = _seg(2000), _seg(3000)
    _store_chunk(store, c1, old_start=0, save_context=[])
    _store_chunk(store, c2, old_start=CHUNK, save_context=list(c1))
    c = _connector(store, link_per_run=True)

    tokens = _seg(0) + _seg(1000) + c1 + c2 + _seg(4000)
    _, meta = _schedule(c, tokens, nc=0)
    m = _m(meta)
    assert set(range(2 * CHUNK, 2 * CHUNK + LINK)) <= m  # run head stitched.
    assert m.isdisjoint(range(3 * CHUNK, 3 * CHUNK + LINK))  # internal: none.


def test_link_per_run_chain_mismatch_stitches_both():
    """Adjacent hits whose chains do NOT chain (independent warms) must get
    per-chunk stitches even in per-run mode -- chains override the old-pos
    heuristic (here old positions LOOK contiguous but contexts differ)."""
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    c1, c2 = _seg(2000), _seg(3000)
    _store_chunk(store, c1, old_start=0, save_context=[])
    # c2 saved at old positions [64,128) but after DIFFERENT content.
    _store_chunk(store, c2, old_start=CHUNK, save_context=_seg(9000))
    c = _connector(store, link_per_run=True)

    tokens = _seg(0) + _seg(1000) + c1 + c2 + _seg(4000)
    _, meta = _schedule(c, tokens, nc=0)
    m = _m(meta)
    assert set(range(2 * CHUNK, 2 * CHUNK + LINK)) <= m
    assert set(range(3 * CHUNK, 3 * CHUNK + LINK)) <= m  # stitched: no proof.
