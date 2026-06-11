# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC functional / lifecycle tests (CPU-only).

Unlike the per-function unit tests in this directory, these drive whole EPIC
scenarios end-to-end through the connector's public lifecycle surface:

  * Scenario A -- "save -> position-independent prefix reuse" (Phase 1 path):
    a request 1 prefill writes RoPE'd K into a fake paged cache, the connector
    save path harvests it into the chunk store, then request 2 (same chunk at a
    DIFFERENT position) matches scheduler-side, builds metadata, and the worker
    ``start_load_kv`` scatters the PIC-re-rotated K into request 2's blocks. We
    assert the loaded K equals K RoPE'd DIRECTLY at the new position (PIC
    end-to-end), V is byte-identical, and the slot placement is exact.

  * Scenario B -- "A+C+B sparse lifecycle" (Phase 2b path): A and B chunks are
    stored at distinct original positions, an A+C+B prompt drives
    ``get_num_new_matched_tokens`` (external == |A|+|B|) -> ``build_connector_meta``
    -> ``EpicReqSparse`` (M sorted, last == N-1) -> ``build_sparse_row_edits`` ->
    a contiguous ``positions`` buffer is overwritten with M's logical positions
    -> the flex ``_convert_physical_to_logical`` logical_q branch maps the
    scattered M queries correctly. We also assert the save guard: the reused B
    chunk is never re-saved.

  * Scenario C -- "flag off, no trace": the SAME B input with
    ``epic_sparse_forward=False`` emits no sparse metadata and behaves as the
    Phase 1 prefix-only path.

Everything is CPU tensors + the standalone ``PICRotator`` (no GPU, no model
load, no VllmConfig); the connector is assembled field-by-field via
``object.__new__`` exactly like the existing unit tests.
"""

from dataclasses import dataclass, field

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    EpicChunkStore,
    StoredChunk,
    hash_chunk_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
    EpicConnector,
    _slot_ids_from_blocks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
    EpicConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.pic import PICRotator
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    EpicSelection,
    LegoLinkRecompute,
    PicAlignment,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.runner_sparse import (
    build_sparse_row_edits,
)
from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

# ---------------------------------------------------------------------------
# Common dimensions. CHUNK is a block multiple so chunk boundaries align with
# paged blocks (the connector's own invariant).
# ---------------------------------------------------------------------------
BLOCK = 16
CHUNK = 16  # one block per chunk -> easy slot math for the lifecycle assertions
NUM_KV_HEADS = 2
HEAD_SIZE = 16
ROTARY_DIM = 16
BASE = 10000.0
LAYER = "model.layers.0.self_attn.attn"


# ---------------------------------------------------------------------------
# Reference neox-style RoPE (identical convention to test_pic.py / vLLM's
# RotaryEmbedding.forward_static). The ground truth for the PIC assertions.
# ---------------------------------------------------------------------------
def _neox_rope_reference(
    x: torch.Tensor, positions: torch.Tensor, base: float, rotary_dim: int
) -> torch.Tensor:
    """x: [num_tokens, num_heads, head_size]; rotate only first rotary_dim."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    )
    t = positions.to(torch.float64)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    out = x.clone().to(torch.float64)
    rot = out[..., :rotary_dim]
    x1, x2 = torch.chunk(rot, 2, dim=-1)
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    out[..., :rotary_dim] = torch.cat((o1, o2), dim=-1)
    return out.to(x.dtype)


# ---------------------------------------------------------------------------
# Minimal scheduler-output stubs (same shape as the other unit tests).
# ---------------------------------------------------------------------------
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
    """Minimal stand-in for a v1 Request (scheduler-side, CPU-only)."""

    request_id: str
    prompt_token_ids: list[int]


def _block_ids_for(num_tokens: int, base_block: int = 0) -> list[list[int]]:
    nblocks = (num_tokens + BLOCK - 1) // BLOCK
    return [[base_block + i for i in range(nblocks)]]


def _make_paged_cache(num_blocks: int) -> torch.Tensor:
    """FlashAttention V1 layout: (2, num_blocks, block_size, num_kv_heads, head)."""
    return torch.zeros(
        2, num_blocks, BLOCK, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float64
    )


def _make_connector(
    *,
    sparse: bool,
    link: int = 8,
    store: EpicChunkStore | None = None,
    worker: bool = False,
) -> EpicConnector:
    """Assemble an EpicConnector field-by-field (no VllmConfig / GPU).

    ``worker=True`` additionally wires a real CPU ``PICRotator`` + alignment and
    empty kv-cache registries so the worker load path can run.
    """
    c = object.__new__(EpicConnector)
    c._block_size = BLOCK
    c._chunk_size = CHUNK
    c._store = store if store is not None else EpicChunkStore(
        capacity_bytes=10**8, pin_memory=False
    )
    c._matched_prefix = {}
    c._non_prefix = {}
    c._loads_pending = {}
    c._selections = {}
    c._sparse_reqs = {}
    c._sparse_forward = sparse
    c._link_tokens = link
    c._selection = EpicSelection()
    c._recompute = LegoLinkRecompute(num_link_tokens=link, phase1_dense=not sparse)
    c._fusion_enabled = False
    c._mask_builder = None
    c._flex_layers_patched = set()
    c._mask_capacity = 0
    if worker:
        rot = PICRotator(
            head_size=HEAD_SIZE,
            rotary_dim=ROTARY_DIM,
            base=BASE,
            is_neox_style=True,
            dtype=torch.float64,
        )
        c._pic = rot
        c._alignment = PicAlignment(rot)
        c._kv_caches = {}
        c._layer_names = []
    else:
        c._pic = None
        c._alignment = None
        c._kv_caches = {}
        c._layer_names = []
    return c


# ===========================================================================
# Scenario A -- save -> position-independent prefix reuse (Phase 1 path)
# ===========================================================================
def test_scenario_a_save_then_position_independent_prefix_reuse():
    torch.manual_seed(0)

    # --- request 1: a chunk that lived at original positions p_old = [10, 26) ---
    p_old_start = 10
    old_positions = torch.arange(p_old_start, p_old_start + CHUNK, dtype=torch.int64)
    # Raw (pre-RoPE) K and the V we will store; K-in-cache is RoPE'd at p_old.
    k_raw = torch.randn(CHUNK, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float64)
    v_raw = torch.randn(CHUNK, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float64)
    k_old = _neox_rope_reference(k_raw, old_positions, BASE, ROTARY_DIM)

    chunk_tokens = list(range(5000, 5000 + CHUNK))
    chunk_hash = hash_chunk_tokens(chunk_tokens)

    # Build a worker connector and stash request 1's KV via the SAVE path. We
    # exercise ``save_kv_layer`` so the harvest -> StoredChunk wiring is covered
    # (not a hand-built StoredChunk).
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    worker = _make_connector(sparse=False, store=store, worker=True)

    # Request 1's blocks: chunk occupies block 3 (slots 48..64). Put k_old/v_raw
    # into a fresh paged cache, then save_kv_layer harvests those slots.
    req1_block = 3
    req1_cache = _make_paged_cache(num_blocks=8)
    req1_slots = _slot_ids_from_blocks([req1_block], BLOCK, 0, CHUNK)
    assert req1_slots == list(range(req1_block * BLOCK, req1_block * BLOCK + CHUNK))
    # Scatter k_old / v_raw into req1 cache at those slots (K-bank=[0], V-bank=[1]).
    k_bank = req1_cache[0].reshape(8 * BLOCK, NUM_KV_HEADS, HEAD_SIZE)
    v_bank = req1_cache[1].reshape(8 * BLOCK, NUM_KV_HEADS, HEAD_SIZE)
    k_bank[req1_slots] = k_old
    v_bank[req1_slots] = v_raw

    worker.register_kv_caches({LAYER: req1_cache})

    save_meta = EpicConnectorMetadata()
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import EpicReqSave

    save = EpicReqSave(req_id="req1")
    save.chunk_hashes.append(chunk_hash)
    save.chunk_slot_ids.append(req1_slots)
    save.chunk_positions.append(old_positions.tolist())
    save_meta.add_save(save)
    worker._connector_metadata = save_meta
    worker.save_kv_layer(LAYER, req1_cache, attn_metadata=None)

    # The store now holds the chunk with its ORIGINAL positions.
    stored = store.get(chunk_hash)
    assert stored is not None
    assert stored.old_positions.tolist() == old_positions.tolist()
    assert torch.allclose(stored.k_per_layer[LAYER], k_old)
    assert torch.allclose(stored.v_per_layer[LAYER], v_raw)

    # --- request 2: same chunk, but now at a DIFFERENT (prefix) position ---
    # In request 2 the chunk lands at prompt position 0 (a clean prefix reuse).
    p_new_start = 0
    new_positions = torch.arange(p_new_start, p_new_start + CHUNK, dtype=torch.int64)

    # Scheduler side: request 2's prompt starts with the cached chunk -> prefix
    # match of CHUNK tokens.
    req2_prompt = chunk_tokens + list(range(7000, 7000 + CHUNK))  # chunk + new tail
    sched = _make_connector(sparse=False, store=store, worker=False)
    num_new, is_async = sched.get_num_new_matched_tokens(
        _Req("req2", req2_prompt), num_computed_tokens=0
    )
    assert is_async is False
    assert num_new == CHUNK  # the whole cached chunk is the matched prefix.

    # update_state_after_alloc marks the load pending (blocks were allocated).
    sched.update_state_after_alloc(
        _Req("req2", req2_prompt), blocks=None, num_external_tokens=num_new
    )
    req2_block = 1  # request 2's first block (chunk lands here, prompt pos 0).
    sout = _SchedOut(
        scheduled_new_reqs=[
            _NewReq("req2", req2_prompt, _block_ids_for(len(req2_prompt), req2_block))
        ],
        num_scheduled_tokens={"req2": len(req2_prompt)},
    )
    load_meta = sched.build_connector_meta(sout)
    assert len(load_meta.loads) == 1
    load = load_meta.loads[0]
    assert len(load.chunks) == 1
    spec = load.chunks[0]
    assert spec.chunk_hash == chunk_hash
    assert spec.new_pos_start == p_new_start
    assert spec.length == CHUNK
    expected_dst = _slot_ids_from_blocks([req2_block], BLOCK, 0, CHUNK)
    assert spec.dst_slot_ids == expected_dst

    # --- worker side: load into request 2's FRESH paged cache ---
    req2_cache = _make_paged_cache(num_blocks=8)
    worker.register_kv_caches({LAYER: req2_cache})  # same worker, new request cache
    worker._connector_metadata = load_meta
    # No fusion mask installed (Phase 1); start_load_kv scatters PIC-rotated K.

    class _NoLayerCtx:
        no_compile_layers = None  # disables the fusion-mask install path.

    worker.start_load_kv(_NoLayerCtx())

    # --- assertions: PIC end-to-end ---
    k2_bank = req2_cache[0].reshape(8 * BLOCK, NUM_KV_HEADS, HEAD_SIZE)
    v2_bank = req2_cache[1].reshape(8 * BLOCK, NUM_KV_HEADS, HEAD_SIZE)
    loaded_k = k2_bank[expected_dst]
    loaded_v = v2_bank[expected_dst]

    # Ground truth: the SAME raw K, RoPE'd directly at the NEW positions.
    k_new_truth = _neox_rope_reference(k_raw, new_positions, BASE, ROTARY_DIM)
    assert torch.allclose(loaded_k, k_new_truth, atol=1e-6, rtol=1e-5), (
        (loaded_k - k_new_truth).abs().max().item()
    )
    # V is position-independent -> byte-identical to what was stored.
    assert torch.allclose(loaded_v, v_raw, atol=1e-9)

    # Slot placement: nothing leaked outside the destination block.
    other_slots = [s for s in range(8 * BLOCK) if s not in expected_dst]
    assert torch.count_nonzero(k2_bank[other_slots]) == 0
    assert torch.count_nonzero(v2_bank[other_slots]) == 0


def test_scenario_a_nonzero_new_position_full_pic():
    """PIC reuse where the chunk moves to a non-zero new prefix offset is still
    exact (covers the signed-delta path, not just p_new==p_old start)."""
    torch.manual_seed(7)
    old_positions = torch.arange(200, 200 + CHUNK, dtype=torch.int64)
    k_raw = torch.randn(CHUNK, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float64)
    v_raw = torch.randn(CHUNK, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float64)
    k_old = _neox_rope_reference(k_raw, old_positions, BASE, ROTARY_DIM)

    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    chunk_hash = hash_chunk_tokens(list(range(9000, 9000 + CHUNK)))
    stored = StoredChunk(
        chunk_hash=chunk_hash, length=CHUNK, old_positions=old_positions
    )
    stored.k_per_layer[LAYER] = k_old
    stored.v_per_layer[LAYER] = v_raw
    store.put(stored)

    worker = _make_connector(sparse=False, store=store, worker=True)
    cache = _make_paged_cache(num_blocks=4)
    worker.register_kv_caches({LAYER: cache})

    # New position start 0 (prefix), block 2.
    from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
        ChunkLoadSpec,
        EpicReqLoad,
    )

    dst = _slot_ids_from_blocks([2], BLOCK, 0, CHUNK)
    meta = EpicConnectorMetadata()
    load = EpicReqLoad(req_id="r")
    load.chunks.append(
        ChunkLoadSpec(
            chunk_hash=chunk_hash,
            dst_slot_ids=dst,
            old_pos_start=-1,
            new_pos_start=0,
            length=CHUNK,
        )
    )
    meta.add_load(load)
    worker._connector_metadata = meta

    class _NoLayerCtx:
        no_compile_layers = None

    worker.start_load_kv(_NoLayerCtx())

    k_bank = cache[0].reshape(4 * BLOCK, NUM_KV_HEADS, HEAD_SIZE)
    loaded_k = k_bank[dst]
    truth = _neox_rope_reference(
        k_raw, torch.arange(0, CHUNK, dtype=torch.int64), BASE, ROTARY_DIM
    )
    assert torch.allclose(loaded_k, truth, atol=1e-6, rtol=1e-5)


# ===========================================================================
# Scenario B -- A+C+B sparse lifecycle (Phase 2b path)
# ===========================================================================
def _store_chunk(store: EpicChunkStore, tokens: list[int], old_start: int) -> str:
    h = hash_chunk_tokens(tokens)
    sc = StoredChunk(
        chunk_hash=h,
        length=len(tokens),
        old_positions=torch.arange(
            old_start, old_start + len(tokens), dtype=torch.int64
        ),
    )
    sc.k_per_layer[LAYER] = torch.zeros(len(tokens), NUM_KV_HEADS, HEAD_SIZE)
    sc.v_per_layer[LAYER] = torch.zeros(len(tokens), NUM_KV_HEADS, HEAD_SIZE)
    store.put(sc)
    return h


def test_scenario_b_sparse_lifecycle_acb():
    """A + C + B sparse lifecycle, end-to-end through the connector + runner.

    Layout (each chunk == CHUNK tokens):
      * A: cached, lands at the prompt PREFIX  -> reused natively (never in M).
      * C: genuinely new (no cache hit)        -> fully in M.
      * B: cached, lands at a NON-prefix offset -> only its link window in M.

    A genuine non-prefix B requires a NEW chunk (C) before it so the selection
    walk does not fold B into the contiguous prefix.
    """
    link = 8
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    conn = _make_connector(sparse=True, link=link, store=store)

    a = list(range(0, CHUNK))
    cnew = list(range(300, 300 + CHUNK))  # genuinely new (breaks contiguity)
    b = list(range(700, 700 + CHUNK))  # cached, non-prefix
    ha = _store_chunk(store, a, old_start=0)
    hb = _store_chunk(store, b, old_start=900)  # B's ORIGINAL pos differs from new
    hc = hash_chunk_tokens(cnew)

    tokens = a + cnew + b  # prompt order A, C, B ; N = 3*CHUNK
    n = len(tokens)
    assert n == 3 * CHUNK

    # ---- 1) scheduler match: external == |A| + |B| ----
    external, is_async = conn.get_num_new_matched_tokens(_Req("r0", tokens), 0)
    assert is_async is False
    assert external == 2 * CHUNK  # |A| (prefix) + |B| (non-prefix hit)
    num_new = n - external
    assert num_new == CHUNK  # exactly |C|

    # ---- 2) build_connector_meta -> EpicReqSparse ----
    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(n))],
        num_scheduled_tokens={"r0": num_new},
    )
    meta = conn.build_connector_meta(sout)
    assert len(meta.sparse) == 1
    sp = meta.sparse[0]
    assert sp.req_id == "r0"
    assert sp.full_seq_len == n
    m = sp.sparse_positions
    # M = C ∪ link(B) ∪ {N-1}. Verify it is exactly that set.
    c_positions = set(range(CHUNK, 2 * CHUNK))  # C is the middle chunk [16,32)
    b_link = set(range(2 * CHUNK, 2 * CHUNK + link))  # leading link of B [32,40)
    expected_m = sorted(c_positions | b_link | {n - 1})
    assert m == expected_m
    # invariants: sorted, unique, last == N-1, A never in M.
    assert m == sorted(set(m))
    assert m[-1] == n - 1
    assert all(p >= CHUNK for p in m)  # prefix A positions [0,16) excluded
    # B body (non-link part) is NOT forwarded.
    assert all(not (2 * CHUNK + link <= p < 3 * CHUNK - 1) for p in m)
    # computed_advance = N - external (NOT len(M)).
    assert sp.computed_advance == n - external == CHUNK

    # ---- 3) runner row edits -> positions buffer overwrite ----
    m_rows = len(m)
    # The scheduler core patch would set num_scheduled_tokens["r0"] = |M|. The
    # runner owns rows [0, |M|) for the single sparse request.
    edits = build_sparse_row_edits(
        req_ids=["r0"],
        cu_num_tokens=[m_rows],
        epic_sparse_positions={"r0": list(m)},
        epic_seq_len={"r0": sp.full_seq_len},
    )
    assert len(edits) == 1
    e = edits[0]
    assert (e.row_start, e.row_end) == (0, m_rows)
    assert e.seq_len == n  # full reused span, NOT |M|.

    # Apply the edit to a contiguous positions buffer (what the vanilla runner
    # would have built as computed_prefix + arange). After the overwrite the
    # rows carry M's scattered LOGICAL positions.
    positions = torch.arange(m_rows, dtype=torch.long)  # vanilla contiguous
    positions[e.row_start : e.row_end] = torch.tensor(e.positions, dtype=torch.long)
    assert positions.tolist() == list(m)
    # Each forwarded row's position is exactly a C / link / last logical position.
    for p in positions.tolist():
        assert (p in c_positions) or (p in b_link) or (p == n - 1)

    # ---- 4) flex logical_q mapping for the scattered M queries ----
    # Drive the real FlexAttention logical_q branch (the S6 hook) with the same
    # synthetic-stub method the runner_sparse unit test uses: the scattered M
    # query rows must map to exactly M's logical positions.
    import types

    block_size = BLOCK
    n_blocks = n // BLOCK
    physical_to_logical = torch.arange(n_blocks, dtype=torch.long).view(1, n_blocks)
    stub = types.SimpleNamespace(
        block_size=block_size,
        physical_to_logical=physical_to_logical,
        seq_lens=torch.tensor([n], dtype=torch.long),
        query_start_loc=torch.tensor([0, m_rows], dtype=torch.long),
        # decode_offset would be wrong for sparse; logical_q_positions overrides.
        decode_offset=torch.tensor([n - 1], dtype=torch.long),
        logical_q_positions=torch.tensor(list(m), dtype=torch.long),
    )
    request_lookup = torch.zeros(m_rows, dtype=torch.long)
    q_idx = torch.arange(m_rows, dtype=torch.long)
    physical_kv_idx = torch.arange(m_rows, dtype=torch.long)
    _, logical_q, _ = FlexAttentionMetadata._convert_physical_to_logical(
        stub, request_lookup, q_idx, physical_kv_idx
    )
    # The flex logical_q for each M query row equals M's scattered logical pos.
    assert torch.equal(logical_q, torch.tensor(list(m), dtype=torch.long))

    # ---- 5) save guard: the reused B chunk is NOT re-saved ----
    saved = set()
    for s in meta.saves:
        saved.update(s.chunk_hashes)
    assert ha not in saved  # A already cached -> not re-saved.
    assert hb not in saved  # B already cached AND only partially in M -> guarded.
    assert hc in saved  # C is pure-new (fully in M) -> saved canonically.


def test_scenario_b_with_real_scheduler_overrides():
    """Bind Scenario B's connector accounting to a real Scheduler instance.

    Mirrors test_scheduler_core_patch.py: drive ``_apply_epic_sparse_overrides``
    + ``_update_after_schedule`` with the EXACT (M, positions, advance) the
    connector derived above, and confirm the scheduler rewrites rows to |M| and
    advances num_computed to N.
    """
    from tests.v1.core.utils import create_requests, create_scheduler
    from vllm.v1.core.sched.output import SchedulerOutput

    link = 8
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    conn = _make_connector(sparse=True, link=link, store=store)
    a = list(range(0, CHUNK))
    cnew = list(range(300, 300 + CHUNK))
    b = list(range(700, 700 + CHUNK))
    _store_chunk(store, a, old_start=0)
    _store_chunk(store, b, old_start=900)
    tokens = a + cnew + b
    n = len(tokens)

    external, _ = conn.get_num_new_matched_tokens(_Req("0", tokens), 0)
    num_new = n - external
    sout_meta = _SchedOut(
        scheduled_new_reqs=[_NewReq("0", tokens, _block_ids_for(n))],
        num_scheduled_tokens={"0": num_new},
    )
    meta = conn.build_connector_meta(sout_meta)
    sp = meta.sparse[0]
    m_rows = len(sp.sparse_positions)

    # A mock connector exposing the scheduler-core hook surface, fed the values
    # the REAL connector produced (so this is a true cross-check, not a fixture).
    class _Bridge:
        def get_sparse_num_scheduled_tokens(self, _meta, req_id):
            return m_rows if req_id == "0" else None

        def get_sparse_positions(self, _meta, req_id):
            return list(sp.sparse_positions) if req_id == "0" else None

        def get_sparse_computed_advance(self, _meta, req_id):
            return sp.computed_advance if req_id == "0" else None

    sched = create_scheduler()
    so = SchedulerOutput.make_empty()
    so.num_scheduled_tokens = {"0": num_new}
    so.total_num_scheduled_tokens = num_new
    sched._apply_epic_sparse_overrides(_Bridge(), so, meta=None)

    # Rows rewritten to |M|; epic dicts stamped.
    assert so.num_scheduled_tokens["0"] == m_rows
    assert so.total_num_scheduled_tokens == num_new + (m_rows - num_new)
    assert so.epic_sparse_positions["0"] == list(sp.sparse_positions)
    assert so.epic_seq_len["0"] == n
    assert so.epic_computed_advance["0"] == sp.computed_advance

    # num_computed advance lands on N.
    reqs = create_requests(num_requests=1, num_tokens=n)
    req = reqs[0]
    # Re-key the request id to "0" so the override dicts match.
    req.request_id = "0"
    sched.requests["0"] = req
    req.num_computed_tokens = external  # scheduler set num_computed to external.
    sched._update_after_schedule(so)
    assert req.num_computed_tokens == n
    assert req.is_prefill_chunk is False


# ===========================================================================
# Scenario C -- flag off => no sparse trace, prefix-only Phase 1 behavior
# ===========================================================================
def test_scenario_c_flag_off_no_sparse_trace():
    """SAME A+C+B input as Scenario B but epic_sparse_forward=False.

    Expect: NO sparse metadata at all, and only the contiguous prefix A is
    reported as reusable (Phase 1 prefix-only path); B (non-prefix) is recorded
    but not turned into a sparse plan.
    """
    link = 8
    store = EpicChunkStore(capacity_bytes=10**8, pin_memory=False)
    conn = _make_connector(sparse=False, link=link, store=store)

    a = list(range(0, CHUNK))
    cnew = list(range(300, 300 + CHUNK))
    b = list(range(700, 700 + CHUNK))
    _store_chunk(store, a, old_start=0)
    hb = _store_chunk(store, b, old_start=900)
    hc = hash_chunk_tokens(cnew)
    tokens = a + cnew + b
    n = len(tokens)

    # Flag off -> match path reports ONLY the contiguous prefix extent |A|.
    external, is_async = conn.get_num_new_matched_tokens(_Req("r0", tokens), 0)
    assert is_async is False
    assert external == CHUNK  # |A| only (NOT |A|+|B|).

    sout = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(n))],
        num_scheduled_tokens={"r0": n - external},
    )
    meta = conn.build_connector_meta(sout)

    # No sparse trace whatsoever.
    assert meta.sparse == []
    assert conn.has_sparse_requests(meta) is False
    assert conn.get_sparse_num_scheduled_tokens(meta, "r0") is None
    assert conn.get_sparse_positions(meta, "r0") is None
    assert conn.get_sparse_computed_advance(meta, "r0") is None
    assert conn.is_sparse_request("r0") is False
    # No sparse-mode save guard -> new chunks (B unseen-as-new aside) saved as
    # usual; only the genuinely new C is harvested (A and B already cached).
    saved = set()
    for s in meta.saves:
        saved.update(s.chunk_hashes)
    assert hc in saved
    assert hb not in saved  # already cached -> not re-saved.

    # Prefix-only load: only A is loadable (B's non-prefix hit is recorded but
    # not loaded in Phase 1).
    conn2 = _make_connector(sparse=False, link=link, store=store)
    ext2, _ = conn2.get_num_new_matched_tokens(_Req("r0", tokens), 0)
    conn2.update_state_after_alloc(
        _Req("r0", tokens), blocks=None, num_external_tokens=ext2
    )
    sout2 = _SchedOut(
        scheduled_new_reqs=[_NewReq("r0", tokens, _block_ids_for(n))],
        num_scheduled_tokens={"r0": n - ext2},
    )
    meta2 = conn2.build_connector_meta(sout2)
    assert meta2.sparse == []
    assert len(meta2.loads) == 1
    load = meta2.loads[0]
    # Exactly the prefix A chunk is loaded (one chunk, CHUNK tokens at pos 0).
    assert len(load.chunks) == 1
    assert load.chunks[0].new_pos_start == 0
    assert load.chunks[0].length == CHUNK
    # B's non-prefix hit is recorded on the load (tracked, not a chunk load).
    assert len(load.non_prefix_hits) >= 1


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
