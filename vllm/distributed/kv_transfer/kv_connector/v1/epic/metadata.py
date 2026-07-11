# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler->Worker metadata for the EPIC connector.

EPIC original passed state through a mutable ``cache_fuse_metadata`` dict that
lived on the model object and threaded through every forward call (see
vllm_epic/vllm/model_executor/models/llama.py and attention/layer.py). That is
exactly the "global dict 관통" the migration brief forbids.

Here the equivalent state is a structured, picklable ``KVConnectorMetadata``
built per scheduler step and bound to the worker via ``forward_context`` /
``bind_connector_metadata`` -- no model-object globals.

Phase 1 only carries *prefix-extent* chunk loads (PIC re-rotary into paged
blocks). Non-prefix hits are recorded but never loaded (Phase 2).
"""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class ChunkLoadSpec:
    """One cached chunk to load into a request's freshly-allocated blocks."""

    chunk_hash: str
    # Destination slot ids (block_id * block_size + offset) in the paged cache,
    # one per token of the chunk. Length == `length`.
    dst_slot_ids: list[int]
    # Absolute position the chunk's K was originally rotated to.
    old_pos_start: int
    # Absolute position the chunk will occupy in the new request.
    new_pos_start: int
    length: int
    # Tokens to skip at the HEAD of the stored chunk before loading (native-
    # computed-region trim): the worker reads stored K/V/old_positions from
    # [src_offset, src_offset + length). 0 == load the chunk from its start.
    # Used when the leading part of a matched chunk is already covered by the
    # EXACT native prefix cache (num_computed_tokens) -- loading approximate
    # store KV over it would both downgrade this request and write into paged
    # blocks that other requests share.
    src_offset: int = 0


@dataclass
class NonPrefixHit:
    """A content-matched chunk that is NOT in the contiguous prefix.

    Phase 1 records these for observability and as the data the Phase 2
    sparse-forward path will consume; it does NOT load them. Loading them now
    would be wasted work: the full (dense) forward re-computes fresh KV for
    every non-prefix token and would overwrite anything we scattered in.
    """

    chunk_hash: str
    # Token offset within the new prompt where the match starts.
    prompt_offset: int
    old_pos_start: int
    length: int
    # Head trim into the stored chunk (see ChunkLoadSpec.src_offset): when the
    # leading part of this hit falls below the native-computed extent, the hit
    # is clamped and this records how many stored tokens to skip at load time.
    src_offset: int = 0
    # Save-time context chain of the STORED chunk (ChainHasher digests of the
    # save prompt's tokens [0, start) / [0, start+len)). None == unknown.
    # Per-run link continuity uses prev.chain_end == cur.chain_start to prove
    # two adjacent hits came from one contiguous warm (same file).
    chain_start: str | None = None
    chain_end: str | None = None


@dataclass
class EpicReqLoad:
    req_id: str
    chunks: list[ChunkLoadSpec] = field(default_factory=list)
    non_prefix_hits: list[NonPrefixHit] = field(default_factory=list)


@dataclass
class EpicReqSave:
    """A request whose prefill KV should be harvested into the chunk store."""

    req_id: str
    # Per-chunk: (chunk_hash, slot ids to gather from, absolute positions).
    chunk_hashes: list[str] = field(default_factory=list)
    chunk_slot_ids: list[list[int]] = field(default_factory=list)
    chunk_positions: list[list[int]] = field(default_factory=list)
    # Per-chunk save-time context chain (chain_before, chain_after): the
    # ChainHasher digests of this prompt's tokens up to the chunk's start/end.
    # Parallel to chunk_hashes; the worker persists these on the StoredChunk
    # so future selections can verify fold/run soundness.
    chunk_chains: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class EpicReqSparse:
    """Per-request sparse-forward descriptor (Phase 2b, S0 plumbing).

    Carries the frozen recompute set M (DESIGN §1, §4.1) from the scheduler to
    the worker / scheduler-core consumer. Pure ints/lists -> pickle-safe across
    the scheduler->worker process boundary; no tensors, no closures.

    Phase 2b's core patches (next batch) consume these:
      * the scheduler reduces ``num_scheduled_tokens`` to ``len(sparse_positions)``
        for this request,
      * the runner builds RoPE ``positions`` from ``sparse_positions`` instead of
        a contiguous range (PHASE2 §2).
    """

    req_id: str
    # Logical positions of M (the tokens actually forwarded). Sorted ascending,
    # last element == full_seq_len - 1. Empty -> dense forward (no sparse M).
    sparse_positions: list[int] = field(default_factory=list)
    # N: the full prompt length this plan describes (logical positions [0, N)).
    full_seq_len: int = 0
    # How much to advance the scheduler's ``num_computed_tokens`` after this
    # step so it converges to N. This is N - external (external = |A| + |B|,
    # the tokens the scheduler already counted as computed via
    # ``num_external_computed_tokens``), NOT len(sparse_positions): M may
    # overlap reused KV positions (link/last tokens inside B), so advancing by
    # len(M) would double-count. 0 means "use the default advance" (the S3
    # scheduler patch only overrides when this is > 0).
    computed_advance: int = 0

    @property
    def is_sparse(self) -> bool:
        return bool(self.sparse_positions)


@dataclass
class FusionMaskPlan:
    """Per-step LegoLink fusion-mask data for the FlexAttention backend.

    This is the picklable, scheduler->worker payload that the worker turns into
    the fixed-size mask metadata tensors (epic/fusion_mask.py). It carries only
    plain Python ints/lists (no tensors, no closures) so it crosses the process
    boundary cleanly and the worker materializes tensors locally.

    Phase 2a: ``gate == False`` and ``reused_offsets`` empty -> the installed
    mask_mod is equivalent to standard causal (full-forward back-compat). Phase
    2b fills ``recompute_offsets`` / ``reused_offsets`` and sets ``gate``.
    """

    # Whether to install the LegoLink mask this step at all.
    enabled: bool = False
    # Max logical position the mask must address (>= max sequence length).
    seq_len: int = 0
    # Logical positions that are M (recompute) query rows. Empty in Phase 2a.
    recompute_offsets: list[int] = field(default_factory=list)
    # Logical positions whose KV participates. Empty -> all positions live
    # (Phase 2a == causal).
    reused_offsets: list[int] = field(default_factory=list)
    # Recompute-row gating. False in Phase 2a (every row forwarded == causal).
    gate: bool = False


@dataclass
class EpicConnectorMetadata(KVConnectorMetadata):
    # Requests needing chunk loads this step.
    loads: list[EpicReqLoad] = field(default_factory=list)
    # Requests whose KV should be saved (harvested) this step.
    saves: list[EpicReqSave] = field(default_factory=list)
    # LegoLink fusion-mask plan for this step's FlexAttention forward (Phase 2a).
    # None / disabled -> the connector leaves the backend's default causal mask
    # untouched (Phase 1 behavior).
    fusion_mask: "FusionMaskPlan | None" = None
    # Per-request sparse-forward descriptors (Phase 2b, S0 plumbing). Empty when
    # epic_sparse_forward is off -> Phase 1/2a behavior unchanged.
    sparse: list[EpicReqSparse] = field(default_factory=list)

    def add_load(self, load: EpicReqLoad) -> None:
        self.loads.append(load)

    def add_save(self, save: EpicReqSave) -> None:
        self.saves.append(save)

    def add_sparse(self, sparse: EpicReqSparse) -> None:
        self.sparse.append(sparse)
