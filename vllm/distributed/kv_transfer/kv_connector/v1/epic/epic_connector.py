# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC KV connector (Phase 1, non-invasive).

Implements position-independent chunk reuse on top of ``KVConnectorBase_V1``,
restricted to the *prefix extent* of new requests (so it stays compatible with
V1's contiguous ``num_computed_tokens`` scheduler and contiguous runner
positions). Non-prefix content matches are tracked only (Phase 2).

Mapping from EPIC original (vllm_epic, vLLM 0.7 / V0) to here (V1):

  * EPIC `cache_fuse_metadata` dict on the model  -> `EpicConnectorMetadata`
    (scheduler-built, bound via forward_context).
  * EPIC `self.hack_kv` dense KV on the model      -> `EpicChunkStore` (CPU).
  * EPIC `rotary_emb(org_pos, fake_q, old_kv[0])`   -> `PICRotator.rotate_keys`
    (delta rotation on already-rotated K; no fake_q hack).
  * EPIC out-of-band chunk match in benchmark       -> content-hash lookup in
    `get_num_new_matched_tokens` reported to the V1 scheduler as a prefix length.

NOT done in Phase 1 (see epic/PHASE2.md): selective recompute (recomp_ratio /
imp_indices), partial-mask fusion attention, sparse (non-prefix) forward.
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    EpicChunkStore,
    EpicSchedulerIndex,
    StoredChunk,
    hash_chunk_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
    ChunkLoadSpec,
    EpicConnectorMetadata,
    EpicReqLoad,
    EpicReqSave,
    EpicReqSparse,
    FusionMaskPlan,
    NonPrefixHit,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.pic import PICRotator
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    AlignmentStrategy,
    EpicSelection,
    LegoLinkMaskBuilder,
    LegoLinkRecompute,
    PicAlignment,
    RecomputePlan,
    RecomputePolicy,
    ReuseSelection,
    SelectionStrategy,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# 8 GB default CPU budget for cached chunks.
DEFAULT_CAPACITY_BYTES = 8 * (1024**3)
DEFAULT_CHUNK_SIZE = 256


class _PromptLenShim:
    """Minimal stand-in exposing ``prompt_token_ids`` so a RecomputePolicy can
    read the exact prompt length N without the V1 ``Request`` (unavailable in
    ``build_connector_meta``; only the new-req descriptor's token ids are)."""

    __slots__ = ("prompt_token_ids",)

    def __init__(self, token_ids: list[int]):
        self.prompt_token_ids = token_ids


def check_scatter_fidelity(
    kv_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_ids: list[int],
) -> tuple[bool, float, bool, float] | None:
    """Pure read-back comparison of a scatter (Implement 2 core, CPU-testable).

    Reads the ``slot_ids`` rows back out of the supported FlashAttention paged
    layout ``(2, num_blocks, block_size, H, D)`` and compares them to the K/V
    that *should* be there. Returns ``(k_allclose, k_maxabsdiff, v_allclose,
    v_maxabsdiff)`` or ``None`` for an unsupported layout. No tensor is mutated.

    This is the load-fidelity oracle: if scatter wrote to the wrong slots, or a
    bad stride/reshape aliases the wrong memory, the read-back diverges from the
    reference and ``allclose`` is False -- the exact in-situ signal the GPU smoke
    needs to separate a layout/aliasing bug from an algorithmic (approximation)
    one. Unit-tested on CPU by injecting a deliberate slot-offset error.
    """
    shape = kv_cache.shape
    if not (shape[0] == 2 and kv_cache.dim() == 5):
        return None
    num_blocks, block_size = shape[1], shape[2]
    k_bank = kv_cache[0].reshape(num_blocks * block_size, *shape[3:])
    v_bank = kv_cache[1].reshape(num_blocks * block_size, *shape[3:])
    slots = torch.as_tensor(slot_ids, device=kv_cache.device, dtype=torch.long)
    k_back = k_bank[slots]
    v_back = v_bank[slots]
    k_ref = k.to(k_bank.dtype)
    v_ref = v.to(v_bank.dtype)
    k_diff = (k_back.float() - k_ref.float()).abs().max().item()
    v_diff = (v_back.float() - v_ref.float()).abs().max().item()
    return (
        bool(torch.allclose(k_back, k_ref)),
        float(k_diff),
        bool(torch.allclose(v_back, v_ref)),
        float(v_diff),
    )


def _slot_ids_from_blocks(
    block_ids: list[int], block_size: int, start_token: int, num_tokens: int
) -> list[int]:
    """Compute paged slot ids for tokens [start_token, start_token+num_tokens)."""
    slots: list[int] = []
    for i in range(start_token, start_token + num_tokens):
        block_idx = i // block_size
        offset = i % block_size
        slots.append(block_ids[block_idx] * block_size + offset)
    return slots


class EpicConnector(KVConnectorBase_V1):
    # --- In-band engagement counters (test/diagnostic only) -----------------
    # CLASS variable so a smoke test running the V1 engine IN-PROCESS
    # (VLLM_ENABLE_V1_MULTIPROCESSING=0) can assert the sparse path actually
    # engaged WITHOUT scraping logs: the SCHEDULER-role connector (EngineCore)
    # and the WORKER-role connector live in the SAME process, share this class
    # dict, and bump distinct keys at the load-bearing sites:
    #   sparse_match  -> a request took the non-prefix sparse branch in
    #                    get_num_new_matched_tokens (scheduler).
    #   sparse_emit   -> a sparse recompute set M was emitted for a request in
    #                    build_connector_meta/_emit_sparse (scheduler).
    #   chunks_loaded -> a cached chunk was actually scattered into the paged
    #                    cache in _load_chunk (worker).
    # Incrementing is GATED on the ``epic_debug_counters`` extra-config so a
    # normal run never touches shared mutable class state (no cross-engine
    # leakage, thread-safety is a non-concern because increment only fires
    # under the in-process single-engine smoke path). Reset between runs via
    # reset_debug_counters().
    debug_counters: dict[str, int] = {
        "sparse_match": 0,
        "sparse_emit": 0,
        "chunks_loaded": 0,
        # Load-fidelity diagnostics (gated on epic_debug_counters; surfaced via
        # the RESULT_JSON counters channel which is reliable regardless of log
        # level / process). These disambiguate the "no check_load log" mystery:
        #   loads_emitted     -> chunk-load specs seen by the worker this run
        #   check_load_calls  -> times the scatter read-back actually ran
        #                        (0 with loads_emitted>0 => epic_debug_check_load
        #                         did not reach the worker)
        #   check_load_mismatch -> read-backs where allclose was False (SCATTER
        #                          / PIC / layout bug)
        #   check_load_skip   -> read-backs skipped (unsupported KV layout =>
        #                        scatter is a NO-OP, a likely root cause)
        "loads_emitted": 0,
        "check_load_calls": 0,
        "check_load_mismatch": 0,
        "check_load_skip": 0,
    }

    # EPIC diagnostics: per-request SELECTION summary so a probe (musique_blend)
    # can SEE whether selection found non-prefix content -- the difference
    # between "LegoLink recompute fired" and "everything silently fell into the
    # contiguous prefix so link is INERT". One dict per get_num_new_matched_tokens
    # call (request_id may repeat across the warm/measured pair; the LAST entry
    # for a given prompt is the measured one). Gated on epic_debug_counters like
    # the counters above. Each entry:
    #   {request_id, N, prefix_extent, num_non_prefix, non_prefix_offsets,
    #    sparse_branch}  -- sparse_branch True iff the non-prefix sparse path was
    # taken (i.e. LegoLink can recompute). num_non_prefix==0 -> LINK INERT.
    debug_selection: list[dict] = []

    @classmethod
    def reset_debug_counters(cls) -> None:
        """Zero every engagement counter (call between smoke runs)."""
        for k in cls.debug_counters:
            cls.debug_counters[k] = 0
        cls.debug_selection = []

    @classmethod
    def _record_selection(cls, entry: dict) -> None:
        cls.debug_selection.append(entry)

    @classmethod
    def _bump_counter(cls, key: str, n: int = 1) -> None:
        cls.debug_counters[key] = cls.debug_counters.get(key, 0) + n

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        extra = self._kv_transfer_config.kv_connector_extra_config or {}

        chunk_size = int(extra.get("epic_chunk_size", DEFAULT_CHUNK_SIZE))
        # Chunk size must be a multiple of block_size so chunk boundaries align
        # with paged blocks (gather/scatter is block-granular friendly and slot
        # math is exact). Round up to the next block multiple.
        if chunk_size % self._block_size != 0:
            chunk_size = (
                (chunk_size + self._block_size - 1)
                // self._block_size
                * self._block_size
            )
        self._chunk_size = max(chunk_size, self._block_size)

        capacity = int(extra.get("epic_cpu_bytes", DEFAULT_CAPACITY_BYTES))
        world_size = vllm_config.parallel_config.world_size
        self._capacity_bytes = capacity // max(world_size, 1)

        # --- role split (root-cause fix) -----------------------------------
        # The connector is instantiated as TWO separate objects in TWO separate
        # processes: SCHEDULER (EngineCore) and WORKER. The WORKER owns the real
        # KV tensors (EpicChunkStore); the SCHEDULER only needs to KNOW which
        # chunks are cached so its selection produces hits. Giving the scheduler
        # its own (always-empty) store was the bug. Instead:
        #   * WORKER   -> EpicChunkStore (holds tensors; save/load run here).
        #   * SCHEDULER-> EpicSchedulerIndex (metadata-only deterministic mirror;
        #     selection/dedup query this).
        # We compute the per-chunk byte dims ONCE so the mirror's byte budget /
        # LRU exactly tracks the worker store (see EpicSchedulerIndex).
        dims = self._derive_chunk_byte_dims(vllm_config, kv_cache_config)
        (self._idx_num_layers, self._idx_num_kv_heads, self._idx_head_size,
            self._idx_dtype_size) = dims

        # Both fields exist on every instance for code clarity, but only the
        # role-appropriate one is constructed/used. (Constructing both would be
        # harmless metadata, but None makes the role intent explicit and trips
        # loudly if the wrong side is touched.)
        self._store: EpicChunkStore | None = None
        self._index: EpicSchedulerIndex | None = None
        if role == KVConnectorRole.SCHEDULER:
            self._index = EpicSchedulerIndex(
                capacity_bytes=self._capacity_bytes,
                num_layers=self._idx_num_layers,
                num_kv_heads=self._idx_num_kv_heads,
                head_size=self._idx_head_size,
                cache_dtype_size=self._idx_dtype_size,
            )
        else:  # KVConnectorRole.WORKER
            self._store = EpicChunkStore(
                capacity_bytes=self._capacity_bytes,
                pin_memory=bool(extra.get("epic_pin_memory", True)),
            )

        # Scheduler-side bookkeeping: req_id -> matched prefix chunk hashes.
        self._matched_prefix: dict[str, list[tuple[str, int]]] = {}
        # req_id -> non-prefix hits (Phase 2 record only).
        self._non_prefix: dict[str, list[NonPrefixHit]] = {}
        # Requests that asked to load this step (after alloc).
        self._loads_pending: dict[str, "Request"] = {}

        # --- Sparse (non-prefix) forward flag (Phase 2b, S0) ---------------
        # OFF by default: every new code path below is inert and Phase 1/2a
        # behavior (dense forward, prefix-only reuse) is byte-for-byte unchanged.
        # ON: the connector derives a sparse recompute set M (LegoLink) per
        # request and emits it on EpicConnectorMetadata.sparse for the scheduler
        # / runner core patches (next batch) to consume.
        self._sparse_forward: bool = bool(
            extra.get("epic_sparse_forward", False)
        )
        # LegoLink link-token count: leading tokens of each non-prefix chunk
        # that are always recomputed (the boundary stitch). DESIGN §4.1.
        self._link_tokens: int = int(extra.get("epic_link_tokens", 8))

        # --- Sparse scheduling-budget guard ---------------------------------
        # The sparse override rewrites num_scheduled_tokens AFTER the scheduler
        # ran its token-budget / long-prefill truncation, and the sparse plan is
        # only correct for a ONE-STEP prefill of the full remaining prompt. So
        # the connector must decline the sparse branch whenever the scheduler
        # would truncate: (a) the pre-override row count N-external must fit the
        # per-step token budget, and (b) the post-override |M| must too (M can
        # exceed N-external by the link tokens overlapping reused B). Mirrors
        # Scheduler.max_num_scheduled_tokens; 0 disables the guard (tests).
        sched = vllm_config.scheduler_config
        self._max_sparse_rows: int = int(
            getattr(sched, "max_num_scheduled_tokens", None)
            or sched.max_num_batched_tokens
        )
        self._long_prefill_threshold: int = int(
            getattr(sched, "long_prefill_token_threshold", 0) or 0
        )

        # --- DIAGNOSTIC: worker load fidelity self-check (default OFF) ------
        # When ``epic_debug_check_load`` is on, the worker re-reads the dst KV
        # slots IMMEDIATELY after scatter and compares them to the source
        # StoredChunk K/V (allclose + per-layer max-abs-diff on the FIRST loaded
        # chunk only, 1 logger.info line). This catches paged-cache layout /
        # stride / aliasing bugs IN SITU: if scatter wrote to the wrong slots or
        # the (2,num_blocks,block_size,H,D) reshape aliases the wrong memory, the
        # read-back diverges from the source. The PIC re-rotary is applied to the
        # source K before scatter, so the check compares against the ALIGNED K
        # (the exact tensor handed to _scatter_kv), isolating scatter fidelity
        # from alignment math. Inert (no read-back, no cost) when off.
        self._debug_check_load: bool = bool(
            extra.get("epic_debug_check_load", False)
        )
        # --- DIAGNOSTIC: in-band engagement counters (default OFF) ----------
        # When on, the load-bearing sparse sites bump EpicConnector.debug_counters
        # (a CLASS dict) so an in-process smoke can assert the sparse path
        # engaged without log scraping. Inert (no shared-state writes) when off.
        self._debug_counters: bool = bool(
            extra.get("epic_debug_counters", False)
        )
        # One-shot latch so the per-layer compare logs only for the FIRST chunk
        # of a step (avoids log spam across many chunks/requests).
        self._check_load_done: bool = False

        # --- Strategy seams (see epic/reuse_strategy.py, epic/DESIGN.md) ---
        # EPIC = exact-hash selection + PIC alignment + static LegoLink recompute.
        # CacheBlend would swap alignment->Identity and recompute->dynamic.
        # Phase 1 routes selection through EpicSelection and alignment through
        # PicAlignment but is otherwise behaviorally identical to before.
        self._selection: SelectionStrategy = EpicSelection()
        # phase1_dense is the inverse of sparse_forward: when sparse is OFF the
        # policy short-circuits to an empty (dense) plan == Phase 1/2a.
        self._recompute: RecomputePolicy = LegoLinkRecompute(
            num_link_tokens=self._link_tokens,
            phase1_dense=not self._sparse_forward,
        )
        # Scheduler-side: req_id -> ReuseSelection from get_num_new_matched_tokens
        # (kept only when sparse_forward is on; needed to derive M at build time).
        self._selections: dict[str, "ReuseSelection"] = {}
        # Scheduler-side: req_id -> (N, external=|A|+|B|) for requests routed
        # through the sparse path this step. Used by the scheduler core hooks
        # (is_sparse_request / get_sparse_computed_advance) and cleared per step
        # in build_connector_meta. Empty when sparse forward is off.
        self._sparse_reqs: dict[str, tuple[int, int]] = {}
        # Scheduler-side: req_id -> native num_computed_tokens at match time.
        # Positions [0, nc) hold EXACT KV from the native prefix cache (whose
        # blocks may be SHARED with other live requests), so build_connector_meta
        # trims every chunk load below nc: loading position-independent
        # (approximate) store KV there would downgrade this request AND poison
        # the shared blocks for everyone else. Cleared per step.
        self._native_computed: dict[str, int] = {}

        # Phase 2a fusion-mask: enable installing the LegoLink FlexAttention
        # mask_mod. OFF by default so Phase 1 behavior (backend default causal)
        # is unchanged unless explicitly opted in via extra config.
        self._fusion_enabled: bool = bool(extra.get("epic_fusion_mask", False))
        # Fixed mask-tensor capacity (== max logical position) so the worker's
        # mask_mod object identity is stable across requests (recompile
        # avoidance). Sized to the longest possible sequence.
        try:
            self._mask_capacity = int(vllm_config.model_config.max_model_len)
        except Exception:  # noqa: BLE001
            self._mask_capacity = 0

        # Worker-side: lazily built once kv caches are registered.
        self._pic: PICRotator | None = None
        self._alignment: AlignmentStrategy | None = None
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        # Worker-side fusion-mask builder (owns fixed-size mask tensors + the
        # single stable mask_mod object). Built lazily on first install.
        self._mask_builder: LegoLinkMaskBuilder | None = None
        # FlexAttention layers we have installed our mask_mod onto (so we can
        # leave non-fusion layers untouched and restore on disable).
        self._flex_layers_patched: set[str] = set()

        # --- Sparse-mode safety gating (Phase 2b, S7) ----------------------
        # When epic_sparse_forward is ON, the worker forward depends on two
        # config invariants that the connector cannot enforce by itself
        # (changing them would be far more invasive than a config check):
        #   1. the FlexAttention backend (the only V1 backend that accepts the
        #      LegoLink logical_mask_mod the connector installs), and
        #   2. enforce_eager (the |M| sparse forward + per-step mask tensor are
        #      not yet validated under CUDA graph / piecewise compile capture).
        # Validate (do NOT silently mutate) so the failure is loud and actionable
        # rather than a wrong-numbers run. Fully inert when sparse is OFF.
        if self._sparse_forward:
            self._validate_sparse_safety(vllm_config)

        if role == KVConnectorRole.WORKER:
            try:
                self._pic = PICRotator.from_vllm_config(vllm_config)
                self._alignment = PicAlignment(self._pic)
            except Exception as e:  # noqa: BLE001
                # Non-Llama / unsupported rope_scaling: disable reuse loads but
                # keep the connector functional (saves still work).
                logger.warning("EPIC PICRotator unavailable, loads disabled: %s", e)
                self._pic = None
                self._alignment = None

    # ===================== chunk byte-dim derivation =====================

    @staticmethod
    def _derive_chunk_byte_dims(
        vllm_config: VllmConfig, kv_cache_config: "KVCacheConfig"
    ) -> tuple[int, int, int, int]:
        """Return (num_layers, num_kv_heads, head_size, cache_dtype_size).

        These four numbers feed ``stored_chunk_nbytes`` so the scheduler index
        and the worker store compute identical per-chunk byte sizes (==
        ``StoredChunk.nbytes()`` for the same chunk length). We read them from the
        ``kv_cache_config`` attention specs when available (the authoritative,
        post-profiling source: num_kv_heads here is already TP-sharded and the
        dtype is the real cache dtype, fp8 included), and fall back to
        ``VllmConfig`` accessors otherwise.

        Drift note: if the worker harvests KV in a dtype that differs from the
        spec dtype (it does ``kv_layer[...].to("cpu")``, preserving the cache
        dtype, so they match in the supported FlashAttention layout), the mirror
        byte budget would diverge. We therefore prefer the spec dtype, which is
        exactly the dtype the worker tensors carry.
        """
        from vllm.v1.kv_cache_interface import AttentionSpec
        from vllm.utils.torch_utils import get_dtype_size

        num_layers = 0
        num_kv_heads = 0
        head_size = 0
        dtype_size = 0
        try:
            for group in kv_cache_config.kv_cache_groups:
                spec = group.kv_cache_spec
                if isinstance(spec, AttentionSpec):
                    num_layers += len(group.layer_names)
                    # All attention layers in a uniform group share dims; take
                    # the group's spec (last write wins -- they agree).
                    num_kv_heads = spec.num_kv_heads
                    head_size = spec.head_size
                    dtype_size = get_dtype_size(spec.dtype)
        except Exception:  # noqa: BLE001
            num_layers = 0

        # Fallbacks from VllmConfig if the kv_cache_config was unavailable /
        # non-attention (keeps the index functional; bytes stay deterministic).
        if num_layers <= 0:
            try:
                num_layers = vllm_config.model_config.get_num_layers(
                    vllm_config.parallel_config
                )
            except Exception:  # noqa: BLE001
                num_layers = 1
        if num_kv_heads <= 0:
            try:
                num_kv_heads = vllm_config.model_config.get_num_kv_heads(
                    vllm_config.parallel_config
                )
            except Exception:  # noqa: BLE001
                num_kv_heads = 1
        if head_size <= 0:
            try:
                head_size = vllm_config.model_config.get_head_size()
            except Exception:  # noqa: BLE001
                head_size = 1
        if dtype_size <= 0:
            try:
                dtype_size = get_dtype_size(vllm_config.model_config.dtype)
            except Exception:  # noqa: BLE001
                dtype_size = 2

        return num_layers, num_kv_heads, head_size, dtype_size

    # ===================== sparse-mode safety gating (S7) =====================

    def _validate_sparse_safety(self, vllm_config: VllmConfig) -> None:
        """Fail fast if sparse-forward prerequisites are not met (Phase 2b, S7).

        Only called when ``epic_sparse_forward`` is on, so it never runs on the
        Phase 1/2a default path. Two checks:

          1. FlexAttention backend. Sparse forward installs a LegoLink
             ``logical_mask_mod`` (see ``_install_fusion_mask``); only the
             FlexAttention V1 backend consumes it. FlashAttention (the V1
             default) silently ignores the mask -> wrong results. We *validate*
             rather than force the backend: vLLM auto-selects the backend
             deep in platform code and overriding it from a connector would be
             far more invasive and fragile than a config assertion.

          2. enforce_eager. The reduced (|M|) sparse forward and the per-step
             fusion-mask tensor have not been validated under CUDA graph /
             piecewise-compile capture; eager is the first-pass safe mode.
             PIECEWISE relaxation is a post-GPU-verification TODO (PHASE2 §S7).
        """
        # --- (1) backend must be FlexAttention -----------------------------
        # vllm_config.attention_config.backend is the *configured* backend
        # (AttentionBackendEnum | None). None == "auto", which on the default
        # platform path resolves to FlashAttention, NOT FlexAttention -> reject
        # so the user explicitly pins FLEX_ATTENTION.
        backend = None
        try:
            backend = vllm_config.attention_config.backend
        except Exception:  # noqa: BLE001
            backend = None

        backend_name = getattr(backend, "name", None)
        if backend_name != "FLEX_ATTENTION":
            chosen = backend_name if backend_name is not None else "auto (default)"
            raise ValueError(
                "EPIC sparse forward (epic_sparse_forward=true) requires the "
                "FlexAttention backend, which is the only V1 backend that "
                "consumes the LegoLink logical_mask_mod the connector installs. "
                f"The configured attention backend is {chosen!r}. "
                "Select it explicitly via the engine argument "
                "--attention-backend FLEX_ATTENTION (serving) or the "
                "attention_backend='FLEX_ATTENTION' kwarg / "
                "attention_config.backend (offline LLM/EngineArgs), then retry. "
                "Note: the legacy VLLM_ATTENTION_BACKEND environment variable "
                "was removed in vLLM v0.22 and is silently ignored. "
                "The connector does not override backend auto-selection on "
                "purpose: doing so from a KV connector would be far more "
                "invasive than this configuration check."
            )

        # --- (2) eager required (first-pass safety) ------------------------
        enforce_eager = False
        try:
            enforce_eager = bool(vllm_config.model_config.enforce_eager)
        except Exception:  # noqa: BLE001
            enforce_eager = False

        if not enforce_eager:
            raise ValueError(
                "EPIC sparse forward (epic_sparse_forward=true) currently "
                "requires enforce_eager=true. The reduced (|M|) sparse forward "
                "and the per-step FlexAttention fusion-mask tensor have not been "
                "validated under CUDA graph / piecewise-compile capture. "
                "Pass --enforce-eager (or enforce_eager=True) and retry. "
                "Relaxing this to PIECEWISE is a post-GPU-verification TODO "
                "(see epic/PHASE2.md S7)."
            )

    # ===================== chunking helpers =====================

    def _split_prompt_into_chunks(
        self, token_ids: list[int]
    ) -> list[tuple[int, int, str]]:
        """Return [(start_token, length, chunk_hash), ...] for full chunks only.

        Only emits whole ``chunk_size`` chunks (a trailing partial chunk is left
        to normal prefill); this keeps hashes position-independent and aligned.
        """
        out: list[tuple[int, int, str]] = []
        n = len(token_ids)
        start = 0
        while start + self._chunk_size <= n:
            chunk = token_ids[start : start + self._chunk_size]
            out.append((start, self._chunk_size, hash_chunk_tokens(chunk)))
            start += self._chunk_size
        return out

    # ===================== scheduler-side =====================

    def _membership(self) -> "EpicSchedulerIndex":
        """The scheduler-side membership oracle (the mirror index).

        All scheduler-side hit tests (selection, save dedup) MUST go through this
        so they see worker-store membership deterministically. Constructed only
        in SCHEDULER role; asserting here makes a wrong-role call fail loudly
        instead of silently missing against an empty worker store.
        """
        assert self._index is not None, (
            "EpicConnector._membership() called off the SCHEDULER role"
        )
        return self._index

    @staticmethod
    def _effective_selection(
        sel: "ReuseSelection", num_computed_tokens: int
    ) -> "ReuseSelection":
        """Fold the NATIVE prefix-cache extent into the selection (sparse mode).

        Positions [0, num_computed_tokens) already hold EXACT KV in the native
        prefix cache -- better than anything the chunk store can offer. So for
        M-derivation / external accounting the effective reused prefix A is
        ``max(store prefix extent, native extent)`` (the user-approved
        "cached_end == dynamic prefix extent" framing): this keeps natively
        computed positions out of M (no pointless recompute, no writes into
        shared native blocks) and lets the sparse path engage even when A is
        cached ONLY natively (store-evicted). Non-prefix hits overlapping the
        native extent are clamped to their part above it, carrying the head
        trim in ``src_offset`` so the worker loads only the uncovered tail.
        Pure function; safe under the multiple-calls-per-request contract.
        """
        nc = max(0, int(num_computed_tokens))
        eff_prefix = max(int(sel.prefix_extent), nc)
        if eff_prefix == sel.prefix_extent:
            return sel
        hits: list[NonPrefixHit] = []
        for h in sel.non_prefix_hits:
            lo = int(h.prompt_offset)
            hi = lo + max(0, int(h.length))
            if hi <= eff_prefix:
                continue  # fully covered by exact native KV -> nothing to load.
            new_lo = max(lo, eff_prefix)
            hits.append(
                NonPrefixHit(
                    chunk_hash=h.chunk_hash,
                    prompt_offset=new_lo,
                    old_pos_start=h.old_pos_start,
                    length=hi - new_lo,
                    src_offset=int(h.src_offset) + (new_lo - lo),
                )
            )
        return ReuseSelection(
            prefix_chunks=list(sel.prefix_chunks),
            prefix_extent=eff_prefix,
            non_prefix_hits=hits,
        )

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        token_ids = list(request.prompt_token_ids or [])
        if not token_ids:
            return 0, False

        chunks = self._split_prompt_into_chunks(token_ids)

        # SelectionStrategy (scheduler side, side-effect free): which content
        # chunks to reuse. EPIC = exact-hash prefix walk; the contiguous prefix
        # is loadable now, non-prefix hits are Phase 2 candidates (recorded).
        # ROOT-CAUSE FIX: query the scheduler-side mirror INDEX, never the worker
        # store (which is None in SCHEDULER role and was always empty before).
        sel = self._selection.select(
            request, num_computed_tokens, self._membership(), chunks
        )

        # Sparse mode: fold the native prefix extent in (see helper docstring)
        # BEFORE anything downstream (hit records, M-derivation stash, external
        # accounting) so every consumer sees one consistent selection.
        if self._sparse_forward:
            sel = self._effective_selection(sel, num_computed_tokens)

        self._non_prefix[request.request_id] = sel.non_prefix_hits
        # Stash the full selection for sparse-M derivation in build_connector_meta
        # (only used when epic_sparse_forward is on). Side-effect-free w.r.t. the
        # return value, so multiple calls per request remain safe.
        if self._sparse_forward:
            self._selections[request.request_id] = sel

        # EPIC diagnostics: record the SELECTION for every request (gated) so a
        # probe can see when a prompt fell ENTIRELY into the contiguous prefix
        # (num_non_prefix==0 -> LegoLink recompute is INERT). ``sparse_branch``
        # is provisional here (set True below only if the sparse path is taken).
        if getattr(self, "_debug_counters", False):
            self._record_selection(
                {
                    "request_id": str(request.request_id),
                    "N": len(token_ids),
                    "prefix_extent": int(sel.prefix_extent),
                    "num_non_prefix": len(sel.non_prefix_hits),
                    "non_prefix_offsets": [
                        int(h.prompt_offset) for h in sel.non_prefix_hits
                    ],
                    "sparse_branch": False,
                }
            )

        # ---- sparse (non-prefix) reuse path (Phase 2b, S3) ----------------
        # When sparse forward is ON *and* this prompt has non-prefix content
        # matches, report external = |A| + |B| (prefix extent + sum of
        # non-prefix hit lengths). The kv_cache_manager.allocate_slots is
        # count-based (total_computed = local + external -> block count), so
        # reporting |A|+|B| makes it allocate the blocks that cover all N
        # logical positions (the genuinely-new |C| tokens plus the reused
        # A/B regions). The scheduler core patch (S3) then overrides the
        # forward-row count to |M| and the num_computed advance to N-external.
        #
        # Invariants the scheduler asserts (:624 external<=N, :670 num_new>0):
        #   * external = |A| + Σ|B| <= N  -- A and the B chunks are disjoint
        #     subranges of [0, N) by construction (prefix walk + chunk hits),
        #     so their total length cannot exceed N.
        #   * num_new = N - external >= 1 -- guaranteed because we only take
        #     this branch when external < N (see `if external < n` below); the
        #     degenerate external == N (whole prompt cached) falls through to
        #     the prefix-only path which itself returns num_new only when > 0.
        if self._sparse_forward and sel.non_prefix_hits:
            n = len(token_ids)
            non_prefix_len = sum(max(0, int(h.length)) for h in sel.non_prefix_hits)
            external = sel.prefix_extent + non_prefix_len
            # Clamp: hits are disjoint subranges of [0, N), but defend against a
            # store-length overestimate so external never exceeds N (assert :624).
            external = min(external, n)
            num_new = max(0, external - num_computed_tokens)
            if 0 < external < n and num_new > 0 and self._sparse_fits_budget(
                request, sel, n, external
            ):
                # Mark this request sparse for the scheduler hooks; remember N
                # and external so the advance (N-external) can be derived later.
                self._matched_prefix[request.request_id] = sel.prefix_chunks
                self._sparse_reqs[request.request_id] = (n, external)
                self._native_computed[request.request_id] = num_computed_tokens
                # EPIC fix (root cause): the non-prefix B chunks counted into
                # ``external`` are treated as computed by the scheduler, so they
                # are NEVER forwarded. Their KV therefore MUST be loaded into the
                # paged cache or M queries attend to uninitialized slots. Mark a
                # load pending here (not only for prefix hits) so
                # build_connector_meta emits ChunkLoadSpecs for B even when there
                # is no prefix hit (the B-only case). See update_state_after_alloc
                # and build_connector_meta.
                self._loads_pending[request.request_id] = request
                # EPIC diagnostics (info, 1 line/req): observability for the GPU
                # smoke (expected B chunk count + offsets vs. external/|M|).
                logger.info(
                    "EPIC sparse match req=%s N=%d prefix_extent=%d "
                    "non_prefix_hits=%d offsets=%s external=%d num_new=%d",
                    request.request_id,
                    n,
                    sel.prefix_extent,
                    len(sel.non_prefix_hits),
                    [int(h.prompt_offset) for h in sel.non_prefix_hits],
                    external,
                    num_new,
                )
                if getattr(self, "_debug_counters", False):
                    self._bump_counter("sparse_match")
                    # Mark the diagnostic entry we just recorded for this request
                    # as having taken the sparse branch (LegoLink can recompute).
                    if self.debug_selection:
                        self.debug_selection[-1]["sparse_branch"] = True
                # Phase 1 loads (prefix chunks) still happen synchronously in
                # start_load_kv; non-prefix B loading now happens there too.
                return num_new, False
            # external == n (fully cached), no genuinely-new tokens, or the
            # one-step sparse prefill does not fit the scheduler token budget:
            # fall through to the prefix-only path (dense), which handles
            # num_new==0. _emit_sparse will NOT fire for this request because
            # no _sparse_reqs entry was registered (match time is the single
            # source of truth for "is sparse").

        # Report only the part of the prefix extent beyond what vLLM computed.
        # Align to block_size (chunk_size already is a block multiple, so the
        # extent is too).
        num_new = max(0, sel.prefix_extent - num_computed_tokens)
        if num_new == 0:
            self._matched_prefix.pop(request.request_id, None)
            return 0, False

        self._matched_prefix[request.request_id] = sel.prefix_chunks
        self._native_computed[request.request_id] = num_computed_tokens
        # Phase 1 loads synchronously inside start_load_kv -> async = False.
        return num_new, False

    def _sparse_fits_budget(
        self,
        request: "Request",
        sel: "ReuseSelection",
        n: int,
        external: int,
    ) -> bool:
        """Whether a ONE-STEP sparse prefill of this request fits the scheduler.

        The sparse plan is only valid when the scheduler schedules the FULL
        remaining prompt (N - external rows pre-override) in a single step and
        the runner then forwards |M| rows post-override. If either exceeds the
        per-step token budget (or the long-prefill threshold would truncate),
        the scheduler would chunk the prefill, the allocated blocks would not
        cover [0, N), and the stamped full-M positions would index out of
        range -- so decline sparse here and fall back to prefix-only reuse.
        """
        budget = int(getattr(self, "_max_sparse_rows", 0) or 0)
        threshold = int(getattr(self, "_long_prefill_threshold", 0) or 0)
        if budget <= 0 and threshold <= 0:
            return True  # no limits configured (unit-test connectors).

        rows_pre = n - external  # rows the scheduler budgets BEFORE override.
        plan = self._recompute.plan_recompute(
            request=request, selection=sel, block_size=self._block_size
        )
        rows_post = len(plan.recompute_offsets)  # |M|, forwarded after override.
        fits = (
            budget <= 0 or (rows_pre <= budget and rows_post <= budget)
        ) and (threshold <= 0 or rows_pre <= threshold)
        if not fits:
            logger.warning(
                "EPIC sparse declined req=%s: one-step prefill does not fit "
                "the scheduler budget (N=%d external=%d rows_pre=%d |M|=%d "
                "budget=%d long_prefill_threshold=%d); falling back to "
                "prefix-only reuse.",
                request.request_id,
                n,
                external,
                rows_pre,
                rows_post,
                budget,
                threshold,
            )
        return fits

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        # A load is pending whenever the scheduler allocated blocks for external
        # KV (num_external_tokens > 0) and we have a match record for the request.
        # In sparse mode the match record is the per-request sparse entry (B-only
        # requests have an empty prefix but still need their B chunks loaded);
        # in prefix-only mode it is _matched_prefix. Covering both here means the
        # B-only case (no prefix hit, only non-prefix B) still registers a load.
        if num_external_tokens > 0 and (
            request.request_id in self._matched_prefix
            or request.request_id in self._sparse_reqs
        ):
            self._loads_pending[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = EpicConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            token_ids = list(new_req.prompt_token_ids or [])
            block_ids = new_req.block_ids[0]

            # ---- sparse recompute set M (Phase 2b, S0/S1) ----
            # OFF by default -> m_set is None and every guard below is inert
            # (Phase 1/2a dense behavior). ON -> derive M via the LegoLink
            # RecomputePolicy and emit it for the scheduler/runner core patches.
            m_set: set[int] | None = None
            if self._sparse_forward:
                m_set = self._emit_sparse(meta, req_id, token_ids)

            # ---- loads ----
            if req_id in self._loads_pending:
                prefix_chunks = self._matched_prefix.get(req_id, [])
                # Native-computed trim: positions [0, nc) hold EXACT KV in the
                # native prefix cache (possibly in blocks SHARED with other
                # requests). Never scatter approximate store KV there -- skip
                # chunks fully below nc and head-trim chunks straddling it.
                nc = self._native_computed.get(req_id, 0)
                load = EpicReqLoad(req_id=req_id)
                for chunk_hash, start_token in prefix_chunks:
                    if start_token + self._chunk_size <= nc:
                        continue  # fully covered by exact native KV.
                    src_off = max(0, nc - start_token)
                    load_start = start_token + src_off
                    length = self._chunk_size - src_off
                    dst_slots = _slot_ids_from_blocks(
                        block_ids, self._block_size, load_start, length
                    )
                    load.chunks.append(
                        ChunkLoadSpec(
                            chunk_hash=chunk_hash,
                            dst_slot_ids=dst_slots,
                            old_pos_start=-1,  # resolved on worker from store
                            new_pos_start=load_start,
                            length=length,
                            src_offset=src_off,
                        )
                    )
                # EPIC fix (root cause): in sparse mode the non-prefix B chunks
                # are counted into ``external`` (computed) and never forwarded,
                # so their KV MUST be scattered into the paged cache here. We turn
                # each non-prefix hit into a real ChunkLoadSpec so the worker's
                # _load_chunk (scatter + PIC delta) populates the B slots. The PIC
                # delta is new_pos - stored.old_positions; setting
                # new_pos_start = h.prompt_offset makes the re-rotary target the
                # chunk's actual position in THIS prompt (old_pos_start stays -1:
                # the worker reads the real old positions from the store, exactly
                # like the prefix case). Outside sparse mode this is inert: the
                # non-prefix hits are only RECORDED (Phase 1 prefix-only path).
                emit_b_loads = self._sparse_forward and req_id in self._sparse_reqs
                for h in self._non_prefix.get(req_id, []):
                    if emit_b_loads:
                        length = int(h.length)
                        dst_slots = _slot_ids_from_blocks(
                            block_ids,
                            self._block_size,
                            int(h.prompt_offset),
                            length,
                        )
                        load.chunks.append(
                            ChunkLoadSpec(
                                chunk_hash=h.chunk_hash,
                                dst_slot_ids=dst_slots,
                                old_pos_start=-1,  # resolved on worker from store
                                new_pos_start=int(h.prompt_offset),
                                length=length,
                                # Head trim carried from _effective_selection
                                # (hit clamped to its part above the native
                                # extent): skip that many stored tokens.
                                src_offset=int(getattr(h, "src_offset", 0)),
                            )
                        )
                    # Always keep the raw hit record (observability / non-sparse).
                    load.non_prefix_hits.append(h)
                meta.add_load(load)
                # EPIC diagnostics (info, 1 line/req): chunk count + slot ranges.
                if load.chunks:
                    ranges = [
                        (s.new_pos_start, s.new_pos_start + s.length)
                        for s in load.chunks
                    ]
                    slot_lo = min(
                        (min(s.dst_slot_ids) for s in load.chunks if s.dst_slot_ids),
                        default=-1,
                    )
                    slot_hi = max(
                        (max(s.dst_slot_ids) for s in load.chunks if s.dst_slot_ids),
                        default=-1,
                    )
                    logger.info(
                        "EPIC load emit req=%s chunks=%d pos_ranges=%s "
                        "slot_range=[%d,%d]",
                        req_id,
                        len(load.chunks),
                        ranges,
                        slot_lo,
                        slot_hi,
                    )

            # ---- saves ----
            # Harvest full content chunks of this prompt that are NOT already
            # cached, so future requests can reuse them position-independently.
            # ROOT-CAUSE FIX: every save the scheduler emits here is registered
            # into the scheduler-side mirror index (_membership()) IMMEDIATELY, so
            # a later request in the SAME or a future step sees the chunk as
            # cached -- deterministically mirroring what the worker store will
            # hold after it executes this EpicReqSave. The dedup test below also
            # queries the index (not the empty worker store).
            index = self._membership()
            save = EpicReqSave(req_id=req_id)
            already_loaded = {
                h for h, _ in self._matched_prefix.get(req_id, [])
            }
            for start, length, h in self._split_prompt_into_chunks(token_ids):
                if h in already_loaded or index.contains(h):
                    # Already cached (or being loaded this step). Refresh LRU on
                    # the index so its eviction order tracks the worker store,
                    # which move_to_end's on the corresponding load/read.
                    index.touch(h)
                    continue
                # ---- S2 save guard (sparse mode only) ----
                # In sparse forward, only the M tokens are freshly computed; the
                # rest of a chunk's KV in the paged cache is reused/rotated KV
                # (aligned for THIS request's positions), not canonical. Saving
                # such a chunk would poison the store. Emit a save only when the
                # whole chunk is in M (every token genuinely recomputed). Non-
                # sparse mode (m_set is None) is unchanged: always save.
                if m_set is not None and not all(
                    pos in m_set for pos in range(start, start + length)
                ):
                    continue
                slots = _slot_ids_from_blocks(
                    block_ids, self._block_size, start, length
                )
                positions = list(range(start, start + length))
                save.chunk_hashes.append(h)
                save.chunk_slot_ids.append(slots)
                save.chunk_positions.append(positions)
                # Mirror the worker store membership: registering with the same
                # byte accounting + LRU policy means the index evicts in lock-step
                # with the worker store (same save sequence -> same eviction).
                # old_pos == start (this prompt's positions) for diagnostics; the
                # worker resolves the real old positions from its own store.
                index.register(h, length, old_pos_start=start)
            if save.chunk_hashes:
                meta.add_save(save)

        # ---- fusion mask plan (Phase 2a) ----
        # Build a LegoLink mask plan covering this step. In Phase 2a the runner
        # still forwards every token, so the plan is a *dense* causal-equivalent
        # mask (gate OFF, reused_offsets empty). It is correct (== standard
        # causal) yet exercises the full mask-install path; Phase 2b will fill
        # recompute/reused offsets and set gate using the SAME wiring.
        if self._fusion_enabled:
            meta.fusion_mask = self._build_fusion_mask_plan(scheduler_output)

        # Reset per-step scheduler state. (_sparse_reqs is consumed by
        # _emit_sparse above, which copies external into EpicReqSparse, so the
        # scheduler core hooks read everything they need off ``meta`` -- the
        # connector stays a pure metadata producer.)
        self._loads_pending.clear()
        self._matched_prefix.clear()
        self._non_prefix.clear()
        self._selections.clear()
        self._sparse_reqs.clear()
        self._native_computed.clear()
        return meta

    def _emit_sparse(
        self,
        meta: EpicConnectorMetadata,
        req_id: str,
        token_ids: list[int],
    ) -> set[int] | None:
        """Derive M for one request and emit it on the metadata (Phase 2b, S1).

        Returns the M set (logical positions) for the S2 save guard, or ``None``
        if there is no sparse plan (degenerate / dense -> caller skips the
        guard, i.e. behaves like non-sparse). Only called when
        ``epic_sparse_forward`` is on.
        """
        n = len(token_ids)
        if n == 0:
            return None

        # EPIC fix (consistency, root cause): match time is the SINGLE source of
        # truth for "is sparse". Emit only for requests that
        # get_num_new_matched_tokens actually REGISTERED sparse this step
        # (_sparse_reqs). Deriving M here from a fresh/stashed selection for an
        # unregistered request used to emit a plan whose external was never
        # counted by the scheduler -> computed_advance defaulted to N-0 -> the
        # post-schedule length invariant (num_computed != N) raised and killed
        # the engine. Cases that fall out naturally now: sparse branch declined
        # (external==N, num_new<=0, budget guard), pure-new prompts, and
        # requests whose match ran in an earlier (deferred) step.
        rec = self._sparse_reqs.get(req_id)
        if rec is None:
            return None

        # The selection recorded at match time. Registration always stashes it
        # (same code path); a missing entry means scheduler-side state was
        # corrupted -- refuse to emit rather than derive M from a fresh select
        # that may disagree with the counted external.
        sel = self._selections.get(req_id)
        if sel is None:
            logger.error(
                "EPIC sparse emit skipped req=%s: registered sparse but no "
                "stashed selection (connector state bug); forwarding dense.",
                req_id,
            )
            return None

        # Defensive: registration requires non-prefix hits, so this cannot
        # trigger unless the stash was mutated. Dense (None) keeps the
        # scheduler/runner consistent either way.
        if not sel.non_prefix_hits:
            logger.info(
                "EPIC sparse skip req=%s: no non-prefix hits -> dense forward",
                req_id,
            )
            return None

        # The policy's _seq_len() reads request.prompt_token_ids first; pass a
        # minimal shim carrying the exact N (the V1 Request isn't in scope in
        # build_connector_meta, only the new-req descriptor's token ids).
        plan: RecomputePlan = self._recompute.plan_recompute(
            request=_PromptLenShim(token_ids),
            selection=sel,
            block_size=self._block_size,
        )
        if not plan.recompute_offsets:
            return None  # dense / degenerate -> no sparse emission, no guard.

        seq_len = plan.seq_len or n
        # computed_advance = N - external. ``external`` is what the scheduler
        # already counted via num_external_computed_tokens (= |A|+|B|), recorded
        # in _sparse_reqs by get_num_new_matched_tokens (guaranteed present by
        # the registration guard above). Advancing by this (not len(M)) avoids
        # double-counting the link/last M tokens that overlap reused B positions.
        external = rec[1]
        computed_advance = max(0, seq_len - external)

        logger.info(
            "EPIC sparse emit req=%s N=%d |M|=%d external=%d advance=%d",
            req_id,
            seq_len,
            len(plan.recompute_offsets),
            external,
            computed_advance,
        )
        meta.add_sparse(
            EpicReqSparse(
                req_id=req_id,
                sparse_positions=list(plan.recompute_offsets),
                full_seq_len=seq_len,
                computed_advance=computed_advance,
            )
        )
        if getattr(self, "_debug_counters", False):
            self._bump_counter("sparse_emit")
        return set(plan.recompute_offsets)

    # ----- core-hook interface (S0): scheduler calls these in the NEXT batch -----
    # Defined here so the scheduler-core patch (S3) has a stable connector API to
    # bind against. They read ONLY the per-step metadata the connector already
    # produced (EpicConnectorMetadata.sparse); they do NOT touch core V1 files.

    def get_sparse_num_scheduled_tokens(
        self, meta: "EpicConnectorMetadata", req_id: str
    ) -> int | None:
        """Number of token rows to forward for ``req_id`` under sparse reuse.

        Returns ``len(M)`` if a sparse plan exists for the request this step,
        else ``None`` (scheduler keeps its normal contiguous count). Pure lookup;
        no side effects.
        """
        for sp in meta.sparse:
            if sp.req_id == req_id and sp.is_sparse:
                return len(sp.sparse_positions)
        return None

    def get_sparse_positions(
        self, meta: "EpicConnectorMetadata", req_id: str
    ) -> list[int] | None:
        """Explicit per-token RoPE positions (M's logical positions) for ``req_id``.

        The runner core patch (next batch) will write these into the captured
        ``positions`` buffer instead of ``computed_prefix + arange`` (PHASE2 §2).
        Returns ``None`` when there is no sparse plan for the request.
        """
        for sp in meta.sparse:
            if sp.req_id == req_id and sp.is_sparse:
                return list(sp.sparse_positions)
        return None

    def has_sparse_requests(self, meta: "EpicConnectorMetadata") -> bool:
        """True iff any request this step has a sparse (non-dense) plan."""
        return any(sp.is_sparse for sp in meta.sparse)

    def get_sparse_computed_advance(
        self, meta: "EpicConnectorMetadata", req_id: str
    ) -> int | None:
        """How much to advance ``num_computed_tokens`` for ``req_id`` this step.

        Returns ``N - external`` (so num_computed converges to N at prefill end)
        if a sparse plan exists, else ``None`` (scheduler keeps the default
        ``+= num_scheduled_tokens`` advance). NOT ``len(M)``: M overlaps reused
        KV positions (link/last tokens inside B), so advancing by len(M) would
        double-count. Pure lookup; no side effects.
        """
        for sp in meta.sparse:
            if sp.req_id == req_id and sp.is_sparse and sp.computed_advance > 0:
                return sp.computed_advance
        return None

    # ----- schedule-time hook (S3): called DURING the scheduling loop, BEFORE
    # build_connector_meta, so it reads live scheduler-side state (_sparse_reqs)
    # rather than the per-step metadata (which isn't built yet at that point).

    def is_sparse_request(self, req_id: str) -> bool:
        """True iff ``req_id`` is being scheduled through the sparse reuse path
        this step (recorded by ``get_num_new_matched_tokens``).

        Used by the scheduler to set ``delay_cache_blocks=True`` so the approx
        (reused/rotated) KV blocks of a sparse request are NOT registered into
        the native prefix cache (avoids poisoning it; DESIGN / S3 brief).
        """
        return req_id in self._sparse_reqs

    def _build_fusion_mask_plan(
        self, scheduler_output: SchedulerOutput
    ) -> FusionMaskPlan | None:
        """Derive this step's LegoLink fusion-mask plan (Phase 2a: dense).

        Uses ``RecomputePolicy.plan_recompute`` so the EPIC strategy seam owns
        what M is. Phase 1/2a's ``LegoLinkRecompute(phase1_dense=True)`` returns
        an empty plan -> dense causal-equivalent mask. The max logical position
        the mask must address is the longest scheduled sequence this step.
        """
        # Upper bound on the logical positions the mask must address this step.
        # The worker further widens capacity from attn_meta.max_seq_len, so an
        # under-estimate here is corrected on the worker; we only need a sane
        # nonzero hint.
        max_seq_len = 0
        for n in scheduler_output.num_scheduled_tokens.values():
            max_seq_len = max(max_seq_len, int(n))
        for new_req in scheduler_output.scheduled_new_reqs:
            max_seq_len = max(max_seq_len, len(new_req.prompt_token_ids or []))
        if max_seq_len <= 0:
            return None

        # Strategy-owned M. With selection=None the policy returns the dense
        # plan in BOTH 2a (phase1_dense) and 2b (sparse) modes -- in sparse
        # mode the fusion mask intentionally stays causal-equivalent (all
        # logical KV positions are live; per-request sparse M travels via
        # EpicReqSparse instead). RecomputePlan.is_sparse is False, so the
        # gate stays OFF and the mask == standard causal.
        plan: RecomputePlan = self._recompute.plan_recompute(
            request=None,  # type: ignore[arg-type]
            selection=None,  # type: ignore[arg-type]
            block_size=self._block_size,
        )
        return FusionMaskPlan(
            enabled=True,
            seq_len=max_seq_len,
            recompute_offsets=list(plan.recompute_offsets),
            reused_offsets=list(plan.reused_offsets),
            gate=plan.is_sparse,
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        # Phase 1 saves are harvested during the prefill forward (save_kv_layer)
        # for new requests; nothing to defer here. Return synchronous-free.
        return False, None

    # ===================== worker-side =====================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv_caches = kv_caches
        self._layer_names = list(kv_caches.keys())

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Scatter cached chunks (PIC-rotated) and install the fusion mask.

        Phase 1 is an eager, one-shot all-layer load. Layer-wise async pipelining
        (overlapping load with compute via wait_for_layer_load) is a TODO.

        Phase 2a additionally installs the LegoLink FlexAttention mask_mod onto
        the attention layers (via forward_context) before the model forward.
        """
        meta = self._get_connector_metadata()
        if not isinstance(meta, EpicConnectorMetadata):
            return

        # DIAGNOSTIC: re-arm the per-step first-chunk check-load latch.
        self._check_load_done = False

        # DIAGNOSTIC (Implement 3): log the effective sparse plan the worker
        # received this step -- the *actual* values flowing into the runner /
        # FlexAttention forward, so a GPU run can confirm sparse_positions and
        # the implied seq_len match what the scheduler intended (len, first3,
        # last3, full_seq_len, computed_advance). Cheap; only fires when sparse
        # descriptors are present.
        for sp in meta.sparse:
            if not sp.is_sparse:
                continue
            pos = sp.sparse_positions
            logger.info(
                "EPIC worker sparse plan req=%s |M|=%d first3=%s last3=%s "
                "full_seq_len=%d computed_advance=%d (implied_seq_len=%d)",
                sp.req_id,
                len(pos),
                pos[:3],
                pos[-3:],
                sp.full_seq_len,
                sp.computed_advance,
                (pos[-1] + 1) if pos else 0,
            )

        # Install / refresh the fusion mask BEFORE chunk loads so it is ready by
        # the time the forward runs. Independent of whether there are loads.
        self._install_fusion_mask(forward_context, meta)

        # DIAGNOSTIC: when check-load is requested, make the load path's state
        # unambiguous at WARNING level -- so "no check_load line" can be told
        # apart from "loads never ran" / "flag never reached the worker".
        # Record the load count on the RELIABLE counters channel (gated on
        # epic_debug_counters, which is surfaced via RESULT_JSON). This shows up
        # in the printed counters dict regardless of log level/process, so
        # loads_emitted>0 with check_load_calls==0 pinpoints whether the
        # check-load flag reached the worker vs whether loads ran at all.
        if getattr(self, "_debug_counters", False):
            self._bump_counter(
                "loads_emitted", sum(len(load.chunks) for load in meta.loads)
            )
        if getattr(self, "_debug_check_load", False):
            n_chunks = sum(len(load.chunks) for load in meta.loads)
            logger.warning(
                "EPIC start_load_kv: debug_check_load=True loads=%d chunks=%d "
                "alignment=%s store=%s -- check_load lines should follow if "
                "chunks>0 (else loads are not happening on the worker)",
                len(meta.loads),
                n_chunks,
                self._alignment is not None,
                self._store is not None,
            )

        if not meta.loads or self._alignment is None or self._store is None:
            return

        for load in meta.loads:
            for spec in load.chunks:
                stored = self._store.get(spec.chunk_hash)
                if stored is None:
                    # MIRROR DRIFT (non-fatal, loud): the scheduler's index said
                    # this chunk was cached and emitted a load for it, but the
                    # worker store has no tensors for it. The scheduler index and
                    # the worker store can legitimately diverge whenever the
                    # WORKER skips a save the scheduler counted as registered --
                    # i.e. any path where save_kv_layer returns early without
                    # calling self._store.put():
                    #   * unsupported / non-FlashAttention KV layout (the
                    #     ``shape[0] == 2 and dim() == 5`` guard in save_kv_layer),
                    #   * a chunk larger than the whole byte budget (EpicChunkStore
                    #     .put drops it; the index also drops oversize chunks, so
                    #     this specific case stays consistent),
                    #   * genuine eviction races between match and load.
                    # In all cases we skip ONLY this chunk's load (the M queries
                    # over its positions will read uninitialized KV in sparse mode
                    # -- a correctness risk that the error log surfaces) and let
                    # the rest of the request load normally.
                    # TODO(drift): report misses back to the scheduler index via
                    # update_connector_output(KVConnectorOutput) so it can evict
                    # the phantom entry. KVConnectorOutput has no generic
                    # connector-data field today (outputs.py:195), so wiring a
                    # typed miss-set is a follow-up; logging is the first-pass.
                    logger.error(
                        "EPIC worker load-miss (mirror drift): chunk_hash=%s "
                        "req=%s new_pos_start=%d length=%d not in worker store; "
                        "skipping this chunk's load (scheduler index believed it "
                        "cached -- likely a worker save skipped for an unsupported "
                        "KV layout). Remaining chunks load normally.",
                        spec.chunk_hash,
                        load.req_id,
                        spec.new_pos_start,
                        spec.length,
                    )
                    continue
                self._load_chunk(stored, spec)

    def _install_fusion_mask(
        self,
        forward_context: "ForwardContext",
        meta: EpicConnectorMetadata,
    ) -> None:
        """Install the LegoLink mask_mod onto FlexAttention layers (Phase 2a).

        Non-invasive: FlexAttention's ``forward`` already auto-picks up a
        ``logical_mask_mod`` attribute set on the attention layer object (see
        ``vllm/v1/attention/backends/flex_attention.py`` ~L1122). We set that
        attribute on each attention layer reachable via
        ``forward_context.no_compile_layers`` (== ``static_forward_context``,
        the ``Attention`` modules that are passed as ``layer`` to
        ``impl.forward``).

        Recompile avoidance: the mask builder hands back the SAME mask_mod
        object every step (only the backing tensor *contents* change), so
        FlexAttention's identity check (``logical_mask_mod is not layer_mask_mod``)
        fires once and the compiled kernel is reused thereafter.
        """
        plan = meta.fusion_mask
        layers = getattr(forward_context, "no_compile_layers", None)
        if not layers:
            return

        if plan is None or not plan.enabled:
            # Fusion disabled this step: remove any mask we previously installed
            # so the backend reverts to its default causal mask. Leave layers we
            # never touched alone.
            for name in self._flex_layers_patched:
                layer = layers.get(name)
                if layer is not None and hasattr(layer, "logical_mask_mod"):
                    try:
                        delattr(layer, "logical_mask_mod")
                    except AttributeError:
                        layer.logical_mask_mod = None
            self._flex_layers_patched.clear()
            return

        # Build/refresh the single stable mask_mod object from the plan tensors.
        # Fixed capacity -> the mask_mod object identity never changes across
        # requests, so FlexAttention reuses its compiled kernel. Tensors live on
        # the KV-cache device so the on-device mask_mod can index them.
        if self._mask_builder is None:
            cap = self._mask_capacity or max(plan.seq_len, 1)
            device = "cpu"
            for kv in self._kv_caches.values():
                device = str(kv.device)
                break
            self._mask_builder = LegoLinkMaskBuilder(capacity=cap, device=device)
        # Translate the picklable plan into a RecomputePlan the builder fills.
        recompute_plan = RecomputePlan(
            recompute_offsets=list(plan.recompute_offsets),
            reused_offsets=list(plan.reused_offsets),
            seq_len=plan.seq_len,
        )
        # No FlexAttentionMetadata here (we run before the backend builds its
        # per-step metadata); pass None and let the builder size from the plan.
        mask_mod = self._mask_builder.build_logical_mask_mod(
            recompute_plan, attn_meta=None  # type: ignore[arg-type]
        )

        for name in self._flex_install_targets(layers):
            layer = layers.get(name)
            if layer is None:
                continue
            # Setting the attribute is the documented FlexAttention hook.
            layer.logical_mask_mod = mask_mod
            self._flex_layers_patched.add(name)

    def _flex_install_targets(self, layers: dict[str, Any]) -> list[str]:
        """Names of attention layers to install the mask on.

        Phase 2a installs on every attention layer (full-forward, mask ==
        causal). Phase 2b can restrict this to check/fusion layers via the
        recompute plan.
        """
        return list(layers.keys())

    def _load_chunk(self, stored: StoredChunk, spec: ChunkLoadSpec) -> None:
        # Head trim (native-computed-region protection): skip the first
        # ``src_offset`` stored tokens; they are covered by exact native KV.
        src_off = max(0, int(getattr(spec, "src_offset", 0)))
        length = min(max(0, stored.length - src_off), spec.length)
        if length == 0:
            return
        if getattr(self, "_debug_counters", False):
            self._bump_counter("chunks_loaded")
        # Defensive defaults: unit tests build a worker via object.__new__ and
        # set attributes by hand, so the diagnostic flags may be absent. getattr
        # keeps the check inert (off) on those partially-built instances.
        debug_check = getattr(self, "_debug_check_load", False)
        check_done = getattr(self, "_check_load_done", True)
        # Destination slots and positions. new_pos_start already points at the
        # FIRST loaded (post-trim) token, so only the source side is offset.
        new_positions = torch.arange(
            spec.new_pos_start, spec.new_pos_start + length, dtype=torch.int64
        )
        old_positions = stored.old_positions[src_off : src_off + length].to(
            torch.int64
        )

        for layer_name in self._layer_names:
            kv_cache = self._kv_caches.get(layer_name)
            if kv_cache is None or layer_name not in stored.k_per_layer:
                continue
            device = kv_cache.device
            k_cpu = stored.k_per_layer[layer_name][src_off : src_off + length]
            v_cpu = stored.v_per_layer[layer_name][src_off : src_off + length]
            k = k_cpu.to(device, non_blocking=True)
            v = v_cpu.to(device, non_blocking=True)

            # AlignmentStrategy: EPIC = PIC delta re-rotary (no fake_q hack;
            # see pic.py). CacheBlend would plug IdentityAlignment here.
            assert self._alignment is not None
            k = self._alignment.align_keys(
                k,
                old_positions.to(device),
                new_positions.to(device),
                layer_name,
            )

            self._scatter_kv(kv_cache, k, v, spec.dst_slot_ids[:length])

            # DIAGNOSTIC (Implement 2): re-read the dst slots and compare to the
            # exact aligned K/V we just scattered. Catches paged-cache layout /
            # stride / aliasing bugs in situ. First loaded chunk only (latched).
            if debug_check and not check_done:
                self._check_scatter_fidelity(
                    kv_cache, k, v, spec.dst_slot_ids[:length], layer_name
                )

        # End-of-chunk: arm the latch so subsequent chunks this step are silent.
        if debug_check:
            self._check_load_done = True

    def _kv_banks(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return flat (k_bank, v_bank) views for the supported FA layout.

        FlashAttention V1 layout: (2, num_blocks, block_size, num_kv_heads,
        head_size); flattened to (num_blocks*block_size, H, D) so a slot index
        (block_id*block_size + offset) indexes a single token row. Returns None
        for any other layout (Phase 1 supports default FlashAttention only).
        """
        shape = kv_cache.shape
        if shape[0] == 2 and kv_cache.dim() == 5:
            num_blocks, block_size = shape[1], shape[2]
            k_bank = kv_cache[0].reshape(num_blocks * block_size, *shape[3:])
            v_bank = kv_cache[1].reshape(num_blocks * block_size, *shape[3:])
            return k_bank, v_bank
        return None

    def _scatter_kv(
        self,
        kv_cache: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_ids: list[int],
    ) -> None:
        """Scatter [num_tokens, heads, head_size] K/V into the paged cache.

        FlashAttention V1 layout: (2, num_blocks, block_size, num_kv_heads,
        head_size). slot = block_id*block_size + offset.
        """
        slots = torch.as_tensor(slot_ids, device=kv_cache.device, dtype=torch.long)
        banks = self._kv_banks(kv_cache)
        if banks is not None:
            k_bank, v_bank = banks
            k_bank[slots] = k.to(k_bank.dtype)
            v_bank[slots] = v.to(v_bank.dtype)
        else:
            # Unknown layout: skip (Phase 1 supports default FlashAttention only).
            logger.warning_once(
                "EPIC: unsupported kv cache layout %s; skipping load.",
                tuple(kv_cache.shape),
            )

    def _check_scatter_fidelity(
        self,
        kv_cache: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_ids: list[int],
        layer_name: str,
    ) -> None:
        """Read back the dst slots and compare to the scattered K/V (Implement 2).

        Logs ONE info line per call with allclose + per-bank max-abs-diff. A
        non-zero diff means the scatter did not land in the slots we read back
        (layout/stride/aliasing bug) -- exactly the failure class that produces a
        correct-looking plan but a corrupted forward. Pure read-back + tensor
        math; no effect on the cache contents.
        """
        if getattr(self, "_debug_counters", False):
            self._bump_counter("check_load_calls")
        result = check_scatter_fidelity(kv_cache, k, v, slot_ids)
        if result is None:
            if getattr(self, "_debug_counters", False):
                self._bump_counter("check_load_skip")
            # Unsupported layout is itself a likely root cause (scatter is a
            # no-op) -> warn, not info, so it survives VLLM_LOGGING_LEVEL=WARNING.
            logger.warning(
                "EPIC check_load layer=%s SKIP (unsupported layout %s) -- "
                "scatter may be a NO-OP for this backend",
                layer_name,
                tuple(kv_cache.shape),
            )
            return
        k_ok, k_diff, v_ok, v_diff = result
        if not (k_ok and v_ok) and getattr(self, "_debug_counters", False):
            self._bump_counter("check_load_mismatch")
        # This only runs when epic_debug_check_load is on (an explicit debug
        # request), so ALWAYS log at WARNING -- otherwise a clean read-back at
        # info is silently dropped under VLLM_LOGGING_LEVEL=WARNING and the user
        # cannot tell "scatter is fine" from "the check never ran".
        logger.warning(
            "EPIC check_load layer=%s n=%d k_allclose=%s k_maxabsdiff=%.3e "
            "v_allclose=%s v_maxabsdiff=%.3e%s",
            layer_name,
            len(slot_ids),
            k_ok,
            k_diff,
            v_ok,
            v_diff,
            "" if (k_ok and v_ok) else "  <-- SCATTER FIDELITY MISMATCH",
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Phase 1 loads synchronously in start_load_kv -> nothing to wait for.
        # TODO(Phase 2): layer-wise async load pipelining.
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Harvest this layer's KV for to-be-saved chunks into the CPU store."""
        meta = self._connector_metadata
        if not isinstance(meta, EpicConnectorMetadata) or not meta.saves:
            return
        if self._store is None:
            # save_kv_layer only runs in WORKER role, where the store exists.
            return
        shape = kv_layer.shape
        # MIRROR-DRIFT SOURCE: if the KV layout is not the supported
        # FlashAttention (2, num_blocks, block_size, ...) shape, the worker skips
        # the save entirely and the chunk NEVER enters the worker store -- but the
        # scheduler index already registered it (it registers unconditionally at
        # save-emit, having no visibility into the worker layout). The result is a
        # phantom index entry whose later load triggers the loud load-miss in
        # start_load_kv. This is an accepted, logged drift for unsupported
        # layouts (Phase 1 targets default FlashAttention only).
        if not (shape[0] == 2 and kv_layer.dim() == 5):
            return
        num_blocks, block_size = shape[1], shape[2]
        k_bank = kv_layer[0].reshape(num_blocks * block_size, *shape[3:])
        v_bank = kv_layer[1].reshape(num_blocks * block_size, *shape[3:])

        for save in meta.saves:
            for ci, chunk_hash in enumerate(save.chunk_hashes):
                slot_ids = save.chunk_slot_ids[ci]
                positions = save.chunk_positions[ci]
                slots = torch.as_tensor(
                    slot_ids, device=kv_layer.device, dtype=torch.long
                )
                k = k_bank[slots].detach().to("cpu")
                v = v_bank[slots].detach().to("cpu")
                k = self._store.maybe_pin(k)
                v = self._store.maybe_pin(v)

                stored = self._store.get(chunk_hash)
                if stored is None:
                    stored = StoredChunk(
                        chunk_hash=chunk_hash,
                        length=len(slot_ids),
                        old_positions=torch.as_tensor(positions, dtype=torch.int64),
                    )
                stored.k_per_layer[layer_name] = k
                stored.v_per_layer[layer_name] = v
                # put() refreshes bytes / LRU; safe to call repeatedly per layer.
                self._store.put(stored)

    def wait_for_save(self):
        # All saves are synchronous CPU copies in save_kv_layer.
        # TODO(Phase 2): async D2H copy with a stream + wait here.
        return
