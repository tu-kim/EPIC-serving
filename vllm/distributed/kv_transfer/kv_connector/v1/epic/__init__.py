# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC (Efficient Position-Independent Caching) KV connector, Phase 1.

Non-contiguous KV reuse (arXiv:2410.15332) ported to vLLM V1 as an isolated
KVConnectorBase_V1 implementation. Phase 1 reuses content-matched chunks only
within the new request's contiguous prefix, with PIC delta re-rotary.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.epic.chunk_store import (
    EpicChunkStore,
    StoredChunk,
    hash_chunk_tokens,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
    EpicConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.fusion_mask import (
    FusionMaskTensors,
    build_legolink_mask_mod,
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
    FusionMaskBuilder,
    IdentityAlignment,
    LegoLinkMaskBuilder,
    LegoLinkRecompute,
    PicAlignment,
    RecomputePlan,
    RecomputePolicy,
    ReuseSelection,
    SelectionStrategy,
)

__all__ = [
    "EpicConnector",
    "EpicConnectorMetadata",
    "ChunkLoadSpec",
    "EpicReqLoad",
    "EpicReqSave",
    "EpicReqSparse",
    "FusionMaskPlan",
    "NonPrefixHit",
    "FusionMaskTensors",
    "build_legolink_mask_mod",
    "EpicChunkStore",
    "StoredChunk",
    "hash_chunk_tokens",
    "PICRotator",
    # strategy seams
    "SelectionStrategy",
    "AlignmentStrategy",
    "RecomputePolicy",
    "FusionMaskBuilder",
    "ReuseSelection",
    "RecomputePlan",
    "EpicSelection",
    "PicAlignment",
    "IdentityAlignment",
    "LegoLinkRecompute",
    "LegoLinkMaskBuilder",
]
