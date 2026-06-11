# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC connector metadata build / serialization tests (CPU-only)."""

import pickle

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
    ChunkLoadSpec,
    EpicConnectorMetadata,
    EpicReqLoad,
    EpicReqSave,
    FusionMaskPlan,
    NonPrefixHit,
)


def test_metadata_is_kvconnector_metadata():
    meta = EpicConnectorMetadata()
    assert isinstance(meta, KVConnectorMetadata)
    assert meta.loads == []
    assert meta.saves == []


def test_build_and_roundtrip():
    meta = EpicConnectorMetadata()
    load = EpicReqLoad(req_id="r0")
    load.chunks.append(
        ChunkLoadSpec(
            chunk_hash="abc",
            dst_slot_ids=[0, 1, 2, 3],
            old_pos_start=10,
            new_pos_start=0,
            length=4,
        )
    )
    load.non_prefix_hits.append(
        NonPrefixHit(chunk_hash="def", prompt_offset=8, old_pos_start=20, length=4)
    )
    meta.add_load(load)

    save = EpicReqSave(req_id="r0")
    save.chunk_hashes.append("abc")
    save.chunk_slot_ids.append([0, 1, 2, 3])
    save.chunk_positions.append([0, 1, 2, 3])
    meta.add_save(save)

    # Phase 2a fusion-mask plan (picklable: plain ints/lists, no tensors).
    meta.fusion_mask = FusionMaskPlan(
        enabled=True,
        seq_len=8,
        recompute_offsets=[],
        reused_offsets=[],
        gate=False,
    )

    # Metadata crosses the scheduler->worker process boundary: must pickle.
    blob = pickle.dumps(meta)
    restored = pickle.loads(blob)

    assert isinstance(restored, EpicConnectorMetadata)
    assert len(restored.loads) == 1
    assert restored.loads[0].req_id == "r0"
    assert restored.loads[0].chunks[0].chunk_hash == "abc"
    assert restored.loads[0].chunks[0].dst_slot_ids == [0, 1, 2, 3]
    assert restored.loads[0].non_prefix_hits[0].prompt_offset == 8
    assert len(restored.saves) == 1
    assert restored.saves[0].chunk_hashes == ["abc"]
    assert restored.saves[0].chunk_positions == [[0, 1, 2, 3]]
    assert restored.fusion_mask is not None
    assert restored.fusion_mask.enabled is True
    assert restored.fusion_mask.seq_len == 8
    assert restored.fusion_mask.gate is False


def test_metadata_default_fusion_mask_none():
    meta = EpicConnectorMetadata()
    assert meta.fusion_mask is None


def test_fusion_mask_plan_sparse_roundtrip():
    """Phase 2b-shaped plan (sparse M + gate) survives pickle."""
    plan = FusionMaskPlan(
        enabled=True,
        seq_len=12,
        recompute_offsets=[4, 5, 8],
        reused_offsets=[0, 1, 2, 3],
        gate=True,
    )
    restored = pickle.loads(pickle.dumps(plan))
    assert restored.recompute_offsets == [4, 5, 8]
    assert restored.reused_offsets == [0, 1, 2, 3]
    assert restored.gate is True
