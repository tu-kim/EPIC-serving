# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the gpu_smoke link-sweep config + helpers (Implement 1).

The link sweep itself needs a GPU + a model, but the config it hands to each
engine and the divergence/head-compare helpers are pure and must be correct
before any GPU time is spent (a wrong extra_config means the sweep silently
measures the wrong thing). We verify here, on CPU, with no tokenizer / GPU.
"""

from tests.v1.kv_connector.unit.epic.gpu_smoke import (
    _LINK_SWEEP,
    _MACHINERY_PASS_THRESHOLD,
    DEFAULT_CHUNK_SIZE,
    _epic_kv_config,
    _first_divergence,
    _head_compare,
)


# --- config build for the sweep -------------------------------------------


def test_link_sweep_includes_b_full_control():
    # The decisive control (M == all of B) requires a link >= the B chunk size.
    assert max(_LINK_SWEEP) >= DEFAULT_CHUNK_SIZE
    # And progressively smaller links to trace the approximation curve.
    assert _LINK_SWEEP == sorted(_LINK_SWEEP, reverse=True)
    assert _MACHINERY_PASS_THRESHOLD > 0.5


def test_epic_kv_config_passes_link_tokens():
    cfg = _epic_kv_config(sparse=True, fusion=True, link_tokens=256)
    extra = cfg["kv_connector_extra_config"]
    assert extra["epic_sparse_forward"] is True
    assert extra["epic_fusion_mask"] is True
    assert extra["epic_link_tokens"] == 256


def test_epic_kv_config_passes_debug_check_load():
    cfg = _epic_kv_config(sparse=True, link_tokens=8, debug_check_load=True)
    extra = cfg["kv_connector_extra_config"]
    assert extra["epic_debug_check_load"] is True
    assert extra["epic_link_tokens"] == 8


def test_epic_kv_config_omits_unset_flags():
    # Dense run: no sparse/link/debug keys leak into the extra config.
    cfg = _epic_kv_config(sparse=False)
    assert "kv_connector_extra_config" not in cfg


def test_epic_kv_config_link_zero_is_emitted():
    # link_tokens=0 is a valid (degenerate) value and must be passed, not dropped
    # by a truthiness check.
    cfg = _epic_kv_config(sparse=True, link_tokens=0)
    assert cfg["kv_connector_extra_config"]["epic_link_tokens"] == 0


# --- divergence / head-compare helpers ------------------------------------


def test_first_divergence_identical():
    assert _first_divergence([1, 2, 3], [1, 2, 3]) == -1


def test_first_divergence_mid():
    assert _first_divergence([1, 2, 9, 4], [1, 2, 3, 4]) == 2


def test_first_divergence_length_mismatch_otherwise_equal():
    assert _first_divergence([1, 2, 3], [1, 2, 3, 4]) == 3


def test_first_divergence_at_zero():
    assert _first_divergence([9], [1]) == 0


class _FakeTok:
    def decode(self, ids):
        return f"<{ids[0]}>"


def test_head_compare_marks_diff_rows():
    out = _head_compare(_FakeTok(), [1, 2, 3], [1, 9, 3], k=3)
    lines = out.splitlines()
    # Header + 3 rows.
    assert len(lines) == 4
    # Row 1 (the differing one) is marked.
    assert "DIFF" in lines[2]
    # The matching rows are not.
    assert "DIFF" not in lines[1]
    assert "DIFF" not in lines[3]


def test_head_compare_handles_length_mismatch():
    out = _head_compare(_FakeTok(), [1, 2], [1, 2, 3], k=3)
    # The third sparse-only row shows dense as '--' and is marked DIFF.
    assert "--" in out
    assert "DIFF" in out
