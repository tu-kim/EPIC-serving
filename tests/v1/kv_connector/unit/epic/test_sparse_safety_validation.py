# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC Phase 2b S7 sparse-mode safety validation tests (CPU-only).

Validates ``EpicConnector._validate_sparse_safety``: when sparse forward is on,
the connector requires the FlexAttention backend and enforce_eager, failing fast
with an actionable error otherwise. The method does not use ``self`` (pure config
inspection), so it is exercised here as an unbound method against lightweight
config stand-ins -- no GPU, no real VllmConfig build, no PICRotator.
"""

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (
    EpicConnector,
)


class _Backend:
    """Stand-in for AttentionBackendEnum: only ``.name`` is read by the check."""

    def __init__(self, name):
        self.name = name


def _config(backend_name, enforce_eager):
    """Minimal duck-typed VllmConfig for _validate_sparse_safety.

    backend_name == None models 'auto' (attention_config.backend is None).
    """
    backend = _Backend(backend_name) if backend_name is not None else None
    return SimpleNamespace(
        attention_config=SimpleNamespace(backend=backend),
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
    )


def _validate(cfg):
    # Unbound call: _validate_sparse_safety reads only its config arg.
    EpicConnector._validate_sparse_safety(None, cfg)


# --------------------------------------------------------------------------
# Pass case
# --------------------------------------------------------------------------


def test_flex_attention_plus_eager_passes():
    _validate(_config("FLEX_ATTENTION", enforce_eager=True))  # no raise.


# --------------------------------------------------------------------------
# Backend failures
# --------------------------------------------------------------------------


def test_flash_attention_backend_rejected():
    with pytest.raises(ValueError) as exc:
        _validate(_config("FLASH_ATTN", enforce_eager=True))
    msg = str(exc.value)
    assert "VLLM_ATTENTION_BACKEND=FLEX_ATTENTION" in msg
    assert "FLASH_ATTN" in msg


def test_auto_backend_rejected():
    # backend None == auto selection -> not guaranteed FlexAttention -> reject.
    with pytest.raises(ValueError) as exc:
        _validate(_config(None, enforce_eager=True))
    assert "VLLM_ATTENTION_BACKEND=FLEX_ATTENTION" in str(exc.value)


# --------------------------------------------------------------------------
# Eager failure
# --------------------------------------------------------------------------


def test_non_eager_rejected():
    with pytest.raises(ValueError) as exc:
        _validate(_config("FLEX_ATTENTION", enforce_eager=False))
    msg = str(exc.value)
    assert "enforce_eager" in msg
    assert "PIECEWISE" in msg  # documents the relaxation TODO.


def test_backend_checked_before_eager():
    # Both wrong: the backend error (the more fundamental misconfig) wins.
    with pytest.raises(ValueError) as exc:
        _validate(_config("FLASH_ATTN", enforce_eager=False))
    assert "VLLM_ATTENTION_BACKEND=FLEX_ATTENTION" in str(exc.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
