# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the gpu_smoke subprocess-isolation machinery (VRAM fix).

gpu_smoke now runs EACH engine in a FRESH subprocess (`gpu_smoke.py
--_worker-json <spec>`) so that in-process engines (which do NOT free device
memory on `del llm`) cannot accumulate VRAM across the dense + 3 sparse runs of
step4. The engine build needs a GPU, but the wiring around it -- the JSON spec
serialise/parse round-trip, the RESULT_JSON line extractor (robust to leading
log noise), and the worker error-json path on a box with no usable engine -- is
pure plumbing and must be correct before any GPU time is spent. We verify it
here, on CPU.
"""

import os
import subprocess
import sys

import pytest

from tests.v1.kv_connector.unit.epic.gpu_smoke import (
    SmokeConfig,
    _RESULT_PREFIX,
    build_worker_spec,
    parse_result_json,
    parse_spec,
    serialize_spec,
)

_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "gpu_smoke.py")
)


# ---------------------------------------------------------------------------
# spec serialise / parse round-trip
# ---------------------------------------------------------------------------


def test_spec_roundtrip_preserves_all_fields():
    spec = build_worker_spec(
        role="sparse",
        model="some/model",
        kv_config={"kv_connector": "EpicConnector", "kv_role": "kv_both"},
        prompts=[[1, 2, 3], [4, 5]],
        warm_prompts=[[9, 8, 7]],
        max_tokens=48,
        enforce_eager=True,
        attention_backend="FLEX_ATTENTION",
        gpu_memory_utilization=0.45,
        max_model_len=2048,
        in_process=True,
        read_counters=True,
    )
    back = parse_spec(serialize_spec(spec))
    assert back == spec
    # spot-check the load-bearing fields survived exactly.
    assert back["prompts"] == [[1, 2, 3], [4, 5]]
    assert back["warm_prompts"] == [[9, 8, 7]]
    assert back["in_process"] is True
    assert back["read_counters"] is True
    assert back["attention_backend"] == "FLEX_ATTENTION"


def test_serialize_spec_is_single_line():
    spec = build_worker_spec(
        role="dense", model="m", kv_config=None, prompts=[[1]],
        max_tokens=4,
    )
    s = serialize_spec(spec)
    assert "\n" not in s  # argv-safe, single line


def test_build_worker_spec_defaults():
    spec = build_worker_spec(
        role="dense", model="m", kv_config=None, prompts=[[1, 2]],
        max_tokens=8,
    )
    assert spec["warm_prompts"] == []
    assert spec["in_process"] is False
    assert spec["read_counters"] is False
    assert spec["attention_backend"] is None
    assert spec["gpu_memory_utilization"] == 0.45
    assert spec["max_model_len"] == 2048


def test_build_worker_spec_copies_prompt_lists():
    # The spec must not alias caller-owned lists (a later parent-side mutation
    # would otherwise corrupt the serialised spec).
    p = [1, 2, 3]
    spec = build_worker_spec(
        role="dense", model="m", kv_config=None, prompts=[p], max_tokens=4,
    )
    p.append(999)
    assert spec["prompts"] == [[1, 2, 3]]


# ---------------------------------------------------------------------------
# RESULT_JSON extraction (robust to leading noise)
# ---------------------------------------------------------------------------


def test_parse_result_json_clean():
    out = f'{_RESULT_PREFIX} {{"ok": true, "token_ids": [[1,2]]}}'
    res = parse_result_json(out)
    assert res["ok"] is True
    assert res["token_ids"] == [[1, 2]]


def test_parse_result_json_ignores_leading_log_noise():
    out = "\n".join([
        "INFO some engine banner",
        "[epic] EPIC sparse match non_prefix_hits=1",
        "a line that merely MENTIONS RESULT_JSON in prose, not a result",
        f'{_RESULT_PREFIX} {{"ok": true, "role": "sparse"}}',
    ])
    res = parse_result_json(out)
    assert res["ok"] is True
    assert res["role"] == "sparse"


def test_parse_result_json_takes_last_when_multiple():
    out = "\n".join([
        f'{_RESULT_PREFIX} {{"ok": false, "stale": true}}',
        "more logs",
        f'{_RESULT_PREFIX} {{"ok": true, "final": true}}',
    ])
    res = parse_result_json(out)
    assert res["ok"] is True
    assert res.get("final") is True


def test_parse_result_json_tolerates_surrounding_whitespace():
    out = f'   {_RESULT_PREFIX}   {{"ok": true}}   '
    assert parse_result_json(out)["ok"] is True


def test_parse_result_json_missing_raises_with_tail():
    with pytest.raises(ValueError) as ei:
        parse_result_json("just\nlogs\nno result line here")
    assert "no RESULT_JSON" in str(ei.value)
    # the tail of stdout is included for debugging.
    assert "no result line here" in str(ei.value)


# ---------------------------------------------------------------------------
# SmokeConfig -> engine kwargs
# ---------------------------------------------------------------------------


def test_smoke_config_engine_kwargs():
    cfg = SmokeConfig(model="m", gpu_memory_utilization=0.6, max_model_len=4096)
    assert cfg.engine_kwargs() == {
        "gpu_memory_utilization": 0.6,
        "max_model_len": 4096,
    }


# ---------------------------------------------------------------------------
# end-to-end worker invocation (no GPU): clean error json, exit 0
# ---------------------------------------------------------------------------


def _run_worker_subprocess(spec: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, _SCRIPT, "--_worker-json", serialize_spec(spec)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_worker_mode_no_gpu_emits_error_json_exit_zero():
    # No usable engine on a CPU box (no GPU / gated model). The worker must NOT
    # crash with a non-zero traceback: it reports {"ok": false, "error": ...} in
    # RESULT_JSON and exits 0, so the parent gets a clean structured failure.
    spec = build_worker_spec(
        role="dense",
        model="meta-llama/Llama-3.2-1B-Instruct",
        kv_config=None,
        prompts=[[1, 2, 3]],
        max_tokens=4,
    )
    proc = _run_worker_subprocess(spec)
    assert proc.returncode == 0, (
        f"worker exited {proc.returncode}; stderr tail:\n"
        + "\n".join(proc.stderr.splitlines()[-10:])
    )
    res = parse_result_json(proc.stdout)
    assert res["ok"] is False
    assert "error" in res and res["error"]
    assert res["role"] == "dense"


def test_worker_mode_bad_spec_emits_error_json_exit_zero():
    # A malformed spec string must also fail closed (error json, exit 0).
    proc = subprocess.run(
        [sys.executable, _SCRIPT, "--_worker-json", "{not valid json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0
    res = parse_result_json(proc.stdout)
    assert res["ok"] is False
    assert "bad spec" in res["error"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
