# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost-model tests for benchmarks/epic_reuse/bench_migration.py.

CPU-only: exercises the accounting (byte math, monotonicity, crossovers),
not the GPU microbench. The model's *predictions* are validated on real
hardware by --run; these tests pin the arithmetic so a refactor can't
silently change what the bench claims to measure.
"""

import pytest

from benchmarks.epic_reuse.bench_migration import (
    STRATEGIES,
    KVGeometry,
    MachineInputs,
    Scenario,
    cost_filekv,
    cost_full,
    cost_hostkv,
    cost_w2w,
    evaluate,
    sweep,
)

GEOM = KVGeometry()  # Llama-3.1-8B defaults


def _scenario(**kw) -> Scenario:
    base = dict(
        history_tokens=16384,
        file_tokens=8192,
        new_tokens=512,
        src_busy=False,
        prefetch_hit=False,
    )
    base.update(kw)
    return Scenario(**base)


def test_kv_bytes_per_token_llama8b():
    # 2 (K+V) * 32 layers * 8 kv-heads * 128 head_dim * 2 bytes = 128 KiB.
    assert GEOM.bytes_per_token == 131072


def test_w2w_moves_whole_history_and_recomputes_only_new():
    s = _scenario()
    c = cost_w2w(s, GEOM, MachineInputs())
    assert c.bytes_moved == s.history_tokens * GEOM.bytes_per_token
    assert c.recompute_tokens == s.new_tokens
    # The copy window IS the interference window on worker1.
    assert c.src_interference_s == pytest.approx(c.detail["copy_s"])


def test_w2w_busy_source_is_strictly_slower():
    m = MachineInputs()
    idle = cost_w2w(_scenario(src_busy=False), GEOM, m)
    busy = cost_w2w(_scenario(src_busy=True), GEOM, m)
    assert busy.ttft_s > idle.ttft_s
    assert busy.src_slowdown < 1.0 and idle.src_slowdown == 1.0


def test_filekv_never_touches_worker1():
    for busy in (False, True):
        c = cost_filekv(_scenario(src_busy=busy), GEOM, MachineInputs())
        assert c.src_interference_s == 0.0
        assert c.src_slowdown == 1.0


def test_filekv_recompute_accounting():
    s = _scenario(history_tokens=4096, file_tokens=2048, new_tokens=256)
    c = cost_filekv(s, GEOM, MachineInputs())
    n_chunks = 2048 // s.chunk_size
    expected = (4096 - 2048) + 256 + n_chunks * s.link_k + 1
    assert c.recompute_tokens == expected


def test_filekv_prefetch_hit_moves_zero_bytes_and_is_not_slower():
    m = MachineInputs()
    miss = cost_filekv(_scenario(prefetch_hit=False), GEOM, m)
    hit = cost_filekv(_scenario(prefetch_hit=True), GEOM, m)
    assert hit.bytes_moved == 0
    assert miss.bytes_moved > 0
    assert hit.ttft_s <= miss.ttft_s


def test_filekv_miss_h2d_overlaps_prefill():
    # Exposed H2D is only the remainder beyond the prefill window.
    m = MachineInputs()
    c = cost_filekv(_scenario(prefetch_hit=False), GEOM, m)
    assert c.ttft_s == pytest.approx(
        c.detail["prefill_s"] + max(0.0, c.detail["h2d_s"] - c.detail["prefill_s"])
    )


def test_hostkv_exact_reuse_zero_interference_full_bytes():
    m = MachineInputs()
    for busy in (False, True):
        c = cost_hostkv(_scenario(src_busy=busy), GEOM, m)
        # Exact prefix reuse: only new tokens recomputed, like w2w.
        assert c.recompute_tokens == 512
        # But zero worker1 contact (D2H was paid at evict time)...
        assert c.src_interference_s == 0.0 and c.src_slowdown == 1.0
        # ...and the whole history moves, unaffected by src_busy.
        assert c.bytes_moved == 16384 * GEOM.bytes_per_token
    idle = cost_hostkv(_scenario(src_busy=False), GEOM, m)
    busy_c = cost_hostkv(_scenario(src_busy=True), GEOM, m)
    assert idle.ttft_s == pytest.approx(busy_c.ttft_s)


def test_hostkv_vs_w2w_flips_on_busy_source():
    """hostkv trades a slower pipe (host_gbps < d2d) for busy-immunity: with
    the source busy enough, hostkv overtakes w2w at identical exactness."""
    m = MachineInputs(d2d_idle_gbps=40.0, d2d_busy_gbps=8.0, host_gbps=14.0)
    s_idle = _scenario(src_busy=False)
    s_busy = _scenario(src_busy=True)
    assert cost_w2w(s_idle, GEOM, m).ttft_s < cost_hostkv(s_idle, GEOM, m).ttft_s
    assert cost_hostkv(s_busy, GEOM, m).ttft_s < cost_w2w(s_busy, GEOM, m).ttft_s


def test_hostkv_vs_filekv_crossover():
    """filekv beats hostkv when the history is nearly all file content and
    staged (recompute ~= links only, zero bytes); hostkv wins at low file
    fraction (its pipe beats recomputing most of the history)."""
    m = MachineInputs()
    hi_file = _scenario(
        history_tokens=131072, file_tokens=129024, new_tokens=256,
        prefetch_hit=True,
    )
    lo_file = _scenario(history_tokens=131072, file_tokens=16384, new_tokens=256)
    assert cost_filekv(hi_file, GEOM, m).ttft_s < cost_hostkv(hi_file, GEOM, m).ttft_s
    assert cost_hostkv(lo_file, GEOM, m).ttft_s < cost_filekv(lo_file, GEOM, m).ttft_s


def test_full_is_upper_bound_on_recompute():
    s = _scenario()
    m = MachineInputs()
    full = cost_full(s, GEOM, m)
    fkv = cost_filekv(s, GEOM, m)
    assert full.recompute_tokens >= fkv.recompute_tokens
    assert full.ttft_s >= fkv.ttft_s


def test_crossover_exists_in_both_directions():
    """The comparison is genuinely scenario-dependent: each strategy must win
    somewhere in the space, otherwise the bench answers nothing."""
    m = MachineInputs()  # PCIe-class defaults
    # Fast interconnect + moderate file fraction -> w2w wins.
    s_w2w = _scenario(history_tokens=131072, file_tokens=65536)
    costs = {c.strategy: c.ttft_s for c in evaluate(s_w2w, GEOM, m)}
    assert min(costs, key=costs.get) == "w2w"
    # Slow interconnect (cross-node / contended, no p2p) + history that is
    # almost entirely file content + prefetch hit -> filekv wins.
    m_slow = MachineInputs(p2p_available=False, staged_idle_gbps=3.0,
                           staged_busy_gbps=1.5, prefill_tokps=20000.0)
    s_fkv = _scenario(
        history_tokens=131072, file_tokens=129024, new_tokens=256,
        src_busy=True, prefetch_hit=True,
    )
    costs = {c.strategy: c.ttft_s for c in evaluate(s_fkv, GEOM, m_slow)}
    assert min(costs, key=costs.get) == "filekv"


def test_w2w_degrades_with_history_but_filekv_with_nonfile_tokens():
    m = MachineInputs()
    # Double the history at fixed file fraction: both grow.
    a = evaluate(_scenario(history_tokens=8192, file_tokens=4096), GEOM, m)
    b = evaluate(_scenario(history_tokens=16384, file_tokens=8192), GEOM, m)
    for x, y in zip(a, b):
        assert y.ttft_s > x.ttft_s
    # Grow ONLY the file part: w2w pays for it (more bytes), filekv nearly
    # free on a prefetch hit (only link heads grow).
    lo = _scenario(history_tokens=16384, file_tokens=4096, prefetch_hit=True)
    hi = _scenario(history_tokens=16384, file_tokens=12288, prefetch_hit=True)
    assert cost_w2w(hi, GEOM, m).bytes_moved == cost_w2w(lo, GEOM, m).bytes_moved
    assert cost_filekv(hi, GEOM, m).ttft_s < cost_filekv(lo, GEOM, m).ttft_s


def test_sweep_snaps_file_tokens_to_chunks_and_labels_winner():
    rows = sweep(GEOM, MachineInputs(), history_grid=[4096], file_frac_grid=[0.3])
    assert rows
    for r in rows:
        assert r["file_tokens"] % 256 == 0
        assert r["winner"] in STRATEGIES
        assert "ttft_hostkv_ms" in r
        # full recompute can never strictly beat filekv in this model.
        assert r["ttft_filekv_ms"] <= r["ttft_full_ms"] + 1e-9
