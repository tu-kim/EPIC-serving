# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turn-migration benchmark: fileKV retrieve vs worker-to-worker KV copy.

Scenario (the challenge this measures)
--------------------------------------
worker1 (GPU0) served turn 9 of a conversation. The frontend routes turn 10
to worker2 (GPU1). worker2 has none of the history KV. Three strategies:

  * ``w2w``     -- copy turn 9's ENTIRE history KV from worker1's paged cache
                   to worker2 (exact prefix reuse; nixl-style transfer).
                   Bytes are large, and worker1 is BUSY serving other requests
                   -- the copy contends for its HBM bandwidth both slowing the
                   copy AND slowing worker1's in-flight decodes.
  * ``filekv``  -- worker2 recomputes the history but pulls the FILE segments
                   (C, F, ... = fileKV chunks) from the CPU store / GPU staging
                   (EPIC non-contiguous reuse). M = non-file tokens + link
                   heads. Zero contact with worker1. H2D is hideable by
                   prefetch (the tool-call schema names the files a turn early).
  * ``full``    -- worker2 recomputes everything (floor/reference).

The crossover depends on: history length, file fraction of the history,
interconnect (NVLink p2p vs PCIe vs through-CPU), source busyness, and
worker2's prefill throughput. This bench measures the machine-specific
inputs and evaluates the model on a grid so the crossover is explicit.

Three layers
------------
1. **microbench** (``--run``, needs >= 2 CUDA devices): measures
     - D2D peer-copy bandwidth GPU0->GPU1, src idle vs src busy
       (busy = HBM-bound triad kernels looping on GPU0),
     - the triad's own throughput solo vs during the copy
       (== worker1's serving degradation, the user's concern),
     - pinned-host H2D bandwidth on GPU1 (the fileKV load path),
     - through-CPU staged copy (D2H on GPU0 + H2D on GPU1) for the
       no-p2p fallback.
2. **cost model** (``--plan-only``, CPU-safe, no torch import): combines
   measured (or assumed) bandwidths with a prefill-throughput number into
   per-strategy TTFT + worker1-interference cost, swept over history length
   x file fraction x src busyness.
3. **end-to-end recipe**: see benchmarks/epic_reuse/README.md (two engines on
   two GPUs; heavy, run after the microbench narrows the interesting grid).

Usage
-----
  # CPU: evaluate the model with assumed/loaded bandwidths
  python -m benchmarks.epic_reuse.bench_migration --plan-only

  # GPU (>=2 devices): measure, then evaluate with the measured numbers
  python -m benchmarks.epic_reuse.bench_migration --run -o migration.json
  python -m benchmarks.epic_reuse.bench_migration --plan-only \
      --measured migration.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------
# KV geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVGeometry:
    """Per-token KV footprint of the serving model.

    Defaults = Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128, fp16).
    """

    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype_bytes: int = 2

    @property
    def bytes_per_token(self) -> int:
        # K and V.
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes


# ---------------------------------------------------------------------------
# Measured (or assumed) machine inputs
# ---------------------------------------------------------------------------


@dataclass
class MachineInputs:
    """Bandwidths in GB/s, throughputs in tok/s. Filled by the microbench or
    left at conservative PCIe-class defaults for --plan-only exploration."""

    # GPU0 -> GPU1 direct copy, source idle / source busy.
    d2d_idle_gbps: float = 40.0
    d2d_busy_gbps: float = 25.0
    # Through-CPU fallback (D2H on src + H2D on dst, pipelined).
    staged_idle_gbps: float = 18.0
    staged_busy_gbps: float = 12.0
    # Pinned-host -> GPU1 (the fileKV CPU-store load path).
    h2d_gbps: float = 20.0
    # worker1 serving throughput multiplier while a copy is in flight
    # (1.0 = no impact; microbench measures triad_during/triad_solo).
    src_slowdown_during_copy: float = 0.85
    # worker2 dense prefill throughput (tok/s). Take from bench_perf on the
    # same model/GPU; the default is a placeholder for grid exploration.
    prefill_tokps: float = 12000.0
    # Fixed per-transfer overhead (handshake, block-table setup), seconds.
    transfer_overhead_s: float = 0.010
    p2p_available: bool = True


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    history_tokens: int  # full turn-9 history (prompt+responses so far)
    file_tokens: int  # tokens of the history covered by fileKV chunks
    new_tokens: int  # turn-10 new user tokens (always recomputed)
    src_busy: bool  # is worker1 actively serving during the copy
    prefetch_hit: bool  # were the fileKV chunks staged on GPU1 a turn early
    chunk_size: int = 256
    link_k: int = 8


@dataclass
class StrategyCost:
    strategy: str
    ttft_s: float  # time until worker2 can start decoding turn 10
    bytes_moved: int  # bytes read from / written to interconnects
    recompute_tokens: int  # tokens worker2 prefills
    src_interference_s: float  # seconds worker1 spends slowed down
    src_slowdown: float  # worker1 throughput multiplier during that window
    detail: dict = field(default_factory=dict)


def _prefill_time(tokens: int, m: MachineInputs) -> float:
    return tokens / m.prefill_tokps if tokens > 0 else 0.0


def cost_w2w(s: Scenario, g: KVGeometry, m: MachineInputs) -> StrategyCost:
    """Copy the whole history KV from worker1, then prefill only new tokens."""
    move = s.history_tokens * g.bytes_per_token
    if m.p2p_available:
        bw = m.d2d_busy_gbps if s.src_busy else m.d2d_idle_gbps
    else:
        bw = m.staged_busy_gbps if s.src_busy else m.staged_idle_gbps
    copy_s = move / (bw * 1e9) + m.transfer_overhead_s
    prefill_s = _prefill_time(s.new_tokens, m)
    return StrategyCost(
        strategy="w2w",
        # Copy must complete before turn-10 prefill can attend to history.
        ttft_s=copy_s + prefill_s,
        bytes_moved=move,
        recompute_tokens=s.new_tokens,
        src_interference_s=copy_s,
        src_slowdown=(m.src_slowdown_during_copy if s.src_busy else 1.0),
        detail={"copy_s": copy_s, "prefill_s": prefill_s, "bw_gbps": bw},
    )


def cost_filekv(s: Scenario, g: KVGeometry, m: MachineInputs) -> StrategyCost:
    """Recompute history minus file segments; load file chunks from fileKV.

    M = (history - file) + new + link heads per file chunk + last token.
    H2D of the file chunks is fully hidden on a prefetch hit (staged a turn
    early on a side stream); on a miss it overlaps the prefill layer-by-layer,
    so the exposed cost is max(prefill, h2d) rather than the sum -- we report
    the conservative serial remainder max(0, h2d - prefill) as exposed.
    """
    n_file_chunks = s.file_tokens // s.chunk_size
    link_tokens = n_file_chunks * s.link_k
    recompute = (s.history_tokens - s.file_tokens) + s.new_tokens + link_tokens + 1
    prefill_s = _prefill_time(recompute, m)
    move = s.file_tokens * g.bytes_per_token
    h2d_s = 0.0 if s.prefetch_hit else move / (m.h2d_gbps * 1e9)
    exposed_h2d = max(0.0, h2d_s - prefill_s)
    return StrategyCost(
        strategy="filekv",
        ttft_s=prefill_s + exposed_h2d,
        bytes_moved=0 if s.prefetch_hit else move,
        recompute_tokens=recompute,
        src_interference_s=0.0,  # never touches worker1
        src_slowdown=1.0,
        detail={
            "prefill_s": prefill_s,
            "h2d_s": h2d_s,
            "exposed_h2d_s": exposed_h2d,
            "link_tokens": link_tokens,
        },
    )


def cost_full(s: Scenario, g: KVGeometry, m: MachineInputs) -> StrategyCost:
    recompute = s.history_tokens + s.new_tokens
    return StrategyCost(
        strategy="full",
        ttft_s=_prefill_time(recompute, m),
        bytes_moved=0,
        recompute_tokens=recompute,
        src_interference_s=0.0,
        src_slowdown=1.0,
    )


def evaluate(s: Scenario, g: KVGeometry, m: MachineInputs) -> list[StrategyCost]:
    return [cost_w2w(s, g, m), cost_filekv(s, g, m), cost_full(s, g, m)]


def sweep(
    g: KVGeometry,
    m: MachineInputs,
    history_grid: list[int],
    file_frac_grid: list[float],
    new_tokens: int = 512,
    chunk_size: int = 256,
    link_k: int = 8,
) -> list[dict]:
    """Grid evaluation. file fraction is snapped down to whole chunks."""
    rows: list[dict] = []
    for hist in history_grid:
        for frac in file_frac_grid:
            file_tokens = int(hist * frac) // chunk_size * chunk_size
            for busy in (False, True):
                for hit in (False, True):
                    s = Scenario(
                        history_tokens=hist,
                        file_tokens=file_tokens,
                        new_tokens=new_tokens,
                        src_busy=busy,
                        prefetch_hit=hit,
                        chunk_size=chunk_size,
                        link_k=link_k,
                    )
                    costs = {c.strategy: c for c in evaluate(s, g, m)}
                    rows.append(
                        {
                            "history": hist,
                            "file_frac": frac,
                            "file_tokens": file_tokens,
                            "src_busy": busy,
                            "prefetch_hit": hit,
                            "ttft_w2w_ms": costs["w2w"].ttft_s * 1e3,
                            "ttft_filekv_ms": costs["filekv"].ttft_s * 1e3,
                            "ttft_full_ms": costs["full"].ttft_s * 1e3,
                            "w2w_bytes_mb": costs["w2w"].bytes_moved / 1e6,
                            "src_interference_ms": costs["w2w"].src_interference_s
                            * 1e3,
                            "src_slowdown": costs["w2w"].src_slowdown,
                            "winner": min(
                                ("w2w", "filekv", "full"),
                                key=lambda k: costs[k].ttft_s,
                            ),
                        }
                    )
    return rows


# ---------------------------------------------------------------------------
# GPU microbench (imports torch lazily; needs >= 2 CUDA devices)
# ---------------------------------------------------------------------------


def _measure(args: argparse.Namespace) -> MachineInputs:
    import torch

    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, (
        "microbench needs >= 2 CUDA devices (got "
        f"{torch.cuda.device_count() if torch.cuda.is_available() else 0})"
    )
    src, dst = torch.device("cuda:0"), torch.device("cuda:1")
    p2p = torch.cuda.can_device_access_peer(0, 1)
    mb = args.transfer_mb * 1024 * 1024
    n = mb // 2  # fp16 elements
    payload_src = torch.randn(n, dtype=torch.float16, device=src)
    payload_dst = torch.empty_like(payload_src, device=dst)
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)

    def timed(fn, iters: int = args.iters) -> float:
        """Median wall seconds per call (events on the participating devices)."""
        import time

        for _ in range(3):
            fn()
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(src)
            torch.cuda.synchronize(dst)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2]

    def gbps(sec: float) -> float:
        return mb / sec / 1e9

    # --- interference workload on src: HBM-bound triad on its own stream ---
    triad_a = torch.randn(args.triad_mb * 1024 * 1024 // 2, dtype=torch.float16, device=src)
    triad_b = torch.empty_like(triad_a)
    busy_stream = torch.cuda.Stream(device=src)

    def triad_once():
        triad_b.copy_(triad_a)
        triad_a.mul_(1.0)  # touch HBM again (read-modify-write)

    def busy_loop(n_iters: int) -> None:
        with torch.cuda.stream(busy_stream):
            for _ in range(n_iters):
                triad_once()

    # triad solo throughput (proxy for worker1 serving speed).
    t_triad_solo = timed(lambda: busy_loop(args.triad_iters))

    # --- D2D copy: idle source ---
    t_d2d_idle = timed(lambda: payload_dst.copy_(payload_src))

    # --- D2D copy while src is busy; also time the triad during the copy ---
    import time as _time

    def d2d_during_busy() -> tuple[float, float]:
        # launch a long busy loop, then copy repeatedly inside its window
        with torch.cuda.stream(busy_stream):
            for _ in range(args.triad_iters * args.iters * 2):
                triad_once()
        t0 = _time.perf_counter()
        for _ in range(args.iters):
            payload_dst.copy_(payload_src)
        torch.cuda.synchronize(dst)
        copy_s = (_time.perf_counter() - t0) / args.iters
        t1 = _time.perf_counter()
        torch.cuda.synchronize(src)  # drain the remaining triad work
        drain_s = _time.perf_counter() - t1
        return copy_s, drain_s

    copy_busy_s, _ = d2d_during_busy()
    torch.cuda.synchronize(src)

    # triad throughput while a continuous copy stream runs (worker1 slowdown).
    copy_stream = torch.cuda.Stream(device=src)

    def triad_during_copy() -> float:
        with torch.cuda.stream(copy_stream):
            for _ in range(args.iters * 4):
                payload_dst.copy_(payload_src)
        t = timed(lambda: busy_loop(args.triad_iters), iters=3)
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        return t

    t_triad_during = triad_during_copy()

    # --- through-CPU staged path (no-p2p fallback): D2H then H2D ---
    def staged():
        host.copy_(payload_src, non_blocking=True)
        torch.cuda.synchronize(src)
        payload_dst.copy_(host, non_blocking=True)

    t_staged_idle = timed(staged)

    # --- pinned H2D on dst (the fileKV load path) ---
    t_h2d = timed(lambda: payload_dst.copy_(host, non_blocking=True))

    # Staged-busy is approximated by scaling staged-idle with the same
    # busy/idle degradation the direct path showed (the D2H leg contends for
    # the same src HBM); a dedicated measurement can replace this later.
    busy_ratio = t_d2d_idle / copy_busy_s if copy_busy_s > 0 else 1.0
    m = MachineInputs(
        d2d_idle_gbps=gbps(t_d2d_idle),
        d2d_busy_gbps=gbps(copy_busy_s),
        staged_idle_gbps=gbps(t_staged_idle),
        staged_busy_gbps=gbps(t_staged_idle) * min(1.0, busy_ratio),
        h2d_gbps=gbps(t_h2d),
        src_slowdown_during_copy=min(1.0, t_triad_solo / t_triad_during),
        prefill_tokps=args.prefill_tokps,
        p2p_available=bool(p2p),
    )
    return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fmt_table(rows: list[dict]) -> str:
    if not rows:
        return "(empty)"
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(_cell(r[c])) for r in rows)) for c in cols}
    out = ["  ".join(c.ljust(widths[c]) for c in cols)]
    for r in rows:
        out.append("  ".join(_cell(r[c]).ljust(widths[c]) for c in cols))
    return "\n".join(out)


def _cell(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-only", action="store_true", help="cost model only (CPU)")
    ap.add_argument("--run", action="store_true", help="GPU microbench (>=2 devices)")
    ap.add_argument("--measured", type=str, default=None, help="JSON from a prior --run")
    ap.add_argument("-o", "--output", type=str, default=None)
    ap.add_argument("--transfer-mb", type=int, default=256)
    ap.add_argument("--triad-mb", type=int, default=512)
    ap.add_argument("--triad-iters", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--prefill-tokps", type=float, default=12000.0,
                    help="worker2 dense prefill throughput (take from bench_perf)")
    ap.add_argument("--history-grid", type=str, default="4096,16384,65536,131072")
    ap.add_argument("--file-frac-grid", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--new-tokens", type=int, default=512)
    ap.add_argument("--link", type=int, default=8)
    args = ap.parse_args()

    geom = KVGeometry()
    if args.run:
        m = _measure(args)
        blob = {"machine": asdict(m), "geometry": asdict(geom)}
        print(json.dumps(blob, indent=2))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(blob, f, indent=2)
        return

    if args.measured:
        with open(args.measured) as f:
            m = MachineInputs(**json.load(f)["machine"])
    else:
        m = MachineInputs(prefill_tokps=args.prefill_tokps)

    rows = sweep(
        geom,
        m,
        history_grid=[int(x) for x in args.history_grid.split(",")],
        file_frac_grid=[float(x) for x in args.file_frac_grid.split(",")],
        new_tokens=args.new_tokens,
        link_k=args.link,
    )
    print(f"# machine: {asdict(m)}")
    print(f"# kv bytes/token: {geom.bytes_per_token}")
    print(_fmt_table(rows))
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"machine": asdict(m), "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
