# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC-reuse benchmark: performance (TTFT + prefill throughput).

Compares three modes on identical data and identical fairness settings
(FLEX_ATTENTION + enforce_eager for all, unless --baseline-backend overrides the
baselines for an extra reference):

  full   : prefix caching OFF (+ per-request cache-bust) -- full-prefill floor.
  prefix : vanilla prefix caching -- reuses A only; recomputes C + B.
  epic   : EpicConnector sparse-forward + fusion mask -- reuses A (native) + B
           (content-hash + PIC), recomputes only M = C + link + last.

Per cell (|A|,|C|,|B|): warm the cache (epic: a warmup prompt seeding B into the
chunk store; prefix: an A-prefix prompt; full: nothing), then time R measured
requests at max_tokens=1 (TTFT == wall time to first token, the prefill cost).

IMPORTANT (process isolation): the attention backend and the connector are fixed
at engine-construction time, so a single process runs exactly ONE mode. Run this
script once per mode (the --modes arg defaults to all three but, when actually
constructing an LLM, only ONE mode may be active per process -- the harness
enforces this by re-exec'ing itself as a subprocess per mode unless
--single-mode is given). On a CPU box use --plan-only to print the full plan
with no GPU.

Usage (GPU):
  # run all three modes (auto subprocess-per-mode) on a prepared JSONL
  python bench_perf.py --data epic_bench.jsonl --model meta-llama/Llama-3.2-1B-Instruct \
      --out perf.csv

  # CPU dry check of the schedule
  python bench_perf.py --data epic_bench.jsonl --plan-only
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass


# EPIC: fork-safety -- force NVML-based CUDA probing and spawn for vLLM child
# processes, so any parent-side CUDA touch (user env, sitecustomize) cannot
# break the forked EngineCore ("Cannot re-initialize CUDA in forked
# subprocess"). Must be set before torch/vllm are imported in this process.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from benchmarks.epic_reuse.common import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LINK_TOKENS,
    BenchRequest,
    backend_for_mode,
    engine_kwargs_for_mode,
    expand_mode_specs,
    parse_int_list,
    parse_mode_spec,
    read_jsonl,
)


@dataclass
class CellResult:
    # ``mode`` here is the mode-spec LABEL (full | prefix | epic@k | reuse-only)
    # so distinct link-k epic runs are kept as separate rows. ``link_k`` is the
    # epic recompute-boundary tokens (-1 for non-epic baselines).
    mode: str
    a: int
    c: int
    b: int
    link: int  # legacy column = the data's request link (kept for back-compat)
    link_k: int
    ttft_mean: float
    ttft_std: float
    prefill_tokens: int
    tokens_per_s: float
    n_measured: int


def _group_by_cell(reqs: list[BenchRequest]) -> dict[tuple[int, int, int], list[BenchRequest]]:
    cells: dict[tuple[int, int, int], list[BenchRequest]] = {}
    for r in reqs:
        cells.setdefault((r.a_tokens, r.c_tokens, r.b_tokens), []).append(r)
    return cells


# ---------------------------------------------------------------------------
# Plan-only (CPU): print the warmup/measure schedule with no engine.
# ---------------------------------------------------------------------------
def plan_only(reqs: list[BenchRequest], specs, measure_reps: int) -> None:
    cells = _group_by_cell(reqs)
    print("=== EPIC perf bench PLAN (no engine constructed) ===")
    labels = [s.label for s in specs]
    print(f"modes={labels} cells={len(cells)} measure_reps={measure_reps}")
    for spec in specs:
        print(f"\n--- mode: {spec.label}  (family={spec.family}, "
              f"link_k={spec.csv_link()}) ---")
        backend = backend_for_mode(spec.family, None)
        print(f"  attention backend: {backend}  enforce_eager: True")
        for (a, c, b), cell_reqs in sorted(cells.items()):
            r0 = cell_reqs[0]
            if spec.is_epic:
                k = spec.link_k
                warm = f"warmup prompt ({len(r0.warmup_token_ids())} tok, B@offset "
                warm += f"{r0.warmup_b_offset_tokens}) -> saves {r0.predicted_b_chunk_hits} B chunk(s)"
                if k == 0:
                    forward = (f"M = C({c}) + last  (reuse-only: B PIC re-rotary, "
                               f"ZERO recompute) [link_k=0]")
                else:
                    forward = (f"M = C({c}) + link({k})xB_chunks + last  "
                               f"(reuses A native + B cached) [link_k={k}]")
            elif spec.family == "prefix":
                warm = f"A-prefix prompt ({a} tok) -> native prefix cache" if a else "no warmup (|A|=0)"
                forward = f"recompute C({c}) + B({b}) + Q; reuse A({a}) only"
            else:  # full
                warm = "no warmup (cache-bust: prefix caching OFF + perturb A)"
                forward = f"full prefill of A({a})+C({c})+B({b})+Q"
            n = len(cell_reqs)
            prompt_len = a + c + b + r0.q_tokens
            print(
                f"  cell A={a} C={c} B={b} (prompt~{prompt_len} tok, {n} reqs): "
                f"\n      warm  : {warm}"
                f"\n      fwd   : {forward}"
                f"\n      measure: {measure_reps} reps @ max_tokens=1 (TTFT)"
            )
    print("\n[plan-only] no GPU used. Drop --plan-only on a CUDA box to run.")


# ---------------------------------------------------------------------------
# Real run for a SINGLE mode (constructs one LLM).
# ---------------------------------------------------------------------------
def run_single_mode(
    family: str,
    label: str,
    reqs: list[BenchRequest],
    *,
    model: str,
    block_size: int,
    chunk_size: int,
    link_tokens: int,
    max_model_len: int,
    gpu_mem_util: float,
    measure_reps: int,
    baseline_backend: str | None,
) -> list[CellResult]:
    backend = backend_for_mode(family, baseline_backend)
    csv_link = link_tokens if family == "epic" else -1
    print(f"[epic-perf] mode={label} (family={family}, link_k={csv_link}) "
          f"backend={backend}")

    # Import vLLM lazily (only when actually running).
    from vllm import LLM, SamplingParams, TokensPrompt

    kwargs = engine_kwargs_for_mode(
        family,
        model=model,
        chunk_size=chunk_size,
        link_tokens=link_tokens,
        block_size=block_size,
        max_model_len=max_model_len,
        gpu_mem_util=gpu_mem_util,
        baseline_backend=baseline_backend,
    )
    llm = LLM(**kwargs)
    sp_ttft = SamplingParams(temperature=0.0, max_tokens=1)

    results: list[CellResult] = []
    cells = _group_by_cell(reqs)

    for (a, c, b), cell_reqs in sorted(cells.items()):
        link = cell_reqs[0].link_tokens
        # ---- warmup: populate the relevant cache ----
        _warm_cell(llm, family, cell_reqs, sp_ttft)

        # ---- measure: TTFT over reqs x reps ----
        ttfts: list[float] = []
        prefill_tokens = a + c + b + cell_reqs[0].q_tokens
        bust = 0
        for rep in range(measure_reps):
            for req in cell_reqs:
                if family == "full":
                    ids = req.busted_prompt_token_ids(bust)
                    bust += 1
                else:
                    ids = req.prompt_token_ids()
                prompt = TokensPrompt(prompt_token_ids=ids)
                t0 = time.perf_counter()
                out = llm.generate([prompt], sp_ttft, use_tqdm=False)
                t1 = time.perf_counter()
                ttfts.append(_ttft_seconds(out[0], t0, t1))

        mean = statistics.fmean(ttfts) if ttfts else 0.0
        std = statistics.pstdev(ttfts) if len(ttfts) > 1 else 0.0
        tps = (prefill_tokens / mean) if mean > 0 else 0.0
        results.append(
            CellResult(
                mode=label, a=a, c=c, b=b, link=link, link_k=csv_link,
                ttft_mean=mean, ttft_std=std,
                prefill_tokens=prefill_tokens, tokens_per_s=tps,
                n_measured=len(ttfts),
            )
        )
        print(
            f"[epic-perf] {label} A={a} C={c} B={b}: "
            f"TTFT {mean*1e3:.2f}+-{std*1e3:.2f} ms  ({tps:.0f} tok/s prefill)"
        )

    del llm
    return results


def _warm_cell(llm, family: str, cell_reqs: list, sp) -> None:
    from vllm import TokensPrompt

    if family == "epic":
        # Seed every DISTINCT B in the cell into the store. In synthetic mode all
        # requests share one B (one warmup); in hf mode each request has its own
        # gold-passage B, so we warm each distinct warmup prompt.
        seen: set[tuple] = set()
        for req in cell_reqs:
            ids = req.warmup_token_ids()
            key = tuple(req.b_ids)
            if key in seen:
                continue
            seen.add(key)
            llm.generate(
                [TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False
            )
    elif family == "prefix":
        # Prime the native prefix cache with A (+ a throwaway tail) so A's blocks
        # are resident. If |A|==0 there is nothing to prime.
        a_ids = cell_reqs[0].a_ids
        if a_ids:
            llm.generate(
                [TokensPrompt(prompt_token_ids=list(a_ids))], sp, use_tqdm=False
            )
    # full: intentionally no warmup.


def _ttft_seconds(out, t0: float, t1: float) -> float:
    """Prefer engine-reported first-token latency; fall back to wall time.

    With max_tokens=1 the wall time t1-t0 already equals prefill+first-token, so
    it is a valid TTFT. If RequestOutput.metrics exposes arrival/first_token we
    use that (excludes client-side overhead).
    """
    m = getattr(out, "metrics", None)
    if m is not None:
        ft = getattr(m, "first_token_time", None)
        arr = getattr(m, "arrival_time", None)
        if ft is not None and arr is not None:
            try:
                d = float(ft) - float(arr)
                if d > 0:
                    return d
            except Exception:  # noqa: BLE001
                pass
    return t1 - t0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(path: str, rows: list[CellResult], append: bool) -> None:
    exists = append and os.path.exists(path)
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(
                ["mode", "A", "C", "B", "link", "link_k", "ttft_mean_s",
                 "ttft_std_s", "prefill_tokens", "tokens_per_s", "n_measured"]
            )
        for r in rows:
            w.writerow([r.mode, r.a, r.c, r.b, r.link, r.link_k,
                        f"{r.ttft_mean:.6f}", f"{r.ttft_std:.6f}",
                        r.prefill_tokens, f"{r.tokens_per_s:.2f}", r.n_measured])


def _load_perf_rows(csv_path: str):
    rows = []
    try:
        with open(csv_path) as f:
            for d in csv.DictReader(f):
                rows.append({
                    "mode": d["mode"], "A": int(d["A"]), "C": int(d["C"]),
                    "B": int(d["B"]),
                    "link_k": int(d.get("link_k", -1)),
                    "ttft": float(d["ttft_mean_s"]),
                })
    except FileNotFoundError:
        pass
    return rows


def print_speedup_table(csv_path: str) -> None:
    """Print epic/prefix and epic/full speedups by |B|.

    ``epic`` here = the BEST epic variant (lowest TTFT) at each cell so the
    headline speedup table is well-defined even with multiple link-k runs.
    """
    rows = _load_perf_rows(csv_path)
    if not rows:
        return
    ttft = {(r["mode"], r["A"], r["C"], r["B"]): r["ttft"] for r in rows}
    cells = sorted({(r["A"], r["C"], r["B"]) for r in rows})
    epic_labels = sorted({r["mode"] for r in rows
                          if r["mode"].startswith("epic@")
                          or r["mode"] == "reuse-only"})
    print("\n=== Speedup (TTFT baseline/epic_best; >1 means EPIC faster) ===")
    print(f"{'A':>6} {'C':>6} {'B':>6} {'epic_best':>10} "
          f"{'epic/prefix':>12} {'epic/full':>10}")
    for (a, c, b) in cells:
        cands = [(lbl, ttft[(lbl, a, c, b)]) for lbl in epic_labels
                 if (lbl, a, c, b) in ttft]
        if cands:
            best_lbl, e = min(cands, key=lambda x: x[1])
        else:
            best_lbl, e = "-", None
        p = ttft.get(("prefix", a, c, b))
        fu = ttft.get(("full", a, c, b))
        sp_p = f"{p / e:.2f}x" if (e and p and e > 0) else "-"
        sp_f = f"{fu / e:.2f}x" if (e and fu and e > 0) else "-"
        print(f"{a:>6} {c:>6} {b:>6} {best_lbl:>10} {sp_p:>12} {sp_f:>10}")


def print_link_k_table(csv_path: str) -> None:
    """k-axis table: for each (A,C,B) cell, TTFT (and speedup vs prefix) as a
    function of the epic link-k. Shows the recompute<->latency tradeoff.
    """
    rows = _load_perf_rows(csv_path)
    if not rows:
        return
    ttft = {(r["mode"], r["A"], r["C"], r["B"]): r["ttft"] for r in rows}
    # link_k present in the data, sorted; reuse-only mapped to k=0 column too.
    ks = sorted({r["link_k"] for r in rows if r["link_k"] >= 0})
    if not ks:
        return
    cells = sorted({(r["A"], r["C"], r["B"]) for r in rows})
    # Build label per k: reuse-only is the canonical k=0 label if present.
    have_reuse = any(r["mode"] == "reuse-only" for r in rows)

    def lbl_for_k(k: int) -> str:
        if k == 0 and have_reuse:
            return "reuse-only"
        return f"epic@{k}"

    print("\n=== TTFT (ms) vs link-k  (per |A|,|C|,|B| cell) ===")
    header = f"{'A':>5} {'C':>5} {'B':>6}" + "".join(
        f"{('k=' + str(k)):>10}" for k in ks) + f"{'prefix':>10}"
    print(header)
    for (a, c, b) in cells:
        line = f"{a:>5} {c:>5} {b:>6}"
        for k in ks:
            v = ttft.get((lbl_for_k(k), a, c, b))
            line += f"{(f'{v*1e3:.1f}' if v else '-'):>10}"
        p = ttft.get(("prefix", a, c, b))
        line += f"{(f'{p*1e3:.1f}' if p else '-'):>10}"
        print(line)


# ---------------------------------------------------------------------------
# CLI / subprocess-per-mode orchestration
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EPIC-reuse perf benchmark.")
    ap.add_argument("--data", required=True, help="JSONL from data_prep.py.")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--out", default="perf.csv")
    ap.add_argument("--modes", default="full,prefix,epic",
                    help="Comma list of mode specs: full,prefix,epic,epic@<k>,"
                    "reuse-only. A bare 'epic' uses --link (or expands across "
                    "--link-sweep if given).")
    ap.add_argument("--link-sweep", default=None,
                    help="Comma list of link-k values (e.g. 0,4,8,32). Expands "
                    "any bare 'epic' mode into one epic@k run per k (separate "
                    "engine each, since the connector fixes k at construction).")
    ap.add_argument("--measure-reps", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    ap.add_argument("--link", type=int, default=DEFAULT_LINK_TOKENS,
                    help="Default epic link-k for a bare 'epic' mode (when no "
                    "--link-sweep). Ignored for explicit epic@k / reuse-only.")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    ap.add_argument("--baseline-backend", default=None,
                    help="Backend for full/prefix baselines (e.g. FLASH_ATTN). "
                    "Default: FLEX_ATTENTION (fair).")
    ap.add_argument("--plan-only", action="store_true",
                    help="Print the schedule and exit (CPU, no engine).")
    # Internal subprocess hooks (one engine == one mode spec).
    ap.add_argument("--single-family", default=None,
                    help="Internal: engine family (full|prefix|epic).")
    ap.add_argument("--single-label", default=None,
                    help="Internal: mode-spec label for CSV/table rows.")
    args = ap.parse_args(argv)

    link_sweep = parse_int_list(args.link_sweep) if args.link_sweep else None
    try:
        specs = expand_mode_specs(
            args.modes, link_sweep=link_sweep, default_link=args.link)
    except ValueError as e:  # noqa: BLE001
        print(f"[epic-perf] {e}")
        return 2
    if not specs:
        print("[epic-perf] no modes to run.")
        return 2

    reqs = read_jsonl(args.data)
    if not reqs:
        print(f"[epic-perf] no requests in {args.data}")
        return 2

    if args.plan_only:
        plan_only(reqs, specs, args.measure_reps)
        return 0

    # ---- single mode-spec (in-process) ----
    if args.single_family:
        link_k = args.link if args.single_family == "epic" else -1
        # The subprocess receives its exact k via --link.
        rows = run_single_mode(
            args.single_family, args.single_label or args.single_family, reqs,
            model=args.model, block_size=args.block_size,
            chunk_size=args.chunk_size, link_tokens=args.link,
            max_model_len=args.max_model_len, gpu_mem_util=args.gpu_mem_util,
            measure_reps=args.measure_reps, baseline_backend=args.baseline_backend,
        )
        write_csv(args.out, rows, append=True)
        return 0

    # ---- orchestrator: subprocess per mode spec (backend/connector and link-k
    # are fixed at engine construction, so each spec MUST be a fresh process) ----
    if os.path.exists(args.out):
        os.remove(args.out)
    for spec in specs:
        print(f"\n[epic-perf] === launching subprocess for mode={spec.label} "
              f"(family={spec.family}, link_k={spec.csv_link()}) ===")
        link_arg = spec.link_k if spec.link_k is not None else args.link
        cmd = [
            sys.executable, "-m", "benchmarks.epic_reuse.bench_perf",
            "--data", args.data, "--model", args.model, "--out", args.out,
            "--single-family", spec.family,
            "--single-label", spec.label,
            "--measure-reps", str(args.measure_reps),
            "--chunk-size", str(args.chunk_size),
            "--block-size", str(args.block_size),
            "--link", str(link_arg),
            "--max-model-len", str(args.max_model_len),
            "--gpu-mem-util", str(args.gpu_mem_util),
        ]
        if args.baseline_backend:
            cmd += ["--baseline-backend", args.baseline_backend]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[epic-perf] mode {spec.label} subprocess failed rc={rc}")
            return rc

    print_speedup_table(args.out)
    print_link_k_table(args.out)
    print(f"\n[epic-perf] done -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
