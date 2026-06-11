# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC-reuse benchmark: plotting (matplotlib).

Reads the CSVs from bench_perf.py / bench_accuracy.py and renders PNGs:

  (a) TTFT vs |B|, one line per mode (|A|,|C| fixed via --fix-a/--fix-c).
  (b) speedup (epic_best/prefix, epic_best/full) vs |B|.
  (c) TTFT vs |A| at a fixed |C| (and chosen |B|).
  (d) accuracy (dense vs epic_best) vs |B|.
  (e) k-tradeoff: accuracy (and F1) + TTFT/speedup vs link-k at a fixed cell.
      This is the EPIC paper's core curve: reuse-only (k=0) is fastest/least
      accurate, accuracy recovers as k grows, dense (prefix) is the ceiling.

matplotlib is optional: if it is not installed, this prints install guidance and
exits 0 (so a CPU CI that imports the module is not penalized).

Usage:
  python plot_results.py --perf perf.csv --acc acc.csv --outdir plots/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict


def _have_mpl():
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _read_perf(path: str):
    rows = []
    with open(path) as f:
        for d in csv.DictReader(f):
            rows.append({
                "mode": d["mode"], "A": int(d["A"]), "C": int(d["C"]),
                "B": int(d["B"]), "link": int(d.get("link", -1)),
                "link_k": int(d.get("link_k", -1)),
                "ttft": float(d["ttft_mean_s"]),
                "tps": float(d["tokens_per_s"]),
            })
    return rows


def _is_epic_label(mode: str) -> bool:
    return mode.startswith("epic@") or mode == "reuse-only"


def _epic_best_ttft(perf_rows, a, c, b):
    """Lowest TTFT among epic specs at a cell (the headline epic point)."""
    cands = [r["ttft"] for r in perf_rows
             if _is_epic_label(r["mode"])
             and r["A"] == a and r["C"] == c and r["B"] == b]
    return min(cands) if cands else None


def _read_acc(path: str):
    rows = []
    with open(path) as f:
        for d in csv.DictReader(f):
            rows.append({
                "mode": d.get("mode", "epic"),
                "A": int(d["A"]), "C": int(d["C"]), "B": int(d["B"]),
                "link_k": int(d.get("link_k", d.get("link", -1))),
                "dense": float(d["dense_needle_acc"]),
                "epic": float(d["epic_needle_acc"]),
                "epic_f1": float(d.get("epic_f1", 0.0)),
                "dense_f1": float(d.get("dense_f1", 0.0)),
            })
    return rows


def plot_perf(perf_rows, outdir, fix_a, fix_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (a) TTFT vs B for the chosen (A,C) slice.
    by_mode = defaultdict(list)
    for r in perf_rows:
        if (fix_a is None or r["A"] == fix_a) and (fix_c is None or r["C"] == fix_c):
            by_mode[r["mode"]].append((r["B"], r["ttft"]))
    if any(by_mode.values()):
        plt.figure()
        for mode, pts in sorted(by_mode.items()):
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1] * 1e3 for p in pts]
            plt.plot(xs, ys, marker="o", label=mode)
        plt.xlabel("|B| (reused passage tokens)")
        plt.ylabel("TTFT (ms)")
        plt.title(f"TTFT vs |B|  (A={fix_a}, C={fix_c})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        p = os.path.join(outdir, "ttft_vs_B.png")
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[epic-plot] wrote {p}")

    # (b) speedup vs B (epic_best = lowest-TTFT epic spec at each cell).
    ttft = {(r["mode"], r["A"], r["C"], r["B"]): r["ttft"] for r in perf_rows}
    cells = sorted({(r["A"], r["C"], r["B"]) for r in perf_rows
                    if (fix_a is None or r["A"] == fix_a)
                    and (fix_c is None or r["C"] == fix_c)})
    sp_p, sp_f = [], []
    for (a, c, b) in cells:
        e = _epic_best_ttft(perf_rows, a, c, b)
        p_ = ttft.get(("prefix", a, c, b))
        fu = ttft.get(("full", a, c, b))
        if e and e > 0:
            if p_:
                sp_p.append((b, p_ / e))
            if fu:
                sp_f.append((b, fu / e))
    if sp_p or sp_f:
        plt.figure()
        if sp_p:
            sp_p.sort()
            plt.plot([x for x, _ in sp_p], [y for _, y in sp_p],
                     marker="o", label="epic/prefix")
        if sp_f:
            sp_f.sort()
            plt.plot([x for x, _ in sp_f], [y for _, y in sp_f],
                     marker="s", label="epic/full")
        plt.axhline(1.0, color="gray", ls="--", alpha=0.6)
        plt.xlabel("|B|")
        plt.ylabel("speedup (x)")
        plt.title(f"EPIC TTFT speedup vs |B|  (A={fix_a}, C={fix_c})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        p = os.path.join(outdir, "speedup_vs_B.png")
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[epic-plot] wrote {p}")

    # (c) TTFT vs A at fixed C (pick the largest B available).
    bs = sorted({r["B"] for r in perf_rows})
    if bs:
        bsel = bs[-1]
        by_mode = defaultdict(list)
        for r in perf_rows:
            if r["B"] == bsel and (fix_c is None or r["C"] == fix_c):
                by_mode[r["mode"]].append((r["A"], r["ttft"]))
        if any(by_mode.values()):
            plt.figure()
            for mode, pts in sorted(by_mode.items()):
                pts.sort()
                plt.plot([x for x, _ in pts], [y * 1e3 for _, y in pts],
                         marker="o", label=mode)
            plt.xlabel("|A| (shared prefix tokens)")
            plt.ylabel("TTFT (ms)")
            plt.title(f"TTFT vs |A|  (C={fix_c}, B={bsel})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            p = os.path.join(outdir, "ttft_vs_A.png")
            plt.savefig(p, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"[epic-plot] wrote {p}")


def plot_acc(acc_rows, outdir, fix_a, fix_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (d) accuracy vs |B|: dense (prefix) ceiling vs epic_best (max epic acc) at
    # each cell. Aggregate over the multiple epic-k rows.
    by_cell_dense: dict = {}
    by_cell_epic_best: dict = {}
    for r in acc_rows:
        if (fix_a is None or r["A"] == fix_a) and (fix_c is None or r["C"] == fix_c):
            cell = (r["A"], r["C"], r["B"])
            by_cell_dense[cell] = r["dense"]
            by_cell_epic_best[cell] = max(
                by_cell_epic_best.get(cell, -1.0), r["epic"])
    pts_d = sorted((b, v) for (_a, _c, b), v in by_cell_dense.items())
    pts_e = sorted((b, v) for (_a, _c, b), v in by_cell_epic_best.items())
    if pts_d or pts_e:
        plt.figure()
        if pts_d:
            plt.plot([x for x, _ in pts_d], [y for _, y in pts_d],
                     marker="o", label="dense (prefix)")
        if pts_e:
            plt.plot([x for x, _ in pts_e], [y for _, y in pts_e],
                     marker="s", label="epic (best k)")
        plt.xlabel("|B|")
        plt.ylabel("accuracy")
        plt.ylim(-0.02, 1.02)
        plt.title(f"Accuracy vs |B|  (A={fix_a}, C={fix_c})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        p = os.path.join(outdir, "accuracy_vs_B.png")
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[epic-plot] wrote {p}")


def plot_k_tradeoff(acc_rows, perf_rows, outdir, fix_a, fix_c, fix_b):
    """(e) The EPIC core curve: at one (A,C,B) cell, accuracy/F1 and TTFT (or
    speedup vs dense) as a function of link-k. x=k, y1=accuracy, y2=TTFT.

    reuse-only (k=0) anchors the low-accuracy/low-latency end; the dense (prefix)
    accuracy is drawn as a horizontal ceiling.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pick the target cell. If not fully specified, choose the cell with the most
    # distinct link-k accuracy points (the most informative sweep).
    epic_acc = [r for r in acc_rows if r["link_k"] >= 0]
    if not epic_acc:
        print("[epic-plot] no link-k accuracy rows; skipping k-tradeoff plot.")
        return

    def _cell_ok(r):
        return ((fix_a is None or r["A"] == fix_a)
                and (fix_c is None or r["C"] == fix_c)
                and (fix_b is None or r["B"] == fix_b))

    candidates = [r for r in epic_acc if _cell_ok(r)]
    if not candidates:
        candidates = epic_acc
    # Group by cell, choose the one with the most ks.
    from collections import defaultdict as _dd
    by_cell = _dd(list)
    for r in candidates:
        by_cell[(r["A"], r["C"], r["B"])].append(r)
    a, c, b = max(by_cell, key=lambda k: len({r["link_k"] for r in by_cell[k]}))
    rows = sorted(by_cell[(a, c, b)], key=lambda r: r["link_k"])
    ks = [r["link_k"] for r in rows]
    accs = [r["epic"] for r in rows]
    f1s = [r["epic_f1"] for r in rows]
    dense_acc = rows[0]["dense"] if rows else None

    # TTFT per k from perf rows (epic@k / reuse-only label).
    ttft_by_k: dict = {}
    for pr in perf_rows or []:
        if pr["A"] == a and pr["C"] == c and pr["B"] == b and pr["link_k"] >= 0:
            ttft_by_k.setdefault(pr["link_k"], pr["ttft"])
    ttfts = [ttft_by_k.get(k) for k in ks]
    prefix_ttft = None
    for pr in perf_rows or []:
        if pr["mode"] == "prefix" and pr["A"] == a and pr["C"] == c and pr["B"] == b:
            prefix_ttft = pr["ttft"]

    fig, ax1 = plt.subplots()
    ax1.plot(ks, accs, marker="o", color="tab:blue", label="epic accuracy")
    ax1.plot(ks, f1s, marker="^", color="tab:cyan", label="epic token-F1")
    if dense_acc is not None:
        ax1.axhline(dense_acc, color="tab:green", ls="--", alpha=0.7,
                    label="dense (prefix) ceiling")
    ax1.set_xlabel("link-k (recompute boundary tokens per B chunk)")
    ax1.set_ylabel("accuracy / F1")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    if any(t is not None for t in ttfts):
        ax2 = ax1.twinx()
        xs = [k for k, t in zip(ks, ttfts) if t is not None]
        ys = [t * 1e3 for t in ttfts if t is not None]
        ax2.plot(xs, ys, marker="s", color="tab:red", label="epic TTFT (ms)")
        if prefix_ttft is not None:
            ax2.axhline(prefix_ttft * 1e3, color="tab:orange", ls=":",
                        alpha=0.7, label="dense TTFT")
        ax2.set_ylabel("TTFT (ms)")
        lines, labels = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + l2, labels + lab2, loc="best", fontsize=8)
    else:
        ax1.legend(loc="best", fontsize=8)

    plt.title(f"Accuracy/TTFT vs link-k  (A={a}, C={c}, B={b})")
    p = os.path.join(outdir, "k_tradeoff.png")
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[epic-plot] wrote {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EPIC-reuse plots.")
    ap.add_argument("--perf", default=None, help="perf.csv from bench_perf.")
    ap.add_argument("--acc", default=None, help="acc.csv from bench_accuracy.")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--fix-a", type=int, default=None,
                    help="Hold |A| fixed for the B-sweep plots.")
    ap.add_argument("--fix-c", type=int, default=None,
                    help="Hold |C| fixed for the B-sweep plots.")
    ap.add_argument("--fix-b", type=int, default=None,
                    help="Hold |B| fixed for the k-tradeoff plot (plot e).")
    args = ap.parse_args(argv)

    if not _have_mpl():
        print("[epic-plot] matplotlib not installed. Install it into the bench "
              "venv to render plots:\n"
              "    <venv>/bin/pip install matplotlib\n"
              "Then re-run. (CSV results are already complete without plots.)")
        return 0

    os.makedirs(args.outdir, exist_ok=True)
    perf_rows = _read_perf(args.perf) if args.perf else None
    acc_rows = _read_acc(args.acc) if args.acc else None
    if perf_rows:
        plot_perf(perf_rows, args.outdir, args.fix_a, args.fix_c)
    if acc_rows:
        plot_acc(acc_rows, args.outdir, args.fix_a, args.fix_c)
        # (e) k-tradeoff needs the accuracy k-sweep; perf is optional (adds TTFT).
        plot_k_tradeoff(acc_rows, perf_rows, args.outdir,
                        args.fix_a, args.fix_c, args.fix_b)
    if not args.perf and not args.acc:
        print("[epic-plot] nothing to plot; pass --perf and/or --acc.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
