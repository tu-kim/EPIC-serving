# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC-reuse benchmark: accuracy (dense/prefix vs epic).

Greedy-decodes the same A+C+B+Q prompts under two modes and compares quality:

  * dense reference  : the ``prefix`` mode (vanilla prefix caching, full B
    recompute) -- the numerically-correct answer the model would give.
  * epic             : EpicConnector sparse-forward + fusion mask (B reused via
    content-hash + PIC; only M recomputed) -- an APPROXIMATION.

Metrics per (|A|,|C|,|B|,link):
  * needle accuracy : fraction of requests whose generated text contains the
    embedded needle answer (the ground-truth code). Computed per mode.
  * exact-match rate : fraction of requests whose epic output == dense output
    (token ids).
  * token-prefix match : mean length of the matching leading token run
    (epic vs dense), normalized by dense length.

The needle answer is the primary signal: EPIC should preserve it even though
its output may differ token-for-token from dense (approximation). A large needle
accuracy GAP (dense high, epic low) at small |C| / large |B| flags an over-
aggressive reuse / mask / PIC bug.

Process isolation mirrors bench_perf: one mode per process; the orchestrator
re-execs itself per mode. ``--plan-only`` prints the comparison plan on CPU.

Usage (GPU):
  python bench_accuracy.py --data epic_bench.jsonl \
      --model meta-llama/Llama-3.2-1B-Instruct --out acc.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
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
    answer_containment,
    backend_for_mode,
    engine_kwargs_for_mode,
    expand_mode_specs,
    parse_int_list,
    read_jsonl,
    token_f1,
)


def _score_request(req: BenchRequest, text: str) -> tuple[bool, float]:
    """Return (answer_hit, token_f1) for a generated ``text`` vs a request's
    ground truth. needle tasks: exact answer substring (back-compat) AND the
    normalized containment; hf tasks: SQuAD-style containment + token-F1.
    """
    golds = req.gold_answers or ([req.needle_answer] if req.needle_answer else [])
    if req.task_type == "hf":
        hit = answer_containment(text, golds)
        f1 = token_f1(text, golds)
        return hit, f1
    # needle: the answer is a unique code; raw substring is the strict signal,
    # but also accept normalized containment so trailing punctuation does not
    # cost a hit. F1 is reported too (degenerate but consistent).
    raw = req.needle_answer in text if req.needle_answer else False
    hit = raw or answer_containment(text, golds)
    f1 = token_f1(text, golds)
    return hit, f1


@dataclass
class CellAccuracy:
    # ``mode`` = mode-spec label (prefix | epic@k | reuse-only).
    mode: str
    a: int
    c: int
    b: int
    link: int
    link_k: int
    needle_acc: float
    f1_mean: float
    n: int


def _group_by_cell(reqs):
    cells = {}
    for r in reqs:
        cells.setdefault((r.a_tokens, r.c_tokens, r.b_tokens), []).append(r)
    return cells


def plan_only(reqs: list[BenchRequest], specs, measure_max_tokens: int) -> None:
    cells = _group_by_cell(reqs)
    labels = [s.label for s in specs]
    print("=== EPIC accuracy bench PLAN (no engine constructed) ===")
    print(f"modes={labels} cells={len(cells)} max_tokens={measure_max_tokens}")
    task = reqs[0].task_type if reqs else "needle"
    scorer = ("answer-containment + SQuAD token-F1" if task == "hf"
              else "needle-answer substring")
    print(f"  task_type={task}  scorer={scorer}")
    for (a, c, b), cell_reqs in sorted(cells.items()):
        r0 = cell_reqs[0]
        examples = [(r.gold_answers[:1] or [r.needle_answer]) for r in cell_reqs[:2]]
        print(
            f"  cell A={a} C={c} B={b} ({len(cell_reqs)} reqs, link={r0.link_tokens}):"
            f"\n      prompt = A({a}) + C({c}) + B({b}) + Q({r0.q_tokens})"
            f"\n      gold examples: {examples}"
            f"\n      compare: dense(prefix) greedy vs each epic/reuse-only spec"
            f" -> {scorer}, exact-match, token-prefix"
        )
    print("\n[plan-only] no GPU used. Drop --plan-only on a CUDA box to run.")


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
    max_tokens: int,
    out_jsonl: str,
) -> list[CellAccuracy]:
    """Generate per-request outputs for one mode spec; dump them to a side JSONL
    so the orchestrator can do the cross-mode (exact-match / prefix) comparison.
    """
    backend = backend_for_mode(family, None)
    csv_link = link_tokens if family == "epic" else -1
    print(f"[epic-acc] mode={label} (family={family}, link_k={csv_link}) "
          f"backend={backend}")

    from vllm import LLM, SamplingParams, TokensPrompt

    kwargs = engine_kwargs_for_mode(
        family, model=model, chunk_size=chunk_size, link_tokens=link_tokens,
        block_size=block_size, max_model_len=max_model_len,
        gpu_mem_util=gpu_mem_util,
    )
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    results: list[CellAccuracy] = []
    cells = _group_by_cell(reqs)
    fout = open(out_jsonl, "w")

    for (a, c, b), cell_reqs in sorted(cells.items()):
        link = cell_reqs[0].link_tokens
        # Warm so reuse actually triggers (epic: seed every distinct B; prefix:
        # prime A). In hf mode each request has its own gold-passage B.
        if family == "epic":
            seen: set[tuple] = set()
            for req in cell_reqs:
                key = tuple(req.b_ids)
                if key in seen:
                    continue
                seen.add(key)
                llm.generate(
                    [TokensPrompt(prompt_token_ids=req.warmup_token_ids())],
                    sp, use_tqdm=False,
                )
        elif family == "prefix" and cell_reqs[0].a_ids:
            llm.generate(
                [TokensPrompt(prompt_token_ids=list(cell_reqs[0].a_ids))],
                sp, use_tqdm=False,
            )

        hits = 0
        f1_sum = 0.0
        for req in cell_reqs:
            prompt = TokensPrompt(prompt_token_ids=req.prompt_token_ids())
            out = llm.generate([prompt], sp, use_tqdm=False)[0]
            text = out.outputs[0].text
            token_ids = list(out.outputs[0].token_ids)
            answer_in, f1 = _score_request(req, text)
            hits += int(answer_in)
            f1_sum += f1
            fout.write(json.dumps({
                "mode": label, "req_id": req.req_id,
                "a": a, "c": c, "b": b, "link": link, "link_k": csv_link,
                "task_type": req.task_type,
                "gold_answers": req.gold_answers,
                "needle_answer": req.needle_answer,
                "answer_in_output": answer_in,
                "token_f1": f1,
                "out_text": text,
                "out_token_ids": token_ids,
            }) + "\n")
        n = len(cell_reqs)
        acc = hits / n if n else 0.0
        f1m = f1_sum / n if n else 0.0
        results.append(CellAccuracy(mode=label, a=a, c=c, b=b, link=link,
                                    link_k=csv_link, needle_acc=acc,
                                    f1_mean=f1m, n=n))
        print(f"[epic-acc] {label} A={a} C={c} B={b}: acc {acc:.3f} "
              f"f1 {f1m:.3f} ({hits}/{n})")

    fout.close()
    del llm
    return results


# ---------------------------------------------------------------------------
# Cross-mode comparison (orchestrator side, CPU)
# ---------------------------------------------------------------------------
def _match_rate(a: list[int], b: list[int]) -> float:
    if not a and not b:
        return 1.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / max(len(a), len(b))


def _prefix_match(a: list[int], b: list[int]) -> float:
    n = min(len(a), len(b))
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    denom = max(len(a), 1)
    return k / denom


def _load_jsonl_by_rid(path: str) -> dict:
    d = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            d[r["req_id"]] = r
    return d


def compare(dense_jsonl: str, epic_jsonls: list[str], out_csv: str) -> None:
    """Compare each epic mode-spec dump against the dense (prefix) reference.

    ``epic_jsonls`` is one path per epic spec (epic@k / reuse-only). The CSV has
    one row per (label, A, C, B, link_k) cell so the k-sweep is fully captured.
    """
    dense = _load_jsonl_by_rid(dense_jsonl)

    # cells keyed by (label, A, C, B, link_k).
    cells: dict[tuple, dict] = {}
    for ejpath in epic_jsonls:
        epic = _load_jsonl_by_rid(ejpath)
        for rid, e in epic.items():
            d = dense.get(rid)
            if d is None:
                continue
            key = (e["mode"], e["a"], e["c"], e["b"], e.get("link_k", -1))
            agg = cells.setdefault(key, {
                "n": 0, "exact": 0, "prefix_sum": 0.0,
                "dense_acc": 0, "epic_acc": 0,
                "dense_f1": 0.0, "epic_f1": 0.0,
            })
            agg["n"] += 1
            agg["exact"] += int(e["out_token_ids"] == d["out_token_ids"])
            agg["prefix_sum"] += _prefix_match(
                e["out_token_ids"], d["out_token_ids"])
            agg["dense_acc"] += int(d["answer_in_output"])
            agg["epic_acc"] += int(e["answer_in_output"])
            agg["dense_f1"] += float(d.get("token_f1", 0.0))
            agg["epic_f1"] += float(e.get("token_f1", 0.0))

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "A", "C", "B", "link_k", "n",
                    "dense_needle_acc", "epic_needle_acc", "acc_gap",
                    "dense_f1", "epic_f1", "f1_gap",
                    "exact_match_rate", "token_prefix_match"])
        for key in sorted(cells):
            label, a, c, b, link_k = key
            agg = cells[key]
            n = agg["n"]
            d_acc = agg["dense_acc"] / n
            e_acc = agg["epic_acc"] / n
            d_f1 = agg["dense_f1"] / n
            e_f1 = agg["epic_f1"] / n
            w.writerow([label, a, c, b, link_k, n,
                        f"{d_acc:.4f}", f"{e_acc:.4f}", f"{d_acc - e_acc:.4f}",
                        f"{d_f1:.4f}", f"{e_f1:.4f}", f"{d_f1 - e_f1:.4f}",
                        f"{agg['exact'] / n:.4f}",
                        f"{agg['prefix_sum'] / n:.4f}"])

    print("\n=== Accuracy (dense=prefix vs epic specs) ===")
    print(f"{'mode':>11} {'A':>5} {'C':>5} {'B':>6} {'k':>4} "
          f"{'dense':>6} {'epic':>6} {'gap':>6} {'eF1':>6} {'exact':>6} "
          f"{'tokpfx':>6}")
    for key in sorted(cells):
        label, a, c, b, link_k = key
        agg = cells[key]
        n = agg["n"]
        d_acc = agg["dense_acc"] / n
        e_acc = agg["epic_acc"] / n
        e_f1 = agg["epic_f1"] / n
        print(f"{label:>11} {a:>5} {c:>5} {b:>6} {link_k:>4} "
              f"{d_acc:>6.3f} {e_acc:>6.3f} {d_acc - e_acc:>6.3f} "
              f"{e_f1:>6.3f} {agg['exact'] / n:>6.3f} "
              f"{agg['prefix_sum'] / n:>6.3f}")

    _print_k_accuracy_table(cells)


def _print_k_accuracy_table(cells: dict) -> None:
    """k-axis: epic accuracy (and F1) as a function of link-k per (A,C,B) cell."""
    ks = sorted({k for (_lbl, _a, _c, _b, k) in cells if k >= 0})
    if not ks:
        return
    # Map (a,c,b) -> {k: (acc, f1)}; reuse-only mapped onto k=0 if present.
    triples = sorted({(a, c, b) for (_lbl, a, c, b, _k) in cells})
    have_reuse = any(lbl == "reuse-only" for (lbl, *_rest) in cells)

    def get(a, c, b, k):
        # Prefer reuse-only label for k=0 if it exists, else epic@k.
        for lbl in (("reuse-only",) if (k == 0 and have_reuse) else ()) + \
                (f"epic@{k}",):
            key = (lbl, a, c, b, k)
            if key in cells:
                agg = cells[key]
                n = agg["n"] or 1
                return agg["epic_acc"] / n, agg["epic_f1"] / n
        return None

    print("\n=== epic accuracy vs link-k  (per |A|,|C|,|B| cell) ===")
    header = f"{'A':>5} {'C':>5} {'B':>6}" + "".join(
        f"{('k=' + str(k)):>10}" for k in ks)
    print(header)
    for (a, c, b) in triples:
        line = f"{a:>5} {c:>5} {b:>6}"
        for k in ks:
            v = get(a, c, b, k)
            cell = f"{v[0]:.2f}/{v[1]:.2f}" if v else "-"
            line += f"{cell:>10}"
        print(line)
    print("(cells show acc/F1; columns are link-k)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EPIC-reuse accuracy benchmark.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--out", default="acc.csv")
    ap.add_argument("--modes", default="epic",
                    help="Epic mode specs to score against the dense (prefix) "
                    "reference: epic,epic@<k>,reuse-only. The prefix dense ref "
                    "is always run. Use --link-sweep to expand a bare 'epic'.")
    ap.add_argument("--link-sweep", default=None,
                    help="Comma list of link-k (e.g. 0,4,8,32); expands a bare "
                    "'epic' into one epic@k run per k.")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    ap.add_argument("--link", type=int, default=DEFAULT_LINK_TOKENS,
                    help="Default epic link-k for a bare 'epic' mode.")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    ap.add_argument("--plan-only", action="store_true")
    # Internal subprocess hooks.
    ap.add_argument("--single-family", default=None)
    ap.add_argument("--single-label", default=None)
    ap.add_argument("--out-jsonl", default=None,
                    help="Internal: per-mode raw output dump path.")
    args = ap.parse_args(argv)

    reqs = read_jsonl(args.data)
    if not reqs:
        print(f"[epic-acc] no requests in {args.data}")
        return 2

    link_sweep = parse_int_list(args.link_sweep) if args.link_sweep else None
    try:
        all_specs = expand_mode_specs(
            args.modes, link_sweep=link_sweep, default_link=args.link)
    except ValueError as e:  # noqa: BLE001
        print(f"[epic-acc] {e}")
        return 2
    # Only the epic family is scored as 'epic'; prefix is forced as the dense ref.
    epic_specs = [s for s in all_specs if s.is_epic]
    if not epic_specs:
        print("[epic-acc] no epic mode specs to score (need epic / epic@k / "
              "reuse-only).")
        return 2

    if args.plan_only:
        from benchmarks.epic_reuse.common import parse_mode_spec as _pms
        plan_only(reqs, [_pms("prefix")] + epic_specs, args.max_tokens)
        return 0

    if args.single_family:
        run_single_mode(
            args.single_family, args.single_label or args.single_family, reqs,
            model=args.model, block_size=args.block_size,
            chunk_size=args.chunk_size, link_tokens=args.link,
            max_model_len=args.max_model_len, gpu_mem_util=args.gpu_mem_util,
            max_tokens=args.max_tokens, out_jsonl=args.out_jsonl,
        )
        return 0

    # Orchestrator: run prefix (dense ref) once, then each epic spec, each in its
    # own process (engine fixes connector + link-k at construction).
    base = os.path.splitext(args.out)[0]
    dense_jsonl = base + ".dense.jsonl"

    def _launch(family: str, label: str, link_k: int, jpath: str) -> int:
        print(f"\n[epic-acc] === subprocess mode={label} "
              f"(family={family}, link_k={link_k}) ===")
        cmd = [
            sys.executable, "-m", "benchmarks.epic_reuse.bench_accuracy",
            "--data", args.data, "--model", args.model,
            "--single-family", family, "--single-label", label,
            "--out-jsonl", jpath,
            "--max-tokens", str(args.max_tokens),
            "--chunk-size", str(args.chunk_size),
            "--block-size", str(args.block_size),
            "--link", str(link_k),
            "--max-model-len", str(args.max_model_len),
            "--gpu-mem-util", str(args.gpu_mem_util),
        ]
        return subprocess.call(cmd)

    rc = _launch("prefix", "prefix", args.link, dense_jsonl)
    if rc != 0:
        print(f"[epic-acc] dense (prefix) subprocess failed rc={rc}")
        return rc

    epic_jsonls: list[str] = []
    for spec in epic_specs:
        jpath = f"{base}.{spec.label.replace('@', '_at_')}.jsonl"
        rc = _launch("epic", spec.label, spec.link_k, jpath)
        if rc != 0:
            print(f"[epic-acc] mode {spec.label} subprocess failed rc={rc}")
            return rc
        epic_jsonls.append(jpath)

    compare(dense_jsonl, epic_jsonls, args.out)
    print(f"\n[epic-acc] done -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
