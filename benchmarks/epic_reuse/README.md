# EPIC-reuse Benchmark Suite

Accuracy + performance benchmark comparing **EPIC non-contiguous KV reuse**
(`vllm/distributed/kv_transfer/kv_connector/v1/epic/`) against **vanilla vLLM
prefix caching** on the same model, data, and fairness settings.

Adds benchmarks only; it does **not** modify any vLLM source.

## Prompt layout: A + C + B + Q

```
prompt = A          C          B            Q
         (shared    (per-req   (shared      (needle
          prefix)    unique)    passage,     question)
                                MIDDLE)
```

| segment | meaning | full | prefix | epic |
|---|---|---|---|---|
| A | shared prefix doc | recompute | reuse (native prefix cache) | reuse (native) |
| C | per-request new context | recompute | recompute | recompute (in M) |
| B | shared passage in the **middle** (non-prefix) | recompute | **recompute** | **reuse** (content-hash + PIC) |
| Q | needle question | recompute | recompute | recompute (in M) |

`prefix` cannot reuse B: a different C in front breaks the prefix-chain hash.
`epic` reuses B by content hash + PIC re-rotary and only recomputes
`M = C ∪ link-tokens(B) ∪ {last}`. **EPIC's win grows as |B|↑ and |C|↓.**

## Modes and mode specs (identical model + conditions)

Base families:

1. **full** — prefix caching OFF (+ per-request cache-bust). Full-prefill floor.
2. **prefix** — vanilla prefix caching only (no connector). Reuses A; recomputes C+B.
   This is the **dense reference** (numerically-correct answer) in accuracy runs.
3. **epic** — `EpicConnector` + `epic_sparse_forward=true` + `epic_fusion_mask=true`.

The epic family is parameterized by **link-k** (`epic_link_tokens`): the number of
leading boundary tokens of each reused non-prefix B chunk that are *recomputed*
to stitch the seam. The connector fixes k at **engine construction**, so each k
is a separate engine == a separate *mode spec*:

| mode spec | meaning |
|---|---|
| `epic` | epic at the default `--link` (k=8 unless overridden). |
| `epic@<k>` | epic at link-k (e.g. `epic@0`, `epic@4`, `epic@32`). |
| `reuse-only` | **alias of `epic@0`**: B is PIC re-rotated and reused with **ZERO recompute** (M = C ∪ {last}). The naive KV-reuse baseline (no recompute, position-fix only). Labeled distinctly in tables/plots. |

Two ways to request a k-sweep:

```bash
--modes full,prefix,epic@0,epic@4,epic@8,epic@32   # explicit
--modes full,prefix,epic --link-sweep 0,4,8,32     # auto-expands the bare 'epic'
```

`reuse-only` (= naive reuse, recompute none) maps to k=0 in the k-axis tables and
plots, so it shares the k=0 column with `epic@0` but carries its own label.

**Fairness:** all three modes run the `FLEX_ATTENTION` backend +
`enforce_eager` (EPIC's hard requirement, applied to the baselines too, so only
the algorithm differs). The harness passes the backend per mode automatically
via the `attention_backend` LLM/EngineArgs kwarg (the legacy
`VLLM_ATTENTION_BACKEND` env var was removed in vLLM v0.22).
`--baseline-backend FLASH_ATTN` adds a non-fair reference for the baselines.

## GPU requirements

- A CUDA box with EPIC-built vLLM (`VLLM_USE_PRECOMPILED=1 uv pip install -e .`).
- A small Llama-family model is recommended (e.g. `meta-llama/Llama-3.2-1B-Instruct`).
- `data_prep.py --dry-run` and every `--plan-only` mode are **CPU-only**.

## Alignment constraint (read this)

EPIC hashes whole `chunk_size` chunks counted from prompt position 0. For B's
chunks to collide between the warmup prompt and the target prompt, **B must
start at a chunk-aligned offset in both**. `data_prep.py` therefore:

- forces `|A|` and `|C|` to multiples of the *effective* chunk size
  (`chunk_size` rounded up to a `block_size` multiple, mirroring the connector),
- forces `|B|` to a chunk multiple,
- builds prompts at the **token-id level** (segments tokenized independently and
  concatenated) so B's ids are byte-identical regardless of subword boundaries,
- statically verifies (`predicted_b_chunk_hits`) that every `B >= chunk` request
  has ≥1 predicted chunk collision, and **fails the dry-run** otherwise.

For fine-grained small sweeps, lower `--chunk-size` (a `block_size` multiple,
default block 16) so e.g. `|C|=64` survives instead of rounding to 256.

## Workflow

```bash
# 0) (one-time) install a tokenizer / matplotlib into the bench venv if needed
#    <venv>/bin/pip install transformers matplotlib

# 1) DATA PREP — synthetic needle (offline). Verifies alignment on CPU.
<venv>/bin/python -m benchmarks.epic_reuse.data_prep \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --a-lens 0,256,1024 --c-lens 64,256,1024 --b-lens 256,1024,4096 \
    --link 8 --chunk-size 256 --block-size 16 \
    --requests-per-cell 5 --out epic_bench.jsonl

#    CPU-only check first (whitespace fallback if no tokenizer):
<venv>/bin/python -m benchmarks.epic_reuse.data_prep --dry-run --out /tmp/chk.jsonl

#    HF real-QA prep (RAG-style A+C+B+Q; B = gold passage, Q = real question):
<venv>/bin/python -m benchmarks.epic_reuse.data_prep \
    --data-mode hf --hf-dataset squad --hf-limit 128 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --a-lens 256 --c-lens 256 --b-lens 512 --chunk-size 256 --out epic_hf.jsonl

# 2) PERF — TTFT + prefill tok/s per mode spec (auto subprocess per spec).
#    k-sweep: one engine per k (connector fixes k at construction).
VLLM_LOGGING_LEVEL=WARNING <venv>/bin/python -m benchmarks.epic_reuse.bench_perf \
    --data epic_bench.jsonl --model meta-llama/Llama-3.2-1B-Instruct \
    --modes full,prefix,reuse-only,epic --link-sweep 0,8,32 \
    --measure-reps 5 --chunk-size 256 --block-size 16 --out perf.csv
#    CPU schedule preview (expands epic@k):
<venv>/bin/python -m benchmarks.epic_reuse.bench_perf --data epic_bench.jsonl \
    --plan-only --link-sweep 0,8,32

# 3) ACCURACY — dense(prefix) vs each epic spec. needle: answer substring;
#    hf: answer-containment + SQuAD token-F1.
<venv>/bin/python -m benchmarks.epic_reuse.bench_accuracy \
    --data epic_hf.jsonl --model meta-llama/Llama-3.2-1B-Instruct \
    --modes reuse-only,epic --link-sweep 0,8,32 \
    --max-tokens 32 --chunk-size 256 --block-size 16 --out acc.csv
#    CPU plan preview:
<venv>/bin/python -m benchmarks.epic_reuse.bench_accuracy --data epic_hf.jsonl \
    --plan-only --link-sweep 0,8,32

# 4) PLOTS (matplotlib; prints guidance if missing).
#    Adds (e) the k-tradeoff curve (accuracy/F1 + TTFT vs link-k).
<venv>/bin/python -m benchmarks.epic_reuse.plot_results \
    --perf perf.csv --acc acc.csv --fix-a 0 --fix-c 256 --fix-b 512 --outdir plots/
```

> Always invoke as a module from the repo root (`python -m benchmarks.epic_reuse.X`)
> so the `benchmarks` package resolves. Use the project venv, never system python.

## Why subprocess-per-mode

The attention backend and the KV connector are fixed at **engine construction**.
A single process can host only one mode. `bench_perf` / `bench_accuracy` re-exec
themselves once per mode (`--single-mode`) and aggregate the CSVs. The EPIC store
is in-process, so warmup→measure for the epic mode happens in the same engine.

## Cache-bust for the `full` baseline

`full` sets `enable_prefix_caching=False` so every request is a true full
prefill. As a belt-and-braces second bust, the harness perturbs the first token
of A per request (`busted_prompt_token_ids`) so even a surviving cache cannot
hit. Length is preserved so timing stays comparable.

## Outputs

- `perf.csv`: `mode,A,C,B,link,link_k,ttft_mean_s,ttft_std_s,prefill_tokens,tokens_per_s,n_measured`
  where `mode` is the spec label (`full`/`prefix`/`epic@k`/`reuse-only`) and
  `link_k` is the epic recompute-boundary tokens (`-1` for baselines). Console:
  a speedup table (baseline / `epic_best`, where `epic_best` is the lowest-TTFT
  epic spec per cell) **and** a `TTFT vs link-k` table.
- `acc.csv`: `mode,A,C,B,link_k,n,dense_needle_acc,epic_needle_acc,acc_gap,dense_f1,epic_f1,f1_gap,exact_match_rate,token_prefix_match`
  — one row per epic spec vs the dense (prefix) reference. Console also prints an
  `epic accuracy vs link-k` table.
- `plots/*.png`: `ttft_vs_B`, `speedup_vs_B`, `ttft_vs_A`, `accuracy_vs_B`, and
  **`k_tradeoff`** (accuracy/F1 + TTFT vs link-k at one cell — the EPIC core figure).

JSONL records carry `task_type` (`"needle"` | `"hf"`) and `gold_answers` (the
accepted answer aliases). Old JSONL without these fields still loads (the single
needle answer is back-filled as the only gold).

## Result interpretation

- **EPIC speedup is largest when |B| is large and |C| is small** — most of the
  prompt is reused B, and only a small M (C + link + last) is recomputed.
- At **|B|=0 or |B|<chunk**, EPIC has nothing to reuse → expect parity with
  `prefix` (and a `predicted_b_chunk_hits=0` warning from data_prep).
- At **|A| large**, both `prefix` and `epic` reuse A natively, so their gap is
  driven purely by B; `full` keeps paying for A every time.
- **Accuracy**: EPIC is an approximation. Watch `acc_gap` (dense − epic needle
  recall) and `f1_gap`. A small gap at large |B| means reuse is safe; a large gap
  (especially with low `token_prefix_match`) flags a mask / PIC / position bug,
  not just approximation noise.

### The link-k tradeoff curve (EPIC's core story)

The `k_tradeoff` plot and the `vs link-k` tables tell the central EPIC story:

- **`reuse-only` (k=0)** — naive KV reuse: B is PIC re-rotated and reused with
  **zero recompute**. Expect the **lowest TTFT** and the **lowest accuracy**: no
  recompute means the seam between non-prefix chunks is never stitched, so the
  model can mis-bind B's content. This is the floor of the curve.
- **k increasing (`epic@4` → `epic@8` → `epic@32`)** — each step recomputes more
  boundary tokens per B chunk: **accuracy/F1 recover** toward dense while **TTFT
  rises** (more recompute). EPIC's claim is that a *small* k recovers most of the
  accuracy for a *fraction* of the recompute cost.
- **`dense` (prefix)** — recomputes all of B: the **accuracy ceiling** and the
  TTFT baseline that EPIC undercuts.

A healthy run shows a monotone accuracy-up / TTFT-up curve between the reuse-only
floor and the dense ceiling. If `reuse-only` already matches dense accuracy, the
task is too easy (B reuse is trivially safe — increase |C| pressure or use harder
HF questions). If even `epic@32` stays far below dense, suspect a PIC / mask bug.

## Data modes

- `--data-mode synthetic` (default, offline): deterministic seeded text, exact
  token lengths, K needle facts in B (`--needles-per-b`), needle question Q.
  `task_type="needle"`, scored by answer substring.
- `--data-mode hf` (real QA, needs `datasets` + network): RAG-style wiring of a
  real dataset into A+C+B+Q.
  - **B** = the **gold passage that contains the answer**, built to the exact
    `b_tokens` length. If the passage is longer it is truncated **sentence-wise
    around the answer span** so the answer sentence is never cut; if shorter it
    is padded with other real passages then deterministic filler.
  - **A** = shared distractor passages (the prefix-reuse target, identical across
    requests). **C** = a per-request distractor passage (genuinely-new tokens).
    **Q** = the real question. `gold_answers` = the dataset answer aliases.
  - Datasets: `--hf-dataset squad` (default, small/fast) or `hotpot_qa`. Friendly
    names map to namespaced repo ids (`rajpurkar/squad`, `hotpotqa/hotpot_qa`).
  - `task_type="hf"`, scored by **answer-containment + SQuAD-style token-F1**
    (normalization: lowercase / strip punctuation / drop articles / squash
    whitespace).
  - If `datasets` is missing or the Hub is unreachable, `data_prep` prints clear
    guidance and **aborts** (no silent synthetic fallback in hf mode) so you get
    real data or an explicit error. Use `--data-mode synthetic` for fully
    offline runs.

The chunk-alignment contract is identical in both modes: B is still built at the
token-id level to exactly `b_tokens` (a chunk multiple) so its content hashes
collide between the warmup and target prompts.

## Turn-migration bench: fileKV vs worker-to-worker copy (`bench_migration.py`)

Answers: *worker1 served turn 9; turn 10 lands on worker2 — is it better to
W2W-copy the whole history KV from (busy) worker1, or recompute the history
on worker2 with fileKV assist?*

Strategies compared: `w2w` (copy full history KV = exact prefix reuse; large
bytes + worker1 HBM interference), `filekv` (recompute non-file history, load
file chunks from the CPU store / GPU staging; zero worker1 contact; H2D hidden
on a prefetch hit), `full` (recompute floor).

Three layers:

1. **Microbench** (needs >= 2 CUDA devices) — measures the machine inputs:
   D2D peer bandwidth with the source idle vs busy (HBM-bound triad load),
   the triad's own slowdown while a copy streams out (== worker1 serving
   degradation), pinned H2D, and the through-CPU staged fallback:
   ```bash
   python -m benchmarks.epic_reuse.bench_migration --run -o migration.json
   ```
2. **Cost model** (CPU, torch-free) — sweeps history x file-fraction x
   src-busy x prefetch-hit and prints per-strategy TTFT, bytes moved, and
   worker1 interference, with the winner per cell:
   ```bash
   python -m benchmarks.epic_reuse.bench_migration --plan-only \
       --measured migration.json --prefill-tokps <from bench_perf>
   ```
   Arithmetic pinned by `tests/.../epic/test_migration_bench.py`.
3. **End-to-end** (2 GPUs, heavy — run after the model narrows the grid):
   - worker1 = engine on GPU0 serving a steady decode load (reuse
     `bench_perf` with a long-running batch).
   - `w2w` arm: while worker1 decodes, copy `history_tokens *
     bytes_per_token` from GPU0 tensors to GPU1 (this bench's copy routine),
     then prefill only the new tokens on a GPU1 engine; record worker1's
     tok/s during the window.
   - `filekv` arm: GPU1 engine with `EpicConnector` + staged chunks
     (`epic_prefetch_gpu_bytes` > 0), full turn-10 prompt; worker1 untouched.
   - Report: turn-10 TTFT, worker1 tok/s dip, and accuracy delta of the two
     arms (w2w is exact; filekv is EPIC-approximate — quality goes through
     `bench_accuracy`).

Honest caveat baked into the model: on PCIe-class defaults `w2w` wins raw
TTFT for moderate file fractions (copying 128 KiB/token at 25-40 GB/s beats
recomputing half the history). `filekv`'s wins concentrate where (a) the
history is mostly file content, (b) the interconnect is slow/contended or
cross-node (no p2p), (c) worker1's KV was already evicted (w2w impossible —
fileKV persists on CPU), or (d) worker1 interference is priced in. The bench
exists to locate that boundary on real hardware instead of asserting it.
