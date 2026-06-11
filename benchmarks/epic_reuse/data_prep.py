# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC-reuse benchmark: data preparation.

Builds a JSONL of token-exact A+C+B+Q requests for the perf/accuracy harnesses,
with the chunk-alignment constraints the EPIC content-hash matcher needs.

Two data modes:
  * synthetic-needle (default, offline): deterministic seeded text, exact token
    lengths via the tokenizer, K needle facts embedded in B, a needle question Q.
  * hf (optional): pull passages/questions from an HF dataset (hotpot_qa/squad).

Alignment contract (see common.py header): |A| and |C| are forced to multiples
of the *effective* chunk size (chunk_size rounded up to a block multiple) so B
begins at a chunk-aligned offset in BOTH the warmup prompt and the target prompt
and its chunk hashes collide. |B| is forced to a chunk multiple too.

CPU-only verification: ``--dry-run`` builds the JSONL using the tokenizer alone
(HF small model, or the offline whitespace fallback) and asserts the predicted
B-chunk collisions are non-zero, so the alignment math is checked without a GPU.

Examples:
  # offline dry-run (no network, no GPU): build + verify a small grid
  python data_prep.py --dry-run --out /tmp/epic_bench.jsonl \
      --a-lens 0,256 --c-lens 64,256 --b-lens 256,512 --chunk-size 256

  # real prep with a model tokenizer
  python data_prep.py --model meta-llama/Llama-3.2-1B-Instruct \
      --out data/epic_bench.jsonl
"""

from __future__ import annotations

import argparse
import sys

from benchmarks.epic_reuse.common import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LINK_TOKENS,
    BenchRequest,
    Grid,
    Needle,
    QAItem,
    build_passage_b_of_exact_tokens,
    build_text_of_exact_tokens,
    effective_chunk_size,
    load_qa_items,
    load_tokenizer,
    make_needles,
    parse_int_list,
    resolve_hf_dataset_id,
    round_up,
    split_into_full_chunks,
    write_jsonl,
)


def _exact(ids: list[int], n: int) -> list[int]:
    """Force an id list to exactly ``n`` tokens (truncate or pad with last id).

    Padding only triggers when the tokenizer under-produced (rare subword
    boundary); for the offline whitespace tokenizer it is exact already.
    """
    if n <= 0:
        return []
    if len(ids) >= n:
        return list(ids[:n])
    pad = ids[-1] if ids else 0
    return list(ids) + [pad] * (n - len(ids))


def _build_b_with_needles(
    tok,
    seed: int,
    b_tokens: int,
    needles: list[Needle],
) -> tuple[str, list[int]]:
    """Build B of exactly ``b_tokens`` tokens with needle facts interleaved.

    The needle sentences are placed at the FRONT of B so they survive any token
    truncation, then filler words pad B to the exact length. B is SHARED across
    all requests that use the same (seed, b_tokens, needles) so its chunks
    collide across warmup/target.
    """
    fact_text = " ".join(n.fact_sentence() for n in needles)
    # Use the needle sentences as a deterministic prefix, then grow with seeded
    # filler to the exact token count.
    prefix_words = fact_text.split(" ")
    return build_text_of_exact_tokens(
        tok, seed, b_tokens, prefix_words=prefix_words
    )


def _build_head(tok, seed: int, n_tokens: int, real_text):
    """Build an A/C head segment of exactly n_tokens.

    If ``real_text`` is given (hf mode), it is used as the leading content with
    seeded synthetic filler padding to the exact token count; otherwise pure
    synthetic seeded words (synthetic mode).
    """
    prefix_words = real_text.split(" ") if real_text else None
    return build_text_of_exact_tokens(
        tok, seed, n_tokens, prefix_words=prefix_words
    )


def _shared_distractor_text(qa_items) -> str:
    """Deterministic shared distractor text for A (same every request): the
    concatenation of the gold passages of the FIRST few QA items (answer leakage
    is acceptable in A because A is shared context, not the targeted gold B for
    the current question).
    """
    if not qa_items:
        return ""
    take = qa_items[:4]
    return " ".join(it.gold_passage for it in take)


def _per_request_distractor_text(qa_items, cursor: int) -> str:
    """Per-request distractor text for C: a passage from a DIFFERENT QA item
    than the one used for this request's B (so C is genuinely-new content).
    """
    if not qa_items:
        return ""
    # Offset by a prime so C's source differs from B's (cursor) source.
    idx = (cursor + 7) % len(qa_items)
    it = qa_items[idx]
    pool = ([it.gold_passage] + list(it.distractor_passages))
    return " ".join(pool)


def _answer_in_b(tok, b_ids: list[int], item: QAItem, eff_chunk: int) -> bool:
    """Check the answer text survives in the (truncated) B by decoding B back.

    Best-effort: requires the tokenizer to expose ``decode``. The whitespace
    fallback cannot decode, so we skip the check there (dry-run length-only).
    """
    decode = getattr(getattr(tok, "_tok", None), "decode", None)
    if decode is None:
        return True  # offline fallback: cannot verify, assume ok
    try:
        from benchmarks.epic_reuse.common import normalize_answer

        text = decode(b_ids)
        nt = normalize_answer(text)
        for g in item.gold_answers or [item.answer_text]:
            ng = normalize_answer(g)
            if ng and ng in nt:
                return True
        return False
    except Exception:  # noqa: BLE001
        return True


def _verify_alignment(
    req: BenchRequest,
    eff_chunk: int,
) -> tuple[int, list[str]]:
    """Intersect warmup & target full-chunk hash sets at the TOKEN-ID level and
    return (num_b_chunk_hits, messages). Pure CPU verification of the contract.

    Token ids (not re-tokenized text) are the source of truth, mirroring exactly
    what the connector hashes from ``request.prompt_token_ids`` at serve time.
    """
    msgs: list[str] = []
    target_ids = req.prompt_token_ids()
    warmup_ids = req.warmup_token_ids()

    target_chunks = split_into_full_chunks(target_ids, eff_chunk)
    warmup_chunks = split_into_full_chunks(warmup_ids, eff_chunk)
    warmup_hashes = {h for _, _, h in warmup_chunks}

    # The B region in the target starts at offset a+c (token counts) -- verify it
    # is chunk-aligned and its chunks are present in the warmup hash set.
    b_offset = req.a_tokens + req.c_tokens
    if b_offset % eff_chunk != 0:
        msgs.append(
            f"req {req.req_id}: B offset {b_offset} NOT a multiple of effective "
            f"chunk size {eff_chunk} -- alignment broken."
        )
    hits = 0
    for start, _, h in target_chunks:
        if start < b_offset:
            continue  # A or C region
        if start >= b_offset + req.b_tokens:
            continue  # Q tail (partial chunk anyway)
        if h in warmup_hashes:
            hits += 1
    return hits, msgs


def _finalize_request(
    *,
    req_id: str,
    a_ids: list[int],
    warm_head_ids: list[int],
    c_ids: list[int],
    b_ids: list[int],
    b_text: str,
    q_ids: list[int],
    q_text: str,
    needle_subject: str,
    needle_answer: str,
    gold_answers: list[str],
    task_type: str,
    eff_chunk: int,
    link: int,
    messages: list[str],
    b_tokens: int,
) -> BenchRequest:
    """Assemble + alignment-verify one BenchRequest. Shared by both data modes."""
    warmup_ids = list(warm_head_ids) + list(b_ids) + list(q_ids)
    req = BenchRequest(
        req_id=req_id,
        a_ids=list(a_ids),
        c_ids=list(c_ids),
        b_ids=list(b_ids),
        q_ids=list(q_ids),
        b_text=b_text,
        q_text=q_text,
        needle_subject=needle_subject,
        needle_answer=needle_answer,
        a_tokens=len(a_ids),
        c_tokens=len(c_ids),
        b_tokens=len(b_ids),
        q_tokens=len(q_ids),
        warmup_ids=warmup_ids,
        warmup_b_offset_tokens=len(warm_head_ids),
        target_b_offset_tokens=len(a_ids) + len(c_ids),
        chunk_size=eff_chunk,
        link_tokens=link,
        predicted_b_chunk_hits=0,
        task_type=task_type,
        gold_answers=list(gold_answers),
    )
    hits, msgs = _verify_alignment(req, eff_chunk)
    req.predicted_b_chunk_hits = hits
    messages.extend(msgs)
    if b_tokens >= eff_chunk and hits == 0:
        messages.append(
            f"req {req.req_id}: predicted 0 B-chunk collisions "
            f"(expected >= 1). EPIC reuse will NOT trigger -- check "
            f"tokenizer determinism / chunk alignment."
        )
    return req


def build_records(
    *,
    tok,
    is_real_tok: bool,
    grid: Grid,
    chunk_size: int,
    block_size: int,
    needles_per_b: int,
    requests_per_cell: int,
    seed: int,
    data_mode: str = "synthetic",
    qa_items: list[QAItem] | None = None,
) -> tuple[list[BenchRequest], list[str]]:
    eff_chunk = effective_chunk_size(chunk_size, block_size)
    messages: list[str] = []
    records: list[BenchRequest] = []
    qa_items = qa_items or []
    qa_cursor = 0  # rotates through QA items for hf mode

    for (a_req, c_req, b_req) in grid.cells():
        # Force chunk alignment for A, C, and B.
        a_tokens = round_up(a_req, eff_chunk) if a_req > 0 else 0
        c_tokens = round_up(c_req, eff_chunk) if c_req > 0 else 0
        b_tokens = round_up(b_req, eff_chunk)
        if (a_tokens, c_tokens, b_tokens) != (a_req, c_req, b_req):
            messages.append(
                f"cell (A={a_req},C={c_req},B={b_req}) adjusted to "
                f"(A={a_tokens},C={c_tokens},B={b_tokens}) for chunk alignment "
                f"(eff_chunk={eff_chunk})."
            )

        # ---- A: SHARED across requests in a cell (prefix reuse target) ----
        # In hf mode A is built from shared distractor passages (still seeded so
        # identical |A| share content); falls back to synthetic words.
        a_distractor_text = _shared_distractor_text(qa_items) if data_mode == "hf" else None
        a_text, a_ids_full = (
            _build_head(tok, 10000 + a_tokens, a_tokens, a_distractor_text)
            if a_tokens > 0
            else ("", [])
        )
        a_ids = _exact(a_ids_full, a_tokens)

        # Warmup head: a DIFFERENT prefix than A but the SAME length (chunk
        # multiple) so B lands at a chunk-aligned offset in the warmup prompt.
        _, warm_head_full = (
            _build_head(tok, 55000 + a_tokens, a_tokens, None)
            if a_tokens > 0
            else ("", [])
        )
        warm_head_ids = _exact(warm_head_full, a_tokens)

        # ---- synthetic-mode B is SHARED across the whole cell ----
        if data_mode != "hf":
            b_seed = 90000 + b_tokens  # depends only on length -> stable B/size
            needles = make_needles(b_seed, needles_per_b)
            b_text, b_ids = _build_b_with_needles(tok, b_seed, b_tokens, needles)
            b_ids = b_ids[:b_tokens]
            if len(b_ids) < b_tokens:
                messages.append(
                    f"cell B={b_tokens}: tokenizer produced only {len(b_ids)} "
                    "ids for B (under target); padding with filler id."
                )
                b_ids = b_ids + [b_ids[-1] if b_ids else 0] * (b_tokens - len(b_ids))

        for r in range(requests_per_cell):
            # C is UNIQUE per request (genuinely-new tokens, always recomputed).
            c_seed = 20000 + (a_tokens * 31 + c_tokens * 17 + b_tokens * 7 + r)
            c_distractor = (
                _per_request_distractor_text(qa_items, qa_cursor)
                if data_mode == "hf"
                else None
            )
            c_text, c_ids_full = (
                _build_head(tok, c_seed, c_tokens, c_distractor)
                if c_tokens > 0
                else ("", [])
            )
            c_ids = _exact(c_ids_full, c_tokens)

            if data_mode == "hf":
                # ---- hf B: per-request gold passage (answer-preserving) ----
                item = qa_items[qa_cursor % len(qa_items)]
                qa_cursor += 1
                pad = list(item.distractor_passages)
                b_text, b_ids = build_passage_b_of_exact_tokens(
                    tok, item.gold_passage, item.answer_text, b_tokens,
                    filler_seed=90000 + b_tokens + r, pad_passages=pad,
                )
                b_ids = b_ids[:b_tokens]
                if len(b_ids) < b_tokens:
                    b_ids = b_ids + [b_ids[-1] if b_ids else 0] * (
                        b_tokens - len(b_ids))
                q_text = " " + item.question.strip()
                q_ids = tok.encode(q_text, add_special_tokens=False)
                gold_answers = item.gold_answers or [item.answer_text]
                needle_subject = "hf"
                needle_answer = item.answer_text
                task_type = "hf"
                if not _answer_in_b(tok, b_ids, item, eff_chunk):
                    messages.append(
                        f"req a{a_tokens}_c{c_tokens}_b{b_tokens}_r{r}: gold "
                        f"answer not found in (possibly truncated) B; consider a "
                        f"larger --b-lens."
                    )
            else:
                needle = needles[r % len(needles)] if needles else Needle("none", "0")
                q_text = needle.question()
                q_ids = tok.encode(q_text, add_special_tokens=False)
                gold_answers = [needle.answer]
                needle_subject = needle.subject
                needle_answer = needle.answer
                task_type = "needle"

            records.append(_finalize_request(
                req_id=f"a{a_tokens}_c{c_tokens}_b{b_tokens}_r{r}",
                a_ids=a_ids, warm_head_ids=warm_head_ids, c_ids=c_ids,
                b_ids=b_ids, b_text=b_text, q_ids=q_ids, q_text=q_text,
                needle_subject=needle_subject, needle_answer=needle_answer,
                gold_answers=gold_answers, task_type=task_type,
                eff_chunk=eff_chunk, link=grid.link, messages=messages,
                b_tokens=b_tokens,
            ))

    return records, messages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="EPIC-reuse benchmark data prep (A+C+B+Q)."
    )
    ap.add_argument("--out", default="epic_bench.jsonl", help="Output JSONL path.")
    ap.add_argument(
        "--model",
        default=None,
        help="HF model id for the tokenizer (omit -> offline whitespace "
        "fallback, valid only for --dry-run length accounting).",
    )
    ap.add_argument(
        "--data-mode",
        choices=("synthetic", "hf"),
        default="synthetic",
        help="synthetic needle (default, offline) or HF dataset passages.",
    )
    ap.add_argument(
        "--hf-dataset", default="squad",
        help="HF dataset for hf mode: 'squad' (default, small/fast) or "
        "'hotpot_qa'. Friendly names map to namespaced repo ids.",
    )
    ap.add_argument("--hf-split", default="validation",
                    help="HF split for hf mode (default: validation).")
    ap.add_argument("--hf-limit", type=int, default=128,
                    help="Max HF examples to pull (rotated across requests).")
    ap.add_argument("--self-test", action="store_true",
                    help="Run scoring + mode-spec self-tests and exit (CPU).")
    ap.add_argument("--a-lens", default="0,256,1024")
    ap.add_argument("--c-lens", default="64,256,1024")
    ap.add_argument("--b-lens", default="256,1024,4096")
    ap.add_argument("--link", type=int, default=DEFAULT_LINK_TOKENS)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    ap.add_argument("--needles-per-b", type=int, default=4)
    ap.add_argument("--requests-per-cell", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build + verify with the tokenizer only (CPU); no GPU/model.",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        from benchmarks.epic_reuse import common as _c

        _c._scoring_self_test()
        # mode-spec round-trips
        assert _c.parse_mode_spec("reuse-only").link_k == 0
        assert _c.parse_mode_spec("epic@32").link_k == 32
        labels = [s.label for s in _c.expand_mode_specs(
            "full,prefix,epic", link_sweep=[0, 8, 32])]
        assert labels == ["full", "prefix", "epic@0", "epic@8", "epic@32"], labels
        print("[epic-bench] self-test PASSED (scoring + mode specs).")
        return 0

    if args.chunk_size % args.block_size != 0:
        eff = effective_chunk_size(args.chunk_size, args.block_size)
        print(
            f"[epic-bench] note: chunk_size {args.chunk_size} is not a multiple "
            f"of block_size {args.block_size}; effective chunk size = {eff} "
            "(mirrors the connector's rounding)."
        )

    grid = Grid(
        a_lens=parse_int_list(args.a_lens),
        c_lens=parse_int_list(args.c_lens),
        b_lens=parse_int_list(args.b_lens),
        link=args.link,
    )

    qa_items = None
    if args.data_mode == "hf":
        repo = resolve_hf_dataset_id(args.hf_dataset)
        try:
            qa_items = load_qa_items(
                args.hf_dataset, split=args.hf_split, limit=args.hf_limit
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[epic-bench] HF dataset {args.hf_dataset!r} (-> {repo!r}) "
                f"unavailable ({type(e).__name__}: {e}).\n"
                "[epic-bench] This usually means no network / 'datasets' not "
                "installed. Install with '<venv>/bin/pip install datasets' and "
                "ensure HF Hub is reachable, or use --data-mode synthetic "
                "(default, fully offline). Synthetic fallback NOT applied "
                "automatically in hf mode -- aborting so you get real data or a "
                "clear error."
            )
            return 2
        if not qa_items:
            print(f"[epic-bench] HF dataset {repo!r} returned 0 usable QA items.")
            return 2
        print(f"[epic-bench] loaded {len(qa_items)} QA items from {repo!r} "
              f"(split={args.hf_split}).")

    tok, is_real = load_tokenizer(args.model, allow_fallback=args.dry_run)
    if not is_real and not args.dry_run:
        print(
            "[epic-bench] ERROR: no real tokenizer loaded and not in --dry-run. "
            "Pass --model <hf-id> for a real prep, or add --dry-run for offline "
            "length verification."
        )
        return 2

    print(
        f"[epic-bench] tokenizer={tok.name} data-mode={args.data_mode} grid: "
        f"A={grid.a_lens} C={grid.c_lens} B={grid.b_lens} link={grid.link} "
        f"chunk={args.chunk_size} block={args.block_size} "
        f"reqs/cell={args.requests_per_cell}"
    )

    records, messages = build_records(
        tok=tok,
        is_real_tok=is_real,
        grid=grid,
        chunk_size=args.chunk_size,
        block_size=args.block_size,
        needles_per_b=args.needles_per_b,
        requests_per_cell=args.requests_per_cell,
        seed=args.seed,
        data_mode=args.data_mode,
        qa_items=qa_items,
    )

    write_jsonl(args.out, records)

    # Report.
    n_cells = len(list(grid.cells()))
    total_hits = sum(r.predicted_b_chunk_hits for r in records)
    zero_hit = sum(
        1
        for r in records
        if r.b_tokens >= effective_chunk_size(args.chunk_size, args.block_size)
        and r.predicted_b_chunk_hits == 0
    )
    print(f"[epic-bench] wrote {len(records)} requests over {n_cells} cells "
          f"-> {args.out}")
    print(f"[epic-bench] predicted total B-chunk collisions: {total_hits}")
    for m in messages[:40]:
        print(f"[epic-bench]   {m}")
    if len(messages) > 40:
        print(f"[epic-bench]   ... ({len(messages) - 40} more messages)")

    if zero_hit > 0:
        print(
            f"[epic-bench] FAIL: {zero_hit} request(s) with B >= chunk had ZERO "
            "predicted collisions -- alignment/determinism broken."
        )
        return 1

    print("[epic-bench] alignment verification PASSED "
          "(every B>=chunk request has >=1 predicted collision).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
