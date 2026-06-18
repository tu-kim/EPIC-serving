# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPIC PIC/reuse end-to-end probe on CacheBlend's *musique* RAG data.

Standalone GPU script (NOT a pytest module -- no ``test_*`` functions, runs from
``__main__``). It answers the question "does EPIC non-prefix reuse PRESERVE the
answer and SAVE prefill time on real multi-hop RAG prompts?" using the exact
dataset CacheBlend ships (``inputs/musique_s.json``). Each musique sample is a
multi-hop question with 10 retrieved Wikipedia contexts and a gold answer list.

WHAT IT DOES (per the task spec)
--------------------------------
For each of N samples we assemble a prompt
``[sys][ctx0][ctx1]...[ctxK][question]`` at the TOKEN level and run it through
four engine modes, each in a FRESH
subprocess (one in-process engine per mode, processing all N samples
sequentially so the per-engine EPIC store accumulates harmlessly -- content
hashes differ across samples):

  * ``full``        -- no connector. Whole prompt full-prefill. == ground truth.
  * ``reuse-only``  -- epic@0. Warm each ctx individually (connector stores its
                       chunks), then issue the whole prompt as a reuse request:
                       the ctx chunks are matched NON-PREFIX and loaded, ZERO
                       recompute (M = question + last token only) -- pure stale
                       reused KV + PIC re-rotary.
  * ``epic@k``      -- same as reuse-only but LegoLink recomputes the leading
                       ``k`` tokens of EACH non-prefix chunk
                       (``epic_link_tokens=k``).

LEGOLINK / SYSTEM SEGMENT (why the prompt starts with ``[sys]``)
----------------------------------------------------------------
EpicSelection walks chunks from prompt position 0 and absorbs every contiguous
hit into the *prefix* extent; only chunks that fall OUTSIDE that contiguous
prefix become ``non_prefix_hits``, and the connector's sparse/LegoLink recompute
path is gated on ``non_prefix_hits`` being non-empty. If we lay out
``[ctx0][ctx1]...[ctxK][question]`` and warm EVERY ctx, then ctx0 (pos 0), ctx1
(pos len0), ... are each a hit starting exactly where the previous chunk ended,
so they ALL fold into the contiguous prefix -> ``non_prefix_hits == []`` ->
LegoLink is INERT: link=k recomputes NOTHING for any k, contexts are reused
STALE, output degrades, and the "speedup" is an artefact of skipping all
recompute. (Confirmed on GPU: link=256 gave 3.48x and garbage output.)

FIX (mirrors CacheBlend's ``[system][docs][query]`` layout): prepend a leading
instruction/system segment that is chunk-padded but is **NOT warmed** into the
store. Its chunk at position 0 is therefore a NON-hit -> contiguity is broken at
the first chunk -> ctx0..K can no longer be absorbed into the prefix and ALL
become non-prefix hits -> LegoLink can engage. The sys segment is included in
both the full and reuse prompts (fair comparison) and is always part of M (new
tokens), as is the question. ``--no-system`` reverts to the buggy all-prefix
layout as a control. Per-request SELECTION diagnostics (prefix_extent /
non_prefix count+offsets / sparse_branch) are surfaced from the connector so the
silent INERT case is flagged loudly rather than mistaken for a working speedup.

RESIDUAL BUG + NONCE FIX (why "not warmed" was not enough)
----------------------------------------------------------
"Not warmed by the bench" does NOT keep the sys chunk a non-hit. The connector's
SAVE path (``epic_connector.py`` ``build_connector_meta``) saves every whole
chunk fully inside M. For a REUSE request the sys segment is entirely in M (it is
recomputed, never loaded), so the FIRST reuse request SAVES the sys chunk into
the per-engine store. Since one mode subprocess processes all N samples against a
single shared store, sample 0's reuse saves sys -> sample 1+ sees sys as a HIT at
position 0 -> the ctx run folds back into the contiguous prefix ->
``non_prefix_hits == 0`` -> LegoLink INERT (and with ``--warmup-discard`` the
MEASURED samples are exactly the post-sample-0 ones that hit this). FIX: make the
sys segment UNIQUE PER SAMPLE by prepending a per-sample NONCE (the sample index
encoded as tokens) to the sys content. The nonce lands in the first sys chunk, so
that chunk's content hash differs per sample and can never collide with a sys
chunk an earlier sample saved -> position 0 is ALWAYS a non-hit -> every ctx
stays non-prefix on every measured sample. The nonce is confined to the sys
region: ctx ids / chunk hashes / byte-identity vs. the warm prefills are
UNCHANGED, and the same sample's full and reuse modes share the same nonce (fair
comparison). This is a deliberate test-side workaround for a KNOWN connector
limitation -- EpicSelection treats any contiguous run of hits from position 0 as
prefix, so a hit sys would drag the docs into the prefix; to MEASURE non-prefix
doc reuse we keep sys intentionally non-hit. The connector is NOT changed.

``full`` runs the connector OFF (sparse off). ``reuse-only`` / ``epic@k`` run
sparse ON + fusion mask + FLEX_ATTENTION + enforce_eager (the S7 safety gate
requires exactly this) and assert in-band engagement
(``epic_debug_counters``: sparse_match>=1, chunks_loaded>=1).

CHUNK ALIGNMENT (why we pad each ctx -- and how this DIFFERS from CacheBlend)
----------------------------------------------------------------------------
The EPIC chunk store hashes WHOLE ``chunk_size`` chunks counted from prompt
position 0; a chunk is only reusable if it is byte-identical AND lands on a
chunk boundary in both the warm prompt (where it was saved) and the reuse
prompt (where it is matched). Re-tokenizing "ctx_a + ctx_b" does NOT keep
ctx_b's ids stable or chunk-aligned (subword boundaries shift with neighbours).
So we assemble at the TOKEN level and PAD EACH ctx up to a multiple of
``chunk_size`` with rotating REAL-WORD filler (never blank space -- blank
collapses output to a trivial constant). Each ctx then occupies an integer
number of whole chunks and its id-slice is byte-identical between the per-ctx
warm prefill and the full reuse prompt -> the content hash collides -> a real
non-prefix hit.

  *** DIFFERENCE vs CacheBlend ***: CacheBlend concatenates the raw contexts
  with NO padding and reuses arbitrary token spans. Our current EPIC
  implementation reuses BLOCK-ALIGNED whole chunks only, so we must pad each ctx
  to a chunk multiple. The padding is wasted compute on the warm side and a
  (small) length overhead on the reuse side; we LOG the waste. Removing the
  alignment constraint (arbitrary-span reuse like CacheBlend) is future work in
  the connector, not this script.

  ``link=k`` is applied PER non-prefix CHUNK. If a ctx spans C chunks, the
  leading k tokens of EACH of its C chunks are recomputed (so the effective
  recompute for that ctx is ~C*k tokens, NOT k). With a LARGE ``--chunk-size``
  (e.g. 512/768) most ctxs fit in ~1 chunk, so "link=k per chunk" ~= "k per
  ctx"; we log chunks-per-ctx so the reader can interpret k correctly.

TIMING
------
Only the MEASURED (reuse / full) request is timed; warm prefills are excluded.
We time the prefill via a ``max_tokens=1`` greedy request's wall-clock around
``llm.generate`` (TTFT-equivalent for a single short request), then a SEPARATE
``max_tokens``-token greedy request produces the answer text scored for answer
containment. All modes use FLEX_ATTENTION + enforce_eager so the only thing
that differs is the algorithm (fairness). ``--warmup-discard`` drops the first
sample's numbers (cold-start compile/autotune) from the aggregate.

OUTPUT
------
A per-sample table (mode | answer_hit | prefill_ms | generated-text head) and an
aggregate table (mode | answer_hit_rate | mean_prefill_ms | speedup vs full).

CPU / no-GPU behaviour: prints guidance and exits 0 (so a structure-only import
check is not penalised). ``--help`` and ``py_compile`` work with no GPU.

REALISTIC RAG REUSE SCENARIOS (``--scenario``; reorder/insert/same)
-------------------------------------------------------------------
On top of the per-ctx warm probe above, this script adds a warm->measure flow
that mirrors a real RAG server and -- crucially -- SPLITS the two reuse
mechanisms so a failure localises. A FIRST request full-prefills
``[system][a][b][c]`` (connector STORES sys/a/b/c by content hash); a SECOND
("measured") request reuses them under ``--scenario``:

  * ``reorder`` -> ``[system][b][a][c][query]``. sys is warmed (a HIT) and
    b,a,c are all hits CONTIGUOUS from the sys boundary, so they fold into the
    prefix (``non_prefix==0``, sparse-forward NOT taken). But every ctx MOVED,
    so the prefix-load applies a NON-ZERO PIC delta re-rotary -> this is the
    FIRST real-data test of PIC re-rotary + load + scatter ALONE.
  * ``insert``  -> ``[system][d][a][c][query]``. ``d`` is a held-out (never
    warmed) ctx of the SAME sample spliced after sys -> its chunk is a non-hit
    -> contiguity breaks -> a,c become NON-PREFIX hits -> the sparse-forward +
    LegoLink recompute path engages (``non_prefix>0``, ``sparse_branch==True``).
  * ``same``    -> ``[system][a][b][c][query]`` (legacy nonce control).

WHAT THIS ANSWERS (the link=255 collapse, split into PIC vs sparse-forward):
  reorder reuse-only CORRECT  => PIC re-rotary/load/scatter are SOUND (first
                                 PIC validation on real data passes).
  reorder fine, insert-epic COLLAPSES => bug is in the SPARSE-FORWARD path, not
                                 PIC.
  reorder ALSO COLLAPSES      => bug is in PIC/load itself.
Per-mode connector counters (loads_emitted, chunks_loaded, sparse_match,
check_load_*) and per-request selection (prefix_extent, non_prefix offsets,
sparse_branch) are printed so reorder's "sparse_match=0" and insert's
"sparse_branch=True" are visible, and the silent INERT case is flagged loudly.

Usage (CUDA box, after ``VLLM_USE_PRECOMPILED=1 uv pip install -e .``):

    # legacy per-ctx probe shape (scenario defaults to 'same'):
    .venv/bin/python benchmarks/epic_reuse/musique_blend.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --num-samples 5 --chunk-size 256 --link 8

    # PIC-only validation (sparse-forward NOT exercised):
    .venv/bin/python benchmarks/epic_reuse/musique_blend.py \
        --scenario reorder --num-samples 5 --chunk-size 256 --link 8

    # sparse-forward + LegoLink validation:
    .venv/bin/python benchmarks/epic_reuse/musique_blend.py \
        --scenario insert --num-samples 5 --chunk-size 256 --link 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass

# fork-safety: probe CUDA via NVML (no driver context in the parent) and force
# spawn for any vLLM child. Identical rationale to gpu_smoke.py -- the parent
# tokenizes + dispatches per-mode workers and must NOT create a CUDA context
# (a forked EngineCore could not re-init it). Set before torch/vllm import.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Make the repo root importable so ``tests.*`` and ``benchmarks.*`` resolve when
# this script is run directly (python benchmarks/epic_reuse/musique_blend.py)
# from any cwd. The repo root is three levels up from this file
# (benchmarks/epic_reuse/musique_blend.py -> repo root).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the gpu_smoke subprocess-isolation + result plumbing verbatim (no
# duplicate implementations): the RESULT_JSON sentinel/parse, the spec
# serialise/parse, and the fork-safe CUDA probe. We only ADD a richer worker
# (per-sample warm groups + timing + block_size) here because the smoke worker
# spec is fixed to a single shared warm list.
from tests.v1.kv_connector.unit.epic.gpu_smoke import (  # noqa: E402
    _RESULT_PREFIX,
    parse_result_json,
)

# Pure helpers reused from the bench common module (tokenizer loader, chunk hash
# mirror, answer normalisation/containment). No re-implementation.
from benchmarks.epic_reuse.common import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    answer_containment,
    effective_chunk_size,
    epic_chunk_hash,
    load_tokenizer,
)

# CacheBlend musique sample file (list[150]; each has ctxs/question/answers).
_MUSIQUE_URL = (
    "https://raw.githubusercontent.com/YaoJiayi/CacheBlend/"
    "refs/heads/main/inputs/musique_s.json"
)
_DEFAULT_DATA_PATH = "/tmp/musique_s.json"

# Rotating REAL-WORD filler pool (NOT blank space). Mirrors common._WORD_POOL /
# gpu_smoke._FILLER_WORDS: distinct words so a ctx padded up to a chunk boundary
# is ordinary text, not a run of identical blank tokens (blank padding collapses
# dense+sparse to the same trivial output -> a vacuous 1.000 answer match).
_FILLER_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu amber basil cedar dawn ember frost grove harbor ivory "
    "jade kite lotus maple nectar opal pearl quartz river slate timber umbra "
    "violet willow xenon yarrow zephyr anchor beacon canyon dune eagle fjord"
).split()


def _log(msg: str) -> None:
    print(f"[epic-musique] {msg}", flush=True)


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    _log(f"FAIL: {msg}")
    sys.exit(1)


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception as e:  # noqa: BLE001
        _log(f"torch/CUDA probe failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Data loading (CacheBlend musique_s.json)
# ---------------------------------------------------------------------------
@dataclass
class MusiqueSample:
    """One musique example: contexts + question + accepted answers."""

    ctxs: list[str]
    question: str
    answers: list[str]


def parse_musique_records(raw: list) -> list[MusiqueSample]:
    """Parse the decoded musique JSON (list of dicts) into MusiqueSamples (pure).

    Each record is ``{"ctxs": [{"title","text"}...], "question", "answers"}``.
    The ctx string used downstream is ``title + "\\n" + text`` when a title is
    present (matches CacheBlend's passage formatting), else just the text.
    Records missing required keys are skipped (defensive against a truncated
    download). Raises ValueError if NOTHING parses.
    """
    out: list[MusiqueSample] = []
    for rec in raw or []:
        if not isinstance(rec, dict):
            continue
        ctxs_raw = rec.get("ctxs")
        question = rec.get("question")
        answers = rec.get("answers")
        if not ctxs_raw or not question or not answers:
            continue
        ctx_strs: list[str] = []
        for c in ctxs_raw:
            if not isinstance(c, dict):
                continue
            title = (c.get("title") or "").strip()
            text = (c.get("text") or "").strip()
            if not text:
                continue
            ctx_strs.append(f"{title}\n{text}" if title else text)
        if not ctx_strs:
            continue
        out.append(
            MusiqueSample(
                ctxs=ctx_strs,
                question=str(question),
                answers=[str(a) for a in answers if str(a)],
            )
        )
    if not out:
        raise ValueError(
            "no usable musique records parsed (file empty / wrong schema?)"
        )
    return out


def load_musique(path: str, *, download: bool = True) -> list[MusiqueSample]:
    """Load musique samples from ``path``; download from CacheBlend if absent.

    Pure file/JSON parsing on top of ``parse_musique_records``; the network
    fetch is the only side effect and is skipped when the file already exists or
    ``download`` is False (in which case a missing file is a clear error).
    """
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(
                f"musique data not found at {path} and download disabled"
            )
        _log(f"musique data absent at {path}; downloading from {_MUSIQUE_URL}")
        try:
            urllib.request.urlretrieve(_MUSIQUE_URL, path)  # noqa: S310
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"failed to download musique data to {path}: {e}. Fetch it "
                f"manually:\n  curl -L -o {path} {_MUSIQUE_URL}"
            ) from e
    with open(path) as f:
        raw = json.load(f)
    return parse_musique_records(raw)


# ---------------------------------------------------------------------------
# Token-level, chunk-aligned prompt assembly
# ---------------------------------------------------------------------------
def _filler_token_pool(encode) -> list[int]:
    """Encode the real-word filler pool into a flat id list for cycling (pure
    apart from the injected ``encode``). Falls back to ``[0]`` if encoding
    yields nothing (degenerate tokenizer)."""
    ids: list[int] = []
    for w in _FILLER_WORDS:
        ids.extend(encode(" " + w))
    return ids or [0]


@dataclass
class AssembledPrompt:
    """A token-level chunk-aligned musique prompt + the metadata the run/test
    asserts on. ``warm_ctx_ids`` are the per-ctx warm prefill prompts (each is a
    chunk-aligned ctx, optionally with a tiny lead so it is a valid prompt); the
    reuse/full prompt is ``prompt_ids``.
    """

    prompt_ids: list[int]            # [sys][ctx0..ctxK][question] (reuse / full)
    warm_ctx_ids: list[list[int]]    # one warm prefill per ctx (seeds store)
    ctx_offsets: list[int]           # start offset of each ctx in prompt_ids
    ctx_token_lens: list[int]        # padded token length of each ctx
    ctx_chunk_counts: list[int]      # whole chunks per ctx
    ctx_chunk_hashes: list[list[str]]  # hashes of each ctx's chunks (in prompt)
    question_len: int
    chunk_size: int
    real_tokens: int                 # ctx tokens BEFORE padding (content)
    pad_tokens: int                  # filler tokens added for alignment
    # Leading NON-warmed system/instruction segment. Chunk-padded so the FIRST
    # ctx still starts on a chunk boundary, but NEVER warmed into the store. Its
    # presence makes prompt position 0 a NON-hit -> breaks contiguity ->
    # EpicSelection cannot absorb ctx0..K into the contiguous prefix, so EVERY
    # ctx becomes a non_prefix hit and the LegoLink recompute path can engage.
    # See the module-level "LEGOLINK / SYSTEM SEGMENT" note and the task spec.
    sys_len: int = 0                 # padded length of the leading sys segment


def assemble_musique_prompt(
    *,
    ctx_token_lists: list[list[int]],
    question_ids: list[int],
    chunk_size: int,
    filler_ids: list[int],
    warm_lead_ids: list[int] | None = None,
    sys_ids: list[int] | None = None,
    sys_nonce_ids: list[int] | None = None,
) -> AssembledPrompt:
    """Assemble ``[sys][ctx0..ctxK][question]`` with EACH ctx (and the leading
    sys segment) padded to a chunk multiple, returning the reuse/full prompt +
    per-ctx warm prefills (pure).

    Alignment invariants (the whole reason this exists):
      * the OPTIONAL leading ``sys_ids`` (instruction/system text) is padded UP
        to a multiple of ``chunk_size`` and prepended to the reuse/full prompt
        but is **NOT warmed** into the store. Because it lands on prompt
        position 0 and is never saved by the bench's WARM step, the intent is
        that its chunk is a NON-hit -> contiguity is broken at the very first
        chunk -> EpicSelection cannot fold ctx0..K into the contiguous prefix,
        so EVERY ctx becomes a non_prefix hit and the LegoLink recompute path
        can engage. With ``sys_ids`` empty (or None) the legacy behaviour is
        preserved: ctx0 starts at position 0 and the whole ctx run is a single
        contiguous prefix (link INERT).

      * RESIDUAL-BUG FIX (sys nonce): "never warmed by the bench" is NOT
        sufficient to keep the sys chunk a non-hit. The connector's SAVE path
        (epic_connector.py build_connector_meta) saves every whole chunk that is
        fully inside M (the freshly computed tokens). For a REUSE request the
        sys segment IS entirely in M (it is recomputed, never loaded), so the
        very first reuse request SAVES the sys chunk into the per-engine store.
        Because one mode subprocess processes all N samples sequentially with a
        single shared store, sample 0's reuse saves sys -> sample 1+ sees sys as
        a HIT at position 0 -> the ctx run folds back into the contiguous prefix
        -> ``non_prefix_hits == 0`` -> LegoLink INERT again (exactly the case the
        --warmup-discard measured samples land in). The fix is to make the sys
        segment UNIQUE PER SAMPLE by prepending ``sys_nonce_ids`` (e.g. the
        sample index encoded as tokens) to the sys content BEFORE padding. The
        nonce lands in the FIRST sys chunk, so that chunk's content hash differs
        per sample and can NEVER collide with a sys chunk an EARLIER sample
        saved -> position 0 is ALWAYS a non-hit -> ctx0..K stay non-prefix ->
        LegoLink engages on every measured sample. The nonce does NOT touch the
        ctx ids (it is confined to the sys region), so ctx chunk hashes and
        byte-identity vs. the warm prefills are UNCHANGED.

        Why this is legitimate for the probe (and a known connector limitation
        we are deliberately working around): the sys instruction is recomputed
        on every request (always in M), so in real serving a SHARED sys would be
        a legitimate reusable prefix. But EpicSelection absorbs any contiguous
        run of hits starting at position 0 into the prefix, so if sys is a hit it
        drags ctx0..K in with it. To MEASURE non-prefix doc reuse we must keep
        sys intentionally non-hit; the per-sample nonce is the simplest way to
        guarantee that without changing the connector. Fairness is preserved
        because the SAME sample's full and reuse modes both use this prompt_ids
        (same nonce), so the comparison is apples-to-apples within a sample.
      * each ctx is padded UP to a multiple of ``chunk_size`` with cycled
        real-word filler, so it occupies whole hashable chunks and the NEXT ctx
        starts on a chunk boundary;
      * the SAME padded ctx id-slice is used in the warm prefill and in the
        reuse prompt -> byte-identical -> EPIC content hash collides -> a real
        non-prefix hit for every ctx chunk;
      * the question is appended AFTER all ctxs as a trailing (partial) chunk --
        never hashed/saved, matching the connector's whole-chunks-only rule.

    Because the sys segment is padded to a chunk multiple, every ctx offset is
    SHIFTED by ``sys_len`` yet remains chunk-aligned, so the per-ctx chunk
    hashes (which depend only on the ctx ids, not their absolute offset) are
    UNCHANGED -> the warm-side hashes still collide with the reuse-side hashes.

    The warm prefill for a ctx is ``warm_lead_ids + padded_ctx``; ``warm_lead``
    is padded to a chunk multiple too so the ctx still starts on a chunk
    boundary in the warm prompt (default lead is empty -> ctx starts at 0). The
    sys segment is deliberately EXCLUDED from the warm prefills (warming it would
    re-introduce the position-0 hit and re-collapse everything into the prefix).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    pool = list(filler_ids) if filler_ids else [0]

    def _fill(n: int) -> list[int]:
        return [pool[i % len(pool)] for i in range(n)]

    def _pad_to_chunk(ids: list[int]) -> tuple[list[int], int, int]:
        """Pad ids up to a chunk multiple. Returns (padded, real, pad)."""
        real = len(ids)
        rem = real % chunk_size
        pad = (chunk_size - rem) % chunk_size
        if real == 0:
            pad = chunk_size  # never emit a zero-length ctx
        return list(ids) + _fill(pad), real, pad

    # Warm lead: pad to a chunk multiple so the ctx begins chunk-aligned in the
    # warm prompt (empty lead -> begins at 0, also aligned).
    lead = list(warm_lead_ids or [])
    if lead:
        lead, _, _ = _pad_to_chunk(lead)

    # Leading system/instruction segment: pad to a chunk multiple so ctx0 stays
    # chunk-aligned in the reuse/full prompt, but DO NOT warm it (see docstring).
    # The per-sample ``sys_nonce_ids`` are prepended to the sys content BEFORE
    # padding so they land in the FIRST sys chunk -> that chunk's content hash is
    # unique per sample -> it can never collide with a sys chunk an earlier
    # sample's reuse request SAVED into the shared store -> position 0 stays a
    # non-hit (see the docstring's RESIDUAL-BUG FIX note). The nonce is confined
    # to the sys region, so ctx ids / hashes / warm prefills are unaffected.
    sys_seg: list[int] = []
    raw_sys = list(sys_nonce_ids or []) + list(sys_ids or [])
    if raw_sys:
        sys_seg, _, _ = _pad_to_chunk(raw_sys)

    prompt_ids: list[int] = list(sys_seg)  # sys occupies prompt positions [0, sys_len)
    warm_ctx_ids: list[list[int]] = []
    ctx_offsets: list[int] = []
    ctx_token_lens: list[int] = []
    ctx_chunk_counts: list[int] = []
    ctx_chunk_hashes: list[list[str]] = []
    total_real = 0
    total_pad = 0

    for ctx_ids in ctx_token_lists:
        padded, real, pad = _pad_to_chunk(ctx_ids)
        total_real += real
        total_pad += pad
        n_chunks = len(padded) // chunk_size
        # Hash each chunk of the padded ctx (these are the hashes EPIC matches).
        hashes = [
            epic_chunk_hash(padded[c * chunk_size:(c + 1) * chunk_size])
            for c in range(n_chunks)
        ]
        ctx_offsets.append(len(prompt_ids))
        ctx_token_lens.append(len(padded))
        ctx_chunk_counts.append(n_chunks)
        ctx_chunk_hashes.append(hashes)
        prompt_ids.extend(padded)
        # Warm prefill: lead (chunk-aligned) + the SAME padded ctx ids.
        warm_ctx_ids.append(list(lead) + list(padded))

    prompt_ids.extend(question_ids)

    return AssembledPrompt(
        prompt_ids=prompt_ids,
        warm_ctx_ids=warm_ctx_ids,
        ctx_offsets=ctx_offsets,
        ctx_token_lens=ctx_token_lens,
        ctx_chunk_counts=ctx_chunk_counts,
        ctx_chunk_hashes=ctx_chunk_hashes,
        question_len=len(question_ids),
        chunk_size=chunk_size,
        real_tokens=total_real,
        pad_tokens=total_pad,
        sys_len=len(sys_seg),
    )


# ---------------------------------------------------------------------------
# REALISTIC RAG REUSE SCENARIOS (reorder / insert / same)
# ---------------------------------------------------------------------------
# These exercise EPIC the way a real RAG server hits it: a FIRST ("warm")
# request full-prefills ``[system][a][b][c]`` so the connector stores the sys,
# a, b, c chunks by content-hash, then a SECOND ("measured") request reuses
# those chunks under one of three layouts. The point is to split the two
# independent reuse mechanisms so a failure localises to ONE of them:
#
#   * ``reorder`` -> ``[system][b][a][c][query]``. b, a, c are the SAME chunks
#     warmed, but at DIFFERENT positions (b now sits where a was, etc.). The sys
#     chunk is ALSO warmed (it is identical in warm and measured -- no nonce),
#     so EpicSelection walks sys, b, a, c as a CONTIGUOUS run of hits from
#     position 0 -> they ALL fold into the prefix -> ``non_prefix == 0`` ->
#     the sparse/LegoLink branch is NOT taken (``sparse_match == 0`` expected).
#     But because every chunk moved, the connector's prefix-load path applies a
#     NON-ZERO PIC delta re-rotary (stored old_pos != new prompt position). So
#     reorder is the FIRST real-data test of PIC re-rotary + load + scatter on
#     their own, with the sparse-forward path deliberately untouched.
#
#   * ``insert`` -> ``[system][d][a][c][query]``. ``d`` is a DIFFERENT,
#     HELD-OUT context of the SAME sample that was NEVER warmed, spliced in
#     right after sys. Its chunk(s) are a NON-hit, so contiguity breaks at the
#     first ctx slot -> ``a`` (now offset by |sys|+|d|) and ``c`` are no longer
#     contiguous with the warmed run -> they become NON-PREFIX hits -> the
#     sparse-forward + LegoLink recompute path engages (``non_prefix > 0``,
#     ``sparse_branch == True`` expected). This isolates the sparse path.
#
#   * ``same`` -> ``[system][a][b][c][query]``, same order as warm. Kept as the
#     legacy control. Here sys IS warmed and a, b, c sit at the SAME positions
#     they were warmed at, so without help the whole run folds into the prefix
#     AND the PIC delta is zero -> nothing interesting. To keep ``same`` doing
#     what the pre-scenario script did (force non-prefix hits via a broken
#     position-0), ``same`` retains the per-sample sys NONCE so sys is a non-hit
#     and a, b, c become non-prefix hits. reorder/insert do NOT use the nonce:
#     the reorder/insert layout itself supplies the contiguity break (insert) or
#     the non-zero PIC delta (reorder), and a warmed sys is REQUIRED for reorder
#     to fold into the (PIC-rotated) prefix.
#
# HOW THIS SEPARATES THE link=255 COLLAPSE BUG (PIC vs sparse-forward):
#   reorder reuse-only correct  => PIC re-rotary + load + scatter are SOUND on
#                                  real data (first PIC validation passes).
#   reorder fine BUT insert-epic broken => the bug lives in the SPARSE-FORWARD /
#                                  LegoLink recompute path, NOT in PIC.
#   reorder ALSO broken         => the bug is in PIC / load itself.

# The three realistic reuse scenarios. ``same`` is the legacy nonce control.
SCENARIOS = ("same", "reorder", "insert")


@dataclass
class ScenarioPrompt:
    """A warm + measured prompt pair for ONE realistic RAG reuse scenario.

    ``warm_ids`` is the SINGLE first-request prompt ``[system][a][b][c]`` that
    is full-prefilled so the connector STORES the sys/a/b/c chunks. ``prompt_ids``
    is the measured second-request prompt for the chosen scenario. The remaining
    fields are diagnostics the run/test asserts on (chunk offsets/hashes per
    segment, which segment was held-out, expected selection outcome).
    """

    scenario: str
    warm_ids: list[int]              # [sys][a][b][c] -- the warm full prefill
    prompt_ids: list[int]            # measured prompt for the scenario
    # Per-segment layout of the MEASURED prompt, in prompt order. Each entry is
    # (label, offset, length, [chunk_hash...]). ``label`` is "sys"/"a"/"b"/"c"/
    # "d"/"query". sys + ctx segments are chunk-aligned; query is a trailing
    # partial chunk (never hashed).
    seg_labels: list[str]
    seg_offsets: list[int]
    seg_lens: list[int]
    seg_chunk_hashes: list[list[str]]
    # Hashes the WARM prefill stores (sys + a + b + c chunks). For ``same`` the
    # warm sys carries the SAME nonce-free sys text, so its hash is recorded but
    # the measured prompt's sys (nonced) will NOT collide with it.
    warm_stored_hashes: set
    sys_len: int
    chunk_size: int
    question_len: int
    held_out_label: str | None       # "d" for insert, else None
    real_tokens: int
    pad_tokens: int
    # Static prediction (pure mirror of EpicSelection) for the MEASURED prompt:
    # which segments fold into the contiguous prefix vs become non-prefix hits.
    expect_prefix_extent: int
    expect_non_prefix_offsets: list[int]

    @property
    def expect_sparse(self) -> bool:
        """Whether the sparse-forward / LegoLink branch is expected to engage
        (i.e. there is at least one non-prefix hit)."""
        return len(self.expect_non_prefix_offsets) > 0


def _mirror_select_offsets(
    chunks: list[tuple[int, int, str]], store_hashes: set
) -> tuple[int, list[int]]:
    """Pure mirror of ``EpicSelection.select``'s prefix/non-prefix walk.

    ``chunks`` = ordered ``[(start, length, hash)]`` for the measured prompt;
    ``store_hashes`` = hashes present in the store after the warm prefill.
    Returns ``(prefix_extent, non_prefix_offsets)``. Used to PREDICT (CPU, no
    torch) whether a scenario will fold into the prefix (reorder/same-via-nonce)
    or expose non-prefix hits (insert) -- the load-bearing diagnostic.
    """
    prefix_extent = 0
    contiguous = True
    non_prefix: list[int] = []
    for start, length, h in chunks:
        hit = h in store_hashes
        if contiguous and start == prefix_extent and hit:
            prefix_extent += length
        else:
            contiguous = False
            if hit:
                non_prefix.append(start)
    return prefix_extent, non_prefix


def assemble_scenario_prompt(
    *,
    scenario: str,
    ctx_token_lists: list[list[int]],
    held_out_ctx_ids: list[int] | None,
    question_ids: list[int],
    chunk_size: int,
    filler_ids: list[int],
    sys_ids: list[int],
    sys_nonce_ids: list[int] | None = None,
) -> ScenarioPrompt:
    """Assemble a warm/measured pair for a realistic RAG reuse ``scenario``.

    ``ctx_token_lists`` are the THREE warmed contexts (a, b, c) -- the function
    requires exactly three so the reorder/insert layouts are unambiguous.
    ``held_out_ctx_ids`` is the FOURTH, never-warmed context ``d`` used ONLY by
    ``insert``. Every segment (sys, a, b, c, d) is padded UP to a ``chunk_size``
    multiple with cycled real-word filler so it occupies whole hashable chunks
    and the next segment starts on a chunk boundary -- identical alignment rule
    to ``assemble_musique_prompt``.

    Warm prompt (ALL scenarios): ``[sys][a][b][c]`` full-prefilled -> stores the
    sys, a, b, c chunks (the connector saves every whole chunk inside M, and a
    reuse-less first prefill computes everything). The warm sys is NONCE-FREE.

    Measured prompt by scenario:
      * ``same``    : ``[sys_nonced][a][b][c][query]``. The measured sys gets a
                      per-sample NONCE so it is a NON-hit -> contiguity breaks at
                      position 0 -> a, b, c all become non-prefix hits (legacy
                      LegoLink-engagement behaviour; PIC delta is ZERO because
                      a, b, c sit where they were warmed).
      * ``reorder`` : ``[sys][b][a][c][query]``. sys is NONCE-FREE so it HITS the
                      warmed sys -> sys, b, a, c form a contiguous prefix hit ->
                      ``non_prefix == 0`` (no sparse branch), but b/a moved so the
                      prefix-load PIC delta is NON-ZERO. Validates PIC alone.
      * ``insert``  : ``[sys][d][a][c][query]``. sys HITS; d is the held-out ctx
                      (NON-hit) spliced after sys -> breaks contiguity -> a, c
                      become non-prefix hits -> sparse branch engages.

    a, b, c are BYTE-IDENTICAL between warm and measured (same padded ids), so
    their content hashes collide regardless of position. Returns a ScenarioPrompt
    with the measured layout, the warm-stored hash set, and a STATIC prediction
    of the prefix/non-prefix split (pure mirror of EpicSelection).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; valid: {SCENARIOS}")
    if len(ctx_token_lists) != 3:
        raise ValueError(
            f"scenario assembly needs exactly 3 warmed ctxs (a,b,c); got "
            f"{len(ctx_token_lists)}"
        )
    if scenario == "insert" and not held_out_ctx_ids:
        raise ValueError("scenario 'insert' requires a held-out ctx 'd'")

    pool = list(filler_ids) if filler_ids else [0]

    def _fill(n: int) -> list[int]:
        return [pool[i % len(pool)] for i in range(n)]

    def _pad_to_chunk(ids: list[int]) -> tuple[list[int], int, int]:
        real = len(ids)
        rem = real % chunk_size
        pad = (chunk_size - rem) % chunk_size
        if real == 0:
            pad = chunk_size
        return list(ids) + _fill(pad), real, pad

    def _hashes(padded: list[int]) -> list[str]:
        n = len(padded) // chunk_size
        return [
            epic_chunk_hash(padded[c * chunk_size:(c + 1) * chunk_size])
            for c in range(n)
        ]

    # --- pad every segment to a chunk multiple (byte-stable per segment) ---
    a_pad, a_real, a_p = _pad_to_chunk(ctx_token_lists[0])
    b_pad, b_real, b_p = _pad_to_chunk(ctx_token_lists[1])
    c_pad, c_real, c_p = _pad_to_chunk(ctx_token_lists[2])
    d_pad: list[int] = []
    d_real = d_p = 0
    if held_out_ctx_ids:
        d_pad, d_real, d_p = _pad_to_chunk(held_out_ctx_ids)

    # Warm sys: NONCE-FREE (so reorder's measured sys collides with it).
    warm_sys_pad, _, warm_sys_p = _pad_to_chunk(list(sys_ids))
    # Measured sys: nonced ONLY for ``same`` (force a position-0 non-hit). For
    # reorder/insert the measured sys is nonce-free so it HITS the warmed sys.
    if scenario == "same":
        meas_sys_raw = list(sys_nonce_ids or []) + list(sys_ids)
    else:
        meas_sys_raw = list(sys_ids)
    meas_sys_pad, _, meas_sys_p = _pad_to_chunk(meas_sys_raw)

    a_h, b_h, c_h = _hashes(a_pad), _hashes(b_pad), _hashes(c_pad)
    d_h = _hashes(d_pad) if d_pad else []
    warm_sys_h = _hashes(warm_sys_pad)

    # --- WARM prompt: [sys][a][b][c] (stores sys,a,b,c chunks) ---
    warm_ids = list(warm_sys_pad) + a_pad + b_pad + c_pad
    warm_stored: set = set(warm_sys_h) | set(a_h) | set(b_h) | set(c_h)

    # --- MEASURED prompt assembly per scenario ---
    # Each entry: (label, padded_ids, chunk_hashes).
    if scenario == "same":
        ordered = [("a", a_pad, a_h), ("b", b_pad, b_h), ("c", c_pad, c_h)]
        held_out_label = None
    elif scenario == "reorder":
        ordered = [("b", b_pad, b_h), ("a", a_pad, a_h), ("c", c_pad, c_h)]
        held_out_label = None
    else:  # insert
        ordered = [("d", d_pad, d_h), ("a", a_pad, a_h), ("c", c_pad, c_h)]
        held_out_label = "d"

    prompt_ids: list[int] = list(meas_sys_pad)
    seg_labels: list[str] = ["sys"]
    seg_offsets: list[int] = [0]
    seg_lens: list[int] = [len(meas_sys_pad)]
    seg_chunk_hashes: list[list[str]] = [_hashes(meas_sys_pad)]
    for label, padded, hashes in ordered:
        seg_labels.append(label)
        seg_offsets.append(len(prompt_ids))
        seg_lens.append(len(padded))
        seg_chunk_hashes.append(hashes)
        prompt_ids.extend(padded)
    q_off = len(prompt_ids)
    prompt_ids.extend(question_ids)
    seg_labels.append("query")
    seg_offsets.append(q_off)
    seg_lens.append(len(question_ids))
    seg_chunk_hashes.append([])

    # --- static prediction of the prefix/non-prefix split (CPU mirror) ---
    chunks: list[tuple[int, int, str]] = []
    for off, ln, hashes in zip(seg_offsets, seg_lens, seg_chunk_hashes):
        for c, h in enumerate(hashes):
            chunks.append((off + c * chunk_size, chunk_size, h))
    expect_prefix, expect_nonpref = _mirror_select_offsets(chunks, warm_stored)

    # padding accounting (content vs filler) over the MEASURED prompt ctxs.
    seg_reals = {"a": a_real, "b": b_real, "c": c_real, "d": d_real}
    seg_pads = {"a": a_p, "b": b_p, "c": c_p, "d": d_p}
    real_tokens = sum(seg_reals[lbl] for lbl in seg_labels if lbl in seg_reals)
    pad_tokens = sum(seg_pads[lbl] for lbl in seg_labels if lbl in seg_pads)

    return ScenarioPrompt(
        scenario=scenario,
        warm_ids=warm_ids,
        prompt_ids=prompt_ids,
        seg_labels=seg_labels,
        seg_offsets=seg_offsets,
        seg_lens=seg_lens,
        seg_chunk_hashes=seg_chunk_hashes,
        warm_stored_hashes=warm_stored,
        sys_len=len(meas_sys_pad),
        chunk_size=chunk_size,
        question_len=len(question_ids),
        held_out_label=held_out_label,
        real_tokens=real_tokens,
        pad_tokens=pad_tokens,
        expect_prefix_extent=expect_prefix,
        expect_non_prefix_offsets=expect_nonpref,
    )


# ---------------------------------------------------------------------------
# Mode -> kv-transfer config + engine kwargs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mode:
    """A measured engine mode. ``link`` is None for ``full`` (no connector)."""

    label: str
    sparse: bool
    link: int | None  # epic_link_tokens (None for full)

    @property
    def is_reuse(self) -> bool:
        return self.sparse


def mode_full() -> Mode:
    return Mode(label="full", sparse=False, link=None)


def mode_reuse_only() -> Mode:
    return Mode(label="reuse-only", sparse=True, link=0)


def mode_epic(k: int) -> Mode:
    if k < 0:
        raise ValueError(f"link k must be >= 0, got {k}")
    return Mode(label=f"epic@{k}", sparse=True, link=k)


def kv_config_for_mode(mode: Mode, *, chunk_size: int) -> dict | None:
    """KV-transfer config dict for a mode.

    full       -> None (no connector; sparse OFF, prefix caching OFF -> a true
                  full prefill ground truth).
    reuse/epic -> EpicConnector, sparse-forward + fusion mask ON, the requested
                  link, and the SAME chunk_size used for assembly (so the
                  connector's chunk boundaries match ours).
    """
    if not mode.sparse:
        return None
    extra: dict = {
        "epic_chunk_size": int(chunk_size),
        "epic_link_tokens": int(mode.link or 0),
        "epic_sparse_forward": True,
        "epic_fusion_mask": True,
        # In-band engagement counters (assert sparse actually fired) -- requires
        # the in-process engine so scheduler + worker share the class state.
        "epic_debug_counters": True,
        # Worker reads back scattered dst slots vs the store (per-layer
        # allclose + max-abs-diff on the first loaded chunk) -- the decisive
        # signal for whether the pure-load B KV is faithful or garbage.
        "epic_debug_check_load": True,
    }
    return {
        "kv_connector": "EpicConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra,
    }


# ---------------------------------------------------------------------------
# Worker spec (one engine = one mode, processes ALL samples sequentially)
# ---------------------------------------------------------------------------
def build_musique_worker_spec(
    *,
    mode: Mode,
    model: str,
    chunk_size: int,
    samples: list[dict],
    max_tokens: int,
    block_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> dict:
    """JSON-serialisable spec for ONE mode's engine run over N samples (pure).

    Each entry of ``samples`` is ``{"warm": [[ids]...], "prompt": [ids],
    "answers": [str]}``. For ``reuse-only`` / ``epic@k`` the worker first runs
    every warm prefill (seeds the store) for a sample, then times that sample's
    reuse prompt; for ``full`` the warm lists are ignored (no connector).

    sparse modes set ``in_process`` so the connector's class-level engagement
    counters are visible to the worker (scheduler + worker share state); ``full``
    does not need it but it is harmless.
    """
    return {
        "kind": "musique",
        "mode_label": mode.label,
        "sparse": bool(mode.sparse),
        "model": model,
        "chunk_size": int(chunk_size),
        "kv_config": kv_config_for_mode(mode, chunk_size=chunk_size),
        "samples": [
            {
                "warm": [[int(t) for t in w] for w in s["warm"]],
                "prompt": [int(t) for t in s["prompt"]],
                "answers": list(s["answers"]),
            }
            for s in samples
        ],
        "max_tokens": int(max_tokens),
        "block_size": int(block_size),
        "max_model_len": int(max_model_len),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        # sparse needs the in-process engine for the in-band counters.
        "in_process": bool(mode.sparse),
        "read_counters": bool(mode.sparse),
    }


def serialize_spec(spec: dict) -> str:
    return json.dumps(spec, separators=(",", ":"))


def parse_spec(s: str) -> dict:
    return json.loads(s)


def _emit_result(result: dict) -> None:
    print(f"{_RESULT_PREFIX} {json.dumps(result)}", flush=True)


# ---------------------------------------------------------------------------
# Aggregation (pure, CPU-testable)
# ---------------------------------------------------------------------------
def speedup(full_ms: float, mode_ms: float) -> float:
    """``full_ms / mode_ms`` -- prefill speedup of a mode vs the full baseline.

    >1 means the mode is FASTER than full. Guards a zero/non-positive mode time
    (returns inf, "infinitely faster") and a non-positive full time (returns
    0.0, undefined baseline). Pure."""
    if mode_ms <= 0.0:
        return float("inf")
    if full_ms <= 0.0:
        return 0.0
    return full_ms / mode_ms


def mean(xs: list[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


@dataclass
class ModeAggregate:
    label: str
    n: int
    answer_hits: int
    mean_prefill_ms: float
    speedup_vs_full: float = 0.0

    @property
    def answer_hit_rate(self) -> float:
        return (self.answer_hits / self.n) if self.n else 0.0


def aggregate_mode(
    label: str, per_sample: list[dict], *, warmup_discard: bool
) -> ModeAggregate:
    """Aggregate one mode's per-sample results into hit-rate + mean prefill.

    ``per_sample`` rows are ``{"answer_hit": bool, "prefill_ms": float, ...}``.
    ``warmup_discard`` drops the FIRST row (cold compile/autotune) from BOTH the
    hit count and the timing mean. Pure."""
    rows = per_sample[1:] if (warmup_discard and len(per_sample) > 1) else per_sample
    n = len(rows)
    hits = sum(1 for r in rows if r.get("answer_hit"))
    ms = mean([float(r["prefill_ms"]) for r in rows if r.get("prefill_ms") is not None])
    return ModeAggregate(label=label, n=n, answer_hits=hits, mean_prefill_ms=ms)


def fill_speedups(aggs: list[ModeAggregate]) -> list[ModeAggregate]:
    """Set ``speedup_vs_full`` on each aggregate using the ``full`` mode's mean
    prefill as the baseline (full speedup == 1.0). If there is no ``full`` row,
    speedups stay 0.0. Pure (mutates + returns the list)."""
    full = next((a for a in aggs if a.label == "full"), None)
    base = full.mean_prefill_ms if full else 0.0
    for a in aggs:
        a.speedup_vs_full = speedup(base, a.mean_prefill_ms) if base > 0 else 0.0
    return aggs


# ---------------------------------------------------------------------------
# Worker (runs inside the per-mode subprocess; needs a GPU)
# ---------------------------------------------------------------------------
def run_musique_worker(spec: dict) -> dict:
    """Build ONE engine for ``spec``'s mode, process all samples, return result.

    Per sample (sparse modes): run every warm prefill (discarded; seeds store),
    then time a ``max_tokens=1`` reuse request (prefill wall-clock) and run a
    ``max_tokens`` request for the answer text. ``full`` skips the warm step. On
    any error returns ``{"ok": False, "error": ...}`` so the parent gets a clean
    structured failure. Device memory is freed by PROCESS EXIT.
    """
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    label = spec.get("mode_label", "?")
    try:
        if spec.get("in_process"):
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        kwargs: dict = dict(
            model=spec["model"],
            enforce_eager=True,
            block_size=int(spec["block_size"]),
            max_model_len=int(spec["max_model_len"]),
            gpu_memory_utilization=float(spec["gpu_memory_utilization"]),
            # FLEX_ATTENTION for ALL modes (fairness + EPIC hard requirement).
            attention_backend="FLEX_ATTENTION",
        )
        sparse = bool(spec.get("sparse"))
        if sparse:
            kwargs["kv_transfer_config"] = spec["kv_config"]
            kwargs["enable_prefix_caching"] = True
        else:
            # full = true ground truth: no reuse of anything.
            kwargs["enable_prefix_caching"] = False
        llm = LLM(**kwargs)

        time_params = SamplingParams(temperature=0.0, max_tokens=1)
        gen_params = SamplingParams(
            temperature=0.0, max_tokens=int(spec["max_tokens"])
        )

        per_sample: list[dict] = []
        for s in spec["samples"]:
            # Warm: seed each ctx's chunks into the per-engine store. Discarded
            # outputs; excluded from timing. Same engine/process as the reuse
            # request so the store is visible.
            if sparse:
                for w in s["warm"]:
                    llm.generate(
                        [TokensPrompt(prompt_token_ids=list(w))], time_params
                    )

            prompt = list(s["prompt"])
            # Timing: a single-token greedy request's wall-clock == prefill TTFT
            # for one short request. Only the MEASURED request is timed.
            t0 = time.perf_counter()
            llm.generate([TokensPrompt(prompt_token_ids=prompt)], time_params)
            prefill_ms = (time.perf_counter() - t0) * 1000.0

            # Answer text: a separate greedy generation (max_tokens tokens).
            outs = llm.generate(
                [TokensPrompt(prompt_token_ids=prompt)], gen_params
            )
            text = outs[0].outputs[0].text
            hit = answer_containment(text, s["answers"])
            per_sample.append(
                {
                    "answer_hit": bool(hit),
                    "prefill_ms": float(prefill_ms),
                    "text": text,
                    "answers": s["answers"],
                }
            )

        result: dict = {
            "ok": True,
            "mode_label": label,
            "per_sample": per_sample,
        }
        if spec.get("read_counters"):
            from vllm.distributed.kv_transfer.kv_connector.v1.epic.epic_connector import (  # noqa: E501
                EpicConnector,
            )

            result["counters"] = dict(EpicConnector.debug_counters)
            # Per-request SELECTION diagnostics (prefix_extent / non_prefix
            # count+offsets / sparse_branch). Lets the parent flag the silent
            # "everything fell into the prefix -> LINK INERT" failure mode.
            result["selection"] = [
                dict(e) for e in EpicConnector.debug_selection
            ]
        return result
    except Exception as e:  # noqa: BLE001
        import traceback

        return {
            "ok": False,
            "mode_label": label,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def run_mode_subprocess(spec: dict) -> dict:
    """Parent side: run one mode's engine in a FRESH subprocess, parse result.

    Reuses gpu_smoke's RESULT_JSON contract (``parse_result_json``). Subprocess
    exit returns ALL device memory (no reliance on Python cleanup), so the four
    modes never accumulate VRAM. stderr is inherited (engine + EPIC logs stream
    live); only stdout is parsed.
    """
    import subprocess
    import tempfile

    # Write the spec to a temp file and pass its PATH (musique contexts are too
    # large to fit in argv -> "Argument list too long").
    fd, spec_path = tempfile.mkstemp(prefix="epic_musique_spec_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialize_spec(spec))
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--_worker-spec-file",
            spec_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    finally:
        try:
            os.unlink(spec_path)
        except OSError:
            pass
    try:
        return parse_result_json(proc.stdout)
    except ValueError as e:
        raise RuntimeError(
            f"worker (mode={spec.get('mode_label')}) produced no RESULT_JSON "
            f"(exit code {proc.returncode}): {e}"
        ) from None


# ---------------------------------------------------------------------------
# Parent orchestration
# ---------------------------------------------------------------------------
def _encode(tok):
    def enc(text: str) -> list[int]:
        # add_special_tokens=False: we control boundaries; a stray BOS inside a
        # ctx would break byte-identity vs. the same ctx in the warm prompt.
        return list(tok.encode(text, add_special_tokens=False))

    return enc


@dataclass
class PreparedSample:
    """A musique sample assembled into chunk-aligned token prompts + the gold
    answers and the assembly metadata (for logging)."""

    assembled: AssembledPrompt
    answers: list[str]
    question: str


def prepare_samples(
    samples: list[MusiqueSample],
    *,
    tok,
    chunk_size: int,
    ctx_per_sample: int,
    system_text: str | None = None,
) -> list[PreparedSample]:
    """Tokenize + chunk-align N musique samples (parent side, CPU-only).

    Each ctx is encoded independently (so its ids are stable) and the prompt is
    assembled at the token level with per-ctx chunk padding. When
    ``system_text`` is non-empty it is tokenized once and prepended (chunk-padded
    but NOT warmed) to EVERY sample's reuse/full prompt -- this is the fix that
    breaks position-0 contiguity so the LegoLink recompute path can engage (see
    ``assemble_musique_prompt``). To keep that position-0 chunk a non-hit DESPITE
    the connector saving the sys chunk on the first reuse request (one mode
    subprocess processes all N samples against one shared store), each sample's
    sys segment is made UNIQUE with a per-sample NONCE (the sample index encoded
    as tokens) prepended to the sys content; see ``assemble_musique_prompt``'s
    RESIDUAL-BUG FIX note. The nonce is confined to the sys region so ctx hashes
    and warm prefills are identical across samples that share a ctx. Returns the
    prepared prompts + answers; logging of padding waste is the caller's job.
    """
    enc = _encode(tok)
    # Tokenize the leading system/instruction segment ONCE (same TEXT for every
    # sample). Empty/None -> no sys (legacy all-prefix behaviour, --no-system).
    sys_ids = enc(system_text) if system_text else []
    prepared: list[PreparedSample] = []
    for idx, s in enumerate(samples):
        ctxs = s.ctxs[:ctx_per_sample]
        ctx_token_lists = [enc(c) for c in ctxs]
        # Per-sample sys NONCE: a sample-unique marker prepended to the sys
        # content so the FIRST sys chunk's content hash differs per sample. Only
        # emitted when there IS a sys segment (no sys -> no position-0 chunk to
        # protect). Encoded from the sample index: deterministic, unique within a
        # run, and -- crucially -- a value the store cannot already hold (sample
        # i's nonce chunk differs from every j<i that was saved). Same sample ->
        # same nonce, so a sample's full and reuse modes share the exact same
        # prompt_ids (fair comparison), and the same idx across the per-mode
        # subprocesses yields the same nonce (consistent within-sample pairing).
        sys_nonce_ids = enc(f"[sample-{idx} sys]\n") if sys_ids else []
        # Question with a small instruction tail so an instruct model answers.
        q_text = (
            f"\n\nQuestion: {s.question}\nAnswer the question using ONLY the "
            "passages above. Answer:"
        )
        q_ids = enc(q_text)
        assembled = assemble_musique_prompt(
            ctx_token_lists=ctx_token_lists,
            question_ids=q_ids,
            chunk_size=chunk_size,
            filler_ids=_filler_token_pool(enc),
            sys_ids=sys_ids,
            sys_nonce_ids=sys_nonce_ids,
        )
        prepared.append(
            PreparedSample(
                assembled=assembled, answers=s.answers, question=s.question
            )
        )
    return prepared


@dataclass
class PreparedScenarioSample:
    """A musique sample assembled for a realistic RAG reuse scenario.

    Unlike ``PreparedSample`` (one warm prefill per ctx, legacy probe), the
    scenario path warms a SINGLE ``[sys][a][b][c]`` prompt and measures the
    scenario-specific reordered/inserted layout. ``assembled`` carries the
    static prefix/non-prefix prediction the run logs and the test asserts on.
    """

    assembled: ScenarioPrompt
    answers: list[str]
    question: str


def prepare_scenario_samples(
    samples: list[MusiqueSample],
    *,
    tok,
    chunk_size: int,
    scenario: str,
    system_text: str,
) -> list[PreparedScenarioSample]:
    """Tokenize + chunk-align N samples for a realistic RAG reuse ``scenario``.

    For every sample we pick the FIRST three contexts as the warmed a, b, c and
    (for ``insert``) the FOURTH context as the held-out d. The warm prompt is
    ``[sys][a][b][c]`` (full-prefilled -> stores sys/a/b/c). The measured prompt
    is the scenario layout (see ``assemble_scenario_prompt``). ``system_text`` is
    REQUIRED (a leading RAG instruction is the whole point); it is nonce-free in
    warm and in reorder/insert (so it HITS), and per-sample nonced ONLY in
    ``same`` (legacy position-0 break). Samples without enough contexts for the
    scenario (>=3, or >=4 for insert) are SKIPPED with a log line.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; valid: {SCENARIOS}")
    if not system_text:
        raise ValueError("scenario mode requires a non-empty --system-prompt")
    enc = _encode(tok)
    sys_ids = enc(system_text)
    need = 4 if scenario == "insert" else 3
    prepared: list[PreparedScenarioSample] = []
    for idx, s in enumerate(samples):
        if len(s.ctxs) < need:
            _log(
                f"skip sample {idx}: scenario {scenario!r} needs >={need} ctxs, "
                f"has {len(s.ctxs)}"
            )
            continue
        abc = [enc(c) for c in s.ctxs[:3]]
        held_out = enc(s.ctxs[3]) if scenario == "insert" else None
        # Nonce only matters for ``same`` (the legacy position-0 break). reorder/
        # insert keep sys nonce-free so the measured sys HITS the warmed sys.
        sys_nonce_ids = enc(f"[sample-{idx} sys]\n") if scenario == "same" else None
        q_text = (
            f"\n\nQuestion: {s.question}\nAnswer the question using ONLY the "
            "passages above. Answer:"
        )
        q_ids = enc(q_text)
        assembled = assemble_scenario_prompt(
            scenario=scenario,
            ctx_token_lists=abc,
            held_out_ctx_ids=held_out,
            question_ids=q_ids,
            chunk_size=chunk_size,
            filler_ids=_filler_token_pool(enc),
            sys_ids=sys_ids,
            sys_nonce_ids=sys_nonce_ids,
        )
        prepared.append(
            PreparedScenarioSample(
                assembled=assembled, answers=s.answers, question=s.question
            )
        )
    if not prepared:
        _fail(
            f"no samples have enough contexts for scenario {scenario!r} "
            f"(need >={need}); raise --num-samples or pick a different scenario"
        )
    return prepared


def _samples_for_spec(prepared: list[PreparedSample]) -> list[dict]:
    return [
        {
            "warm": p.assembled.warm_ctx_ids,
            "prompt": p.assembled.prompt_ids,
            "answers": p.answers,
        }
        for p in prepared
    ]


def _scenario_samples_for_spec(
    prepared: list[PreparedScenarioSample],
) -> list[dict]:
    """Worker-spec sample rows for the scenario path.

    The warm side is a SINGLE prompt ``[sys][a][b][c]`` (wrapped in a one-element
    ``warm`` list so the worker's existing per-warm loop seeds it), and the
    measured prompt is the scenario layout. Same on-the-wire schema as
    ``_samples_for_spec`` so ``run_musique_worker`` is unchanged.
    """
    return [
        {
            "warm": [p.assembled.warm_ids],
            "prompt": p.assembled.prompt_ids,
            "answers": p.answers,
        }
        for p in prepared
    ]


def _print_padding_report(prepared: list[PreparedSample], chunk_size: int) -> None:
    total_real = sum(p.assembled.real_tokens for p in prepared)
    total_pad = sum(p.assembled.pad_tokens for p in prepared)
    total = total_real + total_pad
    waste = (total_pad / total * 100.0) if total else 0.0
    chunk_counts = [c for p in prepared for c in p.assembled.ctx_chunk_counts]
    avg_chunks = mean([float(c) for c in chunk_counts])
    _log(
        f"prompt assembly: {len(prepared)} samples, chunk_size={chunk_size} "
        f"(block-aligned), ctx content tokens={total_real}, padding tokens="
        f"{total_pad} ({waste:.1f}% wasted on chunk alignment), "
        f"avg chunks/ctx={avg_chunks:.2f}"
    )
    if avg_chunks > 1.05:
        _log(
            f"NOTE: avg {avg_chunks:.2f} chunks/ctx -> link=k recomputes k "
            "tokens PER CHUNK, i.e. ~"
            f"{avg_chunks:.1f}*k tokens per ctx. Raise --chunk-size so ctx~=1 "
            "chunk for a clean 'k tokens per ctx' interpretation."
        )
    # CacheBlend-difference reminder (also in the module docstring).
    _log(
        "NOTE: CacheBlend concatenates contexts WITHOUT padding (arbitrary-span "
        "reuse); our EPIC reuses BLOCK-ALIGNED whole chunks, hence the padding "
        "above. This is a current-implementation constraint, not the algorithm."
    )


def _print_sample_table(mode_results: dict[str, dict]) -> None:
    """Per-sample table: one block per mode."""
    _log("PER-SAMPLE RESULTS")
    for label, res in mode_results.items():
        _log(f"  mode={label}")
        _log("    sample | answer_hit | prefill_ms | generated_text_head")
        rows = res.get("per_sample", [])
        for i, r in enumerate(rows):
            head = (r.get("text") or "").replace("\n", " ")[:60]
            ms = r.get("prefill_ms")
            ms_str = f"{ms:9.1f}" if ms is not None else "    n/a"
            _log(
                f"    {i:6d} | {('HIT' if r.get('answer_hit') else 'miss'):>10s}"
                f" | {ms_str} | {head!r}"
            )


def _print_aggregate_table(aggs: list[ModeAggregate], *, chunk_size: int) -> None:
    _log("AGGREGATE (mode | answer_hit_rate | mean_prefill_ms | speedup_vs_full)")
    _log("    mode        | hit_rate (n)   | mean_prefill_ms | speedup_vs_full")
    for a in aggs:
        sp = (
            "  n/a" if a.label == "full"
            else (f"{a.speedup_vs_full:6.2f}x"
                  if a.speedup_vs_full not in (float("inf"),) else "   inf")
        )
        _log(
            f"    {a.label:11s} | {a.answer_hit_rate:5.2f} ({a.answer_hits}/"
            f"{a.n}) | {a.mean_prefill_ms:15.1f} | {sp}"
        )
    # link >= chunk_size CONTROL interpretation. When the per-chunk link covers
    # the WHOLE chunk, every non-prefix chunk's tokens are recomputed -> the
    # forward is effectively full over the reused span. With the system-segment
    # fix engaged (all ctx are non-prefix), epic@k with k>=chunk_size should give
    # answer_hit_rate ~= full and speedup ~= 1x (or <1x from load overhead). If
    # such a mode is instead MUCH faster than full, the recompute did NOT happen
    # (LINK INERT) -- which is exactly the bug this fix targets.
    for a in aggs:
        if a.label.startswith("epic@"):
            try:
                k = int(a.label.split("@", 1)[1])
            except ValueError:
                continue
            if k >= chunk_size:
                verdict = (
                    "as expected (recompute engaged)"
                    if a.speedup_vs_full <= 1.5
                    else "SUSPECT: too fast -> recompute likely did NOT fire "
                         "(LINK INERT?)"
                )
                _log(
                    f"    CONTROL {a.label} (link>=chunk_size {chunk_size}): "
                    f"recomputes every non-prefix chunk -> expect "
                    f"speedup~=1x + hit_rate~=full. Observed speedup="
                    f"{a.speedup_vs_full:.2f}x -> {verdict}."
                )


def _check_sparse_engagement(label: str, res: dict) -> None:
    """Assert the connector actually engaged for a sparse mode (in-band counter
    check, same gate as gpu_smoke step4). Reports loudly; does not crash a run
    that simply had no reusable chunk (e.g. chunk_size > every ctx)."""
    counters = res.get("counters", {})
    if not (counters.get("sparse_match", 0) >= 1
            and counters.get("chunks_loaded", 0) >= 1):
        _log(
            f"WARNING: sparse mode {label!r} engagement counters {counters} "
            "do NOT show sparse_match>=1 AND chunks_loaded>=1. The reuse path "
            "may not have fired (check chunk alignment / byte-identity / "
            "in-process engine). Numbers for this mode are SUSPECT."
        )
    else:
        _log(f"  mode={label}: sparse engaged in-band, counters={counters}")
    _report_selection(label, res)


def _report_selection(label: str, res: dict) -> None:
    """Print the connector's per-request SELECTION diagnostics for a sparse mode.

    The load-bearing observability for this whole fix: it surfaces, per request,
    what the connector ACTUALLY saw -- ``prefix_extent``, the number/offsets of
    non-prefix hits, and whether the sparse (LegoLink-capable) branch fired. The
    silent failure mode this guards is "every ctx folded into the contiguous
    prefix -> num_non_prefix==0 -> LegoLink link is INERT (no recompute, fast but
    stale)". We emit an explicit ``LINK INERT`` warning so the reader sees it
    rather than mistaking a 3.5x speedup for a working reuse path.
    """
    sel = res.get("selection") or []
    if not sel:
        _log(
            f"  mode={label}: NO selection diagnostics recorded "
            "(epic_debug_counters off or no request reached selection?)."
        )
        return
    # Measured requests are the ones with the largest N (sys + all ctx + Q);
    # warm prefills (single ctx) have small N. Report every distinct entry but
    # focus the INERT check on requests that have at least one ctx-sized prompt.
    inert = [e for e in sel if int(e.get("num_non_prefix", 0)) == 0]
    engaged = [e for e in sel if int(e.get("num_non_prefix", 0)) > 0]
    _log(
        f"  mode={label}: selection over {len(sel)} request(s): "
        f"{len(engaged)} with non-prefix hits (LegoLink can recompute), "
        f"{len(inert)} with NONE."
    )
    # Show a few entries (the measured prompts have the largest N).
    for e in sorted(sel, key=lambda x: -int(x.get("N", 0)))[:5]:
        _log(
            f"      req={e.get('request_id')} N={e.get('N')} "
            f"prefix_extent={e.get('prefix_extent')} "
            f"non_prefix={e.get('num_non_prefix')} "
            f"offsets={e.get('non_prefix_offsets')} "
            f"sparse_branch={e.get('sparse_branch')}"
        )
    # The big-prompt (measured) requests are the ones we care about: if the
    # LARGEST-N request had zero non-prefix hits, link is inert for that mode.
    biggest = max(sel, key=lambda x: int(x.get("N", 0)))
    if int(biggest.get("num_non_prefix", 0)) == 0:
        _log(
            f"  *** WARNING: mode={label}: the measured prompt (N="
            f"{biggest.get('N')}) had non_prefix_hits=0 -> LINK INERT. Every "
            "context folded into the contiguous prefix, so LegoLink recomputes "
            "NOTHING regardless of --link. Reuse is STALE (fast but degraded). "
            "Enable the system segment (drop --no-system) to break position-0 "
            "contiguity. ***"
        )


# ---------------------------------------------------------------------------
# Scenario-specific reporting (reorder / insert / same)
# ---------------------------------------------------------------------------
def _scenario_layout_str(scenario: str) -> str:
    return {
        "same": "[sys*][a][b][c][query]  (* per-sample nonce -> sys non-hit)",
        "reorder": "[sys][b][a][c][query]  (sys HIT; b/a/c moved -> PIC delta!=0)",
        "insert": "[sys][d][a][c][query]  (d held-out NON-hit -> a,c non-prefix)",
    }[scenario]


def _print_scenario_intent(scenario: str) -> None:
    """State, per scenario, exactly which reuse mechanism is under test and what
    the connector counters/selection are EXPECTED to show, so the operator can
    immediately classify a result (PIC bug vs sparse-forward bug)."""
    if scenario == "reorder":
        _log(
            "INTENT reorder: b,a,c are all PREFIX hits (sys is warmed) so "
            "EXPECT sparse_match=0, sparse_branch=False, non_prefix=0 -- the "
            "sparse-forward path is NOT exercised. The contexts MOVED, so the "
            "prefix-load applies a NON-ZERO PIC delta re-rotary. This is the "
            "first real-data test of PIC re-rotary + load + scatter ALONE. "
            "reuse-only correct here => PIC is sound."
        )
    elif scenario == "insert":
        _log(
            "INTENT insert: d is a held-out (never-warmed) ctx spliced after "
            "sys, so a,c become NON-PREFIX hits. EXPECT non_prefix>0, "
            "sparse_branch=True, sparse_match>=1 -- the sparse-forward + "
            "LegoLink recompute path IS exercised. Breakage here while reorder "
            "is fine => the bug is in sparse-forward, NOT PIC."
        )
    else:  # same
        _log(
            "INTENT same: legacy control. sys carries a per-sample nonce -> "
            "sys is a NON-hit at position 0 -> a,b,c become non-prefix hits at "
            "their ORIGINAL positions (PIC delta=0). EXPECT non_prefix>0, "
            "sparse_branch=True. Mixes PIC(zero-delta)+sparse; use reorder/"
            "insert to separate the two."
        )


def _print_scenario_padding_report(
    prepared: list[PreparedScenarioSample], chunk_size: int
) -> None:
    total_real = sum(p.assembled.real_tokens for p in prepared)
    total_pad = sum(p.assembled.pad_tokens for p in prepared)
    total = total_real + total_pad
    waste = (total_pad / total * 100.0) if total else 0.0
    _log(
        f"scenario assembly: {len(prepared)} samples, chunk_size={chunk_size} "
        f"(block-aligned), ctx content tokens={total_real}, padding tokens="
        f"{total_pad} ({waste:.1f}% wasted on chunk alignment). "
        "CacheBlend concatenates contexts WITHOUT padding (arbitrary-span "
        "reuse); EPIC reuses BLOCK-ALIGNED whole chunks, hence the padding."
    )


def _print_scenario_selection_prediction(
    prepared: list[PreparedScenarioSample],
) -> None:
    """Print the STATIC (CPU mirror) prediction of the prefix/non-prefix split
    for the measured prompts, so the operator sees what selection SHOULD do
    before any GPU runs -- and can compare it to the connector's in-band
    selection diagnostics afterward."""
    if not prepared:
        return
    p0 = prepared[0].assembled
    n_non = [len(p.assembled.expect_non_prefix_offsets) for p in prepared]
    n_non_set = sorted(set(n_non))
    sparse_expected = any(p.assembled.expect_sparse for p in prepared)
    _log(
        f"PREDICTED selection (CPU mirror of EpicSelection): scenario="
        f"{p0.scenario!r} measured layout segs={p0.seg_labels} "
        f"prefix_extent(sample0)={p0.expect_prefix_extent} "
        f"non_prefix_offsets(sample0)={p0.expect_non_prefix_offsets} "
        f"non_prefix_count_over_samples={n_non_set} "
        f"=> sparse_branch_expected={sparse_expected}"
    )
    if p0.scenario == "reorder" and sparse_expected:
        _log(
            "  *** UNEXPECTED: reorder predicted to have non-prefix hits. The "
            "reordered ctxs should stay a contiguous PREFIX (sys is warmed). "
            "Check ctx chunk alignment. ***"
        )
    if p0.scenario == "insert" and not sparse_expected:
        _log(
            "  *** UNEXPECTED: insert predicted to have NO non-prefix hits. The "
            "held-out d should break contiguity so a,c are non-prefix. Check "
            "that d is genuinely never warmed. ***"
        )


def _check_scenario_engagement(scenario: str, label: str, res: dict) -> None:
    """Report the connector's in-band counters/selection for a sparse mode and
    check them against the scenario's EXPECTATION (reorder: sparse must NOT
    fire; insert/same: sparse must fire). Reports loudly; never crashes a run."""
    counters = res.get("counters", {})
    _log(
        f"  scenario={scenario!r} mode={label!r} counters: "
        f"loads_emitted={counters.get('loads_emitted', 0)} "
        f"chunks_loaded={counters.get('chunks_loaded', 0)} "
        f"sparse_match={counters.get('sparse_match', 0)} "
        f"sparse_emit={counters.get('sparse_emit', 0)} "
        f"check_load_calls={counters.get('check_load_calls', 0)} "
        f"check_load_mismatch={counters.get('check_load_mismatch', 0)} "
        f"check_load_skip={counters.get('check_load_skip', 0)}"
    )
    sel = res.get("selection") or []
    biggest = max(sel, key=lambda x: int(x.get("N", 0))) if sel else {}
    nprefix = int(biggest.get("num_non_prefix", 0)) if biggest else 0
    sbranch = bool(biggest.get("sparse_branch")) if biggest else False
    if scenario == "reorder":
        # reorder should LOAD chunks (prefix-load + PIC) but NOT take the sparse
        # branch. chunks_loaded>=1 proves load+PIC ran; sparse_match==0 proves
        # the sparse-forward path stayed out (so any reorder breakage is PIC).
        if counters.get("chunks_loaded", 0) < 1:
            _log(
                f"  *** WARNING reorder/{label}: chunks_loaded=0 -> the prefix "
                "chunks were NOT loaded. PIC/load path never ran; the measured "
                "prompt may not have hit the warmed sys (check sys byte-identity "
                "warm vs measured). Numbers SUSPECT. ***"
            )
        if counters.get("sparse_match", 0) != 0 or nprefix != 0:
            _log(
                f"  *** NOTE reorder/{label}: sparse_match="
                f"{counters.get('sparse_match', 0)} non_prefix={nprefix} -- "
                "EXPECTED 0 for reorder (all ctxs should fold into the PIC-"
                "rotated prefix). If non-zero, the reorder layout is NOT a pure "
                "prefix-reuse test. ***"
            )
        else:
            _log(
                f"  reorder/{label}: PURE prefix reuse + PIC (sparse_match=0, "
                "non_prefix=0) as intended -- isolates PIC re-rotary/load."
            )
    else:  # insert / same -> sparse path must engage
        if not (counters.get("sparse_match", 0) >= 1
                and counters.get("chunks_loaded", 0) >= 1 and sbranch):
            _log(
                f"  *** WARNING {scenario}/{label}: sparse branch did NOT fully "
                f"engage (sparse_match={counters.get('sparse_match', 0)}, "
                f"chunks_loaded={counters.get('chunks_loaded', 0)}, "
                f"sparse_branch={sbranch}, non_prefix={nprefix}). The recompute "
                "path may be inert; numbers SUSPECT. ***"
            )
        else:
            _log(
                f"  {scenario}/{label}: sparse-forward engaged "
                f"(sparse_match={counters.get('sparse_match', 0)}, "
                f"non_prefix={nprefix}, sparse_branch=True) as intended."
            )
    _report_selection(f"{scenario}/{label}", res)


def _print_scenario_aggregate_table(
    scenario: str, aggs: list[ModeAggregate], *, chunk_size: int
) -> None:
    _log(
        f"AGGREGATE scenario={scenario!r} "
        "(mode | answer_hit_rate | mean_prefill_ms | speedup_vs_full)"
    )
    _log("    mode             | hit_rate (n)   | mean_prefill_ms | speedup")
    for a in aggs:
        is_full = a.label.endswith("/full")
        sp = (
            "  n/a" if is_full
            else (f"{a.speedup_vs_full:6.2f}x"
                  if a.speedup_vs_full not in (float("inf"),) else "   inf")
        )
        _log(
            f"    {a.label:16s} | {a.answer_hit_rate:5.2f} ({a.answer_hits}/"
            f"{a.n}) | {a.mean_prefill_ms:15.1f} | {sp}"
        )


def _print_scenario_verdict(scenario: str) -> None:
    """Print the decision rule this scenario answers (PIC vs sparse-forward)."""
    if scenario == "reorder":
        _log(
            "VERDICT KEY (reorder): reuse-only hit_rate ~= full => PIC re-"
            "rotary + load + scatter are SOUND on real data. reuse-only "
            "DEGRADED here => the bug is in PIC/load (sparse-forward is not "
            "even exercised in reorder)."
        )
    elif scenario == "insert":
        _log(
            "VERDICT KEY (insert): epic@k engages the sparse-forward path. If "
            "reorder was fine but insert-epic collapses => the bug is in the "
            "SPARSE-FORWARD / LegoLink recompute path, NOT PIC. epic@k hit_rate "
            "should climb toward full as k grows (more boundary recompute)."
        )
    else:
        _log(
            "VERDICT KEY (same): control -- mixes zero-delta PIC with sparse-"
            "forward. Use scenario=reorder (PIC only) and scenario=insert "
            "(sparse only) to localise any collapse."
        )


def run_all(args) -> int:
    samples_all = load_musique(args.data, download=not args.no_download)
    samples = samples_all[: args.num_samples]
    if not samples:
        _fail("no samples after slicing --num-samples")

    block_size = DEFAULT_BLOCK_SIZE
    # Force chunk_size to a block_size multiple (the connector does this anyway;
    # we mirror it so our assembly hashes match the connector's chunk grid).
    chunk_size = effective_chunk_size(args.chunk_size, block_size)
    if chunk_size != args.chunk_size:
        _log(
            f"--chunk-size {args.chunk_size} rounded UP to {chunk_size} "
            f"(must be a multiple of block_size {block_size})"
        )

    tok, is_real = load_tokenizer(args.model, allow_fallback=False)
    if not is_real:
        _fail("could not load a real tokenizer; cannot build model prompts")

    # The leading RAG system/instruction segment is REQUIRED for the scenario
    # path: it is the byte-identical leading chunk shared by warm and measured,
    # and (for reorder/insert) the warmed sys HIT is what lets the reordered ctxs
    # fold into the PIC-rotated prefix / the held-out d break contiguity. An
    # empty system prompt is rejected so we never silently lose the layout.
    system_text = args.system_prompt
    if not system_text:
        _fail("--system-prompt must be non-empty for the scenario probe")

    scenario = args.scenario
    prepared = prepare_scenario_samples(
        samples, tok=tok, chunk_size=chunk_size,
        scenario=scenario, system_text=system_text,
    )
    sys_lens = sorted({p.assembled.sys_len for p in prepared})
    _log(
        f"scenario={scenario!r}: RAG system prompt fixed at FRONT "
        f"(padded sys_len={sys_lens} tokens, chunk_size={chunk_size}). "
        f"warm prompt = [sys][a][b][c] (stores sys,a,b,c); "
        f"measured prompt = {_scenario_layout_str(scenario)}."
    )
    _print_scenario_intent(scenario)
    _print_scenario_padding_report(prepared, chunk_size)
    _print_scenario_selection_prediction(prepared)

    # Length sanity vs max_model_len (warm and measured prompts both matter).
    max_prompt = max(
        max(len(p.assembled.prompt_ids), len(p.assembled.warm_ids))
        for p in prepared
    )
    if max_prompt + args.max_tokens + 8 > args.max_model_len:
        _fail(
            f"longest assembled prompt is {max_prompt} tokens + {args.max_tokens}"
            f" gen > --max-model-len {args.max_model_len}. Raise --max-model-len "
            "or lower --chunk-size."
        )

    # Build the mode list: full, reuse-only, epic@<link or sweep>.
    modes: list[Mode] = [mode_full(), mode_reuse_only()]
    link_ks = (
        sorted(set(args.link_sweep)) if args.link_sweep else [args.link]
    )
    for k in link_ks:
        if k > 0:  # epic@0 == reuse-only, already added.
            modes.append(mode_epic(k))

    _log(
        f"modes={[m.label for m in modes]} scenario={scenario!r} "
        f"num_samples={len(prepared)} chunk_size={chunk_size} "
        f"max_tokens={args.max_tokens} warmup_discard={args.warmup_discard}"
    )

    sample_specs = _scenario_samples_for_spec(prepared)
    mode_results: dict[str, dict] = {}
    aggs: list[ModeAggregate] = []
    for mode in modes:
        _log(
            f"=== running scenario={scenario!r} mode={mode.label} "
            f"(fresh subprocess engine) ==="
        )
        spec = build_musique_worker_spec(
            mode=mode,
            model=args.model,
            chunk_size=chunk_size,
            samples=sample_specs,
            max_tokens=args.max_tokens,
            block_size=block_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
        )
        res = run_mode_subprocess(spec)
        if not res.get("ok"):
            _fail(
                f"mode {mode.label} engine failed: {res.get('error')}\n"
                + res.get("traceback", "")
            )
        mode_results[mode.label] = res
        if mode.sparse:
            _check_scenario_engagement(scenario, mode.label, res)
        aggs.append(
            aggregate_mode(
                f"{scenario}/{mode.label}", res["per_sample"],
                warmup_discard=args.warmup_discard,
            )
        )

    # Speedups within this scenario use this scenario's full as the baseline.
    full_label = f"{scenario}/full"
    base = next((a for a in aggs if a.label == full_label), None)
    base_ms = base.mean_prefill_ms if base else 0.0
    for a in aggs:
        a.speedup_vs_full = speedup(base_ms, a.mean_prefill_ms) if base_ms > 0 else 0.0
    _print_sample_table(mode_results)
    _print_scenario_aggregate_table(scenario, aggs, chunk_size=chunk_size)
    _print_scenario_verdict(scenario)
    _log("DONE")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _worker_main(spec_json: str) -> int:
    try:
        spec = parse_spec(spec_json)
    except Exception as e:  # noqa: BLE001
        _emit_result({"ok": False, "mode_label": "?",
                      "error": f"bad spec: {type(e).__name__}: {e}"})
        return 0
    _emit_result(run_musique_worker(spec))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="EPIC PIC/reuse probe on CacheBlend musique data"
    )
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct",
                    help="HF model id (small Llama-family recommended).")
    ap.add_argument("--data", default=_DEFAULT_DATA_PATH,
                    help=f"musique_s.json path (default {_DEFAULT_DATA_PATH}; "
                         "downloaded from CacheBlend if absent).")
    ap.add_argument("--no-download", action="store_true",
                    help="do NOT download musique data if the file is missing.")
    ap.add_argument("--num-samples", type=int, default=5,
                    help="number of musique samples to run (default 5).")
    ap.add_argument("--ctx-per-sample", type=int, default=6,
                    help="contexts per sample, capped from the 10 available "
                         "(default 6; controls prompt length).")
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="EPIC chunk size; rounded UP to a block_size (16) "
                         "multiple (default 256). Larger -> ctx~=1 chunk.")
    ap.add_argument("--link", type=int, default=8,
                    help="epic@k LegoLink leading-recompute tokens per "
                         "non-prefix chunk (default 8).")
    ap.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="same",
        help="Realistic RAG reuse scenario for the MEASURED request after a "
             "warm [sys][a][b][c] full prefill. 'same'=[sys*][a][b][c][query] "
             "(legacy nonce control, * = per-sample sys nonce); "
             "'reorder'=[sys][b][a][c][query] (sys HIT, ctxs moved -> non-zero "
             "PIC delta, NO sparse branch -> isolates PIC re-rotary/load); "
             "'insert'=[sys][d][a][c][query] (d = held-out non-warmed ctx -> a,c "
             "non-prefix -> isolates sparse-forward+LegoLink). reorder vs insert "
             "splits the link=255 collapse into PIC vs sparse-forward bugs.")
    ap.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "You are a helpful assistant. Use the documents to answer the "
            "question.\n"
        ),
        help="Leading RAG instruction/system segment fixed at the FRONT of "
             "both the warm and measured prompts (chunk-padded). It is the "
             "byte-identical leading chunk shared warm<->measured; in reorder/"
             "insert it is WARMED (a HIT) so the reordered ctxs fold into the "
             "PIC-rotated prefix / the held-out d breaks contiguity. In 'same' "
             "a per-sample nonce makes it a non-hit (legacy behaviour). Must be "
             "non-empty.")
    ap.add_argument(
        "--no-system",
        action="store_true",
        help="DEPRECATED in the scenario probe (the scenario layouts REQUIRE a "
             "leading system segment); accepted but ignored.")
    ap.add_argument("--link-sweep", type=str, default=None,
                    help="comma-separated link values to run epic@k for each "
                         "(overrides --link), e.g. '0,8,64'.")
    ap.add_argument("--max-tokens", type=int, default=32,
                    help="greedy generation length for the answer (default 32).")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="max_model_len per engine (default 8192).")
    ap.add_argument("--gpu-mem-util", type=float, default=0.45,
                    help="gpu_memory_utilization per engine (default 0.45).")
    ap.add_argument("--warmup-discard", action="store_true",
                    help="drop the FIRST sample from the aggregate (cold "
                         "compile/autotune warmup).")
    # Spec is passed via a FILE, not argv: musique contexts are thousands of
    # tokens, and serializing them into a command-line argument overruns the OS
    # argv limit ("Argument list too long").
    ap.add_argument("--_worker-spec-file", dest="worker_spec_file", default=None,
                    help=argparse.SUPPRESS)
    return ap


def _parse_link_sweep(arg: str | None) -> list[int] | None:
    if arg is None:
        return None
    out: list[int] = []
    for tok in str(arg).split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v < 0:
            raise ValueError(f"link value must be >= 0, got {v}")
        out.append(v)
    return out or None


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.worker_spec_file is not None:
        with open(args.worker_spec_file, encoding="utf-8") as f:
            return _worker_main(f.read())

    try:
        args.link_sweep = _parse_link_sweep(args.link_sweep)
    except ValueError as e:
        _fail(f"bad --link-sweep: {e}")

    if not _has_cuda():
        _log(
            "No CUDA GPU detected. This script requires a GPU + a built EPIC "
            "vLLM. See vllm/.../epic/PHASE2.md for the build command. Exiting "
            "without running (CPU import/structure check is a no-op here)."
        )
        return 0

    return run_all(args)


if __name__ == "__main__":
    sys.exit(main())
