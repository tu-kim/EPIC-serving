# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the EPIC-reuse benchmark suite.

This module is import-safe on a CPU-only box (no torch / no vLLM import at
module load). It provides:

  * the prompt-layout contract (A + C + B + Q) and its alignment constraints,
  * a deterministic synthetic token-vocabulary builder,
  * a tokenizer abstraction with an offline whitespace fallback so
    ``--dry-run`` / ``--plan-only`` work with no network and no model download,
  * the EPIC chunk-hash (mirrors the connector's ``hash_chunk_tokens`` so the
    bench can *predict* which chunks will collide between warmup and target),
  * the kv-transfer / engine config builders for the three modes.

Nothing here touches a GPU. The heavy stuff (LLM construction) lives in
bench_perf.py / bench_accuracy.py and is only imported under ``--plan-only``
guards there.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Prompt layout contract
# ---------------------------------------------------------------------------
# prompt = A + C + B + Q  (Q is the question; it may be folded into the C tail).
#
#   A : shared prefix document. vLLM native prefix caching can reuse this
#       (prefix-chain hash match).  Reused by BOTH prefix and epic modes.
#   C : per-request unique context. Always recomputed (the genuinely-new tokens).
#   B : shared passage placed in the MIDDLE of the prompt (non-prefix). Native
#       prefix caching can NEVER reuse this (a different C in front breaks the
#       prefix chain). EPIC reuses it via content-hash + PIC re-rotary.
#   Q : the needle question, appended after B.
#
# Alignment constraint (THE crux of correct reuse measurement):
#   The EPIC chunk store hashes whole `chunk_size` chunks counted from prompt
#   position 0. For B's chunks to collide between a warmup prompt and a target
#   prompt, B must START at a chunk-aligned offset in BOTH and its tokens must
#   be byte-identical and chunk-aligned in length. We therefore force
#       |A| % chunk_size == 0   and   |C| % chunk_size == 0
#   so that in the target prompt B begins exactly at offset |A|+|C| (a chunk
#   multiple) and in the warmup prompt B begins at a chunk multiple too.
#   |B| is also forced to a chunk multiple so every B token lives in a full
#   (hashable) chunk.  Q sits after B as a trailing partial chunk (never
#   hashed/saved), matching the connector's "whole chunks only" rule.


# Base mode families. The concrete modes actually run are *mode specs* (see
# ModeSpec / parse_mode_spec below): an epic run is parameterized by its link-k
# (recompute boundary tokens per non-prefix chunk), and the connector fixes k at
# engine construction, so every distinct k is a separate engine == a separate
# mode spec ("epic@<k>"). "reuse-only" is the first-class label for epic@0
# (PIC re-rotary, ZERO recompute of B).
MODES = ("full", "prefix", "epic")

# A reuse-only run is literally epic with link_tokens == 0. We expose it under a
# distinct label in tables/plots but it constructs the same EpicConnector.
REUSE_ONLY_LINK = 0

DEFAULT_BLOCK_SIZE = 16
DEFAULT_CHUNK_SIZE = 256
DEFAULT_LINK_TOKENS = 8


# ---------------------------------------------------------------------------
# Mode specs (mode + link-k variant)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModeSpec:
    """A concrete engine configuration to run.

    ``family`` is one of MODES ("full", "prefix", "epic").
    ``link_k`` is the EPIC LegoLink recompute boundary tokens per non-prefix
    chunk; it is meaningful only for the epic family (None for full/prefix).
    ``label`` is the human/table/plot name:
      * "full", "prefix"            -> family name, link_k None.
      * "epic@<k>"                  -> epic family, link_k == k.
      * "reuse-only"                -> epic family, link_k == 0 (alias of epic@0,
                                       but labeled distinctly so the "no
                                       recompute, position-fix only" baseline
                                       stands out in tables and plots).
    """

    family: str
    label: str
    link_k: Optional[int] = None

    @property
    def is_epic(self) -> bool:
        return self.family == "epic"

    def csv_link(self) -> int:
        """Value to write into the CSV ``link_k`` column (-1 for non-epic)."""
        return self.link_k if self.link_k is not None else -1


def parse_mode_spec(token: str, default_link: int = DEFAULT_LINK_TOKENS) -> ModeSpec:
    """Parse one mode token into a ModeSpec.

    Accepted tokens:
      full | prefix                 (baselines)
      epic                          (epic at default_link)
      epic@<k>                      (epic at link k, k a non-negative int)
      reuse-only                    (epic at link 0, distinct label)
    """
    t = token.strip()
    if t == "reuse-only":
        return ModeSpec(family="epic", label="reuse-only", link_k=REUSE_ONLY_LINK)
    if t in ("full", "prefix"):
        return ModeSpec(family=t, label=t, link_k=None)
    if t == "epic":
        return ModeSpec(family="epic", label=f"epic@{default_link}",
                        link_k=default_link)
    if t.startswith("epic@"):
        suffix = t[len("epic@"):]
        try:
            k = int(suffix)
        except ValueError as e:  # noqa: BLE001
            raise ValueError(f"bad epic link spec {token!r}: {e}") from e
        if k < 0:
            raise ValueError(f"epic link k must be >= 0, got {k} in {token!r}")
        return ModeSpec(family="epic", label=f"epic@{k}", link_k=k)
    raise ValueError(
        f"unknown mode {token!r}; valid: full, prefix, epic, epic@<k>, reuse-only"
    )


def expand_mode_specs(
    modes_csv: str,
    *,
    link_sweep: Optional[list[int]] = None,
    default_link: int = DEFAULT_LINK_TOKENS,
) -> list[ModeSpec]:
    """Expand a ``--modes`` string (+ optional ``--link-sweep``) into ModeSpecs.

    ``--link-sweep`` auto-expands any bare ``epic`` token in the modes list into
    one ``epic@k`` spec per k in the sweep (de-duplicated, order preserved).
    Non-epic tokens and explicit ``epic@k`` / ``reuse-only`` tokens pass through
    unchanged. If the modes list contains no ``epic`` token but a link-sweep is
    given, the sweep is appended as epic@k specs (so ``--link-sweep`` alone is a
    convenient way to request the whole epic-k family).
    """
    tokens = [m.strip() for m in modes_csv.split(",") if m.strip()]
    out: list[ModeSpec] = []
    seen: set[str] = set()

    def _add(spec: ModeSpec) -> None:
        if spec.label not in seen:
            seen.add(spec.label)
            out.append(spec)

    saw_bare_epic = any(t == "epic" for t in tokens)
    for t in tokens:
        if t == "epic" and link_sweep:
            for k in link_sweep:
                _add(parse_mode_spec(f"epic@{k}", default_link))
            continue
        _add(parse_mode_spec(t, default_link))

    if link_sweep and not saw_bare_epic:
        # No bare 'epic' to expand in-place; append the sweep family explicitly
        # only if the user did not already list explicit epic@k tokens.
        already_epic = any(s.is_epic for s in out)
        if not already_epic:
            for k in link_sweep:
                _add(parse_mode_spec(f"epic@{k}", default_link))
    return out


# ---------------------------------------------------------------------------
# Tokenizer abstraction (offline-capable)
# ---------------------------------------------------------------------------
class WhitespaceTokenizer:
    """Trivial reversible tokenizer: one token == one whitespace-split word.

    Used as an offline fallback for ``--dry-run`` / ``--plan-only`` when no HF
    tokenizer can be loaded. Token "ids" are deterministic hashes of the word;
    they are never fed to a real model under this fallback (dry-run only), so
    collision-freedom of the *text* (not the id) is what matters and the
    synthetic vocabulary below guarantees distinct words.
    """

    name = "whitespace-fallback"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        words = text.split(" ")
        return [
            int(hashlib.sha256(w.encode()).hexdigest()[:8], 16)
            for w in words
            if w != ""
        ]

    def __call__(self, text: str, **kw: Any) -> list[int]:
        return self.encode(text)


class HFTokenizerWrap:
    """Thin wrapper so HF and the fallback share one interface."""

    def __init__(self, tok: Any, name: str):
        self._tok = tok
        self.name = name

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)


def load_tokenizer(model: Optional[str], *, allow_fallback: bool = True):
    """Load an HF tokenizer; fall back to whitespace if offline/unavailable.

    Returns ``(tokenizer, is_real)`` where ``is_real`` is False for the
    whitespace fallback (which must NOT be used to build prompts fed to a real
    model -- it is only valid for dry-run length accounting).
    """
    if model:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            return HFTokenizerWrap(tok, model), True
        except Exception as e:  # noqa: BLE001
            if not allow_fallback:
                raise
            print(
                f"[epic-bench] WARNING: could not load HF tokenizer {model!r} "
                f"({type(e).__name__}: {e}). Falling back to the offline "
                "whitespace tokenizer. This is fine for --dry-run length "
                "checks but NOT for a real generation run."
            )
    return WhitespaceTokenizer(), False


# ---------------------------------------------------------------------------
# Deterministic synthetic vocabulary
# ---------------------------------------------------------------------------
# A pool of plain English-ish words. We pick from it by a seeded index so the
# generated text is reproducible and tokenizes to roughly one token/word for
# common tokenizers (we then trim/pad to an exact token count).
_WORD_POOL = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu amber basil cedar dawn ember frost grove harbor ivory "
    "jade kite lotus maple nectar opal pearl quartz river slate timber umbra "
    "violet willow xenon yarrow zephyr anchor beacon canyon dune eagle fjord "
    "glacier hollow island jungle knoll ledge meadow notch oasis prairie "
    "quarry ridge summit tundra valley wetland canopy boulder crater desert"
).split()


def _word_at(seed: int, idx: int) -> str:
    return _WORD_POOL[(seed * 1103515245 + idx * 12345) % len(_WORD_POOL)]


def _make_words(seed: int, n_words: int) -> list[str]:
    return [_word_at(seed, i) for i in range(n_words)]


def build_text_of_exact_tokens(
    tok: Any,
    seed: int,
    n_tokens: int,
    *,
    prefix_words: Optional[list[str]] = None,
) -> tuple[str, list[int]]:
    """Build text whose tokenization is EXACTLY ``n_tokens`` ids.

    Strategy: emit seeded words until the token count meets/exceeds the target,
    then truncate the *token ids* and re-decode is not reversible for HF, so we
    instead grow word-by-word and stop at the largest word boundary that does
    not exceed n_tokens, then top up with single filler words until exact. We
    finally trim at the token level by dropping trailing words if we overshoot.
    Returns ``(text, token_ids)`` with ``len(token_ids) == n_tokens``.
    """
    if n_tokens <= 0:
        return "", []
    words: list[str] = list(prefix_words or [])
    # Grow generously then trim at token granularity.
    i = 0
    # Heuristic over-generate factor; whitespace fallback is ~1 tok/word, HF
    # subword tokenizers can be >1, so cap iterations defensively.
    while True:
        text = " ".join(words)
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) >= n_tokens:
            break
        words.append(_word_at(seed, i))
        i += 1
        if i > n_tokens * 8 + 64:  # safety stop
            break
    # Trim words from the end until tokens <= n_tokens, then we accept the exact
    # boundary by binary-ish shrink.
    while len(words) > 0:
        text = " ".join(words)
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= n_tokens:
            break
        words.pop()
    # Token list may be a hair under n_tokens (subword boundary). Pad with a
    # single repeated filler word until exact; if a single word adds >1 token we
    # may overshoot by a token or two -- correct by truncating the id list and
    # rebuilding a decodable text is impossible for HF, so we instead pad with
    # the rarest 1-token-ish filler and finally hard-truncate ids for accounting.
    ids = tok.encode(" ".join(words), add_special_tokens=False)
    filler = "the"
    guard = 0
    while len(ids) < n_tokens and guard < n_tokens + 16:
        words.append(filler)
        ids = tok.encode(" ".join(words), add_special_tokens=False)
        guard += 1
    text = " ".join(words)
    ids = tok.encode(text, add_special_tokens=False)
    # Final exactness for accounting: truncate ids (the *text* may tokenize to a
    # few extra subwords; callers that need byte-identical B use the ids list,
    # and we expose the truncated ids so length math is exact).
    ids = ids[:n_tokens]
    return text, ids


def _sentence_split(text: str) -> list[str]:
    """Naive sentence splitter (no nltk dep): split on ., !, ? keeping the
    delimiter. Good enough to keep the answer-bearing sentence intact.
    """
    import re as _re_local

    parts = _re_local.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def build_passage_b_of_exact_tokens(
    tok: Any,
    passage: str,
    answer_text: str,
    n_tokens: int,
    *,
    filler_seed: int,
    pad_passages: Optional[list[str]] = None,
) -> tuple[str, list[int]]:
    """Build B from a real gold ``passage`` at EXACTLY ``n_tokens`` ids, keeping
    the answer-bearing sentence(s) intact.

    Strategy:
      * Split the passage into sentences. Always keep the sentence(s) that
        contain ``answer_text`` (the "must-keep" core). If the passage is longer
        than n_tokens, drop sentences from the FAR ends first, never the core.
      * If still too long, hard-truncate the surrounding (non-answer) tokens but
        leave the answer span; if the answer span alone exceeds n_tokens we keep
        it and let the caller widen b_tokens (a message is the caller's job).
      * If too short, append filler from ``pad_passages`` (other real passages)
        then seeded synthetic words, so B reaches exactly n_tokens. Filler is
        appended AFTER the core so the answer is never pushed out by truncation.

    Returns ``(text, ids)`` with ``len(ids) == n_tokens`` (unless the answer core
    alone is longer, in which case ids may exceed and the caller truncates).
    """
    if n_tokens <= 0:
        return "", []

    sents = _sentence_split(passage) or [passage]
    norm_ans = normalize_answer(answer_text)

    core_idx = [
        i for i, s in enumerate(sents)
        if norm_ans and norm_ans in normalize_answer(s)
    ]
    if not core_idx:
        # Answer not found as a sentence substring (e.g. yes/no). Keep the whole
        # passage as core so we never accidentally cut the relevant content.
        core_idx = list(range(len(sents)))
    lo, hi = min(core_idx), max(core_idx)

    # Greedy window expand around [lo, hi] until we would exceed n_tokens.
    def toks(idxs: list[int]) -> int:
        if not idxs:
            return 0
        txt = " ".join(sents[i] for i in idxs)
        return len(tok.encode(txt, add_special_tokens=False))

    chosen = list(range(lo, hi + 1))
    left, right = lo - 1, hi + 1
    # Expand outward alternately while under budget.
    while (left >= 0 or right < len(sents)) and toks(
        ([left] if left >= 0 else []) + chosen + ([right] if right < len(sents) else [])
    ) <= n_tokens:
        if right < len(sents):
            chosen = chosen + [right]
            right += 1
        if left >= 0 and toks([left] + chosen) <= n_tokens:
            chosen = [left] + chosen
            left -= 1
        if left < 0 and right >= len(sents):
            break

    core_text = " ".join(sents[i] for i in chosen)
    core_ids = tok.encode(core_text, add_special_tokens=False)

    if len(core_ids) >= n_tokens:
        # Truncate from the tail but try to keep the answer span. If the answer
        # lives near the front this is safe; otherwise we keep front n_tokens.
        return core_text, core_ids[:n_tokens]

    # Too short: pad with other real passages, then seeded synthetic words.
    pad_chunks = list(pad_passages or [])
    words: list[str] = core_text.split(" ")
    pi = 0
    while pi < len(pad_chunks):
        cand = words + pad_chunks[pi].split(" ")
        if len(tok.encode(" ".join(cand), add_special_tokens=False)) > n_tokens:
            break
        words = cand
        pi += 1
    # Final exact top-up with deterministic seeded filler words.
    i = 0
    while True:
        ids = tok.encode(" ".join(words), add_special_tokens=False)
        if len(ids) >= n_tokens:
            break
        words.append(_word_at(filler_seed, i))
        i += 1
        if i > n_tokens * 8 + 64:
            break
    # Trim word boundary then pad ids exactly (mirrors build_text_of_exact_tokens).
    while words:
        ids = tok.encode(" ".join(words), add_special_tokens=False)
        if len(ids) <= n_tokens:
            break
        words.pop()
    ids = tok.encode(" ".join(words), add_special_tokens=False)
    guard = 0
    while len(ids) < n_tokens and guard < n_tokens + 16:
        words.append("the")
        ids = tok.encode(" ".join(words), add_special_tokens=False)
        guard += 1
    text = " ".join(words)
    ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
    return text, ids


# ---------------------------------------------------------------------------
# Needle facts (the accuracy signal)
# ---------------------------------------------------------------------------
@dataclass
class Needle:
    subject: str
    answer: str

    def fact_sentence(self) -> str:
        return f"The secret code for {self.subject} is {self.answer}."

    def question(self) -> str:
        return f" What is the secret code for {self.subject}?"


def make_needles(seed: int, k: int) -> list[Needle]:
    out: list[Needle] = []
    for j in range(k):
        subj = _word_at(seed + 7, j) + "-" + _word_at(seed + 13, j)
        ans = f"{(seed * 31 + j * 17) % 9000 + 1000}"
        out.append(Needle(subject=subj, answer=ans))
    return out


# ---------------------------------------------------------------------------
# Answer scoring (HF QA tasks): containment + SQuAD-style token F1
# ---------------------------------------------------------------------------
import re as _re
import string as _string

_ARTICLES_RE = _re.compile(r"\b(a|an|the)\b", _re.UNICODE)
_PUNCT_TABLE = {ord(c): None for c in _string.punctuation}
_WS_RE = _re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    """SQuAD normalization: lowercase, strip punctuation, drop articles, squash
    whitespace. Identical in spirit to the official SQuAD eval ``normalize_answer``
    so our token-F1 is comparable to standard reported numbers.
    """
    if s is None:
        return ""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def answer_containment(prediction: str, gold_answers: list[str]) -> bool:
    """True if any normalized gold answer is a substring of the normalized
    prediction (token-boundary aware via the whitespace squash). This is the
    'did the model say the answer somewhere' signal, robust to extra chatter.
    """
    norm_pred = normalize_answer(prediction)
    if not norm_pred:
        return False
    for g in gold_answers or []:
        ng = normalize_answer(g)
        if ng and ng in norm_pred:
            return True
    return False


def _f1_single(prediction: str, gold: str) -> float:
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    # Multiset overlap (SQuAD official semantics).
    common: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for t in gold_toks:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    num_same = 0
    pred_counts: dict[str, int] = {}
    for t in pred_toks:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t, c in pred_counts.items():
        num_same += min(c, gold_counts.get(t, 0))
    common  # noqa: B018 (kept for clarity; unused)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, gold_answers: list[str]) -> float:
    """Max token-F1 over the gold alias list (SQuAD multi-answer convention)."""
    if not gold_answers:
        return 0.0
    return max(_f1_single(prediction, g) for g in gold_answers)


def _scoring_self_test() -> None:
    """Lightweight invariants for the scoring helpers (run via --self-test)."""
    assert normalize_answer("The Denver Broncos!") == "denver broncos"
    assert normalize_answer("  A  CAT,  ") == "cat"
    assert answer_containment("the answer is Denver Broncos.",
                              ["Denver Broncos"]) is True
    assert answer_containment("a panther", ["Carolina Panthers"]) is False
    assert answer_containment("", ["x"]) is False
    # F1: exact match -> 1.0
    assert abs(token_f1("Denver Broncos", ["denver broncos"]) - 1.0) < 1e-9
    # F1: partial overlap (1 of 2 pred toks, 1 of 1 gold) -> P=0.5,R=1 -> 0.666..
    f = _f1_single("denver bears", "denver")
    assert abs(f - (2 * 0.5 * 1.0 / 1.5)) < 1e-9, f
    # F1: no overlap -> 0
    assert token_f1("cats", ["dogs"]) == 0.0
    # F1: best over aliases
    assert abs(token_f1("Broncos", ["Denver Broncos", "Broncos"]) - 1.0) < 1e-9
    # Articles dropped
    assert token_f1("the broncos", ["broncos"]) == 1.0


# ---------------------------------------------------------------------------
# EPIC chunk hash (mirror of connector hash_chunk_tokens) -- predict collisions
# ---------------------------------------------------------------------------
def epic_chunk_hash(token_ids: list[int]) -> str:
    """Byte-for-byte mirror of the connector's ``hash_chunk_tokens``.

    Kept in lockstep so the bench can statically predict which warmup/target
    chunks collide (used by ``--dry-run`` verification). If the connector's
    hashing changes, this must change with it.
    """
    h = hashlib.sha256()
    h.update(b"epic-chunk-v1")
    h.update(len(token_ids).to_bytes(4, "little"))
    for t in token_ids:
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def split_into_full_chunks(
    token_ids: list[int], chunk_size: int
) -> list[tuple[int, int, str]]:
    """Mirror of connector ``_split_prompt_into_chunks`` (whole chunks only)."""
    out: list[tuple[int, int, str]] = []
    n = len(token_ids)
    start = 0
    while start + chunk_size <= n:
        out.append(
            (start, chunk_size, epic_chunk_hash(token_ids[start : start + chunk_size]))
        )
        start += chunk_size
    return out


def round_up(x: int, m: int) -> int:
    if m <= 0:
        return x
    return ((x + m - 1) // m) * m


def effective_chunk_size(chunk_size: int, block_size: int) -> int:
    """Mirror the connector's chunk-size rounding (multiple of block_size)."""
    if chunk_size % block_size != 0:
        chunk_size = round_up(chunk_size, block_size)
    return max(chunk_size, block_size)


# ---------------------------------------------------------------------------
# Engine / kv-transfer config per mode
# ---------------------------------------------------------------------------
def kv_transfer_config_for_mode(
    mode: str,
    *,
    chunk_size: int,
    link_tokens: int,
    cpu_bytes: Optional[int] = None,
) -> Optional[dict]:
    """KV-transfer config dict for an engine in the given mode.

    full / prefix : no connector (None).
    epic          : EpicConnector with sparse-forward + fusion mask on.
    """
    if mode in ("full", "prefix"):
        return None
    if mode == "epic":
        extra: dict[str, Any] = {
            "epic_chunk_size": chunk_size,
            "epic_link_tokens": link_tokens,
            "epic_sparse_forward": True,
            "epic_fusion_mask": True,
        }
        if cpu_bytes is not None:
            extra["epic_cpu_bytes"] = cpu_bytes
        return {
            "kv_connector": "EpicConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": extra,
        }
    raise ValueError(f"unknown mode {mode!r}")


def engine_kwargs_for_mode(
    mode: str,
    *,
    model: str,
    chunk_size: int,
    link_tokens: int,
    block_size: int,
    max_model_len: int,
    gpu_mem_util: float,
    baseline_backend: Optional[str] = None,
) -> dict:
    """Construct LLM(**kwargs) for a mode.

    Fairness: all three modes use enforce_eager. The attention backend is set
    via the VLLM_ATTENTION_BACKEND env var (see ``apply_backend_env``) -- EPIC
    requires FLEX_ATTENTION; the baselines use it too by default so the only
    difference measured is the algorithm. ``baseline_backend`` (full/prefix
    only) lets a user opt a baseline into FLASH_ATTN for an extra reference.

    Prefix-caching toggle:
      * prefix / epic : enable_prefix_caching=True.
      * full          : enable_prefix_caching=False (cache-bust baseline) --
        every request does a full prefill. The data layer ALSO perturbs A per
        request as a belt-and-braces cache-bust (see data_prep).
    """
    kwargs: dict[str, Any] = dict(
        model=model,
        enforce_eager=True,
        block_size=block_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
    )
    if mode == "full":
        kwargs["enable_prefix_caching"] = False
    else:
        kwargs["enable_prefix_caching"] = True
    cfg = kv_transfer_config_for_mode(
        mode, chunk_size=chunk_size, link_tokens=link_tokens
    )
    if cfg is not None:
        kwargs["kv_transfer_config"] = cfg
    return kwargs


def backend_for_mode(mode: str, baseline_backend: Optional[str]) -> str:
    """Attention backend env value for a mode.

    epic           -> FLEX_ATTENTION (hard requirement; connector fails closed).
    full / prefix  -> FLEX_ATTENTION by default (fairness), or the user-supplied
                      baseline_backend (e.g. FLASH_ATTN) for an extra reference.
    """
    if mode == "epic":
        return "FLEX_ATTENTION"
    return baseline_backend or "FLEX_ATTENTION"


def apply_backend_env(mode: str, baseline_backend: Optional[str]) -> str:
    backend = backend_for_mode(mode, baseline_backend)
    os.environ["VLLM_ATTENTION_BACKEND"] = backend
    return backend


# ---------------------------------------------------------------------------
# JSONL request record
# ---------------------------------------------------------------------------
@dataclass
class BenchRequest:
    """One measured request, fully specified at the token level.

    The actual prompt fed to the model is rebuilt from the *text* fields; the
    token-id fields and lengths are for accounting + alignment verification.
    """

    req_id: str
    # Token ids are the SOURCE OF TRUTH fed to the model (TokensPrompt). Building
    # the prompt at the token level (concat of independently-tokenized segments)
    # guarantees B's ids are byte-identical and chunk-aligned between warmup and
    # target regardless of subword boundary effects -- the only robust way to
    # make the EPIC content hashes collide. Texts are kept for human inspection
    # and for the HF accuracy decode (answer-in-output check).
    a_ids: list[int]
    c_ids: list[int]
    b_ids: list[int]
    q_ids: list[int]
    b_text: str
    q_text: str
    needle_subject: str
    needle_answer: str
    a_tokens: int
    c_tokens: int
    b_tokens: int
    q_tokens: int
    # Warmup prompt that seeds B into the EPIC store at a chunk-aligned offset.
    warmup_ids: list[int]
    warmup_b_offset_tokens: int
    target_b_offset_tokens: int
    chunk_size: int
    link_tokens: int
    # Predicted: number of B chunks that should collide (hash match) between
    # warmup and target. Computed statically by data_prep for verification.
    predicted_b_chunk_hits: int
    # Task family for the accuracy scorer. "needle" (default, synthetic) uses
    # exact answer substring; "hf" uses answer-containment + SQuAD token-F1.
    # Defaulted so old JSONL (without these keys) still loads (back-compat).
    task_type: str = "needle"
    # Accepted ground-truth answers (aliases). For needle this is just the one
    # code; for hf it is the dataset's answer list. Defaulted to [] and lazily
    # back-filled from needle_answer in from_json for old records.
    gold_answers: list[str] = field(default_factory=list)

    def prompt_token_ids(self) -> list[int]:
        return list(self.a_ids) + list(self.c_ids) + list(self.b_ids) + list(self.q_ids)

    def warmup_token_ids(self) -> list[int]:
        return list(self.warmup_ids)

    def busted_prompt_token_ids(self, salt: int) -> list[int]:
        """Cache-bust variant for the 'full' mode: perturb the FIRST token of A
        (or of C if |A|==0) so the prefix-chain hash differs every request and
        nothing is reused -- a true full-prefill baseline even if some cache
        survives. Keeps total length identical so timing is comparable.
        """
        ids = self.prompt_token_ids()
        if not ids:
            return ids
        ids = list(ids)
        # XOR a small salt into the first token id, kept in a safe range.
        ids[0] = (ids[0] ^ (salt + 1)) % max(1, _BUST_VOCAB_CAP)
        return ids

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "BenchRequest":
        # Back-compat: drop unknown keys, supply defaults for new ones so an
        # older JSONL (no task_type/gold_answers) still loads, and a newer one
        # with extra fields does not blow up.
        import dataclasses as _dc

        known = {f.name for f in _dc.fields(BenchRequest)}
        kw = {k: v for k, v in d.items() if k in known}
        kw.setdefault("task_type", "needle")
        if not kw.get("gold_answers"):
            # Old records: the single needle answer is the only gold.
            na = kw.get("needle_answer", "")
            kw["gold_answers"] = [na] if na else []
        return BenchRequest(**kw)


# Conservative cap so a busted token id stays a valid id for tiny vocabs is not
# guaranteed; full mode primarily relies on enable_prefix_caching=False. The
# perturbation is a belt-and-braces secondary bust. Set high enough to be a
# no-op for real vocabularies (>32k).
_BUST_VOCAB_CAP = 32000


def write_jsonl(path: str, records: list[BenchRequest]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_json()) + "\n")


def read_jsonl(path: str) -> list[BenchRequest]:
    out: list[BenchRequest] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(BenchRequest.from_json(json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# Grid parsing
# ---------------------------------------------------------------------------
@dataclass
class Grid:
    a_lens: list[int]
    c_lens: list[int]
    b_lens: list[int]
    link: int = DEFAULT_LINK_TOKENS

    def cells(self):
        for a in self.a_lens:
            for c in self.c_lens:
                for b in self.b_lens:
                    yield (a, c, b)


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


# ---------------------------------------------------------------------------
# HF dataset loading (real QA tasks) -> normalized QA items
# ---------------------------------------------------------------------------
@dataclass
class QAItem:
    """One normalized QA example, dataset-agnostic.

    * ``question``            -- the user question text (becomes Q).
    * ``gold_passage``        -- the passage CONTAINING the answer (becomes B).
    * ``gold_answers``        -- accepted answer aliases (ground truth).
    * ``answer_text``         -- the canonical answer (for sentence-preserving
                                  truncation of the gold passage).
    * ``distractor_passages`` -- other (answer-free) passages, used to pad B to a
                                  chunk multiple and to fill shared-A / per-req-C.
    """

    question: str
    gold_passage: str
    gold_answers: list[str]
    answer_text: str
    distractor_passages: list[str] = field(default_factory=list)


# HF dataset id aliases. ``datasets`` 5.x rejects the bare canonical names
# ("squad", "hotpot_qa") -- they must be namespaced repo ids. Map friendly
# names to the real repo ids so the CLI stays ergonomic.
_HF_DATASET_ALIASES = {
    "squad": "rajpurkar/squad",
    "hotpot_qa": "hotpotqa/hotpot_qa",
    "hotpotqa": "hotpotqa/hotpot_qa",
}


def resolve_hf_dataset_id(name: str) -> str:
    return _HF_DATASET_ALIASES.get(name, name)


def load_qa_items(
    dataset: str,
    *,
    split: str = "validation",
    limit: int = 64,
) -> list[QAItem]:
    """Load and normalize a QA dataset into QAItems. Raises on unavailability so
    the caller can print clear offline guidance.

    Supports squad (single context + answer_start) and hotpot_qa distractor
    (multi-passage with supporting_facts marking the gold passage).
    """
    from datasets import load_dataset  # may raise ImportError offline

    repo = resolve_hf_dataset_id(dataset)
    sl = f"{split}[:{int(limit)}]"

    if repo.endswith("hotpot_qa"):
        ds = load_dataset(repo, "distractor", split=sl)
        return [_hotpot_to_qa(r) for r in ds]
    # Default: SQuAD-style schema.
    ds = load_dataset(repo, split=sl)
    out: list[QAItem] = []
    for r in ds:
        out.append(_squad_to_qa(r))
    return out


def _squad_to_qa(r: dict) -> QAItem:
    answers = r.get("answers", {}) or {}
    texts = list(answers.get("text", []) or [])
    # De-dup aliases preserving order.
    seen: set[str] = set()
    golds: list[str] = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            golds.append(t)
    answer_text = golds[0] if golds else ""
    return QAItem(
        question=r.get("question", "") or "",
        gold_passage=r.get("context", "") or "",
        gold_answers=golds,
        answer_text=answer_text,
        distractor_passages=[],  # squad has no extra passages in-record
    )


def _hotpot_to_qa(r: dict) -> QAItem:
    answer = r.get("answer", "") or ""
    ctx = r.get("context", {}) or {}
    titles = list(ctx.get("title", []) or [])
    sentences = list(ctx.get("sentences", []) or [])
    sf = r.get("supporting_facts", {}) or {}
    gold_titles = set(sf.get("title", []) or [])

    # Build one passage string per title; identify the gold passage as the
    # supporting passage that actually contains the answer text (yes/no answers
    # have no source span -> fall back to the first supporting passage).
    passages: list[tuple[str, str]] = []  # (title, text)
    for title, sents in zip(titles, sentences):
        passages.append((title, " ".join(sents)))

    gold_passage = ""
    norm_ans = normalize_answer(answer)
    # Prefer a supporting passage whose text contains the (normalized) answer.
    for title, text in passages:
        if title in gold_titles and norm_ans and norm_ans in normalize_answer(text):
            gold_passage = text
            break
    if not gold_passage:
        for title, text in passages:
            if title in gold_titles:
                gold_passage = text
                break
    if not gold_passage and passages:
        gold_passage = passages[0][1]

    distractors = [text for title, text in passages if text != gold_passage]
    return QAItem(
        question=r.get("question", "") or "",
        gold_passage=gold_passage,
        gold_answers=[answer] if answer else [],
        answer_text=answer,
        distractor_passages=distractors,
    )
