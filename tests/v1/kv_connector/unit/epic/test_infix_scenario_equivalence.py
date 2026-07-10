# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The user's canonical infix scenario, proven end-to-end on CPU (float64).

Prompt layout (6 segments):

    [ A | B | C | D | F | G ]
      A : cached prefix (native KV cache / in-context)     -> reused, not in M
      B : new text                                          -> in M
      C : code file, KV pre-built separately ("fileKV")     -> loaded + PIC
      D : new text                                          -> in M
      F : second code file (fileKV)                         -> loaded + PIC
      G : new query text                                    -> in M

Desired behavior under evaluation:
  1. Only A is reusable by a plain prefix cache; infix reuse additionally pulls
     C and F from the fileKV store.
  2. B + D + G plus the EPIC recompute tokens of C/F (LegoLink heads + last
     token) form ONE concatenated matrix and run in ONE forward step.
  3. RoPE is applied per segment at the true logical positions; C/F keys are
     PIC delta-rotated from their stored positions to their new positions.
  4. Causal masking holds by logical position: a B-row query sees only
     positions before it (never C/D/F/G), a D-row sees A+B+C+its D prefix, etc.
  5. The whole A+B+C+D+F+G prefill completes in one step.

This module drives the REAL ``PICRotator`` and REAL ``LegoLinkRecompute``
through a float64 toy multi-layer RoPE transformer (noise floor ~1e-12) and
checks our procedure against (a) the HF-prototype spec procedure
(isolated-at-absolute-positions + combined Phase B) and (b) the vanilla dense
prefill. See test_spec_procedure_equivalence.py for the underlying math
(store-at-0 + delta rotation == isolated-at-absolute, layer by layer).
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.metadata import (
    NonPrefixHit,
)
from vllm.distributed.kv_transfer.kv_connector.v1.epic.pic import PICRotator
from vllm.distributed.kv_transfer.kv_connector.v1.epic.reuse_strategy import (
    LegoLinkRecompute,
    ReuseSelection,
)
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

# ---------------------------------------------------------------------------
# Toy multi-layer RoPE attention model (float64, deterministic) -- same
# construction as test_spec_procedure_equivalence.py.
# ---------------------------------------------------------------------------

HIDDEN = 16
HEADS = 2
HEAD = 8  # rotary_dim == head_size (full rotary)
LAYERS = 3
BASE = 10000.0
DT = torch.float64

_INV_FREQ = 1.0 / (BASE ** (torch.arange(0, HEAD, 2, dtype=DT) / HEAD))


def _rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    freqs = positions.to(DT)[:, None] * _INV_FREQ[None, :]
    return ApplyRotaryEmb.forward_static(
        x, freqs.cos(), freqs.sin(), is_neox_style=True
    )


def _make_layers(seed: int = 0):
    g = torch.Generator().manual_seed(seed)

    def w():
        return torch.randn(HIDDEN, HIDDEN, generator=g, dtype=DT) / math.sqrt(
            HIDDEN
        )

    return [(w(), w(), w(), w()) for _ in range(LAYERS)]


def _empty_cache(total_len: int):
    return [
        {
            "k": torch.zeros(total_len, HEADS, HEAD, dtype=DT),
            "v": torch.zeros(total_len, HEADS, HEAD, dtype=DT),
            "filled": torch.zeros(total_len, dtype=torch.bool),
        }
        for _ in range(LAYERS)
    ]


def _forward_rows(h, positions, cache, layers):
    """Forward ``h`` rows at ``positions``: per layer write K/V into the cache
    at the rows' logical positions, then attend each query at position p over
    every FILLED slot c with c <= p -- the causal-over-logical-positions mask
    (FlexAttention ``logical_q_positions`` semantics). Returns final hidden.
    """
    total_len = cache[0]["k"].shape[0]
    slot_pos = torch.arange(total_len)
    for li, (wq, wk, wv, wo) in enumerate(layers):
        t = h.shape[0]
        q = _rope((h @ wq).reshape(t, HEADS, HEAD), positions)
        k = _rope((h @ wk).reshape(t, HEADS, HEAD), positions)
        v = (h @ wv).reshape(t, HEADS, HEAD)
        c = cache[li]
        c["k"][positions] = k
        c["v"][positions] = v
        c["filled"][positions] = True
        mask = c["filled"][None, :] & (slot_pos[None, :] <= positions[:, None])
        scores = torch.einsum("thd,chd->htc", q, c["k"]) / math.sqrt(HEAD)
        scores = scores.masked_fill(~mask[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("htc,chd->thd", probs, c["v"]).reshape(t, HIDDEN)
        h = h + out @ wo
    return h


# ---------------------------------------------------------------------------
# The 6-segment layout: [ A | B | C | D | F | G ].
# ---------------------------------------------------------------------------

S_A = 6  # cached prefix (native)
S_B = 5  # new text
N_C = 12  # code file (fileKV)
S_D = 4  # new text
N_F = 9  # code file (fileKV)
S_G = 7  # query

OFF_B = S_A
OFF_C = OFF_B + S_B
OFF_D = OFF_C + N_C
OFF_F = OFF_D + S_D
OFF_G = OFF_F + N_F
TOTAL = OFF_G + S_G


def _embeddings(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(TOTAL, HIDDEN, generator=g, dtype=DT)


class _Req:
    def __init__(self, n: int):
        self.prompt_token_ids = list(range(n))


def _selection() -> ReuseSelection:
    """A is the contiguous prefix; C and F are non-prefix fileKV hits."""
    return ReuseSelection(
        prefix_extent=S_A,
        non_prefix_hits=[
            NonPrefixHit("C", prompt_offset=OFF_C, old_pos_start=0, length=N_C),
            NonPrefixHit("F", prompt_offset=OFF_F, old_pos_start=0, length=N_F),
        ],
    )


def _m_positions(k: int) -> list[int]:
    plan = LegoLinkRecompute(num_link_tokens=k, phase1_dense=False).plan_recompute(
        _Req(TOTAL), _selection(), block_size=16
    )
    return list(plan.recompute_offsets)


def _expected_m(k: int) -> list[int]:
    """B + link(C) + D + link(F) + G, per the user's scenario."""
    m: list[int] = []
    m += range(OFF_B, OFF_B + S_B)
    m += range(OFF_C, OFF_C + min(k, N_C))
    m += range(OFF_D, OFF_D + S_D)
    m += range(OFF_F, OFF_F + min(k, N_F))
    m += range(OFF_G, OFF_G + S_G)
    return m  # G ends at TOTAL-1, so the policy's {N-1} union is absorbed.


# ---------------------------------------------------------------------------
# Procedures.
# ---------------------------------------------------------------------------


def _run_dense(embeds, layers):
    cache = _empty_cache(TOTAL)
    h = _forward_rows(embeds, torch.arange(TOTAL), cache, layers)
    return cache, h


def _run_ours(embeds, layers, k: int):
    """Our vLLM formulation of the infix prefill.

    A: in-context prefill (== what the native prefix cache holds).
    C/F: warmed ISOLATED AT POSITION 0 (a standalone fileKV build), stored as
         post-RoPE K + V, then PIC delta-rotated to their absolute offsets and
         scattered -- exactly EpicConnector._load_chunk.
    M:   REAL LegoLinkRecompute; one combined forward of the M rows at their
         true logical positions under the causal-by-position mask.
    Returns (cache, M positions, M outputs).
    """
    rotator = PICRotator(
        head_size=HEAD, rotary_dim=HEAD, base=BASE, is_neox_style=True, dtype=DT
    )
    cache = _empty_cache(TOTAL)

    # A: native prefix (in-context).
    _forward_rows(embeds[:S_A], torch.arange(S_A), cache, layers)

    # fileKV: C and F warmed isolated at position 0, stored per layer.
    store = {}
    for name, off, n in (("C", OFF_C, N_C), ("F", OFF_F, N_F)):
        warm = _empty_cache(n)
        _forward_rows(embeds[off : off + n], torch.arange(n), warm, layers)
        store[name] = [
            (warm[li]["k"].clone(), warm[li]["v"].clone())
            for li in range(LAYERS)
        ]

    # Load: PIC delta-rotation old->new absolute positions, scatter.
    for name, off, n in (("C", OFF_C, N_C), ("F", OFF_F, N_F)):
        old_pos = torch.arange(n)
        new_pos = torch.arange(off, off + n)
        for li in range(LAYERS):
            k_stored, v_stored = store[name][li]
            cache[li]["k"][off : off + n] = rotator.rotate_keys(
                k_stored, old_pos, new_pos
            )
            cache[li]["v"][off : off + n] = v_stored
            cache[li]["filled"][off : off + n] = True

    # ONE combined sparse forward of M.
    m_pos = torch.tensor(_m_positions(k), dtype=torch.long)
    h_m = _forward_rows(embeds[m_pos], m_pos, cache, layers)
    return cache, m_pos, h_m


def _run_spec(embeds, layers, k: int):
    """HF-prototype spec: files isolated at ABSOLUTE positions + combined
    Phase B over the same Q walk."""
    cache = _empty_cache(TOTAL)
    _forward_rows(embeds[:S_A], torch.arange(S_A), cache, layers)
    for off, n in ((OFF_C, N_C), (OFF_F, N_F)):
        scratch = _empty_cache(TOTAL)
        _forward_rows(
            embeds[off : off + n], torch.arange(off, off + n), scratch, layers
        )
        for li in range(LAYERS):
            cache[li]["k"][off : off + n] = scratch[li]["k"][off : off + n]
            cache[li]["v"][off : off + n] = scratch[li]["v"][off : off + n]
            cache[li]["filled"][off : off + n] = True
    q_pos = torch.tensor(_expected_m(k), dtype=torch.long)
    h_q = _forward_rows(embeds[q_pos], q_pos, cache, layers)
    return cache, q_pos, h_q


ATOL = 1e-10


def _assert_caches_equal(ca, cb, atol=ATOL):
    for li in range(LAYERS):
        assert torch.equal(ca[li]["filled"], cb[li]["filled"])
        sel = ca[li]["filled"]
        torch.testing.assert_close(
            ca[li]["k"][sel], cb[li]["k"][sel], atol=atol, rtol=0
        )
        torch.testing.assert_close(
            ca[li]["v"][sel], cb[li]["v"][sel], atol=atol, rtol=0
        )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_m_is_exactly_b_d_g_plus_file_recompute_heads():
    """Requirement 2: the concatenated forward matrix is B + D + G plus the
    EPIC recompute tokens of C and F -- nothing more (A, C/F bodies excluded)."""
    for k in (0, 1, 3, N_F, N_C, 100):
        assert _m_positions(k) == _expected_m(k), f"k={k}"


@pytest.mark.parametrize("k", [0, 3, 7])
def test_infix_equals_spec_procedure(k):
    """Requirements 3+4+5: one-step sparse prefill with PIC-rotated fileKV and
    causal-by-position masking is EXACTLY the spec's isolated-at-absolute +
    combined Phase B (float64, atol 1e-10) -- cache contents AND M outputs."""
    embeds = _embeddings()
    layers = _make_layers()
    cache_ours, m_ours, h_ours = _run_ours(embeds, layers, k)
    cache_spec, m_spec, h_spec = _run_spec(embeds, layers, k)
    assert torch.equal(m_ours, m_spec)
    _assert_caches_equal(cache_ours, cache_spec)
    torch.testing.assert_close(h_ours, h_spec, atol=ATOL, rtol=0)
    # Requirement 5 invariant: after the single step, every slot [0, N) is
    # filled -- the whole A+B+C+D+F+G prefill completed in one step.
    for li in range(LAYERS):
        assert bool(cache_ours[li]["filled"].all())


def test_full_recompute_degenerates_to_dense():
    """k >= max file length: the infix procedure IS the vanilla dense prefill
    (every fileKV token recomputed and overwritten)."""
    embeds = _embeddings()
    layers = _make_layers()
    k = max(N_C, N_F)
    cache_dense, h_dense = _run_dense(embeds, layers)
    cache_ours, m_pos, h_ours = _run_ours(embeds, layers, k)
    _assert_caches_equal(cache_ours, cache_dense)
    torch.testing.assert_close(
        h_ours, h_dense[m_pos], atol=ATOL, rtol=0
    )


@pytest.mark.parametrize("k", [0, 3])
def test_b_rows_match_dense_exactly_causality(k):
    """Requirement 4, behaviorally: every B row's context ([0, pos)) consists
    ONLY of exact A KV + earlier B rows, so B outputs must equal the vanilla
    dense prefill EXACTLY even at k < file length. Any causal-mask leak of
    C/D/F/G into B, or a wrong RoPE position on a B row, breaks this."""
    embeds = _embeddings()
    layers = _make_layers()
    _, h_dense = _run_dense(embeds, layers)
    _, m_pos, h_ours = _run_ours(embeds, layers, k)

    b_rows = [i for i, p in enumerate(m_pos.tolist()) if OFF_B <= p < OFF_C]
    assert len(b_rows) == S_B
    b_pos = m_pos[b_rows]
    torch.testing.assert_close(
        h_ours[b_rows], h_dense[b_pos], atol=ATOL, rtol=0
    )
    # D/G rows sit downstream of approximate fileKV context; at k=0 they must
    # genuinely diverge from dense (non-vacuousness of the B check above).
    if k == 0:
        g_rows = [i for i, p in enumerate(m_pos.tolist()) if p >= OFF_G]
        assert (h_ours[g_rows] - h_dense[m_pos[g_rows]]).abs().max() > 1e-8


def test_filekv_reuse_is_position_independent():
    """Requirement 3 (PIC): the same stored fileKV chunk is reusable at a
    DIFFERENT absolute offset -- rebuild the prompt with a longer B segment
    (shifting C/D/F/G) and the procedure still matches the spec exactly."""
    global S_B, OFF_C, OFF_D, OFF_F, OFF_G, TOTAL
    saved = (S_B, OFF_C, OFF_D, OFF_F, OFF_G, TOTAL)
    try:
        S_B = 11  # longer B -> every downstream segment shifts by +6.
        OFF_C = OFF_B + S_B
        OFF_D = OFF_C + N_C
        OFF_F = OFF_D + S_D
        OFF_G = OFF_F + N_F
        TOTAL = OFF_G + S_G
        embeds = _embeddings(seed=2)
        layers = _make_layers(seed=3)
        cache_ours, m_ours, h_ours = _run_ours(embeds, layers, 3)
        cache_spec, m_spec, h_spec = _run_spec(embeds, layers, 3)
        assert torch.equal(m_ours, m_spec)
        _assert_caches_equal(cache_ours, cache_spec)
        torch.testing.assert_close(h_ours, h_spec, atol=ATOL, rtol=0)
    finally:
        (S_B, OFF_C, OFF_D, OFF_F, OFF_G, TOTAL) = saved


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
