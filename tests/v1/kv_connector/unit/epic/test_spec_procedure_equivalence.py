# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF-prototype (spec) procedure vs our EPIC procedure: EXACT equivalence (CPU).

The HF prototype spec (§3) prescribes:

  Phase A  every file segment -> ISOLATED forward with KV written at its
           ABSOLUTE positions [F_i, F_i+n_i); text below cached_end ->
           in-context prefill.
  Phase B  ONE combined forward over Q = (per file: first k_i tokens) +
           (text above cached_end), with non-contiguous abs positions and a
           position-based causal mask (q at p may attend cache slot c iff
           c <= p). New KV overwrites file-lead slots; isolated tails survive.

Our vLLM implementation formulates the same computation differently:

  * files are warmed ISOLATED AT POSITION 0 and stored (post-RoPE K + V),
  * on reuse, K is delta-rotated to the new absolute positions by the REAL
    ``PICRotator`` (R(new-old)·R(old) == R(new), the rotation-group identity),
  * M is derived by the REAL ``LegoLinkRecompute`` policy,
  * the sparse forward runs M rows at their true logical positions under a
    causal-over-logical-positions mask (FlexAttention ``logical_q_positions``).

These two formulations are MATHEMATICALLY IDENTICAL: an isolated forward is
position-shift-equivariant (attention depends on relative positions only, so
hidden states and V are position-independent and post-RoPE K differs by a pure
rotation), hence store-at-0 + delta-rotation == direct write at absolute
positions, layer by layer, exactly.

This test proves that equivalence end-to-end on a toy multi-layer RoPE
transformer in float64 (noise floor ~1e-12), driving the REAL PICRotator and
the REAL LegoLinkRecompute. Consequences:

  * FAILURE here == an algorithm-level bug in our formulation (M composition,
    PIC identity usage, mask semantics).
  * PASS here == the formulation is exact; any GPU-side corruption (e.g. the
    musique link<chunk_size collapse) must live in vLLM plumbing
    (scatter/kernel interop, positions wiring), not in the algorithm.

It also covers acceptance §8-1 at toy scale: k == file length degenerates BOTH
procedures to the vanilla dense prefill bit-exactly.
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
# Toy multi-layer RoPE attention model (float64, deterministic).
# ---------------------------------------------------------------------------

HIDDEN = 16
HEADS = 2
HEAD = 8  # rotary_dim == head_size (full rotary)
LAYERS = 3
BASE = 10000.0
DT = torch.float64

_INV_FREQ = 1.0 / (BASE ** (torch.arange(0, HEAD, 2, dtype=DT) / HEAD))


def _rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply neox RoPE at ``positions``. x: [T, HEADS, HEAD]."""
    freqs = positions.to(DT)[:, None] * _INV_FREQ[None, :]  # [T, HEAD//2]
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


def _forward_rows(
    h: torch.Tensor,
    positions: torch.Tensor,
    cache,
    layers,
) -> torch.Tensor:
    """Forward ``h`` rows at ``positions`` against/into ``cache``.

    Per layer: compute this layer's K/V for the rows, WRITE them into the
    cache at their positions (overwriting whatever was there -- this is the
    Phase-B overwrite), then attend each row q at position p over every
    FILLED cache slot c with c <= p (the spec's position-based causal mask ==
    our causal-over-logical-positions FlexAttention mask).
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
        # mask [T, total_len]: filled AND slot position <= query position.
        mask = c["filled"][None, :] & (slot_pos[None, :] <= positions[:, None])
        scores = torch.einsum("thd,chd->htc", q, c["k"]) / math.sqrt(HEAD)
        scores = scores.masked_fill(~mask[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("htc,chd->thd", probs, c["v"]).reshape(t, HIDDEN)
        h = h + out @ wo
    return h


# ---------------------------------------------------------------------------
# Suite layout: [text0 | fileA | text1 | fileB | query]   (infix mode:
# cached_end == s == len(text0); text1 is ABOVE cached_end -> recomputed).
# ---------------------------------------------------------------------------

S_TEXT0 = 5
N_FILE_A = 12
S_TEXT1 = 4
N_FILE_B = 9
S_QUERY = 6

OFF_A = S_TEXT0
OFF_T1 = OFF_A + N_FILE_A
OFF_B = OFF_T1 + S_TEXT1
OFF_Q = OFF_B + N_FILE_B
TOTAL = OFF_Q + S_QUERY  # == E


def _embeddings(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(TOTAL, HIDDEN, generator=g, dtype=DT)


def _spec_q_positions(k: int) -> list[int]:
    """Spec §3 Phase B walk: per file its first min(k, n_i) tokens; per text
    segment the part above cached_end. Strictly increasing, non-contiguous."""
    q: list[int] = []
    q += range(OFF_A, OFF_A + min(k, N_FILE_A))
    q += range(OFF_T1, OFF_T1 + S_TEXT1)
    q += range(OFF_B, OFF_B + min(k, N_FILE_B))
    q += range(OFF_Q, OFF_Q + S_QUERY)
    return q


class _Req:
    def __init__(self, n: int):
        self.prompt_token_ids = list(range(n))


def _our_m_positions(k: int) -> list[int]:
    """M from the REAL LegoLinkRecompute over the equivalent selection.

    Each file is a single content hit (hit length == file length), so the
    per-chunk link rule coincides with the spec's per-file k_i here.
    """
    selection = ReuseSelection(
        prefix_extent=S_TEXT0,
        non_prefix_hits=[
            NonPrefixHit(
                chunk_hash="A",
                prompt_offset=OFF_A,
                old_pos_start=0,
                length=N_FILE_A,
            ),
            NonPrefixHit(
                chunk_hash="B",
                prompt_offset=OFF_B,
                old_pos_start=0,
                length=N_FILE_B,
            ),
        ],
    )
    policy = LegoLinkRecompute(num_link_tokens=k, phase1_dense=False)
    plan = policy.plan_recompute(_Req(TOTAL), selection, block_size=16)
    return list(plan.recompute_offsets)


# ---------------------------------------------------------------------------
# Side A: the HF-prototype spec procedure (isolated at ABSOLUTE positions).
# ---------------------------------------------------------------------------


def _run_spec(embeds: torch.Tensor, layers, k: int):
    cache = _empty_cache(TOTAL)

    # Phase A step 2: text below cached_end, in-context at [0, s).
    _forward_rows(embeds[:S_TEXT0], torch.arange(S_TEXT0), cache, layers)

    # Phase A step 1: every file ISOLATED, at its absolute positions. Run in a
    # scratch cache (so it cannot attend outside itself) and copy the rows in.
    for off, n in ((OFF_A, N_FILE_A), (OFF_B, N_FILE_B)):
        scratch = _empty_cache(TOTAL)
        _forward_rows(embeds[off : off + n], torch.arange(off, off + n),
                      scratch, layers)
        for li in range(LAYERS):
            cache[li]["k"][off : off + n] = scratch[li]["k"][off : off + n]
            cache[li]["v"][off : off + n] = scratch[li]["v"][off : off + n]
            cache[li]["filled"][off : off + n] = True

    # Phase B: one combined forward over the non-contiguous Q.
    q_pos = torch.tensor(_spec_q_positions(k), dtype=torch.long)
    h_out = _forward_rows(embeds[q_pos], q_pos, cache, layers)
    return cache, h_out[-1]


# ---------------------------------------------------------------------------
# Side B: OUR procedure (warm at 0 -> store -> REAL PICRotator -> REAL M).
# ---------------------------------------------------------------------------


def _run_ours(embeds: torch.Tensor, layers, k: int):
    rotator = PICRotator(
        head_size=HEAD, rotary_dim=HEAD, base=BASE, is_neox_style=True, dtype=DT
    )
    cache = _empty_cache(TOTAL)

    # "Native prefix": text0 in-context (identical to spec Phase A step 2).
    _forward_rows(embeds[:S_TEXT0], torch.arange(S_TEXT0), cache, layers)

    # Warm each file ISOLATED AT POSITION 0 (a standalone prefill request) and
    # store its post-RoPE K + V per layer -- exactly what save_kv_layer keeps.
    store: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for name, off, n in (("A", OFF_A, N_FILE_A), ("B", OFF_B, N_FILE_B)):
        warm = _empty_cache(n)
        _forward_rows(embeds[off : off + n], torch.arange(n), warm, layers)
        store[name] = [
            (warm[li]["k"].clone(), warm[li]["v"].clone())
            for li in range(LAYERS)
        ]

    # Load: PIC delta-rotate stored K from old positions [0, n) to the new
    # absolute positions, scatter into this request's slots (_load_chunk).
    for name, off, n in (("A", OFF_A, N_FILE_A), ("B", OFF_B, N_FILE_B)):
        old_pos = torch.arange(n)
        new_pos = torch.arange(off, off + n)
        for li in range(LAYERS):
            k_stored, v_stored = store[name][li]
            cache[li]["k"][off : off + n] = rotator.rotate_keys(
                k_stored, old_pos, new_pos
            )
            cache[li]["v"][off : off + n] = v_stored
            cache[li]["filled"][off : off + n] = True

    # Sparse forward: M rows (REAL LegoLinkRecompute) at their true logical
    # positions under the causal-over-logical-positions mask.
    m_pos = torch.tensor(_our_m_positions(k), dtype=torch.long)
    h_out = _forward_rows(embeds[m_pos], m_pos, cache, layers)
    return cache, h_out[-1]


def _run_dense(embeds: torch.Tensor, layers):
    cache = _empty_cache(TOTAL)
    h_out = _forward_rows(embeds, torch.arange(TOTAL), cache, layers)
    return cache, h_out[-1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

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


def test_m_composition_matches_spec_q_walk():
    """The REAL LegoLinkRecompute M == the spec §3 Phase B Q walk, for every k.

    (The spec's Q always ends with the query segment, so the policy's explicit
    ``{N-1}`` union is absorbed and the sets coincide exactly.)
    """
    for k in (0, 1, 3, 7, N_FILE_B, N_FILE_A, 100):
        assert _our_m_positions(k) == _spec_q_positions(k), f"k={k}"


@pytest.mark.parametrize("k", [0, 3, 7])
def test_ours_equals_spec_exactly(k):
    """Store-at-0 + PIC delta + sparse-M forward == spec isolated-at-absolute
    + combined Phase B, bit-tight in float64 (cache AND last-row output)."""
    embeds = _embeddings()
    layers = _make_layers()

    cache_spec, out_spec = _run_spec(embeds, layers, k)
    cache_ours, out_ours = _run_ours(embeds, layers, k)

    _assert_caches_equal(cache_spec, cache_ours)
    torch.testing.assert_close(out_spec, out_ours, atol=ATOL, rtol=0)

    # Every slot in [0, E) is filled after Phase B (spec §4 layout invariant).
    for li in range(LAYERS):
        assert bool(cache_spec[li]["filled"].all())


def test_full_recompute_degenerates_to_dense():
    """Acceptance §8-1 at toy scale: k >= every n_i -> both procedures ==
    vanilla dense prefill (reuse fully overwritten, nothing approximate)."""
    embeds = _embeddings()
    layers = _make_layers()
    k = max(N_FILE_A, N_FILE_B)

    cache_dense, out_dense = _run_dense(embeds, layers)
    cache_spec, out_spec = _run_spec(embeds, layers, k)
    cache_ours, out_ours = _run_ours(embeds, layers, k)

    _assert_caches_equal(cache_spec, cache_dense)
    _assert_caches_equal(cache_ours, cache_dense)
    torch.testing.assert_close(out_spec, out_dense, atol=ATOL, rtol=0)
    torch.testing.assert_close(out_ours, out_dense, atol=ATOL, rtol=0)


def test_partial_k_is_a_real_approximation():
    """Non-vacuousness guard: at k < n_i the isolated tails make BOTH
    procedures diverge from dense by a clearly-nonzero amount (if this ever
    became ~0 the equivalence tests above would be testing nothing)."""
    embeds = _embeddings()
    layers = _make_layers()

    _, out_dense = _run_dense(embeds, layers)
    _, out_spec = _run_spec(embeds, layers, 3)

    assert (out_spec - out_dense).abs().max() > 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
