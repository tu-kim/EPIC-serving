# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PIC delta-rotation math tests (CPU-only, neox-style RoPE)."""

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.epic.pic import PICRotator


def _neox_rope_reference(
    x: torch.Tensor, positions: torch.Tensor, base: float, rotary_dim: int
) -> torch.Tensor:
    """Reference neox-style RoPE applied directly at absolute `positions`.

    x: [num_tokens, num_heads, head_size]. Rotates only the first rotary_dim.
    Mirrors vLLM's RotaryEmbedding.forward_static convention exactly.
    """
    num_tokens = x.shape[0]
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    )
    t = positions.to(torch.float64)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [num_tokens, rotary_dim//2]
    cos = freqs.cos()
    sin = freqs.sin()

    out = x.clone().to(torch.float64)
    rot = out[..., :rotary_dim]
    x1, x2 = torch.chunk(rot, 2, dim=-1)
    c = cos.unsqueeze(1)  # [num_tokens, 1, rotary_dim//2]
    s = sin.unsqueeze(1)
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    out[..., :rotary_dim] = torch.cat((o1, o2), dim=-1)
    return out.to(x.dtype)


def test_delta_rotation_matches_direct_rope():
    """R(p_new - p_old) . R(p_old) . k == R(p_new) . k for full rotary dim."""
    torch.manual_seed(0)
    num_tokens, num_heads, head_size = 5, 2, 16
    base = 10000.0
    rotary_dim = head_size

    k_raw = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float64)
    old_pos = torch.tensor([0, 1, 2, 3, 4])
    new_pos = torch.tensor([100, 50, 7, 200, 3])

    # K as stored in the cache: RoPE'd at old positions.
    k_old = _neox_rope_reference(k_raw, old_pos, base, rotary_dim)
    # Ground truth: same raw K RoPE'd directly at new positions.
    k_new_truth = _neox_rope_reference(k_raw, new_pos, base, rotary_dim)

    rot = PICRotator(
        head_size=head_size,
        rotary_dim=rotary_dim,
        base=base,
        is_neox_style=True,
        dtype=torch.float64,
    )
    k_new_pic = rot.rotate_keys(k_old, old_pos, new_pos)

    assert torch.allclose(k_new_pic, k_new_truth, atol=1e-6, rtol=1e-5), (
        (k_new_pic - k_new_truth).abs().max().item()
    )


def test_delta_rotation_partial_rotary():
    """Pass-through tail (head_size > rotary_dim) is preserved and aligned."""
    torch.manual_seed(1)
    num_tokens, num_heads, head_size = 4, 3, 16
    rotary_dim = 8
    base = 10000.0

    k_raw = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float64)
    old_pos = torch.tensor([10, 11, 12, 13])
    new_pos = torch.tensor([0, 5, 99, 1])

    k_old = _neox_rope_reference(k_raw, old_pos, base, rotary_dim)
    k_new_truth = _neox_rope_reference(k_raw, new_pos, base, rotary_dim)

    rot = PICRotator(
        head_size=head_size,
        rotary_dim=rotary_dim,
        base=base,
        is_neox_style=True,
        dtype=torch.float64,
    )
    k_new_pic = rot.rotate_keys(k_old, old_pos, new_pos)

    assert torch.allclose(k_new_pic, k_new_truth, atol=1e-6, rtol=1e-5)
    # The pass-through tail must be untouched relative to the original cached K.
    assert torch.allclose(
        k_new_pic[..., rotary_dim:], k_old[..., rotary_dim:], atol=1e-9
    )


def test_zero_delta_is_identity():
    torch.manual_seed(2)
    k = torch.randn(3, 2, 16, dtype=torch.float64)
    pos = torch.tensor([7, 8, 9])
    rot = PICRotator(
        head_size=16, rotary_dim=16, base=10000.0, dtype=torch.float64
    )
    out = rot.rotate_keys(k, pos, pos)
    assert torch.allclose(out, k, atol=1e-9)


def test_negative_delta_supported():
    """new < old must work (analytic cos/sin, not cache index)."""
    torch.manual_seed(3)
    k_raw = torch.randn(2, 1, 16, dtype=torch.float64)
    old_pos = torch.tensor([500, 600])
    new_pos = torch.tensor([1, 2])
    k_old = _neox_rope_reference(k_raw, old_pos, 10000.0, 16)
    k_truth = _neox_rope_reference(k_raw, new_pos, 10000.0, 16)
    rot = PICRotator(head_size=16, rotary_dim=16, base=10000.0, dtype=torch.float64)
    assert torch.allclose(rot.rotate_keys(k_old, old_pos, new_pos), k_truth, atol=1e-6)


# Llama-3.1/3.2 rope_scaling type="llama3" (GPU blocker: loads were disabled
# for these models). llama3 scaling is a static inv_freq transform; positions
# are not scaled, so delta rotation stays exact.
_LLAMA31_SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def test_llama3_inv_freq_matches_vllm_reference():
    """PICRotator's llama3 inv_freq must equal vLLM's
    Llama3RotaryEmbedding._compute_inv_freq exactly (float64 vs float32 ref)."""
    from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
        Llama3RotaryEmbedding,
    )

    head, rdim, base = 128, 128, 500000.0
    # object.__new__: skip CustomOp __init__ (needs vLLM config context);
    # _compute_inv_freq only reads the attrs set below.
    ref = object.__new__(Llama3RotaryEmbedding)
    ref.rotary_dim = rdim
    ref.scaling_factor = _LLAMA31_SCALING["factor"]
    ref.low_freq_factor = _LLAMA31_SCALING["low_freq_factor"]
    ref.high_freq_factor = _LLAMA31_SCALING["high_freq_factor"]
    ref.orig_max_position = _LLAMA31_SCALING[
        "original_max_position_embeddings"
    ]
    ref_inv = ref._compute_inv_freq(base).to(torch.float64)

    rot = PICRotator(
        head_size=head, rotary_dim=rdim, base=base,
        rope_scaling=dict(_LLAMA31_SCALING),
    )
    assert torch.allclose(rot.register_inv_freq, ref_inv, rtol=1e-6, atol=0.0)
    # the transform must actually change something (not the unscaled freqs)
    unscaled = PICRotator(head_size=head, rotary_dim=rdim, base=base)
    assert not torch.allclose(rot.register_inv_freq, unscaled.register_inv_freq)


def test_llama3_delta_rotation_matches_direct_rope():
    """R(new-old)·R(old)·k == R(new)·k with llama3-scaled frequencies."""
    torch.manual_seed(7)
    head, rdim, base = 64, 64, 500000.0
    rot = PICRotator(
        head_size=head, rotary_dim=rdim, base=base,
        dtype=torch.float64, rope_scaling=dict(_LLAMA31_SCALING),
    )
    n, kv_heads = 16, 2
    k_raw = torch.randn(n, kv_heads, head, dtype=torch.float64)
    old_pos = torch.arange(1000, 1000 + n)
    new_pos = torch.arange(40, 40 + n)  # negative delta too

    inv = rot.register_inv_freq

    def direct(k, pos):
        # same math as _neox_rope_reference but with llama3-scaled inv_freq
        freqs = torch.einsum("i,j->ij", pos.to(torch.float64), inv)
        cos, sin = freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)
        out = k.clone().to(torch.float64)
        x1, x2 = torch.chunk(out[..., :rdim], 2, dim=-1)
        out[..., :rdim] = torch.cat(
            (x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1
        )
        return out

    k_old = direct(k_raw, old_pos)
    k_expected = direct(k_raw, new_pos)
    k_rotated = rot.rotate_keys(k_old, old_pos, new_pos)
    assert torch.allclose(k_rotated, k_expected, atol=1e-9)
