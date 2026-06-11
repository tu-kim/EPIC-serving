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
