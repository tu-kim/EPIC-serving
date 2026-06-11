# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PIC (Position-Independent Caching) re-rotary for EPIC.

EPIC original (vllm_epic/vllm/model_executor/models/llama.py) re-rotated cached
K by calling ``self.rotary_emb(org_pos, fake_q, old_kv[0])`` -- i.e. it re-ran
the *full* RoPE forward with a throwaway ("fake") query so the kernel would
rotate K to the original absolute position ``org_pos``. That worked because in
EPIC the cached K was *un-rotated raw* K stashed in ``self.hack_kv`` and then
RoPE'd to the desired position on every reuse.

In Phase 1 here we instead persist *already-rotated* K (the K that lives in the
paged cache after a normal prefill). To place such a chunk at a new position we
apply a **delta rotation**:

    K_new = R(p_new - p_old) . K_old           (K_old = R(p_old) . k_raw)
          = R(p_new - p_old) . R(p_old) . k_raw
          = R(p_new) . k_raw                    (RoPE is a rotation group)

So we never need the raw K and never need the EPIC ``fake_q`` hack -- we apply
RoPE directly to K only, with the *signed* delta ``p_new - p_old`` (which can be
negative). Because deltas can be negative we cannot index the model's precomputed
``cos_sin_cache`` (built only for positions ``[0, max_pos)``); we compute cos/sin
analytically here from ``inv_freq``.

This module is deliberately self-contained / CPU-friendly so the delta-rotation
identity can be unit-tested without a GPU or a full model load.
"""

from typing import Any

import torch

from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb


class PICRotator:
    """Delta-rotates cached keys from their old positions to new positions.

    Only the rotary subspace (``rotary_dim``) of each head is rotated; the
    pass-through tail (``head_size - rotary_dim``) is left untouched, matching
    ``RotaryEmbedding.forward_static``.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        base: float,
        is_neox_style: bool = True,
        dtype: torch.dtype = torch.float32,
        rope_scaling: dict[str, Any] | None = None,
    ) -> None:
        assert rotary_dim % 2 == 0, "rotary_dim must be even"
        assert rotary_dim <= head_size
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        # EPIC change: compute inv_freq ourselves (rather than reusing the
        # model's cos_sin_cache) so we can evaluate cos/sin at signed deltas.
        # Matches RotaryEmbeddingBase._compute_inv_freq.
        # Keep inv_freq in float64 so high-position deltas stay accurate; the
        # final cos/sin are downcast to the key dtype in _cos_sin_for_delta.
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
        )
        # rope_scaling: Phase 1 supports plain "linear" scaling (the common
        # Llama variant). YaRN / longrope etc. are TODO -- we assert-out rather
        # than silently produce wrong rotations.
        self._scaling_factor = 1.0
        if rope_scaling is not None:
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type in ("linear", "su", None):
                factor = rope_scaling.get("factor", 1.0)
                if rope_type == "linear":
                    self._scaling_factor = float(factor)
            elif rope_type == "llama3":
                # Llama-3.1/3.2 rope scaling is a STATIC per-dim transformation
                # of inv_freq (positions themselves are NOT scaled), so the
                # delta-rotation identity R(a)R(b)=R(a+b) still holds exactly
                # per frequency dim -- PIC works unchanged with the transformed
                # inv_freq. Formula mirrors vLLM's
                # Llama3RotaryEmbedding._compute_inv_freq
                # (model_executor/layers/rotary_embedding/llama3_rope.py:33-53)
                # and is verified against it by unit test.
                import math

                factor = float(rope_scaling["factor"])
                low = float(rope_scaling["low_freq_factor"])
                high = float(rope_scaling["high_freq_factor"])
                orig_max = int(rope_scaling["original_max_position_embeddings"])
                low_freq_wavelen = orig_max / low
                high_freq_wavelen = orig_max / high
                wave_len = 2 * math.pi / inv_freq
                if low != high:
                    smooth = (orig_max / wave_len - low) / (high - low)
                else:
                    smooth = torch.zeros_like(wave_len)
                inv_freq = torch.where(
                    wave_len < high_freq_wavelen,
                    inv_freq,
                    torch.where(
                        wave_len > low_freq_wavelen,
                        inv_freq / factor,
                        (1 - smooth) * inv_freq / factor + smooth * inv_freq,
                    ),
                )
            else:
                raise NotImplementedError(
                    f"PICRotator does not support rope_scaling "
                    f"type={rope_type!r}; supported: linear, llama3. "
                    f"See epic/PHASE2.md."
                )
        self.register_inv_freq = inv_freq
        # NOTE: we deliberately do NOT instantiate ApplyRotaryEmb (a CustomOp,
        # which requires an active vLLM config context at construction). We only
        # call its pure staticmethod `forward_static`, so PICRotator is usable
        # standalone (e.g. in unit tests) without a config context.

    def _cos_sin_for_delta(
        self, delta: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin of shape [num_tokens, rotary_dim//2] for `delta`.

        `delta` is a 1-D float/int tensor of per-token signed position deltas.
        """
        # Use the highest precision available (>= float32) for the trig so large
        # |delta| at high positions doesn't lose accuracy; downcast at the end.
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        inv_freq = self.register_inv_freq.to(device=device, dtype=compute_dtype)
        t = delta.to(device=device, dtype=compute_dtype) / self._scaling_factor
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [num_tokens, rotary_dim//2]
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    def rotate_keys(
        self,
        key: torch.Tensor,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return K re-rotated from `old_positions` to `new_positions`.

        Args:
            key: [num_tokens, num_kv_heads, head_size] (already RoPE'd at
                old_positions).
            old_positions: [num_tokens] int positions the K was rotated to.
            new_positions: [num_tokens] int target positions.

        Returns:
            Tensor of the same shape as `key`, rotated to `new_positions`.
        """
        assert key.dim() == 3, "expected [num_tokens, num_kv_heads, head_size]"
        num_tokens = key.shape[0]
        assert old_positions.shape[0] == num_tokens
        assert new_positions.shape[0] == num_tokens

        delta = new_positions.to(torch.float32) - old_positions.to(torch.float32)
        cos, sin = self._cos_sin_for_delta(delta, key.device, key.dtype)

        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        # ApplyRotaryEmb.forward_static expects x: [..., seq, heads, head_size]
        # and cos/sin: [seq, head_size//2]. Our key_rot is [seq, heads, rdim].
        key_rot = ApplyRotaryEmb.forward_static(
            key_rot, cos, sin, is_neox_style=self.is_neox_style
        )
        if key_pass.shape[-1] == 0:
            return key_rot
        return torch.cat((key_rot, key_pass), dim=-1)

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "PICRotator":
        """Build a PICRotator from a VllmConfig (Llama-family models)."""
        hf = vllm_config.model_config.hf_text_config
        head_size = vllm_config.model_config.get_head_size()
        rotary_dim = getattr(hf, "rotary_dim", head_size)
        # Some configs expose partial_rotary_factor.
        partial = getattr(hf, "partial_rotary_factor", 1.0)
        if partial != 1.0:
            rotary_dim = int(head_size * partial)
        base = getattr(hf, "rope_theta", 10000.0)
        rope_scaling = getattr(hf, "rope_scaling", None)
        dtype = vllm_config.model_config.dtype
        return cls(
            head_size=head_size,
            rotary_dim=rotary_dim,
            base=base,
            is_neox_style=True,
            dtype=dtype,
            rope_scaling=rope_scaling,
        )
