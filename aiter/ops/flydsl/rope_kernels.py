# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fused RoPE + KV Cache kernel wrapper for AITER.

Drop-in replacement for ``fused_qk_rope_reshape_and_cache`` from
``aiter.ops.triton.fusions.fused_kv_cache``, using the FlyDSL backend.

Typical speedup: 1.5x over Triton on MI355X (gfx950).

Usage:
    from aiter.ops.flydsl.rope_kernels import flydsl_fused_qk_rope_reshape_and_cache

    q_out, k_out, key_cache, value_cache = flydsl_fused_qk_rope_reshape_and_cache(
        q, k, v, key_cache, value_cache, slot_mapping, pos,
        cos, sin, k_scale, v_scale,
        is_neox=True, flash_layout=True,
    )
"""

import functools
import logging
import os
import sys

import torch

_LOGGER = logging.getLogger(__name__)

# Locate FlyDSL kernels directory.  Priority:
# 1. FLYDSL_ROOT env var (explicit override)
# 2. flydsl package installation directory (pip install flydsl ships kernels/)
# 3. Common development paths
_FLYDSL_ROOT = os.environ.get("FLYDSL_ROOT")
if _FLYDSL_ROOT is None:
    try:
        import flydsl as _flydsl_pkg

        _pkg_dir = os.path.dirname(os.path.dirname(_flydsl_pkg.__file__))
        _kernels_dir = os.path.join(_pkg_dir, "kernels")
        if os.path.isdir(_kernels_dir):
            _FLYDSL_ROOT = _pkg_dir
    except ImportError:
        pass

if _FLYDSL_ROOT and _FLYDSL_ROOT not in sys.path:
    sys.path.insert(0, _FLYDSL_ROOT)

from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module  # noqa: E402


@functools.lru_cache(maxsize=64)
def _get_launch_fn(
    head_dim, num_q_heads, num_kv_heads, block_size, flash_layout, dtype_str
):
    """Compile and cache FlyDSL kernel for given configuration."""
    return build_fused_rope_cache_module(
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        is_neox=True,
        flash_layout=flash_layout,
        dtype_str=dtype_str,
        pos_int64=True,  # AITER always passes int64
    )


def flydsl_fused_qk_rope_reshape_and_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_neox: bool,
    flash_layout: bool,
    apply_scale: bool = True,
    offs: torch.Tensor = None,
    q_out: torch.Tensor = None,
    k_out: torch.Tensor = None,
    output_zeros: bool = True,
    zeros_out: torch.Tensor = None,
):
    """FlyDSL drop-in replacement for fused_qk_rope_reshape_and_cache.

    Supports the same interface as the Triton version. Unsupported features
    (offsets, scale, zeros, non-NeoX) fall back to Triton automatically.
    """
    t, qh, d = q.shape
    _, kh, _ = k.shape

    # ── Fallback conditions: features FlyDSL doesn't support yet ──
    if not is_neox:
        _LOGGER.debug("FlyDSL RoPE: GPT-J style not supported, falling back to Triton")
        return _triton_fallback(
            q,
            k,
            v,
            key_cache,
            value_cache,
            slot_mapping,
            pos,
            cos,
            sin,
            k_scale,
            v_scale,
            is_neox,
            flash_layout,
            apply_scale,
            offs,
            q_out,
            k_out,
            output_zeros,
            zeros_out,
        )

    if offs is not None:
        _LOGGER.debug("FlyDSL RoPE: offsets not supported, falling back to Triton")
        return _triton_fallback(
            q,
            k,
            v,
            key_cache,
            value_cache,
            slot_mapping,
            pos,
            cos,
            sin,
            k_scale,
            v_scale,
            is_neox,
            flash_layout,
            apply_scale,
            offs,
            q_out,
            k_out,
            output_zeros,
            zeros_out,
        )

    if output_zeros or zeros_out is not None:
        _LOGGER.debug("FlyDSL RoPE: zeros output not supported, falling back to Triton")
        return _triton_fallback(
            q,
            k,
            v,
            key_cache,
            value_cache,
            slot_mapping,
            pos,
            cos,
            sin,
            k_scale,
            v_scale,
            is_neox,
            flash_layout,
            apply_scale,
            offs,
            q_out,
            k_out,
            output_zeros,
            zeros_out,
        )

    if apply_scale and (k_scale is not None or v_scale is not None):
        _LOGGER.debug("FlyDSL RoPE: KV scale not supported, falling back to Triton")
        return _triton_fallback(
            q,
            k,
            v,
            key_cache,
            value_cache,
            slot_mapping,
            pos,
            cos,
            sin,
            k_scale,
            v_scale,
            is_neox,
            flash_layout,
            apply_scale,
            offs,
            q_out,
            k_out,
            output_zeros,
            zeros_out,
        )

    # ── Determine dtype ──
    if q.dtype == torch.bfloat16:
        dtype_str = "bf16"
    elif q.dtype == torch.float16:
        dtype_str = "f16"
    else:
        _LOGGER.debug(
            f"FlyDSL RoPE: unsupported dtype {q.dtype}, falling back to Triton"
        )
        return _triton_fallback(
            q,
            k,
            v,
            key_cache,
            value_cache,
            slot_mapping,
            pos,
            cos,
            sin,
            k_scale,
            v_scale,
            is_neox,
            flash_layout,
            apply_scale,
            offs,
            q_out,
            k_out,
            output_zeros,
            zeros_out,
        )

    # ── Determine block_size from cache ──
    if flash_layout:
        block_size = key_cache.shape[1]
    else:
        block_size = key_cache.shape[3]

    # ── Allocate outputs if needed ──
    if q_out is None:
        q_out = torch.empty_like(q)
    if k_out is None:
        k_out = torch.empty_like(k)

    # ── Adapt cos/sin: AITER passes [max_pos, 1, 1, D//2] or [max_pos, D//2] ──
    # FlyDSL kernel uses its own layout algebra with cos_sin_layout = (half_dim, VEC_WIDTH)
    # so we just need contiguous 2D data. squeeze() is zero-copy.
    if cos.ndim == 4:
        cos_2d = cos.squeeze(1).squeeze(1)
        sin_2d = sin.squeeze(1).squeeze(1)
    elif cos.ndim == 2:
        cos_2d = cos
        sin_2d = sin
    else:
        # cos might be [max_pos, D] (full dim, reuse_freqs_front_part)
        cos_2d = cos
        sin_2d = sin

    # ── Get compiled kernel ──
    launch_fn = _get_launch_fn(d, qh, kh, block_size, flash_layout, dtype_str)

    # ── Launch ──
    stream = torch.cuda.current_stream()
    num_tokens = t

    launch_fn(
        q,
        k,
        v,
        pos,
        cos_2d,
        sin_2d,
        slot_mapping,
        key_cache,
        value_cache,
        q_out,
        k_out,
        num_tokens,
        stream=stream,
    )

    return q_out, k_out, key_cache, value_cache


def _triton_fallback(
    q,
    k,
    v,
    key_cache,
    value_cache,
    slot_mapping,
    pos,
    cos,
    sin,
    k_scale,
    v_scale,
    is_neox,
    flash_layout,
    apply_scale,
    offs,
    q_out,
    k_out,
    output_zeros,
    zeros_out,
):
    """Fall back to Triton implementation for unsupported features."""
    from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache

    return fused_qk_rope_reshape_and_cache(
        q,
        k,
        v,
        key_cache,
        value_cache,
        slot_mapping,
        pos,
        cos,
        sin,
        k_scale,
        v_scale,
        is_neox,
        flash_layout,
        apply_scale,
        offs,
        q_out,
        k_out,
        output_zeros,
        zeros_out,
    )
