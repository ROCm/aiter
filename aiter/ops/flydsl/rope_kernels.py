# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fused RoPE + KV Cache kernel wrapper for AITER.

Drop-in replacement for ``fused_qk_rope_reshape_and_cache`` from
``aiter.ops.triton.fusions.fused_kv_cache``, using the FlyDSL backend.

Typical speedup: ~1.45x over Triton on MI300 (gfx942).

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
import os as _os
import sys as _sys

import torch

_LOGGER = logging.getLogger(__name__)

# Prefer dsl2 source kernel (editable install) over vendored copy,
# because the dsl2 kernels/ package includes kernels_common side effects
# needed for correct buffer_ops initialization.

try:
    import flydsl as _flydsl_pkg

    # Walk up from flydsl/__init__.py to find dsl2 repo root with kernels/
    _cur = _os.path.dirname(_flydsl_pkg.__file__)
    _dsl_root = None
    for _ in range(4):
        _cur = _os.path.dirname(_cur)
        if _os.path.isfile(
            _os.path.join(_cur, "kernels", "fused_rope_cache_kernel.py")
        ):
            _dsl_root = _cur
            break
    if _dsl_root and _os.path.isfile(
        _os.path.join(_dsl_root, "kernels", "fused_rope_cache_kernel.py")
    ):
        if _dsl_root not in _sys.path:
            _sys.path.insert(0, _dsl_root)
except Exception:
    pass

try:
    from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module
except ImportError:
    from aiter.ops.flydsl.kernels.fused_rope_cache_kernel import (
        build_fused_rope_cache_module,
    )


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

    # Common fallback args — avoids repeating the 18-arg list at every guard.
    _fb = (
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

    # -- Fallback conditions --
    # FlyDSL soffset codegen has issues for T>1 (prefill). Decode (T=1) is reliable.
    if t > 1:
        _LOGGER.debug("FlyDSL RoPE: T>1 not stable, falling back to Triton")
        return _triton_fallback(*_fb)

    # FlyDSL kernel only supports half-dim cos/sin (reuse_freqs_front_part=True).
    if cos.shape[-1] != d // 2:
        _LOGGER.debug("FlyDSL RoPE: full-dim cos/sin not supported, falling back")
        return _triton_fallback(*_fb)

    if not flash_layout:
        _LOGGER.debug("FlyDSL RoPE: non-flash layout not supported, falling back")
        return _triton_fallback(*_fb)

    if not is_neox:
        _LOGGER.debug("FlyDSL RoPE: GPT-J style not supported, falling back")
        return _triton_fallback(*_fb)

    if offs is not None:
        _LOGGER.debug("FlyDSL RoPE: offsets not supported, falling back")
        return _triton_fallback(*_fb)

    if output_zeros or zeros_out is not None:
        _LOGGER.debug("FlyDSL RoPE: zeros output not supported, falling back")
        return _triton_fallback(*_fb)

    if apply_scale and (k_scale is not None or v_scale is not None):
        _LOGGER.debug("FlyDSL RoPE: KV scale not supported, falling back")
        return _triton_fallback(*_fb)

    # -- Determine dtype --
    if q.dtype == torch.bfloat16:
        dtype_str = "bf16"
    elif q.dtype == torch.float16:
        dtype_str = "f16"
    else:
        _LOGGER.debug(f"FlyDSL RoPE: unsupported dtype {q.dtype}, falling back")
        return _triton_fallback(*_fb)

    # -- Determine block_size (flash_layout guaranteed True here) --
    block_size = key_cache.shape[1]

    # -- Allocate outputs if needed --
    if q_out is None:
        q_out = torch.empty_like(q)
    if k_out is None:
        k_out = torch.empty_like(k)

    # -- Adapt cos/sin to 2D [max_pos, D//2] --
    if cos.ndim == 4:
        cos_2d = cos.squeeze(1).squeeze(1)
        sin_2d = sin.squeeze(1).squeeze(1)
    elif cos.ndim == 2:
        cos_2d = cos
        sin_2d = sin
    else:
        cos_2d = cos
        sin_2d = sin

    # -- Cast positions and slot_mapping to int32 (FlyDSL kernel uses i32) --
    pos_i32 = pos.to(torch.int32) if pos.dtype != torch.int32 else pos
    slot_i32 = (
        slot_mapping.to(torch.int32)
        if slot_mapping.dtype != torch.int32
        else slot_mapping
    )

    # -- Get compiled kernel --
    launch_fn = _get_launch_fn(d, qh, kh, block_size, flash_layout, dtype_str)

    # -- Launch --
    stream = torch.cuda.current_stream()
    num_tokens = t

    launch_fn(
        q,
        k,
        v,
        pos_i32,
        cos_2d,
        sin_2d,
        slot_i32,
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
