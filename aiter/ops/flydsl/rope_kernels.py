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

# Set FLYDSL_ROPE_DISABLE=1 to force Triton fallback (useful for perf A/B comparisons).
_DISABLED = _os.environ.get("FLYDSL_ROPE_DISABLE", "0") == "1"
# Set FLYDSL_ROPE_DEBUG=1 to log the first call's shapes/dtypes (one-shot).
_DEBUG = _os.environ.get("FLYDSL_ROPE_DEBUG", "0") == "1"
_debug_printed = False

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
    head_dim, num_q_heads, num_kv_heads, block_size, flash_layout, dtype_str,
    apply_scale, reuse_freqs_front_part, pos_dtype, x_size,
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
        apply_scale=apply_scale,
        reuse_freqs_front_part=reuse_freqs_front_part,
        pos_dtype=pos_dtype,
        x_size=x_size,
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
    global _debug_printed
    t, qh, d = q.shape
    _, kh, _ = k.shape

    if _DEBUG and not _debug_printed:
        _debug_printed = True
        print(
            f"[FlyDSL RoPE DEBUG] q={tuple(q.shape)} dtype={q.dtype} "
            f"k={tuple(k.shape)} v={tuple(v.shape)}\n"
            f"  key_cache={tuple(key_cache.shape)} dtype={key_cache.dtype} "
            f"  value_cache={tuple(value_cache.shape)}\n"
            f"  cos={tuple(cos.shape)} sin={tuple(sin.shape)}\n"
            f"  pos={tuple(pos.shape)} dtype={pos.dtype} "
            f"  slot_mapping={tuple(slot_mapping.shape)} dtype={slot_mapping.dtype}\n"
            f"  is_neox={is_neox} flash_layout={flash_layout} "
            f"apply_scale={apply_scale} output_zeros={output_zeros} offs={offs is not None}"
        )

    # Common fallback args — avoids repeating the 18-arg list at every guard.
    if _DISABLED:
        return _triton_fallback(q, k, v, key_cache, value_cache, slot_mapping, pos,
                                cos, sin, k_scale, v_scale, is_neox, flash_layout,
                                apply_scale, offs, q_out, k_out, output_zeros, zeros_out)
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

    # -- Adapt cos/sin to 2D [max_pos, cos_dim] --
    if cos.ndim == 4:
        cos_2d = cos.squeeze(1).squeeze(1)
        sin_2d = sin.squeeze(1).squeeze(1)
    else:
        cos_2d = cos
        sin_2d = sin

    # -- Fallback conditions --
    if not is_neox:
        _LOGGER.debug("FlyDSL RoPE: GPT-J style not supported, falling back")
        return _triton_fallback(*_fb)

    if offs is not None:
        _LOGGER.debug("FlyDSL RoPE: offsets not supported, falling back")
        return _triton_fallback(*_fb)

    if output_zeros or zeros_out is not None:
        _LOGGER.debug("FlyDSL RoPE: zeros output not supported, falling back")
        return _triton_fallback(*_fb)

    # Detect half-dim [max_pos, D//2] or full-dim [max_pos, D] cos/sin.
    if cos_2d.shape[-1] == d // 2:
        reuse_freqs_front_part = True
    elif cos_2d.shape[-1] == d:
        reuse_freqs_front_part = False
    else:
        _LOGGER.debug(f"FlyDSL RoPE: unexpected cos/sin shape {cos.shape} for head_dim={d}, falling back")
        return _triton_fallback(*_fb)

    # -- Determine dtype --
    if q.dtype == torch.bfloat16:
        dtype_str = "bf16"
    elif q.dtype == torch.float16:
        dtype_str = "f16"
    else:
        _LOGGER.debug(f"FlyDSL RoPE: unsupported dtype {q.dtype}, falling back")
        return _triton_fallback(*_fb)

    # apply_scale=True only when fp8 KV cache; k_scale/v_scale must be non-None tensors
    _apply_scale = bool(apply_scale and k_scale is not None and v_scale is not None)

    # -- Determine block_size and x_size --
    block_size = key_cache.shape[1] if flash_layout else key_cache.shape[3]
    _x_size = 16 if flash_layout else key_cache.shape[-1]

    # -- Allocate outputs if needed --
    if q_out is None:
        q_out = torch.empty_like(q)
    if k_out is None:
        k_out = torch.empty_like(k)

    # Zero-copy int64 → int32 reinterpret: .view(torch.int32) changes dtype
    # metadata without launching a CUDA cast kernel.  An int64 tensor of
    # shape [T] becomes int32 of shape [2*T]; on little-endian the low 32
    # bits of element i sit at index 2*i.  The kernel compensates via
    # stride-2 indexing when pos_dtype == "i64".
    if pos.dtype == torch.int64:
        pos_i32 = pos.view(torch.int32)       # shape [2*T], zero-copy
        pos_dtype = "i64"
    else:
        pos_i32 = pos                         # already int32
        pos_dtype = "i32"

    if slot_mapping.dtype == torch.int64:
        slot_i32 = slot_mapping.view(torch.int32)  # shape [2*T], zero-copy
    else:
        slot_i32 = slot_mapping                    # already int32

    # -- Scale tensors: FlyDSL requires at least 1D; scalars must be reshaped --
    if _apply_scale:
        k_scale_arg = k_scale if k_scale.dtype == torch.float32 else k_scale.float()
        v_scale_arg = v_scale if v_scale.dtype == torch.float32 else v_scale.float()
        if k_scale_arg.ndim == 0:
            k_scale_arg = k_scale_arg.reshape(1)
        if v_scale_arg.ndim == 0:
            v_scale_arg = v_scale_arg.reshape(1)
    else:
        # Kernel compiled with apply_scale=False ignores these args entirely.
        # torch.empty avoids a fill kernel (unlike torch.ones) — safe in CUDAGraph.
        _placeholder = torch.empty(1, dtype=torch.float32, device=q.device)
        k_scale_arg = _placeholder
        v_scale_arg = _placeholder

    # -- Get compiled kernel --
    launch_fn = _get_launch_fn(d, qh, kh, block_size, flash_layout, dtype_str,
                               _apply_scale, reuse_freqs_front_part, pos_dtype, _x_size)

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
        k_scale_arg,
        v_scale_arg,
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
