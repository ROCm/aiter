# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL GDN gated RMSNorm + out_proj decode interface."""

import functools

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.gdn_gated_rmsnorm_out_proj import (
    CHUNK,
    MAX_TOKENS,
    build_gdn_gated_rmsnorm_out_proj_module,
    check_gdn_gated_rmsnorm_out_proj_shape,
)
from .kernels.tensor_shim import _run_compiled

# The kernel uses v_dot2_f32_bf16 and was validated on gfx950 only.
_SUPPORTED_ARCHS = ("gfx950",)

# Only head_dim 128, the value head size of Qwen3.5 and Qwen3-Next, is validated.
_SUPPORTED_HEAD_DIMS = (128,)

_SUPPORTED_ACTIVATIONS = ("silu", "swish")

# Tile per token count, (cols_per_wave, waves_per_block). These are the fastest tiles
# measured on gfx950 inside HIP graphs with K=2048 and N=8192. At K=1024 with N=4096
# and at K=4096 with N=8192 they are at most 9% slower than the fastest tile of that
# shape. Token counts not listed use _DEFAULT_TILE, which was the fastest tile for 7
# and 8 tokens at K=1024.
_TILE_BY_TOKENS = {1: (1, 8), 2: (2, 8), 3: (2, 4), 4: (2, 8), 5: (2, 8)}
_DEFAULT_TILE = (2, 8)


# The kernel was measured on gfx950 inside HIP graphs against the norm kernel plus the
# vLLM GEMM at K=1024, 2048, and 4096. Every workgroup computes the gated norm of all
# tokens before its first multiply, so that part grows with num_tokens * K, while the
# separate GEMM is bound by the weight read and hardly grows. The kernel is faster up
# to 8 tokens at K=1024 and up to 5 tokens at K=2048 and K=4096, and even or slower
# above that. A K between two measured values uses the limit of the larger one.
_MIN_K = 1024
# No K above 4096 was measured. After the barrier each lane also reads its W values and
# its y values of all tokens, (cols_per_wave + num_tokens) * K / 128 VGPRs, so a larger
# K risks spills. The final ISA at K=4096 has no spills up to 5 tokens.
_MAX_K = 4096


def _max_tokens(k):
    """Return the largest token count at which the kernel runs for this K, or 0."""
    if not _MIN_K <= k <= _MAX_K:
        return 0
    return MAX_TOKENS if k <= 1024 else 5


# y for all tokens lives in LDS as bf16. The cap keeps several workgroups per CU.
_MAX_LDS_BYTES = 64 * 1024

# FlyDSL keeps buffer offsets in 32 bits, so the W buffer must stay below 2 GB.
_MAX_W_BYTES = 2**31


@functools.lru_cache(maxsize=256)
def _pick_tile(m, k, n, head_dim):
    """Return (cols_per_wave, waves_per_block) for this shape, or None."""
    preferred = _TILE_BY_TOKENS.get(m, _DEFAULT_TILE)
    # A tile needs N to be a multiple of cols_per_wave * waves_per_block. These tiles
    # cover an N that is a multiple of 8 or 4 but not of 16.
    fallbacks = [(2, 4), (1, 8), (1, 4)]
    for cols, waves in [preferred] + fallbacks:
        reason = check_gdn_gated_rmsnorm_out_proj_shape(m, k, n, head_dim, cols, waves)
        if reason is None:
            return cols, waves
    return None


@functools.lru_cache(maxsize=256)
def flydsl_gdn_gated_rmsnorm_out_proj_supported(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    out_features: int,
    dtype: torch.dtype = torch.bfloat16,
    activation: str = "silu",
) -> bool:
    """Return True if :func:`flydsl_gdn_gated_rmsnorm_out_proj` supports this shape.

    ``num_heads`` is the number of value heads on this rank and ``out_features`` is
    ``N`` of the out_proj weight. Shapes where the kernel was measured to be slower
    than the separate norm and GEMM return False, so a caller should run those unfused.
    """
    if _runtime_gfx() not in _SUPPORTED_ARCHS:
        return False
    if dtype != torch.bfloat16 or activation not in _SUPPORTED_ACTIVATIONS:
        return False
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        return False
    k = num_heads * head_dim
    if not 1 <= num_tokens <= _max_tokens(k):
        return False
    if k % CHUNK or num_tokens * k * 2 > _MAX_LDS_BYTES:
        return False
    if out_features * k * 2 >= _MAX_W_BYTES:
        return False
    return _pick_tile(num_tokens, k, out_features, head_dim) is not None


def _runtime_gfx():
    """Return the arch of the GPU this process runs on, or "unknown"."""
    # get_gfx() follows GPU_ARCHS, which a build for several archs sets, so it can
    # name another arch than the GPU in use. v_dot2_f32_bf16 does not exist on gfx942.
    try:
        return get_gfx_runtime()
    except KeyError:
        return "unknown"


@functools.cache
def _get_launcher(m, k, n, head_dim, eps, cols_per_wave, waves_per_block):
    return build_gdn_gated_rmsnorm_out_proj_module(
        m, k, n, head_dim, eps, cols_per_wave, waves_per_block
    )


@functools.lru_cache(maxsize=128)
def _validate(
    x_shape,
    x_dtype,
    x_contiguous,
    z_shape,
    z_dtype,
    z_contiguous,
    nw_shape,
    nw_dtype,
    nw_contiguous,
    w_shape,
    w_dtype,
    w_contiguous,
    out_shape,
    out_dtype,
    out_contiguous,
    devices,
    activation,
):
    if len(x_shape) != 3:
        raise ValueError(f"x must be [num_tokens, num_heads, head_dim], got {x_shape}")
    m, num_heads, head_dim = x_shape
    k = num_heads * head_dim
    if z_shape != x_shape:
        raise ValueError(f"z must have the shape of x {x_shape}, got {z_shape}")
    if nw_shape != (head_dim,):
        raise ValueError(f"norm_weight must be [{head_dim}], got {nw_shape}")
    if len(w_shape) != 2 or w_shape[1] != k:
        raise ValueError(f"weight must be [N, {k}], got {w_shape}")
    n = w_shape[0]
    if out_shape != (m, n):
        raise ValueError(f"out must be [{m}, {n}], got {out_shape}")
    for name, dtype in (
        ("x", x_dtype),
        ("z", z_dtype),
        ("norm_weight", nw_dtype),
        ("weight", w_dtype),
        ("out", out_dtype),
    ):
        if dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16, got {dtype}")
    # The kernel reads every tensor as a dense row-major buffer. A strided view would
    # be read at the wrong addresses without any error.
    for name, contiguous in (
        ("x", x_contiguous),
        ("z", z_contiguous),
        ("norm_weight", nw_contiguous),
        ("weight", w_contiguous),
        ("out", out_contiguous),
    ):
        if not contiguous:
            raise ValueError(f"{name} must be contiguous")
    if len(set(devices)) != 1 or devices[0].type != "cuda":
        raise ValueError("all tensors must be on the same GPU")
    if not flydsl_gdn_gated_rmsnorm_out_proj_supported(
        m, num_heads, head_dim, n, x_dtype, activation
    ):
        raise ValueError(
            "flydsl_gdn_gated_rmsnorm_out_proj does not support "
            f"num_tokens={m}, num_heads={num_heads}, head_dim={head_dim}, N={n}, "
            f"activation={activation!r} on {_runtime_gfx()}. Check "
            "flydsl_gdn_gated_rmsnorm_out_proj_supported() first"
        )


def flydsl_gdn_gated_rmsnorm_out_proj(
    x: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    activation: str = "silu",
) -> torch.Tensor:
    """Gated RMSNorm per head followed by the out_proj GEMM, as one decode kernel.

    Computes ``out = y.flatten(-2) @ weight.T`` with
    ``y = bf16(x * rsqrt(mean(x^2, -1) + eps) * norm_weight * silu(z))``, the same
    math as ``RMSNormGated(head_dim, group_size=None, norm_before_gate=True,
    activation="silu")`` followed by an unquantized bf16 linear layer.

    Args:
        x: ``[num_tokens, num_heads, head_dim]`` GDN core output.
        z: ``[num_tokens, num_heads, head_dim]`` output gate.
        norm_weight: ``[head_dim]`` RMSNormGated weight, shared by all heads.
        weight: ``[N, num_heads * head_dim]`` out_proj weight.
        eps: RMSNorm epsilon.
        out: optional preallocated ``[num_tokens, N]`` result.
        activation: gate activation, ``"silu"`` (or its alias ``"swish"``).

    All tensors must be contiguous bf16 on one GPU. Call
    :func:`flydsl_gdn_gated_rmsnorm_out_proj_supported` first. The first call for a
    new shape JIT-compiles the kernel, so run each token count once before capturing
    a HIP graph.
    """
    if x.dim() != 3 or weight.dim() != 2:
        raise ValueError(
            "x must be [num_tokens, num_heads, head_dim] and weight must be [N, K], "
            f"got {tuple(x.shape)} and {tuple(weight.shape)}"
        )
    m, num_heads, head_dim = x.shape
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _validate(
        tuple(x.shape),
        x.dtype,
        x.is_contiguous(),
        tuple(z.shape),
        z.dtype,
        z.is_contiguous(),
        tuple(norm_weight.shape),
        norm_weight.dtype,
        norm_weight.is_contiguous(),
        tuple(weight.shape),
        weight.dtype,
        weight.is_contiguous(),
        tuple(out.shape),
        out.dtype,
        out.is_contiguous(),
        (x.device, z.device, norm_weight.device, weight.device, out.device),
        activation,
    )
    k = num_heads * head_dim
    cols_per_wave, waves_per_block = _pick_tile(m, k, n, head_dim)
    launcher = _get_launcher(
        m, k, n, head_dim, float(eps), cols_per_wave, waves_per_block
    )
    _run_compiled(
        launcher,
        x.view(-1),
        z.view(-1),
        norm_weight,
        weight,
        out,
        # Bound to the tensors' device. current_stream() without an argument returns
        # the stream of the current device, which may be a different GPU.
        torch.cuda.current_stream(x.device),
    )
    return out
