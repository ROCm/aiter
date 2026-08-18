# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Literal

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.fusions.fused_clamp_act_mul import (
    _fused_clamp_silu_mul_kernel as _fused_clamp_silu_mul_gluon_kernel,
)
from aiter.ops.triton._triton_kernels.fusions.fused_clamp_act_mul import (
    _fused_clamp_silu_mul_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# Architectures that have a gluon kernel
_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


def _is_gluon_available():
    """True if the current GPU arch has a Gluon port of this kernel."""
    try:
        arch = get_arch()
        return any(s in arch for s in _GLUON_SUPPORTED_ARCHS)
    except Exception:  # noqa: BLE001
        return False


def _get_config(M: int, N: int, block_size_n: int) -> dict:
    # For gluon
    if M == 128 and N == 4096:
        return {
            "ROWS_PER_PROG": 1,
            "BLOCK_SIZE_M": 1,
            "BLOCK_SIZE_N": 512,
            "num_warps": 2,
            "waves_per_eu": 4,
        }
    if M == 16384 and N == 4096:
        return {
            "ROWS_PER_PROG": 2,
            "BLOCK_SIZE_M": 2,
            "BLOCK_SIZE_N": 256,
            "num_warps": 1,
            "waves_per_eu": 5,
        }
    if M == 4096 and N == 512:
        return {
            "ROWS_PER_PROG": 1,
            "BLOCK_SIZE_M": 4,
            "BLOCK_SIZE_N": 512,
            "num_warps": 1,
            "waves_per_eu": 4,
        }
    if M == 16384 and N == 8192:
        return {
            "ROWS_PER_PROG": 1,
            "BLOCK_SIZE_M": 2,
            "BLOCK_SIZE_N": 512,
            "num_warps": 1,
            "waves_per_eu": 6,
        }
    if M == 8192 and N == 8192:
        return {
            "ROWS_PER_PROG": 3,
            "BLOCK_SIZE_M": 4,
            "BLOCK_SIZE_N": 512,
            "num_warps": 4,
            "waves_per_eu": 6,
        }
    return {
        "ROWS_PER_PROG": 1,
        "BLOCK_SIZE_M": 1,
        "BLOCK_SIZE_N": block_size_n,
        "num_warps": 1,
        "waves_per_eu": 1,
    }


def fused_clamp_act_mul(
    inp: torch.Tensor,
    out: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    swiglu_limit: float = 0,
    activation: Literal["silu", "gelu", "gelu_tanh"] = "silu",
    weights: torch.Tensor | None = None,
    dtype_quant: torch.dtype | None = None,
    transpose_scale: bool = False,
    quant_block_size: int = 128,
    scale_dtype_fmt: Literal["fp32", "ue8m0"] = "fp32",
    shuffle_scale: bool = False,
    block_size_n: int | None = None,
    backend: str | None = None,
):
    """
    Fusion of chunk + activation + multiply + quantize,
    optional FP8 quant/shuffle.

    Splits inp into two halves, gate and up, then computes:

        out = act(clamp(gate)) * clamp(up) * weights

    (clamp and weights if applicable)

    Args:
        inp: input, shape [M, 2N], contiguous. Splits to [M,N] for both gate/up
        out: output buffer, shape [M, N]
        scale: buffer for quant scales
        swiglu_limit: clamp threshold, to skip set <= 0
        activation: which activation to apply to gate, silu/gelu/gelu_tanh
        weights: [M, 1] (broadcast over N) or [M, N] or None
        dtype_quant: dtype to quantize output to, None => output keeps input dtype.
        transpose_scale: store scales as [N_blocks, M] instead of [M, N_blocks]
        quant_block_size: how many columns share one scale, default 128.
            N_blocks = ceil(N / quant_block_size).
        scale_dtype_fmt: scale format, options "fp32" or "ue8m0"
            (ue8m0 requires quant_block_size=32 and FP8 E4M3 dtype_quant)
        shuffle_scale: write scales in the preshuffled layout, padded to
            [ceil(M/256)*256, ceil(N_blocks/8)*8].
            Requires scale_dtype_fmt="ue8m0"
        backend: specify "triton" or "gluon".
            None picks "gluon" on supported architectures, otherwise "triton"

    Returns:
        out if no quant, otherwise (out, scale).

    Constraints:
        N must be a power of two, at least 128, and a multiple of 128.
    """

    # setup inputs
    assert inp.dim() == 2
    M, D = inp.shape
    assert D % 2 == 0
    n_half = D // 2

    HAS_QUANT = dtype_quant is not None

    # validate scale format, pick storage dtype
    assert scale_dtype_fmt in ("fp32", "ue8m0")
    if scale_dtype_fmt == "ue8m0":
        assert HAS_QUANT, "scale_dtype_fmt='ue8m0' requires dtype_quant"
        assert (
            quant_block_size == 32
        ), f"ue8m0 requires quant_block_size=32 got {quant_block_size}"
        assert dtype_quant in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ), f"ue8m0 requires fp8 e4m3, got {dtype_quant}"
        assert not (
            shuffle_scale and transpose_scale
        ), "shuffle_scale incompatible with transpose_scale"
        _scale_storage_dtype = torch.uint8
    else:
        assert (
            not shuffle_scale
        ), "shuffle_scale only valid with scale_dtype_fmt='ue8m0'"
        _scale_storage_dtype = torch.float32

    if HAS_QUANT:
        # handle quant out
        if out is None:
            out = torch.empty((M, n_half), dtype=dtype_quant, device=inp.device)
        else:
            assert out.shape == (M, n_half)
            if out.dtype != dtype_quant:
                _LOGGER.info(
                    "fused_clamp_act_mul: dtype_quant=%s ignored; using out.dtype=%s",
                    dtype_quant,
                    out.dtype,
                )

        # one scale per group of quant_block_size columns, split N blocks
        num_blocks = (n_half + quant_block_size - 1) // quant_block_size

        # determine scale shape
        if shuffle_scale:
            scale_m_pad = (M + 255) // 256 * 256
            scale_n_pad = (num_blocks + 7) // 8 * 8
            if scale is None:
                scale = torch.empty(
                    (scale_m_pad, scale_n_pad),
                    dtype=_scale_storage_dtype,
                    device=inp.device,
                )
            else:
                assert scale.shape == (scale_m_pad, scale_n_pad)
        elif scale is None:
            if transpose_scale:
                scale = torch.empty(
                    (num_blocks, M), dtype=_scale_storage_dtype, device=inp.device
                )
            else:
                scale = torch.empty(
                    (M, num_blocks), dtype=_scale_storage_dtype, device=inp.device
                )
        else:
            if transpose_scale:
                assert scale.shape == (num_blocks, M)
            else:
                assert scale.shape == (M, num_blocks)
    else:
        if out is None:
            out = torch.empty((M, n_half), dtype=inp.dtype, device=inp.device)
        else:
            assert out.shape == (M, n_half)

    assert n_half >= 128
    assert n_half % 128 == 0

    # default block width is the whole row, can change at config
    BLOCK_SIZE_N = triton.next_power_of_2(n_half)

    # handle weight constants
    HAVE_WEIGHTS = weights is not None
    if HAVE_WEIGHTS:
        assert weights.is_cuda and weights.is_contiguous()
        assert weights.shape[0] == M
        if weights.shape[1] == 1:
            WEIGHT_BROADCAST = True
        else:
            assert weights.shape[1] == n_half
            WEIGHT_BROADCAST = False
    else:
        WEIGHT_BROADCAST = False

    # saturation bound for the quantized output
    if HAS_QUANT:
        DTYPE_MAX = (
            torch.finfo(out.dtype).max
            if torch.is_floating_point(out)
            else float(torch.iinfo(out.dtype).max)
        )
    else:
        DTYPE_MAX = 0.0

    HAVE_SWIGLU_CLAMP = swiglu_limit > 0

    # determine scale strides
    scale_n_pad = 0
    if HAS_QUANT:
        if shuffle_scale:
            scale_row_stride = scale.stride(0)
            scale_col_stride = scale.stride(1)
            num_bs_cols = scale.shape[1]
            scale_n_pad = scale.shape[1]
        elif transpose_scale:
            scale_row_stride = scale.stride(1)
            scale_col_stride = scale.stride(0)
            num_bs_cols = scale.shape[0]
        else:
            scale_row_stride = scale.stride(0)
            scale_col_stride = scale.stride(1)
            num_bs_cols = scale.shape[1]
        scale_arg = scale
    else:
        scale_row_stride = 0
        scale_col_stride = 0
        scale_arg = inp

    # choose backend
    if backend is None:
        backend = "gluon" if _is_gluon_available() else "triton"
    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    if backend == "gluon":
        # Config if applicable, otherwise defaults
        config = _get_config(M, n_half, BLOCK_SIZE_N)
        ROWS_PER_PROG = config["ROWS_PER_PROG"]
        BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
        BLOCK_SIZE_N = config["BLOCK_SIZE_N"]
        num_warps = config["num_warps"]
        waves_per_eu = config["waves_per_eu"]

        # ensure quant block can be safely applied to N tile
        assert BLOCK_SIZE_N % quant_block_size == 0, (
            f"BLOCK_SIZE_N ({BLOCK_SIZE_N}) must be a multiple of "
            f"quant_block_size ({quant_block_size})"
        )

        # triton/gluon
        assert (
            BLOCK_SIZE_M & (BLOCK_SIZE_M - 1) == 0
        ), f"BLOCK_SIZE_M ({BLOCK_SIZE_M}) must be a power of two"
        assert (
            BLOCK_SIZE_N & (BLOCK_SIZE_N - 1) == 0
        ), f"BLOCK_SIZE_N ({BLOCK_SIZE_N}) must be a power of two"

        assert (
            _is_gluon_available()
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"

        # (M chunks * rows to process, N tiles)
        _fused_clamp_silu_mul_gluon_kernel[
            (
                triton.cdiv(M, ROWS_PER_PROG * BLOCK_SIZE_M),
                triton.cdiv(n_half, BLOCK_SIZE_N),
            )
        ](
            inp,
            out,
            scale_arg,
            weights if HAVE_WEIGHTS else inp,
            M,
            n_half,
            inp.stride(0),
            inp.stride(1),
            out.stride(0),
            out.stride(1),
            scale_row_stride,
            scale_col_stride,
            weights.stride(0) if HAVE_WEIGHTS else 0,
            weights.stride(1) if HAVE_WEIGHTS else 0,
            swiglu_limit,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            QUANT_BLOCK_SIZE=quant_block_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            SCALE_FMT=scale_dtype_fmt,
            DTYPE_MAX=DTYPE_MAX,
            DTYPE_MIN=-DTYPE_MAX,
            HAVE_WEIGHTS=HAVE_WEIGHTS,
            WEIGHT_BROADCAST=WEIGHT_BROADCAST,
            HAVE_SWIGLU_CLAMP=HAVE_SWIGLU_CLAMP,
            HAS_QUANT=HAS_QUANT,
            ACTIVATION=activation,
            SHUFFLE=shuffle_scale,
            SCALE_N_PAD=scale_n_pad,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            ROWS_PER_PROG=ROWS_PER_PROG,
            cache_modifier=".cg",
        )
    else:
        # only for triton
        if BLOCK_SIZE_N <= 512:
            num_warps = 1
        elif BLOCK_SIZE_N <= 2048:
            num_warps = 4
        else:
            num_warps = 8

        _fused_clamp_silu_mul_kernel[(M,)](
            inp,
            out,
            scale_arg,
            weights if HAVE_WEIGHTS else inp,
            M,
            n_half,
            inp.stride(0),
            inp.stride(1),
            out.stride(0),
            out.stride(1),
            scale_row_stride,
            scale_col_stride,
            weights.stride(0) if HAVE_WEIGHTS else 0,
            weights.stride(1) if HAVE_WEIGHTS else 0,
            swiglu_limit,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            QUANT_BLOCK_SIZE=quant_block_size,
            SCALE_FMT=scale_dtype_fmt,
            DTYPE_MAX=DTYPE_MAX,
            DTYPE_MIN=-DTYPE_MAX,
            HAVE_WEIGHTS=HAVE_WEIGHTS,
            WEIGHT_BROADCAST=WEIGHT_BROADCAST,
            HAVE_SWIGLU_CLAMP=HAVE_SWIGLU_CLAMP,
            HAS_QUANT=HAS_QUANT,
            ACTIVATION=activation,
            SHUFFLE=shuffle_scale,
            SCALE_N_PAD=scale_n_pad,
            num_warps=num_warps,
        )

    if HAS_QUANT:
        if transpose_scale:
            scale = scale.view(M, num_bs_cols)
        return out, scale
    return out
