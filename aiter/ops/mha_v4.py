# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from enum import IntEnum
from typing import Optional

import torch
import triton
from torch import Tensor

from aiter import dtypes
from aiter.ops.triton._triton_kernels.quant.sage_attention_quant import (
    sage_quant_v_amax_finalize_kernel,
    sage_quant_v_amax_partial_kernel,
    sage_quant_v_kernel,
)
from aiter.ops.triton.quant.mxfp6_fmha_pack import (
    fp6_k_raw_buffer_sizes,
    fp6_k_lds_order_views_from_raw,
    reorder_fp6_k_lds_order_triton,
)
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    fp4_v_padded_sequence,
    fp4_v_raw_buffer_size,
    pack_v_mxfp4_colmajor_raw,
)

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx


MHA_V4_LOG2E = 1.4426950408889634


@compile_ops("module_fmha_v4_fwd")
def rotate_activation_mxfp6_quant(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
    multiplier: float,
) -> None:
    """Apply hd128 Walsh-Hadamard rotation and pack directly to MXFP6 E2M3."""


@compile_ops("module_fmha_v4_fwd")
def rotate_activation_mxfp4_quant(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
    multiplier: float,
) -> None:
    """Apply hd128 Walsh-Hadamard rotation and pack directly to MXFP4 E2M1."""


@compile_ops("module_fmha_v4_fwd")
def rotate_activation_mxfp4_quant_k(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
) -> None:
    """Apply hd128 Walsh-Hadamard rotation and pack K in the MXFP4 ASM tile order."""


class AttentionFormat(IntEnum):
    FP32 = 0
    FP16 = 1
    BF16 = 2
    FP8_E4M3 = 3
    FP8 = FP8_E4M3
    FP8_E4M3_FNUZ = 4
    FP8_E5M2 = 5
    FP8_E5M2_FNUZ = 6
    FP6_E2M3 = 7
    MXFP6_E2M3 = FP6_E2M3
    MXFP6 = FP6_E2M3
    FP6_E3M2 = 8
    MXFP6_E3M2 = FP6_E3M2
    MXBF6 = FP6_E3M2
    FP4_E2M1 = 9
    MXFP4 = FP4_E2M1
    INT8 = 10
    UINT8 = 11
    INT4 = 12
    UINT4 = 13


class AttentionScaleMode(IntEnum):
    NONE = 0
    F32_PER_TENSOR = 1
    F32_PER_HEAD = 2
    F32_PER_TOKEN = 3
    F32_PER_CHANNEL = 4
    E8M0_PER_1X32 = 5


_FP8_FORMATS = (AttentionFormat.FP8_E4M3, AttentionFormat.FP8_E4M3_FNUZ)
_MX_FORMATS = (AttentionFormat.FP6_E2M3, AttentionFormat.FP4_E2M1)
_PACKED_QK_WIDTH = {
    AttentionFormat.INT8: 128,
    AttentionFormat.FP8_E4M3: 128,
    AttentionFormat.FP8_E4M3_FNUZ: 128,
    AttentionFormat.FP6_E2M3: 96,
    AttentionFormat.FP4_E2M1: 64,
}


def native_fp8_format() -> AttentionFormat:
    return (
        AttentionFormat.FP8_E4M3_FNUZ
        if get_gfx() == "gfx942"
        else AttentionFormat.FP8_E4M3
    )


def _is_fp8_format(format: AttentionFormat) -> bool:
    return format in _FP8_FORMATS


def _validate_format_contract(
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
) -> None:
    if q_format == AttentionFormat.FP6_E3M2:
        raise NotImplementedError(
            "FP6 E3M2 has a reserved format ID but no kernel row yet"
        )
    if q_format != k_format:
        raise ValueError("MHA v4 currently requires matching Q and K formats")
    if q_format not in _PACKED_QK_WIDTH:
        raise ValueError(f"unsupported Q/K format: {q_format!r}")
    if v_format not in (*_FP8_FORMATS, AttentionFormat.FP4_E2M1):
        raise ValueError(f"unsupported V format: {v_format!r}")
    if q_format == AttentionFormat.INT8 and v_format not in _FP8_FORMATS:
        raise ValueError("INT8 Q/K currently requires FP8 V")
    if q_format in _FP8_FORMATS and v_format != q_format:
        raise ValueError("FP8 Q/K currently requires the same FP8 encoding for V")


def scale_modes_for_formats(
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
) -> tuple[AttentionScaleMode, AttentionScaleMode, AttentionScaleMode]:
    _validate_format_contract(q_format, k_format, v_format)
    if q_format == AttentionFormat.INT8 or q_format in _FP8_FORMATS:
        return (
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
        )
    if q_format in _MX_FORMATS:
        v_scale_mode = (
            AttentionScaleMode.F32_PER_CHANNEL
            if _is_fp8_format(v_format)
            else AttentionScaleMode.E8M0_PER_1X32
        )
        return (
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.E8M0_PER_1X32,
            v_scale_mode,
        )
    raise NotImplementedError(
        f"raw preprocessing is not implemented for Q/K format {q_format.name}"
    )


def _fmha_v4_fwd_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
) -> None:
    del q, k, v, q_descale, k_descale, v_descale
    del q_format, k_format, v_format
    del q_scale_mode, k_scale_mode, v_scale_mode, softmax_scale
    del out


@compile_ops(
    "module_fmha_v4_fwd",
    fc_name="fmha_v4_fwd",
    gen_fake=_fmha_v4_fwd_fake,
)
def _fmha_v4_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
) -> None: ...


@torch.library.custom_op("aiter::mha_v4_fwd_launch", mutates_args=("out",))
def _mha_v4_fwd_launch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
) -> None:
    _fmha_v4_fwd(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        out,
        q_format,
        k_format,
        v_format,
        q_scale_mode,
        k_scale_mode,
        v_scale_mode,
        softmax_scale,
    )


@_mha_v4_fwd_launch.register_fake
def _mha_v4_fwd_launch_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
) -> None:
    del q, k, v, q_descale, k_descale, v_descale, out
    del q_format, k_format, v_format
    del q_scale_mode, k_scale_mode, v_scale_mode, softmax_scale


def mha_v4_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
    q_scale_mode: AttentionScaleMode,
    k_scale_mode: AttentionScaleMode,
    v_scale_mode: AttentionScaleMode,
    softmax_scale: Optional[float] = None,
    out: Optional[Tensor] = None,
    return_lse: bool = False,
) -> Tensor:
    """Launch a dense, non-causal MHA v4 kernel over pre-quantized BSHD operands."""
    if return_lse:
        raise NotImplementedError("MHA v4 kernels do not produce LSE yet")
    expected_scale_modes = scale_modes_for_formats(q_format, k_format, v_format)
    if (q_scale_mode, k_scale_mode, v_scale_mode) != expected_scale_modes:
        raise ValueError(
            "unsupported scale recipe for formats: "
            f"got {(q_scale_mode.name, k_scale_mode.name, v_scale_mode.name)}, "
            f"expected {tuple(mode.name for mode in expected_scale_modes)}"
        )

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("MHA v4 expects BSHD Q, K, and V tensors")
    batch, query_length, query_heads, _ = q.shape
    if k.shape[0] != batch or v.shape[0] != batch:
        raise ValueError("Q, K, and V must have the same batch size")
    if k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
        raise ValueError("K and V must have matching sequence and head dimensions")
    if query_heads != k.shape[2]:
        raise ValueError(
            "MHA v4 initially supports MHA only; Q and KV heads must match"
        )
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("MHA v4 expects GPU tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, and V must be on the same device")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Q, K, and V must have contiguous last dimensions")

    logical_head_dim = 128
    expected_q_width = _PACKED_QK_WIDTH[q_format]
    if q.shape[-1] != expected_q_width or k.shape[-1] != expected_q_width:
        raise ValueError(
            f"{q_format.name} Q/K must have packed width {expected_q_width}"
        )
    if v.shape[-1] != logical_head_dim:
        raise ValueError("MHA v4 currently requires logical V head dimension 128")
    if q_format == AttentionFormat.MXFP4:
        if _is_fp8_format(v_format):
            tiles = (k.shape[1] + 127) // 128
            expected_k_stride = (
                k.shape[2] * tiles * 8192,
                64,
                tiles * 8192,
                1,
            )
            if k.stride() != expected_k_stride:
                raise ValueError("MXFP4/FP8 K must use the coalesced MHA v4 tile layout")
        else:
            tiles = (k.shape[1] + 127) // 128
            expected_k_stride = (
                k.shape[2] * tiles * 8192,
                64,
                tiles * 8192,
                1,
            )
            if k.stride() != expected_k_stride:
                raise ValueError("F4F4 K must use the coalesced MHA v4 tile layout")

    if softmax_scale is None:
        softmax_scale = logical_head_dim**-0.5
    if out is None:
        out = torch.empty(
            (batch, query_length, query_heads, logical_head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
    elif out.shape != (batch, query_length, query_heads, logical_head_dim):
        raise ValueError("out has the wrong shape for MHA v4")
    elif out.dtype != torch.bfloat16 or out.device != q.device:
        raise ValueError("out must be a BF16 tensor on the same device as Q")

    _mha_v4_fwd_launch(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        out,
        int(q_format),
        int(k_format),
        int(v_format),
        int(q_scale_mode),
        int(k_scale_mode),
        int(v_scale_mode),
        softmax_scale,
    )
    return out


@torch.library.custom_op("aiter::mha_v4_quantize_int8", mutates_args=())
def _quantize_int8(input: Tensor) -> tuple[Tensor, Tensor]:
    scale = input.float().abs().max() / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.clamp(torch.round(input.float() / scale), -128, 127).to(
        torch.int8
    )
    return quantized, scale.reshape(1).to(torch.float32)


@_quantize_int8.register_fake
def _quantize_int8_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    return input.new_empty(input.shape, dtype=torch.int8), input.new_empty(
        (1,), dtype=torch.float32
    )


@torch.library.custom_op("aiter::mha_v4_quantize_fp8", mutates_args=())
def _quantize_fp8(input: Tensor) -> tuple[Tensor, Tensor]:
    scale = input.float().abs().max() / torch.finfo(dtypes.fp8).max
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    # The dtype conversion performs the format's native rounding and saturation.
    return (input.float() / scale).to(dtypes.fp8), scale.reshape(1).to(torch.float32)


@_quantize_fp8.register_fake
def _quantize_fp8_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    return input.new_empty(input.shape, dtype=dtypes.fp8), input.new_empty(
        (1,), dtype=torch.float32
    )


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp4", mutates_args=())
def _quantize_mxfp4(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, head_dim = input.shape
    if head_dim != 128 or not input.is_contiguous():
        raise ValueError("MXFP4 quantization requires contiguous hd128 BSHD input")
    quantized = input.new_empty(
        (batch, sequence, heads, head_dim // 2), dtype=torch.uint8
    )
    scale = input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)
    rotate_activation_mxfp4_quant(quantized, scale, input, multiplier)
    return quantized, scale


@_quantize_mxfp4.register_fake
def _quantize_mxfp4_fake(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    del multiplier
    batch, sequence, heads, head_dim = input.shape
    return input.new_empty(
        (batch, sequence, heads, head_dim // 2), dtype=torch.uint8
    ), input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)


def mxfp4_k_raw_buffer_size(batch: int, sequence: int, heads: int) -> int:
    """Return bytes for the coalesced MXFP4 K backing buffer."""
    tiles = (sequence + 127) // 128
    return batch * heads * tiles * 8192


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp4_k_raw", mutates_args=())
def quantize_mxfp4_k(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, head_dim = input.shape
    if head_dim != 128 or not input.is_contiguous():
        raise ValueError("MXFP4 K quantization requires contiguous hd128 BSHD input")
    raw = input.new_empty(
        (mxfp4_k_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
    )
    scale = input.new_empty(
        (batch, sequence, heads, head_dim // 32), dtype=torch.uint8
    )
    rotate_activation_mxfp4_quant_k(raw, scale, input)
    return raw, scale


@quantize_mxfp4_k.register_fake
def _quantize_mxfp4_k_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, head_dim = input.shape
    return input.new_empty(
        (mxfp4_k_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
    ), input.new_empty(
        (batch, sequence, heads, head_dim // 32), dtype=torch.uint8
    )
def mxfp4_k_view(raw: Tensor, scale: Tensor) -> Tensor:
    """Rebuild the logical MXFP4 K view from its contiguous backing buffer."""
    batch, sequence, heads, _ = scale.shape
    tiles = (sequence + 127) // 128
    head_stride = tiles * 8192
    return torch.as_strided(
        raw,
        (batch, sequence, heads, 64),
        (heads * head_stride, 64, head_stride, 1),
    )


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp6_q", mutates_args=())
def _quantize_mxfp6_q(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, head_dim = input.shape
    if head_dim != 128 or not input.is_contiguous():
        raise ValueError(
            "MXFP6 E2M3 Q quantization requires contiguous hd128 BSHD input"
        )
    quantized = input.new_empty(
        (batch, sequence, heads, head_dim // 32 * 24), dtype=torch.uint8
    )
    scale = input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)
    rotate_activation_mxfp6_quant(quantized, scale, input, multiplier)
    return quantized, scale


@_quantize_mxfp6_q.register_fake
def _quantize_mxfp6_q_fake(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    del multiplier
    batch, sequence, heads, head_dim = input.shape
    return input.new_empty(
        (batch, sequence, heads, head_dim // 32 * 24), dtype=torch.uint8
    ), input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp6_k_raw", mutates_args=())
def _quantize_mxfp6_k_raw(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, head_dim = input.shape
    if head_dim != 128 or not input.is_contiguous():
        raise ValueError(
            "MXFP6 E2M3 K quantization requires contiguous hd128 BSHD input"
        )
    packed = input.new_empty(
        (batch, sequence, heads, head_dim // 32 * 24), dtype=torch.uint8
    )
    scale = input.new_empty(
        (batch, sequence, heads, head_dim // 32), dtype=torch.uint8
    )
    rotate_activation_mxfp6_quant(packed, scale, input, 1.0)
    return reorder_fp6_k_lds_order_triton(packed, scale, tile=128, return_raw=True)


@_quantize_mxfp6_k_raw.register_fake
def _quantize_mxfp6_k_raw_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, _ = input.shape
    data_size, scale_size = fp6_k_raw_buffer_sizes(batch, sequence, heads)
    return input.new_empty((data_size,), dtype=torch.uint8), input.new_empty(
        (scale_size,), dtype=torch.uint8
    )


@torch.library.custom_op("aiter::mha_v4_quantize_v_fp8", mutates_args=())
def _quantize_v_fp8(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, head_dim = input.shape
    if head_dim != 128 or not input.is_contiguous():
        raise ValueError("FP8 V quantization requires contiguous hd128 BSHD input")
    fp8_max = torch.finfo(dtypes.fp8).max
    scale_block_k = 256
    scale_blocks = triton.cdiv(sequence, scale_block_k)
    scale_reduce_block = triton.next_power_of_2(scale_blocks)
    partial = input.new_empty(
        (batch * heads, scale_blocks, head_dim), dtype=torch.float32
    )
    scale = input.new_empty((batch, heads, head_dim), dtype=torch.float32)
    sage_quant_v_amax_partial_kernel[(batch * heads * scale_blocks,)](
        input,
        partial,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        sequence,
        heads,
        scale_blocks,
        D=head_dim,
        BLOCK_K=scale_block_k,
        num_warps=8,
    )
    sage_quant_v_amax_finalize_kernel[(triton.cdiv(head_dim, 32), batch * heads)](
        partial,
        scale,
        scale_blocks,
        D=head_dim,
        FP8_MAX=fp8_max,
        BLOCK_N=scale_reduce_block,
        BLOCK_D=32,
        num_warps=4,
    )
    block_k = 64
    blocks = triton.cdiv(sequence, block_k)
    quantized = torch.empty_like(input, dtype=dtypes.fp8)
    sage_quant_v_kernel[(batch * heads * blocks,)](
        input,
        quantized,
        scale,
        input.stride(0),
        input.stride(2),
        input.stride(1),
        input.stride(3),
        scale.stride(0),
        scale.stride(1),
        batch,
        heads,
        blocks,
        sequence,
        D=head_dim,
        BLK_K=block_k,
        num_stages=3,
        num_warps=8,
    )
    return quantized, scale


@_quantize_v_fp8.register_fake
def _quantize_v_fp8_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, _, heads, head_dim = input.shape
    return input.new_empty(input.shape, dtype=dtypes.fp8), input.new_empty(
        (batch, heads, head_dim), dtype=torch.float32
    )


@torch.library.custom_op("aiter::mha_v4_quantize_v_mxfp4_raw_v2", mutates_args=())
def _quantize_v_mxfp4_raw(input: Tensor) -> tuple[Tensor, Tensor]:
    if input.shape[-1] != 128 or not input.is_contiguous():
        raise ValueError("MXFP4 V quantization requires contiguous hd128 BSHD input")
    return pack_v_mxfp4_colmajor_raw(input)


@_quantize_v_mxfp4_raw.register_fake
def _quantize_v_mxfp4_raw_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, _ = input.shape
    tiles = fp4_v_padded_sequence(sequence) // 128
    return input.new_empty(
        (fp4_v_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
    ), input.new_empty((batch, heads, tiles * 512), dtype=torch.uint8)


def _v_mxfp4_view(raw: Tensor, scale: Tensor, sequence: int) -> Tensor:
    batch, heads, _ = scale.shape
    padded_sequence = fp4_v_padded_sequence(sequence)
    return torch.as_strided(
        raw,
        (batch, sequence, heads, 128),
        (heads * padded_sequence * 64, 64, padded_sequence * 64, 1),
    )


@torch.library.custom_op("aiter::mha_v4_launch_mxfp4_coalesced_v2", mutates_args=("out",))
def _launch_mxfp4_coalesced(
    q: Tensor,
    q_descale: Tensor,
    k_data: Tensor,
    k_descale: Tensor,
    v_data: Tensor,
    v_descale: Tensor,
    out: Tensor,
    v_format: int,
    softmax_scale: float,
) -> None:
    resolved_v_format = AttentionFormat(v_format)
    k = mxfp4_k_view(k_data, k_descale)
    v = (
        v_data
        if _is_fp8_format(resolved_v_format)
        else _v_mxfp4_view(v_data, v_descale, k.shape[1])
    )
    v_scale_mode = (
        AttentionScaleMode.F32_PER_CHANNEL
        if _is_fp8_format(resolved_v_format)
        else AttentionScaleMode.E8M0_PER_1X32
    )
    mha_v4_packed(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        AttentionFormat.MXFP4,
        AttentionFormat.MXFP4,
        resolved_v_format,
        AttentionScaleMode.E8M0_PER_1X32,
        AttentionScaleMode.E8M0_PER_1X32,
        v_scale_mode,
        softmax_scale=softmax_scale,
        out=out,
    )


@_launch_mxfp4_coalesced.register_fake
def _launch_mxfp4_coalesced_fake(
    q: Tensor,
    q_descale: Tensor,
    k_data: Tensor,
    k_descale: Tensor,
    v_data: Tensor,
    v_descale: Tensor,
    out: Tensor,
    v_format: int,
    softmax_scale: float,
) -> None:
    del q, q_descale, k_data, k_descale, v_data, v_descale, v_format, softmax_scale
    del out


@torch.library.custom_op("aiter::mha_v4_launch_mxfp6_v2", mutates_args=("out",))
def _launch_mxfp6(
    q: Tensor,
    q_descale: Tensor,
    k_raw: Tensor,
    k_descale_raw: Tensor,
    v_data: Tensor,
    v_descale: Tensor,
    out: Tensor,
    sequence_k: int,
    heads: int,
    v_format: int,
    softmax_scale: float,
) -> None:
    resolved_v_format = AttentionFormat(v_format)
    k, k_descale = fp6_k_lds_order_views_from_raw(
        k_raw, k_descale_raw, q.shape[0], sequence_k, heads
    )
    v = (
        v_data
        if _is_fp8_format(resolved_v_format)
        else _v_mxfp4_view(v_data, v_descale, sequence_k)
    )
    v_scale_mode = (
        AttentionScaleMode.F32_PER_CHANNEL
        if _is_fp8_format(resolved_v_format)
        else AttentionScaleMode.E8M0_PER_1X32
    )
    mha_v4_packed(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        AttentionFormat.MXFP6,
        AttentionFormat.MXFP6,
        resolved_v_format,
        AttentionScaleMode.E8M0_PER_1X32,
        AttentionScaleMode.E8M0_PER_1X32,
        v_scale_mode,
        softmax_scale=softmax_scale,
        out=out,
    )


@_launch_mxfp6.register_fake
def _launch_mxfp6_fake(
    q: Tensor,
    q_descale: Tensor,
    k_raw: Tensor,
    k_descale_raw: Tensor,
    v_data: Tensor,
    v_descale: Tensor,
    out: Tensor,
    sequence_k: int,
    heads: int,
    v_format: int,
    softmax_scale: float,
) -> None:
    del q, q_descale, k_raw, k_descale_raw, v_data, v_descale
    del sequence_k, heads, v_format, softmax_scale
    del out


def mha_v4(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
    softmax_scale: Optional[float] = None,
    out: Optional[Tensor] = None,
    return_lse: bool = False,
) -> Tensor:
    """Quantize BF16 BSHD operands and run dense, non-causal MHA v4."""
    if return_lse:
        raise NotImplementedError("MHA v4 kernels do not produce LSE yet")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("mha_v4 expects BSHD Q, K, and V tensors")
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        raise ValueError("mha_v4 currently expects BF16 Q, K, and V inputs")
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError("mha_v4 currently supports head dimension 128 only")
    q_scale_mode, k_scale_mode, v_scale_mode = scale_modes_for_formats(
        q_format, k_format, v_format
    )
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("mha_v4 currently requires contiguous BSHD inputs")
    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)
    elif out.shape != q.shape or out.dtype != torch.bfloat16 or out.device != q.device:
        raise ValueError("out must match Q's shape/device and have BF16 dtype")

    if q_format == AttentionFormat.INT8 and _is_fp8_format(v_format):
        q_quantized, q_descale = _quantize_int8(q)
        k_quantized, k_descale = _quantize_int8(k)
        v_quantized, v_descale = _quantize_fp8(v)
    elif q_format in _FP8_FORMATS and v_format == q_format:
        q_quantized, q_descale = _quantize_fp8(q)
        k_quantized, k_descale = _quantize_fp8(k)
        v_quantized, v_descale = _quantize_fp8(v)
    elif q_format == AttentionFormat.MXFP4 and v_format in (
        *_FP8_FORMATS,
        AttentionFormat.MXFP4,
    ):
        if softmax_scale is None:
            softmax_scale = 128**-0.5
        q_quantized, q_descale = _quantize_mxfp4(q, softmax_scale * MHA_V4_LOG2E)
        k_quantized, k_descale = quantize_mxfp4_k(k)
        if _is_fp8_format(v_format):
            v_quantized, v_descale = _quantize_v_fp8(v)
        else:
            v_quantized, v_descale = _quantize_v_mxfp4_raw(v)
        _launch_mxfp4_coalesced(
            q_quantized,
            q_descale,
            k_quantized,
            k_descale,
            v_quantized,
            v_descale,
            out,
            int(v_format),
            softmax_scale,
        )
        return out
    elif q_format == AttentionFormat.MXFP6 and v_format in (
        *_FP8_FORMATS,
        AttentionFormat.MXFP4,
    ):
        if softmax_scale is None:
            softmax_scale = 128**-0.5
        q_quantized, q_descale = _quantize_mxfp6_q(q, softmax_scale * MHA_V4_LOG2E)
        k_quantized, k_descale = _quantize_mxfp6_k_raw(k)
        if _is_fp8_format(v_format):
            v_quantized, v_descale = _quantize_v_fp8(v)
        else:
            v_quantized, v_descale = _quantize_v_mxfp4_raw(v)
        _launch_mxfp6(
            q_quantized,
            q_descale,
            k_quantized,
            k_descale,
            v_quantized,
            v_descale,
            out,
            k.shape[1],
            k.shape[2],
            int(v_format),
            softmax_scale,
        )
        return out
    else:
        raise NotImplementedError(
            "raw preprocessing is not implemented yet for "
            f"Q={q_format.name}, K={k_format.name}, V={v_format.name}"
        )

    return mha_v4_packed(
        q_quantized,
        k_quantized,
        v_quantized,
        q_descale,
        k_descale,
        v_descale,
        q_format,
        k_format,
        v_format,
        q_scale_mode,
        k_scale_mode,
        v_scale_mode,
        softmax_scale=softmax_scale,
        out=out,
        return_lse=return_lse,
    )
