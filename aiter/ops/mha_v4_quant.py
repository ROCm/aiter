# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Quantization and packed-layout producers for MHA v4."""

import torch
import triton
from torch import Tensor

from aiter import dtypes
from aiter.jit.core import compile_ops
from aiter.ops.triton._triton_kernels.quant.sage_attention_quant import (
    mha_v4_per_tensor_amax_kernel,
    mha_v4_per_tensor_quant_kernel,
    mha_v4_per_tensor_scale_kernel,
    sage_quant_v_amax_finalize_kernel,
    sage_quant_v_amax_partial_kernel,
    sage_quant_v_kernel,
)
from aiter.ops.triton.quant.mxfp6_fmha_pack import (
    fp6_k_lds_order_views_from_raw,
    fp6_k_raw_buffer_sizes,
    pack_fp6_v_data_scale_views,
)
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    FP4_V_BUFFER_SLACK_BYTES,
    FP4_V_PACKED_BYTES_PER_TOKEN,
    FP4_V_TILE_TOKENS,
    fp4_v_padded_sequence,
    fp4_v_raw_buffer_size,
    pack_v_mxfp4_colmajor_raw,
)

MHA_V4_LOG2E = 1.4426950408889634
MHA_V4_PER_TENSOR_BLOCK_SIZE = 8192
MHA_V4_MXFP4_V_TILE_TOKENS = FP4_V_TILE_TOKENS
MHA_V4_MXFP4_V_PACKED_ROW_BYTES = FP4_V_PACKED_BYTES_PER_TOKEN
MHA_V4_MXFP4_V_SCALE_TILE_BYTES = 512
MHA_V4_MXFP4_V_BUFFER_SLACK_BYTES = FP4_V_BUFFER_SLACK_BYTES
# Dense kernels speculatively gather the overlapping final K-scale dword and one lookahead V-scale
# tile. Keep those reads mapped and zero without changing either scale tensor's logical shape.
MHA_V4_MXFP4_K_SCALE_SLACK_BYTES = 4
MHA_V4_MXFP4_V_SCALE_SLACK_BYTES = MHA_V4_MXFP4_V_SCALE_TILE_BYTES
MHA_V4_MXFP6_V_TILE_TOKENS = 128
MHA_V4_MXFP6_V_PACKED_ROW_BYTES = 96
MHA_V4_MXFP6_V_TILE_BYTES = (
    MHA_V4_MXFP6_V_TILE_TOKENS * MHA_V4_MXFP6_V_PACKED_ROW_BYTES
)
MHA_V4_MXFP6_V_SCALE_TILE_BYTES = 512
MHA_V4_MXFP6_V_BUFFER_SLACK_BYTES = 256


def mha_v4_q_multiplier(softmax_scale: float) -> float:
    """Return the Q multiplier expected by the MX attention quantizers."""
    return softmax_scale * MHA_V4_LOG2E


@compile_ops("module_fmha_v4_fwd", develop=True)
def rotate_activation_hd128(out: Tensor, input: Tensor) -> None:
    """Apply normalized Walsh-Hadamard rotation to contiguous hd128 rows."""


@compile_ops("module_fmha_v4_fwd", develop=True)
def rotate_activation_mxfp8_quant(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
    multiplier: float,
) -> None:
    """Apply hd128 Walsh-Hadamard rotation and quantize directly to MXFP8."""


@compile_ops("module_fmha_v4_fwd", develop=True)
def rotate_activation_mxfp6_quant(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
    multiplier: float,
) -> None:
    """Apply hd128 Walsh-Hadamard rotation and pack directly to MXFP6 E2M3."""


@compile_ops("module_fmha_v4_fwd", develop=True)
def rotate_activation_mxfp6_quant_k(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
) -> None:
    """Rotate and pack hd128 K directly into the MXFP6 LDS-order buffers."""


@compile_ops("module_fmha_v4_fwd", develop=True)
def _quantize_v_mxfp6_fp6_p_hip(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
) -> None:
    """Pack V in the contraction order required by an FP6 P operand."""


@compile_ops("module_fmha_v4_fwd", develop=True)
def rotate_activation_mxfp4_quant(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
    multiplier: float,
) -> None:
    """Apply hd128 Walsh-Hadamard rotation and pack directly to MXFP4 E2M1."""


@compile_ops("module_fmha_v4_fwd", develop=True)
def rotate_activation_mxfp4_quant_k(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
) -> None:
    """Apply hd128 Walsh-Hadamard rotation and pack K in the MXFP4 ASM tile order."""


@compile_ops("module_fmha_v4_fwd", develop=True)
def _quantize_v_mxfp4_fp6_p_hip(
    out: Tensor,
    scale: Tensor,
    input: Tensor,
) -> None:
    """Pack MXFP4 V in the contraction order required by an FP6 P operand."""


def _validate_bshd_hd128(input: Tensor, operation: str) -> tuple[int, int, int, int]:
    if input.dim() != 4 or input.shape[-1] != 128 or not input.is_contiguous():
        raise ValueError(f"{operation} requires contiguous hd128 BSHD input")
    return input.shape


def _quantize_per_tensor(
    input: Tensor, output_dtype: torch.dtype, dtype_max: float, clip: float
) -> tuple[Tensor, Tensor]:
    if not input.is_contiguous():
        raise ValueError("MHA v4 per-tensor quantization requires contiguous input")
    numel = input.numel()
    blocks = triton.cdiv(numel, MHA_V4_PER_TENSOR_BLOCK_SIZE)
    partial = input.new_empty((blocks,), dtype=torch.float32)
    scale = input.new_empty((1,), dtype=torch.float32)
    output = input.new_empty(input.shape, dtype=output_dtype)
    mha_v4_per_tensor_amax_kernel[(blocks,)](
        input,
        partial,
        numel,
        BLOCK_SIZE=MHA_V4_PER_TENSOR_BLOCK_SIZE,
        num_warps=8,
    )
    scale_block = triton.next_power_of_2(blocks)
    mha_v4_per_tensor_scale_kernel[(1,)](
        partial,
        scale,
        blocks,
        dtype_max=dtype_max / clip,
        BLOCK_SIZE=scale_block,
        num_warps=8,
    )
    mha_v4_per_tensor_quant_kernel[(blocks,)](
        input,
        output,
        scale,
        numel,
        IS_INT8=output_dtype == torch.int8,
        BLOCK_SIZE=MHA_V4_PER_TENSOR_BLOCK_SIZE,
        num_warps=8,
    )
    return output, scale


@torch.library.custom_op("aiter::mha_v4_quantize_int8_v2", mutates_args=())
def quantize_int8(input: Tensor, clip: float = 1.0) -> tuple[Tensor, Tensor]:
    """Per-tensor quantize a contiguous tensor to INT8 and return its scale."""
    return _quantize_per_tensor(input, torch.int8, 127.0, clip)


@quantize_int8.register_fake
def _quantize_int8_fake(input: Tensor, clip: float = 1.0) -> tuple[Tensor, Tensor]:
    del clip
    return input.new_empty(input.shape, dtype=torch.int8), input.new_empty(
        (1,), dtype=torch.float32
    )


@torch.library.custom_op("aiter::mha_v4_quantize_fp8", mutates_args=())
def quantize_fp8(input: Tensor) -> tuple[Tensor, Tensor]:
    """Per-tensor quantize a contiguous tensor to native FP8 and return its scale."""
    return _quantize_per_tensor(input, dtypes.fp8, torch.finfo(dtypes.fp8).max, 1.0)


@quantize_fp8.register_fake
def _quantize_fp8_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    return input.new_empty(input.shape, dtype=dtypes.fp8), input.new_empty(
        (1,), dtype=torch.float32
    )


def quantize_fp8_rotated(input: Tensor) -> tuple[Tensor, Tensor]:
    """Apply normalized hd128 Walsh-Hadamard rotation, then per-tensor FP8 quantize."""
    if input.shape[-1] != 128 or not input.is_contiguous():
        raise ValueError("rotated FP8 quantization requires contiguous hd128 input")
    rotated = torch.empty_like(input)
    rotate_activation_hd128(rotated, input)
    return quantize_fp8(rotated)


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp8_q", mutates_args=())
def quantize_mxfp8_q(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    """Rotate and quantize hd128 BSHD Q to MXFP8 data and E8M0 block scales."""
    batch, sequence, heads, head_dim = _validate_bshd_hd128(input, "MXFP8 quantization")
    quantized = input.new_empty(input.shape, dtype=dtypes.fp8)
    scale = input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)
    rotate_activation_mxfp8_quant(quantized, scale, input, multiplier)
    return quantized, scale


@quantize_mxfp8_q.register_fake
def _quantize_mxfp8_q_fake(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    del multiplier
    batch, sequence, heads, head_dim = input.shape
    return input.new_empty(input.shape, dtype=dtypes.fp8), input.new_empty(
        (batch, sequence, heads, head_dim // 32), dtype=torch.uint8
    )


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp8_k", mutates_args=())
def quantize_mxfp8_k(input: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate and quantize hd128 BSHD K to MXFP8 data and E8M0 block scales."""
    batch, sequence, heads, head_dim = _validate_bshd_hd128(
        input, "MXFP8 K quantization"
    )
    quantized = input.new_empty(input.shape, dtype=dtypes.fp8)
    scale = input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)
    rotate_activation_mxfp8_quant(quantized, scale, input, 1.0)
    return quantized, scale


@quantize_mxfp8_k.register_fake
def _quantize_mxfp8_k_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, head_dim = input.shape
    return input.new_empty(input.shape, dtype=dtypes.fp8), input.new_empty(
        (batch, sequence, heads, head_dim // 32), dtype=torch.uint8
    )


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp4", mutates_args=())
def quantize_mxfp4_q(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    """Rotate and pack hd128 BSHD Q as MXFP4 data with E8M0 block scales."""
    batch, sequence, heads, head_dim = _validate_bshd_hd128(input, "MXFP4 quantization")
    quantized = input.new_empty(
        (batch, sequence, heads, head_dim // 2), dtype=torch.uint8
    )
    scale = input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)
    rotate_activation_mxfp4_quant(quantized, scale, input, multiplier)
    return quantized, scale


@quantize_mxfp4_q.register_fake
def _quantize_mxfp4_q_fake(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    del multiplier
    batch, sequence, heads, head_dim = input.shape
    return input.new_empty(
        (batch, sequence, heads, head_dim // 2), dtype=torch.uint8
    ), input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)


def mxfp4_k_raw_buffer_size(batch: int, sequence: int, heads: int) -> int:
    """Return bytes for the coalesced MXFP4 K backing buffer."""
    tiles = (sequence + 127) // 128
    return batch * heads * tiles * 8192


def mxfp4_v_tiles(sequence: int) -> int:
    """Return the number of 128-token MXFP4 V tiles."""
    return (sequence + MHA_V4_MXFP4_V_TILE_TOKENS - 1) // MHA_V4_MXFP4_V_TILE_TOKENS


def mxfp4_v_padded_sequence(sequence: int) -> int:
    """Round a V sequence length up to the 128-token MXFP4 packing tile."""
    return fp4_v_padded_sequence(sequence)


def mxfp4_v_raw_buffer_size(batch: int, sequence: int, heads: int) -> int:
    """Return bytes for the packed MXFP4 V buffer, including view slack."""
    return fp4_v_raw_buffer_size(batch, sequence, heads)


def mxfp6_v_tiles(sequence: int) -> int:
    """Return the number of 128-token MXFP6 V tiles."""
    return (sequence + MHA_V4_MXFP6_V_TILE_TOKENS - 1) // MHA_V4_MXFP6_V_TILE_TOKENS


def mxfp6_v_padded_sequence(sequence: int) -> int:
    """Round a V sequence length up to the 128-token MXFP6 packing tile."""
    return mxfp6_v_tiles(sequence) * MHA_V4_MXFP6_V_TILE_TOKENS


def mxfp6_v_raw_buffer_size(batch: int, sequence: int, heads: int) -> int:
    """Return bytes for the packed MXFP6 V buffer, including view slack."""
    return (
        batch
        * heads
        * mxfp6_v_padded_sequence(sequence)
        * MHA_V4_MXFP6_V_PACKED_ROW_BYTES
        + MHA_V4_MXFP6_V_BUFFER_SLACK_BYTES
    )


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp4_k_raw", mutates_args=())
def quantize_mxfp4_k(input: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate and pack hd128 BSHD K into the coalesced MXFP4 ASM layout."""
    batch, sequence, heads, head_dim = _validate_bshd_hd128(
        input, "MXFP4 K quantization"
    )
    raw = input.new_empty(
        (mxfp4_k_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
    )
    scale_elements = batch * sequence * heads * (head_dim // 32)
    scale_storage = input.new_empty(
        (scale_elements + MHA_V4_MXFP4_K_SCALE_SLACK_BYTES,), dtype=torch.uint8
    )
    scale_storage[scale_elements:].zero_()
    scale = scale_storage[:scale_elements].view(
        batch, sequence, heads, head_dim // 32
    )
    rotate_activation_mxfp4_quant_k(raw, scale, input)
    return raw, scale


@quantize_mxfp4_k.register_fake
def _quantize_mxfp4_k_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, head_dim = input.shape
    scale_elements = batch * sequence * heads * (head_dim // 32)
    scale_storage = input.new_empty(
        (scale_elements + MHA_V4_MXFP4_K_SCALE_SLACK_BYTES,), dtype=torch.uint8
    )
    return input.new_empty(
        (mxfp4_k_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
    ), scale_storage[:scale_elements].view(batch, sequence, heads, head_dim // 32)


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


def mxfp6_k_view(
    raw: Tensor,
    scale_raw: Tensor,
    batch: int,
    sequence: int,
    heads: int,
) -> tuple[Tensor, Tensor]:
    """Rebuild the logical MXFP6 K and scale views from raw backing buffers."""
    return fp6_k_lds_order_views_from_raw(raw, scale_raw, batch, sequence, heads)


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp6_q", mutates_args=())
def quantize_mxfp6_q(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    """Rotate and pack hd128 BSHD Q as MXFP6 E2M3 with E8M0 block scales."""
    batch, sequence, heads, head_dim = _validate_bshd_hd128(
        input, "MXFP6 E2M3 Q quantization"
    )
    quantized = input.new_empty(
        (batch, sequence, heads, head_dim // 32 * 24), dtype=torch.uint8
    )
    scale = input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)
    rotate_activation_mxfp6_quant(quantized, scale, input, multiplier)
    return quantized, scale


@quantize_mxfp6_q.register_fake
def _quantize_mxfp6_q_fake(input: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    del multiplier
    batch, sequence, heads, head_dim = input.shape
    return input.new_empty(
        (batch, sequence, heads, head_dim // 32 * 24), dtype=torch.uint8
    ), input.new_empty((batch, sequence, heads, head_dim // 32), dtype=torch.uint8)


@torch.library.custom_op("aiter::mha_v4_quantize_mxfp6_k_raw", mutates_args=())
def quantize_mxfp6_k(input: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate and pack hd128 BSHD K into raw MXFP6 ASM data and scale buffers."""
    batch, sequence, heads, _ = _validate_bshd_hd128(input, "MXFP6 E2M3 K quantization")
    data_size, scale_size = fp6_k_raw_buffer_sizes(batch, sequence, heads)
    raw = input.new_empty((data_size,), dtype=torch.uint8)
    scale_raw = input.new_empty((scale_size,), dtype=torch.uint8)
    rotate_activation_mxfp6_quant_k(raw, scale_raw, input)
    return raw, scale_raw


@quantize_mxfp6_k.register_fake
def _quantize_mxfp6_k_raw_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, _ = input.shape
    data_size, scale_size = fp6_k_raw_buffer_sizes(batch, sequence, heads)
    return input.new_empty((data_size,), dtype=torch.uint8), input.new_empty(
        (scale_size,), dtype=torch.uint8
    )


@torch.library.custom_op("aiter::mha_v4_quantize_v_fp8", mutates_args=())
def quantize_v_fp8(input: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize hd128 BSHD V to FP8 with one FP32 scale per channel."""
    batch, sequence, heads, head_dim = _validate_bshd_hd128(input, "FP8 V quantization")
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


@quantize_v_fp8.register_fake
def _quantize_v_fp8_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, _, heads, head_dim = input.shape
    return input.new_empty(input.shape, dtype=dtypes.fp8), input.new_empty(
        (batch, heads, head_dim), dtype=torch.float32
    )


@torch.library.custom_op("aiter::mha_v4_quantize_v_mxfp4_raw_v2", mutates_args=())
def quantize_v_mxfp4(input: Tensor) -> tuple[Tensor, Tensor]:
    """Pack hd128 BSHD V into raw column-major MXFP4 data and scale buffers."""
    _validate_bshd_hd128(input, "MXFP4 V quantization")
    return pack_v_mxfp4_colmajor_raw(input)


@quantize_v_mxfp4.register_fake
def _quantize_v_mxfp4_raw_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, _ = input.shape
    tiles = mxfp4_v_tiles(sequence)
    return input.new_empty(
        (mxfp4_v_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
    ), input.new_empty(
        (batch, heads, tiles * MHA_V4_MXFP4_V_SCALE_TILE_BYTES), dtype=torch.uint8
    )


@torch.library.custom_op("aiter::mha_v4_quantize_v_mxfp4_fp6_p_raw", mutates_args=())
def quantize_v_mxfp4_fp6_p(input: Tensor) -> tuple[Tensor, Tensor]:
    """Pack MXFP4 V in the token order consumed by the FP6-P F4F4 kernel."""
    batch, sequence, heads, _ = _validate_bshd_hd128(
        input, "MXFP4 V-for-FP6-P quantization"
    )
    tiles = mxfp4_v_tiles(sequence)
    raw = input.new_empty(
        (mxfp4_v_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
    )
    scale_elements = batch * heads * tiles * MHA_V4_MXFP4_V_SCALE_TILE_BYTES
    scale_storage = input.new_empty(
        (scale_elements + MHA_V4_MXFP4_V_SCALE_SLACK_BYTES,), dtype=torch.uint8
    )
    scale_storage[scale_elements:].zero_()
    scale = scale_storage[:scale_elements].view(
        batch, heads, tiles * MHA_V4_MXFP4_V_SCALE_TILE_BYTES
    )
    _quantize_v_mxfp4_fp6_p_hip(raw, scale, input)
    return raw, scale


@quantize_v_mxfp4_fp6_p.register_fake
def _quantize_v_mxfp4_fp6_p_raw_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, _ = input.shape
    tiles = mxfp4_v_tiles(sequence)
    scale_elements = batch * heads * tiles * MHA_V4_MXFP4_V_SCALE_TILE_BYTES
    scale_storage = input.new_empty(
        (scale_elements + MHA_V4_MXFP4_V_SCALE_SLACK_BYTES,), dtype=torch.uint8
    )
    return input.new_empty(
        (mxfp4_v_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
    ), scale_storage[:scale_elements].view(
        batch, heads, tiles * MHA_V4_MXFP4_V_SCALE_TILE_BYTES
    )


@torch.library.custom_op("aiter::mha_v4_quantize_v_mxfp6", mutates_args=())
def quantize_v_mxfp6(input: Tensor) -> tuple[Tensor, Tensor]:
    """Pack hd128 BSHD V into MXFP6 data and E8M0 scale views."""
    _validate_bshd_hd128(input, "MXFP6 V quantization")
    return pack_fp6_v_data_scale_views(input)


@quantize_v_mxfp6.register_fake
def _quantize_v_mxfp6_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    batch, sequence, heads, head_dim = input.shape
    tiles = mxfp6_v_tiles(sequence)
    head_stride = tiles * MHA_V4_MXFP6_V_TILE_BYTES
    raw = input.new_empty(
        (mxfp6_v_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
    )
    quantized = torch.as_strided(
        raw,
        (batch, sequence, heads, head_dim),
        (heads * head_stride, MHA_V4_MXFP6_V_PACKED_ROW_BYTES, head_stride, 1),
    )
    return quantized, input.new_empty(
        (batch, heads, tiles * MHA_V4_MXFP6_V_SCALE_TILE_BYTES), dtype=torch.uint8
    )


@torch.library.custom_op("aiter::mha_v4_quantize_v_mxfp6_fp6_p", mutates_args=())
def quantize_v_mxfp6_fp6_p(input: Tensor) -> tuple[Tensor, Tensor]:
    """Pack MXFP6 V in the contraction order required by an FP6 P operand."""
    batch, sequence, heads, head_dim = _validate_bshd_hd128(
        input, "FP6-P MXFP6 V quantization"
    )
    tiles = mxfp6_v_tiles(sequence)
    head_stride = tiles * MHA_V4_MXFP6_V_TILE_BYTES
    raw = input.new_empty(
        (mxfp6_v_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
    )
    scale = input.new_empty(
        (batch, heads, tiles * MHA_V4_MXFP6_V_SCALE_TILE_BYTES), dtype=torch.uint8
    )
    _quantize_v_mxfp6_fp6_p_hip(raw, scale, input)
    quantized = torch.as_strided(
        raw,
        (batch, sequence, heads, head_dim),
        (heads * head_stride, MHA_V4_MXFP6_V_PACKED_ROW_BYTES, head_stride, 1),
    )
    return quantized, scale


@quantize_v_mxfp6_fp6_p.register_fake
def _quantize_v_mxfp6_fp6_p_fake(input: Tensor) -> tuple[Tensor, Tensor]:
    return _quantize_v_mxfp6_fake(input)


def mxfp4_v_view(raw: Tensor, scale: Tensor, sequence: int) -> Tensor:
    """Rebuild the logical MXFP4 V view from its contiguous backing buffer."""
    batch, heads, _ = scale.shape
    padded_sequence = mxfp4_v_padded_sequence(sequence)
    return torch.as_strided(
        raw,
        (batch, sequence, heads, 128),
        (
            heads * padded_sequence * MHA_V4_MXFP4_V_PACKED_ROW_BYTES,
            MHA_V4_MXFP4_V_PACKED_ROW_BYTES,
            padded_sequence * MHA_V4_MXFP4_V_PACKED_ROW_BYTES,
            1,
        ),
    )