# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Quantized inverse-Ulysses output all-to-all over MoRI peer writes."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Stream

from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

from .attention_a2a_intranode_kernel import make_attention_a2a_completion_kernel
from .communication_ops_utils import wave_uniform_i64
from .quant_utils import (
    emit_f32_to_e2m1,
    emit_f32_to_e2m3_native,
    emit_f32_to_fp8,
    emit_mx_e8m0_scale,
)
from .tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor

_JIT_CACHE_TAG = "attention_output_a2a_intranode_v13"
_ROWS_PER_WAVE = 4
_LANES_PER_ROW = 16
_VALUES_PER_LANE = 8
_MX_GROUP_SIZE = 32
_MX_GROUPS_PER_HEAD = 4
_MX_TILE_ROWS = 256
_MX_K_TILE = 128
_MX_K_GUARD_TILES = 2
_MX_SCALE_TILE_BYTES = 1024
_MXFP4_TILE_BYTES = 16384
_MXFP6_TILE_BYTES = 24576
_HADAMARD32_NORM = 0.1767578125
_HADAMARD_SAFETY_SHIFT = 3


def _ceil(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _hadamard32(values, group_lane, *, scale_first, round_bf16, safe):
    norm = _HADAMARD32_NORM * (0.125 if safe else 1.0)
    result = [values[index] for index in range(_VALUES_PER_LANE)]
    if scale_first:
        result = [value * fx.Float32(norm) for value in result]
    for stage in range(3):
        shift = 1 << stage
        result = [
            (
                result[index ^ shift] - result[index]
                if index & shift
                else result[index] + result[index ^ shift]
            )
            for index in range(_VALUES_PER_LANE)
        ]
    for shift in (1, 2):
        result = [
            ((group_lane & shift) == 0).select(
                value + value.shuffle_xor(shift, 64),
                value.shuffle_xor(shift, 64) - value,
            )
            for value in result
        ]
    if not scale_first:
        result = [value * fx.Float32(norm) for value in result]
    if round_bf16:
        result = [value.to(fx.BFloat16).to(fx.Float32) for value in result]
    return result


def _group_amax(values, group_lane, *, floor):
    maximum = fx.Float32(floor)
    for value in values:
        maximum = maximum.maximumf(fmath.absf(value))
    for shift in (1, 2):
        maximum = maximum.maximumf(maximum.shuffle_xor(shift, 64))
    return maximum


def _gemm_fp6_scale(amax):
    exponent = (amax.bitcast(fx.Int32) >> 23) & 255
    ordinary = fx.max(fx.min(exponent - 2, 254), 0)
    return (amax == 0.0).select(fx.Int32(127), ordinary)


def make_attention_output_a2a_kernel(
    *,
    rank,
    npes,
    seq_local,
    heads_local,
    head_dim,
    block_num,
    warp_num_per_block,
    fp8_fnuz,
    codec,
):
    """Build the BF16 BSHD -> prequantized inverse-Ulysses push kernel."""
    if head_dim != _LANES_PER_ROW * _VALUES_PER_LANE:
        raise ValueError(f"expected head_dim=128, got {head_dim}")
    if codec not in {"e4m3", "e4m3_external_scale", "mxfp4", "mxfp6"}:
        raise ValueError(f"unsupported attention output codec {codec!r}")
    if (
        codec in {"e4m3_external_scale", "mxfp4", "mxfp6"}
        and current_target().arch != "gfx950"
    ):
        raise RuntimeError(f"{codec} attention output requires gfx950")

    seq_full = seq_local * npes
    heads_total = heads_local * npes
    rows_per_peer = seq_local * heads_local
    input_rows = seq_full * heads_local
    output_rows = seq_local * heads_total
    input_bytes = input_rows * head_dim * 2
    gemm_k = heads_total * head_dim
    pad_rows = _ceil(seq_local, _MX_TILE_ROWS)
    pad_k = _ceil(gemm_k, _MX_K_TILE)
    nk_pad = pad_k // _MX_K_TILE + _MX_K_GUARD_TILES
    if codec in {"e4m3", "e4m3_external_scale"}:
        output_bytes = output_rows * head_dim
        scale_bytes = (1 if codec == "e4m3_external_scale" else output_rows) * 4
    elif codec == "mxfp4":
        output_bytes = seq_local * gemm_k // 2
        scale_columns = _ceil(gemm_k // _MX_GROUP_SIZE, 8)
        scale_bytes = pad_rows * scale_columns
    else:
        output_bytes = pad_rows // _MX_TILE_ROWS * nk_pad * _MXFP6_TILE_BYTES
        scale_bytes = pad_rows // _MX_TILE_ROWS * nk_pad * _MX_SCALE_TILE_BYTES
    fp8_max = 240.0 if fp8_fnuz else 448.0

    shared_storage = fx.struct(
        type(
            "_AttentionOutputA2ASharedStorage",
            (),
            {
                "__annotations__": {
                    "p2p_payload": fx.Array[fx.Int64, npes, 16],
                    "p2p_scale": fx.Array[fx.Int64, npes, 16],
                }
            },
        )
    )

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def attention_output_a2a_push(
        addr_input: fx.Int64,
        addr_input_scale: fx.Int64,
        addr_p2p_payload: fx.Int64,
        addr_p2p_scale: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        warp = tid >> 6
        lane = tid & 63
        row_group = lane >> 4
        row_lane = lane & 15
        global_warp = bid * warp_num_per_block + warp
        global_warps = block_num * warp_num_per_block
        peer_warps = global_warps // npes
        dest_pe = global_warp % npes
        peer_warp = global_warp // npes

        payload_table = ptr_buf_tensor(
            addr_p2p_payload, fx.Int64, num_records_bytes=npes * 8
        )
        scale_table = ptr_buf_tensor(
            addr_p2p_scale, fx.Int64, num_records_bytes=npes * 8
        )
        shared = fx.SharedAllocator().allocate(shared_storage).peek()
        payload_bases = shared.p2p_payload.view(fx.make_layout(npes, 1))
        scale_bases = shared.p2p_scale.view(fx.make_layout(npes, 1))
        if lane < npes:
            fx.memref_store(
                buf_copy_load(payload_table, lane, fx.Int64), payload_bases, lane
            )
            fx.memref_store(
                buf_copy_load(scale_table, lane, fx.Int64), scale_bases, lane
            )
        fx.barrier()

        payload_base = wave_uniform_i64(
            fx.Uint64(fx.memref_load(payload_bases, dest_pe))
        )
        scale_base = wave_uniform_i64(fx.Uint64(fx.memref_load(scale_bases, dest_pe)))
        source = ptr_buf_tensor(
            addr_input,
            fx.BFloat16,
            unit_elems=_VALUES_PER_LANE,
            unit_stride=1,
            num_records_bytes=input_bytes,
        )
        destination = ptr_buf_tensor(
            payload_base,
            fx.Int32,
            unit_elems=2 if codec in {"e4m3", "e4m3_external_scale"} else 1,
            unit_stride=1,
            num_records_bytes=output_bytes,
        )
        scales = ptr_buf_tensor(
            scale_base,
            fx.Float32 if codec in {"e4m3", "e4m3_external_scale"} else fx.Int8,
            num_records_bytes=scale_bytes,
        )
        input_scales = ptr_buf_tensor(
            addr_input_scale,
            fx.Float32,
            num_records_bytes=npes * 4,
        )

        row_step = peer_warps * _ROWS_PER_WAVE
        for peer_row_base in range(peer_warp * _ROWS_PER_WAVE, rows_per_peer, row_step):
            peer_row = peer_row_base + row_group
            valid = peer_row < rows_per_peer
            safe_peer_row = valid.select(peer_row, fx.Int32(0))
            token_local = safe_peer_row // heads_local
            local_head = safe_peer_row % heads_local
            source_token = dest_pe * seq_local + token_local
            source_row = source_token * heads_local + local_head
            values = fx.Vector(
                buf_copy_load(
                    source,
                    source_row * head_dim + row_lane * _VALUES_PER_LANE,
                    fx.BFloat16,
                    _VALUES_PER_LANE,
                )
            ).to(fx.Float32)

            destination_head = rank * heads_local + local_head
            if const_expr(codec in {"e4m3", "e4m3_external_scale"}):
                if const_expr(codec == "e4m3_external_scale"):
                    scale = buf_copy_load(
                        input_scales,
                        dest_pe,
                        fx.Float32,
                    )
                else:
                    amax = fx.Float32(0.0)
                    for index in range_constexpr(_VALUES_PER_LANE):
                        amax = amax.maximumf(fmath.absf(values[index]))
                    for shift in (1, 2, 4, 8):
                        amax = amax.maximumf(amax.shuffle_xor(shift, 64))

                    # Match dynamic_per_group_scaled_quant_kernel's nonzero floor
                    # so zero/subnormal rows get the canonical FP32 descale.
                    amax = amax.maximumf(fx.Float32(1e-10))
                    scale = amax * fx.Float32(1.0 / fp8_max)
                scaled = [
                    (
                        values[index] / scale
                        if const_expr(codec == "e4m3_external_scale")
                        else values[index] * (fx.Float32(1.0) / scale)
                    )
                    for index in range_constexpr(_VALUES_PER_LANE)
                ]
                pairs = [
                    fx.Uint32(
                        emit_f32_to_fp8(
                            scaled[2 * pair]
                            .maximumf(fx.Float32(-fp8_max))
                            .minimumf(fx.Float32(fp8_max)),
                            scaled[2 * pair + 1]
                            .maximumf(fx.Float32(-fp8_max))
                            .minimumf(fx.Float32(fp8_max)),
                        )
                    )
                    & fx.Uint32(0xFFFF)
                    for pair in range_constexpr(_VALUES_PER_LANE // 2)
                ]
                words = fx.Vector.from_elements(
                    [
                        fx.Int32(pairs[0] | (pairs[1] << 16)),
                        fx.Int32(pairs[2] | (pairs[3] << 16)),
                    ],
                    fx.Int32,
                )
                destination_row = token_local * heads_total + destination_head
                if valid:
                    buf_copy_store(
                        destination,
                        destination_row * (head_dim // 4) + row_lane * 2,
                        words,
                        fx.Int32,
                        2,
                    )
                    if const_expr(codec == "e4m3") and row_lane == 0:
                        buf_copy_store(
                            scales,
                            destination_row,
                            scale,
                            fx.Float32,
                        )
            group_lane = row_lane & 3
            head_group = row_lane >> 2
            global_group = destination_head * _MX_GROUPS_PER_HEAD + head_group
            if const_expr(codec == "mxfp4"):
                # A4W4 consumes ordinary per-1x32 MXFP4. Unlike A6W4/A6W6,
                # this contract does not rotate activation or weight blocks.
                rotated = [values[index] for index in range(_VALUES_PER_LANE)]
                amax = _group_amax(rotated, group_lane, floor=1e-10)
                safety_shift = fx.Int32(0)
            else:
                normal_values = _hadamard32(
                    values,
                    group_lane,
                    scale_first=False,
                    round_bf16=False,
                    safe=False,
                )
                normal_amax = _group_amax(
                    normal_values,
                    group_lane,
                    floor=0.0,
                )
                extreme = ((normal_amax.bitcast(fx.Int32) >> 23) & 255) == 255
                safe_values = _hadamard32(
                    values,
                    group_lane,
                    scale_first=True,
                    round_bf16=False,
                    safe=True,
                )
                safe_amax = _group_amax(
                    safe_values,
                    group_lane,
                    floor=0.0,
                )
                rotated = [
                    extreme.select(safe_values[index], normal_values[index])
                    for index in range(_VALUES_PER_LANE)
                ]
                amax = extreme.select(safe_amax, normal_amax)
                safety_shift = extreme.select(
                    fx.Int32(_HADAMARD_SAFETY_SHIFT), fx.Int32(0)
                )

            tile_row = token_local // _MX_TILE_ROWS
            rem = token_local % _MX_TILE_ROWS
            row_block = rem // 16
            row16 = rem % 16
            step = global_group // 4
            k_group = global_group % 4
            block = row_block * 64 + k_group * 16 + row16
            tile = tile_row * nk_pad + step
            scale_upper = rem // 128
            scale_sub = (rem % 128) // 16
            scale_address = (
                tile * _MX_SCALE_TILE_BYTES
                + scale_upper * 512
                + k_group * 128
                + row16 * 8
                + scale_sub
            )

            if const_expr(codec == "mxfp4"):
                raw_scale = fx.Int32(
                    emit_mx_e8m0_scale(
                        amax.ir_value(),
                        mode=MxScaleRoundModeInt.RoundUp,
                        dtype=MxDtypeInt.FP4_E2M1,
                    )
                )
                stored_scale = fx.min(raw_scale + safety_shift, fx.Int32(254))
                conversion_scale = stored_scale - safety_shift
                reciprocal = ((fx.Int32(254) - conversion_scale) << 23).bitcast(
                    fx.Float32
                )
                codes = [
                    fx.Uint32(
                        emit_f32_to_e2m1((rotated[index] * reciprocal).ir_value())
                    )
                    for index in range(_VALUES_PER_LANE)
                ]
                word = fx.Int32(
                    codes[0]
                    | (codes[1] << 4)
                    | (codes[2] << 8)
                    | (codes[3] << 12)
                    | (codes[4] << 16)
                    | (codes[5] << 20)
                    | (codes[6] << 24)
                    | (codes[7] << 28)
                )
                payload_address = (
                    token_local * (gemm_k // 8) + global_group * 4 + group_lane
                )
                scale_columns = _ceil(gemm_k // _MX_GROUP_SIZE, 8)
                scale_swizzle = (
                    (token_local // 32 * scale_columns) * 32
                    + (global_group // 8) * 256
                    + (global_group % 4) * 64
                    + (token_local % 16) * 4
                    + (global_group % 8) // 4 * 2
                    + (token_local % 32) // 16
                )
                if valid:
                    buf_copy_store(
                        destination,
                        payload_address,
                        word,
                        fx.Int32,
                    )
                    if group_lane == 0:
                        buf_copy_store(
                            scales,
                            scale_swizzle,
                            stored_scale.to(fx.Int8),
                            fx.Int8,
                        )
            conversion_scale = _gemm_fp6_scale(amax)
            stored_scale = fx.min(conversion_scale + safety_shift, fx.Int32(254))
            conversion_scale = stored_scale - safety_shift
            scale_bits = (conversion_scale == 0).select(
                fx.Int32(0x00400000),
                conversion_scale << 23,
            )
            forward_scale = scale_bits.bitcast(fx.Float32)
            even_by_lane = [[], [], [], []]
            odd_by_lane = [[], [], [], []]
            for pair in range_constexpr(_VALUES_PER_LANE // 2):
                first = rotated[2 * pair]
                second = rotated[2 * pair + 1]
                first_x1 = first.shuffle_xor(1, 64)
                second_x1 = second.shuffle_xor(1, 64)
                first_x2 = first.shuffle_xor(2, 64)
                second_x2 = second.shuffle_xor(2, 64)
                first_x3 = first_x2.shuffle_xor(1, 64)
                second_x3 = second_x2.shuffle_xor(1, 64)
                for target, value in enumerate((first, first_x1, first_x2, first_x3)):
                    even_by_lane[target].append(value)
                for target, value in enumerate(
                    (second, second_x1, second_x2, second_x3)
                ):
                    odd_by_lane[target].append(value)
            even = [value for lane_values in even_by_lane for value in lane_values]
            odd = [value for lane_values in odd_by_lane for value in lane_values]
            c0_word = (tile * _MXFP6_TILE_BYTES + block * 16) // 4
            c1_word = (tile * _MXFP6_TILE_BYTES + 16384 + block * 8) // 4
            if const_expr(codec == "mxfp6"):  # noqa: SIM102
                if valid & (group_lane == 0):
                    fp6_words = emit_f32_to_e2m3_native(
                        fx.Vector.from_elements(even, fx.Float32),
                        fx.Vector.from_elements(odd, fx.Float32),
                        forward_scale,
                    )
                    for index in range_constexpr(4):
                        buf_copy_store(
                            destination,
                            c0_word + index,
                            (amax == 0.0).select(fx.Int32(0), fp6_words[index]),
                            fx.Int32,
                        )
                    for index in range_constexpr(2):
                        buf_copy_store(
                            destination,
                            c1_word + index,
                            (amax == 0.0).select(fx.Int32(0), fp6_words[4 + index]),
                            fx.Int32,
                        )
                    buf_copy_store(
                        scales,
                        scale_address,
                        stored_scale.to(fx.Int8),
                        fx.Int8,
                    )

    return attention_output_a2a_push


def make_attention_output_packed_copy_kernel(
    *,
    rank,
    npes,
    seq_local,
    heads_local,
    head_dim,
    block_num,
    warp_num_per_block,
    codec,
):
    """Copy canonical local MX packs into peer final-GEMM layouts."""
    heads_total = heads_local * npes
    local_k = heads_local * head_dim
    global_k = heads_total * head_dim
    global_warps = block_num * warp_num_per_block
    peer_warps = global_warps // npes
    if codec == "mxfp6":
        row_tiles = _ceil(seq_local, _MX_TILE_ROWS) // _MX_TILE_ROWS
        local_k_tiles = local_k // _MX_K_TILE
        local_nk_pad = local_k_tiles + _MX_K_GUARD_TILES
        global_nk_pad = global_k // _MX_K_TILE + _MX_K_GUARD_TILES
        source_row_tiles = _ceil(seq_local, _MX_TILE_ROWS) // _MX_TILE_ROWS * npes
        source_payload_bytes = source_row_tiles * local_nk_pad * _MXFP6_TILE_BYTES
        source_scale_bytes = source_row_tiles * local_nk_pad * _MX_SCALE_TILE_BYTES
        output_payload_bytes = row_tiles * global_nk_pad * _MXFP6_TILE_BYTES
        output_scale_bytes = row_tiles * global_nk_pad * _MX_SCALE_TILE_BYTES
        payload_chunks_per_tile = _MXFP6_TILE_BYTES // 16
        scale_chunks_per_tile = _MX_SCALE_TILE_BYTES // 16
    else:
        local_payload_row_bytes = local_k // 2
        global_payload_row_bytes = global_k // 2
        local_scale_columns = local_k // _MX_GROUP_SIZE
        global_scale_columns = global_k // _MX_GROUP_SIZE
        global_scale_columns_padded = _ceil(global_scale_columns, 8)
        source_payload_bytes = seq_local * npes * local_payload_row_bytes
        source_scale_bytes = seq_local * npes * local_scale_columns
        output_payload_bytes = seq_local * global_payload_row_bytes
        output_scale_bytes = (
            _ceil(seq_local, _MX_TILE_ROWS) * global_scale_columns_padded
        )

    shared_storage = fx.struct(
        type(
            "_AttentionOutputPackedCopySharedStorage",
            (),
            {
                "__annotations__": {
                    "p2p_payload": fx.Array[fx.Int64, npes, 16],
                    "p2p_scale": fx.Array[fx.Int64, npes, 16],
                }
            },
        )
    )

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def attention_output_packed_copy(
        addr_source_payload: fx.Int64,
        addr_source_scale: fx.Int64,
        addr_p2p_payload: fx.Int64,
        addr_p2p_scale: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        warp = tid >> 6
        lane = tid & 63
        global_warp = bid * warp_num_per_block + warp
        dest_pe = global_warp % npes
        peer_warp = global_warp // npes

        payload_table = ptr_buf_tensor(
            addr_p2p_payload, fx.Int64, num_records_bytes=npes * 8
        )
        scale_table = ptr_buf_tensor(
            addr_p2p_scale, fx.Int64, num_records_bytes=npes * 8
        )
        shared = fx.SharedAllocator().allocate(shared_storage).peek()
        payload_bases = shared.p2p_payload.view(fx.make_layout(npes, 1))
        scale_bases = shared.p2p_scale.view(fx.make_layout(npes, 1))
        if lane < npes:
            fx.memref_store(
                buf_copy_load(payload_table, lane, fx.Int64),
                payload_bases,
                lane,
            )
            fx.memref_store(
                buf_copy_load(scale_table, lane, fx.Int64),
                scale_bases,
                lane,
            )
        fx.barrier()

        payload_base = wave_uniform_i64(
            fx.Uint64(fx.memref_load(payload_bases, dest_pe))
        )
        scale_base = wave_uniform_i64(fx.Uint64(fx.memref_load(scale_bases, dest_pe)))
        source_payload = ptr_buf_tensor(
            addr_source_payload,
            fx.Int32,
            unit_elems=4,
            unit_stride=1,
            num_records_bytes=source_payload_bytes,
        )
        destination_payload = ptr_buf_tensor(
            payload_base,
            fx.Int32,
            unit_elems=4,
            unit_stride=1,
            num_records_bytes=output_payload_bytes,
        )

        if const_expr(codec == "mxfp6"):
            source_scale = ptr_buf_tensor(
                addr_source_scale,
                fx.Int32,
                unit_elems=4,
                unit_stride=1,
                num_records_bytes=source_scale_bytes,
            )
            destination_scale = ptr_buf_tensor(
                scale_base,
                fx.Int32,
                unit_elems=4,
                unit_stride=1,
                num_records_bytes=output_scale_bytes,
            )
            payload_chunks_per_dest = (
                row_tiles * local_k_tiles * payload_chunks_per_tile
            )
            payload_groups = (payload_chunks_per_dest + 63) // 64
            for group in range(peer_warp, payload_groups, peer_warps):
                item = group * 64 + lane
                if item < payload_chunks_per_dest:
                    chunk = item % payload_chunks_per_tile
                    tile_item = item // payload_chunks_per_tile
                    local_k_tile = tile_item % local_k_tiles
                    local_row_tile = tile_item // local_k_tiles
                    source_row_tile = dest_pe * row_tiles + local_row_tile
                    source_tile = source_row_tile * local_nk_pad + local_k_tile
                    destination_tile = (
                        local_row_tile * global_nk_pad
                        + rank * local_k_tiles
                        + local_k_tile
                    )
                    value = buf_copy_load(
                        source_payload,
                        source_tile * payload_chunks_per_tile * 4 + chunk * 4,
                        fx.Int32,
                        4,
                    )
                    buf_copy_store(
                        destination_payload,
                        destination_tile * payload_chunks_per_tile * 4 + chunk * 4,
                        value,
                        fx.Int32,
                        4,
                    )

            scale_chunks_per_dest = row_tiles * local_k_tiles * scale_chunks_per_tile
            scale_groups = (scale_chunks_per_dest + 63) // 64
            for group in range(peer_warp, scale_groups, peer_warps):
                item = group * 64 + lane
                if item < scale_chunks_per_dest:
                    chunk = item % scale_chunks_per_tile
                    tile_item = item // scale_chunks_per_tile
                    local_k_tile = tile_item % local_k_tiles
                    local_row_tile = tile_item // local_k_tiles
                    source_row_tile = dest_pe * row_tiles + local_row_tile
                    source_tile = source_row_tile * local_nk_pad + local_k_tile
                    destination_tile = (
                        local_row_tile * global_nk_pad
                        + rank * local_k_tiles
                        + local_k_tile
                    )
                    value = buf_copy_load(
                        source_scale,
                        source_tile * scale_chunks_per_tile * 4 + chunk * 4,
                        fx.Int32,
                        4,
                    )
                    buf_copy_store(
                        destination_scale,
                        destination_tile * scale_chunks_per_tile * 4 + chunk * 4,
                        value,
                        fx.Int32,
                        4,
                    )
        else:
            source_scale = ptr_buf_tensor(
                addr_source_scale,
                fx.Int8,
                num_records_bytes=source_scale_bytes,
            )
            destination_scale = ptr_buf_tensor(
                scale_base,
                fx.Int8,
                num_records_bytes=output_scale_bytes,
            )
            payload_chunks_per_row = local_payload_row_bytes // 16
            payload_chunks_per_dest = seq_local * payload_chunks_per_row
            payload_groups = (payload_chunks_per_dest + 63) // 64
            for group in range(peer_warp, payload_groups, peer_warps):
                item = group * 64 + lane
                if item < payload_chunks_per_dest:
                    local_row = item // payload_chunks_per_row
                    chunk = item % payload_chunks_per_row
                    source_row = dest_pe * seq_local + local_row
                    source_chunk = source_row * payload_chunks_per_row + chunk
                    destination_chunk = (
                        local_row * (global_payload_row_bytes // 16)
                        + rank * payload_chunks_per_row
                        + chunk
                    )
                    value = buf_copy_load(
                        source_payload,
                        source_chunk * 4,
                        fx.Int32,
                        4,
                    )
                    buf_copy_store(
                        destination_payload,
                        destination_chunk * 4,
                        value,
                        fx.Int32,
                        4,
                    )

            scale_items_per_dest = seq_local * local_scale_columns
            scale_groups = (scale_items_per_dest + 63) // 64
            for group in range(peer_warp, scale_groups, peer_warps):
                item = group * 64 + lane
                if item < scale_items_per_dest:
                    local_row = item // local_scale_columns
                    local_group = item % local_scale_columns
                    source_row = dest_pe * seq_local + local_row
                    global_group = rank * local_scale_columns + local_group
                    source_index = source_row * local_scale_columns + local_group
                    destination_index = (
                        (local_row // 32 * global_scale_columns_padded) * 32
                        + (global_group // 8) * 256
                        + (global_group % 4) * 64
                        + (local_row % 16) * 4
                        + (global_group % 8) // 4 * 2
                        + (local_row % 32) // 16
                    )
                    value = buf_copy_load(
                        source_scale,
                        source_index,
                        fx.Int8,
                    )
                    buf_copy_store(
                        destination_scale,
                        destination_index,
                        value,
                        fx.Int8,
                    )

    return attention_output_packed_copy


def make_attention_output_packed_copy_jit(
    *,
    rank,
    npes,
    seq_local,
    heads_local,
    head_dim,
    block_num,
    warp_num_per_block,
    codec,
):
    payload = make_attention_output_packed_copy_kernel(
        rank=rank,
        npes=npes,
        seq_local=seq_local,
        heads_local=heads_local,
        head_dim=head_dim,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
        codec=codec,
    )
    completion = make_attention_a2a_completion_kernel(rank=rank, npes=npes)
    key = (
        "packed_copy",
        rank,
        npes,
        seq_local,
        heads_local,
        head_dim,
        block_num,
        warp_num_per_block,
        codec,
        current_target().arch,
        _JIT_CACHE_TAG,
    )

    @flyc.jit
    def launch(
        addr_source_payload: fx.Int64,
        addr_source_scale: fx.Int64,
        addr_p2p_payload: fx.Int64,
        addr_p2p_scale: fx.Int64,
        addr_xdb_mem: fx.Int64,
        addr_p2p_xdb_mem: fx.Int64,
        addr_xdb_flag: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        _ = key
        payload(
            addr_source_payload,
            addr_source_scale,
            addr_p2p_payload,
            addr_p2p_scale,
        ).launch(
            grid=(block_num, 1, 1),
            block=(warp_num_per_block * 64, 1, 1),
            stream=stream,
        )
        for phase in range_constexpr(2):
            completion(
                addr_xdb_mem,
                addr_p2p_xdb_mem,
                addr_xdb_flag,
                phase == 0,
            ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def make_attention_output_a2a_jit(
    *,
    rank,
    npes,
    seq_local,
    heads_local,
    head_dim,
    block_num,
    warp_num_per_block,
    fp8_fnuz,
    codec,
):
    payload = make_attention_output_a2a_kernel(
        rank=rank,
        npes=npes,
        seq_local=seq_local,
        heads_local=heads_local,
        head_dim=head_dim,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
        fp8_fnuz=fp8_fnuz,
        codec=codec,
    )
    completion = make_attention_a2a_completion_kernel(rank=rank, npes=npes)
    key = (
        rank,
        npes,
        seq_local,
        heads_local,
        head_dim,
        block_num,
        warp_num_per_block,
        fp8_fnuz,
        codec,
        current_target().arch,
        _JIT_CACHE_TAG,
    )

    @flyc.jit
    def launch(
        addr_input: fx.Int64,
        addr_input_scale: fx.Int64,
        addr_p2p_payload: fx.Int64,
        addr_p2p_scale: fx.Int64,
        addr_xdb_mem: fx.Int64,
        addr_p2p_xdb_mem: fx.Int64,
        addr_xdb_flag: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        _ = key
        payload(
            addr_input,
            addr_input_scale,
            addr_p2p_payload,
            addr_p2p_scale,
        ).launch(
            grid=(block_num, 1, 1),
            block=(warp_num_per_block * 64, 1, 1),
            stream=stream,
        )
        for phase in range_constexpr(2):
            completion(
                addr_xdb_mem,
                addr_p2p_xdb_mem,
                addr_xdb_flag,
                phase == 0,
            ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch
