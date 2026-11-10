# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Quantized inverse-Ulysses output all-to-all over MoRI peer writes."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Stream

from .attention_a2a_intranode_kernel import make_attention_a2a_completion_kernel
from .communication_ops_utils import wave_uniform_i64
from .quant_utils import emit_f32_to_fp8
from .tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor

_JIT_CACHE_TAG = "attention_output_a2a_intranode"
_ROWS_PER_WAVE = 4
_LANES_PER_ROW = 16
_VALUES_PER_LANE = 8
_MX_GROUP_SIZE = 32
# The per_1x32 quantizer pads the rows of a shuffled A4W4 scale tensor to this.
_A4W4_SCALE_ROW_TILE = 256
# Must equal aiter::kAmaxFloor (csrc/include/quant_amax_floor.h), the value the
# per_1x128 quantizer starts every group amax from.
_AMAX_FLOOR = 1e-10


def _ceil(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _check_compile_target(arch):
    target = flyc.get_backend().target.arch
    if target != arch:
        raise RuntimeError(
            f"FlyDSL compiles for {target}, but the attention output A2A runs on {arch}"
        )


def mxfp6_pack_extents(*, seq_local, heads_local, head_dim, npes):
    """Byte extents of the source and destination A6W6 packs the packed copy addresses.

    Returns (source payload, source scale, destination payload, destination scale).
    The tile geometry comes from the A6W6 packer, which owns the layout, so callers can
    check these against mxfp6_gemm_pack_size.
    """
    from aiter.ops.gemm_op_a6w6 import (
        _K_TILE,
        _PACKED_TILE_BYTES,
        _PADK,
        _SCALE_TILE_BYTES,
        _TILE,
    )

    row_tiles = _ceil(seq_local, _TILE) // _TILE
    local_nk_pad = heads_local * head_dim // _K_TILE + _PADK
    global_nk_pad = heads_local * npes * head_dim // _K_TILE + _PADK
    return (
        npes * row_tiles * local_nk_pad * _PACKED_TILE_BYTES,
        npes * row_tiles * local_nk_pad * _SCALE_TILE_BYTES,
        row_tiles * global_nk_pad * _PACKED_TILE_BYTES,
        row_tiles * global_nk_pad * _SCALE_TILE_BYTES,
    )


def _a4w4_scale_index(row, group, padded_columns):
    """Byte offset of E8M0 scale (row, group) in the shuffled A4W4 scale layout.

    Must match aiter::mx_scale_shuffle_idx (csrc/include/mx_quant_utils.h), the
    layout the per_1x32 quantizer writes with shuffle_scale.
    """
    return (
        (row // 32 * padded_columns) * 32
        + (group // 8) * 256
        + (group % 4) * 64
        + (row % 16) * 4
        + (group % 8) // 4 * 2
        + (row % 32) // 16
    )


def make_attention_output_a2a_kernel(
    *,
    rank,
    npes,
    seq_local,
    heads_local,
    head_dim,
    block_num,
    warp_num_per_block,
    arch,
    codec,
):
    """Build the BF16 BSHD -> E4M3 inverse-Ulysses push kernel."""
    if head_dim != _LANES_PER_ROW * _VALUES_PER_LANE:
        raise ValueError(f"expected head_dim=128, got {head_dim}")
    if codec not in {"e4m3", "e4m3_external_scale"}:
        raise ValueError(f"the push kernel quantizes only E4M3 codecs, got {codec!r}")
    if codec == "e4m3_external_scale" and arch != "gfx950":
        raise RuntimeError(f"{codec} attention output requires gfx950")

    heads_total = heads_local * npes
    rows_per_peer = seq_local * heads_local
    input_rows = seq_local * npes * heads_local
    output_rows = seq_local * heads_total
    input_bytes = input_rows * head_dim * 2
    output_bytes = output_rows * head_dim
    scale_bytes = (1 if codec == "e4m3_external_scale" else output_rows) * 4
    fp8_max = 240.0 if arch == "gfx942" else 448.0

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
            unit_elems=2,
            unit_stride=1,
            num_records_bytes=output_bytes,
        )
        scales = ptr_buf_tensor(
            scale_base,
            fx.Float32,
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
            if const_expr(codec == "e4m3_external_scale"):
                # TorchAO's per-tensor activation quantization: divide by the scale,
                # saturate with NaN-propagating clamps, then cast.
                scale = buf_copy_load(input_scales, dest_pe, fx.Float32)
                clamped = [
                    fx.min(
                        fx.max(values[index] / scale, fx.Float32(-fp8_max)),
                        fx.Float32(fp8_max),
                    )
                    for index in range_constexpr(_VALUES_PER_LANE)
                ]
            else:
                # aiter's per_1x128 quantizer: an amax that ignores NaN, and a clamp
                # that saturates NaN to -fp8_max as its v_med3_f32 does.
                amax = fx.Float32(_AMAX_FLOOR)
                for index in range_constexpr(_VALUES_PER_LANE):
                    amax = fx.maxnumf(amax, fmath.absf(values[index]))
                for shift in (1, 2, 4, 8):
                    amax = fx.maxnumf(amax, fx.gpu.shuffle_xor(amax, shift, 64))
                scale = amax * fx.Float32(1.0 / fp8_max)
                inverse = fx.Float32(1.0) / scale
                clamped = [
                    fx.minnumf(
                        fx.maxnumf(values[index] * inverse, fx.Float32(-fp8_max)),
                        fx.Float32(fp8_max),
                    )
                    for index in range_constexpr(_VALUES_PER_LANE)
                ]
            pairs = [
                fx.Uint32(emit_f32_to_fp8(clamped[2 * pair], clamped[2 * pair + 1]))
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
        from aiter.ops.gemm_op_a6w6 import (
            _K_TILE,
            _PACKED_TILE_BYTES,
            _PADK,
            _SCALE_TILE_BYTES,
            _TILE,
        )

        row_tiles = _ceil(seq_local, _TILE) // _TILE
        local_k_tiles = local_k // _K_TILE
        local_nk_pad = local_k_tiles + _PADK
        global_nk_pad = global_k // _K_TILE + _PADK
        (
            source_payload_bytes,
            source_scale_bytes,
            output_payload_bytes,
            output_scale_bytes,
        ) = mxfp6_pack_extents(
            seq_local=seq_local, heads_local=heads_local, head_dim=head_dim, npes=npes
        )
        tile_chunks = (_PACKED_TILE_BYTES // 16, _SCALE_TILE_BYTES // 16)
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
            _ceil(seq_local, _A4W4_SCALE_ROW_TILE) * global_scale_columns_padded
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
            # Payload and scale tiles share one tile order; only the tile size differs.
            tile_sources = (source_payload, source_scale)
            tile_destinations = (destination_payload, destination_scale)
            for kind in range_constexpr(2):
                chunks_per_tile = tile_chunks[kind]
                chunks_per_dest = row_tiles * local_k_tiles * chunks_per_tile
                chunk_groups = (chunks_per_dest + 63) // 64
                for group in range(peer_warp, chunk_groups, peer_warps):
                    item = group * 64 + lane
                    if item < chunks_per_dest:
                        chunk = item % chunks_per_tile
                        tile_item = item // chunks_per_tile
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
                            tile_sources[kind],
                            source_tile * chunks_per_tile * 4 + chunk * 4,
                            fx.Int32,
                            4,
                        )
                        buf_copy_store(
                            tile_destinations[kind],
                            destination_tile * chunks_per_tile * 4 + chunk * 4,
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
                    destination_index = _a4w4_scale_index(
                        local_row, global_group, global_scale_columns_padded
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
    arch,
    codec,
    spin_timeout_iters=0,
):
    _check_compile_target(arch)
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
    completion = make_attention_a2a_completion_kernel(
        rank=rank, npes=npes, spin_timeout_iters=spin_timeout_iters
    )
    key = (
        "packed_copy",
        rank,
        npes,
        seq_local,
        heads_local,
        head_dim,
        block_num,
        warp_num_per_block,
        arch,
        codec,
        spin_timeout_iters,
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
    arch,
    codec,
    spin_timeout_iters=0,
):
    _check_compile_target(arch)
    payload = make_attention_output_a2a_kernel(
        rank=rank,
        npes=npes,
        seq_local=seq_local,
        heads_local=heads_local,
        head_dim=head_dim,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
        arch=arch,
        codec=codec,
    )
    completion = make_attention_a2a_completion_kernel(
        rank=rank, npes=npes, spin_timeout_iters=spin_timeout_iters
    )
    key = (
        rank,
        npes,
        seq_local,
        heads_local,
        head_dim,
        block_num,
        warp_num_per_block,
        arch,
        codec,
        spin_timeout_iters,
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
