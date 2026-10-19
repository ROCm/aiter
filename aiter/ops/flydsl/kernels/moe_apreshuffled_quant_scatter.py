# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Separate MXFP4 quantization and A-preshuffle scatter kernels for gfx1250.

Quantization writes compact source rows. Scatter gathers those rows into
M16 payload tiles and M16K4 scales. Prequantized MoRI rows enter
scatter directly.
Payload vectors contain four dwords (16 bytes). Address names state bytes or
dwords; K256 offsets count 256-feature groups, not bytes.
"""

import types
from collections.abc import Callable

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels import buffer_ops, tensor_shim
from aiter.ops.flydsl.kernels import gemm_common_gfx1250 as gemm_common
from aiter.ops.flydsl.kernels import (
    moe_fused_route_quant_scatter as route_quant,
)

# Each K1024 slice contains four K256 blocks. Padding and the K32 swizzle
# distribute vector payload accesses and scale transposes across LDS banks.
_M_TILE_ROWS = 16
_K_SLICE_SIZE = 1024
_K256_PER_SLICE = 4
_SCALE_DWORDS_PER_SLICE_ROW = 8
_PAYLOAD_LDS_ROW_DWORDS = 33
_PAYLOAD_LDS_K256_DWORDS = _M_TILE_ROWS * _PAYLOAD_LDS_ROW_DWORDS
_PAYLOAD_LDS_SLICE_DWORDS = _K256_PER_SLICE * _PAYLOAD_LDS_K256_DWORDS
_SCALE_LDS_ROW_DWORDS = _SCALE_DWORDS_PER_SLICE_ROW + 1
_SCALE_LDS_SLICE_DWORDS = _M_TILE_ROWS * _SCALE_LDS_ROW_DWORDS


@fx.struct
class _PayloadLdsK1024:
    """M16 x K1024 tile with one padding dword per M row and K256 slice."""

    payload: fx.Array[fx.Int32, _PAYLOAD_LDS_SLICE_DWORDS, 16]


@fx.struct
class _ScaleLdsK1024:
    """Padded M16 x 8-dword scale transpose tile."""

    scale: fx.Array[fx.Int32, _SCALE_LDS_SLICE_DWORDS, 16]


@fx.struct
class _PayloadLdsK4096:
    """Four K1024 payload slices sharing one ready/reuse barrier pair."""

    payload: fx.Array[fx.Int32, 4 * _PAYLOAD_LDS_SLICE_DWORDS, 16]


@fx.struct
class _ScaleLdsK4096:
    """Four padded scale slices for one K4096 tile."""

    scale: fx.Array[fx.Int32, 4 * _SCALE_LDS_SLICE_DWORDS, 16]


def build_moe_apre_quant_module(
    feat_dim: int,
    wmma_rep: int,
) -> Callable:
    """Build a BF16-to-MXFP4 quantizer that writes compact source rows.

    Args:
        feat_dim: Logical feature count K, divisible by 256.
        wmma_rep: GEMM scale-layout repetition factor.

    Returns:
        A JIT launcher accepting BF16 input, compact payload/scale buffers,
        token and workgroup counts, and an optional stream.

    Raises:
        ValueError: K is not divisible by 256.
        NotImplementedError: The target lacks gfx1250 pk8 or wave32 support.
    """
    quant_layout = route_quant._quant_layout(feat_dim, "fp4", wmma_rep)
    if not quant_layout.use_pk8 or quant_layout.lanes_per_mx_block != 4:
        raise NotImplementedError("A-preshuffle quant requires gfx1250 pk8")
    if feat_dim % 256:
        raise ValueError("A-preshuffle quant requires K divisible by 256")
    if quant_layout.wave_size != 32:
        raise NotImplementedError("A-preshuffle quant requires wave32")

    threads_per_block = 512
    waves_per_block = 16
    num_k256_groups = feat_dim // 256
    payload_bytes_per_row = quant_layout.payload_bytes_per_row
    scale_dwords_per_row = quant_layout.mx_blocks_per_row // 4
    module_name = f"moe_apre_quant_fd{feat_dim}_r{wmma_rep}_fp4_pk8_w16"

    @flyc.kernel(name=module_name, known_block_size=[threads_per_block, 1, 1])
    def quant_kernel(
        hidden: fx.Pointer,
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        num_tokens: Int32,
        num_workgroups: Int32,
    ):
        i32 = T.i32
        f32 = T.f32
        zero_i32 = arith.constant(0, type=i32)
        one_i32 = arith.constant(1, type=i32)
        four_i32 = arith.constant(4, type=i32)
        mantissa_bits_i32 = arith.constant(23, type=i32)
        reciprocal_exponent_bias_i32 = arith.constant(254, type=i32)
        zero_f32 = arith.constant(0.0, type=f32)
        wave_size_i32 = arith.constant(quant_layout.wave_size, type=i32)

        thread_id = fx.Uint32(fx.thread_idx.x)
        block_id = fx.Uint32(fx.block_idx.x)
        wave_id = thread_id // fx.Uint32(quant_layout.wave_size)
        lane_id = thread_id - wave_id * fx.Uint32(quant_layout.wave_size)
        first_source_token = block_id * fx.Uint32(waves_per_block) + wave_id
        source_token_stride = fx.Uint32(num_workgroups) * fx.Uint32(
            waves_per_block
        )
        for source_token in range(
            first_source_token, fx.Uint32(num_tokens), source_token_stride
        ):
            for k256_group in range(
                fx.Uint32(0), fx.Uint32(num_k256_groups), 1
            ):
                mx_block_in_wave = lane_id // fx.Uint32(
                    quant_layout.lanes_per_mx_block
                )
                lane_in_mx_block = lane_id - mx_block_in_wave * fx.Uint32(
                    quant_layout.lanes_per_mx_block
                )
                # Field names follow the shared quant-emitter interface.
                quant_context = types.SimpleNamespace(
                    i32=i32,
                    f32=f32,
                    block_iters=1,
                    payload_base=fx.Int64(ptrtoint(compact_payload)),
                    payload_bytes_per_row=payload_bytes_per_row,
                    hidden_base=fx.Int64(ptrtoint(hidden)),
                    feat_bytes_per_row=feat_dim * 2,
                    feat_row_i32=source_token,
                    payload_num_records=payload_bytes_per_row,
                    prequantized=False,
                    mx_blocks_per_wave_iter=(
                        quant_layout.mx_blocks_per_wave_iter
                    ),
                    mx_blocks_per_row=quant_layout.mx_blocks_per_row,
                    amax_shuffle_dists=quant_layout.amax_shuffle_dists,
                    is_fp8=False,
                    use_native=quant_layout.use_native,
                    use_pk8=True,
                    mx_dtype=quant_layout.mx_dtype,
                    c0_i32=zero_i32,
                    c1_i32=one_i32,
                    c4_i32=four_i32,
                    c23_i32=mantissa_bits_i32,
                    c254_i32=reciprocal_exponent_bias_i32,
                    c0_f32=zero_f32,
                    c_wave=wave_size_i32,
                    c_elems_per_lane=arith.constant(
                        quant_layout.elems_per_lane, type=i32
                    ),
                    c_payload_bytes_per_block=arith.constant(
                        quant_layout.payload_bytes_per_block, type=i32
                    ),
                    c_payload_bytes_per_lane=arith.constant(
                        quant_layout.payload_bytes_per_lane, type=i32
                    ),
                    c_wmma_rep=arith.constant(wmma_rep, type=i32),
                    block_in_wave=mx_block_in_wave,
                    lane_in_block=lane_in_mx_block,
                    is_block_lead=lane_in_mx_block == zero_i32,
                    payload_dests=[
                        types.SimpleNamespace(payload_row_i32=source_token)
                    ],
                    dests=[
                        types.SimpleNamespace(
                            payload_row_i32=source_token,
                            scale_row_dword_base=zero_i32,
                        )
                    ],
                    scale_t=tensor_shim.ptr_buf_tensor(compact_scale, fx.Int8),
                    scale_rsrc=buffer_ops.create_buffer_resource_from_addr(
                        fx.Int64(ptrtoint(compact_scale)),
                        num_records_bytes=route_quant._SCALE_RSRC_MAX_BYTES,
                    ),
                    scale_pack_dwords=True,
                    compact_scale=True,
                    c_scale_dwords_per_row=arith.constant(
                        scale_dwords_per_row, type=i32
                    ),
                    mx_group_base=k256_group,
                )
                route_quant._emit_quant_one_k_group(quant_context, k256_group)

    @flyc.jit
    def launch_quant(
        hidden: fx.Pointer,
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        num_tokens: fx.Int32,
        num_workgroups: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        quant_kernel(
            hidden,
            compact_payload,
            compact_scale,
            num_tokens,
            num_workgroups,
        ).launch(
            grid=(arith.index_cast(T.index, num_workgroups), 1, 1),
            block=(threads_per_block, 1, 1),
            stream=stream,
        )

    launch_quant.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": tensor_shim.AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": (
                tensor_shim.AITER_FLYDSL_KERNARG_PRELOAD_COUNT
            ),
        },
    }
    return launch_quant


def build_moe_apre_scatter_module(
    feat_dim: int,
    wmma_rep: int,
    src_scale_stride_bytes: int = 0,
) -> Callable:
    """Scatter compact MXFP4 rows into M16 A-preshuffled payload and scales.

    Each 16-wave workgroup gathers an M16 tile through padded LDS. Input scale
    rows may have a padded byte stride; output scales use M16K4. K >= 2048
    uses K4096 tiles with next-tile prefetch; smaller inputs use K1024.
    K4096 loads one scale dword per lane and stages four K1024 payload slices
    before one ready/reuse barrier pair. A tail stages only its remaining
    slices and masks unused scale lanes. Eight waves write scales; all 16
    waves transfer payloads. Quantization runs in a separate kernel when needed.

    Args:
        feat_dim: Logical feature count K, divisible by 1024.
        wmma_rep: Number of M16 tiles grouped in the output scale layout.
        src_scale_stride_bytes: Input scale row stride, including transport
            padding. Zero selects the compact stride K / 32.

    Returns:
        A JIT launcher accepting compact and output payload/scale buffers,
        a row-to-token map, token/row/workgroup counts, and an optional stream.
        Negative map entries are padding; fully empty M16 tiles are untouched.

    Raises:
        ValueError: K or the input scale stride has unsupported alignment.
        NotImplementedError: The target lacks gfx1250 pk8 or wave32 support.
    """
    quant_layout = route_quant._quant_layout(feat_dim, "fp4", wmma_rep)
    if not quant_layout.use_pk8 or quant_layout.lanes_per_mx_block != 4:
        raise NotImplementedError("A-preshuffle scatter requires gfx1250 pk8")
    k_tile_size = 4096 if feat_dim >= 2048 else 1024
    if feat_dim % 1024:
        raise ValueError("A-preshuffle scatter requires K divisible by 1024")
    slices_per_k_tile = k_tile_size // _K_SLICE_SIZE
    num_k1024_slices = feat_dim // _K_SLICE_SIZE
    num_k_tiles = (
        num_k1024_slices + slices_per_k_tile - 1
    ) // slices_per_k_tile

    wave_size = quant_layout.wave_size
    if wave_size != 32:
        raise NotImplementedError("A-preshuffle scatter requires wave32")
    threads_per_block = 512
    payload_bytes_per_row = quant_layout.payload_bytes_per_row
    payload_dwords_per_row = payload_bytes_per_row // 4
    scale_dwords_per_row = quant_layout.mx_blocks_per_row // 4
    src_scale_stride_bytes = (
        src_scale_stride_bytes or quant_layout.mx_blocks_per_row
    )
    if (
        src_scale_stride_bytes < quant_layout.mx_blocks_per_row
        or src_scale_stride_bytes % 4
    ):
        raise ValueError(
            "input scale stride must cover the row and be dword aligned"
        )
    source_scale_stride_dwords = src_scale_stride_bytes // 4
    module_name = (
        f"moe_apre_scatter_fd{feat_dim}_r{wmma_rep}_fp4_pk8_w16_tk{k_tile_size}"
    )
    if src_scale_stride_bytes != quant_layout.mx_blocks_per_row:
        module_name += f"_ss{src_scale_stride_bytes}"
    module_name += "_pf_sw8"

    @flyc.kernel(name=module_name, known_block_size=[threads_per_block, 1, 1])
    def scatter_kernel(
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        row_to_token: fx.Pointer,
        num_tokens: Int32,
        num_output_rows: Int32,
        num_workgroups: Int32,
    ):
        i32 = T.i32
        zero_i32 = arith.constant(0, type=i32)
        one_i32 = arith.constant(1, type=i32)
        lds_load_b128, lds_store_b128 = gemm_common.make_lds_copy_ops(128)

        def _sync_lds():
            # Only LDS is exchanged between waves. Keep next-chunk global
            # loads and disjoint output stores in flight across the fence.
            rocdl.s_wait_dscnt(0)
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)

        thread_id = fx.Uint32(fx.thread_idx.x)
        block_id = fx.Uint32(fx.block_idx.x)
        wave_id = thread_id // fx.Uint32(wave_size)
        lane_id = thread_id - wave_id * fx.Uint32(wave_size)

        shared_allocator = fx.SharedAllocator()
        if const_expr(k_tile_size == 4096):
            payload_storage = shared_allocator.allocate(_PayloadLdsK4096).peek()
            scale_storage = shared_allocator.allocate(_ScaleLdsK4096).peek()
        else:
            payload_storage = shared_allocator.allocate(_PayloadLdsK1024).peek()
            scale_storage = shared_allocator.allocate(_ScaleLdsK1024).peek()
        payload_lds = payload_storage.payload.ptr
        payload_lds_base_bytes = fx.Index(fx.ptrtoint(payload_lds))
        scale_lds = scale_storage.scale.ptr
        source_payload_resource = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(compact_payload)),
            num_records_bytes=route_quant._SCALE_RSRC_MAX_BYTES,
        )
        output_payload_resource = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(grouped_payload)),
            num_records_bytes=route_quant._SCALE_RSRC_MAX_BYTES,
        )
        source_scale_size_dwords = fx.Uint32(num_tokens) * fx.Uint32(
            source_scale_stride_dwords
        )
        output_scale_size_dwords = fx.Uint32(num_output_rows) * fx.Uint32(
            scale_dwords_per_row
        )
        output_scale_resource = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(grouped_scale)),
            num_records_bytes=output_scale_size_dwords * fx.Uint32(4),
        )
        source_scale_buffer = tensor_shim.ptr_buf_tensor(
            compact_scale,
            fx.Int32,
            num_records_bytes=source_scale_size_dwords * fx.Uint32(4),
        )
        row_to_token_buffer = tensor_shim.ptr_buf_tensor(row_to_token)

        def _load_payload_slice(k256_offset, source_row):
            """Load one 16-byte vector per lane from a K1024 source slice."""
            source_payload_offset_dwords = (
                source_row * fx.Uint32(payload_dwords_per_row)
                + k256_offset * fx.Uint32(32)
                + lane_id * fx.Uint32(4)
            )
            source_payload_vec = fx.Vector(
                buffer_ops.buffer_load(
                    source_payload_resource,
                    source_payload_offset_dwords,
                    vec_width=4,
                    dtype=i32,
                )
            )
            return source_payload_vec

        def _load_scale_tile(
            first_k1024_slice, active_slices, source_row, row_is_valid
        ):
            scale_lane_is_active = lane_id < fx.Uint32(active_slices * 8)
            source_scale_offset_dwords = (
                row_is_valid & scale_lane_is_active
            ).select(
                source_row * fx.Uint32(source_scale_stride_dwords)
                + fx.Uint32(first_k1024_slice * 8)
                + lane_id,
                source_scale_size_dwords,
            )
            return fx.Int32(source_scale_buffer[source_scale_offset_dwords])

        def _load_k1024_slice(k256_offset, source_row, row_is_valid):
            is_scale_load_lane = lane_id < fx.Uint32(8)
            source_scale_offset_dwords = (
                row_is_valid & is_scale_load_lane
            ).select(
                source_row * fx.Uint32(source_scale_stride_dwords)
                + k256_offset * fx.Uint32(2)
                + lane_id,
                source_scale_size_dwords,
            )
            scale_dword = fx.Int32(
                source_scale_buffer[source_scale_offset_dwords]
            )
            rocdl.sched_barrier(0)
            return _load_payload_slice(k256_offset, source_row), scale_dword

        row_in_m16 = wave_id
        is_scale_load_lane = lane_id < fx.Uint32(_SCALE_DWORDS_PER_SLICE_ROW)

        def _stage_slice_to_lds(
            source_payload_vec, scale_dword, slice_id, row_is_valid
        ):
            """Stage payloads and K1024 scales, zeroing unmapped rows."""
            masked_payload_vec = fx.Vector.from_elements(
                [
                    row_is_valid.select(
                        ArithValue(source_payload_vec[j]), zero_i32
                    )
                    for j in range_constexpr(4)
                ],
                fx.Int32,
            )
            rocdl.sched_barrier(0)

            if const_expr(k_tile_size == 1024):
                # The ninth dword in each scale row breaks stride-8 LDS bank
                # aliasing. Inactive lanes write only to the padding column.
                scale_padding_row = (lane_id + fx.Uint32(8)) % fx.Uint32(16)
                scale_lds_offset_dwords = is_scale_load_lane.select(
                    row_in_m16 * fx.Uint32(_SCALE_LDS_ROW_DWORDS) + lane_id,
                    scale_padding_row * fx.Uint32(_SCALE_LDS_ROW_DWORDS)
                    + fx.Uint32(8),
                )
                scale_lds[
                    fx.Uint32(slice_id * _SCALE_LDS_SLICE_DWORDS)
                    + scale_lds_offset_dwords
                ] = scale_dword
                rocdl.sched_barrier(0)

            k256_in_slice = lane_id // fx.Uint32(8)
            k32_in_k256 = lane_id - k256_in_slice * fx.Uint32(8)
            swizzled_k32 = (
                k32_in_k256 + row_in_m16 // fx.Uint32(4)
            ) % fx.Uint32(8)
            payload_lds_offset_dwords = (
                k256_in_slice * fx.Uint32(_PAYLOAD_LDS_K256_DWORDS)
                + row_in_m16 * fx.Uint32(_PAYLOAD_LDS_ROW_DWORDS)
                + swizzled_k32 * fx.Uint32(4)
            )
            lds_store_b128(
                payload_lds_base_bytes
                + fx.Index(slice_id * _PAYLOAD_LDS_SLICE_DWORDS * 4),
                fx.Index(payload_lds_offset_dwords * fx.Uint32(4)),
                masked_payload_vec,
            )

        def _read_shuffled_slice_from_lds(k256_offset, slice_id, m16_tile_id):
            """Read payload and scale vectors in their destination order."""
            output_slice_offset_dwords = thread_id * fx.Uint32(4)
            output_k256_in_slice = output_slice_offset_dwords // fx.Uint32(512)
            output_k256_offset_dwords = (
                output_slice_offset_dwords
                - output_k256_in_slice * fx.Uint32(512)
            )
            output_k32_in_k256 = output_k256_offset_dwords // fx.Uint32(64)
            output_k32_offset_dwords = (
                output_k256_offset_dwords - output_k32_in_k256 * fx.Uint32(64)
            )
            output_row_in_m16 = output_k32_offset_dwords // fx.Uint32(4)
            output_swizzled_k32 = (
                output_k32_in_k256 + output_row_in_m16 // fx.Uint32(4)
            ) % fx.Uint32(8)
            read_payload_lds_offset_dwords = (
                output_k256_in_slice * fx.Uint32(_PAYLOAD_LDS_K256_DWORDS)
                + output_row_in_m16 * fx.Uint32(_PAYLOAD_LDS_ROW_DWORDS)
                + output_swizzled_k32 * fx.Uint32(4)
            )
            output_payload_vec = lds_load_b128(
                payload_lds_base_bytes
                + fx.Index(slice_id * _PAYLOAD_LDS_SLICE_DWORDS * 4),
                fx.Index(read_payload_lds_offset_dwords * fx.Uint32(4)),
            )

            output_scale_dword = fx.Int32(0)
            # Eight waves each write 64 B for a K1024 scale slice.
            output_scale_column = row_in_m16
            output_scale_row = lane_id
            is_scale_store_lane = (row_in_m16 < fx.Uint32(8)) & (
                lane_id < fx.Uint32(16)
            )
            scale_padding_row = (lane_id + fx.Uint32(8)) % fx.Uint32(16)
            inactive_scale_offset_dwords = scale_padding_row * fx.Uint32(
                _SCALE_LDS_ROW_DWORDS
            ) + fx.Uint32(8)
            if const_expr(k_tile_size == 4096):
                # Wide writes populate useful columns only.
                inactive_scale_offset_dwords = fx.Uint32(0)
            read_scale_lds_offset_dwords = is_scale_store_lane.select(
                output_scale_row * fx.Uint32(_SCALE_LDS_ROW_DWORDS)
                + output_scale_column,
                inactive_scale_offset_dwords,
            )
            if row_in_m16 < fx.Uint32(8):
                output_scale_dword = scale_lds[
                    fx.Uint32(slice_id * _SCALE_LDS_SLICE_DWORDS)
                    + read_scale_lds_offset_dwords
                ]

            scale_k256_in_slice = output_scale_column // fx.Uint32(2)
            scale_k128_in_k256 = (
                output_scale_column - scale_k256_in_slice * fx.Uint32(2)
            )
            output_scale_k256 = k256_offset + scale_k256_in_slice
            scale_row_group = m16_tile_id // fx.Uint32(wmma_rep)
            m16_in_scale_row_group = m16_tile_id - scale_row_group * fx.Uint32(
                wmma_rep
            )
            scale_group_offset_dwords = scale_row_group * fx.Uint32(
                scale_dwords_per_row * wmma_rep * 16
            ) + m16_in_scale_row_group * fx.Uint32(16)
            output_scale_offset_dwords = is_scale_store_lane.select(
                scale_group_offset_dwords
                + output_scale_k256 * fx.Uint32(2 * wmma_rep * 16)
                + scale_k128_in_k256 * fx.Uint32(wmma_rep * 16)
                + output_scale_row,
                output_scale_size_dwords,
            )
            return (
                output_payload_vec,
                output_scale_dword,
                output_scale_offset_dwords,
            )

        def _store_shuffled_slice(
            k256_offset,
            output_payload_vec,
            output_scale_dword,
            output_scale_offset_dwords,
            m16_tile_id,
        ):
            """Store a shuffled K1024 slice to global output buffers."""
            # Non-writing waves still take every payload barrier.
            if row_in_m16 < fx.Uint32(8):
                buffer_ops.buffer_store(
                    output_scale_dword,
                    output_scale_resource,
                    output_scale_offset_dwords * fx.Uint32(4),
                    offset_is_bytes=True,
                )
            buffer_ops.buffer_store(
                output_payload_vec,
                output_payload_resource,
                m16_tile_id * fx.Uint32(16 * payload_dwords_per_row)
                + k256_offset * fx.Uint32(512)
                + thread_id * fx.Uint32(4),
            )

        num_m16_tiles = fx.Uint32(num_output_rows) // fx.Uint32(16)
        for m16_tile_id in range(
            block_id, num_m16_tiles, fx.Uint32(num_workgroups)
        ):
            output_row = m16_tile_id * fx.Uint32(16) + row_in_m16
            # Fetch each row mapping once per wave, then use LDS both to
            # broadcast it across the wave and to let thread 0 detect a fully
            # empty M16 tile.  The resulting branch is uniform for the TG, so
            # it is safe to skip all K-slice barriers together.
            if lane_id == fx.Uint32(0):
                payload_lds[row_in_m16] = row_to_token_buffer[output_row]
            gpu.barrier()
            mapped_token = fx.Int32(payload_lds[row_in_m16])
            row_is_valid = mapped_token >= fx.Int32(0)
            source_row = row_is_valid.select(
                fx.Uint32(mapped_token), fx.Uint32(0)
            )

            if thread_id == fx.Uint32(0):
                tile_has_valid_rows = zero_i32
                for row_to_check in range(16):
                    mapped_row_is_valid = (
                        fx.Int32(payload_lds[fx.Uint32(row_to_check)])
                        >= zero_i32
                    )
                    tile_has_valid_rows = fx.Int32(
                        arith.ori(
                            arith.unwrap(tile_has_valid_rows),
                            arith.unwrap(
                                mapped_row_is_valid.select(one_i32, zero_i32)
                            ),
                        )
                    )
                payload_lds[fx.Uint32(16)] = tile_has_valid_rows
            gpu.barrier()
            tile_has_valid_rows = (
                fx.Int32(payload_lds[fx.Uint32(16)]) != zero_i32
            )

            if tile_has_valid_rows:
                if const_expr(k_tile_size == 1024):
                    k256_offset = fx.Uint32(0)
                    source_payload_vec, scale_dword = _load_k1024_slice(
                        k256_offset, source_row, row_is_valid
                    )
                    _stage_slice_to_lds(
                        source_payload_vec, scale_dword, 0, row_is_valid
                    )
                    _sync_lds()
                    shuffled_slice = _read_shuffled_slice_from_lds(
                        k256_offset, 0, m16_tile_id
                    )
                    _sync_lds()
                    _store_shuffled_slice(
                        k256_offset,
                        shuffled_slice[0],
                        shuffled_slice[1],
                        shuffled_slice[2],
                        m16_tile_id,
                    )
                else:
                    prefetched_scale_dword = _load_scale_tile(
                        0,
                        min(slices_per_k_tile, num_k1024_slices),
                        source_row,
                        row_is_valid,
                    )
                    prefetched_payload_slices = []
                    for slice_id in range_constexpr(
                        min(slices_per_k_tile, num_k1024_slices)
                    ):
                        prefetched_payload_slices.append(
                            _load_payload_slice(
                                fx.Uint32(slice_id * 4), source_row
                            )
                        )
                    for k_tile_id in range_constexpr(num_k_tiles):
                        first_k1024_slice = k_tile_id * slices_per_k_tile
                        active_slices = min(
                            slices_per_k_tile,
                            num_k1024_slices - first_k1024_slice,
                        )
                        next_first_slice = first_k1024_slice + slices_per_k_tile
                        next_active_slices = min(
                            slices_per_k_tile,
                            max(0, num_k1024_slices - next_first_slice),
                        )
                        current_scale_dword = prefetched_scale_dword
                        current_payload_slices = prefetched_payload_slices
                        prefetched_payload_slices = []
                        if const_expr(next_active_slices > 0):
                            prefetched_scale_dword = _load_scale_tile(
                                next_first_slice,
                                next_active_slices,
                                source_row,
                                row_is_valid,
                            )
                            for slice_id in range_constexpr(next_active_slices):
                                prefetched_payload_slices.append(
                                    _load_payload_slice(
                                        fx.Uint32(
                                            (next_first_slice + slice_id) * 4
                                        ),
                                        source_row,
                                    )
                                )
                        # Each lane places its four scale bytes directly
                        # into one of the four padded K1024 LDS slices.
                        scale_slice_id = lane_id // fx.Uint32(8)
                        scale_column = lane_id % fx.Uint32(8)
                        scale_lds[
                            scale_slice_id * fx.Uint32(_SCALE_LDS_SLICE_DWORDS)
                            + row_in_m16 * fx.Uint32(_SCALE_LDS_ROW_DWORDS)
                            + scale_column
                        ] = current_scale_dword
                        for slice_id in range_constexpr(active_slices):
                            _stage_slice_to_lds(
                                current_payload_slices[slice_id],
                                zero_i32,
                                slice_id,
                                row_is_valid,
                            )
                        _sync_lds()
                        shuffled_slices = []
                        for slice_id in range_constexpr(active_slices):
                            shuffled_slices.append(
                                _read_shuffled_slice_from_lds(
                                    fx.Uint32(
                                        (first_k1024_slice + slice_id) * 4
                                    ),
                                    slice_id,
                                    m16_tile_id,
                                )
                            )
                        _sync_lds()
                        for slice_id in range_constexpr(active_slices):
                            _store_shuffled_slice(
                                fx.Uint32((first_k1024_slice + slice_id) * 4),
                                shuffled_slices[slice_id][0],
                                shuffled_slices[slice_id][1],
                                shuffled_slices[slice_id][2],
                                m16_tile_id,
                            )

    @flyc.jit
    def launch_scatter(
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        row_to_token: fx.Pointer,
        num_tokens: fx.Int32,
        num_output_rows: fx.Int32,
        num_workgroups: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        scatter_kernel(
            compact_payload,
            compact_scale,
            grouped_payload,
            grouped_scale,
            row_to_token,
            num_tokens,
            num_output_rows,
            num_workgroups,
        ).launch(
            grid=(arith.index_cast(T.index, num_workgroups), 1, 1),
            block=(threads_per_block, 1, 1),
            stream=stream,
        )

    launch_scatter.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": tensor_shim.AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": (
                tensor_shim.AITER_FLYDSL_KERNARG_PRELOAD_COUNT
            ),
        },
    }
    return launch_scatter
