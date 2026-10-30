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
from flydsl.expr import arith, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels import buffer_ops, tensor_shim
from aiter.ops.flydsl.kernels import gemm_common_gfx1250 as gemm_common
from aiter.ops.flydsl.kernels import (
    moe_fused_route_quant_scatter as route_quant,
)
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr

# Each K1024 slice contains four K256 blocks. Padding and the K32 swizzle
# distribute vector payload accesses and scale transposes across LDS banks.
_M_TILE_ROWS = 16
_MAX_K_TILE_SIZE = 16384
_K_SLICE_SIZE = 1024
_K256_SIZE = 256
_MX_BLOCK_SIZE = 32
_DWORD_BYTES = 4
_FP4_VALUES_PER_BYTE = 2
_PAYLOAD_VECTOR_BYTES = 16
_PAYLOAD_VECTOR_DWORDS = _PAYLOAD_VECTOR_BYTES // _DWORD_BYTES
_K256_PER_SLICE = _K_SLICE_SIZE // _K256_SIZE
_K32_PER_K256 = _K256_SIZE // _MX_BLOCK_SIZE
_ROWS_PER_SWIZZLE_GROUP = 4
# E8M0 has one byte per MX block; four scales fit in one dword.
_SCALE_DWORDS_PER_SLICE_ROW = _K_SLICE_SIZE // _MX_BLOCK_SIZE // _DWORD_BYTES
_SCALE_DWORDS_PER_K256 = _K256_SIZE // _MX_BLOCK_SIZE // _DWORD_BYTES
_PAYLOAD_K256_ROW_DWORDS = _K256_SIZE // _FP4_VALUES_PER_BYTE // _DWORD_BYTES
_PAYLOAD_LDS_ROW_DWORDS = _PAYLOAD_K256_ROW_DWORDS + 1
_PAYLOAD_LDS_K256_DWORDS = _M_TILE_ROWS * _PAYLOAD_LDS_ROW_DWORDS
_PAYLOAD_LDS_SLICE_DWORDS = _K256_PER_SLICE * _PAYLOAD_LDS_K256_DWORDS
_SCALE_LDS_ROW_DWORDS = _SCALE_DWORDS_PER_SLICE_ROW + 1
_SCALE_LDS_SLICE_DWORDS = _M_TILE_ROWS * _SCALE_LDS_ROW_DWORDS


def validate_apre_k(feat_dim: int) -> None:
    """Require complete K1024 slices that fit the full-row LDS specialization."""
    if not 0 < feat_dim <= _MAX_K_TILE_SIZE or feat_dim % _K_SLICE_SIZE:
        raise ValueError(
            "A-preshuffle requires K divisible by "
            f"{_K_SLICE_SIZE} with 0 < K <= {_MAX_K_TILE_SIZE}, got K={feat_dim}"
        )


def build_moe_apre_quant_module(
    feat_dim: int,
    wmma_rep: int,
) -> Callable:
    """Build a BF16-to-MXFP4 quantizer that writes compact source rows.

    Args:
        feat_dim: Logical feature count K, a multiple of 1024 up to 16384.
        wmma_rep: GEMM scale-layout repetition factor.

    Returns:
        A JIT launcher accepting BF16 input, compact payload/scale buffers,
        token and workgroup counts, and an optional stream.

    Raises:
        ValueError: K is outside the supported range or not divisible by 1024.
        NotImplementedError: The target lacks gfx1250 pk8 or wave32 support.
    """
    validate_apre_k(feat_dim)
    quant_layout = route_quant._quant_layout(feat_dim, "fp4", wmma_rep)
    if not quant_layout.use_pk8 or quant_layout.lanes_per_mx_block != 4:
        raise NotImplementedError("A-preshuffle quant requires gfx1250 pk8")
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
        source_token_stride = fx.Uint32(num_workgroups) * fx.Uint32(waves_per_block)
        for source_token in range(
            first_source_token, fx.Uint32(num_tokens), source_token_stride
        ):
            for k256_group in range(fx.Uint32(0), fx.Uint32(num_k256_groups), 1):
                mx_block_in_wave = lane_id // fx.Uint32(quant_layout.lanes_per_mx_block)
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
                    mx_blocks_per_wave_iter=(quant_layout.mx_blocks_per_wave_iter),
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
                    payload_dests=[types.SimpleNamespace(payload_row_i32=source_token)],
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

    Each 16-wave workgroup gathers one M16 x K tile through padded LDS.
    All K1024 slices are loaded asynchronously before one data-ready barrier.
    Ordinary LDS reads and buffer stores write the A-preshuffled output.
    LDS is never reused; there is no persistent loop or cross-tile prefetch.
    Each wave ballots the same M16 row map, so early quit needs no TG barrier.
    Eight waves write scales; all 16 waves transfer payloads.

    Args:
        feat_dim: Logical feature count K, a multiple of 1024 up to 16384.
        wmma_rep: Number of M16 tiles grouped in the output scale layout.
        src_scale_stride_bytes: Input scale row stride, including transport
            padding. Zero selects the compact stride K / 32.

    Returns:
        A JIT launcher accepting compact and output payload/scale buffers,
        a row-to-token map, token/row/workgroup counts, and an optional stream.
        Negative map entries are padding; fully empty M16 tiles are untouched.

    Raises:
        ValueError: K is outside the supported range, not divisible by 1024,
            or the input scale stride is invalid.
        NotImplementedError: The target lacks gfx1250 pk8 or wave32 support.
    """
    validate_apre_k(feat_dim)
    quant_layout = route_quant._quant_layout(feat_dim, "fp4", wmma_rep)
    if not quant_layout.use_pk8 or quant_layout.lanes_per_mx_block != 4:
        raise NotImplementedError("A-preshuffle scatter requires gfx1250 pk8")

    wave_size = quant_layout.wave_size
    if wave_size != 32:
        raise NotImplementedError("A-preshuffle scatter requires wave32")
    threads_per_block = _M_TILE_ROWS * wave_size
    num_k_slices = feat_dim // _K_SLICE_SIZE
    payload_bytes_per_row = quant_layout.payload_bytes_per_row
    payload_dwords_per_row = payload_bytes_per_row // _DWORD_BYTES
    scale_dwords_per_row = quant_layout.mx_blocks_per_row // _DWORD_BYTES
    src_scale_stride_bytes = src_scale_stride_bytes or quant_layout.mx_blocks_per_row
    if (
        src_scale_stride_bytes < quant_layout.mx_blocks_per_row
        or src_scale_stride_bytes % _DWORD_BYTES
    ):
        raise ValueError("input scale stride must cover the row and be dword aligned")
    source_scale_stride_dwords = src_scale_stride_bytes // _DWORD_BYTES
    scale_load_rounds = (scale_dwords_per_row + wave_size - 1) // wave_size
    payload_slice_row_dwords = payload_dwords_per_row // num_k_slices
    output_k256_dwords = _M_TILE_ROWS * _PAYLOAD_K256_ROW_DWORDS
    output_k32_dwords = _M_TILE_ROWS * _PAYLOAD_VECTOR_DWORDS

    @fx.struct
    class ScatterLds:
        payload: fx.Array[
            fx.Int32, num_k_slices * _PAYLOAD_LDS_SLICE_DWORDS, _PAYLOAD_VECTOR_BYTES
        ]
        scale: fx.Array[
            fx.Int32, num_k_slices * _SCALE_LDS_SLICE_DWORDS, _PAYLOAD_VECTOR_BYTES
        ]

    module_name = (
        f"moe_apre_scatter_fd{feat_dim}_r{wmma_rep}_fp4_pk8"
        f"_w{_M_TILE_ROWS}_tk{feat_dim}"
    )
    if src_scale_stride_bytes != quant_layout.mx_blocks_per_row:
        module_name += f"_ss{src_scale_stride_bytes}"
    module_name += "_grid_sw8_async_ballot"

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
        lds_load_b128, _ = gemm_common.make_lds_copy_ops(128)

        thread_id = fx.Uint32(fx.thread_idx.x)
        block_id = fx.Uint32(fx.block_idx.x)
        wave_id = thread_id // fx.Uint32(wave_size)
        lane_id = thread_id - wave_id * fx.Uint32(wave_size)

        shared_allocator = fx.SharedAllocator()
        storage = shared_allocator.allocate(ScatterLds).peek()
        payload_lds = storage.payload.ptr
        payload_lds_base_bytes = fx.Index(fx.ptrtoint(payload_lds))
        scale_lds = storage.scale.ptr
        output_payload_resource = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(grouped_payload)),
            num_records_bytes=route_quant._SCALE_RSRC_MAX_BYTES,
        )
        output_scale_size_dwords = fx.Uint32(num_output_rows) * fx.Uint32(
            scale_dwords_per_row
        )
        output_scale_resource = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(grouped_scale)),
            num_records_bytes=output_scale_size_dwords * fx.Uint32(_DWORD_BYTES),
        )
        row_to_token_buffer = tensor_shim.ptr_buf_tensor(row_to_token)

        row_in_m16 = wave_id

        def load_payload_from_lds(slice_id):
            """Read a 16-byte payload vector in destination order."""
            output_slice_offset_dwords = thread_id * fx.Uint32(_PAYLOAD_VECTOR_DWORDS)
            output_k256_in_slice = output_slice_offset_dwords // fx.Uint32(
                output_k256_dwords
            )
            output_k256_offset_dwords = (
                output_slice_offset_dwords
                - output_k256_in_slice * fx.Uint32(output_k256_dwords)
            )
            output_k32_in_k256 = output_k256_offset_dwords // fx.Uint32(
                output_k32_dwords
            )
            output_k32_offset_dwords = (
                output_k256_offset_dwords
                - output_k32_in_k256 * fx.Uint32(output_k32_dwords)
            )
            output_row_in_m16 = output_k32_offset_dwords // fx.Uint32(
                _PAYLOAD_VECTOR_DWORDS
            )
            output_swizzled_k32 = (
                output_k32_in_k256
                + output_row_in_m16 // fx.Uint32(_ROWS_PER_SWIZZLE_GROUP)
            ) % fx.Uint32(_K32_PER_K256)
            read_payload_lds_offset_dwords = (
                output_k256_in_slice * fx.Uint32(_PAYLOAD_LDS_K256_DWORDS)
                + output_row_in_m16 * fx.Uint32(_PAYLOAD_LDS_ROW_DWORDS)
                + output_swizzled_k32 * fx.Uint32(_PAYLOAD_VECTOR_DWORDS)
            )
            output_payload_vec = lds_load_b128(
                payload_lds_base_bytes
                + fx.Index(slice_id * _PAYLOAD_LDS_SLICE_DWORDS * _DWORD_BYTES),
                fx.Index(read_payload_lds_offset_dwords * fx.Uint32(_DWORD_BYTES)),
            )

            return output_payload_vec

        def load_scale_from_lds(slice_id):
            """Read a scale dword in destination order."""
            output_scale_dword = fx.Int32(0)
            # Eight waves each write 64 B for a K1024 scale slice.
            output_scale_column = row_in_m16
            output_scale_row = lane_id
            is_scale_store_lane = (
                row_in_m16 < fx.Uint32(_SCALE_DWORDS_PER_SLICE_ROW)
            ) & (lane_id < fx.Uint32(_M_TILE_ROWS))
            read_scale_lds_offset_dwords = is_scale_store_lane.select(
                output_scale_row * fx.Uint32(_SCALE_LDS_ROW_DWORDS)
                + output_scale_column,
                fx.Uint32(0),
            )
            if row_in_m16 < fx.Uint32(_SCALE_DWORDS_PER_SLICE_ROW):
                output_scale_dword = scale_lds[
                    fx.Uint32(slice_id * _SCALE_LDS_SLICE_DWORDS)
                    + read_scale_lds_offset_dwords
                ]

            return output_scale_dword

        def store_scale_to_gmem(k256_offset, output_scale_dword, m16_tile_id):
            """Store scales contiguously across the M16 output rows."""
            output_scale_column = row_in_m16
            output_scale_row = lane_id
            is_scale_store_lane = (
                row_in_m16 < fx.Uint32(_SCALE_DWORDS_PER_SLICE_ROW)
            ) & (lane_id < fx.Uint32(_M_TILE_ROWS))
            scale_k256_in_slice = output_scale_column // fx.Uint32(
                _SCALE_DWORDS_PER_K256
            )
            scale_k128_in_k256 = output_scale_column - scale_k256_in_slice * fx.Uint32(
                _SCALE_DWORDS_PER_K256
            )
            output_scale_k256 = k256_offset + scale_k256_in_slice
            scale_row_group = m16_tile_id // fx.Uint32(wmma_rep)
            m16_in_scale_row_group = m16_tile_id - scale_row_group * fx.Uint32(wmma_rep)
            scale_group_offset_dwords = scale_row_group * fx.Uint32(
                scale_dwords_per_row * wmma_rep * _M_TILE_ROWS
            ) + m16_in_scale_row_group * fx.Uint32(_M_TILE_ROWS)
            output_scale_offset_dwords = is_scale_store_lane.select(
                scale_group_offset_dwords
                + output_scale_k256
                * fx.Uint32(_SCALE_DWORDS_PER_K256 * wmma_rep * _M_TILE_ROWS)
                + scale_k128_in_k256 * fx.Uint32(wmma_rep * _M_TILE_ROWS)
                + output_scale_row,
                output_scale_size_dwords,
            )
            if row_in_m16 < fx.Uint32(_SCALE_DWORDS_PER_SLICE_ROW):
                buffer_ops.buffer_store(
                    output_scale_dword,
                    output_scale_resource,
                    output_scale_offset_dwords * fx.Uint32(_DWORD_BYTES),
                    offset_is_bytes=True,
                )

        def store_payload_to_gmem(k256_offset, output_payload_vec, m16_tile_id):
            """Store a 16-byte vector at its contiguous A-preshuffled address."""
            buffer_ops.buffer_store(
                output_payload_vec,
                output_payload_resource,
                m16_tile_id * fx.Uint32(_M_TILE_ROWS * payload_dwords_per_row)
                + k256_offset * fx.Uint32(output_k256_dwords)
                + thread_id * fx.Uint32(_PAYLOAD_VECTOR_DWORDS),
            )

        def copy_payload_from_gmem_to_lds(slice_id, source_row):
            """Direct asynchronous b128 copy; no payload VGPR staging."""
            k256_in_slice = lane_id // fx.Uint32(_K32_PER_K256)
            k32_in_k256 = lane_id % fx.Uint32(_K32_PER_K256)
            swizzled_k32 = (
                k32_in_k256 + row_in_m16 // fx.Uint32(_ROWS_PER_SWIZZLE_GROUP)
            ) % fx.Uint32(_K32_PER_K256)
            dst_dword = (
                fx.Uint32(slice_id * _PAYLOAD_LDS_SLICE_DWORDS)
                + k256_in_slice * fx.Uint32(_PAYLOAD_LDS_K256_DWORDS)
                + row_in_m16 * fx.Uint32(_PAYLOAD_LDS_ROW_DWORDS)
                + swizzled_k32 * fx.Uint32(_PAYLOAD_VECTOR_DWORDS)
            )
            src_dword = (
                source_row * fx.Uint32(payload_dwords_per_row)
                + fx.Uint32(slice_id * payload_slice_row_dwords)
                + lane_id * fx.Uint32(_PAYLOAD_VECTOR_DWORDS)
            )
            gptr = create_llvm_ptr(
                fx.Int64(ptrtoint(compact_payload))
                + fx.Int64(src_dword) * _DWORD_BYTES,
                address_space=1,
            )
            lptr = create_llvm_ptr(
                fx.Int32(payload_lds_base_bytes) + fx.Int32(dst_dword) * _DWORD_BYTES,
                address_space=3,
            )
            rocdl.global_load_async_to_lds_b128(gptr, lptr, 0)

        def copy_scale_from_gmem_to_lds(source_row, dst_scale_lds):
            """Copy the row scales in wave-sized rounds, masking the final tail."""
            for scale_round in range_constexpr(scale_load_rounds):
                column = fx.Uint32(scale_round * wave_size) + lane_id
                if column < fx.Uint32(scale_dwords_per_row):
                    dst_dword = (
                        (column // fx.Uint32(_SCALE_DWORDS_PER_SLICE_ROW))
                        * fx.Uint32(_SCALE_LDS_SLICE_DWORDS)
                        + row_in_m16 * fx.Uint32(_SCALE_LDS_ROW_DWORDS)
                        + column % fx.Uint32(_SCALE_DWORDS_PER_SLICE_ROW)
                    )
                    src_dword = (
                        source_row * fx.Uint32(source_scale_stride_dwords) + column
                    )
                    gptr = create_llvm_ptr(
                        fx.Int64(ptrtoint(compact_scale))
                        + fx.Int64(src_dword) * _DWORD_BYTES,
                        address_space=1,
                    )
                    lptr = create_llvm_ptr(
                        fx.Int32(ptrtoint(dst_scale_lds))
                        + fx.Int32(dst_dword) * _DWORD_BYTES,
                        address_space=3,
                    )
                    rocdl.global_load_async_to_lds_b32(gptr, lptr, 0)

        # Each TG owns one M16 x K tile. LDS is written once and never reused.
        m16_tile_id = block_id
        # Every wave reads the same M16 map. The second lane half repeats the
        # first half's addresses, so all 32 lanes can load without an OOB mask.
        map_row = lane_id % fx.Uint32(_M_TILE_ROWS)
        token_on_lane = fx.Int32(
            row_to_token_buffer[m16_tile_id * fx.Uint32(_M_TILE_ROWS) + map_row]
        )
        valid_rows = fx.Uint32(rocdl.ballot(T.i32, token_on_lane >= fx.Int32(0)))
        tile_has_valid_rows = valid_rows != fx.Uint32(0)
        # Wave w owns row w; broadcast that lane's mapping within the wave.
        mapped_token = fx.Int32(rocdl.readlane(T.i32, token_on_lane, wave_id))

        # Every wave observes the same flag before entering the payload barrier.
        if tile_has_valid_rows:
            # Padding rows are not consumed by the MoE output. Redirect them
            # to token 0 so every wave can issue valid, unconditional loads.
            # A nonempty M16 tile implies that at least one source token exists.
            source_row = fx.Uint32(fx.max(mapped_token, fx.Int32(0)))

            copy_scale_from_gmem_to_lds(source_row, scale_lds)
            for slice_id in range_constexpr(num_k_slices):
                copy_payload_from_gmem_to_lds(slice_id, source_row)
            rocdl.s_wait_asynccnt(0)
            gpu.barrier()

            for slice_id in range_constexpr(num_k_slices):
                k256_offset = fx.Uint32(slice_id * _K256_PER_SLICE)
                output_scale_dword = load_scale_from_lds(slice_id)
                output_payload_vec = load_payload_from_lds(slice_id)
                store_scale_to_gmem(k256_offset, output_scale_dword, m16_tile_id)
                store_payload_to_gmem(k256_offset, output_payload_vec, m16_tile_id)

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
            grid=(
                arith.index_cast(T.index, num_output_rows) // _M_TILE_ROWS,
                1,
                1,
            ),
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
