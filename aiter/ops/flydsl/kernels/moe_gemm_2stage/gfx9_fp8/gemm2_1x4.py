# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE stage2 1x4 down-projection kernel for N256 tiles."""

from functools import partial

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from . import common as fxh
from .common import (
    FlyObjCache,
    _f32_to_bf16,
    get_down_device_config,
    validate_gemm_options,
)

# gfx942 raw-buffer aux bit 1 selects the non-temporal policy.
_DOWN_STORE_CACHE_MODIFIER = 2

# ==================== Task setup helpers ====================


def _flatten_rows(x):
    # The second mode is innermost, so swap modes before flattening for the tiled copy.
    return fx.group(fx.select(x, [1, 0]), 0, -1)


def _gather_sorted_rows(ops, lds_rows, input_rows, sorted_ids, M, TOPK, K):
    """Gather the task's sorted activation rows into LDS; returns the input buffer tensor."""
    input_rows = fx.rocdl.make_buffer_tensor(
        input_rows,
        max_size=False,
        num_records_bytes=fx.Int64(M) * (TOPK * K) * (input_rows.dtype.width // 8),
    )
    copy_atom = ops.get_buffer_copy_atom(input_rows.dtype, 128)
    for dst, row, col in fxh.all_copy_atoms(
        _flatten_rows(lds_rows),
        _flatten_rows(fxh.make_1d_coord_tensor(lds_rows, 0, fx.get_iter(sorted_ids))),
        _flatten_rows(fxh.make_1d_coord_tensor(lds_rows, 1, fx.make_int_tuple(0))),
        atom_bits=128,
        num_threads=256,
    ):
        sorted_id = row[0].bitcast(fx.Uint32)
        source = fxh.atom_tensor(
            input_rows, (sorted_id & 0xFFFFFF, sorted_id >> 24, col[0]), 128
        )
        fx.copy(copy_atom, source, dst)
    fx.gpu.barrier()
    return input_rows


def _act_scale_tensor(p_a_scale, act_quant_type, M, TOPK):
    """[M, TOPK] activation scales as a buffer tensor; per-tensor broadcasts one value."""
    per_tensor = act_quant_type == "per_tensor"
    scale = fx.make_view(
        fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
        fx.make_layout((M, TOPK), (0, 0) if per_tensor else (TOPK, 1)),
    )
    return fx.rocdl.make_buffer_tensor(
        scale,
        max_size=False,
        num_records_bytes=(fx.Int64(1) if per_tensor else fx.Int64(M) * TOPK)
        * (scale.dtype.width // 8),
    )


def _apply_token_scales(
    ops, mm, a_scale, sorted_ids, routing, block_n, block_m, weight_scale=None
):
    """Gather per-token activation scales in the C layout and fold in the routing weights."""
    copy_atom = ops.get_buffer_copy_atom(a_scale.dtype, 32)
    coord_tensor = fx.make_view(
        fx.get_iter(sorted_ids), fx.make_layout((block_n, block_m), (0, 1))
    )
    coords = ops.load_tiled_mma_fragC(mm, coord_tensor, copy_atom_bits=32)
    scales = mm.make_fragment_C(coord_tensor)
    for dst, coord in fxh.all_elements(
        ops.get_tiled_mma_retile(mm, scales, "C", copy_atom=copy_atom), coords
    ):
        sorted_id = coord[0].bitcast(fx.Uint32)
        source = fxh.atom_tensor(a_scale, (sorted_id & 0xFFFFFF, sorted_id >> 24), 32)
        fx.copy(copy_atom, source, dst)
    for scale, weight in fxh.all_elements(scales, routing):
        combined = scale.load() * weight.load()
        if const_expr(weight_scale is not None):
            combined = combined * weight_scale
        scale.store(combined)
    return scales


# ==================== N256 epilogue and scheduling helpers ====================


def _set_priority(value):
    fx.rocdl.sched_barrier(0)
    fx.rocdl.s_setprio(value)
    fx.rocdl.sched_barrier(0)


def _pack_scaled_bf16_pairs(values, scales):
    # Round before packing; a floating-point FMA bias of 0x8000 cannot replace integer rounding.
    packed = _f32_to_bf16(values * scales).bitcast(fx.Uint32)
    return [packed[index] for index in range_constexpr(packed.numel)]


def _store_scaled_bf16(source, scales, destination):
    for src, scale, dst in fxh.all_elements(source, scales, destination):
        dst.store(
            Vec.from_elements(
                _pack_scaled_bf16_pairs(src.load(), scale.load()), fx.Uint32
            ).bitcast(fx.BFloat16)
        )


def _write_cshuffle_vector(
    lds,
    copy_atom,
    wave_lds_base,
    lane_group,
    row_in_8,
    row_half,
    channel_piece,
    packed_bf16,
):
    """Write one packed BF16 vector of an M16 row pair to the wave's swizzled CShuffle LDS."""
    physical_atom = (lane_group * 2 + channel_piece) ^ row_in_8
    lds_offset = wave_lds_base + ((row_half * 8 + row_in_8) * 8 + physical_atom) * 8
    destination = fx.make_view(fx.get_iter(lds) + lds_offset, fx.make_layout(8, 1))
    fragment = fx.make_fragment_like(destination)
    fragment.store(packed_bf16)
    fx.copy(copy_atom, fragment, destination)


def _store_cshuffle_row_pair(
    lds,
    copy_atom,
    store_rsrc,
    store_cache,
    wave_lds_base,
    lane_id,
    wave_id,
    row_pair,
    block_n,
    *,
    block_n_size,
    wave_n,
    row_stride,
):
    """Read one M16 row pair back from CShuffle LDS and store its two rows to global memory."""
    fragments = []
    byte_offsets = []
    fx.rocdl.sched_barrier(0)
    for row_half in range_constexpr(2):
        output_row = (row_pair * 2 + row_half) * 8 + lane_id // 8
        output_atom = lane_id % 8
        physical_atom = output_atom ^ (lane_id // 8)
        lds_offset = (
            wave_lds_base + ((row_half * 8 + lane_id // 8) * 8 + physical_atom) * 8
        )
        source = fx.make_view(fx.get_iter(lds) + lds_offset, fx.make_layout(8, 1))
        fragment = fx.make_fragment_like(source)
        fx.copy(copy_atom, source, fragment)
        fragments.append(fragment)
        output_column = (
            fx.Int64(block_n) * block_n_size
            + fx.Int64(wave_id) * wave_n
            + fx.Int64(output_atom) * 8
        )
        byte_offsets.append(
            ((fx.Int64(output_row) * row_stride + output_column) * 2).to(fx.Int32)
        )

    # Consume the older read first without forcing the newer read complete.
    for index, lgkmcnt in ((0, 1), (1, 0)):
        fx.rocdl.s_waitcnt(lgkmcnt=lgkmcnt)
        fx.rocdl.RawPtrBufferStoreOp(
            Vec(fragments[index].load()).bitcast(fx.Int32).ir_value(),
            store_rsrc,
            byte_offsets[index].ir_value(),
            fx.Int32(0).ir_value(),
            aux=ir.IntegerAttr.get(fx.Int32.ir_type, store_cache),
        )


def _build_moe_gemm2_1x4_n256(
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage="down",
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    down_path="default",
    down_output_padding_bytes=None,
    _task_table=False,
    _store_cache=_DOWN_STORE_CACHE_MODIFIER,
):
    assert stage == "down" and alg == "prefill_1x4" and down_path == "1x4_64x256"
    assert weight_dtype == "fp8"
    # Activation scales default to the weight form; shared checks cover the supported combos.
    act_quant_type, _ = validate_gemm_options(
        weight_dtype,
        weight_quant_type,
        act_quant_type,
        BLOCK_TILE_SIZE_M,
        alg,
        activation,
        swiglu_limit,
    )
    assert BLOCK_TILE_SIZE_M == 64 and BLOCK_TILE_SIZE_N == 256
    assert N % BLOCK_TILE_SIZE_N == 0 and K % 64 == 0
    assert down_output_padding_bytes in (0, 32, 64, 128)
    _, generic_xcd_count = get_down_device_config()
    block_m_per_4wave_group = 64
    activation_bytes = block_m_per_4wave_group * K
    scale_bytes = BLOCK_TILE_SIZE_N * 4 if weight_quant_type == "ptpc" else 0
    cshuffle_bytes = 4 * 16 * 64 * (fx.BFloat16.width // 8)
    assert activation_bytes + scale_bytes + cshuffle_bytes <= 64 * 1024, (
        "1x4_64x256 exceeds gfx942 LDS capacity; "
        f"activation={activation_bytes}B, scale={scale_bytes}B, "
        f"cshuffle={cshuffle_bytes}B"
    )
    output_row_stride = N + down_output_padding_bytes // (fx.BFloat16.width // 8)

    weight_dtype = fx.Float8E4M3FNUZ
    down_ops = FlyObjCache(cache_ir=False)

    @flyc.jit
    def _map_down_task(valid_rows: fx.Int32, task_rows: fx.Constexpr[int]):
        """Interleave down tasks across XCDs."""
        workgroup_idx = fx.Int32(fx.gpu.block_idx.y)
        valid_rows_u32 = fx.Uint32(valid_rows)
        valid_tasks = valid_rows_u32 // task_rows
        valid_tasks += fx.Uint32(valid_rows_u32 % task_rows != 0)

        swizzle_chunk = valid_tasks // generic_xcd_count
        swizzle_limit = swizzle_chunk * generic_xcd_count
        swizzled_e_idx = (
            workgroup_idx % generic_xcd_count
        ) * swizzle_chunk + workgroup_idx // generic_xcd_count
        return fx.Int32(
            arith.select(workgroup_idx < swizzle_limit, swizzled_e_idx, workgroup_idx)
        )

    @flyc.kernel(known_block_size=[256, 1, 1])
    def moe_2stage_down_prefill_1x4_64x256(
        p_input: fx.Pointer,  # fp8 [M, TOPK, K]            K = HIDDEN_STATES//TP
        p_weight: fx.Pointer,  # quantized/bf16 [E, N, K]   N = HIDDEN_STATES
        p_output: fx.Pointer,  # bf16 [M, TOPK, N]
        p_sorted_ids: fx.Pointer,  # int32 [num_tokens_sorted]
        p_sorted_weights: fx.Pointer,  # f32 [num_tokens_sorted]
        p_sorted_expert_ids: fx.Pointer,  # int32 [num_blocks] num_tokens_sorted <= num_blocks * BLOCK_TILE_SIZE_M
        p_num_valid_ids: fx.Pointer,  # int32 [2]  value: (sorting valid rows incl. expert padding, M)
        p_w_scale: fx.Pointer,  # weight fp8 scale (per-output-channel ptpc / per-tensor)
        p_a_scale: fx.Pointer,  # input fp8 scale (per-token ptpc / per-tensor)
        M: fx.Int32,
    ):
        """M64xN256: each workgroup uses four waves to compute one M64 task."""
        max_valid_id = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]
        e_idx = _map_down_task(max_valid_id, block_m_per_4wave_group)
        e_offset = fx.Int64(e_idx)
        if e_idx * block_m_per_4wave_group < max_valid_id:
            if const_expr(_task_table):
                row_begin = fx.Int64(p_sorted_expert_ids[2 * e_idx])
            # 1. Create input, output, sorted-metadata, and weight views for this expert task.
            arg_p_input = fxh.view_as_torch_tensor(p_input, (M, TOPK, K), weight_dtype)
            arg_p_output = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_output, fx.BFloat16)
                + (
                    row_begin * output_row_stride
                    if const_expr(_task_table)
                    else e_offset * (block_m_per_4wave_group * output_row_stride)
                ),
                (block_m_per_4wave_group, output_row_stride),
            )
            output_store_rsrc = fx.rocdl.get_buffer_rsrc(
                fx.rocdl.make_buffer_ptr(
                    fx.get_iter(arg_p_output),
                    num_records_bytes=block_m_per_4wave_group * output_row_stride * 2,
                )
            )

            def sorted_rows(pointer, dtype):
                return fxh.view_as_torch_tensor(
                    fxh._as_ptr(pointer)
                    + (
                        row_begin
                        if const_expr(_task_table)
                        else e_offset * block_m_per_4wave_group
                    ),
                    (block_m_per_4wave_group,),
                    dtype,
                )

            arg_p_sorted_ids = sorted_rows(p_sorted_ids, fx.Int32)
            arg_p_sorted_weights = sorted_rows(p_sorted_weights, fx.Float32)
            expert_id = (
                p_sorted_expert_ids[2 * e_idx + 1]
                if const_expr(_task_table)
                else fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[
                    e_idx
                ]
            )

            # 16bytes/DW4
            element_num = 16 // (weight_dtype.width // 8)
            arg_p_weight = fx.make_view(
                fxh._as_ptr(p_weight, weight_dtype) + fx.Int64(expert_id) * N * K,
                fx.make_layout(
                    (((4, 2, 2, 4, 4, N // 256)), (element_num, K // element_num)),
                    (
                        (element_num, 16 * K, 32 * K, 64 * K, 4 * element_num, 256 * K),
                        (1, 16 * element_num),
                    ),
                ),
            )
            arg_p_weight = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)
            BLOCK_M = block_m_per_4wave_group
            BLOCK_N = 256
            BLOCK_K = 128 if K % 128 == 0 else 64 // (weight_dtype.width // 8)
            WAVE_N = BLOCK_N // 4

            # mask,base,shift, swizzle always in unit of 128b,
            swz_base = ((128 // weight_dtype.width) - 1).bit_length()
            a_swz = fx.SwizzleType.get(3, swz_base, 4)

            shared_allocator = fx.SharedAllocator()
            activation_storage = shared_allocator.allocate(
                fx.Array[weight_dtype, BLOCK_M * K]
            )
            scale_lds = None
            if const_expr(weight_quant_type == "ptpc"):
                scale_storage = shared_allocator.allocate(
                    fx.Array[fx.Float32, BLOCK_N, 16]
                )
                scale_lds = scale_storage.peek().view(fx.make_layout(BLOCK_N, 1))
            cshuffle_storage = shared_allocator.allocate(
                fx.Array[fx.BFloat16, 4 * 16 * 64, 16]
            )
            cshuffle_lds = cshuffle_storage.peek().view(fx.make_layout(4 * 16 * 64, 1))
            ldsA0 = activation_storage.peek().view(
                fx.make_composed_layout(fx.static(a_swz), fxh.torch_layout(BLOCK_M, K))
            )
            # 2. Gather activations into LDS; the barrier starts the four-wave pipeline.
            arg_p_input = _gather_sorted_rows(
                down_ops, ldsA0, arg_p_input, arg_p_sorted_ids, M, TOPK, K
            )

            # (BLOCK_N, BLOCK_K, num_blocks_N, num_blocks_K)
            weight = fx.flat_divide(arg_p_weight, (BLOCK_N, BLOCK_K))
            ldsA = fx.flat_divide(ldsA0, (BLOCK_M, BLOCK_K))
            nBN = fxh.div_up(N, BLOCK_N)
            nBK = fxh.div_up(K, BLOCK_K)
            mm = down_ops.create_thr_mma(weight_dtype, (4, 1, 1))

            c_fake_tensor = fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_ordered_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            fragC = mm.make_fragment_C(c_fake_tensor)
            fragC_bf16 = fx.make_fragment_like(fragC, fx.BFloat16)
            frag_act = mm.make_fragment_B(ldsA[None, None, 0, 0])
            # 3. Prepare weight, activation, routing scales, and output CShuffle resources.
            per_tensor_w_scale = None
            scale_global_rsrc = None
            scale_lds_logical = None
            if const_expr(weight_quant_type == "per_tensor"):
                per_tensor_w_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id, fx.make_layout(1, 1)
                )[0]
            if const_expr(weight_quant_type == "ptpc"):
                scale_global = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id * N, fx.make_layout(N, 1)
                )
                scale_global_rsrc = fx.rocdl.get_buffer_rsrc(
                    fx.rocdl.make_buffer_ptr(
                        fx.get_iter(scale_global),
                        num_records_bytes=N * (fx.Float32.width // 8),
                    )
                )
                scale_lds_logical = fx.make_view(
                    fx.get_iter(scale_lds),
                    fx.make_layout(((4, 2, 2, 4, 4), 1), ((1, 16, 32, 64, 4), 0)),
                )
            scale_lds_copy_atom = down_ops.get_universal_copy_atom(fx.Float32, 128)

            def issue_scale_block_global(block_n):
                lane_id = fx.Int32(fx.thread_idx.x % 64)
                wave_id = fx.Int32(fx.thread_idx.x // 64)
                scale_local_offset = wave_id * WAVE_N + lane_id * 4
                scale_offset = fx.Int32(block_n) * BLOCK_N + scale_local_offset
                # The original buffer_load used element offsets; raw ROCDL uses bytes.
                # Invalid lanes still use out-of-bounds zero filling.
                scale_byte_offset = (lane_id < WAVE_N // 4).select(
                    scale_offset * (fx.Float32.width // 8), fx.Int32(0x7FFFFFFF)
                )
                scale_vec = Vec(
                    fx.rocdl.RawPtrBufferLoadOp(
                        ir.VectorType.get([4], fx.Float32.ir_type),
                        scale_global_rsrc,
                        scale_byte_offset.ir_value(),
                        fx.Int32(0).ir_value(),
                        aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0),
                    ).result
                )
                return scale_vec

            def commit_scale_block_lds(scale_vec, dst=None):
                lane_id = fx.Int32(fx.thread_idx.x % 64)
                wave_id = fx.Int32(fx.thread_idx.x // 64)
                if lane_id < WAVE_N // 4:
                    scale_local_offset = wave_id * WAVE_N + lane_id * 4
                    scale_dst = fx.make_view(
                        fx.get_iter(scale_lds) + scale_local_offset,
                        fx.make_layout(4, 1),
                    )
                    scale_frag = fx.make_fragment_like(scale_dst)
                    scale_frag.store(scale_vec)
                    fx.copy(scale_lds_copy_atom, scale_frag, scale_dst)
                return down_ops.load_tiled_mma_fragC(
                    mm, scale_lds_logical, dst=dst, copy_atom_bits=128
                )

            arg_a_scale = None
            per_tensor_a_scale = None
            if const_expr(
                act_quant_type == "per_tensor" and weight_quant_type == "per_tensor"
            ):
                per_tensor_a_scale = fx.make_view(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                    fx.make_layout(1, 1),
                )[0]
            else:
                arg_a_scale = _act_scale_tensor(p_a_scale, act_quant_type, M, TOPK)

            sorted_weights = fx.make_view(
                fx.get_iter(arg_p_sorted_weights),
                fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            # load rows/token weights using C layout
            frag_sorted_weight = down_ops.load_tiled_mma_fragC(
                mm, sorted_weights, copy_atom_bits=32
            )

            if const_expr(per_tensor_a_scale is not None):
                combined_scale = per_tensor_a_scale * per_tensor_w_scale
                frag_sorted_weight.store(frag_sorted_weight.load() * combined_scale)
            else:
                frag_sorted_weight = _apply_token_scales(
                    down_ops,
                    mm,
                    arg_a_scale,
                    arg_p_sorted_ids,
                    frag_sorted_weight,
                    BLOCK_N,
                    BLOCK_M,
                    per_tensor_w_scale,
                )
            cshuffle_copy_atom = down_ops.get_universal_copy_atom(fx.BFloat16, 128)
            write_cshuffle_vector = partial(
                _write_cshuffle_vector, cshuffle_lds, cshuffle_copy_atom
            )
            store_cshuffle_row_pair = partial(
                _store_cshuffle_row_pair,
                cshuffle_lds,
                cshuffle_copy_atom,
                output_store_rsrc,
                _store_cache,
                block_n_size=BLOCK_N,
                wave_n=WAVE_N,
                row_stride=output_row_stride,
            )

            def store_cshuffle_n256(output, block_n):
                lane_id = fx.Int32(fx.thread_idx.x % 64)
                wave_id = fx.Int32(fx.thread_idx.x // 64)
                lane_group = lane_id // 16
                lane_row = lane_id % 16
                wave_lds_base = wave_id * (16 * 64)

                # All 64 lanes produce one M16 pair, halving the DS writes used
                # by the previous two masked M8 producers.
                for row_pair in range_constexpr(4):
                    row_in_8 = lane_row % 8
                    row_half = lane_row // 8
                    for channel_piece in range_constexpr(2):
                        channels_lo = Vec(
                            output[None, 2 * channel_piece, row_pair].load()
                        )
                        channels_hi = Vec(
                            output[None, 2 * channel_piece + 1, row_pair].load()
                        )
                        write_cshuffle_vector(
                            wave_lds_base,
                            lane_group,
                            row_in_8,
                            row_half,
                            channel_piece,
                            channels_lo.shuffle(channels_hi, list(range(8))),
                        )
                    store_cshuffle_row_pair(
                        wave_lds_base, lane_id, wave_id, row_pair, block_n
                    )

            def postprocess_store_vector_4wave(
                output, sorted_weight, block_n, vector_begin=0, vector_count=8
            ):
                lane_id = fx.Int32(fx.thread_idx.x % 64)
                wave_id = fx.Int32(fx.thread_idx.x // 64)
                lane_group = lane_id // 16
                lane_row = lane_id % 16
                wave_lds_base = wave_id * (16 * 64)

                for vector_index in range_constexpr(
                    vector_begin, vector_begin + vector_count
                ):
                    row_pair = vector_index // 2
                    channel_piece = vector_index % 2
                    row_in_8 = lane_row % 8
                    row_half = lane_row // 8
                    channels_lo = Vec(output[None, 2 * channel_piece, row_pair].load())
                    channels_hi = Vec(
                        output[None, 2 * channel_piece + 1, row_pair].load()
                    )
                    scales_lo = Vec(
                        sorted_weight[None, 2 * channel_piece, row_pair].load()
                    )
                    scales_hi = Vec(
                        sorted_weight[None, 2 * channel_piece + 1, row_pair].load()
                    )
                    packed_bf16 = Vec.from_elements(
                        _pack_scaled_bf16_pairs(channels_lo, scales_lo)
                        + _pack_scaled_bf16_pairs(channels_hi, scales_hi),
                        fx.Uint32,
                    ).bitcast(fx.BFloat16)
                    write_cshuffle_vector(
                        wave_lds_base,
                        lane_group,
                        row_in_8,
                        row_half,
                        channel_piece,
                        packed_bf16,
                    )
                    if const_expr(channel_piece == 1):
                        store_cshuffle_row_pair(
                            wave_lds_base, lane_id, wave_id, row_pair, block_n
                        )

            use_delayed_4wave_store = BLOCK_K == 128 and nBK == 2

            # Prologue: stage0 prepares N block 0 / K core 0.
            _set_priority(0)
            frag_weight = down_ops.load_tiled_mma_fragA(
                mm, weight, [None, None, fx.Int32(0), 0]
            )
            frag_pc_scale = None
            next_frag_pc_scale = None
            if const_expr(weight_quant_type == "ptpc"):
                frag_pc_scale = commit_scale_block_lds(issue_scale_block_global(0))
                next_frag_pc_scale = fx.make_fragment_like(frag_pc_scale)
            frag_weight_slots = [frag_weight, fx.make_fragment_like(frag_weight)]

            def overlap_previous_output_store(block_n, k_core, previous_fragC):
                # Overlap the previous tile's eight store vectors with the two K cores (3 + 5).
                if const_expr(use_delayed_4wave_store):  # noqa: SIM102
                    if block_n > 0:
                        postprocess_store_vector_4wave(
                            previous_fragC,
                            frag_sorted_weight,
                            block_n - 1,
                            *((0, 3), (3, 5))[k_core],
                        )

            def run_k_core_pipeline(
                block_n, k_core, previous_fragC, next_frag_pc_scale
            ):
                current_slot = k_core % 2
                next_slot = (k_core + 1) % 2
                next_scale_vec = None
                overlap_previous_output_store(block_n, k_core, previous_fragC)
                down_ops.load_tiled_mma_fragB(
                    mm, ldsA, [None, None, 0, k_core], frag_act
                )
                # Prefetch the next K core, or K core 0 (and PTPC scales) of the next N tile.
                next_block_n, next_k_core = block_n, k_core + 1
                if const_expr(k_core + 1 == nBK):
                    next_block_n, next_k_core = block_n + 1, 0
                    if const_expr(weight_quant_type == "ptpc"):
                        next_scale_vec = issue_scale_block_global(next_block_n)
                        fx.rocdl.sched_barrier(0)
                down_ops.load_tiled_mma_fragA(
                    mm,
                    weight,
                    [None, None, next_block_n, next_k_core],
                    frag_weight_slots[next_slot],
                )
                _set_priority(3)
                fx.gemm(mm, fragC, frag_weight_slots[current_slot], frag_act, fragC)
                _set_priority(0)
                if const_expr(k_core + 1 == nBK and weight_quant_type == "ptpc"):
                    commit_scale_block_lds(next_scale_vec, next_frag_pc_scale)

            previous_fragC = None
            if const_expr(use_delayed_4wave_store):
                previous_fragC = fx.make_fragment_like(fragC)
                previous_fragC.fill(0)

            # Carry FP8 packets as bits across SCF boundaries. The pinned
            # compiler lowers FP8 memory to integers before lowering scf.for.
            # A delayed output tile, when present, is always the last state.
            loop_state = [frag_weight.load().bitcast(fx.Uint32)]
            if const_expr(weight_quant_type == "ptpc"):
                loop_state.append(frag_pc_scale.load())
            if const_expr(use_delayed_4wave_store):
                loop_state.append(previous_fragC.load())
            for block_n, state in range(
                fx.Int64(0), fx.Int64(nBN), fx.Int64(1), init=loop_state
            ):
                # Restore the next weight/scale and any delayed output tile.
                frag_weight_slots[0].store(state[0].bitcast(weight_dtype))
                if const_expr(weight_quant_type == "ptpc"):
                    frag_pc_scale.store(state[1])
                if const_expr(use_delayed_4wave_store):
                    previous_fragC.store(state[-1])

                # Compute this N tile while retiring the previous tile's stores.
                fragC.fill(0)
                for k_core in range_constexpr(nBK):
                    run_k_core_pipeline(
                        block_n, k_core, previous_fragC, next_frag_pc_scale
                    )

                # Apply channel scale, routing weight and BF16 conversion.
                if const_expr(weight_quant_type == "ptpc"):
                    for fc, fpc in fxh.all_elements(fragC, frag_pc_scale):
                        fc.store(fc.load() * fpc.load())
                if not const_expr(use_delayed_4wave_store):
                    _store_scaled_bf16(fragC, frag_sorted_weight, fragC_bf16)
                    store_cshuffle_n256(fragC_bf16, block_n)

                next_state = [frag_weight_slots[nBK % 2].load().bitcast(fx.Uint32)]
                if const_expr(weight_quant_type == "ptpc"):
                    next_state.append(next_frag_pc_scale.load())
                if const_expr(use_delayed_4wave_store):
                    next_state.append(fragC.load())
                results = yield next_state

            # Drain the final delayed tile after the N loop.
            if const_expr(use_delayed_4wave_store):
                previous_fragC.store(results[-1])
                postprocess_store_vector_4wave(
                    previous_fragC, frag_sorted_weight, fx.Int64(nBN - 1)
                )
            _set_priority(0)

    @flyc.jit
    def launch_prefill_1x4(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        kernel = moe_2stage_down_prefill_1x4_64x256(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            M,
            value_attrs={"passthrough": [["target-features", "-packed-fp32-ops"]]},
        )
        kernel.launch(grid=(1, task_num, 1), block=(256, 1, 1), stream=stream)

    launch_prefill_1x4.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch_prefill_1x4
