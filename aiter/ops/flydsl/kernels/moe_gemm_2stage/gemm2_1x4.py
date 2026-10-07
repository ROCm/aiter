# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE stage2 1x4 down-projection kernels for N64 and N256 tiles."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from . import common as fxh
from .common import FlyObjCache, _f32_to_bf16, get_down_device_config

# gfx942 raw-buffer aux bit 1 selects the non-temporal policy.
_DOWN_STORE_CACHE_MODIFIER = 2


def _build_moe_gemm2_1x4_n64(
    *,
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    act_quant_type,
    E=None,
):
    assert K % 64 == 0, f"down prefill requires K to be divisible by 64, got K={K}"
    assert N % 256 == 0, (
        f"down prefill requires N to be divisible by 256 for paired "
        f"128-wide tiles, got N={N}"
    )
    if weight_dtype == "bf16":
        weight_dtype = fx.BFloat16
    elif weight_dtype == "fp8":
        weight_dtype = fx.Float8E4M3FNUZ

    flyobj = fxh.FlyObjCache()

    @flyc.kernel
    def moe_2stage_down_prefill_1x4_n64(
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
    ):
        e_idx = fx.gpu.block_idx.y

        flyobj.bid = e_idx

        max_valid_id = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]

        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            arg_p_input = fxh.view_as_torch_tensor(p_input, (M, TOPK, K), weight_dtype)
            arg_p_output = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_output, fx.BFloat16)
                + fx.Int64(e_idx) * (BLOCK_TILE_SIZE_M * N),
                (BLOCK_TILE_SIZE_M, N),
            )
            arg_p_sorted_ids = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M,
                (BLOCK_TILE_SIZE_M,),
                fx.Int32,
            )
            arg_p_sorted_weights = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_weights) + e_idx * BLOCK_TILE_SIZE_M,
                (BLOCK_TILE_SIZE_M,),
                fx.Float32,
            )
            expert_id = fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[
                e_idx
            ]

            element_num = 16 // (weight_dtype.width // 8)
            arg_p_weight = fx.make_view(
                fxh._as_ptr(p_weight, weight_dtype) + fx.Int64(expert_id * N * K),
                fx.make_layout(
                    ((16, N // 16), (element_num, K // element_num)),
                    ((element_num, 16 * K), (1, 16 * element_num)),
                ),
            )

            arg_p_weight = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)
            arg_p_output = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False)

            fx.rocdl.make_buffer_tensor(arg_p_sorted_ids, max_size=False)

            BLOCK_M = BLOCK_TILE_SIZE_M
            BLOCK_N = 64
            BLOCK_K = 64 // (weight_dtype.width // 8)

            swz_base = ((128 // weight_dtype.width) - 1).bit_length()
            swz = fx.SwizzleType.get(3, swz_base, 3)

            act_dtype = weight_dtype

            @fx.union
            class SharedStorage:
                A: fx.Array[act_dtype, BLOCK_M * K]
                C: fx.Array[fx.BFloat16, 2 * BLOCK_M * BLOCK_N]

            lds = fx.SharedAllocator().allocate(SharedStorage)
            ldsA0 = lds.A.peek().view(
                fx.make_composed_layout(fx.static(swz), fxh.torch_layout(BLOCK_M, K))
            )
            layoutC = fx.make_composed_layout(
                fx.static(swz),
                fx.make_ordered_layout((BLOCK_M, BLOCK_N, 2), (1, 0, 2)),
            )
            layoutCt = fx.make_composed_layout(
                fx.static(swz), fx.make_ordered_layout((BLOCK_N, BLOCK_M, 2), (0, 1, 2))
            )
            ldsC = lds.C.peek().view(layoutC)
            ldsCt = lds.C.peek().view(layoutCt)

            arg_p_input = fx.rocdl.make_buffer_tensor(
                arg_p_input,
                max_size=False,
                num_records_bytes=fx.Int64(M)
                * (TOPK * K)
                * (arg_p_input.dtype.width // 8),
            )
            cp_atom = flyobj.get_buffer_copy_atom(arg_p_input.dtype, 128)

            def flatten_A(x):
                x = fx.select(x, [1, 0])
                return fx.group(x, 0, -1)

            cp_ldsA0 = flatten_A(ldsA0)
            cp_rows = flatten_A(
                fxh.make_1d_coord_tensor(ldsA0, 0, fx.get_iter(arg_p_sorted_ids))
            )
            cp_cols = flatten_A(
                fxh.make_1d_coord_tensor(ldsA0, 1, fx.make_int_tuple(0))
            )
            for dst, row, col in fxh.all_copy_atoms(
                cp_ldsA0, cp_rows, cp_cols, atom_bits=128, num_threads=256
            ):
                sorted_id = row[0].bitcast(fx.Uint32)
                atom_A = fxh.atom_tensor(
                    arg_p_input, (sorted_id & 0xFFFFFF, sorted_id >> 24, col[0]), 128
                )
                fx.copy(cp_atom, atom_A, dst)
            fx.gpu.barrier()

            weight = fx.flat_divide(arg_p_weight, (BLOCK_N, BLOCK_K))
            ldsA = fx.flat_divide(ldsA0, (BLOCK_M, BLOCK_K))

            nBN = fxh.div_up(N, BLOCK_N)
            nBK = fxh.div_up(K, BLOCK_K)

            mm = flyobj.create_thr_mma(weight_dtype, (4, 1, 1))

            c_fake_tensor = fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_ordered_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            fragC = [
                mm.make_fragment_C(c_fake_tensor),
                mm.make_fragment_C(c_fake_tensor),
            ]
            fragC_bf16 = fx.make_fragment_like(fragC[0], fx.BFloat16)

            frag_act = flyobj.load_tiled_mma_fragB(mm, ldsA, copy_atom_bits=128)
            fx.gpu.barrier()  # Finish reading ldsA before ldsC reuses its storage.

            arg_w_scale = None
            if const_expr(weight_quant_type == "per_tensor"):
                arg_w_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id, fx.make_layout((N, 1), (0, 0))
                )
                arg_w_scale = fx.flat_divide(arg_w_scale, (BLOCK_N, 1))
            if const_expr(weight_quant_type == "ptpc"):
                arg_w_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id * N,
                    fx.make_layout((N, 1), (1, 0)),
                )
                arg_w_scale = fx.flat_divide(arg_w_scale, (BLOCK_N, 1))

            arg_a_scale = None
            if const_expr(act_quant_type == "per_tensor"):
                arg_a_scale = fx.make_view(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                    fx.make_layout((M, TOPK), (0, 0)),
                )
                arg_a_scale = fx.rocdl.make_buffer_tensor(
                    arg_a_scale,
                    max_size=False,
                    num_records_bytes=fx.Int64(1) * (arg_a_scale.dtype.width // 8),
                )
            if const_expr(act_quant_type == "ptpc"):
                arg_a_scale = fx.make_view(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                    fx.make_layout((M, TOPK), (TOPK, 1)),
                )
                arg_a_scale = fx.rocdl.make_buffer_tensor(
                    arg_a_scale,
                    max_size=False,
                    num_records_bytes=fx.Int64(M)
                    * TOPK
                    * (arg_a_scale.dtype.width // 8),
                )

            sorted_weights = fx.make_view(
                fx.get_iter(arg_p_sorted_weights),
                fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            frag_sorted_weight = flyobj.load_tiled_mma_fragC(
                mm, sorted_weights, copy_atom_bits=32
            )

            if fx.const_expr(arg_a_scale is not None):
                cp_atom = flyobj.get_buffer_copy_atom(p_a_scale.dtype, 32)
                coord_tensor = fx.make_view(
                    fx.get_iter(arg_p_sorted_ids),
                    fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
                )
                frag_coord = flyobj.load_tiled_mma_fragC(
                    mm, coord_tensor, copy_atom_bits=32
                )
                frag_pt_scales = mm.make_fragment_C(coord_tensor)
                frag_pt_scalesr = flyobj.get_tiled_mma_retile(
                    mm, frag_pt_scales, "C", copy_atom=cp_atom
                )

                for dst, coord in fxh.all_elements(frag_pt_scalesr, frag_coord):
                    sorted_id = coord[0].bitcast(fx.Uint32)
                    atom_A = fxh.atom_tensor(
                        arg_a_scale,
                        (sorted_id & 0xFFFFFF, sorted_id >> 24),
                        32,
                    )
                    fx.copy(cp_atom, atom_A, dst)

                for frag_pt, frag_sw in fxh.all_elements(
                    frag_pt_scales, frag_sorted_weight
                ):
                    frag_pt.store(frag_pt.load() * frag_sw.load())

                frag_sorted_weight = frag_pt_scales

            def gemm_compute(fragW, fragPCS, fragC):
                fragC.fill(0)
                for k in fx.range_constexpr(nBK):
                    fx.gemm(
                        mm,
                        fragC,
                        fragW[None, None, None, k],
                        frag_act[None, None, None, 0, k],
                        fragC,
                    )
                if fx.const_expr(fragPCS is not None):
                    for fc, fpc in fxh.all_elements(fragC, fragPCS):
                        fc.store(fc.load() * fpc.load())

            fx.make_view(
                fx.get_iter(arg_p_sorted_ids),
                fx.make_layout((BLOCK_M, BLOCK_N), (1, 0)),
            )
            col_tensor = fx.make_view(
                fx.make_int_tuple(0), fx.make_layout((BLOCK_M, N), (0, 1))
            )
            col_tensor = fx.flat_divide(col_tensor, (BLOCK_M, BLOCK_N))

            tcopyLDS, cp_ldsc = flyobj.get_tiled_copy_coalesced_mn(
                ldsC[None, None, 0], copy_atom_bits=128, num_threads=256
            )

            thrv_ldsC = tcopyLDS.partition_S(ldsC)

            copy_atom_ = flyobj.get_universal_copy_atom(fragC_bf16.dtype, 64)
            tcopy = flyobj.get_tiled_mma_copy(copy_atom_, mm, "C")
            fragC_bf16r = flyobj.get_retile(tcopy, fragC_bf16)

            thrv_ldsCt = flyobj.get_partition_D(tcopy, ldsCt)

            def postprocess_store2lds(fragC, ldsc_idx):
                for fc, fsw in fxh.all_elements(fragC, frag_sorted_weight):
                    fc.store(fc.load() * fsw.load())
                vec_f32 = fragC.load()
                fragC_bf16.store(_f32_to_bf16(vec_f32))
                fx.copy(copy_atom_, fragC_bf16r, thrv_ldsCt[None, None, None, ldsc_idx])

            arg_p_output = fx.flat_divide(arg_p_output, (BLOCK_M, BLOCK_N))
            cp_atom_out_128b = flyobj.get_buffer_copy_atom(fx.BFloat16, 128)
            thrv_out = tcopyLDS.partition_D(arg_p_output)
            fragOut = fx.make_fragment_like(thrv_ldsC[None, None, None, 0])

            def postprocess_store2vmem(n, ldsc_idx):
                fx.copy(cp_ldsc, thrv_ldsC[None, None, None, ldsc_idx], fragOut)
                fx.copy(cp_atom_out_128b, fragOut, thrv_out[None, None, None, 0, n])

            def hot_loop_scheduler():
                num_mfma_inst = (BLOCK_M // 16) * (
                    K // (16 if weight_dtype.width == 16 else 32)
                )
                num_stores = BLOCK_M // (256 // (BLOCK_N // 8))
                num_loads = K // ((4 * 8) if weight_dtype.width == 16 else (4 * 16))

                nloads = num_loads
                nstores = num_stores
                mfma_step = num_mfma_inst // (nloads + nstores)

                nmfma = num_mfma_inst - mfma_step * (nloads + nstores)
                if nmfma > 0:
                    fx.rocdl.sched_mfma(nmfma)

                for _ in fx.range_constexpr(nloads):
                    fx.rocdl.sched_mfma(mfma_step)
                    fx.rocdl.sched_group_barrier(0x10, 1, 0)

                for _ in fx.range_constexpr(nstores):
                    fx.rocdl.sched_mfma(mfma_step)
                    fx.rocdl.sched_group_barrier(0x10, 1, 0)

                fx.rocdl.sched_barrier(0)

            frag_weights = [None, None]
            frag_pc_scales = [None, None]
            frag_weights[0] = flyobj.load_tiled_mma_fragA(
                mm, weight, [None, None, 0, None]
            )
            if fx.const_expr(arg_w_scale is not None):
                frag_pc_scales[0] = flyobj.load_tiled_mma_fragC(
                    mm,
                    arg_w_scale,
                    [None, None, 0, 0],
                    copy_atom_bits=32 if weight_quant_type == "per_tensor" else 128,
                )

            gemm_compute(frag_weights[0], frag_pc_scales[0], fragC[0])
            frag_weights[1] = flyobj.load_tiled_mma_fragA(
                mm, weight, [None, None, 1, None]
            )
            if fx.const_expr(arg_w_scale is not None):
                frag_pc_scales[1] = flyobj.load_tiled_mma_fragC(
                    mm,
                    arg_w_scale,
                    [None, None, 1, 0],
                    copy_atom_bits=32 if weight_quant_type == "per_tensor" else 128,
                )

            postprocess_store2lds(fragC[0], 0)
            fx.gpu.barrier()
            for n, state in range(0, nBN - 2, 2, init=[]):
                fxh.asm_mark("aaa")
                postprocess_store2vmem(n, 0)
                flyobj.load_tiled_mma_fragA(
                    mm, weight, [None, None, n + 2, None], frag_weights[0]
                )
                if fx.const_expr(
                    arg_w_scale is not None and weight_quant_type != "per_tensor"
                ):
                    flyobj.load_tiled_mma_fragC(
                        mm, arg_w_scale, [None, None, n + 2, 0], frag_pc_scales[0]
                    )
                gemm_compute(frag_weights[1], frag_pc_scales[1], fragC[1])
                postprocess_store2lds(fragC[1], 1)

                hot_loop_scheduler()
                fx.gpu.barrier()

                fxh.asm_mark("bbb")

                postprocess_store2vmem(n + 1, 1)
                flyobj.load_tiled_mma_fragA(
                    mm, weight, [None, None, n + 3, None], frag_weights[1]
                )

                if fx.const_expr(
                    arg_w_scale is not None and weight_quant_type != "per_tensor"
                ):
                    flyobj.load_tiled_mma_fragC(
                        mm, arg_w_scale, [None, None, n + 3, 0], frag_pc_scales[1]
                    )
                gemm_compute(frag_weights[0], frag_pc_scales[0], fragC[0])
                postprocess_store2lds(fragC[0], 0)

                hot_loop_scheduler()
                fx.gpu.barrier()

            postprocess_store2vmem(nBN - 2, 0)
            gemm_compute(frag_weights[1], frag_pc_scales[1], fragC[1])
            postprocess_store2lds(fragC[1], 1)
            fx.gpu.barrier()
            postprocess_store2vmem(nBN - 1, 1)

    @flyc.jit
    def launch_prefill_1x4_n64(
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
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK
        moe_2stage_down_prefill_1x4_n64(
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
        ).launch(
            grid=(1, task_num, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_prefill_1x4_n64


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
    METADATA_TILE_SIZE_M=None,
    _task_table=False,
    _store_cache=_DOWN_STORE_CACHE_MODIFIER,
):
    assert stage == "down"
    assert down_path == "1x4_64x256"
    # weight_quant_type governs the WEIGHT scale form; act_quant_type governs the ACTIVATION
    # scale form (native-fp8 prefill only) and defaults to weight_quant_type (previous behavior
    # where a single quant_type drove both).
    if act_quant_type is None:
        act_quant_type = weight_quant_type
    assert (
        BLOCK_TILE_SIZE_M <= 256
    ), "BLOCK_SIZE_M must be less than or equal to 256 due to LDS size limit for sorted ids."
    assert weight_dtype in [
        "bf16",
        "fp8",
    ], "weight_dtype must be either 'bf16' or 'fp8'"
    assert weight_quant_type in [
        "no",
        "ptpc",
        "per_tensor",
    ], "weight_quant_type must be either 'no', 'ptpc' or 'per_tensor'"
    assert act_quant_type in [
        "no",
        "ptpc",
        "per_tensor",
    ], "act_quant_type must be either 'no', 'ptpc' or 'per_tensor'"
    assert activation in [
        "silu",
        "swiglu",
    ], "activation must be either 'silu' or 'swiglu'"
    if activation == "swiglu":
        swiglu_limit = float(swiglu_limit) if swiglu_limit else 7.0
    if METADATA_TILE_SIZE_M is None:
        METADATA_TILE_SIZE_M = BLOCK_TILE_SIZE_M
    assert (
        BLOCK_TILE_SIZE_M == METADATA_TILE_SIZE_M
    ), "only gateup prefill_1x4 supports different kernel/metadata M tiles"
    _, generic_xcd_count = get_down_device_config()
    block_m_per_4wave_group = 64
    assert BLOCK_TILE_SIZE_M == 64
    assert BLOCK_TILE_SIZE_N == 256
    assert alg == "prefill_1x4"
    assert N % BLOCK_TILE_SIZE_N == 0
    assert K % 64 == 0
    assert weight_dtype == "fp8"
    assert weight_quant_type in ("ptpc", "per_tensor")
    assert down_output_padding_bytes in (0, 32, 64, 128)
    activation_bytes = block_m_per_4wave_group * K
    scale_bytes = BLOCK_TILE_SIZE_N * 4 if weight_quant_type == "ptpc" else 0
    cshuffle_bytes = 4 * 16 * 64 * (fx.BFloat16.width // 8)
    assert activation_bytes + scale_bytes + cshuffle_bytes <= 64 * 1024, (
        "1x4_64x256 exceeds gfx942 LDS capacity; "
        f"activation={activation_bytes}B, scale={scale_bytes}B, "
        f"cshuffle={cshuffle_bytes}B"
    )
    output_row_stride = N + (
        down_output_padding_bytes // (fx.BFloat16.width // 8)
        if down_output_padding_bytes is not None
        else 0
    )
    # Supported native-fp8 prefill (weight, act) combos: weight ptpc requires act ptpc;
    # weight per_tensor allows act ptpc or per_tensor.
    if weight_dtype == "fp8" and alg == "prefill_1x4":
        assert (weight_quant_type == "ptpc" and act_quant_type == "ptpc") or (
            weight_quant_type == "per_tensor"
            and act_quant_type in ("ptpc", "per_tensor")
        ), (
            f"unsupported prefill quant combo (weight={weight_quant_type}, "
            f"act={act_quant_type})"
        )

    if weight_dtype == "bf16":
        weight_dtype = fx.BFloat16
    elif weight_dtype == "fp8":
        weight_dtype = fx.Float8E4M3FNUZ

    def _pack_scaled_bf16_pairs(values, scales):
        # Round before packing; a floating-point FMA bias of 0x8000 cannot replace integer rounding.
        packed = _f32_to_bf16(values * scales).bitcast(fx.Uint32)
        return [packed[index] for index in range_constexpr(packed.numel)]

    def _store_scaled_bf16(source, scales, destination):
        for src, scale, dst in fxh.all_elements(source, scales, destination):
            dst.store(
                Vec.from_elements(
                    _pack_scaled_bf16_pairs(src.load(), scale.load()),
                    fx.Uint32,
                ).bitcast(fx.BFloat16)
            )

    down_ops = FlyObjCache(cache_ir=False)

    @flyc.jit
    def _map_down_task(
        valid_rows: fx.Int32,
        task_rows: fx.Constexpr[int],
    ):
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
            arith.select(
                workgroup_idx < swizzle_limit,
                swizzled_e_idx,
                workgroup_idx,
            )
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
            arg_p_sorted_ids = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_ids)
                + (
                    row_begin
                    if const_expr(_task_table)
                    else e_offset * block_m_per_4wave_group
                ),
                (block_m_per_4wave_group,),
                fx.Int32,
            )
            arg_p_sorted_weights = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_weights)
                + (
                    row_begin
                    if const_expr(_task_table)
                    else e_offset * block_m_per_4wave_group
                ),
                (block_m_per_4wave_group,),
                fx.Float32,
            )
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
                    (
                        ((4, 2, 2, 4, 4, N // 256)),
                        (element_num, K // element_num),
                    ),
                    (
                        (
                            element_num,
                            16 * K,
                            32 * K,
                            64 * K,
                            4 * element_num,
                            256 * K,
                        ),
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
            arg_p_input = fx.rocdl.make_buffer_tensor(
                arg_p_input,
                max_size=False,
                num_records_bytes=fx.Int64(M)
                * (TOPK * K)
                * (arg_p_input.dtype.width // 8),
            )
            cp_atom = down_ops.get_buffer_copy_atom(arg_p_input.dtype, 128)

            def flatten_A(x):
                # second mode is innermost, so swap before flattening
                # to get the right order for the tiled copy
                x = fx.select(x, [1, 0])
                return fx.group(x, 0, -1)

            cp_ldsA0 = flatten_A(ldsA0)
            cp_rows = flatten_A(
                fxh.make_1d_coord_tensor(ldsA0, 0, fx.get_iter(arg_p_sorted_ids))
            )
            cp_cols = flatten_A(
                fxh.make_1d_coord_tensor(ldsA0, 1, fx.make_int_tuple(0))
            )
            # 2. Gather activations into LDS; the barrier starts the four-wave pipeline.
            for dst, row, col in fxh.all_copy_atoms(
                cp_ldsA0, cp_rows, cp_cols, atom_bits=128, num_threads=256
            ):
                sorted_id = row[0].bitcast(fx.Uint32)
                atom_A = fxh.atom_tensor(
                    arg_p_input,
                    (sorted_id & 0xFFFFFF, sorted_id >> 24, col[0]),
                    128,
                )
                fx.copy(cp_atom, atom_A, dst)
            fx.gpu.barrier()

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
                    fxh._as_ptr(p_w_scale) + expert_id * N,
                    fx.make_layout(N, 1),
                )
                scale_global_rsrc = fx.rocdl.get_buffer_rsrc(
                    fx.rocdl.make_buffer_ptr(
                        fx.get_iter(scale_global),
                        num_records_bytes=N * (fx.Float32.width // 8),
                    )
                )
                scale_lds_logical = fx.make_view(
                    fx.get_iter(scale_lds),
                    fx.make_layout(
                        ((4, 2, 2, 4, 4), 1),
                        ((1, 16, 32, 64, 4), 0),
                    ),
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
                    mm,
                    scale_lds_logical,
                    dst=dst,
                    copy_atom_bits=128,
                )

            arg_a_scale = None
            per_tensor_a_scale = None
            if const_expr(act_quant_type == "per_tensor"):
                if const_expr(weight_quant_type == "per_tensor"):
                    per_tensor_a_scale = fx.make_view(
                        fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                        fx.make_layout(1, 1),
                    )[0]
                else:
                    arg_a_scale = fx.make_view(
                        fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                        fx.make_layout((M, TOPK), (0, 0)),
                    )
                    arg_a_scale = fx.rocdl.make_buffer_tensor(
                        arg_a_scale,
                        max_size=False,
                        num_records_bytes=fx.Int64(1) * (arg_a_scale.dtype.width // 8),
                    )
            if const_expr(act_quant_type == "ptpc"):
                arg_a_scale = fx.make_view(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                    fx.make_layout((M, TOPK), (TOPK, 1)),
                )
                arg_a_scale = fx.rocdl.make_buffer_tensor(
                    arg_a_scale,
                    max_size=False,
                    num_records_bytes=fx.Int64(M)
                    * TOPK
                    * (arg_a_scale.dtype.width // 8),
                )

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
            elif fx.const_expr(arg_a_scale is not None):
                """Load per-token scales and combine them with routing weights."""
                cp_atom = down_ops.get_buffer_copy_atom(p_a_scale.dtype, 32)
                coord_tensor = fx.make_view(
                    fx.get_iter(arg_p_sorted_ids),
                    fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
                )
                frag_coord = down_ops.load_tiled_mma_fragC(
                    mm, coord_tensor, copy_atom_bits=32
                )
                frag_pt_scales = mm.make_fragment_C(coord_tensor)
                frag_pt_scalesr = down_ops.get_tiled_mma_retile(
                    mm, frag_pt_scales, "C", copy_atom=cp_atom
                )

                for dst, coord in fxh.all_elements(frag_pt_scalesr, frag_coord):
                    sorted_id = coord[0].bitcast(fx.Uint32)
                    atom_A = fxh.atom_tensor(
                        arg_a_scale,
                        (sorted_id & 0xFFFFFF, sorted_id >> 24),
                        32,
                    )
                    fx.copy(cp_atom, atom_A, dst)

                # combine per-token scales with per-token weights
                for frag_pt, frag_sw in fxh.all_elements(
                    frag_pt_scales, frag_sorted_weight
                ):
                    combined_scale = frag_pt.load() * frag_sw.load()
                    if const_expr(weight_quant_type == "per_tensor"):
                        combined_scale = combined_scale * per_tensor_w_scale
                    frag_pt.store(combined_scale)
                frag_sorted_weight = frag_pt_scales
            cshuffle_copy_atom = down_ops.get_universal_copy_atom(fx.BFloat16, 128)

            def store_cshuffle_n256(output, block_n, cshuffle_lds_arg):
                lane_id = fx.Int32(fx.thread_idx.x % 64)
                wave_id = fx.Int32(fx.thread_idx.x // 64)
                lane_group = lane_id // 16
                lane_row = lane_id % 16
                wave_lds_base = wave_id * (16 * 64)

                # All 64 lanes produce one M16 pair, halving the DS writes used
                # by the previous two masked M8 producers.
                def write_row_pair(row_pair):
                    row_in_8 = lane_row % 8
                    row_half = lane_row // 8
                    for channel_piece in range_constexpr(2):
                        channels_lo = Vec(
                            output[None, 2 * channel_piece, row_pair].load()
                        )
                        channels_hi = Vec(
                            output[None, 2 * channel_piece + 1, row_pair].load()
                        )
                        packed_bf16 = channels_lo.shuffle(channels_hi, list(range(8)))
                        logical_atom = lane_group * 2 + channel_piece
                        physical_atom = logical_atom ^ row_in_8
                        lds_offset = (
                            wave_lds_base
                            + ((row_half * 8 + row_in_8) * 8 + physical_atom) * 8
                        )
                        lds_dst = fx.make_view(
                            fx.get_iter(cshuffle_lds_arg) + lds_offset,
                            fx.make_layout(8, 1),
                        )
                        lds_frag = fx.make_fragment_like(lds_dst)
                        lds_frag.store(packed_bf16)
                        fx.copy(cshuffle_copy_atom, lds_frag, lds_dst)

                def read_store_row_pair(row_pair):
                    out_frags = []
                    byte_offsets = []
                    fx.rocdl.sched_barrier(0)
                    for row_half in range_constexpr(2):
                        output_row = (row_pair * 2 + row_half) * 8 + lane_id // 8
                        output_atom = lane_id % 8
                        physical_atom = output_atom ^ (lane_id // 8)
                        lds_offset = (
                            wave_lds_base
                            + ((row_half * 8 + lane_id // 8) * 8 + physical_atom) * 8
                        )
                        lds_src = fx.make_view(
                            fx.get_iter(cshuffle_lds_arg) + lds_offset,
                            fx.make_layout(8, 1),
                        )
                        out_frag = fx.make_fragment_like(lds_src)
                        fx.copy(cshuffle_copy_atom, lds_src, out_frag)
                        out_frags.append(out_frag)
                        output_column = (
                            fx.Int64(block_n) * BLOCK_N
                            + fx.Int64(wave_id) * WAVE_N
                            + fx.Int64(output_atom) * 8
                        )
                        byte_offset = (
                            (fx.Int64(output_row) * output_row_stride + output_column)
                            * 2
                        ).to(fx.Int32)
                        byte_offsets.append(byte_offset)

                    # Consume the older read first without forcing the newer read complete.
                    fx.rocdl.s_waitcnt(lgkmcnt=1)
                    fx.rocdl.RawPtrBufferStoreOp(
                        Vec(out_frags[0].load()).bitcast(fx.Int32).ir_value(),
                        output_store_rsrc,
                        byte_offsets[0].ir_value(),
                        fx.Int32(0).ir_value(),
                        aux=ir.IntegerAttr.get(fx.Int32.ir_type, _store_cache),
                    )
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                    fx.rocdl.RawPtrBufferStoreOp(
                        Vec(out_frags[1].load()).bitcast(fx.Int32).ir_value(),
                        output_store_rsrc,
                        byte_offsets[1].ir_value(),
                        fx.Int32(0).ir_value(),
                        aux=ir.IntegerAttr.get(fx.Int32.ir_type, _store_cache),
                    )

                for row_pair in range_constexpr(4):
                    write_row_pair(row_pair)
                    read_store_row_pair(row_pair)

            def postprocess_store_vector_4wave(
                output,
                sorted_weight,
                block_n,
                vector_begin=0,
                vector_count=8,
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
                    logical_atom = lane_group * 2 + channel_piece
                    physical_atom = logical_atom ^ row_in_8
                    lds_offset = (
                        wave_lds_base
                        + ((row_half * 8 + row_in_8) * 8 + physical_atom) * 8
                    )
                    lds_dst = fx.make_view(
                        fx.get_iter(cshuffle_lds) + lds_offset,
                        fx.make_layout(8, 1),
                    )
                    lds_frag = fx.make_fragment_like(lds_dst)
                    lds_frag.store(packed_bf16)
                    fx.copy(cshuffle_copy_atom, lds_frag, lds_dst)

                    if const_expr(channel_piece == 1):
                        out_frags = []
                        byte_offsets = []
                        fx.rocdl.sched_barrier(0)
                        for row_half_out in range_constexpr(2):
                            output_atom = lane_id % 8
                            physical_atom_out = output_atom ^ (lane_id // 8)
                            lds_offset_out = (
                                wave_lds_base
                                + (
                                    (row_half_out * 8 + lane_id // 8) * 8
                                    + physical_atom_out
                                )
                                * 8
                            )
                            lds_src = fx.make_view(
                                fx.get_iter(cshuffle_lds) + lds_offset_out,
                                fx.make_layout(8, 1),
                            )
                            out_frag = fx.make_fragment_like(lds_src)
                            fx.copy(cshuffle_copy_atom, lds_src, out_frag)
                            out_frags.append(out_frag)
                            output_row = (
                                row_pair * 2 + row_half_out
                            ) * 8 + lane_id // 8
                            output_column = (
                                fx.Int64(block_n) * BLOCK_N
                                + fx.Int64(wave_id) * WAVE_N
                                + fx.Int64(output_atom) * 8
                            )
                            byte_offsets.append(
                                (
                                    (
                                        fx.Int64(output_row) * output_row_stride
                                        + output_column
                                    )
                                    * 2
                                ).to(fx.Int32)
                            )
                        fx.rocdl.s_waitcnt(lgkmcnt=1)
                        fx.rocdl.RawPtrBufferStoreOp(
                            Vec(out_frags[0].load()).bitcast(fx.Int32).ir_value(),
                            output_store_rsrc,
                            byte_offsets[0].ir_value(),
                            fx.Int32(0).ir_value(),
                            aux=ir.IntegerAttr.get(fx.Int32.ir_type, _store_cache),
                        )
                        fx.rocdl.s_waitcnt(lgkmcnt=0)
                        fx.rocdl.RawPtrBufferStoreOp(
                            Vec(out_frags[1].load()).bitcast(fx.Int32).ir_value(),
                            output_store_rsrc,
                            byte_offsets[1].ir_value(),
                            fx.Int32(0).ir_value(),
                            aux=ir.IntegerAttr.get(fx.Int32.ir_type, _store_cache),
                        )

            use_delayed_4wave_store = BLOCK_K == 128 and nBK == 2

            def enter_read_write_stage():
                fx.rocdl.sched_barrier(0)
                fx.rocdl.s_setprio(0)
                fx.rocdl.sched_barrier(0)

            def enter_compute_stage():
                fx.rocdl.sched_barrier(0)
                fx.rocdl.s_setprio(3)
                fx.rocdl.sched_barrier(0)

            # Prologue: stage0 prepares N block 0 / K core 0.
            enter_read_write_stage()
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
                if const_expr(use_delayed_4wave_store):  # noqa: SIM102
                    if block_n > 0:
                        if const_expr(k_core == 0):
                            postprocess_store_vector_4wave(
                                previous_fragC,
                                frag_sorted_weight,
                                block_n - 1,
                                vector_begin=0,
                                vector_count=3,
                            )
                        elif const_expr(k_core == 1):
                            postprocess_store_vector_4wave(
                                previous_fragC,
                                frag_sorted_weight,
                                block_n - 1,
                                vector_begin=3,
                                vector_count=5,
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
                if const_expr(k_core + 1 < nBK):
                    down_ops.load_tiled_mma_fragA(
                        mm,
                        weight,
                        [None, None, block_n, k_core + 1],
                        frag_weight_slots[next_slot],
                    )
                else:
                    next_block_n = block_n + 1
                    if const_expr(weight_quant_type == "ptpc"):
                        next_scale_vec = issue_scale_block_global(next_block_n)
                        fx.rocdl.sched_barrier(0)
                    down_ops.load_tiled_mma_fragA(
                        mm,
                        weight,
                        [None, None, next_block_n, 0],
                        frag_weight_slots[next_slot],
                    )
                enter_compute_stage()
                fx.gemm(
                    mm,
                    fragC,
                    frag_weight_slots[current_slot],
                    frag_act,
                    fragC,
                )
                enter_read_write_stage()
                if const_expr(k_core + 1 == nBK and weight_quant_type == "ptpc"):
                    commit_scale_block_lds(next_scale_vec, next_frag_pc_scale)

            previous_fragC = None
            if const_expr(use_delayed_4wave_store):
                previous_fragC = fx.make_fragment_like(fragC)
                previous_fragC.fill(0)

            delayed_output_state_index = 2 if weight_quant_type == "ptpc" else 1
            # Carry FP8 packets as bits across SCF boundaries. The pinned
            # compiler lowers FP8 memory to integers before lowering scf.for.
            if const_expr(weight_quant_type == "ptpc"):
                loop_state = [
                    frag_weight.load().bitcast(fx.Uint32),
                    frag_pc_scale.load(),
                ]
            else:
                loop_state = [frag_weight.load().bitcast(fx.Uint32)]
            if const_expr(use_delayed_4wave_store):
                loop_state.append(previous_fragC.load())
            for block_n, state in range(
                fx.Int64(0),
                fx.Int64(nBN),
                fx.Int64(1),
                init=loop_state,
            ):
                # Restore the next weight/scale and any delayed output tile.
                frag_weight_slots[0].store(state[0].bitcast(weight_dtype))
                if const_expr(weight_quant_type == "ptpc"):
                    frag_pc_scale.store(state[1])
                if const_expr(use_delayed_4wave_store):
                    previous_fragC.store(state[delayed_output_state_index])

                # Compute this N tile while retiring the previous tile's stores.
                fragC.fill(0)
                for k_core in range_constexpr(nBK):
                    run_k_core_pipeline(
                        block_n,
                        k_core,
                        previous_fragC,
                        next_frag_pc_scale,
                    )

                # Apply channel scale, routing weight and BF16 conversion.
                if const_expr(weight_quant_type == "ptpc"):
                    for fc, fpc in fxh.all_elements(fragC, frag_pc_scale):
                        fc.store(fc.load() * fpc.load())
                if not const_expr(use_delayed_4wave_store):
                    _store_scaled_bf16(fragC, frag_sorted_weight, fragC_bf16)
                    store_cshuffle_n256(fragC_bf16, block_n, cshuffle_lds)

                if const_expr(weight_quant_type == "ptpc"):
                    next_state = [
                        frag_weight_slots[nBK % 2].load().bitcast(fx.Uint32),
                        next_frag_pc_scale.load(),
                    ]
                else:
                    next_state = [frag_weight_slots[nBK % 2].load().bitcast(fx.Uint32)]
                if const_expr(use_delayed_4wave_store):
                    next_state.append(fragC.load())
                results = yield next_state

            # Drain the final delayed tile after the N loop.
            if const_expr(use_delayed_4wave_store):
                previous_fragC.store(results[delayed_output_state_index])
                postprocess_store_vector_4wave(
                    previous_fragC,
                    frag_sorted_weight,
                    fx.Int64(nBN - 1),
                )
            fx.rocdl.sched_barrier(0)
            fx.rocdl.s_setprio(0)
            fx.rocdl.sched_barrier(0)

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
