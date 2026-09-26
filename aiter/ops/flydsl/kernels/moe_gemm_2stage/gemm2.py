# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Baseline Down kernels and device-local compilation cache."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr

from .. import moe_gemm_2stage_utils as fxh
from .common import (
    _f32_to_bf16,
    device_context,
    get_device_cache_key,
    resolve_tile_k,
    validate_gemm_options,
)
from .layout_helpers import make_gemm_helpers


def _build_moe_gemm2(
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
):
    TILE_K = 64
    act_quant_type, swiglu_limit = validate_gemm_options(
        weight_dtype,
        weight_quant_type,
        act_quant_type,
        BLOCK_TILE_SIZE_M,
        alg,
        activation,
        swiglu_limit,
    )
    if alg == "prefill_1x4":
        assert K % 64 == 0, f"down prefill requires K to be divisible by 64, got K={K}"
        assert N % 256 == 0, (
            f"down prefill requires N to be divisible by 256 for paired "
            f"128-wide tiles, got N={N}"
        )

    if alg == "splitk":
        assert (
            K % TILE_K == 0
        ), f"down split-K requires K to be divisible by {TILE_K}, got K={K}"

        @fx.struct
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]

    if weight_dtype == "bf16":
        weight_dtype = fx.BFloat16
    elif weight_dtype == "fp8":
        weight_dtype = fx.Float8E4M3FNUZ

    TensorWithIndex, _read_sorted_index, gemm_splitk = make_gemm_helpers(
        K, weight_dtype, BLOCK_TILE_SIZE_M, TOPK
    )

    def _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale):
        if const_expr(weight_dtype != fx.BFloat16):
            if const_expr(weight_quant_type == "ptpc"):
                arg_p_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id * N, fx.make_layout(N, 1)
                )
                scale_tile = fx.flat_divide(
                    arg_p_scale, fx.make_tile(BLOCK_TILE_SIZE_N)
                )[None, blk_n]
                cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
                tiled_copy_scale = fx.make_tiled_copy(
                    cp_atom_scale,
                    fx.make_layout(((16, 4), 4), ((0, 4), 1)),
                    fx.make_tile(16),
                )
                scale_frag_tensor = tiled_copy_scale.get_slice(tid).partition_S(
                    scale_tile
                )
                scale_frag = fx.make_fragment_like(scale_frag_tensor)
                fx.copy(cp_atom_scale, scale_frag_tensor, scale_frag)
                m_reps = fx.size(fx.get_shape(c_frag)[1]).to_py_value()
                n_reps = fx.size(fx.get_shape(c_frag)[2]).to_py_value()
                for n in range_constexpr(n_reps):
                    scale_vec = scale_frag[None, n].load()
                    for m in range_constexpr(m_reps):
                        c_vec = c_frag[None, m, n].load()
                        vec = c_vec * scale_vec
                        c_frag[None, m, n].store(vec)
            elif const_expr(weight_quant_type == "per_tensor"):
                arg_p_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id, fx.make_layout(1, 1)
                )
                scale = arg_p_scale[0]
                c_frag.store(c_frag.load() * scale)

    def _cvt_f32_to_bf16(c_frag):
        c_frag_bf16 = fx.make_fragment_like(c_frag, dtype=fx.BFloat16)
        c_frag_bf16.store(_f32_to_bf16(c_frag.load()))
        return c_frag_bf16

    def _make_down_weight_view(p_weight, expert_id):
        element_num = 16 // (p_weight.dtype.width // 8)
        return fx.make_view(
            p_weight + fx.Int64(expert_id * N * K),
            fx.make_layout(
                ((16, N // 16), (element_num, K // element_num)),
                ((element_num, 16 * K), (1, 16 * element_num)),
            ),
        )

    @flyc.kernel
    def moe_2stage_down_splitk(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        arg_p_input = fx.make_view(
            fxh._as_ptr(p_input), fx.make_layout((M, TOPK, K), (TOPK * K, K, 1))
        )
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        if const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, fxh._as_ptr(p_weight))
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(
                    fx.Int32, fxh._as_ptr(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            arg_p_sorted_expert_ids = fx.recast_iter(
                fx.Int32, fxh._as_ptr(p_sorted_expert_ids)
            )
            expert_id = arg_p_sorted_expert_ids[e_idx]
            arg_p_weight = _make_down_weight_view(p_weight, expert_id)

            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            lds_view = fx.make_view(
                lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1)
            )
            for idx in range(tid, BLOCK_TILE_SIZE_M, 64):
                lds_view[idx] = sorted_ids_buf[idx]
            gpu.barrier()

            cp_atom_weight = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            arg_p_sorted_weights = fx.make_view(
                fx.recast_iter(
                    fx.Float32,
                    fxh._as_ptr(p_sorted_weights) + e_idx * BLOCK_TILE_SIZE_M,
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            sorted_weights_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_weights, max_size=False
            )
            cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
            tiled_copy_sortid_lds = fx.make_tiled_copy(
                cp_atom_lds, fx.make_layout(((16, 4), 1), ((1, 0), 0)), fx.make_tile(16)
            )
            sorted_weights_tensor = tiled_copy_sortid_lds.get_slice(tid).partition_S(
                sorted_weights_buf
            )
            sorted_weight_frag = fx.make_fragment_like(
                sorted_weights_tensor, fx.Float32
            )
            fx.copy(cp_atom_weight, sorted_weights_tensor, sorted_weight_frag)

            c_frag = gemm_splitk(
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                arg_p_weight,
                lds,
                splitk_waves=1,
            )

            _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale)

            sorted_weight_frag_vec = sorted_weight_frag.load()
            for m in range_constexpr(BLOCK_TILE_SIZE_M // 16):
                w = sorted_weight_frag_vec[m]
                v = c_frag[None, m, None].load()
                v *= w
                c_frag[None, m, None].store(v)

            c_frag_bf16 = _cvt_f32_to_bf16(c_frag)

            if const_expr(not USE_ATOMIC_WRITE):
                arg_p_output = fx.make_view(
                    fxh._as_ptr(p_output),
                    fx.make_layout((M, TOPK, N), (TOPK * N, N, 1)),
                )
                arg_p_output = fx.rocdl.make_buffer_tensor(
                    arg_p_output,
                    max_size=False,
                    num_records_bytes=fx.Int64(M) * (TOPK * N * fx.BFloat16.width // 8),
                )
                cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.BFloat16)
                is_atomic_write = False
            else:
                arg_p_output = fx.make_view(
                    fxh._as_ptr(p_output), fx.make_layout((M, N), (N, 1))
                )
                cp_atom_w = fx.make_copy_atom(
                    fx.UniversalAtomic(fx.AtomicOp.Add, fx.BFloat16), fx.BFloat16
                )
                is_atomic_write = True
            c_tiled_g = fx.make_tiled_copy(
                cp_atom_w,
                fx.make_layout(((16, 4), 4), ((1, 64), 16)),
                fx.make_tile(16, 16),
            )
            c_index_frag = _read_sorted_index(
                tiled_copy_sortid_lds, tid, lds.sorted_lds
            )
            c_tensor = TensorWithIndex(
                arg_p_output,
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                c_index_frag,
                c_tiled_g,
                tid,
                is_read_from_mem=False,
                TOPK=TOPK,
                is_atomic_write=is_atomic_write,
            )
            c_tensor.copy(
                cp_atom_w, blk_n, c_tiled_g.get_slice(tid).retile(c_frag_bf16)
            )

    @flyc.kernel
    def moe_2stage_down_batch1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_topk_weights: fx.Pointer,
        p_w_scale: fx.Pointer,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        arg_p_input = fx.make_view(
            fxh._as_ptr(p_input) + fx.Int64(e_idx * K),
            fx.make_layout((BLOCK_TILE_SIZE_M, K), (0, 1)),
        )
        if const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, fxh._as_ptr(p_weight))
        arg_p_topk_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_topk_ids))
        arg_p_topk_weights = fx.recast_iter(fx.Float32, fxh._as_ptr(p_topk_weights))
        expert_id = arg_p_topk_ids[e_idx]
        topk_weight = arg_p_topk_weights[e_idx]
        arg_p_weight = _make_down_weight_view(p_weight, expert_id)

        c_frag = gemm_splitk(
            BLOCK_TILE_SIZE_M,
            BLOCK_TILE_SIZE_N,
            TILE_K,
            blk_n,
            arg_p_input,
            arg_p_weight,
            None,
            splitk_waves=1,
            a_with_index=False,
        )

        _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale)

        c_frag.store(c_frag.load() * topk_weight)

        c_frag_bf16 = _cvt_f32_to_bf16(c_frag)

        arg_p_output = fx.make_view(
            fxh._as_ptr(p_output), fx.make_layout((1, N), (N, 1))
        )
        cp_atom_w = fx.make_copy_atom(
            fx.UniversalAtomic(fx.AtomicOp.Add, fx.BFloat16), fx.BFloat16
        )
        c_tiled_g = fx.make_tiled_copy(
            cp_atom_w,
            fx.make_layout(((16, 4), 4), ((1, 64), 16)),
            fx.make_tile(16, 16),
        )
        c_tile = fx.flat_divide(
            arg_p_output, fx.make_tile(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N)
        )[None, None, None, blk_n]
        c_dst = c_tiled_g.get_slice(tid).partition_S(c_tile)
        c_src = c_tiled_g.get_slice(tid).retile(c_frag_bf16)
        rep_m = fx.size(fx.get_shape(c_src)[1]).to_py_value()
        rep_n = fx.size(fx.get_shape(c_src)[2]).to_py_value()
        if tid % 16 == 0:
            for m in range_constexpr(rep_m):
                for n in range_constexpr(rep_n):
                    reg_vec = c_src[None, m, n].load()
                    ptr_base = fx.get_iter(c_dst[None, m, n, 0])
                    fxh.atomic_add_bf16(ptr_base, reg_vec)

    @flyc.kernel
    def moe_2stage_down_prefill_1x4(
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

        max_valid_id = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]

        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            # Keep the stateless helper trace-local, not an scf.if-carried value.
            tile_ops = fxh.MoETileOps()
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
            cp_atom = tile_ops.get_buffer_copy_atom(arg_p_input.dtype, 128)

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

            mm = tile_ops.create_thr_mma(weight_dtype, (4, 1, 1))

            c_fake_tensor = fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_ordered_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            fragC = [
                mm.make_fragment_C(c_fake_tensor),
                mm.make_fragment_C(c_fake_tensor),
            ]
            fragC_bf16 = fx.make_fragment_like(fragC[0], fx.BFloat16)

            frag_act = tile_ops.load_tiled_mma_fragB(mm, ldsA, copy_atom_bits=128)
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
            frag_sorted_weight = tile_ops.load_tiled_mma_fragC(
                mm, sorted_weights, copy_atom_bits=32
            )

            if fx.const_expr(arg_a_scale is not None):
                cp_atom = tile_ops.get_buffer_copy_atom(p_a_scale.dtype, 32)
                coord_tensor = fx.make_view(
                    fx.get_iter(arg_p_sorted_ids),
                    fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
                )
                frag_coord = tile_ops.load_tiled_mma_fragC(
                    mm, coord_tensor, copy_atom_bits=32
                )
                frag_pt_scales = mm.make_fragment_C(coord_tensor)
                frag_pt_scalesr = tile_ops.get_tiled_mma_retile(
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

            tcopyLDS, cp_ldsc = tile_ops.get_tiled_copy_coalesced_mn(
                ldsC[None, None, 0], copy_atom_bits=128, num_threads=256
            )

            thrv_ldsC = tcopyLDS.partition_S(ldsC)

            copy_atom_ = tile_ops.get_universal_copy_atom(fragC_bf16.dtype, 64)
            tcopy = tile_ops.get_tiled_mma_copy(copy_atom_, mm, "C")
            fragC_bf16r = tile_ops.get_retile(tcopy, fragC_bf16)

            thrv_ldsCt = tile_ops.get_partition_D(tcopy, ldsCt)

            def postprocess_store2lds(fragC, ldsc_idx):
                for fc, fsw in fxh.all_elements(fragC, frag_sorted_weight):
                    fc.store(fc.load() * fsw.load())
                vec_f32 = fragC.load()
                fragC_bf16.store(_f32_to_bf16(vec_f32))
                fx.copy(copy_atom_, fragC_bf16r, thrv_ldsCt[None, None, None, ldsc_idx])

            arg_p_output = fx.flat_divide(arg_p_output, (BLOCK_M, BLOCK_N))
            cp_atom_out_128b = tile_ops.get_buffer_copy_atom(fx.BFloat16, 128)
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
            frag_weights[0] = tile_ops.load_tiled_mma_fragA(
                mm, weight, [None, None, 0, None]
            )
            if fx.const_expr(arg_w_scale is not None):
                frag_pc_scales[0] = tile_ops.load_tiled_mma_fragC(
                    mm,
                    arg_w_scale,
                    [None, None, 0, 0],
                    copy_atom_bits=32 if weight_quant_type == "per_tensor" else 128,
                )

            gemm_compute(frag_weights[0], frag_pc_scales[0], fragC[0])
            frag_weights[1] = tile_ops.load_tiled_mma_fragA(
                mm, weight, [None, None, 1, None]
            )
            if fx.const_expr(arg_w_scale is not None):
                frag_pc_scales[1] = tile_ops.load_tiled_mma_fragC(
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
                tile_ops.load_tiled_mma_fragA(
                    mm, weight, [None, None, n + 2, None], frag_weights[0]
                )
                if fx.const_expr(
                    arg_w_scale is not None and weight_quant_type != "per_tensor"
                ):
                    tile_ops.load_tiled_mma_fragC(
                        mm, arg_w_scale, [None, None, n + 2, 0], frag_pc_scales[0]
                    )
                gemm_compute(frag_weights[1], frag_pc_scales[1], fragC[1])
                postprocess_store2lds(fragC[1], 1)

                hot_loop_scheduler()
                fx.gpu.barrier()

                fxh.asm_mark("bbb")

                postprocess_store2vmem(n + 1, 1)
                tile_ops.load_tiled_mma_fragA(
                    mm, weight, [None, None, n + 3, None], frag_weights[1]
                )

                if fx.const_expr(
                    arg_w_scale is not None and weight_quant_type != "per_tensor"
                ):
                    tile_ops.load_tiled_mma_fragC(
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
    def launch_splitk(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK
        moe_2stage_down_splitk(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            M,
        ).launch(
            grid=(num_n_blocks, task_num, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_batch1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_topk_weights: fx.Pointer,
        p_w_scale: fx.Pointer,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)
        moe_2stage_down_batch1(
            p_input, p_weight, p_output, p_topk_ids, p_topk_weights, p_w_scale
        ).launch(
            grid=(num_n_blocks, task_num, 1),
            block=(64, 1, 1),
            stream=stream,
        )

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
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK
        moe_2stage_down_prefill_1x4(
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

    if const_expr(alg == "prefill_1x4"):
        return launch_prefill_1x4
    if const_expr(alg == "batch1"):
        return launch_batch1
    return launch_splitk


@functools.cache
def _compile_moe_gemm2_cached(
    device_cache_key,
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    alg,
    E,
    USE_ATOMIC_WRITE,
    act_quant_type,
    tile_k,
    activation,
    swiglu_limit,
):
    del device_cache_key
    return _build_moe_gemm2(
        N,
        K,
        weight_dtype,
        weight_quant_type,
        TOPK,
        BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N,
        alg,
        E,
        USE_ATOMIC_WRITE,
        act_quant_type,
        tile_k,
        activation,
        swiglu_limit,
    )


def compile_moe_gemm2(
    *,
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    device=None,
):
    """Return the device-local Down launcher with the baseline launch ABI."""
    with device_context(device):
        return _compile_moe_gemm2_cached(
            get_device_cache_key(device),
            N,
            K,
            weight_dtype,
            weight_quant_type,
            TOPK,
            BLOCK_TILE_SIZE_M,
            BLOCK_TILE_SIZE_N,
            alg,
            E,
            USE_ATOMIC_WRITE,
            act_quant_type,
            resolve_tile_k(tile_k),
            activation,
            swiglu_limit,
        )


compile_moe_gemm2.cache_clear = _compile_moe_gemm2_cached.cache_clear
compile_moe_gemm2.cache_info = _compile_moe_gemm2_cached.cache_info
