# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Down-kernel dispatch and baseline split-K and batch1 implementations."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr

from . import common as fxh
from .common import (
    _f32_to_bf16,
    make_gemm_helpers,
    validate_gemm_options,
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
    down_path="default",
    down_output_padding_bytes=0,
):
    if down_path != "default":
        from .gemm2_1x4 import _build_moe_gemm2_1x4_n256
        from .gemm2_8x1 import _build_moe_gemm2_8x1
        from .gemm2_8x1_compact import _build_moe_gemm2_8x1_compact

        builders = {
            "1x4_64x256": _build_moe_gemm2_1x4_n256,
            "8x1": _build_moe_gemm2_8x1,
            "8x1_compact": _build_moe_gemm2_8x1_compact,
        }
        if down_path not in builders:
            raise ValueError(f"Unsupported Down path: {down_path}")
        return builders[down_path](
            N=N,
            K=K,
            weight_dtype=weight_dtype,
            weight_quant_type=weight_quant_type,
            TOPK=TOPK,
            BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
            BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
            alg=alg,
            E=E,
            USE_ATOMIC_WRITE=USE_ATOMIC_WRITE,
            act_quant_type=act_quant_type,
            tile_k=tile_k,
            activation=activation,
            swiglu_limit=swiglu_limit,
            down_path=down_path,
            down_output_padding_bytes=down_output_padding_bytes,
        )
    if down_output_padding_bytes != 0:
        raise ValueError("The default Down path does not support output padding")
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
        from .gemm2_1x4 import _build_moe_gemm2_1x4_n64

        return _build_moe_gemm2_1x4_n64(
            N=N,
            K=K,
            weight_dtype=weight_dtype,
            weight_quant_type=weight_quant_type,
            TOPK=TOPK,
            BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
            act_quant_type=act_quant_type,
            E=E,
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

    if const_expr(alg == "batch1"):
        return launch_batch1
    return launch_splitk
