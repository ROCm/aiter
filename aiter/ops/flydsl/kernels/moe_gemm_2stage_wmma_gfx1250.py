# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


"""gfx1250 MoE 2-stage fp16 WMMA kernels.

Implements stage1/stage2 single-kernel inline paths using the
``wmma_f32_16x16x32_f16`` instruction for fp16 (and bf16 via host
conversion) inputs.
"""

from __future__ import annotations

import functools

from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from kernels.moe_gemm_2stage import (
    MoeGemm2Mode,
    compile_moe_reduction,
)
from kernels.moe_gemm_2stage_common_gfx1250 import (
    _bf16_to_f16_wrapper,
    _emit_stage1_gate_up_epilogue,
    _emit_stage2_store_epilogue,
    _finalize_alloc_and_launch_2d,
    _make_moe_wave_layout,
    _make_wmma_sub_tiles,
    _moe_out_elem_ty,
    _pick_fp16_single_launch_shape,
    _require_gfx1250,
)

@functools.lru_cache(maxsize=64)
def _compile_stage1_wmma_kernel_impl(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    route_tile_m: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    doweight_stage1: bool,
    out_dtype: str,
    waves_per_eu: int | None,
    expert_sched_mode: bool = True,
):
    """Compile dense stage1 single kernel: route-pack + TDM + WMMA + epilog."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as llvm_dialect
    from flydsl._mlir.dialects import scf
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, gpu, idx2crd, range_constexpr, rocdl, tdm_ops, vector
    from flydsl.expr.typing import T
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

    WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
    WAVE_SIZE = 32
    LDS_PAD_A = 8
    LDS_PAD_B = 8
    elem_bytes = 2

    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"fp16 stage1 single kernel supports out_dtype in ('f16','bf16'), got {out_dtype!r}")
    if (int(model_dim) % int(tile_k)) != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by tile_k={tile_k}")
    if (int(tile_k) % WMMA_K) != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by {WMMA_K}")
    if (int(tile_m) % WMMA_M) != 0 or (int(tile_n) % WMMA_N) != 0:
        raise ValueError(f"tile_m/tile_n must be multiples of 16, got ({tile_m},{tile_n})")

    block_threads = int(m_warp) * int(n_warp) * WAVE_SIZE
    warp_tile_m = int(tile_m) // int(m_warp)
    warp_tile_n = int(tile_n) // int(n_warp)
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    if wmma_m_rep <= 0 or wmma_n_rep <= 0:
        raise ValueError(f"Invalid warp tiling for fp16 single kernel: wmma_m_rep={wmma_m_rep}, wmma_n_rep={wmma_n_rep}")

    n_accs = wmma_m_rep * wmma_n_rep
    num_k_tiles = int(model_dim) // int(tile_k)
    k_wmma_steps = int(tile_k) // WMMA_K
    n_total = int(2 * inter_dim)
    _sub_tiles = _make_wmma_sub_tiles(
        wmma_m_rep=wmma_m_rep, wmma_n_rep=wmma_n_rep, WMMA_M=WMMA_M, is_fp4=False
    )

    lds_a_stride = int(tile_k) + LDS_PAD_A
    lds_b_stride = int(tile_n) + LDS_PAD_B
    lds_a_elems = int(tile_m) * lds_a_stride + LDS_PAD_A
    lds_b_elems = int(tile_k) * lds_b_stride + LDS_PAD_B

    alloc = SmemAllocator(None, arch=str(get_hip_arch()), global_sym_name="moe_fp16_s1_single")
    off_bg = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_bg + lds_b_elems * elem_bytes
    off_bu = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_bu + lds_b_elems * elem_bytes
    off_a = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_a + lds_a_elems * elem_bytes

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def moe_fp16_stage1_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        _ = (arg_scale_x, arg_scale_w, arg_max_token_ids, i32_k_in)
        llvm_dialect.inline_asm(
            None, [],  # void result, no operands
            "s_setreg_imm32_b32 hwreg(26, 4, 1), 1",
            "",
            has_side_effects=True,
        )

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")  # inter tile
        by = gpu.block_id("y")  # expert block

        tokens_idx = arith.index_cast(T.index, i32_tokens_in)
        inter_idx = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids = arith.index_cast(T.index, i32_size_expert_ids_in)

        sorted_num = size_expert_ids * arith.index(int(route_tile_m))
        sorted_nbytes = sorted_num * arith.index(4)
        eid_nbytes = size_expert_ids * arith.index(4)
        x_nbytes = tokens_idx * arith.index(int(model_dim)) * arith.index(2)
        w_nbytes = arith.index(int(experts * n_total * int(model_dim) * 2))

        sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False, num_records_bytes=sorted_nbytes)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_nbytes)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False, num_records_bytes=w_nbytes)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        sw_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=True)

        eid_i32 = buffer_ops.buffer_load(eid_rsrc, arith.index_cast(T.i32, by), vec_width=1, dtype=T.i32)
        eid_ok0 = arith.cmpi(arith.CmpIPredicate.sge, eid_i32, arith.constant(0, type=T.i32))
        eid_ok1 = arith.cmpi(arith.CmpIPredicate.slt, eid_i32, arith.constant(int(experts), type=T.i32))
        eid_ok = arith.andi(eid_ok0, eid_ok1)

        layout_thr = _make_moe_wave_layout(m_warp=m_warp, n_warp=n_warp, WAVE_SIZE=WAVE_SIZE, fx=fx)
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0), fx.get(thr_coord, 1), fx.get(thr_coord, 2), fx.get(thr_coord, 3)
        )
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)
        blk_n = bx * arith.index(int(tile_n))

        base_ptr = alloc.get_base()
        smem_bg = SmemPtr(base_ptr, off_bg, T.f16, shape=(lds_b_elems,))
        smem_bu = SmemPtr(base_ptr, off_bu, T.f16, shape=(lds_b_elems,))
        smem_a = SmemPtr(base_ptr, off_a, T.f16, shape=(lds_a_elems,))
        lds_bg = get_op_result_or_value(smem_bg.get())
        lds_bu = get_op_result_or_value(smem_bu.get())
        lds_a = get_op_result_or_value(smem_a.get())

        def silu(x):
            t = x * (-1.4426950408889634)
            emu = rocdl.exp2(T.f32, t)
            den = 1.0 + emu
            sig = rocdl.rcp(T.f32, den)
            return x * sig

        def pack_a_to_lds(k_base):
            total = int(tile_m * tile_k)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(T.i32, elem),
                    arith.constant(total, type=T.i32),
                )
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    row = elem // arith.index(int(tile_k))
                    col = elem % arith.index(int(tile_k))
                    sorted_row = by * arith.index(int(tile_m)) + row
                    row_in_route = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        arith.index_cast(T.i32, row),
                        arith.constant(int(route_tile_m), type=T.i32),
                    )
                    sorted_row_safe = arith.select(
                        row_in_route,
                        arith.index_cast(T.i32, sorted_row),
                        arith.index_cast(T.i32, by * arith.index(int(route_tile_m))),
                    )
                    fused = buffer_ops.buffer_load(sorted_rsrc, sorted_row_safe, vec_width=1, dtype=T.i32)
                    tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                    tok_ok0 = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                    tok_ok = arith.andi(row_in_route, tok_ok0)
                    x_idx = tok * arith.constant(int(model_dim), type=T.i32) + arith.index_cast(T.i32, k_base + col)
                    x_idx_safe = arith.select(tok_ok, x_idx, arith.constant(0, type=T.i32))
                    x_val = arith.select(
                        tok_ok,
                        buffer_ops.buffer_load(x_rsrc, x_idx_safe, vec_width=1, dtype=T.f16),
                        arith.constant(0.0, type=T.f16),
                    )
                    lds_idx = row * arith.index(lds_a_stride) + col
                    v1 = vector.from_elements(T.vec(1, T.f16), [x_val])
                    vector.store(v1, lds_a, [lds_idx], alignment=2)
                    scf.YieldOp([])

        def copy_b_to_lds(k_base, lds_memref, up_shift):
            eid_idx = arith.index_cast(T.index, eid_i32)
            n_base = eid_idx * arith.index(n_total) + blk_n + arith.index(up_shift)
            total = int(tile_k) * int(tile_n)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(T.i32, elem),
                    arith.constant(total, type=T.i32),
                )
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    k_local = elem // arith.index(int(tile_n))
                    n_local = elem % arith.index(int(tile_n))
                    w_idx = (n_base + n_local) * arith.index(int(model_dim)) + k_base + k_local
                    w_val = buffer_ops.buffer_load(
                        w_rsrc, arith.index_cast(T.i32, w_idx),
                        vec_width=1, dtype=T.f16,
                    )
                    lds_idx = k_local * arith.index(lds_b_stride) + n_local
                    v1 = vector.from_elements(T.vec(1, T.f16), [w_val])
                    vector.store(v1, lds_memref, [lds_idx], alignment=2)
                    scf.YieldOp([])

        def _precompute_a_lane_bases():
            row_stride_off = (warp_m_base + lane16) * arith.index(lds_a_stride)
            k_lane_off = lane_kgrp * arith.index(8)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                a_base = row_stride_off + arith.index(wm * WMMA_M * lds_a_stride) + k_lane_off
                bases.append(a_base)
            return bases

        def _precompute_b_lane_bases():
            lane8 = lane16 % arith.index(8)
            lane_ngrp = lane16 / arith.index(8)
            k_lane_off = (lane_kgrp * arith.index(8) + lane8) * arith.index(lds_b_stride)
            n_lane_off = lane_ngrp * arith.index(8)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                n_col = warp_n_base + arith.index(wn * WMMA_N) + n_lane_off
                bases.append(k_lane_off + n_col)
            return bases

        def load_a_frag(a_base, ks):
            vec8_ty = ir.VectorType.get([8], T.f16)
            off0 = a_base + arith.index(ks * WMMA_K)
            off1 = a_base + arith.index(ks * WMMA_K + 16)
            v0 = vector.load_op(vec8_ty, lds_a, [off0])
            v1 = vector.load_op(vec8_ty, lds_a, [off1])
            return vector.shuffle(v0, v1, list(range(16)))

        def load_b_frag(lds_buf, b_base, ks):
            vec8_ty = ir.VectorType.get([8], T.f16)
            results = []
            for k_half in range_constexpr(2):
                k_row_off = (ks * WMMA_K + k_half * 16) * lds_b_stride
                elem_off = b_base + arith.index(k_row_off)
                v = rocdl.lds_transpose_load(vec8_ty, lds_buf, elem_off, elem_bytes)
                results.append(v)
            return vector.shuffle(results[0], results[1], list(range(16)))

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        acc_gate = [acc_zero] * n_accs
        acc_up = [acc_zero] * n_accs

        _if_eid = scf.IfOp(eid_ok)
        with ir.InsertionPoint(_if_eid.then_block):
            a_bases = _precompute_a_lane_bases()
            b_bases = _precompute_b_lane_bases()
            for kt in range_constexpr(num_k_tiles):
                k_base = fx.Index(kt * int(tile_k))
                pack_a_to_lds(k_base)
                copy_b_to_lds(k_base, lds_bg, 0)
                copy_b_to_lds(k_base, lds_bu, int(inter_dim))
                gpu.barrier()

                for ks in range_constexpr(k_wmma_steps):
                    b_gate_frags = [load_b_frag(lds_bg, b_bases[wn], ks) for wn in range_constexpr(wmma_n_rep)]
                    b_up_frags = [load_b_frag(lds_bu, b_bases[wn], ks) for wn in range_constexpr(wmma_n_rep)]
                    for wm in range_constexpr(wmma_m_rep):
                        a_frag = load_a_frag(a_bases[wm], ks)
                        for wn in range_constexpr(wmma_n_rep):
                            idx = wm * wmma_n_rep + wn
                            acc_gate[idx] = rocdl.wmma_f32_16x16x32_f16(
                                T.vec(8, T.f32),
                                b_gate_frags[wn],
                                a_frag,
                                acc_gate[idx],
                                signA=False,
                                signB=False,
                                modC=0,
                                reuseA=False,
                                reuseB=False,
                            ).result
                            acc_up[idx] = rocdl.wmma_f32_16x16x32_f16(
                                T.vec(8, T.f32),
                                b_up_frags[wn],
                                a_frag,
                                acc_up[idx],
                                signA=False,
                                signB=False,
                                modC=0,
                                reuseA=False,
                                reuseB=False,
                            ).result
                gpu.barrier()

            out_elem_ty = _moe_out_elem_ty(out_dtype, T)

            def _load_gate_up_sub8(acc_idx, _vec_base):
                return acc_gate[acc_idx], acc_up[acc_idx]

            _emit_stage1_gate_up_epilogue(
                sub_tiles=_sub_tiles,
                by=by,
                tile_m=int(tile_m),
                route_tile_m=int(route_tile_m),
                warp_m_base=warp_m_base,
                warp_n_base=warp_n_base,
                blk_n=blk_n,
                lane16=lane16,
                lane_kgrp=lane_kgrp,
                WMMA_N=WMMA_N,
                i32_tokens_in=i32_tokens_in,
                i32_inter_in=i32_inter_in,
                topk=int(topk),
                sorted_rsrc=sorted_rsrc,
                tw_rsrc=sw_rsrc,
                out_rsrc=out_rsrc,
                doweight_stage1=bool(doweight_stage1),
                out_elem_ty=out_elem_ty,
                load_gate_up_sub8=_load_gate_up_sub8,
                silu_fn=silu,
                ir=ir,
                fx=fx,
                arith=arith,
                buffer_ops=buffer_ops,
                scf=scf,
                vector=vector,
                range_constexpr=range_constexpr,
                T=T,
            )
            scf.YieldOp([])

    @flyc.jit
    def launch_fp16_stage1_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = i32_k_in
        ctx = CompilationContext.get_current()
        inter_in = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = (inter_in + fx.Index(int(tile_n) - 1)) // fx.Index(int(tile_n))
        gy = size_expert_ids_in
        launcher = moe_fp16_stage1_single(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        )
        _finalize_alloc_and_launch_2d(
            ctx=ctx,
            alloc=alloc,
            launcher=launcher,
            gx=gx,
            gy=gy,
            block_threads=block_threads,
            stream=stream,
            waves_per_eu=waves_per_eu,
            ir=ir,
        )

    if expert_sched_mode:
        launch_fp16_stage1_single.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

    return launch_fp16_stage1_single


@functools.lru_cache(maxsize=64)
def _compile_stage2_wmma_kernel_impl(
    *,
    inter_dim: int,
    experts: int,
    topk: int,
    route_tile_m: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    doweight_stage2: bool,
    out_dtype: str,
    accumulate: bool,
    waves_per_eu: int | None,
    expert_sched_mode: bool = True,
):
    """Compile fp16 stage2 single kernel: route-pack + TDM + WMMA + epilog."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as llvm_dialect
    from flydsl._mlir.dialects import scf
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, gpu, idx2crd, range_constexpr, rocdl, tdm_ops, vector
    from flydsl.expr.typing import T
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

    WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
    WAVE_SIZE = 32
    LDS_PAD_A = 8
    LDS_PAD_B = 8
    elem_bytes = 2

    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"fp16 stage2 single kernel supports out_dtype in ('f16','bf16'), got {out_dtype!r}")
    if (int(inter_dim) % int(tile_k)) != 0:
        raise ValueError(f"inter_dim={inter_dim} must be divisible by tile_k={tile_k}")
    if (int(tile_k) % WMMA_K) != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by {WMMA_K}")

    block_threads = int(m_warp) * int(n_warp) * WAVE_SIZE
    warp_tile_m = int(tile_m) // int(m_warp)
    warp_tile_n = int(tile_n) // int(n_warp)
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    if wmma_m_rep <= 0 or wmma_n_rep <= 0:
        raise ValueError(f"Invalid warp tiling for fp16 stage2 single kernel: wmma_m_rep={wmma_m_rep}, wmma_n_rep={wmma_n_rep}")

    n_accs = wmma_m_rep * wmma_n_rep
    num_k_tiles = int(inter_dim) // int(tile_k)
    k_wmma_steps = int(tile_k) // WMMA_K
    _sub_tiles = _make_wmma_sub_tiles(
        wmma_m_rep=wmma_m_rep, wmma_n_rep=wmma_n_rep, WMMA_M=WMMA_M, is_fp4=False
    )

    lds_a_stride = int(tile_k) + LDS_PAD_A
    lds_b_stride = int(tile_n) + LDS_PAD_B
    lds_a_elems = int(tile_m) * lds_a_stride + LDS_PAD_A
    lds_b_elems = int(tile_k) * lds_b_stride + LDS_PAD_B

    alloc = SmemAllocator(None, arch=str(get_hip_arch()), global_sym_name="moe_fp16_s2_single")
    off_b = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_b + lds_b_elems * elem_bytes
    off_a = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_a + lds_a_elems * elem_bytes

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def moe_fp16_stage2_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        _ = (arg_scale_x, arg_scale_w, i32_k_in)
        llvm_dialect.inline_asm(
            None, [],  # void result, no operands
            "s_setreg_imm32_b32 hwreg(26, 4, 1), 1",
            "",
            has_side_effects=True,
        )

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")  # n tile
        by = gpu.block_id("y")  # expert block

        tokens_idx = arith.index_cast(T.index, i32_tokens_in)
        n_idx = arith.index_cast(T.index, i32_n_in)
        size_expert_ids = arith.index_cast(T.index, i32_size_expert_ids_in)
        num_valid_i32 = buffer_ops.buffer_load(
            buffer_ops.create_buffer_resource(arg_num_valid_ids, max_size=True),
            arith.constant(0, type=T.i32),
            vec_width=1,
            dtype=T.i32,
        )

        sorted_num = size_expert_ids * arith.index(int(route_tile_m))
        sorted_nbytes = sorted_num * arith.index(4)
        eid_nbytes = size_expert_ids * arith.index(4)
        x_rows = tokens_idx * arith.index(int(topk))
        x_nbytes = x_rows * arith.index(int(inter_dim)) * arith.index(2)
        out_nbytes = tokens_idx * n_idx * arith.index(2)
        if not bool(accumulate):
            out_nbytes = x_rows * n_idx * arith.index(2)

        sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False, num_records_bytes=sorted_nbytes)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_nbytes)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=False, num_records_bytes=out_nbytes)
        sw_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=True)

        eid_i32 = buffer_ops.buffer_load(eid_rsrc, arith.index_cast(T.i32, by), vec_width=1, dtype=T.i32)
        eid_ok0 = arith.cmpi(arith.CmpIPredicate.sge, eid_i32, arith.constant(0, type=T.i32))
        eid_ok1 = arith.cmpi(arith.CmpIPredicate.slt, eid_i32, arith.constant(int(experts), type=T.i32))
        block_row_start = arith.index_cast(T.i32, by * arith.index(int(route_tile_m)))
        block_in_valid = arith.cmpi(arith.CmpIPredicate.slt, block_row_start, num_valid_i32)
        block_ok = arith.andi(block_in_valid, arith.andi(eid_ok0, eid_ok1))

        layout_thr = _make_moe_wave_layout(m_warp=m_warp, n_warp=n_warp, WAVE_SIZE=WAVE_SIZE, fx=fx)
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0), fx.get(thr_coord, 1), fx.get(thr_coord, 2), fx.get(thr_coord, 3)
        )
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)
        blk_n = bx * arith.index(int(tile_n))

        base_ptr = alloc.get_base()
        smem_b = SmemPtr(base_ptr, off_b, T.f16, shape=(lds_b_elems,))
        smem_a = SmemPtr(base_ptr, off_a, T.f16, shape=(lds_a_elems,))
        lds_b = get_op_result_or_value(smem_b.get())
        lds_a = get_op_result_or_value(smem_a.get())

        def pack_a_to_lds(k_base):
            total = int(tile_m * tile_k)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(T.i32, elem),
                    arith.constant(total, type=T.i32),
                )
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    row = elem // arith.index(int(tile_k))
                    col = elem % arith.index(int(tile_k))
                    sorted_row = by * arith.index(int(tile_m)) + row
                    row_i32 = arith.index_cast(T.i32, row)
                    sorted_i32 = arith.index_cast(T.i32, sorted_row)
                    row_in_route = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        row_i32,
                        arith.constant(int(route_tile_m), type=T.i32),
                    )
                    row_in_valid = arith.cmpi(arith.CmpIPredicate.slt, sorted_i32, num_valid_i32)
                    row_ok = arith.andi(row_in_route, row_in_valid)
                    sorted_safe = arith.select(row_ok, sorted_i32, block_row_start)
                    fused = buffer_ops.buffer_load(sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32)
                    tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                    slot = fused >> arith.constant(24, type=T.i32)
                    tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                    slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, slot, arith.constant(0, type=T.i32))
                    slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, slot, arith.constant(int(topk), type=T.i32))
                    ts = tok * arith.constant(int(topk), type=T.i32) + slot
                    ts_ok = arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1))
                    load_ok = arith.andi(row_ok, ts_ok)
                    x_idx = ts * arith.constant(int(inter_dim), type=T.i32) + arith.index_cast(T.i32, k_base + col)
                    x_idx_safe = arith.select(load_ok, x_idx, arith.constant(0, type=T.i32))
                    x_val = arith.select(
                        load_ok,
                        buffer_ops.buffer_load(x_rsrc, x_idx_safe, vec_width=1, dtype=T.f16),
                        arith.constant(0.0, type=T.f16),
                    )
                    lds_idx = row * arith.index(lds_a_stride) + col
                    v1 = vector.from_elements(T.vec(1, T.f16), [x_val])
                    vector.store(v1, lds_a, [lds_idx], alignment=2)
                    scf.YieldOp([])

        def copy_b_to_lds(k_base):
            eid_idx = arith.index_cast(T.index, eid_i32)
            n_base = eid_idx * n_idx + blk_n
            total = int(tile_k) * int(tile_n)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(T.i32, elem),
                    arith.constant(total, type=T.i32),
                )
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    k_local = elem // arith.index(int(tile_n))
                    n_local = elem % arith.index(int(tile_n))
                    w_idx = (n_base + n_local) * arith.index(int(inter_dim)) + k_base + k_local
                    w_val = buffer_ops.buffer_load(
                        w_rsrc, arith.index_cast(T.i32, w_idx),
                        vec_width=1, dtype=T.f16,
                    )
                    lds_idx = k_local * arith.index(lds_b_stride) + n_local
                    v1 = vector.from_elements(T.vec(1, T.f16), [w_val])
                    vector.store(v1, lds_b, [lds_idx], alignment=2)
                    scf.YieldOp([])

        def _precompute_a_lane_bases():
            row_stride_off = (warp_m_base + lane16) * arith.index(lds_a_stride)
            k_lane_off = lane_kgrp * arith.index(8)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                a_base = row_stride_off + arith.index(wm * WMMA_M * lds_a_stride) + k_lane_off
                bases.append(a_base)
            return bases

        def _precompute_b_lane_bases():
            lane8 = lane16 % arith.index(8)
            lane_ngrp = lane16 / arith.index(8)
            k_lane_off = (lane_kgrp * arith.index(8) + lane8) * arith.index(lds_b_stride)
            n_lane_off = lane_ngrp * arith.index(8)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                n_col = warp_n_base + arith.index(wn * WMMA_N) + n_lane_off
                bases.append(k_lane_off + n_col)
            return bases

        def load_a_frag(a_base, ks):
            vec8_ty = ir.VectorType.get([8], T.f16)
            off0 = a_base + arith.index(ks * WMMA_K)
            off1 = a_base + arith.index(ks * WMMA_K + 16)
            v0 = vector.load_op(vec8_ty, lds_a, [off0])
            v1 = vector.load_op(vec8_ty, lds_a, [off1])
            return vector.shuffle(v0, v1, list(range(16)))

        def load_b_frag(b_base, ks):
            vec8_ty = ir.VectorType.get([8], T.f16)
            results = []
            for k_half in range_constexpr(2):
                k_row_off = (ks * WMMA_K + k_half * 16) * lds_b_stride
                elem_off = b_base + arith.index(k_row_off)
                v = rocdl.lds_transpose_load(vec8_ty, lds_b, elem_off, elem_bytes)
                results.append(v)
            return vector.shuffle(results[0], results[1], list(range(16)))

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        acc = [acc_zero] * n_accs

        _if_blk = scf.IfOp(block_ok)
        with ir.InsertionPoint(_if_blk.then_block):
            a_bases = _precompute_a_lane_bases()
            b_bases = _precompute_b_lane_bases()

            for kt in range_constexpr(num_k_tiles):
                k_base = fx.Index(kt * int(tile_k))
                pack_a_to_lds(k_base)
                copy_b_to_lds(k_base)
                gpu.barrier()

                for ks in range_constexpr(k_wmma_steps):
                    b_frags = [load_b_frag(b_bases[wn], ks) for wn in range_constexpr(wmma_n_rep)]
                    for wm in range_constexpr(wmma_m_rep):
                        a_frag = load_a_frag(a_bases[wm], ks)
                        for wn in range_constexpr(wmma_n_rep):
                            idx = wm * wmma_n_rep + wn
                            acc[idx] = rocdl.wmma_f32_16x16x32_f16(
                                T.vec(8, T.f32),
                                b_frags[wn],
                                a_frag,
                                acc[idx],
                                signA=False,
                                signB=False,
                                modC=0,
                                reuseA=False,
                                reuseB=False,
                            ).result
                gpu.barrier()

            out_elem_ty = _moe_out_elem_ty(out_dtype, T)

            def _load_sub8(acc_idx, _vec_base):
                return acc[acc_idx]

            _emit_stage2_store_epilogue(
                sub_tiles=_sub_tiles,
                by=by,
                tile_m=int(tile_m),
                route_tile_m=int(route_tile_m),
                warp_m_base=warp_m_base,
                warp_n_base=warp_n_base,
                blk_n=blk_n,
                lane16=lane16,
                lane_kgrp=lane_kgrp,
                WMMA_N=WMMA_N,
                i32_tokens_in=i32_tokens_in,
                i32_n_in=i32_n_in,
                topk=int(topk),
                num_valid_i32=num_valid_i32,
                block_row_start=block_row_start,
                sorted_rsrc=sorted_rsrc,
                tw_rsrc=sw_rsrc,
                out_rsrc=out_rsrc,
                doweight_stage2=bool(doweight_stage2),
                accumulate=bool(accumulate),
                out_elem_ty=out_elem_ty,
                load_sub8=_load_sub8,
                ir=ir,
                fx=fx,
                arith=arith,
                buffer_ops=buffer_ops,
                scf=scf,
                vector=vector,
                range_constexpr=range_constexpr,
                rocdl=rocdl,
                T=T,
            )
            scf.YieldOp([])

    @flyc.jit
    def launch_fp16_stage2_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = i32_k_in
        ctx = CompilationContext.get_current()
        n_in = arith.index_cast(T.index, i32_n_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = (n_in + fx.Index(int(tile_n) - 1)) // fx.Index(int(tile_n))
        gy = size_expert_ids_in
        launcher = moe_fp16_stage2_single(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
            i32_size_expert_ids_in,
        )
        _finalize_alloc_and_launch_2d(
            ctx=ctx,
            alloc=alloc,
            launcher=launcher,
            gx=gx,
            gy=gy,
            block_threads=block_threads,
            stream=stream,
            waves_per_eu=waves_per_eu,
            ir=ir,
        )

    if expert_sched_mode:
        launch_fp16_stage2_single.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

    return launch_fp16_stage2_single


# ---------------------------------------------------------------------------
# Public API entry points for fp16/bf16
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1024)
def _compile_moe_wmma_gemm(
    *,
    stage: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight: bool,
    in_dtype: str = "fp16",
    out_dtype: str = "f16",
    accumulate: bool = True,
    waves_per_eu: int | None = None,
    expert_sched_mode: bool = True,
):
    _require_gfx1250()
    if waves_per_eu is not None and int(waves_per_eu) < 1:
        raise ValueError(f"waves_per_eu must be >= 1, got {waves_per_eu!r}")
    if in_dtype not in ("fp16", "bf16"):
        raise ValueError(
            f"Unsupported in_dtype for WMMA stage{stage}: {in_dtype!r}, "
            "expected 'fp16' or 'bf16'"
        )

    single_tile_m, single_tile_n, single_m_warp, single_n_warp = _pick_fp16_single_launch_shape(
        int(tile_m), int(tile_n), max_total_warps=8,
    )
    common = dict(
        inter_dim=int(inter_dim),
        experts=int(experts),
        topk=int(topk),
        route_tile_m=int(tile_m),
        tile_m=int(single_tile_m),
        tile_n=int(single_tile_n),
        tile_k=int(tile_k),
        m_warp=int(single_m_warp),
        n_warp=int(single_n_warp),
        out_dtype=out_dtype,
        waves_per_eu=waves_per_eu,
        expert_sched_mode=expert_sched_mode,
    )

    if stage == 1:
        exe = _compile_stage1_wmma_kernel_impl(
            model_dim=int(model_dim), doweight_stage1=bool(doweight), **common,
        )
    else:
        exe = _compile_stage2_wmma_kernel_impl(
            doweight_stage2=bool(doweight), accumulate=bool(accumulate), **common,
        )

    if in_dtype == "bf16":
        return _bf16_to_f16_wrapper(exe, x_arg=1, w_arg=2)
    return exe


def compile_moe_gemm1(*, doweight_stage1, group_size=-1, use_cshuffle_epilog=None,
                       num_buffers=1, use_tdm_gather=True, use_tdm_store=False,
                       inst_prefetch=False, wave_specialized_tdm=False,
                       cluster_m=1, cluster_n=1, **kw):
    return _compile_moe_wmma_gemm(stage=1, doweight=doweight_stage1, **kw)


def compile_moe_gemm2(*, doweight_stage2, accumulate=True, group_size=-1,
                       use_cshuffle_epilog=None,
                       num_buffers=1, use_tdm_gather=True, use_tdm_store=False,
                       inst_prefetch=False, wave_specialized_tdm=False,
                       cluster_m=1, cluster_n=1, **kw):
    return _compile_moe_wmma_gemm(stage=2, doweight=doweight_stage2, accumulate=accumulate, **kw)


def compile_moe_gemm2_ex(*, mode=MoeGemm2Mode.ATOMIC, valid_mask=None, zero_intermediate=True, **kw):
    if mode == MoeGemm2Mode.REDUCE:
        gemm2_exe = compile_moe_gemm2(accumulate=False, **kw)
        out_s = str(kw.get("out_dtype", "f16")).strip().lower()
        if out_s in ("f16", "fp16", "half"):
            dtype_str = "f16"
        elif out_s in ("bf16", "bfloat16"):
            dtype_str = "bf16"
        else:
            dtype_str = "f32"
        reduce_exe = compile_moe_reduction(
            topk=kw["topk"], model_dim=kw["model_dim"],
            dtype_str=dtype_str, use_mask=(valid_mask is not None),
        )
        from kernels.moe_gemm_2stage import _MoeGemm2ReduceWrapper
        return _MoeGemm2ReduceWrapper(
            gemm2_exe=gemm2_exe, reduce_exe=reduce_exe,
            topk=kw["topk"], model_dim=kw["model_dim"],
            out_dtype_str=dtype_str,
            use_mask=(valid_mask is not None),
            zero_intermediate=zero_intermediate,
        )
    return compile_moe_gemm2(accumulate=True, **kw)
