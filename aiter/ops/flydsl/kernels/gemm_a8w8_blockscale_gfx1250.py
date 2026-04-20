# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A8W8 FP8 blockscale GEMM for gfx1250. Y = (X @ W^T) with per-K-block f32 scales."""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    gpu,
    idx2crd,
    range_constexpr,
    rocdl,
    tdm_ops,
    vector,
)
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_mlir_type_size

WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
FRAG_VGPRS = 16
DS_LOADS_PER_FRAG = 4
LDS_PAD_A_BYTES = 16
LDS_PAD_B_BYTES = 16
# Packed E8M0 = 127 per byte → 2^0 = 1.0; makes WMMA scale op a no-op.
E8M0_IDENTITY = 0x7F7F7F7F
_STAGE_NAMES = ("ping", "pong", "pang", "pung")


def _lds_vec_type(memref, total_bits):
    raw_mr = arith.unwrap(memref)
    elem_type = ir.MemRefType(raw_mr.type).element_type
    elem_bits = get_mlir_type_size(elem_type) * 8
    n = total_bits // elem_bits
    return ir.VectorType.get([n], elem_type)


def lds_load_b128(memref, elem_off):
    """ds_load_b128: load 16 bytes from LDS as vector<4xi32>."""
    vec_ty = _lds_vec_type(memref, 128)
    loaded = vector.load_op(vec_ty, memref, [elem_off])
    return vector.bitcast(
        ir.VectorType.get([4], ir.IntegerType.get_signless(32)), loaded
    )


def pipeline_fence(outstanding=0, use_cluster=False):
    """s_wait_tensorcnt + barrier for the N-stage pipeline."""
    tdm_ops.tensor_wait(outstanding)
    if use_cluster:
        gpu.cluster_barrier()
    else:
        gpu.barrier()


def store_acc_vec8_to_buffer(
    acc_vec8, c_rsrc, addr, out_elem=None, offset_is_bytes=False
):
    """Write a vec<8xf32> accumulator to global via buffer_store."""
    if out_elem is not None:
        h_vec = arith.trunc_f(T.vec(8, out_elem), acc_vec8)
        i32_vec = vector.bitcast(T.vec(4, T.i32), h_vec)
        buffer_ops.buffer_store(i32_vec, c_rsrc, addr, offset_is_bytes=offset_is_bytes)
        return 1
    for half in range(2):
        vals = [
            vector.extract(
                acc_vec8, static_position=[half * 4 + vi], dynamic_position=[]
            )
            for vi in range(4)
        ]
        vec4 = vector.from_elements(T.vec(4, T.f32), vals)
        if isinstance(addr, (list, tuple)):
            buffer_ops.buffer_store(vec4, c_rsrc, addr[half])
        else:
            buffer_ops.buffer_store(vec4, c_rsrc, addr)
    return 2


def make_tail_plan(num_buffers, pre_loaded, extra):
    """Return a list of (load_stage, compute_stage, outstanding) tuples for the tail.
    outstanding=-1 on the last step → no barrier follows."""
    steps = pre_loaded + extra
    plan = []
    for i in range(steps):
        compute_stage = (
            i if i < pre_loaded else (i - pre_loaded + num_buffers - 1) % num_buffers
        )
        load_stage = (i + num_buffers - 1) % num_buffers if i < extra else None
        is_last = i == steps - 1
        if is_last:
            outstanding = -1
        else:
            j = i + 1
            next_compute = (
                j
                if j < pre_loaded
                else (j - pre_loaded + num_buffers - 1) % num_buffers
            )
            outstanding = (
                2 * (num_buffers - 2)
                if (load_stage is not None and load_stage != next_compute)
                else 0
            )
        plan.append((load_stage, compute_stage, outstanding))
    return plan


def compile_gemm_a8w8_blockscale(
    *,
    K: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 4,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 0,
    out_dtype: str = "bf16",
):
    if out_dtype not in ("bf16", "fp16", "f32"):
        raise ValueError(
            f"out_dtype must be 'bf16', 'fp16', or 'f32', got {out_dtype!r}"
        )
    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3, or 4, got {num_buffers}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_k != scale_block_k:
        raise ValueError(
            f"tile_k ({tile_k}) must equal scale_block_k ({scale_block_k})"
        )
    if K % tile_k != 0:
        raise ValueError(f"K ({K}) must be divisible by tile_k ({tile_k})")
    if K % scale_block_k != 0:
        raise ValueError(
            f"K ({K}) must be divisible by scale_block_k ({scale_block_k})"
        )

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N}")
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep
    k_wmma_steps = tile_k // WMMA_K

    num_k_tiles = K // tile_k
    if num_k_tiles < num_buffers - 1:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers - 1}, "
            f"got {num_k_tiles}"
        )

    scale_k = K // scale_block_k

    gpu_arch = str(get_hip_arch(timeout_s=300))
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    elem_bytes_d = 2 if out_dtype in ("bf16", "fp16") else 4
    effective_waves_per_eu = waves_per_eu

    lds_a_stride_bytes = tile_k + LDS_PAD_A_BYTES
    lds_b_stride_bytes = tile_k + LDS_PAD_B_BYTES
    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = tile_n * lds_b_stride_bytes

    stage_allocators = []
    stage_a_data_off = []
    stage_b_data_off = []
    for i in range(num_buffers):
        alloc = SmemAllocator(
            None, arch=gpu_arch, global_sym_name=f"a8w8bs_{_STAGE_NAMES[i]}"
        )
        off = alloc._align(alloc.ptr, 16)
        stage_a_data_off.append(off)
        alloc.ptr = off + lds_a_data_bytes
        off = alloc._align(alloc.ptr, 16)
        stage_b_data_off.append(off)
        alloc.ptr = off + lds_b_data_bytes
        stage_allocators.append(alloc)

    pre_loaded = num_buffers - 1
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers
    extra = num_k_tiles - _tail_start - pre_loaded
    _raw_tail_plan = make_tail_plan(num_buffers, pre_loaded, extra)
    TDM_LOADS_PER_STEP = 2
    tail_plan = [
        (ls, cs, o * TDM_LOADS_PER_STEP // 2 if o > 0 else o)
        for ls, cs, o in _raw_tail_plan
    ]

    @flyc.kernel
    def kernel_gemm_a8w8_blockscale(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_x_scale: fx.Tensor,
        arg_w_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16), (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1)
        )
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx = fx.get(thr_coord, 0)
        wave_n_idx = fx.get(thr_coord, 1)
        lane_kgrp = fx.get(thr_coord, 2)
        lane16 = fx.get(thr_coord, 3)

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        n_idx = arith.index_cast(T.index, i32_n.ir_value())
        n_stride = n_idx

        y_total_bytes = m_idx * n_stride * arith.index(elem_bytes_d)
        y_buf = buffer_ops.create_buffer_resource(
            arg_y, num_records_bytes=y_total_bytes
        )

        x_scale_total_bytes = m_idx * arith.index(scale_k) * arith.index(4)
        x_scale_buf = buffer_ops.create_buffer_resource(
            arg_x_scale, num_records_bytes=x_scale_total_bytes
        )

        num_n_scale_blocks = (n_idx + arith.index(scale_block_n - 1)) / arith.index(
            scale_block_n
        )
        w_scale_total_bytes = num_n_scale_blocks * arith.index(scale_k) * arith.index(4)
        w_scale_buf = buffer_ops.create_buffer_resource(
            arg_w_scale, num_records_bytes=w_scale_total_bytes
        )

        identity_scale = arith.constant(E8M0_IDENTITY, type=T.i32)

        stages_a = [
            SmemPtr(
                stage_allocators[i].get_base(),
                stage_a_data_off[i],
                T.f8,
                shape=(lds_a_data_bytes,),
            )
            for i in range(num_buffers)
        ]
        stages_b = [
            SmemPtr(
                stage_allocators[i].get_base(),
                stage_b_data_off[i],
                T.f8,
                shape=(lds_b_data_bytes,),
            )
            for i in range(num_buffers)
        ]
        stages_a_mem = [stages_a[i].get() for i in range(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range(num_buffers)]

        # Build descriptor once per buffer at k_base=0; advance per iter.
        _k_zero = arith.index(0)
        a_desc_bases = [
            tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_x,
                lds_memref=stages_a_mem[i],
                global_offset=(blk_m, _k_zero),
                tensor_shape=(tile_m, tile_k),
                strides=(K, 1),
                tile_shape=(tile_m, tile_k),
                elem_bytes=1,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_A_BYTES,
                num_warps=num_warps,
            )
            for i in range(num_buffers)
        ]
        b_desc_bases = [
            tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w,
                lds_memref=stages_b_mem[i],
                global_offset=(blk_n, _k_zero),
                tensor_shape=(tile_n, tile_k),
                strides=(K, 1),
                tile_shape=(tile_n, tile_k),
                elem_bytes=1,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_B_BYTES,
                num_warps=num_warps,
            )
            for i in range(num_buffers)
        ]

        def copy_a_to_lds(k_base, buffer_idx):
            desc = tdm_ops.advance_tdm_descriptor(a_desc_bases[buffer_idx], k_base)
            tdm_ops.tensor_load_2d(desc)

        def copy_b_to_lds(k_base, buffer_idx):
            desc = tdm_ops.advance_tdm_descriptor(b_desc_bases[buffer_idx], k_base)
            tdm_ops.tensor_load_2d(desc)

        def issue_all_tdm_loads(k_base, buffer_idx):
            rocdl.s_setprio(2)
            copy_a_to_lds(k_base, buffer_idx)
            copy_b_to_lds(k_base, buffer_idx)
            rocdl.s_setprio(0)

        # Pre-combine x_scale * w_scale per (wm, wn) to save a multiply per tile.
        def load_and_combine_scales(k_base):
            kb = k_base / arith.index(scale_block_k)

            x_scales = []
            for wm in range_constexpr(wmma_m_rep):
                row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                idx = row * arith.index(scale_k) + kb
                x_scales.append(
                    buffer_ops.buffer_load(x_scale_buf, idx, vec_width=1, dtype=T.f32)
                )

            w_scales = []
            for wn in range_constexpr(wmma_n_rep):
                col = (
                    blk_n
                    + warp_n_base
                    + arith.index(wn * WMMA_N)
                    + lane_kgrp * arith.index(8)
                )
                n_block = col / arith.index(scale_block_n)
                idx = n_block * arith.index(scale_k) + kb
                w_scales.append(
                    buffer_ops.buffer_load(w_scale_buf, idx, vec_width=1, dtype=T.f32)
                )

            combined = []
            for wm in range_constexpr(wmma_m_rep):
                row_combined = []
                for wn in range_constexpr(wmma_n_rep):
                    row_combined.append(arith.mulf(x_scales[wm], w_scales[wn]))
                combined.append(row_combined)
            return combined

        # lane_kgrp selects K-half: kgrp=0 → bytes [0..63], kgrp=1 → [64..127].
        _k_half_off = lane_kgrp * arith.index(64)

        def _precompute_lane_bases(warp_base, stride_bytes, num_reps, rep_stride_elems):
            row_base_bytes = (warp_base + lane16) * arith.index(stride_bytes)
            bases = []
            for rep in range_constexpr(num_reps):
                base = (
                    row_base_bytes
                    + arith.index(rep * rep_stride_elems * stride_bytes)
                    + _k_half_off
                )
                bases.append(base)
            return bases

        def _load_frag(lds_memref, lane_base, ks):
            k_sub_off = arith.index(ks * WMMA_K)
            off = lane_base + k_sub_off
            v0 = lds_load_b128(lds_memref, off)
            v1 = lds_load_b128(lds_memref, off + arith.index(16))
            v2 = lds_load_b128(lds_memref, off + arith.index(32))
            v3 = lds_load_b128(lds_memref, off + arith.index(48))
            v01 = vector.shuffle(v0, v1, list(range(8)))
            v23 = vector.shuffle(v2, v3, list(range(8)))
            return vector.shuffle(v01, v23, list(range(16)))

        def precompute_a_lane_bases():
            return _precompute_lane_bases(
                warp_m_base, lds_a_stride_bytes, wmma_m_rep, WMMA_M
            )

        def precompute_b_lane_bases():
            return _precompute_lane_bases(
                warp_n_base, lds_b_stride_bytes, wmma_n_rep, WMMA_N
            )

        def load_a_frag(a_lds_memref, a_lane_base, ks=0):
            return _load_frag(a_lds_memref, a_lane_base, ks)

        def load_b_frag(b_lds_memref, b_lane_base, ks=0):
            return _load_frag(b_lds_memref, b_lane_base, ks)

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        # WMMA into a local acc, then scale and add into globals (per-tile
        # scales differ, so no direct accumulation across tiles).
        def compute_tile(global_accs, compute_stage, k_base):
            combined_scales = load_and_combine_scales(k_base)
            a_lds_memref = stages_a_mem[compute_stage]
            b_lds_memref = stages_b_mem[compute_stage]
            a_bases = precompute_a_lane_bases()
            b_bases = precompute_b_lane_bases()
            local_accs = [acc_zero] * n_accs

            for ks in range_constexpr(k_wmma_steps):
                a_frags = [
                    load_a_frag(a_lds_memref, a_bases[wm], ks)
                    for wm in range_constexpr(wmma_m_rep)
                ]
                b_frags = [
                    load_b_frag(b_lds_memref, b_bases[wn], ks)
                    for wn in range_constexpr(wmma_n_rep)
                ]
                rocdl.s_wait_dscnt(0)

                for wm in range_constexpr(wmma_m_rep):
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        # ISA operand order: (B, A, C), reversed from math.
                        local_accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                            T.vec(8, T.f32),
                            b_frags[wn],
                            a_frags[wm],
                            local_accs[idx],
                            identity_scale,
                            identity_scale,
                            fmtA=0,
                            fmtB=0,
                            scaleAType=0,
                            scaleBType=0,
                        )

            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    scale_vec = vector.broadcast(
                        T.vec(8, T.f32), combined_scales[wm][wn]
                    )
                    scaled = arith.mulf(local_accs[idx], scale_vec)
                    global_accs[idx] = arith.addf(global_accs[idx], scaled)

            return global_accs

        # Step 1: Prologue — issue TDM for the first pre_loaded tiles, fence.
        _main_outstanding = 2 * (num_buffers - 2)

        for i in range_constexpr(pre_loaded):
            issue_all_tdm_loads(arith.index(i * tile_k), i)
        pipeline_fence(outstanding=_main_outstanding, use_cluster=False)

        # Step 2: Main loop — per iter, issue future load + compute current.
        main_end = loop_iters * num_buffers * tile_k

        if loop_iters > 0:
            for iv, state in range(0, main_end, num_buffers * tile_k, init=list(accs)):
                rocdl.iglp_opt(1)
                accs_in = list(state)

                for s in range_constexpr(num_buffers):
                    load_buffer = (s + num_buffers - 1) % num_buffers
                    load_k_offset = (s + num_buffers - 1) * tile_k
                    issue_all_tdm_loads(iv + arith.index(load_k_offset), load_buffer)

                    compute_k_base = iv + arith.index(s * tile_k)
                    accs_in = compute_tile(accs_in, s, compute_k_base)

                    pipeline_fence(outstanding=_main_outstanding, use_cluster=False)

                results = yield list(accs_in)
            accs = [results] if n_accs == 1 else list(results)

        # Step 3: compute remaining in-flight tiles per tail_plan.
        for step_idx, (_load_buffer, _compute_buffer, _outstanding) in enumerate(
            tail_plan
        ):
            if _load_buffer is not None:
                tail_load_k = (_tail_start + pre_loaded + step_idx) * tile_k
                issue_all_tdm_loads(arith.index(tail_load_k), _load_buffer)

            compute_k_base = arith.index((_tail_start + step_idx) * tile_k)
            accs = compute_tile(accs, _compute_buffer, compute_k_base)

            if _outstanding != -1:
                pipeline_fence(outstanding=_outstanding, use_cluster=False)

        # Step 4: convert f32 accs to out_dtype, buffer_store to Y.
        _out_elem = (
            T.bf16 if out_dtype == "bf16" else T.f16 if out_dtype == "fp16" else None
        )
        _half_out = out_dtype in ("bf16", "fp16")

        for wm in range_constexpr(wmma_m_rep):
            for wn in range_constexpr(wmma_n_rep):
                idx = wm * wmma_n_rep + wn
                row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                col_base = (
                    blk_n
                    + warp_n_base
                    + arith.index(wn * WMMA_N)
                    + lane_kgrp * arith.index(8)
                )

                if _half_out:
                    c_off_bytes = (row * n_stride + col_base) * arith.index(
                        elem_bytes_d
                    )
                    store_acc_vec8_to_buffer(
                        accs[idx],
                        y_buf,
                        c_off_bytes,
                        out_elem=_out_elem,
                        offset_is_bytes=True,
                    )
                else:
                    offsets = []
                    for half in range_constexpr(2):
                        col = col_base + arith.index(half * 4)
                        offsets.append(row * n_stride + col)
                    store_acc_vec8_to_buffer(accs[idx], y_buf, offsets)

    cache_tag = (
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        scale_block_k,
        scale_block_n,
        num_buffers,
        effective_waves_per_eu,
        l2_prefetch_distance,
        out_dtype,
    )

    @flyc.jit
    def launch_gemm_a8w8_blockscale(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_x_scale: fx.Tensor,
        arg_w_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag

        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            for alloc in stage_allocators:
                alloc.finalized = False
            for alloc in stage_allocators:
                alloc.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))
        gy = _raw((idx_n + arith.index(tile_n - 1)) / arith.index(tile_n))

        launcher = kernel_gemm_a8w8_blockscale(
            arg_y, arg_x, arg_w, arg_x_scale, arg_w_scale, i32_m, i32_n
        )

        if effective_waves_per_eu is not None:
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    wpe = int(effective_waves_per_eu)
                    if wpe >= 1:
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), wpe
                        )

        launcher.launch(
            grid=(gx, gy, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_gemm_a8w8_blockscale


def gemm_a8w8_blockscale(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    y: torch.Tensor = None,
    dtype: torch.dtype = torch.bfloat16,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 4,
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 0,
):
    """Compute Y = (X @ W^T) with per-block f32 scales (A8W8 blockscale)."""
    assert x.ndim == 2 and w.ndim == 2, "X and W must be 2D"
    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w, f"K mismatch: X has {K}, W has {K_w}"

    assert x_scale.ndim == 2 and w_scale.ndim == 2, "scales must be 2D"
    assert x_scale.shape[0] == M, f"x_scale rows {x_scale.shape[0]} != M {M}"
    scale_k_x = x_scale.shape[1]
    scale_n, scale_k_w = w_scale.shape
    assert (
        scale_k_x == scale_k_w
    ), f"scale_k mismatch: x_scale has {scale_k_x}, w_scale has {scale_k_w}"
    scale_k = scale_k_x

    def _next_pow2(n):
        p = 1
        while p < n:
            p *= 2
        return p

    scale_block_k_derived = _next_pow2((K + scale_k - 1) // scale_k)
    scale_block_n_derived = _next_pow2((N + scale_n - 1) // scale_n)

    torch_to_str = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "f32",
    }
    assert dtype in torch_to_str, f"Unsupported output dtype {dtype}"
    out_dtype_str = torch_to_str[dtype]

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)
    else:
        assert y.shape == (M, N), f"y shape {y.shape} != ({M}, {N})"
        assert y.dtype == dtype, f"y dtype {y.dtype} != {dtype}"

    launcher = compile_gemm_a8w8_blockscale(
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        scale_block_k=scale_block_k_derived,
        scale_block_n=scale_block_n_derived,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        out_dtype=out_dtype_str,
    )

    stream = torch.cuda.current_stream(device=x.device).cuda_stream
    launcher(y, x, w, x_scale, w_scale, M, N, stream=stream)
    return y


__all__ = [
    "compile_gemm_a8w8_blockscale",
    "gemm_a8w8_blockscale",
]
