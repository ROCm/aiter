# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""W8A8 blockwise batched GEMM (OOB variant) for gfx1250.

Operation (E8M0 blockwise dequant, both operands fp8):
    C[m,b,n] = sum_k (A_fp8[m,b,k] * a_scale[m,b,k//gk])
                   * (B_fp8[b,n,k] * b_scale[b,n//gn,k//gk])

Layouts :
    layout="mbn":
        A       : [M, B, K]            fp8_e4m3fn
        a_scale : [M, B, K//gk]        uint8 E8M0
        C       : [M, B, N]            bf16 / f16 / f32
    layout="bmn":
        A       : [B, M, K]            fp8_e4m3fn
        a_scale : [B, M, K//gk]        uint8 E8M0
        C       : [B, M, N]            bf16 / f16 / f32
    B       : [B, N, K]            fp8_e4m3fn
    b_scale : [B, N//gn, K//gk]    uint8 E8M0

launch_fn(arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale, M, stream)
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, scf
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    idx2crd,
    range_constexpr,
    rocdl,
    tdm_ops,
)
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity

from .gemm_common_gfx1250 import (
    extract_lds_base_idx,
    get_lds_memref,
    issue_tdm_loads,
    lds_load_b128_raw,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
    store_acc_vec8_to_buffer,
    store_acc_vec8_to_lds,
)
from .pipeline_utils import make_tail_plan, tdm_epilogue_fence_threshold_bytes

WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
ACC_VEC_SIZE = 8
DS_LOADS_PER_A_FRAG = 4
DS_LOADS_PER_B_FRAG = 4

LDS_PAD_A_BYTES = 16
LDS_PAD_B_BYTES = 16
LDS_PAD_D_BYTES = 16

ELEM_BYTES_A = 1  # fp8
ELEM_BYTES_B = 1  # fp8
ELEM_BYTES_SCALE = 1  # E8M0 uint8

LDS_SEGMENT_BYTES = 64 * 1024
LDS_GFX1250_MAX_BYTES = 5 * LDS_SEGMENT_BYTES


@functools.lru_cache(maxsize=256)
def compile_bmm_w8a8_bpreshuffle_gfx1250(
    *,
    B: int = 16,
    M: int = 0,
    N: int = 1024,
    K: int = 4096,
    group_k: int = 128,
    group_n: int = 128,
    tile_m: int = 64,
    tile_n: int = 256,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 2,
    out_dtype: str = "bf16",
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 2,
    use_tdm_store: bool = True,
    expert_sched_mode: bool = True,
    inst_prefetch: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
    wave_specialized_tdm: bool = False,
    atomic_barrier_enable: bool = False,
    split_k: int = 1,
    preshuffle_b: bool = False,
    layout: str = "mbn",
):
    _ = M
    wmma_op = rocdl.wmma_scale_f32_16x16x128_f8f6f4

    if layout not in ("mbn", "bmn"):
        raise ValueError(f"layout must be 'mbn' or 'bmn', got {layout!r}")

    if out_dtype not in ("f32", "f16", "bf16"):
        raise ValueError(
            f"out_dtype must be 'f32', 'bf16', or 'f16', got {out_dtype!r}"
        )
    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3 or 4, got {num_buffers}")
    if tile_k % group_k != 0:
        raise ValueError(f"tile_k ({tile_k}) must be divisible by group_k ({group_k})")
    if tile_n % group_n != 0:
        raise ValueError(f"tile_n ({tile_n}) must be divisible by group_n ({group_n})")
    if K % tile_k != 0:
        raise ValueError(f"K ({K}) must be divisible by tile_k ({tile_k})")
    if N % tile_n != 0:
        raise ValueError(f"N ({N}) must be divisible by tile_n ({tile_n})")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")

    if const_expr(preshuffle_b):
        if N % 16 != 0 or tile_n % 16 != 0:
            raise ValueError(
                f"preshuffle_b requires N and tile_n divisible by 16, got N={N}, tile_n={tile_n}"
            )
        if K % 16 != 0 or tile_k % 16 != 0:
            raise ValueError(
                f"preshuffle_b requires K and tile_k divisible by 16, got K={K}, tile_k={tile_k}"
            )

    # ───────────────────────── split_k ─────────────────────────
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    if K % split_k != 0:
        raise ValueError(f"K must be divisible by split_k={split_k}, got K={K}")
    split_k_chunk = K // split_k
    if split_k_chunk % tile_k != 0:
        raise ValueError(
            f"K/split_k (={split_k_chunk}) must be divisible by tile_k={tile_k}"
        )
    if split_k_chunk % group_k != 0:
        raise ValueError(
            f"K/split_k (={split_k_chunk}) must be divisible by group_k={group_k}"
        )

    use_cluster = cluster_m > 1 or cluster_n > 1
    if const_expr(use_cluster):
        if cluster_m * cluster_n > 16:
            raise ValueError(
                f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}"
            )

    elem_bytes_d = 2 if out_dtype in ("bf16", "f16") else 4
    _effective_l2_pf = (
        max(1, l2_prefetch_distance - 1) if use_cluster else l2_prefetch_distance
    )
    effective_waves_per_eu = waves_per_eu

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    if block_threads > 1024:
        raise ValueError(f"block_threads must be <= 1024, got {block_threads}")

    if wave_specialized_tdm and num_warps < 2:
        raise ValueError(
            f"wave_specialized_tdm requires at least 2 waves, got {num_warps}"
        )

    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N}")

    num_k_tiles = split_k_chunk // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, got {num_k_tiles} (K/split_k={split_k_chunk}, tile_k={tile_k})"
        )

    tdm_store_enabled = use_tdm_store and split_k == 1

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep
    k_wmma_steps = tile_k // WMMA_K

    k_k_blocks = K // group_k
    n_n_blocks = N // group_n
    k_blocks_per_tile = tile_k // group_k

    gy_compile = N // tile_n  # compile-time grid.y (== number of N-tiles)

    lds_a_stride = tile_k + LDS_PAD_A_BYTES
    lds_a_stride_bytes = lds_a_stride
    lds_a_bytes = tile_m * lds_a_stride

    if const_expr(preshuffle_b):
        lds_b_stride = tile_k
        lds_b_stride_bytes = lds_b_stride
        lds_b_bytes = tile_n * tile_k
    else:
        lds_b_stride = tile_k + LDS_PAD_B_BYTES
        lds_b_stride_bytes = lds_b_stride
        lds_b_bytes = tile_n * lds_b_stride

    tdm_desc_num_warps = 1 if wave_specialized_tdm else num_warps

    def _align_up(value: int, align: int) -> int:
        return ((value + align - 1) // align) * align

    stage_layout = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="bmm_fp8_oob_layout"
    )
    stage_a_data_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_a_data_rel_off + lds_a_bytes
    stage_b_data_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_b_data_rel_off + lds_b_bytes
    stage_bytes = _align_up(stage_layout.ptr, 128)

    # ── Multi-stage pipeline schedule (compile-time) ──
    pre_loaded = num_buffers - 1
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers
    extra = num_k_tiles - _tail_start - pre_loaded
    _base_tail_plan = make_tail_plan(num_buffers, pre_loaded, extra)

    _last_compute_stage = _base_tail_plan[-1][1]
    stage_pitch_bytes = _align_up(stage_bytes, 1024)

    arena_alloc = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(f"bmm_w8a8_oob_{tile_m}x{tile_n}x{tile_k}"),
    )

    stage_phys_order = [i for i in range(num_buffers) if i != _last_compute_stage]
    stage_phys_order.append(_last_compute_stage)
    stage_base_off = [0] * num_buffers
    for phys_i, logical_i in enumerate(stage_phys_order):
        stage_base_off[logical_i] = phys_i * stage_pitch_bytes
    arena_alloc.ptr = stage_pitch_bytes * num_buffers
    arena_total_bytes = arena_alloc.ptr

    epilogue_fence_threshold_bytes = tdm_epilogue_fence_threshold_bytes(
        stage_base_off=stage_base_off,
        tail_plan=_base_tail_plan,
        loop_iters=loop_iters,
        extra=extra,
    )

    stage_a_data_off = [
        stage_base_off[i] + stage_a_data_rel_off for i in range(num_buffers)
    ]
    stage_b_data_off = [
        stage_base_off[i] + stage_b_data_rel_off for i in range(num_buffers)
    ]

    use_deep_pipeline = (
        tile_m == 128 and tile_n == 256 and tile_k == 128 and num_buffers == 4
    )
    if const_expr(use_deep_pipeline):
        stage_a_data_off = [0x00000, 0x04800, 0x09000, 0x0D800]
        stage_b_data_off = [0x20000, 0x29000, 0x32000, 0x3B000]
        arena_alloc.ptr = LDS_GFX1250_MAX_BYTES
        arena_total_bytes = arena_alloc.ptr
        epilogue_fence_threshold_bytes = 0

    if const_expr(tdm_store_enabled):
        lds_d_row_stride = warp_tile_n * elem_bytes_d + LDS_PAD_D_BYTES
        warp_d_bytes = warp_tile_m * lds_d_row_stride
        total_d_bytes = num_warps * warp_d_bytes
        d_output_off = 0
        _lds_d_stride_elems = lds_d_row_stride // 2
        _warp_d_elems = warp_d_bytes // 2
        _n_col_d_elems = WMMA_N * elem_bytes_d // 2
        d_need_epilogue_fence = total_d_bytes > epilogue_fence_threshold_bytes
        if total_d_bytes > arena_total_bytes:
            arena_total_bytes = total_d_bytes
            arena_alloc.ptr = total_d_bytes
    check_smem_capacity(arena_total_bytes, gpu_arch)

    TDM_LOADS_PER_STEP = 1 if wave_specialized_tdm else 2
    tail_plan = [
        (ls, cs, o * TDM_LOADS_PER_STEP // 2 if o > 0 else o)
        for ls, cs, o in _base_tail_plan
    ]

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def kernel_bmm_w8a8_bpreshuffle_gfx1250(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_m: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()

        if const_expr(inst_prefetch):
            if rocdl.wave_id() == arith.constant(0, type=T.i32):
                rocdl.s_prefetch_inst_burst(num_pages=10)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bzz = gpu.block_id("z")
        if const_expr(split_k > 1):
            bz = bzz / arith.index(split_k)  # batch
            ks_split_idx = bzz % arith.index(split_k)
            split_k_base = ks_split_idx * arith.index(split_k_chunk)
        else:
            bz = bzz
            ks_split_idx = arith.index(0)
            split_k_base = arith.index(0)

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

        if const_expr(use_cluster):
            local_x, local_y = cluster.compute_cluster_position()
            a_mcast_mask, b_mcast_mask = cluster.compute_mcast_masks(
                local_x, local_y, cluster_m, cluster_n
            )
        else:
            a_mcast_mask = 0
            b_mcast_mask = 0

        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16),
            (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1),
        )
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0),
            fx.get(thr_coord, 1),
            fx.get(thr_coord, 2),
            fx.get(thr_coord, 3),
        )

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        m_idx = arith.index_cast(T.index, i32_m.ir_value())

        if const_expr(layout == "mbn"):
            a_row_stride = B * K
            c_row_stride = B * N
            a_batch_off = bz * arith.index(K)
            c_batch_off = bz * arith.index(N)  #
        else:  # bmn
            a_row_stride = K
            c_row_stride = N
            a_batch_off = bz * m_idx * arith.index(K)
            c_batch_off = bz * m_idx * arith.index(N)
        b_batch_off = bz * arith.index(N)

        c_nrec = m_idx * arith.index(B) * arith.index(N * elem_bytes_d)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)
        if const_expr(split_k > 1):
            c_global_ptr_type = ir.Type.parse("!llvm.ptr<1>")
            c_global_base_i64 = llvm_dialect.PtrToIntOp(
                T.i64,
                fly.extract_aligned_pointer_as_index(
                    c_global_ptr_type, arg_c.__extract_to_ir_values__()[0]
                ),
            ).result

        scale_a_total_elems = m_idx * arith.index(B) * arith.index(k_k_blocks)
        scale_a_nrec = scale_a_total_elems * arith.index(ELEM_BYTES_SCALE)
        scale_a_rsrc = buffer_ops.create_buffer_resource(
            arg_a_scale, num_records_bytes=scale_a_nrec
        )

        scale_b_total_elems = arith.index(B * n_n_blocks * k_k_blocks)
        scale_b_nrec = scale_b_total_elems * arith.index(ELEM_BYTES_SCALE)
        scale_b_rsrc = buffer_ops.create_buffer_resource(
            arg_b_scale, num_records_bytes=scale_b_nrec
        )

        by_i32 = arith.index_cast(T.i32, by)
        bz_i32 = arith.index_cast(T.i32, bz)
        wave_n_index_i32 = arith.index_cast(T.i32, wave_n_idx)
        blk_m_i32 = arith.index_cast(T.i32, blk_m)
        warp_m_i32 = arith.index_cast(T.i32, warp_m_base)
        lane16_i32 = arith.index_cast(T.i32, lane16)

        m_idx_i32 = arith.index_cast(T.i32, m_idx)
        if const_expr(layout == "mbn"):
            scale_a_row_stride_i32 = arith.constant(B * k_k_blocks, type=T.i32)
            scale_a_batch_off_i32 = bz_i32 * arith.constant(k_k_blocks, type=T.i32)
        else:  # bmn
            scale_a_row_stride_i32 = arith.constant(k_k_blocks, type=T.i32)
            scale_a_batch_off_i32 = (
                bz_i32 * m_idx_i32 * arith.constant(k_k_blocks, type=T.i32)
            )
        scale_b_batch_off_i32 = bz_i32 * arith.constant(
            n_n_blocks * k_k_blocks, type=T.i32
        )
        scale_b_k_stride_i32 = arith.constant(k_k_blocks, type=T.i32)
        split_k_scale_off_i32 = arith.index_cast(T.i32, ks_split_idx) * arith.constant(
            split_k_chunk // group_k, type=T.i32
        )

        warp_n_block_base_i32 = by_i32 * arith.constant(
            tile_n // group_n, type=T.i32
        ) + (
            wave_n_index_i32 * arith.constant(warp_tile_n, type=T.i32)
        ) // arith.constant(
            group_n, type=T.i32
        )

        a_oob_bound_i32 = arith.index_cast(T.i32, m_idx)

        # ── TDM descriptor factories ──
        def make_desc_a(memref, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a,
                lds_memref=memref,
                global_offset=(blk_m, k_base + a_batch_off),
                tensor_shape=(tile_m, tile_k),
                strides=(a_row_stride, 1),
                tile_shape=(tile_m, tile_k),
                elem_bytes=1,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_A_BYTES,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=a_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
                oob_outer_bound=a_oob_bound_i32,
            )

        def make_desc_b(memref, k_base):
            if const_expr(preshuffle_b):
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_b,
                    lds_memref=memref,
                    global_offset=(
                        b_batch_off // arith.index(16) + blk_n // arith.index(16),
                        k_base * arith.index(16),
                    ),
                    tensor_shape=(B * N // 16, K * 16),
                    strides=(K * 16, 1),
                    tile_shape=(tile_n // 16, tile_k * 16),
                    elem_bytes=1,
                    pad_interval=0,
                    pad_amount=0,
                    num_warps=tdm_desc_num_warps,
                    workgroup_mask=b_mcast_mask,
                    atomic_barrier_enable=atomic_barrier_enable,
                    early_timeout=True,
                )
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b,
                lds_memref=memref,
                global_offset=(blk_n + b_batch_off, k_base),
                tensor_shape=(tile_n, tile_k),
                strides=(K, 1),
                tile_shape=(tile_n, tile_k),
                elem_bytes=1,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_B_BYTES,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
            )

        # ── A/B LDS fragment loaders ──
        def _precompute_a_lane_bases(lds_ptr):
            row_base = (warp_m_base + lane16) * arith.index(lds_a_stride_bytes)
            k_half_off = lane_kgrp * arith.index(16)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                base = (
                    row_base
                    + arith.index(wm * WMMA_M * lds_a_stride_bytes)
                    + k_half_off
                )
                bases.append(base)
            return lds_ptr, bases

        def _precompute_b_lane_bases(lds_ptr):
            if const_expr(preshuffle_b):
                _ngroup_stride = tile_k * 16
                _n_group_base = arith.index(warp_tile_n // 16) * wave_n_idx
                row_off = lane16 * arith.index(16)
                k_tile_off = lane_kgrp * arith.index(256)
                bases = []
                for wn in range_constexpr(wmma_n_rep):
                    ngroup_off = _n_group_base * arith.index(
                        _ngroup_stride
                    ) + arith.index(wn * _ngroup_stride)
                    bases.append(ngroup_off + row_off + k_tile_off)
                return lds_ptr, bases
            row_base = (warp_n_base + lane16) * arith.index(lds_b_stride_bytes)
            k_half_off = lane_kgrp * arith.index(16)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                base = (
                    row_base
                    + arith.index(wn * WMMA_N * lds_b_stride_bytes)
                    + k_half_off
                )
                bases.append(base)
            return lds_ptr, bases

        def _issue_frag_loads(lds_buffer, lane_base, ks):
            k_byte_off = arith.index(ks * WMMA_K)
            byte_off = lane_base + k_byte_off
            return [
                fx.Vector(lds_load_b128_raw(lds_buffer, byte_off)),
                fx.Vector(lds_load_b128_raw(lds_buffer, byte_off + arith.index(32))),
                fx.Vector(lds_load_b128_raw(lds_buffer, byte_off + arith.index(64))),
                fx.Vector(lds_load_b128_raw(lds_buffer, byte_off + arith.index(96))),
            ]

        def _issue_frag_loads_b(lds_buffer, lane_base, ks):
            """B 的 LDS frag load：preshuffle 时用 256/512-byte 步长，否则回退到 plain。"""
            if const_expr(preshuffle_b):
                _num_tiles = WMMA_K // 16
                k_subtile_off = arith.index(ks * _num_tiles * 256)
                byte_off = lane_base + k_subtile_off
                return [
                    fx.Vector(lds_load_b128_raw(lds_buffer, byte_off)),
                    fx.Vector(
                        lds_load_b128_raw(lds_buffer, byte_off + arith.index(512))
                    ),
                    fx.Vector(
                        lds_load_b128_raw(lds_buffer, byte_off + arith.index(1024))
                    ),
                    fx.Vector(
                        lds_load_b128_raw(lds_buffer, byte_off + arith.index(1536))
                    ),
                ]
            return _issue_frag_loads(lds_buffer, lane_base, ks)

        def _assemble_frag(raw4):
            v0, v1, v2, v3 = raw4
            v01 = v0.shuffle(v1, list(range(8)))
            v23 = v2.shuffle(v3, list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def _pack_e8m0_byte(s_i32):
            s32 = s_i32 & arith.constant(0xFF, type=T.i32)
            return s32 | (s32 << 8) | (s32 << 16) | (s32 << 24)

        def _load_a_scales(kt_idx_i32):
            k_block_base_i32 = split_k_scale_off_i32 + kt_idx_i32 * arith.constant(
                k_blocks_per_tile, type=T.i32
            )
            a_scales = []
            for wm in range_constexpr(wmma_m_rep):
                row_i32 = (
                    blk_m_i32
                    + warp_m_i32
                    + arith.constant(wm * WMMA_M, type=T.i32)
                    + lane16_i32
                )
                row_off = (
                    row_i32 * scale_a_row_stride_i32
                    + scale_a_batch_off_i32
                    + k_block_base_i32
                )
                kb_vals = []
                for kb in range_constexpr(k_blocks_per_tile):
                    off = row_off + arith.constant(kb, type=T.i32)
                    i8_val = buffer_ops.buffer_load(
                        scale_a_rsrc, off, vec_width=1, dtype=T.i8
                    )
                    kb_vals.append(_pack_e8m0_byte(arith.extui(T.i32, i8_val)))
                a_scales.append(kb_vals)
            return a_scales

        _n_nblocks_per_warp = warp_tile_n // group_n if warp_tile_n >= group_n else 1

        def _load_b_scales(kt_idx_i32):
            k_block_base_i32 = split_k_scale_off_i32 + kt_idx_i32 * arith.constant(
                k_blocks_per_tile, type=T.i32
            )
            n_nblocks = max(1, _n_nblocks_per_warp)
            per_nblock = []
            for nb_local in range_constexpr(n_nblocks):
                n_block_i32 = warp_n_block_base_i32 + arith.constant(
                    nb_local, type=T.i32
                )
                row_off = (
                    scale_b_batch_off_i32
                    + n_block_i32 * scale_b_k_stride_i32
                    + k_block_base_i32
                )
                kb_vals = []
                for kb in range_constexpr(k_blocks_per_tile):
                    off = row_off + arith.constant(kb, type=T.i32)
                    i8_val = buffer_ops.buffer_load(
                        scale_b_rsrc, off, vec_width=1, dtype=T.i8
                    )
                    kb_vals.append(_pack_e8m0_byte(arith.extui(T.i32, i8_val)))
                per_nblock.append(kb_vals)
            b_scales = []
            for wn in range_constexpr(wmma_n_rep):
                nb_local = (wn * WMMA_N) // group_n
                b_scales.append(per_nblock[nb_local])
            return b_scales

        def _kb_for(ks):
            return (ks * WMMA_K) // group_k

        # ── WMMA emit ──
        def _emit_wmma(accs, wm, wn, a_frag, b_frag, a_scale, b_scale):
            idx = wm * wmma_n_rep + wn
            accs[idx] = wmma_op(
                T.vec(8, T.f32),
                b_frag,
                a_frag,
                accs[idx],
                b_scale,
                a_scale,
                fmtA=0,
                fmtB=0,
                scaleAType=0,
                fmtScaleA=0,
                scaleBType=0,
                fmtScaleB=0,
            )

        def _l2_prefetch(k_base):
            if const_expr(_effective_l2_pf <= 0):
                return
            pf_k = k_base + arith.index(_effective_l2_pf * tile_k)
            tdm_ops.l2_prefetch_tile(
                arg_a,
                (blk_m, pf_k + a_batch_off),
                (tile_m, tile_k),
                (a_row_stride, 1),
                elem_bytes=1,
                thread_id=tx,
                block_threads=block_threads,
            )
            if const_expr(preshuffle_b):
                tdm_ops.l2_prefetch_tile(
                    arg_b,
                    (
                        b_batch_off // arith.index(16) + blk_n // arith.index(16),
                        pf_k * arith.index(16),
                    ),
                    (tile_n // 16, tile_k * 16),
                    (K * 16, 1),
                    elem_bytes=1,
                    thread_id=tx,
                    block_threads=block_threads,
                )
            else:
                tdm_ops.l2_prefetch_tile(
                    arg_b,
                    (b_batch_off + blk_n, pf_k),
                    (tile_n, tile_k),
                    (K, 1),
                    elem_bytes=1,
                    thread_id=tx,
                    block_threads=block_threads,
                )

        # ── compute one K-tile (A-streaming) ──
        DS_LOADS_PER_FRAG = 4

        def _a_streaming_pipeline(
            accs,
            a_buf,
            a_bases,
            b_buf,
            b_bases,
            a_scales,
            b_scales,
            mid_compute_callback=None,
            emit_filler=None,
        ):
            current_accs = list(accs)

            for ks in range_constexpr(k_wmma_steps):
                kb = _kb_for(ks)
                b_raw = [
                    _issue_frag_loads_b(b_buf, b_bases[wn], ks)
                    for wn in range_constexpr(wmma_n_rep)
                ]
                b_frags = [_assemble_frag(r) for r in b_raw]

                a_raw = _issue_frag_loads(a_buf, a_bases[0], ks)

                for wm in range_constexpr(wmma_m_rep):
                    is_last_wm = const_expr(wm == wmma_m_rep - 1)
                    if const_expr(not is_last_wm):
                        a_raw_next = _issue_frag_loads(a_buf, a_bases[wm + 1], ks)
                        rocdl.s_wait_dscnt(DS_LOADS_PER_FRAG)
                    else:
                        rocdl.s_wait_dscnt(0)
                        if const_expr(
                            ks == k_wmma_steps - 1 and emit_filler is not None
                        ):
                            rocdl.sched_barrier(0)
                            emit_filler()

                    a_frag = _assemble_frag(a_raw)

                    for wn in range_constexpr(wmma_n_rep):
                        _emit_wmma(
                            current_accs,
                            wm,
                            wn,
                            a_frag,
                            b_frags[wn],
                            a_scales[wm][kb],
                            b_scales[wn][kb],
                        )

                    if const_expr(not is_last_wm):
                        a_raw = a_raw_next

                    if const_expr(
                        ks == 0 and wm == 0 and mid_compute_callback is not None
                    ):
                        rocdl.sched_barrier(0)
                        mid_compute_callback()
            return current_accs

        def compute_tile(
            accs_in,
            lds_a_idx,
            lds_b_idx,
            a_scales,
            b_scales,
            mid_compute_callback=None,
            emit_filler=None,
        ):
            a_buf, a_bases = _precompute_a_lane_bases(lds_a_idx)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b_idx)
            return _a_streaming_pipeline(
                accs_in,
                a_buf,
                a_bases,
                b_buf,
                b_bases,
                a_scales,
                b_scales,
                mid_compute_callback=mid_compute_callback,
                emit_filler=emit_filler,
            )

        use_half_streaming_schedule = (wmma_m_rep % 2) == 0 and wmma_m_rep > 1

        def hot_loop_scheduler_half_streaming():
            half_wm = wmma_m_rep // 2
            half_wmma = half_wm * wmma_n_rep
            b_full_loads = wmma_n_rep * DS_LOADS_PER_B_FRAG
            a_half_loads = half_wm * DS_LOADS_PER_A_FRAG
            for ks in range_constexpr(k_wmma_steps):
                if const_expr(ks == 0):
                    rocdl.sched_dsrd(b_full_loads + a_half_loads)
                else:
                    rocdl.sched_dsrd(a_half_loads)
                rocdl.sched_mfma(half_wmma)
                rocdl.sched_dsrd(a_half_loads)
                rocdl.sched_mfma(half_wmma)
                if const_expr(ks < k_wmma_steps - 1):
                    rocdl.sched_dsrd(b_full_loads)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler():
            if const_expr(use_half_streaming_schedule):
                hot_loop_scheduler_half_streaming()
            else:
                rocdl.sched_barrier(0)

        # ── pipeline stage LDS handles ──
        acc_zero = arith.constant_vector(0.0, T.vec(ACC_VEC_SIZE, T.f32))
        accs = [acc_zero] * n_accs

        arena_base_ptr = arena_alloc.get_base()
        lds_a_data_f16 = lds_a_bytes // 2
        lds_b_data_f16 = lds_b_bytes // 2
        stages_a = [
            SmemPtr(arena_base_ptr, stage_a_data_off[i], T.f16, shape=(lds_a_data_f16,))
            for i in range_constexpr(num_buffers)
        ]
        stages_b = [
            SmemPtr(arena_base_ptr, stage_b_data_off[i], T.f16, shape=(lds_b_data_f16,))
            for i in range_constexpr(num_buffers)
        ]
        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]
        stages_a_idx = [
            extract_lds_base_idx(stages_a[i]) for i in range_constexpr(num_buffers)
        ]
        stages_b_idx = [
            extract_lds_base_idx(stages_b[i]) for i in range_constexpr(num_buffers)
        ]

        # ── TDM descriptor dgroup0 lane management ──
        def _dg0_lane(desc, lane):
            return fx.Vector(desc.dgroup0)[lane]

        def _pack_dg0(pred, lds_addr, addr_lo, addr_hi):
            return fx.Vector.from_elements([pred, lds_addr, addr_lo, addr_hi], fx.Int32)

        stages_a_lds_addr = [
            _dg0_lane(make_desc_a(stages_a_mem[i], split_k_base), 1)
            for i in range_constexpr(num_buffers)
        ]
        stages_b_lds_addr = [
            _dg0_lane(make_desc_b(stages_b_mem[i], split_k_base), 1)
            for i in range_constexpr(num_buffers)
        ]

        desc_a_init = make_desc_a(stages_a_mem[0], split_k_base)
        desc_b_init = make_desc_b(stages_b_mem[0], split_k_base)
        addr_lo_a = _dg0_lane(desc_a_init, 2)
        addr_hi_a = _dg0_lane(desc_a_init, 3)
        addr_lo_b = _dg0_lane(desc_b_init, 2)
        addr_hi_b = _dg0_lane(desc_b_init, 3)
        dgroup1_a = desc_a_init.dgroup1
        dgroup1_b = desc_b_init.dgroup1

        adv_a_i32 = fx.Int32(tile_k)
        adv_b_i32 = fx.Int32(tile_k * 16 if preshuffle_b else tile_k)
        pred_const = fx.Int32(1)

        if const_expr(wave_specialized_tdm):
            tdm_wave_id = rocdl.wave_id()
            tdm_wave_is_a = tdm_wave_id == fx.Int32(0)

            def _select_wave_tdm_value(a_value, b_value):
                return arith.select(tdm_wave_is_a, a_value, b_value)

            active_stage_lds_addr = [
                _select_wave_tdm_value(stages_a_lds_addr[i], stages_b_lds_addr[i])
                for i in range_constexpr(num_buffers)
            ]
            active_addr_lo = _select_wave_tdm_value(addr_lo_a, addr_lo_b)
            active_addr_hi = _select_wave_tdm_value(addr_hi_a, addr_hi_b)
            active_dgroup1 = _select_wave_tdm_value(dgroup1_a, dgroup1_b)
            active_adv_i32 = _select_wave_tdm_value(adv_a_i32, adv_b_i32)
            active_pred_const = arith.select(
                tdm_wave_id < fx.Int32(2), fx.Int32(1), fx.Int32(0)
            )

        def _pipeline_fence(outstanding=0):
            pipeline_fence(outstanding=outstanding, use_cluster=use_cluster)

        def _pipeline_fence_signal(outstanding=0):
            pipeline_fence_signal(outstanding=outstanding, use_cluster=use_cluster)

        def _issue_ab(load_stage, addr_box, k_prefetch=None):
            """非 wave-specialized：本 wave 同时发 A 和 B 两条 TDM。addr_box=[[lo_a],[lo_b]]。"""
            dg0_a = _pack_dg0(
                pred_const, stages_a_lds_addr[load_stage], addr_box[0][0], addr_hi_a
            )
            dg0_b = _pack_dg0(
                pred_const, stages_b_lds_addr[load_stage], addr_box[1][0], addr_hi_b
            )
            issue_tdm_loads(
                tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                wave_specialized=False,
            )
            addr_box[0][0] = addr_box[0][0] + adv_a_i32
            addr_box[1][0] = addr_box[1][0] + adv_b_i32
            if const_expr(k_prefetch is not None):
                _l2_prefetch(k_prefetch)

        def _issue_ws(load_stage, addr_box, k_prefetch=None):
            """wave-specialized：每个 loader wave 发自己那条 TDM。addr_box=[[active_lo]]。"""
            dg0 = _pack_dg0(
                active_pred_const,
                active_stage_lds_addr[load_stage],
                addr_box[0][0],
                active_addr_hi,
            )
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, active_dgroup1))
            addr_box[0][0] = addr_box[0][0] + active_adv_i32
            if const_expr(k_prefetch is not None):
                _l2_prefetch(k_prefetch)

        # ── Prologue ──
        if const_expr(wave_specialized_tdm):
            _pro_box = [[active_addr_lo]]
            for i in range_constexpr(pre_loaded):
                _issue_ws(i, _pro_box)
            active_addr_lo = _pro_box[0][0]
        else:
            _pro_box = [[addr_lo_a], [addr_lo_b]]
            for i in range_constexpr(pre_loaded):
                _issue_ab(i, _pro_box)
            addr_lo_a = _pro_box[0][0]
            addr_lo_b = _pro_box[1][0]

        _pipeline_fence(outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2))

        # ── Main loop — fence at top, TDM mid-compute (overlap DMA with WMMA) ──
        _fence_outstanding = TDM_LOADS_PER_STEP * (num_buffers - 2)
        epi_addrs_box = [None]

        if const_expr(loop_iters > 0):
            if const_expr(wave_specialized_tdm):
                init_args = list(accs) + [active_addr_lo]
                for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                    accs_in = list(state[:n_accs])
                    cur_lo = state[n_accs]

                    for buf_idx in range_constexpr(num_buffers):
                        load_stage = (buf_idx + num_buffers - 1) % num_buffers

                        _pipeline_fence_signal(outstanding=_fence_outstanding)
                        pipeline_fence_wait(use_cluster=use_cluster)

                        kt_idx_i32 = arith.index_cast(
                            T.i32,
                            loop_iter * arith.index(num_buffers) + arith.index(buf_idx),
                        )
                        a_scales = _load_a_scales(kt_idx_i32)
                        b_scales = _load_b_scales(kt_idx_i32)

                        addr_box = [[cur_lo]]

                        def _mid_ws(
                            _ls=load_stage,
                            _ab=addr_box,
                            _k_off=(
                                split_k_base
                                + loop_iter * arith.index(num_buffers * tile_k)
                                + arith.index(buf_idx * tile_k)
                            ),
                        ):
                            _issue_ws(_ls, _ab, k_prefetch=_k_off)

                        rocdl.sched_barrier(0)
                        accs_in = compute_tile(
                            accs_in,
                            stages_a_idx[buf_idx],
                            stages_b_idx[buf_idx],
                            a_scales,
                            b_scales,
                            mid_compute_callback=_mid_ws,
                        )
                        cur_lo = addr_box[0][0]
                        hot_loop_scheduler()

                    results = yield list(accs_in) + [cur_lo]

                accs = list(results[:n_accs])
                active_addr_lo = results[n_accs]
            else:
                init_args = list(accs) + [addr_lo_a, addr_lo_b]
                for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                    accs_in = list(state[:n_accs])
                    cur_lo_a = state[n_accs]
                    cur_lo_b = state[n_accs + 1]

                    for buf_idx in range_constexpr(num_buffers):
                        load_stage = (buf_idx + num_buffers - 1) % num_buffers

                        _pipeline_fence_signal(outstanding=_fence_outstanding)
                        pipeline_fence_wait(use_cluster=use_cluster)

                        kt_idx_i32 = arith.index_cast(
                            T.i32,
                            loop_iter * arith.index(num_buffers) + arith.index(buf_idx),
                        )
                        a_scales = _load_a_scales(kt_idx_i32)
                        b_scales = _load_b_scales(kt_idx_i32)

                        addr_boxes = [[cur_lo_a], [cur_lo_b]]

                        def _mid_ab(
                            _ls=load_stage,
                            _ab=addr_boxes,
                            _k_off=(
                                split_k_base
                                + loop_iter * arith.index(num_buffers * tile_k)
                                + arith.index(buf_idx * tile_k)
                            ),
                        ):
                            _issue_ab(_ls, _ab, k_prefetch=_k_off)

                        rocdl.sched_barrier(0)
                        accs_in = compute_tile(
                            accs_in,
                            stages_a_idx[buf_idx],
                            stages_b_idx[buf_idx],
                            a_scales,
                            b_scales,
                            mid_compute_callback=_mid_ab,
                        )
                        cur_lo_a = addr_boxes[0][0]
                        cur_lo_b = addr_boxes[1][0]
                        hot_loop_scheduler()

                    results = yield list(accs_in) + [cur_lo_a, cur_lo_b]

                accs = list(results[:n_accs])
                addr_lo_a = results[n_accs]
                addr_lo_b = results[n_accs + 1]

        # ── Tail ──
        if const_expr(loop_iters > 0):
            _pipeline_fence(outstanding=0)
        elif const_expr(use_cluster):
            cluster.cluster_barrier()

        # ── Epilogue helpers ──
        _out_elem = (
            T.f16 if out_dtype == "f16" else (T.bf16 if out_dtype == "bf16" else None)
        )
        _half_out = out_dtype in ("f16", "bf16")

        def epilogue_lds_stores(final_accs, d_buf, d_base):
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    imm = wm * WMMA_M * _lds_d_stride_elems + wn * _n_col_d_elems
                    store_acc_vec8_to_lds(
                        d_buf, d_base, imm, final_accs[idx], out_elem=_out_elem
                    )

        def epilogue_prepare_addrs():
            addrs = []
            n_stride = arith.index(c_row_stride)
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                    col_base = (
                        c_batch_off
                        + blk_n
                        + warp_n_base
                        + arith.index(wn * WMMA_N)
                        + lane_kgrp * arith.index(8)
                    )
                    if const_expr(_half_out):
                        addrs.append(
                            (row * n_stride + col_base) * arith.index(elem_bytes_d)
                        )
                    else:
                        for half in range_constexpr(2):
                            addrs.append(
                                row * n_stride + col_base + arith.index(half * 4)
                            )
            return addrs

        def epilogue_stores_guarded(final_accs, addrs):
            addr_idx = 0
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    row_in_batch = (
                        blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                    )
                    n_slots = 1 if _half_out else 2
                    addr_arg = (
                        addrs[addr_idx] if _half_out else addrs[addr_idx : addr_idx + 2]
                    )

                    def _emit_store(_acc=final_accs[idx], _addr=addr_arg):
                        if const_expr(_half_out):
                            store_acc_vec8_to_buffer(
                                _acc,
                                c_rsrc,
                                _addr,
                                out_elem=_out_elem,
                                offset_is_bytes=True,
                            )
                        else:
                            store_acc_vec8_to_buffer(_acc, c_rsrc, _addr)

                    if_op = scf.IfOp(row_in_batch < m_idx, [], has_else=False)
                    with ir.InsertionPoint(if_op.then_block):
                        _emit_store()
                        scf.YieldOp([])
                    addr_idx += n_slots

        def _atomic_fadd_global(val, byte_off):
            addr_i64 = llvm_dialect.AddOp(
                c_global_base_i64,
                arith.index_cast(T.i64, byte_off),
                llvm_dialect.IntegerOverflowFlags(0),
            ).result
            ptr = llvm_dialect.IntToPtrOp(c_global_ptr_type, addr_i64).result
            llvm_dialect.AtomicRMWOp(
                llvm_dialect.AtomicBinOp.fadd,
                ptr,
                val.ir_value(),
                llvm_dialect.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )

        def _atomic_add_acc_vec8_to_buffer(acc_vec8, addr):
            if const_expr(_half_out):
                h_vec = fx.Vector(arith.trunc_f(T.vec(8, _out_elem), acc_vec8))
                for pair in range_constexpr(4):
                    pair_vec = fx.Vector.from_elements(
                        [h_vec[pair * 2], h_vec[pair * 2 + 1]]
                    )
                    byte_off = addr + arith.index(pair * 4)
                    _atomic_fadd_global(pair_vec, byte_off)
                return 1
            acc_vec = fx.Vector(acc_vec8)
            for half in range_constexpr(2):
                base_addr = addr[half] if isinstance(addr, (list, tuple)) else addr
                for vi in range_constexpr(4):
                    val = acc_vec[half * 4 + vi]
                    byte_off = (base_addr + arith.index(vi)) * arith.index(4)
                    _atomic_fadd_global(val, byte_off)
            return 2

        def epilogue_atomic_adds(final_accs, addrs):
            addr_idx = 0
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    n_slots = 1 if _half_out else 2
                    addr_arg = (
                        addrs[addr_idx] if _half_out else addrs[addr_idx : addr_idx + 2]
                    )
                    row_in_batch = (
                        blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                    )
                    if_op = scf.IfOp(row_in_batch < m_idx, [], has_else=False)
                    with ir.InsertionPoint(if_op.then_block):
                        _atomic_add_acc_vec8_to_buffer(final_accs[idx], addr_arg)
                        scf.YieldOp([])
                    addr_idx += n_slots

        _tail_had_load = False
        _tail_kt = [loop_iters * num_buffers]
        for _load_stage, _compute_stage, _outstanding in tail_plan:
            _kt_i32 = arith.constant(_tail_kt[0], type=T.i32)
            _tail_kt[0] += 1
            _tail_a_scales = _load_a_scales(_kt_i32)
            _tail_b_scales = _load_b_scales(_kt_i32)

            if const_expr(_outstanding == -1):
                if const_expr(_tail_had_load):
                    _pipeline_fence(outstanding=0)
                if const_expr(tdm_store_enabled):
                    rocdl.sched_barrier(0)
                    accs = compute_tile(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        _tail_a_scales,
                        _tail_b_scales,
                    )
                else:

                    def _emit_epi_addrs():
                        epi_addrs_box[0] = epilogue_prepare_addrs()

                    rocdl.sched_barrier(0)
                    accs = compute_tile(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        _tail_a_scales,
                        _tail_b_scales,
                        emit_filler=_emit_epi_addrs,
                    )
            else:
                _pipeline_fence_signal(outstanding=_outstanding)
                pipeline_fence_wait(use_cluster=use_cluster)

                _tail_mid_cb = None
                if const_expr(_load_stage is not None):
                    _tail_had_load = True
                    if const_expr(wave_specialized_tdm):
                        _tail_box = [[active_addr_lo]]

                        def _tail_mid_ws(_ls=_load_stage, _ab=_tail_box):
                            _issue_ws(_ls, _ab)

                        _tail_mid_cb = _tail_mid_ws
                    else:
                        _tail_box = [[addr_lo_a], [addr_lo_b]]

                        def _tail_mid_ab(_ls=_load_stage, _ab=_tail_box):
                            _issue_ab(_ls, _ab)

                        _tail_mid_cb = _tail_mid_ab

                rocdl.sched_barrier(0)
                accs = compute_tile(
                    accs,
                    stages_a_idx[_compute_stage],
                    stages_b_idx[_compute_stage],
                    _tail_a_scales,
                    _tail_b_scales,
                    mid_compute_callback=_tail_mid_cb,
                )
                hot_loop_scheduler()

                if const_expr(_load_stage is not None):
                    if const_expr(wave_specialized_tdm):
                        active_addr_lo = _tail_box[0][0]
                    else:
                        addr_lo_a = _tail_box[0][0]
                        addr_lo_b = _tail_box[1][0]

        # ── Epilogue dispatch ──
        if const_expr(split_k > 1):
            rocdl.sched_barrier(0)
            epilogue_atomic_adds(accs, epilogue_prepare_addrs())
        elif const_expr(tdm_store_enabled):
            d_lds_base_ptr = arena_base_ptr
            d_lds_f16_count = total_d_bytes // 2  # SmemPtr element type is bf16 (2B)
            d_smem = SmemPtr(
                d_lds_base_ptr, d_output_off, T.bf16, shape=(d_lds_f16_count,)
            )
            d_lds_buffer = get_lds_memref(d_smem)

            warp_lds_off = (
                wave_m_idx * arith.index(n_warp) + wave_n_idx
            ) * arith.index(_warp_d_elems)
            d_lane_base = (
                warp_lds_off
                + lane16 * arith.index(_lds_d_stride_elems)
                + lane_kgrp * arith.index(4 * elem_bytes_d)
            )

            wave_id_idx = arith.index_cast(T.index, rocdl.wave_id())
            d_warp_off_sgpr = wave_id_idx * arith.index(warp_d_bytes) + arith.index(
                d_output_off
            )
            warp_m_off_sgpr = (wave_id_idx / arith.index(n_warp)) * arith.index(
                warp_tile_m
            )
            warp_n_off_sgpr = (wave_id_idx % arith.index(n_warp)) * arith.index(
                warp_tile_n
            )
            d_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_c,
                lds_memref=d_lds_base_ptr,
                global_offset=(
                    blk_m + warp_m_off_sgpr,
                    c_batch_off + blk_n + warp_n_off_sgpr,
                ),
                tensor_shape=(warp_tile_m, warp_tile_n),
                strides=(c_row_stride, 1),
                tile_shape=(warp_tile_m, warp_tile_n),
                elem_bytes=elem_bytes_d,
                pad_interval=warp_tile_n,
                pad_amount=LDS_PAD_D_BYTES // elem_bytes_d,
                num_warps=1,
                lds_byte_offset=d_warp_off_sgpr,
                for_store=True,
            )

            def _emit_tdm_store():
                if const_expr(d_need_epilogue_fence):
                    pipeline_fence(outstanding=0, use_cluster=use_cluster)
                rocdl.sched_barrier(0)
                epilogue_lds_stores(accs, d_lds_buffer, d_lane_base)
                rocdl.s_wait_dscnt(0)
                tdm_ops.tensor_store_2d(d_desc)
                tdm_ops.tensor_wait(0)

            def _emit_buffer_store_guarded():
                rocdl.sched_barrier(0)
                epilogue_stores_guarded(accs, epilogue_prepare_addrs())

            full_tile = (blk_m + arith.index(tile_m)) <= m_idx
            if_op = scf.IfOp(full_tile, [], has_else=True)
            with ir.InsertionPoint(if_op.then_block):
                _emit_tdm_store()
                scf.YieldOp([])
            with ir.InsertionPoint(if_op.else_block):
                _emit_buffer_store_guarded()
                scf.YieldOp([])
        else:
            rocdl.sched_barrier(0)
            epilogue_stores_guarded(accs, epilogue_prepare_addrs())

    cache_tag = (
        B,
        K,
        N,
        group_k,
        group_n,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        out_dtype,
        waves_per_eu,
        l2_prefetch_distance,
        use_tdm_store,
        inst_prefetch,
        expert_sched_mode,
        cluster_m,
        cluster_n,
        wave_specialized_tdm,
        atomic_barrier_enable,
        split_k,
        preshuffle_b,
        layout,
    )

    @flyc.jit
    def launch_bmm_w8a8_bpreshuffle_gfx1250(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_m: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena_alloc.finalized = False
            arena_alloc.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))

        launcher = kernel_bmm_w8a8_bpreshuffle_gfx1250(
            arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale, i32_m
        )
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                if effective_waves_per_eu is not None:
                    _wpe = int(effective_waves_per_eu)
                    if _wpe >= 1:
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), _wpe
                        )
                if use_cluster:
                    op.attributes["rocdl.cluster_dims"] = ir.StringAttr.get(
                        f"{cluster_m},{cluster_n},1"
                    )
        cluster_arg = (cluster_m, cluster_n, 1) if use_cluster else None
        launcher.launch(
            grid=(gx, gy_compile, B * split_k),
            block=(block_threads, 1, 1),
            cluster=cluster_arg,
            stream=stream,
        )

    llvm_opts = {}
    if expert_sched_mode:
        llvm_opts["amdgpu-expert-scheduling-mode"] = True
    if inst_prefetch:
        llvm_opts["amdgpu-inst-prefetch-distance"] = 8
    if llvm_opts:
        launch_bmm_w8a8_bpreshuffle_gfx1250.compile_hints["llvm_options"] = llvm_opts

    return launch_bmm_w8a8_bpreshuffle_gfx1250


__all__ = ["compile_bmm_w8a8_bpreshuffle_gfx1250"]
