# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A8W8 FP8 blockscale GEMM for gfx1250.

Computes Y = X @ W^T with per-K-block f32 scales.
Supports reg_preload / no_op_preload and optional TDM-store output.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import math as math_dialect
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
LDS_PAD_D_BYTES = 16
E8M0_IDENTITY = 0x7F7F7F7F

_STAGE_NAMES = ("ping", "pong", "pang", "pung")


def _lds_vec_type(memref, total_bits):
    """Return a vector type of the right shape to hold `total_bits` of the
    memref's element type (used to size ds_load_bNNN reads)."""
    raw_mr = arith.unwrap(memref)
    elem_type = ir.MemRefType(raw_mr.type).element_type
    elem_bits = get_mlir_type_size(elem_type) * 8
    n = total_bits // elem_bits
    return ir.VectorType.get([n], elem_type)


def lds_load_b128(memref, elem_off):
    """ds_load_b128: load 16 bytes from LDS into a vector<4xi32>."""
    vec_ty = _lds_vec_type(memref, 128)
    loaded = vector.load_op(vec_ty, memref, [elem_off])
    return vector.bitcast(ir.VectorType.get([4], ir.IntegerType.get_signless(32)), loaded)


def lds_store_b128(memref, elem_off, data):
    """ds_store_b128: store 16 bytes to LDS, bitcast to match the memref element type."""
    vec_ty = _lds_vec_type(memref, 128)
    typed_vec = vector.bitcast(vec_ty, data)
    vector.store(typed_vec, memref, [elem_off])


def store_acc_vec8_to_lds(memref, base_elem_off, imm_elem_off, acc_vec8, out_elem=None):
    """Write a vec<8 f32> accumulator to LDS for TDM-store epilogue.

    Half output (out_elem = T.bf16/T.f16): trunc_f → bitcast(vec<4xi32>) → 1
    ds_store_b128 (16 bytes covering all 8 elements).
    f32 output (out_elem = None): two ds_store_b128 calls writing 4 f32 each;
    second store offset by 8 LDS elements (the LDS memref is 16-bit-typed even
    for f32 output, so 8 elems = 16 bytes = 4 f32).
    """
    off = base_elem_off + arith.index(imm_elem_off)
    if out_elem is not None:
        h_vec = arith.trunc_f(T.vec(8, out_elem), acc_vec8)
        i32_vec = vector.bitcast(T.vec(4, T.i32), h_vec)
        lds_store_b128(memref, off, i32_vec)
    else:
        for half in range(2):
            vals = [vector.extract(acc_vec8,
                                   static_position=[half * 4 + vi],
                                   dynamic_position=[]) for vi in range(4)]
            vec4 = vector.from_elements(T.vec(4, T.f32), vals)
            lds_store_b128(memref, off + arith.index(half * 8), vec4)


def store_acc_vec8_to_buffer(acc_vec8, c_rsrc, addr, out_elem=None, offset_is_bytes=False):
    """Write a vec<8xf32> accumulator to global via buffer_store.

    If `out_elem` is a half-precision type (bf16/fp16), truncate f32→half and
    emit a single 16-byte buffer_store of a vec<4xi32>.
    If `out_elem` is None (f32 out), emit two vec<4xf32> stores (one per half).
    """
    if out_elem is not None:
        h_vec = arith.trunc_f(T.vec(8, out_elem), acc_vec8)
        i32_vec = vector.bitcast(T.vec(4, T.i32), h_vec)
        buffer_ops.buffer_store(i32_vec, c_rsrc, addr, offset_is_bytes=offset_is_bytes)
        return 1
    for half in range(2):
        vals = [vector.extract(acc_vec8, static_position=[half * 4 + vi], dynamic_position=[]) for vi in range(4)]
        vec4 = vector.from_elements(T.vec(4, T.f32), vals)
        if isinstance(addr, (list, tuple)):
            buffer_ops.buffer_store(vec4, c_rsrc, addr[half])
        else:
            buffer_ops.buffer_store(vec4, c_rsrc, addr)
    return 2



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
    variant: str = "reg_preload",
    N: int = 0,
    use_tdm_store: bool = False,
):
    if variant not in ("reg_preload", "no_op_preload"):
        raise ValueError(f"variant must be 'reg_preload' or 'no_op_preload', got {variant!r}")
    if out_dtype not in ("bf16", "fp16", "f32"):
        raise ValueError(f"out_dtype must be 'bf16', 'fp16', or 'f32', got {out_dtype!r}")
    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3, or 4, got {num_buffers}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_k % scale_block_k != 0:
        raise ValueError(f"tile_k ({tile_k}) must be a multiple of scale_block_k ({scale_block_k})")
    if scale_block_k % WMMA_K != 0:
        raise ValueError(f"scale_block_k ({scale_block_k}) must be a multiple of {WMMA_K}")
    if K % tile_k != 0:
        raise ValueError(f"K ({K}) must be divisible by tile_k ({tile_k})")
    if K % scale_block_k != 0:
        raise ValueError(f"K ({K}) must be divisible by scale_block_k ({scale_block_k})")
    if use_tdm_store:
        if N <= 0:
            raise ValueError("use_tdm_store=True requires N > 0 (compile-time row stride)")
        if N % tile_n != 0:
            raise ValueError(
                f"use_tdm_store=True requires N ({N}) to be a multiple of tile_n ({tile_n})"
            )

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N}")

    wmma_m_rep = warp_tile_m // WMMA_M                 # WMMA tiles per warp along M
    wmma_n_rep = warp_tile_n // WMMA_N                 # WMMA tiles per warp along N
    n_accs = wmma_m_rep * wmma_n_rep                   # global accumulators per warp
    k_wmma_steps = tile_k // WMMA_K                    # WMMAs per K-tile along K
    scales_per_tile = tile_k // scale_block_k          # scale blocks per K-tile
    wmma_steps_per_scale = scale_block_k // WMMA_K
    wmma_pipeline_depth = min(n_accs, 2)
    acc_coords = [(wm, wn, wm * wmma_n_rep + wn) for wm in range(wmma_m_rep) for wn in range(wmma_n_rep)]

    num_k_tiles = K // tile_k
    scale_k = K // scale_block_k

    if num_k_tiles < num_buffers - 1:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers - 1}, "
            f"got {num_k_tiles}"
        )

    gpu_arch = str(get_hip_arch())
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
        alloc = SmemAllocator(None, arch=gpu_arch, global_sym_name=f"a8w8bs_{_STAGE_NAMES[i]}")
        off = alloc._align(alloc.ptr, 16)
        stage_a_data_off.append(off)
        alloc.ptr = off + lds_a_data_bytes
        off = alloc._align(alloc.ptr, 16)
        stage_b_data_off.append(off)
        alloc.ptr = off + lds_b_data_bytes
        stage_allocators.append(alloc)

    if use_tdm_store:
        lds_d_row_stride_bytes = tile_n * elem_bytes_d + LDS_PAD_D_BYTES
        total_d_bytes = tile_m * lds_d_row_stride_bytes
        _lds_d_stride_elems_d = lds_d_row_stride_bytes // 2  
        _n_col_d_elems_d = WMMA_N * elem_bytes_d // 2

        d_lds_allocator = SmemAllocator(
            None, arch=gpu_arch, global_sym_name="a8w8bs_d_out",
        )
        d_lds_allocator.ptr = total_d_bytes

    prologue_tiles = num_buffers - 1
    main_loop_iters = (num_k_tiles - prologue_tiles) // num_buffers
    extra_tiles = num_k_tiles - main_loop_iters * num_buffers - prologue_tiles
    drain_iters = num_buffers - 2

    MAIN_TDM_OUTSTANDING = (num_buffers - 2) * 2


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
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

        layout_thr = fx.make_layout((m_warp, n_warp, 2, 16), (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1))
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
        y_buf = buffer_ops.create_buffer_resource(arg_y, num_records_bytes=y_total_bytes)

        x_scale_total_bytes = m_idx * arith.index(scale_k) * arith.index(4)
        x_scale_buf = buffer_ops.create_buffer_resource(arg_x_scale, num_records_bytes=x_scale_total_bytes)

        num_n_scale_blocks = (n_idx + arith.index(scale_block_n - 1)) / arith.index(scale_block_n)
        w_scale_total_bytes = num_n_scale_blocks * arith.index(scale_k) * arith.index(4)
        w_scale_buf = buffer_ops.create_buffer_resource(arg_w_scale, num_records_bytes=w_scale_total_bytes)

        identity_scale = arith.constant(E8M0_IDENTITY, type=T.i32)
        scale_zero = arith.constant(0.0, type=T.f32)

        stages_a = [
            SmemPtr(stage_allocators[i].get_base(), stage_a_data_off[i], T.f8, shape=(lds_a_data_bytes,))
            for i in range(num_buffers)
        ]
        stages_b = [
            SmemPtr(stage_allocators[i].get_base(), stage_b_data_off[i], T.f8, shape=(lds_b_data_bytes,))
            for i in range(num_buffers)
        ]
        stages_a_mem = [p.get() for p in stages_a]
        stages_b_mem = [p.get() for p in stages_b]


        def tdm_issue_x(k_base, buffer_idx):
            """Async HBM→LDS copy of one X tile (tile_m × tile_k) at K-offset
            `k_base`. Lands in LDS stage `buffer_idx`. Completes asynchronously;
            pair with tdm_ops.tensor_wait to synchronize."""
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_x,
                lds_memref=stages_a_mem[buffer_idx],
                global_offset=(blk_m, k_base),
                tensor_shape=(tile_m, tile_k),
                strides=(K, 1),
                tile_shape=(tile_m, tile_k),
                elem_bytes=1,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_A_BYTES,
                num_warps=num_warps,
            )
            tdm_ops.tensor_load_2d(desc)

        def tdm_issue_w(k_base, buffer_idx):
            """Async HBM→LDS copy of one W tile (tile_n × tile_k) at K-offset
            `k_base`. Lands in LDS stage `buffer_idx`."""
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w,
                lds_memref=stages_b_mem[buffer_idx],
                global_offset=(blk_n, k_base),
                tensor_shape=(tile_n, tile_k),
                strides=(K, 1),
                tile_shape=(tile_n, tile_k),
                elem_bytes=1,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_B_BYTES,
                num_warps=num_warps,
            )
            tdm_ops.tensor_load_2d(desc)

        def issue_tdm_loads(k_base, buffer_idx):
            """Issue the two async TDM copies (X + W) for one K-tile into LDS
            stage `buffer_idx`. The s_setprio bump + drop makes the two TDM
            issues dispatch back-to-back without interleaving scalar work."""
            tdm_issue_x(k_base, buffer_idx)
            tdm_issue_w(k_base, buffer_idx)

        w_is_wave_uniform = warp_tile_n <= scale_block_n
        if w_is_wave_uniform:
            wave_n_block = (blk_n + warp_n_base) / arith.index(scale_block_n)

        def issue_raw_scales(k_base):
            """Fire buffer_loads for one K-tile's x_scale + w_scale values —
            no multiply. Returns (x_raw, w_raw) flat lists. Callers typically
            issue 1-2 substages ahead of consumption so HBM round-trip latency
            (~500-1000 cyc) hides behind other work. The compute helper applies
            x_scale * w_scale on-use during its FMA fold.

            Indexing:
                x_raw[sc * wmma_m_rep + wm] = x_scale[row=wm, kb=sc]
                w_raw[sc * wmma_n_rep + wn] = w_scale[n_block=wn, kb=sc]
                                              (all same value if w_is_wave_uniform)
            """
            kb_base = k_base / arith.index(scale_block_k)
            x_raw = []
            w_raw = []
            for sc in range_constexpr(scales_per_tile):
                kb = kb_base + arith.index(sc)
                for wm in range_constexpr(wmma_m_rep):
                    row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                    idx = row * arith.index(scale_k) + kb
                    x_raw.append(buffer_ops.buffer_load(x_scale_buf, idx, vec_width=1, dtype=T.f32))
                if w_is_wave_uniform:
                    idx = wave_n_block * arith.index(scale_k) + kb
                    w_val = buffer_ops.buffer_load(w_scale_buf, idx, vec_width=1, dtype=T.f32)
                    for wn in range_constexpr(wmma_n_rep):
                        w_raw.append(w_val)
                else:
                    for wn in range_constexpr(wmma_n_rep):
                        col = blk_n + warp_n_base + arith.index(wn * WMMA_N) + lane_kgrp * arith.index(8)
                        n_block = col / arith.index(scale_block_n)
                        idx = n_block * arith.index(scale_k) + kb
                        w_raw.append(buffer_ops.buffer_load(w_scale_buf, idx, vec_width=1, dtype=T.f32))
            return x_raw, w_raw

        def issue_raw_scales_for_tile(tile_idx):
            """Issue raw scales for a compile-time tile index."""
            return issue_raw_scales(arith.index(tile_idx * tile_k))

        def issue_raw_scales_for_future_tile_rt(future_tile_rt):
            """Runtime-safe raw-scale prefetch for dynamic main-loop tiles.

            If `future_tile_rt` is out of range, issue a safe in-range load and
            then mask results to zero so no out-of-range scale values propagate.
            """
            future_tile_i32 = arith.index_cast(T.i32, future_tile_rt)
            valid_future = arith.cmpi(
                arith.CmpIPredicate.ult,
                future_tile_i32,
                arith.constant(num_k_tiles, type=T.i32),
            )
            safe_tile_i32 = arith.select(valid_future, future_tile_i32, arith.constant(0, type=T.i32))
            safe_tile_idx = arith.index_cast(T.index, safe_tile_i32)
            safe_k_base = safe_tile_idx * arith.index(tile_k)
            raw_x, raw_w = issue_raw_scales(safe_k_base)
            masked_x = [arith.select(valid_future, v, scale_zero) for v in raw_x]
            masked_w = [arith.select(valid_future, v, scale_zero) for v in raw_w]
            return masked_x, masked_w

        # lane_kgrp selects K-half: kgrp=0 → bytes [0..63], kgrp=1 → [64..127].
        k_half_byte_offset = lane_kgrp * arith.index(64)

        def _compute_lane_bases(warp_base, stride_bytes, num_reps, rep_stride_elems):
            """Compute per-lane LDS byte offsets for loading `num_reps` WMMA
            frags along M or N. Returns a list of base offsets indexed by rep."""
            row_base_bytes = (warp_base + lane16) * arith.index(stride_bytes)
            bases = []
            for rep in range_constexpr(num_reps):
                base = row_base_bytes + arith.index(rep * rep_stride_elems * stride_bytes) + k_half_byte_offset
                bases.append(base)
            return bases

        def _load_frag(lds_memref, lane_base, ks):
            """Load one WMMA frag (16 × b128) from LDS into a vector<16xi32>
            per lane, starting at byte offset (lane_base + ks * WMMA_K)."""
            k_sub_off = arith.index(ks * WMMA_K)
            off = lane_base + k_sub_off
            v0 = lds_load_b128(lds_memref, off)
            v1 = lds_load_b128(lds_memref, off + arith.index(16))
            v2 = lds_load_b128(lds_memref, off + arith.index(32))
            v3 = lds_load_b128(lds_memref, off + arith.index(48))
            v01 = vector.shuffle(v0, v1, list(range(8)))
            v23 = vector.shuffle(v2, v3, list(range(8)))
            return vector.shuffle(v01, v23, list(range(16)))

        a_lane_bases = _compute_lane_bases(warp_m_base, lds_a_stride_bytes, wmma_m_rep, WMMA_M)
        b_lane_bases = _compute_lane_bases(warp_n_base, lds_b_stride_bytes, wmma_n_rep, WMMA_N)

        def load_operand_frags(buffer_idx):
            """Load all A/B frags for one K-tile from LDS stage `buffer_idx`.

            Returns (a_frags, b_frags) with indexing:
                a_frags[ks * wmma_m_rep + wm]
                b_frags[ks * wmma_n_rep + wn]

            Fast (ds_read ~100 cyc per b128) but cannot cross the K-loop
            boundary — frags are loaded once per tile.
            """
            a_lds = stages_a_mem[buffer_idx]
            b_lds = stages_b_mem[buffer_idx]
            a_frags = []
            b_frags = []
            for ks in range_constexpr(k_wmma_steps):
                for wm in range_constexpr(wmma_m_rep):
                    a_frags.append(_load_frag(a_lds, a_lane_bases[wm], ks))
                for wn in range_constexpr(wmma_n_rep):
                    b_frags.append(_load_frag(b_lds, b_lane_bases[wn], ks))
            return a_frags, b_frags

        # ═══════════════════════════════════════════════════════════════════
        # HELPERS: WMMA compute + scale FMA
        # ═══════════════════════════════════════════════════════════════════

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))

        def compute_wmma_with_frags(global_accs, a_frags, b_frags, x_raw, w_raw):
            """2-deep rolling WMMA/FMA pipeline (no Python queue state).

            Pattern per scale block:
              - seed temp0/temp1 (or just temp0 when n_accs == 1),
              - fold temp0 and issue one new temp each step,
              - fold the remaining temps at the end.
            """
            def issue_wmma_temp(sc, wm, wn):
                temp = acc_zero
                for ks_inner in range_constexpr(wmma_steps_per_scale):
                    ks = sc * wmma_steps_per_scale + ks_inner
                    a_frag = a_frags[ks * wmma_m_rep + wm]
                    b_frag = b_frags[ks * wmma_n_rep + wn]
                    # ISA operand order: (B, A, C), reversed from math.
                    temp = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                        T.vec(8, T.f32),
                        b_frag, a_frag, temp,
                        identity_scale, identity_scale,
                        fmtA=0, fmtB=0, scaleAType=0, scaleBType=0,
                    )
                return temp

            def compute_scale(wm, wn, sc_x_base, sc_w_base):
                return arith.mulf(x_raw[sc_x_base + wm], w_raw[sc_w_base + wn])

            def wmma_with_scale(temp, wm, wn, idx, sc_x_base, sc_w_base):
                scale = compute_scale(wm, wn, sc_x_base, sc_w_base)
                scale_vec = vector.broadcast(T.vec(8, T.f32), scale)
                global_accs[idx] = math_dialect.fma(temp, scale_vec, global_accs[idx])

            for sc in range_constexpr(scales_per_tile):
                sc_x_base = sc * wmma_m_rep
                sc_w_base = sc * wmma_n_rep

                wm0, wn0, idx0 = acc_coords[0]
                rocdl.s_setprio(2)
                #hold onto a temp wmma to prevent the next instr from using fma on same vgpr (vnop issue)
                temp0 = issue_wmma_temp(sc, wm0, wn0)
                if n_accs > 1:
                    wm1, wn1, idx1 = acc_coords[1]
                    temp1 = issue_wmma_temp(sc, wm1, wn1)
                #Might not need this since dscnt 0 is gone 
                rocdl.s_setprio(0)
                
                #Case where we only have 1 wmma, usually skipped
                if n_accs == 1:
                    wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                    continue

                # fma -> wmma -> repeat 
                for i in range_constexpr(n_accs - wmma_pipeline_depth):
                    #fma done here
                    wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)

                    wm0, wn0, idx0 = wm1, wn1, idx1
                    temp0 = temp1

                    #new wmma
                    wm1, wn1, idx1 = acc_coords[i + wmma_pipeline_depth]
                    #rocdl.s_setprio(2)
                    temp1 = issue_wmma_temp(sc, wm1, wn1)
                    #rocdl.s_setprio(0)

                # drain remaining 
                wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                wmma_with_scale(temp1, wm1, wn1, idx1, sc_x_base, sc_w_base)

            return global_accs

        # N_ACCS         — global accumulators (always carried)
        # N_A_FRAGS      — cur_a operand frags (reg_preload only)
        # N_B_FRAGS      — cur_b operand frags (reg_preload only)
        # N_CUR_X_RAW    — current tile raw x_scales
        # N_CUR_W_RAW    — current tile raw w_scales
        # N_PREFETCH_X   — next tile raw x_scales (prefetched)
        # N_PREFETCH_W   — next tile raw w_scales (prefetched)
        N_ACCS = n_accs
        N_A_FRAGS = wmma_m_rep * k_wmma_steps
        N_B_FRAGS = wmma_n_rep * k_wmma_steps
        N_CUR_X_RAW = scales_per_tile * wmma_m_rep
        N_CUR_W_RAW = scales_per_tile * wmma_n_rep
        N_PREFETCH_X = N_CUR_X_RAW
        N_PREFETCH_W = N_CUR_W_RAW
        zero_x_raw = [scale_zero] * N_CUR_X_RAW
        zero_w_raw = [scale_zero] * N_CUR_W_RAW

        #This packing/unpacking just sends our vars to the next iteration, stores them cleanly kinda
        def _pack_state_reg_preload(accs_, a_, b_, cur_x_, cur_w_, px, pw):
            return list(accs_) + list(a_) + list(b_) + list(cur_x_) + list(cur_w_) + list(px) + list(pw)

        def _unpack_state_reg_preload(state):
            i = 0
            accs_ = list(state[i:i + N_ACCS]); i += N_ACCS
            a_ = list(state[i:i + N_A_FRAGS]); i += N_A_FRAGS
            b_ = list(state[i:i + N_B_FRAGS]); i += N_B_FRAGS
            cur_x_ = list(state[i:i + N_CUR_X_RAW]); i += N_CUR_X_RAW
            cur_w_ = list(state[i:i + N_CUR_W_RAW]); i += N_CUR_W_RAW
            px = list(state[i:i + N_PREFETCH_X]); i += N_PREFETCH_X
            pw = list(state[i:i + N_PREFETCH_W]); i += N_PREFETCH_W
            return accs_, a_, b_, cur_x_, cur_w_, px, pw

        def _pack_state_no_op_preload(accs_, cur_x_, cur_w_, px, pw):
            return list(accs_) + list(cur_x_) + list(cur_w_) + list(px) + list(pw)

        def _unpack_state_no_op_preload(state):
            i = 0
            accs_ = list(state[i:i + N_ACCS]); i += N_ACCS
            cur_x_ = list(state[i:i + N_CUR_X_RAW]); i += N_CUR_X_RAW
            cur_w_ = list(state[i:i + N_CUR_W_RAW]); i += N_CUR_W_RAW
            px = list(state[i:i + N_PREFETCH_X]); i += N_PREFETCH_X
            pw = list(state[i:i + N_PREFETCH_W]); i += N_PREFETCH_W
            return accs_, cur_x_, cur_w_, px, pw

        #PROLOGUE
        # Step 1: Prologue — issue TDM for the first pre_loaded tiles, fence.
        for i in range_constexpr(prologue_tiles):
            issue_tdm_loads(arith.index(i * tile_k), i)

        # Step 3: initial scale preload. Keep raw scales and multiply on-use in
        # the WMMA/FMA helper to avoid carrying a full combined scale array.
        cur_x_raw, cur_w_raw = issue_raw_scales_for_tile(0)
        if num_k_tiles > 1:
            prefetch_x_raw, prefetch_w_raw = issue_raw_scales_for_tile(1)
        else:
            prefetch_x_raw, prefetch_w_raw = zero_x_raw, zero_w_raw

        accs = [acc_zero] * n_accs

        #MAIN LOOP
        # asm is unrolled in flydsl for range_constexpr, so we see more in asm
        

        if variant == "reg_preload": 

            # Wait for prolouge loads 
            tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING)
            gpu.barrier()

            cur_a, cur_b = load_operand_frags(0)

            # Main loop start
            main_loop_end_k = main_loop_iters * num_buffers * tile_k
            if main_loop_iters > 0:
                init_state = _pack_state_reg_preload(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw
                )
                for iter_k_base, state in range(0, main_loop_end_k, num_buffers * tile_k, init=init_state):
                    cur_accs, cur_a, cur_b, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw = (
                        _unpack_state_reg_preload(state)
                    )
                    tile_idx_rt = iter_k_base / arith.index(tile_k)

                    for substage in range_constexpr(num_buffers):
                        # Substage body order (per pseudocode):
                        #   wmma n → TDM n+2 → wait n+1 → ds_load n+1 → scale rotate.
                        # WMMA first frees cur_a/cur_b/cur_x_raw/cur_w_raw regs before ds_load
                        # writes them back (no v_mov rotate), and TDM issue overlaps
                        # with WMMA's tail
                        cur_accs = compute_wmma_with_frags(cur_accs, cur_a, cur_b, cur_x_raw, cur_w_raw)

                        load_buffer = (substage + num_buffers - 1) % num_buffers
                        load_k = iter_k_base + arith.index((substage + num_buffers - 1) * tile_k)
                        issue_tdm_loads(load_k, load_buffer)

                        tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING)
                        gpu.barrier()

                        next_buffer = (substage + 1) % num_buffers
                        cur_a, cur_b = load_operand_frags(next_buffer)

                        cur_x_raw = prefetch_x_raw
                        cur_w_raw = prefetch_w_raw
                        future_tile_rt = tile_idx_rt + arith.index(substage + 2)
                        prefetch_x_raw, prefetch_w_raw = issue_raw_scales_for_future_tile_rt(future_tile_rt)

                    results = yield _pack_state_reg_preload(
                        cur_accs, cur_a, cur_b, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw
                    )
                accs, cur_a, cur_b, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw = (
                    _unpack_state_reg_preload(results)
                )
            else:
                accs = list(accs)

            # Extra tiles: if main loop iterations doesnt cleanly divide in the 
            # const_expr loop then we need this for the final buffers 
            extra_base_tile = main_loop_iters * num_buffers
            for step in range_constexpr(extra_tiles):
                accs = compute_wmma_with_frags(accs, cur_a, cur_b, cur_x_raw, cur_w_raw)

                load_tile = extra_base_tile + step + num_buffers - 1
                load_buffer = load_tile % num_buffers
                issue_tdm_loads(arith.index(load_tile * tile_k), load_buffer)

                tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING)
                gpu.barrier()

                next_tile = extra_base_tile + step + 1
                next_buffer = next_tile % num_buffers
                cur_a, cur_b = load_operand_frags(next_buffer)

                next_cur_x_raw = prefetch_x_raw
                next_cur_w_raw = prefetch_w_raw
                future_tile = extra_base_tile + step + 2
                if future_tile < num_k_tiles:
                    next_prefetch_x, next_prefetch_w = issue_raw_scales_for_tile(future_tile)
                else:
                    next_prefetch_x, next_prefetch_w = zero_x_raw, zero_w_raw

                cur_x_raw = next_cur_x_raw
                cur_w_raw = next_cur_w_raw
                prefetch_x_raw = next_prefetch_x
                prefetch_w_raw = next_prefetch_w

            # EPILOUGE, this is the usual epilogue wiht no tdm ops, just computes the final 
            drain_base_tile = extra_base_tile + extra_tiles
            for drain_i in range_constexpr(drain_iters):
                accs = compute_wmma_with_frags(accs, cur_a, cur_b, cur_x_raw, cur_w_raw)

                outstanding = (num_buffers - 3 - drain_i) * 2
                tdm_ops.tensor_wait(outstanding)
                gpu.barrier()

                next_tile = drain_base_tile + drain_i + 1
                next_buffer = next_tile % num_buffers
                cur_a, cur_b = load_operand_frags(next_buffer)

                next_cur_x_raw = prefetch_x_raw
                next_cur_w_raw = prefetch_w_raw
                future_tile = drain_base_tile + drain_i + 2
                if future_tile < num_k_tiles:
                    next_prefetch_x, next_prefetch_w = issue_raw_scales_for_tile(future_tile)
                else:
                    next_prefetch_x, next_prefetch_w = zero_x_raw, zero_w_raw

                cur_x_raw = next_cur_x_raw
                cur_w_raw = next_cur_w_raw
                prefetch_x_raw = next_prefetch_x
                prefetch_w_raw = next_prefetch_w

            # final wmma
            accs = compute_wmma_with_frags(accs, cur_a, cur_b, cur_x_raw, cur_w_raw)

        else:  # variant 1, not tested a lot 
            main_loop_end_k = main_loop_iters * num_buffers * tile_k
            if main_loop_iters > 0:
                init_state = _pack_state_no_op_preload(accs, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw)
                for iter_k_base, state in range(0, main_loop_end_k, num_buffers * tile_k, init=init_state):
                    cur_accs, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw = (
                        _unpack_state_no_op_preload(state)
                    )
                    tile_idx_rt = iter_k_base / arith.index(tile_k)

                    for substage in range_constexpr(num_buffers):
                        load_buffer = (substage + num_buffers - 1) % num_buffers
                        load_k = iter_k_base + arith.index((substage + num_buffers - 1) * tile_k)
                        issue_tdm_loads(load_k, load_buffer)

                        compute_stage = substage % num_buffers
                        fresh_a, fresh_b = load_operand_frags(compute_stage)

                        cur_accs = compute_wmma_with_frags(cur_accs, fresh_a, fresh_b, cur_x_raw, cur_w_raw)

                        tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING)
                        gpu.barrier()

                        cur_x_raw = prefetch_x_raw
                        cur_w_raw = prefetch_w_raw

                        future_tile_rt = tile_idx_rt + arith.index(substage + 2)
                        prefetch_x_raw, prefetch_w_raw = issue_raw_scales_for_future_tile_rt(future_tile_rt)

                    results = yield _pack_state_no_op_preload(
                        cur_accs, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw
                    )
                accs, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw = (
                    _unpack_state_no_op_preload(results)
                )
            else:
                accs = list(accs)

            # Extra tiles: if main loop iterations doesnt cleanly divide in the 
            # const_expr loop then we need this for the final buffers 
            extra_base_tile = main_loop_iters * num_buffers
            for step in range_constexpr(extra_tiles):
                load_tile = extra_base_tile + step + num_buffers - 1
                load_buffer = load_tile % num_buffers
                issue_tdm_loads(arith.index(load_tile * tile_k), load_buffer)

                compute_stage = (extra_base_tile + step) % num_buffers
                fresh_a, fresh_b = load_operand_frags(compute_stage)

                accs = compute_wmma_with_frags(accs, fresh_a, fresh_b, cur_x_raw, cur_w_raw)

                tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING)
                gpu.barrier()

                cur_x_raw = prefetch_x_raw
                cur_w_raw = prefetch_w_raw

                future_tile = extra_base_tile + step + 2
                if future_tile < num_k_tiles:
                    prefetch_x_raw, prefetch_w_raw = issue_raw_scales_for_tile(future_tile)
                else:
                    prefetch_x_raw, prefetch_w_raw = zero_x_raw, zero_w_raw

            # Drain
            drain_base_tile = extra_base_tile + extra_tiles
            for drain_i in range_constexpr(drain_iters):
                outstanding = (num_buffers - 3 - drain_i) * 2
                tdm_ops.tensor_wait(outstanding)
                gpu.barrier()

                compute_stage = (drain_base_tile + drain_i) % num_buffers
                fresh_a, fresh_b = load_operand_frags(compute_stage)

                accs = compute_wmma_with_frags(accs, fresh_a, fresh_b, cur_x_raw, cur_w_raw)

                cur_x_raw = prefetch_x_raw
                cur_w_raw = prefetch_w_raw

                future_tile = drain_base_tile + drain_i + 2
                if future_tile < num_k_tiles:
                    prefetch_x_raw, prefetch_w_raw = issue_raw_scales_for_tile(future_tile)
                else:
                    prefetch_x_raw, prefetch_w_raw = zero_x_raw, zero_w_raw

            # Final wmma 
            final_tile = drain_base_tile + drain_iters
            final_compute_stage = final_tile % num_buffers
            fresh_a, fresh_b = load_operand_frags(final_compute_stage)
            accs = compute_wmma_with_frags(accs, fresh_a, fresh_b, cur_x_raw, cur_w_raw)


        # Step 4: convert f32 accs to out_dtype, buffer_store to Y.
        if num_buffers > 2:
            rocdl.sched_barrier(0)

        out_elem = T.bf16 if out_dtype == "bf16" else T.f16 if out_dtype == "fp16" else None
        is_half_out = out_dtype in ("bf16", "fp16")

        if use_tdm_store:
            d_lds_elem_ty = T.bf16 if out_dtype != "fp16" else T.f16
            d_lds_elems = total_d_bytes // 2
            d_smem = SmemPtr(d_lds_allocator.get_base(), 0, d_lds_elem_ty,
                             shape=(d_lds_elems,))
            d_lds_buffer = d_smem.get()

            row_lds = warp_m_base + lane16            # warp_m_base = wave_m_idx * warp_tile_m
            col_lds = warp_n_base + lane_kgrp * arith.index(8)  # bf16 col within row
            d_lane_base = (
                row_lds * arith.index(_lds_d_stride_elems_d)
                + col_lds
            )
            if not is_half_out:
                d_lane_base = (
                    row_lds * arith.index(_lds_d_stride_elems_d)
                    + warp_n_base * arith.index(elem_bytes_d // 2)
                    + lane_kgrp * arith.index(4 * elem_bytes_d)
                )

            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    imm = (wm * WMMA_M * _lds_d_stride_elems_d
                           + wn * _n_col_d_elems_d)
                    store_acc_vec8_to_lds(
                        d_lds_buffer, d_lane_base, imm, accs[idx], out_elem=out_elem,
                    )

            rocdl.s_wait_dscnt(0)
            gpu.barrier()

            d_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_y,
                lds_memref=d_lds_buffer,
                global_offset=(blk_m, blk_n),
                tensor_shape=(tile_m, tile_n),
                strides=(N, 1),
                tile_shape=(tile_m, tile_n),
                elem_bytes=elem_bytes_d,
                pad_interval=tile_n,
                pad_amount=LDS_PAD_D_BYTES // elem_bytes_d,
                num_warps=num_warps,
                for_store=True,
            )
            tdm_ops.tensor_store_2d(d_desc)
            tdm_ops.tensor_wait(0)
        else:
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                    col_base = blk_n + warp_n_base + arith.index(wn * WMMA_N) + lane_kgrp * arith.index(8)

                    if is_half_out:
                        c_off_bytes = (row * n_stride + col_base) * arith.index(elem_bytes_d)
                        store_acc_vec8_to_buffer(
                            accs[idx], y_buf, c_off_bytes,
                            out_elem=out_elem, offset_is_bytes=True,
                        )
                    else:
                        offsets = []
                        for half in range_constexpr(2):
                            col = col_base + arith.index(half * 4)
                            offsets.append(row * n_stride + col)
                        store_acc_vec8_to_buffer(accs[idx], y_buf, offsets)


    cache_tag = (
        K, N, tile_m, tile_n, tile_k, m_warp, n_warp, scale_block_k, scale_block_n,
        num_buffers, effective_waves_per_eu, l2_prefetch_distance, out_dtype, variant,
        use_tdm_store,
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
        all_allocators = list(stage_allocators)
        if use_tdm_store:
            all_allocators.append(d_lds_allocator)
        with ir.InsertionPoint(ctx.gpu_module_body):
            for alloc in all_allocators:
                alloc.finalized = False
            for alloc in all_allocators:
                alloc.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))
        gy = _raw((idx_n + arith.index(tile_n - 1)) / arith.index(tile_n))

        launcher = kernel_gemm_a8w8_blockscale(arg_y, arg_x, arg_w, arg_x_scale, arg_w_scale, i32_m, i32_n)

        if effective_waves_per_eu is not None:
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    wpe = int(effective_waves_per_eu)
                    if wpe >= 1:
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), wpe
                        )

        flat_wg_attr = ir.StringAttr.get(f"{block_threads},{block_threads}")
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

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
    variant: str = "reg_preload",
    use_tdm_store: bool = False,
):
    """Compute Y = (X @ W^T) with per-block f32 scales (A8W8 blockscale).

    variant: "reg_preload" (default) or "no_op_preload".
      - "reg_preload"   : loop-carry cur_a/cur_b operand frags across iters.
                          Lowest cycle count when VGPR budget allows.
      - "no_op_preload" : load operand frags fresh from LDS each iter.
                          ~256 VGPRs cheaper; enables larger M/N tiles.
    """
    assert x.ndim == 2 and w.ndim == 2, "X and W must be 2D"
    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w, f"K mismatch: X has {K}, W has {K_w}"

    assert x_scale.ndim == 2 and w_scale.ndim == 2, "scales must be 2D"
    assert x_scale.shape[0] == M, f"x_scale rows {x_scale.shape[0]} != M {M}"
    scale_k_x = x_scale.shape[1]
    scale_n, scale_k_w = w_scale.shape
    assert scale_k_x == scale_k_w, f"scale_k mismatch: x_scale has {scale_k_x}, w_scale has {scale_k_w}"
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

    K_padded = ((K + tile_k - 1) // tile_k) * tile_k
    if K_padded != K:
        pad_size = K_padded - K
        x = torch.nn.functional.pad(x, (0, pad_size))
        w = torch.nn.functional.pad(w, (0, pad_size))
        new_scale_k = K_padded // scale_block_k_derived
        scale_pad = new_scale_k - scale_k
        if scale_pad > 0:
            x_scale = torch.nn.functional.pad(x_scale, (0, scale_pad))
            w_scale = torch.nn.functional.pad(w_scale, (0, scale_pad))
        K = K_padded

    # Pad N up to tile_n so the kernel's WMMAs and stores land inside
    # the allocated output
    N_stride = ((N + tile_n - 1) // tile_n) * tile_n

    if y is not None:
        assert y.shape == (M, N), f"y shape {y.shape} != ({M}, {N})"
        assert y.dtype == dtype, f"y dtype {y.dtype} != {dtype}"

    if N_stride != N:
        y_buf = torch.empty((M, N_stride), dtype=dtype, device=x.device)
    elif y is not None:
        y_buf = y
    else:
        y_buf = torch.empty((M, N), dtype=dtype, device=x.device)

    launcher = compile_gemm_a8w8_blockscale(
        K=K,
        N=N_stride,
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
        variant=variant,
        use_tdm_store=use_tdm_store,
    )

    stream = torch.cuda.current_stream(device=x.device).cuda_stream
    launcher(y_buf, x, w, x_scale, w_scale, M, N_stride, stream=stream)

    if N_stride != N:
        result = y_buf[:, :N]
        if y is not None:
            y.copy_(result)
            return y
        return result
    return y_buf


__all__ = [
    "compile_gemm_a8w8_blockscale",
    "gemm_a8w8_blockscale",
]
