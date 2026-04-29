# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, tdm_ops, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value
from flydsl.expr import idx2crd
from typing import Optional

# WMMA 16×16×32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32

# LDS padding per row (8 elements -> 16 bytes)
LDS_PAD_A = 8
LDS_PAD_B = 8

_STAGE_NAMES = tuple(f"stage{i}" for i in range(16))


def _apply_activation_scalar(val, activation: str):
    """Apply activation to a single f32 scalar."""
    from flydsl._mlir.dialects import math as _math

    if activation == "relu":
        zero = arith.constant(0.0, type=T.f32)
        return arith.select(val > zero, val, zero)

    elif activation in ("silu", "silu_exp2"):
        neg = arith.constant(0.0, type=T.f32) - val
        exp_neg = _math.exp(neg)
        one = arith.constant(1.0, type=T.f32)
        denom = one + exp_neg
        return val / denom

    elif activation == "gelu":
        import math

        inv_sqrt2 = arith.constant(1.0 / math.sqrt(2.0), type=T.f32)
        scaled = val * inv_sqrt2
        erf_val = _math.erf(scaled)
        one = arith.constant(1.0, type=T.f32)
        half = arith.constant(0.5, type=T.f32)
        return half * val * (one + erf_val)

    elif activation == "gelu_tanh":
        import math

        sqrt_2_over_pi = arith.constant(math.sqrt(2.0 / math.pi), type=T.f32)
        coeff = arith.constant(0.044715, type=T.f32)
        one = arith.constant(1.0, type=T.f32)
        two = arith.constant(2.0, type=T.f32)
        half = arith.constant(0.5, type=T.f32)
        x3 = val * val * val
        inner = sqrt_2_over_pi * (val + coeff * x3)
        # tanh(z) = 1 - 2/(1 + exp(2z))
        exp2x = _math.exp(two * inner)
        tanh_val = one - two / (one + exp2x)
        return half * val * (one + tanh_val)

    else:
        return val


def compile_gemm_a16w16(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 32,
    m_warp: int = 2,
    n_warp: int = 4,
    in_dtype: str = "fp16",
    out_dtype: str = None,
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 2,
    activation: Optional[str] = None,
    add_bias: bool = False,
    physical_mk: bool = True,  # True=M-major (row-major X), False=K-major (col-major X)
    physical_kn: bool = False,  # False=N-major (row-major W), True=K-major (col-major/transposed W)
):
    """Compile and return a launch function for the A16W16 GEMM kernel.
    Returns a callable: launch_fn(y, x, w, bias, M, N, stream=stream)
    """
    _ = (M, N)

    # ── Input validation ──
    if num_buffers < 2:
        raise ValueError(f"num_buffers must be >= 2, got {num_buffers}")
    if in_dtype not in ("fp16", "bf16"):
        raise ValueError(f"in_dtype must be 'fp16' or 'bf16', got {in_dtype!r}")

    effective_waves_per_eu = waves_per_eu
    is_f16 = in_dtype == "fp16"

    if out_dtype is None:
        out_dtype = "f16" if is_f16 else "bf16"
    if out_dtype not in ("f32", "f16", "bf16"):
        raise ValueError(
            f"out_dtype must be 'f32', 'f16', or 'bf16', got {out_dtype!r}"
        )

    elem_bytes = 2
    elem_bytes_d = 2 if out_dtype in ("f16", "bf16") else 4
    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE

    # ── Tile dimension validation ──
    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if (tile_k & (tile_k - 1)) != 0:
        raise ValueError(f"tile_k must be a power of 2 for TDM, got {tile_k}")

    # ── Physical layout validation ──
    if physical_kn:
        if N == 0:
            raise ValueError(
                "N must be specified (> 0) at compile time when physical_kn=True, "
                "because it is used as the TDM stride for the (K, N) weight layout"
            )
        if (tile_n & (tile_n - 1)) != 0:
            raise ValueError(
                f"tile_n must be a power of 2 when physical_kn=True "
                f"(TDM pad_interval requirement), got {tile_n}"
            )
    if not physical_mk:
        if M == 0:
            raise ValueError(
                "M must be specified (> 0) at compile time when physical_mk=False, "
                "because it is used as the TDM stride for the (K, M) activation layout"
            )
        if (tile_m & (tile_m - 1)) != 0:
            raise ValueError(
                f"tile_m must be a power of 2 when physical_mk=False "
                f"(TDM pad_interval requirement), got {tile_m}"
            )

    # ── Warp tile dimensions ──
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N}")

    # ── K-dimension tiling ──
    num_k_tiles = K // tile_k
    if num_k_tiles < num_buffers - 1:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers - 1}, "
            f"got {num_k_tiles} (K={K}, tile_k={tile_k})"
        )

    # ── Architecture check ──
    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    wmma_op = rocdl.wmma_f32_16x16x32_f16 if is_f16 else rocdl.wmma_f32_16x16x32_bf16

    # ── Compute repetition counts ──
    k_wmma_steps = tile_k // WMMA_K

    def _elem_type():
        return T.f16 if is_f16 else T.bf16

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep
    n_frags_per_tile = k_wmma_steps * (wmma_n_rep + wmma_m_rep)

    # ── LDS layout ──
    # physical_mk=True:  A in LDS as [tile_m, tile_k + pad]
    # physical_mk=False: A in LDS as [tile_k, tile_m + pad] (K-major)
    if physical_mk:
        lds_a_stride = tile_k + LDS_PAD_A
        lds_a_elems = tile_m * lds_a_stride + LDS_PAD_A
    else:
        lds_a_stride = tile_m + LDS_PAD_A
        lds_a_elems = tile_k * lds_a_stride + LDS_PAD_A

    if physical_kn:
        lds_b_stride = tile_n + LDS_PAD_B
        lds_b_elems = tile_k * lds_b_stride + LDS_PAD_B
    else:
        lds_b_stride = tile_k + LDS_PAD_B
        lds_b_elems = tile_n * lds_b_stride + LDS_PAD_B

    buf_size_elems = lds_a_elems + lds_b_elems

    # ── LDS allocation per pipeline stage ──
    stage_allocators = []
    stage_a_offsets = []
    stage_b_offsets = []
    for i in range(num_buffers):
        name = _STAGE_NAMES[i]
        alloc = SmemAllocator(None, arch=gpu_arch, global_sym_name=f"a16w16_{name}")
        off = alloc._align(alloc.ptr, 16)
        alloc.ptr = off + buf_size_elems * elem_bytes
        stage_allocators.append(alloc)
        stage_a_offsets.append(off)
        stage_b_offsets.append(off + lds_a_elems * elem_bytes)

    # ── Pipeline iteration counts ──
    pre_loaded = num_buffers - 1
    main_loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    tail_tiles = (num_k_tiles - pre_loaded) % num_buffers + pre_loaded

    @flyc.kernel
    def kernel_gemm_a16w16(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()

        # ── Thread/block indexing ──
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

        # ── Thread-to-warp decomposition ──
        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16), (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1)
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
        elem_ty = _elem_type()

        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        n_stride = arith.index_cast(T.index, i32_n.ir_value())

        # Buffer resource descriptors
        y_nrec = m_idx * n_stride * arith.index(elem_bytes_d)
        y_rsrc = buffer_ops.create_buffer_resource(arg_y, num_records_bytes=y_nrec)
        if add_bias:
            bias_rsrc = buffer_ops.create_buffer_resource(arg_bias, max_size=True)

        def make_a_desc(k_base, lds_a_mem_ref):
            """TDM descriptor for A tile. Swaps dims when physical_mk=False (K-major)."""
            if physical_mk:
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_x,
                    lds_memref=lds_a_mem_ref,
                    global_offset=(blk_m, k_base),
                    tensor_shape=(tile_m, tile_k),
                    strides=(K, 1),
                    tile_shape=(tile_m, tile_k),
                    elem_bytes=elem_bytes,
                    pad_interval=tile_k,
                    pad_amount=LDS_PAD_A,
                    num_warps=num_warps,
                )
            else:
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_x,
                    lds_memref=lds_a_mem_ref,
                    global_offset=(k_base, blk_m),
                    tensor_shape=(tile_k, tile_m),
                    strides=(M, 1),
                    tile_shape=(tile_k, tile_m),
                    elem_bytes=elem_bytes,
                    pad_interval=tile_m,
                    pad_amount=LDS_PAD_A,
                    num_warps=num_warps,
                )

        def make_b_desc(k_base, lds_b_mem_ref):
            """TDM descriptor for B tile. Swaps dims when physical_kn=True (K-major)."""
            if physical_kn:
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_w,
                    lds_memref=lds_b_mem_ref,
                    global_offset=(k_base, blk_n),
                    tensor_shape=(tile_k, tile_n),
                    strides=(N, 1),
                    tile_shape=(tile_k, tile_n),
                    elem_bytes=elem_bytes,
                    pad_interval=tile_n,
                    pad_amount=LDS_PAD_B,
                    num_warps=num_warps,
                )
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w,
                lds_memref=lds_b_mem_ref,
                global_offset=(blk_n, k_base),
                tensor_shape=(tile_n, tile_k),
                strides=(K, 1),
                tile_shape=(tile_n, tile_k),
                elem_bytes=elem_bytes,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_B,
                num_warps=num_warps,
            )

        def issue_tdm_load(desc):
            tdm_ops.tensor_load_2d(desc)

        def copy_a_to_lds(k_base, lds_a_mem_ref):
            issue_tdm_load(make_a_desc(k_base, lds_a_mem_ref))

        def copy_b_to_lds(k_base, lds_b_mem_ref):
            issue_tdm_load(make_b_desc(k_base, lds_b_mem_ref))

        def _get_lds_memref(lds_ptr):
            if isinstance(lds_ptr, SmemPtr):
                return get_op_result_or_value(lds_ptr.get())
            return get_op_result_or_value(lds_ptr)

        def _precompute_lane_bases(lds_ptr, warp_base, reps, lds_stride):
            """Pre-compute LDS base addresses for each WMMA rep (M-major layout)."""
            lds_buffer = _get_lds_memref(lds_ptr)
            row_stride_off = (warp_base + lane16) * arith.index(lds_stride)
            k_lane_off = lane_kgrp * arith.index(8)
            bases = []
            for rep in range_constexpr(reps):
                base = (
                    row_stride_off + arith.index(rep * WMMA_M * lds_stride) + k_lane_off
                )
                bases.append(base)
            return lds_buffer, bases

        def _precompute_a_lane_bases(lds_ptr):
            """A fragment lane bases. Uses transpose-load addressing when K-major."""
            if physical_mk:
                return _precompute_lane_bases(
                    lds_ptr, warp_m_base, wmma_m_rep, lds_a_stride
                )

            lds_buffer = _get_lds_memref(lds_ptr)
            lane8 = lane16 % arith.index(8)
            lane_mgrp = lane16 / arith.index(8)
            k_lane_off = (lane_kgrp * arith.index(8) + lane8) * arith.index(
                lds_a_stride
            )
            m_lane_off = lane_mgrp * arith.index(8)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                m_col = warp_m_base + arith.index(wm * WMMA_M) + m_lane_off
                bases.append(k_lane_off + m_col)
            return lds_buffer, bases

        def _precompute_b_lane_bases(lds_ptr):
            """B fragment lane bases. Uses transpose-load addressing when K-major."""
            if not physical_kn:
                return _precompute_lane_bases(
                    lds_ptr, warp_n_base, wmma_n_rep, lds_b_stride
                )

            lds_buffer = _get_lds_memref(lds_ptr)
            lane8 = lane16 % arith.index(8)
            lane_ngrp = lane16 / arith.index(8)
            k_lane_off = (lane_kgrp * arith.index(8) + lane8) * arith.index(
                lds_b_stride
            )
            n_lane_off = lane_ngrp * arith.index(8)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                n_col = warp_n_base + arith.index(wn * WMMA_N) + n_lane_off
                bases.append(k_lane_off + n_col)
            return lds_buffer, bases

        def load_wmma_frag_tr(lds_buffer, b_lane_base, ks):
            """Load WMMA B fragment via ds_load_tr16_b128 (K-major LDS)."""
            vec8_ty = ir.VectorType.get([8], elem_ty)
            results = []
            for k_half in range_constexpr(2):
                k_row_off = (ks * WMMA_K + k_half * 16) * lds_b_stride
                elem_off = b_lane_base + arith.index(k_row_off)
                v = rocdl.lds_transpose_load(vec8_ty, lds_buffer, elem_off, elem_bytes)
                results.append(v)
            return vector.shuffle(results[0], results[1], list(range(16)))

        def load_wmma_frag(lds_buffer, lane_base, ks):
            """Load WMMA fragment via ds_read_b128 (M/N-major LDS)."""
            vec8_ty = ir.VectorType.get([8], elem_ty)
            off0 = lane_base + arith.index(ks * WMMA_K)
            off1 = lane_base + arith.index(ks * WMMA_K + 16)
            v0 = vector.load_op(vec8_ty, lds_buffer, [off0])
            v1 = vector.load_op(vec8_ty, lds_buffer, [off1])
            return vector.shuffle(v0, v1, list(range(16)))

        _load_b_frag = load_wmma_frag_tr if physical_kn else load_wmma_frag

        def load_wmma_frag_tr_a(lds_buffer, a_lane_base, ks):
            """Load WMMA A fragment via ds_load_tr16_b128 (K-major LDS)."""
            vec8_ty = ir.VectorType.get([8], elem_ty)
            results = []
            for k_half in range_constexpr(2):
                k_row_off = (ks * WMMA_K + k_half * 16) * lds_a_stride
                elem_off = a_lane_base + arith.index(k_row_off)
                v = rocdl.lds_transpose_load(vec8_ty, lds_buffer, elem_off, elem_bytes)
                results.append(v)
            return vector.shuffle(results[0], results[1], list(range(16)))

        _load_a_frag = load_wmma_frag_tr_a if not physical_mk else load_wmma_frag

        def load_tile_frags(lds_a_ptr, lds_b_ptr):
            """Bulk-load all k-slice fragments for one tile (no overlap)."""
            a_buf, a_bases = _precompute_a_lane_bases(lds_a_ptr)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b_ptr)
            frags = []
            for ks in range_constexpr(k_wmma_steps):
                for wn in range_constexpr(wmma_n_rep):
                    frags.append(_load_b_frag(b_buf, b_bases[wn], ks))
                for wm in range_constexpr(wmma_m_rep):
                    frags.append(_load_a_frag(a_buf, a_bases[wm], ks))
            return frags

        def wmma_tile(accs_in, tile_frags):
            """Execute all WMMAs for one tile using pre-loaded fragments."""
            current_accs = list(accs_in)
            for ks in range_constexpr(k_wmma_steps):
                base = ks * (wmma_n_rep + wmma_m_rep)
                b_frags = [tile_frags[base + wn] for wn in range_constexpr(wmma_n_rep)]
                a_frags = [
                    tile_frags[base + wmma_n_rep + wm]
                    for wm in range_constexpr(wmma_m_rep)
                ]
                for wm in range_constexpr(wmma_m_rep):
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        # ISA operand order: (B, A, C) — reversed from math
                        current_accs[idx] = wmma_op(
                            T.vec(8, T.f32),
                            b_frags[wn],
                            a_frags[wm],
                            current_accs[idx],
                            signA=False,
                            signB=False,
                            modC=0,
                            reuseA=False,
                            reuseB=(wn > 0),
                        ).result
            return current_accs

        def compute_and_prefetch(accs_in, cur_frags, next_lds_a_ptr, next_lds_b_ptr):
            """Interleave WMMA on current frags with LDS loads for next tile."""
            current_accs = list(accs_in)
            next_a_buf, next_a_bases = _precompute_a_lane_bases(next_lds_a_ptr)
            next_b_buf, next_b_bases = _precompute_b_lane_bases(next_lds_b_ptr)
            next_frags = []

            for ks in range_constexpr(k_wmma_steps):
                # Phase 1: ds_read for next tile
                for wn in range_constexpr(wmma_n_rep):
                    next_frags.append(_load_b_frag(next_b_buf, next_b_bases[wn], ks))
                for wm in range_constexpr(wmma_m_rep):
                    next_frags.append(_load_a_frag(next_a_buf, next_a_bases[wm], ks))

                # Phase 2: WMMA for curr tile
                base = ks * (wmma_n_rep + wmma_m_rep)
                b_frags = [cur_frags[base + wn] for wn in range_constexpr(wmma_n_rep)]
                a_frags = [
                    cur_frags[base + wmma_n_rep + wm]
                    for wm in range_constexpr(wmma_m_rep)
                ]
                for wm in range_constexpr(wmma_m_rep):
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        current_accs[idx] = wmma_op(
                            T.vec(8, T.f32),
                            b_frags[wn],
                            a_frags[wm],
                            current_accs[idx],
                            signA=False,
                            signB=False,
                            modC=0,
                            reuseA=False,
                            reuseB=(wn > 0),
                        ).result

            return current_accs, next_frags

        def _l2_prefetch(k_base):
            """Issue L2 cache prefetch hints for a future K-tile."""
            if l2_prefetch_distance <= 0:
                return
            pf_k = k_base + arith.index(l2_prefetch_distance * tile_k)

            if physical_mk:
                tdm_ops.l2_prefetch_tile(
                    arg_x,
                    (blk_m, pf_k),
                    (tile_m, tile_k),
                    (K, 1),
                    elem_bytes=elem_bytes,
                    thread_id=tx,
                    block_threads=block_threads,
                )
            else:
                tdm_ops.l2_prefetch_tile(
                    arg_x,
                    (pf_k, blk_m),
                    (tile_k, tile_m),
                    (M, 1),
                    elem_bytes=elem_bytes,
                    thread_id=tx,
                    block_threads=block_threads,
                )
            if physical_kn:
                tdm_ops.l2_prefetch_tile(
                    arg_w,
                    (pf_k, blk_n),
                    (tile_k, tile_n),
                    (N, 1),
                    elem_bytes=elem_bytes,
                    thread_id=tx,
                    block_threads=block_threads,
                )
            else:
                tdm_ops.l2_prefetch_tile(
                    arg_w,
                    (blk_n, pf_k),
                    (tile_n, tile_k),
                    (K, 1),
                    elem_bytes=elem_bytes,
                    thread_id=tx,
                    block_threads=block_threads,
                )

        _half_out = out_dtype in ("f16", "bf16")
        _out_elem = (
            T.f16 if out_dtype == "f16" else (T.bf16 if out_dtype == "bf16" else None)
        )
        _bias_elem = T.f16 if is_f16 else T.bf16

        def _widen_to_f32(val):
            from flydsl._mlir.dialects import arith as _std_arith

            return _std_arith.ExtFOp(T.f32, _raw(val)).result

        def _apply_bias_and_activation(accs):
            """Add bias and/or apply activation to accumulators."""
            if not add_bias and activation is None:
                return accs

            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    acc = accs[idx]

                    if add_bias:
                        col_base = (
                            blk_n
                            + warp_n_base
                            + arith.index(wn * WMMA_N)
                            + lane_kgrp * arith.index(8)
                        )
                        col_base_i32 = arith.index_cast(T.i32, col_base)
                        col_base_hi = col_base_i32 + arith.constant(4, type=T.i32)

                        bv_lo = buffer_ops.buffer_load(
                            bias_rsrc, col_base_i32, vec_width=4, dtype=_bias_elem
                        )
                        bv_hi = buffer_ops.buffer_load(
                            bias_rsrc, col_base_hi, vec_width=4, dtype=_bias_elem
                        )

                        bias_elems = []
                        for i in range_constexpr(4):
                            b_val = vector.extract(
                                bv_lo, static_position=[i], dynamic_position=[]
                            )
                            bias_elems.append(_widen_to_f32(b_val))
                        for i in range_constexpr(4):
                            b_val = vector.extract(
                                bv_hi, static_position=[i], dynamic_position=[]
                            )
                            bias_elems.append(_widen_to_f32(b_val))

                        bias_vec = vector.from_elements(T.vec(8, T.f32), bias_elems)
                        acc = acc + bias_vec

                    if activation is not None:
                        new_elems = []
                        for i in range_constexpr(8):
                            val = vector.extract(
                                acc, static_position=[i], dynamic_position=[]
                            )
                            val = _apply_activation_scalar(val, activation)
                            new_elems.append(val)
                        acc = vector.from_elements(T.vec(8, T.f32), new_elems)

                    accs[idx] = acc
            return accs

        def epilogue_stores(final_accs):
            """Write accumulators to global output Y."""
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
                        h_vec = arith.trunc_f(T.vec(8, _out_elem), final_accs[idx])
                        i32_vec = vector.bitcast(T.vec(4, T.i32), h_vec)
                        c_off_bytes = (row * n_stride + col_base) * arith.index(
                            elem_bytes_d
                        )
                        buffer_ops.buffer_store(
                            i32_vec, y_rsrc, c_off_bytes, offset_is_bytes=True
                        )
                    else:
                        for half in range_constexpr(2):
                            vals = [
                                vector.extract(
                                    final_accs[idx],
                                    static_position=[half * 4 + vi],
                                    dynamic_position=[],
                                )
                                for vi in range_constexpr(4)
                            ]
                            vec4 = vector.from_elements(T.vec(4, T.f32), vals)
                            col = col_base + arith.index(half * 4)
                            c_off = row * n_stride + col
                            buffer_ops.buffer_store(vec4, y_rsrc, c_off)

        # Step 1: Initialize accumulators
        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        # Step 2: Set up LDS buffer pointers
        base_ptrs = [sa.get_base() for sa in stage_allocators]
        stages_a = [
            SmemPtr(base_ptrs[i], stage_a_offsets[i], elem_ty, shape=(lds_a_elems,))
            for i in range_constexpr(num_buffers)
        ]
        stages_b = [
            SmemPtr(base_ptrs[i], stage_b_offsets[i], elem_ty, shape=(lds_b_elems,))
            for i in range_constexpr(num_buffers)
        ]
        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]

        # Step 3: Prologue — pre-load tiles into LDS
        for i in range_constexpr(pre_loaded):
            copy_a_to_lds(arith.index(i * tile_k), stages_a_mem[i])
            copy_b_to_lds(arith.index(i * tile_k), stages_b_mem[i])

        # L2 prefetch prologue
        if l2_prefetch_distance > 0:
            for pf_i in range_constexpr(min(l2_prefetch_distance, num_k_tiles)):
                pf_k = arith.index(pf_i * tile_k)
                if physical_mk:
                    tdm_ops.l2_prefetch_tile(
                        arg_x,
                        (blk_m, pf_k),
                        (tile_m, tile_k),
                        (K, 1),
                        elem_bytes=elem_bytes,
                        thread_id=tx,
                        block_threads=block_threads,
                    )
                else:
                    tdm_ops.l2_prefetch_tile(
                        arg_x,
                        (pf_k, blk_m),
                        (tile_k, tile_m),
                        (M, 1),
                        elem_bytes=elem_bytes,
                        thread_id=tx,
                        block_threads=block_threads,
                    )
                if physical_kn:
                    tdm_ops.l2_prefetch_tile(
                        arg_w,
                        (pf_k, blk_n),
                        (tile_k, tile_n),
                        (N, 1),
                        elem_bytes=elem_bytes,
                        thread_id=tx,
                        block_threads=block_threads,
                    )
                else:
                    tdm_ops.l2_prefetch_tile(
                        arg_w,
                        (blk_n, pf_k),
                        (tile_n, tile_k),
                        (K, 1),
                        elem_bytes=elem_bytes,
                        thread_id=tx,
                        block_threads=block_threads,
                    )

        # Prepare first TDM descriptors for main loop
        if main_loop_iters > 0:
            _first_load_stage_0 = pre_loaded % num_buffers
            _first_load_k_0 = pre_loaded * tile_k
            pending_desc_a = make_a_desc(
                arith.index(_first_load_k_0), stages_a_mem[_first_load_stage_0]
            )
            pending_desc_b = make_b_desc(
                arith.index(_first_load_k_0), stages_b_mem[_first_load_stage_0]
            )

        main_end = main_loop_iters * num_buffers * tile_k

        # Wait for pre-loaded tiles
        tdm_ops.tensor_wait(2 * (num_buffers - 2))
        gpu.barrier()

        # Pre-load fragments for first compute tile
        cur_frags = load_tile_frags(stages_a[0], stages_b[0])

        # Step 4: Main loop
        if main_loop_iters > 0:
            init_descs = [
                pending_desc_a.dgroup0,
                pending_desc_a.dgroup1,
                pending_desc_b.dgroup0,
                pending_desc_b.dgroup1,
            ]

            for iv, state in range(
                0,
                main_end,
                num_buffers * tile_k,
                init=list(accs) + init_descs + cur_frags,
            ):
                accs_in = list(state[:n_accs])
                cur_desc_a = tdm_ops.TDMDescriptor2D(state[n_accs], state[n_accs + 1])
                cur_desc_b = tdm_ops.TDMDescriptor2D(
                    state[n_accs + 2], state[n_accs + 3]
                )
                cur_frags = list(state[n_accs + 4 : n_accs + 4 + n_frags_per_tile])

                for s in range_constexpr(num_buffers):
                    issue_tdm_load(cur_desc_a)
                    issue_tdm_load(cur_desc_b)

                    if s == num_buffers - 1:
                        _l2_prefetch(iv + arith.index(s * tile_k))

                    if s < num_buffers - 1:
                        _next_load_stage = (s + 1 + pre_loaded) % num_buffers
                        _next_load_k_off = (s + 1 + pre_loaded) * tile_k
                        cur_desc_a = make_a_desc(
                            iv + arith.index(_next_load_k_off),
                            stages_a_mem[_next_load_stage],
                        )
                        cur_desc_b = make_b_desc(
                            iv + arith.index(_next_load_k_off),
                            stages_b_mem[_next_load_stage],
                        )
                    else:
                        _next_load_stage = pre_loaded % num_buffers
                        _next_load_k_off = pre_loaded * tile_k
                        _next_step = num_buffers * tile_k
                        cur_desc_a = make_a_desc(
                            iv + arith.index(_next_step + _next_load_k_off),
                            stages_a_mem[_next_load_stage],
                        )
                        cur_desc_b = make_b_desc(
                            iv + arith.index(_next_step + _next_load_k_off),
                            stages_b_mem[_next_load_stage],
                        )

                    tdm_ops.tensor_wait(2 * (num_buffers - 2))
                    gpu.barrier()

                    _next_compute = (s + 1) % num_buffers
                    accs_in, cur_frags = compute_and_prefetch(
                        accs_in,
                        cur_frags,
                        stages_a[_next_compute],
                        stages_b[_next_compute],
                    )

                out_descs = [
                    cur_desc_a.dgroup0,
                    cur_desc_a.dgroup1,
                    cur_desc_b.dgroup0,
                    cur_desc_b.dgroup1,
                ]
                results = yield list(accs_in) + out_descs + cur_frags

            accs = list(results[:n_accs])
            cur_frags = list(results[n_accs + 4 : n_accs + 4 + n_frags_per_tile])

        # Step 5: drain remaining tiles
        _tail_base_k = main_loop_iters * num_buffers
        _extra_loads = tail_tiles - pre_loaded

        for t in range_constexpr(tail_tiles - 1):
            if t < _extra_loads:
                load_tile_idx = _tail_base_k + pre_loaded + t
                load_stage = (pre_loaded + t) % num_buffers
                copy_a_to_lds(
                    arith.index(load_tile_idx * tile_k), stages_a_mem[load_stage]
                )
                copy_b_to_lds(
                    arith.index(load_tile_idx * tile_k), stages_b_mem[load_stage]
                )

            _epi_outstanding = 2 * (pre_loaded + min(t + 1, _extra_loads) - t - 1)
            tdm_ops.tensor_wait(_epi_outstanding)
            gpu.barrier()

            _next_epi_stage = (_tail_base_k + t + 1) % num_buffers
            accs, cur_frags = compute_and_prefetch(
                accs, cur_frags, stages_a[_next_epi_stage], stages_b[_next_epi_stage]
            )

        # Step 6: Final WMMA
        accs = wmma_tile(accs, cur_frags)

        # Step 7: Bias, activation, and store
        accs = _apply_bias_and_activation(accs)
        epilogue_stores(accs)

    cache_tag = (
        in_dtype,
        out_dtype,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        effective_waves_per_eu,
        l2_prefetch_distance,
        activation,
        add_bias,
        physical_mk,
        physical_kn,
        M,
        N,
    )

    @flyc.jit
    def launch_gemm_a16w16(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()

        # Finalize LDS allocations
        with ir.InsertionPoint(ctx.gpu_module_body):
            for alloc in stage_allocators:
                alloc.finalized = False
            for alloc in stage_allocators:
                alloc.finalize()

        # Grid dimensions
        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))
        gy = _raw((idx_n + arith.index(tile_n - 1)) / arith.index(tile_n))

        # Emit kernel
        launcher = kernel_gemm_a16w16(arg_y, arg_x, arg_w, arg_bias, i32_m, i32_n)

        # Set waves_per_eu
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                if effective_waves_per_eu is not None:
                    _wpe = int(effective_waves_per_eu)
                    if _wpe >= 1:
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), _wpe
                        )

        launcher.launch(
            grid=(gx, gy, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_gemm_a16w16


def gemm_a16w16(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float16,
    y: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 32,
    m_warp: int = 2,
    n_warp: int = 4,
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 2,
):
    """Compute Y = X @ W^T + bias. Auto-detects physical layout from strides."""
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"x must be fp16/bf16, got {x.dtype}"
    assert w.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"w must be fp16/bf16, got {w.dtype}"
    assert x.shape[1] == w.shape[1], "Incompatible K dimensions"

    M, K = x.shape
    N = w.shape[0]

    # ── Layout ──
    if x.stride(1) == 1:
        physical_mk = True
    elif x.stride(0) == 1:
        physical_mk = False

    if w.stride(1) == 1:
        physical_kn = False
    elif w.stride(0) == 1:
        physical_kn = True

    # ── K-pad ──
    K_padded = ((K + tile_k - 1) // tile_k) * tile_k
    if K_padded != K:
        pad_size = K_padded - K
        if physical_mk:
            x = torch.nn.functional.pad(x, (0, pad_size))
        else:
            x = torch.nn.functional.pad(x.T, (0, 0, 0, pad_size)).T

        if physical_kn:
            if w.stride(1) == 1:
                w = torch.nn.functional.pad(w, (0, 0, 0, pad_size))
            else:
                w = torch.nn.functional.pad(w.T, (0, 0, 0, pad_size)).T
        else:
            w = torch.nn.functional.pad(w, (0, pad_size))
        K = K_padded

    # ── N-padding ──
    N_stride = ((N + tile_n - 1) // tile_n) * tile_n

    # ── Output allocation ──
    if y is not None:
        y_buf = (
            y
            if N_stride == N
            else torch.empty((M, N_stride), device=x.device, dtype=dtype)
        )
    else:
        y_buf = (
            torch.empty((M, N_stride), device=x.device, dtype=dtype)
            if N_stride != N
            else torch.empty((M, N), device=x.device, dtype=dtype)
        )

    if bias is None:
        bias = torch.empty(0, device=x.device, dtype=dtype)

    in_dtype_str = "fp16" if x.dtype == torch.float16 else "bf16"
    if dtype == torch.float16:
        out_dtype_str = "f16"
    elif dtype == torch.bfloat16:
        out_dtype_str = "bf16"
    else:
        out_dtype_str = "f32"

    launch_fn = compile_gemm_a16w16(
        M=M if not physical_mk else 0,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        in_dtype=in_dtype_str,
        out_dtype=out_dtype_str,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        activation=activation,
        add_bias=(bias.numel() > 0),
        physical_mk=physical_mk,
        physical_kn=physical_kn,
    )

    launch_fn(y_buf, x, w, bias, M, N_stride, stream=torch.cuda.current_stream())

    if N_stride != N:
        result = y_buf[:, :N]
        if y is not None:
            y.copy_(result)
            return y
        return result
    return y_buf


__all__ = ["compile_gemm_a16w16", "gemm_a16w16"]
