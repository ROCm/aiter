# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    tdm_ops,
    vector,
)
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.expr import idx2crd
from typing import Optional

# WMMA 16×16×32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32

# LDS row padding (8 elems = 16 bytes)
LDS_PAD_A = 8
LDS_PAD_B = 8

_STAGE_NAMES = ("ping", "pong", "pang", "pung")

_SCHED_ALLOW_SALU = 1 << 2


def _disable_unroll_on_enclosing_loop():
    """Tag the enclosing scf.for with llvm.loop metadata so it stays rolled at ASM level."""
    block = ir.InsertionPoint.current.block
    op = block.owner
    if op.name != "scf.for":
        return
    anno = ir.Attribute.parse(
        "#llvm.loop_annotation<unroll = <disable = true>, disableNonforced = true>"
    )
    op.attributes["loop_annotation"] = anno


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
    physical_kn: bool = False,  # False=N-major (row-major W), True=K-major (transposed W)
    loop_carried_load_percent: Optional[int] = None,
    kernarg_preload: bool = False,
    use_manual_barrier: bool = False,
    split_k: int = 1,
    sched_strategy: Optional[str] = None,
    barrier_signal_wait_latency: Optional[int] = None,
    main_loop_unroll: bool = False,
    variant: str = "bandwidth_bound",
):
    """Compile the A16W16 GEMM kernel; returns launch_fn(y, x, w, bias, M, N, stream=stream).

    variant:
        "bandwidth_bound" (default) — steady-state loads the whole next fragment
            bank up front, then runs all WMMAs (burst load, ~2 banks live).
        "compute_bound" — in-place fragment rotation: each fragment's next-tile
            ds_load is hoisted to right after its last WMMA use so the load
            co-executes with the remaining WMMAs (~1 bank live + trickle).
    """
    _ = (M, N)

    if not (2 <= num_buffers <= 8):
        raise ValueError(f"num_buffers must be between 2 and 8, got {num_buffers}")
    if variant not in ("bandwidth_bound", "compute_bound"):
        raise ValueError(
            "variant must be 'bandwidth_bound' or 'compute_bound', " f"got {variant!r}"
        )
    if in_dtype not in ("fp16", "bf16"):
        raise ValueError(f"in_dtype must be 'fp16' or 'bf16', got {in_dtype!r}")
    # Experimental LLVM scheduling levers (injected via llvm_options below).
    if sched_strategy is not None and sched_strategy not in (
        "max-ilp",
        "max-memory-clause",
    ):
        raise ValueError(
            "sched_strategy must be None, 'max-ilp', or 'max-memory-clause', "
            f"got {sched_strategy!r}"
        )

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

    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    if K % split_k != 0:
        raise ValueError(f"K must be divisible by split_k={split_k}, got K={K}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if (tile_k & (tile_k - 1)) != 0:
        raise ValueError(f"tile_k must be a power of 2 for TDM, got {tile_k}")

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

    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N}")

    # split_k>1 splits K into split_k grid-z chunks; num_k_tiles is per-split.
    split_k_chunk = K // split_k
    if split_k_chunk % tile_k != 0:
        raise ValueError(
            f"K/split_k must be divisible by tile_k={tile_k}, got {split_k_chunk}"
        )
    num_k_tiles = split_k_chunk // tile_k
    if num_k_tiles < num_buffers - 1:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers - 1}, "
            f"got {num_k_tiles} (K={K}, tile_k={tile_k})"
        )

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    wmma_op = rocdl.wmma_f32_16x16x32_f16 if is_f16 else rocdl.wmma_f32_16x16x32_bf16

    k_wmma_steps = tile_k // WMMA_K

    def _elem_type():
        return T.f16 if is_f16 else T.bf16

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep
    N_A_FRAGS = wmma_m_rep * k_wmma_steps
    N_B_FRAGS = wmma_n_rep * k_wmma_steps

    # LDS layout: A is [tile_m, tile_k+pad] (M-major) or [tile_k, tile_m+pad] (K-major)
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

    lds_a_data_bytes = lds_a_elems * elem_bytes
    lds_b_data_bytes = lds_b_elems * elem_bytes

    # Unified LDS allocator: contiguous [A0..A_nb-1 | B0..B_nb-1] ring slots
    unified_alloc = SmemAllocator(None, arch=gpu_arch, global_sym_name="a16w16_unified")
    unified_a_off = unified_alloc._align(unified_alloc.ptr, 16)
    unified_alloc.ptr = unified_a_off + num_buffers * lds_a_data_bytes
    unified_b_off = unified_alloc._align(unified_alloc.ptr, 16)
    unified_alloc.ptr = unified_b_off + num_buffers * lds_b_data_bytes

    stage_a_data_off = [
        unified_a_off + i * lds_a_data_bytes for i in range(num_buffers)
    ]
    stage_b_data_off = [
        unified_b_off + i * lds_b_data_bytes for i in range(num_buffers)
    ]
    stage_allocators = [unified_alloc] * num_buffers

    # TDMs per K-tile: A + B
    _TDMS_PER_TILE = 2

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

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)
        # split_k>1: grid-z block picks the K-chunk origin (compile-time gated).
        if const_expr(split_k > 1):
            bz = gpu.block_id("z")
            split_k_base = bz * arith.index(split_k_chunk)

        # Thread -> warp decomposition
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
        if const_expr(add_bias):
            bias_rsrc = buffer_ops.create_buffer_resource(arg_bias, max_size=True)

        def make_a_desc(k_base, lds_a_mem_ref):
            """TDM descriptor for A tile. Swaps dims when physical_mk=False (K-major)."""
            if const_expr(physical_mk):
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
            if const_expr(physical_kn):
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

        # Unified LDS pointers (all NB slots) + per-stage aliases
        big_a = SmemPtr(
            unified_alloc.get_base(),
            unified_a_off,
            elem_ty,
            shape=(num_buffers * lds_a_elems,),
        )
        big_b = SmemPtr(
            unified_alloc.get_base(),
            unified_b_off,
            elem_ty,
            shape=(num_buffers * lds_b_elems,),
        )
        big_a_mem = big_a.get()
        big_b_mem = big_b.get()
        stages_a = [
            SmemPtr(
                unified_alloc.get_base(),
                stage_a_data_off[i],
                elem_ty,
                shape=(lds_a_elems,),
            )
            for i in range(num_buffers)
        ]
        stages_b = [
            SmemPtr(
                unified_alloc.get_base(),
                stage_b_data_off[i],
                elem_ty,
                shape=(lds_b_elems,),
            )
            for i in range(num_buffers)
        ]
        stages_a_mem = [p.get() for p in stages_a]
        stages_b_mem = [p.get() for p in stages_b]

        # TDM descriptors built once at entry; lo32 advances per K-tile, LDS base per ring slot.
        if const_expr(split_k > 1):
            _desc_a_init = make_a_desc(split_k_base, stages_a_mem[0])
            _desc_b_init = make_b_desc(split_k_base, stages_b_mem[0])
        else:
            _desc_a_init = make_a_desc(arith.index(0), stages_a_mem[0])
            _desc_b_init = make_b_desc(arith.index(0), stages_b_mem[0])
        dgroup1_a = _desc_a_init.dgroup1
        dgroup1_w = _desc_b_init.dgroup1
        addr_hi_a = vector.extract(
            _desc_a_init.dgroup0, static_position=[3], dynamic_position=[]
        )
        addr_hi_w = vector.extract(
            _desc_b_init.dgroup0, static_position=[3], dynamic_position=[]
        )
        addr_lo_init_a = vector.extract(
            _desc_a_init.dgroup0, static_position=[2], dynamic_position=[]
        )
        addr_lo_init_w = vector.extract(
            _desc_b_init.dgroup0, static_position=[2], dynamic_position=[]
        )
        a_lds_base_i32 = vector.extract(
            _desc_a_init.dgroup0, static_position=[1], dynamic_position=[]
        )
        b_lds_base_i32 = vector.extract(
            _desc_b_init.dgroup0, static_position=[1], dynamic_position=[]
        )
        slot_stride_a_i32 = arith.constant(lds_a_data_bytes, type=T.i32)
        slot_stride_b_i32 = arith.constant(lds_b_data_bytes, type=T.i32)
        slot_stride_a_elems_i32 = arith.constant(lds_a_elems, type=T.i32)
        slot_stride_b_elems_i32 = arith.constant(lds_b_elems, type=T.i32)

        # Per-K-tile lo32 byte advance (tile_k*bytes row-major; outer-stride*bytes if K-major).
        adv_a_i32 = arith.constant(
            (tile_k if physical_mk else tile_k * M) * elem_bytes, type=T.i32
        )
        adv_w_i32 = arith.constant(
            (tile_k if not physical_kn else tile_k * N) * elem_bytes, type=T.i32
        )
        pred_const = arith.constant(1, type=T.i32)

        def _buf_idx_to_i32(buf_idx):
            """Accept either Python int (prologue/drain) or i32 SSA (main loop)."""
            if const_expr(isinstance(buf_idx, int)):
                return arith.constant(buf_idx, type=T.i32)
            return buf_idx

        def issue_tdm_loads(buf_idx, lo_a, lo_w):
            """Issue A+B TDMs for one K-tile into LDS slot buf_idx; returns advanced (lo_a, lo_w)."""
            buf_i32 = _buf_idx_to_i32(buf_idx)
            a_addr = arith.addi(a_lds_base_i32, arith.muli(buf_i32, slot_stride_a_i32))
            b_addr = arith.addi(b_lds_base_i32, arith.muli(buf_i32, slot_stride_b_i32))
            dg0_a = vector.from_elements(
                T.vec(4, T.i32), [pred_const, a_addr, lo_a, addr_hi_a]
            )
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a))
            dg0_w = vector.from_elements(
                T.vec(4, T.i32), [pred_const, b_addr, lo_w, addr_hi_w]
            )
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_w, dgroup1_w))
            return arith.addi(lo_a, adv_a_i32), arith.addi(lo_w, adv_w_i32)

        def _precompute_lane_bases(warp_base, reps, lds_stride):
            """Per-lane LDS element offsets for `reps` WMMA frags (M/N-major)."""
            row_stride_off = (warp_base + lane16) * arith.index(lds_stride)
            k_lane_off = lane_kgrp * arith.index(8)
            bases = []
            for rep in range_constexpr(reps):
                base = (
                    row_stride_off + arith.index(rep * WMMA_M * lds_stride) + k_lane_off
                )
                bases.append(base)
            return bases

        def _precompute_a_lane_bases():
            """A fragment lane bases. Uses transpose-load addressing when K-major."""
            if const_expr(physical_mk):
                return _precompute_lane_bases(warp_m_base, wmma_m_rep, lds_a_stride)

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
            return bases

        def _precompute_b_lane_bases():
            """B fragment lane bases. Uses transpose-load addressing when K-major."""
            if const_expr(not physical_kn):
                return _precompute_lane_bases(warp_n_base, wmma_n_rep, lds_b_stride)

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
            return bases

        a_lane_bases = _precompute_a_lane_bases()
        b_lane_bases = _precompute_b_lane_bases()

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

        def load_a_frags(buf_idx):
            """Load all A frags for one K-tile from LDS slot buf_idx; index a_frags[ks*wmma_m_rep+wm]."""
            if const_expr(isinstance(buf_idx, int)):
                slot_off_a = arith.index(buf_idx * lds_a_elems)
            else:
                slot_off_a = arith.index_cast(
                    T.index, arith.muli(buf_idx, slot_stride_a_elems_i32)
                )
            a_frags = []
            for ks in range_constexpr(k_wmma_steps):
                for wm in range_constexpr(wmma_m_rep):
                    a_frags.append(
                        _load_a_frag(big_a_mem, a_lane_bases[wm] + slot_off_a, ks)
                    )
            return a_frags

        def load_b_frags(buf_idx):
            """Load all B frags for one K-tile from LDS slot buf_idx; index b_frags[ks*wmma_n_rep+wn]."""
            if const_expr(isinstance(buf_idx, int)):
                slot_off_b = arith.index(buf_idx * lds_b_elems)
            else:
                slot_off_b = arith.index_cast(
                    T.index, arith.muli(buf_idx, slot_stride_b_elems_i32)
                )
            b_frags = []
            for ks in range_constexpr(k_wmma_steps):
                for wn in range_constexpr(wmma_n_rep):
                    b_frags.append(
                        _load_b_frag(big_b_mem, b_lane_bases[wn] + slot_off_b, ks)
                    )
            return b_frags

        def wmma_tile(accs_in, a_frags, b_frags):
            """Execute all WMMAs for one tile (ks-outer accumulate); reuseB=(wn>0) hint kept."""
            current_accs = list(accs_in)
            for ks in range_constexpr(k_wmma_steps):
                a_f = [
                    a_frags[ks * wmma_m_rep + wm] for wm in range_constexpr(wmma_m_rep)
                ]
                b_f = [
                    b_frags[ks * wmma_n_rep + wn] for wn in range_constexpr(wmma_n_rep)
                ]
                for wm in range_constexpr(wmma_m_rep):
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        # ISA operand order: (B, A, C) — reversed from math
                        current_accs[idx] = wmma_op(
                            T.vec(8, T.f32),
                            b_f[wn],
                            a_f[wm],
                            current_accs[idx],
                            modC=0,
                            reuseA=False,
                            reuseB=(wn > 0),
                        ).result
            return current_accs

        def _load_one_a_frag(buf_i32, ks, wm):
            """ds_load a single A fragment (ks, wm) from LDS ring slot buf_i32 (i32 SSA)."""
            slot_off_a = arith.index_cast(
                T.index, arith.muli(buf_i32, slot_stride_a_elems_i32)
            )
            return _load_a_frag(big_a_mem, a_lane_bases[wm] + slot_off_a, ks)

        def _load_one_b_frag(buf_i32, ks, wn):
            """ds_load a single B fragment (ks, wn) from LDS ring slot buf_i32 (i32 SSA)."""
            slot_off_b = arith.index_cast(
                T.index, arith.muli(buf_i32, slot_stride_b_elems_i32)
            )
            return _load_b_frag(big_b_mem, b_lane_bases[wn] + slot_off_b, ks)

        def _wmma_rotate(accs_in, a_frags, b_frags, next_buf_i32):
            """compute_bound steady-state: run this tile's WMMAs, and the instant a
            fragment reaches its last use, issue its next-tile ds_load so the load
            co-executes with the remaining WMMAs. Returns (accs, next_a, next_b) where
            next_* are the freshly rotated-in fragments for the next K-tile."""
            current_accs = list(accs_in)
            next_a = [None] * N_A_FRAGS
            next_b = [None] * N_B_FRAGS
            for ks in range_constexpr(k_wmma_steps):
                for wm in range_constexpr(wmma_m_rep):
                    a_f = a_frags[ks * wmma_m_rep + wm]
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        current_accs[idx] = wmma_op(
                            T.vec(8, T.f32),
                            b_frags[ks * wmma_n_rep + wn],
                            a_f,
                            current_accs[idx],
                            modC=0,
                            reuseA=False,
                            reuseB=(wn > 0),
                        ).result
                    rocdl.sched_barrier(_SCHED_ALLOW_SALU)
                    next_a[ks * wmma_m_rep + wm] = _load_one_a_frag(
                        next_buf_i32, ks, wm
                    )
                rocdl.sched_barrier(_SCHED_ALLOW_SALU)
                for wn in range_constexpr(wmma_n_rep):
                    next_b[ks * wmma_n_rep + wn] = _load_one_b_frag(
                        next_buf_i32, ks, wn
                    )
            return current_accs, next_a, next_b

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
            if const_expr(not add_bias and activation is None):
                return accs

            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    acc = accs[idx]

                    if const_expr(add_bias):
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

                    if const_expr(activation is not None):
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
            """Write accumulators to Y (buffer_store, or atomic-fadd if split_k>1)."""
            if const_expr(split_k > 1):
                zero_i32 = fx.Int32(0)
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

                    if const_expr(_half_out):
                        if const_expr(split_k > 1):
                            h_vec = arith.trunc_f(T.vec(8, _out_elem), final_accs[idx])
                            c_off_bytes = (row * n_stride + col_base) * arith.index(
                                elem_bytes_d
                            )
                            pair_ty = T.vec(2, _out_elem)
                            for pair in range_constexpr(4):
                                e0 = vector.extract(
                                    h_vec,
                                    static_position=[pair * 2],
                                    dynamic_position=[],
                                )
                                e1 = vector.extract(
                                    h_vec,
                                    static_position=[pair * 2 + 1],
                                    dynamic_position=[],
                                )
                                pair_vec = vector.from_elements(pair_ty, [e0, e1])
                                byte_off = arith.index_cast(
                                    T.i32, c_off_bytes + arith.index(pair * 4)
                                )
                                rocdl.raw_ptr_buffer_atomic_fadd(
                                    pair_vec, y_rsrc, byte_off, zero_i32, zero_i32
                                )
                        else:
                            h_vec = arith.trunc_f(T.vec(8, _out_elem), final_accs[idx])
                            i32_vec = vector.bitcast(T.vec(4, T.i32), h_vec)
                            c_off_bytes = (row * n_stride + col_base) * arith.index(
                                elem_bytes_d
                            )
                            buffer_ops.buffer_store(
                                i32_vec, y_rsrc, c_off_bytes, offset_is_bytes=True
                            )
                    else:
                        if const_expr(split_k > 1):
                            for half in range_constexpr(2):
                                base = row * n_stride + col_base + arith.index(half * 4)
                                for vi in range_constexpr(4):
                                    val = vector.extract(
                                        final_accs[idx],
                                        static_position=[half * 4 + vi],
                                        dynamic_position=[],
                                    )
                                    byte_off = arith.index_cast(
                                        T.i32,
                                        (base + arith.index(vi)) * arith.index(4),
                                    )
                                    rocdl.raw_ptr_buffer_atomic_fadd(
                                        val, y_rsrc, byte_off, zero_i32, zero_i32
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

        # Accumulators
        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        # State carry: [accs (N_ACCS) | cur_a frags (N_A_FRAGS) | cur_b frags (N_B_FRAGS)]
        N_ACCS = n_accs

        def _pack_state(accs_, a_, b_):
            return list(accs_) + list(a_) + list(b_)

        def _unpack_state(state):
            i = 0
            accs_ = list(state[i : i + N_ACCS])
            i += N_ACCS
            a_ = list(state[i : i + N_A_FRAGS])
            i += N_A_FRAGS
            b_ = list(state[i : i + N_B_FRAGS])
            i += N_B_FRAGS
            return accs_, a_, b_

        # ── Prologue (memory-bound): fill NB-1 LDS buffers, then load tile 0 ──
        lo_a = addr_lo_init_a
        lo_w = addr_lo_init_w
        for i in range_constexpr(num_buffers - 1):
            lo_a, lo_w = issue_tdm_loads(i, lo_a, lo_w)

        # Wait retires tile 0 (NB-2 in flight); the 2-iter WAR gap lets the main loop use one barrier.
        tdm_ops.tensor_wait((num_buffers - 2) * _TDMS_PER_TILE)
        gpu.barrier()

        cur_a = load_a_frags(0)
        cur_b = load_b_frags(0)

        # ── Main loop (wait-first, unrolled by 2): ping-pong two register banks, prefetch T+1/T+2 ──
        main_loop_iters = num_k_tiles - (num_buffers - 1)
        nb_const_i32 = arith.constant(num_buffers, type=T.i32)
        one_i32 = arith.constant(1, type=T.i32)
        load_idx_init = arith.constant(num_buffers - 1, type=T.i32)
        compute_idx_init = arith.constant(0, type=T.i32)

        def _wg_barrier():
            # gpu.barrier() drains ds_loads via fences; manual variant is a bare s_barrier (no fences, faster but racy).
            if const_expr(use_manual_barrier):
                rocdl.s_barrier_signal(-1)
                rocdl.s_barrier_wait(-1)
            else:
                gpu.barrier()

        def _run_tile(accs_, ca, cb, lo_a_, lo_w_, cidx, lidx, do_prefetch):
            """Compute K-tile cidx from carried frags; if do_prefetch, load cidx+1 into a fresh bank."""
            load_buf_i32 = arith.remui(lidx, nb_const_i32)

            # Refill first, then wait: keeps NB-2 tiles' TDM in flight (depth-2).
            lo_a_, lo_w_ = issue_tdm_loads(load_buf_i32, lo_a_, lo_w_)
            if const_expr(num_buffers == 2):
                # NB=2: only 2 slots -> drain the just-issued TDM (RAW hazard).
                tdm_ops.tensor_wait(0)
            else:
                tdm_ops.tensor_wait((num_buffers - 2) * _TDMS_PER_TILE)

            # Single barrier: RAW wall for this tile's T+1 ds_reads and (across the backedge) WAR wall for the next refill.
            _wg_barrier()

            if const_expr(do_prefetch):
                next_buf_i32 = arith.remui(arith.addi(cidx, one_i32), nb_const_i32)
                # Preload next tile's A+B, then WMMA (loads co-execute with it); raises peak liveness to ~2 banks.
                na = load_a_frags(next_buf_i32)
                nb = load_b_frags(next_buf_i32)
                accs_ = wmma_tile(accs_, ca, cb)
                nbank = (na, nb)
            else:
                accs_ = wmma_tile(accs_, ca, cb)
                nbank = None
            return accs_, nbank, lo_a_, lo_w_

        if const_expr(variant == "compute_bound"):
            if const_expr(main_loop_iters > 0):
                init_state = _pack_state(accs, cur_a, cur_b) + [
                    lo_a,
                    lo_w,
                    load_idx_init,
                    compute_idx_init,
                ]
                results = init_state
                for step, state in range(0, main_loop_iters, 1, init=init_state):
                    _disable_unroll_on_enclosing_loop()
                    cidx = state[-1]
                    lidx = state[-2]
                    lw = state[-3]
                    lx = state[-4]
                    p_accs, ca, cb = _unpack_state(state[:-4])

                    load_buf_i32 = arith.remui(lidx, nb_const_i32)
                    lx, lw = issue_tdm_loads(load_buf_i32, lx, lw)
                    if const_expr(num_buffers == 2):
                        tdm_ops.tensor_wait(0)
                    else:
                        tdm_ops.tensor_wait((num_buffers - 2) * _TDMS_PER_TILE)
                    _wg_barrier()

                    next_buf_i32 = arith.remui(arith.addi(cidx, one_i32), nb_const_i32)
                    p_accs, next_a, next_b = _wmma_rotate(p_accs, ca, cb, next_buf_i32)
                    cidx1 = arith.addi(cidx, one_i32)
                    lidx1 = arith.addi(lidx, one_i32)

                    new_state = _pack_state(p_accs, next_a, next_b) + [
                        lx,
                        lw,
                        lidx1,
                        cidx1,
                    ]
                    results = yield new_state

                accs, cur_a, cur_b = _unpack_state(results[:-4])
            else:
                accs = list(accs)
        elif const_expr(main_loop_iters > 0):
            if const_expr(main_loop_unroll):
                # Unroll-by-2: 2 tiles/trip, ping-pong 2 register banks + odd leftover.
                lo_a_s = lo_a
                lo_w_s = lo_w
                load_idx_s = load_idx_init
                compute_idx_s = compute_idx_init
                num_pairs = main_loop_iters // 2

                if const_expr(num_pairs > 0):
                    init_state = _pack_state(accs, cur_a, cur_b) + [
                        lo_a_s,
                        lo_w_s,
                        load_idx_s,
                        compute_idx_s,
                    ]
                    results = init_state
                    for pair_step, state in range(0, num_pairs, 1, init=init_state):
                        _disable_unroll_on_enclosing_loop()
                        cidx = state[-1]
                        lidx = state[-2]
                        lw = state[-3]
                        lx = state[-4]
                        p_accs, b0a, b0b = _unpack_state(state[:-4])

                        # sub-0: tile T from bank0, prefetch T+1 -> bank1.
                        p_accs, bank1, lx, lw = _run_tile(
                            p_accs, b0a, b0b, lx, lw, cidx, lidx, do_prefetch=True
                        )
                        b1a, b1b = bank1
                        cidx1 = arith.addi(cidx, one_i32)
                        lidx1 = arith.addi(lidx, one_i32)

                        # sub-1: tile T+1 from bank1, prefetch T+2 -> bank0 (yielded).
                        p_accs, bank0, lx, lw = _run_tile(
                            p_accs, b1a, b1b, lx, lw, cidx1, lidx1, do_prefetch=True
                        )
                        nb0a, nb0b = bank0
                        cidx2 = arith.addi(cidx1, one_i32)
                        lidx2 = arith.addi(lidx1, one_i32)

                        new_state = _pack_state(p_accs, nb0a, nb0b) + [
                            lx,
                            lw,
                            lidx2,
                            cidx2,
                        ]
                        results = yield new_state

                    accs, cur_a, cur_b = _unpack_state(results[:-4])
                    lo_a_s = results[-4]
                    lo_w_s = results[-3]
                    load_idx_s = results[-2]
                    compute_idx_s = results[-1]
                else:
                    accs = list(accs)

                # Leftover odd tile: compute the carried bank, prefetch first drain tile.
                if const_expr(main_loop_iters % 2 == 1):
                    accs, leftover_bank, lo_a_s, lo_w_s = _run_tile(
                        accs,
                        cur_a,
                        cur_b,
                        lo_a_s,
                        lo_w_s,
                        compute_idx_s,
                        load_idx_s,
                        do_prefetch=True,
                    )
                    cur_a, cur_b = leftover_bank
                    compute_idx_s = arith.addi(compute_idx_s, one_i32)
            else:
                # Single tile/trip: compute T from carried bank, prefetch T+1.
                init_state = _pack_state(accs, cur_a, cur_b) + [
                    lo_a,
                    lo_w,
                    load_idx_init,
                    compute_idx_init,
                ]
                results = init_state
                for step, state in range(0, main_loop_iters, 1, init=init_state):
                    _disable_unroll_on_enclosing_loop()
                    cidx = state[-1]
                    lidx = state[-2]
                    lw = state[-3]
                    lx = state[-4]
                    p_accs, ca, cb = _unpack_state(state[:-4])

                    # tile T from the carried bank, prefetch T+1 -> next bank (yielded).
                    p_accs, bank, lx, lw = _run_tile(
                        p_accs, ca, cb, lx, lw, cidx, lidx, do_prefetch=True
                    )
                    next_a, next_b = bank
                    cidx1 = arith.addi(cidx, one_i32)
                    lidx1 = arith.addi(lidx, one_i32)

                    new_state = _pack_state(p_accs, next_a, next_b) + [
                        lx,
                        lw,
                        lidx1,
                        cidx1,
                    ]
                    results = yield new_state

                accs, cur_a, cur_b = _unpack_state(results[:-4])
        else:
            accs = list(accs)
            # No main loop ran — drain starts at compute_idx = 0.

        # ── Drain (fully unrolled): consume carried frags, prefetch next bank per tile; final tile does no wait/barrier/ds_load ──
        drain_count_d = (
            num_buffers - 1
            if main_loop_iters > 0
            else min(num_buffers - 1, num_k_tiles)
        )
        drain_base = main_loop_iters if main_loop_iters > 0 else 0

        def _drain_wait_for(tile_idx):
            # Wait for compile-time drain tile tile_idx to land.
            tdm_ops.tensor_wait(max(0, num_k_tiles - 1 - tile_idx) * _TDMS_PER_TILE)

        accs = list(accs)
        drain_a, drain_b = cur_a, cur_b
        for j in range_constexpr(drain_count_d):
            tile_idx = drain_base + j
            if const_expr(j < drain_count_d - 1):
                # Not the last tile: wait + barrier + prefetch tile_idx+1, interleaved with the current WMMA.
                next_tile = tile_idx + 1
                _drain_wait_for(next_tile)
                gpu.barrier()
                next_a = load_a_frags(next_tile % num_buffers)
                accs = wmma_tile(accs, drain_a, drain_b)
                next_b = load_b_frags(next_tile % num_buffers)
                drain_a, drain_b = next_a, next_b
            else:
                # Final K-tile: compute the carried frags — no wait, barrier, or ds_loads.
                accs = wmma_tile(accs, drain_a, drain_b)

        # ── Epilogue: bias, activation, and store ──
        if const_expr(num_buffers > 2):
            rocdl.sched_barrier(0)
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
        loop_carried_load_percent,
        kernarg_preload,
        use_manual_barrier,
        split_k,
        sched_strategy,
        barrier_signal_wait_latency,
        main_loop_unroll,
        variant,
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
        gz = split_k  # 1 at default -> grid tuple identical to original

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

        # Flat work-group size hint.
        flat_wg_attr = ir.StringAttr.get(f"{block_threads},{block_threads}")
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        # Experimental: amdgpu-loop-carried-load-percent passthrough.
        if loop_carried_load_percent is not None:
            lcv = ir.ArrayAttr.get(
                [
                    ir.ArrayAttr.get(
                        [
                            ir.StringAttr.get("amdgpu-loop-carried-load-percent"),
                            ir.StringAttr.get(str(int(loop_carried_load_percent))),
                        ]
                    )
                ]
            )
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    op.attributes["passthrough"] = lcv

        # Mark kernel args inreg so AMDGPU preloads them into user SGPRs.
        if kernarg_preload:
            inreg_attr = ir.UnitAttr.get()
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    num_args = len(op.regions[0].blocks[0].arguments)
                    per_arg = [
                        ir.DictAttr.get({"llvm.inreg": inreg_attr})
                        for _ in range(num_args)
                    ]
                    op.attributes["arg_attrs"] = ir.ArrayAttr.get(per_arg)

        launcher.launch(
            grid=(gx, gy, gz),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    # Backend cl::opt hints; only options present in THIS build are set.
    _llvm_opts = {
        "amdgpu-expert-scheduling-mode": True,  # valid bool cl::opt (GFX12+ only)
    }
    # Experimental scheduling levers (in-process LLVM); None => cl::init default.
    if sched_strategy is not None:
        # cl::opt<str>: max-ilp (GCNMaxILP) or max-memory-clause.
        _llvm_opts["amdgpu-sched-strategy"] = sched_strategy
    if barrier_signal_wait_latency is not None:
        # cl::opt<unsigned> init(35): synthetic barrier signal->wait latency.
        _llvm_opts["amdgpu-barrier-signal-wait-latency"] = barrier_signal_wait_latency
    launch_gemm_a16w16.compile_hints["llvm_options"] = _llvm_opts

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
    kernarg_preload: bool = False,
    loop_carried_load_percent: Optional[int] = None,
    use_manual_barrier: bool = False,
    split_k: int = 1,
    sched_strategy: Optional[str] = None,
    barrier_signal_wait_latency: Optional[int] = None,
    main_loop_unroll: bool = False,
    variant: str = "bandwidth_bound",
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

    # Layout from strides
    if x.stride(1) == 1:
        physical_mk = True
    elif x.stride(0) == 1:
        physical_mk = False

    if w.stride(1) == 1:
        physical_kn = False
    elif w.stride(0) == 1:
        physical_kn = True

    # K-pad up to a tile_k multiple
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

    # N-pad up to a tile_n multiple
    N_stride = ((N + tile_n - 1) // tile_n) * tile_n

    # split_k>1 half-out: accumulate in fp32 workspace, cast at end (bf16 atomic lossy).
    _half_dtype = dtype in (torch.float16, torch.bfloat16)
    _splitk_f32_accum = split_k > 1 and _half_dtype

    in_dtype_str = "fp16" if x.dtype == torch.float16 else "bf16"
    if dtype == torch.float16:
        out_dtype_str = "f16"
    elif dtype == torch.bfloat16:
        out_dtype_str = "bf16"
    else:
        out_dtype_str = "f32"
    buf_dtype = dtype
    if _splitk_f32_accum:
        out_dtype_str = "f32"  # kernel accumulates fp32 partials
        buf_dtype = torch.float32  # fp32 workspace, cast to `dtype` after launch

    # split_k>1 atomic-fadd needs a zeroed buffer; fp32-accum uses a fresh workspace.
    _alloc = torch.zeros if split_k > 1 else torch.empty
    if y is not None and not _splitk_f32_accum:
        y_buf = (
            y
            if N_stride == N
            else _alloc((M, N_stride), device=x.device, dtype=buf_dtype)
        )
        if split_k > 1 and y_buf is y:
            y_buf.zero_()
    else:
        y_buf = (
            _alloc((M, N_stride), device=x.device, dtype=buf_dtype)
            if N_stride != N
            else _alloc((M, N), device=x.device, dtype=buf_dtype)
        )

    if bias is None:
        bias = torch.empty(0, device=x.device, dtype=dtype)

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
        kernarg_preload=kernarg_preload,
        loop_carried_load_percent=loop_carried_load_percent,
        use_manual_barrier=use_manual_barrier,
        split_k=split_k,
        sched_strategy=sched_strategy,
        barrier_signal_wait_latency=barrier_signal_wait_latency,
        main_loop_unroll=main_loop_unroll,
        variant=variant,
    )

    launch_fn(y_buf, x, w, bias, M, N_stride, stream=torch.cuda.current_stream())

    if _splitk_f32_accum:
        # Cast the fp32 accumulation workspace back to the requested half dtype.
        result = (y_buf[:, :N] if N_stride != N else y_buf).to(dtype)
        if y is not None:
            y.copy_(result)
            return y
        return result

    if N_stride != N:
        result = y_buf[:, :N]
        if y is not None:
            y.copy_(result)
            return y
        return result
    return y_buf


__all__ = ["compile_gemm_a16w16", "gemm_a16w16"]
