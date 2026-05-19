# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A8W8 FP8 blockscale GEMM for gfx1250.

Computes Y = X @ W^T with per-K-block f32 scales.
Supports reg_preload / no_op_preload variants and optional TDM-store output.

Variants:
  - reg_preload    : default. Operand frags loop-carried across K-tiles.
                       * W-scales: bulk-load K-tiles' W-scales into VGPRs
                         (each buffer_load_b32 covers up to 32 K-blocks).
                         scale_k <= 32 → one bulk load at kernel entry +
                         per-K-tile v_readlane. scale_k > 32 → a cur/prefetch
                         chunk chain in the loop carry. Requires
                         w_is_wave_uniform.
                       * X-scales: TDM-staged into LDS (num_buffers stages,
                         aligned with X+W tile stages), then ds_read_b32 into
                         VGPRs in lane16 layout.
  - no_op_preload  : operand frags loaded fresh per K-tile (lower VGPR cost).
                     Uses the legacy per-K-block buffer_load scale path.
"""

from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import math as math_dialect, scf as scf_dialect
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
    return vector.bitcast(
        ir.VectorType.get([4], ir.IntegerType.get_signless(32)), loaded
    )


def lds_load_b32_f32(memref, elem_off):
    """ds_load_b32: load 4 bytes from LDS as a single f32. The memref's
    element type may be smaller (e.g. f8 for byte-addressed staging); we
    read the right number of element units to cover 32 bits and bitcast."""
    vec_ty = _lds_vec_type(memref, 32)
    loaded = vector.load_op(vec_ty, memref, [elem_off])
    as_f32_vec = vector.bitcast(ir.VectorType.get([1], ir.F32Type.get()), loaded)
    return vector.extract(as_f32_vec, static_position=[0], dynamic_position=[])


def lds_store_b128(memref, elem_off, data):
    """ds_store_b128: store 16 bytes to LDS, bitcast to match the memref element type."""
    vec_ty = _lds_vec_type(memref, 128)
    typed_vec = vector.bitcast(vec_ty, data)
    vector.store(typed_vec, memref, [elem_off])


def _disable_unroll_on_enclosing_loop():
    """Attach `loop_annotation = #llvm.loop_annotation<unroll = <disable = true>, disableNonforced = true>`
    to the scf.for op that owns the current insertion point's block.

    This survives scf-to-cf -> cf-to-llvm and becomes `!llvm.loop` metadata
    on the back-edge cf.cond_br, which prevents LLVM from peeling 1-iter
    loops or fully unrolling small constant-bounded loops.

    Call as the first statement inside the body of any FlyDSL `for ... in
    range(...)` loop you want to keep visible at ASM level.
    """
    block = ir.InsertionPoint.current.block
    op = block.owner
    if op.name != "scf.for":
        return
    anno = ir.Attribute.parse(
        "#llvm.loop_annotation<unroll = <disable = true>, " "disableNonforced = true>"
    )
    op.attributes["loop_annotation"] = anno


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
            vals = [
                vector.extract(
                    acc_vec8, static_position=[half * 4 + vi], dynamic_position=[]
                )
                for vi in range(4)
            ]
            vec4 = vector.from_elements(T.vec(4, T.f32), vals)
            lds_store_b128(memref, off + arith.index(half * 8), vec4)


def store_acc_vec8_to_buffer(
    acc_vec8, c_rsrc, addr, out_elem=None, offset_is_bytes=False
):
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
    loop_carried_load_percent: Optional[int] = None,
    kernarg_preload: bool = False,
):
    if variant not in (
        "reg_preload",
        "no_op_preload",
        "experimental",
        "experimental_unroll2",
        "manual",
    ):
        raise ValueError(
            f"variant must be 'reg_preload', 'no_op_preload', 'experimental', "
            f"'experimental_unroll2', or 'manual', got {variant!r}"
        )
    if const_expr(variant == "reg_preload"):
        _w_is_wave_uniform = (tile_n // n_warp) <= scale_block_n
        if not _w_is_wave_uniform:
            raise ValueError(
                f"variant='reg_preload' requires warp_tile_n ({tile_n // n_warp}) "
                f"<= scale_block_n ({scale_block_n}) (W-scale must be wave-uniform)"
            )
        # scale_k > 32 → multi-chunk prefetch chain (issued at entry chunks 0+1,
        # advanced per iteration in the main loop). No upper bound.
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
    if tile_k % scale_block_k != 0:
        raise ValueError(
            f"tile_k ({tile_k}) must be a multiple of scale_block_k ({scale_block_k})"
        )
    if scale_block_k % WMMA_K != 0:
        raise ValueError(
            f"scale_block_k ({scale_block_k}) must be a multiple of {WMMA_K}"
        )
    if K % tile_k != 0:
        raise ValueError(f"K ({K}) must be divisible by tile_k ({tile_k})")
    if K % scale_block_k != 0:
        raise ValueError(
            f"K ({K}) must be divisible by scale_block_k ({scale_block_k})"
        )
    if use_tdm_store:
        if N <= 0:
            raise ValueError(
                "use_tdm_store=True requires N > 0 (compile-time row stride)"
            )
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

    wmma_m_rep = warp_tile_m // WMMA_M  # WMMA tiles per warp along M
    wmma_n_rep = warp_tile_n // WMMA_N  # WMMA tiles per warp along N
    n_accs = wmma_m_rep * wmma_n_rep  # global accumulators per warp
    k_wmma_steps = tile_k // WMMA_K  # WMMAs per K-tile along K
    scales_per_tile = tile_k // scale_block_k  # scale blocks per K-tile
    wmma_steps_per_scale = scale_block_k // WMMA_K
    wmma_pipeline_depth = min(n_accs, 2)
    acc_coords = [
        (wm, wn, wm * wmma_n_rep + wn)
        for wm in range(wmma_m_rep)
        for wn in range(wmma_n_rep)
    ]

    num_k_tiles = K // tile_k
    scale_k = K // scale_block_k

    # W-scale chunking: 1 buffer_load_b32 covers 32 K-blocks; lazy chain when scale_k > 32.
    _USES_REG_W = variant in (
        "reg_preload",
        "experimental",
        "experimental_unroll2",
        "manual",
    )
    NUM_W_CHUNKS = (scale_k + 31) // 32 if _USES_REG_W else 1
    USES_W_CHUNK_PREFETCH = _USES_REG_W and NUM_W_CHUNKS > 1

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
    lds_b_stride_bytes = tile_k * 16
    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = (tile_n // 16) * lds_b_stride_bytes

    # X-scale LDS (TDM-staged): tile_m rows × scales_per_tile × 4B per stage.
    USES_X_SCALE_TDM = variant in (
        "reg_preload",
        "experimental",
        "experimental_unroll2",
        "manual",
    )
    lds_x_scale_row_bytes = scales_per_tile * 4
    lds_x_scale_data_bytes = tile_m * lds_x_scale_row_bytes if USES_X_SCALE_TDM else 0

    # Unified LDS allocator: lays out [A0|A1|...|A_nb-1 | B0|...|B_nb-1 |
    # X0|...|X_nb-1] in a single contiguous global LDS region. This lets
    # us address slot i with `region_base + i * slot_stride_bytes`, so
    # helpers can take an SSA `buf_idx_i32` instead of a Python int and the
    # main loop can be a single-tile-per-iter scf.for (Gluon style).
    unified_alloc = SmemAllocator(None, arch=gpu_arch, global_sym_name="a8w8bs_unified")
    unified_a_off = unified_alloc._align(unified_alloc.ptr, 16)
    unified_alloc.ptr = unified_a_off + num_buffers * lds_a_data_bytes
    unified_b_off = unified_alloc._align(unified_alloc.ptr, 16)
    unified_alloc.ptr = unified_b_off + num_buffers * lds_b_data_bytes
    if USES_X_SCALE_TDM:
        unified_x_scale_off = unified_alloc._align(unified_alloc.ptr, 16)
        unified_alloc.ptr = unified_x_scale_off + num_buffers * lds_x_scale_data_bytes
    else:
        unified_x_scale_off = 0

    # Per-region per-slot byte offsets within the unified allocator. Used
    # only by the prologue/drain (compile-time buf_idx); the main loop uses
    # SSA buf_idx and computes addresses arithmetically.
    stage_a_data_off = [
        unified_a_off + i * lds_a_data_bytes for i in range(num_buffers)
    ]
    stage_b_data_off = [
        unified_b_off + i * lds_b_data_bytes for i in range(num_buffers)
    ]
    if USES_X_SCALE_TDM:
        stage_x_scale_off = [
            unified_x_scale_off + i * lds_x_scale_data_bytes for i in range(num_buffers)
        ]
    else:
        stage_x_scale_off = []
    # Compatibility shim: every site that wants `stage_allocators[i]` is
    # really asking for the unified base.
    stage_allocators = [unified_alloc] * num_buffers

    if use_tdm_store:
        lds_d_row_stride_bytes = tile_n * elem_bytes_d + LDS_PAD_D_BYTES
        total_d_bytes = tile_m * lds_d_row_stride_bytes
        _lds_d_stride_elems_d = lds_d_row_stride_bytes // 2
        _n_col_d_elems_d = WMMA_N * elem_bytes_d // 2

        d_lds_allocator = SmemAllocator(
            None,
            arch=gpu_arch,
            global_sym_name="a8w8bs_d_out",
        )
        d_lds_allocator.ptr = total_d_bytes

    prologue_tiles = num_buffers - 1
    main_loop_iters = (num_k_tiles - prologue_tiles) // num_buffers
    extra_tiles = num_k_tiles - main_loop_iters * num_buffers - prologue_tiles
    drain_iters = num_buffers - 2

    # TDMs per tile: 2 (X+W) for no_op_preload, 3 (X+W+scale) for reg_preload.
    _TDMS_PER_TILE_EXP = 3 if USES_X_SCALE_TDM else 2
    MAIN_TDM_OUTSTANDING = (num_buffers - 2) * 2
    MAIN_TDM_OUTSTANDING_EXPERIMENTAL = (num_buffers - 2) * _TDMS_PER_TILE_EXP

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

        scale_zero = arith.constant(0.0, type=T.f32)

        # One memref per region spanning all `num_buffers` slots. ds_loads
        # for slot i use `lane_base + i * lds_<region>_data_bytes` (offset is
        # SSA in the main loop, Python int in prologue/drain).
        big_a = SmemPtr(
            unified_alloc.get_base(),
            unified_a_off,
            T.f8,
            shape=(num_buffers * lds_a_data_bytes,),
        )
        big_b = SmemPtr(
            unified_alloc.get_base(),
            unified_b_off,
            T.f8,
            shape=(num_buffers * lds_b_data_bytes,),
        )
        big_a_mem = big_a.get()
        big_b_mem = big_b.get()
        # Per-stage memref aliases (slot 0 sub-views) used only for building
        # the TDM descriptors — they need a per-slot memref to compute
        # alignments. The actual TDM lds_addr i32 we use for slot i is
        # derived from slot-0's i32 + i * slot_size below.
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
        stages_a_mem = [p.get() for p in stages_a]
        stages_b_mem = [p.get() for p in stages_b]

        # TDM descriptors built once at entry; GROUP1 + addr_hi stay in SGPRs, lo32 advances per K-tile.
        def _make_desc_x(lds_mem, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_x,
                lds_memref=lds_mem,
                global_offset=(blk_m, k_base),
                tensor_shape=(tile_m, tile_k),
                strides=(K, 1),
                tile_shape=(tile_m, tile_k),
                elem_bytes=1,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_A_BYTES,
                num_warps=num_warps,
            )

        def _make_desc_w(lds_mem, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w,
                lds_memref=lds_mem,
                global_offset=(blk_n // arith.index(16), k_base * arith.index(16)),
                tensor_shape=(tile_n // 16, tile_k * 16),
                strides=(K * 16, 1),
                tile_shape=(tile_n // 16, tile_k * 16),
                elem_bytes=1,
                pad_interval=tile_k * 16,
                pad_amount=0,
                num_warps=num_warps,
            )

        _desc_x_init = _make_desc_x(stages_a_mem[0], arith.index(0))
        _desc_w_init = _make_desc_w(stages_b_mem[0], arith.index(0))

        dgroup1_x = _desc_x_init.dgroup1
        dgroup1_w = _desc_w_init.dgroup1
        addr_hi_x = vector.extract(
            _desc_x_init.dgroup0, static_position=[3], dynamic_position=[]
        )
        addr_hi_w = vector.extract(
            _desc_w_init.dgroup0, static_position=[3], dynamic_position=[]
        )
        addr_lo_x_init = vector.extract(
            _desc_x_init.dgroup0, static_position=[2], dynamic_position=[]
        )
        addr_lo_w_init = vector.extract(
            _desc_w_init.dgroup0, static_position=[2], dynamic_position=[]
        )

        # Slot-0 LDS i32 base addresses (pre-extracted from descriptor).
        # Slot i LDS address = slot_0_base + i * slot_stride_bytes (computed
        # arithmetically at runtime when buf_idx is SSA).
        a_lds_base_i32 = vector.extract(
            _make_desc_x(stages_a_mem[0], arith.index(0)).dgroup0,
            static_position=[1],
            dynamic_position=[],
        )
        b_lds_base_i32 = vector.extract(
            _make_desc_w(stages_b_mem[0], arith.index(0)).dgroup0,
            static_position=[1],
            dynamic_position=[],
        )
        slot_stride_a_i32 = arith.constant(lds_a_data_bytes, type=T.i32)
        slot_stride_b_i32 = arith.constant(lds_b_data_bytes, type=T.i32)

        # K-axis innermost (stride=1, elem=1B): per-tile lo32 advance = tile_k bytes.
        adv_x_i32 = arith.constant(tile_k, type=T.i32)
        adv_w_i32 = arith.constant(tile_k * 16, type=T.i32)
        pred_const = arith.constant(1, type=T.i32)

        def _buf_idx_to_i32(buf_idx):
            """Accept either Python int or i32 SSA; return i32 SSA."""
            if const_expr(isinstance(buf_idx, int)):
                return arith.constant(buf_idx, type=T.i32)
            else:
                return buf_idx

        def issue_tdm_loads(buf_idx, lo_x, lo_w):
            """Issue X+W TDMs for one K-tile into LDS stage `buf_idx`.
            `buf_idx` may be a Python int (prologue/drain) or an i32 SSA
            value (main loop). Returns advanced (lo_x, lo_w)."""
            buf_i32 = _buf_idx_to_i32(buf_idx)
            a_addr = arith.addi(a_lds_base_i32, arith.muli(buf_i32, slot_stride_a_i32))
            b_addr = arith.addi(b_lds_base_i32, arith.muli(buf_i32, slot_stride_b_i32))
            dg0_x = vector.from_elements(
                T.vec(4, T.i32), [pred_const, a_addr, lo_x, addr_hi_x]
            )
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_x, dgroup1_x))
            dg0_w = vector.from_elements(
                T.vec(4, T.i32), [pred_const, b_addr, lo_w, addr_hi_w]
            )
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0_w, dgroup1_w))
            return arith.addi(lo_x, adv_x_i32), arith.addi(lo_w, adv_w_i32)

        # ── X-scale TDM descriptor + LDS staging (hoisted) ──────────────────
        if const_expr(USES_X_SCALE_TDM):
            # One big memref spanning all num_buffers slots.
            big_x_scale = SmemPtr(
                unified_alloc.get_base(),
                unified_x_scale_off,
                T.f8,
                shape=(num_buffers * lds_x_scale_data_bytes,),
            )
            big_x_scale_mem = big_x_scale.get()
            # Per-stage memref for descriptor construction (slot 0).
            stages_x_scale = [
                SmemPtr(
                    stage_allocators[i].get_base(),
                    stage_x_scale_off[i],
                    T.f8,
                    shape=(lds_x_scale_data_bytes,),
                )
                for i in range(num_buffers)
            ]
            stages_x_scale_mem = [p.get() for p in stages_x_scale]

            def _make_desc_x_scale(lds_mem, kb_base):
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_x_scale,
                    lds_memref=lds_mem,
                    global_offset=(blk_m, kb_base),
                    tensor_shape=(tile_m, scales_per_tile),
                    strides=(scale_k, 1),
                    tile_shape=(tile_m, scales_per_tile),
                    elem_bytes=4,
                    pad_interval=scales_per_tile,
                    pad_amount=0,
                    num_warps=num_warps,
                )

            _desc_x_scale_init = _make_desc_x_scale(
                stages_x_scale_mem[0], arith.index(0)
            )
            dgroup1_x_scale = _desc_x_scale_init.dgroup1
            addr_hi_x_scale = vector.extract(
                _desc_x_scale_init.dgroup0, static_position=[3], dynamic_position=[]
            )
            addr_lo_x_scale_init = vector.extract(
                _desc_x_scale_init.dgroup0, static_position=[2], dynamic_position=[]
            )

            x_scale_lds_base_i32 = vector.extract(
                _desc_x_scale_init.dgroup0,
                static_position=[1],
                dynamic_position=[],
            )
            slot_stride_x_scale_i32 = arith.constant(lds_x_scale_data_bytes, type=T.i32)

            # K-axis advance per K-tile: scales_per_tile K-blocks × 4 B/elem.
            adv_x_scale_i32 = arith.constant(scales_per_tile * 4, type=T.i32)

            def issue_x_scale_tdm(buf_idx, lo_x_scale):
                """Issue one x_scale K-tile TDM load into LDS stage `buf_idx`.
                `buf_idx` may be Python int or i32 SSA. Returns lo_x_scale
                advanced by scales_per_tile*4 bytes."""
                buf_i32 = _buf_idx_to_i32(buf_idx)
                xs_addr = arith.addi(
                    x_scale_lds_base_i32,
                    arith.muli(buf_i32, slot_stride_x_scale_i32),
                )
                dg0 = vector.from_elements(
                    T.vec(4, T.i32),
                    [pred_const, xs_addr, lo_x_scale, addr_hi_x_scale],
                )
                tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, dgroup1_x_scale))
                return arith.addi(lo_x_scale, adv_x_scale_i32)

            def ds_read_x_scales(buf_idx):
                """Read this warp's x_scales for one K-tile from LDS stage
                `buf_idx` (Python int or i32 SSA). Returns flat list with
                indexing `out[sc * wmma_m_rep + wm]`. Per-lane: ds_read_b32
                at slot_byte_off + (warp_m_base + wm*WMMA_M + lane16) *
                scales_per_tile*4 + sc*4."""
                if const_expr(isinstance(buf_idx, int)):
                    slot_byte_off = arith.index(buf_idx * lds_x_scale_data_bytes)
                else:
                    slot_byte_off = arith.index_cast(
                        T.index,
                        arith.muli(buf_idx, slot_stride_x_scale_i32),
                    )
                out = []
                row_stride_bytes = scales_per_tile * 4
                for sc in range_constexpr(scales_per_tile):
                    for wm in range_constexpr(wmma_m_rep):
                        row = warp_m_base + arith.index(wm * WMMA_M) + lane16
                        off = (
                            slot_byte_off
                            + row * arith.index(row_stride_bytes)
                            + arith.index(sc * 4)
                        )
                        out.append(lds_load_b32_f32(big_x_scale_mem, off))
                return out

        w_is_wave_uniform = warp_tile_n <= scale_block_n
        # Hoisted out of `if const_expr(w_is_wave_uniform):` so closures wrapped by the
        # AST rewriter for downstream branches can resolve it.
        wave_n_block = (blk_n + warp_n_base) / arith.index(scale_block_n)

        # Bulk W-scale load: 1 buffer_load_b32 covers 32 K-blocks; chunk-prefetch chain when scale_k > 32.
        if const_expr(_USES_REG_W):
            lane_id_full = lane_kgrp * arith.index(16) + lane16

            def _issue_w_chunk_const(chunk_i):
                """Issue one bulk W-scale load for compile-time chunk_i."""
                offset = arith.index(chunk_i * 32)
                idx = wave_n_block * arith.index(scale_k) + lane_id_full + offset
                return buffer_ops.buffer_load(
                    w_scale_buf, idx, vec_width=1, dtype=T.f32
                )

            def _issue_w_chunk_runtime(chunk_idx_i32):
                """Issue one bulk W-scale load for runtime chunk_idx_i32.
                Index is clamped to NUM_W_CHUNKS-1 so out-of-range issues are
                cache-cheap re-loads of the last chunk (never readlane'd)."""
                clamped_i32 = arith.minui(
                    chunk_idx_i32,
                    arith.constant(NUM_W_CHUNKS - 1, type=T.i32),
                )
                offset_i32 = arith.muli(clamped_i32, arith.constant(32, type=T.i32))
                offset = arith.index_cast(T.index, offset_i32)
                idx = wave_n_block * arith.index(scale_k) + lane_id_full + offset
                return buffer_ops.buffer_load(
                    w_scale_buf, idx, vec_width=1, dtype=T.f32
                )

            # Deferred to prologue; _w_readlane resolves these at call time.
            bulk_w_cur = None
            bulk_w_prefetch = None
            cur_chunk_idx_i32 = arith.constant(0, type=T.i32)

        def _w_readlane(kb_i32):
            """Fetch w_scale[wave_n_block, kb] for the experimental variant.
            Single-chunk: direct readlane from bulk_w_cur. Multi-chunk:
            picks bulk_w_cur or bulk_w_prefetch based on kb's chunk index
            (vs. the loop-carried cur_chunk_idx_i32), then readlanes."""
            if const_expr(NUM_W_CHUNKS == 1):
                return rocdl.readlane(T.f32, bulk_w_cur, kb_i32)
            kb_chunk_i32 = arith.shrui(kb_i32, arith.constant(5, type=T.i32))
            lane_in_chunk_i32 = arith.andi(kb_i32, arith.constant(31, type=T.i32))
            is_cur = arith.cmpi(arith.CmpIPredicate.eq, kb_chunk_i32, cur_chunk_idx_i32)
            chosen = arith.select(is_cur, bulk_w_cur, bulk_w_prefetch)
            return rocdl.readlane(T.f32, chosen, lane_in_chunk_i32)

        def issue_raw_scales(k_base):
            """Fire buffer_loads for one K-tile's x_scale + w_scale values —
            no multiply. Returns (x_raw, w_raw) flat lists.

            reg_preload / no_op_preload variant: lane16-strided x_scale loads
            (one per wm, 16 unique rows replicated across the upper-half lanes)
            and a direct buffer_load per K-block for w_scale.

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
                    x_raw.append(
                        buffer_ops.buffer_load(
                            x_scale_buf, idx, vec_width=1, dtype=T.f32
                        )
                    )
                if const_expr(w_is_wave_uniform):
                    idx = wave_n_block * arith.index(scale_k) + kb
                    w_val = buffer_ops.buffer_load(
                        w_scale_buf, idx, vec_width=1, dtype=T.f32
                    )
                    for wn in range_constexpr(wmma_n_rep):
                        w_raw.append(w_val)
                else:
                    for wn in range_constexpr(wmma_n_rep):
                        col = (
                            blk_n
                            + warp_n_base
                            + arith.index(wn * WMMA_N)
                            + lane_kgrp * arith.index(8)
                        )
                        n_block = col / arith.index(scale_block_n)
                        idx = n_block * arith.index(scale_k) + kb
                        w_raw.append(
                            buffer_ops.buffer_load(
                                w_scale_buf, idx, vec_width=1, dtype=T.f32
                            )
                        )
            return x_raw, w_raw

        def issue_raw_scales_for_tile(tile_idx):
            """Issue raw scales for a compile-time tile index (reg_preload)."""
            return issue_raw_scales(arith.index(tile_idx * tile_k))

        def issue_raw_scales_for_future_tile_rt(future_tile_rt):
            """Runtime-safe raw-scale prefetch for dynamic main-loop tiles
            (reg_preload). If `future_tile_rt` is out of range, issue a safe
            in-range load and mask results to zero.
            """
            future_tile_i32 = arith.index_cast(T.i32, future_tile_rt)
            valid_future = arith.cmpi(
                arith.CmpIPredicate.ult,
                future_tile_i32,
                arith.constant(num_k_tiles, type=T.i32),
            )
            safe_tile_i32 = arith.select(
                valid_future, future_tile_i32, arith.constant(0, type=T.i32)
            )
            safe_tile_idx = arith.index_cast(T.index, safe_tile_i32)
            safe_k_base = safe_tile_idx * arith.index(tile_k)
            raw_x, raw_w = issue_raw_scales(safe_k_base)
            masked_x = [arith.select(valid_future, v, scale_zero) for v in raw_x]
            masked_w = [arith.select(valid_future, v, scale_zero) for v in raw_w]
            return masked_x, masked_w

        # W-scale issue: chunk-cached readlane (wave-uniform) or per-(wn) buffer_load.
        def issue_w_raw_scales_experimental(k_base):
            """Returns w_raw flat list, indexed [sc * wmma_n_rep + wn] =
            w_scale[n_block=wn, kb=sc]. All wn entries equal when
            w_is_wave_uniform."""
            kb_base = k_base / arith.index(scale_block_k)
            w_raw = []
            for sc in range_constexpr(scales_per_tile):
                kb = kb_base + arith.index(sc)
                if const_expr(w_is_wave_uniform):
                    kb_i32 = arith.index_cast(T.i32, kb)
                    w_val = _w_readlane(kb_i32)
                    for wn in range_constexpr(wmma_n_rep):
                        w_raw.append(w_val)
                else:
                    for wn in range_constexpr(wmma_n_rep):
                        col = (
                            blk_n
                            + warp_n_base
                            + arith.index(wn * WMMA_N)
                            + lane_kgrp * arith.index(8)
                        )
                        n_block = col / arith.index(scale_block_n)
                        idx = n_block * arith.index(scale_k) + kb
                        w_raw.append(
                            buffer_ops.buffer_load(
                                w_scale_buf, idx, vec_width=1, dtype=T.f32
                            )
                        )
            return w_raw

        def issue_w_raw_scales_for_tile_experimental(tile_idx):
            """W-scales for a compile-time tile index (experimental)."""
            return issue_w_raw_scales_experimental(arith.index(tile_idx * tile_k))

        def issue_w_raw_scales_for_future_tile_rt_experimental(future_tile_rt):
            """Runtime-safe W-scale prefetch for dynamic main-loop tiles
            (experimental). Out-of-range future tiles get zero-masked."""
            future_tile_i32 = arith.index_cast(T.i32, future_tile_rt)
            valid_future = arith.cmpi(
                arith.CmpIPredicate.ult,
                future_tile_i32,
                arith.constant(num_k_tiles, type=T.i32),
            )
            safe_tile_i32 = arith.select(
                valid_future, future_tile_i32, arith.constant(0, type=T.i32)
            )
            safe_tile_idx = arith.index_cast(T.index, safe_tile_i32)
            safe_k_base = safe_tile_idx * arith.index(tile_k)
            raw_w = issue_w_raw_scales_experimental(safe_k_base)
            masked_w = [arith.select(valid_future, v, scale_zero) for v in raw_w]
            return masked_w

        # lane_kgrp selects K-half: kgrp=0 → bytes [0..63], kgrp=1 → [64..127].
        k_half_byte_offset = lane_kgrp * arith.index(64)

        b_k_half_byte_offset = lane_kgrp * arith.index(256)

        def _compute_lane_bases(warp_base, stride_bytes, num_reps, rep_stride_elems):
            """Compute per-lane LDS byte offsets for loading `num_reps` WMMA
            frags along M or N. Returns a list of base offsets indexed by rep."""
            row_base_bytes = (warp_base + lane16) * arith.index(stride_bytes)
            bases = []
            for rep in range_constexpr(num_reps):
                base = (
                    row_base_bytes
                    + arith.index(rep * rep_stride_elems * stride_bytes)
                    + k_half_byte_offset
                )
                bases.append(base)
            return bases

        def _compute_b_lane_bases_preshuffled(warp_base, num_reps):
            bases = []
            for rep in range_constexpr(num_reps):
                stripe_offset = (
                    (warp_base + arith.index(rep * WMMA_N))
                    / arith.index(WMMA_N)
                    * arith.index(lds_b_stride_bytes)
                )
                base = stripe_offset + b_k_half_byte_offset + lane16 * arith.index(16)
                bases.append(base)
            return bases

        def _load_frag(
            lds_memref, lane_base, ks, cycle_stride_bytes=16, ks_stride_bytes=WMMA_K
        ):
            """Load one WMMA frag (16 × b128) from LDS into a vector<16xi32>
            per lane, starting at byte offset (lane_base + ks * WMMA_K)."""
            k_sub_off = arith.index(ks * ks_stride_bytes)
            off = lane_base + k_sub_off
            v0 = lds_load_b128(lds_memref, off)
            v1 = lds_load_b128(lds_memref, off + arith.index(cycle_stride_bytes))
            v2 = lds_load_b128(lds_memref, off + arith.index(cycle_stride_bytes * 2))
            v3 = lds_load_b128(lds_memref, off + arith.index(cycle_stride_bytes * 3))
            v01 = vector.shuffle(v0, v1, list(range(8)))
            v23 = vector.shuffle(v2, v3, list(range(8)))
            return vector.shuffle(v01, v23, list(range(16)))

        a_lane_bases = _compute_lane_bases(
            warp_m_base, lds_a_stride_bytes, wmma_m_rep, WMMA_M
        )
        b_lane_bases = _compute_b_lane_bases_preshuffled(warp_n_base, wmma_n_rep)

        def load_operand_frags(buffer_idx):
            """Load all A/B frags for one K-tile from LDS stage
            `buffer_idx` (Python int or i32 SSA).

            Returns (a_frags, b_frags) with indexing:
                a_frags[ks * wmma_m_rep + wm]
                b_frags[ks * wmma_n_rep + wn]
            """
            if const_expr(isinstance(buffer_idx, int)):
                slot_off_a = arith.index(buffer_idx * lds_a_data_bytes)
                slot_off_b = arith.index(buffer_idx * lds_b_data_bytes)
            else:
                slot_off_a = arith.index_cast(
                    T.index, arith.muli(buffer_idx, slot_stride_a_i32)
                )
                slot_off_b = arith.index_cast(
                    T.index, arith.muli(buffer_idx, slot_stride_b_i32)
                )
            a_frags = []
            b_frags = []
            for ks in range_constexpr(k_wmma_steps):
                for wm in range_constexpr(wmma_m_rep):
                    a_frags.append(
                        _load_frag(big_a_mem, a_lane_bases[wm] + slot_off_a, ks)
                    )
                for wn in range_constexpr(wmma_n_rep):
                    b_frags.append(
                        _load_frag(
                            big_b_mem,
                            b_lane_bases[wn] + slot_off_b,
                            ks,
                            cycle_stride_bytes=512,
                            ks_stride_bytes=2048,
                        )
                    )
            return a_frags, b_frags

        def load_operand_frags_with_xscale_interleave(buffer_idx):
            """Same as load_operand_frags but issues the X-scale ds_read
            *between* K-step 0's frags and K-step 1's frags, so X-scale lands
            in registers as early as possible — ready for the first FMA right
            after WMMA #1 completes, instead of being placed near the tail of
            the ds_load burst by the LLVM scheduler.

            Returns (a_frags, b_frags, x_raw).
            """
            if const_expr(isinstance(buffer_idx, int)):
                slot_off_a = arith.index(buffer_idx * lds_a_data_bytes)
                slot_off_b = arith.index(buffer_idx * lds_b_data_bytes)
            else:
                slot_off_a = arith.index_cast(
                    T.index, arith.muli(buffer_idx, slot_stride_a_i32)
                )
                slot_off_b = arith.index_cast(
                    T.index, arith.muli(buffer_idx, slot_stride_b_i32)
                )
            a_frags = []
            b_frags = []
            x_raw = None
            for ks in range_constexpr(k_wmma_steps):
                for wm in range_constexpr(wmma_m_rep):
                    a_frags.append(
                        _load_frag(big_a_mem, a_lane_bases[wm] + slot_off_a, ks)
                    )
                for wn in range_constexpr(wmma_n_rep):
                    b_frags.append(
                        _load_frag(
                            big_b_mem,
                            b_lane_bases[wn] + slot_off_b,
                            ks,
                            cycle_stride_bytes=512,
                            ks_stride_bytes=2048,
                        )
                    )
                # Drop the X-scale ds_read into the gap right after K-step 0's
                # 8 frag ds_loads. sched_barrier pins it so the LLVM scheduler
                # can't slide it back to the tail of the ds_load burst.
                if const_expr(ks == 0):
                    rocdl.sched_barrier(0)
                    x_raw = ds_read_x_scales(buffer_idx)
                    rocdl.sched_barrier(0)
            return a_frags, b_frags, x_raw

        # ═══════════════════════════════════════════════════════════════════
        # HELPERS: WMMA compute + scale FMA
        # ═══════════════════════════════════════════════════════════════════

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))

        def compute_wmma_with_frags(global_accs, a_frags, b_frags, x_raw, w_raw):
            """2-deep rolling WMMA/FMA pipeline (reg_preload / no_op_preload).

            Non-transposed WMMA: ISA operand order (B, A, C). Output layout is
            lane→(row=lane16, col_band=lane_kgrp*8+0..7), so each lane's vec<8>
            shares one row → x_scale broadcasts as a scalar.

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
                    temp = rocdl.wmma_f32_16x16x128_fp8_fp8(
                        T.vec(8, T.f32),
                        b_frag,
                        a_frag,
                        temp,
                    ).result
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
                # rocdl.s_setprio(2)
                # hold onto a temp wmma to prevent the next instr from using fma on same vgpr (vnop issue)
                temp0 = issue_wmma_temp(sc, wm0, wn0)
                if const_expr(n_accs > 1):
                    wm1, wn1, idx1 = acc_coords[1]
                    temp1 = issue_wmma_temp(sc, wm1, wn1)
                # Might not need this since dscnt 0 is gone
                # rocdl.s_setprio(0)

                if const_expr(n_accs == 1):
                    wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                else:
                    for i in range_constexpr(n_accs - wmma_pipeline_depth):
                        wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                        wm0, wn0, idx0 = wm1, wn1, idx1
                        temp0 = temp1
                        wm1, wn1, idx1 = acc_coords[i + wmma_pipeline_depth]
                        temp1 = issue_wmma_temp(sc, wm1, wn1)
                    wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                    wmma_with_scale(temp1, wm1, wn1, idx1, sc_x_base, sc_w_base)

            return global_accs

        # ─────────────────────────────────────────────────────────────────
        # Split-out primitive helpers (used by compute_wmma_with_frags_experimental
        # below — and intended to be called directly from the main loop when
        # hand-orchestrating WMMA/scale/FMA ordering, e.g. to force WMMAs in
        # K-step source order so each one's `s_wait_dscnt` is minimal).
        # ─────────────────────────────────────────────────────────────────

        def issue_wmma_step(sc, wm, wn, a_frags, b_frags):
            """Issue WMMA(s) for scale-block `sc`, M-rep `wm`, N-rep `wn`.

            For each ks_inner in [0, wmma_steps_per_scale), accumulates one
            16×16×128 fp8 WMMA partial product into `temp` (seed = acc_zero,
            so iter 0 is effectively WMMA(B, A, 0)). For our shape
            wmma_steps_per_scale == 1 so this issues exactly one WMMA.

            Returns the partial-sum vec<8 f32> (one per lane), to be scaled
            and folded into a global accumulator by `apply_scale`.
            """
            temp = acc_zero
            for ks_inner in range_constexpr(wmma_steps_per_scale):
                ks = sc * wmma_steps_per_scale + ks_inner
                a_frag = a_frags[ks * wmma_m_rep + wm]
                b_frag = b_frags[ks * wmma_n_rep + wn]
                # ISA operand order: (B, A, C), reversed from math.
                temp = rocdl.wmma_f32_16x16x128_fp8_fp8(
                    T.vec(8, T.f32),
                    b_frag,
                    a_frag,
                    temp,
                ).result
            return temp

        def compute_scale_step(sc, wm, wn, x_raw, w_raw):
            """Compute the combined per-row × per-col fp32 scale for
            scale-block `sc`, M-rep `wm`, N-rep `wn`. Returns an fp32 scalar
            (one per lane, but constant across the 8 outputs of one WMMA so
            broadcast at FMA time)."""
            sc_x_base = sc * wmma_m_rep
            sc_w_base = sc * wmma_n_rep
            return arith.mulf(x_raw[sc_x_base + wm], w_raw[sc_w_base + wn])

        def apply_scale(temp, scale, acc):
            """FMA: returns `temp * broadcast(scale) + acc`. `scale` is an
            fp32 scalar; broadcast to vec<8 f32> for the packed FMA. `acc`
            is the running accumulator (vec<8 f32>) — caller is responsible
            for storing the returned value back into its accumulator slot
            (typically `global_accs[idx] = apply_scale(...)`)."""
            scale_vec = vector.broadcast(T.vec(8, T.f32), scale)
            return math_dialect.fma(temp, scale_vec, acc)

        def compute_wmma_with_frags_experimental(
            global_accs, a_frags, b_frags, x_raw, w_raw
        ):
            """2-deep rolling WMMA/FMA pipeline (experimental variant).

            Non-transposed WMMA: ISA operand order (B, A, C). Same WMMA
            output layout as reg_preload — each lane's vec<8> shares one row,
            so the per-row x_scale broadcasts as a scalar at FMA time.

            Pattern per scale block matches the reg_preload version. Kept as
            a separate function so the experimental path can diverge from
            reg_preload independently (e.g., scale-apply rewrites or
            instruction scheduling experiments).

            Now built on top of the split-out helpers (issue_wmma_step,
            compute_scale_step, apply_scale) so the main loop can call those
            directly when it needs hand-tuned ordering.
            """

            def issue_wmma_temp(sc, wm, wn):
                return issue_wmma_step(sc, wm, wn, a_frags, b_frags)

            def compute_scale(wm, wn, sc_x_base, sc_w_base):
                # Local shim for backward-compat with the rolling-pipeline
                # body below — note: takes pre-computed sc_x_base / sc_w_base,
                # whereas compute_scale_step takes `sc` and computes those.
                return arith.mulf(x_raw[sc_x_base + wm], w_raw[sc_w_base + wn])

            def wmma_with_scale(temp, wm, wn, idx, sc_x_base, sc_w_base):
                scale = compute_scale(wm, wn, sc_x_base, sc_w_base)
                global_accs[idx] = apply_scale(temp, scale, global_accs[idx])

            for sc in range_constexpr(scales_per_tile):
                sc_x_base = sc * wmma_m_rep
                sc_w_base = sc * wmma_n_rep

                wm0, wn0, idx0 = acc_coords[0]
                # rocdl.s_setprio(2)
                temp0 = issue_wmma_temp(sc, wm0, wn0)
                if const_expr(n_accs > 1):
                    wm1, wn1, idx1 = acc_coords[1]
                    temp1 = issue_wmma_temp(sc, wm1, wn1)
                # rocdl.s_setprio(0)

                if const_expr(n_accs == 1):
                    wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                else:
                    for i in range_constexpr(n_accs - wmma_pipeline_depth):
                        wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                        wm0, wn0, idx0 = wm1, wn1, idx1
                        temp0 = temp1
                        wm1, wn1, idx1 = acc_coords[i + wmma_pipeline_depth]
                        temp1 = issue_wmma_temp(sc, wm1, wn1)
                    wmma_with_scale(temp0, wm0, wn0, idx0, sc_x_base, sc_w_base)
                    wmma_with_scale(temp1, wm1, wn1, idx1, sc_x_base, sc_w_base)

            return global_accs

        def hand_ordered_wmma_fma(accs, a_frags, b_frags, x_raw, w_raw):
            """Hand-ordered K-step WMMA→FMA chain (n_accs == 1 only).

            Forces LLVM to keep WMMAs in source K-step order
            (K0→K1→K2→K3) by inserting `rocdl.sched_barrier(0)` at the
            end of each iteration body. Without this, the scheduler
            reorders to (K0→K3→K1→K2), which forces `s_wait_dscnt 0x6`
            (drain 27 of 33) before WMMA #1 because WMMA #2 (=K3) needs
            positions 26–33.

            With K-step order pinned, each WMMA only waits for *its*
            K-step's frags retired:
                WMMA #0 (K0): dscnt 0x19 (drain 8)
                WMMA #1 (K1): dscnt 0x10 (drain 17)
                WMMA #2 (K2): dscnt 0x8  (drain 25)
                WMMA #3 (K3): dscnt 0x0  (drain 33)
            WMMA #0 fires ~60 cyc earlier; subsequent WMMAs overlap the
            tail of the ds_load drain.

            For n_accs > 1 falls back to the existing rolling-pipeline
            path (compute_wmma_with_frags_experimental).
            """
            if const_expr(n_accs == 1):
                accs = list(accs)
                wm, wn, idx = acc_coords[0]
                for sc in range_constexpr(scales_per_tile):
                    temp = issue_wmma_step(sc, wm, wn, a_frags, b_frags)
                    scale = compute_scale_step(sc, wm, wn, x_raw, w_raw)
                    accs[idx] = apply_scale(temp, scale, accs[idx])
                    # Pin the (WMMA, FMA) for this sc — next sc's WMMA
                    # cannot be hoisted above this barrier, preserving
                    # source K-step order in the emitted ASM.
                    rocdl.sched_barrier(0)
                return accs
            else:
                return compute_wmma_with_frags_experimental(
                    accs, a_frags, b_frags, x_raw, w_raw
                )

        def compute_wmma_with_frags_manual(global_accs, a_frags, b_frags, x_raw, w_raw):
            """Hand-scheduled compute used by variant='manual'.

            Per scale block, emit the WMMAs as one contiguous block and the
            scale FMAs as a separate block, with `rocdl.sched_barrier(0)`
            between them. The barrier prevents the LLVM scheduler from
            interleaving an FMA right after the WMMA that wrote its temp,
            which would force a v_nop on gfx1250 (the FMA reads a VGPR
            still being written by the WMMA's pipeline stage).

            A trailing sched_barrier pins the FMA block against the next
            scale block's WMMA block. The result is:

                [WMMA #0 .. WMMA #(n_accs-1)]   ← block 1
                sched_barrier
                [FMA  #0 .. FMA  #(n_accs-1)]   ← block 2
                sched_barrier
                (next sc's WMMA block …)

            Compared to compute_wmma_with_frags_experimental (2-deep
            rolling WMMA/FMA pipeline), this trades some WMMA→FMA overlap
            for elimination of the v_nop class. Intended as a
            perf-exploration baseline; see kernel-trace-analysis to
            measure the trade.
            """
            for sc in range_constexpr(scales_per_tile):
                # Block 1: all WMMAs for this scale block.
                temps = []
                for wm, wn, _idx in acc_coords:
                    temps.append(issue_wmma_step(sc, wm, wn, a_frags, b_frags))

                rocdl.sched_barrier(0)

                # Block 2: all FMAs (scale apply) for this scale block.
                for i, (wm, wn, idx) in enumerate(acc_coords):
                    scale = compute_scale_step(sc, wm, wn, x_raw, w_raw)
                    global_accs[idx] = apply_scale(temps[i], scale, global_accs[idx])

                rocdl.sched_barrier(0)

            return global_accs

        def load_operand_frags_chunked(buffer_idx):
            """A/B operand-frag ds_load split into two bursts with a
            sched_barrier between, so the LLVM scheduler can't merge them
            into one >16-load burst. Used by variant='manual'.

            Burst 1: all A frags  (= 4 * wmma_m_rep * k_wmma_steps ds_load_b128)
            Burst 2: all B frags  (= 4 * wmma_n_rep * k_wmma_steps ds_load_b128)

            Returns (a_frags, b_frags) with the same indexing convention as
            load_operand_frags.
            """
            if const_expr(isinstance(buffer_idx, int)):
                slot_off_a = arith.index(buffer_idx * lds_a_data_bytes)
                slot_off_b = arith.index(buffer_idx * lds_b_data_bytes)
            else:
                slot_off_a = arith.index_cast(
                    T.index, arith.muli(buffer_idx, slot_stride_a_i32)
                )
                slot_off_b = arith.index_cast(
                    T.index, arith.muli(buffer_idx, slot_stride_b_i32)
                )
            a_frags = []
            for ks in range_constexpr(k_wmma_steps):
                for wm in range_constexpr(wmma_m_rep):
                    a_frags.append(
                        _load_frag(big_a_mem, a_lane_bases[wm] + slot_off_a, ks)
                    )
            rocdl.sched_barrier(0)
            b_frags = []
            for ks in range_constexpr(k_wmma_steps):
                for wn in range_constexpr(wmma_n_rep):
                    b_frags.append(
                        _load_frag(
                            big_b_mem,
                            b_lane_bases[wn] + slot_off_b,
                            ks,
                            cycle_stride_bytes=512,
                            ks_stride_bytes=2048,
                        )
                    )
            return a_frags, b_frags

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
        # Carry shape for x_raw is wmma_m_rep entries per sc across all
        # variants — experimental's TDM scheme materializes from LDS to the
        # same lane16 layout that reg_preload / no_op_preload already use.
        N_CUR_X_RAW = scales_per_tile * wmma_m_rep
        N_CUR_W_RAW = scales_per_tile * wmma_n_rep
        N_PREFETCH_X = N_CUR_X_RAW
        N_PREFETCH_W = N_CUR_W_RAW
        zero_x_raw = [scale_zero] * N_CUR_X_RAW
        zero_w_raw = [scale_zero] * N_CUR_W_RAW

        # This packing/unpacking just sends our vars to the next iteration, stores them cleanly kinda
        def _pack_state_reg_preload(accs_, a_, b_, cur_x_, cur_w_, px, pw):
            return (
                list(accs_)
                + list(a_)
                + list(b_)
                + list(cur_x_)
                + list(cur_w_)
                + list(px)
                + list(pw)
            )

        def _unpack_state_reg_preload(state):
            i = 0
            accs_ = list(state[i : i + N_ACCS])
            i += N_ACCS
            a_ = list(state[i : i + N_A_FRAGS])
            i += N_A_FRAGS
            b_ = list(state[i : i + N_B_FRAGS])
            i += N_B_FRAGS
            cur_x_ = list(state[i : i + N_CUR_X_RAW])
            i += N_CUR_X_RAW
            cur_w_ = list(state[i : i + N_CUR_W_RAW])
            i += N_CUR_W_RAW
            px = list(state[i : i + N_PREFETCH_X])
            i += N_PREFETCH_X
            pw = list(state[i : i + N_PREFETCH_W])
            i += N_PREFETCH_W
            return accs_, a_, b_, cur_x_, cur_w_, px, pw

        def _pack_state_no_op_preload(accs_, cur_x_, cur_w_, px, pw):
            return list(accs_) + list(cur_x_) + list(cur_w_) + list(px) + list(pw)

        def _unpack_state_no_op_preload(state):
            i = 0
            accs_ = list(state[i : i + N_ACCS])
            i += N_ACCS
            cur_x_ = list(state[i : i + N_CUR_X_RAW])
            i += N_CUR_X_RAW
            cur_w_ = list(state[i : i + N_CUR_W_RAW])
            i += N_CUR_W_RAW
            px = list(state[i : i + N_PREFETCH_X])
            i += N_PREFETCH_X
            pw = list(state[i : i + N_PREFETCH_W])
            i += N_PREFETCH_W
            return accs_, cur_x_, cur_w_, px, pw

        # Experimental loop carry: same as reg_preload minus prefetch_x_raw
        # (X-scale prefetch lives in LDS, not VGPRs).
        def _pack_state_experimental(accs_, a_, b_, cur_x_, cur_w_, pw):
            return (
                list(accs_)
                + list(a_)
                + list(b_)
                + list(cur_x_)
                + list(cur_w_)
                + list(pw)
            )

        def _unpack_state_experimental(state):
            i = 0
            accs_ = list(state[i : i + N_ACCS])
            i += N_ACCS
            a_ = list(state[i : i + N_A_FRAGS])
            i += N_A_FRAGS
            b_ = list(state[i : i + N_B_FRAGS])
            i += N_B_FRAGS
            cur_x_ = list(state[i : i + N_CUR_X_RAW])
            i += N_CUR_X_RAW
            cur_w_ = list(state[i : i + N_CUR_W_RAW])
            i += N_CUR_W_RAW
            pw = list(state[i : i + N_PREFETCH_W])
            i += N_PREFETCH_W
            return accs_, a_, b_, cur_x_, cur_w_, pw

        # Prologue: issue TDMs for the first prologue_tiles, X-scale interleaved per-tile.
        lo_x = addr_lo_x_init
        lo_w = addr_lo_w_init
        if const_expr(USES_X_SCALE_TDM):
            lo_x_scale = addr_lo_x_scale_init

        # Boost wave priority for the TDM issue burst to compress wave-dispatch skew.
        rocdl.s_setprio(2)
        for i in range_constexpr(prologue_tiles):
            lo_x, lo_w = issue_tdm_loads(i, lo_x, lo_w)
            if const_expr(USES_X_SCALE_TDM):
                lo_x_scale = issue_x_scale_tdm(i, lo_x_scale)
        rocdl.s_setprio(0)

        # MAIN LOOP:
        # asm is unrolled in flydsl for range_constexpr, so we see more in asm
        # NOTE on `accs = [acc_zero] * n_accs`: this Python list creation emits
        # no IR by itself — the actual `v_mov v_, 0` zero-inits are placed by
        # LLVM near first use. Each variant declares it close to its first WMMA
        # call so the source order matches execution order.

        if const_expr(variant == "reg_preload"):
            # Bulk W-scale buffer_load deferred to here (after prologue TDM issues).
            bulk_w_cur = _issue_w_chunk_const(0)
            if const_expr(USES_W_CHUNK_PREFETCH):
                bulk_w_prefetch = _issue_w_chunk_const(1)
            else:
                bulk_w_prefetch = bulk_w_cur

            # Single wait: retires tile-0 X+W+S; leaves just-issued tiles pending.
            tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
            gpu.barrier()

            # Issue ds_loads with X-scale interleaved after K-step 0's frags,
            # so X-scale is in registers by the time WMMA #1 completes and the
            # first FMA can fire without a long dscnt drain.
            cur_a, cur_b, cur_x_raw = load_operand_frags_with_xscale_interleave(0)

            # sched_barrier pins readlane after wait so vmcnt is near zero.
            rocdl.sched_barrier(0)
            cur_w_raw = issue_w_raw_scales_for_tile_experimental(0)
            if const_expr(num_k_tiles > 1):
                prefetch_w_raw = issue_w_raw_scales_for_tile_experimental(1)
            else:
                prefetch_w_raw = zero_w_raw

            # Accumulator init lives here (just before main-loop entry) so
            # source order matches execution order. Note: emits no IR by
            # itself; v_mov_b32 zero-inits are placed lazily by LLVM near
            # first use (the first FMA in main-loop iter 0).
            accs = [acc_zero] * n_accs

            # ──────────────────────────────────────────────────────────────
            # Gluon-style main loop: ONE tile per iteration. The loop body is
            # `wmma → issue tdm → wait → ds_read next → swap`. No outer/inner
            # nesting, no scf.if dispatch — all helpers take SSA `buf_idx_i32`
            # via runtime address arithmetic on the unified LDS allocator.
            # Hitcount in rcv is uniform across the body (= main loop iters).
            # ──────────────────────────────────────────────────────────────
            main_loop_iters_g = num_k_tiles - (num_buffers - 1)

            # Loop-carried indices: load_idx starts at NB-1 (next tile to issue),
            # compute_idx starts at 0 (next tile to consume from VGPRs).
            load_idx_init = arith.constant(num_buffers - 1, type=T.i32)
            compute_idx_init = arith.constant(0, type=T.i32)

            if const_expr(main_loop_iters_g > 0):
                init_state = _pack_state_experimental(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw, prefetch_w_raw
                )
                if const_expr(USES_W_CHUNK_PREFETCH):
                    init_state = init_state + [
                        bulk_w_cur,
                        bulk_w_prefetch,
                        cur_chunk_idx_i32,
                    ]
                init_state = init_state + [
                    lo_x,
                    lo_w,
                    lo_x_scale,
                    load_idx_init,
                    compute_idx_init,
                ]

                nb_const_i32 = arith.constant(num_buffers, type=T.i32)
                one_i32 = arith.constant(1, type=T.i32)
                two_i32 = arith.constant(2, type=T.i32)

                for tile_step, state in range(0, main_loop_iters_g, 1, init=init_state):
                    _disable_unroll_on_enclosing_loop()
                    cur_compute_idx = state[-1]
                    cur_load_idx = state[-2]
                    cur_lo_x_scale = state[-3]
                    cur_lo_w = state[-4]
                    cur_lo_x = state[-5]
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        cur_chunk_idx_i32 = state[-6]
                        bulk_w_prefetch = state[-7]
                        bulk_w_cur = state[-8]
                        _reg_state = state[:-8]
                    else:
                        _reg_state = state[:-5]
                    (
                        cur_accs,
                        cur_a,
                        cur_b,
                        cur_x_raw,
                        cur_w_raw,
                        prefetch_w_raw,
                    ) = _unpack_state_experimental(_reg_state)

                    # SSA buf indices for this iteration.
                    load_buf_i32 = arith.remui(cur_load_idx, nb_const_i32)
                    next_compute_idx = arith.addi(cur_compute_idx, one_i32)
                    next_buf_i32 = arith.remui(next_compute_idx, nb_const_i32)

                    # WMMA on cur tile.
                    cur_accs = compute_wmma_with_frags_experimental(
                        cur_accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                    )

                    # Issue TDMs for tile load_idx.
                    cur_lo_x, cur_lo_w = issue_tdm_loads(
                        load_buf_i32, cur_lo_x, cur_lo_w
                    )
                    cur_lo_x_scale = issue_x_scale_tdm(load_buf_i32, cur_lo_x_scale)

                    # Wait for tile compute_idx+1 to land in LDS.
                    tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
                    gpu.barrier()

                    # Pre-load tile compute_idx+1 into VGPRs.
                    next_x_raw = ds_read_x_scales(next_buf_i32)
                    next_a, next_b = load_operand_frags(next_buf_i32)

                    cur_a = next_a
                    cur_b = next_b
                    cur_x_raw = next_x_raw
                    cur_w_raw = prefetch_w_raw

                    # Prefetch W-scale for compute_idx+2 (zero-masked if past
                    # num_k_tiles).
                    future_tile_i32 = arith.addi(cur_compute_idx, two_i32)
                    future_tile_idx = arith.index_cast(T.index, future_tile_i32)
                    prefetch_w_raw = issue_w_raw_scales_for_future_tile_rt_experimental(
                        future_tile_idx
                    )

                    # W-chunk advance: trigger when next_compute_idx crosses
                    # a 32-K-block boundary.
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        next_kb_i32 = arith.muli(
                            next_compute_idx,
                            arith.constant(scales_per_tile, type=T.i32),
                        )
                        next_chunk_i32 = arith.shrui(
                            next_kb_i32, arith.constant(5, type=T.i32)
                        )
                        need_advance = arith.cmpi(
                            arith.CmpIPredicate.ne,
                            next_chunk_i32,
                            cur_chunk_idx_i32,
                        )
                        new_bulk_w_cur = arith.select(
                            need_advance, bulk_w_prefetch, bulk_w_cur
                        )
                        target_chunk_i32 = arith.addi(next_chunk_i32, one_i32)
                        new_bulk_w_prefetch = _issue_w_chunk_runtime(target_chunk_i32)
                        bulk_w_cur = new_bulk_w_cur
                        bulk_w_prefetch = new_bulk_w_prefetch
                        cur_chunk_idx_i32 = next_chunk_i32

                    new_load_idx = arith.addi(cur_load_idx, one_i32)
                    new_state = _pack_state_experimental(
                        cur_accs,
                        cur_a,
                        cur_b,
                        cur_x_raw,
                        cur_w_raw,
                        prefetch_w_raw,
                    )
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        new_state = new_state + [
                            bulk_w_cur,
                            bulk_w_prefetch,
                            cur_chunk_idx_i32,
                        ]
                    new_state = new_state + [
                        cur_lo_x,
                        cur_lo_w,
                        cur_lo_x_scale,
                        new_load_idx,
                        next_compute_idx,
                    ]
                    results = yield new_state

                final_compute_idx = results[-1]
                lo_x_scale = results[-3]
                lo_w = results[-4]
                lo_x = results[-5]
                if const_expr(USES_W_CHUNK_PREFETCH):
                    cur_chunk_idx_i32 = results[-6]
                    bulk_w_prefetch = results[-7]
                    bulk_w_cur = results[-8]
                    _reg_results = results[:-8]
                else:
                    _reg_results = results[:-5]
                (
                    accs,
                    cur_a,
                    cur_b,
                    cur_x_raw,
                    cur_w_raw,
                    prefetch_w_raw,
                ) = _unpack_state_experimental(_reg_results)
            else:
                accs = list(accs)
                # No main loop ran — drain starts at compute_idx = 0.
                final_compute_idx = arith.constant(0, type=T.i32)

            # Drain (NB-2 iterations). compute_idx is now SSA, advance by
            # arith.addi each step. Body: WMMA on cur → wait → ds_read next.
            drain_compute_idx = final_compute_idx
            nb_const_i32_d = arith.constant(num_buffers, type=T.i32)
            one_i32_d = arith.constant(1, type=T.i32)
            two_i32_d = arith.constant(2, type=T.i32)
            for drain_i in range_constexpr(drain_iters):
                outstanding = (num_buffers - 3 - drain_i) * _TDMS_PER_TILE_EXP

                accs = compute_wmma_with_frags_experimental(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                )

                tdm_ops.tensor_wait(outstanding)
                gpu.barrier()

                next_compute_idx = arith.addi(drain_compute_idx, one_i32_d)
                next_buf_i32 = arith.remui(next_compute_idx, nb_const_i32_d)

                next_x_raw = ds_read_x_scales(next_buf_i32)
                cur_a, cur_b = load_operand_frags(next_buf_i32)
                cur_x_raw = next_x_raw

                cur_w_raw = prefetch_w_raw
                future_tile_i32 = arith.addi(drain_compute_idx, two_i32_d)
                future_tile_idx = arith.index_cast(T.index, future_tile_i32)
                prefetch_w_raw = issue_w_raw_scales_for_future_tile_rt_experimental(
                    future_tile_idx
                )

                drain_compute_idx = next_compute_idx

            # Final WMMA on the last rotated tile.
            accs = compute_wmma_with_frags_experimental(
                accs, cur_a, cur_b, cur_x_raw, cur_w_raw
            )

        elif const_expr(variant == "experimental"):
            # ───────────────────────────────────────────────────────────────
            # EXPERIMENTAL variant: Gluon pipelined ordering.
            #
            # Body order in each main-loop iter (vs reg_preload's
            # WMMA→issue→wait→ds_load):
            #
            #   issue TDM (n+nb) → wait n+1 → ds_load(next_*) → WMMA(cur_*) → swap
            #
            # ds_load writes next_* and WMMA reads cur_* — no dep between
            # them, both sit post-barrier in the same BB region. Scheduler is
            # free to interleave them, like the drain naturally does. The
            # `cur ← next` swap at the tail is just SSA renaming, no
            # instruction cost.
            #
            # Adds rocdl.iglp_opt(1) and sched_barrier hints so the compiler
            # weaves DS_READ with WMMA after the seed is in.
            # ───────────────────────────────────────────────────────────────
            accs = [acc_zero] * n_accs

            bulk_w_cur = _issue_w_chunk_const(0)
            if const_expr(USES_W_CHUNK_PREFETCH):
                bulk_w_prefetch = _issue_w_chunk_const(1)
            else:
                bulk_w_prefetch = bulk_w_cur

            # Wait for tile 0 to land, then seed cur from it.
            tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
            gpu.barrier()

            cur_a, cur_b = load_operand_frags(0)
            cur_x_raw = ds_read_x_scales(0)

            rocdl.sched_barrier(0)
            cur_w_raw = issue_w_raw_scales_for_tile_experimental(0)
            if const_expr(num_k_tiles > 1):
                prefetch_w_raw = issue_w_raw_scales_for_tile_experimental(1)
            else:
                prefetch_w_raw = zero_w_raw

            main_loop_iters_g = num_k_tiles - (num_buffers - 1)
            load_idx_init = arith.constant(num_buffers - 1, type=T.i32)
            compute_idx_init = arith.constant(0, type=T.i32)

            if const_expr(main_loop_iters_g > 0):
                init_state = _pack_state_experimental(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw, prefetch_w_raw
                )
                if const_expr(USES_W_CHUNK_PREFETCH):
                    init_state = init_state + [
                        bulk_w_cur,
                        bulk_w_prefetch,
                        cur_chunk_idx_i32,
                    ]
                init_state = init_state + [
                    lo_x,
                    lo_w,
                    lo_x_scale,
                    load_idx_init,
                    compute_idx_init,
                ]

                nb_const_i32 = arith.constant(num_buffers, type=T.i32)
                one_i32 = arith.constant(1, type=T.i32)
                two_i32 = arith.constant(2, type=T.i32)

                for tile_step, state in range(0, main_loop_iters_g, 1, init=init_state):
                    _disable_unroll_on_enclosing_loop()
                    # iglp_opt(0) at top of body: build [2 DS, 1 WMMA] groups
                    # — closer to our 52:32 DS:WMMA (~1.6:1) ratio than
                    # iglp_opt(1)'s 1:1 steady-state weave.
                    rocdl.iglp_opt(0)

                    cur_compute_idx = state[-1]
                    cur_load_idx = state[-2]
                    cur_lo_x_scale = state[-3]
                    cur_lo_w = state[-4]
                    cur_lo_x = state[-5]
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        cur_chunk_idx_i32 = state[-6]
                        bulk_w_prefetch = state[-7]
                        bulk_w_cur = state[-8]
                        _reg_state = state[:-8]
                    else:
                        _reg_state = state[:-5]
                    (
                        cur_accs,
                        cur_a,
                        cur_b,
                        cur_x_raw,
                        cur_w_raw,
                        prefetch_w_raw,
                    ) = _unpack_state_experimental(_reg_state)

                    load_buf_i32 = arith.remui(cur_load_idx, nb_const_i32)
                    next_compute_idx = arith.addi(cur_compute_idx, one_i32)
                    next_buf_i32 = arith.remui(next_compute_idx, nb_const_i32)

                    # Issue TDMs for tile load_idx (= compute_idx + nb).
                    cur_lo_x, cur_lo_w = issue_tdm_loads(
                        load_buf_i32, cur_lo_x, cur_lo_w
                    )
                    cur_lo_x_scale = issue_x_scale_tdm(load_buf_i32, cur_lo_x_scale)

                    # Wait for tile compute_idx+1 to land.
                    tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
                    gpu.barrier()

                    # ds_load tile compute_idx+1 into NEXT VGPRs (independent
                    # of cur — wmma below reads cur, scheduler can interleave).
                    next_x_raw = ds_read_x_scales(next_buf_i32)
                    next_a, next_b = load_operand_frags(next_buf_i32)

                    # WMMA on cur tile — runs concurrently with the ds_loads
                    # above (different VGPR streams, no data dep).
                    cur_accs = compute_wmma_with_frags_experimental(
                        cur_accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                    )

                    # Swap cur ← next (SSA rename, no instruction).
                    cur_a = next_a
                    cur_b = next_b
                    cur_x_raw = next_x_raw
                    cur_w_raw = prefetch_w_raw

                    future_tile_i32 = arith.addi(cur_compute_idx, two_i32)
                    future_tile_idx = arith.index_cast(T.index, future_tile_i32)
                    prefetch_w_raw = issue_w_raw_scales_for_future_tile_rt_experimental(
                        future_tile_idx
                    )

                    if const_expr(USES_W_CHUNK_PREFETCH):
                        next_kb_i32 = arith.muli(
                            next_compute_idx,
                            arith.constant(scales_per_tile, type=T.i32),
                        )
                        next_chunk_i32 = arith.shrui(
                            next_kb_i32, arith.constant(5, type=T.i32)
                        )
                        need_advance = arith.cmpi(
                            arith.CmpIPredicate.ne,
                            next_chunk_i32,
                            cur_chunk_idx_i32,
                        )
                        new_bulk_w_cur = arith.select(
                            need_advance, bulk_w_prefetch, bulk_w_cur
                        )
                        target_chunk_i32 = arith.addi(next_chunk_i32, one_i32)
                        new_bulk_w_prefetch = _issue_w_chunk_runtime(target_chunk_i32)
                        bulk_w_cur = new_bulk_w_cur
                        bulk_w_prefetch = new_bulk_w_prefetch
                        cur_chunk_idx_i32 = next_chunk_i32

                    # Trailing sched_barrier(0): pin the schedule so iglp_opt's
                    # weaving doesn't get unraveled by later passes.
                    rocdl.sched_barrier(0)

                    new_load_idx = arith.addi(cur_load_idx, one_i32)
                    new_state = _pack_state_experimental(
                        cur_accs,
                        cur_a,
                        cur_b,
                        cur_x_raw,
                        cur_w_raw,
                        prefetch_w_raw,
                    )
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        new_state = new_state + [
                            bulk_w_cur,
                            bulk_w_prefetch,
                            cur_chunk_idx_i32,
                        ]
                    new_state = new_state + [
                        cur_lo_x,
                        cur_lo_w,
                        cur_lo_x_scale,
                        new_load_idx,
                        next_compute_idx,
                    ]
                    results = yield new_state

                final_compute_idx = results[-1]
                lo_x_scale = results[-3]
                lo_w = results[-4]
                lo_x = results[-5]
                if const_expr(USES_W_CHUNK_PREFETCH):
                    cur_chunk_idx_i32 = results[-6]
                    bulk_w_prefetch = results[-7]
                    bulk_w_cur = results[-8]
                    _reg_results = results[:-8]
                else:
                    _reg_results = results[:-5]
                (
                    accs,
                    cur_a,
                    cur_b,
                    cur_x_raw,
                    cur_w_raw,
                    prefetch_w_raw,
                ) = _unpack_state_experimental(_reg_results)
            else:
                accs = list(accs)
                final_compute_idx = arith.constant(0, type=T.i32)

            # Drain (NB-2 iters). Same DS-load-then-WMMA shape as the main
            # loop body, no new TDM issues.
            drain_compute_idx = final_compute_idx
            nb_const_i32_d = arith.constant(num_buffers, type=T.i32)
            one_i32_d = arith.constant(1, type=T.i32)
            two_i32_d = arith.constant(2, type=T.i32)
            for drain_i in range_constexpr(drain_iters):
                outstanding = (num_buffers - 3 - drain_i) * _TDMS_PER_TILE_EXP

                tdm_ops.tensor_wait(outstanding)
                gpu.barrier()

                next_compute_idx = arith.addi(drain_compute_idx, one_i32_d)
                next_buf_i32 = arith.remui(next_compute_idx, nb_const_i32_d)

                next_x_raw = ds_read_x_scales(next_buf_i32)
                next_a, next_b = load_operand_frags(next_buf_i32)

                accs = compute_wmma_with_frags_experimental(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                )

                cur_a = next_a
                cur_b = next_b
                cur_x_raw = next_x_raw
                cur_w_raw = prefetch_w_raw

                future_tile_i32 = arith.addi(drain_compute_idx, two_i32_d)
                future_tile_idx = arith.index_cast(T.index, future_tile_i32)
                prefetch_w_raw = issue_w_raw_scales_for_future_tile_rt_experimental(
                    future_tile_idx
                )
                drain_compute_idx = next_compute_idx

            accs = compute_wmma_with_frags_experimental(
                accs, cur_a, cur_b, cur_x_raw, cur_w_raw
            )

        elif const_expr(variant == "experimental_unroll2"):
            # Reorder acc_coords list (wn-major traversal) — keeps idx
            # mapping as wm*wmma_n_rep + wn so output writes still work,
            # but changes the rolling pipeline's access pattern. Goal:
            # nudge the allocator's placement of accs across VGPR halves
            # to reduce s_set_vgpr_msb count. Mutate in-place so the
            # closure-captured ref in compute_wmma sees the new order.
            acc_coords[:] = [
                (wm, wn, wm * wmma_n_rep + wn)
                for wn in range(wmma_n_rep)
                for wm in range(wmma_m_rep)
            ]

            # ───────────────────────────────────────────────────────────────
            # EXPERIMENTAL_UNROLL2: same body as `experimental` but unrolled
            # by 2 with register rotation, eliminating the cross-iter v_movs.
            #
            # Pair body:
            #   ─── Half A ───
            #   issue TDM (n+nb-1) → wait n+1 → ds_load(next_*)
            #   WMMA(cur_*)                       ← cur_* dead after this
            #   ─── Half B ───
            #   issue TDM (n+nb)   → wait n+2 → ds_load(cur_*)
            #                                     ↑ rewrites cur_*'s physical
            #                                       reg (just freed by WMMA above)
            #   WMMA(next_*)                      ← next_* dead after this
            #
            # Net effect: cur_* gets the just-loaded data at end of pair body
            # naturally, no `v_mov` swap needed. Each pair iter advances
            # compute_idx by 2.
            # ───────────────────────────────────────────────────────────────
            accs = [acc_zero] * n_accs

            bulk_w_cur = _issue_w_chunk_const(0)
            if const_expr(USES_W_CHUNK_PREFETCH):
                bulk_w_prefetch = _issue_w_chunk_const(1)
            else:
                bulk_w_prefetch = bulk_w_cur

            tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
            gpu.barrier()

            cur_a, cur_b = load_operand_frags(0)
            cur_x_raw = ds_read_x_scales(0)

            rocdl.sched_barrier(0)
            cur_w_raw = issue_w_raw_scales_for_tile_experimental(0)
            if const_expr(num_k_tiles > 1):
                prefetch_w_raw = issue_w_raw_scales_for_tile_experimental(1)
            else:
                prefetch_w_raw = zero_w_raw

            main_loop_iters_g = num_k_tiles - (num_buffers - 1)
            pair_iters = main_loop_iters_g // 2
            leftover = main_loop_iters_g % 2

            load_idx_init = arith.constant(num_buffers - 1, type=T.i32)
            compute_idx_init = arith.constant(0, type=T.i32)

            if const_expr(pair_iters > 0):
                init_state = _pack_state_experimental(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw, prefetch_w_raw
                )
                if const_expr(USES_W_CHUNK_PREFETCH):
                    init_state = init_state + [
                        bulk_w_cur,
                        bulk_w_prefetch,
                        cur_chunk_idx_i32,
                    ]
                init_state = init_state + [
                    lo_x,
                    lo_w,
                    lo_x_scale,
                    load_idx_init,
                    compute_idx_init,
                ]

                nb_const_i32 = arith.constant(num_buffers, type=T.i32)
                one_i32 = arith.constant(1, type=T.i32)
                two_i32 = arith.constant(2, type=T.i32)
                three_i32 = arith.constant(3, type=T.i32)

                # Instruction counts per half (for the schedule markers below).
                # Half A and Half B each: scales_per_tile * n_accs WMMAs, with
                # 1 v_mul + 4 v_pk_fma per wmma_with_scale = 5 VALU per WMMA.
                # ds reads = X-scale (ds_b32) + operand frag b128s.
                _hot_n_wmmas_half = scales_per_tile * n_accs
                _hot_n_ds_half = (
                    scales_per_tile * wmma_m_rep
                    + k_wmma_steps * (wmma_m_rep + wmma_n_rep) * DS_LOADS_PER_FRAG
                )
                _hot_valu_per_wmma = 5
                _hot_extra_ds = max(0, _hot_n_ds_half - _hot_n_wmmas_half)
                _hot_base_ds = (
                    _hot_n_ds_half // _hot_n_wmmas_half if _hot_n_wmmas_half > 0 else 0
                )

                def hot_loop_scheduler_half():
                    """Per-half markers: [N DS, 1 WMMA, 5 VALU] × n_wmmas.
                    Fuses v_pk_fmas + v_muls into the wmma+ds_load weave so
                    the scale-FMAs don't pile up in walls after the wmma
                    cluster. Front-loads DS reads to keep dscnt full early."""
                    for w in range_constexpr(_hot_n_wmmas_half):
                        if const_expr(w < _hot_extra_ds):
                            ds_count = _hot_base_ds + 1
                        else:
                            ds_count = _hot_base_ds
                        if const_expr(ds_count > 0):
                            rocdl.sched_group_barrier(0x100, ds_count, 0)  # DS_READ
                        rocdl.sched_group_barrier(0x008, 1, 0)  # MFMA/WMMA
                        rocdl.sched_group_barrier(0x002, _hot_valu_per_wmma, 0)  # VALU

                for pair_step, state in range(0, pair_iters, 1, init=init_state):
                    _disable_unroll_on_enclosing_loop()

                    cur_compute_idx = state[-1]
                    cur_load_idx = state[-2]
                    cur_lo_x_scale = state[-3]
                    cur_lo_w = state[-4]
                    cur_lo_x = state[-5]
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        cur_chunk_idx_i32 = state[-6]
                        bulk_w_prefetch = state[-7]
                        bulk_w_cur = state[-8]
                        _reg_state = state[:-8]
                    else:
                        _reg_state = state[:-5]
                    (
                        cur_accs,
                        cur_a,
                        cur_b,
                        cur_x_raw,
                        cur_w_raw,
                        prefetch_w_raw,
                    ) = _unpack_state_experimental(_reg_state)

                    # SSA buf indices.
                    load_buf_a = arith.remui(cur_load_idx, nb_const_i32)
                    next_compute_idx_a = arith.addi(cur_compute_idx, one_i32)
                    next_buf = arith.remui(next_compute_idx_a, nb_const_i32)
                    load_idx_b = arith.addi(cur_load_idx, one_i32)
                    load_buf_b = arith.remui(load_idx_b, nb_const_i32)
                    next_compute_idx_b = arith.addi(cur_compute_idx, two_i32)
                    cur_after_buf = arith.remui(next_compute_idx_b, nb_const_i32)

                    # ─── Half A ───
                    cur_lo_x, cur_lo_w = issue_tdm_loads(load_buf_a, cur_lo_x, cur_lo_w)
                    cur_lo_x_scale = issue_x_scale_tdm(load_buf_a, cur_lo_x_scale)
                    tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
                    gpu.barrier()

                    next_x_raw = ds_read_x_scales(next_buf)
                    next_a, next_b = load_operand_frags(next_buf)

                    cur_accs = compute_wmma_with_frags_experimental(
                        cur_accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                    )

                    # Half A schedule markers: weave its DS reads + WMMAs +
                    # v_pk_fmas. Markers match instructions bottom-up to here.
                    hot_loop_scheduler_half()

                    # Inter-half fence: prevents the scheduler from blending
                    # Half A and Half B's operand frags (otherwise 3 frag
                    # sets are simultaneously live → spills).
                    rocdl.sched_barrier(0)

                    # ─── Half B ───
                    cur_lo_x, cur_lo_w = issue_tdm_loads(load_buf_b, cur_lo_x, cur_lo_w)
                    cur_lo_x_scale = issue_x_scale_tdm(load_buf_b, cur_lo_x_scale)
                    tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
                    gpu.barrier()

                    new_cur_x_raw = ds_read_x_scales(cur_after_buf)
                    new_cur_a, new_cur_b = load_operand_frags(cur_after_buf)

                    # WMMA on next_* (using prefetch_w for tile compute_idx+1).
                    cur_accs = compute_wmma_with_frags_experimental(
                        cur_accs, next_a, next_b, next_x_raw, prefetch_w_raw
                    )

                    # Half B schedule markers (same pattern as Half A).
                    hot_loop_scheduler_half()

                    # Rotate registers.
                    cur_a = new_cur_a
                    cur_b = new_cur_b
                    cur_x_raw = new_cur_x_raw

                    # W-scale: 2 new prefetches per pair iter.
                    new_w_for_compute_p2 = arith.index_cast(T.index, next_compute_idx_b)
                    cur_w_raw = issue_w_raw_scales_for_future_tile_rt_experimental(
                        new_w_for_compute_p2
                    )
                    new_w_for_compute_p3 = arith.index_cast(
                        T.index, arith.addi(cur_compute_idx, three_i32)
                    )
                    prefetch_w_raw = issue_w_raw_scales_for_future_tile_rt_experimental(
                        new_w_for_compute_p3
                    )

                    # W-chunk advance: check once per pair (advancing by 2 can
                    # cross at most one 32-block boundary).
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        next_kb_i32 = arith.muli(
                            next_compute_idx_b,
                            arith.constant(scales_per_tile, type=T.i32),
                        )
                        next_chunk_i32 = arith.shrui(
                            next_kb_i32, arith.constant(5, type=T.i32)
                        )
                        need_advance = arith.cmpi(
                            arith.CmpIPredicate.ne,
                            next_chunk_i32,
                            cur_chunk_idx_i32,
                        )
                        new_bulk_w_cur = arith.select(
                            need_advance, bulk_w_prefetch, bulk_w_cur
                        )
                        target_chunk_i32 = arith.addi(next_chunk_i32, one_i32)
                        new_bulk_w_prefetch = _issue_w_chunk_runtime(target_chunk_i32)
                        bulk_w_cur = new_bulk_w_cur
                        bulk_w_prefetch = new_bulk_w_prefetch
                        cur_chunk_idx_i32 = next_chunk_i32

                    rocdl.sched_barrier(0)

                    new_load_idx = arith.addi(cur_load_idx, two_i32)
                    new_state = _pack_state_experimental(
                        cur_accs,
                        cur_a,
                        cur_b,
                        cur_x_raw,
                        cur_w_raw,
                        prefetch_w_raw,
                    )
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        new_state = new_state + [
                            bulk_w_cur,
                            bulk_w_prefetch,
                            cur_chunk_idx_i32,
                        ]
                    new_state = new_state + [
                        cur_lo_x,
                        cur_lo_w,
                        cur_lo_x_scale,
                        new_load_idx,
                        next_compute_idx_b,
                    ]
                    results = yield new_state

                final_compute_idx = results[-1]
                final_load_idx = results[-2]
                lo_x_scale = results[-3]
                lo_w = results[-4]
                lo_x = results[-5]
                if const_expr(USES_W_CHUNK_PREFETCH):
                    cur_chunk_idx_i32 = results[-6]
                    bulk_w_prefetch = results[-7]
                    bulk_w_cur = results[-8]
                    _reg_results = results[:-8]
                else:
                    _reg_results = results[:-5]
                (
                    accs,
                    cur_a,
                    cur_b,
                    cur_x_raw,
                    cur_w_raw,
                    prefetch_w_raw,
                ) = _unpack_state_experimental(_reg_results)
            else:
                accs = list(accs)
                final_compute_idx = arith.constant(0, type=T.i32)
                final_load_idx = arith.constant(num_buffers - 1, type=T.i32)

            # Leftover: 1 single-iter step if main_loop_iters_g was odd.
            # Same shape as one experimental iter (issues TDM, swaps via
            # SSA rename — at this point we're outside the pair loop so the
            # one v_mov for the last leftover is unavoidable but cheap).
            leftover_compute_idx = final_compute_idx
            leftover_load_idx = final_load_idx
            if const_expr(leftover > 0):
                nb_const_i32_l = arith.constant(num_buffers, type=T.i32)
                one_i32_l = arith.constant(1, type=T.i32)
                two_i32_l = arith.constant(2, type=T.i32)
                load_buf_l = arith.remui(leftover_load_idx, nb_const_i32_l)
                next_compute_idx_l = arith.addi(leftover_compute_idx, one_i32_l)
                next_buf_l = arith.remui(next_compute_idx_l, nb_const_i32_l)

                lo_x, lo_w = issue_tdm_loads(load_buf_l, lo_x, lo_w)
                lo_x_scale = issue_x_scale_tdm(load_buf_l, lo_x_scale)
                tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
                gpu.barrier()

                next_x_raw = ds_read_x_scales(next_buf_l)
                next_a, next_b = load_operand_frags(next_buf_l)

                accs = compute_wmma_with_frags_experimental(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                )

                cur_a = next_a
                cur_b = next_b
                cur_x_raw = next_x_raw
                cur_w_raw = prefetch_w_raw
                future_tile_l = arith.addi(leftover_compute_idx, two_i32_l)
                future_tile_idx_l = arith.index_cast(T.index, future_tile_l)
                prefetch_w_raw = issue_w_raw_scales_for_future_tile_rt_experimental(
                    future_tile_idx_l
                )
                leftover_compute_idx = next_compute_idx_l

            # Drain (NB-2 iters) — same as experimental.
            drain_compute_idx = leftover_compute_idx
            nb_const_i32_d = arith.constant(num_buffers, type=T.i32)
            one_i32_d = arith.constant(1, type=T.i32)
            two_i32_d = arith.constant(2, type=T.i32)
            for drain_i in range_constexpr(drain_iters):
                outstanding = (num_buffers - 3 - drain_i) * _TDMS_PER_TILE_EXP

                tdm_ops.tensor_wait(outstanding)
                gpu.barrier()

                next_compute_idx = arith.addi(drain_compute_idx, one_i32_d)
                next_buf_i32 = arith.remui(next_compute_idx, nb_const_i32_d)

                next_x_raw = ds_read_x_scales(next_buf_i32)
                next_a, next_b = load_operand_frags(next_buf_i32)

                accs = compute_wmma_with_frags_experimental(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                )

                cur_a = next_a
                cur_b = next_b
                cur_x_raw = next_x_raw
                cur_w_raw = prefetch_w_raw
                future_tile_i32 = arith.addi(drain_compute_idx, two_i32_d)
                future_tile_idx = arith.index_cast(T.index, future_tile_i32)
                prefetch_w_raw = issue_w_raw_scales_for_future_tile_rt_experimental(
                    future_tile_idx
                )
                drain_compute_idx = next_compute_idx

            accs = compute_wmma_with_frags_experimental(
                accs, cur_a, cur_b, cur_x_raw, cur_w_raw
            )

        elif const_expr(variant == "manual"):
            # ───────────────────────────────────────────────────────────────
            # MANUAL variant: same TDM/X-scale/W-readlane plumbing as
            # `experimental`, but the main-loop body uses a hand-scheduled
            # compute (compute_wmma_with_frags_manual) plus a chunked
            # operand-frag ds_read (load_operand_frags_chunked).
            #
            # Two scheduling goals:
            #   1. Keep ds_load bursts ≤ ~16 dwords. The chunked load splits
            #      the per-tile A/B ds_read into two bursts (A then B) with
            #      a sched_barrier between, so the 17th ds_load doesn't
            #      stall waiting for one of the first 16 to retire.
            #   2. Group WMMAs and FMAs into separate blocks so an FMA
            #      doesn't land immediately after the WMMA that wrote its
            #      temp (avoids v_nops on gfx1250).
            #
            # Pipeline shape mirrors `experimental` (1-tile-per-iter
            # Gluon-style): wmma → issue tdm → wait → ds_read next → swap.
            # ───────────────────────────────────────────────────────────────

            bulk_w_cur = _issue_w_chunk_const(0)
            if const_expr(USES_W_CHUNK_PREFETCH):
                bulk_w_prefetch = _issue_w_chunk_const(1)
            else:
                bulk_w_prefetch = bulk_w_cur

            tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
            gpu.barrier()

            cur_a, cur_b, cur_x_raw = load_operand_frags_with_xscale_interleave(0)

            rocdl.sched_barrier(0)
            cur_w_raw = issue_w_raw_scales_for_tile_experimental(0)
            if const_expr(num_k_tiles > 1):
                prefetch_w_raw = issue_w_raw_scales_for_tile_experimental(1)
            else:
                prefetch_w_raw = zero_w_raw

            accs = [acc_zero] * n_accs

            main_loop_iters_g = num_k_tiles - (num_buffers - 1)

            load_idx_init = arith.constant(num_buffers - 1, type=T.i32)
            compute_idx_init = arith.constant(0, type=T.i32)

            if const_expr(main_loop_iters_g > 0):
                init_state = _pack_state_experimental(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw, prefetch_w_raw
                )
                if const_expr(USES_W_CHUNK_PREFETCH):
                    init_state = init_state + [
                        bulk_w_cur,
                        bulk_w_prefetch,
                        cur_chunk_idx_i32,
                    ]
                init_state = init_state + [
                    lo_x,
                    lo_w,
                    lo_x_scale,
                    load_idx_init,
                    compute_idx_init,
                ]

                nb_const_i32 = arith.constant(num_buffers, type=T.i32)
                one_i32 = arith.constant(1, type=T.i32)
                two_i32 = arith.constant(2, type=T.i32)

                for tile_step, state in range(0, main_loop_iters_g, 1, init=init_state):
                    _disable_unroll_on_enclosing_loop()
                    cur_compute_idx = state[-1]
                    cur_load_idx = state[-2]
                    cur_lo_x_scale = state[-3]
                    cur_lo_w = state[-4]
                    cur_lo_x = state[-5]
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        cur_chunk_idx_i32 = state[-6]
                        bulk_w_prefetch = state[-7]
                        bulk_w_cur = state[-8]
                        _reg_state = state[:-8]
                    else:
                        _reg_state = state[:-5]
                    (
                        cur_accs,
                        cur_a,
                        cur_b,
                        cur_x_raw,
                        cur_w_raw,
                        prefetch_w_raw,
                    ) = _unpack_state_experimental(_reg_state)

                    load_buf_i32 = arith.remui(cur_load_idx, nb_const_i32)
                    next_compute_idx = arith.addi(cur_compute_idx, one_i32)
                    next_buf_i32 = arith.remui(next_compute_idx, nb_const_i32)

                    # WMMA+FMA on cur tile (grouped per sc, sched_barrier
                    # between blocks to dodge v_nops).
                    cur_accs = compute_wmma_with_frags_manual(
                        cur_accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                    )

                    # Issue TDMs for tile load_idx.
                    cur_lo_x, cur_lo_w = issue_tdm_loads(
                        load_buf_i32, cur_lo_x, cur_lo_w
                    )
                    cur_lo_x_scale = issue_x_scale_tdm(load_buf_i32, cur_lo_x_scale)

                    # Wait for tile compute_idx+1 to land in LDS.
                    tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
                    gpu.barrier()

                    # Chunked ds_read of next tile's operand frags + x_scales.
                    # x_scale first (cheap, 1 ds_read_b32 per wm × sc), then
                    # A burst, then B burst — sched_barriers between bursts
                    # so each burst stays under ~16 ds_loads.
                    next_x_raw = ds_read_x_scales(next_buf_i32)
                    rocdl.sched_barrier(0)
                    next_a, next_b = load_operand_frags_chunked(next_buf_i32)

                    cur_a = next_a
                    cur_b = next_b
                    cur_x_raw = next_x_raw
                    cur_w_raw = prefetch_w_raw

                    # Prefetch W-scale for compute_idx+2 (zero-masked if past
                    # num_k_tiles).
                    future_tile_i32 = arith.addi(cur_compute_idx, two_i32)
                    future_tile_idx = arith.index_cast(T.index, future_tile_i32)
                    prefetch_w_raw = issue_w_raw_scales_for_future_tile_rt_experimental(
                        future_tile_idx
                    )

                    if const_expr(USES_W_CHUNK_PREFETCH):
                        next_kb_i32 = arith.muli(
                            next_compute_idx,
                            arith.constant(scales_per_tile, type=T.i32),
                        )
                        next_chunk_i32 = arith.shrui(
                            next_kb_i32, arith.constant(5, type=T.i32)
                        )
                        need_advance = arith.cmpi(
                            arith.CmpIPredicate.ne,
                            next_chunk_i32,
                            cur_chunk_idx_i32,
                        )
                        new_bulk_w_cur = arith.select(
                            need_advance, bulk_w_prefetch, bulk_w_cur
                        )
                        target_chunk_i32 = arith.addi(next_chunk_i32, one_i32)
                        new_bulk_w_prefetch = _issue_w_chunk_runtime(target_chunk_i32)
                        bulk_w_cur = new_bulk_w_cur
                        bulk_w_prefetch = new_bulk_w_prefetch
                        cur_chunk_idx_i32 = next_chunk_i32

                    new_load_idx = arith.addi(cur_load_idx, one_i32)
                    new_state = _pack_state_experimental(
                        cur_accs,
                        cur_a,
                        cur_b,
                        cur_x_raw,
                        cur_w_raw,
                        prefetch_w_raw,
                    )
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        new_state = new_state + [
                            bulk_w_cur,
                            bulk_w_prefetch,
                            cur_chunk_idx_i32,
                        ]
                    new_state = new_state + [
                        cur_lo_x,
                        cur_lo_w,
                        cur_lo_x_scale,
                        new_load_idx,
                        next_compute_idx,
                    ]
                    results = yield new_state

                final_compute_idx = results[-1]
                lo_x_scale = results[-3]
                lo_w = results[-4]
                lo_x = results[-5]
                if const_expr(USES_W_CHUNK_PREFETCH):
                    cur_chunk_idx_i32 = results[-6]
                    bulk_w_prefetch = results[-7]
                    bulk_w_cur = results[-8]
                    _reg_results = results[:-8]
                else:
                    _reg_results = results[:-5]
                (
                    accs,
                    cur_a,
                    cur_b,
                    cur_x_raw,
                    cur_w_raw,
                    prefetch_w_raw,
                ) = _unpack_state_experimental(_reg_results)
            else:
                accs = list(accs)
                final_compute_idx = arith.constant(0, type=T.i32)

            # Drain (NB-2 iters) — same shape as experimental, manual compute.
            drain_compute_idx = final_compute_idx
            nb_const_i32_d = arith.constant(num_buffers, type=T.i32)
            one_i32_d = arith.constant(1, type=T.i32)
            two_i32_d = arith.constant(2, type=T.i32)
            for drain_i in range_constexpr(drain_iters):
                outstanding = (num_buffers - 3 - drain_i) * _TDMS_PER_TILE_EXP

                accs = compute_wmma_with_frags_manual(
                    accs, cur_a, cur_b, cur_x_raw, cur_w_raw
                )

                tdm_ops.tensor_wait(outstanding)
                gpu.barrier()

                next_compute_idx = arith.addi(drain_compute_idx, one_i32_d)
                next_buf_i32 = arith.remui(next_compute_idx, nb_const_i32_d)

                next_x_raw = ds_read_x_scales(next_buf_i32)
                rocdl.sched_barrier(0)
                cur_a, cur_b = load_operand_frags_chunked(next_buf_i32)
                cur_x_raw = next_x_raw

                cur_w_raw = prefetch_w_raw
                future_tile_i32 = arith.addi(drain_compute_idx, two_i32_d)
                future_tile_idx = arith.index_cast(T.index, future_tile_i32)
                prefetch_w_raw = issue_w_raw_scales_for_future_tile_rt_experimental(
                    future_tile_idx
                )

                drain_compute_idx = next_compute_idx

            # Final WMMA on the last rotated tile.
            accs = compute_wmma_with_frags_manual(
                accs, cur_a, cur_b, cur_x_raw, cur_w_raw
            )

        else:  # variant 1 (no_op_preload), not tested a lot
            accs = [acc_zero] * n_accs

            # Initial scale preload (shares the reg_preload helpers — operand
            # frags are loaded fresh per K-tile, but scales still use the
            # lane16-strided layout).
            cur_x_raw, cur_w_raw = issue_raw_scales_for_tile(0)
            if const_expr(num_k_tiles > 1):
                prefetch_x_raw, prefetch_w_raw = issue_raw_scales_for_tile(1)
            else:
                prefetch_x_raw, prefetch_w_raw = zero_x_raw, zero_w_raw

            main_loop_end_k = main_loop_iters * num_buffers * tile_k
            if const_expr(main_loop_iters > 0):
                init_state = _pack_state_no_op_preload(
                    accs, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw
                ) + [lo_x, lo_w]
                for iter_k_base, state in range(
                    0, main_loop_end_k, num_buffers * tile_k, init=init_state
                ):
                    cur_lo_x = state[-2]
                    cur_lo_w = state[-1]
                    cur_accs, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw = (
                        _unpack_state_no_op_preload(state[:-2])
                    )
                    tile_idx_rt = iter_k_base / arith.index(tile_k)

                    for substage in range_constexpr(num_buffers):
                        load_buffer = (substage + num_buffers - 1) % num_buffers
                        cur_lo_x, cur_lo_w = issue_tdm_loads(
                            load_buffer, cur_lo_x, cur_lo_w
                        )

                        compute_stage = substage % num_buffers
                        fresh_a, fresh_b = load_operand_frags(compute_stage)

                        cur_accs = compute_wmma_with_frags(
                            cur_accs, fresh_a, fresh_b, cur_x_raw, cur_w_raw
                        )

                        tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING)
                        gpu.barrier()

                        cur_x_raw = prefetch_x_raw
                        cur_w_raw = prefetch_w_raw

                        future_tile_rt = tile_idx_rt + arith.index(substage + 2)
                        prefetch_x_raw, prefetch_w_raw = (
                            issue_raw_scales_for_future_tile_rt(future_tile_rt)
                        )

                    results = yield _pack_state_no_op_preload(
                        cur_accs, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw
                    ) + [cur_lo_x, cur_lo_w]
                accs, cur_x_raw, cur_w_raw, prefetch_x_raw, prefetch_w_raw = (
                    _unpack_state_no_op_preload(results[:-2])
                )
                lo_x = results[-2]
                lo_w = results[-1]
            else:
                accs = list(accs)

            # Extra tiles: if main loop iterations doesnt cleanly divide in the
            # const_expr loop then we need this for the final buffers
            extra_base_tile = main_loop_iters * num_buffers
            for step in range_constexpr(extra_tiles):
                load_tile = extra_base_tile + step + num_buffers - 1
                load_buffer = load_tile % num_buffers
                lo_x, lo_w = issue_tdm_loads(load_buffer, lo_x, lo_w)

                compute_stage = (extra_base_tile + step) % num_buffers
                fresh_a, fresh_b = load_operand_frags(compute_stage)

                accs = compute_wmma_with_frags(
                    accs, fresh_a, fresh_b, cur_x_raw, cur_w_raw
                )

                tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING)
                gpu.barrier()

                cur_x_raw = prefetch_x_raw
                cur_w_raw = prefetch_w_raw

                future_tile = extra_base_tile + step + 2
                if const_expr(future_tile < num_k_tiles):
                    prefetch_x_raw, prefetch_w_raw = issue_raw_scales_for_tile(
                        future_tile
                    )
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

                accs = compute_wmma_with_frags(
                    accs, fresh_a, fresh_b, cur_x_raw, cur_w_raw
                )

                cur_x_raw = prefetch_x_raw
                cur_w_raw = prefetch_w_raw

                future_tile = drain_base_tile + drain_i + 2
                if const_expr(future_tile < num_k_tiles):
                    prefetch_x_raw, prefetch_w_raw = issue_raw_scales_for_tile(
                        future_tile
                    )
                else:
                    prefetch_x_raw, prefetch_w_raw = zero_x_raw, zero_w_raw

            # Final wmma
            final_tile = drain_base_tile + drain_iters
            final_compute_stage = final_tile % num_buffers
            fresh_a, fresh_b = load_operand_frags(final_compute_stage)
            accs = compute_wmma_with_frags(accs, fresh_a, fresh_b, cur_x_raw, cur_w_raw)

        # Step 4: convert f32 accs to out_dtype, buffer_store to Y.
        if const_expr(num_buffers > 2):
            rocdl.sched_barrier(0)

        out_elem = (
            T.bf16 if out_dtype == "bf16" else T.f16 if out_dtype == "fp16" else None
        )
        is_half_out = out_dtype in ("bf16", "fp16")

        if use_tdm_store:
            d_lds_elem_ty = T.bf16 if out_dtype != "fp16" else T.f16
            d_lds_elems = total_d_bytes // 2
            d_smem = SmemPtr(
                d_lds_allocator.get_base(), 0, d_lds_elem_ty, shape=(d_lds_elems,)
            )
            d_lds_buffer = d_smem.get()

            row_lds = warp_m_base + lane16  # warp_m_base = wave_m_idx * warp_tile_m
            col_lds = warp_n_base + lane_kgrp * arith.index(8)  # bf16 col within row
            d_lane_base = row_lds * arith.index(_lds_d_stride_elems_d) + col_lds
            if not is_half_out:
                d_lane_base = (
                    row_lds * arith.index(_lds_d_stride_elems_d)
                    + warp_n_base * arith.index(elem_bytes_d // 2)
                    + lane_kgrp * arith.index(4 * elem_bytes_d)
                )

            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    imm = wm * WMMA_M * _lds_d_stride_elems_d + wn * _n_col_d_elems_d
                    store_acc_vec8_to_lds(
                        d_lds_buffer,
                        d_lane_base,
                        imm,
                        accs[idx],
                        out_elem=out_elem,
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
                    col_base = (
                        blk_n
                        + warp_n_base
                        + arith.index(wn * WMMA_N)
                        + lane_kgrp * arith.index(8)
                    )

                    if is_half_out:
                        c_off_bytes = (row * n_stride + col_base) * arith.index(
                            elem_bytes_d
                        )
                        store_acc_vec8_to_buffer(
                            accs[idx],
                            y_buf,
                            c_off_bytes,
                            out_elem=out_elem,
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
        N,
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
        variant,
        use_tdm_store,
        loop_carried_load_percent,
        kernarg_preload,
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

        flat_wg_attr = ir.StringAttr.get(f"{block_threads},{block_threads}")
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        # Experimental, loop_carried_load_percent
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

        # Mark kernel args as inreg so AMDGPU preloads them into user SGPRs at dispatch.
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
    loop_carried_load_percent: Optional[int] = None,
    kernarg_preload: bool = False,
):
    """Compute Y = (X @ W^T) with per-block f32 scales (A8W8 blockscale).

    variant: "reg_preload" (default) or "no_op_preload".
      - "reg_preload"   : loop-carry cur_a/cur_b operand frags across iters.
                          W-scales bulk-loaded via buffer_load_b32 + readlane,
                          X-scales TDM-staged into LDS. Requires
                          w_is_wave_uniform.
      - "no_op_preload" : operand frags loaded fresh from LDS each iter
                          (~256 VGPRs cheaper). Uses the legacy per-K-block
                          buffer_load scale path.
    """
    assert x.ndim == 2 and w.ndim == 2, "X and W must be 2D"
    M, K = x.shape
    if getattr(w, "is_shuffled", False):
        N = w.shape[0] * 16
        K_w = w.shape[1] // 16
    else:
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

    K_padded = ((K + tile_k - 1) // tile_k) * tile_k
    if K_padded != K:
        assert not getattr(w, "is_shuffled", False), (
            "K padding is not supported for preshuffled W; pass K that is a "
            "multiple of tile_k or shuffle the padded W."
        )
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
        loop_carried_load_percent=loop_carried_load_percent,
        kernarg_preload=kernarg_preload,
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
