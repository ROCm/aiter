# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A8W8 FP8 blockscale GEMM for gfx1250.

Computes Y = X @ W^T with per-K-block f32 scales.
Supports the `compute_bound` variant and optional TDM-store output.

Variant:
  - compute_bound : default. Operand frags loop-carried across K-tiles via
                  `_pack_state_experimental` / `_unpack_state_experimental`.
                    * W-scales: bulk-load K-tiles' W-scales into VGPRs
                      (each buffer_load_b32 covers up to 32 K-blocks).
                      scale_k <= 32 → one bulk load at kernel entry +
                      per-K-tile v_readlane. scale_k > 32 → a cur/prefetch
                      chunk chain in the loop carry. Requires
                      w_is_wave_uniform.
                    * X-scales: TDM-staged into LDS (num_buffers stages,
                      aligned with X+W tile stages), then ds_read_b32 into
                      VGPRs in lane16 layout.
"""

from types import SimpleNamespace
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import math as math_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    const_expr,
    gpu,
    idx2crd,
    range_constexpr,
    rocdl,
    tdm_ops,
)
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import check_smem_capacity


def _i32_const(v):
    return _raw(arith.constant(int(v), type=T.i32))


def _to_i32_offset(offset):
    if isinstance(offset, int):
        return _i32_const(offset)
    offset = arith.unwrap(offset)
    if not isinstance(offset.type, ir.IntegerType) or offset.type.width != 32:
        return _raw(arith.index_cast(T.i32, offset))
    return offset


def _buffer_load(rsrc, offset, vec_width=1, dtype=None):
    if dtype is None:
        dtype = T.f32
    off = _to_i32_offset(offset)
    off = _raw(arith.muli(off, _i32_const(dtype.width // 8)))
    result_type = dtype if vec_width == 1 else T.vec(vec_width, dtype)
    return rocdl.RawPtrBufferLoadOp(
        result_type, rsrc, off, _i32_const(0), _i32_const(0)
    ).result


def _buffer_store(data, rsrc, offset, offset_is_bytes=False):
    data = arith.unwrap(data)
    off = _to_i32_offset(offset)
    if not offset_is_bytes:
        data_type = data.type
        element_type = (
            data_type.element_type if hasattr(data_type, "element_type") else data_type
        )
        off = _raw(arith.muli(off, _i32_const(element_type.width // 8)))
    rocdl.RawPtrBufferStoreOp(data, rsrc, off, _i32_const(0), _i32_const(0))


def _make_buffer_rsrc(tensor, num_records_bytes):
    buf_tensor = fx.rocdl.make_buffer_tensor(
        tensor, max_size=True, num_records_bytes=num_records_bytes
    )
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(buf_tensor))


buffer_ops = SimpleNamespace(
    buffer_load=_buffer_load,
    buffer_store=_buffer_store,
    create_buffer_resource=_make_buffer_rsrc,
)

WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
FRAG_VGPRS = 16
DS_LOADS_PER_FRAG = 4
LDS_PAD_A_BYTES = 16
LDS_PAD_B_BYTES = 16
_STAGE_NAMES = ("ping", "pong", "pang", "pung")


def _align_up(n, a):
    return ((n + a - 1) // a) * a


def lds_load_b128(lds_i8_ptr, byte_off):
    return arith.unwrap(
        fx.ptr_load(
            fx.add_offset(lds_i8_ptr, byte_off),
            result_type=ir.VectorType.get([4], ir.IntegerType.get_signless(32)),
        )
    )


def lds_load_f32(lds_f32_ptr, elem_off):
    return arith.unwrap(fx.ptr_load(fx.add_offset(lds_f32_ptr, elem_off)))


def lds_store_b128(lds_i8_ptr, byte_off, data):
    fx.ptr_store(fx.Vector(arith.unwrap(data)), fx.add_offset(lds_i8_ptr, byte_off))


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


def store_acc_vec8_to_lds(
    lds_i8_ptr, base_elem_off, imm_elem_off, acc_vec8, out_elem=None
):
    off = (base_elem_off + arith.index(imm_elem_off)) * arith.index(2)
    if out_elem is not None:
        h_vec = arith.trunc_f(T.vec(8, out_elem), acc_vec8)
        i32_vec = fx.Vector(h_vec).bitcast(fx.Int32).ir_value()
        lds_store_b128(lds_i8_ptr, off, i32_vec)
    else:
        for half in range(2):
            vals = [fx.Vector(acc_vec8)[half * 4 + vi] for vi in range(4)]
            vec4 = fx.Vector.from_elements(vals, fx.Float32).ir_value()
            lds_store_b128(lds_i8_ptr, off + arith.index(half * 16), vec4)


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
        i32_vec = fx.Vector(h_vec).bitcast(fx.Int32).ir_value()
        buffer_ops.buffer_store(i32_vec, c_rsrc, addr, offset_is_bytes=offset_is_bytes)
        return 1
    for half in range(2):
        vals = [fx.Vector(acc_vec8)[half * 4 + vi] for vi in range(4)]
        vec4 = fx.Vector.from_elements(vals, fx.Float32).ir_value()
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
    waves_per_eu: int | None = None,
    l2_prefetch_distance: int = 0,
    out_dtype: str = "bf16",
    variant: str = "compute_bound",
    N: int = 0,
    use_tdm_store: bool = False,
    loop_carried_load_percent: int | None = None,
    kernarg_preload: bool = False,
    split_k: int = 1,
):
    if variant not in ("compute_bound", "memory_bound"):
        raise ValueError(
            f"variant must be 'compute_bound' or 'memory_bound', got {variant!r}"
        )
    if const_expr(variant in ("compute_bound", "memory_bound")):
        _w_is_wave_uniform = (tile_n // n_warp) <= scale_block_n
        if not _w_is_wave_uniform:
            raise ValueError(
                f"variant={variant!r} requires warp_tile_n ({tile_n // n_warp}) "
                f"<= scale_block_n ({scale_block_n}) (W-scale must be wave-uniform)"
            )
        # scale_k > 32 -> multi-chunk prefetch chain (chunks 0+1 at entry, advanced per iter).
    if out_dtype not in ("bf16", "fp16", "f32"):
        raise ValueError(
            f"out_dtype must be 'bf16', 'fp16', or 'f32', got {out_dtype!r}"
        )
    if not (2 <= num_buffers <= 8):
        raise ValueError(f"num_buffers must be between 2 and 8, got {num_buffers}")
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
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    if split_k > 1:
        if variant != "memory_bound":
            raise ValueError(
                f"split_k > 1 is only supported for variant='memory_bound', got {variant!r}"
            )
        if use_tdm_store:
            raise ValueError("split_k > 1 is incompatible with use_tdm_store")
        if out_dtype != "f32":
            raise ValueError(
                "split_k > 1 requires out_dtype='f32' (partials accumulate via f32 "
                "atomic-fadd; the gemm_a8w8_blockscale wrapper sets this automatically)"
            )
        if K % split_k != 0:
            raise ValueError(f"K ({K}) must be divisible by split_k ({split_k})")
        if (K // split_k) % tile_k != 0:
            raise ValueError(
                f"K/split_k ({K // split_k}) must be a multiple of tile_k ({tile_k})"
            )
        if (K // split_k) % scale_block_k != 0:
            raise ValueError(
                f"K/split_k ({K // split_k}) must be a multiple of scale_block_k ({scale_block_k})"
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
    acc_coords = [
        (wm, wn, wm * wmma_n_rep + wn)
        for wm in range(wmma_m_rep)
        for wn in range(wmma_n_rep)
    ]

    split_k_chunk = K // split_k
    num_k_tiles = split_k_chunk // tile_k
    scale_k = K // scale_block_k
    scale_k_per_split = split_k_chunk // scale_block_k

    # W-scale chunking: 1 buffer_load_b32 covers 32 K-blocks; lazy chain when scale_k > 32 (both variants bulk-load).
    _USES_REG_W = True
    NUM_W_CHUNKS = (scale_k_per_split + 31) // 32
    USES_W_CHUNK_PREFETCH = NUM_W_CHUNKS > 1

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
    # B preshuffled (cycle-major): each stripe (16 N) = tile_k cycles x 16 bytes = tile_k*16 contiguous, no padding.
    lds_b_stride_bytes = tile_k * 16  # per-stripe size in LDS bytes
    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = (tile_n // 16) * lds_b_stride_bytes

    # X-scale LDS (TDM-staged): tile_m rows × scales_per_tile × 4B per stage.
    USES_X_SCALE_TDM = True
    lds_x_scale_row_bytes = scales_per_tile * 4
    lds_x_scale_data_bytes = tile_m * lds_x_scale_row_bytes

    # Unified LDS arena: contiguous [A0..A_nb-1 | B0..B_nb-1 | X0..X_nb-1]; slot i = region_base + i*slot_stride_bytes (SSA buf_idx).
    unified_a_off = 0
    unified_b_off = _align_up(unified_a_off + num_buffers * lds_a_data_bytes, 16)
    if USES_X_SCALE_TDM:
        unified_x_scale_off = _align_up(
            unified_b_off + num_buffers * lds_b_data_bytes, 16
        )
        lds_staging_bytes = _align_up(
            unified_x_scale_off + num_buffers * lds_x_scale_data_bytes, 16
        )
    else:
        unified_x_scale_off = 0
        lds_staging_bytes = _align_up(
            unified_b_off + num_buffers * lds_b_data_bytes, 16
        )

    if use_tdm_store:
        lds_d_row_stride_bytes = tile_n * elem_bytes_d
        total_d_bytes = tile_m * lds_d_row_stride_bytes
        _lds_d_stride_elems_d = lds_d_row_stride_bytes // 2
        _n_col_d_elems_d = WMMA_N * elem_bytes_d // 2
        unified_d_off = _align_up(lds_staging_bytes, 16)
        lds_arena_bytes = unified_d_off + total_d_bytes
    else:
        unified_d_off = 0
        lds_arena_bytes = lds_staging_bytes
    check_smem_capacity(lds_arena_bytes, gpu_arch)

    # 3 TDMs per tile (X + W + X-scale) for both surviving variants.
    _TDMS_PER_TILE_EXP = 3
    # EXPERIMENT: main loop issues the TDM BEFORE the wait (fuller memory pipe: keeps
    # NB-1 tiles in flight across the barrier instead of dipping to NB-2). The wait now
    # sees the just-issued tile -> target is (NB-1)*3, matching the prologue. This
    # re-exposes the WAR (the TDM refill precedes the barrier) -- see the main-loop note.
    if variant == "memory_bound":
        # MEMORY-BOUND (memory_bound): prologue fills NB-1 buffers (not NB), so the waits
        # are one tile shorter than compute_bound -- prologue retires tile-0 leaving
        # NB-1 pending, and the main-loop wait sits ABOVE its own refill.
        MAIN_TDM_OUTSTANDING_EXPERIMENTAL = max(0, num_buffers - 3) * _TDMS_PER_TILE_EXP
        REG_PROLOGUE_WAIT = (num_buffers - 2) * _TDMS_PER_TILE_EXP
    else:
        MAIN_TDM_OUTSTANDING_EXPERIMENTAL = (num_buffers - 1) * _TDMS_PER_TILE_EXP
        REG_PROLOGUE_WAIT = (num_buffers - 1) * _TDMS_PER_TILE_EXP

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
        if const_expr(split_k > 1):
            bz = gpu.block_id("z")
            split_k_base = bz * arith.index(split_k_chunk)
            split_kb_base = bz * arith.index(scale_k_per_split)
        else:
            split_k_base = arith.index(0)
            split_kb_base = arith.index(0)

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
        buffer_ops.create_buffer_resource(
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

        lds_base_ptr = fx.SharedAllocator(static=True).allocate(lds_arena_bytes)._ptr

        def _lds_region_ptr(byte_off):
            return fx.add_offset(lds_base_ptr, byte_off)

        big_a_mem = _lds_region_ptr(unified_a_off)
        big_b_mem = _lds_region_ptr(unified_b_off)
        slot_stride_a_i32 = arith.constant(lds_a_data_bytes, type=T.i32)
        slot_stride_b_i32 = arith.constant(lds_b_data_bytes, type=T.i32)

        def _slot_byte_off(buf_idx, region_off, slot_bytes):
            if const_expr(isinstance(buf_idx, int)):
                return region_off + buf_idx * slot_bytes
            return arith.index_cast(T.index, buf_idx) * arith.index(
                slot_bytes
            ) + arith.index(region_off)

        def _imm64(k_tile, mul):
            if const_expr(isinstance(k_tile, int)):
                return fx.Int64(k_tile * mul)
            as_index = arith.index_cast(
                T.index, arith.muli(k_tile, arith.constant(mul, type=T.i32))
            )
            return fx.Int64(arith.index_cast(T.i64, as_index))

        def _byte_view(tensor, byte_off, shape, stride):
            return fx.Tensor(
                fx.make_view(
                    fx.add_offset(
                        fx.recast_iter(fx.Int8, fx.get_iter(tensor)), byte_off
                    ),
                    fx.make_layout(shape, stride),
                )
            )

        blk_m64 = fx.Int64(blk_m)
        blk_n64 = fx.Int64(blk_n)
        split_k_base64 = fx.Int64(split_k_base)
        x_base_bytes = blk_m64 * fx.Int64(K) + split_k_base64
        w_base_bytes = (blk_n64 // fx.Int64(16)) * fx.Int64(
            K * 16
        ) + split_k_base64 * fx.Int64(16)

        gX = _byte_view(arg_x, x_base_bytes, (tile_m, tile_k), (tile_k, 1))
        gW = _byte_view(
            arg_w, w_base_bytes, (tile_n // 16, tile_k * 16), (tile_k * 16, 1)
        )
        atom_x = fx.rocdl.make_tdm_atom(
            gX,
            [None, None],
            strides=[fx.Int64(K), None],
            num_warps=num_warps,
            pad_interval=tile_k,
            pad_amount=LDS_PAD_A_BYTES,
        )
        atom_w = fx.rocdl.make_tdm_atom(
            gW,
            [None, None],
            strides=[fx.Int64(K * 16), None],
            num_warps=num_warps,
        )

        def issue_tdm_loads(buf_idx, k_tile):
            fx.copy(
                atom_x,
                gX,
                fx.Tensor(
                    fx.make_view(
                        _lds_region_ptr(
                            _slot_byte_off(buf_idx, unified_a_off, lds_a_data_bytes)
                        ),
                        fx.make_layout((tile_m, tile_k), (lds_a_stride_bytes, 1)),
                    )
                ),
                imm_offset=_imm64(k_tile, tile_k),
            )
            fx.copy(
                atom_w,
                gW,
                fx.Tensor(
                    fx.make_view(
                        _lds_region_ptr(
                            _slot_byte_off(buf_idx, unified_b_off, lds_b_data_bytes)
                        ),
                        fx.make_layout(
                            (tile_n // 16, tile_k * 16), (lds_b_stride_bytes, 1)
                        ),
                    )
                ),
                imm_offset=_imm64(k_tile, tile_k * 16),
            )

        # ── X-scale TDM descriptor + LDS staging (hoisted) ──────────────────
        if const_expr(USES_X_SCALE_TDM):
            big_x_scale_mem = fx.recast_iter(
                fx.Float32, _lds_region_ptr(unified_x_scale_off)
            )

            xs_row_bytes = scales_per_tile * 4
            xs_base_bytes = blk_m64 * fx.Int64(scale_k * 4) + fx.Int64(
                split_kb_base
            ) * fx.Int64(4)
            gXS = _byte_view(
                arg_x_scale, xs_base_bytes, (tile_m, xs_row_bytes), (xs_row_bytes, 1)
            )
            atom_xs = fx.rocdl.make_tdm_atom(
                gXS,
                [None, None],
                strides=[fx.Int64(scale_k * 4), None],
                num_warps=num_warps,
            )

            def issue_x_scale_tdm(buf_idx, k_tile):
                fx.copy(
                    atom_xs,
                    gXS,
                    fx.Tensor(
                        fx.make_view(
                            _lds_region_ptr(
                                _slot_byte_off(
                                    buf_idx,
                                    unified_x_scale_off,
                                    lds_x_scale_data_bytes,
                                )
                            ),
                            fx.make_layout((tile_m, xs_row_bytes), (xs_row_bytes, 1)),
                        )
                    ),
                    imm_offset=_imm64(k_tile, xs_row_bytes),
                )

            def ds_read_x_scales(buf_idx):
                """Read this warp's x_scales for one K-tile from LDS stage
                `buf_idx`. Offsets are in f32 elements so consecutive reads are
                visibly adjacent and can pair into ds_load_2addr_b32."""
                slot_elems = lds_x_scale_data_bytes // 4
                if const_expr(isinstance(buf_idx, int)):
                    slot_off = arith.index(buf_idx * slot_elems)
                else:
                    slot_off = arith.index_cast(
                        T.index,
                        arith.muli(buf_idx, arith.constant(slot_elems, type=T.i32)),
                    )
                out = []
                for sc in range_constexpr(scales_per_tile):
                    for wm in range_constexpr(wmma_m_rep):
                        row = warp_m_base + arith.index(wm * WMMA_M) + lane16
                        off = (
                            slot_off
                            + row * arith.index(scales_per_tile)
                            + arith.index(sc)
                        )
                        out.append(lds_load_f32(big_x_scale_mem, off))
                return out

        w_is_wave_uniform = warp_tile_n <= scale_block_n
        # Hoisted out of the w_is_wave_uniform const_expr so AST-rewriter closures downstream can resolve it.
        wave_n_block = (blk_n + warp_n_base) / arith.index(scale_block_n)

        # Bulk W-scale load: 1 buffer_load_b32 covers 32 K-blocks; chunk-prefetch chain when scale_k > 32.
        if const_expr(_USES_REG_W):
            lane_id_full = lane_kgrp * arith.index(16) + lane16

            def _issue_w_chunk_const(chunk_i):
                """Issue one bulk W-scale load for compile-time chunk_i."""
                offset = arith.index(chunk_i * 32)
                idx = wave_n_block * arith.index(scale_k) + lane_id_full + offset
                if const_expr(split_k > 1):
                    idx = idx + split_kb_base
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
                if const_expr(split_k > 1):
                    idx = idx + split_kb_base
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

        # W-scale issue: chunk-cached readlane (wave-uniform) or per-(wn) buffer_load.

        # lane_kgrp selects K-half: kgrp=0 → bytes [0..63], kgrp=1 → [64..127].
        k_half_byte_offset = lane_kgrp * arith.index(64)

        # Preshuffled B (cycle-major, natural K per shuffle_weight_gfx1250): kgrp selects a contiguous 4-cycle K-half (kgrp0=cycles 0-3, kgrp1=4-7); cycle=256B, K-half=1024B.
        b_k_half_byte_offset = lane_kgrp * arith.index(1024)

        def _compute_lane_bases(warp_base, stride_bytes, num_reps, rep_stride_elems):
            """Compute per-lane LDS byte offsets for loading `num_reps` WMMA
            frags along M or N. Returns a list of base offsets indexed by rep.
            (Used for A — M-major LDS layout.)"""
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
            """Compute per-lane LDS byte offsets for B in cycle-major
            preshuffled layout. Each rep = one WMMA tile along N, which
            corresponds to one 16-N stripe in LDS (size lds_b_stride_bytes
            = tile_k * 16). Within a stripe, lane reads at:
                stripe_offset + b_k_half_byte_offset + lane16 * 16
            where lane16 * 16 = byte offset within the cycle (16 bytes per
            N value per cycle).
            """
            bases = []
            for rep in range_constexpr(num_reps):
                # warp_base is the starting N-element index; each rep advances by WMMA_N=16
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
            per lane, starting at byte offset (lane_base + ks * ks_stride_bytes).

            cycle_stride_bytes:
              - 16  for M-major A (default; consecutive b128s are adjacent)
              - 256 for cycle-major preshuffled B (consecutive b128s are one
                    cycle apart; natural-K shuffle_weight_gfx1250 layout)
            ks_stride_bytes:
              - WMMA_K (=128) for M-major A
              - 2048 for cycle-major preshuffled B (one WMMA = 8 cycles = 2048 B)
            """
            k_sub_off = arith.index(ks * ks_stride_bytes)
            off = lane_base + k_sub_off
            v0 = lds_load_b128(lds_memref, off)
            v1 = lds_load_b128(lds_memref, off + arith.index(cycle_stride_bytes))
            v2 = lds_load_b128(lds_memref, off + arith.index(cycle_stride_bytes * 2))
            v3 = lds_load_b128(lds_memref, off + arith.index(cycle_stride_bytes * 3))
            v01 = fx.Vector(v0).shuffle(fx.Vector(v1), list(range(8)))
            v23 = fx.Vector(v2).shuffle(fx.Vector(v3), list(range(8)))
            return v01.shuffle(v23, list(range(16))).ir_value()

        a_lane_bases = _compute_lane_bases(
            warp_m_base, lds_a_stride_bytes, wmma_m_rep, WMMA_M
        )
        b_lane_bases = _compute_b_lane_bases_preshuffled(warp_n_base, wmma_n_rep)

        # HELPERS: WMMA compute + scale FMA

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fx.Float8E4M3FN, fx.Float32)
        )

        def _rmem_vec(v, n, dtype):
            t = fx.make_rmem_tensor(n, dtype)
            t.store(fx.Vector(arith.unwrap(v)))
            return t

        def _rmem_frag_lists(a_frags, b_frags):
            return (
                [_rmem_vec(f, 16, fx.Int32) for f in a_frags],
                [_rmem_vec(f, 16, fx.Int32) for f in b_frags],
            )

        # Split-out primitive helpers (also callable directly from the main loop to hand-order WMMA/scale/FMA for minimal s_wait_dscnt).

        def issue_wmma_step(sc, wm, wn, a_frags, b_frags):
            """Issue WMMA(s) for scale-block `sc`, M-rep `wm`, N-rep `wn`.

            For each ks_inner in [0, wmma_steps_per_scale), accumulates one
            16×16×128 fp8 WMMA partial product into `temp` (seed = acc_zero,
            so iter 0 is effectively WMMA(B, A, 0)). For our shape
            wmma_steps_per_scale == 1 so this issues exactly one WMMA.

            Returns the partial-sum vec<8 f32> (one per lane), to be scaled
            and folded into a global accumulator by `apply_scale`.
            """
            temp_t = _rmem_vec(acc_zero, 8, fx.Float32)
            for ks_inner in range_constexpr(wmma_steps_per_scale):
                ks = sc * wmma_steps_per_scale + ks_inner
                a_t = a_frags[ks * wmma_m_rep + wm]
                b_t = b_frags[ks * wmma_n_rep + wn]
                # ISA operand order: (B, A, C), reversed from math.
                fx.gemm(wmma_atom, temp_t, b_t, a_t, temp_t)
            return arith.unwrap(temp_t.load())

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
            scale_vec = fx.Vector.filled(8, fx.Float32(scale), fx.Float32).ir_value()
            return math_dialect.fma(temp, scale_vec, acc)

        # State layout: N_ACCS accs | N_A_FRAGS cur_a | N_B_FRAGS cur_b | N_CUR_X_RAW | N_CUR_W_RAW | N_PREFETCH_X | N_PREFETCH_W (frags compute_bound only).
        N_ACCS = n_accs
        N_A_FRAGS = wmma_m_rep * k_wmma_steps
        N_B_FRAGS = wmma_n_rep * k_wmma_steps
        # x_raw carry: wmma_m_rep entries per sc (lane16-strided layout).
        N_CUR_X_RAW = scales_per_tile * wmma_m_rep
        N_CUR_W_RAW = scales_per_tile * wmma_n_rep
        N_PREFETCH_W = N_CUR_W_RAW
        zero_w_raw = [scale_zero] * N_CUR_W_RAW

        # Prologue + accs init live inside each variant branch.
        if const_expr(variant == "compute_bound"):
            # ───── compute_bound-only helpers ─────
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
                                cycle_stride_bytes=256,
                                ks_stride_bytes=2048,
                            )
                        )
                    # X-scale ds_read after K-step 0's frag loads (unpinned).
                    if const_expr(ks == 0):
                        x_raw = ds_read_x_scales(buffer_idx)
                return a_frags, b_frags, x_raw

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
                                cycle_stride_bytes=256,
                                ks_stride_bytes=2048,
                            )
                        )
                return a_frags, b_frags

            def load_a_frags(buffer_idx):
                # A-only load — exact split of load_operand_frags's A half.
                if const_expr(isinstance(buffer_idx, int)):
                    slot_off_a = arith.index(buffer_idx * lds_a_data_bytes)
                else:
                    slot_off_a = arith.index_cast(
                        T.index, arith.muli(buffer_idx, slot_stride_a_i32)
                    )
                a_frags = []
                for ks in range_constexpr(k_wmma_steps):
                    for wm in range_constexpr(wmma_m_rep):
                        a_frags.append(
                            _load_frag(big_a_mem, a_lane_bases[wm] + slot_off_a, ks)
                        )
                return a_frags

            def load_b_frags(buffer_idx):
                # B-only load, split out so the main loop can prefetch B into its own vgprs (double-buffered) at the n+1 tensor_wait.
                if const_expr(isinstance(buffer_idx, int)):
                    slot_off_b = arith.index(buffer_idx * lds_b_data_bytes)
                else:
                    slot_off_b = arith.index_cast(
                        T.index, arith.muli(buffer_idx, slot_stride_b_i32)
                    )
                b_frags = []
                for ks in range_constexpr(k_wmma_steps):
                    for wn in range_constexpr(wmma_n_rep):
                        b_frags.append(
                            _load_frag(
                                big_b_mem,
                                b_lane_bases[wn] + slot_off_b,
                                ks,
                                cycle_stride_bytes=256,
                                ks_stride_bytes=2048,
                            )
                        )
                return b_frags

            def compute_wmma_with_frags_experimental(
                global_accs, a_frags, b_frags, x_raw, w_raw
            ):
                """Flat WMMA/FMA (fold-lag removed): issue each WMMA then fold immediately.

                Non-transposed WMMA: ISA operand order (B, A, C). Same WMMA
                output layout as compute_bound — each lane's vec<8> shares one row,
                so the per-row x_scale broadcasts as a scalar at FMA time.

                Pattern per scale block matches the compute_bound version. Kept as
                a separate function so the experimental path can diverge from
                compute_bound independently (e.g., scale-apply rewrites or
                instruction scheduling experiments).

                Now built on top of the split-out helpers (issue_wmma_step,
                compute_scale_step, apply_scale) so the main loop can call those
                directly when it needs hand-tuned ordering.
                """

                a_frag_ts, b_frag_ts = _rmem_frag_lists(a_frags, b_frags)

                def issue_wmma_temp(sc, wm, wn):
                    return issue_wmma_step(sc, wm, wn, a_frag_ts, b_frag_ts)

                def compute_scale(wm, wn, sc_x_base, sc_w_base):
                    # Local shim for the rolling-pipeline body below: takes pre-computed sc_x_base/sc_w_base (compute_scale_step takes sc).
                    return arith.mulf(x_raw[sc_x_base + wm], w_raw[sc_w_base + wn])

                def wmma_with_scale(temp, wm, wn, idx, sc_x_base, sc_w_base):
                    scale = compute_scale(wm, wn, sc_x_base, sc_w_base)
                    global_accs[idx] = apply_scale(temp, scale, global_accs[idx])

                for sc in range_constexpr(scales_per_tile):
                    sc_x_base = sc * wmma_m_rep
                    sc_w_base = sc * wmma_n_rep

                    # FOLD-LAG REMOVED: issue+fold each WMMA immediately in acc_coords order (no 2-deep pipeline); independent chains let coexec overlap; bit-identical to pipelined.
                    for wm, wn, idx in acc_coords:
                        temp = issue_wmma_temp(sc, wm, wn)
                        wmma_with_scale(temp, wm, wn, idx, sc_x_base, sc_w_base)

                return global_accs

            # PROLOGUE — pre-fill state for main-loop iter 0.
            for i in range_constexpr(num_buffers):
                issue_tdm_loads(i, i)
                issue_x_scale_tdm(i, i)

            # Bulk W-scale buffer_load deferred to here (after prologue TDM issues).
            bulk_w_cur = _issue_w_chunk_const(0)
            if const_expr(USES_W_CHUNK_PREFETCH):
                bulk_w_prefetch = _issue_w_chunk_const(1)
            else:
                bulk_w_prefetch = bulk_w_cur

            # Single wait: retires tile-0; leaves NB-1 tiles pending (NB filled).
            tdm_ops.tensor_wait(REG_PROLOGUE_WAIT)
            gpu.barrier()

            # ds_loads with X-scale interleaved after K-step 0's frags, so X-scale is ready when WMMA #1 completes (no long dscnt drain).
            cur_a, cur_b, cur_x_raw = load_operand_frags_with_xscale_interleave(0)

            cur_w_raw = issue_w_raw_scales_for_tile_experimental(0)
            if const_expr(num_k_tiles > 1):
                prefetch_w_raw = issue_w_raw_scales_for_tile_experimental(1)
            else:
                prefetch_w_raw = zero_w_raw

            # Accumulator init here (before main-loop entry) for source/execution order; emits no IR (LLVM places v_mov zero-inits lazily near first use).
            accs = [acc_zero] * n_accs

            # MAIN LOOP — one K-tile per iter: wmma(cur) → tdm(T+2) → wait(T+1) → ds_read next.
            main_loop_iters_g = num_k_tiles - num_buffers

            # Loop-carried indices: load_idx = next tile to issue, compute_idx = 0 (next tile to consume).
            load_idx_init = arith.constant(num_buffers, type=T.i32)
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
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        cur_chunk_idx_i32 = state[-3]
                        bulk_w_prefetch = state[-4]
                        bulk_w_cur = state[-5]
                        _reg_state = state[:-5]
                    else:
                        _reg_state = state[:-2]
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

                    # WAR wall: barrier BEFORE the TDMs, so every wave finished reading
                    # buffer (load_idx % NB) (its previous-iter ds_read) before this refill.
                    gpu.barrier()

                    # Issue TDMs for tile load_idx BEFORE the wait: keeps NB-1 tiles in
                    # flight across the SECOND barrier (vs dipping to NB-2) -> fuller
                    # memory pipe for the mb case. WAR-safe now via the barrier above
                    # (two barriers/iter: one guards the refill, one guards the ds_read).
                    issue_tdm_loads(load_buf_i32, cur_load_idx)
                    issue_x_scale_tdm(load_buf_i32, cur_load_idx)

                    # Wait for tile compute_idx+1 to land in LDS.
                    tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)
                    gpu.barrier()

                    # Pre-load tile compute_idx+1 into VGPRs; double-buffer B first (own vgprs) so ds_reads overlap this tile's WMMAs, then A.
                    next_b = load_b_frags(next_buf_i32)
                    next_x_raw = ds_read_x_scales(next_buf_i32)
                    next_a = load_a_frags(next_buf_i32)

                    cur_a = next_a
                    cur_b = next_b
                    cur_x_raw = next_x_raw
                    cur_w_raw = prefetch_w_raw

                    # Prefetch W-scale for compute_idx+2 (zero-masked if past num_k_tiles).
                    future_tile_i32 = arith.addi(cur_compute_idx, two_i32)
                    future_tile_idx = arith.index_cast(T.index, future_tile_i32)
                    prefetch_w_raw = issue_w_raw_scales_for_future_tile_rt_experimental(
                        future_tile_idx
                    )

                    # W-chunk advance: trigger when next_compute_idx crosses a 32-K-block boundary.
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
                        new_load_idx,
                        next_compute_idx,
                    ]
                    results = yield new_state

                final_compute_idx = results[-1]
                if const_expr(USES_W_CHUNK_PREFETCH):
                    cur_chunk_idx_i32 = results[-3]
                    bulk_w_prefetch = results[-4]
                    bulk_w_cur = results[-5]
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

            # EPILOGUE - scf.for drain, small carry (accs + tile idx, frags reloaded fresh); descending tensorcnt via range_constexpr if-ladder; drain_count_d tiles, no new TDMs.
            drain_compute_idx = final_compute_idx
            nb_const_i32_d = arith.constant(num_buffers, type=T.i32)
            one_i32_d = arith.constant(1, type=T.i32)
            drain_count_d = (
                num_buffers if main_loop_iters_g > 0 else min(num_buffers, num_k_tiles)
            )
            drain_init = list(accs) + [drain_compute_idx]
            for _drain_i, dstate in range(0, drain_count_d, 1, init=drain_init):
                _disable_unroll_on_enclosing_loop()
                accs_d = list(dstate[:N_ACCS])
                cur_dci_d = dstate[-1]
                drain_buf_i32 = arith.remui(cur_dci_d, nb_const_i32_d)

                # cur tile == num_k_tiles-1-_k  ->  _k later tiles still in flight.
                for _k in range_constexpr(drain_count_d):
                    _is_k = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        cur_dci_d,
                        arith.constant(num_k_tiles - 1 - _k, type=T.i32),
                    )
                    if _is_k:
                        tdm_ops.tensor_wait(_k * _TDMS_PER_TILE_EXP)
                gpu.barrier()

                # Reload this tile's operands FRESH (not carried across the loop).
                cur_a_d, cur_b_d, cur_x_d = load_operand_frags_with_xscale_interleave(
                    drain_buf_i32
                )
                cur_w_d = issue_w_raw_scales_for_future_tile_rt_experimental(
                    arith.index_cast(T.index, cur_dci_d)
                )
                accs_d = compute_wmma_with_frags_experimental(
                    accs_d, cur_a_d, cur_b_d, cur_x_d, cur_w_d
                )
                dresults = yield list(accs_d) + [arith.addi(cur_dci_d, one_i32_d)]
            accs = list(dresults[:N_ACCS])

        elif const_expr(variant == "memory_bound"):
            # ───── compute_bound-only helpers ─────

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

            def issue_w_raw_scales_experimental(k_base):
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
                return issue_w_raw_scales_experimental(arith.index(tile_idx * tile_k))

            def issue_w_raw_scales_for_future_tile_rt_experimental(future_tile_rt):
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

            def load_operand_frags_with_xscale_interleave(buffer_idx):
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
                                cycle_stride_bytes=256,
                                ks_stride_bytes=2048,
                            )
                        )
                    # Pin X-scale ds_read right after K-step 0's frag loads.
                    if const_expr(ks == 0):
                        x_raw = ds_read_x_scales(buffer_idx)
                return a_frags, b_frags, x_raw

            def load_operand_frags(buffer_idx):
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
                                cycle_stride_bytes=256,
                                ks_stride_bytes=2048,
                            )
                        )
                return a_frags, b_frags

            def load_a_frags(buffer_idx):
                if const_expr(isinstance(buffer_idx, int)):
                    slot_off_a = arith.index(buffer_idx * lds_a_data_bytes)
                else:
                    slot_off_a = arith.index_cast(
                        T.index, arith.muli(buffer_idx, slot_stride_a_i32)
                    )
                a_frags = []
                for ks in range_constexpr(k_wmma_steps):
                    for wm in range_constexpr(wmma_m_rep):
                        a_frags.append(
                            _load_frag(big_a_mem, a_lane_bases[wm] + slot_off_a, ks)
                        )
                return a_frags

            def load_b_frags(buffer_idx):
                # B-only load, split out so the main loop can prefetch B into its
                # OWN vgprs (double-buffered) the moment the n+1 tensor_wait fires.
                if const_expr(isinstance(buffer_idx, int)):
                    slot_off_b = arith.index(buffer_idx * lds_b_data_bytes)
                else:
                    slot_off_b = arith.index_cast(
                        T.index, arith.muli(buffer_idx, slot_stride_b_i32)
                    )
                b_frags = []
                for ks in range_constexpr(k_wmma_steps):
                    for wn in range_constexpr(wmma_n_rep):
                        b_frags.append(
                            _load_frag(
                                big_b_mem,
                                b_lane_bases[wn] + slot_off_b,
                                ks,
                                cycle_stride_bytes=256,
                                ks_stride_bytes=2048,
                            )
                        )
                return b_frags

            def compute_wmma_with_frags_experimental(
                global_accs, a_frags, b_frags, x_raw, w_raw
            ):

                a_frag_ts, b_frag_ts = _rmem_frag_lists(a_frags, b_frags)

                def issue_wmma_temp(sc, wm, wn):
                    return issue_wmma_step(sc, wm, wn, a_frag_ts, b_frag_ts)

                def compute_scale(wm, wn, sc_x_base, sc_w_base):
                    return arith.mulf(x_raw[sc_x_base + wm], w_raw[sc_w_base + wn])

                def wmma_with_scale(temp, wm, wn, idx, sc_x_base, sc_w_base):
                    scale = compute_scale(wm, wn, sc_x_base, sc_w_base)
                    global_accs[idx] = apply_scale(temp, scale, global_accs[idx])

                for sc in range_constexpr(scales_per_tile):
                    sc_x_base = sc * wmma_m_rep
                    sc_w_base = sc * wmma_n_rep

                    # No manual fold-lag / WMMA software-pipeline: issue each WMMA
                    # and fold it immediately, in acc_coords order. The 64
                    # (WMMA -> scale-fma) chains are independent, so the coexec
                    # scheduler is free to overlap them however it likes.
                    for wm, wn, idx in acc_coords:
                        temp = issue_wmma_temp(sc, wm, wn)
                        wmma_with_scale(temp, wm, wn, idx, sc_x_base, sc_w_base)

                return global_accs

            def precompute_scales_experimental(x_raw, w_raw):
                # Hoisted scale products (the v_muls): scales[sc][idx] = x_raw*w_raw,
                # computed up front so the WMMA->fold chain never stalls on a v_mul.
                scales = []
                for sc in range_constexpr(scales_per_tile):
                    sc_x_base = sc * wmma_m_rep
                    sc_w_base = sc * wmma_n_rep
                    sc_row = [None] * n_accs
                    for wm, wn, idx in acc_coords:
                        sc_row[idx] = arith.mulf(
                            x_raw[sc_x_base + wm], w_raw[sc_w_base + wn]
                        )
                    scales.append(sc_row)
                return scales

            def compute_wmma_with_precomputed_scales(
                global_accs, a_frags, b_frags, scales
            ):
                # Like compute_wmma_with_frags_experimental but folds with precomputed
                # scales (v_muls already issued) -- bit-identical result, decoupled sched.
                a_frag_ts, b_frag_ts = _rmem_frag_lists(a_frags, b_frags)
                for sc in range_constexpr(scales_per_tile):
                    for wm, wn, idx in acc_coords:
                        temp = issue_wmma_step(sc, wm, wn, a_frag_ts, b_frag_ts)
                        global_accs[idx] = apply_scale(
                            temp, scales[sc][idx], global_accs[idx]
                        )
                return global_accs

            # PROLOGUE — pre-fill state for main-loop iter 0.
            # Boost wave priority for the TDM issue burst to compress wave-dispatch skew.
            # MEMORY-BOUND: fill NB-1 buffers in the prologue (not all NB) -- leaves a
            # 2-iter gap between a buffer's ds_read and its refill (single-barrier safe).
            for i in range_constexpr(num_buffers - 1):
                issue_tdm_loads(i, i)
                issue_x_scale_tdm(i, i)

            bulk_w_chunks = [_issue_w_chunk_const(c) for c in range(NUM_W_CHUNKS)]
            bulk_w_cur = bulk_w_chunks[0]
            bulk_w_prefetch = (
                bulk_w_chunks[1] if USES_W_CHUNK_PREFETCH else bulk_w_chunks[0]
            )

            # Single wait: retires tile-0 X+W+S; leaves NB-1 tiles pending.
            tdm_ops.tensor_wait(REG_PROLOGUE_WAIT)
            gpu.barrier()

            cur_a, cur_b, cur_x_raw = load_operand_frags_with_xscale_interleave(0)

            cur_w_raw = issue_w_raw_scales_for_tile_experimental(0)
            if const_expr(num_k_tiles > 1):
                prefetch_w_raw = issue_w_raw_scales_for_tile_experimental(1)
            else:
                prefetch_w_raw = zero_w_raw

            accs = [acc_zero] * n_accs

            # MEMORY-BOUND: prologue fills NB-1 tiles, so the steady loop runs one MORE
            # iteration than fills-NB, and the first tile it issues is NB-1 (lookahead
            # = NB-1).
            main_loop_iters_g = num_k_tiles - num_buffers + 1

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
                    if const_expr(USES_W_CHUNK_PREFETCH):
                        cur_chunk_idx_i32 = state[-3]
                        bulk_w_prefetch = state[-4]
                        bulk_w_cur = state[-5]
                        _reg_state = state[:-5]
                    else:
                        _reg_state = state[:-2]
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

                    # Wait at the TOP: land tile compute_idx+1 before any of its
                    # ds_reads. The wait is ABOVE the TDM, so it does NOT count this
                    # iter's refill -> target (NB-3)*3; pipe holds NB-2 tiles in flight.
                    tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING_EXPERIMENTAL)

                    # Issue TDMs for tile load_idx (refill buffer load_idx%NB). NB-1
                    # prologue fill puts this refill one buffer behind (last ds_read 2
                    # iters ago), so the barrier below is its WAR wall across the backedge.
                    issue_tdm_loads(load_buf_i32, cur_load_idx)
                    issue_x_scale_tdm(load_buf_i32, cur_load_idx)

                    # Barrier AFTER the TDM: RAW wall for ALL tile compute_idx+1 ds_reads
                    # below (X-scale + B + A), AND (across the backedge) the WAR wall for
                    # the next iter's refill. The NB-1 gap lets one barrier cover both.
                    gpu.barrier()

                    # Scale multiplies for the CURRENT tile, hoisted before the WMMA so
                    # the v_muls are off the WMMA->fold chain (from cur_x/cur_w regs).
                    scales = precompute_scales_experimental(cur_x_raw, cur_w_raw)

                    # ds_load tile compute_idx+1's X-scale (A-scale) early, before WMMA.
                    next_x_raw = ds_read_x_scales(next_buf_i32)

                    # WMMA on cur tile (cur frags), folding the precomputed scales.
                    cur_accs = compute_wmma_with_precomputed_scales(
                        cur_accs, cur_a, cur_b, scales
                    )

                    # ds_load tile compute_idx+1's B and A frags (double-buffer B first).
                    next_b = load_b_frags(next_buf_i32)
                    next_a = load_a_frags(next_buf_i32)

                    cur_a = next_a
                    cur_b = next_b
                    cur_x_raw = next_x_raw
                    cur_w_raw = prefetch_w_raw

                    future_tile_i32 = arith.addi(cur_compute_idx, two_i32)
                    future_tile_idx = arith.index_cast(T.index, future_tile_i32)
                    prefetch_w_raw = issue_w_raw_scales_for_future_tile_rt_experimental(
                        future_tile_idx
                    )

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
                        new_load_idx,
                        next_compute_idx,
                    ]
                    results = yield new_state

                final_compute_idx = results[-1]
                if const_expr(USES_W_CHUNK_PREFETCH):
                    cur_chunk_idx_i32 = results[-3]
                    bulk_w_prefetch = results[-4]
                    bulk_w_cur = results[-5]
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

            # EPILOGUE — runtime scf.for drain, SMALL carry (accs + tile index only;
            # frags reloaded FRESH each iter, so the loop-carried set is small, not the
            # full A/B/scale state). Descending tensorcnt via a range_constexpr
            # if-ladder: the runtime tile index picks which compile-time-immediate
            # tensor_wait fires (never hardcoded 0). The small carry is what keeps the
            # runtime scf.if off the backend's crash path. One loop of drain_count_d
            # tiles (folds the old rotate-loop + final WMMA); min(...) clamps the
            # num_k_tiles <= num_buffers path. No new TDMs issued.
            drain_compute_idx = final_compute_idx
            nb_const_i32_d = arith.constant(num_buffers, type=T.i32)
            one_i32_d = arith.constant(1, type=T.i32)
            # MEMORY-BOUND: NB-1 fill -> lookahead is NB-1 tiles, so the drain computes
            # NB-1 tiles (not NB); min(...) clamps the degenerate num_k_tiles == NB-1 path.
            drain_count_d = (
                num_buffers - 1
                if main_loop_iters_g > 0
                else min(num_buffers - 1, num_k_tiles)
            )
            drain_init = list(accs) + [drain_compute_idx]
            for _drain_i, dstate in range(0, drain_count_d, 1, init=drain_init):
                _disable_unroll_on_enclosing_loop()
                accs_d = list(dstate[:N_ACCS])
                cur_dci_d = dstate[-1]
                drain_buf_i32 = arith.remui(cur_dci_d, nb_const_i32_d)

                # cur tile == num_k_tiles-1-_k  ->  _k later tiles still in flight.
                for _k in range_constexpr(drain_count_d):
                    _is_k = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        cur_dci_d,
                        arith.constant(num_k_tiles - 1 - _k, type=T.i32),
                    )
                    if _is_k:
                        tdm_ops.tensor_wait(_k * _TDMS_PER_TILE_EXP)
                gpu.barrier()

                # Reload this tile's operands FRESH (not carried across the loop).
                cur_a_d, cur_b_d, cur_x_d = load_operand_frags_with_xscale_interleave(
                    drain_buf_i32
                )
                cur_w_d = issue_w_raw_scales_for_future_tile_rt_experimental(
                    arith.index_cast(T.index, cur_dci_d)
                )
                accs_d = compute_wmma_with_frags_experimental(
                    accs_d, cur_a_d, cur_b_d, cur_x_d, cur_w_d
                )
                dresults = yield list(accs_d) + [arith.addi(cur_dci_d, one_i32_d)]
            accs = list(dresults[:N_ACCS])

        # Step 4: convert f32 accs to out_dtype, buffer_store to Y.
        if const_expr(num_buffers > 2):
            rocdl.sched_barrier(0)

        out_elem = (
            T.bf16 if out_dtype == "bf16" else T.f16 if out_dtype == "fp16" else None
        )
        is_half_out = out_dtype in ("bf16", "fp16")

        if use_tdm_store:
            d_lds_buffer = _lds_region_ptr(unified_d_off)

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

            d_elem_ty = (
                fx.Float32
                if not is_half_out
                else (fx.Float16 if out_dtype == "fp16" else fx.BFloat16)
            )
            gY = fx.Tensor(
                fx.make_view(
                    fx.add_offset(fx.get_iter(arg_y), blk_m64 * fx.Int64(N) + blk_n64),
                    fx.make_layout((tile_m, tile_n), (N, 1)),
                )
            )
            atom_d = fx.rocdl.make_tdm_atom(
                gY,
                [None, None],
                strides=[fx.Int64(N), None],
                num_warps=num_warps,
            )
            fx.copy(
                atom_d,
                fx.Tensor(
                    fx.make_view(
                        fx.recast_iter(d_elem_ty, _lds_region_ptr(unified_d_off)),
                        fx.make_layout((tile_m, tile_n), (tile_n, 1)),
                    )
                ),
                gY,
            )
            tdm_ops.tensor_wait(0)
        else:
            if const_expr(split_k > 1):
                zero_i32_s = arith.constant(0, type=T.i32)
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

                    if const_expr(split_k > 1):
                        for half in range_constexpr(2):
                            col_h = col_base + arith.index(half * 4)
                            for vi in range_constexpr(4):
                                val = fx.Vector(accs[idx])[half * 4 + vi].ir_value()
                                byte_off = arith.index_cast(
                                    T.i32,
                                    (row * n_stride + col_h + arith.index(vi))
                                    * arith.index(4),
                                )
                                rocdl.raw_ptr_buffer_atomic_fadd(
                                    val, y_buf, byte_off, zero_i32_s, zero_i32_s
                                )
                    elif is_half_out:
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
        split_k,
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
            grid=(gx, gy, split_k),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_gemm_a8w8_blockscale.compile_hints["llvm_options"] = {
        "unroll-threshold": 0,
    }

    # Commented out until coexec branch is merged.
    # launch_gemm_a8w8_blockscale.compile_hints["llvm_options"] = {
    #     "amdgpu-expert-scheduling-mode": True,
    #     "amdgpu-anti-hints-for-va-vdst": True,
    #     "amdgpu-enable-static-simulator": True,
    #     "amdgpu-static-sim-inline": True,
    #     "amdgpu-sched-strategy": "coexec",
    #     # "amdgpu-block-carried-latency": EnumOpt("all"),  # enable per the note above
    # }

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
    waves_per_eu: int | None = None,
    l2_prefetch_distance: int = 0,
    variant: str = "compute_bound",
    use_tdm_store: bool = False,
    loop_carried_load_percent: int | None = None,
    kernarg_preload: bool = False,
    split_k: int = 1,
):
    """Compute Y = (X @ W^T) with per-block f32 scales (A8W8 blockscale).

    variant: "compute_bound" (default) or "memory_bound".
      - "compute_bound" : prologue fills all NB buffers; loop-carries
                        operand frags, 2 barriers/iter.
      - "memory_bound"  : prologue fills NB-1 buffers, 1 barrier/iter.
                        W-scales bulk-loaded via buffer_load_b32 + readlane,
                        X-scales TDM-staged into LDS. Requires
                        w_is_wave_uniform.
    """
    assert x.ndim == 2 and w.ndim == 2, "X and W must be 2D"
    M, K = x.shape
    # W is always preshuffled to (N//16, K*16) for the gfx1250 fragment path; recover logical (N, K) from that shape.
    N = w.shape[0] * 16
    K_w = w.shape[1] // 16
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

    _splitk_f32_accum = split_k > 1 and dtype in (torch.bfloat16, torch.float16)
    buf_dtype = torch.float32 if _splitk_f32_accum else dtype
    if _splitk_f32_accum:
        out_dtype_str = "f32"

    K_padded = ((K + tile_k - 1) // tile_k) * tile_k
    if K_padded != K:
        pad_size = K_padded - K
        x = torch.nn.functional.pad(x, (0, pad_size))
        w = torch.nn.functional.pad(w, (0, pad_size * 16))
        new_scale_k = K_padded // scale_block_k_derived
        scale_pad = new_scale_k - scale_k
        if scale_pad > 0:
            x_scale = torch.nn.functional.pad(x_scale, (0, scale_pad))
            w_scale = torch.nn.functional.pad(w_scale, (0, scale_pad))
        K = K_padded

    # Pad N up to tile_n so the kernel's WMMAs and stores land inside the allocated output.
    N_stride = ((N + tile_n - 1) // tile_n) * tile_n

    if y is not None:
        assert y.shape == (M, N), f"y shape {y.shape} != ({M}, {N})"
        assert y.dtype == dtype, f"y dtype {y.dtype} != {dtype}"

    _alloc = torch.zeros if split_k > 1 else torch.empty
    if _splitk_f32_accum:
        y_buf = _alloc((M, N_stride), dtype=buf_dtype, device=x.device)
    elif N_stride != N:
        y_buf = _alloc((M, N_stride), dtype=dtype, device=x.device)
    elif y is not None:
        y_buf = y
        if split_k > 1:
            y_buf.zero_()
    else:
        y_buf = _alloc((M, N), dtype=dtype, device=x.device)

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
        split_k=split_k,
    )

    stream = torch.cuda.current_stream(device=x.device).cuda_stream
    launcher(y_buf, x, w, x_scale, w_scale, M, N_stride, stream=stream)

    if _splitk_f32_accum:
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


__all__ = [
    "compile_gemm_a8w8_blockscale",
    "gemm_a8w8_blockscale",
]
