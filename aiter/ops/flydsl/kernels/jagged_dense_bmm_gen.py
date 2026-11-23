# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernel for jagged_dense_bmm_broadcast_add (grouped BF16 GEMM + bias).

For each group ``b`` over ``[seq_offsets[b], seq_offsets[b+1])``::

    Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b]

Dense is ``(B, K, N)``, N-contiguous, transposed in registers. Supports a uniform
grid launch with optional XCD remap, and a compact launch for skewed
sequence lengths that maps each workgroup to its group on device.
"""

from __future__ import annotations

import ctypes
import functools
import math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.typing import T
from flydsl.utils import env

from aiter.jit.utils.chip_info import (
    get_lds_capacity_bytes,
    get_num_xcds,
    load_hip_runtime,
)

from .tensor_shim import _run_compiled, buf_scalar_load

BLOCK_M = 128
BLOCK_N = 128
STAGES_A = 2
_COPY_VEC = 16 // torch.bfloat16.itemsize

_INT32_MAX = 2**31 - 1
# An HSA dispatch packet holds each axis's work-item count in a uint32.
_MAX_DISPATCH_ITEMS = 2**32 - 1
# A buffer descriptor holds num_records in 32 bits.
_MAX_BUFFER_BYTES = 2**32 - 1
# A buffer offset is 32 bits, so a tile row past num_records can wrap back into range.
_BUFFER_OFFSET_SPAN = 2**32
_HIP_ATTR_MAX_GRID_DIM_X = 29  # hipDeviceAttributeMaxGridDimX; Y and Z follow.


@dataclass(frozen=True)
class ArchTraits:
    mma_mnk: tuple[int, int, int]
    k_perm_shape: tuple[int, int, int]
    k_perm_stride: tuple[int, int, int]
    lds_bytes: int
    a_swizzle: tuple[int, int, int]
    wave_size: int
    min_block_k: int
    a_lds_dma: bool
    b_stages: int

    @property
    def k_tile_gran(self) -> int:
        return math.prod(self.k_perm_shape)


ARCH_TRAITS = {
    "gfx942": ArchTraits(
        mma_mnk=(16, 16, 16),
        k_perm_shape=(4, 4, 2),
        k_perm_stride=(1, 8, 4),
        lds_bytes=get_lds_capacity_bytes("gfx942"),
        a_swizzle=(3, 3, 3),
        wave_size=64,
        min_block_k=64,
        a_lds_dma=False,
        b_stages=3,
    ),
    "gfx950": ArchTraits(
        mma_mnk=(16, 16, 32),
        k_perm_shape=(8, 4, 1),
        k_perm_stride=(1, 8, 0),
        lds_bytes=get_lds_capacity_bytes("gfx950"),
        a_swizzle=(3, 3, 3),
        wave_size=64,
        min_block_k=64,
        a_lds_dma=True,
        b_stages=2,
    ),
}

_DEFAULT_STREAM = fx.Stream(None)


def _f32_to_bf16_rne(x):
    """fp32 -> bf16, round to nearest even, with no NaN check.

    A NaN input must have a zero low 16 bits (bf16-derived or the hardware
    default NaN). Any other NaN can round into Inf.
    """
    n = x.numel
    bits = x.bitcast(fx.Int32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    halves = rounded.bitcast(fx.Int16)
    hi = halves.shuffle(halves, list(range(1, 2 * n, 2)))
    return fx.Vector(hi.bitcast(fx.BFloat16), x.shape, fx.BFloat16)


def _xcd_remap(xy, num_rows, num_cols, C, W, nXCD):
    xy = fx.Int32(xy)
    cnXCD = fx.Int32(nXCD)
    cC = fx.Int32(C)
    cW = fx.Int32(W)
    c_num_cols = fx.Int32(num_cols)
    c_num_rows = fx.Int32(num_rows)

    xcd = xy % cnXCD
    local = xy // cnXCD
    chunk_idx = local // cC
    pos = local % cC
    xy_g_remap = chunk_idx * (cnXCD * cC) + xcd * cC + pos

    total = c_num_rows * c_num_cols
    period = fx.Int32(nXCD * C)
    prefix = total - (total % period)
    xy_g = (xy < prefix).select(xy_g_remap, xy)

    tids_per_grp = cW * c_num_cols
    group_id = xy_g // tids_per_grp
    first_row = group_id * cW
    remaining = c_num_rows - first_row
    win_h = (remaining < cW).select(remaining, cW)  # last window may be short
    tid_in_grp = xy_g % tids_per_grp
    row = first_row + (tid_in_grp % win_h)
    col = tid_in_grp // win_h
    return row, col


_LAUNCHERS: dict = {}


def _row_major(t):
    # FlyDSL compiles the first stride-1 axis as static, and a contiguous tensor
    # may carry any stride on a size-1 dim.
    strides = [1] * t.dim()
    for i in range(t.dim() - 1, 0, -1):
        strides[i - 1] = strides[i] * t.shape[i]
    return t if t.stride() == tuple(strides) else t.as_strided(t.shape, strides)


def _build_launcher(
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    THREADS,
    XCD_C,
    XCD_W,
    COMPACT,
    NXCD,
    TRAITS,
):
    MMA_N = TRAITS.mma_mnk[1]
    K_STEP = TRAITS.k_tile_gran
    K_ATOMS = K_STEP // TRAITS.mma_mnk[2]
    B_STAGES = TRAITS.b_stages
    N_BLOCKS = N // BLOCK_N  # output column-tiles per group (compile-time)
    # >= 2 adjacent n per lane, so each B load is at least 32-bit.
    WAVES = THREADS // TRAITS.wave_size
    N_WARPS = min(WAVES, BLOCK_N // (MMA_N * 2))
    M_WARPS = WAVES // N_WARPS
    B_REPS = BLOCK_N // (MMA_N * N_WARPS)
    smem_bytes = BLOCK_M * BLOCK_K * STAGES_A * 2  # bf16 A double-buffer staging

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def jdbba_kernel(
        C: fx.Tensor,  # out    (L, N)   bf16
        A: fx.Tensor,  # jagged (L, K)   bf16
        B: fx.Tensor,  # dense  (n_groups * K, N) bf16, view of (n_groups, K, N)
        BIAS: fx.Tensor,  # bias   (n_groups * N,)   bf16
        SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
        L: fx.Int32,
        n_groups: fx.Int32,
        search_steps: fx.Int32,
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s_A: fx.TiledCopy,
        tiled_copy_s2g_C: fx.TiledCopy,
    ):
        tid = fx.thread_idx.x
        pid_mn, _, raw_b = fx.block_idx
        seq_buf = fx.rocdl.make_buffer_tensor(SEQ_OFFSETS, max_size=True)

        def load_seq(i):
            return fx.Int32(buf_scalar_load(seq_buf, i))

        if fx.const_expr(COMPACT):
            if fx.const_expr(XCD_C > 1):
                grid_x = fx.Int32(fx.grid_dim.x)
                raw_xy = fx.Int32(raw_b) * grid_x + fx.Int32(pid_mn)
                tile_row, n_col = _xcd_remap(
                    raw_xy, grid_x, N_BLOCKS, XCD_C, XCD_W, NXCD
                )
                tile_row = fx.Int32(fx.rocdl.readfirstlane(T.i32, tile_row))
                block_n_idx = fx.Int32(fx.rocdl.readfirstlane(T.i32, n_col))
            else:
                tile_row = fx.Int32(pid_mn)
                block_n_idx = fx.Int32(raw_b)
            # Group g owns M-tile slots [f(g), f(g+1)), with
            # f(g) = seq_offsets[g] // BLOCK_M + g. f is strictly increasing and
            # gives each group >= ceil(M_g / BLOCK_M) slots; surplus slots fail
            # start_m < M_b.
            c_bm_shift = fx.Int32(BLOCK_M.bit_length() - 1)

            def slot_start(g, seq_g):
                return (seq_g >> c_bm_shift) + g

            c_one = fx.Int32(1)
            for _, state in range(
                fx.Int32(0),
                search_steps,
                c_one,
                init=[fx.Int32(0), n_groups - c_one],
            ):
                lo, hi = state[0], state[1]
                mid = (lo + hi + c_one) >> c_one
                take = slot_start(mid, load_seq(mid)) <= tile_row
                results = yield [take.select(mid, lo), take.select(hi, mid - c_one)]
            off_b = results[0]
            seq_start = load_seq(off_b)
            seq_end = load_seq(off_b + c_one)
            block_m_idx = tile_row - slot_start(off_b, seq_start)
        else:
            grid_x = fx.Int32(fx.grid_dim.x)
            bm_tiles = grid_x // fx.Int32(N_BLOCKS)
            num_rows = bm_tiles * fx.Int32(fx.grid_dim.z)
            raw_xy = fx.Int32(raw_b) * grid_x + fx.Int32(pid_mn)
            row, col = _xcd_remap(raw_xy, num_rows, N_BLOCKS, XCD_C, XCD_W, NXCD)
            row = fx.Int32(fx.rocdl.readfirstlane(T.i32, row))
            col = fx.Int32(fx.rocdl.readfirstlane(T.i32, col))

            off_b = row // bm_tiles
            block_m_idx = row % bm_tiles
            block_n_idx = col
            seq_start = load_seq(off_b)
            seq_end = load_seq(off_b + fx.Int32(1))
        M_b = seq_end - seq_start
        start_m = fx.Int32(block_m_idx) * fx.Int32(BLOCK_M)

        if start_m < M_b:
            tile_start = seq_start + start_m
            tile_rows = M_b - start_m
            tile_rows = (tile_rows < fx.Int32(BLOCK_M)).select(
                tile_rows, fx.Int32(BLOCK_M)
            )
            rows_left = L - tile_start
            tile_rows = (rows_left < tile_rows).select(rows_left, tile_rows)
            tile_rows = (tile_rows < fx.Int32(0)).select(fx.Int32(0), tile_rows)
            # A VGPR num_records makes every buffer access a waterfall loop.
            tile_rows = fx.Int32(fx.rocdl.readfirstlane(T.i32, tile_rows))
            a_row_off = fx.Int64(tile_start) * fx.Int64(K)
            c_row_off = fx.Int64(tile_start) * fx.Int64(N)
            A_g = fx.make_view(
                fx.add_offset(fx.get_iter(A), fx.make_int_tuple(a_row_off)),
                fx.get_layout(A),
            )
            C_g = fx.make_view(
                fx.add_offset(fx.get_iter(C), fx.make_int_tuple(c_row_off)),
                fx.get_layout(C),
            )
            b_row_off = fx.Int64(off_b) * fx.Int64(N) * fx.Int64(K)
            B_g = fx.make_view(
                fx.add_offset(fx.get_iter(B), fx.make_int_tuple(b_row_off)),
                fx.make_layout((N, K), (1, N)),
            )

            # Rows past tile_rows fall outside num_records: loads read 0, stores drop.
            A_buf = fx.rocdl.make_buffer_tensor(
                A_g, num_records_bytes=fx.Int64(tile_rows) * fx.Int64(K * 2)
            )
            B_buf = fx.rocdl.make_buffer_tensor(B_g, max_size=True)
            C_buf = fx.rocdl.make_buffer_tensor(
                C_g, num_records_bytes=fx.Int64(tile_rows) * fx.Int64(N * 2)
            )

            gA_k = fx.flat_divide(A_buf, (BLOCK_M, BLOCK_K))[None, None, 0, None]
            gB_k = fx.flat_divide(B_buf, (BLOCK_N, BLOCK_K))[
                None, None, block_n_idx, None
            ]
            gC = fx.flat_divide(C_buf, (BLOCK_M, BLOCK_N))[None, None, 0, block_n_idx]

            bias_elem_off = fx.Int32(off_b) * fx.Int32(N)
            BIAS_g = fx.make_view(
                fx.add_offset(fx.get_iter(BIAS), fx.make_int_tuple(bias_elem_off)),
                fx.get_layout(BIAS),
            )
            BIAS_buf = fx.rocdl.make_buffer_tensor(BIAS_g, max_size=True)
            gBias2d = fx.make_view(
                fx.get_iter(BIAS_buf), fx.make_layout((BLOCK_M, N), (0, 1))
            )
            gBias = fx.flat_divide(gBias2d, (BLOCK_M, BLOCK_N))[
                None, None, 0, block_n_idx
            ]

            thr_mma = tiled_mma.thr_slice(tid)
            thr_copy_g2s_A = tiled_copy_g2s_A.get_slice(tid)

            uni_copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            buffer_copy_128b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

            tiled_copy_s2r_A = fx.make_tiled_copy_A(uni_copy_128b, tiled_mma)
            thr_copy_s2r_A = tiled_copy_s2r_A.get_slice(tid)
            b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy(B_REPS * MMA_N), fx.BFloat16)
            tv_mma_B = tiled_mma.tv_layout_B_tiled
            k_in_atom, n_rep, k_rep = 0, 1, 2
            ring_modes = [n_rep, k_in_atom] + ([k_rep] if K_ATOMS > 1 else [])
            vals_g2r_B = fx.coalesce(fx.select(fx.coalesce(tv_mma_B[1]), ring_modes))
            tv_g2r_B = fx.group(fx.append(tv_mma_B[0], vals_g2r_B), 0, tv_mma_B[0].rank)
            thr_copy_g2r_B = fx.make_tiled_copy(
                b_copy, tv_g2r_B, fx.make_tile(BLOCK_N, K_STEP)
            ).get_slice(tid)

            a_swizzle = fx.static(fx.SwizzleType.get(*TRAITS.a_swizzle))
            a_smem = fx.make_ordered_layout((BLOCK_M, BLOCK_K, STAGES_A), (1, 0, 2))
            composed_layout_A = fx.make_composed_layout(a_swizzle, a_smem)
            sA = fx.make_view(fx.get_dyn_shared(fx.BFloat16), composed_layout_A)

            thr_gA_k = thr_copy_g2s_A.partition_S(gA_k)
            thr_sA = thr_copy_g2s_A.partition_D(sA)
            thr_sA_s2r = thr_copy_s2r_A.partition_S(sA)
            thr_gB_k = thr_copy_g2r_B.partition_S(gB_k)

            copy_frag_A = fx.make_fragment_like(thr_sA[None, None, None, 0])

            mma_frag_A = thr_mma.make_fragment_A(sA[None, None, 0])
            mma_frag_C = thr_mma.make_fragment_C(gC)

            mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)

            def k_coord(kc):
                return (None, kc) if K_ATOMS > 1 else kc

            # Ring slots hold (n-rep, k) as loaded; each K32 step shuffles
            # them into the MFMA's K-packed (k, n-rep) operand.
            b_ring = [
                fx.make_fragment_like(thr_gB_k[None, None, None, 0])
                for _ in range(B_STAGES)
            ]
            mma_frag_B = thr_mma.make_fragment_B(gB_k[None, None, 0])
            mma_to_ring_B = fx.composition(fx.left_inverse(vals_g2r_B), tv_mma_B[1])
            b_shuffle = [
                fx.get_scalar(fx.crd2idx(i, mma_to_ring_B))
                for i in range(fx.get_scalar(fx.size(mma_to_ring_B)))
            ]

            def b_dst(slot):
                return b_ring[slot]

            def b_operand(slot, kc):
                raw = b_ring[slot][None, None, kc].load()
                dst = mma_frag_B[None, None, k_coord(kc)]
                dst.store(raw.shuffle(raw, b_shuffle).reshape(dst.shape.to_py_value()))
                return dst

            gA_k_stride = fx.get_scalar(gA_k.stride[2])
            gB_k_stride = fx.get_scalar(gB_k.stride[2])

            if fx.const_expr(TRAITS.a_lds_dma):
                dma_bits = 128
                dma_vec = dma_bits // fx.BFloat16.width
                dma_rounds = BLOCK_M * BLOCK_K // (THREADS * dma_vec)
                dma_atom = fx.make_copy_atom(
                    fx.rocdl.cdna4.BufferLoadAsyncLDS(dma_bits), dma_bits
                )
                unit = fx.make_layout(1, 1)
                # LDS DMA writes lane i's chunk at a wave-uniform base + i * dma_bits/8.
                a_phys = fx.make_ordered_layout(
                    (dma_vec, THREADS, dma_rounds, STAGES_A), (0, 1, 2, 3)
                )
                # The XOR swizzle is its own inverse: slot p holds logical swizzle(p).
                a_slot_to_global = fx.make_composed_layout(
                    fx.composition(
                        fx.make_layout((BLOCK_M, BLOCK_K), (K, 1)),
                        fx.left_inverse(fx.slice(a_smem, (None, None, 0))),
                    ),
                    fx.make_composed_layout(a_swizzle, a_phys),
                )
                a_dma_src = fx.get_iter(A_buf)
                a_dma_off = [
                    fx.get_scalar(fx.crd2idx((0, tid, rnd, 0), a_slot_to_global))
                    for rnd in fx.range_constexpr(dma_rounds)
                ]
                a_dma_wave = fx.Int32(
                    fx.rocdl.readfirstlane(T.i32, tid // TRAITS.wave_size)
                )

                def a_load(k_tile, stage):
                    for rnd in fx.range_constexpr(dma_rounds):
                        lds_off = fx.crd2idx(
                            (0, a_dma_wave * TRAITS.wave_size, rnd, stage), a_phys
                        )
                        fx.copy(
                            dma_atom,
                            fx.make_view(
                                fx.add_offset(
                                    a_dma_src, a_dma_off[rnd] + k_tile * BLOCK_K
                                ),
                                unit,
                            ),
                            fx.make_view(
                                fx.add_offset(
                                    fx.get_dyn_shared(fx.BFloat16),
                                    fx.get_scalar(lds_off),
                                ),
                                unit,
                            ),
                        )
                    fx.rocdl.asyncmark()

                def a_commit(stage):
                    fx.rocdl.wait_asyncmark(0)

            else:

                def a_load(k_tile, stage):
                    fx.copy(
                        buffer_copy_128b,
                        thr_gA_k[None, None, None, 0],
                        copy_frag_A,
                        soffset=fx.Int32(k_tile) * gA_k_stride,
                    )

                def a_commit(stage):
                    fx.copy(
                        uni_copy_128b,
                        copy_frag_A,
                        thr_sA[None, None, None, stage],
                    )

            def run_pipeline_stage(
                a_read_stage, b_read_slot, a_next_k=None, b_next_k=None
            ):
                a_write_stage = a_read_stage ^ 1
                if fx.const_expr(a_next_k is not None):
                    a_load(a_next_k, a_write_stage)
                if fx.const_expr(b_next_k is not None):
                    b_fetch_slot = (b_read_slot + B_STAGES - 1) % B_STAGES
                    fx.copy(
                        b_copy,
                        thr_gB_k[None, None, None, 0],
                        b_dst(b_fetch_slot),
                        soffset=fx.Int32(b_next_k) * gB_k_stride,
                    )

                for block_k_iter in fx.range_constexpr(BLOCK_K // K_STEP):
                    fx.copy(
                        uni_copy_128b,
                        thr_sA_s2r[None, None, block_k_iter, a_read_stage],
                        mma_frag_A_retile[None, None, block_k_iter],
                    )
                    frag_A_k = mma_frag_A[None, None, k_coord(block_k_iter)]
                    frag_B_k = b_operand(b_read_slot, block_k_iter)
                    fx.gemm(
                        tiled_mma,
                        mma_frag_C,
                        frag_A_k,
                        frag_B_k,
                        mma_frag_C,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )

                if fx.const_expr(a_next_k is not None):
                    a_commit(a_write_stage)
                fx.gpu.barrier()

            mma_frag_C_bf16 = fx.make_fragment_like(mma_frag_C, fx.BFloat16.ir_type)
            thr_copy_r2s_C = fx.make_tiled_copy_C(
                fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16), tiled_mma
            ).get_slice(tid)
            mma_frag_C_retile = thr_copy_r2s_C.retile(mma_frag_C_bf16)

            thr_gBias = thr_copy_r2s_C.partition_S(gBias)
            bias_frag = fx.make_fragment_like(mma_frag_C, fx.BFloat16.ir_type)
            fx.copy(
                fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16),
                thr_gBias,
                thr_copy_r2s_C.retile(bias_frag),
            )

            n_tiles = K // BLOCK_K
            a_load(0, 0)
            fx.copy(
                b_copy,
                thr_gB_k[None, None, None, 0],
                b_dst(0),
            )
            for b_pre in fx.range_constexpr(1, min(B_STAGES - 1, n_tiles)):
                fx.copy(
                    b_copy,
                    thr_gB_k[None, None, None, 0],
                    b_dst(b_pre),
                    soffset=fx.Int32(b_pre) * gB_k_stride,
                )
            mma_frag_C.store(bias_frag.load().to(fx.Float32))
            a_commit(0)
            fx.gpu.barrier()

            for k_tile in fx.range_constexpr(n_tiles):
                a_slot = k_tile % 2
                b_slot = k_tile % B_STAGES
                a_next = k_tile + 1 if k_tile + 1 < n_tiles else None
                b_next = (
                    (k_tile + B_STAGES - 1)
                    if (k_tile + B_STAGES - 1) < n_tiles
                    else None
                )
                run_pipeline_stage(a_slot, b_slot, a_next_k=a_next, b_next_k=b_next)

            acc = mma_frag_C.load()
            mma_frag_C_bf16.store(_f32_to_bf16_rne(acc))

            sC = fx.make_view(
                fx.get_dyn_shared(fx.BFloat16),
                fx.make_ordered_layout((BLOCK_M, BLOCK_N), (1, 0)),
            )
            thr_sC = thr_copy_r2s_C.partition_D(sC)

            fx.copy(
                fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16),
                mma_frag_C_retile,
                thr_sC,
            )
            fx.gpu.barrier()

            thr_copy_s2g_C = tiled_copy_s2g_C.get_slice(tid)
            thr_sC_read = thr_copy_s2g_C.partition_S(sC)
            thr_gC = thr_copy_s2g_C.partition_D(gC)
            cs_frag = fx.make_fragment_like(thr_sC_read)
            fx.copy(
                fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
                thr_sC_read,
                cs_frag,
            )
            fx.copy(
                fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16),
                cs_frag,
                thr_gC,
            )

    @flyc.jit
    def _launch(
        C: fx.Tensor,
        A: fx.Tensor,
        B: fx.Tensor,
        BIAS: fx.Tensor,
        SEQ_OFFSETS: fx.Tensor,
        L: int,
        grid_m: int,
        n_groups: int,
        search_steps: int,
        stream: fx.Stream = _DEFAULT_STREAM,
    ):
        n_perm = fx.make_layout((MMA_N, N_WARPS, B_REPS), (B_REPS, MMA_N * B_REPS, 1))
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(*TRAITS.mma_mnk, fx.BFloat16)),
            fx.make_layout((M_WARPS, N_WARPS, 1), (1, M_WARPS, 0)),
            fx.make_tile(
                None,
                n_perm,
                fx.make_layout(TRAITS.k_perm_shape, TRAITS.k_perm_stride),
            ),
        )
        val_per_thr = _COPY_VEC
        thrs_col = BLOCK_K // val_per_thr
        thrs_row = THREADS // thrs_col
        tiled_copy_g2s_A = fx.make_tiled_copy_tv(
            fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
            fx.make_layout((thrs_row, thrs_col), (thrs_col, 1)),
            fx.make_layout((1, val_per_thr), (1, 1)),
        )

        c_val = _COPY_VEC
        c_thrs_n = BLOCK_N // c_val  # threads along N
        c_thrs_m = THREADS // c_thrs_n  # threads along M
        tiled_copy_s2g_C = fx.make_tiled_copy_tv(
            fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16),
            fx.make_layout((c_thrs_m, c_thrs_n), (c_thrs_n, 1)),
            fx.make_layout((1, c_val), (1, 1)),
        )

        epi_smem_bytes = max(smem_bytes, BLOCK_M * BLOCK_N * 2)

        grid = _launch_grid(COMPACT, grid_m, n_groups, N_BLOCKS)
        jdbba_kernel(
            C,
            A,
            B,
            BIAS,
            SEQ_OFFSETS,
            L,
            n_groups,
            search_steps,
            tiled_mma,
            tiled_copy_g2s_A,
            tiled_copy_s2g_C,
        ).launch(grid=grid, block=(THREADS, 1, 1), smem=epi_smem_bytes, stream=stream)

    return _launch


def _launch_grid(compact, grid_m, n_groups, n_blocks):
    if compact:
        return (grid_m, 1, n_blocks)
    return (grid_m * n_blocks, 1, n_groups)


@functools.cache
def _max_grid_dims(device_index: int) -> tuple[int, int, int]:
    libhip = load_hip_runtime()
    dims = []
    for axis in range(3):
        val = ctypes.c_int(0)
        attr = _HIP_ATTR_MAX_GRID_DIM_X + axis
        err = libhip.hipDeviceGetAttribute(ctypes.byref(val), attr, device_index)
        if err != 0:
            raise RuntimeError(
                f"hipDeviceGetAttribute(MaxGridDim{'XYZ'[axis]}) failed with "
                f"error {err} for device {device_index}"
            )
        dims.append(val.value)
    return tuple(dims)


def _check_launch(
    *,
    grid: tuple[int, int, int],
    threads: int,
    block_m: int,
    block_n: int,
    n_groups: int,
    max_seq_len: int,
    L: int,
    K: int,
    N: int,
    compact: bool,
    device_index: int,
) -> None:
    if compact:
        names = (f"A.shape[0] // {block_m} + n_groups", "1", f"N / {block_n}")
    else:
        names = (f"ceil(max_seq_len / {block_m}) * N / {block_n}", "1", "n_groups")
    # A compile-only run has no device to ask and never launches.
    if not env.compile.compile_only:
        limits = zip(grid, names, _max_grid_dims(device_index), (threads, 1, 1))
        for axis, (dim, name, max_dim, block) in zip("xyz", limits):
            limit = min(max_dim, _MAX_DISPATCH_ITEMS // block)
            if dim > limit:
                hint = ""
                if name == "n_groups":
                    hint = "; the skew path (uniform_seqlen=False) has no such limit"
                raise ValueError(
                    f"grid {axis} = {name} = {dim} is over the device limit "
                    f"{limit}{hint}"
                )
    workgroups = math.prod(grid)
    if workgroups > _INT32_MAX:
        raise ValueError(
            f"{workgroups} workgroups overflow the kernel's Int32 workgroup index "
            f"(max {_INT32_MAX})"
        )
    if n_groups * N > _INT32_MAX:
        raise ValueError(
            f"n_groups * N = {n_groups * N} overflows the kernel's Int32 BIAS "
            f"offset (max {_INT32_MAX})"
        )
    if L > _INT32_MAX:
        raise ValueError(f"L = {L} rows overflows the kernel's Int32 row index")
    if block_m * max(K, N) * 2 > _BUFFER_OFFSET_SPAN:
        raise ValueError(
            f"the {block_m} tile rows of A or C span {block_m} * max(K, N) * 2 = "
            f"{block_m * max(K, N) * 2} bytes, past the 4 GiB a 32-bit buffer "
            "offset reaches"
        )
    if compact:
        tile_rows, rows = block_m, f"{block_m}"
    else:
        tile_rows = min(block_m, max_seq_len)
        rows = f"min({block_m}, max_seq_len)"
    if tile_rows * max(K, N) * 2 > _MAX_BUFFER_BYTES:
        raise ValueError(
            f"one tile of A or C is {rows} * max(K, N) * 2 = "
            f"{tile_rows * max(K, N) * 2} bytes, past the 4 GiB of a 32-bit "
            "buffer descriptor"
        )
    if K * N * 2 > _BUFFER_OFFSET_SPAN:
        raise ValueError(
            f"one group of B is K * N * 2 = {K * N * 2} bytes, past the 4 GiB "
            "a 32-bit buffer offset reaches"
        )


def check_config(
    *,
    K: int,
    N: int,
    block_m: int,
    block_n: int,
    block_k: int,
    threads: int,
    xcd_c: int,
    xcd_w: int,
    n_xcd: int,
    traits: ArchTraits,
) -> None:
    """Raise ValueError for a config the kernel cannot run exactly."""
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError(f"block_m must be a power of two, got {block_m}")
    # No partial-tile path in N: N_BLOCKS floors, so a non-divisor block_n
    # would silently drop the trailing N % block_n output columns.
    if block_n <= 0 or N % block_n != 0:
        raise ValueError(f"block_n={block_n} must be a positive divisor of N={N}")
    if block_n % 32:
        raise ValueError(f"block_n must be a multiple of 32, got {block_n}")
    if block_k <= 0 or block_k % traits.k_tile_gran != 0:
        raise ValueError(
            f"block_k must be a positive multiple of {traits.k_tile_gran}, "
            f"got {block_k}"
        )
    if traits.a_lds_dma and block_k & (block_k - 1):
        raise ValueError(f"A LDS DMA needs a power-of-two block_k, got {block_k}")
    if block_k < traits.min_block_k:
        mma = "x".join(map(str, traits.mma_mnk))
        raise ValueError(
            f"block_k must be >= {traits.min_block_k} for MFMA {mma}, got {block_k}"
        )
    # n_tiles = K // block_k floors, silently dropping the K tail.
    if K % block_k != 0:
        raise ValueError(f"block_k={block_k} must divide K={K}")
    a_smem = block_m * block_k * STAGES_A * 2
    epi_smem = max(a_smem, block_m * block_n * 2)
    if epi_smem > traits.lds_bytes:
        raise ValueError(
            f"block_k={block_k} needs {epi_smem} bytes LDS "
            f"(max {traits.lds_bytes}); "
            f"A-stage staging is {a_smem} bytes"
        )
    if threads <= 0 or threads % traits.wave_size:
        raise ValueError(
            f"threads must be a positive multiple of {traits.wave_size}, got {threads}"
        )
    mma_m, mma_n = traits.mma_mnk[:2]
    waves = threads // traits.wave_size
    n_warps = min(waves, block_n // (mma_n * 2))
    m_warps = waves // n_warps
    if waves % n_warps or block_n % (mma_n * n_warps) or block_m % (mma_m * m_warps):
        raise ValueError(
            f"threads={threads} splits into {m_warps}x{n_warps} MFMA waves, "
            f"which do not tile block_m={block_m} x block_n={block_n}"
        )
    for operand, cols in (("A copy block_k", block_k), ("C copy block_n", block_n)):
        thr_cols = cols // _COPY_VEC
        if threads % thr_cols or block_m % (threads // thr_cols):
            raise ValueError(
                f"{operand}={cols} does not tile: {threads} threads in rows of "
                f"{thr_cols} must cover block_m={block_m} rows exactly"
            )
    if xcd_c < 1 or xcd_w < 1:
        raise ValueError(f"xcd_c and xcd_w must be >= 1, got {xcd_c} and {xcd_w}")
    if n_xcd * xcd_c > _INT32_MAX:
        raise ValueError(
            f"xcd_c={xcd_c} times {n_xcd} XCDs overflows the XCD remap's Int32 "
            f"period (max {_INT32_MAX})"
        )
    n_blocks = N // block_n
    if xcd_w * n_blocks > _INT32_MAX:
        raise ValueError(
            f"xcd_w={xcd_w} times {n_blocks} N-tiles overflows the XCD remap's "
            f"Int32 window (max {_INT32_MAX})"
        )


def jagged_dense_bmm(
    C,
    A,
    B,
    BIAS,
    SEQ_OFFSETS,
    n_groups: int,
    max_seq_len: int,
    stream: fx.Stream = _DEFAULT_STREAM,
    *,
    xcd_c: int,
    xcd_w: int,
    block_k: int,
    threads: int,
    uniform_seqlen: bool = True,
    block_m: int | None = None,
    block_n: int | None = None,
    traits: ArchTraits,
):
    if B.dim() != 3:
        raise ValueError(f"B must be 3-D (n_groups, K, N), got {B.dim()}-D")
    K, N = B.shape[1], B.shape[2]
    B = B.view(n_groups * K, N)
    bmn = BLOCK_M if block_m is None else int(block_m)
    bnn = BLOCK_N if block_n is None else int(block_n)
    block_k = int(block_k)
    nthreads = int(threads)
    compact = not uniform_seqlen
    if compact:
        # Slot-count bound from A.shape[0] >= SEQ_OFFSETS[-1]; reading
        # SEQ_OFFSETS[-1] would need a host sync.
        grid_m = A.shape[0] // bmn + n_groups
    else:
        grid_m = (max_seq_len + bmn - 1) // bmn

    with torch.cuda.device(A.device):
        n_xcd = get_num_xcds()
        check_config(
            K=K,
            N=N,
            block_m=bmn,
            block_n=bnn,
            block_k=block_k,
            threads=nthreads,
            xcd_c=xcd_c,
            xcd_w=xcd_w,
            n_xcd=n_xcd,
            traits=traits,
        )
        _check_launch(
            grid=_launch_grid(compact, grid_m, n_groups, N // bnn),
            threads=nthreads,
            block_m=bmn,
            block_n=bnn,
            n_groups=n_groups,
            max_seq_len=max_seq_len,
            L=C.shape[0],
            K=K,
            N=N,
            compact=compact,
            device_index=A.device.index,
        )
        search_steps = (n_groups - 1).bit_length()
        launch_args = (
            *map(_row_major, (C, A, B, BIAS, SEQ_OFFSETS)),
            C.shape[0],
            grid_m,
            n_groups,
            search_steps,
            stream,
        )
        key = (
            N,
            K,
            bmn,
            bnn,
            block_k,
            nthreads,
            xcd_c,
            xcd_w,
            compact,
            n_xcd,
            traits,
        )
        launch = _LAUNCHERS.get((A.device.index, key))
        if launch is None:
            launch = _LAUNCHERS[A.device.index, key] = _build_launcher(*key)
        _run_compiled(launch, *launch_args)
