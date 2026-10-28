# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernel for jagged_dense_bmm_broadcast_add (grouped BF16 GEMM + bias).

For each group ``b`` over ``[seq_offsets[b], seq_offsets[b+1])``::

    Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b]

Dense weight is host-pre-transposed to ``(B * N, K)``. Supports a uniform
grid launch with optional XCD remap, and a compact ``TILE_MAP`` launch for
skewed sequence lengths.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import (
    get_gfx_runtime,
    get_lds_capacity_bytes,
    get_num_xcds,
)
from aiter.ops.flydsl.kernels import buffer_ops

from .tensor_shim import _run_compiled

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
STAGES_A = 2
B_STAGES = 3
THREADS = 256

_DEFAULT_STREAM = fx.Stream(None)

# K16 bf16 atom. The permutation covers two K16 steps behind one 128-bit load,
# so the tiled GEMM advances 32 K per step.
_K16_PERM_SHAPE = (4, 4, 2)
_K16_PERM_STRIDE = (1, 8, 4)
_K16_STEP = 32
_K16_SWIZZLE = (3, 3, 3)
_K16_MIN_BLOCK_K = 64

# CDNA4's K32 atom consumes one 32-wide step. Its flat K mode needs direct
# per-step slicing, unlike the nested K mode of the K16 record.
_K32_PERM_SHAPE = (8, 4)
_K32_PERM_STRIDE = (1, 8)
_K32_STEP = 32
_K32_SWIZZLE = (3, 3, 3)
_K32_MIN_BLOCK_K = 64


@dataclass(frozen=True)
class JdbbaArchTraits:
    """Compile-time MFMA, K permutation, LDS swizzle, and LDS budget."""

    arch: str
    mfma_k: int
    k_perm_shape: tuple[int, ...]
    k_perm_stride: tuple[int, ...]
    k_step: int
    k_slice_nested: bool
    swizzle: tuple[int, int, int]
    lds_bytes: int
    min_block_k: int

    @property
    def key(self) -> tuple:
        return (
            self.arch,
            self.mfma_k,
            self.k_perm_shape,
            self.k_perm_stride,
            self.k_step,
            self.k_slice_nested,
            self.swizzle,
            self.lds_bytes,
            self.min_block_k,
        )


def _runtime_arch() -> str:
    return get_gfx_runtime().split(":", 1)[0]


def jdbba_arch_traits(arch: str) -> JdbbaArchTraits:
    """Traits for a dispatch arch. The LDS budget comes from ``chip_info``."""
    arch = arch.split(":", 1)[0]
    if arch not in ("gfx942", "gfx950"):
        raise ValueError(f"jagged_dense_bmm has no arch traits for {arch}")
    if arch == "gfx950":
        return JdbbaArchTraits(
            arch=arch,
            mfma_k=32,
            k_perm_shape=_K32_PERM_SHAPE,
            k_perm_stride=_K32_PERM_STRIDE,
            k_step=_K32_STEP,
            k_slice_nested=False,
            swizzle=_K32_SWIZZLE,
            lds_bytes=get_lds_capacity_bytes(arch),
            min_block_k=_K32_MIN_BLOCK_K,
        )
    return JdbbaArchTraits(
        arch=arch,
        mfma_k=16,
        k_perm_shape=_K16_PERM_SHAPE,
        k_perm_stride=_K16_PERM_STRIDE,
        k_step=_K16_STEP,
        k_slice_nested=True,
        swizzle=_K16_SWIZZLE,
        lds_bytes=get_lds_capacity_bytes(arch),
        min_block_k=_K16_MIN_BLOCK_K,
    )


def smem_bytes(block_m: int, block_n: int, block_k: int) -> int:
    """LDS bytes for the double-buffered A tile and the epilogue C tile."""
    a_smem = block_m * block_k * STAGES_A * 2
    return max(a_smem, block_m * block_n * 2)


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

    if isinstance(num_rows, int):
        total = num_rows * num_cols  # compile-time
        period = nXCD * C  # compile-time
        prefix = total - (total % period)  # largest [0,prefix) divisible by period
        if fx.const_expr(prefix == total):
            xy_g = xy_g_remap
        else:
            xy_g = (xy < fx.Int32(prefix)).select(xy_g_remap, xy)
    else:
        total = c_num_rows * c_num_cols  # runtime: total launched blocks
        period = fx.Int32(nXCD * C)  # compile-time product
        prefix = total - (total % period)  # runtime largest multiple of period <= total
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


# Per-call sizes (L, n_groups, max_seq_len) are launch-grid dimensions, not
# part of this key, so a new length reuses the compiled launcher.
_LAUNCHER_CACHE_MAX = 64
_LAUNCHER_CACHE: OrderedDict = OrderedDict()


def _build_launcher(
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    THREADS,
    XCD_C,
    XCD_W,
    NUM_XCDS,
    traits: JdbbaArchTraits,
    COMPACT=False,
):
    mfma_k = traits.mfma_k
    k_step = traits.k_step
    k_perm_shape = traits.k_perm_shape
    k_perm_stride = traits.k_perm_stride
    k_slice_nested = traits.k_slice_nested
    swizzle = traits.swizzle
    N_BLOCKS = N // BLOCK_N  # output column-tiles per group (compile-time)
    a_smem = BLOCK_M * BLOCK_K * STAGES_A * 2  # bf16 A double-buffer staging

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def jdbba_kernel(
        C: fx.Tensor,  # out    (L, N)   bf16
        A: fx.Tensor,  # jagged (L, K)   bf16
        B: fx.Tensor,  # dense  (B_groups * N, K) bf16 (pre-transposed, tall)
        BIAS: fx.Tensor,  # bias   (B_groups * N,)   bf16
        SEQ_OFFSETS: fx.Tensor,  # (B_groups + 1,) int32
        TILE_MAP: fx.Tensor,  # compact: (total_occ_tiles, 4) int32 [off_b, m_idx, seq_start, seq_end]; else dummy
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s_A: fx.TiledCopy,
        tiled_copy_s2g_C: fx.TiledCopy,
    ):
        tid = fx.thread_idx.x
        pid_mn, _, raw_b = fx.block_idx

        if fx.const_expr(COMPACT):
            # Scalar int32 record. The offset is an element index, so this stays
            # on buffer_ops.
            tile_rsrc = buffer_ops.create_buffer_resource(TILE_MAP, max_size=True)
            # One XCD has nothing to spread across, so the grid coordinate is
            # the tile. A larger count remaps with that count, not a fixed 8.
            if fx.const_expr(NUM_XCDS > 1 and XCD_C > 1):
                total_occ = fx.grid_dim.x
                raw_xy = raw_b * total_occ + pid_mn
                tile_row, n_col = _xcd_remap(
                    raw_xy, total_occ, N_BLOCKS, XCD_C, XCD_W, NUM_XCDS
                )
                tile_row = fx.Int32(fx.rocdl.readfirstlane(T.i32, tile_row))
                block_n_idx = fx.Int32(fx.rocdl.readfirstlane(T.i32, n_col))
            else:
                tile_row = pid_mn
                block_n_idx = raw_b
            rec = fx.Vector(
                buffer_ops.buffer_load(
                    tile_rsrc, tile_row * fx.Int32(4), vec_width=4, is_scalar=True
                )
            )
            off_b = fx.Int32(rec[0])
            block_m_idx = fx.Int32(rec[1])
            seq_start = fx.Int32(rec[2])
            seq_end = fx.Int32(rec[3])
        else:
            # Grid is (bm_tiles * N_BLOCKS, 1, n_groups). Recover both sizes
            # here so a new batch or max_seq_len does not rebuild the launcher.
            grid_x = fx.grid_dim.x
            bm_tiles = fx.Int32(fx.Uint32(grid_x) // fx.Uint32(N_BLOCKS))
            if fx.const_expr(NUM_XCDS > 1 and XCD_C > 1):
                n_groups_rt = fx.grid_dim.z
                raw_xy = raw_b * grid_x + pid_mn
                row, col = _xcd_remap(
                    raw_xy, bm_tiles * n_groups_rt, N_BLOCKS, XCD_C, XCD_W, NUM_XCDS
                )
                row = fx.Int32(fx.rocdl.readfirstlane(T.i32, row))
                col = fx.Int32(fx.rocdl.readfirstlane(T.i32, col))
                off_b = fx.Int32(fx.Uint32(row) // fx.Uint32(bm_tiles))
                block_m_idx = fx.Int32(fx.Uint32(row) % fx.Uint32(bm_tiles))
                block_n_idx = col
            else:
                off_b = raw_b
                block_m_idx = fx.Int32(fx.Uint32(pid_mn) // fx.Uint32(N_BLOCKS))
                block_n_idx = fx.Int32(fx.Uint32(pid_mn) % fx.Uint32(N_BLOCKS))

            # Scalar int32 metadata. The offset is an element index, so this stays
            # on buffer_ops.
            seq_rsrc = buffer_ops.create_buffer_resource(SEQ_OFFSETS, max_size=True)
            seq_start = fx.Int32(
                buffer_ops.buffer_load(seq_rsrc, off_b, vec_width=1, is_scalar=True)
            )
            seq_end = fx.Int32(
                buffer_ops.buffer_load(seq_rsrc, off_b + 1, vec_width=1, is_scalar=True)
            )
        M_b = seq_end - seq_start
        start_m = block_m_idx * BLOCK_M

        if start_m < M_b:
            a_row_off = fx.Int64(seq_start) * fx.Int64(K)
            c_row_off = fx.Int64(seq_start) * fx.Int64(N)
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
                fx.get_layout(B),
            )

            A_buf = fx.rocdl.make_buffer_tensor(
                A_g,
                num_records_bytes=fx.Int64(M_b) * fx.Int64(K) * fx.Int64(2),
            )
            B_buf = fx.rocdl.make_buffer_tensor(B_g, max_size=True)
            C_buf = fx.rocdl.make_buffer_tensor(
                C_g,
                num_records_bytes=fx.Int64(M_b) * fx.Int64(N) * fx.Int64(2),
            )

            gA_k = fx.flat_divide(A_buf, (BLOCK_M, BLOCK_K))[
                None, None, block_m_idx, None
            ]
            gB_k = fx.flat_divide(B_buf, (BLOCK_N, BLOCK_K))[
                None, None, block_n_idx, None
            ]
            gC = fx.flat_divide(C_buf, (BLOCK_M, BLOCK_N))[
                None, None, block_m_idx, block_n_idx
            ]

            bias_elem_off = off_b * N
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

            thr_copy_s2r_A = fx.make_tiled_copy_A(uni_copy_128b, tiled_mma).get_slice(
                tid
            )
            thr_copy_g2r_B = fx.make_tiled_copy_B(
                buffer_copy_128b, tiled_mma
            ).get_slice(tid)

            composed_layout_A = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(swizzle[0], swizzle[1], swizzle[2])),
                fx.make_ordered_layout((BLOCK_M, BLOCK_K, STAGES_A), (1, 0, 2)),
            )
            sA = fx.make_view(fx.get_dyn_shared(fx.BFloat16), composed_layout_A)

            thr_gA_k = thr_copy_g2s_A.partition_S(gA_k)
            thr_sA = thr_copy_g2s_A.partition_D(sA)
            thr_sA_s2r = thr_copy_s2r_A.partition_S(sA)
            thr_gB_k = thr_copy_g2r_B.partition_S(gB_k)

            copy_frag_A = fx.make_fragment_like(thr_sA[None, None, None, 0])

            mma_frag_A = thr_mma.make_fragment_A(sA[None, None, 0])
            mma_frag_B = thr_mma.make_fragment_B(gB_k, stages=B_STAGES)
            mma_frag_C = thr_mma.make_fragment_C(gC)

            mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)
            mma_frag_B_retile = thr_copy_g2r_B.retile(mma_frag_B)

            gA_k_stride = fx.get_scalar(gA_k.stride[2])
            gB_k_stride = fx.get_scalar(gB_k.stride[2])

            def run_pipeline_stage(
                a_read_stage, b_read_slot, a_next_k=None, b_next_k=None
            ):
                a_write_stage = a_read_stage ^ 1
                if fx.const_expr(a_next_k is not None):
                    fx.copy(
                        buffer_copy_128b,
                        thr_gA_k[None, None, None, 0],
                        copy_frag_A,
                        soffset=fx.Int32(a_next_k) * gA_k_stride,
                    )
                if fx.const_expr(b_next_k is not None):
                    b_fetch_slot = (b_read_slot + (B_STAGES - 1)) % B_STAGES
                    fx.copy(
                        buffer_copy_128b,
                        thr_gB_k[None, None, None, 0],
                        mma_frag_B_retile[None, None, None, b_fetch_slot],
                        soffset=fx.Int32(b_next_k) * gB_k_stride,
                    )

                for block_k_iter in fx.range_constexpr(BLOCK_K // k_step):
                    fx.copy(
                        uni_copy_128b,
                        thr_sA_s2r[None, None, block_k_iter, a_read_stage],
                        mma_frag_A_retile[None, None, block_k_iter],
                    )
                    if fx.const_expr(k_slice_nested):
                        frag_A_k = mma_frag_A[None, None, (None, block_k_iter)]
                        frag_B_k = mma_frag_B[
                            None, None, (None, block_k_iter), b_read_slot
                        ]
                    else:
                        frag_A_k = mma_frag_A[None, None, block_k_iter]
                        frag_B_k = mma_frag_B[None, None, block_k_iter, b_read_slot]
                    fx.gemm(
                        tiled_mma,
                        mma_frag_C,
                        frag_A_k,
                        frag_B_k,
                        mma_frag_C,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )

                if fx.const_expr(a_next_k is not None):
                    fx.copy(
                        uni_copy_128b,
                        copy_frag_A,
                        thr_sA[None, None, None, a_write_stage],
                    )
                fx.gpu.barrier()

            n_tiles = K // BLOCK_K
            fx.copy(buffer_copy_128b, thr_gA_k[None, None, None, 0], copy_frag_A)
            fx.copy(
                buffer_copy_128b,
                thr_gB_k[None, None, None, 0],
                mma_frag_B_retile[None, None, None, 0],
            )
            if fx.const_expr(n_tiles > 1):
                fx.copy(
                    buffer_copy_128b,
                    thr_gB_k[None, None, None, 0],
                    mma_frag_B_retile[None, None, None, 1],
                    soffset=fx.Int32(1) * gB_k_stride,
                )
            mma_frag_C.fill(0)
            fx.copy(uni_copy_128b, copy_frag_A, thr_sA[None, None, None, 0])
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

            mma_frag_C_bf16 = fx.make_fragment_like(mma_frag_C, fx.BFloat16.ir_type)
            thr_copy_r2s_C = fx.make_tiled_copy_C(
                fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16), tiled_mma
            ).get_slice(tid)
            mma_frag_C_retile = thr_copy_r2s_C.retile(mma_frag_C_bf16)

            thr_gBias = thr_copy_r2s_C.partition_S(gBias)
            bias_frag = fx.make_fragment_like(thr_gBias)
            fx.copy(
                fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16),
                thr_gBias,
                bias_frag,
            )
            acc = mma_frag_C.load()
            # Same element count, different logical shape: relabel the bias
            # fragment so the add stays a flat vector op instead of broadcasting.
            bias = fx.Vector(bias_frag.load(), acc.shape, fx.BFloat16)
            mma_frag_C_bf16.store((acc + bias).to(fx.BFloat16))

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
        TILE_MAP: fx.Tensor,
        total_occ_tiles: int,
        n_groups: int,
        max_seq_len: int,
        stream: fx.Stream = _DEFAULT_STREAM,
    ):
        n_warps_total = THREADS // 64
        n_warps = min(n_warps_total, BLOCK_N // 16)
        m_warps = n_warps_total // n_warps
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(16, 16, mfma_k, fx.BFloat16)),
            fx.make_layout((m_warps, n_warps, 1), (1, m_warps, 0)),
            fx.make_tile(None, None, fx.make_layout(k_perm_shape, k_perm_stride)),
        )
        val_per_thr = 8  # 16B / bf16
        thrs_col = BLOCK_K // val_per_thr
        thrs_row = THREADS // thrs_col
        tiled_copy_g2s_A = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
            fx.make_layout(
                ((thrs_col, thrs_row), (1, val_per_thr)),
                ((thrs_row * val_per_thr, 1), (1, thrs_row)),
            ),
            fx.make_tile(thrs_row, BLOCK_K),
        )

        c_val = 8
        c_thrs_n = BLOCK_N // c_val  # threads along N
        c_thrs_m = THREADS // c_thrs_n  # threads along M
        tiled_copy_s2g_C = fx.make_tiled_copy(
            fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16),
            fx.make_layout(
                ((c_thrs_n, c_thrs_m), (1, c_val)),
                ((c_thrs_m * c_val, 1), (1, c_thrs_m)),
            ),
            fx.make_tile(c_thrs_m, BLOCK_N),
        )

        epi_smem_bytes = max(a_smem, BLOCK_M * BLOCK_N * 2)

        bm = (max_seq_len + BLOCK_M - 1) // BLOCK_M
        if fx.const_expr(COMPACT):
            grid = (total_occ_tiles, 1, N_BLOCKS)
        else:
            grid = (bm * N_BLOCKS, 1, n_groups)
        jdbba_kernel(
            C,
            A,
            B,
            BIAS,
            SEQ_OFFSETS,
            TILE_MAP,
            tiled_mma,
            tiled_copy_g2s_A,
            tiled_copy_s2g_C,
        ).launch(grid=grid, block=(THREADS, 1, 1), smem=epi_smem_bytes, stream=stream)

    return _launch


def _cache_get(key):
    launch = _LAUNCHER_CACHE.get(key)
    if launch is not None:
        _LAUNCHER_CACHE.move_to_end(key)
    return launch


def _cache_put(key, launch):
    _LAUNCHER_CACHE[key] = launch
    _LAUNCHER_CACHE.move_to_end(key)
    while len(_LAUNCHER_CACHE) > _LAUNCHER_CACHE_MAX:
        _LAUNCHER_CACHE.popitem(last=False)


def jagged_dense_bmm(
    C,
    A,
    B,
    BIAS,
    SEQ_OFFSETS,
    n_groups: int,
    max_seq_len: int,
    stream: fx.Stream = _DEFAULT_STREAM,
    xcd_c: int | None = None,
    xcd_w: int | None = None,
    num_xcds: int | None = None,
    uniform_seqlen: bool = True,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    threads: int | None = None,
    tile_map=None,
    total_occ_tiles: int | None = None,
):
    N = B.shape[0] // n_groups
    K = B.shape[1]
    bmn = BLOCK_M if block_m is None else int(block_m)
    bnn = BLOCK_N if block_n is None else int(block_n)
    if bmn <= 0:
        raise ValueError(f"block_m must be positive, got {bmn}")
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    # No partial-tile path in N: N_BLOCKS floors, and the uniform kernel
    # divides the grid by that count, so a non-divisor block_n would drop
    # the trailing N % block_n output columns.
    if bnn <= 0 or N % bnn != 0:
        raise ValueError(f"block_n={bnn} must be a positive divisor of N={N}")
    traits = jdbba_arch_traits(_runtime_arch())
    if block_k is None:
        block_k = BLOCK_K
    else:
        block_k = int(block_k)
        if block_k <= 0 or block_k % traits.k_step != 0:
            raise ValueError(
                f"block_k must be a positive multiple of {traits.k_step}, got {block_k}"
            )
        if block_k < traits.min_block_k:
            raise ValueError(
                f"block_k must be >= {traits.min_block_k} for MFMA 16x16x{traits.mfma_k}, "
                f"got {block_k}"
            )
    # These depend on block_m / block_n too, so they run for the default block_k
    # as well: n_tiles = K // block_k floors, silently dropping the K tail.
    if K % block_k != 0:
        raise ValueError(f"block_k={block_k} must divide K={K}")
    epi_smem = smem_bytes(bmn, bnn, block_k)
    if epi_smem > traits.lds_bytes:
        raise ValueError(
            f"block_k={block_k} needs {epi_smem} bytes LDS "
            f"(max {traits.lds_bytes} on {traits.arch})"
        )
    if num_xcds is None:
        num_xcds = int(get_num_xcds())
    else:
        num_xcds = int(num_xcds)
    if num_xcds < 1:
        raise ValueError(f"num_xcds must be positive, got {num_xcds}")
    if xcd_c is None or xcd_w is None:
        # One owner for the analytic XCD choice. Imported here so this module
        # can load without the dispatch table.
        from aiter.ops.flydsl.jagged_dense_bmm_dispatch import analytic_config

        filled = analytic_config(
            uniform_seqlen=uniform_seqlen,
            n_groups=n_groups,
            reduction_k=K,
            num_xcds=num_xcds,
        )
        if xcd_c is None:
            xcd_c = filled["xcd_c"]
        if xcd_w is None:
            xcd_w = filled["xcd_w"]
    nthreads = THREADS if threads is None else threads
    compact = tile_map is not None

    tmap = SEQ_OFFSETS if tile_map is None else tile_map
    tot = 0 if total_occ_tiles is None else int(total_occ_tiles)

    # n_groups, max_seq_len, and tot stay launch arguments. They size the grid
    # inside _launch and must not specialize the kernel.
    key = (
        N,
        K,
        bmn,
        bnn,
        block_k,
        nthreads,
        xcd_c,
        xcd_w,
        num_xcds,
        traits.key,
        compact,
    )
    launch_args = (C, A, B, BIAS, SEQ_OFFSETS, tmap, tot, n_groups, max_seq_len, stream)

    launch = _cache_get(key)
    if launch is None:
        launch = _build_launcher(
            N,
            K,
            bmn,
            bnn,
            block_k,
            nthreads,
            xcd_c,
            xcd_w,
            num_xcds,
            traits,
            compact,
        )
        _cache_put(key, launch)
    _run_compiled(launch, *launch_args)
