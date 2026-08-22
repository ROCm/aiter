# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-K reduce kernel for the FlyDSL HGEMM family.

Tile-agnostic companion to the split-K path in `splitk_hgemm.py` /
`small_m_hgemm.py`: the main kernel writes each split's **fp32** accumulator
into its own disjoint slice of a workspace laid out ``[SLOTS, m, N]``;
this kernel sums the slices, folds bias once, casts to the output dtype and
writes C exactly once.

Ordering between the two launches comes from the stream (a dependency edge
between two nodes inside a captured graph). Nothing is shared between blocks,
so there is no coordination state a CUDA-graph replay can poison.

Design mirrors `csrc/opus_gemm/include/gfx950/splitk_reduce_gfx950.cuh` (that
one is C++ with no python binding, so it is a reference, not a callable).

Layout contract (must match the main kernel):
    * The workspace is unpadded ``[SLOTS, m, N]``: slice ``s`` starts at element
      ``s * m * N``. The main kernel masks its stores to ``row < m``, because
      this kernel only ever reads rows ``[0, m)`` of a slot (grid.y covers m
      real rows), so writing a grid-aligned M_PAD tile would be pure write
      amplification -- 32x at m=1 with BLOCK_M=32.
    * ``SLOTS = SPLIT_K * BLOCK_K_WARPS`` for the generic hgemm family (each
      slice-K warp group gets its own slot so the slice combine is also fp32),
      ``SLOTS = SPLIT_K`` for small_m (no slice-K there).

Grid: ``(ceil(N / (VEC * THREADS)), m, 1)``; each thread owns ``VEC`` fp32 lanes
along N (one ``buffer_load_dwordx4`` per slot for ``VEC=4``).
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import T

from .tensor_shim import GTensor, get_dtype_in_kernel

__all__ = [
    "REDUCE_VEC",
    "compile_splitk_reduce_kernel",
    "reduce_block_threads",
]

# fp32 lanes per thread. 4 == one buffer_load_dwordx4 per split slot, and the
# bf16 store is a single dwordx2. Every supported N is a multiple of 16 (tile_n
# is a multiple of 16 and N % tile_n == 0), so VEC=4 never produces a partial
# vector tail.
REDUCE_VEC = 4
REDUCE_MIN_THREADS = 64
REDUCE_MAX_THREADS = 256


def reduce_block_threads(n: int, vec: int = REDUCE_VEC) -> int:
    """Block size for the reduce grid: enough waves to cover N, capped."""
    lanes = max(1, (n + vec - 1) // vec)
    threads = ((lanes + 63) // 64) * 64
    return max(REDUCE_MIN_THREADS, min(REDUCE_MAX_THREADS, threads))


@functools.lru_cache(maxsize=4096)
def compile_splitk_reduce_kernel(
    dtype: str,
    n: int,
    SLOTS: int,
    HAS_BIAS: bool = False,
    VEC: int = REDUCE_VEC,
    THREADS: int | None = None,
):
    if SLOTS < 2:
        raise ValueError(f"split-K reduce needs SLOTS >= 2, got {SLOTS}")
    if n % VEC != 0:
        raise ValueError(f"split-K reduce needs n % {VEC} == 0, got n={n}")
    if THREADS is None:
        THREADS = reduce_block_threads(n, VEC)

    TILE_N = VEC * THREADS
    N_TILES = (n + TILE_N - 1) // TILE_N

    KERNEL_NAME = (
        f"splitk_reduce_{dtype}_n{n}_s{SLOTS}_v{VEC}x{THREADS}"
        f"{'_BIAS' if HAS_BIAS else ''}"
    )

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def splitk_reduce_kernel(
        C: fx.Pointer,
        WS: fx.Pointer,
        BIAS: fx.Pointer,
        m: fx.Int32,
    ):
        dtype_ = get_dtype_in_kernel(dtype)
        f32_vec_t = T.vec(VEC, T.f32)
        out_vec_t = T.vec(VEC, dtype_)

        C_ = GTensor(C, dtype=dtype_, shape=(-1, n))
        WS_ = GTensor(WS, dtype=T.f32, shape=(-1, n))
        if const_expr(HAS_BIAS):
            BIAS_ = GTensor(BIAS, dtype=dtype_, shape=(n,))

        tid = fx.Index(fx.thread_idx.x)
        row = fx.Index(fx.block_idx.y)
        n_base = fx.Index(fx.block_idx.x) * TILE_N + tid * VEC

        # Slot stride: the main kernel writes an unpadded [SLOTS, m, N]
        # workspace, masking its stores to row < m.
        m_rows = fx.Index(m)

        in_range = arith.cmpi(arith.CmpIPredicate.ult, n_base, fx.Index(n))
        in_range_if = scf.IfOp(in_range, results_=[], has_else=False)
        with ir.InsertionPoint(in_range_if.then_block):
            acc = WS_.vec_load((row, n_base), VEC)
            # Fixed summation order -> bit-reproducible across runs.
            for s in range_constexpr(1, SLOTS):
                part = WS_.vec_load((row + m_rows * s, n_base), VEC)
                acc = arith.addf(acc, part)
            if const_expr(HAS_BIAS):
                bias_vec = BIAS_.vec_load((n_base,), VEC)
                acc = arith.addf(acc, arith.extf(f32_vec_t, bias_vec))
            C_.vec_store((row, n_base), arith.truncf(out_vec_t, acc), VEC)
            scf.YieldOp([])

    @flyc.jit
    def launch_splitk_reduce_kernel(
        C: fx.Pointer,
        WS: fx.Pointer,
        BIAS: fx.Pointer,
        m: fx.Int32,
        stream: fx.Stream,
    ):
        splitk_reduce_kernel._func.__name__ = KERNEL_NAME
        splitk_reduce_kernel(C, WS, BIAS, m).launch(
            grid=(N_TILES, m, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch_splitk_reduce_kernel
