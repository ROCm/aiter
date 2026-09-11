# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Fused atomic split-K epilogue shared by the gfx1250 a8w8 GEMM kernels."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from .gemm_common_gfx1250 import workgroup_barrier

# rows of C pushed per unrolled batch of LDS reads
EPI_UNROLL = 16
# pk_add_bf16 is 32-bit: 2 elems/thread puts lane L at base + L*4, so one
# instruction covers exactly one fully-written 128 B line.
EPI_VEC = 2
FLAG_STRIDE_I32 = 32


def emit_atomic_splitk_epilogue(
    *,
    elem,
    tid,
    block,
    tile_m,
    tile_n,
    c_lds_row,
    lds_base_ptr,
    gc_base,
    c_off_rt,
    ldc64,
    split_k,
    split_idx,
    mn_oob,
    bounded_m,
    flat_tile,
    arg_flag,
):
    """Accumulate the LDS-staged C tile into C with device-scope atomics.

    Chunked ownership: split j owns chunk j, establishes it with atomic_swap (so
    C is never zeroed) and publishes flag[tile][j], then accumulates into chunks
    j+1, j+2, ...  Every split publishes its own chunk at the same moment, so the
    serialised prefix is one chunk rather than a whole tile.

    ``bounded_m`` says M is not a whole multiple of ``tile_m``, so the last tile
    is partial and rows past ``mn_oob`` must be skipped, as the TDM descriptor
    bound did for the store this replaces.  It is a compile-time flag so the
    aligned case emits a single unguarded body.
    """
    lanes_per_row = tile_n // EPI_VEC
    rows_per_iter = block // lanes_per_row
    lds_c = fx.recast_iter(elem, lds_base_ptr)
    r0 = fx.Int32(tid) // lanes_per_row
    cx = (fx.Int32(tid) % lanes_per_row) * EPI_VEC
    ch_rows = tile_m // split_k
    fp = fx.recast_iter(fx.PointerType.get(T.i32, arg_flag.address_space), arg_flag)
    fbase = flat_tile * split_k * FLAG_STRIDE_I32

    def _flag_ptr(idx):
        return fx.to_llvm_ptr(fx.add_offset(fp, fbase + idx * FLAG_STRIDE_I32))

    def _emit_row(binop, gptr, vec):
        for pi in range_constexpr(EPI_VEC // 2):
            pair = fx.Vector.from_elements([vec[pi * 2], vec[pi * 2 + 1]], elem)
            # xchg has no <2 x bf16> form: swap as i32.
            val = (
                pair.bitcast(fx.Int32)[0]
                if const_expr(binop == llvm_dialect.AtomicBinOp.xchg)
                else pair
            )
            # lowers to global_atomic_pk_add_bf16 / _swap_b32 SCOPE_DEV
            # (no-return), executed inside GL2: no writeback handshake.
            llvm_dialect.atomicrmw(
                binop,
                fx.to_llvm_ptr(fx.add_offset(gptr, pi * 2)),
                as_ir_value(val),
                llvm_dialect.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )

    @functools.lru_cache(maxsize=4)
    def _bounded_emitter(binop):
        """Close over ``binop`` -- passing it as a jit arg makes it a runtime
        value on a later trace and breaks the const_expr test in _emit_row."""

        @flyc.jit
        def _f(gptr, vec, row):
            if row < mn_oob:
                _emit_row(binop, gptr, vec)

        return _f

    def _emit_rows(binop, row_base, bounded):
        n_iter = ch_rows // rows_per_iter
        unroll = min(EPI_UNROLL, n_iter)
        # Row offsets are uniform, so the compiler would strength-reduce the
        # per-row addresses into a serial s_add_nc_u64 chain (41% of the epilogue
        # on nb4).  Precomputed independent deltas avoid it.
        row_delta = [
            fx.Int64(u * rows_per_iter) * ldc64 for u in range_constexpr(unroll)
        ]
        grp_delta = [
            fx.Int64(g * unroll * rows_per_iter) * ldc64
            for g in range_constexpr(n_iter // unroll)
        ]
        base_off = c_off_rt + fx.Int64(row_base + r0) * ldc64 + fx.Int64(cx)
        for blk_i in range_constexpr(n_iter // unroll):
            rows = [
                row_base + (r0 + (blk_i * unroll + u) * rows_per_iter)
                for u in range_constexpr(unroll)
            ]
            vecs = [
                fx.Vector(
                    fx.ptr_load(
                        fx.add_offset(lds_c, rows[u] * c_lds_row + cx),
                        result_type=T.vec(EPI_VEC, elem.ir_type),
                    )
                )
                for u in range_constexpr(unroll)
            ]
            for u in range_constexpr(unroll):
                gptr = fx.add_offset(
                    gc_base, base_off + grp_delta[blk_i] + row_delta[u]
                )
                if const_expr(bounded):
                    _bounded_emitter(binop)(gptr, vecs[u], rows[u])
                else:
                    _emit_row(binop, gptr, vecs[u])

    @flyc.jit
    def _publish():
        if tid == fx.Int32(0):
            # monotonic suffices: GL2 already orders the device-scope atomics.
            llvm_dialect.StoreOp(
                as_ir_value(fx.Int32(split_k - 1)),
                _flag_ptr(split_idx),
                alignment=4,
                ordering=llvm_dialect.AtomicOrdering.monotonic,
                syncscope="agent",
            )

    @flyc.jit
    def _release(cc):
        """Count this WG off the peer's slot; the last one leaves it at 0."""
        if tid == fx.Int32(0):
            llvm_dialect.atomicrmw(
                llvm_dialect.AtomicBinOp.add,
                _flag_ptr(cc),
                as_ir_value(fx.Int32(-1)),
                llvm_dialect.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )

    @flyc.jit
    def _spin(cc):
        def _load():
            return fx.Int32(
                llvm_dialect.LoadOp(
                    T.i32,
                    _flag_ptr(cc),
                    alignment=4,
                    ordering=llvm_dialect.AtomicOrdering.monotonic,
                    syncscope="agent",
                ).result
            )

        cur = _load()
        while cur == fx.Int32(0):
            cur = _load()

    _emit_rows(llvm_dialect.AtomicBinOp.xchg, split_idx * ch_rows, bounded_m)
    workgroup_barrier(use_cluster=False)
    _publish()
    for c in range_constexpr(1, split_k):
        cc = (split_idx + fx.Int32(c)) & fx.Int32(split_k - 1)
        # every thread polls its own copy: a broadcast read of one line, cheaper
        # than tid0 polling behind a workgroup barrier.
        _spin(cc)
        _emit_rows(llvm_dialect.AtomicBinOp.fadd, cc * ch_rows, bounded_m)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)
    for c in range_constexpr(1, split_k):
        _release((split_idx + fx.Int32(c)) & fx.Int32(split_k - 1))
