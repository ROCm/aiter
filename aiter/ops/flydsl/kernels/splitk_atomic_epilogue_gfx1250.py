# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Fused atomic split-K epilogue shared by the gfx1250 a8w8 GEMM kernels."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

# rows of C pushed per unrolled batch of LDS reads
EPI_UNROLL = 16
# pk_add_bf16 is 32-bit: 2 elems/thread puts lane L at base + L*4, so one
# instruction covers exactly one fully-written 128 B line.
EPI_VEC = 2
SPIN_CLAIM = 4096  # polls before taking over a slot whose owner never arrived
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

    @functools.lru_cache(maxsize=8)
    def _group_emitter(binop, unroll, row_step):
        @flyc.jit
        def _f(gptrs, vecs, first_row, last_row):
            if last_row < mn_oob:  # whole group live: straight line
                for u in range_constexpr(unroll):
                    _emit_row(binop, gptrs[u], vecs[u])
            elif first_row < mn_oob:  # the one straddling group
                for u in range_constexpr(unroll):
                    _bounded_emitter(binop)(gptrs[u], vecs[u], first_row + u * row_step)
            # else: every row of this group is past mn_oob -- emit nothing

        return _f

    def _emit_rows(binop, row_base, bounded):
        n_iter = ch_rows // rows_per_iter
        unroll = min(EPI_UNROLL, n_iter)
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
            if const_expr(bounded):
                gptrs = [
                    fx.add_offset(gc_base, base_off + grp_delta[blk_i] + row_delta[u])
                    for u in range_constexpr(unroll)
                ]
                _group_emitter(binop, unroll, rows_per_iter)(
                    gptrs, vecs, rows[0], rows[-1]
                )
            else:
                for u in range_constexpr(unroll):
                    gptr = fx.add_offset(
                        gc_base, base_off + grp_delta[blk_i] + row_delta[u]
                    )
                    _emit_row(binop, gptr, vecs[u])

    @flyc.jit
    def _publish(cc):
        if tid == fx.Int32(0):
            # monotonic suffices: GL2 already orders the device-scope atomics.
            llvm_dialect.StoreOp(
                as_ir_value(fx.Int32(split_k - 1)),
                _flag_ptr(cc),
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

    def _load_flag(cc):
        return fx.Int32(
            llvm_dialect.LoadOp(
                T.i32,
                _flag_ptr(cc),
                alignment=4,
                ordering=llvm_dialect.AtomicOrdering.monotonic,
                syncscope="agent",
            ).result
        )

    my_token = fx.Int32(0) - split_idx - fx.Int32(1)

    @flyc.jit
    def _claim(cc):
        if tid == fx.Int32(0):
            llvm_dialect.AtomicCmpXchgOp(
                _flag_ptr(cc),
                as_ir_value(fx.Int32(0)),
                as_ir_value(my_token),
                llvm_dialect.AtomicOrdering.monotonic,
                llvm_dialect.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )

    @flyc.jit
    def _await(cc):
        cur = _load_flag(cc)
        n = fx.Int32(0)
        while (cur < fx.Int32(1)) & (cur != my_token):
            if n == fx.Int32(SPIN_CLAIM):
                _claim(cc)
            n = n + fx.Int32(1)
            cur = _load_flag(cc)
        return cur

    @functools.lru_cache(maxsize=4)
    def _rolled_emitter(binop, bounded):
        n_iter = ch_rows // rows_per_iter

        @flyc.jit
        def _f(row_base):
            for i in range(fx.Int32(0), fx.Int32(n_iter), fx.Int32(1)):
                row = row_base + r0 + i * fx.Int32(rows_per_iter)
                vec = fx.Vector(
                    fx.ptr_load(
                        fx.add_offset(lds_c, row * c_lds_row + cx),
                        result_type=T.vec(EPI_VEC, elem.ir_type),
                    )
                )
                gptr = fx.add_offset(
                    gc_base, c_off_rt + fx.Int64(row) * ldc64 + fx.Int64(cx)
                )
                if const_expr(bounded):
                    if row < mn_oob:
                        _emit_row(binop, gptr, vec)
                else:
                    _emit_row(binop, gptr, vec)

        return _f

    @functools.lru_cache(maxsize=2)
    def _contrib_emitter(bounded, hot_xchg):
        hot = (
            llvm_dialect.AtomicBinOp.xchg if hot_xchg else llvm_dialect.AtomicBinOp.fadd
        )
        cold = (
            llvm_dialect.AtomicBinOp.fadd if hot_xchg else llvm_dialect.AtomicBinOp.xchg
        )

        @flyc.jit
        def _f(cur, row_base, cc):
            if cur == my_token:
                if const_expr(hot_xchg):
                    _emit_rows(hot, row_base, bounded)
                else:
                    _rolled_emitter(cold, bounded)(row_base)
                rocdl.s_wait_storecnt(0)
                rocdl.s_barrier_signal(-1)
                rocdl.s_barrier_wait(-1)
                _publish(cc)
            else:
                if const_expr(hot_xchg):
                    _rolled_emitter(cold, bounded)(row_base)
                else:
                    _emit_rows(hot, row_base, bounded)

        return _f

    def _contribute(cc, row_base, hot_xchg):
        _contrib_emitter(bounded_m, hot_xchg)(_await(cc), row_base, cc)

    _claim(split_idx)
    _contribute(split_idx, split_idx * ch_rows, True)
    for c in range_constexpr(1, split_k):
        cc = (split_idx + fx.Int32(c)) & fx.Int32(split_k - 1)
        _contribute(cc, cc * ch_rows, False)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)
    for c in range_constexpr(1, split_k):
        _release((split_idx + fx.Int32(c)) & fx.Int32(split_k - 1))
