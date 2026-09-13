# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Scaled MFMA, scale loading and epilogues for MXFP8 MoE."""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from .mxfp4_gemm_common import _swiglu_mul_batch


def make_scale_preshuffled_s2r(scale_arg, rows, K, n_tiles):
    class ScalePreshuffledS2R:
        """Coalesced reader for ``shuffle_scale_w4``-packed E8M0 -- no LDS staging."""

        def __init__(self):
            assert n_tiles % 2 == 0, "shuffle_scale_w4 pairs tiles two at a time"
            self.n_pairs = n_tiles // 2
            self.k1_stride = K // 256  # i32 groups of 64 per 32-row super-row
            self.lane = fx.thread_idx.x % 64
            # Same byte count as the raw layout, just permuted.
            t_i8 = fx.rocdl.make_buffer_tensor(
                scale_arg,
                max_size=False,
                num_records_bytes=fx.Int64(rows) * fx.Int64(K // 32),
            )
            i32_ptr = fx.PointerType.get(
                elem_ty=fx.Int32.ir_type,
                address_space=fx.rocdl.TargetAddressSpace.BufferDesc,
                alignment=4,
            )
            iter_i32 = fx.recast_iter(i32_ptr, fx.get_iter(t_i8))
            n_i32 = fx.Int32(rows) * fx.Int32(K // 128)
            self.g_div = fx.logical_divide(
                fx.Tensor(fx.make_view(iter_i32, fx.make_layout(n_i32, 1))),
                fx.make_layout(1, 1),
            )
            self.atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            # Two register sets: the caller prefetches the next K-pair while the
            # current one is still feeding MFMAs.
            self.regs = [
                [
                    fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
                    for _ in range_constexpr(self.n_pairs)
                ]
                for _ in range_constexpr(2)
            ]

        def read(self, row_base16, k):
            """``n_tiles`` scale operands for K-step ``k``; tiles of a pair share one."""
            k1 = k // 2  # compile-time; k % 2 is the k_pack the opsel encodes
            regs = self.regs[k1 % 2]
            words = []
            for p in range_constexpr(self.n_pairs):
                # row_base16 is even (wave offsets are multiples of 32 rows), so the
                # 32-row super-row is row_base16 // 2 + p and n_pack is the tile parity.
                n1 = row_base16 // 2 + p
                base = fx.rocdl.readfirstlane(
                    fx.Int32.ir_type, (n1 * self.k1_stride + k1) * 64
                )
                fx.copy(
                    self.atom,
                    fx.slice(self.g_div, (None, fx.Int32(base) + self.lane)),
                    regs[p],
                )
                w = fx.Int32(regs[p].load()[0])
                words += [w, w]
            return words

    return ScalePreshuffledS2R()


def make_mx_mfma(n_tiles_a, n_tiles_b):
    class MxMfma:
        """16x16x128 scaled MFMA with per-tile packed scales and byte selectors.

        One ``(opsel_a, opsel_b)`` atom per byte pair: in the ``shuffle_scale_w4``
        layout a single i32 carries the E8M0 of two 16-row tiles x two K-steps, and
        the byte is picked by ``opsel`` -- a compile-time atom field -- so the hot
        loop emits no byte-select instructions at all.
        """

        def __init__(self):
            # opsel = k_pack * 2 + tile_in_pair, so both operands share k_pack.
            self.atoms = {
                (kp * 2 + ia, kp * 2 + jb): fx.make_mma_atom(
                    fx.rocdl.cdna4.MFMA_Scale(
                        16,
                        16,
                        128,
                        fx.Float8E4M3FN,
                        fx.Float8E4M3FN,
                        opsel_a=kp * 2 + ia,
                        opsel_b=kp * 2 + jb,
                    )
                )
                for kp in range_constexpr(2)
                for ia in range_constexpr(2)
                for jb in range_constexpr(2)
            }
            self.zero_value = Vec.filled(4, 0.0, fx.Float32)
            self.n_tiles_a = n_tiles_a
            self.n_tiles_b = n_tiles_b

        def idx(self, i, j):
            return i * self.n_tiles_b + j

        def _operand(self, value, words=8):
            frag = fx.make_rmem_tensor(words, fx.Int32)
            frag.store(Vec(value))
            return frag

        def _accum(self, value):
            frag = fx.make_rmem_tensor(4, fx.Float32)
            frag.store(Vec(value))
            return frag

        def _atom_for(self, k_pack, i, j):
            return self.atoms[(k_pack * 2 + i % 2, k_pack * 2 + j % 2)]

        def call(self, a, b, c, sa, sb, *, k_pack, set_prio=True):
            assert len(a) == self.n_tiles_a and len(sa) == self.n_tiles_a
            assert len(b) == self.n_tiles_b and len(sb) == self.n_tiles_b
            assert len(c) == self.n_tiles_a * self.n_tiles_b

            a_frags = [self._operand(a[i]) for i in range_constexpr(self.n_tiles_a)]
            b_frags = [self._operand(b[j]) for j in range_constexpr(self.n_tiles_b)]
            c_frags = [
                self._accum(c[i])
                for i in range_constexpr(self.n_tiles_a * self.n_tiles_b)
            ]
            if const_expr(set_prio):
                rocdl.s_setprio(1)
            for i in range_constexpr(self.n_tiles_a):
                for j in range_constexpr(self.n_tiles_b):
                    cf = c_frags[self.idx(i, j)]
                    atom = self._atom_for(k_pack, i, j)
                    fx.gemm(
                        atom,
                        cf,
                        a_frags[i],
                        b_frags[j],
                        cf,
                        scale_a=sa[i],
                        scale_b=sb[j],
                    )
            if const_expr(set_prio):
                rocdl.s_setprio(0)
                rocdl.s_barrier()
            return [
                c_frags[i].load().ir_value()
                for i in range_constexpr(self.n_tiles_a * self.n_tiles_b)
            ]

    return MxMfma()


def make_mx_pipeline_mma(mfma, a_sc, b_sc, a_base16, b_base16, k_iters):
    """MX scale prefetch and byte selection for the shared eight-wave schedule."""

    # Keep scale emission in this callable so FlyDSL tracks its source dependency.
    class MxPipelineMma:
        def __init__(self):
            self.mfma = mfma
            self.zero_value = mfma.zero_value
            self.n_tiles_a, self.n_tiles_b = mfma.n_tiles_a, mfma.n_tiles_b
            self.a_sc, self.b_sc = a_sc, b_sc
            self.a_base16, self.b_base16 = a_base16, b_base16
            self.k_iters = k_iters
            self.prefetched = {}

        def prefetch(self, k):
            if const_expr(k >= self.k_iters or k // 2 in self.prefetched):
                return
            words = {
                w: (self.a_sc if w[0] == "a" else self.b_sc).read(
                    (self.a_base16 if w[0] == "a" else self.b_base16)[int(w[1])], k
                )
                for w in ("a0", "a1", "b0", "b1")
            }
            self.prefetched = {**self.prefetched, k // 2: words}

        def call(self, a, b, c, *, k, a_half, b_half, set_prio=True):
            scales = self.prefetched[k // 2]
            return self.mfma.call(
                a,
                b,
                c,
                scales[f"a{a_half}"],
                scales[f"b{b_half}"],
                k_pack=k % 2,
                set_prio=set_prio,
            )

    return MxPipelineMma()


def _store_factory(
    *,
    activation=False,
    transpose=False,
    swiglu_limit=7.0,
    topk=1,
    apply_weight=False,
):
    def factory(C, route_ids, route_weights, cols, idx, n_tiles_a, n_tiles_b, scratch):
        cols = cols // 2 if activation else cols
        tile_n = n_tiles_b * (8 if activation else 16)
        tile_m = n_tiles_a * 16
        lane = fx.thread_idx.x % 64
        wave = fx.thread_idx.x // 64
        num_waves = fx.known_block_size()[0] // fx.num_warp_threads()
        # SharedAllocator fields are independent LDS globals, not one contiguous array.
        base = fx.Int32(fx.ptrtoint(scratch[0]))
        for i in range_constexpr(1, num_waves):
            base = (wave == i).select(fx.Int32(fx.ptrtoint(scratch[i])), base)
        ptr = fx.recast_iter(fx.BFloat16, fx.inttoptr(scratch[0].type, base))
        out = fx.rocdl.make_buffer_tensor(
            C,
            max_size=False,
            num_records_bytes=fx.Int64(fx.size(C.shape).unpack()) * 2,
        )
        out = fx.logical_divide(out, fx.make_layout(8, 1))
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        weights = fx.rocdl.make_buffer_tensor(route_weights, max_size=False)

        def scratch_at(row, col, width):
            offset = row * tile_n + (col ^ ((row % (tile_n // 8)) * 8))
            return fx.make_view(ptr + offset, fx.make_layout(width, 1))

        def output_row(sorted_row):
            route = route_ids[sorted_row]
            return (route & 0xFFFFFF) * topk + ((route >> 24) & 0xFF)

        def maybe_weight(values, sorted_row):
            if const_expr(apply_weight):
                return (values.to(fx.Float32) * weights[sorted_row]).to(fx.BFloat16)
            return values

        def store(c_frag, base_row, base_col):
            for ti in range_constexpr(n_tiles_a):
                row = ti * 16 + lane // 16 * 4
                for tj in range_constexpr(n_tiles_b // 2 if activation else n_tiles_b):
                    col = tj * 16 + lane % 16
                    value = Vec(c_frag[idx(ti, tj * 2 if activation else tj)])
                    if const_expr(transpose):
                        offset = row // 4 * (tile_n * 4) + col * 4
                        fx.make_view(ptr + offset, fx.make_layout(4, 1)).store(
                            value.to(fx.BFloat16)
                        )
                    else:
                        if const_expr(activation):
                            up = Vec(c_frag[idx(ti, tj * 2 + 1)])
                            activated = _swiglu_mul_batch(
                                [value[i] for i in range_constexpr(4)],
                                [up[i] for i in range_constexpr(4)],
                                limit=swiglu_limit,
                            )
                        for i in range_constexpr(4):
                            v = activated[i] if activation else value[i]
                            scratch_at(row + i, col, 1).store(
                                Vec.filled(1, v.to(fx.BFloat16), fx.BFloat16)
                            )
            rocdl.s_waitcnt(lgkmcnt=0)
            if const_expr(activation):
                base_col = base_col // 2
            if const_expr(transpose):
                for step in range_constexpr(tile_m * tile_n // (64 * 32)):
                    linear = lane * 32 + step * 64 * 32
                    row, col = linear // (tile_n * 4) * 4, linear // 4 % tile_n
                    values = fx.make_view(ptr + linear, fx.make_layout(32, 1)).load()
                    for i in range_constexpr(4):
                        sorted_row = base_row + row + i
                        reg = fx.make_rmem_tensor(8, fx.BFloat16)
                        reg.store(
                            maybe_weight(
                                Vec.from_elements(
                                    [values[i + j * 4] for j in range_constexpr(8)],
                                    fx.BFloat16,
                                ),
                                sorted_row,
                            )
                        )
                        offset = output_row(sorted_row) * cols + base_col + col
                        fx.copy(atom, reg, fx.slice(out, (None, offset >> 3)))
            else:
                for step in range_constexpr(tile_m * tile_n // (64 * 8)):
                    linear = lane * 8 + step * 64 * 8
                    row, col = linear // tile_n, linear % tile_n
                    sorted_row = base_row + row
                    reg = fx.make_rmem_tensor(8, fx.BFloat16)
                    reg.store(maybe_weight(scratch_at(row, col, 8).load(), sorted_row))
                    offset = output_row(sorted_row) * cols + base_col + col
                    fx.copy(atom, reg, fx.slice(out, (None, offset >> 3)))

        return store

    return factory
