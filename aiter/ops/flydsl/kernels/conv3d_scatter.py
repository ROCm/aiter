# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Advanced Micro Devices, Inc.
#
# ruff: noqa: B023
# The store builds small closures over the tile loops and calls each one
# inside the same iteration, so the loop variable always holds the current
# value. Binding them as default arguments is not possible -- `bias_val` only
# exists on the has_bias path -- and per-line waivers do not survive
# `ruff format`, which moves the flagged column onto continuation lines.

"""The scatter side of the implicit-GEMM conv3d.

Where ``conv3d_im2col`` owns the gather that makes A look like a matrix, this
owns the scatter that turns C back into a convolution's output. The GEMM
produces a (npq, K) tile; NCDHW wants it as (N, K, Do, Ho, Wo), so a row has
to be decomposed back into its sample and spatial position and the store
strides along K instead of along the row. That, plus what the tail of an
over-provisioned grid must not write, plus split-K accumulating into fp32
staging with atomics, is everything the kernel's epilogue used to spell out
inline and now calls ``store`` for.

Only ``n == 1``, contiguous NCDHW, no split-K and a small enough output let
the four accumulator values of an MFMA atom land contiguously; that case takes
a single 64-bit store and every other one stores element by element. Which
case applies is decided once, in the plan.
"""

from typing import NamedTuple

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from .conv3d_gfx950_utils import (
    BF16_BYTES,
    MFMA_C_VALUES,
    OOB_SENTINEL_ELEM,
    buffer_atomic_add,
)
from .conv3d_im2col import ConvGeometry

# Split-K accumulates through a buffer descriptor, whose num_records is a
# 32-bit byte count, so the fp32 staging buffer has to fit one.
SPLITK_MAX_STAGING_BYTES = 0xFFFFFFFF


class OutputScatterPlan(NamedTuple):
    """What the epilogue knows before the kernel runs.

    A NamedTuple for the same reason ``Im2colPlan`` is one: FlyDSL keys a
    compiled kernel on the scalar values its closure captures, and anything
    that is not a scalar or a tuple silently drops out of that key.
    """

    n: int
    k: int
    kg: int
    groups: int
    out_ndhwc: bool
    has_bias: bool

    # The grid C is produced on, shared with the gather and the launch config.
    geom: ConvGeometry

    # MFMA atoms per wave along M and N: the epilogue walks them.
    mi_m: int
    mi_n: int

    # How the store is done, decided once below.
    use_splitk: bool
    big_out: bool
    y_bytes: int
    row_chk: bool
    n_tail: bool
    need_chk: bool
    route_store: bool
    vec_store: bool


def make_output_scatter_plan(
    param, geom, *, kg, mi_m, mi_n, use_splitk, row_chk, n_tail
):
    """An OutputScatterPlan for one problem and launch config, or an assertion.

    ``row_chk`` and ``n_tail`` say whether the grid over-provisions M and N,
    which only the caller's grid arithmetic knows; everything else about how C
    is written follows from the problem and is derived here.
    """
    n, k, out_ndhwc = param.n, param.k, param.out_ndhwc
    npq, dhw = geom.npq, geom.dhw

    big_out = (n * k * geom.do * geom.ho * geom.wo * BF16_BYTES) > 0x7FFFFFFF
    y_bytes = npq * k * (4 if use_splitk else BF16_BYTES)

    assert not use_splitk or npq * k * 4 <= SPLITK_MAX_STAGING_BYTES, (
        f"split-K staging {npq * k * 4}B exceeds the {SPLITK_MAX_STAGING_BYTES}B buffer window"
    )

    need_chk = row_chk or n_tail
    return OutputScatterPlan(
        n=n,
        k=k,
        kg=kg,
        groups=param.groups,
        out_ndhwc=out_ndhwc,
        has_bias=param.has_bias,
        geom=geom,
        mi_m=mi_m,
        mi_n=mi_n,
        use_splitk=use_splitk,
        big_out=big_out,
        y_bytes=y_bytes,
        row_chk=row_chk,
        n_tail=n_tail,
        need_chk=need_chk,
        # Routing a masked element to the OOB sentinel drops it without a
        # branch, but only where the store goes through a buffer descriptor:
        # split-K atomics and the 64-bit BIG_OUT path have no such address.
        route_store=need_chk and not use_splitk and not big_out,
        # The four values of an MFMA atom are consecutive rows, so they are
        # only contiguous in memory where a row's neighbour is the next
        # spatial position: NCDHW, one sample, one store, no staging.
        vec_store=(
            (n == 1)
            and (not use_splitk)
            and (dhw % MFMA_C_VALUES == 0)
            and (not big_out)
            and (not out_ndhwc)
        ),
    )


class OutputScatter:
    """The epilogue of one kernel: build with the other descriptors, then store.

    Stateless per block, unlike the gather -- the accumulator is written once,
    so the block's coordinates are arguments of ``store`` rather than of a
    separate bind.
    """

    def __init__(self, plan, y, bias, elem_ty):
        self._plan = plan
        self._y = y
        self._elem_ty = elem_ty

        y_buf = fx.rocdl.make_buffer_tensor(y, num_records_bytes=plan.y_bytes)
        if const_expr(plan.use_splitk):
            # buffer_atomic_add needs the raw !llvm.ptr<8> descriptor, not a tensor.
            self._y_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(y_buf))
        else:
            self._y_div = fx.logical_divide(
                fx.Tensor(
                    fx.make_view(
                        fx.get_iter(y_buf),
                        fx.make_layout(plan.geom.npq * plan.k, 1),
                    )
                ),
                fx.make_layout(1, 1),
            )
            self._y_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), elem_ty)
            self._y_reg_1 = fx.make_rmem_tensor(1, elem_ty)
            if const_expr(plan.vec_store):
                self._y_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), elem_ty)
                self._y_reg_4 = fx.make_rmem_tensor(MFMA_C_VALUES, elem_ty)
        if const_expr(plan.has_bias):
            self._bias_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(bias), fx.make_layout(1, 1)
            )
            self._bias_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            self._bias_reg = fx.make_rmem_tensor(1, fx.Float32)

        # A type, not an operation: constructing it emits no IR.
        self._big_st_ptr_ty = fx.PointerType.get(
            elem_ty.ir_type, fx.AddressSpace.Global, BF16_BYTES
        )

    def store(self, acc, *, m_offset, n_offset, n_local, c_row, c_col):
        """Write this block's accumulator out.

        ``c_row`` / ``c_col`` are the tile-local coordinates of the
        accumulator's elements, taken from the same tiled_mma that owns
        ``acc`` so the epilogue cannot drift from the MMA's own partitioning.
        """
        plan = self._plan
        elem_ty = self._elem_ty
        if const_expr(plan.big_out):
            self._y_elem_base = fx.Int64(fx.ptrtoint(fx.get_iter(self._y)))

        if const_expr(plan.has_bias and not plan.use_splitk):
            bias_vals = self._load_bias(n_offset, n_local, c_col)

        for mi in range_constexpr(plan.mi_m):
            row_base = m_offset + fx.get_scalar(c_row[MFMA_C_VALUES * mi])
            for ni in range_constexpr(plan.mi_n):
                col, col_loc = self._cols(ni, n_offset, n_local, c_col)
                a = fx.Vector(acc[None, mi, ni].load())
                if const_expr(plan.has_bias and not plan.use_splitk):
                    bias_val = bias_vals[ni]

                if const_expr(plan.vec_store):
                    row0 = fx.Int64(row_base)
                    off_nk0 = col * plan.geom.dhw + row0

                    def _emit_vec():
                        vals = []
                        for i in range_constexpr(MFMA_C_VALUES):
                            cval = (
                                (a[i] + bias_val) if const_expr(plan.has_bias) else a[i]
                            )
                            vals.append(cval.to(elem_ty))
                        v4 = fx.Vector.from_elements(vals, dtype=elem_ty)
                        fx.memref_store_vec(v4, self._y_reg_4)
                        fx.copy(
                            self._y_atom_4,
                            self._y_reg_4,
                            fx.slice(
                                self._y_div,
                                (None, self._route(off_nk0, row0, col_loc)),
                            ),
                        )

                    if const_expr(plan.need_chk and not plan.route_store):
                        if self._valid(row0, col_loc):
                            _emit_vec()
                    else:
                        _emit_vec()
                    continue

                for i in range_constexpr(MFMA_C_VALUES):
                    row = fx.Int64(row_base + i)
                    off_sk = row * plan.k + col
                    off_nk = self._off_nk(row, col, off_sk)

                    def _emit():
                        if const_expr(plan.use_splitk):
                            off_b = fx.Int32(off_sk * 4)
                            z0 = fx.Int32(0)
                            buffer_atomic_add(a[i], self._y_rsrc, off_b, z0, z0)
                        else:
                            cval = (
                                (a[i] + bias_val).to(elem_ty)
                                if const_expr(plan.has_bias)
                                else a[i].to(elem_ty)
                            )
                            if const_expr(plan.big_out):
                                self._big_store(fx.Int64(off_nk), cval)
                            else:
                                fx.memref_store_vec(
                                    fx.Vector.filled(1, cval, elem_ty), self._y_reg_1
                                )
                                fx.copy(
                                    self._y_atom_1,
                                    self._y_reg_1,
                                    fx.slice(
                                        self._y_div,
                                        (None, self._route(off_nk, row, col_loc)),
                                    ),
                                )

                    if const_expr(plan.need_chk and not plan.route_store):
                        if self._valid(row, col_loc):
                            _emit()
                    else:
                        _emit()

    def _load_bias(self, n_offset, n_local, c_col):
        """One bias value per MFMA column block, indexed by global out-channel."""
        plan = self._plan
        bias_vals = []
        for ni in range_constexpr(plan.mi_n):
            col, col_loc = self._cols(ni, n_offset, n_local, c_col)
            col_i = fx.Int32(col)
            if const_expr(plan.n_tail):
                col_i = (col_loc < fx.Int64(plan.kg)).select(col_i, fx.Int32(0))
            fx.copy(
                self._bias_atom, fx.slice(self._bias_div, (None, col_i)), self._bias_reg
            )
            bias_vals.append(fx.Float32(fx.memref_load_vec(self._bias_reg)[0]))
        return bias_vals

    def _cols(self, ni, n_offset, n_local, c_col):
        """Global out-channel for MFMA column block ni, and its index within the group."""
        plan = self._plan
        col_off = fx.Int64(fx.get_scalar(c_col[MFMA_C_VALUES * plan.mi_m * ni]))
        col = n_offset + col_off
        return col, ((n_local + col_off) if const_expr(plan.groups > 1) else col)

    def _off_nk(self, row, col, off_sk):
        """The GEMM's (row, col) as an element offset into y."""
        plan = self._plan
        dhw = plan.geom.dhw
        # NDHWC is already (npq, k) row-major, so the scatter is off_sk.
        if const_expr(plan.out_ndhwc):
            return off_sk
        if const_expr(plan.n == 1):
            return col * dhw + row
        n_idx = row // dhw
        return n_idx * (plan.k * dhw) + col * dhw + (row % dhw)

    def _valid(self, row, col_loc):
        plan = self._plan
        if const_expr(plan.row_chk and plan.n_tail):
            return (row < fx.Int64(plan.geom.npq)) & (col_loc < fx.Int64(plan.kg))
        if const_expr(plan.row_chk):
            return row < fx.Int64(plan.geom.npq)
        return col_loc < fx.Int64(plan.kg)

    def _route(self, off, row, col_loc):
        if const_expr(not self._plan.route_store):
            return fx.Int32(off)
        return self._valid(row, col_loc).select(
            fx.Int32(off), fx.Int32(OOB_SENTINEL_ELEM)
        )

    def _big_store(self, off_nk_i64, value):
        # BIG_OUT means y is past what a buffer descriptor's 32-bit voffset
        # reaches, so there is no buffer-resource form to route this through
        # and the store is addressed by a flat 64-bit address instead. That
        # is also why this path gives up y_div and the store copy atoms.
        addr = self._y_elem_base + off_nk_i64 * fx.Int64(BF16_BYTES)
        fx.ptr_store(value, fx.inttoptr(self._big_st_ptr_ty, addr))
