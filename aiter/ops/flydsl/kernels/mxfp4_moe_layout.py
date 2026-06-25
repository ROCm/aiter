# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Shared FlyDSL **layout-API** primitives for the MXFP4 MoE GEMM ports.

The BM32 GEMM core is identical between gemm1 (up/gate-proj) and gemm2 (down-proj):
the CK-preshuffled B weight and its e8m0 B-scale share the same on-disk layout, and
the scaled 16x16x128 fp4 MFMA uses the same opsel pattern. This module factors out
those layout-API building blocks so both ``mxfp4_gemm1_v2`` and ``mxfp4_gemm2_v2``
reuse one definition:

  * ``b_copy_atom`` / ``bscale_copy_atom`` -- the BufferCopy atoms (the nt/cached
    policy rides on the B atom's ``cache_modifier``).
  * ``bq_view`` / ``bscale_view`` -- ``fx.make_layout`` views over the preshuffled
    weight / scale, anchored at a UNIFORM per-(wave) base so the per-lane
    (klane,nlane) + K-tile become layout axes (a VGPR voffset at copy time, not a
    divergent-pointer waterfall).
  * ``bq_frag_tmpl`` / ``bscale_frag_tmpl`` -- register-fragment templates sliced
    from those views (i32<4:1> for B, i32<1:1> for B-scale).
  * ``scale_mma_atoms`` / ``gemm_mma`` -- the pre-built (opselA,opselB) MFMA_Scale
    atom set and the one-mfma ``fx.gemm`` wrapper (scales ride scale_a=/scale_b=).

Callers keep their own fragment bookkeeping (per-stage vs per-tile), A-side LDS /
ds-read / A-scale, and epilogue -- only these B/B-scale/MMA pieces are shared.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import arith, rocdl
from flydsl.expr.typing import Float4E2M1FN
from flydsl.expr.typing import T

# i32 layout strides (BK=256-derived, K-independent; bytes/4):
#   B payload : klane[0,4)->64, nlane[0,16)->4, K_tile->512, half[0,2)->256, kpack4->1
#   B scale   : klane[0,4)->16, nlane[0,16)->1, K_tile->kBS_stride_k0_dw, unit->1
_BQ_LAYOUT_SHAPE = (4, 16, None, 2, 4)  # K_tile filled in per call
_BQ_LAYOUT_STRIDE = (64, 4, 512, 256, 1)
_BS_LAYOUT_STRIDE = (16, 1, None, 1)  # K_tile stride filled in per call


def _raw(v):
    """Unwrap an fx value to a raw ir.Value."""
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def b_copy_atom(nontemporal):
    """BufferCopy128b (4x i32 = one 128b weight chunk). nt rides cache_modifier."""
    return fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if nontemporal else 0), 32)


def bscale_copy_atom():
    """BufferCopy32b (1x i32 e8m0 scale word); always cached (scales reuse heavily)."""
    return fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), 32)


def bq_view(arg_bq, row_elems, KH4, K_TILES_TOTAL):
    """Layout view over the preshuffled B weight for one N-row tile.

    ``row_elems`` = (e*N_OUT + col): the logical N-row index into the weight. The
    uniform per-(wave) base is ``readfirstlane(row_elems * KH4)`` (KH4 = K_HALF//4,
    the i32 col stride); the per-lane (klane,nlane), K-tile, K-half, and kpack4 are
    layout axes -> a VGPR voffset at copy time. The byte base is zext'd before *4
    (it can exceed a signed i32). Index ``view[lane//16, lane%16, kt, half, None]``
    -> an i32<4:1> (16B = 32 fp4) slice for fx.copy / fx.gemm.
    """
    col_base = rocdl.readfirstlane(T.i32, _raw(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=16
    )
    off_i64 = fx.Int64(arith.ExtUIOp(T.i64, _raw(col_base)).result)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq) + off_i64 * fx.Int64(4))
    shape = (4, 16, K_TILES_TOTAL, 2, 4)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, _BQ_LAYOUT_STRIDE)))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def bscale_view(arg_bscale, base_dw, K_TILES_TOTAL, k0_stride_dw=64):
    """Layout view over the e8m0 B-scale for one n-pack word.

    ``base_dw`` = (e*kBS_per_expert_dw + np*kBS_stride_n0_dw): the uniform per-(wave)
    dword base (readfirstlane'd here). The per-lane (klane,nlane) and the K-tile are
    layout axes; the full K-tile rides the voffset (no hi/lo soffset split). Index
    ``view[lane//16, lane%16, kt, None]`` -> an i32<1:1> scale word.
    """
    base_dw = rocdl.readfirstlane(T.i32, _raw(base_dw))
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    off_i64 = fx.Int64(arith.ExtUIOp(T.i64, _raw(base_dw)).result)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bscale) + off_i64 * fx.Int64(4))
    shape = (4, 16, K_TILES_TOTAL, 1)
    stride = (16, 1, k0_stride_dw, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, stride)))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def bq_frag_tmpl(view):
    """i32<4:1> fragment template sliced from a bq_view (16B = 32 fp4)."""
    return view[0, 0, 0, 0, None]


def bscale_frag_tmpl(view):
    """i32<1:1> fragment template sliced from a bscale_view (one e8m0 word)."""
    return view[0, 0, 0, None]


def scale_mma_atoms():
    """Pre-build all 16 (opselA,opselB) scaled-MFMA atoms (opsel is a TYPE param, so
    one atom per pair; built once at trace time). cbsz/blgp(=4 for fp4) are inferred
    from Float4E2M1FN."""
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, Float4E2M1FN, opsel_a=osa, opsel_b=osb
            )
        )
        for osa in range(4)
        for osb in range(4)
    }


def gemm_mma(atoms, a_frag, b_frag, c_frag, opsel_a, opsel_b, sa, sb):
    """One scaled MFMA via fx.gemm over rank-1 register fragments (-> one
    MmaAtomCall). C accumulates in place (d == c); scales ride scale_a=/scale_b=."""
    fx.gemm(
        atoms[(opsel_a, opsel_b)],
        c_frag,
        a_frag,
        b_frag,
        c_frag,
        scale_a=sa,
        scale_b=sb,
    )
