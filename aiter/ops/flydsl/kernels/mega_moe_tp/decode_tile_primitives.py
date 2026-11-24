# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Low-level per-tile building blocks for decode-shape (8-128 row, E128 top4) MoE GEMM1/GEMM2
compute at H6144/I768: scaled-MFMA wrappers (MXFP4 A4W4 and MXFP8 A8W8 variants), the SwiGLU
activation, MXFP4/MXFP8 row quantization and a byte-addressed LDS pointer helper.

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4/gemm1_a4w4_id_rows.py`` (the MXFP4
``_mfma``/``_hold``/``_raw``/``_lds_ptr``/``_fp4``/``_swiglu``), its sibling
``shared_gemm1_a8w8.py`` (the MXFP8 variants, suffixed ``8`` below) and ``shared_quant.py``
(``_e8m0_fp8``/``_fp8``), all Apache-2.0 "Copyright contributors to the amd-kernels project".
Every function is a thin, self-contained wrapper around one AMD scaled-MFMA / cvt ISA
instruction or a small fixed activation formula; none carries scheduling state of its own
(the tile loop, prefetch ring and LDS layout that call these live in the model-side fused
kernel, not in amd-kernels). A raw inline-asm / LLVM-intrinsic call is kept here, per aiter's
own FlyDSL rule, because FlyDSL has no published wrapper for ``v_mfma_scale_f32_16x16x128_f8f6f4``
(the gfx950 scaled-MFMA op with explicit ``op_sel``/``cbsz``/``blgp`` immediates these call
sites need) or for ``llvm.amdgcn.cvt.scalef32.pk.fp4.f32`` / ``...pk.fp8.f32``.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops


def _raw(x):
    """Unwrap only at LLVM intrinsic and inline-assembly boundaries."""
    return x.ir_value() if hasattr(x, "ir_value") else x


def _lds_ptr(base, off):
    """Byte-address the aligned LDS exchange and transpose regions."""
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(base + off)))


def _hold(xs):
    """Keep the required inline-MFMA latency before any accumulator reads (MXFP4 variant:
    always >1 output in this kernel's call sites, matches the amd-kernels gemm1_a4w4_id_rows
    body one-for-one)."""
    ty = ir.Type.parse("!llvm.struct<(" + ", ".join(["vector<4xf32>"] * len(xs)) + ")>")
    cons = ",".join(["=a"] * len(xs) + [str(i) for i in range(len(xs))])
    v = llvm.inline_asm(
        ty,
        [_raw(x) for x in xs],
        "s_nop 7\ns_nop 7\ns_nop 7",
        cons,
        has_side_effects=True,
    )
    return [llvm.extractvalue(T.vec(4, T.f32), v, [i]) for i in range(len(xs))]


def _hold8(xs):
    """``_hold``, with the single-output guard amd-kernels' shared_gemm1_a8w8.py adds (an
    LLVM inline-asm struct type cannot have exactly one member)."""
    if len(xs) == 1:
        return [
            llvm.inline_asm(
                T.vec(4, T.f32),
                [_raw(xs[0])],
                "s_nop 7\ns_nop 7\ns_nop 7",
                "=a,0",
                has_side_effects=True,
            )
        ]
    return _hold(xs)


def _mfma(a, b, acc, sa, sb, b_byte, transpose=False):
    """Native gfx950 FP4 scaled MFMA into AGPRs (MXFP4 A4W4 routed-expert tile);
    ``transpose`` selects GEMM1's C layout."""
    ty = T.vec(4, T.f32)
    a_byte = 0
    if transpose:
        a, b, sa, sb = b, a, sb, sa
        a_byte, b_byte = b_byte, 0
    asm = (
        f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {'$0' if acc is not None else '0'}, $3, $4 "
        f"op_sel:[{a_byte % 2},{b_byte % 2},0] op_sel_hi:[{a_byte // 2},{b_byte // 2},0] cbsz:4 blgp:4"
    )
    ops = [_raw(a), _raw(b), _raw(sa), _raw(sb)]
    cons = "=a,v,v,v,v"
    if acc is not None:
        ops.append(_raw(acc))
        cons += ",0"
    return llvm.inline_asm(ty, ops, asm, cons, has_side_effects=True)


def _mfma8(a, b, acc, sa, sb, b_byte, transpose=False):
    """Native gfx950 MXFP8 scaled MFMA into AGPRs (shared-expert gate/up/down tile);
    ``transpose`` selects GEMM1's C layout. Differs from ``_mfma`` only in the ``cbsz``/
    ``blgp`` immediates (0 here, selecting the A8W8 operand-format encoding vs. ``_mfma``'s
    A4W4 one) -- same instruction family, two operand formats, kept as two functions to match
    amd-kernels' own source split (gemm1_a4w4_id_rows.py vs shared_gemm1_a8w8.py) one-for-one.
    """
    ty = T.vec(4, T.f32)
    a_byte = 0
    if transpose:
        a, b, sa, sb = b, a, sb, sa
        a_byte, b_byte = b_byte, 0
    asm = (
        f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {'$0' if acc is not None else '0'}, $3, $4 "
        f"op_sel:[{a_byte % 2},{b_byte % 2},0] op_sel_hi:[{a_byte // 2},{b_byte // 2},0] cbsz:0 blgp:0"
    )
    ops = [_raw(a), _raw(b), _raw(sa), _raw(sb)]
    cons = "=a,v,v,v,v"
    if acc is not None:
        ops.append(_raw(acc))
        cons += ",0"
    return llvm.inline_asm(ty, ops, asm, cons, has_side_effects=True)


def _fp4(old, a, b, scale, sel):
    """FP32 pair -> packed MXFP4 (FlyDSL has no wrapper for this intrinsic)."""
    return fx.Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
            [_raw(old), _raw(a), _raw(b), _raw(scale), _raw(fx.Int32(sel))],
            [],
            [],
        )
    )


def _e8m0_fp8(amax):
    """e8m0 exponent of ceil_pow2(amax / 448), floored at 1; 255 (NaN / Inf) is not bumped."""
    u = (amax * fx.Float32(1.0 / 448.0)).bitcast(fx.Int32)
    e = (u >> 23) & 255
    bump = ((u & 0x7FFFFF) != 0) & (e < 255)
    e = bump.select(e + 1, e)
    return (e < 1).select(fx.Int32(1), e)


def _fp8(old, a, b, scale, hi):
    """``v_cvt_scalef32_pk_fp8_f32``: (a, b) / scale -> 2 e4m3 into the low (``hi`` False) or
    high half of ``old``."""
    ty = ir.VectorType.get([2], ir.IntegerType.get_signless(16))
    old_v = llvm.bitcast(ty, fx.as_ir_value(old))
    res = llvm.call_intrinsic(
        ty,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            old_v,
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(scale),
            fx.Boolean(hi).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(llvm.bitcast(T.i32, res))


def _swiglu(g, u):
    """SwiGLU-OAI in fp32: g <= 7, -7 <= u <= 7, g * sigmoid(1.702 g) * (u + 1), the sigmoid
    by exp2 + rcp. Identical formula in both the MXFP4 routed-expert tile and the MXFP8
    shared-expert tile (amd-kernels keeps two copies; this is the one shared definition).
    """
    g = (g < fx.Float32(7)).select(g, fx.Float32(7))
    u = (u < fx.Float32(7)).select(u, fx.Float32(7))
    u = (u > fx.Float32(-7)).select(u, fx.Float32(-7))
    t = (g * fx.Float32(1.702)) * fx.Float32(-1.4426950408889634)
    e = fx.Float32(fx.rocdl.exp2(T.f32, _raw(t)))
    r = fx.Float32(fx.rocdl.rcp(T.f32, _raw(fx.Float32(1) + e)))
    return (g * r) * (u + fx.Float32(1))


def _load8(rsrc, off, soff, plane=16):
    """Two 16-byte loads ``plane`` dwords apart as one 8-dword MFMA operand (MXFP8 tiles'
    wider K-step load)."""
    lo = Vec(
        buffer_ops.buffer_load(
            rsrc, off, vec_width=4, dtype=fx.Int32, soffset_bytes=soff
        )
    )
    hi = Vec(
        buffer_ops.buffer_load(
            rsrc, off + plane, vec_width=4, dtype=fx.Int32, soffset_bytes=soff
        )
    )
    return Vec.from_elements(
        [lo[j] for j in range(4)] + [hi[j] for j in range(4)], fx.Int32
    )


# ---------------------------------------------------------------------------
# Pure-Python shape/tiling arithmetic (no device code): ported from amd-kernels'
# experts_a4w4_small.py and sort_a16w4.py. Kept alongside the device primitives above because
# moe_one_launch-style call sites need both from one place; no GPU kernel logic lives here.
# ---------------------------------------------------------------------------

# BF16 entry with the shared expert: GEMM1 builds its 16-row blocks from topk_ids ("id-rows")
# at these row counts.
ID_ROWS_FROM, ID_ROWS_MAX = 16, 96


def id_rows_for(n):
    return ID_ROWS_FROM <= n <= ID_ROWS_MAX


def id_pairs_for(n):
    """Routing pair slots GEMM1 scans in the id-rows tier (a trace-time int, >= 4 * n)."""
    return 64 if n <= 16 else 128 if n <= 32 else 256 if n <= 64 else 512


def small_config_for(n):
    return 4, (16 if n <= 256 else 32)


def shared_fold_tile_for(n):
    """(rows, GEMM1 K-waves, GEMM1 prefetch depth) of the shared tiles that lead the routed
    launches."""
    if n <= 96:
        return 16, 4, 2
    if n <= 128:
        return 16, 2, 2
    return (64, 4, 1) if n <= 384 else (64, 2, 1)


def shared_g2_columns_for(n):
    """Columns per shared down tile inside the routed GEMM2 launch."""
    return 64 if n <= 16 else 256


def shared_fold_down_rows_for(sh_bm, bm):
    """Rows per folded shared down tile: at most the routed tile's rows."""
    return min(sh_bm, bm)


def max_sorted_rows(n_tokens: int, E: int, topk: int, block_m: int) -> int:
    """Aiter's bound (``moe_3stage_sort.cuh`` contract): every active expert pads by
    < block_m rows. Ported from amd-kernels' ``common/sort.py`` (pure arithmetic, no device
    code)."""
    active = min(E, n_tokens * topk)
    cumsum_max = n_tokens * topk + active * (block_m - 1)
    return ((cumsum_max + block_m - 1) // block_m) * block_m
