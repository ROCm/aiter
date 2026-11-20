# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""mHC step-end tail for hidden 5120, hc_mult 4, gfx950: post-mix, collapse, RMSNorm.

For ``T`` rows, four residual streams ``R[h]`` of ``H = 5120`` columns, the last
sub-layer output ``y`` and its seam's gates (post ``p``, comb ``c[h][j]``), the
carried pre gate ``q`` and the norm weight ``w``, one launch computes (fp32 math,
bf16 roundings where the tensors are bf16)::

    R'_j = bf16(p_j * y + sum_h c[h][j] * R_h)      HAS_POST (else R' = R)
    x1   = bf16(sum_j q_j * R'_j)
    out  = bf16(x1 * rsqrt(mean(x1^2) + eps) * w)
    mean = bf16((R'_0 + R'_1 + R'_2 + R'_3) / 4)     MEAN (auxiliary hidden state)

and optionally stores ``R'`` (``KEEP_RESIDUAL``).

One workgroup per row: 640 lanes (10 waves) own 8 of the 5120 columns each, of all
four streams, so the post-mix, the collapse and the normalization are lane-local;
the square sum is folded by DPP inside a wave and in a fixed order across the
waves through LDS, so repeated calls are bitwise identical.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import AITER_FLYDSL_KERNARG_PRELOAD

H = 5120
N_STREAMS = 4
ROW_DW = H // 2  # dwords of one bf16 row
NTHR = H // 8  # one lane per 8 columns
WAVES = NTHR // 64
KERNARG_PRELOAD_COUNT = 16

# DPP controls: row_shr:n = 0x110 + n, row_bcast:15 = 0x142, row_bcast:31 = 0x143
DPP_ROW_SHR = 0x110
DPP_ROW_BCAST15 = 0x142
DPP_ROW_BCAST31 = 0x143


def _dpp_f32(v, ctrl, row_mask=0xF, old=None):
    if old is None:
        # No fx spelling of an undef operand. With bound_ctrl every lane of an enabled
        # row is written (0 for an invalid source), so the old value is dead and need
        # not be materialized.
        old_v = llvm.mlir_undef(T.i32)
    else:
        old_v = fx.Float32(old).bitcast(fx.Int32).ir_value()
    src = fx.Float32(v).bitcast(fx.Int32).ir_value()
    r = rocdl.update_dpp(T.i32, old_v, src, ctrl, row_mask, 0xF, True)
    return fx.Int32(r).bitcast(fx.Float32)


def _wave_sum_lane63(acc):
    """row_shr 8/4/2/1 adds, then row_bcast 15 / 31: the wave sum lands in lane 63."""
    for sh in (8, 4, 2, 1):
        acc = acc + _dpp_f32(acc, DPP_ROW_SHR + sh)
    acc = _dpp_f32(acc, DPP_ROW_BCAST15, row_mask=0xA, old=acc) + acc
    return acc + _dpp_f32(acc, DPP_ROW_BCAST31)


def _unpack8(v):
    """Four dwords of packed bf16 -> eight fp32 values."""
    out = []
    for i in range(4):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _pack8(vals):
    """Eight fp32 values -> four dwords of bf16 (round to nearest even)."""
    return [
        fx.Int32(
            rocdl.cvt_pk_bf16_f32(
                fx.Float32(vals[2 * i]).ir_value(),
                fx.Float32(vals[2 * i + 1]).ir_value(),
            )
        )
        for i in range(4)
    ]


@functools.cache
def compile_mhc_post_collapse_rmsnorm(has_post: bool, keep_residual: bool, mean: bool):
    """``@flyc.jit`` launcher of one variant; the grid is one workgroup per row."""
    assert has_post or not keep_residual, "keep_residual needs the post-mix"
    name = "mhc_post_collapse_rmsnorm_h5120" + "".join(
        t
        for t, c in (
            ("_post", has_post),
            ("_keep", keep_residual),
            ("_mean", mean),
        )
        if c
    )

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, WAVES]

    @flyc.kernel(name=name, known_block_size=[NTHR, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_eps: fx.Int32,
        arg_res: fx.Int64,  # [m, 4, H] bf16
        arg_y: fx.Int64,  # [m, H] bf16
        arg_pm: fx.Int64,  # [m, 4] f32
        arg_cm: fx.Int64,  # [m, 4, 4] f32
        arg_pre: fx.Int64,  # [m, 4] f32
        arg_nw: fx.Int64,  # [H] bf16
        arg_rout: fx.Int64,  # [m, 4, H] bf16
        arg_out: fx.Int64,  # [m, H] bf16
        arg_mean: fx.Int64,  # [m, H] bf16
    ):
        tx = fx.thread_idx.x
        row = fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tx // 64))
        smem = fx.SharedAllocator().allocate(Shared).peek()
        part = smem.part.ptr
        m64 = fx.Int64(i32_m)

        def rsrc(addr, nbytes):
            return buffer_ops.create_buffer_resource_from_addr(
                addr, num_records_bytes=nbytes
            )

        def ldf(r, off, n):
            v = buffer_ops.buffer_load(r, off, vec_width=n, dtype=fx.Float32)
            return Vec(v) if n > 1 else fx.Float32(v)

        def ld4(r, off):
            return Vec(buffer_ops.buffer_load(r, off, vec_width=4, dtype=fx.Int32))

        rr = rsrc(arg_res, m64 * (N_STREAMS * H * 2))
        nwr = rsrc(arg_nw, H * 2)
        outr = rsrc(arg_out, m64 * (H * 2))
        qv = ldf(rsrc(arg_pre, m64 * 16), row * 4, 4)
        rs = [
            ld4(rr, row * (N_STREAMS * ROW_DW) + s * ROW_DW + tx * 4)
            for s in range_constexpr(N_STREAMS)
        ]
        w = _unpack8(ld4(nwr, tx * 4))
        if fx.const_expr(has_post):
            yv = _unpack8(ld4(rsrc(arg_y, m64 * (H * 2)), row * ROW_DW + tx * 4))
            pv = ldf(rsrc(arg_pm, m64 * 16), row * 4, 4)
            cmr = rsrc(arg_cm, m64 * 64)
            cv = [ldf(cmr, row * 16 + 4 * h, 4) for h in range_constexpr(N_STREAMS)]
            f = [_unpack8(r) for r in rs]
            pk = []
            for j in range_constexpr(N_STREAMS):
                vals = []
                for e in range_constexpr(8):
                    a = pv[j] * yv[e]
                    for h in range_constexpr(N_STREAMS):
                        a = fx.math.fma(cv[h][j], f[h][e], a)
                    vals.append(a)
                pk.append(_pack8(vals))
            if fx.const_expr(keep_residual):
                ror = rsrc(arg_rout, m64 * (N_STREAMS * H * 2))
                for j in range_constexpr(N_STREAMS):
                    buffer_ops.buffer_store(
                        Vec.from_elements(pk[j], fx.Int32),
                        ror,
                        row * (N_STREAMS * ROW_DW) + j * ROW_DW + tx * 4,
                    )
        else:
            pk = [[fx.Int32(r[k]) for k in range(4)] for r in rs]
        gf = [_unpack8(p) for p in pk]
        if fx.const_expr(mean):
            mv = [
                ((gf[0][e] + gf[1][e]) + gf[2][e] + gf[3][e]) * fx.Float32(0.25)
                for e in range_constexpr(8)
            ]
            buffer_ops.buffer_store(
                Vec.from_elements(_pack8(mv), fx.Int32),
                rsrc(arg_mean, m64 * (H * 2)),
                row * ROW_DW + tx * 4,
            )
        x1v = []
        for e in range_constexpr(8):
            a = qv[0] * gf[0][e]
            for s in range_constexpr(1, N_STREAMS):
                a = fx.math.fma(qv[s], gf[s][e], a)
            x1v.append(a)
        x1 = _unpack8(_pack8(x1v))
        acc = fx.Float32(0.0)
        for e in range_constexpr(8):
            acc = fx.math.fma(x1[e], x1[e], acc)
        acc = _wave_sum_lane63(acc)
        if lane == 63:
            part[wave] = acc
        gpu.barrier()
        ss = part[0]
        for i in range_constexpr(1, WAVES):
            ss = ss + part[i]
        eps = i32_eps.bitcast(fx.Float32)
        # v_rsq_f32: the precision the reference contract was validated with
        rstd = fx.Float32(rocdl.rsq(T.f32, (ss * fx.Float32(1.0 / H) + eps).ir_value()))
        yp = _pack8([(x1[e] * rstd) * w[e] for e in range_constexpr(8)])
        buffer_ops.buffer_store(
            Vec.from_elements(yp, fx.Int32), outr, row * ROW_DW + tx * 4
        )

    @flyc.jit
    def launch(
        arg_res: fx.Int64,
        arg_y: fx.Int64,
        arg_pm: fx.Int64,
        arg_cm: fx.Int64,
        arg_pre: fx.Int64,
        arg_nw: fx.Int64,
        arg_rout: fx.Int64,
        arg_out: fx.Int64,
        arg_mean: fx.Int64,
        i32_m: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m,
            i32_eps,
            arg_res,
            arg_y,
            arg_pm,
            arg_cm,
            arg_pre,
            arg_nw,
            arg_rout,
            arg_out,
            arg_mean,
        ).launch(grid=(fx.Int64(i32_m), 1, 1), block=(NTHR, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": KERNARG_PRELOAD_COUNT,
        }
    }
    return launch
