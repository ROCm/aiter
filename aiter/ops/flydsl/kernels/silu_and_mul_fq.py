# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused gate-activation-and-mul + quantization + sorted-scale write kernel (FlyDSL).

Designed for split-K MOE stage1 post-processing:

  input   : tmp_out  (token_num * topk, inter_dim * 2) bf16
            topk_ids (token_num * topk) i32, optional
            bias     (expert, inter_dim * 2) f32, optional
  sorted  : sorted_token_ids (sorted_len,) i32 -- packed (token<<0 | slot<<24)
            num_valid_ids    (1,) i32
  output  : out              raw byte buffer (FP4x2, FP8, or BF16 depending on quant_mode)
            out_scale_sorted raw byte buffer -- tiled E8M0 scale (quant_mode fp4/fp8 only)

Compile options:
  quant_mode : "fp4" | "fp8" | "none"
  gui_layout : False -> gate-up separated  [gate_0:N, up_0:N]
               True  -> block-interleaved  [gate_0:16, up_0:16, gate_16:32, ...]
  act        : "silu" | "swiglu"
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Int32, T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.quant_utils import emit_f32_to_e2m1, emit_mx_e8m0_scale
from aiter.utility.mx_types import (
    MX_DEFAULT_ROUND_MODE as _DEFAULT_MODE,
)
from aiter.utility.mx_types import (
    MxDtypeInt as _D,
)

BLOCK_THREADS = 256
WARP_SIZE = 64


def build_silu_and_mul_fq_module(
    inter_dim: int,
    topk: int,
    quant_mode: str = "fp4",
    gui_layout: bool = False,
    act: str = "silu",
    enable_bias: bool = False,
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
):
    """Return a JIT launcher for fused gate activation + optional quant + scale sort.

    Parameters
    ----------
    inter_dim : int
        Output columns of stage1 (after activation). Input has inter_dim*2 cols.
        Must be divisible by 32 (quant block size).
    topk : int
        Number of expert slots per token.
    quant_mode : str
        "fp4"  -> MXFP4 output + e8m0 scale (tiled layout)
        "fp8"  -> MXFP8 output + e8m0 scale (tiled layout). Element dtype is
                  arch-dependent: e4m3fnuz (gfx942) or e4m3fn (gfx950+); the
                  E8M0 RoundUp scale formula picks ``max_pos`` accordingly.
        "none" -> bf16 output, no quantization (out_scale_sorted ignored)
    gui_layout : bool
        False -> input is gate-up separated  [gate_0:N | up_0:N]
        True  -> input is block-interleaved  [gate_0:16, up_0:16, gate_16:32, ...]
    """
    assert inter_dim % 32 == 0, f"inter_dim={inter_dim} must be divisible by 32"
    _need_fp4 = quant_mode == "fp4"
    _need_fp8 = quant_mode == "fp8"
    _need_quant = _need_fp4 or _need_fp8
    assert _need_fp4 or _need_fp8 or quant_mode == "none"
    if act not in ("silu", "swiglu", "situv2"):
        raise ValueError(f"Unsupported activation for split-K path: {act!r}")

    scale_cols = inter_dim // 32
    ELEMS_PER_THREAD = (inter_dim + BLOCK_THREADS - 1) // BLOCK_THREADS
    # VEC (a thread's contiguous vector) must be a power of two so it evenly
    # divides both the 32-element quant block and the 16-element gate/up block;
    # round up to the next power of two (even isn't enough: inter_dim=1536 gives
    # VEC=6, which divides neither). Cap at 8 (dwordx4/128-bit); VEC=16 fails
    # instruction selection. Wider inter_dim uses more COLS_PER_ITER iterations.
    VEC = 2
    while VEC < ELEMS_PER_THREAD:
        VEC *= 2
    VEC = min(VEC, 8)
    assert 32 % VEC == 0, f"VEC={VEC} must divide 32 evenly"
    if gui_layout:
        assert VEC <= 16, f"VEC={VEC} must be <=16 for block-interleave layout"
    THREADS_PER_QUANT_BLK = 32 // VEC
    SHUFFLE_DISTS = []
    d = 1
    while d < THREADS_PER_QUANT_BLK:
        SHUFFLE_DISTS.append(d)
        d *= 2

    elem_bytes_bf16 = 2

    # All four MXFP4/MXFP8 scale modes share NV ROUND_UP today (industry default,
    # 0% max-value clipping). FP8 dtype follows the HW FP8 variant: gfx942 ships
    # e4m3fnuz (max=240), gfx950+ ships OCP e4m3fn (max=448).
    _mx_dtype = (
        _D.FP4_E2M1
        if _need_fp4
        else (
            (_D.FP8_E4M3_FNUZ if get_hip_arch() == "gfx942" else _D.FP8_E4M3)
            if _need_fp8
            else _D.FP4_E2M1
        )
    )

    @flyc.kernel
    def silu_and_mul_fq_kernel(
        x: fx.Pointer,
        out_buf: fx.Pointer,
        out_scale_sorted: fx.Pointer,
        sorted_ids: fx.Pointer,
        num_valid_ids: fx.Pointer,
        topk_ids: fx.Pointer,
        bias: fx.Pointer,
        token_num: Int32,
        swiglu_limit_f: fx.Float32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32

        c0_f32 = arith.constant(0.0, type=f32)
        c1_f32 = arith.constant(1.0, type=f32)

        inter_dim2 = inter_dim * 2
        n32_sort = scale_cols * 32

        def _ptr_buffer_resource(ptr):
            addr_i64 = fx.Int64(fx.ptrtoint(ptr))
            return buffer_ops.create_buffer_resource_from_addr(addr_i64.ir_value())

        in_rsrc = _ptr_buffer_resource(x)
        out_rsrc = _ptr_buffer_resource(out_buf)
        scale_rsrc = _ptr_buffer_resource(out_scale_sorted)
        tid_rsrc = _ptr_buffer_resource(sorted_ids)
        nv_rsrc = _ptr_buffer_resource(num_valid_ids)
        if enable_bias:
            topk_rsrc = _ptr_buffer_resource(topk_ids)
            bias_rsrc = _ptr_buffer_resource(bias)

            def _load_bias_scalar(offset):
                return buffer_ops.buffer_load(bias_rsrc, offset, vec_width=1, dtype=f32)

        num_valid = fx.Int32(
            buffer_ops.buffer_load(nv_rsrc, 0, vec_width=1, dtype=T.i32)
        )
        fused_tid_val = fx.Int32(
            buffer_ops.buffer_load(tid_rsrc, bid, vec_width=1, dtype=T.i32)
        )
        token_id = fused_tid_val & 0xFFFFFF
        slot_id = fused_tid_val >> 24
        is_valid = (bid < num_valid) & (token_id < token_num) & (slot_id < topk)

        # FP4/FP8 scale and f32->fp4 conversion are shared with
        # mixed_moe_gemm_2stage; helpers live in
        # aiter.ops.flydsl.kernels.quant_utils.
        _f32_to_e2m1 = emit_f32_to_e2m1

        COLS_PER_ITER = BLOCK_THREADS * VEC

        for iter_idx in range_constexpr(
            (inter_dim + COLS_PER_ITER - 1) // COLS_PER_ITER
        ):
            col0 = tid * VEC + iter_idx * COLS_PER_ITER

            if col0 < inter_dim:
                if is_valid:
                    in_row = token_id * topk + slot_id
                    if enable_bias:
                        # sorted_ids encodes token and slot, not expert. Use topk_ids
                        # to recover the expert-specific bias row for this token slot.
                        expert_id = fx.Int32(
                            buffer_ops.buffer_load(
                                topk_rsrc, in_row, vec_width=1, dtype=T.i32
                            )
                        )
                        bias_row = expert_id * inter_dim2
                    in_row_byte_base = in_row * (inter_dim2 * elem_bytes_bf16)

                    vec_dw = VEC * elem_bytes_bf16 // 4

                    if const_expr(gui_layout):
                        # Block-interleaved (block=16):
                        #   [gate_0:16, up_0:16, gate_16:32, up_16:32, ...]
                        block_idx = col0 >> 4
                        offset_in_blk = col0 & 15
                        gate_col = block_idx * 32 + offset_in_blk
                        up_col = gate_col + 16
                    else:
                        # Gate-up separated: gate at col0, up at col0 + inter_dim
                        gate_col = col0
                        up_col = col0 + inter_dim

                    gate_byte = in_row_byte_base + gate_col * elem_bytes_bf16
                    up_byte = in_row_byte_base + up_col * elem_bytes_bf16
                    gate_dw = gate_byte >> 2
                    up_dw = up_byte >> 2

                    vec_bf16_ty = T.vec(VEC, T.bf16)
                    vec_f32_ty = T.vec(VEC, f32)

                    if const_expr(vec_dw == 1):
                        gate_raw = buffer_ops.buffer_load(
                            in_rsrc, gate_dw, vec_width=1, dtype=T.i32
                        )
                        up_raw = buffer_ops.buffer_load(
                            in_rsrc, up_dw, vec_width=1, dtype=T.i32
                        )
                        gate_bf16 = fx.Vector.from_elements(
                            [gate_raw], dtype=fx.Int32
                        ).bitcast(fx.BFloat16)
                        up_bf16 = fx.Vector.from_elements(
                            [up_raw], dtype=fx.Int32
                        ).bitcast(fx.BFloat16)
                    else:
                        gate_raw = buffer_ops.buffer_load(
                            in_rsrc, gate_dw, vec_width=vec_dw, dtype=T.i32
                        )
                        up_raw = buffer_ops.buffer_load(
                            in_rsrc, up_dw, vec_width=vec_dw, dtype=T.i32
                        )
                        gate_bf16 = fx.Vector(gate_raw).bitcast(fx.BFloat16)
                        up_bf16 = fx.Vector(up_raw).bitcast(fx.BFloat16)
                    gate_f32 = gate_bf16.extf(vec_f32_ty)
                    up_f32 = up_bf16.extf(vec_f32_ty)

                    neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                    swiglu_neg_alpha_log2e = arith.constant(
                        -1.4426950408889634 * 1.702, type=f32
                    )
                    # ``swiglu_limit`` is a runtime f32 scalar.  The host passes the
                    # clamp bound (7.0 default for swiglu) or +inf to disable the
                    # clamp (silu without a configured limit).  ``min(x, lim)`` is
                    # expressed via the wrapped ``maximumf`` + negation so the kernel
                    # never bakes the limit as a compile-time constant.
                    _neg_limit = -swiglu_limit_f

                    # The helpers below are re-defined per unrolled ``iter_idx`` on
                    # purpose: they close over SSA values emitted at this insertion
                    # point.  Bind those values as defaults so each definition
                    # captures *its own* iteration's values (also silences B023).
                    def _fmin(x, _neg_limit=_neg_limit):
                        # min(x, lim) == -max(-x, -lim)
                        return -((-x).maximumf(_neg_limit))

                    def _sigmoid_s(x, neg_log2e=neg_log2e):
                        emu = fx.rocdl.exp2(f32, x * neg_log2e)
                        return fx.rocdl.rcp(f32, c1_f32 + emu)

                    def _tanh_s(x):
                        # tanh(x) = 2*sigmoid(2x) - 1
                        two = arith.constant(2.0, type=f32)
                        return two * _sigmoid_s(two * x) - c1_f32

                    # SiTUv2 scale params are compile-time constants (folded via
                    # arith.constant), consistent with the main gemm1 kernel's
                    # compile-time situ_beta/situ_linear_beta model.
                    _sv2_beta_f32 = arith.constant(float(situ_beta), type=f32)
                    _sv2_beta_rcp = arith.constant(1.0 / float(situ_beta), type=f32)
                    _sv2_linbeta_f32 = arith.constant(float(situ_linear_beta), type=f32)
                    _sv2_linbeta_rcp = arith.constant(
                        1.0 / float(situ_linear_beta), type=f32
                    )

                    def _situv2_elem(
                        g,
                        u,
                        _sv2_beta_f32=_sv2_beta_f32,
                        _sv2_beta_rcp=_sv2_beta_rcp,
                        _sv2_linbeta_f32=_sv2_linbeta_f32,
                        _sv2_linbeta_rcp=_sv2_linbeta_rcp,
                    ):
                        # beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)
                        situ_g = (
                            _sv2_beta_f32 * _tanh_s(g * _sv2_beta_rcp) * _sigmoid_s(g)
                        )
                        up_sc = _sv2_linbeta_f32 * _tanh_s(u * _sv2_linbeta_rcp)
                        return situ_g * up_sc

                    act_vals = []
                    for vi in range_constexpr(VEC):
                        g = gate_f32[vi].ir_value()
                        u = up_f32[vi].ir_value()

                        if enable_bias:
                            bias_col = col0 + vi
                            g = g + _load_bias_scalar(bias_row + bias_col)
                            u = u + _load_bias_scalar(bias_row + inter_dim + bias_col)
                        if const_expr(act == "situv2"):
                            # SiTUv2: no clamp (tanh self-saturates).
                            act_vals.append(_situv2_elem(g, u))
                            continue
                        # gate: upper-clamped only; linear: clamped to [-lim, lim].
                        gate = _fmin(g)
                        linear = _fmin(u).maximumf(_neg_limit)
                        if const_expr(act == "swiglu"):
                            t = gate * swiglu_neg_alpha_log2e
                        else:
                            t = gate * neg_log2e

                        emu = fx.rocdl.exp2(f32, t)
                        den = c1_f32 + emu
                        sig = fx.rocdl.rcp(f32, den)
                        if const_expr(act == "swiglu"):
                            act_v = gate * sig * (linear + c1_f32)
                        else:
                            act_v = gate * sig * linear
                        act_vals.append(act_v)

                    if const_expr(_need_quant):
                        local_max = c0_f32
                        for vi in range_constexpr(VEC):
                            abs_v = fx.math.absf(act_vals[vi])
                            local_max = arith.maximumf(local_max, abs_v)

                        for sh_dist in SHUFFLE_DISTS:
                            peer = local_max.shuffle_xor(
                                arith.constant(sh_dist, type=T.i32),
                                arith.constant(64, type=T.i32),
                            )
                            local_max = arith.maximumf(local_max, peer)

                        # NV ROUND_UP / torchao RCEIL: scale = ceil_pow2(amax / max_pos),
                        # 0% max-value clipping. Same formula for FP4 / FP8; only
                        # max_pos differs (selected by ``_mx_dtype``).
                        e8m0_biased = emit_mx_e8m0_scale(
                            local_max, mode=_DEFAULT_MODE, dtype=_mx_dtype
                        )
                        quant_exp = arith.constant(254, type=T.i32) - e8m0_biased
                        quant_scale = (
                            quant_exp << arith.constant(23, type=T.i32)
                        ).bitcast(f32)

                        if const_expr(_need_fp4):
                            out_row_byte_base = in_row * (inter_dim // 2)
                            out_byte_off = out_row_byte_base + (col0 >> 1)

                            fp4_vals = []
                            for vi in range_constexpr(VEC):
                                scaled_v = act_vals[vi] * quant_scale
                                fp4_vals.append(fx.Int32(_f32_to_e2m1(scaled_v)))

                            packed_i32 = fp4_vals[0] | (fp4_vals[1] << 4)
                            for k in range_constexpr(1, VEC // 2):
                                byte_k = fp4_vals[2 * k] | (fp4_vals[2 * k + 1] << 4)
                                packed_i32 = packed_i32 | (byte_k << (k * 8))

                            _pack_bytes = VEC // 2
                            if const_expr(_pack_bytes == 1):
                                buffer_ops.buffer_store(
                                    packed_i32.to(fx.Int8),
                                    out_rsrc,
                                    out_byte_off,
                                    offset_is_bytes=True,
                                )
                            elif const_expr(_pack_bytes == 2):
                                buffer_ops.buffer_store(
                                    packed_i32.to(fx.Int16),
                                    out_rsrc,
                                    out_byte_off,
                                    offset_is_bytes=True,
                                )
                            else:
                                buffer_ops.buffer_store(
                                    packed_i32,
                                    out_rsrc,
                                    out_byte_off,
                                    offset_is_bytes=True,
                                )
                        else:
                            out_row_byte_base = in_row * inter_dim
                            out_byte_off = out_row_byte_base + col0

                            scaled_vals = []
                            for vi in range_constexpr(VEC):
                                scaled_vals.append(act_vals[vi] * quant_scale)

                            if const_expr(VEC <= 4):
                                packed_i32 = arith.constant(0, type=T.i32)
                                for _w in range_constexpr(VEC // 2):
                                    packed_i32 = fx.rocdl.cvt_pk_fp8_f32(
                                        T.i32,
                                        scaled_vals[2 * _w],
                                        scaled_vals[2 * _w + 1],
                                        packed_i32,
                                        _w,
                                    )
                                if const_expr(VEC == 2):
                                    buffer_ops.buffer_store(
                                        fx.Int32(packed_i32).to(fx.Int16),
                                        out_rsrc,
                                        out_byte_off,
                                        offset_is_bytes=True,
                                    )
                                else:
                                    buffer_ops.buffer_store(
                                        packed_i32,
                                        out_rsrc,
                                        out_byte_off,
                                        offset_is_bytes=True,
                                    )
                            else:
                                for _wg in range_constexpr(VEC // 4):
                                    _b = _wg * 4
                                    packed_w = arith.constant(0, type=T.i32)
                                    packed_w = fx.rocdl.cvt_pk_fp8_f32(
                                        T.i32,
                                        scaled_vals[_b],
                                        scaled_vals[_b + 1],
                                        packed_w,
                                        0,
                                    )
                                    packed_w = fx.rocdl.cvt_pk_fp8_f32(
                                        T.i32,
                                        scaled_vals[_b + 2],
                                        scaled_vals[_b + 3],
                                        packed_w,
                                        1,
                                    )
                                    word_off = out_byte_off + _wg * 4
                                    buffer_ops.buffer_store(
                                        packed_w,
                                        out_rsrc,
                                        word_off,
                                        offset_is_bytes=True,
                                    )

                        # E8M0 scale write: the 6-way index split (d0..d5) maps the
                        # (row, col/32) block to its tiled position in the sorted
                        # scale buffer; kept in sync with the host moe_mxfp4_sort.
                        if (col0 & 31) == 0:
                            row_s = bid
                            col_s = col0 >> 5
                            d0 = row_s >> 5
                            d1 = (row_s >> 4) & 1
                            d2 = row_s & 15
                            d3 = col_s >> 3
                            d4 = (col_s >> 2) & 1
                            d5 = col_s & 3
                            s_byte_off = (
                                d0 * n32_sort
                                + d3 * 256
                                + d5 * 64
                                + d2 * 4
                                + d4 * 2
                                + d1
                            )
                            e8m0_i8 = arith.TruncIOp(T.i8, e8m0_biased)
                            buffer_ops.buffer_store(
                                e8m0_i8,
                                scale_rsrc,
                                s_byte_off,
                                offset_is_bytes=True,
                            )

                    else:
                        out_row_byte_base = in_row * (inter_dim * elem_bytes_bf16)
                        out_byte_off = out_row_byte_base + col0 * elem_bytes_bf16
                        out_dw_off = out_byte_off >> 2
                        act_f32_vec = fx.Vector.from_elements(
                            act_vals, dtype=fx.Float32
                        )
                        act_bf16_vec = act_f32_vec.truncf(vec_bf16_ty)
                        act_i32 = fx.Vector(act_bf16_vec).bitcast(fx.Int32)
                        vec_dw_out = VEC * elem_bytes_bf16 // 4
                        if const_expr(vec_dw_out == 1):
                            buffer_ops.buffer_store(act_i32[0], out_rsrc, out_dw_off)
                        else:
                            buffer_ops.buffer_store(act_i32, out_rsrc, out_dw_off)

                else:
                    # Padding row: zero the E8M0 scale so the sorted scale buffer
                    # has no stale entries for invalid (padded) token slots.
                    if const_expr(_need_quant) and (col0 & 31) == 0:
                        row_s_p = bid
                        col_s_p = col0 >> 5
                        d0_p = row_s_p >> 5
                        d1_p = (row_s_p >> 4) & 1
                        d2_p = row_s_p & 15
                        d3_p = col_s_p >> 3
                        d4_p = (col_s_p >> 2) & 1
                        d5_p = col_s_p & 3
                        s_byte_off_p = (
                            d0_p * n32_sort
                            + d3_p * 256
                            + d5_p * 64
                            + d2_p * 4
                            + d4_p * 2
                            + d1_p
                        )
                        buffer_ops.buffer_store(
                            arith.constant(0, type=T.i8),
                            scale_rsrc,
                            s_byte_off_p,
                            offset_is_bytes=True,
                        )

    @flyc.jit
    def launch_silu_and_mul_fq(
        x: fx.Pointer,
        out_buf: fx.Pointer,
        out_scale_sorted: fx.Pointer,
        sorted_ids: fx.Pointer,
        num_valid_ids: fx.Pointer,
        topk_ids: fx.Pointer,
        bias: fx.Pointer,
        token_num: fx.Int32,
        num_sorted_rows: fx.Int32,
        swiglu_limit_f: fx.Float32,
        stream: fx.Stream,
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        idx_rows = fx.Int64(num_sorted_rows)
        launcher = silu_and_mul_fq_kernel(
            x,
            out_buf,
            out_scale_sorted,
            sorted_ids,
            num_valid_ids,
            topk_ids,
            bias,
            token_num,
            swiglu_limit_f,
        )
        launcher.launch(
            grid=(idx_rows, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_silu_and_mul_fq
