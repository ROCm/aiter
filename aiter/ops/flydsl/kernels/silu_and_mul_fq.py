# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused gate-activation-and-mul + quantization + sorted-scale write kernel (FlyDSL).

Split-K MOE stage1 post-processing. Inputs: tmp_out (token_num*topk, inter_dim*2)
bf16, optional topk_ids i32 / bias (expert, inter_dim*2) f32; sorted_token_ids
(sorted_len,) i32 packed (token<<0 | slot<<24) + num_valid_ids. Output: out (FP4x2,
FP8, or BF16 per quant_mode) + out_scale_sorted tiled E8M0 scale (fp4/fp8 only).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

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

    inter_dim: stage1 output cols (input has inter_dim*2); must be divisible by 32.
    quant_mode: "fp4"/"fp8" -> MX output + tiled e8m0 scale (fp8 elem dtype is
    arch-dependent, e4m3fnuz gfx942 / e4m3fn gfx950+); "none" -> bf16, no scale.
    gui_layout: False -> gate-up separated [gate_0:N | up_0:N]; True -> interleaved
    [gate_0:16, up_0:16, ...].
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
    # VEC (per-thread contiguous vector) must be a power of two to evenly divide
    # both the 32-elem quant block and the 16-elem gate/up block; even isn't enough
    # (inter_dim=1536 -> VEC=6 divides neither). Cap at 8 (128-bit); VEC=16 fails
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

    # FP8 dtype follows the HW variant: gfx942 e4m3fnuz (max=240), gfx950+ OCP
    # e4m3fn (max=448); the E8M0 RoundUp scale formula picks max_pos accordingly.
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

        c0_f32 = fx.Float32(0.0)
        c1_f32 = fx.Float32(1.0)

        inter_dim2 = inter_dim * 2
        n32_sort = scale_cols * 32

        _AS_G = fx.AddressSpace.Global

        def _i32_ptr(ptr):
            pt = fx.PointerType.get(T.i32, address_space=_AS_G, alignment=4)
            return fx.inttoptr(pt, fx.Int64(fx.ptrtoint(ptr)))

        # Scalar E8M0 scale writes go through an indexed store on this i8 pointer
        # over the sorted scale buffer (computed byte offset).
        scale_i8_ptr = fx.inttoptr(
            fx.PointerType.get(T.i8, address_space=_AS_G, alignment=1),
            fx.Int64(fx.ptrtoint(out_scale_sorted)),
        )
        sorted_ids_p = _i32_ptr(sorted_ids)
        num_valid_p = _i32_ptr(num_valid_ids)
        if enable_bias:
            topk_ids_p = _i32_ptr(topk_ids)
            bias_f32_p = fx.inttoptr(
                fx.PointerType.get(f32, address_space=_AS_G, alignment=4),
                fx.Int64(fx.ptrtoint(bias)),
            )

            def _load_bias_scalar(offset):
                return bias_f32_p[offset]

        num_valid = num_valid_p[fx.Int32(0)]
        fused_tid_val = sorted_ids_p[bid]
        token_id = fused_tid_val & 0xFFFFFF
        slot_id = fused_tid_val >> 24
        is_valid = (bid < num_valid) & (token_id < token_num) & (slot_id < topk)

        _f32_to_e2m1 = emit_f32_to_e2m1

        COLS_PER_ITER = BLOCK_THREADS * VEC
        # Views span N_ITERS whole tiles (>= inter_dim); the ragged tail past
        # inter_dim is dropped by each view's num_records descriptor.
        N_ITERS = (inter_dim + COLS_PER_ITER - 1) // COLS_PER_ITER
        PAD_COLS = N_ITERS * COLS_PER_ITER

        # Tiled-copy setup: thread tid owns contiguous cols [tid*VEC, (tid+1)*VEC),
        # which keeps the quant block's THREADS_PER_QUANT_BLK threads shuffle-adjacent
        # so the amax shuffle_xor reduction stays correct.
        def _view_bt(base_i64, elem_ir, shape, strides, nbytes):
            pt = fx.PointerType.get(elem_ir, address_space=_AS_G, alignment=2)
            view = fx.make_view(
                fx.inttoptr(pt, base_i64), fx.make_layout(shape, strides)
            )
            return fx.rocdl.make_buffer_tensor(view, num_records_bytes=fx.Int64(nbytes))

        # Copy-atom width must equal the per-thread transfer size, else BufferCopy
        # over-copies (128b atom = 8 bf16 regardless of VEC). Select class by bytes.
        _COPY_BY_BYTES = {
            1: fx.rocdl.BufferCopy8b,
            2: fx.rocdl.BufferCopy16b,
            4: fx.rocdl.BufferCopy32b,
            8: fx.rocdl.BufferCopy64b,
            16: fx.rocdl.BufferCopy128b,
        }

        def _copy_atom(nbytes, elem):
            return fx.make_copy_atom(_COPY_BY_BYTES[nbytes](), elem)

        _copy_ld = _copy_atom(VEC * elem_bytes_bf16, fx.BFloat16)

        # E8M0 scale scatter: one byte per 32 cols to a computed byte offset in the
        # sorted scale buffer, via an indexed store through the i8 scale pointer.
        if const_expr(_need_quant):

            def _store_scale_byte(byte_val_i8, byte_off):
                scale_i8_ptr[fx.Int32(byte_off)] = byte_val_i8

        # gui=True: gate/up are the swiglu (N0,16) stride-(32,1) interleave; one
        # iteration covers BM_iter=COLS_PER_ITER//NLANE_G block-rows of that view.
        # gui=False: gate/up are contiguous [inter_dim]; use a [1,inter_dim] view.
        NLANE_G = 16
        if const_expr(gui_layout):
            _NT_G = NLANE_G // VEC
            _BM_G = COLS_PER_ITER // NLANE_G
            _tile_ld, _tv_ld = fx.make_layout_tv(
                fx.make_layout((_BM_G, _NT_G), (_NT_G, 1)),
                fx.make_layout((1, VEC), (VEC, 1)),
            )
        else:
            _tile_ld, _tv_ld = fx.make_layout_tv(
                fx.make_layout((1, BLOCK_THREADS), (BLOCK_THREADS, 1)),
                fx.make_layout((1, VEC), (VEC, 1)),
            )
        _thr_ld = fx.make_tiled_copy(_copy_ld, _tv_ld, _tile_ld).get_slice(tid)

        # Store TV: contiguous [1, cols_per_row], store_unit elems/thread
        # (VEC bf16 / VEC bytes fp8 / VEC//2 bytes fp4).
        if const_expr(_need_fp4):
            _store_unit = VEC // 2
        else:
            _store_unit = VEC
        _st_elem = fx.Int8 if const_expr(_need_quant) else fx.BFloat16
        _st_elem_bytes = 1 if const_expr(_need_quant) else elem_bytes_bf16
        _copy_st = _copy_atom(_store_unit * _st_elem_bytes, _st_elem)
        _tile_st, _tv_st = fx.make_layout_tv(
            fx.make_layout((1, BLOCK_THREADS), (BLOCK_THREADS, 1)),
            fx.make_layout((1, _store_unit), (_store_unit, 1)),
        )
        _thr_st = fx.make_tiled_copy(_copy_st, _tv_st, _tile_st).get_slice(tid)

        for iter_idx in range_constexpr(
            (inter_dim + COLS_PER_ITER - 1) // COLS_PER_ITER
        ):
            col0 = tid * VEC + iter_idx * COLS_PER_ITER

            def _part_ld(buf, _iter_idx=iter_idx):
                if const_expr(gui_layout):
                    sub = fx.slice(
                        fx.zipped_divide(buf, _tile_ld), (None, (_iter_idx, 0))
                    )
                else:
                    sub = fx.slice(
                        fx.zipped_divide(buf, _tile_ld), (None, (0, _iter_idx))
                    )
                return _thr_ld.partition_S(sub)

            def _part_st(buf, _iter_idx=iter_idx):
                sub = fx.slice(fx.zipped_divide(buf, _tile_st), (None, (0, _iter_idx)))
                return _thr_st.partition_D(sub)

            if col0 < inter_dim:
                if is_valid:
                    in_row = token_id * topk + slot_id
                    if enable_bias:
                        # sorted_ids encodes token+slot, not expert; use topk_ids to
                        # recover the expert-specific bias row for this token slot.
                        expert_id = topk_ids_p[in_row]
                        bias_row = expert_id * inter_dim2
                    in_row_byte_base = in_row * (inter_dim2 * elem_bytes_bf16)

                    vec_bf16_ty = T.vec(VEC, T.bf16)
                    vec_f32_ty = T.vec(VEC, f32)

                    # Base-fold the runtime per-row byte offset into the input ptr,
                    # then load gate/up through TV-tiled copies.
                    x_row_i64 = fx.Int64(fx.ptrtoint(x)) + fx.Int64(in_row_byte_base)
                    _row_nb = inter_dim2 * elem_bytes_bf16
                    if const_expr(gui_layout):
                        # Interleaved [gate_0:16, up_0:16, ...] -> (N0,16) stride(32,1);
                        # up view is the same view shifted +16 elems (32 bytes).
                        _PAD_N0 = PAD_COLS // NLANE_G
                        gate_bt = _view_bt(
                            x_row_i64,
                            fx.BFloat16.ir_type,
                            (_PAD_N0, NLANE_G),
                            (2 * NLANE_G, 1),
                            _row_nb,
                        )
                        up_bt = _view_bt(
                            x_row_i64 + fx.Int64(NLANE_G * elem_bytes_bf16),
                            fx.BFloat16.ir_type,
                            (_PAD_N0, NLANE_G),
                            (2 * NLANE_G, 1),
                            _row_nb - NLANE_G * elem_bytes_bf16,
                        )
                    else:
                        # Separated [gate_0:N | up_0:N] -> two contiguous [1,N] views.
                        gate_bt = _view_bt(
                            x_row_i64,
                            fx.BFloat16.ir_type,
                            (1, PAD_COLS),
                            (PAD_COLS, 1),
                            inter_dim * elem_bytes_bf16,
                        )
                        up_bt = _view_bt(
                            x_row_i64 + fx.Int64(inter_dim * elem_bytes_bf16),
                            fx.BFloat16.ir_type,
                            (1, PAD_COLS),
                            (PAD_COLS, 1),
                            inter_dim * elem_bytes_bf16,
                        )

                    p_gate = _part_ld(gate_bt)
                    p_up = _part_ld(up_bt)
                    gate_frag = fx.make_fragment_like(p_gate)
                    up_frag = fx.make_fragment_like(p_up)
                    fx.copy(_copy_ld, p_gate, gate_frag)
                    fx.copy(_copy_ld, p_up, up_frag)
                    gate_bf16 = fx.Vector(fx.memref_load_vec(gate_frag)).reshape((VEC,))
                    up_bf16 = fx.Vector(fx.memref_load_vec(up_frag)).reshape((VEC,))
                    gate_f32 = gate_bf16.extf(vec_f32_ty)
                    up_f32 = up_bf16.extf(vec_f32_ty)

                    neg_log2e = fx.Float32(-1.4426950408889634)
                    swiglu_neg_alpha_log2e = fx.Float32(-1.4426950408889634 * 1.702)
                    # swiglu_limit is a runtime f32 scalar (clamp bound, or +inf to
                    # disable); min(x, lim) via maximumf + negation so the limit is
                    # never baked as a compile-time constant.
                    _neg_limit = -swiglu_limit_f

                    # Helpers are re-defined per unrolled iter_idx to close over the
                    # SSA values at this insertion point; bind as defaults so each
                    # captures its own iteration's values (also silences B023).
                    def _fmin(x, _neg_limit=_neg_limit):
                        # min(x, lim) == -max(-x, -lim)
                        return -((-x).maximumf(_neg_limit))

                    def _sigmoid_s(x, neg_log2e=neg_log2e):
                        emu = fx.Float32(fx.rocdl.exp2(f32, (x * neg_log2e).ir_value()))
                        return fx.Float32(fx.rocdl.rcp(f32, (c1_f32 + emu).ir_value()))

                    def _tanh_s(x):
                        # tanh(x) = 2*sigmoid(2x) - 1
                        two = fx.Float32(2.0)
                        return two * _sigmoid_s(two * x) - c1_f32

                    # SiTUv2 scale params are compile-time constants (like the main
                    # gemm1 kernel's situ_beta/situ_linear_beta model).
                    _sv2_beta_f32 = fx.Float32(float(situ_beta))
                    _sv2_beta_rcp = fx.Float32(1.0 / float(situ_beta))
                    _sv2_linbeta_f32 = fx.Float32(float(situ_linear_beta))
                    _sv2_linbeta_rcp = fx.Float32(1.0 / float(situ_linear_beta))

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
                        g = gate_f32[vi]
                        u = up_f32[vi]

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

                        emu = fx.Float32(fx.rocdl.exp2(f32, t.ir_value()))
                        den = c1_f32 + emu
                        sig = fx.Float32(fx.rocdl.rcp(f32, den.ir_value()))
                        if const_expr(act == "swiglu"):
                            act_v = gate * sig * (linear + c1_f32)
                        else:
                            act_v = gate * sig * linear
                        act_vals.append(act_v)

                    if const_expr(_need_quant):
                        local_max = c0_f32
                        for vi in range_constexpr(VEC):
                            abs_v = fx.math.absf(act_vals[vi])
                            local_max = local_max.maximumf(abs_v)

                        for sh_dist in SHUFFLE_DISTS:
                            peer = local_max.shuffle_xor(
                                fx.Int32(sh_dist), fx.Int32(64)
                            )
                            local_max = local_max.maximumf(peer)

                        # NV ROUND_UP: scale = ceil_pow2(amax / max_pos). Same formula
                        # for FP4/FP8; only max_pos differs (selected by _mx_dtype).
                        e8m0_biased = fx.Int32(
                            emit_mx_e8m0_scale(
                                local_max.ir_value(),
                                mode=_DEFAULT_MODE,
                                dtype=_mx_dtype,
                            )
                        )
                        quant_exp = fx.Int32(254) - e8m0_biased
                        quant_scale = (quant_exp << fx.Int32(23)).bitcast(fx.Float32)

                        if const_expr(_need_fp4):
                            # Pack VEC values into VEC//2 fp4 nibble bytes.
                            fp4_vals = []
                            for vi in range_constexpr(VEC):
                                scaled_v = act_vals[vi] * quant_scale
                                fp4_vals.append(
                                    fx.Int32(_f32_to_e2m1(scaled_v.ir_value()))
                                )

                            packed_i32 = fp4_vals[0] | (fp4_vals[1] << 4)
                            for k in range_constexpr(1, VEC // 2):
                                byte_k = fp4_vals[2 * k] | (fp4_vals[2 * k + 1] << 4)
                                packed_i32 = packed_i32 | (byte_k << (k * 8))

                            out_row_i8 = fx.Int64(fx.ptrtoint(out_buf)) + fx.Int64(
                                in_row * (inter_dim // 2)
                            )
                            out_bt = _view_bt(
                                out_row_i8,
                                fx.Int8.ir_type,
                                (1, PAD_COLS // 2),
                                (PAD_COLS // 2, 1),
                                inter_dim // 2,
                            )
                            # Keep exactly VEC//2 bytes (avoid over-storing the
                            # unused high bytes of packed_i32 into the neighbor).
                            _nb = VEC // 2
                            if const_expr(_nb == 1):
                                packed_i8 = fx.Vector.from_elements(
                                    [packed_i32.to(fx.Int8)], dtype=fx.Int8
                                )
                            elif const_expr(_nb == 2):
                                packed_i8 = fx.Vector.from_elements(
                                    [packed_i32.to(fx.Int16)], dtype=fx.Int16
                                ).bitcast(fx.Int8)
                            else:
                                packed_i8 = fx.Vector.from_elements(
                                    [packed_i32], dtype=fx.Int32
                                ).bitcast(fx.Int8)
                            p_out = _part_st(out_bt)
                            of = fx.make_fragment_like(p_out)
                            fx.memref_store_vec(packed_i8, of)
                            fx.copy(_copy_st, of, p_out)
                        else:
                            # Pack VEC values into VEC fp8 bytes.
                            scaled_vals = []
                            for vi in range_constexpr(VEC):
                                scaled_vals.append(act_vals[vi] * quant_scale)

                            # Each cvt_pk_fp8_f32 packs 2 f32 -> 2 fp8 bytes (one
                            # i16 halfword); assemble VEC//2 halfwords -> VEC bytes.
                            fp8_half = []
                            for _h in range_constexpr(VEC // 2):
                                _pk = fx.rocdl.cvt_pk_fp8_f32(
                                    T.i32,
                                    scaled_vals[2 * _h].ir_value(),
                                    scaled_vals[2 * _h + 1].ir_value(),
                                    fx.Int32(0).ir_value(),
                                    0,
                                )
                                fp8_half.append(fx.Int32(_pk).to(fx.Int16))

                            out_row_i8 = fx.Int64(fx.ptrtoint(out_buf)) + fx.Int64(
                                in_row * inter_dim
                            )
                            out_bt = _view_bt(
                                out_row_i8,
                                fx.Int8.ir_type,
                                (1, PAD_COLS),
                                (PAD_COLS, 1),
                                inter_dim,
                            )
                            packed_i8 = fx.Vector.from_elements(
                                fp8_half, dtype=fx.Int16
                            ).bitcast(fx.Int8)
                            p_out = _part_st(out_bt)
                            of = fx.make_fragment_like(p_out)
                            fx.memref_store_vec(packed_i8, of)
                            fx.copy(_copy_st, of, p_out)

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
                            _store_scale_byte(e8m0_biased.to(fx.Int8), s_byte_off)

                    else:
                        # bf16 output: truncate VEC f32 and store the bf16 vector.
                        act_f32_vec = fx.Vector.from_elements(
                            act_vals, dtype=fx.Float32
                        )
                        act_bf16_vec = act_f32_vec.truncf(vec_bf16_ty)
                        out_row_i64 = fx.Int64(fx.ptrtoint(out_buf)) + fx.Int64(
                            in_row * (inter_dim * elem_bytes_bf16)
                        )
                        out_bt = _view_bt(
                            out_row_i64,
                            fx.BFloat16.ir_type,
                            (1, PAD_COLS),
                            (PAD_COLS, 1),
                            inter_dim * elem_bytes_bf16,
                        )
                        p_out = _part_st(out_bt)
                        of = fx.make_fragment_like(p_out)
                        fx.memref_store_vec(fx.Vector(act_bf16_vec), of)
                        fx.copy(_copy_st, of, p_out)

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
                        _store_scale_byte(fx.Int8(0), s_byte_off_p)

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
