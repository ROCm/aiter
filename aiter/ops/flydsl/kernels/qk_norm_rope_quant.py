# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused per-token RMSNorm + GPT-J RoPE + optional FP8 quant (FlyDSL).

Q + KV in one launch (grid Y = num_tokens, grid X = num_q_heads + 1: bid_x in
[0, H) handle Q heads, bid_x == H handles KV). Hard-wired D=512, VEC=8,
BLOCK_THREADS=64: one wave per block, so reductions are wave-local (shuffle_xor,
no LDS/barrier). FP8 fast-path uses the rstd-cancellation algebra (matches the
Triton kernel in ``atom/model_ops/v4_kernels/qk_norm_rope_maybe_quant.py``):

    scale  = abs_max(x_norm) * SQRT2 / FP8_MAX     (sqrt(2) upper bound on rope mag)
    factor = FP8_MAX / (abs_max(x_in) * SQRT2)     (rstd cancels algebraically)

(The weighted KV path carries the per-channel weight into amax and factor.)
Public API ``flydsl_qk_norm_rope_quant``; ``compile_flydsl_qk_norm_rope_quant``
returns the cached launcher for pre-allocated callers.
"""

# do NOT add `from __future__ import annotations`: PEP 563 stringifies
# annotations, defeating flydsl's JitFunction._make_cache_key runtime detection
# (is_runtime = hasattr(ann, "__get_c_pointers__")). Int32 params like
# kv_in_row_stride / num_tokens would then be baked into the cache key per value,
# forcing a fresh ~30-70ms JIT compile per batch size / KV stride.

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, ptrtoint, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.arith import FastMathFlags  # fastmath kwarg object for fx fp ops
from flydsl.expr.typing import Int32, ReductionOp, Stream, T

# JIT-free MX-format int mirrors (keep module import JIT-free; aiter.utility
# .dtypes transitively JITs module_aiter_core, unbuilt during setup.py AOT walk).
from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale
from aiter.utility.mx_types import (
    MX_DEFAULT_ROUND_MODE as _DEFAULT_MODE,
)
from aiter.utility.mx_types import (
    MxDtypeInt as _D,
)

from .tensor_shim import _run_compiled

BLOCK_THREADS = 64  # 1 wave64
_SQRT2 = math.sqrt(2.0)


@lru_cache(maxsize=1)
def _fp8_const():
    """Lazy-resolve per-GFX native fp8 algebra coefficients.

    ``aiter.utility.dtypes.fp8`` is e4m3fnuz (max 240) on gfx942 and e4m3fn
    (max 448) on gfx950+; ``cvt_pk_fp8_f32`` emits in that native format, so
    FP8_MAX must track it or the stored dequant scale desyncs from downstream
    consumers reading the tensor as ``aiter.dtypes.fp8``. Cached, not at import.
    """
    from aiter.utility import dtypes as aiter_dtypes

    fp8_dtype = aiter_dtypes.fp8
    fp8_max = float(torch.finfo(fp8_dtype).max)
    return {
        "dtype": fp8_dtype,
        "max": fp8_max,
        "max_over_sqrt2": fp8_max / _SQRT2,  # forward-factor coefficient
        "inv_max_sqrt2": _SQRT2 / fp8_max,  # stored-scale coefficient
    }


# group_size == head_dim -> per-row scale (single scale per token-head).
GROUP_SIZE_OPTIONS = (32, 64, 128)

SCALE_DTYPE_FP32 = "fp32"
SCALE_DTYPE_E8M0 = "e8m0"
SCALE_DTYPE_OPTIONS = (SCALE_DTYPE_FP32, SCALE_DTYPE_E8M0)

_TORCH_DTYPE_FOR_SCALE = {
    SCALE_DTYPE_FP32: torch.float32,
    SCALE_DTYPE_E8M0: torch.uint8,  # no native torch e8m0 dtype; reinterpret as uint8
}


def _bf16_row_view(base_i64, num_elems, nbytes):
    """Build a ``(1, num_elems)`` bf16 buffer-tensor view at a folded base ptr.

    ``base_i64`` already includes the per-token / per-head byte shift;
    num_records bounds the OOB clamp (base + off with an i32 voffset)."""
    pt = fx.PointerType.get(
        fx.BFloat16.ir_type, address_space=fx.AddressSpace.Global, alignment=2
    )
    view = fx.make_view(
        fx.inttoptr(pt, base_i64), fx.make_layout((1, num_elems), (num_elems, 1))
    )
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=fx.Int64(nbytes))


# 1-element copy atoms sized per scalar dtype; the OOB-checked buffer tensor
# clamps a stray voffset, so a max-size resource over a wide 1-D span is safe.
_SCALAR_MAX_RECORDS = 1 << 24


def _scalar_view(base_i64, fx_dt):
    """A ``(N, 1)`` buffer-tensor view at i64 base ``base_i64`` for scalar
    element-indexed loads/stores (slice ``(idx, None)`` keeps one dim)."""
    pt = fx.PointerType.get(
        fx_dt.ir_type, address_space=fx.AddressSpace.Global, alignment=fx_dt.width // 8
    )
    view = fx.make_view(
        fx.inttoptr(pt, fx.Int64(base_i64)),
        fx.make_layout((_SCALAR_MAX_RECORDS, 1), (1, 1)),
    )
    return fx.rocdl.make_buffer_tensor(view, max_size=True)


def _scalar_load(base_i64, idx, fx_dt, copy_bits):
    """Load one ``fx_dt`` element at element index ``idx`` from base ``base_i64``."""
    buf = _scalar_view(base_i64, fx_dt)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy(copy_bits), fx_dt)
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx_dt)
    fx.copy_atom_call(atom, fx.slice(buf, (idx, None)), r)
    return fx.memref_load_vec(r)[0]


def _scalar_store(base_i64, idx, val, fx_dt, copy_bits):
    """Store one ``fx_dt`` value at element index ``idx`` into base ``base_i64``."""
    buf = _scalar_view(base_i64, fx_dt)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy(copy_bits), fx_dt)
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx_dt)
    fx.memref_store_vec(fx.Vector.from_elements([val.ir_value()], dtype=fx_dt), r)
    fx.copy_atom_call(atom, r, fx.slice(buf, (idx, None)))


def _store_bf16_tiled(vals_list, p_dst, copy, vec):
    """Convert VEC fp32 values to a bf16 fragment and tiled-copy to ``p_dst``.

    ``p_dst`` is this thread's D-partition of a ``(1, D)`` bf16 row view; the
    partition already places lane ``tid`` at element ``tid*VEC``."""
    raw = [v.ir_value() if hasattr(v, "ir_value") else v for v in vals_list]
    f32v = fx.Vector.from_elements(raw, dtype=fx.Float32)
    bf16v = f32v.truncf(T.vec(vec, T.bf16))
    frag = fx.make_fragment_like(p_dst)
    fx.memref_store_vec(bf16v, frag)
    fx.copy(copy, frag, p_dst)


def _store_fp8_packed(
    vals_list, out_base_i64, row_base_bytes, idx, vec, *, skip_fnuz_clamp=False
):
    """Pack VEC fp32 -> VEC fp8 via cvt_pk_fp8_f32 and store 8 bytes per thread
    (2 packed dwords = one i64) at ``out_base + row_base_bytes + idx*8``.

    fnuz clamp rationale: on e4m3fnuz, cvt_pk_fp8_f32 returns 0x80 (NaN) for
    inputs rounding to -0, which propagates as NaN through attention -- so clamp
    v in (-2^-8, 0) to +0. On gfx950+ (OCP e4m3fn) 0x80 is -0, not NaN, so
    ``skip_fnuz_clamp=True`` elides the clamp (~4 ALU ops/elem).
    """
    i32 = T.i32

    if skip_fnuz_clamp:
        safe = [v.ir_value() if hasattr(v, "ir_value") else v for v in vals_list]
    else:
        c0 = fx.Float32(0.0)
        c_neg_uf = fx.Float32(-(2.0**-8))
        safe = []
        for v in vals_list:
            vv = v if hasattr(v, "ir_value") else fx.Float32(v)
            is_tn = (vv < c0) & (vv > c_neg_uf)
            safe.append(is_tn.select(c0, vv).ir_value())

    assert vec == 8, "fp8 store helper hardcoded for VEC=8"
    p0 = fx.Int32(0).ir_value()
    p0 = fx.rocdl.cvt_pk_fp8_f32(i32, safe[0], safe[1], p0, 0)
    p0 = fx.rocdl.cvt_pk_fp8_f32(i32, safe[2], safe[3], p0, 1)
    p1 = fx.Int32(0).ir_value()
    p1 = fx.rocdl.cvt_pk_fp8_f32(i32, safe[4], safe[5], p1, 0)
    p1 = fx.rocdl.cvt_pk_fp8_f32(i32, safe[6], safe[7], p1, 1)

    # Fold row_base_bytes into the base ptr; store one i64 (= 2 packed dwords)
    # at i64-element index idx (idx*8 bytes) via the OOB-checked buffer tensor.
    base = out_base_i64 + fx.Int64(row_base_bytes)
    packed_i64 = fx.Vector.from_elements([p0, p1], dtype=fx.Int32).bitcast(fx.Int64)[0]
    _scalar_store(base, idx, fx.Int64(packed_i64), fx.Int64, 64)


# ============================================================================
# Kernel builder
# ============================================================================


def _build_kernel(
    *,
    num_q_heads: int,
    head_dim: int,
    rope_head_dim: int,
    quant: bool,
    group_size: int,
    scale_dtype: str,
    q_weighted: bool,
    kv_write: bool = False,
    paged: bool = False,
):
    """Build the @flyc.kernel + @flyc.jit launcher for a given config.

    Shape constants are captured via closure (not module globals) so launchers
    for different configs coexist. quant=True writes fp8 with one scale per
    ``group_size``-wide block (per-row when group_size == head_dim); scale_dtype
    picks the stored encoding. q_weighted applies a per-channel Q weight.
    """
    H = num_q_heads
    D = head_dim
    RD = rope_head_dim
    NOPE = D - RD
    VEC = D // BLOCK_THREADS
    ROPE_THREAD_LO = NOPE // VEC
    PAIRS_PER_THREAD = VEC // 2

    assert (
        D % BLOCK_THREADS == 0
    ), f"D={D} must be divisible by BLOCK_THREADS={BLOCK_THREADS}"
    assert NOPE % VEC == 0, f"NOPE={NOPE} must be divisible by VEC={VEC}"
    assert RD % 2 == 0, "rope_head_dim must be even (GPT-J pair layout)"
    assert RD % VEC == 0, f"RD={RD} must be divisible by VEC={VEC}"
    # Hard-wired VEC=8 (D=512): atom widths and _store_fp8_packed assume it.
    assert VEC == 8, (
        f"VEC={VEC} unsupported (D={D}); only D=512 / VEC=8 is implemented. "
        "Atom widths and fp8 packing assume VEC=8 -- generalising requires "
        "a wider refactor."
    )

    # group_size must divide D and be a multiple of VEC (so a thread's VEC-wide
    # slice never crosses a group boundary).
    assert (
        group_size > 0 and D % group_size == 0
    ), f"group_size {group_size} must divide head_dim {D}"
    assert (
        group_size % VEC == 0
    ), f"group_size {group_size} must be a multiple of VEC {VEC}"
    TPG = group_size // VEC  # threads per group
    NG = D // group_size  # number of groups per row
    assert (
        TPG > 0 and (TPG & (TPG - 1)) == 0
    ), f"TPG {TPG} must be a power of 2 (for butterfly reduce)"
    assert (
        scale_dtype in SCALE_DTYPE_OPTIONS
    ), f"scale_dtype {scale_dtype!r} must be one of {SCALE_DTYPE_OPTIONS}"

    log2_block = int(math.log2(BLOCK_THREADS))
    log2_tpg = int(math.log2(TPG))
    # In the butterfly loop, sumsq shuffles at offsets [BLOCK/2, ..., 1].
    # amax must NOT cross groups -> only shuffles at offsets < TPG -> only at
    # the last log2(TPG) loop iterations (sh_exp >= amax_start_step).
    amax_start_step = log2_block - log2_tpg

    elem_dtype = fx.BFloat16
    is_e8m0 = scale_dtype == SCALE_DTYPE_E8M0

    # FP8 element dtype follows the arch (matches _fp8_const); emit_mx_e8m0_scale
    # uses it to pick the right max_pos reciprocal.
    _is_fnuz = _fp8_const()["dtype"] == torch.float8_e4m3fnuz
    _fp8_mx_dtype = _D.FP8_E4M3_FNUZ if _is_fnuz else _D.FP8_E4M3

    # Kernel name: only flags that affect the compiled binary.
    _name_parts = ["qk_norm_rope", f"H{H}", f"D{D}", f"RD{RD}"]
    if q_weighted:
        _name_parts.append("qw")
    if quant:
        _name_parts.append(f"g{group_size}")
        _name_parts.append(scale_dtype)
    if kv_write:
        _name_parts.append("kvw")
    if paged:
        _name_parts.append("paged")
    _name_parts.append("flydsl")
    _kname = "_".join(_name_parts)

    @flyc.kernel(name=_kname)
    def kernel(
        q_in: fx.Pointer,  # [T, H, D]         bf16, contig (H, D)
        kv_in: fx.Pointer,  # [T, D]            bf16, may be strided
        q_weight: fx.Tensor,  # [D]               bf16 (dummy when not q_weighted)
        kv_weight: fx.Tensor,  # [D]               bf16
        cos_cache: fx.Tensor,  # [max_pos, RD/2]   bf16
        sin_cache: fx.Tensor,  # [max_pos, RD/2]   bf16
        positions: fx.Pointer,  # [T]               i64
        q_out: fx.Pointer,  # [T, H, D]         bf16 or fp8
        kv_out: fx.Pointer,  # [T, D]            bf16 or fp8
        q_scale: fx.Pointer,  # [T, H, NG]        f32 or uint8 (e8m0)
        kv_scale: fx.Pointer,  # [T, NG]           f32 or uint8 (e8m0)
        kv_in_row_stride: Int32,  # KV row stride in bf16 elements
        swa_kv: fx.Pointer,  # [num_slots, cache_size, D] bf16 (dummy if not kv_write)
        swa_index: fx.Pointer,  # paged: block_tables; direct: [T] dest rows
        batch_id_per_token: fx.Pointer,  # [T] i32, -1 sentinel (dummy if not kv_write)
        swa_slot_stride: Int32,  # bf16 elements (= cache_size * D)
        swa_pos_stride: Int32,  # bf16 elements (= D)
        swa_num_rows: Int32,  # rows in the SWA pool; bounds the final row
        swa_cache_size: Int32,  # ring slot count
    ):
        f32 = T.f32
        fm_fast = FastMathFlags.fast

        full_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 16)
        rope_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(64), 16)
        full_lay = fx.make_layout(VEC, 1)
        rope_lay = fx.make_layout(PAIRS_PER_THREAD, 1)

        def load_vec(
            div_tensor, idx, *, layout=full_lay, atom=full_atom, dt=elem_dtype
        ):
            r = fx.make_rmem_tensor(layout, dt)
            fx.copy_atom_call(atom, fx.slice(div_tensor, (None, idx)), r)
            return fx.memref_load_vec(r)

        bid_x = fx.block_idx.x  # 0..H-1 (Q head) or H (KV)
        bid_t = fx.block_idx.y  # token id (chunked at MAX_GRID_Y per launch)
        tid = fx.thread_idx.x

        # TV-tiled copy over one (1, D) bf16 row: thread t owns VEC contiguous
        # head elems [t*VEC, (t+1)*VEC) -- the SAME thread->element map the
        # shuffle_xor reductions assume.
        row_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        row_tile_mn, row_tv = fx.make_layout_tv(
            fx.make_layout((1, BLOCK_THREADS), (BLOCK_THREADS, 1)),
            fx.make_layout((1, VEC), (VEC, 1)),
        )
        row_thr = fx.make_tiled_copy(row_copy, row_tv, row_tile_mn).get_slice(tid)

        def _row_part_S(buf):
            return row_thr.partition_S(
                fx.slice(fx.zipped_divide(buf, row_tile_mn), (None, (0, 0)))
            )

        def _row_part_D(buf):
            return row_thr.partition_D(
                fx.slice(fx.zipped_divide(buf, row_tile_mn), (None, (0, 0)))
            )

        pos_val_i64 = _scalar_load(fx.Int64(ptrtoint(positions)), bid_t, fx.Int64, 64)
        pos_i32 = fx.Int64(pos_val_i64).to(fx.Int32)

        # cos/sin buffer tensors (rope-threads only)
        cos_buf = fx.rocdl.make_buffer_tensor(cos_cache)
        sin_buf = fx.rocdl.make_buffer_tensor(sin_cache)
        cos_row = fx.slice(cos_buf, (pos_i32, None))
        sin_row = fx.slice(sin_buf, (pos_i32, None))
        cos_div = fx.logical_divide(cos_row, rope_lay)
        sin_div = fx.logical_divide(sin_row, rope_lay)

        def wave_reduce_add(x):
            w = fx.Float32(x)
            for sh_exp in range_constexpr(int(math.log2(BLOCK_THREADS))):
                off = BLOCK_THREADS // (2 << sh_exp)
                peer = w.shuffle_xor(off, BLOCK_THREADS)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def emit_body(
            *,
            weighted: bool,
            x_f32_vec,
            w_f32_vec,  # None for Q
            bf16_out_part,  # D-partition of the (1,D) out row view (when not quant)
            fp8_out_base,  # (out_base_i64, row_base_bytes_within_token) when quant
            scale_base_i64,  # i64 base of the scale ptr (None when not quant)
            scale_base_off,  # base elem-offset; per-lane adds (tid // TPG)
            swa_out_part=None,  # D-partition of swa ring row view when kv_write
            do_swa=None,  # i1 predicate (batch_id >= 0); None when no kv_write
        ):
            """RMSNorm + GPT-J RoPE (+ optional FP8 quant) for this block's row.
            ``x_f32_vec`` / ``w_f32_vec`` are VEC-wide fp32 vectors from the
            caller (``w_f32_vec`` None for Q)."""
            x2 = x_f32_vec * x_f32_vec
            sq_local = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)

            if const_expr(quant):
                if const_expr(weighted):
                    xw = x_f32_vec * w_f32_vec
                    am_local = fmath.absf(xw).reduce(ReductionOp.MAX)
                else:
                    am_local = fmath.absf(x_f32_vec).reduce(ReductionOp.MAX)

                # Fused wave reduce: interleave sumsq-ADD (full row, RMSNorm
                # scope=D) and amax-MAX (one quant group, TPG threads) shuffles
                # so the scheduler overlaps both chains. amax only in tail steps
                # where shuffle offset < TPG; earlier steps would cross groups.
                w_sq = fx.Float32(sq_local)
                w_am = fx.Float32(am_local)
                for sh_exp in range_constexpr(log2_block):
                    off = BLOCK_THREADS // (2 << sh_exp)
                    w_sq = w_sq.addf(
                        w_sq.shuffle_xor(off, BLOCK_THREADS), fastmath=fm_fast
                    )
                    if const_expr(sh_exp >= amax_start_step):
                        w_am = w_am.maximumf(w_am.shuffle_xor(off, BLOCK_THREADS))
                sq_block = w_sq
                am_group = w_am  # per-group after partial butterfly
            else:
                sq_block = wave_reduce_add(sq_local)

            rstd = fmath.rsqrt(sq_block * (1.0 / D) + 1e-6, fastmath=fm_fast)

            if const_expr(quant):
                am_safe = fx.Float32(am_group).maximumf(fx.Float32(1e-12))

                if const_expr(is_e8m0):
                    # MX E8M0 RoundUp scale (NV ROUND_UP / torchao RCEIL, as in
                    # silu_and_mul_fq / mixed_moe_gemm_2stage). amax_post folds
                    # rstd (per-row) and SQRT2 (post-RoPE bound) so the forward
                    # factor bounds x_norm by the target fp8 max_pos.
                    amax_post = (am_safe * rstd * _SQRT2).ir_value()

                    e8m0_biased = fx.Int32(
                        emit_mx_e8m0_scale(
                            amax_post, mode=_DEFAULT_MODE, dtype=_fp8_mx_dtype
                        )
                    )
                    # quant_scale = 2^(127 - e8m0_biased) for x_norm; applied to
                    # x_in directly, so absorb rstd: factor = rstd * quant_scale.
                    quant_exp = fx.Int32(254) - e8m0_biased
                    quant_scale = (quant_exp << fx.Int32(23)).bitcast(fx.Float32)
                    factor = rstd * quant_scale
                    scale_to_store = e8m0_biased.to(fx.Int8)
                    scale_store_dt, scale_store_bits = fx.Int8, 8
                else:
                    # FP32 scale with the rstd-cancellation trick.
                    # scale_val = amax * rstd * SQRT2 / FP8_MAX  (stored)
                    # factor   = FP8_MAX / (amax * SQRT2)        (applied to x_in)
                    # The rstd factor cancels algebraically: store(out) =
                    # x_in * factor -> dequant: x_norm = scale * out = x_in * rstd.
                    rcp_am = fx.Float32(fx.rocdl.rcp(f32, am_safe))
                    _fc = _fp8_const()
                    factor = rcp_am * _fc["max_over_sqrt2"]
                    scale_to_store = am_safe * rstd * _fc["inv_max_sqrt2"]
                    scale_store_dt, scale_store_bits = fx.Float32, 32

                # Group-leader lanes (tid & (TPG-1) == 0) write the scale at
                # scale_base_off + tid/TPG. NOTE: a masked store sets offset
                # 0x7FFFFFFF on masked-off lanes -> OOB fault on gfx950; use the
                # predicated lane-leader branch instead.
                group_idx = tid >> fx.Int32(log2_tpg)
                lane_in_group = tid & fx.Int32(TPG - 1)
                if lane_in_group == 0:
                    my_scale_off = scale_base_off + group_idx
                    _scalar_store(
                        scale_base_i64,
                        my_scale_off,
                        scale_to_store,
                        scale_store_dt,
                        scale_store_bits,
                    )

            # Scale-multiply for ALL threads (hoisted out of rope/nope split).
            scaled = []
            for vi in range_constexpr(VEC):
                xi = x_f32_vec[vi]
                if const_expr(weighted):
                    xi = xi * w_f32_vec[vi]
                if const_expr(quant):
                    scaled.append(xi * factor)
                else:
                    scaled.append(xi * rstd)

            # Round-trip scaled values through rmem so both branches share them.
            out_rmem = fx.make_rmem_tensor(full_lay, fx.Float32)
            scaled_raw = [s.ir_value() for s in scaled]
            scaled_vec = fx.Vector.from_elements(scaled_raw, dtype=fx.Float32)
            fx.memref_store_vec(scaled_vec, out_rmem)

            # ROPE branch: load from rmem, rotate the GPT-J pairs, store back.
            is_rope = tid >= fx.Int32(ROPE_THREAD_LO)
            if is_rope:
                rope_rel = tid - fx.Int32(ROPE_THREAD_LO)
                cos_vec = load_vec(cos_div, rope_rel, layout=rope_lay, atom=rope_atom)
                sin_vec = load_vec(sin_div, rope_rel, layout=rope_lay, atom=rope_atom)
                cos_f32 = cos_vec.to(fx.Float32)
                sin_f32 = sin_vec.to(fx.Float32)

                cur = fx.memref_load_vec(out_rmem)
                rope_elems = []
                for k in range_constexpr(PAIRS_PER_THREAD):
                    e = cur[2 * k]
                    o = cur[2 * k + 1]
                    c = cos_f32[k]
                    s = sin_f32[k]
                    rope_elems.append((e * c - o * s).ir_value())
                    rope_elems.append((e * s + o * c).ir_value())
                rotated_vec = fx.Vector.from_elements(rope_elems, dtype=fx.Float32)
                fx.memref_store_vec(rotated_vec, out_rmem)

            # Unified store: all threads read the final row from rmem and write.
            final = fx.memref_load_vec(out_rmem)
            final_list = [final[i] for i in range(VEC)]

            if const_expr(quant):
                out_base, row_base = fp8_out_base
                _store_fp8_packed(
                    final_list,
                    out_base,
                    row_base,
                    tid,
                    VEC,
                    skip_fnuz_clamp=not _is_fnuz,
                )
            else:
                _store_bf16_tiled(final_list, bf16_out_part, row_copy, VEC)
                if const_expr(kv_write) and do_swa:
                    _store_bf16_tiled(final_list, swa_out_part, row_copy, VEC)

        # Runtime dispatch bid_x < H (Q heads) vs == H (KV). Each in/out base
        # folds bid_t (and the runtime kv row stride) into the descriptor base
        # ptr in i64, so the in-thread voffset stays small (bounded by D) and
        # large H*D configs never overflow i32 (>4GiB spans).
        if bid_x < fx.Int32(H):
            # ---------- Q path ----------
            head_idx = bid_x
            q_in_base = (
                fx.Int64(ptrtoint(q_in))
                + fx.Int64(bid_t) * fx.Int64(H * D * 2)
                + fx.Int64(head_idx) * fx.Int64(D * 2)
            )
            q_in_row = _bf16_row_view(q_in_base, D, D * 2)
            q_in_part = _row_part_S(q_in_row)
            q_frag = fx.make_fragment_like(q_in_part)
            fx.copy(row_copy, q_in_part, q_frag)
            # Round-trip through full_lay rmem so downstream sees the same vector
            # type as the buffer_load path; otherwise the tiled-copy fragment's
            # memref canonicalizes to ub.poison, which ROCm LLVM cannot lower.
            q_rmem = fx.make_rmem_tensor(full_lay, elem_dtype)
            fx.memref_store_vec(fx.memref_load_vec(q_frag), q_rmem)
            x_vec = fx.memref_load_vec(q_rmem)
            x_f32 = x_vec.to(fx.Float32)

            if const_expr(q_weighted):
                qw_buf = fx.rocdl.make_buffer_tensor(q_weight)
                qw_div = fx.logical_divide(qw_buf, full_lay)
                qw_vec = load_vec(qw_div, tid)
                qw_f32 = qw_vec.to(fx.Float32)
            else:
                qw_f32 = None

            if const_expr(quant):
                # fp8 store: fold only the per-token base into the i64 base;
                # row_base_bytes = head_idx*D added per-lane at store.
                qo_base = fx.Int64(ptrtoint(q_out)) + fx.Int64(bid_t) * fx.Int64(H * D)
                row_base_bytes = head_idx * D
                qs_base = fx.Int64(ptrtoint(q_scale))
                # q_scale (T, H, NG) flat; per-lane adds group_idx in emit_body.
                scale_base_off_q = bid_t * (H * NG) + head_idx * NG
                emit_body(
                    weighted=q_weighted,
                    x_f32_vec=x_f32,
                    w_f32_vec=qw_f32,
                    bf16_out_part=None,
                    fp8_out_base=(qo_base, row_base_bytes),
                    scale_base_i64=qs_base,
                    scale_base_off=scale_base_off_q,
                )
            else:
                # bf16 q_out: per-token + per-head base -> (1,D) tiled-copy store.
                qo_base = (
                    fx.Int64(ptrtoint(q_out))
                    + fx.Int64(bid_t) * fx.Int64(H * D * 2)
                    + fx.Int64(head_idx) * fx.Int64(D * 2)
                )
                qo_part = _row_part_D(_bf16_row_view(qo_base, D, D * 2))
                emit_body(
                    weighted=q_weighted,
                    x_f32_vec=x_f32,
                    w_f32_vec=qw_f32,
                    bf16_out_part=qo_part,
                    fp8_out_base=None,
                    scale_base_i64=None,
                    scale_base_off=None,
                )
        else:
            # ---------- KV path ----------
            # KV is often a strided slice (V4: split of qkv_a). The runtime row
            # stride (bf16 elems) is folded into the base ptr as bytes, then the
            # token row is a plain contiguous (1, D) tiled-copy like Q.
            kv_in_base = fx.Int64(ptrtoint(kv_in)) + fx.Int64(bid_t) * (
                fx.Int64(kv_in_row_stride) * fx.Int64(2)
            )
            kv_in_row = _bf16_row_view(kv_in_base, D, D * 2)
            kv_in_part = _row_part_S(kv_in_row)
            kv_frag = fx.make_fragment_like(kv_in_part)
            fx.copy(row_copy, kv_in_part, kv_frag)
            # Round-trip through full_lay rmem (see Q path note re: ub.poison).
            kv_rmem = fx.make_rmem_tensor(full_lay, elem_dtype)
            fx.memref_store_vec(fx.memref_load_vec(kv_frag), kv_rmem)
            x_vec = fx.memref_load_vec(kv_rmem)

            kvw_buf = fx.rocdl.make_buffer_tensor(kv_weight)
            w_div = fx.logical_divide(kvw_buf, full_lay)
            w_vec = load_vec(w_div, tid)
            x_f32 = x_vec.to(fx.Float32)
            w_f32 = w_vec.to(fx.Float32)

            if const_expr(quant):
                # fp8 kv_out: fold per-token base into the i64 base (already at
                # token base -> row_base_bytes = 0).
                kvo_base = fx.Int64(ptrtoint(kv_out)) + fx.Int64(bid_t) * fx.Int64(D)
                row_base_bytes = 0
                kvs_base = fx.Int64(ptrtoint(kv_scale))
                # kv_scale (T, NG) flat; per-lane adds group_idx in emit_body.
                scale_base_off_kv = bid_t * NG
                emit_body(
                    weighted=True,
                    x_f32_vec=x_f32,
                    w_f32_vec=w_f32,
                    bf16_out_part=None,
                    fp8_out_base=(kvo_base, row_base_bytes),
                    scale_base_i64=kvs_base,
                    scale_base_off=scale_base_off_kv,
                )
            else:
                # bf16 kv_out: per-token (1,D) row view + tiled-copy store.
                kvo_base = fx.Int64(ptrtoint(kv_out)) + fx.Int64(bid_t) * fx.Int64(
                    D * 2
                )
                kvo_part = _row_part_D(_bf16_row_view(kvo_base, D, D * 2))

                # ---- Fused SWA scatter setup (kv_write only) ----
                # Target row comes from `swa_index`, whose meaning `paged`
                # selects: a block table to walk, or this token's row outright.
                # batch_id has a -1 sentinel on CG-pad tokens; clamp to 0 to keep
                # the load in-bounds and gate the store on do_swa.
                swa_out_part = None
                do_swa = None
                if const_expr(kv_write):
                    bid_i32 = fx.Int32(
                        _scalar_load(
                            fx.Int64(ptrtoint(batch_id_per_token)),
                            bid_t,
                            fx.Int32,
                            32,
                        )
                    )
                    # A stale token carries a negative position. `divsi`/`remsi`
                    # truncate toward zero, so pos=-300 with block_size=128 gives
                    # blk=-2 and in_blk=-44: the block index passes an upper-bound
                    # test, reads the PREVIOUS request's table entry, and lands the
                    # write 44 rows before that block. The C++ sibling's first gate
                    # is `bid < 0 || pos < 0`; match it, in both modes.
                    pos_ok = pos_i32 >= fx.Int32(0)
                    do_swa = (bid_i32 >= fx.Int32(0)) & pos_ok
                    bid_safe = (bid_i32 >= fx.Int32(0)).select(bid_i32, fx.Int32(0))
                    pos_safe = pos_ok.select(pos_i32, fx.Int32(0))
                    if const_expr(paged):
                        # paged / content-addressed SWA (DeepSeek-V4 #1417):
                        # swa_kv is the FLAT [num_pages, D] pool; `swa_index` is
                        # block_tables[bs, max_blocks], swa_slot_stride=max_blocks,
                        # swa_cache_size=block_size, swa_pos_stride=D. Physical row
                        # = block_tables[bid, pos//bs]*bs + pos%bs.
                        blk = pos_safe // swa_cache_size
                        # The other two gates the C++ sibling applies (see
                        # `swa_row_for_token`): a position past the table's last
                        # block would read the NEXT request's entry, and `-1`
                        # marks a block outside the window. Clamp for the load,
                        # gate the store on the unclamped test. The wrapper
                        # pins stride(0) == max_blocks, so the row stride is
                        # both the table width and the distance between two
                        # requests' rows.
                        blk_ok = blk < swa_slot_stride
                        bt_off = bid_safe * swa_slot_stride + blk_ok.select(
                            blk, fx.Int32(0)
                        )
                        phys = fx.Int32(
                            _scalar_load(
                                fx.Int64(ptrtoint(swa_index)),
                                bt_off,
                                fx.Int32,
                                32,
                            )
                        )
                        in_blk = pos_safe % swa_cache_size
                        row_raw = phys * swa_cache_size + in_blk
                        # Bound the final row against the pool, as the C++ does:
                        # a phys id left over from a larger pool would otherwise
                        # write past the end of swa_kv.
                        row_ok = (
                            blk_ok & (phys >= fx.Int32(0)) & (row_raw < swa_num_rows)
                        )
                        do_swa = do_swa & row_ok
                        # Clamp too: a negative row would move the descriptor
                        # base backwards, out of this tensor entirely.
                        row = row_ok.select(row_raw, fx.Int32(0))
                    else:
                        # Direct: the caller already resolved this token's row.
                        # A -1 row means "skip", same as a -1 batch id -- a caller
                        # whose window this kernel cannot express needs a way to
                        # say so per token, not just per request.
                        dest = fx.Int32(
                            _scalar_load(
                                fx.Int64(ptrtoint(swa_index)),
                                bid_t,
                                fx.Int32,
                                32,
                            )
                        )
                        # Bounded against the pool as well: the caller owns the
                        # row, but a stale entry must not write past swa_kv.
                        dest_ok = (dest >= fx.Int32(0)) & (dest < swa_num_rows)
                        do_swa = do_swa & dest_ok
                        row = dest_ok.select(dest, fx.Int32(0))
                    # Fold the row's byte offset into the base ptr; the per-token
                    # row is then a plain (1, D) tiled-copy store like kv_out.
                    # Widen BEFORE the element product: a unified V4 pool runs to
                    # ~150M rows, so at D=512 `row * swa_pos_stride` passes 2^31
                    # 3% of the way in, and the windows this writes live at the
                    # far end. The base address is 64-bit; the descriptor's own
                    # offset field is not.
                    swa_base = fx.Int64(ptrtoint(swa_kv)) + fx.Int64(row) * fx.Int64(
                        swa_pos_stride
                    ) * fx.Int64(2)
                    swa_out_part = _row_part_D(_bf16_row_view(swa_base, D, D * 2))

                emit_body(
                    weighted=True,
                    x_f32_vec=x_f32,
                    w_f32_vec=w_f32,
                    bf16_out_part=kvo_part,
                    fp8_out_base=None,
                    scale_base_i64=None,
                    scale_base_off=None,
                    swa_out_part=swa_out_part,
                    do_swa=do_swa,
                )

    # Named launcher so the flydsl disk cache dir is
    # launch_qk_norm_rope_quant_<hash>/ instead of the generic launcher_<hash>/.
    @flyc.jit
    def launch_qk_norm_rope_quant(
        q_in: fx.Pointer,
        kv_in: fx.Pointer,
        q_weight: fx.Tensor,
        kv_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        positions: fx.Pointer,
        q_out: fx.Pointer,
        kv_out: fx.Pointer,
        q_scale: fx.Pointer,
        kv_scale: fx.Pointer,
        kv_in_row_stride: fx.Int32,
        swa_kv: fx.Pointer,
        swa_index: fx.Pointer,
        batch_id_per_token: fx.Pointer,
        swa_slot_stride: fx.Int32,
        swa_pos_stride: fx.Int32,
        swa_num_rows: fx.Int32,
        swa_cache_size: fx.Int32,
        num_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        idx_tokens = fx.Int64(num_tokens)
        k = kernel(
            q_in,
            kv_in,
            q_weight,
            kv_weight,
            cos_cache,
            sin_cache,
            positions,
            q_out,
            kv_out,
            q_scale,
            kv_scale,
            kv_in_row_stride,
            swa_kv,
            swa_index,
            batch_id_per_token,
            swa_slot_stride,
            swa_pos_stride,
            swa_num_rows,
            swa_cache_size,
        )
        k.launch(
            grid=(H + 1, idx_tokens, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_qk_norm_rope_quant


# ============================================================================
# Cached compile + public API
# ============================================================================

# Empirically best occupancy on MI355X V4-Pro (sweep): waves_per_eu=8 +
# fast/unsafe fp math, no regression at large T.
_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 8,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


@lru_cache(maxsize=32)
def compile_flydsl_qk_norm_rope_quant(
    *,
    num_q_heads: int,
    head_dim: int,
    rope_head_dim: int,
    quant: bool,
    group_size: int,
    scale_dtype: str,
    q_weighted: bool,
    kv_write: bool = False,
    paged: bool = False,
):
    """Compile (and cache) the launcher for a given config. Returns the
    @flyc.jit launcher; call it directly to skip the per-call torch overhead in
    ``flydsl_qk_norm_rope_quant`` (bounded lru; sibling ops do the same)."""
    launcher = _build_kernel(
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        quant=quant,
        group_size=group_size,
        scale_dtype=scale_dtype,
        q_weighted=q_weighted,
        paged=paged,
        kv_write=kv_write,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


def flydsl_qk_norm_rope_quant(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_q_heads: int,
    head_dim: int,
    rope_head_dim: int,
    q_weight: torch.Tensor | None = None,
    quant: bool = False,
    quant_group_size: int | None = None,
    scale_dtype: str = SCALE_DTYPE_FP32,
    q_out: torch.Tensor | None = None,
    kv_out: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
    swa_kv: torch.Tensor | None = None,
    swa_dest_rows: torch.Tensor | None = None,
    batch_id_per_token: torch.Tensor | None = None,
    swa_block_tables: torch.Tensor | None = None,
    swa_block_size: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Fused RMSNorm + GPT-J RoPE + optional FP8 quant for Q and KV in one launch.

    Args:
        q: Q activations, bf16, ``[T, H*D]`` (``.view``-reshaped) or ``[T, H, D]``;
            must be contig in the (H, D) inner dims.
        kv: KV pre-RoPE/norm, ``[T, D]`` bf16. May be a strided view (the row
            stride from ``kv.stride(0)`` is passed through).
        kv_weight: per-channel KV RMSNorm weight, ``[D]`` bf16.
        cos_cache, sin_cache: RoPE cos/sin, last dim RD/2, any leading shape that
            ``view``-reshapes to ``[max_pos, RD/2]`` (bf16).
        positions: per-token RoPE indices, ``[T]`` int64.
        num_q_heads / head_dim / rope_head_dim: H / D / RD (RD-tail is rotated,
            first D-RD elements pass through as NOPE).
        q_weight: optional per-channel Q RMSNorm weight ``[D]`` bf16; None ->
            weightless Q (V4-Pro default).
        quant: if True write fp8 in the per-GFX native encoding from
            ``aiter.dtypes.fp8`` (e4m3fnuz gfx942 / e4m3fn gfx950); else bf16.
        quant_group_size: 1xG scale block width; default head_dim (per-row).
            Kernel needs G a multiple of head_dim//BLOCK_THREADS (8), so
            typical sub-row choices are {32, 64, 128}.
        scale_dtype: ``"fp32"`` (default) or ``"e8m0"`` (MX uint8).
        q_out, kv_out, q_scale, kv_scale: output buffers (allocated if None);
            q_out [T,H,D], kv_out [T,D], q_scale [T,H,NG], kv_scale [T,NG] with
            NG = head_dim // quant_group_size; fp32 or uint8 (e8m0) scales.
        stream: launch stream (default current). Do NOT leave at
            ``fx.Stream(None)`` under CUDA-graph capture (NULL stream -> empty
            captured graph).
        swa_kv: optional flat bf16 SWA pool (bf16 only, incompatible with quant);
            the post-norm/rope KV row is also scattered into it in the same
            launch, fusing the standalone swa_write. ``[num_rows, D]`` with
            ``swa_dest_rows``, ``[num_pages, D]`` with ``swa_block_tables``.
        swa_dest_rows: ``[T]`` int32 -- the row THIS TOKEN's KV goes to, one entry
            per token (not per request: under speculation one request writes
            several positions in a launch and they do not share a row). ``-1``
            skips the token, letting a caller whose window layout this kernel
            cannot express resolve the row itself. Required w/ swa_kv unless
            ``swa_block_tables`` is given.
        batch_id_per_token: ``[T]`` int32, -1 on CG-pad tokens (store gated off);
            required w/ swa_kv.

    Returns:
        (q_out, kv_out, q_scale_or_None, kv_scale_or_None); scales None when
        ``quant=False``.
    """
    # ---- gfx1250 dispatch (wave32) ----
    from aiter.jit.utils.chip_info import get_gfx as _get_gfx

    if _get_gfx() == "gfx1250":
        from .qk_norm_rope_quant_gfx1250 import flydsl_qk_norm_rope_quant_gfx1250

        return flydsl_qk_norm_rope_quant_gfx1250(
            q=q,
            kv=kv,
            kv_weight=kv_weight,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            positions=positions,
            num_q_heads=num_q_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            q_weight=q_weight,
            quant=quant,
            quant_group_size=quant_group_size,
            scale_dtype=scale_dtype,
            q_out=q_out,
            kv_out=kv_out,
            q_scale=q_scale,
            kv_scale=kv_scale,
            swa_kv=swa_kv,
            swa_dest_rows=swa_dest_rows,
            batch_id_per_token=batch_id_per_token,
            swa_block_tables=swa_block_tables,
            swa_block_size=swa_block_size,
            stream=stream,
        )

    # User-facing inputs use raise (not stripped under python -O); internal
    # codegen invariants in _build_kernel stay as asserts.
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must be bf16, got {q.dtype}")
    if kv.dtype != torch.bfloat16:
        raise TypeError(f"kv must be bf16, got {kv.dtype}")
    if kv_weight.dtype != torch.bfloat16:
        raise TypeError(f"kv_weight must be bf16, got {kv_weight.dtype}")
    if kv.stride(-1) != 1:
        raise ValueError(f"kv must be dense in the last dim, stride={kv.stride()}")
    # KV loads cast bf16->dword; the >>1 in the offset is only correct when the
    # per-row byte offset is dword-aligned, i.e. the bf16 row stride is even.
    if kv.stride(0) % 2 != 0:
        raise ValueError(
            "kv row stride (in bf16 elements) must be even for dword-cast "
            f"buffer loads, got kv.stride(0)={kv.stride(0)}"
        )
    if positions.dtype != torch.int64:
        raise TypeError(f"positions must be int64, got {positions.dtype}")
    if scale_dtype not in SCALE_DTYPE_OPTIONS:
        raise ValueError(f"scale_dtype {scale_dtype!r} not in {SCALE_DTYPE_OPTIONS}")
    if q_weight is not None and q_weight.dtype != torch.bfloat16:
        raise TypeError(f"q_weight must be bf16, got {q_weight.dtype}")

    H, D, RD = num_q_heads, head_dim, rope_head_dim
    T_tok = q.shape[0]
    G = quant_group_size if quant_group_size is not None else D
    NG = D // G
    if D % G != 0:
        raise ValueError(f"head_dim {D} must be divisible by quant_group_size {G}")
    q_weighted = q_weight is not None
    # Kernel always binds q_weight; pass a dummy when unused (const_expr gate
    # DCEs the load but the param binding still needs a valid tensor).
    q_weight_arg = q_weight if q_weighted else kv_weight

    # Normalize Q to [T, H, D] (the kernel expects 3D).
    if q.dim() == 2:
        if q.shape[1] != H * D:
            raise ValueError(f"q shape {tuple(q.shape)} != [T, H*D={H * D}]")
        if not q.is_contiguous():
            raise ValueError("2D q must be contiguous to .view as [T,H,D]")
        q_view = q.view(T_tok, H, D)
    else:
        if q.dim() != 3 or q.shape != (T_tok, H, D):
            raise ValueError(
                f"q shape {tuple(q.shape)} != (T, H, D)=({T_tok}, {H}, {D})"
            )
        q_view = q
        # Kernel indexes q_in as dense [T,H,D]; reject non-dense (H,D) tails
        # (strided views would silently read the wrong elements).
        if q_view.stride(-1) != 1 or q_view.stride(-2) != D:
            raise ValueError(
                "3D q must be contiguous in the (H, D) inner block "
                f"(stride(-1)==1 and stride(-2)==D={D}), got stride={q_view.stride()}"
            )

    # Normalize cos/sin to 2D [max_pos, RD/2]. Accept any shape whose last
    # dim is RD/2 (DeepSeek-V4 stores [max_pos, 1, 1, RD/2]).
    if cos_cache.shape[-1] != RD // 2:
        raise ValueError(
            f"cos_cache last dim {cos_cache.shape[-1]} != RD/2 ({RD // 2})"
        )
    if sin_cache.shape != cos_cache.shape:
        raise ValueError("cos/sin shape mismatch")
    if not (cos_cache.is_contiguous() and sin_cache.is_contiguous()):
        raise ValueError("cos/sin must be contiguous")
    cos_2d = cos_cache.view(cos_cache.shape[0], RD // 2)
    sin_2d = sin_cache.view(sin_cache.shape[0], RD // 2)

    out_dtype = _fp8_const()["dtype"] if quant else torch.bfloat16
    if q_out is None:
        q_out = torch.empty((T_tok, H, D), dtype=out_dtype, device=q.device)
    if kv_out is None:
        kv_out = torch.empty((T_tok, D), dtype=out_dtype, device=kv.device)

    # Scale buffers are always bound (kernel reads the param regardless of
    # quant); allocate dummies when not quant.
    scale_torch_dtype = _TORCH_DTYPE_FOR_SCALE[scale_dtype]
    if quant:
        if q_scale is None:
            q_scale = torch.empty(
                (T_tok, H, NG), dtype=scale_torch_dtype, device=q.device
            )
        if kv_scale is None:
            kv_scale = torch.empty(
                (T_tok, NG), dtype=scale_torch_dtype, device=kv.device
            )
        q_scale_arg, kv_scale_arg = q_scale, kv_scale
    else:
        q_scale_arg = q.new_empty(1, dtype=scale_torch_dtype)
        kv_scale_arg = q.new_empty(1, dtype=scale_torch_dtype)

    # ---- Fused SWA cache-write (BF16 only) ----
    # Two modes, both writing the post-norm/rope KV row in the same launch
    # (no separate swa_write launch, no kv HBM round-trip; bf16 only):
    #   direct (swa_dest_rows): flat pool swa_kv[swa_dest_rows[t], :]
    #   paged  (swa_block_tables): flat pool swa_kv[bt[bid, pos//bs]*bs + pos%bs]
    #                              (DeepSeek-V4 #1417; repurposes the ring
    #                              scalars: swa_slot_stride=max_blocks,
    #                              swa_cache_size=block_size, swa_pos_stride=D).
    paged = swa_block_tables is not None
    kv_write = swa_kv is not None
    if kv_write and quant:
        raise ValueError("kv_write (swa_kv) is BF16 only; not supported with quant")
    if kv_write:
        if batch_id_per_token is None:
            raise ValueError("kv_write requires batch_id_per_token")
        if swa_kv.dtype != torch.bfloat16:
            raise TypeError(f"swa_kv must be bf16, got {swa_kv.dtype}")
        if not swa_kv.is_contiguous():
            raise ValueError("swa_kv must be contiguous")
        if batch_id_per_token.dim() != 1 or batch_id_per_token.dtype != torch.int32:
            raise TypeError("batch_id_per_token must be 1-D int32")
        if batch_id_per_token.shape[0] < T_tok:
            raise ValueError(
                f"batch_id_per_token len {batch_id_per_token.shape[0]} < T={T_tok}"
            )
    if kv_write and paged:
        if swa_block_size is None:
            raise ValueError("paged SWA write requires swa_block_size")
        if swa_kv.dim() != 2 or swa_kv.shape[1] != D:
            raise ValueError(
                f"paged swa_kv must be flat [num_pages, D={D}], got {tuple(swa_kv.shape)}"
            )
        if swa_block_tables.dim() != 2 or swa_block_tables.dtype != torch.int32:
            raise TypeError("swa_block_tables must be 2-D [bs, max_blocks] int32")
        # The kernel indexes the table as `bid * stride(0) + blk` and bounds
        # `blk` against stride(0). For that bound to be the table WIDTH (what
        # the C++ sibling checks, via size(1)) and not merely the distance
        # between rows, the table must be dense: a `full[:, :n]` view would
        # leave stride(0) > n, letting a far-future position read a stale
        # padding column instead of being skipped.
        if swa_block_tables.stride(1) != 1:
            raise ValueError("swa_block_tables must be contiguous in the last dim")
        if swa_block_tables.stride(0) != swa_block_tables.shape[1]:
            raise ValueError(
                "swa_block_tables must be densely packed "
                f"(stride(0)={swa_block_tables.stride(0)} != "
                f"max_blocks={swa_block_tables.shape[1]}); a padded view would "
                "widen the in-kernel block-index bound past the real table"
            )
        swa_slot_stride = swa_block_tables.stride(0)  # == max_blocks
        swa_pos_stride = swa_kv.stride(0)  # = D (flat pool row stride)
        swa_num_rows = swa_kv.shape[0]
        swa_cache_size = swa_block_size
        swa_kv_arg = swa_kv
        ssm_arg = swa_block_tables
        bid_arg = batch_id_per_token
    elif kv_write:
        if swa_dest_rows is None:
            raise ValueError("direct kv_write requires swa_dest_rows")
        if swa_kv.dim() != 2 or swa_kv.shape[1] != D:
            raise ValueError(
                f"direct swa_kv must be flat [num_rows, D={D}], "
                f"got {tuple(swa_kv.shape)}"
            )
        if swa_dest_rows.dim() != 1 or swa_dest_rows.dtype != torch.int32:
            raise TypeError("swa_dest_rows must be 1-D int32")
        if swa_dest_rows.shape[0] < T_tok:
            raise ValueError(
                f"swa_dest_rows len {swa_dest_rows.shape[0]} < T={T_tok}; it is "
                "indexed by token, not by request"
            )
        swa_slot_stride = 0
        swa_pos_stride = swa_kv.stride(0)
        swa_num_rows = swa_kv.shape[0]
        swa_cache_size = 1
        swa_kv_arg = swa_kv
        ssm_arg = swa_dest_rows
        bid_arg = batch_id_per_token
    else:
        # 1-elem dummies so the kernel param binding has valid pointers.
        swa_slot_stride = 0
        swa_pos_stride = 0
        swa_num_rows = 0
        swa_cache_size = 1
        swa_kv_arg = kv_out  # bf16 dummy
        ssm_arg = q.new_empty(1, dtype=torch.int32)
        bid_arg = q.new_empty(1, dtype=torch.int32)

    launcher = compile_flydsl_qk_norm_rope_quant(
        num_q_heads=H,
        head_dim=D,
        rope_head_dim=RD,
        quant=quant,
        group_size=G,
        scale_dtype=scale_dtype,
        q_weighted=q_weighted,
        kv_write=kv_write,
        paged=paged,
    )

    if stream is None:
        stream = torch.cuda.current_stream()
    fx_stream = Stream(stream)

    def _ptr_arg(t):
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    # HW grid Y is a 16-bit field on AMD HIP -> cap 65535 blocks/launch and
    # chunk tokens in Python. (flydsl's ``if cond: return`` does NOT early-exit
    # inside a @flyc.kernel body -- the rest still runs with bid_t past
    # num_tokens, faulting at tail blocks -- so a single folded launch is out.)
    MAX_GRID_Y = 65535
    for start in range(0, T_tok, MAX_GRID_Y):
        n = min(MAX_GRID_Y, T_tok - start)
        end = start + n
        args = (
            _ptr_arg(q_view[start:end]),
            _ptr_arg(kv[start:end]),
            q_weight_arg,
            kv_weight,
            cos_2d,
            sin_2d,
            _ptr_arg(positions[start:end]),
            _ptr_arg(q_out[start:end]),
            _ptr_arg(kv_out[start:end]),
            _ptr_arg(q_scale_arg[start:end] if quant else q_scale_arg),
            _ptr_arg(kv_scale_arg[start:end] if quant else kv_scale_arg),
            kv.stride(0),
            # swa_kv is global (absolute rows), so unsliced. `swa_index` is
            # sliced iff it is indexed by token: block tables are per request
            # and stay whole, destination rows are per token and must follow
            # the chunk exactly as positions and batch_id_per_token do.
            _ptr_arg(swa_kv_arg),
            _ptr_arg(ssm_arg[start:end] if (kv_write and not paged) else ssm_arg),
            _ptr_arg(bid_arg[start:end] if kv_write else bid_arg),
            swa_slot_stride,
            swa_pos_stride,
            swa_num_rows,
            swa_cache_size,
            n,
            fx_stream,
        )
        _run_compiled(launcher, *args)

    return q_out, kv_out, (q_scale if quant else None), (kv_scale if quant else None)
