# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused per-token RMSNorm + GPT-J RoPE + optional FP8 quant (FlyDSL).

Q + KV combined into a single kernel launch (grid Y = num_tokens, grid X =
num_q_heads + 1: bid_x ∈ [0, H) handle Q heads, bid_x == H handles KV).

Hard-coded MVP shape: D=512, RD=64, BLOCK_THREADS=64. Each block uses one
wave (64 threads × 8 bf16 = 512 elems = D), so reductions are wave-local
(shuffle_xor, no LDS, no barrier).

Layout per block:
  - thread t ∈ [0, ROPE_THREAD_LO) owns NOPE elements [t*8, t*8+8)
  - thread t ∈ [ROPE_THREAD_LO, 64) owns ROPE elements [t*8, t*8+8) which
    form ``PAIRS_PER_THREAD`` GPT-J pairs (2k, 2k+1)

GPT-J RoPE with REUSE_FREQS_FRONT_PART=True: cos/sin shape (..., RD/2),
each pair (2k, 2k+1) shares cos[k], sin[k]. Each rope-thread loads
PAIRS_PER_THREAD cos + PAIRS_PER_THREAD sin (one dwordx2 buffer load each).

FP8 fast-path uses the rstd-cancellation algebra (matches the Triton kernel
in ``atom/model_ops/v4_kernels/qk_norm_rope_maybe_quant.py``):

    scale  = abs_max(x_norm) * SQRT2 / FP8_MAX     (sqrt(2) upper bound on rope mag)
    factor = FP8_MAX / (abs_max(x_in) * SQRT2)     (rstd cancels algebraically)
    out_nope = x_in * factor              -> fp8
    out_pe   = (pe_in * factor) RoPEd     -> fp8

(For the weighted KV path the algebra carries the per-channel weight: amax
is taken over |x_in * w|, factor multiplies in w on the store side.)

Public API: ``flydsl_qk_norm_rope_quant`` (torch-friendly, allocates outputs,
binds current stream, handles strided KV and 4D cos/sin views). Internal
``compile_flydsl_qk_norm_rope_quant`` returns the cached launcher for callers
who already have all buffers and want the lowest-overhead path.
"""

# NOTE: do NOT add `from __future__ import annotations` to this file.
# PEP 563 turns all annotations into strings, which defeats flydsl's
# JitFunction._make_cache_key runtime detection:
#   is_runtime = hasattr(ann, "__get_c_pointers__")
# A string like 'fx.Int32' fails that check, so flydsl treats the
# `kv_in_row_stride` and `num_tokens` Int32 parameters as compile-time
# constants and embeds their VALUE in the cache key. Every distinct
# batch size / KV stride then triggers a fresh ~30-70ms JIT compile
# instead of hitting the in-memory CallState cache.

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch

# NOTE: ``aiter.utility.dtypes`` transitively imports ``aiter.ops.enum``,
# whose ``ActivationType = type(_ActivationType(0))`` triggers a JIT call
# into ``module_aiter_core``. That JIT module is not yet built when
# setup.py's AOT-compile pass walks the package, so importing dtypes at
# module load time crashes setup with ``KeyError: 'module_aiter_core'``.
# Defer the import until the first runtime call instead — sibling modules
# (moe_kernels._get_dtypes, gemm_kernels._get_dtypes) use the same pattern.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, vector, buffer_ops
from flydsl.expr import math as fmath
from flydsl.expr.arith import ArithValue, CmpFPredicate
from flydsl.expr.typing import T, Int32, Stream
from flydsl.expr.vector import ReductionOp
from flydsl._mlir.dialects import llvm, rocdl

from .tensor_shim import GTensor, _to_raw

# --- shape constants (V4-Pro MVP) -------------------------------------------
BLOCK_THREADS = 64  # 1 wave64

# SQRT2 has no aiter dependency, so it stays at module level.
_SQRT2 = math.sqrt(2.0)


@lru_cache(maxsize=1)
def _fp8_const():
    """Lazy-resolve fp8 algebra coefficients (per-GFX native fp8).

    ``aiter.utility.dtypes.fp8`` selects e4m3fnuz on gfx942 MI300 and
    e4m3fn on gfx950 MI355 / gfx1250. ``cvt_pk_fp8_f32`` emits bytes in
    the per-gfx native format, so FP8_MAX must track that — hardcoding
    e4m3fnuz's 240 on gfx950 would (a) clip outputs to a stricter range
    than needed and (b) leave the stored dequant scale inconsistent with
    downstream consumers reading the tensor as ``aiter.dtypes.fp8``.
    Cached on first call (kernel build / launcher call), not at import.
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


# --- supported quant-group sizes (1 × group_size block-scales) --------------
# group_size == head_dim → per-row scale (single scale per token-head).
GROUP_SIZE_OPTIONS = (32, 64, 128)

# --- scale-dtype constants --------------------------------------------------
SCALE_DTYPE_FP32 = "fp32"
SCALE_DTYPE_E8M0 = "e8m0"
SCALE_DTYPE_OPTIONS = (SCALE_DTYPE_FP32, SCALE_DTYPE_E8M0)

# E8M0 encoding (matches the convention in silu_and_mul_fq / mixed_moe_gemm).
# For e4m3fnuz (FP8_MAX = 240 ≈ 2^7.9): headroom = 7 keeps factor * amax_safe
# ≤ 2^7 = 128 < FP8_MAX with sufficient SQRT2 margin.
_E8M0_HEADROOM = 7

_TORCH_DTYPE_FOR_SCALE = {
    SCALE_DTYPE_FP32: torch.float32,
    SCALE_DTYPE_E8M0: torch.uint8,  # no native torch e8m0 dtype; reinterpret as uint8
}


# ============================================================================
# Store helpers (module-level so they're easy to reuse / unit-test)
# ============================================================================


def _store_bf16_vec_g(vals_list, g_out, row_off_elems, idx, vec):
    """Convert VEC fp32 values to a bf16 vector and store via a GTensor whose
    base is already shifted per-token. ``row_off_elems`` is this head's row
    offset within the token (i32 elements); ``idx`` is the lane id."""
    vec_t = T.vec(vec, T.f32)
    raw = [v.ir_value() if hasattr(v, "ir_value") else v for v in vals_list]
    f32v = vector.from_elements(vec_t, raw)
    bf16v = f32v.truncf(T.vec(vec, T.bf16))
    my_off = ArithValue(row_off_elems) + ArithValue(idx) * arith.constant(
        vec, type=T.i32
    )
    g_out.store(my_off, bf16v, vec_size=vec)


def _store_fp8_packed(vals_list, out_rsrc, row_base_bytes, idx, vec):
    """Pack VEC fp32 -> VEC fp8 (e4m3fnuz) via cvt_pk_fp8_f32 and store.

    Emits one ``buffer_store_dwordx2`` per thread (VEC=8 → 2 dwords = 8 bytes).

    Workaround for the e4m3fnuz NaN encoding 0x80: cvt_pk_fp8_f32 returns
    0x80 (NaN) for inputs that round to negative zero, which propagates
    through downstream attention as NaN. Clamp v ∈ (-2^-8, 0) to +0 first.
    """
    f32 = T.f32
    i32 = T.i32
    c0 = arith.constant(0.0, type=f32)
    c_neg_uf = arith.constant(-(2.0**-8), type=f32)
    c8 = arith.constant(8, type=i32)

    safe = []
    for v in vals_list:
        vv = v.ir_value() if hasattr(v, "ir_value") else v
        is_tn = arith.andi(
            arith.cmpf(CmpFPredicate.OLT, vv, c0),
            arith.cmpf(CmpFPredicate.OGT, vv, c_neg_uf),
        )
        safe.append(arith.select(is_tn, c0, vv))

    # Pack each pair (s[2i], s[2i+1]) into a packed-fp8 i32, then
    # combine 4 fp8 into one i32 via cvt_pk_fp8_f32 (lane 0 + lane 1).
    assert vec == 8, "fp8 store helper hardcoded for VEC=8"
    p0 = arith.constant(0, type=i32)
    p0 = rocdl.cvt_pk_fp8_f32(i32, safe[0], safe[1], p0, 0)
    p0 = rocdl.cvt_pk_fp8_f32(i32, safe[2], safe[3], p0, 1)
    p1 = arith.constant(0, type=i32)
    p1 = rocdl.cvt_pk_fp8_f32(i32, safe[4], safe[5], p1, 0)
    p1 = rocdl.cvt_pk_fp8_f32(i32, safe[6], safe[7], p1, 1)

    off_bytes = row_base_bytes + ArithValue(idx) * c8
    vec2_i32 = T.vec(2, i32)
    store_vec = vector.from_elements(vec2_i32, [p0, p1])
    buffer_ops.buffer_store(store_vec, out_rsrc, off_bytes, offset_is_bytes=True)


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
):
    """Build the @flyc.kernel + @flyc.jit launcher for a given config.

    All shape constants are captured via closure (NOT module globals), so two
    launchers with different (H, D, RD, group_size, scale_dtype, q_weighted)
    coexist safely. Returns the launcher.

    quant=True writes fp8 (e4m3fnuz) with one scale per ``group_size``-wide
    block of D. When ``group_size == head_dim`` the scale degenerates to
    per-row (NG=1). scale_dtype controls the stored scale encoding
    (``"fp32"`` or ``"e8m0"``).

    q_weighted=True applies a per-channel weight to Q after RMSNorm (same
    pattern as KV). Default False keeps Q weightless (V4-Pro convention).
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
    # Current MVP is hard-wired to VEC=8 (= D=512 with BLOCK_THREADS=64):
    # - ``BufferCopy128b`` atom expects 16 bytes / thread
    # - rope ``BufferCopy(64)`` atom expects 8 bytes / thread (= 4 bf16 pairs)
    # - ``_store_fp8_packed`` is hand-rolled for VEC=8 → 2 dwords
    # Supporting other D values needs the atom widths + fp8 packing pattern
    # generalised. Reject other VECs with a clear message rather than dump
    # core inside LLVM lowering.
    assert VEC == 8, (
        f"VEC={VEC} unsupported (D={D}); only D=512 / VEC=8 is implemented. "
        "Atom widths and fp8 packing assume VEC=8 — generalising requires "
        "a wider refactor."
    )

    # --- quant-group layout ------------------------------------------------
    # group_size must divide D evenly AND be a multiple of VEC (so a single
    # thread's VEC-wide slice never crosses a group boundary).
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
    # amax must NOT cross groups → only shuffles at offsets < TPG → only at
    # the last log2(TPG) loop iterations (sh_exp >= amax_start_step).
    amax_start_step = log2_block - log2_tpg

    elem_dtype = fx.BFloat16
    is_e8m0 = scale_dtype == SCALE_DTYPE_E8M0

    # Kernel name: only include flags that affect the compiled binary.
    # Default (not quant, not q_weighted) → "qk_norm_rope_H16_D512_RD64_flydsl"
    _name_parts = ["qk_norm_rope", f"H{H}", f"D{D}", f"RD{RD}"]
    if q_weighted:
        _name_parts.append("qw")
    if quant:
        _name_parts.append(f"g{group_size}")
        _name_parts.append(scale_dtype)
    _name_parts.append("flydsl")
    _kname = "_".join(_name_parts)

    @flyc.kernel(name=_kname)
    def kernel(
        q_in: fx.Tensor,  # [T, H, D]         bf16, contig (H, D)
        kv_in: fx.Tensor,  # [T, D]            bf16, may be strided
        q_weight: fx.Tensor,  # [D]               bf16 (dummy when not q_weighted)
        kv_weight: fx.Tensor,  # [D]               bf16
        cos_cache: fx.Tensor,  # [max_pos, RD/2]   bf16
        sin_cache: fx.Tensor,  # [max_pos, RD/2]   bf16
        positions: fx.Tensor,  # [T]               i64
        q_out: fx.Tensor,  # [T, H, D]         bf16 or fp8
        kv_out: fx.Tensor,  # [T, D]            bf16 or fp8
        q_scale: fx.Tensor,  # [T, H, NG]        f32 or uint8 (e8m0)
        kv_scale: fx.Tensor,  # [T, NG]           f32 or uint8 (e8m0)
        kv_in_row_stride: Int32,  # KV row stride in bf16 elements
    ):
        f32 = T.f32
        i32 = T.i32
        fm_fast = arith.FastMathFlags.fast

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

        # --- shared: load position (i64 -> i32) ---
        pos_rsrc = buffer_ops.create_buffer_resource(positions, max_size=True)
        pos_val_i64 = buffer_ops.buffer_load(pos_rsrc, bid_t, vec_width=1, dtype=T.i64)
        pos_i32 = arith.trunci(i32, pos_val_i64)

        # --- shared: cos/sin buffer tensors (used by rope-threads only) ---
        cos_buf = fx.rocdl.make_buffer_tensor(cos_cache)
        sin_buf = fx.rocdl.make_buffer_tensor(sin_cache)
        cos_row = fx.slice(cos_buf, (pos_i32, None))
        sin_row = fx.slice(sin_buf, (pos_i32, None))
        cos_div = fx.logical_divide(cos_row, rope_lay)
        sin_div = fx.logical_divide(sin_row, rope_lay)

        def wave_reduce_add(x):
            w = _to_raw(x)
            for sh_exp in range_constexpr(int(math.log2(BLOCK_THREADS))):
                off = BLOCK_THREADS // (2 << sh_exp)
                peer = _to_raw(ArithValue(w).shuffle_xor(off, BLOCK_THREADS))
                w = arith.AddFOp(w, peer, fastmath=fm_fast).result
            return w

        def emit_body(
            *,
            weighted: bool,
            x_f32_vec,
            w_f32_vec,  # None for Q
            bf16_out_g,  # GTensor with per-token shifted base (when not quant)
            bf16_out_row_off,  # i32 element offset of this head's row within token
            fp8_out_rsrc,  # (rsrc_token_shifted, row_base_bytes_within_token) when quant
            scale_rsrc,
            scale_base_off,  # base elem-offset; per-lane adds (tid // TPG)
        ):
            """Apply RMSNorm + GPT-J RoPE (+ optional FP8 quant) for the row
            held by this block. ``x_f32_vec`` and (optional) ``w_f32_vec`` are
            VEC-wide fp32 vectors already loaded by the caller."""
            x2 = x_f32_vec * x_f32_vec
            sq_local = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)

            if const_expr(quant):
                if const_expr(weighted):
                    xw = x_f32_vec * w_f32_vec
                    am_local = fmath.absf(xw).reduce(ReductionOp.MAX)
                else:
                    am_local = fmath.absf(x_f32_vec).reduce(ReductionOp.MAX)

                # Fused wave reduce: interleave sumsq-ADD and amax-MAX
                # shuffles in one loop so the LLVM scheduler can overlap the
                # two shuffle chains (each shuffle has ~4-cycle XCC latency
                # on gfx950; running them serially doubles latency).
                #
                # sumsq reduces over the FULL row (RMSNorm scope = D).
                # amax reduces over a single QUANT GROUP (TPG threads,
                # = group_size elements). Both can interleave in the loop's
                # "tail" steps where shuffle offset < TPG; earlier steps do
                # sumsq-only (amax would cross group boundaries).
                w_sq = _to_raw(sq_local)
                w_am = _to_raw(am_local)
                for sh_exp in range_constexpr(log2_block):
                    off = BLOCK_THREADS // (2 << sh_exp)
                    peer_sq = _to_raw(ArithValue(w_sq).shuffle_xor(off, BLOCK_THREADS))
                    w_sq = arith.AddFOp(w_sq, peer_sq, fastmath=fm_fast).result
                    if const_expr(sh_exp >= amax_start_step):
                        peer_am = _to_raw(
                            ArithValue(w_am).shuffle_xor(off, BLOCK_THREADS)
                        )
                        w_am = arith.maximumf(w_am, peer_am)
                sq_block = w_sq
                am_group = w_am  # per-group after partial butterfly
            else:
                sq_block = wave_reduce_add(sq_local)

            rstd = fmath.rsqrt(sq_block * (1.0 / D) + 1e-6, fastmath=fm_fast)

            if const_expr(quant):
                am_safe = arith.maximumf(am_group, arith.constant(1e-12, type=f32))

                if const_expr(is_e8m0):
                    # silu_and_mul_fq-style e8m0 encoding. amax_post incorporates
                    # rstd (per-row) and SQRT2 (post-RoPE upper bound) so the
                    # forward factor applied to x_norm (= x_in * rstd) bounds
                    # the result by 2^_E8M0_HEADROOM ≤ FP8_MAX.
                    c_sqrt2 = arith.constant(_SQRT2, type=f32)
                    amax_post = am_safe * rstd * c_sqrt2

                    amax_i32 = amax_post.bitcast(T.i32)
                    bits_up = (
                        amax_i32 + arith.constant(0x400000, type=T.i32)
                    ) & arith.constant(0xFF800000, type=T.i32)
                    exp_field = bits_up >> arith.constant(23, type=T.i32)
                    # Subtract HEADROOM only. The IEEE bias (+127) is absorbed
                    # by ``quant_exp = 254 - e8m0_biased`` below (254 = 127+127).
                    # The stored byte is the IEEE biased-exp of the dequant
                    # scale (MX e8m0 convention: byte b → scale 2^(b-127)).
                    e8m0_biased_signed = exp_field - arith.constant(
                        _E8M0_HEADROOM, type=T.i32
                    )
                    e8m0_biased = arith.maxsi(
                        e8m0_biased_signed, arith.constant(0, type=T.i32)
                    )
                    e8m0_biased = arith.minsi(
                        e8m0_biased, arith.constant(255, type=T.i32)
                    )
                    # quant_scale = 2^(127 - e8m0_biased) for x_norm. We apply
                    # to x_in directly, so absorb the per-row rstd: factor =
                    # rstd * quant_scale.
                    quant_exp = arith.constant(254, type=T.i32) - e8m0_biased
                    quant_scale = (quant_exp << arith.constant(23, type=T.i32)).bitcast(
                        T.f32
                    )
                    factor = rstd * quant_scale
                else:
                    # FP32 scale with the rstd-cancellation trick.
                    # scale_val = amax * rstd * SQRT2 / FP8_MAX  (stored)
                    # factor   = FP8_MAX / (amax * SQRT2)        (applied to x_in)
                    # The rstd factor cancels algebraically: store(out) =
                    # x_in * factor → dequant: x_norm = scale * out = x_in * rstd.
                    rcp_am = llvm.call_intrinsic(
                        f32, "llvm.amdgcn.rcp.f32", [am_safe], [], []
                    )
                    _fc = _fp8_const()
                    factor = arith.constant(_fc["max_over_sqrt2"], type=f32) * rcp_am
                    scale_val = (
                        am_safe * rstd * arith.constant(_fc["inv_max_sqrt2"], type=f32)
                    )

                # Group-leader lanes (one per quant group) write the scale.
                # Predicate: tid & (TPG-1) == 0. For TPG=64 (per-row) this is
                # `tid == 0`; for TPG<64 multiple lanes fire concurrently.
                # Per-lane scale_off = scale_base_off + (tid / TPG).
                # NOTE: tried buffer_ops.buffer_store(mask=...) for
                # predication but the mask path sets offset to 0x7FFFFFFF on
                # masked-off lanes → OOB GPU fault on gfx950. Stay with scf.if.
                group_idx = tid >> fx.Int32(log2_tpg)
                lane_in_group = tid & fx.Int32(TPG - 1)
                if lane_in_group == 0:
                    my_scale_off = scale_base_off + ArithValue(group_idx)
                    if const_expr(is_e8m0):
                        e8m0_i8 = arith.TruncIOp(T.i8, e8m0_biased).result
                        buffer_ops.buffer_store(e8m0_i8, scale_rsrc, my_scale_off)
                    else:
                        buffer_ops.buffer_store(scale_val, scale_rsrc, my_scale_off)

            is_rope = tid >= fx.Int32(ROPE_THREAD_LO)
            if is_rope:
                # ---- ROPE path: 8 elements in this thread = 4 GPT-J pairs ----
                rope_rel = tid - fx.Int32(ROPE_THREAD_LO)
                cos_vec = load_vec(cos_div, rope_rel, layout=rope_lay, atom=rope_atom)
                sin_vec = load_vec(sin_div, rope_rel, layout=rope_lay, atom=rope_atom)
                cos_f32 = cos_vec.to(fx.Float32)
                sin_f32 = sin_vec.to(fx.Float32)

                # pre-rotate values: x * factor (fp8) or x * rstd (bf16),
                # with optional kv weight.
                pe = []
                for vi in range_constexpr(VEC):
                    xi = x_f32_vec[vi]
                    if const_expr(weighted):
                        xi = xi * w_f32_vec[vi]
                    if const_expr(quant):
                        pe.append(xi * factor)
                    else:
                        pe.append(xi * rstd)

                # GPT-J pair rotate: new_2k = e*c - o*s; new_2k+1 = e*s + o*c
                rope_out = []
                for k in range_constexpr(PAIRS_PER_THREAD):
                    e = pe[2 * k]
                    o = pe[2 * k + 1]
                    c = cos_f32[k]
                    s = sin_f32[k]
                    rope_out.append(e * c - o * s)
                    rope_out.append(e * s + o * c)

                if const_expr(quant):
                    rsrc, row_base = fp8_out_rsrc
                    _store_fp8_packed(rope_out, rsrc, row_base, tid, VEC)
                else:
                    _store_bf16_vec_g(rope_out, bf16_out_g, bf16_out_row_off, tid, VEC)
            else:
                # ---- NOPE path: direct scaled store ----
                scaled = []
                for vi in range_constexpr(VEC):
                    xi = x_f32_vec[vi]
                    if const_expr(weighted):
                        xi = xi * w_f32_vec[vi]
                    if const_expr(quant):
                        scaled.append(xi * factor)
                    else:
                        scaled.append(xi * rstd)
                if const_expr(quant):
                    rsrc, row_base = fp8_out_rsrc
                    _store_fp8_packed(scaled, rsrc, row_base, tid, VEC)
                else:
                    _store_bf16_vec_g(scaled, bf16_out_g, bf16_out_row_off, tid, VEC)

        # ============ runtime dispatch on bid_x < H ============
        # Per-token byte offsets fold ``bid_t`` into the buffer descriptor
        # base so the runtime offset within each load/store stays in i32
        # range. This lets the kernel handle arbitrary T (only HW grid Y
        # limits T per launch) without the bf16 element offset overflowing
        # signed i32 at H*D = 65k+ per token.
        # Per-token byte offset, computed in index type (= platform pointer
        # width, 64-bit on AMD). GTensor.get_llvm_ptr does
        # arith.index_cast(i64, ...) on this value, which is only valid when
        # the input is index-typed. Doing the math in index avoids large
        # H*D configs (e.g. H=128 D=512 → 128 KB/token, max offset 8.6 GiB
        # at bid_t=65534) silently producing garbage if we feed i64.
        bid_t_idx = arith.index_cast(T.index, _to_raw(bid_t))
        q_tok_off_bytes = arith.MulIOp(
            bid_t_idx, arith.constant(H * D * 2, type=T.index)
        ).result

        if bid_x < fx.Int32(H):
            # ---------- Q path ----------
            head_idx = bid_x
            # Q in: per-token shifted base via GTensor. Each thread reads VEC
            # bf16 at (head_idx, tid*VEC) — element offset is bounded by H*D
            # = 64K (fits i32 with huge headroom).
            q_in_tok = GTensor(
                q_in,
                dtype=T.bf16,
                shape=(H, D),
                static_bytes_offset_i64=q_tok_off_bytes,
            )
            q_my_off = ArithValue(head_idx) * arith.constant(D, type=i32) + ArithValue(
                tid
            ) * arith.constant(VEC, type=i32)
            raw_x_vec = q_in_tok.load(q_my_off, vec_size=VEC)
            # Round-trip through rmem so the rest of emit_body (.to/.reduce)
            # sees a Fly-wrapped vec instead of a raw MLIR vec.
            q_rmem = fx.make_rmem_tensor(full_lay, elem_dtype)
            fx.memref_store_vec(raw_x_vec, q_rmem)
            x_vec = fx.memref_load_vec(q_rmem)
            x_f32 = x_vec.to(fx.Float32)

            # Optional per-channel Q weight (RMSNorm gamma for Q). Loaded only
            # when q_weighted=True; otherwise q_weight tensor is a dummy and
            # never read.
            if const_expr(q_weighted):
                qw_buf = fx.rocdl.make_buffer_tensor(q_weight)
                qw_div = fx.logical_divide(qw_buf, full_lay)
                qw_vec = load_vec(qw_div, tid)
                qw_f32 = qw_vec.to(fx.Float32)
            else:
                qw_f32 = None

            row_off_q_elems = ArithValue(head_idx) * arith.constant(D, type=i32)
            if const_expr(quant):
                # Per-token shifted base for q_out (fp8 = 1 byte/elem).
                q_tok_off_fp8 = arith.MulIOp(
                    bid_t_idx, arith.constant(H * D, type=T.index)
                ).result
                qo_g_tmp = GTensor(
                    q_out,
                    dtype=T.i8,
                    shape=(H, D),
                    static_bytes_offset_i64=q_tok_off_fp8,
                )
                qo_rsrc = qo_g_tmp.rsrc
                # row_base_bytes is now token-relative (head_idx * D bytes for fp8).
                row_base_bytes = ArithValue(head_idx) * arith.constant(D, type=i32)
                qs_rsrc = buffer_ops.create_buffer_resource(q_scale, max_size=True)
                # q_scale layout (T, H, NG) flat: bid_t * H*NG + head_idx * NG.
                # Per-lane adds group_idx inside emit_body.
                scale_base_off_q = ArithValue(bid_t) * arith.constant(
                    H * NG, type=i32
                ) + ArithValue(head_idx) * arith.constant(NG, type=i32)
                emit_body(
                    weighted=q_weighted,
                    x_f32_vec=x_f32,
                    w_f32_vec=qw_f32,
                    bf16_out_g=None,
                    bf16_out_row_off=None,
                    fp8_out_rsrc=(qo_rsrc, row_base_bytes),
                    scale_rsrc=qs_rsrc,
                    scale_base_off=scale_base_off_q,
                )
            else:
                # Per-token shifted base for q_out (bf16 = 2 bytes/elem).
                # Reuses q_tok_off_bytes computed above (the bf16 byte offset).
                qo_g = GTensor(
                    q_out,
                    dtype=T.bf16,
                    shape=(H, D),
                    static_bytes_offset_i64=q_tok_off_bytes,
                )
                emit_body(
                    weighted=q_weighted,
                    x_f32_vec=x_f32,
                    w_f32_vec=qw_f32,
                    bf16_out_g=qo_g,
                    bf16_out_row_off=row_off_q_elems,
                    fp8_out_rsrc=None,
                    scale_rsrc=None,
                    scale_base_off=None,
                )
        else:
            # ---------- KV path ----------
            # KV is often a strided slice of a wider tensor (V4: kv = split of
            # qkv_a → row stride = q_lora + head_dim). fx.slice/logical_divide
            # do not pull stride from torch.Tensor metadata, so use raw
            # buffer_ops with the explicit kv_in_row_stride argument, then
            # round-trip through an rmem tensor to get a Fly-wrapped vec that
            # the rest of emit_body (.to/.reduce/[i]) expects.
            kv_rsrc = buffer_ops.create_buffer_resource(kv_in, max_size=True)
            kv_off_elems = ArithValue(bid_t) * ArithValue(
                kv_in_row_stride
            ) + ArithValue(tid) * arith.constant(VEC, type=i32)
            kv_off_dw = kv_off_elems >> arith.constant(1, type=i32)
            vec_bf16xV = T.vec(VEC, T.bf16)
            x_raw = buffer_ops.buffer_load(
                kv_rsrc, kv_off_dw, vec_width=VEC // 2, dtype=i32
            )
            x_vec_bf16_raw = vector.bitcast(vec_bf16xV, x_raw)
            kv_rmem = fx.make_rmem_tensor(full_lay, elem_dtype)
            fx.memref_store_vec(x_vec_bf16_raw, kv_rmem)
            x_vec = fx.memref_load_vec(kv_rmem)

            kvw_buf = fx.rocdl.make_buffer_tensor(kv_weight)
            w_div = fx.logical_divide(kvw_buf, full_lay)
            w_vec = load_vec(w_div, tid)
            x_f32 = x_vec.to(fx.Float32)
            w_f32 = w_vec.to(fx.Float32)

            if const_expr(quant):
                # Per-token shifted base for kv_out (fp8 = 1 byte/elem).
                kv_tok_off_fp8 = arith.MulIOp(
                    bid_t_idx, arith.constant(D, type=T.index)
                ).result
                kvo_g_tmp = GTensor(
                    kv_out,
                    dtype=T.i8,
                    shape=(D,),
                    static_bytes_offset_i64=kv_tok_off_fp8,
                )
                kvo_rsrc = kvo_g_tmp.rsrc
                row_base_bytes = arith.constant(0, type=i32)  # already at token base
                kvs_rsrc = buffer_ops.create_buffer_resource(kv_scale, max_size=True)
                # kv_scale layout (T, NG) flat: bid_t * NG. Per-lane adds
                # group_idx inside emit_body.
                scale_base_off_kv = ArithValue(bid_t) * arith.constant(NG, type=i32)
                emit_body(
                    weighted=True,
                    x_f32_vec=x_f32,
                    w_f32_vec=w_f32,
                    bf16_out_g=None,
                    bf16_out_row_off=None,
                    fp8_out_rsrc=(kvo_rsrc, row_base_bytes),
                    scale_rsrc=kvs_rsrc,
                    scale_base_off=scale_base_off_kv,
                )
            else:
                # Per-token shifted base for kv_out (bf16 = 2 bytes/elem).
                kv_tok_off_bf16 = arith.MulIOp(
                    bid_t_idx, arith.constant(D * 2, type=T.index)
                ).result
                kvo_g = GTensor(
                    kv_out,
                    dtype=T.bf16,
                    shape=(D,),
                    static_bytes_offset_i64=kv_tok_off_bf16,
                )
                emit_body(
                    weighted=True,
                    x_f32_vec=x_f32,
                    w_f32_vec=w_f32,
                    bf16_out_g=kvo_g,
                    bf16_out_row_off=arith.constant(0, type=i32),
                    fp8_out_rsrc=None,
                    scale_rsrc=None,
                    scale_base_off=None,
                )

    # Name the launcher explicitly so the flydsl disk cache directory becomes
    # `~/.flydsl/cache/launch_qk_norm_rope_quant_<hash>/` instead of the
    # generic `launcher_<hash>/`, which collides visually with every other
    # @flyc.jit function in the codebase.
    @flyc.jit
    def launch_qk_norm_rope_quant(
        q_in: fx.Tensor,
        kv_in: fx.Tensor,
        q_weight: fx.Tensor,
        kv_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        positions: fx.Tensor,
        q_out: fx.Tensor,
        kv_out: fx.Tensor,
        q_scale: fx.Tensor,
        kv_scale: fx.Tensor,
        kv_in_row_stride: fx.Int32,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        idx_tokens = arith.index_cast(T.index, _to_raw(num_tokens))
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

# Empirically (sweep on MI355X V4-Pro shape) ``waves_per_eu=8, fast_fp_math
# =True, unsafe_fp_math=True`` gives the best occupancy at small/mid T with
# no measurable regression at large T. See logs_claude/sweep_hints.py.
_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 8,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


# Bounded to keep parity with sibling flydsl ops (see fmha_kernels._get_kernel).
# In V4-Pro deployment only a handful of (H, D, RD, quant, group_size,
# scale_dtype, q_weighted) combinations actually fire, so 32 leaves wide
# headroom while preventing unbounded growth from sweep/test enumeration.
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
):
    """Compile (and cache) the launcher for a given config.

    Cache key includes (H, D, RD, quant, group_size, scale_dtype, q_weighted).
    Returns the @flyc.jit launcher; call it directly if you've already
    allocated outputs and want to avoid the per-call torch-side overhead in
    ``flydsl_qk_norm_rope_quant``.
    """
    launcher = _build_kernel(
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        quant=quant,
        group_size=group_size,
        scale_dtype=scale_dtype,
        q_weighted=q_weighted,
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
    q_weight: Optional[torch.Tensor] = None,
    quant: bool = False,
    quant_group_size: Optional[int] = None,
    scale_dtype: str = SCALE_DTYPE_FP32,
    q_out: Optional[torch.Tensor] = None,
    kv_out: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    kv_scale: Optional[torch.Tensor] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Fused RMSNorm + GPT-J RoPE + optional FP8 quant for Q and KV in one launch.

    Args:
        q: Q activations, shape ``[T, H*D]`` (will be ``.view``-reshaped to
            ``[T, H, D]``) or already ``[T, H, D]``. Must be bf16 and contig
            in the (H, D) inner dims.
        kv: KV pre-RoPE/norm, shape ``[T, D]``, bf16. May be a strided view
            of a wider tensor (e.g. the KV half of a ``torch.split``); the
            row stride is read from ``kv.stride(0)`` and passed through.
        kv_weight: per-channel RMSNorm weight for KV, shape ``[D]``, bf16.
        cos_cache, sin_cache: RoPE cos/sin, last dim ``rope_head_dim/2``,
            any leading shape that ``view``-reshapes to ``[max_pos, RD/2]``
            (e.g. ``[max_pos, 1, 1, RD/2]`` from DeepSeek-V4). bf16.
        positions: per-token RoPE position indices, shape ``[T]``, int64.
        num_q_heads: H (per-rank Q head count).
        head_dim: D (per-head hidden dim).
        rope_head_dim: RD (size of the RoPE-rotated tail; first D-RD elements
            are passed through as NOPE).
        q_weight: optional per-channel RMSNorm weight for Q, shape ``[D]``,
            bf16. When ``None`` (default, V4-Pro), Q is weightless. When
            provided, applied just like ``kv_weight``.
        quant: if True, write fp8 in the per-GFX native encoding selected by
            ``aiter.dtypes.fp8`` (typically ``e4m3fnuz`` on gfx942 and
            ``e4m3fn`` on gfx950); else bf16.
        quant_group_size: width of the 1×G scale block. Defaults to
            ``head_dim`` (per-row scale). Any value that divides ``head_dim``
            is accepted by the wrapper; the underlying kernel currently
            requires ``G`` to be a multiple of ``head_dim // BLOCK_THREADS``
            (= 8 for V4-Pro at D=512, BLOCK_THREADS=64), so the typical
            sub-row choices are ``{32, 64, 128}``.
        scale_dtype: ``"fp32"`` (default) or ``"e8m0"`` (MX-format uint8).
        q_out, kv_out, q_scale, kv_scale: output buffers; allocated if None.
            ``q_out`` shape ``[T, H, D]``, ``kv_out`` shape ``[T, D]``,
            ``q_scale`` shape ``[T, H, NG]``, ``kv_scale`` shape ``[T, NG]``
            where ``NG = head_dim // quant_group_size``. Scale dtype is
            ``torch.float32`` for ``scale_dtype="fp32"``, ``torch.uint8``
            for ``"e8m0"`` (reinterpret as e8m0 downstream).
        stream: torch CUDA stream to launch on. Defaults to the current
            stream. **Must NOT be left at ``fx.Stream(None)`` default in
            caller code unless you accept the default-stream pitfall under
            CUDA-graph capture** (NULL stream → empty captured graph).

    Returns:
        (q_out, kv_out, q_scale_or_None, kv_scale_or_None)
        Scales are ``None`` when ``quant=False``.
    """
    # Validate user-facing inputs with raise (not assert) so the checks are
    # not stripped under ``python -O``. Internal codegen invariants inside
    # _build_kernel/_store_*_vec_g remain as asserts on purpose.
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must be bf16, got {q.dtype}")
    if kv.dtype != torch.bfloat16:
        raise TypeError(f"kv must be bf16, got {kv.dtype}")
    if kv_weight.dtype != torch.bfloat16:
        raise TypeError(f"kv_weight must be bf16, got {kv_weight.dtype}")
    if kv.stride(-1) != 1:
        raise ValueError(f"kv must be dense in the last dim, stride={kv.stride()}")
    # The KV inner loop casts bf16 vectors to dword (i32) and computes the
    # buffer-load offset as ``(row * kv.stride(0) + tid * VEC) >> 1``. That
    # ``>> 1`` is only correct when the byte offset is dword-aligned for every
    # row, which requires the row stride (in bf16 elements) to be even.
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
    # Kernel always reads the q_weight parameter; pass a 1-elem dummy when
    # q_weighted=False (the const_expr gate inside the kernel ensures the
    # load is dead-code-eliminated, but the parameter binding still needs a
    # valid tensor).
    q_weight_arg = q_weight if q_weighted else kv_weight

    # Normalize Q to [T, H, D] (the kernel expects 3D).
    if q.dim() == 2:
        if q.shape[1] != H * D:
            raise ValueError(f"q shape {tuple(q.shape)} != [T, H*D={H*D}]")
        if not q.is_contiguous():
            raise ValueError("2D q must be contiguous to .view as [T,H,D]")
        q_view = q.view(T_tok, H, D)
    else:
        if q.dim() != 3 or q.shape != (T_tok, H, D):
            raise ValueError(
                f"q shape {tuple(q.shape)} != (T, H, D)=({T_tok}, {H}, {D})"
            )
        q_view = q
        # The kernel linearly indexes q_in as if it were dense [T,H,D] with
        # the (H,D) inner block contiguous. Strided views (e.g. a slice of a
        # wider tensor along an inner axis) would silently read the wrong
        # elements, so reject anything that is not dense in the (H,D) tail.
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

    # Scale buffers must always be passed to the launcher (the kernel reads
    # the parameter regardless of QUANT_*). Allocate dummies when not quant.
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

    launcher = compile_flydsl_qk_norm_rope_quant(
        num_q_heads=H,
        head_dim=D,
        rope_head_dim=RD,
        quant=quant,
        group_size=G,
        scale_dtype=scale_dtype,
        q_weighted=q_weighted,
    )

    if stream is None:
        stream = torch.cuda.current_stream()
    fx_stream = Stream(stream)

    # HW grid Y is a 16-bit field on AMD HIP → cap 65535 blocks/launch. The
    # kernel uses per-token GTensor base-shift so each chunk's resource span
    # is small (just the chunk's tokens), but the grid Y dim itself is HW-
    # bounded. We tried folding T across gridY+gridZ to do a single launch,
    # but flydsl's ``if cond: return`` does NOT actually early-exit inside a
    # @flyc.kernel body (the rest of the kernel still runs with bid_t past
    # num_tokens, causing OOB memory faults at tail blocks). Wrapping the
    # full kernel body in a positive ``if bid_t < num_tokens:`` works but
    # requires indenting ~400 lines. The Python-loop chunk is the pragmatic
    # solution — overhead is one launch per 65k tokens.
    MAX_GRID_Y = 65535
    for start in range(0, T_tok, MAX_GRID_Y):
        n = min(MAX_GRID_Y, T_tok - start)
        end = start + n
        launcher(
            q_view[start:end],
            kv[start:end],
            q_weight_arg,
            kv_weight,
            cos_2d,
            sin_2d,
            positions[start:end],
            q_out[start:end],
            kv_out[start:end],
            q_scale_arg[start:end] if quant else q_scale_arg,
            kv_scale_arg[start:end] if quant else kv_scale_arg,
            kv.stride(0),
            n,
            stream=fx_stream,
        )

    return q_out, kv_out, (q_scale if quant else None), (kv_scale if quant else None)
