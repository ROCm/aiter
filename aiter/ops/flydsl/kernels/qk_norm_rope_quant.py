# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused per-token RMSNorm + GPT-J RoPE + optional FP8 quant (FlyDSL).

wave64 (gfx942/gfx950) and wave32 (gfx1250) in one file, dispatched by get_gfx().
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
from flydsl.expr.arith import FastMathFlags
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Int32, ReductionOp, Stream, T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops

# JIT-free MX-format int mirrors (keep module import JIT-free; aiter.utility
# .dtypes transitively JITs module_aiter_core, unbuilt during setup.py AOT walk).
from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale
from aiter.utility.mx_types import (
    MX_DEFAULT_ROUND_MODE as _DEFAULT_MODE,
)
from aiter.utility.mx_types import (
    MxDtypeInt as _D,
)

from .tensor_shim import GTensor, _run_compiled


def _imin(a, b):
    return (a < b).select(a, b)


def _imax(a, b):
    return (a > b).select(a, b)


def _idiv(a, b):
    """Truncating integer divide. Signed ``//`` maps to arith.floordivsi, a
    longer expansion; every dividend here is provably non-negative, so an
    unsigned divide (divui) is both correct and cheaper."""
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


_STATIC_ADAPTOR_CACHE = {}
_STATIC_ADAPTOR_CACHE_MAX = 64


def _cached_from_dlpack(t: torch.Tensor):
    """Adaptor cache for STATIC tensors only (weights, cos/sin).

    The cached adaptor holds a DLPack reference to ``t``, so a cached entry
    keeps that tensor's allocation alive until the entry is evicted. That is
    free for model-lifetime tensors, whose ``data_ptr`` is stable so the cache
    actually hits -- but it is pure loss for a per-call activation: the pointer
    differs every call (0% hit rate) while up to ``_STATIC_ADAPTOR_CACHE_MAX``
    of them stay pinned, which under serving is tens of GiB of allocator blocks
    the rest of the model can no longer reuse. Activations must therefore build
    their adaptor uncached, via ``flyc.from_dlpack`` directly.
    """
    key = (
        int(t.data_ptr()),
        str(t.device),
        str(t.dtype),
        tuple(t.shape),
        tuple(t.stride()),
        int(t.storage_offset()),
    )
    cached = _STATIC_ADAPTOR_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_STATIC_ADAPTOR_CACHE) >= _STATIC_ADAPTOR_CACHE_MAX:
        _STATIC_ADAPTOR_CACHE.clear()
    adaptor = flyc.from_dlpack(t)
    _STATIC_ADAPTOR_CACHE[key] = adaptor
    return adaptor


BLOCK_THREADS_W64 = 64  # 1 wave64 (gfx942 / gfx950)
BLOCK_THREADS_W32 = 32  # 1 wave32 (CDNA5 / gfx1250)
ROWS_PER_WG = 32
ROWS_PER_WG_SMALL = 4
SMALL_T_THRESHOLD = 96
USE_TDM_PREFILL = True
TDM_MIN_ROWS = 32768
# gfx1250 cache-policy bit 0 reduces the T=512 BF16 output-store drain; larger
# T already reaches steady-state HBM bandwidth and keeps the default policy.
TDM_DECODE_STORE_CACHE_MODIFIER = 1


_SQRT2 = math.sqrt(2.0)


@lru_cache(maxsize=1)
def _fp8_const():
    """Lazy-resolve per-GFX fp8 coefficients. Cached; not resolved at import."""
    from aiter.utility import dtypes as aiter_dtypes

    fp8_dtype = aiter_dtypes.fp8
    fp8_max = float(torch.finfo(fp8_dtype).max)
    return {
        "dtype": fp8_dtype,
        "max": fp8_max,
        "max_over_sqrt2": fp8_max / _SQRT2,  # forward-factor coefficient
        "inv_max_sqrt2": _SQRT2 / fp8_max,  # stored-scale coefficient
    }


SCALE_DTYPE_FP32 = "fp32"
SCALE_DTYPE_E8M0 = "e8m0"
SCALE_DTYPE_OPTIONS = (SCALE_DTYPE_FP32, SCALE_DTYPE_E8M0)

_TORCH_DTYPE_FOR_SCALE = {
    SCALE_DTYPE_FP32: torch.float32,
    SCALE_DTYPE_E8M0: torch.uint8,  # no native torch e8m0 dtype; reinterpret as uint8
}


# Empirically best occupancy on MI355X V4-Pro (sweep): waves_per_eu=8 +
# fast/unsafe fp math, no regression at large T.
_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 8,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


def _bf16_row_view(base_i64, num_elems, nbytes):
    """Build a (1, num_elems) bf16 buffer-tensor view at a folded base ptr."""
    pt = fx.PointerType.get(
        fx.BFloat16.ir_type, address_space=fx.AddressSpace.Global, alignment=2
    )
    view = fx.make_view(
        fx.inttoptr(pt, base_i64), fx.make_layout((1, num_elems), (num_elems, 1))
    )
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=fx.Int64(nbytes))


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
    fx.copy(atom, fx.slice(buf, (idx, None)), r)
    return fx.memref_load_vec(r)[0]


def _scalar_store(base_i64, idx, val, fx_dt, copy_bits):
    """Store one ``fx_dt`` value at element index ``idx`` into base ``base_i64``."""
    buf = _scalar_view(base_i64, fx_dt)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy(copy_bits), fx_dt)
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx_dt)
    fx.memref_store_vec(fx.Vector.from_elements([val], dtype=fx_dt), r)
    fx.copy(atom, r, fx.slice(buf, (idx, None)))


def _store_bf16_tiled(vals_list, p_dst, copy, vec):
    """Convert VEC fp32 → bf16 and tiled-copy to ``p_dst``."""
    f32v = fx.Vector.from_elements(vals_list, dtype=fx.Float32)
    bf16v = f32v.truncf(T.vec(vec, T.bf16))
    frag = fx.make_fragment_like(p_dst)
    fx.memref_store_vec(bf16v, frag)
    fx.copy(copy, frag, p_dst)


def _store_bf16_vec(vals, out_rsrc, row_base_bytes, idx, vec, cache_modifier=0):
    """Convert VEC fp32 → bf16 dwords and raw buffer_store (split for VEC>8)."""
    if isinstance(vals, (list, tuple)):
        vals = fx.Vector.from_elements(vals, dtype=fx.Float32)
    bf16v = vals.to(fx.BFloat16)

    dwords = vec // 2
    bf16_as_i32 = bf16v.bitcast(fx.Int32)
    off_bytes = row_base_bytes + idx * (vec * 2)

    if const_expr(dwords <= 4):
        buffer_ops.buffer_store(
            bf16_as_i32.ir_value(),
            out_rsrc,
            off_bytes,
            cache_modifier=cache_modifier,
            offset_is_bytes=True,
        )
    else:
        lo = fx.Vector.from_elements([bf16_as_i32[i] for i in range(4)], dtype=fx.Int32)
        hi = fx.Vector.from_elements(
            [bf16_as_i32[i] for i in range(4, dwords)], dtype=fx.Int32
        )
        buffer_ops.buffer_store(
            lo.ir_value(),
            out_rsrc,
            off_bytes,
            cache_modifier=cache_modifier,
            offset_is_bytes=True,
        )
        buffer_ops.buffer_store(
            hi.ir_value(),
            out_rsrc,
            off_bytes + 16,
            cache_modifier=cache_modifier,
            offset_is_bytes=True,
        )


def _store_fp8_packed_w64(
    vals_list, out_base_i64, row_base_bytes, idx, vec, *, skip_fnuz_clamp=False
):
    """Pack VEC fp32 → fp8 and buffer-store 8 bytes (2 packed dwords) per thread."""
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

    base = out_base_i64 + fx.Int64(row_base_bytes)
    packed_i64 = fx.Vector.from_elements([p0, p1], dtype=fx.Int32).bitcast(fx.Int64)[0]
    _scalar_store(base, idx, fx.Int64(packed_i64), fx.Int64, 64)


def _store_fp8_packed_w32(
    vals_list, out_rsrc, row_base_bytes, idx, vec, *, skip_fnuz_clamp=False
):
    """Pack VEC fp32 → fp8 and raw buffer_store (VEC in {8, 16})."""
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

    n_dwords = vec // 4
    assert n_dwords in (2, 4), f"VEC={vec} -> n_dwords={n_dwords} unsupported"
    dword_list = []
    for dw_idx in range_constexpr(n_dwords):
        base = dw_idx * 4
        pk = fx.Int32(0).ir_value()
        pk = fx.rocdl.cvt_pk_fp8_f32(i32, safe[base + 0], safe[base + 1], pk, 0)
        pk = fx.rocdl.cvt_pk_fp8_f32(i32, safe[base + 2], safe[base + 3], pk, 1)
        dword_list.append(pk)

    off_bytes = row_base_bytes + idx * vec
    store_vec = fx.Vector.from_elements(dword_list, dtype=fx.Int32)
    buffer_ops.buffer_store(
        store_vec.ir_value(), out_rsrc, off_bytes, offset_is_bytes=True
    )


@lru_cache(maxsize=32)
def _build_kernel_w64(
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
    BLOCK_THREADS = BLOCK_THREADS_W64  # 1 wave64
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
    assert VEC == 8, (
        f"VEC={VEC} unsupported (D={D}); only D=512 / VEC=8 is implemented. "
        "Atom widths and fp8 packing assume VEC=8 -- generalising requires "
        "a wider refactor."
    )

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
    amax_start_step = log2_block - log2_tpg

    elem_dtype = fx.BFloat16
    is_e8m0 = scale_dtype == SCALE_DTYPE_E8M0

    _is_fnuz = _fp8_const()["dtype"] == torch.float8_e4m3fnuz
    _fp8_mx_dtype = _D.FP8_E4M3_FNUZ if _is_fnuz else _D.FP8_E4M3

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
            fx.copy(atom, fx.slice(div_tensor, (None, idx)), r)
            return fx.memref_load_vec(r)

        bid_x = fx.block_idx.x  # 0..H-1 (Q head) or H (KV)
        bid_t = fx.block_idx.y  # token id (chunked at MAX_GRID_Y per launch)
        tid = fx.thread_idx.x

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
                    amax_post = (am_safe * rstd * _SQRT2).ir_value()

                    e8m0_biased = fx.Int32(
                        emit_mx_e8m0_scale(
                            amax_post, mode=_DEFAULT_MODE, dtype=_fp8_mx_dtype
                        )
                    )
                    quant_exp = fx.Int32(254) - e8m0_biased
                    quant_scale = (quant_exp << fx.Int32(23)).bitcast(fx.Float32)
                    factor = rstd * quant_scale
                    scale_to_store = e8m0_biased.to(fx.Int8)
                    scale_store_dt, scale_store_bits = fx.Int8, 8
                else:
                    rcp_am = fx.Float32(fx.rocdl.rcp(f32, am_safe))
                    _fc = _fp8_const()
                    factor = rcp_am * _fc["max_over_sqrt2"]
                    scale_to_store = am_safe * rstd * _fc["inv_max_sqrt2"]
                    scale_store_dt, scale_store_bits = fx.Float32, 32

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

            scaled = []
            for vi in range_constexpr(VEC):
                xi = x_f32_vec[vi]
                if const_expr(weighted):
                    xi = xi * w_f32_vec[vi]
                if const_expr(quant):
                    scaled.append(xi * factor)
                else:
                    scaled.append(xi * rstd)

            out_rmem = fx.make_rmem_tensor(full_lay, fx.Float32)
            scaled_vec = fx.Vector.from_elements(scaled, dtype=fx.Float32)
            fx.memref_store_vec(scaled_vec, out_rmem)

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
                    rope_elems.append(e * c - o * s)
                    rope_elems.append(e * s + o * c)
                rotated_vec = fx.Vector.from_elements(rope_elems, dtype=fx.Float32)
                fx.memref_store_vec(rotated_vec, out_rmem)

            final = fx.memref_load_vec(out_rmem)
            final_list = [final[i] for i in range(VEC)]

            if const_expr(quant):
                out_base, row_base = fp8_out_base
                _store_fp8_packed_w64(
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

        if bid_x < fx.Int32(H):
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
                qo_base = fx.Int64(ptrtoint(q_out)) + fx.Int64(bid_t) * fx.Int64(H * D)
                row_base_bytes = head_idx * D
                qs_base = fx.Int64(ptrtoint(q_scale))
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
            kv_in_base = fx.Int64(ptrtoint(kv_in)) + fx.Int64(bid_t) * (
                fx.Int64(kv_in_row_stride) * fx.Int64(2)
            )
            kv_in_row = _bf16_row_view(kv_in_base, D, D * 2)
            kv_in_part = _row_part_S(kv_in_row)
            kv_frag = fx.make_fragment_like(kv_in_part)
            fx.copy(row_copy, kv_in_part, kv_frag)
            kv_rmem = fx.make_rmem_tensor(full_lay, elem_dtype)
            fx.memref_store_vec(fx.memref_load_vec(kv_frag), kv_rmem)
            x_vec = fx.memref_load_vec(kv_rmem)

            kvw_buf = fx.rocdl.make_buffer_tensor(kv_weight)
            w_div = fx.logical_divide(kvw_buf, full_lay)
            w_vec = load_vec(w_div, tid)
            x_f32 = x_vec.to(fx.Float32)
            w_f32 = w_vec.to(fx.Float32)

            if const_expr(quant):
                kvo_base = fx.Int64(ptrtoint(kv_out)) + fx.Int64(bid_t) * fx.Int64(D)
                row_base_bytes = 0
                kvs_base = fx.Int64(ptrtoint(kv_scale))
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
                kvo_base = fx.Int64(ptrtoint(kv_out)) + fx.Int64(bid_t) * fx.Int64(
                    D * 2
                )
                kvo_part = _row_part_D(_bf16_row_view(kvo_base, D, D * 2))

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
                        dest = fx.Int32(
                            _scalar_load(
                                fx.Int64(ptrtoint(swa_index)),
                                bid_t,
                                fx.Int32,
                                32,
                            )
                        )
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

    launch_qk_norm_rope_quant.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launch_qk_norm_rope_quant


@lru_cache(maxsize=32)
def _build_kernel_w32(
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
    rows_per_wg: int = ROWS_PER_WG,
):
    """Build the wave32 @flyc.kernel + @flyc.jit launcher for a given config."""
    BLOCK_THREADS = BLOCK_THREADS_W32  # 1 wave32
    H = num_q_heads
    D = head_dim
    RD = rope_head_dim
    NOPE = D - RD
    VEC = D // BLOCK_THREADS
    ROPE_THREAD_LO = NOPE // VEC
    PAIRS_PER_THREAD = VEC // 2
    ROWS_PER_WG = rows_per_wg

    assert (
        D % BLOCK_THREADS == 0
    ), f"D={D} must be divisible by BLOCK_THREADS={BLOCK_THREADS}"
    assert NOPE % VEC == 0, f"NOPE={NOPE} must be divisible by VEC={VEC}"
    assert RD % 2 == 0, "rope_head_dim must be even (GPT-J pair layout)"
    assert RD % VEC == 0, f"RD={RD} must be divisible by VEC={VEC}"
    assert VEC in (2, 4, 8, 16), (
        f"VEC={VEC} unsupported (D={D}, BLOCK_THREADS={BLOCK_THREADS}); "
        "supported set: {2, 4, 8, 16}."
    )

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
    amax_start_step = log2_block - log2_tpg

    elem_dtype = fx.BFloat16
    is_e8m0 = scale_dtype == SCALE_DTYPE_E8M0

    _fp8_mx_dtype = _D.FP8_E4M3

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
    if ROWS_PER_WG > 1:
        _name_parts.append(f"r{ROWS_PER_WG}")
    _name_parts.append("w32")
    _name_parts.append("flydsl")
    _kname = "_".join(_name_parts)

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_THREADS, ROWS_PER_WG, 1])
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
        num_tokens: Int32,  # valid tokens in this launch chunk (for tail clamp)
    ):
        f32 = T.f32
        i32 = T.i32

        full_lay = fx.make_layout(VEC, 1)
        if const_expr(VEC <= 8):
            full_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 16)

            def _load_weight_tensor(weight_tensor, tid_val):
                """Load VEC bf16 from a 1D weight fx.Tensor at tid*VEC."""
                wbuf = fx.rocdl.make_buffer_tensor(weight_tensor)
                wdiv = fx.logical_divide(wbuf, full_lay)
                r = fx.make_rmem_tensor(full_lay, elem_dtype)
                fx.copy(full_atom, fx.slice(wdiv, (None, tid_val)), r)
                return fx.memref_load_vec(r)

        else:
            half_lay = fx.make_layout(VEC // 2, 1)
            half_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 16)

            def _load_weight_tensor(weight_tensor, tid_val):
                wbuf = fx.rocdl.make_buffer_tensor(weight_tensor)
                wdiv = fx.logical_divide(wbuf, half_lay)
                r0 = fx.make_rmem_tensor(half_lay, elem_dtype)
                r1 = fx.make_rmem_tensor(half_lay, elem_dtype)
                fx.copy(half_atom, fx.slice(wdiv, (None, tid_val * 2)), r0)
                fx.copy(half_atom, fx.slice(wdiv, (None, tid_val * 2 + 1)), r1)
                v0 = fx.memref_load_vec(r0).to(fx.Float32)
                v1 = fx.memref_load_vec(r1).to(fx.Float32)
                combined = v0.shuffle(v1, list(range(VEC)))
                rmem = fx.make_rmem_tensor(full_lay, fx.Float32)
                fx.memref_store_vec(combined, rmem)
                return fx.memref_load_vec(rmem)

        def _load_bf16_raw(rsrc, off_dw):
            """Load VEC bf16 from a raw buffer resource at dword offset.
            Returns list of VEC f32 scalars. Splits into dwordx4 chunks
            when VEC > 8 (dwords > 4)."""
            dwords = VEC // 2
            out = []
            if const_expr(dwords <= 4):
                raw = buffer_ops.buffer_load(rsrc, off_dw, vec_width=dwords, dtype=i32)
                vec_bf16 = fx.Vector(raw).bitcast(fx.BFloat16)
                for i in range_constexpr(VEC):
                    out.append(vec_bf16[i].to(fx.Float32))
            else:
                half_dw = 4
                half_bf16 = half_dw * 2  # 8 bf16 per chunk
                for chunk in range_constexpr(dwords // half_dw):
                    r = buffer_ops.buffer_load(
                        rsrc,
                        off_dw + (chunk * half_dw),
                        vec_width=half_dw,
                        dtype=i32,
                    )
                    vbf16 = fx.Vector(r).bitcast(fx.BFloat16)
                    for i in range_constexpr(half_bf16):
                        out.append(vbf16[i].to(fx.Float32))
            return out

        bid_x = fx.block_idx.x  # 0..H-1 (Q head) or H (KV)
        bid_t = fx.block_idx.y  # workgroup index along the token-chunk dim
        tid = fx.thread_idx.x
        tid_y = fx.thread_idx.y  # wave within workgroup -> token selector

        tok = _imin(bid_t * ROWS_PER_WG + tid_y, num_tokens - 1)
        bid_t = tok  # all downstream token offsets use the clamped token
        bid_t_idx = fx.Int64(tok)

        def _ptr_buffer_resource(ptr, num_records_bytes=None):
            addr = fx.ptrtoint(ptr)
            addr_i64 = fx.Int64(addr)
            if num_records_bytes is None:
                return buffer_ops.create_buffer_resource_from_addr(addr_i64)
            return buffer_ops.create_buffer_resource_from_addr(
                addr_i64, num_records_bytes=num_records_bytes
            )

        pos_rsrc = _ptr_buffer_resource(positions)
        pos_val_i64 = buffer_ops.buffer_load(pos_rsrc, bid_t, vec_width=1, dtype=T.i64)
        pos_i32 = fx.Int32(pos_val_i64.trunci(i32))

        rope_lay = fx.make_layout(PAIRS_PER_THREAD, 1)
        rope_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(PAIRS_PER_THREAD * 16), 16)
        cos_buf = fx.rocdl.make_buffer_tensor(cos_cache)
        sin_buf = fx.rocdl.make_buffer_tensor(sin_cache)
        cos_row = fx.slice(cos_buf, (pos_i32, None))
        sin_row = fx.slice(sin_buf, (pos_i32, None))
        cos_div = fx.logical_divide(cos_row, rope_lay)
        sin_div = fx.logical_divide(sin_row, rope_lay)

        def wave_reduce_add(x):
            w = fx.Float32(x)
            for sh_exp in range_constexpr(log2_block):
                off = BLOCK_THREADS // (2 << sh_exp)
                w = w + w.shuffle_xor(off, BLOCK_THREADS)
            return w

        def emit_body(
            *,
            weighted: bool,
            x_f32_vec,
            w_f32_vec,  # None for Q
            bf16_out_rsrc,  # raw buffer resource for bf16 store (when not quant)
            bf16_out_row_base_bytes,  # byte offset within token for this head
            fp8_out_rsrc,  # (rsrc_token_shifted, row_base_bytes_within_token) when quant
            scale_rsrc,
            scale_base_off,  # base elem-offset; per-lane adds (tid // TPG)
            swa_out_rsrc=None,  # raw buffer resource for SWA scatter (kv_write only)
            swa_out_row_base_bytes=None,  # byte offset for SWA target row
            do_swa=None,  # i1 predicate (batch_id >= 0); None when no kv_write
        ):
            """Apply RMSNorm + GPT-J RoPE (+ optional FP8 quant) for the row
            held by this block. ``x_f32_vec`` and (optional) ``w_f32_vec`` are
            VEC-wide fp32 vectors already loaded by the caller."""
            is_rope_t = tid >= fx.Int32(ROPE_THREAD_LO)
            rope_rel = _imax(tid - fx.Int32(ROPE_THREAD_LO), fx.Int32(0))
            cos_rmem = fx.make_rmem_tensor(rope_lay, elem_dtype)
            sin_rmem = fx.make_rmem_tensor(rope_lay, elem_dtype)
            fx.copy(rope_atom, fx.slice(cos_div, (None, rope_rel)), cos_rmem)
            fx.copy(rope_atom, fx.slice(sin_div, (None, rope_rel)), sin_rmem)
            cos_raw = fx.memref_load_vec(cos_rmem)
            sin_raw = fx.memref_load_vec(sin_rmem)

            x2 = x_f32_vec * x_f32_vec
            sq_local = x2.reduce(ReductionOp.ADD)

            if const_expr(quant):
                if const_expr(weighted):
                    xw = x_f32_vec * w_f32_vec
                    am_local = fmath.absf(xw).reduce(ReductionOp.MAX)
                else:
                    am_local = fmath.absf(x_f32_vec).reduce(ReductionOp.MAX)

                w_sq = fx.Float32(sq_local)
                w_am = fx.Float32(am_local)
                for sh_exp in range_constexpr(log2_block):
                    off = BLOCK_THREADS // (2 << sh_exp)
                    w_sq = w_sq + w_sq.shuffle_xor(off, BLOCK_THREADS)
                    if const_expr(sh_exp >= amax_start_step):
                        w_am = w_am.maximumf(w_am.shuffle_xor(off, BLOCK_THREADS))
                sq_block = w_sq
                am_group = w_am
            else:
                sq_block = wave_reduce_add(sq_local)

            rstd = fmath.rsqrt(sq_block * (1.0 / D) + 1e-6)

            if const_expr(quant):
                am_safe = am_group.maximumf(fx.Float32(1e-12))

                # Both the norm factor and the stored scale are computed here,
                # unconditionally. The runtime lane guard below must contain only
                # the store: a value produced inside an scf.if has to be yielded,
                # and a const_expr-only branch leaves it undefined on the other
                # path, which the AST rewriter rejects.
                if const_expr(is_e8m0):
                    amax_post = (am_safe * rstd * _SQRT2).ir_value()
                    e8m0_biased = fx.Int32(
                        emit_mx_e8m0_scale(
                            amax_post, mode=_DEFAULT_MODE, dtype=_fp8_mx_dtype
                        )
                    )
                    quant_scale = ((fx.Int32(254) - e8m0_biased) << 23).bitcast(
                        fx.Float32
                    )
                    factor = rstd * quant_scale
                    scale_store = e8m0_biased.to(fx.Int8)
                else:
                    rcp_am = fx.Float32(fx.rocdl.rcp(f32, am_safe))
                    _fc = _fp8_const()
                    factor = fx.Float32(_fc["max_over_sqrt2"]) * rcp_am
                    scale_store = am_safe * rstd * fx.Float32(_fc["inv_max_sqrt2"])

                group_idx = tid >> log2_tpg
                if (tid & (TPG - 1)) == 0:
                    buffer_ops.buffer_store(
                        scale_store, scale_rsrc, scale_base_off + group_idx
                    )

            scaled = []
            for vi in range_constexpr(VEC):
                xi = x_f32_vec[vi]
                if const_expr(weighted):
                    xi = xi * w_f32_vec[vi]
                if const_expr(quant):
                    scaled.append(xi * factor)
                else:
                    scaled.append(xi * rstd)

            cos_f32 = cos_raw.to(fx.Float32)
            sin_f32 = sin_raw.to(fx.Float32)
            if const_expr(PAIRS_PER_THREAD == 1):
                cos_vals = [cos_f32]
                sin_vals = [sin_f32]
            else:
                cos_vals = [cos_f32[i] for i in range(PAIRS_PER_THREAD)]
                sin_vals = [sin_f32[i] for i in range(PAIRS_PER_THREAD)]

            rotated = list(scaled)
            for k in range_constexpr(PAIRS_PER_THREAD):
                e = scaled[2 * k]
                o = scaled[2 * k + 1]
                c = cos_vals[k]
                s = sin_vals[k]
                rotated[2 * k] = e * c - o * s
                rotated[2 * k + 1] = e * s + o * c

            final_list = [
                is_rope_t.select(rotated[i], scaled[i]) for i in range_constexpr(VEC)
            ]

            if const_expr(quant):
                rsrc, row_base = fp8_out_rsrc
                _store_fp8_packed_w32(
                    final_list, rsrc, row_base, tid, VEC, skip_fnuz_clamp=True
                )
            else:
                _store_bf16_vec(
                    final_list, bf16_out_rsrc, bf16_out_row_base_bytes, tid, VEC
                )
                if const_expr(kv_write) and do_swa:
                    _store_bf16_vec(
                        final_list,
                        swa_out_rsrc,
                        swa_out_row_base_bytes,
                        tid,
                        VEC,
                    )

        q_tok_off_bytes = bid_t_idx * fx.Int64(H * D * 2)

        if bid_x < H:
            head_idx = bid_x
            q_in_rsrc = _ptr_buffer_resource(q_in)
            q_row_off_elems = bid_t * (H * D) + head_idx * D + tid * VEC
            q_off_dw = q_row_off_elems >> 1
            q_f32_list = _load_bf16_raw(q_in_rsrc, q_off_dw)
            q_f32_fly_vec = fx.Vector.from_elements(q_f32_list, dtype=fx.Float32)
            q_rmem = fx.make_rmem_tensor(full_lay, fx.Float32)
            fx.memref_store_vec(q_f32_fly_vec, q_rmem)
            x_f32 = fx.memref_load_vec(q_rmem)

            if const_expr(q_weighted):
                qw_vec = _load_weight_tensor(q_weight, tid)
                qw_f32 = qw_vec.to(fx.Float32)
            else:
                qw_f32 = None

            if const_expr(quant):
                q_tok_off_fp8 = bid_t_idx * fx.Int64(H * D)
                qo_g_tmp = GTensor(
                    q_out,
                    dtype=T.i8,
                    shape=(H, D),
                    static_bytes_offset_i64=q_tok_off_fp8,
                )
                qo_rsrc = qo_g_tmp.rsrc
                row_base_bytes = head_idx * D
                qs_rsrc = _ptr_buffer_resource(q_scale)
                scale_base_off_q = bid_t * (H * NG) + head_idx * NG
                emit_body(
                    weighted=q_weighted,
                    x_f32_vec=x_f32,
                    w_f32_vec=qw_f32,
                    bf16_out_rsrc=None,
                    bf16_out_row_base_bytes=None,
                    fp8_out_rsrc=(qo_rsrc, row_base_bytes),
                    scale_rsrc=qs_rsrc,
                    scale_base_off=scale_base_off_q,
                )
            else:
                qo_g_tmp = GTensor(
                    q_out,
                    dtype=T.bf16,
                    shape=(H, D),
                    static_bytes_offset_i64=q_tok_off_bytes,
                )
                qo_rsrc = qo_g_tmp.rsrc
                row_base_bytes_q = head_idx * (D * 2)
                emit_body(
                    weighted=q_weighted,
                    x_f32_vec=x_f32,
                    w_f32_vec=qw_f32,
                    bf16_out_rsrc=qo_rsrc,
                    bf16_out_row_base_bytes=row_base_bytes_q,
                    fp8_out_rsrc=None,
                    scale_rsrc=None,
                    scale_base_off=None,
                )
        else:
            kv_rsrc = _ptr_buffer_resource(kv_in)
            kv_off_elems = bid_t * kv_in_row_stride + tid * VEC
            kv_off_dw = kv_off_elems >> 1

            kv_f32_list = _load_bf16_raw(kv_rsrc, kv_off_dw)
            kv_f32_fly_vec = fx.Vector.from_elements(kv_f32_list, dtype=fx.Float32)
            kv_rmem = fx.make_rmem_tensor(full_lay, fx.Float32)
            fx.memref_store_vec(kv_f32_fly_vec, kv_rmem)
            x_vec_f32 = fx.memref_load_vec(kv_rmem)

            w_vec = _load_weight_tensor(kv_weight, tid)
            w_f32 = w_vec.to(fx.Float32)

            if const_expr(quant):
                kv_tok_off_fp8 = bid_t_idx * fx.Int64(D)
                kvo_g_tmp = GTensor(
                    kv_out,
                    dtype=T.i8,
                    shape=(D,),
                    static_bytes_offset_i64=kv_tok_off_fp8,
                )
                kvo_rsrc = kvo_g_tmp.rsrc
                row_base_bytes = fx.Int32(0)
                kvs_rsrc = _ptr_buffer_resource(kv_scale)
                scale_base_off_kv = bid_t * NG
                emit_body(
                    weighted=True,
                    x_f32_vec=x_vec_f32,
                    w_f32_vec=w_f32,
                    bf16_out_rsrc=None,
                    bf16_out_row_base_bytes=None,
                    fp8_out_rsrc=(kvo_rsrc, row_base_bytes),
                    scale_rsrc=kvs_rsrc,
                    scale_base_off=scale_base_off_kv,
                )
            else:
                kv_tok_off_bf16 = bid_t_idx * fx.Int64(D * 2)
                kvo_g_tmp = GTensor(
                    kv_out,
                    dtype=T.bf16,
                    shape=(D,),
                    static_bytes_offset_i64=kv_tok_off_bf16,
                )
                kvo_rsrc = kvo_g_tmp.rsrc
                row_base_bytes_kv = fx.Int32(0)

                swa_rsrc = None
                swa_row_base = None
                do_swa = None
                if const_expr(kv_write):
                    bid_rsrc = _ptr_buffer_resource(batch_id_per_token)
                    bid_i32 = fx.Int32(
                        buffer_ops.buffer_load(bid_rsrc, bid_t, vec_width=1, dtype=i32)
                    )
                    # A stale token carries a negative position. `divsi`/`remsi`
                    # truncate toward zero, so pos=-300 with block_size=128 gives
                    # blk=-2 and in_blk=-44: the block index passes an upper-bound
                    # test, reads the PREVIOUS request's table entry, and lands
                    # the write 44 rows before that block. The C++ sibling's first
                    # gate is `bid < 0 || pos < 0`; match it, in both modes.
                    pos_ok = pos_i32 >= 0
                    do_swa = (bid_i32 >= 0) & pos_ok
                    bid_safe = _imax(bid_i32, fx.Int32(0))
                    pos_safe = pos_ok.select(pos_i32, fx.Int32(0))
                    if const_expr(paged):
                        blk = _idiv(pos_safe, swa_cache_size)
                        # The other two gates the C++ sibling applies (see
                        # `swa_row_for_token`): a position past the table's last
                        # block would read the NEXT request's entry, and `-1`
                        # marks a block outside the window. Clamp for the load,
                        # gate the store on the unclamped test. The wrapper pins
                        # stride(0) == max_blocks, so the row stride is both the
                        # table width and the distance between two requests' rows.
                        blk_ok = blk < swa_slot_stride
                        blk_safe = blk_ok.select(blk, fx.Int32(0))
                        bt_off = bid_safe * swa_slot_stride + blk_safe
                        bt_rsrc = _ptr_buffer_resource(swa_index)
                        phys = fx.Int32(
                            buffer_ops.buffer_load(
                                bt_rsrc, bt_off, vec_width=1, dtype=i32
                            )
                        )
                        in_blk = pos_safe % swa_cache_size
                        row = phys * swa_cache_size + in_blk
                        # Bound the final row against the pool, as the C++ does:
                        # a phys id left over from a larger pool would otherwise
                        # write past the end of swa_kv.
                        row_ok = blk_ok & (phys >= 0) & (row < swa_num_rows)
                        do_swa = do_swa & row_ok
                        # Clamp too: a negative row would move the descriptor
                        # base backwards, out of this tensor entirely.
                        row_safe = row_ok.select(row, fx.Int32(0))
                    else:
                        row_rsrc = _ptr_buffer_resource(swa_index)
                        row = fx.Int32(
                            buffer_ops.buffer_load(
                                row_rsrc, bid_t, vec_width=1, dtype=i32
                            )
                        )
                        dest_ok = (row >= 0) & (row < swa_num_rows)
                        do_swa = do_swa & dest_ok
                        row_safe = dest_ok.select(row, fx.Int32(0))
                    # The row index fits 32 bits; `row * D * 2` does not. A
                    # unified V4 pool runs to ~150M rows, so a 32-bit byte
                    # offset wraps 3% of the way in, and the sliding windows
                    # this writes live at the far end. Widen before either
                    # multiply, and let it reach the descriptor through the
                    # base address (`static_bytes_offset_i64`) rather than
                    # through the 32-bit offset field, whose window is 4 GiB.
                    swa_off_bytes = (
                        fx.Int64(row_safe) * fx.Int64(swa_pos_stride) * fx.Int64(2)
                    )
                    swa_g_tmp = GTensor(
                        swa_kv,
                        dtype=T.bf16,
                        shape=(D,),
                        static_bytes_offset_i64=swa_off_bytes,
                    )
                    swa_rsrc = swa_g_tmp.rsrc
                    swa_row_base = fx.Int32(0)

                emit_body(
                    weighted=True,
                    x_f32_vec=x_vec_f32,
                    w_f32_vec=w_f32,
                    bf16_out_rsrc=kvo_rsrc,
                    bf16_out_row_base_bytes=row_base_bytes_kv,
                    fp8_out_rsrc=None,
                    scale_rsrc=None,
                    scale_base_off=None,
                    swa_out_rsrc=swa_rsrc,
                    swa_out_row_base_bytes=swa_row_base,
                    do_swa=do_swa,
                )

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
        grid_y = fx.Int64(_idiv(num_tokens + ROWS_PER_WG - 1, fx.Int32(ROWS_PER_WG)))
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
            num_tokens,
        )
        k.launch(
            grid=(H + 1, grid_y, 1),
            block=(BLOCK_THREADS, ROWS_PER_WG, 1),
            stream=stream,
        )

    launch_qk_norm_rope_quant.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launch_qk_norm_rope_quant


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
    """Fused RMSNorm + GPT-J RoPE + optional FP8 quant for Q and KV in one launch."""
    D, RD = head_dim, rope_head_dim
    G = quant_group_size if quant_group_size is not None else D
    kv_write = swa_kv is not None
    # Validated ahead of the wave32 dispatch below so both wave widths share
    # this one copy. User-facing inputs use raise (not stripped under python
    # -O); internal codegen invariants in _build_kernel stay as asserts.
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
    if D % G != 0:
        raise ValueError(f"head_dim {D} must be divisible by quant_group_size {G}")
    # cos/sin are view-reshaped to 2D [max_pos, RD/2] further down; accept any
    # leading shape whose last dim is RD/2 (DeepSeek-V4 stores [max_pos,1,1,RD/2]).
    if cos_cache.shape[-1] != RD // 2:
        raise ValueError(
            f"cos_cache last dim {cos_cache.shape[-1]} != RD/2 ({RD // 2})"
        )
    if sin_cache.shape != cos_cache.shape:
        raise ValueError("cos/sin shape mismatch")
    if not (cos_cache.is_contiguous() and sin_cache.is_contiguous()):
        raise ValueError("cos/sin must be contiguous")
    if kv_write and quant:
        raise ValueError("kv_write (swa_kv) is BF16 only; not supported with quant")
    from aiter.jit.utils.chip_info import get_gfx as _get_gfx

    is_gfx1250 = _get_gfx() == "gfx1250"

    H = num_q_heads
    T_tok = q.shape[0]
    NG = D // G
    q_weighted = q_weight is not None
    # Kernel always binds q_weight; pass a dummy when unused (const_expr gate
    # DCEs the load but the param binding still needs a valid tensor).
    q_weight_arg = q_weight if q_weighted else kv_weight

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

    paged = swa_block_tables is not None
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
        swa_slot_stride = 0
        swa_pos_stride = 0
        swa_num_rows = 0
        swa_cache_size = 1
        swa_kv_arg = kv_out  # bf16 dummy
        ssm_arg = q.new_empty(1, dtype=torch.int32)
        bid_arg = q.new_empty(1, dtype=torch.int32)

    if is_gfx1250:
        tdm_quant = quant
        use_tdm = (
            USE_TDM_PREFILL
            and (not quant or T_tok * H >= 65536)
            and T_tok * H >= TDM_MIN_ROWS
            and (D % 256 == 0)
            and (H & (H - 1) == 0)
        )
        if use_tdm:
            num_rows = T_tok * H
            tiles_per_wg, num_buffers, rows_per_tile = _tdm_tiles_per_wg(
                num_rows, head_dim=D
            )
            launcher = _build_kernel_w32_tdm(
                num_q_heads=H,
                head_dim=D,
                rope_head_dim=RD,
                num_buffers=num_buffers,
                tiles_per_wg=tiles_per_wg,
                rows_per_tile=rows_per_tile,
                q_weighted=q_weighted,
                store_cache_modifier=(
                    TDM_DECODE_STORE_CACHE_MODIFIER
                    if (T_tok, H, D, RD) == (512, 128, 512, 64)
                    else 0
                ),
                quant=tdm_quant,
                group_size=G,
                scale_dtype=scale_dtype,
                kv_write=kv_write,
                paged=paged,
            )
            if stream is None:
                stream = torch.cuda.current_stream()
            has_direct = getattr(launcher, "_direct_call_state", None) is not None

            def _t_ptr(t):
                return (
                    int(t.data_ptr())
                    if has_direct
                    else flyc.from_c_void_p(fx.Uint8, t.data_ptr())
                )

            q_2d = q_view.reshape(num_rows, D)
            per_wg = rows_per_tile * tiles_per_wg
            args = (
                # q is a per-call activation: uncached, else the adaptor cache
                # pins its allocation (see _cached_from_dlpack).
                flyc.from_dlpack(q_2d),
                _t_ptr(kv),
                _cached_from_dlpack(cos_2d),
                _cached_from_dlpack(sin_2d),
                _t_ptr(positions),
                _t_ptr(q_out.view(num_rows, D)),
                _t_ptr(kv_out),
                _t_ptr(q_scale_arg.view(-1)),
                _t_ptr(kv_scale_arg.view(-1)),
                _cached_from_dlpack(q_weight_arg.reshape(-1)),
                _cached_from_dlpack(kv_weight.reshape(-1)),
                _t_ptr(swa_kv_arg),
                _t_ptr(ssm_arg),
                _t_ptr(bid_arg),
                num_rows,
                T_tok,
                (num_rows + per_wg - 1) // per_wg,
                kv.stride(0),
                swa_slot_stride,
                swa_pos_stride,
                swa_num_rows,
                swa_cache_size,
                stream if has_direct else Stream(stream),
            )
            _run_compiled(launcher, *args)
            return (
                q_out,
                kv_out,
                (q_scale if quant else None),
                (kv_scale if quant else None),
            )

        rows_per_wg = ROWS_PER_WG_SMALL if T_tok <= SMALL_T_THRESHOLD else ROWS_PER_WG
        launcher = _build_kernel_w32(
            num_q_heads=H,
            head_dim=D,
            rope_head_dim=RD,
            quant=quant,
            group_size=G,
            scale_dtype=scale_dtype,
            q_weighted=q_weighted,
            kv_write=kv_write,
            paged=paged,
            rows_per_wg=rows_per_wg,
        )
    else:
        launcher = _build_kernel_w64(
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

    if is_gfx1250:
        q_weight_static = _cached_from_dlpack(q_weight_arg)
        kv_weight_static = _cached_from_dlpack(kv_weight)
        cos_static = _cached_from_dlpack(cos_2d)
        sin_static = _cached_from_dlpack(sin_2d)
        has_direct = getattr(launcher, "_direct_call_state", None) is not None

        def _ptr_arg(t):
            return (
                int(t.data_ptr())
                if has_direct
                else flyc.from_c_void_p(fx.Uint8, t.data_ptr())
            )

        def _stream_arg():
            return stream if has_direct else Stream(stream)

        wt_args = (q_weight_static, kv_weight_static, cos_static, sin_static)
    else:

        def _ptr_arg(t):
            return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

        def _stream_arg():
            return Stream(stream)

        wt_args = (q_weight_arg, kv_weight, cos_2d, sin_2d)

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
            *wt_args,
            _ptr_arg(positions[start:end]),
            _ptr_arg(q_out[start:end]),
            _ptr_arg(kv_out[start:end]),
            _ptr_arg(q_scale_arg[start:end] if quant else q_scale_arg),
            _ptr_arg(kv_scale_arg[start:end] if quant else kv_scale_arg),
            kv.stride(0),
            _ptr_arg(swa_kv_arg),
            _ptr_arg(ssm_arg[start:end] if (kv_write and not paged) else ssm_arg),
            _ptr_arg(bid_arg[start:end] if kv_write else bid_arg),
            swa_slot_stride,
            swa_pos_stride,
            swa_num_rows,
            swa_cache_size,
            n,
            _stream_arg(),
        )
        _run_compiled(launcher, *args)

    return q_out, kv_out, (q_scale if quant else None), (kv_scale if quant else None)


ROWS_PER_TILE = 8  # = waves per WG (256-thread WG)
ROWS_PER_TILE_SPARSE = 4  # halve the WG so a thin grid still gets 2 WGs/CU
NUM_BUFFERS = 6  # K: rotating LDS buffers (1 WG at K=6: 6*8KB=48KB LDS)
NUM_BUFFERS_DENSE = 2  # K once the grid is deep enough to hide a short prologue
TILES_PER_WG = 16  # CT: one H=128 token/WG; keeps cos/sin hoisting enabled
TDM_DENSE_MIN_ROWS = 131072  # num_rows from which K=2 beats K=6
TDM_RT8_MIN_ROWS = 65536  # num_rows where RT=8 first reaches 2 WGs per 256 CUs


def _tdm_tiles_per_wg(num_rows, head_dim=512):
    """Pick (CT, K, RT). Both knobs ask the same thing: can a workgroup's load
    latency be hidden by a neighbour, or must it cover its own? Thin grids need
    a smaller WG (more of them) and a deeper prologue; dense grids need neither.
    Thresholds assume a 256-CU part.
    """
    del head_dim  # tile bytes are implied by RT * D in the kernel
    k = NUM_BUFFERS_DENSE if num_rows >= TDM_DENSE_MIN_ROWS else NUM_BUFFERS
    rt = ROWS_PER_TILE if num_rows >= TDM_RT8_MIN_ROWS else ROWS_PER_TILE_SPARSE
    return TILES_PER_WG, k, rt


@lru_cache(maxsize=32)
def _build_kernel_w32_tdm(
    *,
    num_q_heads,
    head_dim,
    rope_head_dim,
    num_buffers=NUM_BUFFERS,
    tiles_per_wg=TILES_PER_WG,
    rows_per_tile=ROWS_PER_TILE,
    hoist_cs=True,
    q_weighted=False,
    store_cache_modifier=0,
    quant=False,
    group_size=None,
    scale_dtype="fp32",
    kv_write=False,
    paged=False,
):
    H, D, RD = num_q_heads, head_dim, rope_head_dim
    NOPE = D - RD
    BLOCK_THREADS = BLOCK_THREADS_W32  # TDM prefill is gfx1250-only
    VEC = D // BLOCK_THREADS
    ROPE_LO = NOPE // VEC
    PAIRS = VEC // 2
    K = num_buffers
    CT = tiles_per_wg
    RT = rows_per_tile
    G = D if group_size is None else group_size
    NG = D // G
    TPG = G // VEC
    log2_tpg = int(math.log2(TPG))
    is_e8m0 = scale_dtype == "e8m0"
    if CT < K:
        raise ValueError(
            f"TDM prologue issues K={K} tiles but WG only owns CT={CT}; "
            "need CT >= K to avoid overlapping the next WG's rows"
        )
    eb = 2
    log2b = int(math.log2(BLOCK_THREADS))
    amax_start_step = log2b - log2_tpg
    log2H = int(math.log2(H)) if (H & (H - 1)) == 0 else None
    fp8_mx_dtype = _D.FP8_E4M3
    EVEN = list(range(0, VEC, 2))
    ODD = list(range(1, VEC, 2))
    ILV = [(i // 2) if i % 2 == 0 else (PAIRS + i // 2) for i in range(VEC)]
    tile_bytes = RT * D * eb
    ARENA = K * tile_bytes  # K rotating tile buffers

    name = f"qk_norm_rope_tdm_H{H}_D{D}_RD{RD}_k{K}_ct{CT}_r{RT}"
    if hoist_cs:
        name += "_h"
    if q_weighted:
        name += "_qw"
    if store_cache_modifier:
        name += f"_sc{store_cache_modifier}"
    if quant:
        name += f"_q8g{G}_{scale_dtype}"
    if kv_write:
        name += "_kvw"
    if paged:
        name += "_paged"
    name += "_fused_w32_flydsl"

    @flyc.kernel(name=name, known_block_size=[BLOCK_THREADS, RT, 1])
    def kernel(
        q_in: fx.Tensor,  # [T*H, D] bf16
        kv_in: fx.Pointer,  # [T, D] bf16, may be strided
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        positions: fx.Pointer,  # [T] i64
        q_out: fx.Pointer,  # [T*H, D] bf16
        kv_out: fx.Pointer,  # [T, D] bf16
        q_scale: fx.Pointer,  # [T*H, NG] fp32/u8 (dummy when not quant)
        kv_scale: fx.Pointer,  # [T, NG] fp32/u8 (dummy when not quant)
        q_weight: fx.Tensor,  # [D] bf16 (dummy when not q_weighted)
        kv_weight: fx.Tensor,  # [D] bf16
        swa_kv: fx.Pointer,  # [num_slots, cache_size, D] bf16 (dummy if not kv_write)
        swa_index: fx.Pointer,  # paged: block_tables; direct: [T] dest rows
        batch_id_per_token: fx.Pointer,  # [T] i32, -1 sentinel (dummy if not kv_write)
        num_rows: Int32,  # T*H
        num_tokens: Int32,  # T
        gx_q: Int32,  # workgroups owned by the Q path
        kv_in_row_stride: Int32,  # KV row stride in bf16 elements
        swa_slot_stride: Int32,  # bf16 elements (= cache_size * D)
        swa_pos_stride: Int32,  # bf16 elements (= D)
        swa_num_rows: Int32,  # rows in the SWA pool; bounds the final row
        swa_cache_size: Int32,  # ring slot count
    ):
        i32 = T.i32
        tid = fx.thread_idx.x
        wave = fx.thread_idx.y
        g = fx.block_idx.x

        lds_ptr = fx.SharedAllocator(static=False).allocate(ARENA)._ptr
        lds_idx = fx.index_cast(T.index, fx.ptrtoint(lds_ptr))
        lds_i32 = fx.index_cast(T.i32, lds_idx)  # i32 form for pointer math
        bufs = [k * tile_bytes for k in range(K)]
        lds_load_b128, _ = make_lds_copy_ops(128)
        lds_bf16_ptr_ty = fx.PointerType.get(
            elem_ty=fx.BFloat16.ir_type,
            address_space=fx.AddressSpace.Shared,
            alignment=16,
        )

        cos_rsrc = buffer_ops.create_buffer_resource(cos_cache, max_size=True)
        sin_rsrc = buffer_ops.create_buffer_resource(sin_cache, max_size=True)
        is_rope = tid >= ROPE_LO
        rope_rel = _imax(tid - ROPE_LO, fx.Int32(0))

        def _ptr_res(ptr):
            return buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(ptr))
            )

        pos_rsrc = _ptr_res(positions)

        def _concat(chunks):
            v = chunks[0]
            for c in range_constexpr(len(chunks) - 1):
                v = v.shuffle(chunks[c + 1], list(range(8 * (c + 2))))
            return v.to(fx.Float32)

        def _load_tensor_vec(tensor, tid_val):
            """Load VEC bf16 from a 1D fx.Tensor via buffer_load chunks."""
            rsrc = buffer_ops.create_buffer_resource(tensor, max_size=True)
            return _concat(
                [
                    fx.Vector(
                        buffer_ops.buffer_load(
                            rsrc,
                            tid_val * VEC + (c * 8),
                            vec_width=8,
                            dtype=T.bf16,
                        )
                    )
                    for c in range(VEC // 8)
                ]
            )

        def load_pos(tok):
            return fx.Int32(
                buffer_ops.buffer_load(pos_rsrc, tok, vec_width=1, dtype=T.i64).trunci(
                    i32
                )
            )

        def load_cs(tok):
            """cos/sin for this token, as PAIRS-wide f32 vectors."""
            return _cs_from_pos(load_pos(tok))

        def issue_pos(tok):
            """Fire position buffer_load (returns raw i64, no wait)."""
            return buffer_ops.buffer_load(pos_rsrc, tok, vec_width=1, dtype=T.i64)

        def _cs_from_pos(pos_i32):
            """Load cos/sin given an already-resolved position value."""
            cs = pos_i32 * (RD // 2) + rope_rel * PAIRS
            if const_expr(PAIRS == 1):
                return tuple(
                    fx.Vector.from_elements(
                        [buffer_ops.buffer_load(r, cs, vec_width=1, dtype=T.bf16)],
                        fx.BFloat16,
                    ).to(fx.Float32)
                    for r in (cos_rsrc, sin_rsrc)
                )
            return tuple(
                fx.Vector(
                    buffer_ops.buffer_load(r, cs, vec_width=PAIRS, dtype=T.bf16)
                ).to(fx.Float32)
                for r in (cos_rsrc, sin_rsrc)
            )

        def norm_rope_store(
            xv,
            wv,
            cosv,
            sinv,
            out_rsrc,
            row_base_bytes,
            scale_rsrc=None,
            scale_idx=None,
            swa_rsrc=None,
            swa_row_base=None,
            do_swa=None,
        ):
            """RMSNorm + RoPE and optional per-row FP8 quantization."""
            sq = (xv * xv).reduce(ReductionOp.ADD)
            if const_expr(quant):
                xw = xv * wv if wv is not None else xv
                am = fmath.absf(xw).reduce(ReductionOp.MAX)
            for sh in range_constexpr(log2b):
                o = BLOCK_THREADS // (2 << sh)
                sq = sq + sq.shuffle_xor(o, BLOCK_THREADS)
                if const_expr(quant and sh >= amax_start_step):
                    am = am.maximumf(am.shuffle_xor(o, BLOCK_THREADS))
            if const_expr(quant and is_e8m0):
                rstd_q = fmath.rsqrt(sq * (1.0 / D) + 1e-6)
                e8m0_biased = fx.Int32(
                    emit_mx_e8m0_scale(
                        (am.maximumf(fx.Float32(1e-12)) * rstd_q * _SQRT2).ir_value(),
                        mode=_DEFAULT_MODE,
                        dtype=fp8_mx_dtype,
                    )
                )
                factor = rstd_q * ((fx.Int32(254) - e8m0_biased) << 23).bitcast(
                    fx.Float32
                )
                scaled = xw * factor
                scale_store = e8m0_biased.to(fx.Int8)
            elif const_expr(quant):
                am_safe = am.maximumf(fx.Float32(1e-12))
                rcp_am = fx.Float32(fx.rocdl.rcp(T.f32, am_safe))
                fc = _fp8_const()
                factor = fx.Float32(fc["max_over_sqrt2"]) * rcp_am
                scaled = xw * factor
            else:
                rstd = fmath.rsqrt(sq * (1.0 / D) + 1e-6)
                scaled = (xv * wv * rstd) if wv is not None else (xv * rstd)

            # Only RD / D lanes own the RoPE tail (4 of 32 for D=512,
            # RD=64). Keep the passthrough value in register memory so an
            # scf.if can update it without an SSA dominance violation, and
            # avoid running all RoPE FMAs in the other 28 lanes.
            out_rmem = fx.make_rmem_tensor(fx.make_layout(VEC, 1), fx.Float32)
            fx.memref_store_vec(scaled, out_rmem)
            if is_rope:
                cur = fx.memref_load_vec(out_rmem)
                ev = cur.shuffle(cur, EVEN)
                od = cur.shuffle(cur, ODD)
                rot = (ev * cosv - od * sinv).shuffle(ev * sinv + od * cosv, ILV)
                fx.memref_store_vec(rot, out_rmem)
            final = fx.memref_load_vec(out_rmem)
            if const_expr(quant):
                _store_fp8_packed_w32(
                    [final[i] for i in range_constexpr(VEC)],
                    out_rsrc,
                    row_base_bytes,
                    tid,
                    VEC,
                    skip_fnuz_clamp=True,
                )
                group_idx = tid >> log2_tpg
                if (tid & (TPG - 1)) == 0:
                    if const_expr(is_e8m0):
                        buffer_ops.buffer_store(
                            scale_store, scale_rsrc, scale_idx + group_idx
                        )
                    else:
                        # FP8 values depend only on amax. Defer the RMS scale
                        # path until after the wide output store so rsqrt
                        # latency overlaps memory retirement.
                        buffer_ops.buffer_store(
                            am_safe
                            * fmath.rsqrt(sq * (1.0 / D) + 1e-6)
                            * fx.Float32(fc["inv_max_sqrt2"]),
                            scale_rsrc,
                            scale_idx + group_idx,
                        )
            else:
                _store_bf16_vec(
                    final,
                    out_rsrc,
                    row_base_bytes,
                    tid,
                    VEC,
                    cache_modifier=store_cache_modifier,
                )
                # Same bytes as kv_out, scattered into the SWA ring.
                if const_expr(kv_write) and do_swa:
                    _store_bf16_vec(final, swa_rsrc, swa_row_base, tid, VEC)

        def emit_q():
            q_scale_rsrc = _ptr_res(q_scale) if const_expr(quant) else None
            nr_m1 = num_rows - 1
            tile_base = g * CT  # first tile index this WG owns
            wg_row0 = tile_base * RT  # first row this WG owns
            out_row_bytes = D if const_expr(quant) else D * 2
            # num_records is 32-bit, so one descriptor only reaches 4 GiB; bias
            # the base per WG so the offset spans CT*RT rows, not the tensor.
            q_out_rsrc = GTensor(
                q_out,
                dtype=T.bf16,
                shape=(CT * RT, D),
                static_bytes_offset_i64=fx.Int64(wg_row0) * fx.Int64(out_row_bytes),
            ).rsrc
            wave_lds_off = wave * (D * eb)
            row_elem = tid * VEC  # LDS read pos within that slot

            qw = _load_tensor_vec(q_weight, tid) if const_expr(q_weighted) else None

            q_row0 = fx.Tensor(
                fx.make_view(fx.get_iter(q_in), fx.make_layout((1, D), (D, 1)))
            )
            tdm_atom = fx.rocdl.make_tdm_atom(
                q_row0,
                [num_rows, None],
                strides=[D, None],
                num_warps=1,
            )

            def row_of(tile_idx):
                return _imin(tile_idx * RT + wave, nr_m1)

            def issue(buf, tile_idx):
                dst = fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(lds_bf16_ptr_ty, lds_i32 + wave_lds_off + buf),
                        fx.make_layout((1, D), (D, 1)),
                    )
                )
                fx.copy(
                    tdm_atom,
                    q_row0,
                    dst,
                    imm_offset=fx.Int64(row_of(tile_idx)) * (D * eb),
                )

            def compute_store(buf, tile_idx, cosv, sinv):
                xv = _concat(
                    [
                        fx.Vector(
                            lds_load_b128(
                                lds_idx,
                                buf + wave_lds_off + (row_elem + c * 8) * eb,
                            )
                        ).bitcast(fx.BFloat16)
                        for c in range(VEC // 8)
                    ]
                )
                norm_rope_store(
                    xv,
                    qw,
                    cosv,
                    sinv,
                    q_out_rsrc,
                    (row_of(tile_idx) - wg_row0) * out_row_bytes,
                    q_scale_rsrc,
                    row_of(tile_idx) * NG,
                )

            def tok_of(tile_idx):
                my_row = row_of(tile_idx)
                if const_expr(log2H is not None):
                    return my_row >> log2H
                return _idiv(my_row, fx.Int32(H))

            def cs_of(tile_idx):
                return load_cs(tok_of(tile_idx))

            GROUP = (H // RT) if (log2H is not None and H % RT == 0) else 1
            # A WG may cover one or more complete tokens, or an aligned fraction
            # of one token. Both cases keep cos/sin constant within the WG.
            do_hoist = hoist_cs and GROUP > 1 and (CT % GROUP == 0 or GROUP % CT == 0)

            for k in range_constexpr(K):
                issue(bufs[k], tile_base + k)

            cs_cache = [None, None]
            if const_expr(do_hoist):
                pending_pos = [issue_pos(tok_of(tile_base + 0))]
                cs_cache[0], cs_cache[1] = _cs_from_pos(
                    fx.Int32(pending_pos[0].trunci(i32))
                )
            for i in range_constexpr(CT):
                # Tile i consumes TDM load #i (issued in tile order: K in the
                # prologue, then one per iteration while i + K < CT). In steady
                # state K + i are issued, so leaving K-1 outstanding retires
                # exactly #0..#i. Once the issues stop the issued count freezes
                # at CT and K-1 is too loose: #i is only guaranteed retired with
                # at most CT-1-i left, reaching 0 on the last tile. Using K-1
                # throughout lets a drain tile read an LDS buffer whose load has
                # not landed; it stays latent only while the per-tile compute
                # happens to outlast the load.
                tdm_ops.tensor_wait(min(K - 1, CT - 1 - i))
                fx.rocdl.s_wait_dscnt(0)
                if const_expr(do_hoist):
                    cosv, sinv = cs_cache[0], cs_cache[1]
                else:
                    cosv, sinv = cs_of(tile_base + i)
                compute_store(bufs[i % K], tile_base + i, cosv, sinv)
                if const_expr(i + K < CT):
                    issue(bufs[i % K], tile_base + i + K)  # reuse after read
                if const_expr(do_hoist and (i + 1) % GROUP == 0 and i + 1 < CT):
                    pending_pos[0] = issue_pos(tok_of(tile_base + i + 1))
                    cs_cache[0], cs_cache[1] = _cs_from_pos(
                        fx.Int32(pending_pos[0].trunci(i32))
                    )

        def emit_kv():
            gk = g - gx_q
            tok = _imin(gk * RT + wave, num_tokens - 1)
            kv_rsrc = _ptr_res(kv_in)
            xv = _concat(
                [
                    fx.Vector(
                        buffer_ops.buffer_load(
                            kv_rsrc,
                            tok * kv_in_row_stride + tid * VEC + (c * 8),
                            vec_width=8,
                            dtype=T.bf16,
                        )
                    )
                    for c in range(VEC // 8)
                ]
            )
            wv = _load_tensor_vec(kv_weight, tid)
            pos_i32 = load_pos(tok)
            cosv, sinv = _cs_from_pos(pos_i32)

            swa_rsrc = None
            swa_row_base = None
            do_swa = None
            if const_expr(kv_write):
                # Gates mirror the wave32 path / C++ sibling exactly.
                bid_i32 = fx.Int32(
                    buffer_ops.buffer_load(
                        _ptr_res(batch_id_per_token), tok, vec_width=1, dtype=i32
                    )
                )
                pos_ok = pos_i32 >= 0
                do_swa = (bid_i32 >= 0) & pos_ok
                bid_safe = _imax(bid_i32, fx.Int32(0))
                pos_safe = pos_ok.select(pos_i32, fx.Int32(0))
                if const_expr(paged):
                    blk = _idiv(pos_safe, swa_cache_size)
                    blk_ok = blk < swa_slot_stride
                    blk_safe = blk_ok.select(blk, fx.Int32(0))
                    phys = fx.Int32(
                        buffer_ops.buffer_load(
                            _ptr_res(swa_index),
                            bid_safe * swa_slot_stride + blk_safe,
                            vec_width=1,
                            dtype=i32,
                        )
                    )
                    row = phys * swa_cache_size + (pos_safe % swa_cache_size)
                    row_ok = blk_ok & (phys >= 0) & (row < swa_num_rows)
                    do_swa = do_swa & row_ok
                    row_safe = row_ok.select(row, fx.Int32(0))
                else:
                    row = fx.Int32(
                        buffer_ops.buffer_load(
                            _ptr_res(swa_index), tok, vec_width=1, dtype=i32
                        )
                    )
                    dest_ok = (row >= 0) & (row < swa_num_rows)
                    do_swa = do_swa & dest_ok
                    row_safe = dest_ok.select(row, fx.Int32(0))
                # row fits 32 bits, row*D*2 does not: widen before the multiply.
                swa_rsrc = GTensor(
                    swa_kv,
                    dtype=T.bf16,
                    shape=(D,),
                    static_bytes_offset_i64=(
                        fx.Int64(row_safe) * fx.Int64(swa_pos_stride) * fx.Int64(2)
                    ),
                ).rsrc
                swa_row_base = fx.Int32(0)

            norm_rope_store(
                xv,
                wv,
                cosv,
                sinv,
                _ptr_res(kv_out),
                tok * (D if const_expr(quant) else D * 2),
                _ptr_res(kv_scale) if const_expr(quant) else None,
                tok * NG,
                swa_rsrc,
                swa_row_base,
                do_swa,
            )

        if g < gx_q:
            emit_q()
        else:
            emit_kv()

    @flyc.jit
    def launch(
        q_in: fx.Tensor,
        kv_in: fx.Pointer,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        positions: fx.Pointer,
        q_out: fx.Pointer,
        kv_out: fx.Pointer,
        q_scale: fx.Pointer,
        kv_scale: fx.Pointer,
        q_weight: fx.Tensor,
        kv_weight: fx.Tensor,
        swa_kv: fx.Pointer,
        swa_index: fx.Pointer,
        batch_id_per_token: fx.Pointer,
        num_rows: fx.Int32,
        num_tokens: fx.Int32,
        gx_q: fx.Int32,
        kv_in_row_stride: fx.Int32,
        swa_slot_stride: fx.Int32,
        swa_pos_stride: fx.Int32,
        swa_num_rows: fx.Int32,
        swa_cache_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        gx_kv = _idiv(num_tokens + (RT - 1), fx.Int32(RT))
        gx = gx_q + gx_kv
        k = kernel(
            q_in,
            kv_in,
            cos_cache,
            sin_cache,
            positions,
            q_out,
            kv_out,
            q_scale,
            kv_scale,
            q_weight,
            kv_weight,
            swa_kv,
            swa_index,
            batch_id_per_token,
            num_rows,
            num_tokens,
            gx_q,
            kv_in_row_stride,
            swa_slot_stride,
            swa_pos_stride,
            swa_num_rows,
            swa_cache_size,
        )
        k.launch(
            grid=(fx.Int64(gx), 1, 1),
            block=(BLOCK_THREADS, RT, 1),
            stream=stream,
        )

    launch.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launch
