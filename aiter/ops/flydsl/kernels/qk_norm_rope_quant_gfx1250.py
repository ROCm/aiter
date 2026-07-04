# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused per-token RMSNorm + GPT-J RoPE + optional FP8 quant (FlyDSL).

gfx1250 (wave32) variant — BLOCK_THREADS=32, VEC=D/32 (=16 for D=512).

Q + KV combined into a single kernel launch (grid Y = num_tokens, grid X =
num_q_heads + 1: bid_x ∈ [0, H) handle Q heads, bid_x == H handles KV).

Layout per block (D=512, BLOCK_THREADS=32):
  - VEC = D // 32 = 16
  - thread t ∈ [0, ROPE_THREAD_LO) owns NOPE elements [t*16, t*16+16)
  - thread t ∈ [ROPE_THREAD_LO, 32) owns ROPE elements [t*16, t*16+16) which
    form ``PAIRS_PER_THREAD=8`` GPT-J pairs (2k, 2k+1)

Key wave32 changes vs wave64 original:
  - BLOCK_THREADS = 32 (1 wave32 on RDNA4/gfx1250)
  - VEC = 16 for D=512 (was 8); all load/store paths generalized for VEC>8
  - shuffle_xor width = 32; butterfly reduction log2(32)=5 passes (was 6)
  - FP8 packing: VEC/4 dwords (4 dwords for VEC=16) stored as dwordx4
  - BF16 store: VEC=16 → 8 dwords, split into 2× dwordx4 via extract_strided_slice
  - BF16 load: VEC=16 → 8 dwords, split into 2× dwordx4 buffer_load
  - Rope CopyAtom: BufferCopy128b (16 bytes = 8 pairs for PAIRS_PER_THREAD=8)
  - No preshuffle (gfx1250 uses linear FP8 layout)

Public API: ``flydsl_qk_norm_rope_quant_gfx1250`` (torch-friendly, allocates
outputs, binds current stream, handles strided KV and 4D cos/sin views).
Internal ``compile_flydsl_qk_norm_rope_quant_gfx1250`` returns the cached
launcher for callers who already have all buffers and want the lowest-overhead
path.
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
# Defer the import until the first runtime call instead -- sibling modules
# (moe_kernels._get_dtypes, gemm_kernels._get_dtypes) use the same pattern.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, vector, buffer_ops
from flydsl.expr import math as fmath
from flydsl.expr.arith import ArithValue, CmpFPredicate, CmpIPredicate
from flydsl.expr.typing import T, Int32, Stream
from flydsl.expr.vector import ReductionOp
from flydsl._mlir.dialects import llvm, rocdl

from .tensor_shim import GTensor, _to_raw, _run_compiled

# JIT-free MX-format mode/dtype int mirrors. ``aiter.utility.mx_types``'s
# pybind11 ``MxScaleRoundMode`` / ``MxDtype`` lazy-load on first attribute
# access; we only pull the int classes here so module import stays JIT-free
# (mirrors the FlyDSL AOT-friendly pattern in ``quant_utils``).
from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale
from aiter.utility.mx_types import (
    MxDtypeInt as _D,
    MX_DEFAULT_ROUND_MODE as _DEFAULT_MODE,
)

_STATIC_ADAPTOR_CACHE = {}
_STATIC_ADAPTOR_CACHE_MAX = 64


def _cached_from_dlpack(t: torch.Tensor):
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


# --- shape constants (gfx1250 wave32) -----------------------------------------
BLOCK_THREADS = 32  # 1 wave32 on RDNA4/gfx1250

# Waves (rows/tokens) per workgroup. gfx1250 caps resident workgroups per CU
# below the 64-waves/CU ceiling, so a 1-wave workgroup leaves occupancy on the
# table even at VGPR=32. Packing ROWS_PER_WG independent waves into one
# workgroup lets a handful of workgroups reach full occupancy and, crucially,
# overlaps one row's load->reduce->store latency with another row's memory ops
# (the kernel is latency-bound on that chain, not bandwidth- or ALU-bound).
# Each wave (thread_idx.y) still processes exactly one (head, token) row.
#
# Adaptive by token count (chosen per launch in the public API):
#   - Large T (prefill): ROWS_PER_WG (=32, full 1024-thread workgroup) maximises
#     waves/workgroup so few workgroups saturate 64 waves/CU.
#   - Small T (decode, T <= SMALL_T_THRESHOLD): a smaller R yields MORE
#     workgroups (grid = (H+1) * ceil(T/R)), spreading across more CUs when the
#     total row count is too small to fill the machine at R=32.
ROWS_PER_WG = 32
ROWS_PER_WG_SMALL = 4
# At/below this token count, R=32 launches < ~256 workgroups (65*ceil(T/32)),
# leaving CUs idle; the small-R variant fills more of them. R=4 (128 threads
# per WG) yields many small workgroups that spread across CUs and reduce
# memory contention under arg-rotation / changing data_ptr scenarios.
SMALL_T_THRESHOLD = 96

# SQRT2 has no aiter dependency, so it stays at module level.
_SQRT2 = math.sqrt(2.0)


@lru_cache(maxsize=1)
def _fp8_const():
    """Lazy-resolve fp8 algebra coefficients (per-GFX native fp8).

    ``aiter.utility.dtypes.fp8`` selects e4m3fnuz on gfx942 MI300 and
    e4m3fn on gfx950 MI355 / gfx1250. ``cvt_pk_fp8_f32`` emits bytes in
    the per-gfx native format, so FP8_MAX must track that -- hardcoding
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


# --- supported quant-group sizes (1 x group_size block-scales) --------------
# group_size == head_dim -> per-row scale (single scale per token-head).
GROUP_SIZE_OPTIONS = (32, 64, 128)

# --- scale-dtype constants --------------------------------------------------
SCALE_DTYPE_FP32 = "fp32"
SCALE_DTYPE_E8M0 = "e8m0"
SCALE_DTYPE_OPTIONS = (SCALE_DTYPE_FP32, SCALE_DTYPE_E8M0)

_TORCH_DTYPE_FOR_SCALE = {
    SCALE_DTYPE_FP32: torch.float32,
    SCALE_DTYPE_E8M0: torch.uint8,  # no native torch e8m0 dtype; reinterpret as uint8
}


# ============================================================================
# Store helpers (module-level so they're easy to reuse / unit-test)
# ============================================================================


def _store_bf16_vec(vals_list, out_rsrc, row_base_bytes, idx, vec):
    """Convert VEC fp32 values to bf16, reinterpret as i32 dwords, and store
    via raw buffer_store. Handles VEC>8 by splitting into dwordx4 chunks.

    ``row_base_bytes`` is byte offset to the start of this head's row within
    the token (already token-shifted). ``idx`` is the lane id (tid).
    """
    i32 = T.i32
    f32 = T.f32
    vec_f32 = T.vec(vec, f32)
    vec_bf16 = T.vec(vec, T.bf16)
    raw = [v.ir_value() if hasattr(v, "ir_value") else v for v in vals_list]
    f32v = vector.from_elements(vec_f32, raw)
    bf16v = f32v.truncf(vec_bf16)

    # bf16 -> i32 dwords: VEC bf16 = VEC/2 dwords
    dwords = vec // 2
    bf16_as_i32 = vector.bitcast(T.vec(dwords, i32), bf16v)
    off_bytes = row_base_bytes + ArithValue(idx) * (vec * 2)

    if const_expr(dwords <= 4):
        buffer_ops.buffer_store(bf16_as_i32, out_rsrc, off_bytes, offset_is_bytes=True)
    else:
        # dwords > 4 (VEC=16 → dwords=8): split into 2× dwordx4
        lo = vector.extract_strided_slice(
            T.vec(4, i32), bf16_as_i32, offsets=[0], sizes=[4], strides=[1]
        )
        hi = vector.extract_strided_slice(
            T.vec(4, i32), bf16_as_i32, offsets=[4], sizes=[4], strides=[1]
        )
        buffer_ops.buffer_store(lo, out_rsrc, off_bytes, offset_is_bytes=True)
        buffer_ops.buffer_store(
            hi, out_rsrc, ArithValue(off_bytes) + 16, offset_is_bytes=True
        )


def _store_fp8_packed(
    vals_list, out_rsrc, row_base_bytes, idx, vec, *, skip_fnuz_clamp=False
):
    """Pack VEC fp32 -> VEC fp8 via cvt_pk_fp8_f32 and store.

    Generalized for VEC in {8, 16}: packs into VEC//4 dwords, stores as a
    single vector (dwordx2 for VEC=8, dwordx4 for VEC=16).

    Workaround for the e4m3fnuz NaN encoding 0x80: cvt_pk_fp8_f32 returns
    0x80 (NaN) for inputs that round to negative zero, which propagates
    through downstream attention as NaN. Clamp v in (-2^-8, 0) to +0 first.

    On gfx950+ (OCP e4m3fn), 0x80 encodes -0 (not NaN), so the clamp is
    unnecessary.  Pass ``skip_fnuz_clamp=True`` to elide it and save ~4
    ALU ops per element.
    """
    f32 = T.f32
    i32 = T.i32

    if skip_fnuz_clamp:
        safe = [v.ir_value() if hasattr(v, "ir_value") else v for v in vals_list]
    else:
        c0 = arith.constant(0.0, type=f32)
        c_neg_uf = arith.constant(-(2.0**-8), type=f32)
        safe = []
        for v in vals_list:
            vv = v.ir_value() if hasattr(v, "ir_value") else v
            is_tn = ArithValue(arith.cmpf(CmpFPredicate.OLT, vv, c0)) & arith.cmpf(
                CmpFPredicate.OGT, vv, c_neg_uf
            )
            safe.append(is_tn.select(c0, vv))

    # Pack each group of 4 fp32 into one i32 dword via cvt_pk_fp8_f32.
    n_dwords = vec // 4
    assert n_dwords in (2, 4), f"VEC={vec} -> n_dwords={n_dwords} unsupported"
    dword_list = []
    for dw_idx in range_constexpr(n_dwords):
        base = dw_idx * 4
        pk = arith.constant(0, type=i32)
        pk = rocdl.cvt_pk_fp8_f32(i32, safe[base + 0], safe[base + 1], pk, 0)
        pk = rocdl.cvt_pk_fp8_f32(i32, safe[base + 2], safe[base + 3], pk, 1)
        dword_list.append(pk)

    off_bytes = row_base_bytes + ArithValue(idx) * vec
    store_vec_ty = T.vec(n_dwords, i32)
    store_vec = vector.from_elements(store_vec_ty, dword_list)
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
    kv_write: bool = False,
    rows_per_wg: int = ROWS_PER_WG,
):
    """Build the @flyc.kernel + @flyc.jit launcher for a given config.

    All shape constants are captured via closure (NOT module globals), so two
    launchers with different (H, D, RD, group_size, scale_dtype, q_weighted)
    coexist safely. Returns the launcher.

    quant=True writes fp8 with one scale per ``group_size``-wide block of D.
    When ``group_size == head_dim`` the scale degenerates to per-row (NG=1).
    scale_dtype controls the stored scale encoding (``"fp32"`` or ``"e8m0"``).

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
    # Local rebind so every ROWS_PER_WG reference below picks up the per-build
    # value (adaptive: R=32 for prefill, R=16 for small-T decode).
    ROWS_PER_WG = rows_per_wg

    assert (
        D % BLOCK_THREADS == 0
    ), f"D={D} must be divisible by BLOCK_THREADS={BLOCK_THREADS}"
    assert NOPE % VEC == 0, f"NOPE={NOPE} must be divisible by VEC={VEC}"
    assert RD % 2 == 0, "rope_head_dim must be even (GPT-J pair layout)"
    assert RD % VEC == 0, f"RD={RD} must be divisible by VEC={VEC}"
    # gfx1250 wave32: VEC = D/32. D=512 → VEC=16, D=128 → VEC=4.
    assert VEC in (2, 4, 8, 16), (
        f"VEC={VEC} unsupported (D={D}, BLOCK_THREADS={BLOCK_THREADS}); "
        "supported set: {2, 4, 8, 16}."
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
    # amax must NOT cross groups -> only shuffles at offsets < TPG -> only at
    # the last log2(TPG) loop iterations (sh_exp >= amax_start_step).
    amax_start_step = log2_block - log2_tpg

    elem_dtype = fx.BFloat16
    is_e8m0 = scale_dtype == SCALE_DTYPE_E8M0

    # gfx1250 ships OCP e4m3fn (max_pos=448).
    _fp8_mx_dtype = _D.FP8_E4M3

    # Kernel name: include w32 suffix to avoid JIT cache collision with wave64.
    _name_parts = ["qk_norm_rope", f"H{H}", f"D{D}", f"RD{RD}"]
    if q_weighted:
        _name_parts.append("qw")
    if quant:
        _name_parts.append(f"g{group_size}")
        _name_parts.append(scale_dtype)
    if kv_write:
        _name_parts.append("kvw")
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
        state_slot_mapping: fx.Pointer,  # [bs] i32 (dummy if not kv_write)
        batch_id_per_token: fx.Pointer,  # [T] i32, -1 sentinel (dummy if not kv_write)
        swa_slot_stride: Int32,  # bf16 elements (= cache_size * D)
        swa_pos_stride: Int32,  # bf16 elements (= D)
        swa_cache_size: Int32,  # ring slot count
        num_tokens: Int32,  # valid tokens in this launch chunk (for tail clamp)
    ):
        f32 = T.f32
        i32 = T.i32
        fm_fast = arith.FastMathFlags.fast

        # --- vector load helpers (generalized for VEC ∈ {2..16}) ---
        # CopyAtom-based loads work for VEC ≤ 8 (BufferCopy128b = 16 bytes
        # = 8 bf16). For VEC=16 (32 bytes/thread), we use raw buffer_load
        # split into dwordx4 chunks, matching the compress_attn gfx1250 pattern.

        # Full-row loads via CopyAtom (for weight tensors).
        # VEC ≤ 8 → BufferCopy128b (16 bytes) is sufficient.
        # VEC = 16 → need 32 bytes; use raw buffer_load split instead.
        full_lay = fx.make_layout(VEC, 1)
        if const_expr(VEC <= 8):
            full_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 16)

            def _load_weight_tensor(weight_tensor, tid_val):
                """Load VEC bf16 from a 1D weight fx.Tensor at tid*VEC."""
                wbuf = fx.rocdl.make_buffer_tensor(weight_tensor)
                wdiv = fx.logical_divide(wbuf, full_lay)
                r = fx.make_rmem_tensor(full_lay, elem_dtype)
                fx.copy_atom_call(full_atom, fx.slice(wdiv, (None, tid_val)), r)
                return fx.memref_load_vec(r)

        else:

            def _load_weight_tensor(weight_tensor, tid_val):
                """Load VEC bf16 from a 1D weight fx.Tensor via raw buffer_load.
                Splits into dwordx4 chunks for VEC=16."""
                wrsrc = buffer_ops.create_buffer_resource(weight_tensor, max_size=True)
                tid_x_vec = ArithValue(tid_val) * VEC
                off_dw = tid_x_vec >> 1
                f32_list = _load_bf16_raw(wrsrc, off_dw)
                f32_vec = vector.from_elements(T.vec(VEC, f32), f32_list)
                rmem = fx.make_rmem_tensor(full_lay, fx.Float32)
                fx.memref_store_vec(f32_vec, rmem)
                return fx.memref_load_vec(rmem)

        def _load_bf16_raw(rsrc, off_dw):
            """Load VEC bf16 from a raw buffer resource at dword offset.
            Returns list of VEC f32 scalars. Splits into dwordx4 chunks
            when VEC > 8 (dwords > 4)."""
            dwords = VEC // 2
            out = []
            if const_expr(dwords <= 4):
                raw = buffer_ops.buffer_load(rsrc, off_dw, vec_width=dwords, dtype=i32)
                vec_bf16 = vector.bitcast(T.vec(VEC, T.bf16), raw)
                for i in range_constexpr(VEC):
                    bf16_v = vector.extract(
                        vec_bf16, static_position=[i], dynamic_position=[]
                    )
                    out.append(arith.extf(f32, bf16_v))
            else:
                half_dw = 4
                half_bf16 = half_dw * 2  # 8 bf16 per chunk
                for chunk in range_constexpr(dwords // half_dw):
                    r = buffer_ops.buffer_load(
                        rsrc,
                        ArithValue(off_dw) + (chunk * half_dw),
                        vec_width=half_dw,
                        dtype=i32,
                    )
                    vbf16 = vector.bitcast(T.vec(half_bf16, T.bf16), r)
                    for i in range_constexpr(half_bf16):
                        bf16_v = vector.extract(
                            vbf16, static_position=[i], dynamic_position=[]
                        )
                        out.append(arith.extf(f32, bf16_v))
            return out

        bid_x = fx.block_idx.x  # 0..H-1 (Q head) or H (KV)
        bid_t = fx.block_idx.y  # workgroup index along the token-chunk dim
        tid = fx.thread_idx.x
        tid_y = fx.thread_idx.y  # wave within workgroup -> token selector

        # Each workgroup owns ROWS_PER_WG consecutive waves; wave tid_y handles
        # token (bid_t*ROWS_PER_WG + tid_y). Clamp OOB tail waves to the last
        # valid token instead of branching: they recompute an already-valid row
        # and write byte-identical results (idempotent), so there is no OOB
        # access and no divergent bounds check on the hot path.
        tok = ArithValue(bid_t) * ROWS_PER_WG + ArithValue(tid_y)
        _nt_m1 = _to_raw(ArithValue(_to_raw(num_tokens)) - 1)
        tok = ArithValue(arith.minsi(_to_raw(tok), _nt_m1))
        bid_t = tok  # all downstream token offsets use the clamped token
        bid_t_idx = arith.index_cast(T.index, _to_raw(tok))

        def _ptr_buffer_resource(ptr, num_records_bytes=None):
            addr = fx.ptrtoint(ptr)
            addr_i64 = arith.index_cast(T.i64, addr)
            if num_records_bytes is None:
                return buffer_ops.create_buffer_resource_from_addr(addr_i64)
            return buffer_ops.create_buffer_resource_from_addr(
                addr_i64, num_records_bytes=num_records_bytes
            )

        # --- shared: load position (i64 -> i32) ---
        pos_rsrc = _ptr_buffer_resource(positions)
        pos_val_i64 = buffer_ops.buffer_load(pos_rsrc, bid_t, vec_width=1, dtype=T.i64)
        pos_i32 = arith.trunci(i32, pos_val_i64)

        # --- shared: cos/sin buffer resources (all threads load, NOPE
        # threads clamp index to 0 so the load is in-bounds/harmless) ---
        cos_rsrc = buffer_ops.create_buffer_resource(cos_cache, max_size=True)
        sin_rsrc = buffer_ops.create_buffer_resource(sin_cache, max_size=True)
        c_half_rd = arith.constant(RD // 2, type=i32)
        cos_sin_row_base = ArithValue(pos_i32) * c_half_rd

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
            # ---- Issue cos/sin loads EARLY so they overlap with the
            # sumsq reduction ALU below (latency hiding). ----
            is_rope_t = arith.cmpi(
                CmpIPredicate.sge,
                _to_raw(tid),
                arith.constant(ROPE_THREAD_LO, type=i32),
            )
            rope_rel_raw = ArithValue(tid) - arith.constant(ROPE_THREAD_LO, type=i32)
            rope_rel = arith.maxsi(_to_raw(rope_rel_raw), arith.constant(0, type=i32))
            cs_off = cos_sin_row_base + ArithValue(rope_rel) * arith.constant(
                PAIRS_PER_THREAD, type=i32
            )
            if const_expr(PAIRS_PER_THREAD == 1):
                cos_raw = buffer_ops.buffer_load(
                    cos_rsrc, cs_off, vec_width=1, dtype=T.bf16
                )
                sin_raw = buffer_ops.buffer_load(
                    sin_rsrc, cs_off, vec_width=1, dtype=T.bf16
                )
            else:
                cos_raw = buffer_ops.buffer_load(
                    cos_rsrc, cs_off, vec_width=PAIRS_PER_THREAD, dtype=T.bf16
                )
                sin_raw = buffer_ops.buffer_load(
                    sin_rsrc, cs_off, vec_width=PAIRS_PER_THREAD, dtype=T.bf16
                )

            # ---- RMSNorm: sumsq reduction (overlaps with cos/sin loads) ----
            x2 = x_f32_vec * x_f32_vec
            sq_local = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)

            if const_expr(quant):
                if const_expr(weighted):
                    xw = x_f32_vec * w_f32_vec
                    am_local = fmath.absf(xw).reduce(ReductionOp.MAX)
                else:
                    am_local = fmath.absf(x_f32_vec).reduce(ReductionOp.MAX)

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
                am_group = w_am
            else:
                sq_block = wave_reduce_add(sq_local)

            rstd = fmath.rsqrt(sq_block * (1.0 / D) + 1e-6, fastmath=fm_fast)

            if const_expr(quant):
                am_safe = arith.maximumf(am_group, arith.constant(1e-12, type=f32))

                if const_expr(is_e8m0):
                    c_sqrt2 = arith.constant(_SQRT2, type=f32)
                    amax_post = am_safe * rstd * c_sqrt2

                    e8m0_biased = emit_mx_e8m0_scale(
                        amax_post, mode=_DEFAULT_MODE, dtype=_fp8_mx_dtype
                    )
                    quant_exp = arith.constant(254, type=T.i32) - e8m0_biased
                    quant_scale = (quant_exp << 23).bitcast(T.f32)
                    factor = rstd * quant_scale
                else:
                    rcp_am = llvm.call_intrinsic(
                        f32, "llvm.amdgcn.rcp.f32", [am_safe], [], []
                    )
                    _fc = _fp8_const()
                    factor = arith.constant(_fc["max_over_sqrt2"], type=f32) * rcp_am
                    scale_val = (
                        am_safe * rstd * arith.constant(_fc["inv_max_sqrt2"], type=f32)
                    )

                group_idx = tid >> log2_tpg
                lane_in_group = tid & (TPG - 1)
                if lane_in_group == 0:
                    my_scale_off = scale_base_off + ArithValue(group_idx)
                    if const_expr(is_e8m0):
                        e8m0_i8 = arith.TruncIOp(T.i8, e8m0_biased).result
                        buffer_ops.buffer_store(e8m0_i8, scale_rsrc, my_scale_off)
                    else:
                        buffer_ops.buffer_store(scale_val, scale_rsrc, my_scale_off)

            # ---- Scale-multiply ----
            scaled = []
            for vi in range_constexpr(VEC):
                xi = x_f32_vec[vi]
                if const_expr(weighted):
                    xi = xi * w_f32_vec[vi]
                if const_expr(quant):
                    scaled.append(xi * factor)
                else:
                    scaled.append(xi * rstd)

            # ---- Extract cos/sin values (loads issued early, now consumed) ----
            if const_expr(PAIRS_PER_THREAD == 1):
                cos_vals = [arith.extf(f32, cos_raw)]
                sin_vals = [arith.extf(f32, sin_raw)]
            else:
                cos_vals = [
                    arith.extf(
                        f32,
                        vector.extract(
                            cos_raw, static_position=[i], dynamic_position=[]
                        ),
                    )
                    for i in range(PAIRS_PER_THREAD)
                ]
                sin_vals = [
                    arith.extf(
                        f32,
                        vector.extract(
                            sin_raw, static_position=[i], dynamic_position=[]
                        ),
                    )
                    for i in range(PAIRS_PER_THREAD)
                ]

            scaled_raw = [_to_raw(s) for s in scaled]
            rotated = list(scaled_raw)
            for k in range_constexpr(PAIRS_PER_THREAD):
                e = scaled_raw[2 * k]
                o = scaled_raw[2 * k + 1]
                c = cos_vals[k]
                s = sin_vals[k]
                rotated[2 * k] = arith.subf(
                    arith.MulFOp(e, c, fastmath=fm_fast).result,
                    arith.MulFOp(o, s, fastmath=fm_fast).result,
                )
                rotated[2 * k + 1] = arith.AddFOp(
                    arith.MulFOp(e, s, fastmath=fm_fast).result,
                    arith.MulFOp(o, c, fastmath=fm_fast).result,
                    fastmath=fm_fast,
                ).result

            final_list = [
                arith.select(is_rope_t, rotated[i], scaled_raw[i])
                for i in range_constexpr(VEC)
            ]

            if const_expr(quant):
                rsrc, row_base = fp8_out_rsrc
                _store_fp8_packed(
                    final_list, rsrc, row_base, tid, VEC, skip_fnuz_clamp=True
                )
            else:
                _store_bf16_vec(
                    final_list, bf16_out_rsrc, bf16_out_row_base_bytes, tid, VEC
                )
                if const_expr(kv_write):
                    if do_swa:
                        _store_bf16_vec(
                            final_list,
                            swa_out_rsrc,
                            swa_out_row_base_bytes,
                            tid,
                            VEC,
                        )

        # ============ runtime dispatch on bid_x < H ============
        q_tok_off_bytes = arith.MulIOp(
            bid_t_idx, arith.constant(H * D * 2, type=T.index)
        ).result

        if bid_x < H:
            # ---------- Q path ----------
            head_idx = bid_x
            # Q load: use raw buffer_load to handle VEC=16 correctly.
            q_in_rsrc = _ptr_buffer_resource(q_in)
            # Per-token byte offset → dword offset for Q input.
            q_row_off_elems = (
                ArithValue(bid_t) * (H * D)
                + ArithValue(head_idx) * D
                + ArithValue(tid) * VEC
            )
            q_off_dw = q_row_off_elems >> 1
            q_f32_list = _load_bf16_raw(q_in_rsrc, q_off_dw)
            q_f32_fly_vec = vector.from_elements(T.vec(VEC, f32), q_f32_list)
            q_rmem = fx.make_rmem_tensor(full_lay, fx.Float32)
            fx.memref_store_vec(q_f32_fly_vec, q_rmem)
            x_f32 = fx.memref_load_vec(q_rmem)

            # Optional per-channel Q weight.
            if const_expr(q_weighted):
                qw_vec = _load_weight_tensor(q_weight, tid)
                qw_f32 = qw_vec.to(fx.Float32)
            else:
                qw_f32 = None

            if const_expr(quant):
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
                row_base_bytes = ArithValue(head_idx) * D
                qs_rsrc = _ptr_buffer_resource(q_scale)
                scale_base_off_q = (
                    ArithValue(bid_t) * (H * NG) + ArithValue(head_idx) * NG
                )
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
                # BF16 store via raw buffer_store (handles VEC=16).
                qo_g_tmp = GTensor(
                    q_out,
                    dtype=T.bf16,
                    shape=(H, D),
                    static_bytes_offset_i64=q_tok_off_bytes,
                )
                qo_rsrc = qo_g_tmp.rsrc
                row_base_bytes_q = ArithValue(head_idx) * (D * 2)
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
            # ---------- KV path ----------
            kv_rsrc = _ptr_buffer_resource(kv_in)
            kv_off_elems = (
                ArithValue(bid_t) * ArithValue(kv_in_row_stride) + ArithValue(tid) * VEC
            )
            kv_off_dw = kv_off_elems >> 1

            kv_f32_list = _load_bf16_raw(kv_rsrc, kv_off_dw)
            kv_f32_fly_vec = vector.from_elements(T.vec(VEC, f32), kv_f32_list)
            kv_rmem = fx.make_rmem_tensor(full_lay, fx.Float32)
            fx.memref_store_vec(kv_f32_fly_vec, kv_rmem)
            x_vec_f32 = fx.memref_load_vec(kv_rmem)

            w_vec = _load_weight_tensor(kv_weight, tid)
            w_f32 = w_vec.to(fx.Float32)

            if const_expr(quant):
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
                row_base_bytes = arith.constant(0, type=i32)
                kvs_rsrc = _ptr_buffer_resource(kv_scale)
                scale_base_off_kv = ArithValue(bid_t) * NG
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
                kv_tok_off_bf16 = arith.MulIOp(
                    bid_t_idx, arith.constant(D * 2, type=T.index)
                ).result
                kvo_g_tmp = GTensor(
                    kv_out,
                    dtype=T.bf16,
                    shape=(D,),
                    static_bytes_offset_i64=kv_tok_off_bf16,
                )
                kvo_rsrc = kvo_g_tmp.rsrc
                row_base_bytes_kv = arith.constant(0, type=i32)

                # ---- Fused SWA scatter setup (kv_write only) ----
                swa_rsrc = None
                swa_row_base = None
                do_swa = None
                if const_expr(kv_write):
                    bid_rsrc = _ptr_buffer_resource(batch_id_per_token)
                    bid_i32 = buffer_ops.buffer_load(
                        bid_rsrc, bid_t, vec_width=1, dtype=i32
                    )
                    do_swa = ArithValue(bid_i32) >= 0
                    bid_safe = arith.maxsi(bid_i32, arith.constant(0, type=i32))
                    slot_rsrc = _ptr_buffer_resource(state_slot_mapping)
                    slot = buffer_ops.buffer_load(
                        slot_rsrc, bid_safe, vec_width=1, dtype=i32
                    )
                    ring = arith.remsi(pos_i32, _to_raw(swa_cache_size))
                    swa_off_elems = ArithValue(slot) * ArithValue(
                        swa_slot_stride
                    ) + ArithValue(ring) * ArithValue(swa_pos_stride)
                    swa_off_bytes = arith.index_cast(
                        T.index, _to_raw(swa_off_elems)
                    ) * arith.constant(2, type=T.index)
                    swa_g_tmp = GTensor(
                        swa_kv,
                        dtype=T.bf16,
                        shape=(D,),
                        static_bytes_offset_i64=swa_off_bytes,
                    )
                    swa_rsrc = swa_g_tmp.rsrc
                    swa_row_base = arith.constant(0, type=i32)

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
        state_slot_mapping: fx.Pointer,
        batch_id_per_token: fx.Pointer,
        swa_slot_stride: fx.Int32,
        swa_pos_stride: fx.Int32,
        swa_cache_size: fx.Int32,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # grid.y = ceil(num_tokens / ROWS_PER_WG): each workgroup covers
        # ROWS_PER_WG tokens (one per wave via thread_idx.y).
        _nt = ArithValue(_to_raw(num_tokens))
        _gy_i32 = arith.divsi(
            _to_raw(_nt + ROWS_PER_WG - 1),
            arith.constant(ROWS_PER_WG, type=T.i32),
        )
        idx_grid_y = arith.index_cast(T.index, _gy_i32)
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
            state_slot_mapping,
            batch_id_per_token,
            swa_slot_stride,
            swa_pos_stride,
            swa_cache_size,
            num_tokens,
        )
        k.launch(
            grid=(H + 1, idx_grid_y, 1),
            block=(BLOCK_THREADS, ROWS_PER_WG, 1),
            stream=stream,
        )

    return launch_qk_norm_rope_quant


# ============================================================================
# Cached compile + public API
# ============================================================================

_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 8,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


@lru_cache(maxsize=32)
def compile_flydsl_qk_norm_rope_quant_gfx1250(
    *,
    num_q_heads: int,
    head_dim: int,
    rope_head_dim: int,
    quant: bool,
    group_size: int,
    scale_dtype: str,
    q_weighted: bool,
    kv_write: bool = False,
    rows_per_wg: int = ROWS_PER_WG,
):
    """Compile (and cache) the gfx1250 wave32 launcher for a given config."""
    launcher = _build_kernel(
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        quant=quant,
        group_size=group_size,
        scale_dtype=scale_dtype,
        q_weighted=q_weighted,
        kv_write=kv_write,
        rows_per_wg=rows_per_wg,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


def flydsl_qk_norm_rope_quant_gfx1250(
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
    swa_kv: Optional[torch.Tensor] = None,
    state_slot_mapping: Optional[torch.Tensor] = None,
    batch_id_per_token: Optional[torch.Tensor] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Fused RMSNorm + GPT-J RoPE + optional FP8 quant (gfx1250 wave32).

    Same API as ``flydsl_qk_norm_rope_quant`` — see that docstring for full
    parameter descriptions. This variant compiles with BLOCK_THREADS=32 for
    RDNA4/gfx1250 wave32 hardware.
    """
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must be bf16, got {q.dtype}")
    if kv.dtype != torch.bfloat16:
        raise TypeError(f"kv must be bf16, got {kv.dtype}")
    if kv_weight.dtype != torch.bfloat16:
        raise TypeError(f"kv_weight must be bf16, got {kv_weight.dtype}")
    if kv.stride(-1) != 1:
        raise ValueError(f"kv must be dense in the last dim, stride={kv.stride()}")
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
        if q_view.stride(-1) != 1 or q_view.stride(-2) != D:
            raise ValueError(
                "3D q must be contiguous in the (H, D) inner block "
                f"(stride(-1)==1 and stride(-2)==D={D}), got stride={q_view.stride()}"
            )

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
    kv_write = swa_kv is not None
    if kv_write:
        if quant:
            raise ValueError("kv_write (swa_kv) is BF16 only; not supported with quant")
        if state_slot_mapping is None or batch_id_per_token is None:
            raise ValueError(
                "kv_write requires state_slot_mapping and batch_id_per_token"
            )
        if swa_kv.dim() != 3 or swa_kv.shape[2] != D:
            raise ValueError(f"swa_kv must be [S, C, D={D}], got {tuple(swa_kv.shape)}")
        if swa_kv.dtype != torch.bfloat16:
            raise TypeError(f"swa_kv must be bf16, got {swa_kv.dtype}")
        if not swa_kv.is_contiguous():
            raise ValueError("swa_kv must be contiguous")
        if state_slot_mapping.dim() != 1 or state_slot_mapping.dtype != torch.int32:
            raise TypeError("state_slot_mapping must be 1-D int32")
        if batch_id_per_token.dim() != 1 or batch_id_per_token.dtype != torch.int32:
            raise TypeError("batch_id_per_token must be 1-D int32")
        if batch_id_per_token.shape[0] < T_tok:
            raise ValueError(
                f"batch_id_per_token len {batch_id_per_token.shape[0]} < T={T_tok}"
            )
        swa_slot_stride = swa_kv.stride(0)
        swa_pos_stride = swa_kv.stride(1)
        swa_cache_size = swa_kv.shape[1]
        swa_kv_arg = swa_kv
        ssm_arg = state_slot_mapping
        bid_arg = batch_id_per_token
    else:
        swa_slot_stride = 0
        swa_pos_stride = 0
        swa_cache_size = 1
        swa_kv_arg = kv_out  # bf16 dummy
        ssm_arg = q.new_empty(1, dtype=torch.int32)
        bid_arg = q.new_empty(1, dtype=torch.int32)

    # Adaptive workgroup packing: small-T (decode) launches too few workgroups
    # at R=32 to fill all CUs, so use the small-R variant there; large-T
    # (prefill) keeps R=32 for max waves/workgroup. Selected once per launch by
    # total token count (chunking below stays within one regime for real cases).
    rows_per_wg = ROWS_PER_WG_SMALL if T_tok <= SMALL_T_THRESHOLD else ROWS_PER_WG

    launcher = compile_flydsl_qk_norm_rope_quant_gfx1250(
        num_q_heads=H,
        head_dim=D,
        rope_head_dim=RD,
        quant=quant,
        group_size=G,
        scale_dtype=scale_dtype,
        q_weighted=q_weighted,
        kv_write=kv_write,
        rows_per_wg=rows_per_wg,
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    def _has_direct_state():
        return getattr(launcher, "_direct_call_state", None) is not None

    def _ptr_arg(t):
        if _has_direct_state():
            return int(t.data_ptr())
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    def _stream_arg():
        if _has_direct_state():
            return stream
        return Stream(stream)

    q_weight_static = _cached_from_dlpack(q_weight_arg)
    kv_weight_static = _cached_from_dlpack(kv_weight)
    cos_static = _cached_from_dlpack(cos_2d)
    sin_static = _cached_from_dlpack(sin_2d)

    MAX_GRID_Y = 65535
    for start in range(0, T_tok, MAX_GRID_Y):
        n = min(MAX_GRID_Y, T_tok - start)
        end = start + n
        args = (
            _ptr_arg(q_view[start:end]),
            _ptr_arg(kv[start:end]),
            q_weight_static,
            kv_weight_static,
            cos_static,
            sin_static,
            _ptr_arg(positions[start:end]),
            _ptr_arg(q_out[start:end]),
            _ptr_arg(kv_out[start:end]),
            _ptr_arg(q_scale_arg[start:end] if quant else q_scale_arg),
            _ptr_arg(kv_scale_arg[start:end] if quant else kv_scale_arg),
            kv.stride(0),
            _ptr_arg(swa_kv_arg),
            _ptr_arg(ssm_arg),
            _ptr_arg(bid_arg[start:end] if kv_write else bid_arg),
            swa_slot_stride,
            swa_pos_stride,
            swa_cache_size,
            n,
            _stream_arg(),
        )
        _run_compiled(launcher, *args)

    return q_out, kv_out, (q_scale if quant else None), (kv_scale if quant else None)
