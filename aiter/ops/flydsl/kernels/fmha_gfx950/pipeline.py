# SPDX-License-Identifier: MIT
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Per-block forward pass: shared primitives, traits, and the kernel context."""

import math as host_math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, vector
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from aiter.ops.flydsl.kernels import buffer_ops

_LOG2E = host_math.log2(host_math.e)
_LN2 = 1.0 / _LOG2E

# log2 of e4m3's largest finite value, 448.
_P_HEADROOM_LOG2 = 8.807354922057604


LDS_BYTES_GFX950 = 160 * 1024


# The dual-wave 8-wave CTA fixes the q-block height; callers need it to count
# q-blocks before any traits object exists.
DUALWAVE_SWP_BLOCK_M = 256


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only."""
    rocdl.s_waitcnt(vmcnt=n)


def _s_setprio(val):
    rocdl.s_setprio(val)


def _read_exec_i64():
    """Read the current wave exec mask, matching Clang's builtin lowering."""
    true_i1 = fx.Boolean(True).ir_value()
    return rocdl.ballot(T.i64, true_i1)


def _ds_read_tr8_b64_imm(result_type, addr_i32, imm_offset=0):
    imm = int(imm_offset)
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    raw = llvm.inline_asm(
        raw_type,
        [as_mlir_value(addr_i32)],
        f"ds_read_b64_tr_b8 $0, $1 offset:{imm}\n",
        "=v,v,~{memory}",
        has_side_effects=True,
    )
    return vector.BitCastOp(result_type, raw).result


def _bitcast_i32(value):
    return as_mlir_value(fx.Float32(value).bitcast(fx.Int32).ir_value())


def _bitcast_f32(value):
    return as_mlir_value(fx.Int32(value).bitcast(fx.Float32).ir_value())


def _attn_mask_vec2_imm(rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32):
    """Causal pair mask: ``rel < thr ? -inf : score``, on the f32 bit patterns."""
    rel = fx.Int32(rel_i32)
    neg_inf = fx.Int32(neg_inf_i32)
    out_x = (rel < fx.Int32(thr_x)).select(neg_inf, fx.Int32(x_ref_i32))
    out_y = (rel < fx.Int32(thr_y)).select(neg_inf, fx.Int32(y_ref_i32))
    return out_x.ir_value(), out_y.ir_value()


def _reduction_pair(v_f32):
    v_i32 = _bitcast_i32(v_f32)
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_ty, v_i32, v_i32, False, True)
    lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
    rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
    return _bitcast_f32(lhs_i32), _bitcast_f32(rhs_i32)


def _anchor_scalar_f32(x):
    """Pin a scalar f32 at the current source position (no-op asm)."""
    x_ir = as_mlir_value(x)
    return llvm.inline_asm(
        x_ir.type,
        [x_ir],
        "",
        "=v,0",
        has_side_effects=True,
    )


def _anchor_v_o(traits, v_o):
    """Pin v_o accumulators at the current source position."""
    acc_irs = [as_mlir_value(v_o[dc]) for dc in range_constexpr(traits.D_CHUNKS)]
    ret_ty = ir.Type.parse(
        f"!llvm.struct<({', '.join(['vector<16xf32>'] * traits.D_CHUNKS)})>"
    )
    constraints = ",".join(
        ["=v"] * traits.D_CHUNKS + [str(i) for i in range(traits.D_CHUNKS)]
    )
    ret = llvm.inline_asm(
        ret_ty,
        acc_irs,
        "",
        constraints,
        has_side_effects=True,
    )
    return [
        llvm.extractvalue(acc_irs[dc].type, ret, [dc])
        for dc in range_constexpr(traits.D_CHUNKS)
    ]


def _score_pair_to_lists(v_s):
    s_lo, s_hi = v_s
    return (
        [Vec(s_lo)[r] for r in range_constexpr(16)],
        [Vec(s_hi)[r] for r in range_constexpr(16)],
    )


def _score_lists_to_vecs(v_s_lists):
    s_lo, s_hi = v_s_lists
    return (
        Vec.from_elements([as_mlir_value(v) for v in s_lo], fx.Float32).ir_value(),
        Vec.from_elements([as_mlir_value(v) for v in s_hi], fx.Float32).ir_value(),
    )


def _reduce_score_pair(v_s, initial, reducer, fm_fast):
    s_lo, s_hi = v_s
    acc = initial
    for r in range_constexpr(16):
        acc = reducer(acc, s_lo[r], fm_fast)
    for r in range_constexpr(16):
        acc = reducer(acc, s_hi[r], fm_fast)
    return acc


def _lane_pair_reduce(v, reducer, fm_fast):
    lhs, rhs = _reduction_pair(v)
    return reducer(lhs, rhs, fm_fast)


def _score_pair_max(v_s, neg_inf, fm_fast):
    reducer = lambda a, b, _fm: fx.maxnumf(a, b)
    return _lane_pair_reduce(
        _reduce_score_pair(v_s, neg_inf, reducer, fm_fast), reducer, fm_fast
    )


def _score_pair_sum(v_s, zero_f, fm_fast):
    s_lo, s_hi = _score_lists_to_vecs(v_s)
    tile = Vec(s_lo) + Vec(s_hi)
    reducer = lambda a, b, _fm: a + b
    return _lane_pair_reduce(
        tile.reduce("add", init_val=zero_f, fastmath=fm_fast), reducer, fm_fast
    )


def _scale_sub_score_pair(v_s, row_max_raw, scale, zero_f, fm_fast, bias=None):
    """Fused softmax-scale + row-max subtraction (optimization 1-A).

    Returns ``scale * (v_s - row_max_raw) + bias`` per element via a single FMA
    (``fma(s, scale, bias - scale*row_max_raw)``), so the fp8 QK MMA can emit raw
    (un-scaled) logits and reduce_max can run in the raw domain (scale > 0 is
    order-preserving). Replaces the separate post-QK scale multiply + subtract.
    ``-inf`` masked lanes stay ``-inf`` (scale > 0), matching the un-fused path.

    ``bias`` lands in the FMA's addend, so a caller needing ``exp2`` to produce
    ``2**bias * P`` pays nothing -- see ``DualwaveFp8SoftmaxHelper.sub_m``.
    """
    s_lo, s_hi = v_s
    neg_scaled_max = zero_f - scale * row_max_raw
    if bias is not None:
        # exp2 lands on 2**bias * P instead of P, at no extra instruction: the
        # FMA's addend absorbs it.
        neg_scaled_max = neg_scaled_max + bias
    scale_v = Vec.from_elements([scale], fx.Float32).broadcast_to(16)
    nsm_v = Vec.from_elements([neg_scaled_max], fx.Float32).broadcast_to(16)
    lo = fx.fma(Vec(s_lo), scale_v, nsm_v, fastmath=fm_fast)
    hi = fx.fma(Vec(s_hi), scale_v, nsm_v, fastmath=fm_fast)
    return as_mlir_value(lo), as_mlir_value(hi)


def _exp2_score_slice(v_s, start):
    if const_expr(start == 0):
        s_lo = [Vec(v_s[0])[r] for r in range_constexpr(16)]
        lo_partial = []
        for r in range_constexpr(16):
            lo_partial.append(rocdl.exp2(T.f32, as_mlir_value(s_lo[r])))
        return Vec.from_elements(lo_partial, fx.Float32).ir_value(), v_s[1]

    lo_partial = [Vec(v_s[0])[r] for r in range_constexpr(16)]
    hi_full = []
    for r in range_constexpr(16):
        hi_full.append(rocdl.exp2(T.f32, as_mlir_value(Vec(v_s[1])[r])))
    return lo_partial, hi_full


def _scale_o_accs(v_o, scale_scalar, traits):
    scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
    for dc in range_constexpr(traits.D_CHUNKS):
        v_o[dc] = Vec(v_o[dc]) * scale_vec


def _causal_pair_thresholds(kv_vectorized):
    if const_expr(kv_vectorized):
        return [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (16, 17),
            (18, 19),
            (20, 21),
            (22, 23),
        ]
    return [
        (0, 1),
        (2, 3),
        (8, 9),
        (10, 11),
        (16, 17),
        (18, 19),
        (24, 25),
        (26, 27),
    ]


def _apply_dualwave_causal_mask_pair(s_values, rel_i32, neg_inf_i32, pair_thresholds):
    for p in range_constexpr(len(pair_thresholds)):
        thr_x, thr_y = pair_thresholds[p]
        idx_x = p * 2
        idx_y = p * 2 + 1
        x_bits = _bitcast_i32(s_values[idx_x])
        y_bits = _bitcast_i32(s_values[idx_y])
        new_x, new_y = _attn_mask_vec2_imm(
            rel_i32, neg_inf_i32, thr_x, thr_y, x_bits, y_bits
        )
        s_values[idx_x] = _bitcast_f32(new_x)
        s_values[idx_y] = _bitcast_f32(new_y)


def _cu_load(div, idx, cu_atom, cu_v1i32):
    """Load cu_seqlens[idx] into an SGPR. ``idx`` must be wave-uniform."""
    v = fly.copy_atom_call_ssa(
        [cu_v1i32], cu_atom, fx.slice(div, (None, fx.Int32(idx)))
    )
    return fx.Index(
        rocdl.readfirstlane(T.i32, as_mlir_value(fx.Int32(Vec(v, (1,), fx.Int32)[0])))
    )


def _make_ws_rsrc(ws_base_i64, byte_offset, nrec_bytes):
    addr_i64 = as_mlir_value(ws_base_i64 + fx.Int64(byte_offset))
    return buffer_ops.create_buffer_resource_from_addr(
        addr_i64, num_records_bytes=as_mlir_value(fx.Int64(nrec_bytes))
    )


def _p_headroom_log2(traits):
    """Exponent bias `sub_m` folds into P, so l_row carries a 2**this factor."""
    h = _P_HEADROOM_LOG2
    if traits.DUALWAVE_SWP_LAZY_RESCALE:
        h -= traits.DUALWAVE_SWP_RESCALE_THRESHOLD
    return max(0.0, h)


def _store_lse(ctx, row, m_scaled, l_row, in_range, is_writer):
    """LSE = m*ln2 + ln(l) - headroom*ln2, undoing the P headroom l_row carries.

    ``m_scaled`` is the row max already scaled by ``c_logit_scale`` (log2 domain,
    softmax scale folded); the result is a natural log. Dense LSE is [B, H, Sq]
    (per-batch slice), varlen [H, total_q] (per-head), so lse_stride_h is Sq resp.
    total_q. One writer per row; everyone else aims past num_records, which drops
    the store.
    """
    traits = ctx.traits
    if const_expr(traits.VARLEN):
        slice_elems = ctx.lse_stride_h_v
        slice_off = ctx.q_head_idx * slice_elems
        local = ctx.q_tok_base + row
    else:
        slice_elems = traits.NUM_HEADS_Q * ctx.lse_stride_h_v
        slice_off = ctx.batch_idx * slice_elems
        local = ctx.q_head_idx * ctx.lse_stride_h_v + row
    lse_rsrc = _make_ws_rsrc(
        fx.Int64(fx.ptrtoint(fx.get_iter(ctx.LSE))), slice_off * 4, slice_elems * 4
    )
    lse = (
        fx.Float32(m_scaled) * fx.Float32(_LN2)
        + fx.log(fx.Float32(l_row), fastmath=ctx.fm_fast)
        + fx.Float32(-_p_headroom_log2(traits) * _LN2)
    )
    off = is_writer.select(in_range.select(local, slice_elems), slice_elems)
    buffer_ops.buffer_store(lse.ir_value(), lse_rsrc, fx.Int32(off).ir_value())


def _buffer_load_128(elem_index, _load_atom_128, q_div, q_load_i32x4_type):
    """128-bit global->register load (buffer_load_dwordx4) from Q."""
    return fly.copy_atom_call_ssa(
        [q_load_i32x4_type],
        _load_atom_128,
        fx.slice(q_div, (None, fx.Int32(elem_index))),
    )


def _buffer_load_lds_128(
    src_div, lds_byte_addr, src_elem, soffset_elems, _dma_atom, _lds_ptr_ty
):
    """128-bit global->LDS DMA; `src_elem` is voffset, `soffset_elems` is scaled by the atom."""
    lds_ptr = fx.inttoptr(_lds_ptr_ty, fx.Int32(lds_byte_addr))
    dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
    src = fx.slice(src_div, (None, fx.Int32(src_elem)))
    fx.copy(_dma_atom, src, dst, soffset=fx.Int32(soffset_elems))


def _buffer_store_128(
    pack_i32_vec, elem_index, _o_store_reg_128, _store_atom_128, o_div
):
    """128-bit register->global store (buffer_store_dwordx4) into O."""
    fx.memref_store_vec(pack_i32_vec, _o_store_reg_128)
    fx.copy(
        _store_atom_128, _o_store_reg_128, fx.slice(o_div, (None, fx.Int32(elem_index)))
    )


@flyc.jit
def _stagger_extra_barrier_if_one(stagger_i32):
    """Emit `sched_barrier(0); s_barrier;` only when stagger == 1."""
    if fx.Int32(stagger_i32) != fx.Int32(0):
        rocdl.sched_barrier(0)
        rocdl.s_barrier()


@dataclass(frozen=True)
class DualwaveSwpFp8Traits:
    """Pure compile-time tile/layout constants for the gfx950 DUALWAVE_SWP fp8 kernel."""

    BLOCK_M: int
    BLOCK_N: int
    WARP_SIZE: int
    NUM_WAVES: int
    BLOCK_SIZE: int
    ROWS_PER_WAVE: int
    HEAD_DIM: int
    HEAD_DIM_V: int
    D_CHUNK: int
    D_CHUNKS: int
    PV_K_STEPS: int
    NUM_HEADS_Q: int
    NUM_HEADS_KV: int
    GQA_GROUP_SIZE: int
    CAUSAL: bool
    DAZ: bool
    DUALWAVE_SWP_LAZY_RESCALE: bool
    DUALWAVE_SWP_SETPRIO: bool
    DUALWAVE_SWP_ENABLE_STAGGER: bool
    NUM_KV_SPLITS: int
    SPLITK: bool
    VARLEN: bool
    CROSS_SEQLEN: bool
    DEFAULT_STRIDE_Q_N: int
    DEFAULT_STRIDE_KV_N: int
    DEFAULT_STRIDE_V_N: int
    DEFAULT_STRIDE_O_N: int
    QLDS: bool
    K_BAND_CHUNK: tuple[int, ...]
    K_BAND_BASE: tuple[int, ...]
    K_BAND_LINE_STRIDE: tuple[int, ...]
    K_BAND_GLOBAL_D: tuple[int, ...]
    K_WS_BAND: tuple[int, ...]
    K_WS_OFF: tuple[int, ...]
    DMA_BYTES: int
    ELEM_BYTES: int
    OUT_ELEM_BYTES: int
    VEC_KV: int
    LANE_SPLIT_KV: int
    SMEM_K_TILE_ELEMS: int
    NUM_PREFETCH_K: int
    DUALWAVE_SWP_KV_PER_BUFFER: int
    LDS_KV_TOTAL_SIZE: int
    DUALWAVE_SWP_K_BUF_BASE: tuple[int, int]
    VT_BF16_TOTAL: int
    DUALWAVE_SWP_RESCALE_THRESHOLD: float
    SCHED_MFMA_MASK: int
    SCHED_DS_READ_MASK: int
    NEG_INF_F32_BITS: int
    BATCH_INTERLEAVE_GROUP: int = 1
    RETURN_LSE: bool = False

    @property
    def cache_tag(self):
        """The independent builder arguments, and nothing derived from them.

        Every other trait is a pure function of these (verified by enumerating the
        whole argument grid and checking this tuple stays injective), so adding one
        cannot separate two builds that would otherwise share a binary.
        """
        return (
            "fp8_e4m3_dualwave_swp",
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.HEAD_DIM,
            self.HEAD_DIM_V,
            self.BLOCK_M,
            self.CAUSAL,
            self.DAZ,
            self.DUALWAVE_SWP_LAZY_RESCALE,
            self.DUALWAVE_SWP_RESCALE_THRESHOLD,
            self.DUALWAVE_SWP_SETPRIO,
            self.DUALWAVE_SWP_ENABLE_STAGGER,
            self.NUM_KV_SPLITS,
            self.VARLEN,
            self.CROSS_SEQLEN,
            self.BATCH_INTERLEAVE_GROUP,
            self.RETURN_LSE,
        )


def _make_dualwave_swp_fp8_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    rescale_threshold,
    head_dim_v=None,
    block_m=256,
    causal=True,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    dualwave_swp_setprio=True,
    dualwave_swp_enable_stagger=True,
    num_kv_splits=1,
    varlen=False,
    cross_seqlen=False,
    batch_interleave_group=1,
    return_lse=False,
):
    """Build gfx950 DUALWAVE_SWP fp8 compile-time layout traits.

    ``head_dim`` is the QK reduction width (a multiple of 64: the QK MFMA is
    32x32x64) and ``head_dim_v`` the V/output width, tiled in 32-wide D_CHUNKs.
    """
    if head_dim_v is None:
        head_dim_v = head_dim
    if head_dim % 64:
        raise RuntimeError(
            f"fp8 flash attention needs head_dim % 64 == 0, got head_dim={head_dim}"
        )
    # D_CHUNKS == head_dim_v // 32 must land in [2, 6]: below 2 `_anchor_v_o`
    # aborts LLVM, above 6 the high D_CHUNKs come back wrong.
    if head_dim_v % 32 or not 64 <= head_dim_v <= 192:
        raise RuntimeError(
            "fp8 flash attention needs 64 <= head_dim_v <= 192 and head_dim_v % 32 == 0, "
            f"got head_dim_v={head_dim_v} (head_dim={head_dim})"
        )
    block_n = 64
    k_sub_n = 32
    warp_size = 64
    rows_per_wave = 32
    if block_m % rows_per_wave or block_m // rows_per_wave not in (4, 8):
        raise RuntimeError(
            f"fp8 flash attention supports block_m 128 (4 waves) or 256 (8 waves), got {block_m}"
        )
    num_waves = block_m // rows_per_wave
    block_size = num_waves * warp_size

    d_chunk = 32
    d_chunks = head_dim_v // d_chunk
    pv_k_step = 16
    pv_k_steps = k_sub_n // pv_k_step

    gqa_group_size = num_heads // num_kv_heads
    default_stride_q_n = num_heads * head_dim
    default_stride_kv_n = num_kv_heads * head_dim
    default_stride_v_n = num_kv_heads * head_dim_v
    default_stride_o_n = num_heads * head_dim_v

    # fp8: Q/K/V are 1B; O is bf16 (2B). ELEM_BYTES=1 drives the fp8 address math.
    elem_bytes = 1
    out_elem_bytes = 2
    vec_kv = 16 // elem_bytes
    lane_split_kv = 8
    smem_k_pad = 16 // elem_bytes

    rows_per_wave_dma = -(-block_n // num_waves)
    k_band_chunk, k_band_base, k_band_line_stride, k_band_global_d = [], [], [], []
    _off, _cursor = 0, 0
    while _off < head_dim:
        chunk = min(128, head_dim - _off)
        line_stride = rows_per_wave_dma * chunk + smem_k_pad
        k_band_chunk.append(chunk)
        k_band_base.append(_cursor)
        k_band_line_stride.append(line_stride)
        k_band_global_d.append(_off)
        _cursor += num_waves * line_stride
        _off += chunk
    smem_k_tile_elems = _cursor
    k_ws_band, k_ws_off = [], []
    for _bi, chunk in enumerate(k_band_chunk):
        for _o in range(0, chunk, 64):
            k_ws_band.append(_bi)
            k_ws_off.append(_o)
    num_prefetch_k = 6
    dualwave_swp_kv_per_buffer = smem_k_tile_elems
    lds_kv_total_size = num_prefetch_k * dualwave_swp_kv_per_buffer
    dualwave_swp_k_buf_base = tuple(
        i * dualwave_swp_kv_per_buffer for i in range(num_prefetch_k)
    )

    # The +128 covers the alignment the DMA base is rounded up to.
    eb_bf = 2
    fp8_v_tile_bytes = (block_n // 8) * (head_dim_v // 16) * 128
    vt_bf16_total = num_prefetch_k * (fp8_v_tile_bytes // eb_bf) + 128

    splitk = num_kv_splits > 1

    qlds = head_dim <= 128

    lds_bytes = lds_kv_total_size * elem_bytes + vt_bf16_total * eb_bf
    if qlds:
        lds_bytes += block_m * head_dim * elem_bytes
    if lds_bytes > LDS_BYTES_GFX950:
        raise RuntimeError(
            f"fp8 flash attention head_dim={head_dim}/head_dim_v={head_dim_v} at block_m={block_m} "
            f"needs {lds_bytes} B of LDS, over the {LDS_BYTES_GFX950} B gfx950 workgroup limit. "
            "Largest head_dim_v that fits: 192 at head_dim 64/128/192, 160 at 256, 96 at 320; "
            "head_dim 384 and above never fits."
        )

    return DualwaveSwpFp8Traits(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        WARP_SIZE=warp_size,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=block_size,
        ROWS_PER_WAVE=rows_per_wave,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=head_dim_v,
        D_CHUNK=d_chunk,
        D_CHUNKS=d_chunks,
        PV_K_STEPS=pv_k_steps,
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        CAUSAL=causal,
        DAZ=bool(daz),
        DUALWAVE_SWP_LAZY_RESCALE=bool(dualwave_swp_lazy_rescale),
        DUALWAVE_SWP_SETPRIO=bool(dualwave_swp_setprio),
        DUALWAVE_SWP_ENABLE_STAGGER=bool(dualwave_swp_enable_stagger),
        NUM_KV_SPLITS=num_kv_splits,
        SPLITK=splitk,
        VARLEN=bool(varlen),
        CROSS_SEQLEN=bool(cross_seqlen),
        DEFAULT_STRIDE_Q_N=default_stride_q_n,
        DEFAULT_STRIDE_KV_N=default_stride_kv_n,
        DEFAULT_STRIDE_V_N=default_stride_v_n,
        DEFAULT_STRIDE_O_N=default_stride_o_n,
        QLDS=bool(qlds),
        K_BAND_CHUNK=tuple(k_band_chunk),
        K_BAND_BASE=tuple(k_band_base),
        K_BAND_LINE_STRIDE=tuple(k_band_line_stride),
        K_BAND_GLOBAL_D=tuple(k_band_global_d),
        K_WS_BAND=tuple(k_ws_band),
        K_WS_OFF=tuple(k_ws_off),
        DMA_BYTES=16,
        ELEM_BYTES=elem_bytes,
        OUT_ELEM_BYTES=out_elem_bytes,
        VEC_KV=vec_kv,
        LANE_SPLIT_KV=lane_split_kv,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        NUM_PREFETCH_K=num_prefetch_k,
        DUALWAVE_SWP_KV_PER_BUFFER=dualwave_swp_kv_per_buffer,
        LDS_KV_TOTAL_SIZE=lds_kv_total_size,
        DUALWAVE_SWP_K_BUF_BASE=dualwave_swp_k_buf_base,
        VT_BF16_TOTAL=vt_bf16_total,
        DUALWAVE_SWP_RESCALE_THRESHOLD=rescale_threshold,
        SCHED_MFMA_MASK=0x008,
        SCHED_DS_READ_MASK=0x100,
        NEG_INF_F32_BITS=0xFF800000,
        BATCH_INTERLEAVE_GROUP=int(batch_interleave_group),
        RETURN_LSE=bool(return_lse),
    )


def dualwave_fp8_dma_per_iter(traits):
    rows_per_wave = -(-traits.BLOCK_N // traits.NUM_WAVES)
    k_instr = sum(
        -(-(rows_per_wave * (chunk // traits.VEC_KV)) // traits.WARP_SIZE)
        for chunk in traits.K_BAND_CHUNK
    )
    num_dma_v = (traits.BLOCK_N * (traits.HEAD_DIM_V // 16) * 16) // (
        traits.WARP_SIZE * traits.VEC_KV * traits.ELEM_BYTES
    )
    v_instr_min = num_dma_v // traits.NUM_WAVES
    return 2 * k_instr + 2 * v_instr_min


def _init_dualwave_thread_mapping(ctx):
    """Set block/wave/lane/head indices on a dualwave-style context.

    Sets h_idx / q_block_idx / batch_idx / split_idx and the lane decomposition."""
    traits = ctx.traits
    batch_interleave_group = traits.BATCH_INTERLEAVE_GROUP
    if const_expr(batch_interleave_group > 1):
        linear_head_batch = fx.Index(gpu.block_idx.x)
        ctx.h_idx = linear_head_batch % traits.NUM_HEADS_Q
        ctx.batch_idx = (
            fx.Index(gpu.block_idx.z) * batch_interleave_group
            + linear_head_batch // traits.NUM_HEADS_Q
        )
        ctx.q_block_idx = fx.Index(gpu.block_idx.y)
    else:
        ctx.h_idx = fx.Index(gpu.block_idx.x)
        ctx.q_block_idx = fx.Index(gpu.block_idx.y)
    if const_expr(traits.SPLITK):
        ctx.bz_idx = fx.Index(gpu.block_idx.z)
        ctx.batch_idx = ctx.bz_idx // traits.NUM_KV_SPLITS
        ctx.split_idx = ctx.bz_idx % traits.NUM_KV_SPLITS
    elif const_expr(batch_interleave_group > 1):
        ctx.split_idx = None
    else:
        ctx.batch_idx = fx.Index(gpu.block_idx.z)
        ctx.split_idx = None
    ctx.tid = fx.Index(gpu.thread_idx.x)

    ctx.wave_id = ctx.tid // traits.WARP_SIZE
    ctx.lane = ctx.tid % traits.WARP_SIZE
    ctx.lane_mod_32 = ctx.lane % 32
    ctx.lane_div_32 = ctx.lane // 32

    _tid_i32 = fx.Int32(ctx.tid)
    _wave_id_uni_i32 = rocdl.readfirstlane(
        T.i32,
        (_tid_i32 // fx.Int32(traits.WARP_SIZE)).ir_value(),
    )
    # Two stagger groups, whatever the wave count.
    ctx.stagger_i32 = arith.divsi(
        _wave_id_uni_i32, as_mlir_value(fx.Int32(traits.NUM_WAVES // 2))
    )
    ctx.wave_id_uni = fx.Index(_wave_id_uni_i32)

    ctx.wave_q_offset = ctx.wave_id * traits.ROWS_PER_WAVE
    ctx.q_start = ctx.q_block_idx * traits.BLOCK_M

    ctx.h_kv_idx = ctx.h_idx % traits.NUM_HEADS_KV
    ctx.group_id = ctx.h_idx // traits.NUM_HEADS_KV
    ctx.q_head_idx = ctx.h_kv_idx * traits.GQA_GROUP_SIZE + ctx.group_id
    ctx.kv_head_idx = ctx.h_kv_idx


def _init_dualwave_q_row(ctx):
    """Set q_row / q_row_i32 / q_start_pos_i32 on a dualwave-style context."""
    traits = ctx.traits
    ctx.q_row_in_block = ctx.wave_q_offset + ctx.lane_mod_32
    ctx.q_start_pos_i32 = fx.Int32(ctx.q_start + ctx.wave_id_uni * traits.ROWS_PER_WAVE)
    ctx.q_row = ctx.q_start + ctx.q_row_in_block
    ctx.q_row_i32 = fx.Int32(ctx.q_row)


class DualwaveFp8KernelContext:
    """Shared per-kernel state for the gfx950 dualwave fp8 attention helpers.

    Raw fp8 Q/K/V
    (i8 buffer views), per-tensor Q/K/V descale scalars applied to the fp32 logits,
    and a bf16 ``vt`` LDS scratch for HIPREC PV."""

    def __init__(
        self,
        traits_or_ctx,
        Q=None,
        K=None,
        V=None,
        O=None,
        Workspace=None,
        CuSeqQ=None,
        CuSeqKv=None,
        QDescale=None,
        KDescale=None,
        VDescale=None,
        LSE=None,
        seq_len=None,
        seq_len_kv=None,
        stride_q_n=None,
        stride_kv_n=None,
        head_dim_runtime=None,
        lse_stride_h=None,
    ):
        if isinstance(traits_or_ctx, DualwaveFp8KernelContext):
            self.__dict__.update(traits_or_ctx.__dict__)
            self.ctx_ref = getattr(traits_or_ctx, "ctx_ref", traits_or_ctx)
            return
        self.ctx_ref = self
        self.traits = traits_or_ctx
        self.Q = Q
        self.K = K
        self.V = V
        self.O = O
        self.Workspace = Workspace
        self.CuSeqQ = CuSeqQ
        self.CuSeqKv = CuSeqKv
        self.QDescale = QDescale
        self.KDescale = KDescale
        self.VDescale = VDescale
        self.LSE = LSE
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.stride_q_n = stride_q_n
        self.stride_kv_n = stride_kv_n
        self.head_dim_runtime = head_dim_runtime
        self.lse_stride_h = lse_stride_h

    def init_types_and_constants(self):
        traits = self.traits
        self.elem_dtype = fx.Float8E4M3FN
        self.fm_fast = fx.arith.FastMathFlags.fast
        self.v4i32_type = Vec.make_type(4, fx.Int32)
        self.v16f32_type = Vec.make_type(16, fx.Float32)
        self.v2i32_type = Vec.make_type(2, fx.Int32)
        self.NUM_DMA_K = len(traits.K_BAND_CHUNK)
        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_neg_floor = fx.Float32(-3.0e38)
        self.c_zero_f = fx.Float32(0.0)
        self.c_rescale_thr_f = fx.Float32(traits.DUALWAVE_SWP_RESCALE_THRESHOLD)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)

    def init_runtime_indices(self):
        traits = self.traits
        self.seq_len_v = fx.Index(self.seq_len)
        self.seq_len_kv_v = fx.Index(self.seq_len_kv)
        if const_expr(traits.RETURN_LSE):
            self.lse_stride_h_v = fx.Index(self.lse_stride_h)
        self.stride_q_n_v = fx.Index(self.stride_q_n)
        self.stride_kv_n_v = fx.Index(self.stride_kv_n)
        if traits.HEAD_DIM_V == traits.HEAD_DIM:
            self.stride_v_n_v = self.stride_kv_n_v
            self.stride_o_n_v = self.stride_q_n_v
        else:
            self.stride_v_n_v = traits.DEFAULT_STRIDE_V_N
            self.stride_o_n_v = traits.DEFAULT_STRIDE_O_N

    def init_causal_lpt_order(self):
        """Issue causal q-blocks longest-first by reversing the q-block grid axis.

        Causal work per q-block grows with the block index and workgroups dispatch in
        flattened-id order, so the natural order issues the heaviest block last and the
        makespan carries its tail. Must run after init_thread_mapping and before
        init_sequence_lengths / init_tile_bounds / init_q_row read q_start.
        """
        traits = self.traits
        num_q_blocks = (self.seq_len_v + traits.BLOCK_M - 1) // traits.BLOCK_M
        self.q_block_idx = num_q_blocks - 1 - self.q_block_idx
        self.q_start = self.q_block_idx * traits.BLOCK_M

    def init_lds(self, shared_storage):
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.kv.ptr))
        self.lds_kv_base_ptr = lds.kv.ptr.llvm_ptr
        self.lds_vt_base_idx = fx.Index(fx.ptrtoint(lds.vt.ptr))
        self.lds_vt_base_ptr = lds.vt.ptr.llvm_ptr
        self.lds_q_base_idx = fx.Index(fx.ptrtoint(lds.q.ptr))
        self.lds_q_base_ptr = lds.q.ptr.llvm_ptr

    def init_thread_mapping(self):
        _init_dualwave_thread_mapping(self)

    def init_dma_thread_offsets(self):
        # Emitted after descriptors/atoms (matching the original schedule) so the
        # d_bucket ``v_and`` lands at the same ISA position.
        traits = self.traits
        self.lane_in_warp = self.tid % traits.WARP_SIZE
        self.n_in_warp = self.lane_in_warp // traits.LANE_SPLIT_KV
        self.d_bucket = self.lane_in_warp % traits.LANE_SPLIT_KV

    def init_sequence_lengths(self):
        traits = self.traits
        if const_expr(traits.VARLEN):
            _cuq_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(self.CuSeqQ), fx.make_layout(1, 1)
            )
            _cuk_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(self.CuSeqKv), fx.make_layout(1, 1)
            )
            _cu_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            _cu_v1i32 = Vec.make_type(1, fx.Int32)

            self.q_tok_base = _cu_load(_cuq_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.q_tok_end = _cu_load(_cuq_div, self.batch_idx + 1, _cu_atom, _cu_v1i32)
            self.kv_tok_base = _cu_load(_cuk_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.kv_tok_end = _cu_load(
                _cuk_div, self.batch_idx + 1, _cu_atom, _cu_v1i32
            )
            self.seqlen_q_v = self.q_tok_end - self.q_tok_base
            self.seqlen_kv_v = self.kv_tok_end - self.kv_tok_base
            self.seqlen_kv_i32 = fx.Int32(self.seqlen_kv_v)
        else:
            self.q_tok_base = self.batch_idx * self.seq_len_v
            self.kv_tok_base = self.batch_idx * self.seq_len_kv_v
            self.q_tok_end = (self.batch_idx + 1) * self.seq_len_v
            self.kv_tok_end = (self.batch_idx + 1) * self.seq_len_kv_v
            self.seqlen_q_v = self.seq_len_v
            self.seqlen_kv_v = self.seq_len_kv_v
            self.seqlen_kv_i32 = self.seq_len_kv
        self.delta_i32 = fx.Int32(self.seqlen_kv_i32 - fx.Int32(self.seqlen_q_v))
        self.q_gmem_elem_offset = (
            self.q_tok_base + self.q_start
        ) * self.stride_q_n_v + self.q_head_idx * traits.HEAD_DIM
        self.kv_gmem_elem_offset = (
            self.kv_tok_base * self.stride_kv_n_v + self.kv_head_idx * traits.HEAD_DIM
        )
        self.v_gmem_elem_offset = (
            self.kv_tok_base * self.stride_v_n_v + self.kv_head_idx * traits.HEAD_DIM_V
        )

    def init_descriptors(self):
        traits = self.traits
        eb = traits.ELEM_BYTES
        q_nrec_bytes = as_mlir_value(self.q_tok_end * self.stride_q_n_v * eb)
        kv_nrec_bytes = as_mlir_value(self.kv_tok_end * self.stride_kv_n_v * eb)
        v_nrec_bytes = as_mlir_value(self.kv_tok_end * self.stride_v_n_v * eb)
        o_nrec_bytes = as_mlir_value(
            self.q_tok_end * self.stride_o_n_v * traits.OUT_ELEM_BYTES
        )

        def _make_buf_div(tensor, nrec_bytes):
            # fp8 Q/K/V buffer views are i8-typed so DMA and register loads share one
            # byte view.
            bt = fx.rocdl.make_buffer_tensor(tensor, num_records_bytes=nrec_bytes)
            it = fx.get_iter(bt)
            i8_ptr_ty = fx.PointerType.get(
                elem_ty=fx.Int8.ir_type,
                address_space=fx.PointerType(it.type).address_space,
                alignment=fx.PointerType(it.type).alignment,
            )
            bt = fx.Tensor(
                fx.make_view(fx.recast_iter(i8_ptr_ty, it), fx.get_layout(bt))
            )
            return fx.logical_divide(bt, fx.make_layout(1, 1))

        self.q_div = _make_buf_div(self.Q, q_nrec_bytes)
        self.k_div = _make_buf_div(self.K, kv_nrec_bytes)
        self.v_div = _make_buf_div(self.V, v_nrec_bytes)
        self.o_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(self.O, num_records_bytes=o_nrec_bytes),
            fx.make_layout(1, 1),
        )

    def init_atoms_and_lds_ptrs(self):
        traits = self.traits
        self.load_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.load_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
        self.store_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
        self.store_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.o_store_reg = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
        self.o_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        # fp8 global->LDS DMA uses i8 destination typing; K/V LDS reads are byte-addressed.
        self.lds_ptr_ty = fx.PointerType.get(fx.Int8.ir_type, 2, traits.DMA_BYTES)

    def init_descale(self):
        def _load_scale_scalar(tensor):
            _div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(tensor), fx.make_layout(1, 1)
            )
            _atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            _v = fly.copy_atom_call_ssa(
                [Vec.make_type(1, fx.Float32)],
                _atom,
                fx.slice(_div, (None, fx.Int32(0))),
            )
            return fx.Float32(Vec(_v, (1,), fx.Float32)[0])

        head_dim_f32 = fx.Float32(self.head_dim_runtime)
        c_log2e_f = fx.Float32(_LOG2E)
        c_sm_scale_log2e = fx.rsqrt(head_dim_f32, fastmath=self.fm_fast) * c_log2e_f
        _qd = _load_scale_scalar(self.QDescale)
        _kd = _load_scale_scalar(self.KDescale)
        self.vd_fp8 = _load_scale_scalar(self.VDescale)
        # fp8 feeds raw Q/K into the MFMA, so q/k descale * softmax scale multiplies
        # the fp32 logits after QK.
        self.c_logit_scale = c_sm_scale_log2e * (_qd * _kd)

    def init_tile_bounds(self):
        traits = self.traits
        kv_tile_size = traits.BLOCK_N
        num_kv_tiles = (self.seqlen_kv_v + kv_tile_size - 1) // kv_tile_size
        if const_expr(traits.CAUSAL):
            causal_end_raw_i32 = (
                fx.Int32(self.q_start + traits.BLOCK_M) + self.delta_i32
            )
            causal_end_i32 = fx.Int32(
                (causal_end_raw_i32 > fx.Int32(0)).select(
                    causal_end_raw_i32, fx.Int32(0)
                )
            )
            causal_num_tiles = (
                fx.Index(causal_end_i32) + kv_tile_size - 1
            ) // kv_tile_size
            max_num_tiles = fx.Index(
                (causal_num_tiles < num_kv_tiles).select(causal_num_tiles, num_kv_tiles)
            )
        else:
            causal_end_raw_i32 = None
            max_num_tiles = num_kv_tiles
        # Pipeline needs an EVEN tile count >= 4; extra tiles read 0 (num_records) and are masked.
        max_num_tiles = ((max_num_tiles + 1) // 2) * 2
        max_num_tiles = fx.Index((max_num_tiles < 4).select(4, max_num_tiles))
        self.max_num_tiles = max_num_tiles
        if const_expr(traits.SPLITK):
            chunk = (
                (
                    (max_num_tiles + (traits.NUM_KV_SPLITS - 1)) // traits.NUM_KV_SPLITS
                    + 1
                )
                // 2
                * 2
            )
            chunk = fx.Index((chunk < 6).select(6, chunk))
            split_t0 = self.split_idx * chunk
            split_t_end = split_t0 + chunk
            split_t_end = fx.Index(
                (split_t_end < max_num_tiles).select(split_t_end, max_num_tiles)
            )
            split_t_end = fx.Index(
                (max_num_tiles - split_t_end < 4).select(max_num_tiles, split_t_end)
            )
            self.split_nonempty = split_t0 + 4 <= max_num_tiles
        else:
            split_t0 = 0
            split_t_end = max_num_tiles
            self.split_nonempty = None

        if const_expr(traits.VARLEN or (traits.CAUSAL and traits.CROSS_SEQLEN)):
            active = None
            if const_expr(traits.VARLEN):
                active = self.q_start < self.seqlen_q_v
            if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
                in_mask = causal_end_raw_i32 > fx.Int32(0)
                active = in_mask if active is None else (active & in_mask)
            split_t_end = fx.Index(active.select(split_t_end, split_t0))

        self.split_t0 = split_t0
        self.split_t_end = split_t_end

    def init_workspace_io(self):
        if const_expr(self.traits.SPLITK):
            self.ws_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(self.Workspace), fx.make_layout(1, 1)
            )
            self.ws_store_atom_32 = fx.make_copy_atom(
                fx.rocdl.BufferCopy32b(), fx.Int32
            )
            self.ws_store_reg_32 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            self.ws_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)

    def ws_store_f32(self, f32_val, elem_index):
        pack = Vec.from_elements([fx.Float32(f32_val)], fx.Float32).bitcast(fx.Int32)
        fx.memref_store_vec(pack, self.ws_store_reg_32)
        fx.copy(
            self.ws_store_atom_32,
            self.ws_store_reg_32,
            fx.slice(self.ws_div, (None, fx.Int32(elem_index))),
        )

    def ws_store_quad_i32(self, dwords, elem_index):
        pack = Vec.from_elements([fx.Int32(v) for v in dwords], fx.Int32)
        fx.memref_store_vec(pack, self.ws_store_reg_128)
        fx.copy(
            self.store_atom_128,
            self.ws_store_reg_128,
            fx.slice(self.ws_div, (None, fx.Int32(elem_index))),
        )

    def init_q_row(self):
        _init_dualwave_q_row(self)

    def k_buf_base(self, buf_id):
        traits = self.traits
        if const_expr(isinstance(buf_id, int)):
            return traits.DUALWAVE_SWP_K_BUF_BASE[buf_id]
        return buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER

    def buffer_load_128(self, elem_index):
        return _buffer_load_128(
            elem_index, self.load_atom_128, self.q_div, self.v4i32_type
        )

    def buffer_load_lds_128(self, src_div, lds_byte_addr, src_elem, soffset_elems):
        _buffer_load_lds_128(
            src_div,
            lds_byte_addr,
            src_elem,
            soffset_elems,
            _dma_atom=self.dma_atom,
            _lds_ptr_ty=self.lds_ptr_ty,
        )

    def buffer_store_128(self, pack_i32_vec, elem_index):
        _buffer_store_128(
            pack_i32_vec,
            elem_index,
            self.o_store_reg_128,
            self.store_atom_128,
            self.o_div,
        )

    def global_idx_q(self, token_idx, col):
        return (
            (self.q_tok_base + token_idx) * self.stride_q_n_v
            + self.q_head_idx * self.traits.HEAD_DIM
            + col
        )

    def global_idx_o(self, token_idx, col):
        """Element index into O, which is HEAD_DIM_V wide (not HEAD_DIM)."""
        return (
            (self.q_tok_base + token_idx) * self.stride_o_n_v
            + self.q_head_idx * self.traits.HEAD_DIM_V
            + col
        )

    def read_i32x8_lds(self, base_ptr, byte_row):
        halves = []
        for h in range_constexpr(2):
            p = buffer_ops.get_element_ptr(
                base_ptr, byte_offset=fx.Int32(byte_row + h * 16), elem_type=T.i8
            )
            halves.append(
                Vec(llvm.LoadOp(Vec.make_type(4, fx.Int32), p, alignment=16).result)
            )
        return halves[0].shuffle(halves[1], [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
