# SPDX-License-Identifier: MIT
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Shared helpers for the gfx950 dual-wave, software-pipelined fp8 flash attention.

Migrated from FlyDSL ``kernels/attention/flash_attn_utils.py``, restricted to the
symbols the fp8 (e4m3fn) kernel actually reaches: low-level ROCDL/MLIR
primitives, the fp8 tile/layout traits, the software-pipeline stages
(load / GEMM / softmax / store), and the split-K combine pass. The bf16/f16
dual-wave, the gfx942 generic path, paged KV, and the bias/ALiBi helpers are not
part of the fp8 path and were left behind.
"""

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


# gfx950 (MI350/MI355X): 8 XCDs, each with a private ~4 MB L2.
NUM_XCD_GFX950 = 8


LDS_BYTES_GFX950 = 160 * 1024


MIN_Q_BLOCKS_XCD_SWIZZLE = 64


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
    """gfx950 ds_read_b64_tr_b8 (8-bit transpose) with immediate byte offset.

    Returns 64 bits = 8 fp8 (the fp8 analog of ds_read_b64_tr_b16's 4 bf16),
    used for the fp8 V transpose load.
    """
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


def _concat_vectors(lhs, rhs):
    lhs_vec = Vec(lhs)
    rhs_vec = Vec(rhs)
    return lhs_vec.shuffle(
        rhs_vec,
        list(range(lhs_vec.numel)) + [lhs_vec.numel + i for i in range(rhs_vec.numel)],
    )


def _bitcast_i32(value):
    return as_mlir_value(fx.Float32(value).bitcast(fx.Int32).ir_value())


def _bitcast_f32(value):
    return as_mlir_value(fx.Int32(value).bitcast(fx.Float32).ir_value())


def _attn_mask_vec2_imm(rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32):
    """DUALWAVE_SWP pair mask asm: 2 compares followed by 2 cndmasks."""
    asm_str = (
        f"v_cmp_lt_i32_e64 $0, $6, {int(thr_x)}\n\t"
        f"v_cmp_lt_i32_e64 $1, $6, {int(thr_y)}\n\t"
        "v_cndmask_b32_e64 $2, $4, $7, $0\n\t"
        "v_cndmask_b32_e64 $3, $5, $7, $1"
    )
    ret_struct_ty = ir.Type.parse("!llvm.struct<(i64, i64, i32, i32)>")
    ret = llvm.inline_asm(
        ret_struct_ty,
        [
            as_mlir_value(x_ref_i32),
            as_mlir_value(y_ref_i32),
            as_mlir_value(rel_i32),
            as_mlir_value(neg_inf_i32),
        ],
        asm_str,
        "=s,=s,=v,=v,2,3,v,v,~{vcc}",
        has_side_effects=True,
    )
    return llvm.extractvalue(T.i32, ret, [2]), llvm.extractvalue(T.i32, ret, [3])


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


def _anchor_v_p(traits, v_p, elem_dtype):
    p_lo, p_hi = v_p
    p_lo_all = _concat_vectors(p_lo[0], p_lo[1])
    p_hi_all = _concat_vectors(p_hi[0], p_hi[1])
    p_all = _concat_vectors(p_lo_all, p_hi_all)
    p_all_ir = as_mlir_value(p_all)
    p_all_anchored = llvm.inline_asm(
        p_all_ir.type,
        [p_all_ir],
        "",
        "=v,0",
        has_side_effects=True,
    )
    p_vec = Vec(p_all_anchored, (traits.PV_K_STEPS * 2 * 8,), elem_dtype)
    anchored_lo = []
    anchored_hi = []
    for pks in range_constexpr(traits.PV_K_STEPS):
        lo_base = pks * 8
        hi_base = traits.PV_K_STEPS * 8 + pks * 8
        anchored_lo.append(
            p_vec.shuffle(p_vec, [lo_base + i for i in range(8)]).ir_value()
        )
        anchored_hi.append(
            p_vec.shuffle(p_vec, [hi_base + i for i in range(8)]).ir_value()
        )
    return anchored_lo, anchored_hi


def _v_pair_to_vec32(v):
    return _concat_vectors(v[0], v[1]).ir_value()


def _v_vec32_to_pair(v):
    v_vec = Vec(v, (32,), fx.Float32)
    v_lo = v_vec.shuffle(v_vec, [i for i in range(16)]).ir_value()
    v_hi = v_vec.shuffle(v_vec, [16 + i for i in range(16)]).ir_value()
    return v_lo, v_hi


def _v_p_to_vec32(v_p):
    p_lo, p_hi = v_p
    p_lo_all = _concat_vectors(p_lo[0], p_lo[1])
    p_hi_all = _concat_vectors(p_hi[0], p_hi[1])
    return _concat_vectors(p_lo_all, p_hi_all).ir_value()


def _v_vec32_to_p(traits, v_p_all, elem_dtype):
    p_vec = Vec(v_p_all, (traits.PV_K_STEPS * 2 * 8,), elem_dtype)
    p_lo = []
    p_hi = []
    for pks in range_constexpr(traits.PV_K_STEPS):
        lo_base = pks * 8
        hi_base = traits.PV_K_STEPS * 8 + pks * 8
        p_lo.append(p_vec.shuffle(p_vec, [lo_base + i for i in range(8)]).ir_value())
        p_hi.append(p_vec.shuffle(p_vec, [hi_base + i for i in range(8)]).ir_value())
    return p_lo, p_hi


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


def _pack_p_v8_slices(traits, v_p, pack_v8_fn):
    lo_partial_list, hi_full = v_p
    p_lo_packs = []
    p_hi_packs = []
    for pks in range_constexpr(traits.PV_K_STEPS):
        p_base = pks * 8
        lo_slice = [lo_partial_list[p_base + s] for s in range_constexpr(8)]
        hi_slice = hi_full[p_base : p_base + 8]
        p_lo_packs.append(pack_v8_fn(lo_slice))
        p_hi_packs.append(pack_v8_fn(hi_slice))
    return p_lo_packs, p_hi_packs


def _safe_l_inv(l_row, zero_f):
    l_inv = rocdl.rcp(T.f32, as_mlir_value(l_row))
    return (fx.Float32(l_row) > zero_f).select(l_inv, zero_f)


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


def _init_dualwave_thread_mapping(ctx):
    """Set block/wave/lane/head indices on a dualwave-style context.

    Shared verbatim by DualwaveKernelContext and DualwaveFp8KernelContext."""
    traits = ctx.traits
    batch_interleave_group = getattr(traits, "BATCH_INTERLEAVE_GROUP", 1)
    # Swizzled Head-first Mapping (arXiv:2511.02132): the grid is head-fast, so one
    # head's q-blocks scatter across all XCDs and each re-streams its K/V. Re-derive
    # (head, q_block) with head as the slow axis to keep them on one XCD. Bijective,
    # so output is bit-identical; split-K's third grid axis would not survive it.
    # Non-causal only: under a causal mask q-block i does work proportional to i, so
    # making q_block the fast axis clusters unequal work and costs 7% (measured).
    if const_expr(
        traits.XCD_SWIZZLE
        and not traits.SPLITK
        and not traits.CAUSAL
        and traits.NUM_HEADS_Q % NUM_XCD_GFX950 == 0
    ):
        num_q_blocks = fx.Index(gpu.grid_dim.y)
        linear_wg = fx.Index(gpu.block_idx.x) + fx.Index(gpu.block_idx.y) * fx.Index(
            traits.NUM_HEADS_Q
        )
        ctx.h_idx = linear_wg // num_q_blocks
        ctx.q_block_idx = linear_wg % num_q_blocks
    elif const_expr(batch_interleave_group > 1):
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
    DTYPE_STR: str
    WAVES_PER_EU: int
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
    DUALWAVE_SWP_V_BUF_BASE: tuple[int, int]
    VT_BF16_TOTAL: int
    DUALWAVE_SWP_RESCALE_THRESHOLD: float
    SCHED_MFMA_MASK: int
    SCHED_DS_READ_MASK: int
    NEG_INF_F32_BITS: int
    XCD_SWIZZLE: bool = False
    BATCH_INTERLEAVE_GROUP: int = 1

    @property
    def cache_tag(self):
        return (
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.HEAD_DIM,
            self.CAUSAL,
            self.DTYPE_STR,
            self.WAVES_PER_EU,
            self.DAZ,
            self.DUALWAVE_SWP_LAZY_RESCALE,
            self.DUALWAVE_SWP_RESCALE_THRESHOLD,
            self.DUALWAVE_SWP_SETPRIO,
            self.DUALWAVE_SWP_ENABLE_STAGGER,
            self.NUM_KV_SPLITS,
            self.SPLITK,
            self.VARLEN,
            self.CROSS_SEQLEN,
            self.HEAD_DIM_V,
            self.QLDS,
            self.K_BAND_CHUNK,
            "fp8_wide_qk_hiprec_pv",
            self.ELEM_BYTES,
            self.OUT_ELEM_BYTES,
            self.LANE_SPLIT_KV,
            self.VT_BF16_TOTAL,
            self.NUM_PREFETCH_K,
            self.XCD_SWIZZLE,
            self.BATCH_INTERLEAVE_GROUP,
            self.BLOCK_M,
            self.BLOCK_SIZE,
            self.NUM_WAVES,
        )


def _make_dualwave_swp_fp8_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    rescale_threshold,
    head_dim_v=None,
    block_m=256,
    causal=True,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    dualwave_swp_setprio=True,
    dualwave_swp_enable_stagger=True,
    num_kv_splits=1,
    varlen=False,
    cross_seqlen=False,
    xcd_swizzle=False,
    batch_interleave_group=1,
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
    dualwave_swp_v_buf_base = tuple(
        smem_k_tile_elems + i * dualwave_swp_kv_per_buffer
        for i in range(num_prefetch_k)
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
        DTYPE_STR="fp8",
        WAVES_PER_EU=waves_per_eu,
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
        DUALWAVE_SWP_V_BUF_BASE=dualwave_swp_v_buf_base,
        VT_BF16_TOTAL=vt_bf16_total,
        DUALWAVE_SWP_RESCALE_THRESHOLD=rescale_threshold,
        SCHED_MFMA_MASK=0x008,
        SCHED_DS_READ_MASK=0x100,
        NEG_INF_F32_BITS=0xFF800000,
        XCD_SWIZZLE=bool(xcd_swizzle),
        BATCH_INTERLEAVE_GROUP=int(batch_interleave_group),
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


class DualwaveFp8KernelContext:
    """Shared per-kernel state for the gfx950 dualwave fp8 attention helpers.

    Mirrors ``DualwaveKernelContext`` but for the fp8 single path: raw fp8 Q/K/V
    (i8 buffer views), per-tensor Q/K/V descale scalars applied to the fp32 logits,
    and a bf16 ``vt`` LDS scratch for HIPREC PV."""

    def __init__(
        self,
        traits_or_ctx,
        Q=None,
        K=None,
        V=None,
        O=None,
        DebugCounts=None,
        CuSeqQ=None,
        CuSeqKv=None,
        QDescale=None,
        KDescale=None,
        VDescale=None,
        seq_len=None,
        seq_len_kv=None,
        stride_q_n=None,
        stride_kv_n=None,
        head_dim_runtime=None,
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
        self.DebugCounts = DebugCounts
        self.CuSeqQ = CuSeqQ
        self.CuSeqKv = CuSeqKv
        self.QDescale = QDescale
        self.KDescale = KDescale
        self.VDescale = VDescale
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.stride_q_n = stride_q_n
        self.stride_kv_n = stride_kv_n
        self.head_dim_runtime = head_dim_runtime

    def init_types_and_constants(self):
        traits = self.traits
        self.elem_dtype = fx.Float8E4M3FN
        self.fm_fast = fx.arith.FastMathFlags.fast
        self.v4i32_type = Vec.make_type(4, fx.Int32)
        self.v4f16_type = Vec.make_type(4, self.elem_dtype)
        self.v16f32_type = Vec.make_type(16, fx.Float32)
        self.v2i32_type = Vec.make_type(2, fx.Int32)
        self.p_elem = fx.BFloat16
        self.v4bf16_type = Vec.make_type(4, fx.BFloat16)
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
        self.stride_q_n_v = fx.Index(self.stride_q_n)
        self.stride_kv_n_v = fx.Index(self.stride_kv_n)
        if traits.HEAD_DIM_V == traits.HEAD_DIM:
            self.stride_v_n_v = self.stride_kv_n_v
            self.stride_o_n_v = self.stride_q_n_v
        else:
            self.stride_v_n_v = fx.Index(traits.DEFAULT_STRIDE_V_N)
            self.stride_o_n_v = fx.Index(traits.DEFAULT_STRIDE_O_N)

    def init_causal_lpt_order(self):
        """Issue causal q-blocks longest-first by reversing the q-block grid axis.

        Causal work per q-block grows with the block index and workgroups dispatch in
        flattened-id order, so the natural order issues the heaviest block last and the
        makespan carries its tail. Must run after init_thread_mapping and before
        init_sequence_lengths / init_tile_bounds / init_q_row read q_start.
        """
        traits = self.traits
        num_q_blocks = (self.seq_len_v + traits.BLOCK_M - 1) // traits.BLOCK_M
        self.q_block_idx = num_q_blocks - fx.Index(1) - self.q_block_idx
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
            self.q_tok_end = _cu_load(
                _cuq_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32
            )
            self.kv_tok_base = _cu_load(_cuk_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.kv_tok_end = _cu_load(
                _cuk_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32
            )
            self.seqlen_q_v = self.q_tok_end - self.q_tok_base
            self.seqlen_kv_v = self.kv_tok_end - self.kv_tok_base
            self.seqlen_kv_i32 = fx.Int32(self.seqlen_kv_v)
        else:
            self.q_tok_base = self.batch_idx * self.seq_len_v
            self.kv_tok_base = self.batch_idx * self.seq_len_kv_v
            self.q_tok_end = (self.batch_idx + fx.Index(1)) * self.seq_len_v
            self.kv_tok_end = (self.batch_idx + fx.Index(1)) * self.seq_len_kv_v
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

        head_dim_f32 = fx.Float32(fx.Int32(self.head_dim_runtime))
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
        max_num_tiles = ((max_num_tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
        max_num_tiles = fx.Index(
            (max_num_tiles < fx.Index(4)).select(fx.Index(4), max_num_tiles)
        )
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
            chunk = fx.Index((chunk < fx.Index(6)).select(fx.Index(6), chunk))
            split_t0 = self.split_idx * chunk
            split_t_end = split_t0 + chunk
            split_t_end = fx.Index(
                (split_t_end < max_num_tiles).select(split_t_end, max_num_tiles)
            )
            split_t_end = fx.Index(
                (max_num_tiles - split_t_end < fx.Index(4)).select(
                    max_num_tiles, split_t_end
                )
            )
            self.split_nonempty = split_t0 + fx.Index(4) <= max_num_tiles
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
                fx.rocdl.make_buffer_tensor(self.DebugCounts), fx.make_layout(1, 1)
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

    def v_buf_base(self, buf_id):
        traits = self.traits
        if const_expr(isinstance(buf_id, int)):
            return traits.DUALWAVE_SWP_V_BUF_BASE[buf_id]
        return traits.SMEM_K_TILE_ELEMS + buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER

    def v_pair_to_vec32(self, v):
        return _v_pair_to_vec32(v)

    def v_vec32_to_pair(self, v):
        return _v_vec32_to_pair(v)

    def bf16_trunc_pack_v8(self, f32_vals):
        # HIPREC carries P/V as v8 bf16 regardless of the fp8 element dtype:
        # pack 8 f32 -> 4 cvt_pk_bf16 dwords.
        pairs = []
        for j in range_constexpr(4):
            pairs.append(rocdl.cvt_pk_bf16_f32(f32_vals[j * 2], f32_vals[j * 2 + 1]))
        return Vec.from_elements(pairs, fx.Int32).bitcast(fx.BFloat16).ir_value()

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


class DualwaveFp8QLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def stage_q_to_lds(self):
        traits = self.traits
        chunks_per_row = traits.HEAD_DIM // 16  # 16-byte DMA chunks per Q row
        total_chunks = (traits.BLOCK_M * traits.HEAD_DIM) // 16
        for p in range_constexpr(total_chunks // traits.BLOCK_SIZE):
            c = self.tid + fx.Index(p * traits.BLOCK_SIZE)
            row = c // fx.Index(chunks_per_row)
            dchunk = c % fx.Index(chunks_per_row)
            src_elem = (
                self.q_gmem_elem_offset
                + row * self.stride_q_n_v
                + dchunk * fx.Index(16)
            )
            lds_addr = self.lds_q_base_idx + c * fx.Index(16)
            self.buffer_load_lds_128(self.q_div, lds_addr, src_elem, 0)

    def load_all_wide(self, q_row_in_block):
        traits = self.traits
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            byte_row = (
                q_row_in_block * fx.Index(traits.HEAD_DIM) + fx.Index(ws * 64) + d_base
            )
            packs.append(self.read_i32x8_lds(self.lds_q_base_ptr, fx.Int32(byte_row)))
        return packs


class DualwaveFp8GemmHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _mfma_acc_fp8_wide(self, a_i32x8, b_i32x8, c_v16):
        # Wide fp8 QK: mfma_scale (32x32x64) with unit E8M0 scales, i32x8 operands.
        return rocdl.mfma_scale_f32_32x32x64_f8f6f4(
            self.v16f32_type,
            [
                as_mlir_value(a_i32x8),
                as_mlir_value(b_i32x8),
                as_mlir_value(c_v16),
                0,
                0,
                0,
                as_mlir_value(fx.Int32(0x7F7F7F7F)),
                0,
                as_mlir_value(fx.Int32(0x7F7F7F7F)),
            ],
        )

    def _pack_fp8_i32x8(self, f32_vals):
        c0 = llvm.mlir_poison(T.i32)
        words = []
        for g in range_constexpr(8):
            base = g * 4
            w = rocdl.cvt_pk_fp8_f32(
                T.i32,
                as_mlir_value(f32_vals[base]),
                as_mlir_value(f32_vals[base + 1]),
                c0,
                0,
            )
            w = rocdl.cvt_pk_fp8_f32(
                T.i32,
                as_mlir_value(f32_vals[base + 2]),
                as_mlir_value(f32_vals[base + 3]),
                w,
                1,
            )
            words.append(fx.Int32(w))
        return Vec.from_elements(words, fx.Int32).ir_value()

    def _v_concat_i32x8(self, v_v, dc):
        words = []
        for ks in range_constexpr(4):
            v2 = Vec(
                llvm.bitcast(self.v2i32_type, as_mlir_value(v_v[ks][dc])),
                (2,),
                fx.Int32,
            )
            words.append(fx.Int32(v2[0]))
            words.append(fx.Int32(v2[1]))
        return Vec.from_elements(words, fx.Int32).ir_value()

    def _load_q_wide_lds(self):
        traits = self.traits
        q_row_in_block = self.ctx_ref.q_row_in_block
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            byte_row = (
                q_row_in_block * fx.Index(traits.HEAD_DIM) + fx.Index(ws * 64) + d_base
            )
            packs.append(self.read_i32x8_lds(self.lds_q_base_ptr, fx.Int32(byte_row)))
        return packs

    def _load_q_wide_global(self):
        """Pull this lane's Q operands straight from global into VGPRs (head_dim > 128)."""
        traits = self.traits
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            elem = self.global_idx_q(self.ctx_ref.q_row, fx.Index(ws * 64) + d_base)
            lo = self.buffer_load_128(elem)
            hi = self.buffer_load_128(elem + fx.Index(16))
            packs.append(Vec(lo).shuffle(Vec(hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
        return packs

    def load_q_wide(self):
        if const_expr(self.traits.QLDS):
            return self._load_q_wide_lds()
        return self._load_q_wide_global()

    def qk(self, v_k, q_wide=None):
        traits = self.traits
        k_lo, k_hi = v_k
        q_all_wide = self._load_q_wide_lds() if q_wide is None else q_wide
        v_s_lo = self.c_zero_v16f32
        v_s_hi = self.c_zero_v16f32
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            q_w = q_all_wide[ws]
            v_s_lo = self._mfma_acc_fp8_wide(k_lo[ws], q_w, v_s_lo)
            v_s_hi = self._mfma_acc_fp8_wide(k_hi[ws], q_w, v_s_hi)
        n_ds = const_expr(traits.HEAD_DIM // 64 * 4)
        n_mfma = const_expr(traits.HEAD_DIM // 64 * 2)
        rocdl.sched_group_barrier(traits.SCHED_DS_READ_MASK, n_ds // 2, 12)
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, 12)
        rocdl.sched_group_barrier(traits.SCHED_DS_READ_MASK, n_ds // 2, 12)
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, n_mfma - 1, 12)
        return (v_s_lo, v_s_hi)

    def cast_p_fp8_direct(self, v_p):
        lo_partial_list, hi_full = v_p
        f32 = []
        for pks in range_constexpr(self.traits.PV_K_STEPS):
            p_base = pks * 8
            f32 += [lo_partial_list[p_base + s] for s in range_constexpr(8)]
        for pks in range_constexpr(self.traits.PV_K_STEPS):
            p_base = pks * 8
            f32 += [hi_full[p_base + s] for s in range_constexpr(8)]
        return self._pack_fp8_i32x8(f32)

    def _pv_fp8_direct(self, p_fp8, v_v, v_o):
        v_o = _anchor_v_o(self.traits, v_o)
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_op = self._v_concat_i32x8(v_v, dc)
            v_o[dc] = self._mfma_acc_fp8_wide(v_op, p_fp8, v_o[dc])
        return v_o

    def pv(self, v_p, v_v, v_o):
        return self._pv_fp8_direct(v_p, v_v, v_o)


class DualwaveFp8KvGmemToLdsLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_k(self, tile_start, buf_id):
        """DMA one K tile into LDS, one pass per head-dim band.

        A band's LDS line is this wave's n-rows of `chunk` bytes, row-contiguous,
        so the QK read indexes it as (band, row, 64-byte slice). The 64-byte tail
        band at head_dim 192 runs on the low 32 lanes and moves no padding.
        """
        traits = self.traits
        eb = traits.ELEM_BYTES
        k_lds_byte_base = self.lds_kv_base_idx + self.k_buf_base(buf_id) * eb
        rows_per_wave = -(-traits.BLOCK_N // traits.NUM_WAVES)
        for d in range_constexpr(self.NUM_DMA_K):
            lanes_per_row = traits.K_BAND_CHUNK[d] // traits.VEC_KV
            slots = rows_per_wave * lanes_per_row
            band_base = (
                k_lds_byte_base
                + fx.Index(traits.K_BAND_BASE[d] * eb)
                + self.wave_id_uni * (traits.K_BAND_LINE_STRIDE[d] * eb)
            )
            for pas in range_constexpr(-(-slots // traits.WARP_SIZE)):
                slot = self.lane_in_warp + fx.Index(pas * traits.WARP_SIZE)
                n_in_tile = (slot // lanes_per_row) * traits.NUM_WAVES + self.wave_id
                global_d = (
                    slot % lanes_per_row
                ) * traits.VEC_KV + traits.K_BAND_GLOBAL_D[d]
                src_elem = (
                    self.kv_gmem_elem_offset + n_in_tile * self.stride_kv_n_v + global_d
                )
                lds_addr = band_base + fx.Index(
                    pas * traits.WARP_SIZE * traits.VEC_KV * eb
                )
                active = min(slots - pas * traits.WARP_SIZE, traits.WARP_SIZE)
                if const_expr(active == traits.WARP_SIZE):
                    self.buffer_load_lds_128(
                        self.k_div, lds_addr, src_elem, tile_start * self.stride_kv_n_v
                    )
                else:
                    self._load_k_band_partial_wave(
                        lds_addr, src_elem, tile_start, active
                    )

    def _load_k_band_partial_wave(self, lds_addr, src_elem, tile_start, active_lanes):
        soffset = tile_start * self.stride_kv_n_v
        k_div = self.k_div

        @flyc.jit
        def _run():
            if self.lane_in_warp < fx.Index(active_lanes):
                self.buffer_load_lds_128(k_div, lds_addr, src_elem, soffset)

        _run()

    def load_v(self, tile_start, buf_id):
        self._stage_v_fp8_block_dma(tile_start, buf_id)

    def _stage_v_fp8_block_dma(self, tile_start, buf_id):
        traits = self.traits
        nbands = traits.HEAD_DIM_V // 16
        v_tile_bytes = (traits.BLOCK_N // 8) * nbands * 128
        buf_off = buf_id * v_tile_bytes
        aligned_base = (
            (self.lds_vt_base_idx + fx.Index(127)) // fx.Index(128)
        ) * fx.Index(128)
        # The tile is BLOCK_N * nbands 16-byte slots, and one DMA instruction moves a
        # whole wave of them. Hand out instructions, not row-groups: a wave's LDS
        # destination is then always a full WARP_SIZE*16 span, so nothing has to be
        # masked off inside a wave. buffer_load...lds strides the LDS write by lane
        # regardless of exec, so an intra-wave mask would still write past the span.
        per_dma = traits.WARP_SIZE * traits.VEC_KV * traits.ELEM_BYTES
        slots_per_group = 8 * nbands
        num_dma = (traits.BLOCK_N * nbands * 16) // per_dma
        passes = -(-num_dma // traits.NUM_WAVES)
        for pas in range_constexpr(passes):
            dma_id = self.wave_id_uni + fx.Index(pas * traits.NUM_WAVES)
            slot = dma_id * fx.Index(traits.WARP_SIZE) + self.lane
            lds_addr = aligned_base + fx.Index(buf_off) + dma_id * fx.Index(per_dma)
            grp = slot // fx.Index(slots_per_group)
            rem = slot % fx.Index(slots_per_group)
            dest_n = fx.Int32(grp * fx.Index(8) + rem % fx.Index(8))
            w16 = dest_n % fx.Int32(16)
            c_add = (w16 >= fx.Int32(4)) & (w16 < fx.Int32(8))
            c_sub = (w16 >= fx.Int32(8)) & (w16 < fx.Int32(12))
            n = (
                dest_n
                + c_add.select(fx.Int32(4), fx.Int32(0))
                - c_sub.select(fx.Int32(4), fx.Int32(0))
            )
            d_block = rem // fx.Index(8)
            src_elem = (
                self.v_gmem_elem_offset
                + fx.Index(n) * self.stride_v_n_v
                + d_block * fx.Index(16)
            )
            if const_expr(num_dma % traits.NUM_WAVES == 0 or pas < passes - 1):
                self.buffer_load_lds_128(
                    self.v_div, lds_addr, src_elem, tile_start * self.stride_v_n_v
                )
            else:
                self._load_v_group_if_in_tile(
                    lds_addr, src_elem, tile_start, dma_id, num_dma
                )

    def _load_v_group_if_in_tile(self, lds_addr, src_elem, tile_start, grp, groups):
        soffset = tile_start * self.stride_v_n_v
        v_div = self.v_div

        @flyc.jit
        def _run():
            if grp < fx.Index(groups):
                self.buffer_load_lds_128(v_div, lds_addr, src_elem, soffset)

        _run()


class DualwaveFp8KvLdsToVgprLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_k(self, buf_id):
        # Read K in the wide 32x32x64 QK operand layout (32 contiguous head-dim/lane,
        # two N-strips, two head-dim halves).
        traits = self.traits
        k_base = self.k_buf_base(buf_id)
        d_base = self.lane_div_32 * 32
        n_lo = self.lane_mod_32
        n_hi = self.lane_mod_32 + 32

        rows_per_line = traits.NUM_WAVES

        def _read_strip(key):
            out = []
            for ws in range_constexpr(traits.HEAD_DIM // 64):
                b = traits.K_WS_BAND[ws]
                line = (key % rows_per_line) * traits.K_BAND_LINE_STRIDE[b]
                row = line + (key // rows_per_line) * traits.K_BAND_CHUNK[b]
                addr = (
                    k_base + traits.K_BAND_BASE[b] + row + traits.K_WS_OFF[ws] + d_base
                )
                out.append(self.read_i32x8_lds(self.lds_kv_base_ptr, addr))
            return out

        return (_read_strip(n_lo), _read_strip(n_hi))

    def load_v(self, buf_id):
        return self._load_v_fp8_block(buf_id)

    def _load_v_fp8_block(self, buf_id):
        traits = self.traits
        v_tile_bytes = (traits.BLOCK_N // 8) * (traits.HEAD_DIM_V // 16) * 128
        buf_off = buf_id * v_tile_bytes
        nbands = traits.HEAD_DIM_V // 16
        rh = (self.lane % fx.Index(32)) // fx.Index(16)
        l16 = self.lane % fx.Index(16)
        lane_hi = self.lane // fx.Index(32)
        aligned_base = (
            (self.lds_vt_base_idx + fx.Index(127)) // fx.Index(128)
        ) * fx.Index(128)
        base = fx.Int32(
            aligned_base
            + buf_off
            + rh * fx.Index(128)
            + l16 * fx.Index(8)
            + lane_hi * fx.Index(nbands * 128)
        )

        def _tr8(imm):
            r = _ds_read_tr8_b64_imm(self.v2i32_type, base, imm)
            return llvm.bitcast(T.i64, as_mlir_value(Vec(r)))

        packs = [[None] * traits.D_CHUNKS for _ in range(4)]
        for dc in range_constexpr(traits.D_CHUNKS):
            for ks in range_constexpr(4):
                imm0 = (2 * ks * nbands + dc * 2) * 128
                packs[ks][dc] = _tr8(imm0)
        return packs


class DualwaveFp8SoftmaxHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _attn_mask_vec2_imm(
        self, rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32
    ):
        return _attn_mask_vec2_imm(
            rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32
        )

    def v_s_vec_to_lists(self, v_s):
        return _score_pair_to_lists(v_s)

    def _causal_mask_inplace(self, v_s, tile_idx):
        traits = self.traits
        s_lo, s_hi = v_s
        kv_tile_start = tile_idx * traits.BLOCK_N
        kv_start_i32 = fx.Int32(kv_tile_start)
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(4)
        # q_row_i32 is set by init_q_row (called after helper construction), so read
        # it from the live ctx.
        rel_lo_i32 = fx.Int32(
            self.ctx_ref.q_row_i32 + self.delta_i32 - kv_start_i32 - lane_off_i32
        )
        rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
        neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)
        pair_thresholds = _causal_pair_thresholds(False)
        _apply_dualwave_causal_mask_pair(s_lo, rel_lo_i32, neg_inf_i32, pair_thresholds)
        _apply_dualwave_causal_mask_pair(s_hi, rel_hi_i32, neg_inf_i32, pair_thresholds)

    def causal_mask_prologue_if_needed(self, v_s, tile_idx=None, kv_end_pos=None):
        if tile_idx is None:
            tile_idx = fx.Index(0)
        if kv_end_pos is None:
            kv_end_pos = self.traits.BLOCK_N

        @flyc.jit
        def _run(v_s, tile_idx=tile_idx, kv_end_pos=kv_end_pos):
            s_lo, s_hi = v_s
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self._causal_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _run(v_s)

    def causal_mask_pair_if_needed(self, v_s_a, v_s_b, tile_a):
        """Causal-mask a BN128 tile pair under one scalar-uniform branch.

        Branching per sub-tile would put two scf.if regions in the body on every
        iteration, splitting it into five basic blocks and blocking the QK/softmax/PV
        interleave the sched_group_barriers ask for. Branching once on the pair's end
        position keeps every strictly-below-diagonal pair a single straight-line block.
        Masking sub-tile `a` inside the taken branch even when only `b` needs it is a
        no-op beyond the VALU compares, and happens in at most one pair per q-block.

        This replaces seq_pad_mask_if_needed: with delta = seqlen_kv - seqlen_q the
        largest key any row may attend to is seqlen_kv - 1, so every padding column is
        already strictly above the diagonal.
        """
        traits = self.traits
        kv_end_pos = (tile_a + fx.Index(2)) * traits.BLOCK_N

        @flyc.jit
        def _run(v_s_a, v_s_b, tile_a=tile_a, kv_end_pos=kv_end_pos):
            a_lo, a_hi = v_s_a
            b_lo, b_hi = v_s_b
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
                a_l, a_h = self.v_s_vec_to_lists(v_s_a)
                self._causal_mask_inplace((a_l, a_h), tile_a)
                a_lo, a_hi = _score_lists_to_vecs((a_l, a_h))
                b_l, b_h = self.v_s_vec_to_lists(v_s_b)
                self._causal_mask_inplace((b_l, b_h), tile_a + fx.Index(1))
                b_lo, b_hi = _score_lists_to_vecs((b_l, b_h))
            return a_lo, a_hi, b_lo, b_hi

        a_lo, a_hi, b_lo, b_hi = _run(v_s_a, v_s_b)
        return (a_lo, a_hi), (b_lo, b_hi)

    def _seq_pad_mask_inplace(self, v_s_lists, tile_idx):
        traits = self.traits
        s_lo, s_hi = v_s_lists
        kv_tile_start = tile_idx * traits.BLOCK_N
        col_base = fx.Int32(kv_tile_start) + fx.Int32(self.lane_div_32) * fx.Int32(4)
        for r in range_constexpr(16):
            thr = (r // 4) * 8 + (r % 4)
            col_lo = col_base + fx.Int32(thr)
            col_hi = col_lo + fx.Int32(32)
            s_lo[r] = (col_lo < self.seqlen_kv_i32).select(s_lo[r], self.c_neg_inf)
            s_hi[r] = (col_hi < self.seqlen_kv_i32).select(s_hi[r], self.c_neg_inf)

    def seq_pad_mask_if_needed(self, v_s, tile_idx=None):
        if tile_idx is None:
            tile_idx = fx.Index(0)

        @flyc.jit
        def _run(v_s, tile_idx=tile_idx):
            s_lo, s_hi = v_s
            kv_tile_end = (tile_idx + fx.Index(1)) * self.traits.BLOCK_N
            if fx.Int32(kv_tile_end) > self.seqlen_kv_i32:
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self._seq_pad_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _run(v_s)

    def reduce_max(self, v_s):
        return _score_pair_max(v_s, self.c_neg_inf, self.fm_fast)

    def max2(self, a, b):
        return fx.maxnumf(a, b)

    def floor_masked_max(self, row_max):
        return fx.maxnumf(row_max, self.c_neg_floor)

    # log2 of e4m3's largest finite value, 448.
    _P_HEADROOM_LOG2 = 8.807354922057604

    def sub_m(self, v_s, row_max):
        # P is cast to e4m3, whose smallest subnormal is 2**-9, so a softmax
        # over thousands of keys loses its tail to flush-to-zero -- while l_row,
        # summed before the cast, still counts it. Scaling P up first uses the
        # format's whole range; l_row scales with it, so the output is unchanged
        # apart from the tail that survives. Free: it rides the FMA's addend.
        #
        # Available headroom is bounded by how large exp2 gets: the lazy path
        # holds the running max until a tile exceeds it by RESCALE_THRESHOLD, so
        # exp2 <= 2**THRESHOLD there; the eager path rebases every tile.
        headroom = self._P_HEADROOM_LOG2
        if const_expr(self.traits.DUALWAVE_SWP_LAZY_RESCALE):
            headroom -= self.traits.DUALWAVE_SWP_RESCALE_THRESHOLD
        bias = fx.Float32(headroom) if headroom > 0.0 else None
        return _scale_sub_score_pair(
            v_s, row_max, self.c_logit_scale, self.c_zero_f, self.fm_fast, bias
        )

    def exp2(self, v_s, start):
        return _exp2_score_slice(v_s, start)

    def tile_sum(self, v_p):
        return _score_pair_sum(v_p, self.c_zero_f, self.fm_fast)

    def reduce_sum(self, l_row, v_p):
        return l_row + self.tile_sum(v_p)

    def cast_p(self, v_p):
        # Pack the finished softmax probabilities into v8 bf16 P packs for PV.
        return _pack_p_v8_slices(self.traits, v_p, self.bf16_trunc_pack_v8)

    def scale_o(self, v_o, scale_scalar):
        _scale_o_accs(v_o, scale_scalar, self.traits)

    def scale_v_p(self, v_p, scale_scalar):
        # P is v8 bf16 (HIPREC): ext to f32, scale, repack bf16.
        p_lo, p_hi = v_p
        out_lo, out_hi = [], []
        for src, dst in ((p_lo, out_lo), (p_hi, out_hi)):
            for pk in src:
                f32 = Vec(
                    llvm.FPExtOp(
                        Vec.make_type(8, fx.Float32), as_mlir_value(pk)
                    ).result,
                    (8,),
                    fx.Float32,
                )
                scaled = [fx.Float32(f32[i]) * scale_scalar for i in range(8)]
                dst.append(self.bf16_trunc_pack_v8(scaled))
        return out_lo, out_hi

    def anchor_v_p(self, v_p):
        return _anchor_v_p(self.traits, v_p, elem_dtype=self.p_elem)

    def anchor_v_o(self, v_o):
        return _anchor_v_o(self.traits, v_o)

    def anchor_scalar_f32(self, x):
        return _anchor_scalar_f32(x)

    def safe_l_inv(self, l_row):
        return _safe_l_inv(l_row, self.c_zero_f)

    def rescale_from_tile_max(self, m_row, m_tile_max):
        row_max = fx.maxnumf(m_row, m_tile_max)
        diff_scaled = (m_row - row_max) * self.c_logit_scale
        rescale = rocdl.exp2(T.f32, as_mlir_value(diff_scaled))
        return row_max, rescale

    def apply_l_rescale(self, l_row, rescale):
        return l_row * rescale

    def rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        m_new, corr = self.rescale_from_tile_max(m_row, m_tile_max)
        self.scale_o(v_o, corr)
        v_o = self.anchor_v_o(v_o)
        v_p = self.scale_v_p(v_p, corr)
        l_row = self.apply_l_rescale(l_row, corr)
        return v_o, m_new, l_row, v_p

    def v_p_to_vec32(self, v_p):
        # P packs are (p_lo[0..1], p_hi[0..1]) v8 bf16; concat into one v32 SSA value
        # for the scf.if loop-carry.
        return _v_p_to_vec32(v_p)

    def v_vec32_to_p(self, v_p_all):
        return _v_vec32_to_p(self.traits, v_p_all, elem_dtype=self.p_elem)

    def _lazy_correction(self, v_o, m_row, m_tile_max):
        """Monotonic: a downward rebase gives corr > 1 and repeated ones overflow."""
        m_new, corr = self.rescale_from_tile_max(m_row, m_tile_max)
        scaled_accs = list(v_o)
        self.scale_o(scaled_accs, corr)
        return (
            m_new,
            corr,
            [as_mlir_value(scaled_accs[i]) for i in range(self.traits.D_CHUNKS)],
        )

    def lazy_rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        @flyc.jit
        def _run(v_o, m_row, l_row, m_tile_max, v_p):
            m_diff = m_tile_max - m_row
            m_diff_scaled = m_diff * self.c_logit_scale
            below = fx.Float32(m_diff_scaled) <= self.c_rescale_thr_f
            ballot = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = arith.cmpi(
                arith.CmpIPredicate.eq, as_mlir_value(ballot), _read_exec_i64()
            )
            all_below = llvm.intr_expect(
                all_below, arith.constant(1, type=ir.IntegerType.get_signless(1))
            )

            o_out = [
                as_mlir_value(v_o[dc]) for dc in range_constexpr(self.traits.D_CHUNKS)
            ]
            m_out = as_mlir_value(m_row)
            l_out = as_mlir_value(l_row)
            vp_out = self.v_p_to_vec32(v_p)
            if fx.Boolean(all_below):
                pass
            else:
                m_new, corr, scaled_accs = self._lazy_correction(v_o, m_row, m_tile_max)
                o_out = [
                    as_mlir_value(scaled_accs[dc])
                    for dc in range_constexpr(self.traits.D_CHUNKS)
                ]
                vp_out = self.v_p_to_vec32(self.scale_v_p(v_p, corr))
                l_out = as_mlir_value(l_row * corr)
                m_out = self.anchor_scalar_f32(m_new)
            return (o_out, m_out, l_out, self.v_vec32_to_p(vp_out))

        return _run(v_o, m_row, l_row, m_tile_max, v_p)

    def lazy_correct_o(self, v_o, m_row, l_row, m_tile_max):
        @flyc.jit
        def _run(v_o, m_row, l_row, m_tile_max):
            m_diff = m_tile_max - m_row
            m_diff_scaled = m_diff * self.c_logit_scale
            below = fx.Float32(m_diff_scaled) <= self.c_rescale_thr_f
            ballot = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = arith.cmpi(
                arith.CmpIPredicate.eq, as_mlir_value(ballot), _read_exec_i64()
            )
            all_below = llvm.intr_expect(
                all_below, arith.constant(1, type=ir.IntegerType.get_signless(1))
            )

            o_out = [
                as_mlir_value(v_o[dc]) for dc in range_constexpr(self.traits.D_CHUNKS)
            ]
            m_out = as_mlir_value(m_row)
            l_out = as_mlir_value(l_row)
            if fx.Boolean(all_below):
                pass
            else:
                m_new, corr, scaled_accs = self._lazy_correction(v_o, m_row, m_tile_max)
                o_out = [
                    as_mlir_value(scaled_accs[dc])
                    for dc in range_constexpr(self.traits.D_CHUNKS)
                ]
                l_out = as_mlir_value(l_row * corr)
                m_out = self.anchor_scalar_f32(m_new)
            return (o_out, m_out, l_out)

        return _run(v_o, m_row, l_row, m_tile_max)


class DualwaveFp8StoreHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _o_pack_2dw(self, v_o, dc, store_group):
        r_base = store_group * 4
        lo = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base], Vec(v_o[dc])[r_base + 1])
        hi = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base + 2], Vec(v_o[dc])[r_base + 3])
        return lo, hi

    def _swap_half_partner(self, dw):
        pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
        swapped = rocdl.permlane32_swap(
            pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False
        )
        lo_res = llvm.extractvalue(T.i32, swapped, [0])
        hi_res = llvm.extractvalue(T.i32, swapped, [1])
        return (self.lane_div_32 != fx.Index(0)).select(lo_res, hi_res)

    def _packed_o_128_dwords(self, v_o, dc, g):
        is_hi_half = self.lane_div_32 != fx.Index(0)
        d0_a, d1_a = self._o_pack_2dw(v_o, dc, 2 * g)
        d0_b, d1_b = self._o_pack_2dw(v_o, dc, 2 * g + 1)
        y0_a, y1_a = self._swap_half_partner(d0_a), self._swap_half_partner(d1_a)
        y0_b, y1_b = self._swap_half_partner(d0_b), self._swap_half_partner(d1_b)
        w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
        w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
        w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
        w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
        return w0, w1, w2, w3

    def _packed_o_128_vec(self, v_o, dc, g):
        return Vec.from_elements(
            [fx.Int32(w) for w in self._packed_o_128_dwords(v_o, dc, g)], fx.Int32
        )

    def store_final_o(self, v_o, q_row):
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                o_pack = self._packed_o_128_vec(v_o, dc, g)
                d_col = (dc * self.traits.D_CHUNK) + (2 * g + self.lane_div_32) * 8
                o_global = self.global_idx_o(q_row, d_col)
                self.buffer_store_128(o_pack, o_global)

    def store_splitk_partial_o(self, v_o, m_row, l_row, q_row):
        m_row = fx.Float32(m_row) * self.c_logit_scale
        split_z = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
        o_part_row_base = (
            (split_z * self.traits.NUM_HEADS_Q + self.q_head_idx) * self.seq_len_v
            + q_row
        ) * (self.traits.HEAD_DIM_V // 2)
        grid_z = fx.Index(gpu.grid_dim.z)
        mrow_base = (
            grid_z
            * self.traits.NUM_HEADS_Q
            * self.seq_len_v
            * (self.traits.HEAD_DIM_V // 2)
        )
        lrow_base = mrow_base + grid_z * self.traits.NUM_HEADS_Q * self.seq_len_v
        ml_row_idx = (
            split_z * self.traits.NUM_HEADS_Q + self.q_head_idx
        ) * self.seq_len_v + q_row

        @flyc.jit
        def _store_splitk_partial_if_qrow():
            if q_row < self.seq_len_v:
                for dc in range_constexpr(self.traits.D_CHUNKS):
                    for g in range_constexpr(2):
                        dw_col = (
                            dc * (self.traits.D_CHUNK // 2)
                            + (2 * g + self.lane_div_32) * 4
                        )
                        self.ws_store_quad_i32(
                            self._packed_o_128_dwords(v_o, dc, g),
                            o_part_row_base + dw_col,
                        )
                if self.lane < fx.Index(32):
                    self.ws_store_f32(m_row, mrow_base + ml_row_idx)
                    self.ws_store_f32(l_row, lrow_base + ml_row_idx)

        _store_splitk_partial_if_qrow()

    def store_empty_split(self):
        @flyc.jit
        def _store_empty_split():
            if self.max_num_tiles < self.split_t0 + fx.Index(4):
                q_row_e = self.q_start + self.wave_q_offset + self.lane_mod_32
                split_z_e = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
                o_row_base_e = (
                    (split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx)
                    * self.seq_len_v
                    + q_row_e
                ) * (self.traits.HEAD_DIM_V // 2)
                grid_z_e = fx.Index(gpu.grid_dim.z)
                mrow_base_e = (
                    grid_z_e
                    * self.traits.NUM_HEADS_Q
                    * self.seq_len_v
                    * (self.traits.HEAD_DIM_V // 2)
                )
                lrow_base_e = (
                    mrow_base_e + grid_z_e * self.traits.NUM_HEADS_Q * self.seq_len_v
                )
                ml_row_e = (
                    split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx
                ) * self.seq_len_v + q_row_e
                if q_row_e < self.seq_len_v:
                    c_zero_i = fx.Int32(0)
                    for dc in range_constexpr(self.traits.D_CHUNKS):
                        for g in range_constexpr(2):
                            dw_col = (
                                dc * (self.traits.D_CHUNK // 2)
                                + (2 * g + self.lane_div_32) * 4
                            )
                            self.ws_store_quad_i32(
                                [c_zero_i, c_zero_i, c_zero_i, c_zero_i],
                                o_row_base_e + dw_col,
                            )
                    if self.lane < fx.Index(32):
                        self.ws_store_f32(fx.Float32(-1e30), mrow_base_e + ml_row_e)
                        self.ws_store_f32(self.c_zero_f, lrow_base_e + ml_row_e)

        _store_empty_split()


class DualwaveSplitKCombineContext:
    """Shared per-kernel state for the split-K combine pass."""

    def __init__(
        self,
        traits_or_ctx,
        O=None,
        WS=None,
        batch_size=None,
        seq_len=None,
        stride_o_n=None,
        LSE=None,
        Sink=None,
        CuSeqQ=None,
    ):
        if isinstance(traits_or_ctx, DualwaveSplitKCombineContext):
            self.__dict__.update(traits_or_ctx.__dict__)
            self.ctx_ref = getattr(traits_or_ctx, "ctx_ref", traits_or_ctx)
            return

        self.ctx_ref = self
        self.traits = traits_or_ctx
        self.O = O
        self.WS = WS
        self.LSE = LSE
        self.Sink = Sink
        self.CuSeqQ = CuSeqQ
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stride_o_n = stride_o_n

    def init_types_and_constants(self):
        self.elem_dtype = fx.BFloat16  # fp8 in, bf16 out
        self.fm_fast = fx.arith.FastMathFlags.fast
        self.c_zero_f = fx.Float32(0.0)
        self.c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)
        # LSE store folds the log2->ln conversion (m_max is sm_scale*log2e-scaled).
        self.c_ln2_f = fx.Float32(1.0 / _LOG2E)

    def init_runtime_indices(self):
        self.seq_len_v = fx.Index(self.seq_len)
        self.stride_o_n_v = fx.Index(self.stride_o_n)
        self.batch_size_v = fx.Index(self.batch_size)

    def init_thread_mapping(self, combine_rows_per_block, combine_lanes_per_row):
        traits = self.traits
        self.tid = fx.Index(gpu.thread_idx.x)
        self.blk = fx.Index(gpu.block_idx.x)
        self.batch_idx = fx.Index(gpu.block_idx.y)
        self.col = (self.tid % combine_lanes_per_row) * 4
        rows_per_batch = self.seq_len_v * traits.NUM_HEADS_Q
        row_raw = self.blk * combine_rows_per_block + self.tid // combine_lanes_per_row
        threads_in_use = fx.Index(combine_rows_per_block * combine_lanes_per_row)
        self.row = (self.tid < threads_in_use).select(row_raw, rows_per_batch)
        self.row_valid = self.row < rows_per_batch
        self.q_head_idx = self.row // self.seq_len_v
        self.seq_idx = self.row % self.seq_len_v

    def init_workspace(self):
        traits = self.traits
        z_total = self.batch_size_v * traits.NUM_KV_SPLITS
        self.ws_opart_per_split_elems = (
            fx.Index(traits.NUM_HEADS_Q)
            * self.seq_len_v
            * fx.Index(traits.HEAD_DIM_V // 2)
        )
        self.ws_ml_per_split_elems = fx.Index(traits.NUM_HEADS_Q) * self.seq_len_v
        self.ws_opart_per_split_bytes = self.ws_opart_per_split_elems * fx.Index(4)
        self.ws_ml_per_split_bytes = self.ws_ml_per_split_elems * fx.Index(4)
        self.ws_mrow_abs_bytes = z_total * self.ws_opart_per_split_bytes
        self.ws_lrow_abs_bytes = (
            self.ws_mrow_abs_bytes + z_total * self.ws_ml_per_split_bytes
        )
        self.local_ml_idx = self.q_head_idx * self.seq_len_v + self.seq_idx
        self.local_o_base = (
            self.q_head_idx * self.seq_len_v + self.seq_idx
        ) * fx.Index(traits.HEAD_DIM_V // 2)
        self.ws_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(self.WS)))

    def init_descriptors(self):
        if const_expr(self.traits.VARLEN):
            _cuq_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(self.CuSeqQ), fx.make_layout(1, 1)
            )
            _cu_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            _cu_v1i32 = Vec.make_type(1, fx.Int32)
            q_tok_base = _cu_load(_cuq_div, self.batch_idx, _cu_atom, _cu_v1i32)
            q_tok_end = _cu_load(
                _cuq_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32
            )
            batch_byte_off = q_tok_base * self.stride_o_n_v * fx.Index(2)
            nrec_bytes = (q_tok_end - q_tok_base) * self.stride_o_n_v * fx.Index(2)
        else:
            per_batch_elems = self.seq_len_v * self.stride_o_n_v
            batch_byte_off = self.batch_idx * per_batch_elems * fx.Index(2)
            nrec_bytes = per_batch_elems * fx.Index(2)
        self.o_nrec_bytes = nrec_bytes
        self.o_rsrc = buffer_ops.create_buffer_resource_from_addr(
            as_mlir_value(
                fx.Int64(fx.ptrtoint(fx.get_iter(self.O))) + fx.Int64(batch_byte_off)
            ),
            num_records_bytes=as_mlir_value(fx.Int64(nrec_bytes)),
        )
        self.load_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)

    def workspace_resource(self, byte_offset, nrec_bytes):
        return _make_ws_rsrc(self.ws_base_i64, byte_offset, nrec_bytes)

    def split_z(self, split_i):
        return self.batch_idx * self.traits.NUM_KV_SPLITS + split_i

    def opart_resource(self, split_z):
        return self.workspace_resource(
            split_z * self.ws_opart_per_split_bytes, self.ws_opart_per_split_bytes
        )

    def mrow_resource(self, split_z):
        return self.workspace_resource(
            self.ws_mrow_abs_bytes + split_z * self.ws_ml_per_split_bytes,
            self.ws_ml_per_split_bytes,
        )

    def lrow_resource(self, split_z):
        return self.workspace_resource(
            self.ws_lrow_abs_bytes + split_z * self.ws_ml_per_split_bytes,
            self.ws_ml_per_split_bytes,
        )


class DualwaveSplitKCombineHelper(DualwaveSplitKCombineContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_ml_rows(self):
        m_s = []
        l_s = []
        for i in range_constexpr(self.traits.NUM_KV_SPLITS):
            split_z_i = self.split_z(i)
            m_f32 = buffer_ops.buffer_load(
                self.mrow_resource(split_z_i),
                as_mlir_value(fx.Int32(self.local_ml_idx)),
                vec_width=1,
                dtype=T.f32,
            )
            l_f32 = buffer_ops.buffer_load(
                self.lrow_resource(split_z_i),
                as_mlir_value(fx.Int32(self.local_ml_idx)),
                vec_width=1,
                dtype=T.f32,
            )
            m_s.append(m_f32)
            l_s.append(l_f32)
        return m_s, l_s

    def reduce_m_max(self, m_s):
        m_max = m_s[0]
        for i in range_constexpr(self.traits.NUM_KV_SPLITS - 1):
            m_max = fx.maxnumf(m_max, m_s[i + 1])
        return m_max

    def fold_sink(self, m_max, bias_log2e):
        sink_rsrc = buffer_ops.create_buffer_resource_from_addr(
            as_mlir_value(fx.Int64(fx.ptrtoint(fx.get_iter(self.Sink)))),
            num_records_bytes=as_mlir_value(fx.Int64(self.traits.NUM_HEADS_Q * 4)),
        )
        sink_f32 = buffer_ops.buffer_load(
            sink_rsrc,
            as_mlir_value(fx.Int32(self.q_head_idx)),
            vec_width=1,
            dtype=T.f32,
        )

        sink_log2 = sink_f32 * fx.Float32(bias_log2e)
        m_new = fx.maxnumf(m_max, sink_log2)
        sink_w = rocdl.exp2(T.f32, as_mlir_value(sink_log2 - m_new))
        return m_new, sink_w

    def add_sink_den(self, den, sink_w):
        return den + sink_w

    def init_accumulators(self):
        return as_mlir_value(self.c_zero_v4f32), as_mlir_value(self.c_zero_f)

    def accumulate_split(self, acc, den, split_i, m_i, l_i, m_max):
        orsrc_i = self.opart_resource(self.split_z(split_i))
        local_o_idx_i = self.local_o_base + self.col // 2

        @flyc.jit
        def _accum_split(acc, den):
            if fx.Float32(l_i) > fx.Float32(0.0):
                w = rocdl.exp2(T.f32, as_mlir_value(m_i - m_max))
                wl = w * l_i
                den = den + wl
                o2_raw = buffer_ops.buffer_load(
                    orsrc_i,
                    as_mlir_value(fx.Int32(local_o_idx_i)),
                    vec_width=2,
                    dtype=T.i32,
                )
                o2_i32 = ir.Value(o2_raw)
                o4 = Vec(o2_i32, (2,), fx.Int32).bitcast(self.elem_dtype).to(fx.Float32)
                w4 = Vec.from_elements([fx.Float32(wl)], fx.Float32).broadcast_to(4)
                acc = acc + w4 * o4
            return acc, den

        return _accum_split(acc, den)

    def accumulate_splits(self, m_s, l_s, m_max):
        acc, den = self.init_accumulators()
        for i in range_constexpr(self.traits.NUM_KV_SPLITS):
            acc, den = self.accumulate_split(acc, den, i, m_s[i], l_s[i], m_max)
        return acc, den

    def pack_output(self, acc, den):
        inv_rcp = rocdl.rcp(T.f32, den)
        inv = (fx.Float32(den) > self.c_zero_f).select(inv_rcp, self.c_zero_f)
        inv4 = Vec.from_elements([fx.Float32(inv)], fx.Float32).broadcast_to(4)
        out4 = Vec(acc * inv4, (4,), fx.Float32)
        lo = rocdl.cvt_pk_bf16_f32(out4[0], out4[1])
        hi = rocdl.cvt_pk_bf16_f32(out4[2], out4[3])
        return Vec.from_elements([fx.Int32(lo), fx.Int32(hi)], fx.Int32)

    def store_lse(self, m_max, den):
        # Combined LSE = m_max * ln2 + ln(den); den = sum_s 2^(m_s - m_max) * l_s
        # completes the natural-log, scale-folded LSE. One lane (col == 0) writes.
        lse_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(self.LSE)))
        lse_per_batch_elems = fx.Index(self.traits.NUM_HEADS_Q) * self.seq_len_v
        lse_per_batch_bytes = lse_per_batch_elems * fx.Index(4)
        lse_rsrc = _make_ws_rsrc(
            lse_base_i64, self.batch_idx * lse_per_batch_bytes, lse_per_batch_bytes
        )
        lse_val = m_max * self.c_ln2_f + fx.log(den, fastmath=self.fm_fast)
        lse_in_range = self.row_valid.select(self.local_ml_idx, lse_per_batch_elems)
        lse_off = fx.Index(
            (self.col == fx.Index(0)).select(lse_in_range, lse_per_batch_elems)
        )
        buffer_ops.buffer_store(
            as_mlir_value(fx.Float32(lse_val)),
            lse_rsrc,
            as_mlir_value(fx.Int32(lse_off)),
        )

    def store_output(self, o_pack):
        o_global = (
            self.seq_idx * self.stride_o_n_v
            + self.q_head_idx * self.traits.HEAD_DIM_V
            + self.col
        )
        # Out-of-range rows aim past num_records, which the buffer drops.
        o_off = self.row_valid.select(o_global * fx.Index(2), self.o_nrec_bytes)
        buffer_ops.buffer_store(
            o_pack.ir_value(),
            self.o_rsrc,
            as_mlir_value(fx.Int32(o_off)),
            offset_is_bytes=True,
        )


@flyc.jit
def _stagger_extra_barrier_if_one(stagger_i32):
    """Emit `sched_barrier(0); s_barrier;` only when stagger == 1."""
    if fx.Int32(stagger_i32) != fx.Int32(0):
        rocdl.sched_barrier(0)
        rocdl.s_barrier()


def dualwave_splitk_workspace_elems(
    batch_size, num_heads, seq_len, num_kv_splits, head_dim=128
):
    """fp32 elements needed for the split-K workspace: O_partial + Mrow + Lrow.

    O_partial is stored as kernel-native 16-bit (bf16/fp16), two columns per
    fp32 slot; Mrow/Lrow stay fp32.
    """
    rows = batch_size * num_kv_splits * num_heads * seq_len
    return rows * (head_dim // 2) + 2 * rows
