# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared DUALWAVE_SWP flash-attention helpers, vendored from FlyDSL.

Consumed by ``flash_attn_fp8_gfx950.py``. Kept flat and un-suffixed rather
than under an arch directory because these are dtype/layout helpers, not
gfx950-specific code; a bf16 dualwave kernel would use the same ones.

Upstream: FlyDSL ``kernels/attention/flash_attn_utils.py`` at tag v0.3.0
(5675194f). ``flydsl.kernels`` is not shipped in the wheel (``kernels/`` sits
outside ``python/``, which is what ``find_packages`` scans), so aiter vendors
these sources rather than importing them.

Contents are the fp8-reachable subset of that file: 56 of 153 top-level
symbols, 1807 of 5535 lines, taken as the transitive AST closure of what
``flash_attn_fp8_gfx950.py`` imports. The bf16 ``Dualwave*`` hierarchy is
deliberately absent -- note that puts ``DualwavePageIdLoader``, the paged
block-table loader, outside this file.

Regenerate with ``scripts/vendor_fp8_attention.py`` (vault issue directory
``projects/aiter/issues/flydsl-unified-attention``) when the FlyDSL pin moves,
rather than hand-patching against a new upstream.
"""

import math as host_math

import os

from dataclasses import dataclass

import flydsl.compiler as flyc

import flydsl.expr as fx

from flydsl._mlir import ir

from flydsl._mlir.dialects import fly, llvm, vector

from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace as _TargetAddressSpace

from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch

from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl

from flydsl.expr import math as fmath

from flydsl.expr.typing import T

from flydsl.expr.typing import Vector as Vec

from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from flydsl.utils.smem_allocator import SmemPtr

from aiter.ops.flydsl.kernels import buffer_ops



def dtype_to_elem_type(dtype_str: str):
    """Map a dtype string to its FlyDSL numeric type.

    Local to this package rather than reusing
    ``aiter.ops.flydsl.kernels.kernels_common.dtype_to_elem_type``: that copy
    predates fp8 and returns MLIR ``T.*`` types, while these kernels need fx
    types for ``Vec.make_type``. Upstream equivalent is
    ``kernels/common/kernels_common.py:72`` at v0.3.0.
    """
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    if dtype_str == "fp8":
        return fx.Float8E4M3FN
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', 'bf16', or 'fp8')")


_LOG2E = host_math.log2(host_math.e)

NUM_XCD_GFX950 = 8

# s_waitcnt bitfield encoding. Byte-identical to upstream
# kernels/attention/flash_attn_utils.py:37-40 at v0.3.0; vendored because the
# fp8 kernel called only the full-drain form, so the AST closure never pulled
# these in.
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3


def stagger_extra_barrier_if_zero(stagger_i32):
    """Emit `s_barrier` only on waves with stagger == 0.

    Hand-written asm rather than a flyc.jit conditional: the branch must not be
    sunk, duplicated, or fenced, and this form emits no sched_barrier. Byte-
    identical to upstream flash_attn_utils.py:5489 at v0.3.0.
    """
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [stagger_i32],
        ("s_cmp_eq_u32 $0, 0\n\ts_cbranch_scc0 1f\n\ts_barrier\n\t1:"),
        "s",
        has_side_effects=True,
    )


@flyc.jit
def stagger_extra_barrier_if_one(stagger_i32):
    """Emit `sched_barrier(0); s_barrier` only on waves with stagger != 0.

    Mirrors upstream flash_attn_utils.py:5500. Carries the sched_barrier the
    _if_zero form omits, matching the cluster-boundary shape it stands in for.
    """
    if fx.Int32(stagger_i32) != fx.Int32(0):
        rocdl.sched_barrier(0)
        rocdl.s_barrier()


def waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7).

    vmcnt retires in issue order, so `n` means "at most n loads outstanding",
    i.e. everything but the last n has landed.
    """
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    rocdl.s_waitcnt(val)

def _read_exec_i64():
    """Read the current wave exec mask, matching Clang's builtin lowering."""
    true_i1 = fx.Boolean(True).ir_value()
    return rocdl.ballot(T.i64, true_i1)

def _ds_read_tr16_b64_imm(result_type, addr_i32, imm_offset=0):
    """gfx950 ds_read_b64_tr_b16 with DUALWAVE_SWP immediate byte offset."""
    imm = int(imm_offset)
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    raw = llvm.inline_asm(
        raw_type,
        [as_mlir_value(addr_i32)],
        f"ds_read_b64_tr_b16 $0, $1 offset:{imm}\n",
        "=v,v,~{memory}",
        has_side_effects=True,
    )
    return vector.BitCastOp(result_type, raw).result

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

def _fadd(a, b, fm_fast):
    return arith.addf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)

def _fsub(a, b, fm_fast):
    return arith.subf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)

def _fmul(a, b, fm_fast):
    return arith.mulf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)

def _fmax(a, b, fm_fast):
    return arith.MaxNumFOp(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast).result

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
    ret_ty = ir.Type.parse(f"!llvm.struct<({', '.join(['vector<16xf32>'] * traits.D_CHUNKS)})>")
    constraints = ",".join(["=v"] * traits.D_CHUNKS + [str(i) for i in range(traits.D_CHUNKS)])
    ret = llvm.inline_asm(
        ret_ty,
        acc_irs,
        "",
        constraints,
        has_side_effects=True,
    )
    return [llvm.extractvalue(acc_irs[dc].type, ret, [dc]) for dc in range_constexpr(traits.D_CHUNKS)]

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
        anchored_lo.append(p_vec.shuffle(p_vec, [lo_base + i for i in range(8)]).ir_value())
        anchored_hi.append(p_vec.shuffle(p_vec, [hi_base + i for i in range(8)]).ir_value())
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
    return _lane_pair_reduce(_reduce_score_pair(v_s, neg_inf, _fmax, fm_fast), _fmax, fm_fast)

def _score_pair_sum(v_s, zero_f, fm_fast):
    return _lane_pair_reduce(_reduce_score_pair(v_s, zero_f, _fadd, fm_fast), _fadd, fm_fast)

def _scale_sub_score_pair(v_s, row_max_raw, scale, zero_f, fm_fast):
    """Fused softmax-scale + row-max subtraction (optimization 1-A).

    Returns ``scale * (v_s - row_max_raw)`` per element via a single FMA
    (``fma(s, scale, -scale*row_max_raw)``), so the fp8 QK MMA can emit raw
    (un-scaled) logits and reduce_max can run in the raw domain (scale > 0 is
    order-preserving). Replaces the separate post-QK scale multiply + subtract.
    ``-inf`` masked lanes stay ``-inf`` (scale > 0), matching the un-fused path.
    """
    s_lo, s_hi = v_s
    neg_scaled_max = _fsub(zero_f, _fmul(scale, row_max_raw, fm_fast), fm_fast)
    lo = [fmath.fma(s_lo[r], scale, neg_scaled_max, fastmath=fm_fast) for r in range_constexpr(16)]
    hi = [fmath.fma(s_hi[r], scale, neg_scaled_max, fastmath=fm_fast) for r in range_constexpr(16)]
    return Vec.from_elements(lo, fx.Float32).ir_value(), Vec.from_elements(hi, fx.Float32).ir_value()

def _exp2_score_slice(v_s, start, length):
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

def _scale_o_accs(v_o, scale_scalar, traits, fm_fast):
    scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
    for dc in range_constexpr(traits.D_CHUNKS):
        v_o[dc] = _fmul(Vec(v_o[dc]), scale_vec, fm_fast)

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
        new_x, new_y = _attn_mask_vec2_imm(rel_i32, neg_inf_i32, thr_x, thr_y, x_bits, y_bits)
        s_values[idx_x] = _bitcast_f32(new_x)
        s_values[idx_y] = _bitcast_f32(new_y)

def _cu_load(div, idx, cu_atom, cu_v1i32):
    v = fly.copy_atom_call_ssa([cu_v1i32], cu_atom, fx.slice(div, (None, fx.Int32(idx))))
    return fx.Index(Vec(v, (1,), fx.Int32)[0])

def _make_page_view(
    base_iter, base_iter_ty, align, page_id, page_byte_stride, page_nrec_bytes, page_layout, elem_ir, buf_flags_i32
):
    """Buffer descriptor covering exactly one KV page.

    Vendored from upstream ``flash_attn_utils._make_page_view`` (v0.3.0), which is
    already dtype-parametric via ``elem_ir`` -- the fp8 caller passes i8 to match its
    byte-indexed dense descriptors. ``num_records`` is one page, so a read past the
    page end returns 0 instead of the neighbouring page: the bound is the OOB guard.
    """
    base_i64 = fx.Int64(fx.ptrtoint(base_iter))
    off_i64 = fx.Int64(page_id * page_byte_stride)
    shifted = fx.inttoptr(base_iter_ty, base_i64 + off_i64)
    buf_ptr_ty = fx.PointerType.get(elem_ty=elem_ir, address_space=_TargetAddressSpace.BufferDesc, alignment=align)
    buf_ptr = fx.make_ptr(
        buf_ptr_ty,
        [shifted, fx.Int16(0).ir_value(), page_nrec_bytes.ir_value(), buf_flags_i32.ir_value()],
    )
    return fx.logical_divide(fx.make_view(buf_ptr, page_layout), fx.make_layout(1, 1))

def _paged_bt_byte_offset(tile_idx, split_t0):
    """Byte offset of `tile_idx`'s page-id entry in the LDS block-table cache."""
    return fx.Int32((tile_idx - split_t0) * fx.Index(4))

def _make_ws_rsrc(ws_base_i64, byte_offset, nrec_bytes):
    addr_i64 = as_mlir_value(ws_base_i64 + fx.Int64(byte_offset))
    return buffer_ops.create_buffer_resource_from_addr(addr_i64, num_records_bytes=as_mlir_value(fx.Int64(nrec_bytes)))

def _buffer_load_128(elem_index, _load_atom_128, q_div, q_load_i32x4_type):
    """128-bit global->register load (buffer_load_dwordx4) from Q."""
    return fly.copy_atom_call_ssa([q_load_i32x4_type], _load_atom_128, fx.slice(q_div, (None, fx.Int32(elem_index))))

def _buffer_load_lds_128(src_div, lds_byte_addr, src_elem, soffset_elems, _dma_atom, _lds_ptr_ty):
    """128-bit global->LDS DMA; `src_elem` is voffset, `soffset_elems` is scaled by the atom."""
    lds_ptr = fx.inttoptr(_lds_ptr_ty, fx.Int32(lds_byte_addr))
    dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
    src = fx.slice(src_div, (None, fx.Int32(src_elem)))
    fx.copy(_dma_atom, src, dst, soffset=fx.Int32(soffset_elems))

def _buffer_store_128(pack_i32_vec, elem_index, _o_store_reg_128, _store_atom_128, o_div):
    """128-bit register->global store (buffer_store_dwordx4) into O."""
    fx.memref_store_vec(pack_i32_vec, _o_store_reg_128)
    fx.copy(_store_atom_128, _o_store_reg_128, fx.slice(o_div, (None, fx.Int32(elem_index))))

def _init_dualwave_thread_mapping(ctx):
    """Set block/wave/lane/head indices on a dualwave-style context.

    Shared verbatim by DualwaveKernelContext and DualwaveFp8KernelContext."""
    traits = ctx.traits
    # Swizzled Head-first Mapping (arXiv:2511.02132): the grid is head-fast, so one
    # head's q-blocks scatter across all XCDs and each re-streams its K/V. Re-derive
    # (head, q_block) with head as the slow axis to keep them on one XCD. Bijective,
    # so output is bit-identical; split-K's third grid axis would not survive it.
    # Non-causal only: under a causal mask q-block i does work proportional to i, so
    # making q_block the fast axis clusters unequal work and costs 7% (measured).
    # grid.x is NUM_HEADS_KV under GQA packing (the group rides in M) and
    # NUM_HEADS_Q otherwise. The swizzle below linearises (x, y), so it must use
    # the ACTUAL x extent -- using NUM_HEADS_Q while the grid launched
    # NUM_HEADS_KV makes the mapping non-bijective and silently drops every head
    # above num_kv_heads (found 2026-08-07: heads 0-3 correct, 4-7 at cos 0.00).
    _grid_x_extent = traits.NUM_HEADS_KV if traits.GQA_PACK_M else traits.NUM_HEADS_Q
    if const_expr(not traits.SPLITK and not traits.CAUSAL and _grid_x_extent % NUM_XCD_GFX950 == 0):
        num_q_blocks = fx.Index(gpu.grid_dim.y)
        linear_wg = fx.Index(gpu.block_idx.x) + fx.Index(gpu.block_idx.y) * fx.Index(_grid_x_extent)
        ctx.h_idx = linear_wg // num_q_blocks
        ctx.q_block_idx = linear_wg % num_q_blocks
    else:
        ctx.h_idx = fx.Index(gpu.block_idx.x)
        ctx.q_block_idx = fx.Index(gpu.block_idx.y)
    if const_expr(traits.SPLITK):
        ctx.bz_idx = fx.Index(gpu.block_idx.z)
        ctx.batch_idx = ctx.bz_idx // traits.NUM_KV_SPLITS
        ctx.split_idx = ctx.bz_idx % traits.NUM_KV_SPLITS
    else:
        ctx.batch_idx = fx.Index(gpu.block_idx.z)
        ctx.split_idx = None
    ctx.tid = fx.Index(gpu.thread_idx.x)

    ctx.wave_id = ctx.tid // traits.WARP_SIZE
    ctx.lane = ctx.tid % traits.WARP_SIZE
    ctx.lane_mod_32 = ctx.lane % 32
    ctx.lane_div_32 = ctx.lane // 32

    _tid_i32 = as_mlir_value(fx.Int32(ctx.tid))
    _wave_id_uni_i32 = rocdl.readfirstlane(
        T.i32,
        arith.divsi(_tid_i32, as_mlir_value(fx.Int32(traits.WARP_SIZE))),
    )
    ctx.stagger_i32 = arith.divsi(_wave_id_uni_i32, as_mlir_value(fx.Int32(4)))
    ctx.wave_id_uni = fx.Index(_wave_id_uni_i32)

    ctx.wave_q_offset = ctx.wave_id * traits.ROWS_PER_WAVE
    ctx.q_start = ctx.q_block_idx * traits.BLOCK_Q

    if const_expr(traits.GQA_PACK_M):
        # grid.x is num_kv_heads: the whole GQA group rides in M, so the q-head
        # is a function of the M row, not of the workgroup. q_head_idx is left
        # unset -- every consumer must use q_head_of_row(), and a stale scalar
        # read would silently address one head for all 16.
        ctx.kv_head_idx = ctx.h_idx
        ctx.h_kv_idx = ctx.h_idx
        ctx.group_id = None
        ctx.q_head_base = ctx.h_idx * traits.GQA_GROUP_SIZE
        ctx.q_head_idx = None
    else:
        ctx.h_kv_idx = ctx.h_idx % traits.NUM_HEADS_KV
        ctx.group_id = ctx.h_idx // traits.NUM_HEADS_KV
        ctx.q_head_idx = ctx.h_kv_idx * traits.GQA_GROUP_SIZE + ctx.group_id
        ctx.kv_head_idx = ctx.h_kv_idx
        ctx.q_head_base = None

def _init_dualwave_q_row(ctx):
    """Set q_row / q_row_i32 / q_start_pos_i32 on a dualwave-style context.

    ``q_row_in_block`` is the M row. Under GQA_PACK_M that row decomposes into
    a query position and a head within the group, exactly as Triton does at
    unified_attention.py:139:

        query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    so ``q_row`` (the TOKEN index) advances only once per GQA_GROUP_SIZE rows,
    and the head advances within them. Without packing the two coincide and
    ``q_row_in_block`` is itself the token offset.
    """
    traits = ctx.traits
    ctx.q_row_in_block = ctx.wave_q_offset + ctx.lane_mod_32
    if const_expr(traits.GQA_PACK_M):
        g = fx.Index(traits.GQA_GROUP_SIZE)
        ctx.q_pos_in_block = ctx.q_row_in_block // g
        ctx.q_head_in_group = ctx.q_row_in_block % g
        ctx.q_head_row = ctx.q_head_base + ctx.q_head_in_group
        # Causal bounds are per query POSITION; all heads of a position share a
        # row of the score matrix, so the wave's start position divides too.
        ctx.q_start_pos_i32 = fx.Int32(
            ctx.q_start + (ctx.wave_id_uni * traits.ROWS_PER_WAVE) // g)
    else:
        ctx.q_pos_in_block = ctx.q_row_in_block
        ctx.q_head_in_group = None
        ctx.q_head_row = ctx.q_head_idx
        ctx.q_start_pos_i32 = fx.Int32(
            ctx.q_start + ctx.wave_id_uni * traits.ROWS_PER_WAVE)
    ctx.q_row = ctx.q_start + ctx.q_pos_in_block
    ctx.q_row_i32 = fx.Int32(ctx.q_row)

@dataclass(frozen=True)
class DualwaveSwpFp8Traits:
    """Pure compile-time tile/layout constants for the gfx950 DUALWAVE_SWP fp8 kernel.

    fp8 runs a single path: WIDE QK (32x32x64 mfma_scale) feeding HIPREC PV (fp8 V
    dequantized into a bf16 ``vt`` LDS scratch, then a bf16 PV MMA). The ``*_BF``
    fields describe that bf16 vt layout; ``ELEM_BYTES`` is 1 (Q/K/V are fp8)."""

    BLOCK_M: int
    BLOCK_N: int
    K_SUB_N: int
    WARP_SIZE: int
    NUM_WAVES: int
    BLOCK_SIZE: int
    ROWS_PER_WAVE: int
    HEAD_DIM: int
    D_CHUNK: int
    D_CHUNKS: int
    PV_K_STEPS: int
    NUM_HEADS_Q: int
    NUM_HEADS_KV: int
    GQA_GROUP_SIZE: int
    # GQA_PACK_M factors the M dimension as BLOCK_Q query POSITIONS x
    # GQA_GROUP_SIZE heads, instead of BLOCK_M positions x 1 head. That is what
    # lets one kernel serve prefill and decode in a single launch: at Sq=1 a
    # single token fills the tile because the tile's width is heads, and at
    # Sq=4023 the same tile holds BLOCK_Q positions. Mirrors Triton's
    # `query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv`
    # (aiter/ops/triton/_triton_kernels/attention/unified_attention.py:139).
    # Off: the legacy one-head-per-workgroup shape, BLOCK_Q == BLOCK_M.
    GQA_PACK_M: bool
    BLOCK_Q: int
    CAUSAL: bool
    DTYPE_STR: str
    WAVES_PER_EU: int
    DAZ: bool
    DUALWAVE_SWP_LAZY_RESCALE: bool
    DUALWAVE_SWP_SETPRIO: bool
    DUALWAVE_SWP_DEBUG_LAZY_COUNTS: bool
    DUALWAVE_SWP_ENABLE_STAGGER: bool
    NUM_KV_SPLITS: int
    SPLITK: bool
    VARLEN: bool
    CROSS_SEQLEN: bool
    PAGED: bool
    PAGED_BT_LDS_SIZE: int
    PREFETCH_BOUND: str
    PV_SPREAD: bool
    FP8_PV: bool
    FP8_PV_DIRECT: bool
    BN128: bool
    BN128_PF: bool
    QREG: bool
    VDMA: bool
    DEFAULT_STRIDE_Q_N: int
    DEFAULT_STRIDE_KV_N: int
    DMA_BYTES: int
    ELEM_BYTES: int
    OUT_ELEM_BYTES: int
    D_128B_SIZE: int
    VEC_KV: int
    LANE_SPLIT_KV: int
    SMEM_N_RPT: int
    SMEM_D_RPT: int
    SMEM_K_LINE_STRIDE: int
    SMEM_K_TILE_ELEMS: int
    NUM_PREFETCH_K: int
    DUALWAVE_SWP_KV_PER_BUFFER: int
    LDS_KV_TOTAL_SIZE: int
    DUALWAVE_SWP_K_BUF_BASE: tuple[int, int]
    DUALWAVE_SWP_V_BUF_BASE: tuple[int, int]
    # bf16 vt scratch layout (HIPREC V dequant target + transpose read strides).
    EB_BF: int
    D128_BF: int
    VEC_BF: int
    SDRPT_BF: int
    SNRPT_BF: int
    VLS_BF: int
    VT_BF16_ELEMS: int
    VT_BF16_TOTAL: int
    URV_GRPK_BF: int
    URV_GRP_N_BF: int
    URV_LANE_LO_BF: int
    URV_LANE_HI_BF: int
    URV_STEPK_BF: int
    URV_DC_AXIS0_BF: int
    URV_DC_AXIS1_BF: int
    URV_I5_BF: int
    DUALWAVE_SWP_RESCALE_THRESHOLD: float
    SCHED_MFMA_MASK: int
    SCHED_VALU_MASK: int
    SCHED_EXP_MASK: int
    SCHED_DS_READ_MASK: int
    LDS_SCOPE_NAMES: tuple[str, str, str, str]
    NEG_INF_F32_BITS: int
    LGKMCNT_0_ONLY: int

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
            self.DUALWAVE_SWP_SETPRIO,
            self.DUALWAVE_SWP_DEBUG_LAZY_COUNTS,
            self.DUALWAVE_SWP_ENABLE_STAGGER,
            self.NUM_KV_SPLITS,
            self.SPLITK,
            self.VARLEN,
            self.CROSS_SEQLEN,
            # PAGED must stay in the tag: dense and paged differ only in const_expr
            # branches, so without it both hash to one key and the builder's cache
            # hands back whichever compiled first.
            self.PAGED,
            self.PAGED_BT_LDS_SIZE,
            self.PREFETCH_BOUND,
            self.PV_SPREAD,
            "fp8_wide_qk_hiprec_pv",
            self.ELEM_BYTES,
            self.OUT_ELEM_BYTES,
            self.LANE_SPLIT_KV,
            self.VT_BF16_ELEMS,
            self.VT_BF16_TOTAL,
            self.FP8_PV,
            self.FP8_PV_DIRECT,
            self.NUM_PREFETCH_K,
            # BLOCK_M/NUM_WAVES became parameters 2026-08-07. They change the
            # tile shape and the CTA width, so they must key the cache -- two
            # block_m values would otherwise hash alike and the builder would
            # hand back whichever compiled first. Same class of bug the PAGED
            # note above records.
            self.BLOCK_M,
            self.NUM_WAVES,
            self.BLOCK_SIZE,
            self.GQA_PACK_M,
            self.BLOCK_Q,
            self.BN128,
            self.BN128_PF,
            self.QREG,
            self.VDMA,
        )

def _make_dualwave_swp_fp8_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    causal=True,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    dualwave_swp_setprio=True,
    dualwave_swp_debug_lazy_counts=False,
    dualwave_swp_enable_stagger=True,
    num_kv_splits=1,
    varlen=False,
    cross_seqlen=False,
    bn128=None,
    paged=False,
    prefetch_bound="none",
    pv_spread=False,
    num_waves=8,
    num_prefetch_k_override=None,
    gqa_pack_m=None,
):
    """Build gfx950 DUALWAVE_SWP fp8 compile-time layout traits (dtype fixed to fp8).

    ``bn128`` selects the deep-pipeline shape (6-deep K prefetch ring, Q in
    registers, DMA V staging, direct packed-i32x8 PV). ``None`` keeps
    upstream's behaviour of deriving it from ``num_kv_splits``/``varlen``;
    pass True or False to choose it independently of those. See the comment at
    the derivation site for why the two are separable.

    ``paged`` addresses KV through a block table instead of contiguously. Page
    size is not a parameter: it is structurally ``BLOCK_N`` (64), which is why
    the host pins ``_PAGED_PAGE_SIZE = 64``. Only the linear cache layout is
    supported here -- the vectorized 5D layout is bf16-only upstream and the
    target workload reports ``SHUFFLED_KV_CACHE_0``.

    ``prefetch_bound="clamp"`` bounds the ring's forward prefetch; it defaults
    off so the dense path stays bit-identical to upstream, and it is not a perf
    win (see its use site).

    ``num_waves`` sets the CTA width and, with it, ``block_m = num_waves * 32``
    (``rows_per_wave`` is pinned by the MFMA tile, so this is the only way to
    move ``block_m``). Default 8 -> block_m 256, the shape every prefill number
    in this issue was measured at.

    A 2026-08-06 note here claimed block_m could not usefully be lowered
    because "the KV ring alone is 96.8 KB against an 80 KB half-budget, at any
    block_m". Re-derived 2026-08-07: that figure is npf=4, but the fp8 path
    runs npf=6, and the ring does not depend on block_m at all
    (``smem_k_tile_elems`` is a function of ``block_n`` and ``head_dim`` only).
    The real numbers per CU, against 160 KB of LDS on gfx950:

    | npf | block_m=256 | block_m=32 |
    |-----|-------------|------------|
    |   6 |    129.0 KB |    101.0 KB |
    |   4 |     96.8 KB |     68.8 KB |
    |   3 |     80.6 KB |     52.6 KB |

    So the old conclusion (block_m alone cannot reach 2 workgroups/CU) holds,
    but the reason is the Q tile, not the ring: at block_m=256 the 32 KB of Q
    swamps any ring saving, and at npf=6 no block_m fits twice. Lowering both
    together does fit. Ring depth is ``num_prefetch_k``, still derived from
    ``bn128`` -- shallowing it for fp8 is unmeasured, which is what the
    block_m/npf sweep is for.

    Still absent: there is no short-KV prologue variant: the prologue stages
    exactly the 4 tiles the minimum two loop iterations consume, so there is
    nothing to trim.
    """
    # Tile shape and wave geometry follow the gfx950 dual-wave CTA.
    block_n = 64
    k_sub_n = 32
    warp_size = 64
    rows_per_wave = 32
    num_waves = int(num_waves)
    assert num_waves in (1, 2, 4, 8), f"num_waves must be a power of two <= 8, got {num_waves}"
    # The fp8 V staging path distributes BLOCK_N rows over waves as
    # `n = wave_id * 8 + ...` (_stage_v_fp8_block / _stage_v_fp8_block_dma), a
    # literal 8 rather than a BLOCK_N // NUM_WAVES. At num_waves < 8 that
    # covers only part of the tile and the rest is silently stale -- wrong
    # output, not a crash. K staging is fine (it uses NUM_WAVES). Lift this by
    # deriving the V stride from BLOCK_N // num_waves in both stagers.
    assert not (num_waves != 8), (
        f"fp8 V staging is hardcoded to 8 waves; num_waves={num_waves} would "
        "stage a partial V tile. See _stage_v_fp8_block."
    )
    block_size = num_waves * warp_size
    block_m = num_waves * rows_per_wave

    d_chunk = 32
    d_chunks = head_dim // d_chunk
    pv_k_step = 16
    pv_k_steps = k_sub_n // pv_k_step

    gqa_group_size = num_heads // num_kv_heads
    if gqa_pack_m is None:
        # Default on wherever it is valid. Measured 2026-08-07 against the
        # legacy factorisation, fp8 paged+varlen causal on MI355X: mixed batch
        # 1.15-1.81x, decode 2.1-11.8x, prefill 0.99x (unchanged). It also wins
        # or ties against Triton unified_attention everywhere, where the legacy
        # shape lost the mixed batch at 0.60-0.97x.
        #
        # Two exclusions. GQA 1:1 is MHA -- packing is the identity there, so it
        # only costs a distinct cache entry. Split-K is refused below.
        gqa_pack_m = (gqa_group_size > 1) and (num_kv_splits <= 1)
    gqa_pack_m = bool(gqa_pack_m)
    if gqa_pack_m:
        assert block_m % gqa_group_size == 0, (
            f"block_m={block_m} must be divisible by gqa_group_size="
            f"{gqa_group_size} to pack the group into M")
        # The split-K partial store and its combine index the workspace by a
        # SCALAR q_head_idx (store_splitk_partial_o and the combine kernel),
        # which does not exist under packing -- the head varies per M row. That
        # is a real but separate change; refuse rather than write every head of
        # the group into one head's workspace rows.
        assert num_kv_splits <= 1, (
            "gqa_pack_m with num_kv_splits > 1 is not implemented: the split-K "
            "workspace is indexed by a scalar q_head_idx per workgroup")
        block_q = block_m // gqa_group_size
    else:
        block_q = block_m
    default_stride_q_n = num_heads * head_dim
    default_stride_kv_n = num_kv_heads * head_dim

    # fp8: Q/K/V are 1B; O is bf16 (2B). ELEM_BYTES=1 drives the fp8 address math.
    elem_bytes = 1
    out_elem_bytes = 2
    d_128b_size = 128 // elem_bytes
    vec_kv = 16 // elem_bytes
    lane_split_kv = 8
    smem_linear_wave = warp_size * 16 // elem_bytes
    smem_n_per_wave = smem_linear_wave // d_128b_size
    smem_n_rpt = block_n // smem_n_per_wave
    smem_d_rpt = head_dim // d_128b_size
    smem_k_pad = 16 // elem_bytes
    smem_v_pad = 64 // elem_bytes
    smem_k_line_stride = smem_linear_wave + smem_k_pad
    smem_v_line_stride = smem_linear_wave + smem_v_pad
    smem_k_tile_elems = smem_n_rpt * smem_d_rpt * smem_k_line_stride
    smem_v_tile_elems = smem_n_rpt * smem_d_rpt * smem_v_line_stride
    # BN128 is the deep-pipeline shape: a 6-deep K prefetch ring, Q held in
    # registers, DMA V staging, and the packed-i32x8 direct PV path. Upstream
    # derives it as `(num_kv_splits <= 1) and (not varlen)`, which conflates
    # two unrelated things -- none of what BN128 selects depends on how Q is
    # packed or on whether the KV range is split. Measured 2026-08-06: forcing
    # FP8_PV_DIRECT true under varlen makes the kernel build and compute real
    # attention (cos 0.96), and moves the split-K failure to an unrelated
    # workspace fault. See the vault issue's bf16-template-assessment.
    #
    # So the pipeline shape is its own parameter here. It still defaults to
    # upstream's expression, because the deep ring has only ever been measured
    # dense (1804-1858 TFLOPS at NPF=6) and the shallow path is unexercised on
    # the fp8 side -- callers opt in rather than inherit a silent change.
    if bn128 is None:
        bn128 = (num_kv_splits <= 1) and (not varlen)
    bn128 = bool(bn128)
    bn128_pf = bn128
    qreg = bn128_pf
    vdma = bn128_pf
    deep_ring = bn128
    num_prefetch_k = (6 if bn128_pf else 4) if deep_ring else 2
    if num_prefetch_k_override is not None:
        # Ring depth independent of the BN128 shape. BN128 selects four things
        # at once (ring depth, Q in registers, V DMA staging, direct PV) and
        # only the first is an LDS-budget knob; the fp8 body needs the other
        # three.
        #
        # Measured 2026-08-07, and the floor is 6, not 2. This knob resizes the
        # LDS ring, but the BN128 body's prefetch distance is hardcoded: the
        # prologue stages tiles t0..t0+3 (flash_attn_fp8_gfx950.py:306-323) and
        # the main loop prefetches at +4/+5 (`_ring_wrap(a_buf + 4/5)`, :398-399),
        # so six buffers must be live simultaneously. Below that the modulo
        # aliases a buffer being prefetched onto one still being read:
        #   npf=5 -> bit-identical but 2.3% slower (aliasing not yet fatal)
        #   npf=4 -> WRONG OUTPUT, maxerr 4.2e-02 against npf=6
        #   npf=3 -> memory fault
        # So the ring cannot be shallowed without also shortening the prefetch
        # distance in the body, which is the pipeline's whole point.
        #
        # That closes off 2 workgroups/CU entirely for this pipeline: at npf=6
        # the ring (48.8 KB) plus the bf16 V staging (48.2 KB) is 97.0 KB before
        # a single Q row, already over the 80 KB half-budget. Shrinking block_m
        # 256 -> 16 saves 30 KB and lands at 99.0 KB, still 1 workgroup/CU. So
        # the M-dimension refactor must be justified by issue rate and grid
        # shape, NOT by occupancy -- there is no second workgroup to be had
        # without restructuring the V staging too.
        num_prefetch_k = int(num_prefetch_k_override)
        assert 2 <= num_prefetch_k <= 6, (
            f"num_prefetch_k must be in [2, 6], got {num_prefetch_k}")
        if num_prefetch_k < 6:
            import warnings
            warnings.warn(
                f"num_prefetch_k={num_prefetch_k} < 6 aliases live ring buffers "
                "in the BN128 body (wrong output at 4, faults at 3). Probe only.",
                stacklevel=2)
    if bn128_pf:
        dualwave_swp_kv_per_buffer = smem_k_tile_elems
    else:
        dualwave_swp_kv_per_buffer = smem_k_tile_elems + smem_v_tile_elems
    lds_kv_total_size = num_prefetch_k * dualwave_swp_kv_per_buffer
    dualwave_swp_k_buf_base = tuple(i * dualwave_swp_kv_per_buffer for i in range(num_prefetch_k))
    dualwave_swp_v_buf_base = tuple(smem_k_tile_elems + i * dualwave_swp_kv_per_buffer for i in range(num_prefetch_k))

    # bf16 vt scratch layout: HIPREC dequantizes fp8 V into these positions so the
    # proven bf16 V transpose read (ds_read_tr16) + bf16 PV MMA are reused unchanged.
    eb_bf = 2
    d128_bf = 128 // eb_bf
    vec_bf = 16 // eb_bf
    slw_bf = warp_size * 16 // eb_bf
    snrpt_bf = block_n // (slw_bf // d128_bf)
    sdrpt_bf = head_dim // d128_bf
    vls_bf = slw_bf + 64 // eb_bf
    vt_bf16_elems = snrpt_bf * sdrpt_bf * vls_bf
    fp8_v_tile_bytes = (block_n // 8) * (head_dim // 16) * 128
    if bn128_pf:
        vt_bf16_total = num_prefetch_k * (fp8_v_tile_bytes // eb_bf) + 128
    else:
        vt_bf16_total = (2 if deep_ring else num_prefetch_k) * vt_bf16_elems

    splitk = num_kv_splits > 1

    fp8_pv = os.getenv("FLYDSL_FA_FP8_PV", "0") == "1"
    fp8_pv_direct = bn128
    if fp8_pv_direct:
        fp8_pv = True

    return DualwaveSwpFp8Traits(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        K_SUB_N=k_sub_n,
        WARP_SIZE=warp_size,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=block_size,
        ROWS_PER_WAVE=rows_per_wave,
        HEAD_DIM=head_dim,
        D_CHUNK=d_chunk,
        D_CHUNKS=d_chunks,
        PV_K_STEPS=pv_k_steps,
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        GQA_PACK_M=gqa_pack_m,
        BLOCK_Q=block_q,
        CAUSAL=causal,
        DTYPE_STR="fp8",
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        DUALWAVE_SWP_LAZY_RESCALE=bool(dualwave_swp_lazy_rescale),
        DUALWAVE_SWP_SETPRIO=bool(dualwave_swp_setprio),
        DUALWAVE_SWP_DEBUG_LAZY_COUNTS=bool(dualwave_swp_debug_lazy_counts),
        DUALWAVE_SWP_ENABLE_STAGGER=bool(dualwave_swp_enable_stagger),
        NUM_KV_SPLITS=num_kv_splits,
        SPLITK=splitk,
        VARLEN=bool(varlen),
        CROSS_SEQLEN=bool(cross_seqlen),
        PAGED=bool(paged),
        # 2048 int32 entries = 8 KB of LDS, matching bf16. Caps a split's tile
        # count; the host checks the window before dispatching.
        PAGED_BT_LDS_SIZE=2048,
        PREFETCH_BOUND=str(prefetch_bound),
        PV_SPREAD=bool(pv_spread),
        FP8_PV=fp8_pv,
        FP8_PV_DIRECT=bool(fp8_pv_direct),
        BN128=bool(bn128),
        BN128_PF=bool(bn128_pf),
        QREG=bool(qreg),
        VDMA=bool(vdma),
        DEFAULT_STRIDE_Q_N=default_stride_q_n,
        DEFAULT_STRIDE_KV_N=default_stride_kv_n,
        DMA_BYTES=16,
        ELEM_BYTES=elem_bytes,
        OUT_ELEM_BYTES=out_elem_bytes,
        D_128B_SIZE=d_128b_size,
        VEC_KV=vec_kv,
        LANE_SPLIT_KV=lane_split_kv,
        SMEM_N_RPT=smem_n_rpt,
        SMEM_D_RPT=smem_d_rpt,
        SMEM_K_LINE_STRIDE=smem_k_line_stride,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        NUM_PREFETCH_K=num_prefetch_k,
        DUALWAVE_SWP_KV_PER_BUFFER=dualwave_swp_kv_per_buffer,
        LDS_KV_TOTAL_SIZE=lds_kv_total_size,
        DUALWAVE_SWP_K_BUF_BASE=dualwave_swp_k_buf_base,
        DUALWAVE_SWP_V_BUF_BASE=dualwave_swp_v_buf_base,
        EB_BF=eb_bf,
        D128_BF=d128_bf,
        VEC_BF=vec_bf,
        SDRPT_BF=sdrpt_bf,
        SNRPT_BF=snrpt_bf,
        VLS_BF=vls_bf,
        VT_BF16_ELEMS=vt_bf16_elems,
        VT_BF16_TOTAL=vt_bf16_total,
        URV_GRPK_BF=4 * vls_bf,
        URV_GRP_N_BF=16,
        URV_LANE_LO_BF=4,
        URV_LANE_HI_BF=vls_bf,
        URV_STEPK_BF=128,
        URV_DC_AXIS0_BF=snrpt_bf * vls_bf,
        URV_DC_AXIS1_BF=32,
        URV_I5_BF=d128_bf,
        DUALWAVE_SWP_RESCALE_THRESHOLD=8.0,
        SCHED_MFMA_MASK=0x008,
        SCHED_VALU_MASK=0x002,
        SCHED_EXP_MASK=0x400,
        SCHED_DS_READ_MASK=0x100,
        LDS_SCOPE_NAMES=("lds_k0", "lds_k1", "lds_v0", "lds_v1"),
        NEG_INF_F32_BITS=0xFF800000,
        LGKMCNT_0_ONLY=0xC07F,
    )

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
        O=None,  # noqa: E741
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
        BlockTable=None,
        block_table_stride=None,
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
        self.BlockTable = BlockTable
        self.block_table_stride = block_table_stride

    def init_types_and_constants(self):
        traits = self.traits
        self.elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)
        self.fm_fast = fx.arith.FastMathFlags.fast
        self.v4i32_type = Vec.make_type(4, fx.Int32)
        self.v4f16_type = Vec.make_type(4, self.elem_dtype)
        self.v16f32_type = Vec.make_type(16, fx.Float32)
        self.v2i32_type = Vec.make_type(2, fx.Int32)
        self.p_elem = fx.BFloat16
        self.v4bf16_type = Vec.make_type(4, fx.BFloat16)
        self.NUM_DMA_K = traits.SMEM_D_RPT
        self.NUM_DMA_V = traits.SMEM_D_RPT
        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_neg_floor = fx.Float32(-3.0e38)
        self.c_zero_f = fx.Float32(0.0)
        self.c_eight_f = fx.Float32(traits.DUALWAVE_SWP_RESCALE_THRESHOLD)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)

    def init_runtime_indices(self):
        self.seq_len_v = fx.Index(self.seq_len)
        self.seq_len_kv_v = fx.Index(self.seq_len_kv)
        self.stride_q_n_v = fx.Index(self.stride_q_n)
        self.stride_kv_n_v = fx.Index(self.stride_kv_n)

    def init_causal_lpt_order(self):
        """Issue causal q-blocks longest-first by reversing the q-block grid axis.

        Causal work per q-block grows with the block index and workgroups dispatch in
        flattened-id order, so the natural order issues the heaviest block last and the
        makespan carries its tail. Must run after init_thread_mapping and before
        init_sequence_lengths / init_tile_bounds / init_q_row read q_start.
        """
        traits = self.traits
        # BLOCK_Q, not BLOCK_M: blocks are counted and started in query
        # POSITIONS. These coincide unless GQA packing is on, where a BLOCK_M-row
        # tile covers BLOCK_M // group positions -- using BLOCK_M here computes
        # too few blocks and strides q_start by the group size, so only
        # q_block_idx 0 lands correctly (found 2026-08-07: exactly the first 16
        # of 256 positions right at GQA 16:1). Must stay in sync with the same
        # expression in _init_dualwave_thread_mapping and the launcher's grid.y.
        num_q_blocks = (self.seq_len_v + traits.BLOCK_Q - 1) // traits.BLOCK_Q
        self.q_block_idx = num_q_blocks - fx.Index(1) - self.q_block_idx
        self.q_start = self.q_block_idx * traits.BLOCK_Q

    def init_lds(self, shared_storage):
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.kv.ptr))
        self.lds_kv_base_ptr = buffer_ops.create_llvm_ptr(self.lds_kv_base_idx, address_space=3)
        self.lds_vt_base_idx = fx.Index(fx.ptrtoint(lds.vt.ptr))
        self.lds_vt_base_ptr = buffer_ops.create_llvm_ptr(self.lds_vt_base_idx, address_space=3)
        self.lds_q_base_idx = fx.Index(fx.ptrtoint(lds.q.ptr))
        self.lds_q_base_ptr = buffer_ops.create_llvm_ptr(self.lds_q_base_idx, address_space=3)
        if const_expr(self.traits.PAGED):
            self.lds_bt_base_idx = fx.Index(fx.ptrtoint(lds.bt.ptr))
            self.lds_bt_base_ptr = buffer_ops.create_llvm_ptr(self.lds_bt_base_idx, address_space=3)
        else:
            self.lds_bt_base_idx = None
            self.lds_bt_base_ptr = None

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
            _cuq_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(self.CuSeqQ), fx.make_layout(1, 1))
            _cuk_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(self.CuSeqKv), fx.make_layout(1, 1))
            _cu_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            _cu_v1i32 = Vec.make_type(1, fx.Int32)

            self.q_tok_base = _cu_load(_cuq_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.q_tok_end = _cu_load(_cuq_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32)
            self.kv_tok_base = _cu_load(_cuk_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.kv_tok_end = _cu_load(_cuk_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32)
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
        if const_expr(traits.GQA_PACK_M):
            # Head is per M row here (stage_q_to_lds adds it), so the base
            # carries only the group's first head.
            _q_head_term = self.q_head_base * traits.HEAD_DIM
        else:
            _q_head_term = self.q_head_idx * traits.HEAD_DIM
        self.q_gmem_elem_offset = (
            self.q_tok_base + self.q_start
        ) * self.stride_q_n_v + _q_head_term
        # Dense fp8 carries the batch token base in the voffset, because its K/V
        # descriptors span the whole tensor. Under paging the descriptor is rebased
        # per page, so the token base is meaningless there -- keeping it would push
        # every batch above 0 past a one-page num_records and silently read zeros.
        self.kv_head_elem_offset = self.kv_head_idx * traits.HEAD_DIM
        if const_expr(traits.PAGED):
            self.kv_gmem_elem_offset = self.kv_head_elem_offset
        else:
            self.kv_gmem_elem_offset = self.kv_tok_base * self.stride_kv_n_v + self.kv_head_elem_offset

    def init_descriptors(self):
        traits = self.traits
        eb = traits.ELEM_BYTES
        q_nrec_bytes = as_mlir_value(self.q_tok_end * self.stride_q_n_v * eb)
        kv_nrec_bytes = as_mlir_value(self.kv_tok_end * self.stride_kv_n_v * eb)
        o_nrec_bytes = as_mlir_value(self.q_tok_end * self.stride_q_n_v * traits.OUT_ELEM_BYTES)

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
            bt = fx.Tensor(fx.make_view(fx.recast_iter(i8_ptr_ty, it), fx.get_layout(bt)))
            return fx.logical_divide(bt, fx.make_layout(1, 1))

        self.q_div = _make_buf_div(self.Q, q_nrec_bytes)
        self.o_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(self.O, num_records_bytes=o_nrec_bytes), fx.make_layout(1, 1)
        )

        if const_expr(traits.PAGED):
            # No whole-tensor K/V view under paging: each tile builds its own
            # page-bounded descriptor. Left None so any missed call site faults
            # rather than reading the dense view at a page-relative offset.
            self.k_div = None
            self.v_div = None
            # Byte quantities throughout: the page view is i8-typed to match the
            # dense descriptors, so buffer offsets are unscaled and `page_elems`
            # is already a byte count at ELEM_BYTES=1. Upstream's BF16_BYTES here
            # would double the stride and land every page but the first too far.
            page_elems = fx.Index(traits.BLOCK_N) * self.stride_kv_n_v * fx.Index(eb)
            self.page_byte_stride = page_elems
            self.page_nrec_bytes = fx.Int64(self.page_byte_stride)
            self.page_layout = fx.make_layout(fx.Int32(page_elems), fx.Int32(1))
            self.buf_flags_i32 = fx.Int32(buffer_ops._get_buffer_flags())
            self.page_elem_ir = fx.Int8.ir_type
            # Raw tensor iters, as upstream's _kv_src_div does: the page view
            # rebases the pointer itself and sets its own num_records, so a
            # whole-tensor buffer descriptor would be built and discarded.
            self.k_iter = fx.get_iter(self.K)
            self.v_iter = fx.get_iter(self.V)
            self.block_table_stride_v = fx.Index(self.block_table_stride)
            self.bt_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(self.BlockTable), fx.make_layout(1, 1))
            self.bt_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            self.bt_v1i32 = Vec.make_type(1, fx.Int32)
        else:
            self.k_div = _make_buf_div(self.K, kv_nrec_bytes)
            self.v_div = _make_buf_div(self.V, kv_nrec_bytes)
            self.page_byte_stride = None
            self.page_nrec_bytes = None
            self.page_layout = None
            self.buf_flags_i32 = None
            self.page_elem_ir = None
            self.k_iter = None
            self.v_iter = None
            self.block_table_stride_v = None
            self.bt_div = None
            self.bt_atom = None
            self.bt_v1i32 = None

    def kv_page_div(self, which, page_id):
        """Per-tile page-bounded K or V descriptor. `which` is "k" or "v"."""
        base_iter = self.k_iter if which == "k" else self.v_iter
        return _make_page_view(
            base_iter,
            base_iter.type,
            fx.PointerType(base_iter.type).alignment,
            page_id,
            self.page_byte_stride,
            self.page_nrec_bytes,
            self.page_layout,
            self.page_elem_ir,
            self.buf_flags_i32,
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
        self.bf16_mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16))
        self.v_fp8_load64_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)

    def init_descale(self):
        def _load_scale_scalar(tensor):
            _div = fx.logical_divide(fx.rocdl.make_buffer_tensor(tensor), fx.make_layout(1, 1))
            _atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            _v = fly.copy_atom_call_ssa([Vec.make_type(1, fx.Float32)], _atom, fx.slice(_div, (None, fx.Int32(0))))
            return fx.Float32(Vec(_v, (1,), fx.Float32)[0])

        head_dim_f32 = fx.Float32(fx.Int32(self.head_dim_runtime))
        c_log2e_f = fx.Float32(_LOG2E)
        c_sm_scale_log2e = fx.Float32(
            arith.mulf(
                as_mlir_value(fmath.rsqrt(head_dim_f32, fastmath=self.fm_fast)),
                as_mlir_value(c_log2e_f),
                fastmath=self.fm_fast,
            )
        )
        _qd = _load_scale_scalar(self.QDescale)
        _kd = _load_scale_scalar(self.KDescale)
        self.vd_fp8 = _load_scale_scalar(self.VDescale)
        # fp8 feeds raw Q/K into the MFMA, so q/k descale * softmax scale multiplies
        # the fp32 logits after QK.
        self.c_logit_scale = fx.Float32(
            arith.mulf(
                as_mlir_value(c_sm_scale_log2e),
                as_mlir_value(arith.mulf(as_mlir_value(_qd), as_mlir_value(_kd), fastmath=self.fm_fast)),
                fastmath=self.fm_fast,
            )
        )

    def init_tile_bounds(self):
        traits = self.traits
        kv_tile_size = traits.BLOCK_N
        num_kv_tiles = (self.seqlen_kv_v + kv_tile_size - 1) // kv_tile_size
        # Retained for the paged block-table stage, which zero-fills past the
        # real tile count so out-of-range page ids are 0 rather than garbage.
        self.num_kv_tiles = num_kv_tiles
        if const_expr(traits.CAUSAL):
            # BLOCK_Q, not BLOCK_M: this is how far the block reaches in query
            # POSITIONS, which is what bounds the causal KV extent. Under GQA
            # packing a BLOCK_M-row tile spans only BLOCK_M // group positions,
            # and using BLOCK_M here over-runs the tile count by the group size.
            causal_end_i32 = fx.Int32(self.q_start + traits.BLOCK_Q) + self.delta_i32
            causal_end_i32 = fx.Int32((causal_end_i32 > fx.Int32(0)).select(causal_end_i32, fx.Int32(0)))
            causal_num_tiles = (fx.Index(causal_end_i32) + kv_tile_size - 1) // kv_tile_size
            max_num_tiles = fx.Index((causal_num_tiles < num_kv_tiles).select(causal_num_tiles, num_kv_tiles))
        else:
            max_num_tiles = num_kv_tiles
        # Pipeline needs an EVEN tile count >= 4; extra tiles read 0 (num_records) and are masked.
        max_num_tiles = ((max_num_tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
        max_num_tiles = fx.Index((max_num_tiles < fx.Index(4)).select(fx.Index(4), max_num_tiles))
        self.max_num_tiles = max_num_tiles
        if const_expr(traits.SPLITK):
            chunk = ((max_num_tiles + (traits.NUM_KV_SPLITS - 1)) // traits.NUM_KV_SPLITS + 1) // 2 * 2
            chunk = fx.Index((chunk < fx.Index(6)).select(fx.Index(6), chunk))
            split_t0 = self.split_idx * chunk
            split_t_end = split_t0 + chunk
            split_t_end = fx.Index((split_t_end < max_num_tiles).select(split_t_end, max_num_tiles))
            split_t_end = fx.Index((max_num_tiles - split_t_end < fx.Index(4)).select(max_num_tiles, split_t_end))
            self.split_nonempty = split_t0 + fx.Index(4) <= max_num_tiles
        else:
            split_t0 = 0
            split_t_end = max_num_tiles
            self.split_nonempty = None
        self.split_t0 = split_t0
        self.split_t_end = split_t_end

    def compute_active_guard(self):
        """Predicate for whether this workgroup's Q block is in range.

        The grid is sized ``ceil(max_seqlen_q / BLOCK_M)`` in y for every batch
        entry, because one launch has to cover the longest sequence. Under
        varlen that overshoots every shorter sequence, and those excess blocks
        have ``q_start`` past the end of their own sequence -- left to run they
        write into the next packed sequence's rows.

        Mirrors ``DualwaveKernelContext.compute_active_guard`` on the bf16 side
        (upstream ``flash_attn_utils.py:3503`` at v0.3.0), which the fp8 context
        never had: it computes ``seqlen_q_v`` but used it only for
        ``delta_i32``. Returns None when no guard is needed, matching bf16 so
        the dense path emits exactly the same code as before.
        """
        traits = self.traits
        if const_expr(traits.SPLITK):
            return self.split_nonempty
        if const_expr(traits.VARLEN):
            return self.q_start < self.seqlen_q_v
        return None

    def init_active_guard(self):
        self.active = self.compute_active_guard()

    def init_workspace_io(self):
        if const_expr(self.traits.SPLITK):
            self.ws_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(self.DebugCounts), fx.make_layout(1, 1))
            self.ws_store_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            self.ws_store_reg_32 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            self.ws_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)

    def ws_store_f32(self, f32_val, elem_index):
        pack = Vec.from_elements([fx.Float32(f32_val)], fx.Float32).bitcast(fx.Int32)
        fx.memref_store_vec(pack, self.ws_store_reg_32)
        fx.copy(self.ws_store_atom_32, self.ws_store_reg_32, fx.slice(self.ws_div, (None, fx.Int32(elem_index))))

    def ws_store_quad_i32(self, dwords, elem_index):
        pack = Vec.from_elements([fx.Int32(v) for v in dwords], fx.Int32)
        fx.memref_store_vec(pack, self.ws_store_reg_128)
        fx.copy(self.store_atom_128, self.ws_store_reg_128, fx.slice(self.ws_div, (None, fx.Int32(elem_index))))

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
        # HIPREC carries P/V as v8 bf16 regardless of the fp8 element dtype: pack
        # 8 f32 -> 4 cvt_pk_bf16 dwords. (The generic _bf16_trunc_pack_v8 branches on
        # DTYPE_STR and would take the fp16 path for fp8, so keep this bf16-only pack.)
        pairs = []
        for j in range_constexpr(4):
            pairs.append(rocdl.cvt_pk_bf16_f32(f32_vals[j * 2], f32_vals[j * 2 + 1]))
        return Vec.from_elements(pairs, fx.Int32).bitcast(fx.BFloat16).ir_value()

    def buffer_load_128(self, elem_index):
        return _buffer_load_128(elem_index, self.load_atom_128, self.q_div, self.v4i32_type)

    def buffer_load_lds_128(self, src_div, lds_byte_addr, src_elem, soffset_elems):
        _buffer_load_lds_128(
            src_div, lds_byte_addr, src_elem, soffset_elems, _dma_atom=self.dma_atom, _lds_ptr_ty=self.lds_ptr_ty
        )

    def buffer_store_128(self, pack_i32_vec, elem_index):
        _buffer_store_128(pack_i32_vec, elem_index, self.o_store_reg_128, self.store_atom_128, self.o_div)

    def global_idx_q(self, token_idx, col):
        # q_head_row is the per-M-row head under GQA_PACK_M and the scalar
        # workgroup head otherwise; q_head_idx is None in the packed mode
        # precisely so a missed call site faults here instead of silently
        # writing every head of the group to one head's rows.
        return (
            (self.q_tok_base + token_idx) * self.stride_q_n_v
            + self.ctx_head_row() * self.traits.HEAD_DIM
            + col
        )

    def ctx_head_row(self):
        ref = getattr(self, "ctx_ref", self)
        return ref.q_head_row

    def read_i32x8_lds(self, base_ptr, byte_row):
        halves = []
        for h in range_constexpr(2):
            p = buffer_ops.get_element_ptr(base_ptr, byte_offset=fx.Int32(byte_row + h * 16), elem_type=T.i8)
            halves.append(Vec(llvm.LoadOp(Vec.make_type(4, fx.Int32), p, alignment=16).result))
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
            if const_expr(traits.GQA_PACK_M):
                # M row -> (query position, head in group). Consecutive rows are
                # consecutive HEADS of one token, so they are HEAD_DIM apart in
                # gmem, not stride_q_n apart -- staging them as if contiguous
                # would load one token's data into all 16 head rows.
                g = fx.Index(traits.GQA_GROUP_SIZE)
                src_elem = (
                    self.q_gmem_elem_offset
                    + (row // g) * self.stride_q_n_v
                    + (row % g) * fx.Index(traits.HEAD_DIM)
                    + dchunk * fx.Index(16)
                )
            else:
                src_elem = self.q_gmem_elem_offset + row * self.stride_q_n_v + dchunk * fx.Index(16)
            lds_addr = self.lds_q_base_idx + c * fx.Index(16)
            self.buffer_load_lds_128(self.q_div, lds_addr, src_elem, 0)

    def load_all_wide(self, q_row_in_block):
        traits = self.traits
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            byte_row = q_row_in_block * fx.Index(traits.HEAD_DIM) + fx.Index(ws * 64) + d_base
            packs.append(self.read_i32x8_lds(self.lds_q_base_ptr, fx.Int32(byte_row)))
        return packs

class DualwaveFp8GemmHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _mfma_acc_fp8_wide(self, a_i32x8, b_i32x8, c_v16):
        # Wide fp8 QK: mfma_scale (32x32x64) with unit E8M0 scales, i32x8 operands.
        return rocdl.mfma_scale_f32_32x32x64_f8f6f4(
            self.v16f32_type,
            as_mlir_value(a_i32x8),
            as_mlir_value(b_i32x8),
            as_mlir_value(c_v16),
            0,
            0,
            0,
            as_mlir_value(fx.Int32(0x7F7F7F7F)),
            0,
            as_mlir_value(fx.Int32(0x7F7F7F7F)),
        ).result

    def _mfma_acc_bf16(self, a_v8, b_v8, c_v16):
        return fly.mma_atom_call_ssa([self.v16f32_type], self.bf16_mma_atom, a_v8, b_v8, c_v16)

    def _v8bf16_to_f32(self, v8):
        f32 = Vec(llvm.FPExtOp(Vec.make_type(8, fx.Float32), as_mlir_value(v8)).result, (8,), fx.Float32)
        return [f32[i] for i in range_constexpr(8)]

    def _pack_fp8_i32x8(self, f32_vals):
        c0 = as_mlir_value(fx.Int32(0))
        words = []
        for g in range_constexpr(8):
            base = g * 4
            w = rocdl.cvt_pk_fp8_f32(T.i32, as_mlir_value(f32_vals[base]), as_mlir_value(f32_vals[base + 1]), c0, 0)
            w = rocdl.cvt_pk_fp8_f32(T.i32, as_mlir_value(f32_vals[base + 2]), as_mlir_value(f32_vals[base + 3]), w, 1)
            words.append(fx.Int32(w))
        return Vec.from_elements(words, fx.Int32).ir_value()

    def _p_to_fp8_i32x8(self, v_p):
        p_lo, p_hi = v_p
        f32 = []
        for pk in (p_lo[0], p_lo[1], p_hi[0], p_hi[1]):
            f32 += self._v8bf16_to_f32(pk)
        return self._pack_fp8_i32x8(f32)

    def _v_concat_i32x8(self, v_v, dc):
        words = []
        for ks in range_constexpr(4):
            v2 = Vec(llvm.bitcast(self.v2i32_type, as_mlir_value(v_v[ks][dc])), (2,), fx.Int32)
            words.append(fx.Int32(v2[0]))
            words.append(fx.Int32(v2[1]))
        return Vec.from_elements(words, fx.Int32).ir_value()

    def _v_to_fp8_i32x8(self, v_v, dc):
        f32 = []
        for step in range_constexpr(4):
            f32 += self._v8bf16_to_f32(v_v[step][dc])
        return self._pack_fp8_i32x8(f32)

    def _pv_fp8(self, v_p, v_v, v_o):
        p_fp8 = self._p_to_fp8_i32x8(v_p)
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_op = self._v_concat_i32x8(v_v, dc)
            v_o[dc] = self._mfma_acc_fp8_wide(v_op, p_fp8, v_o[dc])
        return v_o

    def _pv_step_fp8(self, step, v_p, v_v, v_o):
        if const_expr(step == 0):
            self._pv_p_fp8_cache = self._p_to_fp8_i32x8(v_p)
        v_op = self._v_concat_i32x8(v_v, step)
        v_o[step] = self._mfma_acc_fp8_wide(v_op, self._pv_p_fp8_cache, v_o[step])
        return v_o

    def _load_q_wide_lds(self):
        traits = self.traits
        q_row_in_block = self.ctx_ref.q_row_in_block
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            byte_row = q_row_in_block * fx.Index(traits.HEAD_DIM) + fx.Index(ws * 64) + d_base
            packs.append(self.read_i32x8_lds(self.lds_q_base_ptr, fx.Int32(byte_row)))
        return packs

    def load_q_wide(self):
        return self._load_q_wide_lds()

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
        if const_expr(traits.QREG):
            n_ds = const_expr(traits.HEAD_DIM // 64 * 4)
            n_mfma = const_expr(traits.HEAD_DIM // 64 * 2)
            rocdl.sched_group_barrier(traits.SCHED_DS_READ_MASK, n_ds // 2, 12)
            rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, 12)
            rocdl.sched_group_barrier(traits.SCHED_DS_READ_MASK, n_ds // 2, 12)
            rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, n_mfma - 1, 12)
        return (v_s_lo, v_s_hi)

    def pv_step_k(self, step, v_p, v_v, v_o):
        if const_expr(self.traits.FP8_PV):
            return self._pv_step_fp8(step, v_p, v_v, v_o)
        # HIPREC PV: P and V are both v8 bf16, accumulated by a bf16 MMA.
        v_p_lo, v_p_hi = v_p
        v_pk = v_v[step]
        if const_expr(step < 2):
            p_pk = v_p_lo[step]
        else:
            p_pk = v_p_hi[step - 2]
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_o[dc] = self._mfma_acc_bf16(v_pk[dc], p_pk, v_o[dc])
        return v_o

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
        if const_expr(self.traits.PV_SPREAD):
            # The D_CHUNKS MFMAs above are mutually independent -- each
            # accumulates a different v_o[dc] over a different D chunk -- and
            # their operand prep is pure bitcasts the compiler hoists, so they
            # emit back to back with nothing between. The matrix pipe is
            # asynchronous: issuing into a full queue blocks rather than
            # retiring faster. Measured across three distinct compiled kernels
            # (causal paged+varlen at M=1384 and M=8192, non-causal dense at
            # M=2048, all identical): the first MFMA of a run stalls 0% and
            # every one after it stalls 94%, for 90% stalled overall.
            #
            # Ask the scheduler to alternate one MFMA with surrounding VALU
            # instead, which is the cadence the hand-written ASM kernel keeps
            # (84 of its 127 MFMA gaps are exactly 6 instructions, none are
            # back to back). sched_group_barrier is a scheduling hint, not
            # control flow, so the loop stays one basic block -- an scf.if here
            # would split it into five and break the QK/softmax/PV interleave
            # the rest of the schedule depends on.
            #
            # Not shape-specific: D_CHUNKS is HEAD_DIM // D_CHUNK = 4 for every
            # configuration the builder accepts (D=128 is enforced), so the
            # clustering and this fix are structurally identical everywhere.
            for _ in range_constexpr(self.traits.D_CHUNKS):
                rocdl.sched_group_barrier(self.traits.SCHED_MFMA_MASK, 1, 13)
                rocdl.sched_group_barrier(self.traits.SCHED_VALU_MASK, 4, 13)
        return v_o

    def pv(self, v_p, v_v, v_o):
        if const_expr(self.traits.FP8_PV_DIRECT):
            return self._pv_fp8_direct(v_p, v_v, v_o)
        if const_expr(self.traits.FP8_PV):
            return self._pv_fp8(v_p, v_v, v_o)
        for step in range_constexpr(4):
            v_o = self.pv_step_k(step, v_p, v_v, v_o)
        return v_o

class DualwaveFp8PageIdLoader(DualwaveFp8KernelContext):
    """Block-table staging and page-id lookup for the paged fp8 path.

    Vendored from upstream ``DualwavePageIdLoader`` (v0.3.0) with the base class
    swapped to the fp8 context; the arithmetic is unchanged, and it was only ever
    coupled to bf16 by inheritance -- every field it reads is named explicitly.

    Unlike bf16, the fp8 kernel body does not thread page ids through its pipeline.
    bf16 splits issue (``load_page_id_lds``) from finish (``finish_page_id``) across
    ~30 call sites to keep the ``s_waitcnt lgkmcnt(0)`` drain away from in-flight
    K/V ``ds_read``s. The fp8 BN128 ring issues four tile loads per iteration, so
    that pattern would drain four times per iteration; instead the KV loader looks
    the page id up from ``tile_start`` at its own call site. Correctness first --
    the scheduling cost of the drain is measured, not assumed.
    """

    def __init__(self, ctx):
        super().__init__(ctx)

    def load_block_table_to_lds(self):
        traits = self.traits
        tid = self.tid
        split_t0 = self.split_t0
        split_t_end = self.split_t_end
        num_kv_tiles = self.num_kv_tiles
        batch_idx = self.batch_idx
        block_table_stride_v = self.block_table_stride_v
        lds_bt_base_ptr = self.lds_bt_base_ptr
        bt_div = self.bt_div
        bt_atom = self.bt_atom
        bt_v1i32 = self.bt_v1i32

        @flyc.jit
        def _load_block_table_to_lds():
            segment_tiles = split_t_end - split_t0
            for pass_id in range_constexpr(traits.PAGED_BT_LDS_SIZE // traits.BLOCK_SIZE):
                local_tile = tid + fx.Index(pass_id * traits.BLOCK_SIZE)
                if local_tile < segment_tiles:
                    tile_idx = split_t0 + local_tile
                    byte_off = as_mlir_value(fx.Int32(local_tile * fx.Index(4)))
                    dst = buffer_ops.get_element_ptr(lds_bt_base_ptr, byte_offset=byte_off, elem_type=T.i8)
                    llvm.StoreOp(as_mlir_value(fx.Int32(0)), dst)
                    if tile_idx < num_kv_tiles:
                        row_idx = batch_idx * block_table_stride_v + tile_idx
                        v = fly.copy_atom_call_ssa([bt_v1i32], bt_atom, fx.slice(bt_div, (None, fx.Int32(row_idx))))
                        page_id_i32 = as_mlir_value(fx.Int32(Vec(v, (1,), fx.Int32)[0]))
                        llvm.StoreOp(page_id_i32, dst)

        _load_block_table_to_lds()

    def load_page_id_lds(self, tile_idx):
        src = buffer_ops.get_element_ptr(
            self.lds_bt_base_ptr,
            byte_offset=as_mlir_value(_paged_bt_byte_offset(tile_idx, split_t0=self.split_t0)),
            elem_type=T.i8,
        )
        return llvm.LoadOp(T.i32, src).result

    def finish_page_id(self, v):
        # readfirstlane: the page id lands in an SGPR buffer-descriptor base, so it
        # must be wave-uniform. It already is -- every lane reads the same LDS slot.
        rocdl.s_waitcnt(self.traits.LGKMCNT_0_ONLY)
        v = rocdl.readfirstlane(T.i32, v)
        return fx.Index(fx.Int32(v))

    def page_id_for_tile(self, tile_idx):
        # Clamp into the staged window. The BN128 ring prefetches up to 5 tiles
        # past split_t_end, and those tiles' LDS slots hold no valid page id.
        # Dense absorbs the same overrun via num_records; paged cannot, because
        # the page id picks the descriptor base before any bound applies -- an
        # unstaged slot sends the DMA outside the page pool entirely. Clamping
        # re-reads the last live page instead; the data is wrong but in-bounds,
        # and the epilogue masks these tiles out regardless.
        last = self.split_t_end - fx.Index(1)
        safe = fx.Index((tile_idx < last).select(tile_idx, last))
        return self.finish_page_id(self.load_page_id_lds(safe))

class DualwaveFp8KvGmemToLdsLoader(DualwaveFp8PageIdLoader):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _kv_src(self, which, tile_start):
        """Resolve (descriptor, tile_soffset) for a KV tile.

        Dense keeps one whole-tensor descriptor and moves with soffset; paged
        rebases per page and the tile term vanishes -- the page view already
        starts at the tile. Callers that fold the tile into their voffset must
        use ``tile_voffset`` instead of adding it themselves.
        """
        if const_expr(self.traits.PAGED):
            page_id = self.page_id_for_tile(tile_start // fx.Index(self.traits.BLOCK_N))
            return self.kv_page_div(which, page_id), 0
        div = self.k_div if which == "k" else self.v_div
        return div, tile_start * self.stride_kv_n_v

    def tile_voffset(self, tile_start):
        """Tile term for the V paths that carry it in the voffset, not the soffset."""
        if const_expr(self.traits.PAGED):
            return fx.Index(0)
        return tile_start * self.stride_kv_n_v

    def load_k(self, tile_start, buf_id):
        traits = self.traits
        eb = traits.ELEM_BYTES
        k_div, k_soffset = self._kv_src("k", tile_start)
        k_lds_byte_base = self.lds_kv_base_idx + self.k_buf_base(buf_id) * eb
        for d in range_constexpr(self.NUM_DMA_K):
            lds_addr = (
                k_lds_byte_base
                + self.wave_id_uni * (traits.SMEM_K_LINE_STRIDE * eb)
                + (d * traits.SMEM_N_RPT * traits.SMEM_K_LINE_STRIDE * eb)
            )
            n_in_tile = self.n_in_warp * traits.NUM_WAVES + self.wave_id
            global_d = self.d_bucket * traits.VEC_KV + (d * traits.D_128B_SIZE)
            src_elem = self.kv_gmem_elem_offset + n_in_tile * self.stride_kv_n_v + global_d
            self.buffer_load_lds_128(k_div, lds_addr, src_elem, k_soffset)

    def load_v(self, tile_start, buf_id):
        if const_expr(self.traits.FP8_PV):
            self._stage_v_fp8_block(tile_start, buf_id)
        else:
            self._stage_vt_dequant_fp8(tile_start, buf_id)

    def zero_v_fp8_lds(self):
        traits = self.traits
        v_tile_bytes = (traits.BLOCK_N // 8) * (traits.HEAD_DIM // 16) * 128
        total = 2 * v_tile_bytes
        aligned_base = ((self.lds_vt_base_idx + fx.Index(127)) // fx.Index(128)) * fx.Index(128)
        zero = Vec.from_elements([fx.Int32(0) for _ in range_constexpr(4)], fx.Int32)
        per = total // (traits.BLOCK_SIZE)  # bytes per thread
        for i in range_constexpr(per // 16):
            off = aligned_base + self.tid * fx.Index(per) + fx.Index(i * 16)
            p = buffer_ops.create_llvm_ptr(off, address_space=3)
            llvm.StoreOp(as_mlir_value(zero), p, alignment=16)

    def _stage_v_fp8_block(self, tile_start, buf_id):
        traits = self.traits
        if const_expr(traits.VDMA):
            return self._stage_v_fp8_block_dma(tile_start, buf_id)
        v_tile_bytes = (traits.BLOCK_N // 8) * (traits.HEAD_DIM // 16) * 128
        buf_off = buf_id * v_tile_bytes
        n = self.wave_id * fx.Index(8) + self.lane // fx.Index(8)
        d_block = self.lane % fx.Index(8)
        v_div, _ = self._kv_src("v", tile_start)
        src_elem = (
            self.kv_gmem_elem_offset
            + n * self.stride_kv_n_v
            + d_block * fx.Index(16)
            + self.tile_voffset(tile_start)
        )
        v16 = fly.copy_atom_call_ssa(
            [Vec.make_type(4, fx.Int32)], self.load_atom_128, fx.slice(v_div, (None, fx.Int32(src_elem)))
        )
        n_i = fx.Int32(n)
        w16 = n_i % fx.Int32(16)
        c_add = (w16 >= fx.Int32(4)) & (w16 < fx.Int32(8))
        c_sub = (w16 >= fx.Int32(8)) & (w16 < fx.Int32(12))
        dest_n = n_i + c_add.select(fx.Int32(4), fx.Int32(0)) - c_sub.select(fx.Int32(4), fx.Int32(0))
        dest_wave = fx.Index(dest_n // fx.Int32(8))
        dest_m = fx.Index(dest_n % fx.Int32(8))
        block = dest_wave * fx.Index(8) + self.lane % fx.Index(8)
        aligned_base = ((self.lds_vt_base_idx + fx.Index(127)) // fx.Index(128)) * fx.Index(128)
        byte_off = aligned_base + fx.Index(buf_off) + block * fx.Index(128) + fx.Index(16) * dest_m
        lds_ptr = buffer_ops.create_llvm_ptr(byte_off, address_space=3)
        llvm.StoreOp(as_mlir_value(Vec(v16)), lds_ptr, alignment=16)

    def _stage_v_fp8_block_dma(self, tile_start, buf_id):
        traits = self.traits
        v_tile_bytes = (traits.BLOCK_N // 8) * (traits.HEAD_DIM // 16) * 128
        buf_off = buf_id * v_tile_bytes
        aligned_base = ((self.lds_vt_base_idx + fx.Index(127)) // fx.Index(128)) * fx.Index(128)
        lds_addr = aligned_base + fx.Index(buf_off) + self.wave_id_uni * fx.Index(1024)
        dest_n = fx.Int32(self.wave_id * fx.Index(8) + self.lane % fx.Index(8))
        w16 = dest_n % fx.Int32(16)
        c_add = (w16 >= fx.Int32(4)) & (w16 < fx.Int32(8))
        c_sub = (w16 >= fx.Int32(8)) & (w16 < fx.Int32(12))
        n = dest_n + c_add.select(fx.Int32(4), fx.Int32(0)) - c_sub.select(fx.Int32(4), fx.Int32(0))
        d_block = self.lane // fx.Index(8)
        v_div, v_soffset = self._kv_src("v", tile_start)
        src_elem = self.kv_gmem_elem_offset + fx.Index(n) * self.stride_kv_n_v + d_block * fx.Index(16)
        self.buffer_load_lds_128(v_div, lds_addr, src_elem, v_soffset)

    def _stage_vt_dequant_fp8(self, tile_start, buf_id):
        # Dequantize fp8 V into the exact bf16 V staging positions. The two d-iters
        # load 8 fp8 at D offsets 64 apart; a contiguous 16B load would gather wrong.
        traits = self.traits
        vt_buf = buf_id * traits.VT_BF16_ELEMS
        n_in_tile = self.n_in_warp * traits.NUM_WAVES + self.wave_id
        v_div, _ = self._kv_src("v", tile_start)
        tile_off = self.tile_voffset(tile_start)
        for d in range_constexpr(traits.SDRPT_BF):
            global_d = self.d_bucket * traits.VEC_BF + (d * traits.D128_BF)
            src_elem = self.kv_gmem_elem_offset + n_in_tile * self.stride_kv_n_v + global_d + tile_off
            v_i32x2 = fly.copy_atom_call_ssa(
                [self.v2i32_type], self.v_fp8_load64_atom, fx.slice(v_div, (None, fx.Int32(src_elem)))
            )
            v_words = Vec(v_i32x2, (2,), fx.Int32)
            bf = []
            for w in range_constexpr(2):
                word = as_mlir_value(fx.Int32(v_words[w]))
                lo2 = Vec(rocdl.cvt_pk_f32_fp8(Vec.make_type(2, fx.Float32), word, False), (2,), fx.Float32)
                hi2 = Vec(rocdl.cvt_pk_f32_fp8(Vec.make_type(2, fx.Float32), word, True), (2,), fx.Float32)
                for e in (lo2[0], lo2[1], hi2[0], hi2[1]):
                    bf.append(fx.Float32(e) * self.vd_fp8)
            v8bf = self.bf16_trunc_pack_v8(bf)
            byte_off = (
                vt_buf
                + self.wave_id_uni * traits.VLS_BF
                + d * traits.SNRPT_BF * traits.VLS_BF
                + self.lane * traits.VEC_BF
            ) * traits.EB_BF
            lds_ptr = buffer_ops.get_element_ptr(self.lds_vt_base_ptr, byte_offset=byte_off, elem_type=T.i8)
            llvm.StoreOp(as_mlir_value(v8bf), lds_ptr, alignment=16)

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

        def _read_strip(key):
            row = (key % 8) * traits.SMEM_K_LINE_STRIDE + (key // 8) * traits.D_128B_SIZE
            return [
                self.read_i32x8_lds(self.lds_kv_base_ptr, k_base + row + ws * 64 + d_base)
                for ws in range_constexpr(traits.HEAD_DIM // 64)
            ]

        return (_read_strip(n_lo), _read_strip(n_hi))

    def load_v(self, buf_id):
        if const_expr(self.traits.FP8_PV):
            return self._load_v_fp8_block(buf_id)
        # Read all V packs from the bf16 vt scratch for buffer `buf_id`.
        traits = self.traits
        urv = (
            self.lane_div_32 * traits.URV_GRPK_BF
            + ((self.lane % 16) // 4) * traits.URV_LANE_HI_BF
            + ((self.lane // 16) % 2) * traits.URV_GRP_N_BF
            + (self.lane % 4) * traits.URV_LANE_LO_BF
        )
        packs = [[None] * traits.D_CHUNKS for _ in range(4)]
        for dc in range_constexpr(traits.D_CHUNKS):
            dc_off = (dc // 2) * traits.URV_DC_AXIS0_BF + (dc % 2) * traits.URV_DC_AXIS1_BF
            for k_substep in range_constexpr(4):
                imm_lo = (k_substep * traits.URV_STEPK_BF + dc_off) * traits.EB_BF
                byte0 = (urv + buf_id * traits.VT_BF16_ELEMS) * traits.EB_BF + self.lds_vt_base_idx
                a = _ds_read_tr16_b64_imm(self.v4bf16_type, fx.Int32(byte0), imm_lo)
                b = _ds_read_tr16_b64_imm(self.v4bf16_type, fx.Int32(byte0), imm_lo + traits.URV_I5_BF * traits.EB_BF)
                packs[k_substep][dc] = Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return packs

    def _load_v_fp8_block(self, buf_id):
        traits = self.traits
        v_tile_bytes = (traits.BLOCK_N // 8) * (traits.HEAD_DIM // 16) * 128
        buf_off = buf_id * v_tile_bytes
        nbands = traits.HEAD_DIM // 16  # 8
        rh = (self.lane % fx.Index(32)) // fx.Index(16)
        l16 = self.lane % fx.Index(16)
        lane_hi = self.lane // fx.Index(32)
        aligned_base = ((self.lds_vt_base_idx + fx.Index(127)) // fx.Index(128)) * fx.Index(128)
        base = fx.Int32(
            aligned_base + buf_off + rh * fx.Index(128) + l16 * fx.Index(8) + lane_hi * fx.Index(nbands * 128)
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

    def _attn_mask_vec2_imm(self, rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32):
        return _attn_mask_vec2_imm(rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32)

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
        rel_lo_i32 = fx.Int32(self.ctx_ref.q_row_i32 + self.delta_i32 - kv_start_i32 - lane_off_i32)
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
        return _fmax(a, b, self.fm_fast)

    def floor_masked_max(self, row_max):
        return _fmax(row_max, self.c_neg_floor, self.fm_fast)

    def sub_m(self, v_s, row_max):
        return _scale_sub_score_pair(v_s, row_max, self.c_logit_scale, self.c_zero_f, self.fm_fast)

    def exp2(self, v_s, start, length):
        return _exp2_score_slice(v_s, start, length)

    def tile_sum(self, v_p):
        return _score_pair_sum(v_p, self.c_zero_f, self.fm_fast)

    def reduce_sum(self, l_row, v_p):
        return _fadd(l_row, self.tile_sum(v_p), self.fm_fast)

    def cast_p(self, v_p):
        # Pack the finished softmax probabilities into v8 bf16 P packs for PV.
        return _pack_p_v8_slices(self.traits, v_p, self.bf16_trunc_pack_v8)

    def scale_o(self, v_o, scale_scalar):
        _scale_o_accs(v_o, scale_scalar, self.traits, self.fm_fast)

    def scale_v_p(self, v_p, scale_scalar):
        # P is v8 bf16 (HIPREC): ext to f32, scale, repack bf16.
        p_lo, p_hi = v_p
        out_lo, out_hi = [], []
        for src, dst in ((p_lo, out_lo), (p_hi, out_hi)):
            for pk in src:
                f32 = Vec(llvm.FPExtOp(Vec.make_type(8, fx.Float32), as_mlir_value(pk)).result, (8,), fx.Float32)
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
        row_max = _fmax(m_row, m_tile_max, self.fm_fast)
        diff_scaled = _fmul(_fsub(m_row, row_max, self.fm_fast), self.c_logit_scale, self.fm_fast)
        rescale = rocdl.exp2(T.f32, as_mlir_value(diff_scaled))
        return row_max, rescale

    def apply_l_rescale(self, l_row, rescale):
        return _fmul(l_row, rescale, self.fm_fast)

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

    def lazy_rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        @flyc.jit
        def _run(v_o, m_row, l_row, m_tile_max, v_p):
            m_diff = _fsub(m_tile_max, m_row, self.fm_fast)
            m_diff_scaled = _fmul(m_diff, self.c_logit_scale, self.fm_fast)
            below = fx.Float32(m_diff_scaled) <= self.c_eight_f
            ballot = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = arith.cmpi(arith.CmpIPredicate.eq, as_mlir_value(ballot), _read_exec_i64())
            all_below = llvm.intr_expect(all_below, arith.constant(1, type=ir.IntegerType.get_signless(1)))

            o0, o1, o2, o3 = (
                as_mlir_value(v_o[0]),
                as_mlir_value(v_o[1]),
                as_mlir_value(v_o[2]),
                as_mlir_value(v_o[3]),
            )
            m_out = as_mlir_value(m_row)
            l_out = as_mlir_value(l_row)
            vp_out = self.v_p_to_vec32(v_p)
            if fx.Boolean(all_below):
                pass
            else:
                corr = rocdl.exp2(T.f32, as_mlir_value(_fsub(self.c_zero_f, m_diff_scaled, self.fm_fast)))
                scaled_accs = list(v_o)
                self.scale_o(scaled_accs, corr)
                o0, o1, o2, o3 = (
                    as_mlir_value(scaled_accs[0]),
                    as_mlir_value(scaled_accs[1]),
                    as_mlir_value(scaled_accs[2]),
                    as_mlir_value(scaled_accs[3]),
                )
                vp_out = self.v_p_to_vec32(self.scale_v_p(v_p, corr))
                l_out = as_mlir_value(_fmul(l_row, corr, self.fm_fast))
                m_out = self.anchor_scalar_f32(m_tile_max)
            return ([o0, o1, o2, o3], m_out, l_out, self.v_vec32_to_p(vp_out))

        return _run(v_o, m_row, l_row, m_tile_max, v_p)

    def lazy_correct_o(self, v_o, m_row, l_row, m_tile_max):
        @flyc.jit
        def _run(v_o, m_row, l_row, m_tile_max):
            m_diff = _fsub(m_tile_max, m_row, self.fm_fast)
            m_diff_scaled = _fmul(m_diff, self.c_logit_scale, self.fm_fast)
            below = fx.Float32(m_diff_scaled) <= self.c_eight_f
            ballot = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = arith.cmpi(arith.CmpIPredicate.eq, as_mlir_value(ballot), _read_exec_i64())
            all_below = llvm.intr_expect(all_below, arith.constant(1, type=ir.IntegerType.get_signless(1)))

            o0, o1, o2, o3 = (
                as_mlir_value(v_o[0]),
                as_mlir_value(v_o[1]),
                as_mlir_value(v_o[2]),
                as_mlir_value(v_o[3]),
            )
            m_out = as_mlir_value(m_row)
            l_out = as_mlir_value(l_row)
            if fx.Boolean(all_below):
                pass
            else:
                corr = rocdl.exp2(T.f32, as_mlir_value(_fsub(self.c_zero_f, m_diff_scaled, self.fm_fast)))
                scaled_accs = list(v_o)
                self.scale_o(scaled_accs, corr)
                o0, o1, o2, o3 = (
                    as_mlir_value(scaled_accs[0]),
                    as_mlir_value(scaled_accs[1]),
                    as_mlir_value(scaled_accs[2]),
                    as_mlir_value(scaled_accs[3]),
                )
                l_out = as_mlir_value(_fmul(l_row, corr, self.fm_fast))
                m_out = self.anchor_scalar_f32(m_tile_max)
            return ([o0, o1, o2, o3], m_out, l_out)

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
        swapped = rocdl.permlane32_swap(pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False)
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
        return Vec.from_elements([fx.Int32(w) for w in self._packed_o_128_dwords(v_o, dc, g)], fx.Int32)

    def store_final_o(self, v_o, q_row):
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                o_pack = self._packed_o_128_vec(v_o, dc, g)
                d_col = (dc * self.traits.D_CHUNK) + (2 * g + self.lane_div_32) * 8
                o_global = self.global_idx_q(q_row, d_col)
                self.buffer_store_128(o_pack, o_global)

    def store_splitk_partial_o(self, v_o, m_row, l_row, q_row):
        split_z = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
        o_part_row_base = ((split_z * self.traits.NUM_HEADS_Q + self.q_head_idx) * self.seq_len_v + q_row) * (
            self.traits.HEAD_DIM // 2
        )
        grid_z = fx.Index(gpu.grid_dim.z)
        mrow_base = grid_z * self.traits.NUM_HEADS_Q * self.seq_len_v * (self.traits.HEAD_DIM // 2)
        lrow_base = mrow_base + grid_z * self.traits.NUM_HEADS_Q * self.seq_len_v
        ml_row_idx = (split_z * self.traits.NUM_HEADS_Q + self.q_head_idx) * self.seq_len_v + q_row

        @flyc.jit
        def _store_splitk_partial_if_qrow():
            if q_row < self.seq_len_v:
                for dc in range_constexpr(self.traits.D_CHUNKS):
                    for g in range_constexpr(2):
                        dw_col = dc * (self.traits.D_CHUNK // 2) + (2 * g + self.lane_div_32) * 4
                        self.ws_store_quad_i32(self._packed_o_128_dwords(v_o, dc, g), o_part_row_base + dw_col)
                if self.lane < fx.Index(32):
                    self.ws_store_f32(self.splitk_m_out(m_row), mrow_base + ml_row_idx)
                    self.ws_store_f32(l_row, lrow_base + ml_row_idx)

        _store_splitk_partial_if_qrow()

    def splitk_m_out(self, m_row):
        """Convert m_row into the units the combine pass expects.

        Inside the kernel c_logit_scale -- which folds the q/k descales with
        sm_scale*log2e -- is only ever applied to a difference (s - m), never to
        m itself, so m_row is carried in raw un-descaled logit units and the
        scale cancels among its consumers. The combine is a separate kernel with
        no descale tensors: it computes exp2(m_i - m_max) directly, so it needs m
        pre-scaled. Unscaled, the fp8 gaps are ~1e5 and every exp2 but the
        largest split's flushes to zero, silently dropping those partials.

        bf16 has no descale, so its c_logit_scale is just sm_scale*log2e and the
        two conventions coincide -- which is why this only bites fp8.
        """
        if const_expr(not self.traits.FP8_PV):
            return m_row
        return fx.Float32(_fmul(as_mlir_value(m_row), as_mlir_value(self.c_logit_scale), self.fm_fast))

    def store_empty_split(self):
        @flyc.jit
        def _store_empty_split():
            if self.max_num_tiles < self.split_t0 + fx.Index(4):
                q_row_e = self.q_start + self.wave_q_offset + self.lane_mod_32
                split_z_e = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
                o_row_base_e = ((split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx) * self.seq_len_v + q_row_e) * (
                    self.traits.HEAD_DIM // 2
                )
                grid_z_e = fx.Index(gpu.grid_dim.z)
                mrow_base_e = grid_z_e * self.traits.NUM_HEADS_Q * self.seq_len_v * (self.traits.HEAD_DIM // 2)
                lrow_base_e = mrow_base_e + grid_z_e * self.traits.NUM_HEADS_Q * self.seq_len_v
                ml_row_e = (split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx) * self.seq_len_v + q_row_e
                if q_row_e < self.seq_len_v:
                    c_zero_i = fx.Int32(0)
                    for dc in range_constexpr(self.traits.D_CHUNKS):
                        for g in range_constexpr(2):
                            dw_col = dc * (self.traits.D_CHUNK // 2) + (2 * g + self.lane_div_32) * 4
                            self.ws_store_quad_i32([c_zero_i, c_zero_i, c_zero_i, c_zero_i], o_row_base_e + dw_col)
                    if self.lane < fx.Index(32):
                        self.ws_store_f32(fx.Float32(-1e30), mrow_base_e + ml_row_e)
                        self.ws_store_f32(self.c_zero_f, lrow_base_e + ml_row_e)

        _store_empty_split()

class DualwaveSplitKCombineContext:
    """Shared per-kernel state for the split-K combine pass."""

    def __init__(
        self,
        traits_or_ctx,
        O=None,  # noqa: E741
        WS=None,
        batch_size=None,
        seq_len=None,
        stride_q_n=None,
        LSE=None,
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
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stride_q_n = stride_q_n

    def init_types_and_constants(self):
        self.elem_dtype = dtype_to_elem_type(self.traits.DTYPE_STR)
        # Element type of the split-K partials in the workspace, and of the
        # final output. NOT the input dtype: the main kernel packs partials
        # with cvt_pk_bf16_f32 whatever it reads (see _o_pack_2dw), the row
        # stride is HEAD_DIM // 2 i32, and store_output scales its offset by 2.
        # For an fp8 kernel elem_dtype is one byte, so using it here unpacks 8
        # elements where 4 are meant and packs 4 into one i32 instead of two.
        self.part_dtype = fx.BFloat16 if self.traits.DTYPE_STR == "fp8" else self.elem_dtype
        self.fm_fast = fx.arith.FastMathFlags.fast
        self.c_zero_f = fx.Float32(0.0)
        self.c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)
        # LSE store folds the log2->ln conversion (m_max is sm_scale*log2e-scaled).
        self.c_ln2_f = fx.Float32(1.0 / _LOG2E)

    def init_runtime_indices(self):
        self.seq_len_v = fx.Index(self.seq_len)
        self.stride_q_n_v = fx.Index(self.stride_q_n)
        self.batch_size_v = fx.Index(self.batch_size)

    def init_thread_mapping(self, combine_rows_per_block, combine_lanes_per_row):
        traits = self.traits
        self.tid = fx.Index(gpu.thread_idx.x)
        self.blk = fx.Index(gpu.block_idx.x)
        self.row = self.blk * combine_rows_per_block + self.tid // combine_lanes_per_row
        self.col = (self.tid % combine_lanes_per_row) * 4
        heads_per_batch = self.seq_len_v * traits.NUM_HEADS_Q
        self.batch_idx = self.row // heads_per_batch
        rem = self.row % heads_per_batch
        self.q_head_idx = rem // self.seq_len_v
        self.seq_idx = rem % self.seq_len_v

    def init_workspace(self):
        traits = self.traits
        z_total = self.batch_size_v * traits.NUM_KV_SPLITS
        self.ws_opart_per_split_elems = fx.Index(traits.NUM_HEADS_Q) * self.seq_len_v * fx.Index(traits.HEAD_DIM // 2)
        self.ws_ml_per_split_elems = fx.Index(traits.NUM_HEADS_Q) * self.seq_len_v
        self.ws_opart_per_split_bytes = self.ws_opart_per_split_elems * fx.Index(4)
        self.ws_ml_per_split_bytes = self.ws_ml_per_split_elems * fx.Index(4)
        self.ws_mrow_abs_bytes = z_total * self.ws_opart_per_split_bytes
        self.ws_lrow_abs_bytes = self.ws_mrow_abs_bytes + z_total * self.ws_ml_per_split_bytes
        self.local_ml_idx = self.q_head_idx * self.seq_len_v + self.seq_idx
        self.local_o_base = (self.q_head_idx * self.seq_len_v + self.seq_idx) * fx.Index(traits.HEAD_DIM // 2)
        self.ws_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(self.WS)))

    def init_descriptors(self):
        per_batch_elems = self.seq_len_v * self.stride_q_n_v
        batch_byte_off = self.batch_idx * per_batch_elems * fx.Index(2)
        self.o_rsrc = buffer_ops.create_buffer_resource_from_addr(
            as_mlir_value(fx.Int64(fx.ptrtoint(fx.get_iter(self.O))) + fx.Int64(batch_byte_off)),
            num_records_bytes=as_mlir_value(fx.Int64(per_batch_elems * fx.Index(2))),
        )
        self.load_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)

    def workspace_resource(self, byte_offset, nrec_bytes):
        return _make_ws_rsrc(self.ws_base_i64, byte_offset, nrec_bytes)

    def split_z(self, split_i):
        return self.batch_idx * self.traits.NUM_KV_SPLITS + split_i

    def opart_resource(self, split_z):
        return self.workspace_resource(split_z * self.ws_opart_per_split_bytes, self.ws_opart_per_split_bytes)

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
            m_max = _fmax(m_max, m_s[i + 1], self.fm_fast)
        return m_max

    def init_accumulators(self):
        return as_mlir_value(self.c_zero_v4f32), as_mlir_value(self.c_zero_f)

    def accumulate_split(self, acc, den, split_i, m_i, l_i, m_max):
        orsrc_i = self.opart_resource(self.split_z(split_i))
        local_o_idx_i = self.local_o_base + self.col // 2

        @flyc.jit
        def _accum_split(acc, den):
            if fx.Float32(l_i) > fx.Float32(0.0):
                w = rocdl.exp2(T.f32, as_mlir_value(_fsub(m_i, m_max, self.fm_fast)))
                wl = _fmul(w, l_i, self.fm_fast)
                den = _fadd(den, wl, self.fm_fast)
                o2_raw = buffer_ops.buffer_load(
                    orsrc_i,
                    as_mlir_value(fx.Int32(local_o_idx_i)),
                    vec_width=2,
                    dtype=T.i32,
                )
                o2_i32 = ir.Value(o2_raw)
                o4 = Vec(o2_i32, (2,), fx.Int32).bitcast(self.part_dtype).to(fx.Float32)
                w4 = Vec.from_elements([fx.Float32(wl)], fx.Float32).broadcast_to(4)
                acc = _fadd(acc, _fmul(w4, o4, self.fm_fast), self.fm_fast)
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
        out4 = Vec(_fmul(acc, inv4, self.fm_fast), (4,), fx.Float32)
        # part_dtype, not elem_dtype -- an fp8 kernel still writes bf16 out.
        if const_expr(self.part_dtype is fx.BFloat16):
            lo = rocdl.cvt_pk_bf16_f32(out4[0], out4[1])
            hi = rocdl.cvt_pk_bf16_f32(out4[2], out4[3])
        else:
            o_f16 = []
            for i in range_constexpr(4):
                o_f16.append(fx.Float32(out4[i]).to(self.part_dtype))
            pack = Vec.from_elements(o_f16, self.part_dtype).bitcast(fx.Int32)
            lo, hi = as_mlir_value(pack[0]), as_mlir_value(pack[1])
        return Vec.from_elements([fx.Int32(lo), fx.Int32(hi)], fx.Int32)

    def store_lse(self, m_max, den):
        # Combined LSE = m_max * ln2 + ln(den); den = sum_s 2^(m_s - m_max) * l_s
        # completes the natural-log, scale-folded LSE. One lane (col == 0) writes.
        lse_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(self.LSE)))
        lse_per_batch_elems = fx.Index(self.traits.NUM_HEADS_Q) * self.seq_len_v
        lse_per_batch_bytes = lse_per_batch_elems * fx.Index(4)
        lse_rsrc = _make_ws_rsrc(lse_base_i64, self.batch_idx * lse_per_batch_bytes, lse_per_batch_bytes)
        lse_val = _fadd(
            _fmul(m_max, self.c_ln2_f, self.fm_fast),
            fmath.log(as_mlir_value(den), fastmath=self.fm_fast),
            self.fm_fast,
        )
        lse_off = fx.Index((self.col == fx.Index(0)).select(self.local_ml_idx, lse_per_batch_elems))
        buffer_ops.buffer_store(as_mlir_value(fx.Float32(lse_val)), lse_rsrc, as_mlir_value(fx.Int32(lse_off)))

    def store_output(self, o_pack):
        o_global = self.seq_idx * self.stride_q_n_v + self.q_head_idx * self.traits.HEAD_DIM + self.col
        buffer_ops.buffer_store(
            o_pack.ir_value(),
            self.o_rsrc,
            as_mlir_value(fx.Int32(o_global * fx.Index(2))),
            offset_is_bytes=True,
        )

def _scale_sched_pairs(pairs, head_dim):
    return max(1, (pairs + 1) // 2) if head_dim == 64 else pairs

def _sched_barrier_pairs(traits, pairs, valu_cnt, group):
    """Emit `pairs` × {1 MFMA + valu_cnt VALU} sched_group_barrier groups."""
    pairs = _scale_sched_pairs(pairs, traits.HEAD_DIM)
    for _ in range_constexpr(pairs):
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, group)
        rocdl.sched_group_barrier(traits.SCHED_VALU_MASK, valu_cnt, group)

def _sched_barrier_exp_pairs(traits, pairs, exp_cnt, group):
    """Emit `pairs` × {1 MFMA + exp_cnt EXP} sched_group_barrier groups."""
    pairs = _scale_sched_pairs(pairs, traits.HEAD_DIM)
    for _ in range_constexpr(pairs):
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, group)
        rocdl.sched_group_barrier(traits.SCHED_EXP_MASK, exp_cnt, group)

def dualwave_splitk_workspace_elems(batch_size, num_heads, seq_len, num_kv_splits, head_dim=128):
    """fp32 elements needed for the split-K workspace: O_partial + Mrow + Lrow.

    O_partial is stored as kernel-native 16-bit (bf16/fp16), two columns per
    fp32 slot; Mrow/Lrow stay fp32.
    """
    rows = batch_size * num_kv_splits * num_heads * seq_len
    return rows * (head_dim // 2) + 2 * rows
