# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MHA Forward Prefill kernel — ``m32x8`` design, gfx1250 (MI400 / mi450).

A fresh, clean FlyDSL kernel written in the high-level layout-algebra style
(tiled copy / tiled MMA + ``SharedAllocator``) — deliberately independent of the
hand-tuned, assembly-mirroring ``fmha_kernel.py`` (inline ASM, raw TDM
descriptors, ``set_vgpr_bank`` hints, per-WMMA schedule tables).

``m32x8`` names the threadgroup shape: **8 waves per threadgroup**, each wave
owning a **32-row** Q span (2 adjacent 16-row WMMA tiles). gfx1250 runs wave32,
so a threadgroup is ``8 * 32 = 256`` threads and ``BLOCK_M = 32 * 8 = 256`` Q
rows. (The leading ``32`` is per-wave Q rows; ``16`` is the WMMA M dimension.)

Layout support — two device kernels over one shared compute core (option B):
  - ``kn_fmha_fwd_prefill_a16w16_m32x8_thd``  — varlen THD, driven by ``cu_seqlens``.
  - ``kn_fmha_fwd_prefill_a16w16_m32x8_bshd`` — batched BSHD, uniform ``seq_len`` scalar
    (no ``cu_seqlens`` tensors → nothing transient to bake into a CUDA graph).
Both resolve their per-workgroup base offsets + sequence bounds, then call the
layout-agnostic ``_core_attention`` helper.

Scope — v1 (this file is intentionally config-agnostic in its name):
  - ``qk_hdim in {128, 192, 256}`` (D_qk), ``v_hdim == 128`` (D_v), ``n_block == 64``
  - dtype: bf16 for Q/K/V/O
  - grouped-query attention (GQA): ``gqa = nheads_q // nheads_k``
  - causal and non-causal

``qk_hdim``, ``v_hdim`` and the dtype are compile-time (build-time) parameters
captured by the builder closure, so they never appear in the file name and can
be generalized later without changing the runtime kernel signatures.

Target: gfx1250, wave32, 8 waves per threadgroup (256 threads).
"""

import functools
from enum import IntEnum

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl._mlir.dialects import scf
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl.expr import arith, gpu, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.jit.utils.chip_info import get_lds_capacity_bytes
from aiter.ops.flydsl.kernels import buffer_ops

from ..tensor_shim import _run_compiled

# Runtime `if` helper the AST rewriter lowers dynamic conditions to. Called
# explicitly here since _core_attention is a module-level helper (outside the
# rewriter's @flyc.kernel scope), keeping side-effect guards free of raw scf.IfOp.
scf_if_dispatch = ReplaceIfWithDispatch.scf_if_dispatch

# Q/K/V staging managers (own their LDS swizzles + async copy schedules). They are
# self-contained: this kernel maintains its own arch constants below and passes the
# config each manager needs through its constructor.
from flydsl.expr.rocdl import tdm_ops

# Single source of truth for gfx1250 Expert Scheduling Mode 2 (DEP_MODE=2). Lives
# in fmha_b16_buffer_managers. Under mode 2 the LLVM setreg (via the
# amdgpu-expert-scheduling-mode hint, set in _ensure_*_kernel) makes LLVM insert all
# depctr covers itself for the plain intrinsics the kernel emits.
from .fmha_b16_buffer_managers import (
    ENABLE_SCHED_MODE2,
    KManager16bV1,
    KManager16bV2,
    OManager16bV1,
    OManager16bV2,
    OManager16bV3,
    QManager16bV1,
    QManager16bV2,
    VManager16bV1,
    VManager16bV2,
    _async_load_to_lds,
    _ir,
)

# ============================================================================
# Threadgroup / arch constants
# ============================================================================

WAVE_SIZE = 32  # gfx1250 kernels run wave32
NUM_WAVES = 8  # "m32x8" — 8 waves per threadgroup
BLOCK_SIZE = WAVE_SIZE * NUM_WAVES  # 256 threads

# "m32x8": each wave owns WMMA_ROW_PER_WAVE adjacent 16-row (WMMA M) Q sub-tiles →
# BLOCK_M = 16 * 2 * 8 = 256 Q rows per threadgroup. Each wave's 2 tiles are
# contiguous: warp i owns rows [i*32, i*32+32) = tiles 2i, 2i+1.
WMMA_M = 16  # query rows per WMMA tile (the "m16" in m32x8)
WMMA_N = 16  # kv rows per WMMA tile (the S^T=K@Q^T output's n_block-direction axis)
WMMA_K = 32  # WMMA contraction depth (bf16 v_wmma_f32_16x16x32); d-tile width
WMMA_ROW_PER_WAVE = 2  # Q WMMA tiles per wave (the "x2" step from m16x8 to m32x8)
BLOCK_M = WMMA_M * WMMA_ROW_PER_WAVE * NUM_WAVES  # 256


class WarpType(IntEnum):
    """Warp-specialization role (compile-time). gfx1250 pairs wave i with wave i+4 on
    one SIMD; the low half (waves 0..3) and high half (waves 4..7) run different
    main-loop preamble orderings so one wave drives memory while its SIMD-mate computes.
    """

    LO_WARP = 0  # waves 0..NUM_WAVES/2-1
    HI_WARP = 1  # waves NUM_WAVES/2..NUM_WAVES-1


DEFAULT_QK_HDIM = 128
DEFAULT_V_HDIM = 128
DEFAULT_DTYPE = "bf16"
_DTYPE_MAP = {"bf16": fx.BFloat16, "fp16": fx.Float16}
_TORCH_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}
SUPPORTED_QK_HDIM = (128, 192, 256)

# KV sequence block (columns of one QK GEMM tile). Configurable; 64 for now.
N_BLOCK_CHOICES = (32, 64, 128, 256)
DEFAULT_N_BLOCK = 64

# Ping-pong K LDS buffers the main loop rotates through (double-buffered prefetch).
N_KV_PP = 2

# Each K/V ping-pong block is floored to this many bytes (reserved headroom).
MIN_KV_BLK_BYTES = 64 * 1024

# log2(e): exp(x) = exp2(x * LOG2E). Softmax uses the native ISA exp2 intrinsic.
LOG2E = 1.4426950408889634

# Deferred oaccu rescale (FAv4 innovation, hk_mla spec §9.1.1). Rescaling the
# running O accumulator by corr = exp(m_prev - m_new) is a full-width VALU pass
# (d_tiles*8 f32/lane) every tile, but corr == 1 when the running max doesn't
# move. So keep m STALE while the tile's row max stays within RESCALE_THRESHOLD
# logit units of it: P = exp2(S - m_stale) accumulates against the un-rescaled
# oaccu/denom, staying consistent. The per-lane test is promoted to wave-uniform
# via ballot (any lane over threshold => the whole wave rescales), so the caller
# can gate the wide multiply with one non-divergent scf.if. Since our exp is
# exp2((s-m)*LOG2E) = e^(s-m) (natural logits), threshold 8.0 => defer until the
# max would move by e^8 ~ 2981x, far under the e^88 fp32 exp overflow wall.
# Set ENABLE_DEFER_RESCALE=False (or threshold < 0) to always rescale.
ENABLE_DEFER_RESCALE = True
RESCALE_THRESHOLD = 8.0

# Running-max seed: a finite big-negative (not -inf) so a fully-masked row keeps m
# finite -> softmax's (m_prev - m_new) and fma(s, .., -m) never hit -inf arithmetic
# (NaN). exp2(big_neg - real) still underflows to 0, so it zeroes the empty seed like
# -inf did. Masked scores stay -inf (p = exp2(-inf) = 0); only the max seed changes.
BIG_NEG = -1.0e30

# Compile-time Q/K/V loader select. False = V1 (Q ring async + swizzled LDS; K/V cluster_load_async +
# swizzled LDS); True = V2 (Q per-warp TDM; K/V TDM global->LDS; all row-major padded LDS, HW OOB,
# fewer address VGPRs). Gates all three loaders (Q, K and V); O is selected separately by O_VARIANT.
USE_TDM_LOADER = True
# O writer variant (decoupled from USE_TDM_LOADER): "v1" swizzled LDS + buffer_store (fastest so
# far), "v2" TDM store (padding ignored -> contiguous LDS -> bank conflict, slow), "v3" padded LDS +
# global_store_async_from_lds_b128.
O_VARIANT = "v3"

# NOTE: the remaining tiling constants (chunk sizes, K/V write-tile + V swizzle
# granularity) live inside fmha_b16_buffer_managers.py — they are intrinsic to the
# managers' LDS layouts, so the kernel no longer declares them here.


# ============================================================================
# Small device helpers
# ============================================================================


def _warp_id():
    """Wave (warp) index within the workgroup, matching opus ``waveid_in_workgroup()``."""
    return fx.Int32(rocdl.wave_id())


def _named_barrier_pair(warp_idx):
    """2-wave SIMD-pair rendezvous (warp `i` ↔ `i+4`) for warp specialization.

    TODO(named-barrier): wire the gfx1250 named barrier — one `@__nbar[warp_idx & 3]`
    per SIMD pair (`s_barrier_signal_var(ptr, 2)` + `s_barrier_wait`), joined once in
    the prologue. Currently a NO-OP: the LO/HI split only reorders each wave's own
    self-contained load→compute, so it stays correct under the workgroup `gpu.barrier()`
    alone; the named barrier is a perf-only SIMD-issue rendezvous, added after the
    standalone probe validates allocation + per-pair sync."""
    del warp_idx


def _lane_id():
    """Lane index within the wave (wave32), matching opus ``lane_id()``."""
    return fx.Int32(
        rocdl_dialect.mbcnt_lo(T.i32, fx.Int32(-1).ir_value(), fx.Int32(0).ir_value())
    )


def _load_seqlen_pair(ptr_tensor, idx):
    """Load ``ptr_tensor[idx]`` and ``ptr_tensor[idx + 1]`` (adjacent i32s) as one
    ``vector<2xi32>``; returns ``(start, end)`` as ``fx.Int32``.

    The two values are contiguous and the address is uniform (derived from
    ``block_id``), so a single 64-bit load should lower to one ``s_load_b64``.
    """
    p = fx.get_iter(ptr_tensor)
    pair = fx.ptr_load(p + fx.Int64(idx), result_type=fx.Vector.make_type(2, fx.Int32))
    return fx.Int32(pair[0]), fx.Int32(pair[1])


def _load_sink_logit(ptr_sink, q_head_idx, num_heads_q):
    """Load this lane's per-head sink logit ``sink[q_head_idx]`` from the 1-D
    ``[num_heads_q]`` fp32 ``sink`` — one extra ``exp(sink)`` term in the softmax
    denominator, in the scaled-score domain (same units as S).

    Uses a flat ``llvm.load`` (not ``buffer_load``): ``buffer_load`` re-scales the
    offset (``offset * element_bytes``) INTERNALLY, so a flat load keeps the address
    arithmetic SSA-visible for LLVM to order/cover under sched mode 2. Safe without a
    HW bounds check because ``q_head_idx = kv_head*gqa_ratio + row_idx%gqa_ratio`` is
    always ``< num_heads_q`` (in-bounds by construction)."""
    del num_heads_q  # in-bounds by construction; no buffer bounds check needed
    sink_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_sink)))
    byte_off = fx.Int64(q_head_idx) * fx.Int64(4)
    addr = sink_base_i64 + byte_off
    gptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
    return fx.Float32(llvm_dialect.load(ir.F32Type.get(), gptr))


def _packed_tile_indices(gqa_ratio, warp_idx, lane_idx):
    """Map this lane's rows in the packed ``(seq, q_head_in_group)`` tile to global
    indices; returns ``(kv_head, q_head_idx, seq_idx)`` where ``kv_head`` is a
    scalar ``fx.Int32`` (shared) and ``q_head_idx`` / ``seq_idx`` are length-R
    lists (one per q-WMMA-tile owned by this wave; R = WMMA_ROW_PER_WAVE).

    GQA head x seq packing:
      block_id x -> tile over one kv-head's ``(seq, q_head_in_group)`` plane
      block_id y -> kv_head
    ``q_head_in_group`` is the fast axis, so the ``% / //`` use the small (often
    power-of-two) ``gqa_ratio``. Each of the ``BLOCK_M`` rows is an independent
    query sharing this kv-head's K/V. The R tiles a wave owns are contiguous:
    ``warp_row0 = block_x*BLOCK_M + warp_idx*(R*WMMA_M)`` and tile ``qt`` starts
    at ``warp_row0 + qt*WMMA_M``.
    """
    kv_head = fx.Int32(gpu.block_id("y"))
    warp_row0 = fx.Int32(gpu.block_id("x")) * BLOCK_M + warp_idx * (
        WMMA_ROW_PER_WAVE * WMMA_M
    )
    q_head_idx = []
    seq_idx = []
    for qt in range(WMMA_ROW_PER_WAVE):
        row_idx = warp_row0 + qt * WMMA_M + lane_idx % WMMA_M
        q_head_idx.append(kv_head * gqa_ratio + row_idx % gqa_ratio)
        seq_idx.append(row_idx // gqa_ratio)
    return kv_head, q_head_idx, seq_idx


# ============================================================================
# Compute stages — EMPTY, unwired. Implemented and tested one at a time; the KV
# streaming driver below lands (and is tested) first with these left inert.
# ============================================================================


def _wmma(a, b, c):
    """v_wmma_f32_16x16x32_{bf16,f16} (gfx1250, wave32): C[16x16 f32] = A[16x32] @
    B[32x16] + C. No fdsl wrapper exists for this op (only mfma/fp8/f4), so we call
    the raw ODS builder locally.

    a/b: v16 16-bit fragments; c: v8 f32 accumulator; returns the v8 f32 result
    (raw MLIR value, feed straight back as ``c`` to accumulate)."""
    v8f32 = fx.Vector.make_type(8, fx.Float32)
    wmma = (
        rocdl_dialect.wmma_f32_16x16x32_f16
        if a.dtype is fx.Float16
        else rocdl_dialect.wmma_f32_16x16x32_bf16
    )
    # modC defaults to WMMACModifier::none (== the old modC=0); omit it.
    return wmma(v8f32, _ir(a), _ir(b), _ir(c), reuseA=False, reuseB=False).result


def _qk_gemm(*, k_values, q_frags_list, n_block):
    """GEMM1: S^T = K @ Q^T for one resident KV tile, for all R q-WMMA-tiles this
    wave owns. K is **shared** across the q-tiles (loaded once), so each K fragment
    is shuffled once and fed into R independent WMMA chains.

    WMMA convention (gfx1250): S^T[kv,q] = K @ Q^T with **K = A-operand** (src_a)
    and **Q = B-operand** (src_b). Contract d in ``NDT = qk_hdim//WMMA_K`` tiles;
    produce ``NKV = n_block//WMMA_N`` kv-tiles. GPU-verified accumulator layout:
    lane ``l`` element ``si`` holds S^T[kv = kv_tile*WMMA_N + (l//16)*8 + si,
    q = l%16] (kv on the C-row / M axis, q on the C-col / N axis).

    ``q_frags_list`` is a length-R list; entry ``qt`` is that q-tile's NDT v16-bf16
    Q fragments. Returns ``s_acc_list``: a length-R list, each a list of NKV
    v8-f32 accumulators (== P^T for that q-tile).

    ``k_values`` is the already-burst-loaded flat ``(kv, dt, half)`` list of
    v8-bf16 ds_load results (from ``k_mgr.load_k_to_reg``, kept OUT of the WMMA stream so
    there is no wmma->ds_load issue bubble). Each K fragment is the two 16-col halves of a
    d-tile shuffled into a v16 fragment matching the Q frag layout.

    NOTE: the ``s_wait_dscnt(0)`` that drains the K ds_load burst is now issued by the
    (warp-specialized) main-loop preamble before this call — under warp specialization the
    LO and HI warps drain at different points, so the wait can't live inside the gemm.
    """
    R = len(q_frags_list)
    NKV = n_block // WMMA_N  # output kv tiles (WMMA_N kv rows each)
    NDT = len(q_frags_list[0])  # contraction d-tiles (== qk_hdim // WMMA_K)

    # Consume in (kv, dt, half) order: a (half=0, half=1) pair shuffles into a v16
    # K fragment (shared by all q-tiles); NDT d-tiles accumulate into one kv-tile's
    # s_acc, independently per q-tile.
    s_acc_list = [[None] * NKV for _ in range(R)]
    j = 0
    for kv in range(NKV):
        for dt in range(NDT):
            lo = k_values[j]
            hi = k_values[j + 1]
            j += 2
            k_frag = lo.shuffle(hi, list(range(16)))
            for qt in range(R):
                acc = (
                    s_acc_list[qt][kv]
                    if dt > 0
                    else fx.Vector.filled(8, 0.0, fx.Float32)
                )
                s_acc_list[qt][kv] = _wmma(k_frag, q_frags_list[qt][dt], acc)
    return s_acc_list


def _tree_reduce_multi(lists, op3, op2):
    """Balanced 3-way tree reduction of R independent lists in lockstep, returning one result
    per list. Per list the critical path is ~ceil(log3(N)) vs N-1 for a left-fold, and op3 =
    nested op2 so the backend fuses it (v_max3_f32 for max). Each layer's combines are emitted
    POSITION-MAJOR across the lists (list0[pos], list1[pos], ...) so the R independent ops sit
    adjacent in the IR -> the backend can dual-issue them and hide one row's cross-lane /
    latency bubble behind the other's work."""
    curs = [list(v) for v in lists]
    while max(len(c) for c in curs) > 1:
        nxts = [[] for _ in curs]
        idxs = [0] * len(curs)
        while any(idxs[k] < len(curs[k]) for k in range(len(curs))):
            for k in range(len(curs)):
                cur, i, n = curs[k], idxs[k], len(curs[k])
                if i >= n:
                    continue
                if n - i >= 3:
                    nxts[k].append(op3(cur[i], cur[i + 1], cur[i + 2]))
                    idxs[k] += 3
                elif n - i == 2:
                    nxts[k].append(op2(cur[i], cur[i + 1]))
                    idxs[k] += 2
                else:
                    nxts[k].append(cur[i])
                    idxs[k] += 1
        curs = nxts
    return [c[0] for c in curs]


def _softmax(
    *,
    s_list,
    m_prev_list,
    d_prev_list,
    lane_idx,
    n_block,
    kv_pos_base=None,
    q_max_list=None,
    q_min_list=None,
    kv_len=None,
    elem_dtype,
):
    """Online-softmax update for one KV tile, for ALL R q-WMMA-tiles this wave owns.

    The R rows are independent (each owns its S, running m/d, and mask bounds) but share
    the tile's K/V. Processing them together lets the two rows' balanced max-tree and
    sum-tree reductions emit INTERLEAVED (position-major across rows, via
    ``_tree_reduce_multi``) so the backend can dual-issue row0/row1 combines and hide each
    other's cross-lane permlanex16 latency. ``s_list[r]`` already includes softmax_scale
    (folded into Q), so exp uses plain LOG2E.

    Layout (from ``_qk_gemm``): ``s_list[r]`` is a list of ``NKV = n_block//WMMA_N`` v8-f32
    accumulators; this lane owns query ``q = warp*16 + l%16`` and, in tile ``kvt``, the kv
    rows ``kvt*16 + (l//16)*8 + [0..8)`` (its half). The peer lane ``l^16`` holds the other
    8-row half of the same q, so the row max/sum reduce locally over (kvt, i) then across
    the ``shuffle_xor(16)`` partner.

    Masking (per element, per row r, sequence-relative ``kv_pos = kv_pos_base + (l//16)*8 +
    kvt*16 + i``; all bounds fx.Int32): ``q_max_list[r]`` masks ``kv_pos > q_max`` (band
    upper edge = ``q_seq + (kv_len-q_len) + window_right``; clamped to ``kv_len-1`` on the
    last tile to fold the tail); ``q_min_list[r]`` masks ``kv_pos < q_min`` (band lower edge
    = ``q_seq + (kv_len-q_len) - window_left``); kv_len (only when q_max is None) masks
    ``kv_pos >= kv_len`` (standalone tail for the non-causal case). A None bound skips it.

    Args: ``s_list``/``m_prev_list``/``d_prev_list``/``q_max_list``/``q_min_list`` are
    length-R lists (R = WMMA_ROW_PER_WAVE); the q_*_list default to all-None. m_prev/d_prev
    are fx.Float32 shared by the l<->l^16 pair.

    Returns 5 length-R lists ``(p, m_new, d_new, corr, do_rescale)`` — per row: p = NKV v8
    **bf16** P^T = exp(S^T - m_new); m_new = updated running max, STALE (== m_prev) when the
    deferred-rescale ballot did not fire (FAv4 §9.1.1); d_new = corr*d_prev + rowsum(p);
    corr = exp(m_prev - m_new) (== 1 on the stale path); do_rescale = wave-uniform i1 (None
    when deferral is compiled out).
    """
    NKV = n_block // WMMA_N
    f32 = ir.F32Type.get()
    fast = arith.FastMathFlags.fast
    neg_inf = fx.Float32(float("-inf"))
    zero = fx.Float32(0.0)
    log2e = fx.Float32(LOG2E)

    def fmax(a, b):
        return fx.Float32(arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fast).result)

    def fadd(a, b):
        return fx.Float32(arith.addf(_raw(a), _raw(b), fastmath=fast))

    # fast-math WITHOUT reassoc: LLVM's Reassociate pass otherwise re-linearizes the
    # sum tree back into a serial chain (max survives — Reassociate ignores maxnum).
    _FF = arith.FastMathFlags
    _no_reassoc = _FF.nnan | _FF.ninf | _FF.nsz | _FF.arcp | _FF.contract | _FF.afn

    def fadd_t(a, b):
        return fx.Float32(arith.addf(_raw(a), _raw(b), fastmath=_no_reassoc))

    def fsub(a, b):
        return fx.Float32(arith.subf(_raw(a), _raw(b), fastmath=fast))

    def fmul(a, b):
        return fx.Float32(arith.mulf(_raw(a), _raw(b), fastmath=fast))

    def exp2(x):
        return fx.Float32(rocdl.exp2(f32, _raw(x)))

    # permlanex16 selectors: identity cross-16 gather (nibbles 0..15) => lane l<->l^16.
    sel_lo, sel_hi = _raw(fx.Int32(0x76543210)), _raw(fx.Int32(0xFEDCBA98))

    def peer(v):  # cross-lane reduce partner: lane l <-> l^16 (the other kv half)
        return fx.Float32(
            rocdl_dialect.permlanex16(
                f32,
                _raw(v),
                _raw(v),
                sel_lo,
                sel_hi,
                fi=False,
                bound_control=False,
            )
        )

    khalf = lane_idx // fx.Int32(WMMA_M)  # 0/1: which 8-row kv half this lane owns

    R = len(s_list)
    q_max_list = q_max_list if q_max_list is not None else [None] * R
    q_min_list = q_min_list if q_min_list is not None else [None] * R

    # ---- Pass 1 (all R rows): masked S values, flattened (kvt, i) order. Built for every
    # row first so the row max-trees below emit INTERLEAVED. ----
    s_masked_list = []
    for r in range(R):
        s = s_list[r]
        q_max, q_min = q_max_list[r], q_min_list[r]
        s_masked = []
        for kvt in range(NKV):
            svec = fx.Vector(_ir(s[kvt]))
            for i in range(8):
                sval = fx.Float32(svec[i])
                if q_max is not None or q_min is not None or kv_len is not None:
                    kv_pos = (
                        kv_pos_base + khalf * fx.Int32(8) + fx.Int32(kvt * WMMA_N + i)
                    )
                    if q_max is not None:
                        ubound = (
                            q_max
                            if kv_len is None
                            else fx.min(q_max, kv_len - fx.Int32(1))
                        )
                        sval = (kv_pos > ubound).select(neg_inf, sval)
                    if q_min is not None:
                        sval = (kv_pos < q_min).select(neg_inf, sval)
                    if kv_len is not None and q_max is None:
                        sval = (kv_pos >= kv_len).select(neg_inf, sval)
                s_masked.append(sval)
        s_masked_list.append(s_masked)

    # ---- Row max: the R rows' balanced max-trees emitted INTERLEAVED (position-major
    # across rows) so the backend dual-issues row0/row1 combines and hides the cross-lane
    # permlanex16 latency. ----
    max3 = lambda a, b, c: fmax(fmax(a, b), c)
    local_max_list = _tree_reduce_multi(s_masked_list, max3, fmax)

    # ---- Per row: peer reduce + deferred-rescale decision + corr / neg_m. ----
    m_new_list, corr_list, neg_m_list, do_rescale_list = [], [], [], []
    for r in range(R):
        m_prev, q_min = m_prev_list[r], q_min_list[r]
        row_max = fmax(local_max_list[r], peer(local_max_list[r]))
        m_full = fmax(m_prev, row_max)

        # Deferred oaccu rescale (FAv4, hk_mla spec 9.1.1): keep m STALE while the running
        # max barely moves (< RESCALE_THRESHOLD logits) so the caller SKIPS the wide
        # `o_acc *= corr` multiply. Ballot promotes the per-lane test to wave-uniform (non-
        # divergent branch). ORDERED OGT: a fully-masked lane's -inf - -inf = NaN never
        # forces a rescale. Safe stale path: row_max - m_prev <= 8 -> p <= e^8, no overflow.
        if ENABLE_DEFER_RESCALE and RESCALE_THRESHOLD >= 0.0:
            # `>` lowers to ordered OGT, so a fully-masked lane's -inf - -inf = NaN
            # compares false and never forces a rescale.
            need = fsub(row_max, m_prev) > fx.Float32(RESCALE_THRESHOLD)
            mask = rocdl.ballot(fx.Int32.ir_type, need)
            do_rescale = fx.Int32(mask) != fx.Int32(0)
            m_new = do_rescale.select(m_full, m_prev)
        else:
            do_rescale = None
            m_new = m_full

        # corr = exp(m_prev - m_new); neg_m = -(m_new * log2e) for the fused p exp.
        # m is seeded to BIG_NEG (finite), so m_prev/m_new never reach -inf: a fully
        # masked row (row_max=-inf) keeps m_new=BIG_NEG, giving corr=exp2(0)=1 and a
        # finite neg_m (p=exp2(-inf)=0). No (-inf)-(-inf) / -inf+inf, so no clamp needed.
        corr = exp2(fmul(fsub(m_prev, m_new), log2e))
        neg_m = fsub(zero, fmul(m_new, log2e))
        m_new_list.append(m_new)
        corr_list.append(corr)
        neg_m_list.append(neg_m)
        do_rescale_list.append(do_rescale)

    # ---- Pass 2 (all R rows): p = exp(S - m_new) (bf16, per tile) + flat p for the sum
    # tree. Built for every row first so the row sum-trees below emit INTERLEAVED. ----
    p_list, p_flat_list = [], []
    for r in range(R):
        neg_m, s_masked = neg_m_list[r], s_masked_list[r]
        p, p_flat, idx = [], [], 0
        for kvt in range(NKV):
            pe = []
            for i in range(8):
                # exp2(s*log2e - m_new*log2e) via one fma.
                pj = exp2(
                    fx.Float32(fmath.fma(_raw(s_masked[idx]), _raw(log2e), _raw(neg_m)))
                )
                pe.append(pj)
                p_flat.append(pj)
                idx += 1
            p.append(fx.Vector.from_elements(pe, fx.Float32).to(elem_dtype))
        p_list.append(p)
        p_flat_list.append(p_flat)

    # ---- Row sum: R rows' balanced sum-trees emitted INTERLEAVED. fadd_t (fast-math minus
    # reassoc) so LLVM's Reassociate does NOT re-linearize the tree into a serial chain. ----
    add3 = lambda a, b, c: fadd_t(fadd_t(a, b), c)
    local_sum_list = _tree_reduce_multi(p_flat_list, add3, fadd_t)

    d_new_list = []
    for r in range(R):
        d_new_list.append(
            fadd(
                fmul(corr_list[r], d_prev_list[r]),
                fadd(local_sum_list[r], peer(local_sum_list[r])),
            )
        )
    return p_list, m_new_list, d_new_list, corr_list, do_rescale_list


def _pv_gemm(*, v_values, p_list, v_hdim, n_block, o_acc_list=None):
    """GEMM2: O^T = V^T @ P^T for one resident KV tile, for all R q-WMMA-tiles this
    wave owns. V is **shared** across the q-tiles (transpose-loaded once), so each
    V fragment is shuffled once and fed into R independent WMMA chains.

    WMMA convention (gfx1250): D[M=d, N=q] with **A = V^T** (src_a, transpose-loaded
    via ds_load_tr16_b128) and **B = P^T** (src_b, the bf16 softmax output). Contract
    kv in ``nkt = n_block//WMMA_K`` tiles (K=32); produce ``d_tiles = v_hdim//WMMA_M``
    output d-tiles (M axis). Lane ``l`` element ``si`` of tile ``dt`` holds
    O[q = l%16, d = dt*WMMA_M + (l//16)*8 + si] — the OManager16b frag layout.

    ``p_list`` is a length-R list; entry ``qt`` is that q-tile's list of softmax
    kv-tiles (bf16 P^T B-operands). ``o_acc_list`` is either None or a length-R
    list of running O accumulators (each ``d_tiles`` v8-f32, already rescaled by
    ``corr``). Returns ``out_list``: a length-R list of updated O accumulators.

    ``v_values`` is the already-burst-loaded flat ``(dt, kt, half)`` list of v8-bf16
    transpose-load results (from ``v_mgr.load_v_to_reg``, kept OUT of the WMMA stream so there is
    no wmma->ds_load issue bubble); a single ``s_wait_dscnt(0)`` drains that burst
    here before any WMMA. Online accumulation: each tile's PV adds onto the running
    o_acc. Each WMMA operand is a v16 bf16 fragment = two 16-wide halves shuffled:
    A from V-tiles (kv, kv+16), B from softmax tiles (p[2kt], p[2kt+1]).
    """
    R = len(p_list)
    d_tiles = v_hdim // WMMA_M  # output d-tiles (M axis, WMMA_M d rows each)
    nkt = n_block // WMMA_K  # kv contraction tiles (K=32 kv each)

    # Drain the whole V transpose burst once; every v_values entry is now resident.
    rocdl.s_wait_dscnt(0)

    out_list = [[None] * d_tiles for _ in range(R)]
    for dt in range(d_tiles):
        accs = [
            (
                o_acc_list[qt][dt]
                if o_acc_list is not None
                else fx.Vector.filled(8, 0.0, fx.Float32)
            )
            for qt in range(R)
        ]
        j = dt * nkt * 2
        for kt in range(nkt):
            # A-operand: V^T frag = two 16-kv transpose-load tiles -> v16 bf16
            # (shared across q-tiles).
            v_lo = v_values[j]
            v_hi = v_values[j + 1]
            j += 2
            v_frag = v_lo.shuffle(v_hi, list(range(16)))
            for qt in range(R):
                # B-operand: P^T frag = two consecutive softmax kv-tiles -> v16 bf16.
                p = p_list[qt]
                p_frag = p[2 * kt].shuffle(p[2 * kt + 1], list(range(16)))
                accs[qt] = _wmma(v_frag, p_frag, accs[qt])
        for qt in range(R):
            out_list[qt][dt] = accs[qt]
    return out_list


# ============================================================================
# Shared, layout-agnostic compute core
# ============================================================================


def _alloc_lds():
    """Allocate the full per-CU LDS once and return its base (fx.Int32). Called once per
    kernel body before the warp-type dispatch so both ``_core_attention`` traces share the
    single SharedAllocator flydsl permits; K/V, Q, and the O epilogue all carve this base.
    """
    smem = fx.SharedAllocator().allocate(get_lds_capacity_bytes("gfx1250"))
    return fx.Int32(fx.ptrtoint(smem.peek().ptr))


def _core_attention(
    *,
    qk_hdim,
    v_hdim,
    n_block,  # compile-time KV block width (columns of one QK GEMM tile)
    mask_left,  # compile-time: bound the left band edge (finite window_left)
    mask_right,  # compile-time: bound the right band edge (causal or finite window_right)
    return_lse,
    has_sink,  # compile-time: fold a per-head sink logit into the softmax denom
    gqa_ratio,  # compile-time GQA group size = nheads_q // nheads_kv
    ptr_O,
    ptr_Q,
    ptr_K,
    ptr_V,
    ptr_LSE,
    ptr_sink,  # [nheads_q] fp32 per-head sink logits; read only when has_sink
    softmax_scale,
    stride_q_seq,
    stride_k_seq,
    stride_v_seq,
    stride_o_seq,
    stride_q_head,
    stride_k_head,
    stride_v_head,
    stride_o_head,
    # LSE addressing (element strides + per-batch bound), resolved by the caller.
    # Only consumed when return_lse; the caller may pass anything otherwise.
    stride_lse_seq,
    stride_lse_head,
    lse_base_elems,  # first element offset of this batch's LSE slab
    lse_num_records_bytes,  # buffer-resource bound (below the 0x7FFFFFFF drop)
    # Per-batch token ranges (fx.Int32), resolved by the caller:
    q_start,  # first Q token index of this batch in the global tensor
    q_len,  # valid Q tokens in this batch
    kv_start,  # first K/V token index of this batch
    kv_len,  # valid K/V tokens in this batch
    # Sliding-window bounds (runtime fx.Int32, >= 0). window_left read only when
    # mask_left, window_right only when mask_right. Causal == mask_right, window_right=0.
    window_left,
    window_right,
    warp_idx,  # runtime fx.Int32 wave index
    warp_type,  # compile-time WarpType (LO_WARP / HI_WARP)
    lds_base,  # LDS base (fx.Int32), allocated once by the caller (_alloc_lds)
    elem_dtype,  # compile-time fx.BFloat16 / fx.Float16 for Q/K/V/P/O fragments
):
    """Layout-agnostic m32x8 compute — empty scaffold.

    Shared by the THD and BSHD kernel entries. The caller resolves the per-batch
    token ranges (``q_start``/``q_len`` and ``kv_start``/``kv_len``) — the only
    part that differs between varlen and batched layouts — and passes them here.

    Warp-specialized: the caller dispatches on runtime ``warp_type`` and traces this
    body TWICE (once per compile-time ``warp_type``); the two instantiations differ in
    the ``main_loop`` preamble ordering (LO drives K load, HI shadows it) and rendezvous
    on a 2-wave named barrier (``_named_barrier_pair``, currently a no-op stub).
    """
    lane_idx = _lane_id()
    kv_head, q_head_idx, seq_idx = _packed_tile_indices(gqa_ratio, warp_idx, lane_idx)

    # K/V staging: N_KV_PP ping-pong slots ([K.pp0|V.pp0][K.pp1|V.pp1]), blocks floored
    # at 64KB. O reuses a non-current slot; Q time-shares slot 1, so the slot must also
    # be >= the Q staging footprint (at qk_hdim=256 that exceeds K|V+128KB -> we grow the
    # slot, "allocating additional space for K|V"; still occupancy=1). slot_bytes is
    # compile-time (no allocation; lds_base is passed in).
    if USE_TDM_LOADER:
        q_mgr = QManager16bV2(
            qk_hdim=qk_hdim,
            gqa_ratio=gqa_ratio,
            num_waves=NUM_WAVES,
            q_tiles_per_wave=WMMA_ROW_PER_WAVE,
            elem_dtype=elem_dtype,
        )
        k_mgr = KManager16bV2(
            qk_hdim=qk_hdim,
            n_block=n_block,
            num_waves=NUM_WAVES,
            elem_dtype=elem_dtype,
        )
        v_mgr = VManager16bV2(
            v_hdim=v_hdim, n_block=n_block, num_waves=NUM_WAVES, elem_dtype=elem_dtype
        )
    else:
        q_mgr = QManager16bV1(
            qk_hdim=qk_hdim,
            gqa_ratio=gqa_ratio,
            num_waves=NUM_WAVES,
            q_tiles_per_wave=WMMA_ROW_PER_WAVE,
            elem_dtype=elem_dtype,
        )
        k_mgr = KManager16bV1(
            qk_hdim=qk_hdim,
            n_block=n_block,
            num_waves=NUM_WAVES,
            elem_dtype=elem_dtype,
        )
        v_mgr = VManager16bV1(
            v_hdim=v_hdim, n_block=n_block, num_waves=NUM_WAVES, elem_dtype=elem_dtype
        )
    k_blk_bytes = max(k_mgr.get_lds_size_in_byte(), MIN_KV_BLK_BYTES)
    v_blk_bytes = max(v_mgr.get_lds_size_in_byte(), MIN_KV_BLK_BYTES)
    slot_bytes = max(k_blk_bytes + v_blk_bytes, q_mgr.get_lds_size_in_byte())

    def _k_lds_buf(
        pp,
    ):  # K base of ping-pong slot ``pp`` (int or fx.Int32; folds when const)
        if isinstance(pp, int):
            pp = fx.Int32(pp)
        return lds_base + pp * fx.Int32(slot_bytes)

    def _v_lds_buf(pp):  # V base of ping-pong slot ``pp`` (== K base + k_blk_bytes)
        if isinstance(pp, int):
            pp = fx.Int32(pp)
        return lds_base + pp * fx.Int32(slot_bytes) + fx.Int32(k_blk_bytes)

    # ---- Q staging TIME-SHARES slot 1: Q's LDS base = slot-1 base (kv_base +
    # slot_bytes). Q is loaded + drained into VGPR in the prologue, then dead; the
    # main loop's first slot-1 prefetch reuses the region. Safe with zero new sync —
    # the prologue drains Q (part2) -> s_wait_asynccnt(0) -> gpu.barrier() BEFORE the
    # loop, and prologue K/V loads target slot 0. slot_bytes >= Q footprint by
    # construction (see above), so Q always fits in slot 1. ----
    q_lds_base = lds_base + fx.Int32(slot_bytes)

    q_mgr.load_q_to_vgpr_part1(
        ptr_Q=ptr_Q,
        stride_q_seq=stride_q_seq,
        stride_q_head=stride_q_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        block_x=fx.Int32(gpu.block_id("x")),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds=q_lds_base,
    )

    # ---- This WG's KV tiles span relative kv [start_tile*n_block, kv_len_wg).
    # Packed row r maps to seq r//gqa_ratio; a query at seq s attends the band
    # [s+causal_off-window_left, s+causal_off+window_right] (causal_off=kv_len-q_len).
    #
    # Right edge (mask_right): kv_len_wg clips to the WG's max query's attend-limit so
    # we don't run tiles fully past the band. Non-mask_right: all kv (kv_len).
    # Left edge (mask_left): start_tile skips whole tiles before the WG's min query's
    # band start. Non-mask_left: start at tile 0.
    block_x = fx.Int32(gpu.block_id("x"))
    causal_off = kv_len - q_len
    if mask_right:
        wg_max_seq = (block_x * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1)) // fx.Int32(
            gqa_ratio
        )
        wg_max_seq = fx.min(wg_max_seq, q_len - fx.Int32(1))
        kv_len_wg = wg_max_seq + causal_off + window_right + fx.Int32(1)
        kv_len_wg = fx.min(kv_len_wg, kv_len)
        kv_len_wg = fx.max(kv_len_wg, fx.Int32(1))
    else:
        kv_len_wg = kv_len

    # Tile range [start_tile, n_tiles): n_tiles from the right-clipped kv_len_wg;
    # start_tile skips whole tiles before the WG's min query's band start. The
    # defensive min() keeps start_tile a valid buffer index even for an over-launched
    # WG whose whole band is empty (its per-element masks zero the work anyway).
    n_tiles = fx.ceildiv(kv_len_wg, fx.Int32(n_block))
    if mask_left:
        wg_min_seq = (block_x * fx.Int32(BLOCK_M)) // fx.Int32(gqa_ratio)
        kv_lo = fx.max(wg_min_seq + causal_off - window_left, fx.Int32(0))
        start_tile = kv_lo // fx.Int32(n_block)
        start_tile = fx.min(start_tile, fx.Int32(n_tiles) - fx.Int32(1))
    else:
        start_tile = fx.Int32(0)

    def _kv_valid(blk_row0):
        # How many rows of [blk_row0, blk_row0+n_block) are in-bounds, clamped to
        # the WG's effective KV length kv_len_wg (0..n_block). Past the end -> 0 (a
        # harmless clamped load that is never consumed).
        rem = fx.max(kv_len_wg - blk_row0, fx.Int32(0))
        return fx.min(rem, fx.Int32(n_block))

    # ---- Prologue (reordered for the mode-2 hang investigation): compute all K/V
    # addresses AND the loop-init in the Q global-load shadow, then run part2 (Q
    # ds_load), then issue the K/V cluster_loads LAST — so NOTHING runs between the
    # loads and the prologue barrier below. Sequence: (1) part1 [above] -> (2) KMgr
    # param calc -> (3) loop init -> (4) part2 -> (5) K cluster_load -> (6) V
    # cluster_load.

    # (2) KMgr param calc — pure address arithmetic (no memory op), hoisted into the
    # Q global-load shadow.
    #
    # Ping-pong parity is LOCAL to this WG's tile stream: the prologue always loads the
    # first tile (start_tile) into buffer 0, and the main loop selects buffers by the
    # 0-based LOCAL iteration index (not the absolute tile index), so start_tile parity
    # is irrelevant. This lets the 2x-unrolled loop pick buffers at COMPILE time (even
    # local iter -> buffer 0, odd -> buffer 1) and reach buffer 1 from buffer-0 ds_load
    # pointers via a constant immediate offset.
    start_pp = 0
    start_row0 = start_tile * fx.Int32(n_block)
    if USE_TDM_LOADER:
        # V2: build the TDM copy views for the first tile (pure), run Q part2, then issue
        # the K/V TDM copies and drain with tensor_wait before the prologue barrier.
        k_views = k_mgr.load_views(
            ptr_lds=_k_lds_buf(start_pp),
            ptr_K=ptr_K,
            stride_k_seq=stride_k_seq,
            stride_k_head=stride_k_head,
            kv_head=kv_head,
            kv_row0=kv_start + start_row0,
            kv_valid=_kv_valid(start_row0),
        )
        v_views = v_mgr.load_views(
            ptr_lds=_v_lds_buf(start_pp),
            ptr_V=ptr_V,
            stride_v_seq=stride_v_seq,
            stride_v_head=stride_v_head,
            kv_head=kv_head,
            kv_row0=kv_start + start_row0,
            kv_valid=_kv_valid(start_row0),
        )
        q_frags = q_mgr.load_q_to_vgpr_part2(scale=softmax_scale)
        for _v in k_views:
            fx.copy_atom_call(*_v)
        for _v in v_views:
            fx.copy_atom_call(*_v)
        tdm_ops.tensor_wait(0)
        gpu.barrier()
    else:
        k_gptrs, k_lds_ptrs, k_imm_offs = k_mgr.global_load_ptrs(
            ptr_lds=_k_lds_buf(start_pp),
            ptr_K=ptr_K,
            stride_k_seq=stride_k_seq,
            stride_k_head=stride_k_head,
            kv_head=kv_head,
            kv_row0=kv_start + start_row0,
            kv_valid=_kv_valid(start_row0),
            warp_idx=warp_idx,
            lane_idx=lane_idx,
        )
        v_gptrs, v_lds_ptrs, v_imm_offs = v_mgr.global_load_ptrs(
            ptr_lds=_v_lds_buf(start_pp),
            ptr_V=ptr_V,
            stride_v_seq=stride_v_seq,
            stride_v_head=stride_v_head,
            kv_head=kv_head,
            kv_row0=kv_start + start_row0,
            kv_valid=_kv_valid(start_row0),
            warp_idx=warp_idx,
            lane_idx=lane_idx,
        )
        # (3) QMgr part2 — Q ds_load LDS->VGPR (drains the part1 Q async), issued AHEAD of
        # the cluster_loads so its Q-scaling reg reuse leaves the load shadow.
        q_frags = q_mgr.load_q_to_vgpr_part2(scale=softmax_scale)
        # (4)+(5) Issue K then V cluster_loads LAST as ONE packed burst before the compiler
        # reuses their source address VGPRs (part2 clobbers them). Only the cross-wave
        # s_barrier is an immovable wall that packs the burst (mode-2 async-source-WAR fix).
        _async_load_to_lds(k_gptrs, k_lds_ptrs, cluster=True, imm_offs=k_imm_offs)
        _async_load_to_lds(v_gptrs, v_lds_ptrs, cluster=True, imm_offs=v_imm_offs)
        rocdl.s_wait_asynccnt(0)
        gpu.barrier()

    # (7) Loop init — MOVED to after the prologue barrier (ordering experiment). Online-
    # softmax seed + O accumulators (iter_args) and loop bounds.
    #
    # Loop-carried state (scf.for_ iter_args): the online-softmax running max ``m`` and
    # denom ``d`` (per-lane f32), followed by the ``d_tiles`` fp32 O accumulators. Seed
    # m=-inf, d=0, O=0: the first tile's corr=exp2(m_prev-m_new)=0 zeroes the
    # (already-zero) O before its PV adds in — the standard flash seed. (Fully-masked
    # leading tiles under a finite-left window would make exp2(-inf-(-inf))=NaN;
    # _softmax sanitizes that on the q_min path.)
    #
    # Attention sink (compile-time): the sink is one extra ``exp(sink)`` term in the
    # softmax denominator. Fold it in by seeding m=sink[q_head] and d=1.0 (=exp(sink-
    # sink)); the rescales carry that d seed to exactly exp(sink - m_final), the sink
    # denom term. (Without a sink, m=-inf makes the first tile's corr zero the d seed,
    # so d=1 would equal d=0 — the no-sink path keeps d=0 to stay byte-for-byte.)
    d_tiles = v_hdim // WMMA_M
    R = WMMA_ROW_PER_WAVE
    _QS = 2 + d_tiles  # per-q-tile carried state: [m, d, O_0 .. O_{d_tiles-1}]
    if has_sink:
        num_heads_q = gpu.grid_dim.y * fx.Int32(gqa_ratio)
        m_init = [
            _load_sink_logit(ptr_sink, q_head_idx[qt], num_heads_q) for qt in range(R)
        ]
        d_init = [fx.Float32(1.0) for _ in range(R)]
    else:
        m_init = [fx.Float32(BIG_NEG) for _ in range(R)]
        d_init = [fx.Float32(0.0) for _ in range(R)]
    # _init = R copies of [m, d, O_tile0 .. O_tile{d_tiles-1}] — per q-tile running max,
    # denom, then one v8-f32 O accumulator per 16-wide output-dim tile (this lane's
    # partial O[q, d]), all zero. The R q-tiles have independent online-softmax state.
    _init = []
    for qt in range(R):
        _init += [
            _raw(m_init[qt]),
            _raw(d_init[qt]),
        ] + [_raw(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(d_tiles)]

    # ---- ds_load LDS base pointers for both ping-pong buffers, carried as iter_args and
    # swapped curr<->next each iteration (buffer selected by pointer). Base count per mgr
    # is manager-defined (V1: 2, V2: 1) — carried generically. Same machinery for V1/V2;
    # only the global->LDS ISSUE (_addr_phase/_prefetch/_drain) differs. ----
    k_lds_ld_curr = k_mgr.ds_load_ptrs(ptr_lds=_k_lds_buf(0), lane_idx=lane_idx)
    v_lds_ld_curr = v_mgr.ds_load_ptrs(ptr_lds=_v_lds_buf(0), lane_idx=lane_idx)
    k_lds_ld_next = k_mgr.ds_load_ptrs(ptr_lds=_k_lds_buf(1), lane_idx=lane_idx)
    v_lds_ld_next = v_mgr.ds_load_ptrs(ptr_lds=_v_lds_buf(1), lane_idx=lane_idx)
    _NKB = len(k_lds_ld_curr)  # ds bases per K buffer (V1: 2, V2: 1)
    _NVB = len(v_lds_ld_curr)
    _PTR_BASE = len(_init)
    _init = _init + k_lds_ld_curr + k_lds_ld_next + v_lds_ld_curr + v_lds_ld_next

    # ========================================================================
    # Main KV loop -- stream tiles [start_tile, n_tiles) through the N_KV_PP ping-pong
    # ring, one tile per iteration. The buffer is selected by the carried curr ds
    # pointers, swapped curr<->next at the end of each `main_loop`.
    #
    # Each `main_loop` call is one tile: it drains outstanding async + barriers (its KV
    # is then GUARANTEED resident -- start_tile from the prologue, every later tile from
    # the previous call's top-of-body bulk prefetch into the OTHER buffer), issues the
    # tile t+1 prefetch UP FRONT into the other buffer, then computes QK->softmax->PV on
    # its own buffer while that async copy runs, drained at the next call's top.
    #
    # TODO(perf): go finer still -- per-write-tile async_load interleaved between the
    # QK/softmax/PV ops (order tuned by thread trace) rather than one bulk burst.
    # ========================================================================
    def main_loop(t, state, *, mask_left, mask_right, kv_len):
        # mask_left/mask_right/kv_len shadow the closure flags: the caller splits the
        # tile stream into a mask-free clean region + boundary loops and passes None for
        # any edge this sub-loop provably doesn't cross (compile-time gate).
        #
        # Runtime ping-pong: this tile reads its curr buffer (carried curr pointers); the
        # tile t+1 prefetch writes the next buffer, and curr<->next are swapped in the yield.
        nxt_pp = (t - start_tile + fx.Int32(1)) % fx.Int32(2)

        kv_tile_start = t * fx.Int32(
            n_block
        )  # this tile's first (batch-relative) kv row

        # Unpack loop-carried state — R independent per-q-tile (m, d, O) groups,
        # then the shared K/V ds pointers.
        m_prev = [fx.Float32(state[qt * _QS + 0]) for qt in range(R)]
        d_prev = [fx.Float32(state[qt * _QS + 1]) for qt in range(R)]
        o_acc = [
            [fx.Vector(state[qt * _QS + 2 + dt]) for dt in range(d_tiles)]
            for qt in range(R)
        ]
        k_curr = list(state[_PTR_BASE + 0 * _NKB : _PTR_BASE + 1 * _NKB])
        k_next = list(state[_PTR_BASE + 1 * _NKB : _PTR_BASE + 2 * _NKB])
        _VB0 = _PTR_BASE + 2 * _NKB
        v_curr = list(state[_VB0 + 0 * _NVB : _VB0 + 1 * _NVB])
        v_next = list(state[_VB0 + 1 * _NVB : _VB0 + 2 * _NVB])

        # Warp-specialized preamble: same pieces, ordered so the SIMD-mate pair (i / i+4)
        # staggers K load vs prefetch around `_named_barrier_pair`. Correctness is
        # warp-type-independent (each wave reads its own resident K under the workgroup
        # barrier); the rendezvous is a perf-only stagger. The READ (load_k_to_reg(k_curr))
        # and ping-pong pointer machinery are UNIFORM; only the global->LDS ISSUE branches
        # by USE_TDM_LOADER (V1 cluster_load_async / V2 TDM copy).
        nxt = t + fx.Int32(1)
        nxt_row0 = nxt * fx.Int32(n_block)
        nxt_valid = _kv_valid(nxt_row0)

        def _addr_phase():
            # Pure (no memory op) -> hoistable: V2 the TDM copy views, V1 the per-lane
            # global/LDS pointer lists, for tile t+1's K/V into the nxt_pp buffer.
            if USE_TDM_LOADER:
                k_views = k_mgr.load_views(
                    ptr_lds=_k_lds_buf(nxt_pp),
                    ptr_K=ptr_K,
                    stride_k_seq=stride_k_seq,
                    stride_k_head=stride_k_head,
                    kv_head=kv_head,
                    kv_row0=kv_start + nxt_row0,
                    kv_valid=nxt_valid,
                )
                v_views = v_mgr.load_views(
                    ptr_lds=_v_lds_buf(nxt_pp),
                    ptr_V=ptr_V,
                    stride_v_seq=stride_v_seq,
                    stride_v_head=stride_v_head,
                    kv_head=kv_head,
                    kv_row0=kv_start + nxt_row0,
                    kv_valid=nxt_valid,
                )
                return (k_views, v_views)
            k_g, k_l, k_i = k_mgr.global_load_ptrs(
                ptr_lds=_k_lds_buf(nxt_pp),
                ptr_K=ptr_K,
                stride_k_seq=stride_k_seq,
                stride_k_head=stride_k_head,
                kv_head=kv_head,
                kv_row0=kv_start + nxt_row0,
                kv_valid=nxt_valid,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
            )
            v_g, v_l, v_i = v_mgr.global_load_ptrs(
                ptr_lds=_v_lds_buf(nxt_pp),
                ptr_V=ptr_V,
                stride_v_seq=stride_v_seq,
                stride_v_head=stride_v_head,
                kv_head=kv_head,
                kv_row0=kv_start + nxt_row0,
                kv_valid=nxt_valid,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
            )
            return (k_g, k_l, k_i, v_g, v_l, v_i)

        def _drain_barrier():
            if USE_TDM_LOADER:
                tdm_ops.tensor_wait(0)
            else:
                rocdl.s_wait_asynccnt(0)
            rocdl.sched_barrier(0)
            gpu.barrier()
            rocdl.sched_barrier(0)

        def _prefetch(addr):
            # Skip t+1 prefetch on the last tile: a dead copy into the O-epilogue slot
            # races the epilogue O write across waves.
            def _issue():
                if USE_TDM_LOADER:
                    k_views, v_views = addr
                    for _v in k_views:
                        fx.copy_atom_call(*_v)
                    for _v in v_views:
                        fx.copy_atom_call(*_v)
                else:
                    k_g, k_l, k_i, v_g, v_l, v_i = addr
                    _async_load_to_lds(k_g, k_l, cluster=True, imm_offs=k_i)
                    _async_load_to_lds(v_g, v_l, cluster=True, imm_offs=v_i)

            scf_if_dispatch(nxt < fx.Int32(n_tiles), _issue)

        # Address VALU up front (no barrier dependency) so it overlaps the drain; only
        # the async issue in _prefetch must stay after the barrier.
        addr = _addr_phase()
        if warp_type == WarpType.LO_WARP:
            _drain_barrier()
            k_values = k_mgr.load_k_to_reg(k_curr)
            _prefetch(addr)
            _named_barrier_pair(warp_idx)
        else:
            _drain_barrier()
            _prefetch(addr)
            _named_barrier_pair(warp_idx)
            k_values = k_mgr.load_k_to_reg(k_curr)

        # Fence the K burst out of the WMMA stream (no wmma<-ds_load bubble). No explicit
        # s_wait_dscnt: the K ds_load is SSA-visible, so mode-2 inserts the dscnt cover
        # itself, placed optimally so the WMMA can issue as soon as its operands land.
        rocdl.sched_barrier(0)

        # ---- GEMM1: S^T = K @ Q^T for this KV tile (== P^T pre-softmax); consumes the
        # pre-loaded k_values (K burst already drained by the preamble above). ----
        s_list = _qk_gemm(
            k_values=k_values,
            q_frags_list=q_frags,
            n_block=n_block,
        )

        # ---- Burst ALL this-tile V transpose ds_loads (out of the WMMA stream), now
        # that QK has consumed the K burst. Issued BEFORE softmax so the ds_load
        # latency hides under softmax's VALU; the values are consumed by _pv_gemm
        # after the rescale. ----
        rocdl.sched_barrier(0)
        v_values = v_mgr.load_v_to_reg(v_curr)
        rocdl.sched_barrier(0)

        # ---- Softmax: online update over this KV tile's kv axis, INDEPENDENTLY per
        # q-tile. This lane's query (tile qt) attends [q_min, q_max] (batch-relative
        # kv): q_max = seq+causal_off+window_right, q_min = seq+causal_off-window_left
        # (causal_off = kv_len-q_len). None bounds are skipped -> a clean-region tile
        # passes all-None and does zero per-element masking. kv_len is passed only on
        # the last tile (folds the OOB tail into the q_max clamp / standalone tail
        # mask). K/V are shared, but each q-tile has its own S and running m/d. ----
        # Softmax for ALL R q-tiles in ONE call so the rows' max/sum tree reductions emit
        # INTERLEAVED (ILP): the rows are independent (own S, m, d) but share this tile's K/V.
        q_max_list = [
            seq_idx[qt] + causal_off + window_right if mask_right else None
            for qt in range(R)
        ]
        q_min_list = [
            seq_idx[qt] + causal_off - window_left if mask_left else None
            for qt in range(R)
        ]
        p_list, m_new_list, d_new_list, corr_list, do_rescale_list = _softmax(
            s_list=s_list,
            m_prev_list=m_prev,
            d_prev_list=d_prev,
            lane_idx=lane_idx,
            n_block=n_block,
            kv_pos_base=kv_tile_start,
            q_max_list=q_max_list,
            q_min_list=q_min_list,
            kv_len=kv_len,
            elem_dtype=elem_dtype,
        )

        # ---- Rescale each q-tile's running O by its corr, then GEMM2 accumulates this
        # tile. When deferral is active (do_rescale is a wave-uniform i1) the wide
        # `o_acc *= corr` multiply (d_tiles*8 f32/lane) is gated behind a non-divergent
        # scf.if that fires only when the running max actually moved; on the stale path
        # corr == 1 so the else-branch passes o_acc through untouched. do_rescale is
        # None -> deferral compiled out, keep the unconditional multiply. Each q-tile
        # decides its own deferred-rescale. ----
        o_resc_list = []
        for qt in range(R):
            corr_vec = fx.Vector.from_elements(
                [corr_list[qt]], fx.Float32
            ).broadcast_to(8)
            o_vecs = [fx.Vector(_ir(o_acc[qt][dt])) for dt in range(d_tiles)]
            if do_rescale_list[qt] is None:
                o_resc = [ov * corr_vec for ov in o_vecs]
            else:
                # Gate the wide multiply behind a wave-uniform scf.if (via the file's
                # scf_if_dispatch idiom): the then-branch rescales, the omitted
                # else-branch auto-passes o_acc through unchanged.
                o_resc = list(
                    scf_if_dispatch(
                        do_rescale_list[qt],
                        lambda *_a, _ov=o_vecs, _cv=corr_vec: [ov * _cv for ov in _ov],
                        result_names=tuple(f"o{qt}_{dt}" for dt in range(d_tiles)),
                        result_values=o_vecs,
                    )
                )
            o_resc_list.append(o_resc)

        o_new_list = _pv_gemm(
            v_values=v_values,
            p_list=p_list,
            v_hdim=v_hdim,
            n_block=n_block,
            o_acc_list=o_resc_list,
        )

        # Yield state — R updated (m, d, O) groups, then the shared K/V ds pointers
        # swapped curr<->next (4 s_swap_b32).
        out = []
        for qt in range(R):
            out += [_raw(m_new_list[qt]), _raw(d_new_list[qt])] + [
                _raw(o) for o in o_new_list[qt]
            ]
        # Swap curr<->next ds bases (manager-defined count each).
        out += k_next + k_curr + v_next + v_curr
        return out

    # ---- Stream tiles [start_tile, n_tiles) through 3 sub-loops split by the attention
    # band so interior tiles fully inside the band skip masking. clean_lo/clean_hi are
    # runtime split points, but the mask on/off per sub-loop is COMPILE-TIME (each loop
    # traces main_loop once with fixed None-ness). Ping-pong swap state threads
    # continuously through all three; buffer parity is by LOCAL iteration index, so the
    # split leaves it intact.
    #   [start_tile, clean_lo) left boundary   (emitted only when mask_left)
    #   [clean_lo,   clean_hi) clean, no mask
    #   [clean_hi,   n_tiles)  right boundary + kv_len tail (last tile)
    n_iter = fx.Int32(n_tiles) - start_tile
    n_last = fx.Int32(n_tiles) - fx.Int32(1)  # last tile always carries the kv_len tail

    # clean_hi = first tile that could need RIGHT masking = the WG's earliest query's
    # diagonal tile ((min q_max + 1)//n_block). Kept <= n_last so the tail tile stays in
    # the right loop, and >= start_tile for a valid partition.
    if mask_right:
        wg_min_seq = (block_x * fx.Int32(BLOCK_M)) // fx.Int32(gqa_ratio)
        qmax_min = fx.max(wg_min_seq + causal_off + window_right, fx.Int32(0))
        clean_hi = (qmax_min + fx.Int32(1)) // fx.Int32(n_block)
    else:
        clean_hi = fx.Int32(n_tiles)
    clean_hi = fx.max(fx.min(clean_hi, n_last), start_tile)

    # clean_lo = first tile fully at/above the WG's latest query's window start
    # (ceildiv(max q_min, n_block)); clamped into [start_tile, clean_hi].
    if mask_left:
        wg_max_seq = fx.min(
            (block_x * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1))
            // fx.Int32(gqa_ratio),
            q_len - fx.Int32(1),
        )
        qmin_max = fx.max(wg_max_seq + causal_off - window_left, fx.Int32(0))
        clean_lo = (qmin_max + fx.Int32(n_block - 1)) // fx.Int32(n_block)
    else:
        clean_lo = start_tile
    clean_lo = fx.min(fx.max(clean_lo, start_tile), clean_hi)

    def _run_tiles(state, lo_i32, hi_i32, *, mask_left, mask_right, kv_len):
        _lo = arith.index_cast(T.index, arith.unwrap(lo_i32))
        _hi = arith.index_cast(T.index, arith.unwrap(hi_i32))
        _step = arith.index(1)
        for _iv, _iargs, _res in scf.for_(_lo, _hi, _step, iter_args=state):
            t0 = fx.Int32(arith.index_cast(T.i32, _iv))
            scf.yield_(
                main_loop(
                    t0,
                    list(_iargs),
                    mask_left=mask_left,
                    mask_right=mask_right,
                    kv_len=kv_len,
                )
            )
        return _res

    state = _init
    if mask_left:
        state = _run_tiles(
            state,
            start_tile,
            clean_lo,
            mask_left=mask_left,
            mask_right=mask_right,
            kv_len=None,
        )
    state = _run_tiles(
        state, clean_lo, clean_hi, mask_left=None, mask_right=None, kv_len=None
    )
    state = _run_tiles(
        state,
        clean_hi,
        fx.Int32(n_tiles),
        mask_left=mask_left,
        mask_right=mask_right,
        kv_len=kv_len,
    )
    final = state

    # ========================================================================
    # Epilogue: normalize O by the running denom d, then reshape+store to VRAM.
    # o_final[dt] lane l elem si = sum_kv P[q,kv] V[kv, dt*16+(l//16)*8+si]
    # (unnormalized); divide by the per-query denom d (peer-consistent across the
    # lane pair) to finish softmax. OManager16b masks rows with seq >= q_len.
    # ========================================================================
    # O staging reuses the NON-CURRENT K|V slot. Local-parity ring: n_iter tiles occupy
    # local indices 0..n_iter-1, so the LAST tile lives in buffer (n_iter-1)%N_KV_PP and
    # the free (non-current) slot is n_iter%N_KV_PP (base 0KB or 128KB). With the last
    # iteration's dead prefetch skipped, no wave ever writes that slot near the end: the
    # last load into it was local tile n_iter-2 (issued during local tile n_iter-3,
    # consumed at n_iter-2), and the top-of-body barrier at local tile n_iter-1 already
    # synchronized every wave past that read. So the slot is idle here -- no cross-wave
    # barrier needed. Only the V1 loader issues async loads (asynccnt); under TDM (V2/V3)
    # K/V/Q load via tensorcnt, so nothing increments asynccnt and s_wait_asynccnt(0) is a
    # pure no-op -- keep it only for V1 as a defensive per-wave WAR guard (retire any
    # still-inflight async load into this slot before O's LDS write). The R q-tiles
    # serialize through the same O ring (s_wait_dscnt(0) between them).
    _OMgr = {"v1": OManager16bV1, "v2": OManager16bV2, "v3": OManager16bV3}[O_VARIANT]
    o_mgr = _OMgr(
        v_hdim=v_hdim,
        gqa_ratio=gqa_ratio,
        num_waves=NUM_WAVES,
        q_tiles_per_wave=R,
        elem_dtype=elem_dtype,
    )
    assert (
        o_mgr.get_lds_size_in_byte() <= slot_bytes
    ), f"O ring budget {o_mgr.get_lds_size_in_byte()}B exceeds K|V slot {slot_bytes}B"
    non_cur_pp = n_iter % fx.Int32(N_KV_PP)
    if not USE_TDM_LOADER:
        rocdl.s_wait_asynccnt(
            0
        )  # V1-only WAR: retire inflight async loads before slot reuse
    # O strides are in ELEMENTS (OManager multiplies by _BF16_BYTES itself). Both V1/V2
    # take ptr_O and build their own store descriptor internally (V1 a bounded buffer
    # resource for the masked buffer_store; V2 the TDM store atom with HW OOB drop).
    o_lds_base = _k_lds_buf(non_cur_pp)
    for qt in range(R):
        # Normalize this q-tile's O by its running denom d, then reshape+store to VRAM.
        # o_final[dt] lane l elem si = sum_kv P[q,kv] V[kv, dt*16+(l//16)*8+si]
        # (unnormalized); divide by the per-query denom d (peer-consistent across the
        # lane pair) to finish softmax. OManager16b masks rows with seq >= q_len.
        d_final = fx.Float32(final[qt * _QS + 1])
        o_final = [fx.Vector(final[qt * _QS + 2 + dt]) for dt in range(d_tiles)]
        # Fully-masked row (d_final==0): 1/0=inf, o_final=0, 0*inf=NaN -> guard to O=0.
        inv = (d_final > fx.Float32(0.0)).select(
            fx.Float32(1.0) / d_final, fx.Float32(0.0)
        )
        inv_vec = fx.Vector.from_elements([inv], fx.Float32).broadcast_to(8)
        # NOTE (mode-2): tying o_final through va_vdst here (to cover the final PV-wmma
        # writeback -> this normalize mul) was MEASURED HARMFUL: 8192nc 1/80 -> 9/80 with
        # the same PV fence present. Either va_vdst doesn't reliably track the wmma
        # writeback or the added drain reshuffles RA into a new race. Left uncovered.
        o_norm = [o_final[dt] * inv_vec for dt in range(d_tiles)]
        if qt > 0:
            rocdl.s_wait_dscnt(0)  # drain prev q-tile's O ring/DS ops before reuse
        o_mgr.store_o_to_vram(
            ptr_O=ptr_O,
            o_base_elems=fx.Int32(0),
            stride_o_seq=stride_o_seq,
            stride_o_head=stride_o_head,
            q_start=q_start,
            q_len=q_len,
            kv_head=kv_head,
            block_x=block_x,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            ptr_lds=o_lds_base,
            o_frags=o_norm,
            qtile=qt,
        )

    # ---- LSE store (optional). LSE = m_final + ln(d_final) in the scaled-score
    # domain (softmax_scale is folded into Q, so S already carries it) — matches
    # torch.logsumexp(scale * Q @ K^T, dim=kv). Each query q = warp*R*16 + qt*16 + l%16
    # is held identically by the lane pair (l, l^16); store once from the khalf==0
    # lanes, masked by seq < q_len. buffer_store redirects mask-drops to byte
    # 0x7FFFFFFF, so lse_rsrc is bounded. Emitted per q-tile.
    if return_lse:
        khalf0 = (lane_idx // fx.Int32(WMMA_M)) == fx.Int32(0)
        lse_rsrc = buffer_ops.create_buffer_resource(
            ptr_LSE, num_records_bytes=lse_num_records_bytes
        )
        for qt in range(R):
            m_final = fx.Float32(final[qt * _QS + 0])
            d_final = fx.Float32(final[qt * _QS + 1])
            # fx.log2 lowers to the HW v_log_f32 (base-2), so scale by ln2 (= 1/LOG2E)
            # to get the natural log for LSE = m + ln(d).
            ln_d = fx.log2(d_final) * fx.Float32(1.0 / LOG2E)
            lse_val = m_final + ln_d
            lse_mask = khalf0 & (seq_idx[qt] < q_len)
            lse_off_el = (
                lse_base_elems
                + seq_idx[qt] * stride_lse_seq
                + q_head_idx[qt] * stride_lse_head
            )
            # Pre-mask the offset (OOB rows -> 0x7fffffff) and pass mask=None so the
            # store maps 1:1 to a single buffer_store with masking already SSA-visible.
            lse_off_masked = lse_mask.select(
                lse_off_el * fx.Int32(4), fx.Int32(0x7FFFFFFF)
            )
            buffer_ops.buffer_store(
                lse_val, lse_rsrc, lse_off_masked, mask=None, offset_is_bytes=True
            )


def _zero_fill_attention(
    *,
    v_hdim,
    gqa_ratio,
    return_lse,
    has_sink,
    ptr_sink,
    ptr_O,
    ptr_LSE,
    stride_o_seq,
    stride_o_head,
    stride_lse_seq,
    stride_lse_head,
    lse_num_records_bytes,
    q_start,
    q_len,
    elem_dtype,
):
    """q_len>0 with kv_len==0 (cross-attention): softmax over an empty KV set, so O=0 for
    this WG's valid query rows. LSE=-inf, or (with a sink) LSE=sink[head] since the only
    surviving softmax term is exp(sink) (sink value is 0, O stays 0). Flat coalesced b128
    write — consecutive lanes write consecutive 16-byte O chunks (no WMMA layout)."""
    tid = _warp_id() * fx.Int32(WAVE_SIZE) + _lane_id()
    kv_head = fx.Int32(gpu.block_id("y"))
    row0 = fx.Int32(gpu.block_id("x")) * fx.Int32(BLOCK_M)
    g = fx.Int32(gqa_ratio)
    _CH = 8  # bf16 per b128 store
    cpr = v_hdim // _CH  # b128 chunks per O row

    # i64: an i32 product (large total_q * stride) can overflow negative, then the
    # descriptor sign-extends it to a huge bound, defeating the 0x7FFFFFFF OOB drop.
    o_num_records_bytes = (
        fx.Int64(q_start + q_len) * fx.Int64(stride_o_seq) * fx.Int64(2)
    )
    o_rsrc = buffer_ops.create_buffer_resource(
        ptr_O, num_records_bytes=o_num_records_bytes
    )
    zero_o = fx.Vector.filled(_CH, 0.0, elem_dtype)
    for r in range(BLOCK_M * cpr // BLOCK_SIZE):
        cix = fx.Int32(r * BLOCK_SIZE) + tid  # flat b128-chunk index this round
        prow = row0 + cix // fx.Int32(cpr)
        d = (cix % fx.Int32(cpr)) * fx.Int32(_CH)
        seq = prow // g
        head = kv_head * g + prow % g
        off = (q_start + seq) * stride_o_seq + head * stride_o_head + d
        off_masked = (seq < q_len).select(off * fx.Int32(2), fx.Int32(0x7FFFFFFF))
        buffer_ops.buffer_store(
            zero_o, o_rsrc, off_masked, mask=None, offset_is_bytes=True
        )

    if return_lse:
        lse_rsrc = buffer_ops.create_buffer_resource(
            ptr_LSE, num_records_bytes=lse_num_records_bytes
        )
        prow = row0 + tid  # one LSE per packed row (BLOCK_SIZE threads == BLOCK_M)
        seq = prow // g
        head = kv_head * g + prow % g
        if has_sink:
            num_heads_q = gpu.grid_dim.y * g
            lse_val = _load_sink_logit(ptr_sink, head, num_heads_q)
        else:
            lse_val = fx.Float32(float("-inf"))
        off = (q_start + seq) * stride_lse_seq + head * stride_lse_head
        off_masked = (seq < q_len).select(off * fx.Int32(4), fx.Int32(0x7FFFFFFF))
        buffer_ops.buffer_store(
            lse_val, lse_rsrc, off_masked, mask=None, offset_is_bytes=True
        )


# ============================================================================
# Builder — one device kernel per (layout, config)
# ============================================================================


@functools.cache
def build_fmha_fwd_prefill_a16w16_m32x8(
    *,
    layout: str = "thd",
    qk_hdim: int = DEFAULT_QK_HDIM,
    v_hdim: int = DEFAULT_V_HDIM,
    n_block: int = DEFAULT_N_BLOCK,
    dtype_str: str = DEFAULT_DTYPE,
    mask_left: bool = False,
    mask_right: bool = False,
    return_lse: bool = False,
    has_sink: bool = False,
    gqa_ratio: int = 1,
):
    """Build the m32x8 device kernel for a given layout + config.

    ``layout`` is ``"thd"`` (varlen) or ``"bshd"`` (batched). Compile-time
    parameters are captured here and baked into the traced kernel. ``gqa_ratio``
    (= ``nheads_q // nheads_kv``) is compile-time so the per-lane ``% / //`` fold
    to shift/and when it is a power of two.
    """
    assert layout in ("thd", "bshd"), f"layout must be thd|bshd, got {layout!r}"
    # qk_hdim in {128,192,256} (D_qk, WMMA_K multiple); v_hdim fixed at 128 (D_v).
    assert (
        qk_hdim in SUPPORTED_QK_HDIM and v_hdim == 128
    ), f"supports qk_hdim in {SUPPORTED_QK_HDIM} with v_hdim==128, got {qk_hdim}/{v_hdim}"
    assert (
        dtype_str in _DTYPE_MAP
    ), f"dtype_str must be in {list(_DTYPE_MAP)}, got {dtype_str!r}"
    ELEM_DTYPE = _DTYPE_MAP[dtype_str]
    assert gqa_ratio >= 1, f"gqa_ratio must be >= 1, got {gqa_ratio}"
    assert (
        n_block in N_BLOCK_CHOICES
    ), f"n_block must be in {N_BLOCK_CHOICES}, got {n_block}"

    QK_HDIM = qk_hdim
    V_HDIM = v_hdim
    N_BLOCK = int(n_block)
    MASK_LEFT = bool(mask_left)
    MASK_RIGHT = bool(mask_right)
    RET_LSE = bool(return_lse)
    HAS_SINK = bool(has_sink)
    GQA_RATIO = int(gqa_ratio)

    if layout == "thd":

        @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
        def kn_fmha_fwd_prefill_a16w16_m32x8_thd(
            ptr_O: fx.Pointer,
            ptr_Q: fx.Pointer,
            ptr_K: fx.Pointer,
            ptr_V: fx.Pointer,
            ptr_LSE: fx.Pointer,
            ptr_sink: fx.Pointer,
            ptr_cu_seqlens_q: fx.Pointer,
            ptr_cu_seqlens_k: fx.Pointer,
            softmax_scale: fx.Float32,
            stride_q_seq: fx.Int32,
            stride_k_seq: fx.Int32,
            stride_v_seq: fx.Int32,
            stride_o_seq: fx.Int32,
            stride_q_head: fx.Int32,
            stride_k_head: fx.Int32,
            stride_v_head: fx.Int32,
            stride_o_head: fx.Int32,
            stride_lse_seq: fx.Int32,
            stride_lse_head: fx.Int32,
            window_left: fx.Int32,
            window_right: fx.Int32,
            max_seqlen_q: fx.Int32,
            max_seqlen_k: fx.Int32,
        ):
            """Varlen THD entry — empty scaffold.

            THD: this batch's token ranges come from cu_seqlens (batch = grid.z).
            """
            batch = fx.Int32(gpu.block_id("z"))
            q_start, q_end = _load_seqlen_pair(ptr_cu_seqlens_q, batch)
            kv_start, kv_end = _load_seqlen_pair(ptr_cu_seqlens_k, batch)
            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            # LSE is [total_q, nheads_q]: base = q_start*stride_lse_seq; every valid
            # element offset is < (q_start+q_len)*stride_lse_seq (< the 0x7FFFFFFF drop).
            lse_base_elems = q_start * stride_lse_seq
            lse_num_records_bytes = (
                fx.Int64(q_start + q_len) * fx.Int64(stride_lse_seq) * fx.Int64(4)
            )

            # An empty batch (no queries OR no keys) must NOT enter the core:
            # kv_len==0 gives an empty softmax denom (d=0) and the epilogue would
            # write O/0 = NaN to that batch's query rows; q_len==0 has no rows to
            # write. Self-attn's kv_len==0 implies q_len==0, so this only skips
            # genuinely empty work. (varlen may carry a per-batch kv_len==0 tail.)
            if (q_len > fx.Int32(0)) & (kv_len > fx.Int32(0)):
                _ca_kw = {
                    "qk_hdim": QK_HDIM,
                    "v_hdim": V_HDIM,
                    "n_block": N_BLOCK,
                    "mask_left": MASK_LEFT,
                    "mask_right": MASK_RIGHT,
                    "return_lse": RET_LSE,
                    "has_sink": HAS_SINK,
                    "gqa_ratio": GQA_RATIO,
                    "ptr_O": ptr_O,
                    "ptr_Q": ptr_Q,
                    "ptr_K": ptr_K,
                    "ptr_V": ptr_V,
                    "ptr_LSE": ptr_LSE,
                    "ptr_sink": ptr_sink,
                    "softmax_scale": softmax_scale,
                    "stride_q_seq": stride_q_seq,
                    "stride_k_seq": stride_k_seq,
                    "stride_v_seq": stride_v_seq,
                    "stride_o_seq": stride_o_seq,
                    "stride_q_head": stride_q_head,
                    "stride_k_head": stride_k_head,
                    "stride_v_head": stride_v_head,
                    "stride_o_head": stride_o_head,
                    "stride_lse_seq": stride_lse_seq,
                    "stride_lse_head": stride_lse_head,
                    "lse_base_elems": lse_base_elems,
                    "lse_num_records_bytes": lse_num_records_bytes,
                    "q_start": q_start,
                    "q_len": q_len,
                    "kv_start": kv_start,
                    "kv_len": kv_len,
                    "window_left": window_left,
                    "window_right": window_right,
                    "elem_dtype": ELEM_DTYPE,
                }
                # Warp specialization: LO warp (waves 0..N/2-1) vs HI warp (N/2..N-1).
                lds_base = _alloc_lds()
                warp_idx = _warp_id()
                if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):
                    _core_attention(
                        warp_idx=warp_idx,
                        warp_type=WarpType.LO_WARP,
                        lds_base=lds_base,
                        **_ca_kw,
                    )
                else:
                    _core_attention(
                        warp_idx=warp_idx,
                        warp_type=WarpType.HI_WARP,
                        lds_base=lds_base,
                        **_ca_kw,
                    )
            elif q_len > fx.Int32(0):
                # Cross-attention tail: q_len>0 but kv_len==0 -> O=0, LSE=-inf (or sink).
                _zero_fill_attention(
                    v_hdim=V_HDIM,
                    gqa_ratio=GQA_RATIO,
                    return_lse=RET_LSE,
                    has_sink=HAS_SINK,
                    ptr_sink=ptr_sink,
                    ptr_O=ptr_O,
                    ptr_LSE=ptr_LSE,
                    stride_o_seq=stride_o_seq,
                    stride_o_head=stride_o_head,
                    stride_lse_seq=stride_lse_seq,
                    stride_lse_head=stride_lse_head,
                    lse_num_records_bytes=lse_num_records_bytes,
                    q_start=q_start,
                    q_len=q_len,
                    elem_dtype=ELEM_DTYPE,
                )

        return kn_fmha_fwd_prefill_a16w16_m32x8_thd

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def kn_fmha_fwd_prefill_a16w16_m32x8_bshd(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        stride_lse_batch: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
    ):
        """Batched BSHD entry — empty scaffold.

        Uniform sequence lengths (``seq_len_q`` / ``seq_len_k``) replace
        cu_seqlens — nothing transient, so this path is CUDA-graph safe.
        Token base is batch_idx * seq_len (batch = grid.z).
        """
        batch = fx.Int32(gpu.block_id("z"))

        # LSE is [B, nheads_q, seq_q]: base = batch*stride_lse_batch; every valid
        # element offset is < base + stride_lse_batch (< the 0x7FFFFFFF drop).
        lse_base_elems = batch * stride_lse_batch
        lse_num_records_bytes = fx.Int64(lse_base_elems + stride_lse_batch) * fx.Int64(
            4
        )

        _ca_kw = {
            "qk_hdim": QK_HDIM,
            "v_hdim": V_HDIM,
            "n_block": N_BLOCK,
            "mask_left": MASK_LEFT,
            "mask_right": MASK_RIGHT,
            "return_lse": RET_LSE,
            "has_sink": HAS_SINK,
            "gqa_ratio": GQA_RATIO,
            "ptr_O": ptr_O,
            "ptr_Q": ptr_Q,
            "ptr_K": ptr_K,
            "ptr_V": ptr_V,
            "ptr_LSE": ptr_LSE,
            "ptr_sink": ptr_sink,
            "softmax_scale": softmax_scale,
            "stride_q_seq": stride_q_seq,
            "stride_k_seq": stride_k_seq,
            "stride_v_seq": stride_v_seq,
            "stride_o_seq": stride_o_seq,
            "stride_q_head": stride_q_head,
            "stride_k_head": stride_k_head,
            "stride_v_head": stride_v_head,
            "stride_o_head": stride_o_head,
            "stride_lse_seq": stride_lse_seq,
            "stride_lse_head": stride_lse_head,
            "lse_base_elems": lse_base_elems,
            "lse_num_records_bytes": lse_num_records_bytes,
            "q_start": batch * seq_len_q,
            "q_len": seq_len_q,
            "kv_start": batch * seq_len_k,
            "kv_len": seq_len_k,
            "window_left": window_left,
            "window_right": window_right,
            "elem_dtype": ELEM_DTYPE,
        }
        # Warp specialization: LO warp (waves 0..N/2-1) vs HI warp (N/2..N-1).
        lds_base = _alloc_lds()
        warp_idx = _warp_id()
        if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):
            _core_attention(
                warp_idx=warp_idx,
                warp_type=WarpType.LO_WARP,
                lds_base=lds_base,
                **_ca_kw,
            )
        else:
            _core_attention(
                warp_idx=warp_idx,
                warp_type=WarpType.HI_WARP,
                lds_base=lds_base,
                **_ca_kw,
            )

    return kn_fmha_fwd_prefill_a16w16_m32x8_bshd


# ============================================================================
# Launch wrappers + host entries
# ============================================================================

_launch_fns = (
    {}
)  # {(layout, mask_left, mask_right, return_lse, has_sink, gqa_ratio): fn}


def _ensure_thd_kernel(
    mask_left: bool,
    mask_right: bool,
    return_lse: bool,
    has_sink: bool,
    gqa_ratio: int,
    qk_hdim: int = DEFAULT_QK_HDIM,
    dtype_str: str = DEFAULT_DTYPE,
):
    key = (
        "thd",
        bool(mask_left),
        bool(mask_right),
        bool(return_lse),
        bool(has_sink),
        int(gqa_ratio),
        int(qk_hdim),
        str(dtype_str),
    )
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_a16w16_m32x8(
        layout="thd",
        qk_hdim=qk_hdim,
        mask_left=mask_left,
        mask_right=mask_right,
        return_lse=return_lse,
        has_sink=has_sink,
        gqa_ratio=gqa_ratio,
        dtype_str=dtype_str,
    )

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        ptr_cu_seqlens_q: fx.Pointer,
        ptr_cu_seqlens_k: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = 256 (8 waves x wave32).
        grid_x = arith.index_cast(
            T.index,
            arith.ceildivui(
                arith.unwrap(max_seqlen_q * gqa_ratio),
                arith.constant(BLOCK_M, type=T.i32),
            ),
        )
        grid_y = arith.index_cast(T.index, num_heads_kv)
        grid_z = arith.index_cast(T.index, batch_size)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_sink,
            ptr_cu_seqlens_q,
            ptr_cu_seqlens_k,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            stride_lse_seq,
            stride_lse_head,
            window_left,
            window_right,
            max_seqlen_q,
            max_seqlen_k,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": ENABLE_SCHED_MODE2,
        # "amdgpu-sched-strategy": "coexec",  # gfx1250 co-exec sched; revisit after named barrier
    }
    _launch.compile_hints["waves_per_eu"] = 2
    _launch_fns[key] = _launch


def _ensure_bshd_kernel(
    mask_left: bool,
    mask_right: bool,
    return_lse: bool,
    has_sink: bool,
    gqa_ratio: int,
    qk_hdim: int = DEFAULT_QK_HDIM,
    dtype_str: str = DEFAULT_DTYPE,
):
    key = (
        "bshd",
        bool(mask_left),
        bool(mask_right),
        bool(return_lse),
        bool(has_sink),
        int(gqa_ratio),
        int(qk_hdim),
        str(dtype_str),
    )
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_a16w16_m32x8(
        layout="bshd",
        qk_hdim=qk_hdim,
        mask_left=mask_left,
        mask_right=mask_right,
        return_lse=return_lse,
        has_sink=has_sink,
        gqa_ratio=gqa_ratio,
        dtype_str=dtype_str,
    )

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        stride_lse_batch: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = 256 (8 waves x wave32).
        grid_x = arith.index_cast(
            T.index,
            arith.ceildivui(
                arith.unwrap(seq_len_q * gqa_ratio),
                arith.constant(BLOCK_M, type=T.i32),
            ),
        )
        grid_y = arith.index_cast(T.index, num_heads_kv)
        grid_z = arith.index_cast(T.index, batch_size)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_sink,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            stride_lse_seq,
            stride_lse_head,
            stride_lse_batch,
            window_left,
            window_right,
            seq_len_q,
            seq_len_k,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": ENABLE_SCHED_MODE2,
        # "amdgpu-sched-strategy": "coexec",  # gfx1250 co-exec sched; revisit after named barrier
    }
    _launch.compile_hints["waves_per_eu"] = 2
    _launch_fns[key] = _launch


def flash_attn_varlen_m32x8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    out=None,
    return_lse=False,
    sink=None,
    lse=None,
):
    """Host entry — varlen THD, qk_hdim in {128,192,256} / v_hdim=128, bf16 or fp16.

    ``window_size`` (optional): ``(left, right)`` sliding-window bounds. ``-1`` =
    infinite on that side; ``(-1, -1)`` = full attention. ``causal`` forces
    ``right=0``. Finiteness is baked into the kernel (compile-time ``mask_left`` /
    ``mask_right``); the window magnitudes are runtime args, so one variant serves
    any window value.

    ``sink`` (optional): 1-D ``[nheads_q]`` fp32 per-head sink logits in the
    scaled-score domain — one extra ``exp(sink)`` term in the softmax denominator.
    Presence is baked into the kernel at compile time (``has_sink``).

    ``lse`` (optional): caller-provided ``[total_q, nheads_q]`` fp32 output buffer,
    used only when ``return_lse``; allocated here when ``return_lse`` and None.
    """
    assert q.dtype in _TORCH_DTYPE_MAP.values(), f"Expected bf16 or fp16, got {q.dtype}"
    assert (
        k.dtype == q.dtype and v.dtype == q.dtype
    ), f"q/k/v dtype must match, got {q.dtype}/{k.dtype}/{v.dtype}"
    dtype_str = "bf16" if q.dtype == torch.bfloat16 else "fp16"
    qk_hdim = q.shape[-1]
    assert (
        qk_hdim in SUPPORTED_QK_HDIM
    ), f"Expected qk_hdim in {SUPPORTED_QK_HDIM}, got {qk_hdim}"
    assert v.shape[-1] == 128, f"Expected v_hdim=128, got {v.shape[-1]}"

    total_q_tokens = q.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    nheads_q = q.shape[1]
    nheads_k = k.shape[1]
    assert (
        nheads_q % nheads_k == 0
    ), f"nheads_q={nheads_q} must be a multiple of nheads_k={nheads_k}"
    gqa = nheads_q // nheads_k

    has_sink = sink is not None
    if has_sink:
        assert sink.dtype == torch.float32, f"sink must be fp32, got {sink.dtype}"
        assert (
            sink.dim() == 1 and sink.shape[0] == nheads_q
        ), f"sink must be [nheads_q={nheads_q}], got {tuple(sink.shape)}"
    # ptr_sink is only read when has_sink; pass q as a valid placeholder otherwise.
    sink_ptr = sink if has_sink else q

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    # Sliding window: causal forces right=0. Finiteness (>=0) is compile-time
    # (mask_left/mask_right); the magnitudes ride along as runtime Int32 args.
    win_left, win_right = int(window_size[0]), int(window_size[1])
    if causal:
        win_right = 0
    mask_left = win_left >= 0
    mask_right = win_right >= 0
    window_left = max(win_left, 0)
    window_right = max(win_right, 0)

    if out is None:
        out = torch.empty(
            (total_q_tokens, nheads_q, 128), dtype=q.dtype, device=q.device
        )
    if return_lse:
        if lse is None:
            lse = torch.empty(
                (total_q_tokens, nheads_q), dtype=torch.float32, device=q.device
            )
        lse_ptr = lse
        stride_lse_seq = lse.stride(0)
        stride_lse_head = lse.stride(1)
    else:
        lse_ptr = q
        stride_lse_seq = 0
        stride_lse_head = 0

    # Q/K/V/O strides in ELEMENTS (TDM loaders consume them directly).
    stride_q_seq = q.stride(0)
    stride_k_seq = k.stride(0)
    stride_v_seq = v.stride(0)
    stride_o_seq = out.stride(0)
    stride_q_head = q.stride(1)
    stride_k_head = k.stride(1)
    stride_v_head = v.stride(1)
    stride_o_head = out.stride(1)

    _ensure_thd_kernel(
        mask_left,
        mask_right,
        bool(return_lse),
        has_sink,
        gqa,
        qk_hdim=qk_hdim,
        dtype_str=dtype_str,
    )

    _run_compiled(
        _launch_fns[
            (
                "thd",
                mask_left,
                mask_right,
                bool(return_lse),
                has_sink,
                gqa,
                qk_hdim,
                dtype_str,
            )
        ],
        out,
        q,
        k,
        v,
        lse_ptr,
        sink_ptr,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        stride_q_seq,
        stride_k_seq,
        stride_v_seq,
        stride_o_seq,
        stride_q_head,
        stride_k_head,
        stride_v_head,
        stride_o_head,
        stride_lse_seq,
        stride_lse_head,
        window_left,
        window_right,
        max_seqlen_q,
        max_seqlen_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out


def flash_attn_batch_m32x8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    out=None,
    return_lse=False,
    sink=None,
    lse=None,
):
    """Host entry — batched BSHD ``[B, S, H, D]``, qk_hdim in {128,192,256} / v_hdim=128, bf16 or fp16.

    Uses the dedicated BSHD kernel with a uniform ``seq_len`` scalar (no
    cu_seqlens), so there is nothing transient to bake into a CUDA graph.

    ``window_size`` (optional): ``(left, right)`` sliding-window bounds. ``-1`` =
    infinite on that side; ``(-1, -1)`` = full attention. ``causal`` forces
    ``right=0``. Finiteness is baked into the kernel (compile-time ``mask_left`` /
    ``mask_right``); the window magnitudes are runtime args.

    ``sink`` (optional): 1-D ``[nheads_q]`` fp32 per-head sink logits in the
    scaled-score domain — one extra ``exp(sink)`` term in the softmax denominator.
    Presence is baked into the kernel at compile time (``has_sink``).

    ``lse`` (optional): caller-provided ``[B, nheads_q, S_q]`` fp32 output buffer,
    used only when ``return_lse``; allocated here when ``return_lse`` and None.
    """
    assert q.dtype in _TORCH_DTYPE_MAP.values(), f"Expected bf16 or fp16, got {q.dtype}"
    assert (
        k.dtype == q.dtype and v.dtype == q.dtype
    ), f"q/k/v dtype must match, got {q.dtype}/{k.dtype}/{v.dtype}"
    dtype_str = "bf16" if q.dtype == torch.bfloat16 else "fp16"
    assert q.dim() == 4, f"Expected 4D BSHD tensor, got rank {q.dim()}"
    qk_hdim = q.shape[-1]
    assert (
        qk_hdim in SUPPORTED_QK_HDIM
    ), f"Expected qk_hdim in {SUPPORTED_QK_HDIM}, got {qk_hdim}"
    assert v.shape[-1] == 128, f"Expected v_hdim=128, got {v.shape[-1]}"

    batch, seq_len_q, nheads_q, _ = q.shape
    seq_len_k = k.shape[1]
    nheads_k = k.shape[2]
    assert (
        nheads_q % nheads_k == 0
    ), f"nheads_q={nheads_q} must be a multiple of nheads_k={nheads_k}"
    gqa = nheads_q // nheads_k

    has_sink = sink is not None
    if has_sink:
        assert sink.dtype == torch.float32, f"sink must be fp32, got {sink.dtype}"
        assert (
            sink.dim() == 1 and sink.shape[0] == nheads_q
        ), f"sink must be [nheads_q={nheads_q}], got {tuple(sink.shape)}"
    # ptr_sink is only read when has_sink; pass q as a valid placeholder otherwise.
    sink_ptr = sink if has_sink else q

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    # Sliding window: causal forces right=0. Finiteness (>=0) is compile-time
    # (mask_left/mask_right); the magnitudes ride along as runtime Int32 args.
    win_left, win_right = int(window_size[0]), int(window_size[1])
    if causal:
        win_right = 0
    mask_left = win_left >= 0
    mask_right = win_right >= 0
    window_left = max(win_left, 0)
    window_right = max(win_right, 0)

    if out is None:
        out = torch.empty(
            (batch, seq_len_q, nheads_q, 128), dtype=q.dtype, device=q.device
        )
    if return_lse:
        if lse is None:
            lse = torch.empty(
                (batch, nheads_q, seq_len_q), dtype=torch.float32, device=q.device
            )
        lse_ptr = lse
        stride_lse_seq = lse.stride(2)
        stride_lse_head = lse.stride(1)
        stride_lse_batch = lse.stride(0)
    else:
        lse_ptr = q
        stride_lse_seq = 0
        stride_lse_head = 0
        stride_lse_batch = 0

    # Empty tensor — skip the launch (host-known dims, no device sync). No queries: out
    # has no rows to write. No keys (seq_len_k==0, seq_len_q>0): softmax over an empty KV
    # set -> O=0. LSE=-inf, or (with a sink) LSE=sink[head] since the only surviving
    # softmax term is exp(sink) (sink value is 0, so O stays 0).
    if seq_len_q == 0 or seq_len_k == 0:
        if seq_len_q > 0 and seq_len_k == 0:
            out.zero_()
            if return_lse:
                if sink is not None:
                    lse.copy_(
                        sink.to(device=lse.device, dtype=lse.dtype)
                        .view(1, -1, 1)
                        .expand_as(lse)
                    )
                else:
                    lse.fill_(float("-inf"))
        return (out, lse) if return_lse else out

    # BSHD: seq is dim 1, head dim 2 — the per-batch base is derived in-kernel as
    # batch_idx * seq_len. Q/K/V/O strides in ELEMENTS (TDM loaders consume directly).
    stride_q_seq = q.stride(1)
    stride_k_seq = k.stride(1)
    stride_v_seq = v.stride(1)
    stride_o_seq = out.stride(1)
    stride_q_head = q.stride(2)
    stride_k_head = k.stride(2)
    stride_v_head = v.stride(2)
    stride_o_head = out.stride(2)

    _ensure_bshd_kernel(
        mask_left,
        mask_right,
        bool(return_lse),
        has_sink,
        gqa,
        qk_hdim=qk_hdim,
        dtype_str=dtype_str,
    )

    _run_compiled(
        _launch_fns[
            (
                "bshd",
                mask_left,
                mask_right,
                bool(return_lse),
                has_sink,
                gqa,
                qk_hdim,
                dtype_str,
            )
        ],
        out,
        q,
        k,
        v,
        lse_ptr,
        sink_ptr,
        softmax_scale,
        stride_q_seq,
        stride_k_seq,
        stride_v_seq,
        stride_o_seq,
        stride_q_head,
        stride_k_head,
        stride_v_head,
        stride_o_head,
        stride_lse_seq,
        stride_lse_head,
        stride_lse_batch,
        window_left,
        window_right,
        seq_len_q,
        seq_len_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out
