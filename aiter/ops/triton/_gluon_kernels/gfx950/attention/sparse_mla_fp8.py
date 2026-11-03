# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950) sparse MLA on a per-tensor fp8 KV cache ("fp8_scalar") with fp8
dots. sparse_mla_fwd (attention/sparse_mla.py) launches it for every call it can
address (_sparse_mla_fp8_launch); every other call runs the generic kernel in
sparse_mla.py, whose split merge (_sparse_mla_reduce) and output store
(_epilogue_store) these calls reuse.

Geometry (ROPE):
    0   one 512-byte row per key, V = the whole row: DeepSeek-V4 / V4.1 (RoPE inside
        the row, per-head sink) and GLM-5.3-Flash (no RoPE, no sink)
    64  512 latent bytes + a 64-byte RoPE tail per key, V = the latent (GLM-5.2);
        QK adds one 16x16x32 MFMA chain over the RoPE dims
q is fp8 with one scale for the tensor, or bf16 quantized in the prologue (one
scale per head). The cache scale folds into the score scale (K side) and the final
normalization (V side).

A program is one query row, G heads (8..64) and one split of the row's keys. G and
the wave count NW fix the layouts (_layouts), and the layouts fix how the waves
share the work, so there is one loop body for every head count:
    QK  waves = HW head groups x KG key groups, HW = min(NW, GM / 16), KG = NW / HW
        (GM = max(G, 16) MFMA rows; G = 8 pads to 16)
    PV  waves split the 512 output dims
With KG = 1 (64 heads on 4 waves) a wave owns whole rows: QK runs as two 32-key
halves, since a whole tile's K operand would be 128 registers, and the row sum is
reduced in-wave every tile. With KG > 1 the row sum is deferred (a [GM, 64]
accumulator, reduced once).

Per tile, from a 2-slot LDS ring of [64, 512] fp8 tiles that direct-to-LDS gathers
fill one tile ahead:
    wait tile t | gather t + 1 | row ids of t + 2 | QK | mask | softmax |
    P, alpha -> LDS, barrier | rescale if a row max moved | PV
The softmax runs in base 2 with a lagging row max: it moves only when a score
beats it by more than 2^LAZY, so most tiles skip the accumulator rescale (LAZY = 0
is the exact max). P goes to PV as e4m3, scaled by 2^LOG2_PS = 448 / 2^LAZY, so it
reaches the format's largest value even with the max lagging; the scale is folded
into the subtracted max, so p is exp2 of one FMA, and cancels in the normalization.

Key lists (SEG):
    0  one list of pool row ids, a flat [rows, 512 + ROPE] pool
    1  paged caches (slot s is row o + (s >> b) * stride + (s & (2^b - 1))) with
       up to two lists, DeepSeek-V4's SWA window (main) and top-k (extra), read as
       one list in one loop: both caches in one allocation, under one scale
    2  the same two lists from two allocations or under two scales: two loops
       carrying (acc, m, l), the second in the extra cache's V units
The kernel takes no -1 sentinels in the lists; the caller sends those to the
generic kernel.

Split-K programs write the generic kernel's partials (base-2 row max, row sum, bf16
unnormalized acc in V units) for _sparse_mla_reduce. The output store can apply
DeepSeek-V4's inverse RoPE and MXFP8 quantization, here or in the merge.
"""

from dataclasses import dataclass

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import _core as _gl_core

from aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_mla import (
    _epilogue_store,
)

LOG2E: gl.constexpr = gl.constexpr(1.4426950408889634)
_LOG2_448 = 8.807354922057604  # log2 of e4m3's largest value
BN: gl.constexpr = gl.constexpr(64)  # keys per tile
D: gl.constexpr = gl.constexpr(512)  # latent width = V width

# Triton 3.8 spells shared_memory_descriptor.reinterpret as _reinterpret and has
# no gl.amd.hint.sched_barrier; both settled at compile time.
_HAS_REINTERPRET = gl.constexpr(
    hasattr(_gl_core.shared_memory_descriptor, "reinterpret")
)
_HAS_SCHED_BARRIER = gl.constexpr(
    hasattr(gl.amd, "hint") and hasattr(gl.amd.hint, "sched_barrier")
)
# No AGPRs. Triton 3.9's LLVM infers this; 3.8's does not for kernels with async
# copies and splits the budget into 128 VGPRs + 128 AGPRs at 2 waves/SIMD.
_FN_ATTRS = "amdgpu-agpr-alloc=0"


@gluon.jit
def _sched_barrier():
    if _HAS_SCHED_BARRIER:
        gl.amd.hint.sched_barrier()


@gluon.jit
def _relayout(smem, layout: gl.constexpr):
    """The same LDS bytes under another shared layout."""
    if _HAS_REINTERPRET:
        r = smem.reinterpret(smem.dtype, smem.shape, layout)
    else:
        r = smem._reinterpret(smem.dtype, smem.shape, layout)
    return r


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Layouts:
    GM: int  # MFMA rows, max(G, 16)
    KG: int  # QK key groups; 1: a wave owns whole rows
    QK: object  # 16x16x128 QK MFMA
    QR: object  # 16x16x32 MFMA over the RoPE dims
    PV: object  # PV MFMA, waves on the 512 output dims
    PV_SCALED: bool  # PV is a 32x32x64 scaled MFMA, else a 16x16x32 one
    KW_PV: int  # PV operands' k_width
    GATHER: object  # [64, 512] tile gather
    GATHER_R: object  # [64, ROPE] RoPE tail gather
    K_SH: object  # ring tile as written, and read as K
    V_SH: object  # ring tile read as V^T
    KR_SH: object  # RoPE tile
    P_SH: object  # P as the softmax writes it
    P_PV_SH: object  # P as PV reads it


@gluon.constexpr_function
def _layouts(G, NW, ROPE):
    """Every layout of a program with G heads on NW waves. Compile time only."""
    GM = max(G, 16)
    HW = min(NW, GM // 16)
    KG = NW // HW
    qk = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [HW, KG])
    qr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [HW, KG])
    # The ring tile's LDS image: a [64, 512] tile is 15 address bits (key, dim).
    # Bits 0-3 are a 16 B run of dims, bits 4-9 one 1 KB block, the rest pick the
    # block, with a pad after every KB. The orders below make the 32-lane passes
    # of the K reads (ds_read_b128) and the V^T reads (ds_read_b64_tr_b8) hit
    # distinct banks.
    if GM >= 32:
        # 32x32x64 scaled PV; key3 is block bit 3, moved 256 B by the 32 B pad
        pv = gl.amd.AMDMFMALayout(
            4, [32, 32, 64], True, [1, NW], tiles_per_warp=[1, 512 // 32 // NW]
        )
        kw_pv = 16
        pv_scaled = True
        pad = 32
        kv = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [0, 32],
              [0, 64], [0, 128], [0, 256], [16, 0], [8, 0], [32, 0]]  # fmt: skip
        vv = kv
        pp = None
    else:
        # 16x16x32 PV, whose V^T pass varies key3: V and P are read through views
        # with key bits 3 and 4 swapped (PV sums over keys, so a key permutation
        # common to P and V is exact)
        pv = gl.amd.AMDMFMALayout(
            4, [16, 16, 32], True, [1, NW], tiles_per_warp=[1, 512 // 16 // NW]
        )
        kw_pv = 8
        pv_scaled = False
        pad = 16
        kv = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [0, 32],
              [0, 64], [16, 0], [0, 128], [0, 256], [32, 0], [8, 0]]  # fmt: skip
        vv = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [0, 32],
              [0, 64], [8, 0], [0, 128], [0, 256], [32, 0], [16, 0]]  # fmt: skip
        pp = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 8], [0, 32]]
        pp = pp + [[1 << i, 0] for i in range(GM.bit_length() - 1)]
    k_sh = gl.PaddedSharedLayout([[1024, pad]], kv, [], [64, 512])
    v_sh = gl.PaddedSharedLayout([[1024, pad]], vv, [], [64, 512])
    p_sh = gl.PaddedSharedLayout.with_identity_for([[64, 16]], [GM, 64], [1, 0])
    if pp is None:
        p_pv_sh = p_sh
    else:
        p_pv_sh = gl.PaddedSharedLayout([[64, 16]], pp, [], [GM, 64])
    # Gather: a lane per 16 B of one 1 KB block (the copy writes each wave's
    # 64 x 16 B contiguously), the block bits on registers and waves. Dims before
    # keys on registers: register order is issue order, and a key bit early makes
    # the next gather wait on every gather in flight.
    nwb = NW.bit_length() - 1
    block = sorted(kv[10:], key=lambda b: b[0] != 0)
    ga = gl.DistributedLinearLayout(
        reg_bases=kv[:4] + block[: 5 - nwb],
        lane_bases=kv[4:10],
        warp_bases=block[5 - nwb :],
        block_bases=[],
        shape=[64, 512],
    )
    kr_sh = k_sh
    gr = ga
    if ROPE > 0:
        assert ROPE == 64 and KG > 1, "the RoPE tile runs with key-split QK"
        rb = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0], [0, 16],
              [0, 32], [16, 0], [32, 0]]  # fmt: skip
        kr_sh = gl.PaddedSharedLayout([[1024, 16]], rb, [], [64, ROPE])
        nr = min(nwb, 2)
        gr = gl.DistributedLinearLayout(
            reg_bases=rb[:4] + rb[10:][: 2 - nr],
            lane_bases=rb[4:10],
            warp_bases=rb[10:][2 - nr :] + [[0, 0]] * (nwb - nr),
            block_bases=[],
            shape=[64, ROPE],
        )
    return _Layouts(
        GM=GM, KG=KG, QK=qk, QR=qr, PV=pv, PV_SCALED=pv_scaled, KW_PV=kw_pv,
        GATHER=ga, GATHER_R=gr, K_SH=k_sh, V_SH=v_sh, KR_SH=kr_sh, P_SH=p_sh,
        P_PV_SH=p_pv_sh,
    )  # fmt: skip


@gluon.constexpr_function
def _log2_ps(LAZY):
    """log2 of the P scale: a score at most 2^LAZY above the lagging max gives
    P = 448, e4m3's largest value."""
    return _LOG2_448 - LAZY


@gluon.constexpr_function
def _inv_ps(LAZY):
    return 2.0 ** (LAZY - _LOG2_448)


# ---------------------------------------------------------------------------
# Key lists and gathers
#
# A key list (ix) is one of
#   (ids,)                 a flat pool's row ids
#   (main, extra, n_main)  two lists of paged slot ids read as their
#                          concatenation, main for its first n_main keys; a list
#                          is (ids, o, stride, b), slot s being row
#                          o + (s >> b) * stride + (s & (2^b - 1))
# The code dispatches on len(ix): tuple members lose constexpr-ness in a callee.
# ---------------------------------------------------------------------------


@gluon.jit
def _key_lists(
    indptr_ptr,
    idx_ptr,
    x_indptr_ptr,
    x_idx_ptr,
    row,
    mo,
    mstr,
    xo,
    xstr,
    SEG: gl.constexpr,
    MB_SH: gl.constexpr,
    XB_SH: gl.constexpr,
):
    """(start, end, ix) of a row read in one loop (SEG 0 / 1). start and end are
    the positions the concatenated list would have (masks and splits only)."""
    start = gl.load(indptr_ptr + row)
    end = gl.load(indptr_ptr + row + 1)
    if SEG == 0:
        ix = (idx_ptr + start,)
    else:
        main = (idx_ptr + start, mo, mstr, MB_SH)
        if x_indptr_ptr is None:  # one paged list
            ix = (main, main, end - start)
        else:
            es = gl.load(x_indptr_ptr + row)
            ee = gl.load(x_indptr_ptr + row + 1)
            ix = (main, (x_idx_ptr + es, xo, xstr, XB_SH), end - start)
            start = start + es
            end = end + ee
    return start, end, ix


@gluon.jit
def _rows(ix, k, n, KL: gl.constexpr):
    """Start loading the row ids of keys k .. k + 64 (relative to the list start);
    keys past n - 1 clamp onto it. Returns what _rid resolves after the wait that
    covers these loads. Flat ids load straight in the gather's key layout KL;
    paged ones one key per lane, mapped and spread to KL by _rid: two loads plus
    the mapping per gathered register cost more than the loop has to spare."""
    if len(ix) > 1:
        cl: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
        j = gl.minimum(k + gl.arange(0, BN, layout=cl), n - 1)
        n_main = ix[2]
        # no `other`: a masked buffer load reads 0, while other=0 is a select right
        # after the load, an s_waitcnt vmcnt(0) behind the gathers in flight
        sm = gl.amd.cdna4.buffer_load(ix[0][0], j, mask=j < n_main)
        sx = gl.amd.cdna4.buffer_load(ix[1][0], j - n_main, mask=j >= n_main)
        r = (sm, sx, k, n)
    else:
        kg = gl.arange(0, BN, layout=KL)
        r = (gl.amd.cdna4.buffer_load(ix[0], gl.minimum(k + kg, n - 1)),)
    return r


@gluon.jit
def _slot_row(s, lst):
    """Row of paged slot s of list lst = (ids, o, stride, b)."""
    return lst[1] + (s >> lst[3]) * lst[2] + (s & ((1 << lst[3]) - 1))


@gluon.jit
def _rid(ix, r, KL: gl.constexpr):
    """The row ids in layout KL, from _rows' r."""
    if len(ix) > 1:
        # The ids are lane-uniform: left alone, the backend moves them to SGPRs
        # (v_readlane) right after the load, an s_waitcnt behind the gathers in
        # flight. A non-pure VGPR copy keeps that math after the step's wait.
        sm = gl.inline_asm_elementwise(
            "v_mov_b32 $0, $1", "=v,v", [r[0]], dtype=gl.int32, is_pure=False, pack=1
        )
        sx = gl.inline_asm_elementwise(
            "v_mov_b32 $0, $1", "=v,v", [r[1]], dtype=gl.int32, is_pure=False, pack=1
        )
        k = r[2]
        n = r[3]
        kc = gl.arange(0, BN, layout=sm.type.layout)
        in_main = gl.minimum(k + kc, n - 1) < ix[2]
        rid = gl.where(in_main, _slot_row(sm, ix[0]), _slot_row(sx, ix[1]))
        rid = gl.convert_layout(rid, KL)
    else:
        rid = r[0]
    return rid


@gluon.jit
def _copy(dst, kv_ptr, rid, cols, RS: gl.constexpr, OFF: gl.constexpr, BIG: gl.constexpr):
    """Bytes OFF + cols of rows rid (row pitch RS) into dst, no commit."""
    if BIG:
        # pools past 2 GB: a 64-bit row base (buffer offsets are i32)
        rp = kv_ptr + rid.to(gl.int64) * RS + OFF
        gl.amd.cdna4.async_copy.global_load_to_shared(dst, rp[:, None] + cols[None, :])
    else:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dst, kv_ptr, (rid * RS + OFF)[:, None] + cols[None, :]
        )


@gluon.jit
def _gather(
    kv_dst, kr_dst, kv_ptr, ix, r, cols, G: gl.constexpr, ROPE: gl.constexpr,
    BIG: gl.constexpr,
):  # fmt: skip
    """One tile (_rows' r of list ix) into a ring slot: the latent into kv_dst, the
    RoPE tail into kr_dst; cols: their byte columns in the gather layouts. One
    commit group."""
    L: gl.constexpr = _layouts(G, gl.num_warps(), ROPE)
    rid = _rid(ix, r, gl.SliceLayout(1, L.GATHER))
    _copy(kv_dst, kv_ptr, rid, cols[0], D + ROPE, 0, BIG)
    if ROPE > 0:
        rid_r = gl.convert_layout(rid, gl.SliceLayout(1, L.GATHER_R))
        _copy(kr_dst, kv_ptr, rid_r, cols[1], D + ROPE, D, BIG)
    gl.amd.cdna4.async_copy.commit_group()


# ---------------------------------------------------------------------------
# Q
# ---------------------------------------------------------------------------


@gluon.jit
def _load_q(
    q_src, C0: gl.constexpr, NC: gl.constexpr, G: gl.constexpr, GM: gl.constexpr,
    layout: gl.constexpr,
):  # fmt: skip
    """Columns C0 .. C0 + NC of the program's GM heads in layout. q_src = (the
    first head's row, head stride). Rows G .. GM (G = 8 on 16-row tiles) read 0
    (a masked buffer load: no `other` select)."""
    hq = gl.arange(0, GM, layout=gl.SliceLayout(1, layout))
    cq = gl.arange(0, NC, layout=gl.SliceLayout(0, layout))
    off = hq[:, None] * q_src[1] + cq[None, :]
    if G < GM:
        q = gl.amd.cdna4.buffer_load(q_src[0] + C0, off, mask=(hq < G)[:, None])
    else:
        q = gl.amd.cdna4.buffer_load(q_src[0] + C0, off)
    return q


@gluon.jit
def _q_fp8(q_src, G: gl.constexpr, ROPE: gl.constexpr):
    """(latent, RoPE) operands of fp8 q, loaded straight into the operand layouts."""
    L: gl.constexpr = _layouts(G, gl.num_warps(), ROPE)
    q_op: gl.constexpr = gl.DotOperandLayout(0, L.QK, 16)
    q_dot = _load_q(q_src, 0, D, G, L.GM, q_op).to(gl.float8e4nv, bitcast=True)
    if ROPE > 0:
        qr_op: gl.constexpr = gl.DotOperandLayout(0, L.QR, 8)
        qr_dot = _load_q(q_src, D, ROPE, G, L.GM, qr_op).to(gl.float8e4nv, bitcast=True)
    else:
        qr_dot = q_dot
    return q_dot, qr_dot


@gluon.jit
def _q_bf16(q_src, G: gl.constexpr, ROPE: gl.constexpr):
    """bf16 q quantized to e4m3 with one scale per head (over latent and RoPE):
    ((latent, RoPE) operands, the scale per row)."""
    NW: gl.constexpr = gl.num_warps()
    L: gl.constexpr = _layouts(G, NW, ROPE)
    ql: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [NW, 1], [1, 0])
    q = _load_q(q_src, 0, D, G, L.GM, ql).to(gl.float32)
    amax = gl.max(gl.abs(q), axis=1)
    if ROPE > 0:
        rl: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NW, 1], [1, 0])
        qr = _load_q(q_src, D, ROPE, G, L.GM, rl).to(gl.float32)
        amax_r = gl.convert_layout(gl.max(gl.abs(qr), axis=1), amax.type.layout)
        amax = gl.maximum(amax, amax_r)
    s = gl.maximum(amax, 1e-30) / 448.0
    q_dot = gl.convert_layout(
        (q / s[:, None]).to(gl.float8e4nv), gl.DotOperandLayout(0, L.QK, 16)
    )
    if ROPE > 0:
        sr = gl.convert_layout(s, gl.SliceLayout(1, rl))
        qr_dot = gl.convert_layout(
            (qr / sr[:, None]).to(gl.float8e4nv), gl.DotOperandLayout(0, L.QR, 8)
        )
    else:
        qr_dot = q_dot
    return (q_dot, qr_dot), gl.convert_layout(s, gl.SliceLayout(1, L.QK))


# ---------------------------------------------------------------------------
# One tile, one loop
# ---------------------------------------------------------------------------


@gluon.jit
def _mask(S, k0, end):
    """-inf on the keys k0 + column of S past end (the row's last tile only)."""
    kcol = gl.arange(0, S.shape[1], layout=gl.SliceLayout(0, S.type.layout))
    if k0 + S.shape[1] > end:
        S = gl.where((k0 + kcol < end)[None, :], S, float("-inf"))
    return S


@gluon.jit
def _qk(kvt, q_dot, k0: gl.constexpr, NK: gl.constexpr, QK_L: gl.constexpr):
    """Scores of tile keys k0 .. k0 + NK: 16x16x128 scaled MFMAs, unit scales."""
    k_op: gl.constexpr = gl.DotOperandLayout(1, QK_L, 16)
    k = kvt.slice(k0, NK, dim=0).permute([1, 0]).load(k_op)
    acc = gl.zeros([q_dot.shape[0], NK], gl.float32, QK_L)
    return gl.amd.cdna4.mfma_scaled(
        q_dot, None, "e4m3", k.to(gl.float8e4nv, bitcast=True), None, "e4m3", acc
    )


@gluon.jit
def _tile(
    kvt,
    krt,
    p_smem,
    a_smem,
    q,
    c,
    acc,
    m_i,
    l_i,
    k0,
    end,
    G: gl.constexpr,
    ROPE: gl.constexpr,
    LAZY: gl.constexpr,
):
    """QK, softmax and PV of the tile in kvt (its RoPE part in krt): keys k0 ..
    k0 + 64 of a row ending at end. q: the (latent, RoPE) operands; c: the base-2
    score scale per row; l_i: the row sum, [GM] (KG = 1) or the deferred [GM, 64]
    accumulator."""
    L: gl.constexpr = _layouts(G, gl.num_warps(), ROPE)
    HALVES: gl.constexpr = L.KG == 1
    HB: gl.constexpr = BN // 2

    # QK and the row max of the scaled scores (c > 0: scale the max, not S)
    if HALVES:
        S0 = _mask(_qk(kvt, q[0], 0, HB, L.QK), k0, end)
        _sched_barrier()  # one half's K operand live at a time
        S1 = _mask(_qk(kvt, q[0], HB, HB, L.QK), k0 + HB, end)
        mx = gl.maximum(gl.max(S0, axis=1), gl.max(S1, axis=1))
    else:
        S0 = _qk(kvt, q[0], 0, BN, L.QK)
        if ROPE > 0:
            # + the RoPE dims, 16x16x32 fp8 into the same scores
            kr_op: gl.constexpr = gl.DotOperandLayout(1, L.QR, 8)
            kr = krt.permute([1, 0]).load(kr_op).to(gl.float8e4nv, bitcast=True)
            S0 = gl.convert_layout(
                gl.amd.cdna4.mfma(q[1], kr, gl.convert_layout(S0, L.QR)), L.QK
            )
        S0 = _mask(S0, k0, end)
        S1 = S0
        mx = gl.max(S0, axis=1)

    # online softmax in base 2: the max moves only when beaten by 2^LAZY, P is
    # scaled through the subtracted max; a row with no key so far keeps m = -inf
    # (subtract 0, rescale by 1)
    m_cand = gl.maximum(m_i, mx * c)
    m_new = gl.where(m_cand > m_i + LAZY, m_cand, m_i)
    live = m_new > float("-inf")
    alpha = gl.where(live, gl.exp2(m_i - m_new), 1.0)
    m_sub = gl.where(live, m_new - _log2_ps(LAZY), 0.0)
    c2 = gl.broadcast(c[:, None], S0)[0]
    nm = gl.broadcast(-m_sub[:, None], S0)[0]
    p0 = gl.exp2(gl.fma(S0, c2, nm))
    if HALVES:
        p1 = gl.exp2(gl.fma(S1, c2, nm))
        l_i = l_i * alpha + gl.sum(p0, axis=1) + gl.sum(p1, axis=1)
        p_smem.slice(0, HB, dim=1).store(p0.to(gl.float8e4nv).to(gl.uint8, bitcast=True))
        p_smem.slice(HB, HB, dim=1).store(p1.to(gl.float8e4nv).to(gl.uint8, bitcast=True))
    else:
        # deferred row sum: no cross-wave reduction per tile
        l_i = l_i * gl.broadcast(alpha[:, None], S0)[0] + p0
        p_smem.store(p0.to(gl.float8e4nv).to(gl.uint8, bitcast=True))
    a_smem.store(alpha)
    gl.barrier()  # P and alpha published

    # PV, after rescaling acc if any row's max moved (a wave-uniform branch)
    alpha_pv = a_smem.load(gl.SliceLayout(1, L.PV))
    if gl.min(alpha_pv, axis=0) < 1.0:
        acc = acc * alpha_pv[:, None]
    p_op: gl.constexpr = gl.DotOperandLayout(0, L.PV, L.KW_PV)
    v_op: gl.constexpr = gl.DotOperandLayout(1, L.PV, L.KW_PV)
    v = _relayout(kvt, L.V_SH).load(v_op).to(gl.float8e4nv, bitcast=True)  # V^T
    p = _relayout(p_smem, L.P_PV_SH).load(p_op).to(gl.float8e4nv, bitcast=True)
    if L.PV_SCALED:
        acc = gl.amd.cdna4.mfma_scaled(p, None, "e4m3", v, None, "e4m3", acc)
    else:
        acc = gl.amd.cdna4.mfma(p, v, acc)
    return acc, m_new, l_i


@gluon.jit
def _split_tiles(n_tiles, split, num_splits):
    """This split's tiles [t0, t1) of n_tiles."""
    per = gl.cdiv(n_tiles, num_splits)
    t0 = gl.minimum(split * per, n_tiles)
    t1 = gl.minimum(t0 + per, n_tiles)
    return t0, t1


@gluon.jit
def _loop(
    q,
    q_src,
    kv_ptr,
    ix,
    start,
    n,
    t0,
    t1,
    c,
    init,
    G: gl.constexpr,
    ROPE: gl.constexpr,
    LAZY: gl.constexpr,
    BIG: gl.constexpr,
):
    """Tiles t0 .. t1 of the n-key list ix (start: its position in the
    concatenated list, for the masks). q: the (latent, RoPE) operands, or None to
    load fp8 q from q_src behind the first gather. init: None or the (acc, m, l)
    to continue from. Returns (acc, m, l), l as _tile keeps it."""
    L: gl.constexpr = _layouts(G, gl.num_warps(), ROPE)
    GM: gl.constexpr = L.GM
    KL: gl.constexpr = gl.SliceLayout(1, L.GATHER)

    # the 2-slot ring, one allocation per slot (static addresses)
    kv0 = gl.allocate_shared_memory(gl.uint8, [BN, D], L.K_SH)
    kv1 = gl.allocate_shared_memory(gl.uint8, [BN, D], L.K_SH)
    # the gathers' byte columns, made up front: made in _gather they cost the
    # SEG 2 kernel at 64 heads a spill
    cg = gl.arange(0, D, layout=gl.SliceLayout(0, L.GATHER))
    if ROPE > 0:
        kr0 = gl.allocate_shared_memory(gl.uint8, [BN, ROPE], L.KR_SH)
        kr1 = gl.allocate_shared_memory(gl.uint8, [BN, ROPE], L.KR_SH)
        cols = (cg, gl.arange(0, ROPE, layout=gl.SliceLayout(0, L.GATHER_R)))
    else:
        kr0 = kv0
        kr1 = kv1
        cols = (cg, cg)
    p_smem = gl.allocate_shared_memory(gl.uint8, [GM, BN], L.P_SH)
    a_smem = gl.allocate_shared_memory(
        gl.float32, [GM], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )

    if init is None:
        m_i = gl.full([GM], float("-inf"), gl.float32, gl.SliceLayout(1, L.QK))
        if L.KG == 1:
            l_i = gl.zeros([GM], gl.float32, gl.SliceLayout(1, L.QK))
        else:
            l_i = gl.zeros([GM, BN], gl.float32, L.QK)
        acc = gl.zeros([GM, D], gl.float32, L.PV)
    else:
        acc = init[0]
        m_i = init[1]
        l_i = init[2]

    # r: the row ids of the next tile to gather
    n_s = gl.minimum(n, t1 * BN)  # this split's keys end here
    kb = t0 * BN
    r = _rows(ix, kb, n_s, KL)
    _gather(kv0, kr0, kv_ptr, ix, r, cols, G, ROPE, BIG)
    r = _rows(ix, kb + BN, n_s, KL)
    if q is None:
        q = _q_fp8(q_src, G, ROPE)

    # unrolled x 2 over the two slots; an odd tile count runs one fully masked
    # tile, which reads nothing of the next split (its keys end at n_s)
    end_s = start + n_s
    for it in range(0, gl.cdiv(t1 - t0, 2)):
        t = it * 2
        gl.amd.cdna4.async_copy.wait_group(0)  # tile t landed (+ the barrier)
        _gather(kv1, kr1, kv_ptr, ix, r, cols, G, ROPE, BIG)
        r = _rows(ix, kb + (t + 2) * BN, n_s, KL)
        acc, m_i, l_i = _tile(
            kv0, kr0, p_smem, a_smem, q, c, acc, m_i, l_i, start + kb + t * BN,
            end_s, G, ROPE, LAZY,
        )  # fmt: skip
        gl.amd.cdna4.async_copy.wait_group(0)
        _gather(kv0, kr0, kv_ptr, ix, r, cols, G, ROPE, BIG)
        r = _rows(ix, kb + (t + 3) * BN, n_s, KL)
        acc, m_i, l_i = _tile(
            kv1, kr1, p_smem, a_smem, q, c, acc, m_i, l_i, start + kb + (t + 1) * BN,
            end_s, G, ROPE, LAZY,
        )  # fmt: skip
    gl.amd.cdna4.async_copy.wait_group(0)
    return acc, m_i, l_i


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@gluon.jit
def _program(G: gl.constexpr, XH: gl.constexpr):
    """(row, first head) of this program. XH > 1: a row's XH head blocks run at
    program ids L, L + 8, .. (one XCD, back to back, so the later blocks' gathers
    hit its L2); the grid pads the rows to a multiple of 8."""
    if XH > 1:
        L = gl.program_id(0)
        row = (L // (8 * XH)) * 8 + L % 8
        h0 = (L // 8) % XH * G
    else:
        row = gl.program_id(0)
        h0 = gl.program_id(1) * G
    return row, h0


@gluon.jit
def _store_partial(
    acc,
    m_i,
    l_i,
    v_scale,
    row,
    split,
    h0,
    num_heads,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    G: gl.constexpr,
    LAZY: gl.constexpr,
):
    """This split's (m, l, acc) for _sparse_mla_reduce: [C, P, H] f32 row max (base
    2) and row sum, [C, P, H, 512] bf16 unnormalized acc in V units, P the launched
    split count. acc = None writes an empty split (m = -inf, l = 0)."""
    NW: gl.constexpr = gl.num_warps()
    GM: gl.constexpr = m_i.shape[0]
    base = (row * gl.num_programs(2) + split).to(gl.int64) * num_heads + h0
    hs = gl.arange(0, GM, layout=m_i.type.layout)
    gl.store(part_m_ptr + base + hs, m_i, mask=hs < G)
    gl.store(part_l_ptr + base + hs, l_i * _inv_ps(LAZY), mask=hs < G)  # l carries PS
    if acc is not None:
        o_l: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [NW, 1], [1, 0])
        o = gl.convert_layout((acc * (v_scale * _inv_ps(LAZY))).to(gl.bfloat16), o_l)
        ho = gl.arange(0, GM, layout=gl.SliceLayout(1, o_l))
        co = gl.arange(0, D, layout=gl.SliceLayout(0, o_l))
        gl.amd.cdna4.buffer_store(
            o, part_acc_ptr + base * D, ho[:, None] * D + co[None, :],
            mask=(ho < G)[:, None],
        )  # fmt: skip


@gluon.jit
def _finish(
    acc,
    m_i,
    l_i,
    v_scale,
    row,
    h0,
    num_heads,
    sink_ptr,
    lse_ptr,
    out_ptr,
    out_stride0,
    out_stride1,
    epi,
    G: gl.constexpr,
    LAZY: gl.constexpr,
    INV_ROPE: gl.constexpr,
    OUT_MXFP8: gl.constexpr,
):
    """Normalize (the sink is one more term in the denominator), store the rows
    and, with lse_ptr, the natural-log LSE. An empty row (m = -inf, l = 0) comes
    out 0. epi = (positions, cos_sin, its stride, E8M0 scales, their row stride)."""
    NW: gl.constexpr = gl.num_warps()
    GM: gl.constexpr = acc.shape[0]
    LOG2_PS: gl.constexpr = _log2_ps(LAZY)  # l carries the P scale
    hs = gl.arange(0, GM, layout=m_i.type.layout)
    hm = hs < G
    if sink_ptr is not None:
        sink = gl.load(sink_ptr + h0 + hs, mask=hm, other=0.0) * LOG2E
        # an empty row gets l = inf here, so 0
        r = v_scale / (l_i + gl.exp2(sink - m_i + LOG2_PS))
    else:
        r = gl.where(l_i > 0.0, v_scale / l_i, 0.0)
    if lse_ptr is not None:
        # relative to max(m, sink): finite for an empty row with a sink
        m_f = m_i
        if sink_ptr is not None:
            m_f = gl.maximum(m_i, sink)
        l_t = gl.where(m_i > float("-inf"), l_i * gl.exp2(m_i - m_f - LOG2_PS), 0.0)
        if sink_ptr is not None:
            l_t = l_t + gl.exp2(sink - m_f)
        gl.store(
            lse_ptr + row.to(gl.int64) * num_heads + h0 + hs,
            (m_f + gl.log2(l_t)) * 0.6931471805599453,
            mask=hm,
        )
    x = acc * gl.convert_layout(r, gl.SliceLayout(1, acc.type.layout))[:, None]
    if INV_ROPE or OUT_MXFP8:
        # rows G .. GM are padding, so the head bound is h0 + G
        epi_l: gl.constexpr = gl.BlockedLayout([1, 32], [16, 4], [NW // 4, 4], [1, 0])
        _epilogue_store(
            x, out_ptr, out_stride0, out_stride1, row, h0, h0 + G, epi_l, epi[0],
            epi[1], epi[2], epi[3], epi[4], 64, INV_ROPE, OUT_MXFP8,
        )  # fmt: skip
    else:
        # 16 B per lane, row-contiguous
        o_l: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [NW, 1], [1, 0])
        o = gl.convert_layout(x.to(gl.bfloat16), o_l)
        ho = gl.arange(0, GM, layout=gl.SliceLayout(1, o_l))
        co = gl.arange(0, D, layout=gl.SliceLayout(0, o_l))
        gl.amd.cdna4.buffer_store(
            o,
            out_ptr + row.to(gl.int64) * out_stride0 + h0 * out_stride1,
            ho[:, None] * out_stride1 + co[None, :],
            mask=(ho < G)[:, None],
        )


@gluon.jit(do_not_specialize=["num_queries", "num_splits"])
def _sparse_mla_fp8(
    q_ptr,  # [C, H, 512 + ROPE] fp8 (u8 view) or bf16
    kv_ptr,  # u8 base of the cache(s): rows of 512 + ROPE bytes
    out_ptr,  # [C, H, 512] bf16 or MXFP8 e4m3 data; None when split
    indptr_ptr,  # [C + 1] int32, the main list
    idx_ptr,  # int32 row ids (SEG 0) or slot ids (SEG 1 / 2)
    sink_ptr,  # [H] f32 or None
    q_scale_ptr,  # [1] f32 for fp8 q, None for bf16 q
    ks_ptr,  # [1] f32 cache scale
    lse_ptr,  # [C, H] f32 or None; None when split
    part_m_ptr,  # split partials (_store_partial); None unless split
    part_l_ptr,
    part_acc_ptr,
    # SEG 1 / 2: the extra list and the paged maps; SEG 2: the extra cache, scale
    x_indptr_ptr,
    x_idx_ptr,
    mo,
    mstr,
    xo,
    xstr,
    x_kv_ptr,
    x_ks_ptr,
    # output epilogue: positions and cos_sin (+ stride) for the inverse RoPE,
    # E8M0 scales (+ stride) for MXFP8; None when off
    pos_ptr,
    cs_ptr,
    cs_stride,
    osc_ptr,
    osc_stride,
    num_queries,
    num_heads,
    sm_scale_log2,
    num_splits,
    q_stride0,
    q_stride1,
    out_stride0,
    out_stride1,
    G: gl.constexpr,
    ROPE: gl.constexpr,
    LAZY: gl.constexpr,
    BIG: gl.constexpr,
    SEG: gl.constexpr,
    MB_SH: gl.constexpr,
    XB_SH: gl.constexpr,
    XH: gl.constexpr,
):
    L: gl.constexpr = _layouts(G, gl.num_warps(), ROPE)
    ROW_L: gl.constexpr = gl.SliceLayout(1, L.QK)  # one value per head
    SPLIT: gl.constexpr = part_m_ptr is not None
    Q_BF16: gl.constexpr = q_scale_ptr is None
    row, h0 = _program(G, XH)
    if XH > 1:
        if row >= num_queries:
            return
    split = gl.program_id(2)
    if SPLIT:
        if split >= num_splits:  # padding of the launched split count
            m0 = gl.full([L.GM], float("-inf"), gl.float32, ROW_L)
            _store_partial(
                None, m0, gl.zeros_like(m0), 0.0, row, split, h0, num_heads,
                part_m_ptr, part_l_ptr, part_acc_ptr, G, LAZY,
            )  # fmt: skip
            return

    # the base-2 score scale per row: q scale x cache scale x softmax scale
    k_scale = gl.load(ks_ptr)
    q_src = (q_ptr + row.to(gl.int64) * q_stride0 + h0 * q_stride1, q_stride1)
    if Q_BF16:
        # quantized before the ring is live, so its LDS scratch aliases the ring
        q, q_s = _q_bf16(q_src, G, ROPE)
        c = q_s * (k_scale * sm_scale_log2)
    else:
        q = None  # loaded by the loop, behind its first gather
        c = gl.zeros([L.GM], gl.float32, ROW_L) + (
            gl.load(q_scale_ptr) * k_scale * sm_scale_log2
        )

    if SEG == 2:
        # the SWA window from kv, then the top-k from x_kv; a split takes a range
        # of the concatenated tiles, so it may cover part of either list or both
        if not Q_BF16:
            q = _q_fp8(q_src, G, ROPE)
        ms = gl.load(indptr_ptr + row)
        na = gl.load(indptr_ptr + row + 1) - ms
        es = gl.load(x_indptr_ptr + row)
        nb = gl.load(x_indptr_ptr + row + 1) - es
        main = (idx_ptr + ms, mo, mstr, MB_SH)
        extra = (x_idx_ptr + es, xo, xstr, XB_SH)
        if x_ks_ptr is None:
            v_scale = k_scale
            c_b = c
        else:
            v_scale = gl.load(x_ks_ptr)
            c_b = c * (v_scale / k_scale)
        nta = gl.cdiv(na, BN)
        u0, u1 = _split_tiles(nta + gl.cdiv(nb, BN), split, num_splits)
        st = _loop(
            q, q_src, kv_ptr, (main, main, na), 0, na, gl.minimum(u0, nta),
            gl.minimum(u1, nta), c, None, G, ROPE, LAZY, BIG,
        )  # fmt: skip
        if x_ks_ptr is not None:  # the first list's V in the second one's units
            st = (st[0] * (k_scale / v_scale), st[1], st[2])
        acc, m_i, l_i = _loop(
            q, q_src, x_kv_ptr, (extra, extra, 0), 0, nb, gl.maximum(u0 - nta, 0),
            gl.maximum(u1 - nta, 0), c_b, st, G, ROPE, LAZY, BIG,
        )  # fmt: skip
    else:
        start, end, ix = _key_lists(
            indptr_ptr, idx_ptr, x_indptr_ptr, x_idx_ptr, row, mo, mstr, xo, xstr,
            SEG, MB_SH, XB_SH,
        )  # fmt: skip
        t0, t1 = _split_tiles(gl.cdiv(end - start, BN), split, num_splits)
        acc, m_i, l_i = _loop(
            q, q_src, kv_ptr, ix, start, end - start, t0, t1, c, None, G, ROPE, LAZY,
            BIG,
        )  # fmt: skip
        v_scale = k_scale
    if L.KG > 1:  # the deferred row sum
        l_i = gl.sum(l_i, axis=1)

    if SPLIT:
        _store_partial(
            acc, m_i, l_i, v_scale, row, split, h0, num_heads, part_m_ptr, part_l_ptr,
            part_acc_ptr, G, LAZY,
        )  # fmt: skip
    else:
        _finish(
            acc, m_i, l_i, v_scale, row, h0, num_heads, sink_ptr, lse_ptr, out_ptr,
            out_stride0, out_stride1, (pos_ptr, cs_ptr, cs_stride, osc_ptr, osc_stride),
            G, LAZY, pos_ptr is not None, osc_ptr is not None,
        )  # fmt: skip
