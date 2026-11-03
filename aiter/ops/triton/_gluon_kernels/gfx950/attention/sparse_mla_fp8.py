# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950) sparse MLA on a per-tensor fp8 KV cache ("fp8_scalar") with fp8
dots. sparse_mla_fwd (attention/sparse_mla.py) launches it for every call it can
address; every other call runs the generic kernel in sparse_mla.py, whose split
merge (_sparse_mla_reduce) and output store (_epilogue_store) this one reuses.

Geometry (ROPE):
    0   one 512-byte row per key, V = the whole row: DeepSeek-V4 / V4.1 (RoPE inside
        the row, per-head sink) and GLM-5.3-Flash (no RoPE, no sink)
    64  512 latent bytes + a 64-byte RoPE tail per key, V = the latent (GLM-5.2);
        QK adds one 16x16x32 MFMA chain over the RoPE dims
q is fp8 with one scale for the tensor, or bf16 quantized in the prologue (one
scale per head). The cache scale folds into the score scale (K side) and the final
normalization (V side).

A program is one query row, G heads (8..64) and one split of the row's keys. G and
the wave count NW fix the layouts (_Config), and the layouts fix how the waves
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
The lists hold no -1 sentinels; the caller sends those calls to the generic kernel.

Split-K programs write the generic kernel's partials (base-2 row max, row sum, bf16
unnormalized acc in V units) for _sparse_mla_reduce. The output store can apply
DeepSeek-V4's inverse RoPE and MXFP8 quantization, here or in the merge.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import _core as _gl_core
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_mla import (
    _epilogue_store,
)
from aiter.ops.triton.utils.common_utils import strip_annotate

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


@aggregate
@strip_annotate
class _Config:
    """What a program fixes at compile time: G heads on NW waves, the RoPE tail
    width, LAZY, BIG (64-bit gather addresses) and the layouts."""

    G: gl.constexpr
    NW: gl.constexpr
    ROPE: gl.constexpr
    LAZY: gl.constexpr
    BIG: gl.constexpr
    GM: gl.constexpr  # MFMA rows, max(G, 16)
    KG: gl.constexpr  # QK key groups; 1: a wave owns whole rows
    LOG2_PS: gl.constexpr  # P scale: a score 2^LAZY above the lagging max gives 448
    INV_PS: gl.constexpr
    QK: gl.constexpr  # 16x16x128 QK MFMA
    QR: gl.constexpr  # 16x16x32 MFMA over the RoPE dims
    PV: gl.constexpr  # PV MFMA, waves on the 512 output dims
    PV_SCALED: gl.constexpr  # PV is a 32x32x64 scaled MFMA, else a 16x16x32 one
    KW_PV: gl.constexpr  # PV operands' k_width
    GATHER: gl.constexpr  # [64, 512] tile gather
    GATHER_R: gl.constexpr  # [64, ROPE] RoPE tail gather
    K_SH: gl.constexpr  # ring tile as written, and read as K
    V_SH: gl.constexpr  # ring tile read as V^T
    KR_SH: gl.constexpr  # RoPE tile
    P_SH: gl.constexpr  # P as the softmax writes it
    P_PV_SH: gl.constexpr  # P as PV reads it
    ROW_L: gl.constexpr  # one value per head
    KEY_L: gl.constexpr  # one value per gathered key
    WIDE_L: gl.constexpr  # [GM, 512] rows, 16 B per lane

    @gluon.constexpr_function
    def __init__(self, G, NW, ROPE, LAZY, BIG):
        GM = max(G, 16)
        HW = min(NW, GM // 16)
        KG = NW // HW
        qk = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [HW, KG])
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
            pad = 16
            kv = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [0, 32],
                  [0, 64], [16, 0], [0, 128], [0, 256], [32, 0], [8, 0]]  # fmt: skip
            vv = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [0, 32],
                  [0, 64], [8, 0], [0, 128], [0, 256], [32, 0], [16, 0]]  # fmt: skip
            pp = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 8], [0, 32]]
            pp = pp + [[1 << i, 0] for i in range(GM.bit_length() - 1)]
        k_sh = gl.PaddedSharedLayout([[1024, pad]], kv, [], [64, 512])
        p_sh = gl.PaddedSharedLayout.with_identity_for([[64, 16]], [GM, 64], [1, 0])
        # Gather: a lane per 16 B of one 1 KB block (the copy writes each wave's
        # 64 x 16 B contiguously), the block bits on registers and waves. Dims before
        # keys on registers: register order is issue order, and a key bit early makes
        # the next gather wait on every gather in flight.
        nwb = NW.bit_length() - 1
        block = sorted(kv[10:], key=lambda b: b[0] != 0)
        gather = gl.DistributedLinearLayout(
            reg_bases=kv[:4] + block[: 5 - nwb],
            lane_bases=kv[4:10],
            warp_bases=block[5 - nwb :],
            block_bases=[],
            shape=[64, 512],
        )
        kr_sh = k_sh
        gather_r = gather
        if ROPE > 0:
            assert ROPE == 64 and KG > 1, "the RoPE tile runs with key-split QK"
            rb = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0], [0, 16],
                  [0, 32], [16, 0], [32, 0]]  # fmt: skip
            kr_sh = gl.PaddedSharedLayout([[1024, 16]], rb, [], [64, ROPE])
            nr = min(nwb, 2)
            gather_r = gl.DistributedLinearLayout(
                reg_bases=rb[:4] + rb[10:][: 2 - nr],
                lane_bases=rb[4:10],
                warp_bases=rb[10:][2 - nr :] + [[0, 0]] * (nwb - nr),
                block_bases=[],
                shape=[64, ROPE],
            )

        self.G = gl.constexpr(G)
        self.NW = gl.constexpr(NW)
        self.ROPE = gl.constexpr(ROPE)
        self.LAZY = gl.constexpr(LAZY)
        self.BIG = gl.constexpr(BIG)
        self.GM = gl.constexpr(GM)
        self.KG = gl.constexpr(KG)
        self.LOG2_PS = gl.constexpr(_LOG2_448 - LAZY)
        self.INV_PS = gl.constexpr(2.0 ** (LAZY - _LOG2_448))
        self.QK = gl.constexpr(qk)
        self.QR = gl.constexpr(gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [HW, KG]))
        self.PV = gl.constexpr(pv)
        self.PV_SCALED = gl.constexpr(GM >= 32)
        self.KW_PV = gl.constexpr(kw_pv)
        self.GATHER = gl.constexpr(gather)
        self.GATHER_R = gl.constexpr(gather_r)
        self.K_SH = gl.constexpr(k_sh)
        self.V_SH = gl.constexpr(
            gl.PaddedSharedLayout([[1024, pad]], vv, [], [64, 512])
        )
        self.KR_SH = gl.constexpr(kr_sh)
        self.P_SH = gl.constexpr(p_sh)
        if pp is None:
            self.P_PV_SH = gl.constexpr(p_sh)
        else:
            self.P_PV_SH = gl.constexpr(
                gl.PaddedSharedLayout([[64, 16]], pp, [], [GM, 64])
            )
        self.ROW_L = gl.constexpr(gl.SliceLayout(1, qk))
        self.KEY_L = gl.constexpr(gl.SliceLayout(1, gather))
        self.WIDE_L = gl.constexpr(gl.BlockedLayout([1, 8], [1, 64], [NW, 1], [1, 0]))


# A key list (ix) is one of
#   (ids,)                 a flat pool's row ids
#   (main, extra, n_main)  two lists of paged slot ids read as their concatenation,
#                          main for its first n_main keys; a list is
#                          (ids, o, stride, b), slot s being row
#                          o + (s >> b) * stride + (s & (2^b - 1))
# The code dispatches on len(ix): tuple members lose constexpr-ness in a callee.


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
        # no other=: a masked buffer load reads 0, while other=0 is a select right
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
def _copy(
    dst, kv_ptr, rid, cols, RS: gl.constexpr, OFF: gl.constexpr, BIG: gl.constexpr
):
    """Bytes OFF + cols of rows rid (row pitch RS) into dst, no commit."""
    if BIG:
        # a 64-bit row base: buffer offsets are i32
        rp = kv_ptr + rid.to(gl.int64) * RS + OFF
        gl.amd.cdna4.async_copy.global_load_to_shared(dst, rp[:, None] + cols[None, :])
    else:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dst, kv_ptr, (rid * RS + OFF)[:, None] + cols[None, :]
        )


@gluon.jit
def _gather(cfg, kv_dst, kr_dst, kv_ptr, ix, r, cols):
    """One tile into a ring slot, the latent into kv_dst and the RoPE tail into
    kr_dst, as one commit group. cols: their byte columns in the gather layouts."""
    rid = _rid(ix, r, cfg.KEY_L)
    _copy(kv_dst, kv_ptr, rid, cols[0], D + cfg.ROPE, 0, cfg.BIG)
    if cfg.ROPE > 0:
        rid_r = gl.convert_layout(rid, gl.SliceLayout(1, cfg.GATHER_R))
        _copy(kr_dst, kv_ptr, rid_r, cols[1], D + cfg.ROPE, D, cfg.BIG)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _mask(S, k0, end):
    """-inf on the keys k0 + column of S past end (the row's last tile only)."""
    kcol = gl.arange(0, S.shape[1], layout=gl.SliceLayout(0, S.type.layout))
    if k0 + S.shape[1] > end:
        S = gl.where((k0 + kcol < end)[None, :], S, float("-inf"))
    return S


@gluon.jit
def _qk(cfg, kvt, q_dot, k0: gl.constexpr, NK: gl.constexpr):
    """Scores of tile keys k0 .. k0 + NK: 16x16x128 scaled MFMAs, unit scales."""
    k_op: gl.constexpr = gl.DotOperandLayout(1, cfg.QK, 16)
    k = kvt.slice(k0, NK, dim=0).permute([1, 0]).load(k_op)
    acc = gl.zeros([q_dot.shape[0], NK], gl.float32, cfg.QK)
    return gl.amd.cdna4.mfma_scaled(
        q_dot, None, "e4m3", k.to(gl.float8e4nv, bitcast=True), None, "e4m3", acc
    )


@gluon.jit
def _tile(cfg, kvt, krt, p_smem, a_smem, q, c, acc, m_i, l_i, k0, end):
    """QK, softmax and PV of the tile in kvt (its RoPE part in krt): keys k0 ..
    k0 + 64 of a row ending at end. q: the (latent, RoPE) operands; c: the base-2
    score scale per row; l_i: the row sum, [GM] (KG = 1) or the deferred [GM, 64]
    accumulator."""
    HALVES: gl.constexpr = cfg.KG == 1
    HB: gl.constexpr = BN // 2

    # c > 0, so the max of the scaled scores is the scaled max
    if HALVES:
        S0 = _mask(_qk(cfg, kvt, q[0], 0, HB), k0, end)
        _sched_barrier()  # one half's K operand live at a time
        S1 = _mask(_qk(cfg, kvt, q[0], HB, HB), k0 + HB, end)
        mx = gl.maximum(gl.max(S0, axis=1), gl.max(S1, axis=1))
    else:
        S0 = _qk(cfg, kvt, q[0], 0, BN)
        if cfg.ROPE > 0:
            kr_op: gl.constexpr = gl.DotOperandLayout(1, cfg.QR, 8)
            kr = krt.permute([1, 0]).load(kr_op).to(gl.float8e4nv, bitcast=True)
            S0 = gl.convert_layout(
                gl.amd.cdna4.mfma(q[1], kr, gl.convert_layout(S0, cfg.QR)), cfg.QK
            )
        S0 = _mask(S0, k0, end)
        S1 = S0
        mx = gl.max(S0, axis=1)

    # A row with no key so far keeps m = -inf: subtract 0, rescale by 1.
    m_cand = gl.maximum(m_i, mx * c)
    m_new = gl.where(m_cand > m_i + cfg.LAZY, m_cand, m_i)
    live = m_new > float("-inf")
    alpha = gl.where(live, gl.exp2(m_i - m_new), 1.0)
    m_sub = gl.where(live, m_new - cfg.LOG2_PS, 0.0)
    c2 = gl.broadcast(c[:, None], S0)[0]
    nm = gl.broadcast(-m_sub[:, None], S0)[0]
    p0 = gl.exp2(gl.fma(S0, c2, nm))
    if HALVES:
        p1 = gl.exp2(gl.fma(S1, c2, nm))
        l_i = gl.fma(l_i, alpha, gl.sum(p0, axis=1) + gl.sum(p1, axis=1))
        p_smem.slice(0, HB, dim=1).store(
            p0.to(gl.float8e4nv).to(gl.uint8, bitcast=True)
        )
        p_smem.slice(HB, HB, dim=1).store(
            p1.to(gl.float8e4nv).to(gl.uint8, bitcast=True)
        )
    else:
        # Deferred row sum: no cross-wave reduction per tile. Rounded twice on
        # purpose (the launch turns contraction off): as v_pk_fma_f32 it is
        # slower than v_pk_mul + v_pk_add.
        l_i = l_i * gl.broadcast(alpha[:, None], S0)[0] + p0
        p_smem.store(p0.to(gl.float8e4nv).to(gl.uint8, bitcast=True))
    a_smem.store(alpha)
    gl.barrier()

    # a uniform branch: on most tiles no row max moved
    alpha_pv = a_smem.load(gl.SliceLayout(1, cfg.PV))
    if gl.min(alpha_pv, axis=0) < 1.0:
        acc = acc * alpha_pv[:, None]
    p_op: gl.constexpr = gl.DotOperandLayout(0, cfg.PV, cfg.KW_PV)
    v_op: gl.constexpr = gl.DotOperandLayout(1, cfg.PV, cfg.KW_PV)
    v = _relayout(kvt, cfg.V_SH).load(v_op).to(gl.float8e4nv, bitcast=True)
    p = _relayout(p_smem, cfg.P_PV_SH).load(p_op).to(gl.float8e4nv, bitcast=True)
    if cfg.PV_SCALED:
        acc = gl.amd.cdna4.mfma_scaled(p, None, "e4m3", v, None, "e4m3", acc)
    else:
        acc = gl.amd.cdna4.mfma(p, v, acc)
    return acc, m_new, l_i


@aggregate
@strip_annotate
class _Program:
    """One query row, G heads from h0, one split of the row's keys."""

    cfg: _Config
    row: gl.tensor
    h0: gl.tensor
    split: gl.tensor
    num_splits: gl.tensor
    num_heads: gl.tensor
    q_ptr: gl.tensor  # head h0 of the row's q
    q_stride: gl.tensor  # between heads

    @gluon.jit
    def initialize(
        cfg, XH: gl.constexpr, q_ptr, q_stride0, q_stride1, num_heads, num_splits
    ):
        # XH > 1: a row's XH head blocks run at program ids L, L + 8, .. (one XCD,
        # back to back, so the later blocks' gathers hit its L2); the grid pads the
        # rows to a multiple of 8.
        if XH > 1:
            L = gl.program_id(0)
            row = (L // (8 * XH)) * 8 + L % 8
            h0 = (L // 8) % XH * cfg.G
        else:
            row = gl.program_id(0)
            h0 = gl.program_id(1) * cfg.G
        return _Program(
            cfg, row, h0, gl.program_id(2), num_splits, num_heads,
            q_ptr + row.to(gl.int64) * q_stride0 + h0 * q_stride1, q_stride1,
        )  # fmt: skip

    @gluon.jit
    def tiles(self, n_tiles):
        """This split's tiles [t0, t1) of n_tiles."""
        per = gl.cdiv(n_tiles, self.num_splits)
        t0 = gl.minimum(self.split * per, n_tiles)
        t1 = gl.minimum(t0 + per, n_tiles)
        return t0, t1

    @gluon.jit
    def load_q(self, C0: gl.constexpr, NC: gl.constexpr, layout: gl.constexpr):
        """Columns C0 .. C0 + NC of the program's GM heads; rows G .. GM read 0."""
        G: gl.constexpr = self.cfg.G
        GM: gl.constexpr = self.cfg.GM
        hq = gl.arange(0, GM, layout=gl.SliceLayout(1, layout))
        cq = gl.arange(0, NC, layout=gl.SliceLayout(0, layout))
        off = hq[:, None] * self.q_stride + cq[None, :]
        if G < GM:
            q = gl.amd.cdna4.buffer_load(self.q_ptr + C0, off, mask=(hq < G)[:, None])
        else:
            q = gl.amd.cdna4.buffer_load(self.q_ptr + C0, off)
        return q

    @gluon.jit
    def q_fp8(self):
        """(latent, RoPE) operands of fp8 q, loaded straight into the operand layouts."""
        cfg = self.cfg
        q_op: gl.constexpr = gl.DotOperandLayout(0, cfg.QK, 16)
        q_dot = self.load_q(0, D, q_op).to(gl.float8e4nv, bitcast=True)
        if cfg.ROPE > 0:
            qr_op: gl.constexpr = gl.DotOperandLayout(0, cfg.QR, 8)
            qr_dot = self.load_q(D, cfg.ROPE, qr_op).to(gl.float8e4nv, bitcast=True)
        else:
            qr_dot = q_dot
        return q_dot, qr_dot

    @gluon.jit
    def q_bf16(self):
        """bf16 q as e4m3 with one scale per head (over latent and RoPE):
        ((latent, RoPE) operands, the scale per row)."""
        cfg = self.cfg
        q = self.load_q(0, D, cfg.WIDE_L).to(gl.float32)
        amax = gl.max(gl.abs(q), axis=1)
        if cfg.ROPE > 0:
            rl: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [cfg.NW, 1], [1, 0])
            qr = self.load_q(D, cfg.ROPE, rl).to(gl.float32)
            amax_r = gl.convert_layout(gl.max(gl.abs(qr), axis=1), amax.type.layout)
            amax = gl.maximum(amax, amax_r)
        s = gl.maximum(amax, 1e-30) / 448.0
        q_dot = gl.convert_layout(
            (q / s[:, None]).to(gl.float8e4nv), gl.DotOperandLayout(0, cfg.QK, 16)
        )
        if cfg.ROPE > 0:
            sr = gl.convert_layout(s, gl.SliceLayout(1, rl))
            qr_dot = gl.convert_layout(
                (qr / sr[:, None]).to(gl.float8e4nv), gl.DotOperandLayout(0, cfg.QR, 8)
            )
        else:
            qr_dot = q_dot
        return (q_dot, qr_dot), gl.convert_layout(s, cfg.ROW_L)

    @gluon.jit
    def loop(self, q, kv_ptr, ix, start, n, t0, t1, c, init):
        """Tiles t0 .. t1 of the n-key list ix (start: its position in the
        concatenated list, for the masks). q: the (latent, RoPE) operands, or None
        to load fp8 q behind the first gather. init: None or the (acc, m, l) to
        continue from. Returns (acc, m, l), l as _tile keeps it."""
        cfg = self.cfg
        GM: gl.constexpr = cfg.GM
        ROPE: gl.constexpr = cfg.ROPE

        # one allocation per slot, so the slots have static addresses
        kv0 = gl.allocate_shared_memory(gl.uint8, [BN, D], cfg.K_SH)
        kv1 = gl.allocate_shared_memory(gl.uint8, [BN, D], cfg.K_SH)
        # made once here: made in _gather, the columns cost the SEG 2 kernel at 64
        # heads a spill
        cg = gl.arange(0, D, layout=gl.SliceLayout(0, cfg.GATHER))
        if ROPE > 0:
            kr0 = gl.allocate_shared_memory(gl.uint8, [BN, ROPE], cfg.KR_SH)
            kr1 = gl.allocate_shared_memory(gl.uint8, [BN, ROPE], cfg.KR_SH)
            cols = (cg, gl.arange(0, ROPE, layout=gl.SliceLayout(0, cfg.GATHER_R)))
        else:
            kr0 = kv0
            kr1 = kv1
            cols = (cg, cg)
        p_smem = gl.allocate_shared_memory(gl.uint8, [GM, BN], cfg.P_SH)
        a_smem = gl.allocate_shared_memory(
            gl.float32, [GM], gl.SwizzledSharedLayout(1, 1, 1, [0])
        )

        if init is None:
            m_i = gl.full([GM], float("-inf"), gl.float32, cfg.ROW_L)
            if cfg.KG == 1:
                l_i = gl.zeros([GM], gl.float32, cfg.ROW_L)
            else:
                l_i = gl.zeros([GM, BN], gl.float32, cfg.QK)
            acc = gl.zeros([GM, D], gl.float32, cfg.PV)
        else:
            acc = init[0]
            m_i = init[1]
            l_i = init[2]

        # r: the row ids of the next tile to gather
        n_s = gl.minimum(n, t1 * BN)
        kb = t0 * BN
        r = _rows(ix, kb, n_s, cfg.KEY_L)
        _gather(cfg, kv0, kr0, kv_ptr, ix, r, cols)
        r = _rows(ix, kb + BN, n_s, cfg.KEY_L)
        if q is None:
            q = self.q_fp8()

        # Unrolled over the two slots. An odd tile count runs one fully masked
        # tile, which reads nothing of the next split: its keys end at n_s.
        end_s = start + n_s
        for it in range(gl.cdiv(t1 - t0, 2)):
            t = it * 2
            # tile t landed; the wait's barrier also retires the reads of the other slot
            gl.amd.cdna4.async_copy.wait_group(0)
            _gather(cfg, kv1, kr1, kv_ptr, ix, r, cols)
            r = _rows(ix, kb + (t + 2) * BN, n_s, cfg.KEY_L)
            acc, m_i, l_i = _tile(
                cfg, kv0, kr0, p_smem, a_smem, q, c, acc, m_i, l_i,
                start + kb + t * BN, end_s,
            )  # fmt: skip
            gl.amd.cdna4.async_copy.wait_group(0)
            _gather(cfg, kv0, kr0, kv_ptr, ix, r, cols)
            r = _rows(ix, kb + (t + 3) * BN, n_s, cfg.KEY_L)
            acc, m_i, l_i = _tile(
                cfg, kv1, kr1, p_smem, a_smem, q, c, acc, m_i, l_i,
                start + kb + (t + 1) * BN, end_s,
            )  # fmt: skip
        gl.amd.cdna4.async_copy.wait_group(0)
        return acc, m_i, l_i

    @gluon.jit
    def store_partial(
        self, acc, m_i, l_i, v_scale, part_m_ptr, part_l_ptr, part_acc_ptr
    ):
        """This split's (m, l, acc) for _sparse_mla_reduce: [C, P, H] f32 row max
        (base 2) and row sum, [C, P, H, 512] bf16 unnormalized acc in V units, P
        the launched split count. acc = None writes an empty split."""
        cfg = self.cfg
        base = (self.row * gl.num_programs(2) + self.split).to(gl.int64)
        base = base * self.num_heads + self.h0
        hs = gl.arange(0, cfg.GM, layout=m_i.type.layout)
        gl.store(part_m_ptr + base + hs, m_i, mask=hs < cfg.G)
        gl.store(part_l_ptr + base + hs, l_i * cfg.INV_PS, mask=hs < cfg.G)
        if acc is not None:
            o = gl.convert_layout(
                (acc * (v_scale * cfg.INV_PS)).to(gl.bfloat16), cfg.WIDE_L
            )
            ho = gl.arange(0, cfg.GM, layout=gl.SliceLayout(1, cfg.WIDE_L))
            co = gl.arange(0, D, layout=gl.SliceLayout(0, cfg.WIDE_L))
            gl.amd.cdna4.buffer_store(
                o, part_acc_ptr + base * D, ho[:, None] * D + co[None, :],
                mask=(ho < cfg.G)[:, None],
            )  # fmt: skip

    @gluon.jit
    def finish(
        self, acc, m_i, l_i, v_scale, sink_ptr, lse_ptr, out_ptr, out_stride0,
        out_stride1, epi,
    ):  # fmt: skip
        """Normalize (the sink is one more term in the denominator), store the
        rows and, with lse_ptr, the natural-log LSE. An empty row comes out 0.
        epi = (positions, cos_sin, its stride, E8M0 scales, their row stride)."""
        cfg = self.cfg
        G: gl.constexpr = cfg.G
        INV_ROPE: gl.constexpr = epi[0] is not None
        OUT_MXFP8: gl.constexpr = epi[3] is not None
        row = self.row
        h0 = self.h0
        hs = gl.arange(0, cfg.GM, layout=m_i.type.layout)
        hm = hs < G
        if sink_ptr is not None:
            sink = gl.load(sink_ptr + h0 + hs, mask=hm, other=0.0) * LOG2E
            # an empty row gets l = inf here, so 0
            r = v_scale / (l_i + gl.exp2(sink - m_i + cfg.LOG2_PS))
        else:
            r = gl.where(l_i > 0.0, v_scale / l_i, 0.0)
        if lse_ptr is not None:
            # relative to max(m, sink): finite for an empty row with a sink
            m_f = m_i
            if sink_ptr is not None:
                m_f = gl.maximum(m_i, sink)
            l_t = gl.where(
                m_i > float("-inf"), l_i * gl.exp2(m_i - m_f - cfg.LOG2_PS), 0.0
            )
            if sink_ptr is not None:
                l_t = l_t + gl.exp2(sink - m_f)
            gl.store(
                lse_ptr + row.to(gl.int64) * self.num_heads + h0 + hs,
                (m_f + gl.log2(l_t)) * 0.6931471805599453,
                mask=hm,
            )
        x = acc * gl.convert_layout(r, gl.SliceLayout(1, acc.type.layout))[:, None]
        if INV_ROPE or OUT_MXFP8:
            # rows G .. GM are padding, so the head bound is h0 + G
            epi_l: gl.constexpr = gl.BlockedLayout(
                [1, 32], [16, 4], [cfg.NW // 4, 4], [1, 0]
            )
            _epilogue_store(
                x, out_ptr, out_stride0, out_stride1, row, h0, h0 + G, epi_l, epi[0],
                epi[1], epi[2], epi[3], epi[4], 64, INV_ROPE, OUT_MXFP8,
            )  # fmt: skip
        else:
            o = gl.convert_layout(x.to(gl.bfloat16), cfg.WIDE_L)
            ho = gl.arange(0, cfg.GM, layout=gl.SliceLayout(1, cfg.WIDE_L))
            co = gl.arange(0, D, layout=gl.SliceLayout(0, cfg.WIDE_L))
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
    part_m_ptr,  # split partials (_Program.store_partial); None unless split
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
    cfg = _Config(G, gl.num_warps(), ROPE, LAZY, BIG)
    SPLIT: gl.constexpr = part_m_ptr is not None
    Q_BF16: gl.constexpr = q_scale_ptr is None
    pgm = _Program.initialize(
        cfg, XH, q_ptr, q_stride0, q_stride1, num_heads, num_splits
    )
    if XH > 1:
        if pgm.row >= num_queries:
            return
    if SPLIT:
        if pgm.split >= num_splits:  # padding of the launched split count
            m0 = gl.full([cfg.GM], float("-inf"), gl.float32, cfg.ROW_L)
            pgm.store_partial(
                None, m0, gl.zeros_like(m0), 0.0, part_m_ptr, part_l_ptr, part_acc_ptr
            )
            return

    # base-2 score scale per row: q scale x cache scale x softmax scale
    k_scale = gl.load(ks_ptr)
    if Q_BF16:
        # quantized before the ring is live, so its LDS scratch aliases the ring
        q, q_s = pgm.q_bf16()
        c = q_s * (k_scale * sm_scale_log2)
    else:
        q = None  # loaded by the loop, behind its first gather
        c = gl.zeros([cfg.GM], gl.float32, cfg.ROW_L) + (
            gl.load(q_scale_ptr) * k_scale * sm_scale_log2
        )

    if SEG == 2:
        # The SWA window from kv, then the top-k from x_kv. A split takes a range of
        # the concatenated tiles, so it may cover part of either list or both.
        if not Q_BF16:
            q = pgm.q_fp8()
        row = pgm.row
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
        u0, u1 = pgm.tiles(nta + gl.cdiv(nb, BN))
        st = pgm.loop(
            q, kv_ptr, (main, main, na), 0, na, gl.minimum(u0, nta),
            gl.minimum(u1, nta), c, None,
        )  # fmt: skip
        if x_ks_ptr is not None:  # the first list's V in the second one's units
            st = (st[0] * (k_scale / v_scale), st[1], st[2])
        acc, m_i, l_i = pgm.loop(
            q, x_kv_ptr, (extra, extra, 0), 0, nb, gl.maximum(u0 - nta, 0),
            gl.maximum(u1 - nta, 0), c_b, st,
        )  # fmt: skip
    else:
        start, end, ix = _key_lists(
            indptr_ptr, idx_ptr, x_indptr_ptr, x_idx_ptr, pgm.row, mo, mstr, xo, xstr,
            SEG, MB_SH, XB_SH,
        )  # fmt: skip
        t0, t1 = pgm.tiles(gl.cdiv(end - start, BN))
        acc, m_i, l_i = pgm.loop(q, kv_ptr, ix, start, end - start, t0, t1, c, None)
        v_scale = k_scale
    if cfg.KG > 1:
        l_i = gl.sum(l_i, axis=1)

    if SPLIT:
        pgm.store_partial(acc, m_i, l_i, v_scale, part_m_ptr, part_l_ptr, part_acc_ptr)
    else:
        pgm.finish(
            acc, m_i, l_i, v_scale, sink_ptr, lse_ptr, out_ptr, out_stride0, out_stride1,
            (pos_ptr, cs_ptr, cs_stride, osc_ptr, osc_stride),
        )  # fmt: skip
