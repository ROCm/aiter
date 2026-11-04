# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950) sparse MLA on a per-tensor fp8 KV cache ("fp8_scalar") with fp8
dots. sparse_mla_fwd (attention/sparse_mla.py) launches it whenever it can address
the call
"""

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import _core as _gl_core
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_mla import (
    _epilogue_store,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.common_utils import strip_annotate

LOG2E: gl.constexpr = gl.constexpr(1.4426950408889634)
LN2: gl.constexpr = gl.constexpr(0.6931471805599453)
E4M3_MAX: gl.constexpr = gl.constexpr(448.0)
_LOG2_E4M3_MAX = math.log2(448.0)
BN: gl.constexpr = gl.constexpr(64)  # keys per tile
D: gl.constexpr = gl.constexpr(512)  # latent width = V width
INV_ROPE_DIM: gl.constexpr = gl.constexpr(64)  # dims the output's inverse RoPE turns

# Triton 3.8 spells shared_memory_descriptor.reinterpret as _reinterpret and has
# no gl.amd.hint.sched_barrier (sched_barrier falls back to an asm fence); both
# settled at compile time.
_HAS_REINTERPRET = gl.constexpr(
    hasattr(_gl_core.shared_memory_descriptor, "reinterpret")
)
_HAS_SCHED_BARRIER = gl.constexpr(
    hasattr(gl.amd, "hint") and hasattr(gl.amd.hint, "sched_barrier")
)


@gluon.jit
def sched_barrier():
    if _HAS_SCHED_BARRIER:
        gl.amd.hint.sched_barrier()
    else:
        # Triton 3.8 has no sched_barrier. An empty asm fence keeps memory ops in
        # place but not MFMAs, so the K read-ahead only partly holds there.
        fence_l: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
        gl.inline_asm_elementwise(
            "; fence", "=v,v", [gl.full([64], 0, gl.int32, fence_l)], dtype=gl.int32,
            is_pure=False, pack=1,
        )  # fmt: skip


# The loop's f32 math as unpacked VOP3 ops. Left to itself LLVM pairs them into
# v_pk_*, which cannot issue while an MFMA is in flight; these can, and two waves
# can dual-issue them.
@gluon.jit
def fma(a, b, c):
    return gl.inline_asm_elementwise(
        "v_fma_f32 $0, $1, $2, $3", "=v,v,v,v", [a, b, c], dtype=gl.float32,
        is_pure=True, pack=1,
    )  # fmt: skip


@gluon.jit
def mul(a, b):
    return gl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2", "=v,v,v", [a, b], dtype=gl.float32, is_pure=True,
        pack=1,
    )  # fmt: skip


@gluon.jit
def add(a, b):
    return gl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2", "=v,v,v", [a, b], dtype=gl.float32, is_pure=True,
        pack=1,
    )  # fmt: skip


# For operands a v_exp_f32 may have just written: LLVM pads the trans forwarding
# hazard for its own instructions but not for inline asm, which reads stale lanes.
@gluon.jit
def add_after_trans(a, b):
    return gl.inline_asm_elementwise(
        "s_nop 0\n\tv_add_f32 $0, $1, $2", "=v,v,v", [a, b], dtype=gl.float32,
        is_pure=True, pack=1,
    )  # fmt: skip


@gluon.jit
def fma_after_trans(a, b, c):
    return gl.inline_asm_elementwise(
        "s_nop 0\n\tv_fma_f32 $0, $1, $2, $3", "=v,v,v,v", [a, b, c],
        dtype=gl.float32, is_pure=True, pack=1,
    )  # fmt: skip


@gluon.jit
def relayout(smem, layout: gl.constexpr):
    """The same LDS bytes under another shared layout."""
    if _HAS_REINTERPRET:
        r = smem.reinterpret(smem.dtype, smem.shape, layout)
    else:
        r = smem._reinterpret(smem.dtype, smem.shape, layout)
    return r


@aggregate
@strip_annotate
class Config:
    """Compile-time shape and layouts. Shared layouts end in _SH, distributed
    ones in _L."""

    BLOCK_H: gl.constexpr  # heads per program
    NUM_WARPS: gl.constexpr
    ROPE_TAIL: gl.constexpr  # RoPE bytes after the 512 latent bytes
    LAZY_LOG2: gl.constexpr
    ADDR64: gl.constexpr  # cache past 2 GB
    BLOCK_M: gl.constexpr  # MFMA rows, max(BLOCK_H, 16)
    QK_KEY_WAVES: gl.constexpr  # waves sharing a tile's keys in QK
    WHOLE_ROWS: gl.constexpr  # QK_KEY_WAVES == 1
    P_SCALE_LOG2: gl.constexpr
    INV_P_SCALE: gl.constexpr
    QK_MFMA: gl.constexpr
    ROPE_MFMA: gl.constexpr
    PV_MFMA: gl.constexpr
    PV_SCALED: gl.constexpr  # 32x32x64 scaled PV, else 16x16x32
    PV_K_WIDTH: gl.constexpr
    GATHER_L: gl.constexpr
    GATHER_ROPE_L: gl.constexpr
    K_SH: gl.constexpr  # ring tile as written and read as K
    V_SH: gl.constexpr  # ring tile read as V^T
    ROPE_SH: gl.constexpr
    P_SH: gl.constexpr  # P as the softmax writes it
    P_PV_SH: gl.constexpr  # P as PV reads it
    ROW_L: gl.constexpr  # one value per head
    KEY_L: gl.constexpr  # one value per gathered key
    WIDE_L: gl.constexpr  # [BLOCK_M, 512], 16 B per lane

    @gluon.constexpr_function
    def __init__(self, BLOCK_H, NUM_WARPS, ROPE_TAIL, LAZY_LOG2, ADDR64):
        BLOCK_M = max(BLOCK_H, 16)
        head_waves = min(NUM_WARPS, BLOCK_M // 16)
        key_waves = NUM_WARPS // head_waves
        qk = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [head_waves, key_waves])
        # LDS image of a [64, 512] tile: a 16 B run of dims, then 1 KB blocks with a
        # pad after each. The bit orders keep the 32-lane passes of the K reads
        # (ds_read_b128) and V^T reads (ds_read_b64_tr_b8) on distinct banks.
        if BLOCK_M >= 32:
            # 32x32x64 scaled PV; key3 is block bit 3, moved 256 B by the 32 B pad
            pv = gl.amd.AMDMFMALayout(
                4, [32, 32, 64], True, [1, NUM_WARPS],
                tiles_per_warp=[1, 512 // 32 // NUM_WARPS],
            )  # fmt: skip
            pv_k_width = 16
            pad = 32
            kv = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [0, 32],
                  [0, 64], [0, 128], [0, 256], [16, 0], [8, 0], [32, 0]]  # fmt: skip
            vv = kv
            pp = None
        else:
            # 16x16x32 PV, whose V^T pass varies key3: read V and P with key bits 3
            # and 4 swapped (the same permutation on both, so PV is unchanged)
            pv = gl.amd.AMDMFMALayout(
                4, [16, 16, 32], True, [1, NUM_WARPS],
                tiles_per_warp=[1, 512 // 16 // NUM_WARPS],
            )  # fmt: skip
            pv_k_width = 8
            pad = 16
            kv = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [0, 32],
                  [0, 64], [16, 0], [0, 128], [0, 256], [32, 0], [8, 0]]  # fmt: skip
            vv = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [0, 32],
                  [0, 64], [8, 0], [0, 128], [0, 256], [32, 0], [16, 0]]  # fmt: skip
            pp = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 8], [0, 32]]
            pp = pp + [[1 << i, 0] for i in range(BLOCK_M.bit_length() - 1)]
        k_sh = gl.PaddedSharedLayout([[1024, pad]], kv, [], [64, 512])
        p_sh = gl.PaddedSharedLayout.with_identity_for(
            [[64, 16]], [BLOCK_M, 64], [1, 0]
        )
        # A lane per 16 B of a 1 KB block. Dims come before keys on registers, since
        # register order is issue order and an early key bit makes the next gather
        # wait on all of them.
        warp_bits = NUM_WARPS.bit_length() - 1
        block = sorted(kv[10:], key=lambda b: b[0] != 0)
        reg_blocks = len(block) - warp_bits
        gather = gl.DistributedLinearLayout(
            reg_bases=kv[:4] + block[:reg_blocks],
            lane_bases=kv[4:10],
            warp_bases=block[reg_blocks:],
            block_bases=[],
            shape=[64, 512],
        )
        rope_sh = k_sh
        gather_rope = gather
        if ROPE_TAIL > 0:
            assert ROPE_TAIL == 64 and key_waves > 1, "the RoPE tile needs key waves"
            rb = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0], [0, 16],
                  [0, 32], [16, 0], [32, 0]]  # fmt: skip
            rope_sh = gl.PaddedSharedLayout([[1024, 16]], rb, [], [64, ROPE_TAIL])
            # the tail has two block bits, so waves 4-7 copy the same bytes again
            nr = min(warp_bits, 2)
            gather_rope = gl.DistributedLinearLayout(
                reg_bases=rb[:4] + rb[10:][: 2 - nr],
                lane_bases=rb[4:10],
                warp_bases=rb[10:][2 - nr :] + [[0, 0]] * (warp_bits - nr),
                block_bases=[],
                shape=[64, ROPE_TAIL],
            )

        self.BLOCK_H = gl.constexpr(BLOCK_H)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.ROPE_TAIL = gl.constexpr(ROPE_TAIL)
        self.LAZY_LOG2 = gl.constexpr(LAZY_LOG2)
        self.ADDR64 = gl.constexpr(ADDR64)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.QK_KEY_WAVES = gl.constexpr(key_waves)
        self.WHOLE_ROWS = gl.constexpr(key_waves == 1)
        self.P_SCALE_LOG2 = gl.constexpr(_LOG2_E4M3_MAX - LAZY_LOG2)
        self.INV_P_SCALE = gl.constexpr(2.0 ** (LAZY_LOG2 - _LOG2_E4M3_MAX))
        self.QK_MFMA = gl.constexpr(qk)
        self.ROPE_MFMA = gl.constexpr(
            gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [head_waves, key_waves])
        )
        self.PV_MFMA = gl.constexpr(pv)
        self.PV_SCALED = gl.constexpr(BLOCK_M >= 32)
        self.PV_K_WIDTH = gl.constexpr(pv_k_width)
        self.GATHER_L = gl.constexpr(gather)
        self.GATHER_ROPE_L = gl.constexpr(gather_rope)
        self.K_SH = gl.constexpr(k_sh)
        self.V_SH = gl.constexpr(
            gl.PaddedSharedLayout([[1024, pad]], vv, [], [64, 512])
        )
        self.ROPE_SH = gl.constexpr(rope_sh)
        self.P_SH = gl.constexpr(p_sh)
        if pp is None:
            self.P_PV_SH = gl.constexpr(p_sh)
        else:
            self.P_PV_SH = gl.constexpr(
                gl.PaddedSharedLayout([[64, 16]], pp, [], [BLOCK_M, 64])
            )
        self.ROW_L = gl.constexpr(gl.SliceLayout(1, qk))
        self.KEY_L = gl.constexpr(gl.SliceLayout(1, gather))
        self.WIDE_L = gl.constexpr(
            gl.BlockedLayout([1, 8], [1, 64], [NUM_WARPS, 1], [1, 0])
        )


# A key list ix is (ids,) for a flat pool, or (main, extra, n_main) for paged
# slot ids, read as main's first n_main keys then extra's. A paged list is
# (ids, row0, stride, block_log2): slot s is row
# row0 + (s >> block_log2) * stride + s % 2^block_log2.
# Dispatch is on len(ix), since tuple members lose constexpr-ness in a callee.


@gluon.jit
def one_loop_list(
    indptr_ptr,
    idx_ptr,
    x_indptr_ptr,
    x_idx_ptr,
    row,
    main_row0,
    main_stride,
    extra_row0,
    extra_stride,
    PAGED: gl.constexpr,
    MAIN_BLOCK_LOG2: gl.constexpr,
    EXTRA_BLOCK_LOG2: gl.constexpr,
):
    """(start, end, ix) of a row read in one loop. start and end only feed the
    masks and the split."""
    start = gl.load(indptr_ptr + row)
    end = gl.load(indptr_ptr + row + 1)
    if not PAGED:
        ix = (idx_ptr + start,)
    else:
        main = (idx_ptr + start, main_row0, main_stride, MAIN_BLOCK_LOG2)
        if x_indptr_ptr is None:  # one paged list
            ix = (main, main, end - start)
        else:
            x_start = gl.load(x_indptr_ptr + row)
            x_end = gl.load(x_indptr_ptr + row + 1)
            extra = (x_idx_ptr + x_start, extra_row0, extra_stride, EXTRA_BLOCK_LOG2)
            ix = (main, extra, end - start)
            start = start + x_start
            end = end + x_end
    return start, end, ix


@gluon.jit
def load_row_ids(ix, k, n, KL: gl.constexpr):
    """Start loading the row ids of keys k .. k + 64, clamped to n - 1; resolve
    them with resolve_row_ids after the next wait. An empty list loads index -1,
    which the buffer load reads as 0: one harmless gather of row 0. Paged ids load
    one per lane and are spread to KL later, which is cheaper than mapping every
    gathered register."""
    if len(ix) > 1:
        main, extra, n_main = ix
        cl: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
        j = gl.minimum(k + gl.arange(0, BN, layout=cl), n - 1)
        # no other=: a masked buffer load reads 0, while other=0 is a select right
        # after the load, an s_waitcnt vmcnt(0) behind the gathers in flight
        s_main = gl.amd.cdna4.buffer_load(main[0], j, mask=j < n_main)
        s_extra = gl.amd.cdna4.buffer_load(extra[0], j - n_main, mask=j >= n_main)
        r = (s_main, s_extra, k, n)
    else:
        kg = gl.arange(0, BN, layout=KL)
        r = (gl.amd.cdna4.buffer_load(ix[0], gl.minimum(k + kg, n - 1)),)
    return r


@gluon.jit
def slot_to_row(s, lst):
    _, row0, stride, block_log2 = lst
    return row0 + (s >> block_log2) * stride + (s & ((1 << block_log2) - 1))


@gluon.jit
def resolve_row_ids(ix, r, KL: gl.constexpr):
    """The row ids in layout KL, from load_row_ids' r."""
    if len(ix) > 1:
        main, extra, n_main = ix
        # The ids are lane-uniform: left alone, the backend moves them to SGPRs
        # (v_readlane) right after the load, an s_waitcnt behind the gathers in
        # flight. A non-pure VGPR copy keeps that math after the step's wait.
        s_main = gl.inline_asm_elementwise(
            "v_mov_b32 $0, $1", "=v,v", [r[0]], dtype=gl.int32, is_pure=False, pack=1
        )
        s_extra = gl.inline_asm_elementwise(
            "v_mov_b32 $0, $1", "=v,v", [r[1]], dtype=gl.int32, is_pure=False, pack=1
        )
        k = r[2]
        n = r[3]
        keys = gl.arange(0, BN, layout=s_main.type.layout)
        in_main = gl.minimum(k + keys, n - 1) < n_main
        rid = gl.where(in_main, slot_to_row(s_main, main), slot_to_row(s_extra, extra))
        rid = gl.convert_layout(rid, KL)
    else:
        rid = r[0]
    return rid


@gluon.jit
def copy_rows(
    dst, kv_ptr, rid, cols, PITCH: gl.constexpr, OFF: gl.constexpr, ADDR64: gl.constexpr
):
    """Bytes OFF + cols of rows rid (row pitch PITCH) into dst, no commit."""
    if ADDR64:
        # a 64-bit row base: buffer offsets are i32
        rp = kv_ptr + rid.to(gl.int64) * PITCH + OFF
        gl.amd.cdna4.async_copy.global_load_to_shared(dst, rp[:, None] + cols[None, :])
    else:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dst, kv_ptr, (rid * PITCH + OFF)[:, None] + cols[None, :]
        )


@gluon.jit
def gather_tile(cfg, kv_dst, rope_dst, kv_ptr, ix, r, cols):
    """One tile into a ring slot (latent and RoPE tail), as one commit group."""
    rid = resolve_row_ids(ix, r, cfg.KEY_L)
    PITCH: gl.constexpr = D + cfg.ROPE_TAIL
    copy_rows(kv_dst, kv_ptr, rid, cols[0], PITCH, 0, cfg.ADDR64)
    if cfg.ROPE_TAIL > 0:
        rid_r = gl.convert_layout(rid, gl.SliceLayout(1, cfg.GATHER_ROPE_L))
        copy_rows(rope_dst, kv_ptr, rid_r, cols[1], PITCH, D, cfg.ADDR64)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def mask_tail(S, k0, end):
    """-inf on the keys k0 + column of S past end (the row's last tile only)."""
    kcol = gl.arange(0, S.shape[1], layout=gl.SliceLayout(0, S.type.layout))
    if k0 + S.shape[1] > end:
        S = gl.where((k0 + kcol < end)[None, :], S, float("-inf"))
    return S


@gluon.jit
def qk_tile(cfg, kvt, q_dot):
    """Scores of the whole tile: 16x16x128 scaled MFMAs, unit scales."""
    k_op: gl.constexpr = gl.DotOperandLayout(1, cfg.QK_MFMA, 16)
    k = kvt.permute([1, 0]).load(k_op)
    acc = gl.zeros([q_dot.shape[0], BN], gl.float32, cfg.QK_MFMA)
    return gl.amd.cdna4.mfma_scaled(
        q_dot, None, "e4m3", k.to(gl.float8e4nv, bitcast=True), None, "e4m3", acc
    )


@gluon.jit
def k_chunk(cfg, kvt, i: gl.constexpr, NK: gl.constexpr):
    """K chunk i of the tile, as the QK MFMA's B operand: keys (i // 4) * NK ..
    + NK, dims (i % 4) * 128 .. + 128."""
    k_op: gl.constexpr = gl.DotOperandLayout(1, cfg.QK_MFMA, 16)
    kc = kvt.slice((i // 4) * NK, NK, dim=0).slice((i % 4) * 128, 128, dim=1)
    return kc.permute([1, 0]).load(k_op).to(gl.float8e4nv, bitcast=True)


@gluon.jit
def qk_halves(cfg, kvt, qc):
    """Scores of the tile's two 32-key halves, one accumulator each, q in 128-dim
    chunks qc. Each K chunk is read before the previous chunk's MFMAs; the fences
    hold that order, since left alone LLVM moves every read down to its MFMA and
    waits lgkmcnt(0) there."""
    HB: gl.constexpr = BN // 2
    S0 = gl.zeros([qc[0].shape[0], HB], gl.float32, cfg.QK_MFMA)
    S1 = S0
    k = k_chunk(cfg, kvt, 0, HB)
    for i in gl.static_range(8):
        kn = k
        if i + 1 < 8:
            kn = k_chunk(cfg, kvt, i + 1, HB)
        sched_barrier()
        if i < 4:
            S0 = gl.amd.cdna4.mfma_scaled(qc[i], None, "e4m3", k, None, "e4m3", S0)
        else:
            S1 = gl.amd.cdna4.mfma_scaled(qc[i - 4], None, "e4m3", k, None, "e4m3", S1)
        sched_barrier()
        k = kn
    return S0, S1


@gluon.jit
def tile_scores(cfg, kvt, ropet, q, k0, end):
    """(S0, S1, row max): the two 32-key halves with WHOLE_ROWS, else the whole
    tile in S0 (S1 = S0)."""
    if cfg.WHOLE_ROWS:
        # K read-ahead only here: with key waves the fences and chunked q cost more
        HB: gl.constexpr = BN // 2
        S0, S1 = qk_halves(cfg, kvt, q[0])
        S0 = mask_tail(S0, k0, end)
        S1 = mask_tail(S1, k0 + HB, end)
        mx = gl.maximum(gl.max(S0, axis=1), gl.max(S1, axis=1))
    else:
        S0 = qk_tile(cfg, kvt, q[0])
        if cfg.ROPE_TAIL > 0:
            kr_op: gl.constexpr = gl.DotOperandLayout(1, cfg.ROPE_MFMA, 8)
            kr = ropet.permute([1, 0]).load(kr_op).to(gl.float8e4nv, bitcast=True)
            # Its own accumulator, added after: chaining onto S0 makes this MFMA
            # read another shape's MFMA result, a hazard Triton 3.8's LLVM pads
            # with no wait states (nondeterministic scores).
            sr = gl.amd.cdna4.mfma(
                q[1], kr, gl.zeros(S0.shape, gl.float32, cfg.ROPE_MFMA)
            )
            S0 = S0 + gl.convert_layout(sr, cfg.QK_MFMA)
        S0 = mask_tail(S0, k0, end)
        S1 = S0
        mx = gl.max(S0, axis=1)
    return S0, S1, mx


@gluon.jit
def softmax_to_lds(cfg, S0, S1, mx, c, m_i, l_i, p_smem, alpha_smem):
    """Writes P (e4m3) and the rescale alpha to LDS, returns (m, l). l is
    [BLOCK_M], or with key waves a [BLOCK_M, 64] sum reduced at the end."""
    # c > 0, so the max of the scaled scores is the scaled max
    m_cand = gl.maximum(m_i, mx * c)
    m_new = gl.where(m_cand > m_i + cfg.LAZY_LOG2, m_cand, m_i)
    # A row with no key so far keeps m = -inf: subtract 0, rescale by 1.
    live = m_new > float("-inf")
    alpha = gl.where(live, gl.exp2(m_i - m_new), 1.0)
    m_sub = gl.where(live, m_new - cfg.P_SCALE_LOG2, 0.0)
    c2 = gl.broadcast(c[:, None], S0)[0]
    nm = gl.broadcast(-m_sub[:, None], S0)[0]
    p0 = gl.exp2(fma(S0, c2, nm))
    if cfg.WHOLE_ROWS:
        HB: gl.constexpr = BN // 2
        p1 = gl.exp2(fma(S1, c2, nm))
        l_i = fma(l_i, alpha, gl.reduce(add_after_trans(p0, p1), 1, add))
        p_smem.slice(0, HB, dim=1).store(
            p0.to(gl.float8e4nv).to(gl.uint8, bitcast=True)
        )
        p_smem.slice(HB, HB, dim=1).store(
            p1.to(gl.float8e4nv).to(gl.uint8, bitcast=True)
        )
    else:
        # deferred row sum: no cross-wave reduction per tile
        l_i = fma_after_trans(l_i, gl.broadcast(alpha[:, None], S0)[0], p0)
        p_smem.store(p0.to(gl.float8e4nv).to(gl.uint8, bitcast=True))
    alpha_smem.store(alpha)
    return m_new, l_i


@gluon.jit
def tile_pv(cfg, kvt, p_smem, alpha_smem, acc):
    """acc * alpha + P V, once a barrier has published P."""
    # a uniform branch: on most tiles no row max moved
    alpha = alpha_smem.load(gl.SliceLayout(1, cfg.PV_MFMA))
    if gl.min(alpha, axis=0) < 1.0:
        acc = mul(acc, gl.broadcast(alpha[:, None], acc)[0])
    p_op: gl.constexpr = gl.DotOperandLayout(0, cfg.PV_MFMA, cfg.PV_K_WIDTH)
    v_op: gl.constexpr = gl.DotOperandLayout(1, cfg.PV_MFMA, cfg.PV_K_WIDTH)
    v = relayout(kvt, cfg.V_SH).load(v_op).to(gl.float8e4nv, bitcast=True)
    p = relayout(p_smem, cfg.P_PV_SH).load(p_op).to(gl.float8e4nv, bitcast=True)
    if cfg.PV_SCALED:
        acc = gl.amd.cdna4.mfma_scaled(p, None, "e4m3", v, None, "e4m3", acc)
    else:
        acc = gl.amd.cdna4.mfma(p, v, acc)
    return acc


@gluon.jit
def attend_tile(cfg, kvt, ropet, p_smem, alpha_smem, q, c, acc, m_i, l_i, k0, end):
    """One tile: keys k0 .. k0 + 64 of a row ending at end."""
    S0, S1, mx = tile_scores(cfg, kvt, ropet, q, k0, end)
    m_i, l_i = softmax_to_lds(cfg, S0, S1, mx, c, m_i, l_i, p_smem, alpha_smem)
    gl.barrier()
    acc = tile_pv(cfg, kvt, p_smem, alpha_smem, acc)
    return acc, m_i, l_i


@aggregate
@strip_annotate
class Program:
    """One query row, BLOCK_H heads from h0, one split of the row's keys."""

    cfg: Config
    row: gl.tensor
    h0: gl.tensor
    split: gl.tensor
    num_splits: gl.tensor
    num_heads: gl.constexpr
    q_ptr: gl.tensor  # head h0 of the row's q
    q_stride: gl.constexpr  # between heads

    @gluon.jit
    def initialize(
        cfg,
        HEAD_BLOCKS: gl.constexpr,
        NUM_XCDS: gl.constexpr,
        q_ptr,
        q_stride0: gl.constexpr,
        q_stride1: gl.constexpr,
        num_heads: gl.constexpr,
        num_splits,
    ):
        # A row's head blocks run at program ids L, L + NUM_XCDS, ..: on one XCD,
        # so the later blocks hit the first one's gathers in L2.
        L = gl.program_id(0)
        row = (L // (NUM_XCDS * HEAD_BLOCKS)) * NUM_XCDS + L % NUM_XCDS
        h0 = (L // NUM_XCDS) % HEAD_BLOCKS * cfg.BLOCK_H
        return Program(
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
        """Columns C0 .. C0 + NC; padding heads read 0."""
        BLOCK_H: gl.constexpr = self.cfg.BLOCK_H
        BLOCK_M: gl.constexpr = self.cfg.BLOCK_M
        hq = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, layout))
        cq = gl.arange(0, NC, layout=gl.SliceLayout(0, layout))
        off = hq[:, None] * self.q_stride + cq[None, :]
        if BLOCK_H < BLOCK_M:
            q = gl.amd.cdna4.buffer_load(
                self.q_ptr + C0, off, mask=(hq < BLOCK_H)[:, None]
            )
        else:
            q = gl.amd.cdna4.buffer_load(self.q_ptr + C0, off)
        return q

    @gluon.jit
    def q_fp8(self):
        """(latent, RoPE) dot operands; the latent in four 128-dim chunks with
        WHOLE_ROWS. Without a RoPE tail the second is a placeholder."""
        cfg = self.cfg
        q_op: gl.constexpr = gl.DotOperandLayout(0, cfg.QK_MFMA, 16)
        if cfg.WHOLE_ROWS:
            q_dot = (
                self.load_q(0, 128, q_op).to(gl.float8e4nv, bitcast=True),
                self.load_q(128, 128, q_op).to(gl.float8e4nv, bitcast=True),
                self.load_q(256, 128, q_op).to(gl.float8e4nv, bitcast=True),
                self.load_q(384, 128, q_op).to(gl.float8e4nv, bitcast=True),
            )
        else:
            q_dot = self.load_q(0, D, q_op).to(gl.float8e4nv, bitcast=True)
        if cfg.ROPE_TAIL > 0:
            qr_op: gl.constexpr = gl.DotOperandLayout(0, cfg.ROPE_MFMA, 8)
            qr_dot = self.load_q(D, cfg.ROPE_TAIL, qr_op).to(
                gl.float8e4nv, bitcast=True
            )
        elif cfg.WHOLE_ROWS:
            qr_dot = q_dot[0]
        else:
            qr_dot = q_dot
        return q_dot, qr_dot

    @gluon.jit
    def q_bf16(self):
        """q_fp8's operands from bf16 q, plus the per-head scale."""
        cfg = self.cfg
        if cfg.WHOLE_ROWS:
            ql: gl.constexpr = gl.BlockedLayout(
                [1, 8], [4, 16], [cfg.NUM_WARPS, 1], [1, 0]
            )
            q0 = self.load_q(0, 128, ql).to(gl.float32)
            q1 = self.load_q(128, 128, ql).to(gl.float32)
            q2 = self.load_q(256, 128, ql).to(gl.float32)
            q3 = self.load_q(384, 128, ql).to(gl.float32)
            amax = gl.maximum(
                gl.maximum(gl.max(gl.abs(q0), axis=1), gl.max(gl.abs(q1), axis=1)),
                gl.maximum(gl.max(gl.abs(q2), axis=1), gl.max(gl.abs(q3), axis=1)),
            )
        else:
            q = self.load_q(0, D, cfg.WIDE_L).to(gl.float32)
            amax = gl.max(gl.abs(q), axis=1)
        if cfg.ROPE_TAIL > 0:
            rl: gl.constexpr = gl.BlockedLayout(
                [1, 8], [8, 8], [cfg.NUM_WARPS, 1], [1, 0]
            )
            qr = self.load_q(D, cfg.ROPE_TAIL, rl).to(gl.float32)
            amax_r = gl.convert_layout(gl.max(gl.abs(qr), axis=1), amax.type.layout)
            amax = gl.maximum(amax, amax_r)
        s = gl.maximum(amax, 1e-30) / E4M3_MAX
        q_op: gl.constexpr = gl.DotOperandLayout(0, cfg.QK_MFMA, 16)
        if cfg.WHOLE_ROWS:
            q_dot = (
                gl.convert_layout((q0 / s[:, None]).to(gl.float8e4nv), q_op),
                gl.convert_layout((q1 / s[:, None]).to(gl.float8e4nv), q_op),
                gl.convert_layout((q2 / s[:, None]).to(gl.float8e4nv), q_op),
                gl.convert_layout((q3 / s[:, None]).to(gl.float8e4nv), q_op),
            )
            qr_dot = q_dot[0]
        else:
            q_dot = gl.convert_layout((q / s[:, None]).to(gl.float8e4nv), q_op)
            qr_dot = q_dot
        if cfg.ROPE_TAIL > 0:
            sr = gl.convert_layout(s, gl.SliceLayout(1, rl))
            qr_dot = gl.convert_layout(
                (qr / sr[:, None]).to(gl.float8e4nv),
                gl.DotOperandLayout(0, cfg.ROPE_MFMA, 8),
            )
        return (q_dot, qr_dot), gl.convert_layout(s, cfg.ROW_L)

    @gluon.jit
    def loop(self, q, kv_ptr, ix, start, n, t0, t1, c, init):
        """Tiles t0 .. t1 of the n-key list ix, continuing from init = (acc, m, l)
        if given. q = None loads fp8 q here, behind the first gather."""
        cfg = self.cfg
        BLOCK_M: gl.constexpr = cfg.BLOCK_M
        ROPE_TAIL: gl.constexpr = cfg.ROPE_TAIL

        # one allocation per slot, so the slots have static addresses
        kv0 = gl.allocate_shared_memory(gl.uint8, [BN, D], cfg.K_SH)
        kv1 = gl.allocate_shared_memory(gl.uint8, [BN, D], cfg.K_SH)
        # built once here; inside gather_tile they cost a spill at 64 heads
        cg = gl.arange(0, D, layout=gl.SliceLayout(0, cfg.GATHER_L))
        if ROPE_TAIL > 0:
            rope0 = gl.allocate_shared_memory(gl.uint8, [BN, ROPE_TAIL], cfg.ROPE_SH)
            rope1 = gl.allocate_shared_memory(gl.uint8, [BN, ROPE_TAIL], cfg.ROPE_SH)
            cr = gl.arange(0, ROPE_TAIL, layout=gl.SliceLayout(0, cfg.GATHER_ROPE_L))
            cols = (cg, cr)
        else:
            rope0 = kv0
            rope1 = kv1
            cols = (cg, cg)
        p_smem = gl.allocate_shared_memory(gl.uint8, [BLOCK_M, BN], cfg.P_SH)
        alpha_smem = gl.allocate_shared_memory(
            gl.float32, [BLOCK_M], gl.SwizzledSharedLayout(1, 1, 1, [0])
        )

        if init is None:
            m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, cfg.ROW_L)
            if cfg.WHOLE_ROWS:
                l_i = gl.zeros([BLOCK_M], gl.float32, cfg.ROW_L)
            else:
                l_i = gl.zeros([BLOCK_M, BN], gl.float32, cfg.QK_MFMA)
            acc = gl.zeros([BLOCK_M, D], gl.float32, cfg.PV_MFMA)
        else:
            acc, m_i, l_i = init

        # r: the row ids of the next tile to gather
        n_s = gl.minimum(n, t1 * BN)
        kb = t0 * BN
        if q is None:
            # its latency hides behind the list bounds -> row ids -> gather chain
            q = self.q_fp8()
        r = load_row_ids(ix, kb, n_s, cfg.KEY_L)
        gather_tile(cfg, kv0, rope0, kv_ptr, ix, r, cols)
        r = load_row_ids(ix, kb + BN, n_s, cfg.KEY_L)

        # Unrolled over the two slots. An odd tile count runs one fully masked
        # tile, which reads nothing of the next split: its keys end at n_s.
        end_s = start + n_s
        for it in range(gl.cdiv(t1 - t0, 2)):
            t = it * 2
            # tile t landed; the wait's barrier also retires the reads of the other slot
            gl.amd.cdna4.async_copy.wait_group(0)
            gather_tile(cfg, kv1, rope1, kv_ptr, ix, r, cols)
            r = load_row_ids(ix, kb + (t + 2) * BN, n_s, cfg.KEY_L)
            acc, m_i, l_i = attend_tile(
                cfg, kv0, rope0, p_smem, alpha_smem, q, c, acc, m_i, l_i,
                start + kb + t * BN, end_s,
            )  # fmt: skip
            gl.amd.cdna4.async_copy.wait_group(0)
            gather_tile(cfg, kv0, rope0, kv_ptr, ix, r, cols)
            r = load_row_ids(ix, kb + (t + 3) * BN, n_s, cfg.KEY_L)
            acc, m_i, l_i = attend_tile(
                cfg, kv1, rope1, p_smem, alpha_smem, q, c, acc, m_i, l_i,
                start + kb + (t + 1) * BN, end_s,
            )  # fmt: skip
        gl.amd.cdna4.async_copy.wait_group(0)
        return acc, m_i, l_i

    @gluon.jit
    def loop_two_lists(
        self, q, kv_ptr, x_kv_ptr, indptr_ptr, idx_ptr, x_indptr_ptr, x_idx_ptr,
        main_row0, main_stride, extra_row0, extra_stride, c, k_scale, x_ks_ptr,
        MAIN_BLOCK_LOG2: gl.constexpr, EXTRA_BLOCK_LOG2: gl.constexpr,
    ):  # fmt: skip
        """The SWA list from kv, then the top-k list from x_kv. A split's tile range
        may cover either list or both. Returns (acc, m, l, acc's V scale)."""
        row = self.row
        main_start = gl.load(indptr_ptr + row)
        n_main = gl.load(indptr_ptr + row + 1) - main_start
        x_start = gl.load(x_indptr_ptr + row)
        n_extra = gl.load(x_indptr_ptr + row + 1) - x_start
        main = (idx_ptr + main_start, main_row0, main_stride, MAIN_BLOCK_LOG2)
        extra = (x_idx_ptr + x_start, extra_row0, extra_stride, EXTRA_BLOCK_LOG2)
        if x_ks_ptr is None:
            x_scale = k_scale
            c_extra = c
        else:
            x_scale = gl.load(x_ks_ptr)
            c_extra = c * (x_scale / k_scale)
        main_tiles = gl.cdiv(n_main, BN)
        u0, u1 = self.tiles(main_tiles + gl.cdiv(n_extra, BN))
        st = self.loop(
            q, kv_ptr, (main, main, n_main), 0, n_main, gl.minimum(u0, main_tiles),
            gl.minimum(u1, main_tiles), c, None,
        )  # fmt: skip
        if x_ks_ptr is not None:  # the first list's V in the second one's units
            st = (st[0] * (k_scale / x_scale), st[1], st[2])
        acc, m_i, l_i = self.loop(
            q, x_kv_ptr, (extra, extra, 0), 0, n_extra,
            gl.maximum(u0 - main_tiles, 0), gl.maximum(u1 - main_tiles, 0), c_extra,
            st,
        )  # fmt: skip
        return acc, m_i, l_i, x_scale

    @gluon.jit
    def store_rows(self, x, base_ptr, row_stride):
        """x as bf16 rows, BLOCK_H heads at base_ptr + head * row_stride."""
        cfg = self.cfg
        o = gl.convert_layout(x.to(gl.bfloat16), cfg.WIDE_L)
        ho = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.WIDE_L))
        co = gl.arange(0, D, layout=gl.SliceLayout(0, cfg.WIDE_L))
        gl.amd.cdna4.buffer_store(
            o, base_ptr, ho[:, None] * row_stride + co[None, :],
            mask=(ho < cfg.BLOCK_H)[:, None],
        )  # fmt: skip

    @gluon.jit
    def store_partial(
        self, acc, m_i, l_i, v_scale, part_m_ptr, part_l_ptr, part_acc_ptr
    ):
        """This split's base-2 row max, row sum and unnormalized acc for
        _sparse_mla_reduce. acc = None writes an empty split."""
        cfg = self.cfg
        base = (self.row * gl.num_programs(2) + self.split).to(gl.int64)
        base = base * self.num_heads + self.h0
        hs = gl.arange(0, cfg.BLOCK_M, layout=m_i.type.layout)
        gl.store(part_m_ptr + base + hs, m_i, mask=hs < cfg.BLOCK_H)
        gl.store(part_l_ptr + base + hs, l_i * cfg.INV_P_SCALE, mask=hs < cfg.BLOCK_H)
        if acc is not None:
            x = acc * (v_scale * cfg.INV_P_SCALE)
            self.store_rows(x, part_acc_ptr + base * D, D)

    @gluon.jit
    def finish(
        self, acc, m_i, l_i, v_scale, sink_ptr, lse_ptr, out_ptr, out_stride0,
        out_stride1, epi,
    ):  # fmt: skip
        """Normalize (the sink adds to the denominator) and store the rows, and the
        natural-log LSE if asked. An empty row comes out 0."""
        cfg = self.cfg
        BLOCK_H: gl.constexpr = cfg.BLOCK_H
        pos_ptr, cs_ptr, cs_stride, osc_ptr, osc_stride = epi
        INV_ROPE: gl.constexpr = pos_ptr is not None
        OUT_MXFP8: gl.constexpr = osc_ptr is not None
        row = self.row
        h0 = self.h0
        hs = gl.arange(0, cfg.BLOCK_M, layout=m_i.type.layout)
        hm = hs < BLOCK_H
        if sink_ptr is not None:
            sink = gl.load(sink_ptr + h0 + hs, mask=hm, other=0.0) * LOG2E
            # an empty row gets l = inf here, so 0 (assumes a finite sink)
            r = v_scale / (l_i + gl.exp2(sink - m_i + cfg.P_SCALE_LOG2))
        else:
            r = gl.where(l_i > 0.0, v_scale / l_i, 0.0)
        if lse_ptr is not None:
            # relative to max(m, sink): finite for an empty row with a sink
            m_f = m_i
            if sink_ptr is not None:
                m_f = gl.maximum(m_i, sink)
            l_t = gl.where(
                m_i > float("-inf"), l_i * gl.exp2(m_i - m_f - cfg.P_SCALE_LOG2), 0.0
            )
            if sink_ptr is not None:
                l_t = l_t + gl.exp2(sink - m_f)
            gl.store(
                lse_ptr + row.to(gl.int64) * self.num_heads + h0 + hs,
                (m_f + gl.log2(l_t)) * LN2,
                mask=hm,
            )
        x = acc * gl.convert_layout(r, gl.SliceLayout(1, acc.type.layout))[:, None]
        if INV_ROPE or OUT_MXFP8:
            # 32 dims per lane = one MXFP8 scale block
            epi_l: gl.constexpr = gl.BlockedLayout(
                [1, 32], [16, 4], [cfg.NUM_WARPS // 4, 4], [1, 0]
            )
            _epilogue_store(
                x, out_ptr, out_stride0, out_stride1, row, h0, h0 + BLOCK_H, epi_l,
                pos_ptr, cs_ptr, cs_stride, osc_ptr, osc_stride, INV_ROPE_DIM,
                INV_ROPE, OUT_MXFP8,
            )  # fmt: skip
        else:
            base = out_ptr + row.to(gl.int64) * out_stride0 + h0 * out_stride1
            self.store_rows(x, base, out_stride1)


_sparse_mla_fp8_repr = make_kernel_repr(
    "_sparse_mla_fp8", ["BLOCK_H", "ROPE_TAIL", "LAZY_LOG2", "ADDR64", "HEAD_BLOCKS"]
)


@gluon.jit(repr=_sparse_mla_fp8_repr, do_not_specialize=["num_queries", "num_splits"])
def _sparse_mla_fp8(
    q_ptr,  # [C, H, 512 + ROPE_TAIL] fp8 (u8 view) or bf16
    kv_ptr,  # u8 base of the cache(s): rows of 512 + ROPE_TAIL bytes
    out_ptr,  # [C, H, 512] bf16 or MXFP8 e4m3 data; None when split
    indptr_ptr,  # [C + 1] int32, the main list
    idx_ptr,  # int32 row ids (flat pool) or slot ids
    sink_ptr,  # [H] f32 or None
    q_scale_ptr,  # [1] f32 for fp8 q, None for bf16 q
    ks_ptr,  # [1] f32 cache scale
    lse_ptr,  # [C, H] f32 or None; None when split
    part_m_ptr,  # split partials (Program.store_partial); None unless split
    part_l_ptr,
    part_acc_ptr,
    # paged lists: the extra list, both lists' slot -> row maps (row0, stride);
    # two loops add the extra cache and its scale (None: kv's scale)
    x_indptr_ptr,
    x_idx_ptr,
    main_row0,
    main_stride: gl.constexpr,
    extra_row0,
    extra_stride: gl.constexpr,
    x_kv_ptr,
    x_ks_ptr,
    # output epilogue, positions and cos_sin (+ stride)
    # for the inverse RoPE, E8M0 scales (+ stride) for MXFP8; None when off
    pos_ptr,
    cos_sin_ptr,
    cs_stride: gl.constexpr,
    out_scale_ptr,
    os_stride0: gl.constexpr,
    num_queries,
    num_heads: gl.constexpr,
    sm_scale,
    num_splits,
    q_stride0: gl.constexpr,
    q_stride1: gl.constexpr,
    out_stride0: gl.constexpr,
    out_stride1: gl.constexpr,
    BLOCK_H: gl.constexpr,
    ROPE_TAIL: gl.constexpr,
    LAZY_LOG2: gl.constexpr,
    ADDR64: gl.constexpr,
    MAIN_BLOCK_LOG2: gl.constexpr,
    EXTRA_BLOCK_LOG2: gl.constexpr,
    HEAD_BLOCKS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    cfg = Config(BLOCK_H, gl.num_warps(), ROPE_TAIL, LAZY_LOG2, ADDR64)
    SPLIT: gl.constexpr = part_m_ptr is not None
    Q_BF16: gl.constexpr = q_scale_ptr is None
    PAGED: gl.constexpr = main_stride is not None
    TWO_LOOPS: gl.constexpr = x_kv_ptr is not None
    pgm = Program.initialize(
        cfg, HEAD_BLOCKS, NUM_XCDS, q_ptr, q_stride0, q_stride1, num_heads, num_splits
    )
    if HEAD_BLOCKS > 1 and pgm.row >= num_queries:  # rows padded to the XCD count
        return
    if SPLIT and pgm.split >= num_splits:  # padding of the launched split count
        m0 = gl.full([cfg.BLOCK_M], float("-inf"), gl.float32, cfg.ROW_L)
        pgm.store_partial(
            None, m0, gl.zeros_like(m0), 0.0, part_m_ptr, part_l_ptr, part_acc_ptr
        )
        return

    # base-2 score scale per row: q scale x cache scale x softmax scale
    sm_scale_log2 = sm_scale * LOG2E
    k_scale = gl.load(ks_ptr)
    if Q_BF16:
        # quantized before the ring is live, so its LDS scratch aliases the ring
        q, q_s = pgm.q_bf16()
        c = q_s * (k_scale * sm_scale_log2)
    else:
        q = None  # loaded by the loop
        c = gl.zeros([cfg.BLOCK_M], gl.float32, cfg.ROW_L) + (
            gl.load(q_scale_ptr) * k_scale * sm_scale_log2
        )

    if TWO_LOOPS:
        if not Q_BF16:
            # once, for both loops
            q = pgm.q_fp8()
        acc, m_i, l_i, v_scale = pgm.loop_two_lists(
            q, kv_ptr, x_kv_ptr, indptr_ptr, idx_ptr, x_indptr_ptr, x_idx_ptr,
            main_row0, main_stride, extra_row0, extra_stride, c, k_scale, x_ks_ptr,
            MAIN_BLOCK_LOG2, EXTRA_BLOCK_LOG2,
        )  # fmt: skip
    else:
        start, end, ix = one_loop_list(
            indptr_ptr, idx_ptr, x_indptr_ptr, x_idx_ptr, pgm.row, main_row0,
            main_stride, extra_row0, extra_stride, PAGED, MAIN_BLOCK_LOG2,
            EXTRA_BLOCK_LOG2,
        )  # fmt: skip
        t0, t1 = pgm.tiles(gl.cdiv(end - start, BN))
        acc, m_i, l_i = pgm.loop(q, kv_ptr, ix, start, end - start, t0, t1, c, None)
        v_scale = k_scale
    if not cfg.WHOLE_ROWS:
        l_i = gl.sum(l_i, axis=1)  # the deferred row sum

    if SPLIT:
        pgm.store_partial(acc, m_i, l_i, v_scale, part_m_ptr, part_l_ptr, part_acc_ptr)
    else:
        pgm.finish(
            acc, m_i, l_i, v_scale, sink_ptr, lse_ptr, out_ptr, out_stride0, out_stride1,
            (pos_ptr, cos_sin_ptr, cs_stride, out_scale_ptr, os_stride0),
        )  # fmt: skip
