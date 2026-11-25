# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Decode rewrite (<= 16 rows) of the router GEMM + bf16 shared expert + fold-1 route tiles
# (`_gemm_a16w16_router_shared_expert_topk_kernel`): same outputs, bit for bit, in about 2 us less a
# launch on gfx1250. The old kernel is latency bound, not bandwidth bound: a tile's time is its CU's
# load time (per-CU ingress, and the number of TDM ops, not the HBM floor) plus the in-kernel handoffs.
# This one changes where the time goes, not what is computed:
#   * every phase-1 unit is 8 weight rows x full K (zero rows past 8 are not fetched: TDM bounds), so 161
#     to 256 CUs pull the 5.3 MB of router + gate/up instead of 81, and every K tile of a unit is in
#     flight at once, as few wide TDM loads ([*, 512]);
#   * the K chain of every unit is the a16w16 one (one fp32 accumulator, K in order, 256-wide WMMA
#     steps), so the logits, h and the shared output are bitwise those of the separate GEMMs;
#   * router -> route handoff: the router units store their logits (bf16) into a sentinel-filled
#     scratch and the route tiles poll that row directly (no store wait, no counter, one round trip);
#   * top-8 as one tree reduction whose combine merges two sorted 8-lists (bitonic), not 8 dependent
#     max reductions;
#   * token masks packed two experts a word (rows <= 16): the mask poll is one 16-byte load a lane;
#   * the route tiles' count-out (the last one restores the shared state) is issued after their last
#     load and only consumed at the end;
#   * the down tiles stage their Wd slice (16 x 512) in LDS first thing, then poll for h.
# Roles by program id (one warp a program, every program resident):
#   [0, NQ)                 route tile q (RPT routes of row q // (8 // RPT)); programs < N_R also run
#                           router unit `pid` first
#   [PQ0, PQ0 + N_GU)       gate/up unit (8 interleaved rows -> 4 h columns), h stored, wave arrive
#   [P - N_D, P)            down tile (16 output columns), may share a program with a gate/up unit
# Workspaces: ws (shared with the original kernel: [0] timeouts, M-block counters), fws (this kernel's:
# [1] unused, [3] route tiles out, [256:384] packed token masks), lsc ([16, LSC_STRIDE] bf16 logits
# scratch, all 0xFFFF between launches). Every launch leaves them as it found them.

import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from aiter.ops.triton._gluon_kernels.gfx1250.gemm.fused.fused_gemm_a16w16_router_shared_expert import (
    WS_HEADER,
    WS_MAX_MB,
    _store_tile,
    _wave_arrive,
)
from aiter.ops.triton._gluon_kernels.gfx1250.gemm.fused.fused_gemm_a16w16_router_shared_expert_topk import (
    EXPERTS,
    FEAT,
    SPIN_MAX,
    TMOFF,
    TOPK,
    _bcnt,
    _exp_f32,
    _pick,
)


@gluon.jit
def _vzero(x):
    """0 in every lane, opaque and divergent (a VGPR inline asm): an atomic on ptr + _vzero(..) is not turned into one
    lane's atomic + readfirstlane, so its return is waited for only where it is used."""
    return gl.inline_asm_elementwise("v_mov_b32 $0, 0", "=v,v", [x], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _vkeep(x):
    """x through an opaque move: the uses of x below it (and the wait for x) stay below it."""
    return gl.inline_asm_elementwise("v_mov_b32 $0, $1", "=v,v", [x], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _xcd_pid(pid, P: gl.constexpr, XCD_RANKS: gl.constexpr):
    """The program's logical id under XCD_RANKS (4 bits a XCD: its rank in dispatch-start order): the grid has
    P8 = 8 * ceil(P / 8) programs, program pid runs on XCD pid % 8, and logical ids fill the XCDs by rank, so the
    low ids (router units, route tiles: the critical chain) start first. -1: a padding program."""
    P8: gl.constexpr = (P + 7) // 8 * 8
    lid = ((XCD_RANKS >> ((pid % 8) * 4)) & 0xF) * (P8 // 8) + pid // 8
    return gl.where(lid < P, lid, -1)


@gluon.jit
def _split_poll(cnt_ptr, target, ws_ptr, CNT_SPLIT: gl.constexpr, NUM_WARPS: gl.constexpr):
    """Wait until the CNT_SPLIT arrival counters (128 B apart) sum to target: one volatile load of them a poll."""
    L: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    lane = gl.arange(0, 32, layout=L)
    ptrs = cnt_ptr + gl.where(lane < CNT_SPLIT, lane, 0) * 32
    c = gl.sum(gl.where(lane < CNT_SPLIT, gl.load(ptrs, volatile=True), 0), axis=0)
    n = 0
    while (c < target) & (n < SPIN_MAX):
        c = gl.sum(gl.where(lane < CNT_SPLIT, gl.load(ptrs, volatile=True), 0), axis=0)
        n += 1
    if n >= SPIN_MAX:
        gl.atomic_add(ws_ptr, 1, sem="relaxed", scope="gpu")


@gluon.jit
def _ce(x, y):
    return gl.maximum(x, y), gl.minimum(x, y)


@gluon.jit
def _merge8(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7):
    """Top 8 (descending) of two descending 8-lists: bitonic split, then a bitonic merge."""
    c0 = gl.maximum(a0, b7)
    c1 = gl.maximum(a1, b6)
    c2 = gl.maximum(a2, b5)
    c3 = gl.maximum(a3, b4)
    c4 = gl.maximum(a4, b3)
    c5 = gl.maximum(a5, b2)
    c6 = gl.maximum(a6, b1)
    c7 = gl.maximum(a7, b0)
    c0, c4 = _ce(c0, c4)
    c1, c5 = _ce(c1, c5)
    c2, c6 = _ce(c2, c6)
    c3, c7 = _ce(c3, c7)
    c0, c2 = _ce(c0, c2)
    c1, c3 = _ce(c1, c3)
    c4, c6 = _ce(c4, c6)
    c5, c7 = _ce(c5, c7)
    c0, c1 = _ce(c0, c1)
    c2, c3 = _ce(c2, c3)
    c4, c5 = _ce(c4, c5)
    c6, c7 = _ce(c6, c7)
    return c0, c1, c2, c3, c4, c5, c6, c7


@gluon.jit
def _top8(keys, wk):
    """The 8 largest (distinct) keys of a 1-D tensor, descending, as an [8] tensor in wk's layout."""
    z = keys * 0
    r = gl.reduce((keys, z, z, z, z, z, z, z), 0, _merge8)
    w = gl.where(wk == 0, r[0], 0).to(gl.uint32)
    w = gl.where(wk == 1, r[1], w)
    w = gl.where(wk == 2, r[2], w)
    w = gl.where(wk == 3, r[3], w)
    w = gl.where(wk == 4, r[4], w)
    w = gl.where(wk == 5, r[5], w)
    w = gl.where(wk == 6, r[6], w)
    w = gl.where(wk == 7, r[7], w)
    return w


@gluon.jit
def _unit_acc(x_ptr, w_ptr, M, N, K: gl.constexpr, stride_xm, nb,
              BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BKT: gl.constexpr, BKC: gl.constexpr,
              SHARED_A: gl.constexpr, SHARED_B: gl.constexpr, WMMA: gl.constexpr, OP_A: gl.constexpr,
              OP_B: gl.constexpr, ROWS: gl.constexpr):
    """fp32 [BLOCK_M, BLOCK_N] = x @ w[nb*ROWS : nb*ROWS+ROWS]^T (columns past ROWS are zero): all of K in
    flight as [*, BKT] TDM loads, WMMA on BKC-wide slices in K order (the a16w16 K chain)."""
    NKT: gl.constexpr = K // BKT
    NC: gl.constexpr = BKT // BKC
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=x_ptr, shape=(M, K), strides=(stride_xm, 1), block_shape=(BLOCK_M, BKT), layout=SHARED_A)
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=w_ptr + nb * ROWS * K, shape=(gl.minimum(N - nb * ROWS, ROWS), K), strides=(K, 1),
        block_shape=(BLOCK_N, BKT), layout=SHARED_B)
    a_buf = gl.allocate_shared_memory(x_ptr.type.element_ty, shape=[NKT, BLOCK_M, BKT], layout=SHARED_A)
    b_buf = gl.allocate_shared_memory(w_ptr.type.element_ty, shape=[NKT, BLOCK_N, BKT], layout=SHARED_B)
    for k in gl.static_range(NKT):
        gl.amd.gfx1250.tdm.async_load(b_desc, [0, k * BKT], b_buf.index(k))
        gl.amd.gfx1250.tdm.async_load(a_desc, [0, k * BKT], a_buf.index(k))
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
    for k in gl.static_range(NKT):
        gl.amd.gfx1250.tdm.async_wait((NKT - 1 - k) * 2)
        for c in gl.static_range(NC):
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_buf.index(k).slice(c * BKC, BKC, 1), OP_A)
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buf.index(k).slice(c * BKC, BKC, 1).permute([1, 0]), OP_B)
            acc = gl.amd.gfx1250.wmma(cur_a, cur_b, acc)
    return acc


@gluon.jit
def _unit_issue(x_ptr, w_ptr, M, N, K: gl.constexpr, stride_xm, nb,
                BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BKT: gl.constexpr,
                SHARED_A: gl.constexpr, SHARED_B: gl.constexpr, ROWS: gl.constexpr):
    """`_unit_acc`'s TDM loads (all of K in flight); returns the LDS buffers for `_unit_mma`."""
    NKT: gl.constexpr = K // BKT
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=x_ptr, shape=(M, K), strides=(stride_xm, 1), block_shape=(BLOCK_M, BKT), layout=SHARED_A)
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=w_ptr + nb * ROWS * K, shape=(gl.minimum(N - nb * ROWS, ROWS), K), strides=(K, 1),
        block_shape=(BLOCK_N, BKT), layout=SHARED_B)
    a_buf = gl.allocate_shared_memory(x_ptr.type.element_ty, shape=[NKT, BLOCK_M, BKT], layout=SHARED_A)
    b_buf = gl.allocate_shared_memory(w_ptr.type.element_ty, shape=[NKT, BLOCK_N, BKT], layout=SHARED_B)
    for k in gl.static_range(NKT):
        gl.amd.gfx1250.tdm.async_load(b_desc, [0, k * BKT], b_buf.index(k))
        gl.amd.gfx1250.tdm.async_load(a_desc, [0, k * BKT], a_buf.index(k))
    return a_buf, b_buf


@gluon.jit
def _unit_mma(a_buf, b_buf, K: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BKT: gl.constexpr,
              BKC: gl.constexpr, WMMA: gl.constexpr, OP_A: gl.constexpr, OP_B: gl.constexpr):
    """`_unit_acc`'s K chain on the buffers `_unit_issue` filled (the same WMMA sequence: bitwise)."""
    NKT: gl.constexpr = K // BKT
    NC: gl.constexpr = BKT // BKC
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
    for k in gl.static_range(NKT):
        gl.amd.gfx1250.tdm.async_wait((NKT - 1 - k) * 2)
        for c in gl.static_range(NC):
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_buf.index(k).slice(c * BKC, BKC, 1), OP_A)
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buf.index(k).slice(c * BKC, BKC, 1).permute([1, 0]), OP_B)
            acc = gl.amd.gfx1250.wmma(cur_a, cur_b, acc)
    return acc


@gluon.jit
def _rstd_rows(ssq_ptr, stride_sm, M, eps, K: gl.constexpr, BLOCK_M: gl.constexpr, NSSQ: gl.constexpr,
               NUM_WARPS: gl.constexpr, OUT: gl.constexpr):
    """ATOM_FOLD_POSTNORM: the post-attention RMSNorm's 1 / rms of rows [0, BLOCK_M) from the producer GEMM's
    per-tile sums of squares ssq[row, 0:NSSQ], in SliceLayout(1, OUT)."""
    L: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NUM_WARPS, 1], [1, 0])
    r = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, L))
    c = gl.arange(0, NSSQ, layout=gl.SliceLayout(0, L))
    v = gl.load(ssq_ptr + r[:, None] * stride_sm + c[None, :], mask=(r < M)[:, None], other=0.0)
    rs = gl.rsqrt(gl.sum(v, axis=1) / K + eps)
    return gl.convert_layout(rs, gl.SliceLayout(1, OUT))


@gluon.jit
def _route_tile_v2(
    u, x_ptr, logits_ptr, nvr_ptr, tw_ptr, tid_ptr, rows_ptr, mm_ptr, psum_ptr, pay_ptr, scale_ptr, fws_ptr,
    ws_ptr, lsc_ptr, M: gl.constexpr, stride_xm, LSC_STRIDE: gl.constexpr, ALIGN_M: gl.constexpr,
    WMMA_REP: gl.constexpr, HAS_NVR: gl.constexpr, NUM_WARPS: gl.constexpr, RPT: gl.constexpr,
    ssq_ptr=None, nw_ptr=None, stride_sm=0, eps=0.0, NSSQ: gl.constexpr = 1, PRENORM: gl.constexpr = 0,
    LATE_WAIT: gl.constexpr = 0,
):
    """fold 1's route tile (topkGatingSoftmax<renorm> top-8, routing tables, MXFP4 activation quant to the
    grouped rows), bit for bit, with the handoffs above."""
    G: gl.constexpr = TOPK // RPT  # route tiles a row
    NT: gl.constexpr = M * G
    CH: gl.constexpr = (EXPERTS + NT - 1) // NT  # masked_m / psum entries a tile writes
    gl.static_assert(ALIGN_M >= M)
    gl.static_assert(M <= 16)
    t = u // G
    g = u - t * G

    # ---- 1. MXFP4 quant of x[t] (needs nothing: runs under the GEMMs) ---------------------------------
    QL: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [NUM_WARPS, 1], [1, 0])
    SL: gl.constexpr = gl.BlockedLayout([1, 1], [8, 4], [NUM_WARPS, 1], [1, 0])
    qr = gl.arange(0, FEAT // 32, layout=gl.SliceLayout(1, QL))
    qc = gl.arange(0, 32, layout=gl.SliceLayout(0, QL))
    xv = gl.amd.gfx1250.buffer_load(x_ptr + t.to(gl.int64) * stride_xm, qr[:, None] * 32 + qc[None, :])
    if PRENORM:
        # x is the un-normalized residual sum: apply the post-attention RMSNorm (as _gluon_fused_rms_kernel)
        PNL: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
        sci = gl.arange(0, NSSQ, layout=PNL)
        rs = gl.rsqrt(gl.sum(gl.load(ssq_ptr + t.to(gl.int64) * stride_sm + sci), axis=0) / FEAT + eps)
        wv = gl.amd.gfx1250.buffer_load(nw_ptr, qr[:, None] * 32 + qc[None, :]).to(gl.float32) + 1.0
        xv = (xv.to(gl.float32) * rs * wv).to(xv.dtype)
    amax = gl.max(gl.abs(xv.to(gl.float32)), axis=1)
    work = (amax * 0.16666667163372040).to(gl.int32, bitcast=True)  # * 1/6 (0x3E2AAAAB)
    bexp = (work >> 23) & 0xFF
    e8 = gl.where((work & 0x7FFFFF) != 0, bexp + 1, bexp)  # RoundUp e8m0
    e8 = gl.minimum(gl.maximum(e8, 0), 255)
    e8s = gl.convert_layout(e8, gl.SliceLayout(1, SL))
    sc4 = gl.full([FEAT // 32, 4], 0, gl.int32, SL) + e8s[:, None]
    payload = gl.amd.gfx1250.scaled_downcast(xv, sc4.to(gl.uint8), "e2m1", axis=1)  # [64, 16] u8
    VL: gl.constexpr = gl.BlockedLayout([1, 16], [32, 1], [NUM_WARPS, 1], [1, 0])
    payload = gl.convert_layout(payload, VL).to(gl.int32)
    VL3: gl.constexpr = gl.BlockedLayout([1, 4, 4], [32, 1, 1], [NUM_WARPS, 1, 1], [2, 1, 0])
    p3 = gl.convert_layout(gl.reshape(payload, [FEAT // 32, 4, 4]), VL3)
    bsh = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, VL3)))
    payload = gl.sum(p3 << (bsh[None, None, :] * 8), axis=2)  # [64, 4] dwords
    PW: gl.constexpr = payload.type.layout
    pr = gl.arange(0, FEAT // 32, layout=gl.SliceLayout(1, PW))
    pc = gl.arange(0, 4, layout=gl.SliceLayout(0, PW))
    poffs = pr[:, None] * 4 + pc[None, :]
    DL: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [NUM_WARPS, 1], [1, 0])
    e8d = gl.convert_layout(gl.reshape(e8, [FEAT // 128, 4]), DL)
    sh = gl.arange(0, 4, layout=gl.SliceLayout(0, DL))
    sdw = gl.sum(e8d << (sh[None, :] * 8), axis=1)  # [16] e8m0 dwords
    sdi = gl.arange(0, FEAT // 128, layout=gl.SliceLayout(1, DL))

    if HAS_NVR:
        nvr = gl.load(nvr_ptr)
        nvr = gl.minimum(gl.where(nvr < 0, 0, nvr), M)
    else:
        nvr = M

    # ---- 2. row t's logits: the scratch row is the flag (no 0xFFFF sentinel left), then top-8 ---------
    KL: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])
    cols = gl.arange(0, EXPERTS, layout=KL)
    lrow = lsc_ptr + t * LSC_STRIDE + cols
    lg = gl.load(lrow, cache_modifier=".cg", volatile=True)
    n_r = 0
    while (gl.max((lg.to(gl.int16, bitcast=True) == -1).to(gl.int32), axis=0) > 0) & (n_r < SPIN_MAX):
        lg = gl.load(lrow, cache_modifier=".cg", volatile=True)
        n_r += 1
    if n_r >= SPIN_MAX:
        gl.atomic_add(ws_ptr, 1, sem="relaxed", scope="gpu")
    b = (lg.to(gl.float32) + 0.0).to(gl.uint32, bitcast=True)  # -0 -> +0
    flip = (b.to(gl.int32, bitcast=True) >> 31).to(gl.uint32, bitcast=True) | 0x80000000
    keys = ((b ^ flip) & 0xFFFF0000) | (255 - cols).to(gl.uint32)  # bf16 order bits | 255 - col
    KO: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])  # 8 in-thread
    wk = gl.arange(0, TOPK, layout=KO)
    wins = _top8(keys, wk)
    top_e = 255 - (wins & 0xFF).to(gl.int32)
    is_pad = t >= nvr
    out_e = gl.where(is_pad, -1, top_e)

    # ---- 3. publish the tile's routes: packed token masks, expert e -> word e >> 1, bit 16 (e & 1) + t -
    LS: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])  # 8 values on lanes 0..7
    wl = gl.arange(0, TOPK, layout=LS)
    e_l = gl.convert_layout(out_e, LS)
    mine = (wl >= g * RPT) & (wl < g * RPT + RPT)
    e0 = gl.where(e_l >= 0, e_l, 0)
    gl.atomic_or(fws_ptr + TMOFF + (e0 >> 1), gl.full([TOPK], 1, gl.int32, LS) << (t + 16 * (e0 & 1)),
                 mask=(e_l >= 0) & mine, sem="relaxed", scope="gpu")
    hi = wins & 0xFFFF0000
    vb = gl.where(hi >= 0x80000000, hi ^ 0x80000000, (hi ^ 0xFFFFFFFF) & 0xFFFF0000)
    top_v = vb.to(gl.float32, bitcast=True)
    v0 = gl.max(top_v, axis=0)
    numer = _exp_f32(top_v - v0)
    acc = _pick(numer, wk, 0)
    for k in gl.static_range(1, TOPK):
        acc = acc + _pick(numer, wk, k)
    nz = acc != 0.0
    scl = gl.where(nz, gl.fdiv(1.0, acc, ieee_rounding=True), 1.0)
    w = gl.where(nz, numer * scl, numer)
    out_w = gl.where(is_pad, 0.0, w)
    gl.amd.gfx1250.buffer_store(gl.convert_layout(out_w, LS), tw_ptr + t * TOPK, wl, mask=mine)
    gl.amd.gfx1250.buffer_store(e_l, tid_ptr + t * TOPK, wl, mask=mine)

    # ---- 4. every valid route landed: nvr * 8 bits in the 128 mask words (one 16-byte load a lane) -----
    EL: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])  # expert e on lane e // 8
    ex = gl.arange(0, EXPERTS, layout=EL)
    EL2: gl.constexpr = gl.BlockedLayout([4], [32], [NUM_WARPS], [0])
    wj = gl.arange(0, EXPERTS // 2, layout=EL2)
    tw2 = gl.load(fws_ptr + TMOFF + wj, cache_modifier=".cg", volatile=True)
    n_it = 0
    while (gl.sum(_bcnt(tw2), axis=0) != nvr * TOPK) & (n_it < SPIN_MAX):
        tw2 = gl.load(fws_ptr + TMOFF + wj, cache_modifier=".cg", volatile=True)
        n_it += 1
    if n_it >= SPIN_MAX:
        gl.atomic_add(ws_ptr, 1, sem="relaxed", scope="gpu")
    tm = gl.convert_layout(gl.reshape(gl.join(tw2 & 0xFFFF, (tw2 >> 16) & 0xFFFF), [EXPERTS]), EL)
    # count this tile out now (after its last read of the shared state); the old value is used at the end
    CL: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NUM_WARPS, 1], [1, 0])
    c_lanes = gl.arange(0, 32, layout=gl.SliceLayout(0, CL))
    c_waves = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(1, CL))
    c_lead = (c_waves[:, None] + c_lanes[None, :]) == 0
    c_zero = c_waves[:, None] * 0 + c_lanes[None, :] * 0
    if LATE_WAIT:
        c_used = gl.atomic_add(fws_ptr + 3 + _vzero(c_zero), c_zero + 1, mask=c_lead, sem="relaxed", scope="gpu")
    else:
        c_used = gl.atomic_add(fws_ptr + 3 + c_zero, c_zero + 1, mask=c_lead, sem="relaxed", scope="gpu")

    # ---- 5. routing ---------------------------------------------------------------------------------
    cnt = _bcnt(tm)
    nzp = (tm != 0).to(gl.int32)
    start = ALIGN_M * (gl.cumsum(nzp, axis=0) - nzp)
    cm = (ex >= u * CH) & (ex < u * CH + CH)
    gl.amd.gfx1250.buffer_store(cnt, mm_ptr, ex, mask=cm)
    gl.amd.gfx1250.buffer_store(start + cnt, psum_ptr, ex, mask=cm)
    eg = gl.where(e_l >= 0, e_l, 0)
    st_k = gl.gather(start, eg, 0)
    tm_k = gl.gather(tm, eg, 0)
    rows = gl.where(e_l >= 0, st_k + _bcnt(tm_k & ((1 << t) - 1)), -1)
    gl.amd.gfx1250.buffer_store(rows, rows_ptr + t * TOPK, wl, mask=mine)
    rows_r = gl.convert_layout(rows, KO)

    # ---- 6. payload + e8m0 to the grouped rows ------------------------------------------------------
    if t < nvr:
        for k in gl.static_range(RPT):
            r = _pick(rows_r, wk, g * RPT + k)
            gl.amd.gfx1250.buffer_store(payload, pay_ptr, r * (FEAT // 8) + poffs)
            tile = r // (16 * WMMA_REP)
            rit = r - tile * (16 * WMMA_REP)
            base = tile * (16 * WMMA_REP) * 16 + (rit // 16) * 16 + (rit % 16)
            gl.amd.gfx1250.buffer_store(sdw, scale_ptr, base + sdi * (WMMA_REP * 16))

    # ---- 7. the last tile out restores the shared state for the next launch ---------------------------
    if LATE_WAIT:
        c_used = _vkeep(c_used)
    c_last = c_lead & (c_used == NT - 1)
    gl.atomic_xchg(fws_ptr + 3 + c_zero, c_zero, mask=c_last, sem="relaxed", scope="gpu")
    if gl.sum(c_last.to(gl.int32), axis=None) > 0:
        gl.amd.gfx1250.buffer_store(gl.full([EXPERTS // 2], 0, gl.int32, EL2), fws_ptr + TMOFF, wj, cache=".cg")
        RL: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [NUM_WARPS, 1], [1, 0])
        rr0 = gl.arange(0, 16, layout=gl.SliceLayout(1, RL))
        rc0 = gl.arange(0, EXPERTS, layout=gl.SliceLayout(0, RL))
        gl.amd.gfx1250.buffer_store(gl.full([16, EXPERTS], -1, gl.int16, RL).to(gl.bfloat16, bitcast=True),
                                    lsc_ptr, rr0[:, None] * LSC_STRIDE + rc0[None, :], mask=rr0[:, None] < M,
                                    cache=".cg")


@gluon.jit
def _gemm_a16w16_router_shared_expert_topk_v2_kernel(
    x_ptr, wr_ptr, wgu_ptr, wd_ptr, logits_ptr, h_ptr, out_ptr, ws_ptr, fws_ptr, lsc_ptr, nvr_ptr, tw_ptr,
    tid_ptr, rows_ptr, mm_ptr, psum_ptr, pay_ptr, scale_ptr,
    M: gl.constexpr, stride_xm, stride_lm, stride_hm, stride_om,
    K: gl.constexpr, NR: gl.constexpr, NR_ROWS: gl.constexpr, NGU: gl.constexpr, ND: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N1: gl.constexpr, BLOCK_K1: gl.constexpr, BKT1: gl.constexpr,
    ROWS_R: gl.constexpr, ROWS_G: gl.constexpr, BLOCK_N2: gl.constexpr, BLOCK_K2: gl.constexpr,
    BKT2: gl.constexpr, P: gl.constexpr, PQ0: gl.constexpr, NUM_WARPS: gl.constexpr,
    SHARED_A1: gl.constexpr, SHARED_B1: gl.constexpr, SHARED_A2: gl.constexpr, SHARED_B2: gl.constexpr,
    WMMA1: gl.constexpr, OP_A1: gl.constexpr, OP_B1: gl.constexpr, WMMA2: gl.constexpr, OP_A2: gl.constexpr,
    OP_B2: gl.constexpr, STORE1: gl.constexpr, TIMEOUT_NS: gl.constexpr, H_CACHE: gl.constexpr,
    LSC_STRIDE: gl.constexpr, ALIGN_M: gl.constexpr, WMMA_REP: gl.constexpr, HAS_NVR: gl.constexpr,
    RPT: gl.constexpr,
    ssq_ptr=None, nw_ptr=None, stride_sm=0, eps=0.0, NSSQ: gl.constexpr = 1, PRENORM: gl.constexpr = 0,
    LATE_WAIT: gl.constexpr = 0, XCD_RANKS: gl.constexpr = 0, CNT_SPLIT: gl.constexpr = 1,
    EARLY_RSTD: gl.constexpr = 0,
):
    # EARLY_RSTD (AITER_ROUTER_SHARED_V2_EARLY_RSTD, with PRENORM): the post-norm rstd's loads go out after the units'
    # weight loads, not after their GEMM. Bitwise.
    # CNT_SPLIT (AITER_ROUTER_SHARED_V2_CNT_SPLIT): the gate/up arrivals go to CNT_SPLIT counters 128 B apart (not
    # all to one L2 line); the down tiles poll them with one load. Bitwise.
    # LATE_WAIT (AITER_ROUTER_SHARED_V2_LATE_WAIT): the count-out atomics' returns are waited for where they are used
    # (the route tiles' at the end, the down tiles count out right after their poll), not right after the atomic.
    # XCD_RANKS (AITER_ROUTER_SHARED_V2_XCD_ORDER): logical program ids by XCD start order (_xcd_pid). Both bitwise.
    # PRENORM (ATOM_FOLD_POSTNORM): x is the residual sum h + r, not its post-attention RMSNorm; ssq holds its
    # rows' per-tile sums of squares (the producer GEMM's epilogue) and nw the norm weight. The router and gate/up
    # weights come pre-scaled by (1 + nw) (the caller's), so x_norm @ W^T = rstd * (x @ W'^T): the units scale
    # their accumulators by rstd; the route tiles normalize their row before the MXFP4 quant.
    KD: gl.constexpr = NGU // 2
    N_GU: gl.constexpr = NGU // ROWS_G
    N_R: gl.constexpr = (NR + ROWS_R - 1) // ROWS_R
    N_D: gl.constexpr = ND // BLOCK_N2
    NQ: gl.constexpr = M * (TOPK // RPT)
    HALF: gl.constexpr = BLOCK_N1 // 2
    HG: gl.constexpr = ROWS_G // 2  # h columns of a gate/up unit
    NKT2: gl.constexpr = KD // BKT2
    NC2: gl.constexpr = BKT2 // BLOCK_K2
    pid = gl.program_id(axis=0)
    if XCD_RANKS != 0:
        pid = _xcd_pid(pid, P, XCD_RANKS)
        if pid < 0:
            return
    cnt_ptr = ws_ptr + WS_HEADER
    use_ptr = cnt_ptr + WS_MAX_MB
    zi = pid * 0

    # down tile: its Wd slice [BLOCK_N2, KD] goes to LDS before anything else
    d = P - 1 - pid
    is_down = d < N_D
    dd = gl.minimum(d, N_D - 1)
    b2_buf = gl.allocate_shared_memory(wd_ptr.type.element_ty, shape=[NKT2, BLOCK_N2, BKT2], layout=SHARED_B2)
    b2_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=wd_ptr + dd * BLOCK_N2 * KD, shape=(ND - dd * BLOCK_N2, KD), strides=(KD, 1),
        block_shape=(BLOCK_N2, BKT2), layout=SHARED_B2)
    if is_down:
        for k in gl.static_range(NKT2):
            gl.amd.gfx1250.tdm.async_load(b2_desc, [0, k * BKT2], b2_buf.index(k))

    # router unit: logits of ROWS_R router rows -> output and the flag scratch
    if pid < N_R:
        if EARLY_RSTD and PRENORM:
            # the rstd's loads under the weight loads (the same values and multiply)
            a_b, b_b = _unit_issue(x_ptr, wr_ptr, M, NR_ROWS, K, stride_xm, pid, BLOCK_M, BLOCK_N1, BKT1,
                                   SHARED_A1, SHARED_B1, ROWS_R)
            rs_r = _rstd_rows(ssq_ptr, stride_sm, M, eps, K, BLOCK_M, NSSQ, NUM_WARPS, WMMA1)
            acc = _unit_mma(a_b, b_b, K, BLOCK_M, BLOCK_N1, BKT1, BLOCK_K1, WMMA1, OP_A1, OP_B1)
            acc = acc * rs_r[:, None]
        else:
            acc = _unit_acc(x_ptr, wr_ptr, M, NR_ROWS, K, stride_xm, pid, BLOCK_M, BLOCK_N1, BKT1, BLOCK_K1,
                            SHARED_A1, SHARED_B1, WMMA1, OP_A1, OP_B1, ROWS_R)
            if PRENORM:
                acc = acc * _rstd_rows(ssq_ptr, stride_sm, M, eps, K, BLOCK_M, NSSQ, NUM_WARPS, WMMA1)[:, None]
        _store_tile(acc, lsc_ptr, M, pid * ROWS_R + ROWS_R, LSC_STRIDE, zi, pid * ROWS_R, WMMA1, ".cg")
        _store_tile(acc, logits_ptr, M, gl.minimum(pid * ROWS_R + ROWS_R, NR), stride_lm, zi, pid * ROWS_R,
                    WMMA1, "")
    if pid < NQ:
        _route_tile_v2(pid, x_ptr, logits_ptr, nvr_ptr, tw_ptr, tid_ptr, rows_ptr, mm_ptr, psum_ptr, pay_ptr,
                       scale_ptr, fws_ptr, ws_ptr, lsc_ptr, M, stride_xm, LSC_STRIDE, ALIGN_M, WMMA_REP, HAS_NVR,
                       NUM_WARPS, RPT, ssq_ptr, nw_ptr, stride_sm, eps, NSSQ, PRENORM, LATE_WAIT)
    # gate/up unit: h[:, g*HG : g*HG+HG] = silu(gate) * up (gate, up rounded to bf16 first, as
    # gemm_a16w16_gated), device-scope stores, then one arrival a wave
    g = pid - PQ0
    if (g >= 0) & (g < N_GU):
        if EARLY_RSTD and PRENORM:
            a_b, b_b = _unit_issue(x_ptr, wgu_ptr, M, NGU, K, stride_xm, g, BLOCK_M, BLOCK_N1, BKT1,
                                   SHARED_A1, SHARED_B1, ROWS_G)
            rs_g = _rstd_rows(ssq_ptr, stride_sm, M, eps, K, BLOCK_M, NSSQ, NUM_WARPS, WMMA1)
            acc = _unit_mma(a_b, b_b, K, BLOCK_M, BLOCK_N1, BKT1, BLOCK_K1, WMMA1, OP_A1, OP_B1)
            acc = acc * rs_g[:, None]
        else:
            acc = _unit_acc(x_ptr, wgu_ptr, M, NGU, K, stride_xm, g, BLOCK_M, BLOCK_N1, BKT1, BLOCK_K1,
                            SHARED_A1, SHARED_B1, WMMA1, OP_A1, OP_B1, ROWS_G)
            if PRENORM:
                acc = acc * _rstd_rows(ssq_ptr, stride_sm, M, eps, K, BLOCK_M, NSSQ, NUM_WARPS, WMMA1)[:, None]
        gate, up = gl.split(gl.reshape(acc, [BLOCK_M, HALF, 2]))
        gate = gate.to(h_ptr.type.element_ty).to(gl.float32)
        up = up.to(h_ptr.type.element_ty).to(gl.float32)
        hv = gate / (1.0 + gl.exp(-gate)) * up
        hv = gl.convert_layout(hv, STORE1)
        _store_tile(hv, h_ptr, M, g * HG + HG, stride_hm, zi, g * HG, STORE1, H_CACHE)
        if CNT_SPLIT > 1:
            _wave_arrive(cnt_ptr + (g % CNT_SPLIT) * 32, NUM_WARPS)
        else:
            _wave_arrive(cnt_ptr, NUM_WARPS)
    # down tile: wait for every gate/up unit, h -> LDS, WMMA over K = I in order, store
    if is_down:
        if CNT_SPLIT > 1:
            _split_poll(cnt_ptr, N_GU * NUM_WARPS, ws_ptr, CNT_SPLIT, NUM_WARPS)
        else:
            ok = gl.atomic_poll(cnt_ptr, N_GU * NUM_WARPS, sem="relaxed", scope="gpu", timeout_ns=TIMEOUT_NS)
            if not ok:
                gl.atomic_add(ws_ptr, 1, sem="relaxed", scope="gpu")
        L: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NUM_WARPS, 1], [1, 0])
        lanes = gl.arange(0, 32, layout=gl.SliceLayout(0, L))
        waves = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(1, L))
        lead = (waves[:, None] + lanes[None, :]) == 0
        zero = waves[:, None] * 0 + lanes[None, :] * 0
        if LATE_WAIT:
            # the poll above was this tile's last read of the counter: count it out now, use the value at the end
            used = gl.atomic_add(use_ptr + _vzero(zero), zero + 1, mask=lead, sem="relaxed", scope="gpu")
        a2_buf = gl.allocate_shared_memory(h_ptr.type.element_ty, shape=[NKT2, BLOCK_M, BKT2], layout=SHARED_A2)
        a2_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=h_ptr, shape=(M, KD), strides=(stride_hm, 1), block_shape=(BLOCK_M, BKT2), layout=SHARED_A2)
        for k in gl.static_range(NKT2):
            gl.amd.gfx1250.tdm.async_load(a2_desc, [0, k * BKT2], a2_buf.index(k), cache_modifier=H_CACHE)
        acc2 = gl.zeros((BLOCK_M, BLOCK_N2), dtype=gl.float32, layout=WMMA2)
        for k in gl.static_range(NKT2):
            gl.amd.gfx1250.tdm.async_wait(NKT2 - 1 - k)
            for c in gl.static_range(NC2):
                cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    a2_buf.index(k).slice(c * BLOCK_K2, BLOCK_K2, 1), OP_A2)
                cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    b2_buf.index(k).slice(c * BLOCK_K2, BLOCK_K2, 1).permute([1, 0]), OP_B2)
                acc2 = gl.amd.gfx1250.wmma(cur_a, cur_b, acc2)
        _store_tile(acc2, out_ptr, M, ND, stride_om, zi, d * BLOCK_N2, WMMA2, "")
        # count this consumer; the last one clears the counters for the next launch
        if LATE_WAIT:
            used = _vkeep(used)
        else:
            used = gl.atomic_add(use_ptr + zero, zero + 1, mask=lead, sem="relaxed", scope="gpu")
        last = lead & (used == N_D - 1)
        gl.atomic_xchg(cnt_ptr + zero, zero, mask=last, sem="relaxed", scope="gpu")
        for j in gl.static_range(1, CNT_SPLIT):
            gl.atomic_xchg(cnt_ptr + j * 32 + zero, zero, mask=last, sem="relaxed", scope="gpu")
        gl.atomic_xchg(use_ptr + zero, zero, mask=last, sem="relaxed", scope="gpu")


# ---------------------------------------------------------------------------------------------------------
# Wide decode variant (17 .. 48 rows: the verify step of 5 .. 12 requests x 4 tokens). Same handoffs and K chains
# as the kernel above (so every output is bitwise the same as the native pair); what changes with the row count
# (MB = ceil(M / 16) M blocks of 16 rows):
#   * router units: one a (8 router rows, M block): N_R * MB of them, each loads its block of x only, so the logits
#     of every block land at once (route tiles share these programs, as above);
#   * gate/up units: 16 weight rows (no half-empty WMMA) on half of the M blocks each (2 * NGU / 16 programs): B
#     once, every A block of the half in flight, then the blocks' epilogues in order;
#   * down tiles: every M block of h in flight, then a K chain a block;
#   * token masks: WPE = ceil(M / 32) planes of 256 words (row t, expert e -> word (t >> 5) * 256 + e, bit t & 31):
#     fws[256:256 + 256 * WPE], polled as one or two 1-D loads a tile;
#   * routing as moe_topk_route_quant for any power-of-two align_m: start[e] = exclusive scan of
#     align_up(count, ALIGN_M), the route's grouped row = start[e] + (rows of expert e below t), token-major;
#   * the last tile out re-arms the logits scratch 16 rows at a time (runtime loop).
# Four M blocks (49 .. 64 rows) do not fit the LDS (> 320 KB).
# ---------------------------------------------------------------------------------------------------------


@gluon.jit
def _unit_epi(acc, out0_ptr, out1_ptr, M, stride0, stride1, N0, N1, row0, col0, BLOCK_M: gl.constexpr,
              HALF: gl.constexpr, IS_GU: gl.constexpr, WMMA: gl.constexpr, STORE1: gl.constexpr,
              H_CACHE: gl.constexpr):
    if IS_GU:
        gate, up = gl.split(gl.reshape(acc, [BLOCK_M, HALF, 2]))
        gate = gate.to(out0_ptr.type.element_ty).to(gl.float32)
        up = up.to(out0_ptr.type.element_ty).to(gl.float32)
        hv = gate / (1.0 + gl.exp(-gate)) * up
        hv = gl.convert_layout(hv, STORE1)
        _store_tile(hv, out0_ptr, M, N0, stride0, row0, col0, STORE1, H_CACHE)
    else:
        _store_tile(acc, out0_ptr, M, N0, stride0, row0, col0, WMMA, ".cg")
        _store_tile(acc, out1_ptr, M, N1, stride1, row0, col0, WMMA, "")


@gluon.jit
def _unit_mb(x_ptr, w_ptr, M, N, K: gl.constexpr, stride_xm, nb, out0_ptr, out1_ptr, stride0, stride1, N0, N1,
             col0, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BKT: gl.constexpr, BKC: gl.constexpr,
             SHARED_A: gl.constexpr, SHARED_B: gl.constexpr, WMMA: gl.constexpr, OP_A: gl.constexpr,
             OP_B: gl.constexpr, ROWS: gl.constexpr, MB: gl.constexpr, IS_GU: gl.constexpr, STORE1: gl.constexpr,
             H_CACHE: gl.constexpr, m0=0):
    """`_unit_acc` over MB M blocks from row m0 (B once, every A block in flight), then each block's epilogue:
    router (IS_GU False): logits -> out0 (the flag scratch, .cg) and out1 (logits); gate/up: h -> out0."""
    NKT: gl.constexpr = K // BKT
    NC: gl.constexpr = BKT // BKC
    HALF: gl.constexpr = BLOCK_N // 2
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=x_ptr, shape=(M, K), strides=(stride_xm, 1), block_shape=(BLOCK_M, BKT), layout=SHARED_A)
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=w_ptr + nb * ROWS * K, shape=(gl.minimum(N - nb * ROWS, ROWS), K), strides=(K, 1),
        block_shape=(BLOCK_N, BKT), layout=SHARED_B)
    a_buf = gl.allocate_shared_memory(x_ptr.type.element_ty, shape=[MB * NKT, BLOCK_M, BKT], layout=SHARED_A)
    b_buf = gl.allocate_shared_memory(w_ptr.type.element_ty, shape=[NKT, BLOCK_N, BKT], layout=SHARED_B)
    for k in gl.static_range(NKT):
        gl.amd.gfx1250.tdm.async_load(b_desc, [0, k * BKT], b_buf.index(k))
        gl.amd.gfx1250.tdm.async_load(a_desc, [m0, k * BKT], a_buf.index(k))
    for mb in gl.static_range(1, MB):
        for k in gl.static_range(NKT):
            gl.amd.gfx1250.tdm.async_load(a_desc, [m0 + mb * BLOCK_M, k * BKT], a_buf.index(mb * NKT + k))
    zi = nb * 0 + m0
    for mb in gl.static_range(MB):
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
        for k in gl.static_range(NKT):
            if mb == 0:
                gl.amd.gfx1250.tdm.async_wait((NKT - 1 - k) * 2 + (MB - 1) * NKT)
            else:
                gl.amd.gfx1250.tdm.async_wait((MB - 1 - mb) * NKT + (NKT - 1 - k))
            for c in gl.static_range(NC):
                cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    a_buf.index(mb * NKT + k).slice(c * BKC, BKC, 1), OP_A)
                cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    b_buf.index(k).slice(c * BKC, BKC, 1).permute([1, 0]), OP_B)
                acc = gl.amd.gfx1250.wmma(cur_a, cur_b, acc)
        _unit_epi(acc, out0_ptr, out1_ptr, M, stride0, stride1, N0, N1, zi + mb * BLOCK_M, col0, BLOCK_M, HALF,
                  IS_GU, WMMA, STORE1, H_CACHE)


@gluon.jit
def _route_tile_wide(
    u, x_ptr, nvr_ptr, tw_ptr, tid_ptr, rows_ptr, mm_ptr, psum_ptr, pay_ptr, scale_ptr, fws_ptr,
    ws_ptr, lsc_ptr, M: gl.constexpr, stride_xm, LSC_STRIDE: gl.constexpr, ALIGN_M: gl.constexpr,
    WMMA_REP: gl.constexpr, HAS_NVR: gl.constexpr, NUM_WARPS: gl.constexpr, RPT: gl.constexpr, WPE: gl.constexpr,
    LATE_WAIT: gl.constexpr = 0,
):
    """`_route_tile_v2` for 17 .. 64 rows (token mask planes, any power-of-two align_m)."""
    G: gl.constexpr = TOPK // RPT  # route tiles a row
    NT: gl.constexpr = M * G
    CH: gl.constexpr = (EXPERTS + NT - 1) // NT  # masked_m / psum entries a tile writes
    gl.static_assert(M <= 32 * WPE)
    t = u // G
    g = u - t * G

    # ---- 1. MXFP4 quant of x[t] (needs nothing: runs under the GEMMs) ---------------------------------
    QL: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [NUM_WARPS, 1], [1, 0])
    SL: gl.constexpr = gl.BlockedLayout([1, 1], [8, 4], [NUM_WARPS, 1], [1, 0])
    qr = gl.arange(0, FEAT // 32, layout=gl.SliceLayout(1, QL))
    qc = gl.arange(0, 32, layout=gl.SliceLayout(0, QL))
    xv = gl.amd.gfx1250.buffer_load(x_ptr + t.to(gl.int64) * stride_xm, qr[:, None] * 32 + qc[None, :])
    amax = gl.max(gl.abs(xv.to(gl.float32)), axis=1)
    work = (amax * 0.16666667163372040).to(gl.int32, bitcast=True)  # * 1/6 (0x3E2AAAAB)
    bexp = (work >> 23) & 0xFF
    e8 = gl.where((work & 0x7FFFFF) != 0, bexp + 1, bexp)  # RoundUp e8m0
    e8 = gl.minimum(gl.maximum(e8, 0), 255)
    e8s = gl.convert_layout(e8, gl.SliceLayout(1, SL))
    sc4 = gl.full([FEAT // 32, 4], 0, gl.int32, SL) + e8s[:, None]
    payload = gl.amd.gfx1250.scaled_downcast(xv, sc4.to(gl.uint8), "e2m1", axis=1)  # [64, 16] u8
    VL: gl.constexpr = gl.BlockedLayout([1, 16], [32, 1], [NUM_WARPS, 1], [1, 0])
    payload = gl.convert_layout(payload, VL).to(gl.int32)
    VL3: gl.constexpr = gl.BlockedLayout([1, 4, 4], [32, 1, 1], [NUM_WARPS, 1, 1], [2, 1, 0])
    p3 = gl.convert_layout(gl.reshape(payload, [FEAT // 32, 4, 4]), VL3)
    bsh = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, VL3)))
    payload = gl.sum(p3 << (bsh[None, None, :] * 8), axis=2)  # [64, 4] dwords
    PW: gl.constexpr = payload.type.layout
    pr = gl.arange(0, FEAT // 32, layout=gl.SliceLayout(1, PW))
    pc = gl.arange(0, 4, layout=gl.SliceLayout(0, PW))
    poffs = pr[:, None] * 4 + pc[None, :]
    DL: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [NUM_WARPS, 1], [1, 0])
    e8d = gl.convert_layout(gl.reshape(e8, [FEAT // 128, 4]), DL)
    sh = gl.arange(0, 4, layout=gl.SliceLayout(0, DL))
    sdw = gl.sum(e8d << (sh[None, :] * 8), axis=1)  # [16] e8m0 dwords
    sdi = gl.arange(0, FEAT // 128, layout=gl.SliceLayout(1, DL))

    if HAS_NVR:
        nvr = gl.load(nvr_ptr)
        nvr = gl.minimum(gl.where(nvr < 0, 0, nvr), M)
    else:
        nvr = M

    # ---- 2. row t's logits from the flag scratch, top-8 -----------------------------------------------
    KL: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])
    cols = gl.arange(0, EXPERTS, layout=KL)
    lrow = lsc_ptr + t * LSC_STRIDE + cols
    lg = gl.load(lrow, cache_modifier=".cg", volatile=True)
    n_r = 0
    while (gl.max((lg.to(gl.int16, bitcast=True) == -1).to(gl.int32), axis=0) > 0) & (n_r < SPIN_MAX):
        lg = gl.load(lrow, cache_modifier=".cg", volatile=True)
        n_r += 1
    if n_r >= SPIN_MAX:
        gl.atomic_add(ws_ptr, 1, sem="relaxed", scope="gpu")
    b = (lg.to(gl.float32) + 0.0).to(gl.uint32, bitcast=True)  # -0 -> +0
    flip = (b.to(gl.int32, bitcast=True) >> 31).to(gl.uint32, bitcast=True) | 0x80000000
    keys = ((b ^ flip) & 0xFFFF0000) | (255 - cols).to(gl.uint32)  # bf16 order bits | 255 - col
    KO: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])  # 8 in-thread
    wk = gl.arange(0, TOPK, layout=KO)
    wins = _top8(keys, wk)
    top_e = 255 - (wins & 0xFF).to(gl.int32)
    is_pad = t >= nvr
    out_e = gl.where(is_pad, -1, top_e)

    # ---- 3. publish the tile's routes: expert e, row t -> word e * WPE + (t >> 5), bit t & 31 ------------
    LS: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])  # 8 values on lanes 0..7
    wl = gl.arange(0, TOPK, layout=LS)
    e_l = gl.convert_layout(out_e, LS)
    mine = (wl >= g * RPT) & (wl < g * RPT + RPT)
    e0 = gl.where(e_l >= 0, e_l, 0)
    gl.atomic_or(fws_ptr + TMOFF + (t >> 5) * EXPERTS + e0, gl.full([TOPK], 1, gl.int32, LS) << (t & 31),
                 mask=(e_l >= 0) & mine, sem="relaxed", scope="gpu")
    hi = wins & 0xFFFF0000
    vb = gl.where(hi >= 0x80000000, hi ^ 0x80000000, (hi ^ 0xFFFFFFFF) & 0xFFFF0000)
    top_v = vb.to(gl.float32, bitcast=True)
    v0 = gl.max(top_v, axis=0)
    numer = _exp_f32(top_v - v0)
    acc = _pick(numer, wk, 0)
    for k in gl.static_range(1, TOPK):
        acc = acc + _pick(numer, wk, k)
    nz = acc != 0.0
    scl = gl.where(nz, gl.fdiv(1.0, acc, ieee_rounding=True), 1.0)
    w = gl.where(nz, numer * scl, numer)
    out_w = gl.where(is_pad, 0.0, w)
    gl.amd.gfx1250.buffer_store(gl.convert_layout(out_w, LS), tw_ptr + t * TOPK, wl, mask=mine)
    gl.amd.gfx1250.buffer_store(e_l, tid_ptr + t * TOPK, wl, mask=mine)

    # ---- 4. every valid route landed: nvr * 8 bits in the WPE planes of 256 mask words -----------------
    EL: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])  # expert e on lane e // 8
    ex = gl.arange(0, EXPERTS, layout=EL)
    w0 = gl.load(fws_ptr + TMOFF + ex, cache_modifier=".cg", volatile=True)
    if WPE == 2:
        w1 = gl.load(fws_ptr + TMOFF + EXPERTS + ex, cache_modifier=".cg", volatile=True)
        nb_ = gl.sum(_bcnt(w0) + _bcnt(w1), axis=0)
    else:
        nb_ = gl.sum(_bcnt(w0), axis=0)
    n_it = 0
    while (nb_ != nvr * TOPK) & (n_it < SPIN_MAX):
        w0 = gl.load(fws_ptr + TMOFF + ex, cache_modifier=".cg", volatile=True)
        if WPE == 2:
            w1 = gl.load(fws_ptr + TMOFF + EXPERTS + ex, cache_modifier=".cg", volatile=True)
            nb_ = gl.sum(_bcnt(w0) + _bcnt(w1), axis=0)
        else:
            nb_ = gl.sum(_bcnt(w0), axis=0)
        n_it += 1
    if n_it >= SPIN_MAX:
        gl.atomic_add(ws_ptr, 1, sem="relaxed", scope="gpu")
    CL: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NUM_WARPS, 1], [1, 0])
    c_lanes = gl.arange(0, 32, layout=gl.SliceLayout(0, CL))
    c_waves = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(1, CL))
    c_lead = (c_waves[:, None] + c_lanes[None, :]) == 0
    c_zero = c_waves[:, None] * 0 + c_lanes[None, :] * 0
    if LATE_WAIT:
        c_used = gl.atomic_add(fws_ptr + 3 + _vzero(c_zero), c_zero + 1, mask=c_lead, sem="relaxed", scope="gpu")
    else:
        c_used = gl.atomic_add(fws_ptr + 3 + c_zero, c_zero + 1, mask=c_lead, sem="relaxed", scope="gpu")

    # ---- 5. routing ---------------------------------------------------------------------------------
    low = (1 << (t & 31)) - 1
    eg = gl.where(e_l >= 0, e_l, 0)
    w0k = gl.gather(w0, eg, 0)
    if WPE == 2:
        cnt = _bcnt(w0) + _bcnt(w1)
        w1k = gl.gather(w1, eg, 0)
        bl_k = gl.where(t < 32, _bcnt(w0k & low), _bcnt(w0k) + _bcnt(w1k & low))
    else:
        cnt = _bcnt(w0)
        bl_k = _bcnt(w0k & low)
    padded = (cnt + (ALIGN_M - 1)) & (-ALIGN_M)
    start = gl.cumsum(padded, axis=0) - padded
    cm = (ex >= u * CH) & (ex < u * CH + CH)
    gl.amd.gfx1250.buffer_store(cnt, mm_ptr, ex, mask=cm)
    gl.amd.gfx1250.buffer_store(start + cnt, psum_ptr, ex, mask=cm)
    st_k = gl.gather(start, eg, 0)
    rows = gl.where(e_l >= 0, st_k + bl_k, -1)
    gl.amd.gfx1250.buffer_store(rows, rows_ptr + t * TOPK, wl, mask=mine)
    rows_r = gl.convert_layout(rows, KO)

    # ---- 6. payload + e8m0 to the grouped rows ------------------------------------------------------
    if t < nvr:
        for k in gl.static_range(RPT):
            r = _pick(rows_r, wk, g * RPT + k)
            gl.amd.gfx1250.buffer_store(payload, pay_ptr, r * (FEAT // 8) + poffs)
            tile = r // (16 * WMMA_REP)
            rit = r - tile * (16 * WMMA_REP)
            base = tile * (16 * WMMA_REP) * 16 + (rit // 16) * 16 + (rit % 16)
            gl.amd.gfx1250.buffer_store(sdw, scale_ptr, base + sdi * (WMMA_REP * 16))

    # ---- 7. the last tile out restores the shared state for the next launch ---------------------------
    if LATE_WAIT:
        c_used = _vkeep(c_used)
    c_last = c_lead & (c_used == NT - 1)
    gl.atomic_xchg(fws_ptr + 3 + c_zero, c_zero, mask=c_last, sem="relaxed", scope="gpu")
    if gl.sum(c_last.to(gl.int32), axis=None) > 0:
        for pl in gl.static_range(WPE):
            gl.amd.gfx1250.buffer_store(gl.full([EXPERTS], 0, gl.int32, EL), fws_ptr + TMOFF + pl * EXPERTS, ex,
                                        cache=".cg")
        RL: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [NUM_WARPS, 1], [1, 0])
        rr0 = gl.arange(0, 16, layout=gl.SliceLayout(1, RL))
        rc0 = gl.arange(0, EXPERTS, layout=gl.SliceLayout(0, RL))
        neg = gl.full([16, EXPERTS], -1, gl.int16, RL).to(gl.bfloat16, bitcast=True)
        for r0 in range(0, M, 16):
            gl.amd.gfx1250.buffer_store(neg, lsc_ptr + r0 * LSC_STRIDE, rr0[:, None] * LSC_STRIDE + rc0[None, :],
                                        mask=rr0[:, None] < M - r0, cache=".cg")


@gluon.jit
def _gemm_a16w16_router_shared_expert_topk_v2w_kernel(
    x_ptr, wr_ptr, wgu_ptr, wd_ptr, logits_ptr, h_ptr, out_ptr, ws_ptr, fws_ptr, lsc_ptr, nvr_ptr, tw_ptr,
    tid_ptr, rows_ptr, mm_ptr, psum_ptr, pay_ptr, scale_ptr,
    M: gl.constexpr, stride_xm, stride_lm, stride_hm, stride_om,
    K: gl.constexpr, NR: gl.constexpr, NR_ROWS: gl.constexpr, NGU: gl.constexpr, ND: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N1: gl.constexpr, BLOCK_K1: gl.constexpr, BKT1: gl.constexpr,
    ROWS_R: gl.constexpr, ROWS_G: gl.constexpr, BLOCK_N2: gl.constexpr, BLOCK_K2: gl.constexpr,
    BKT2: gl.constexpr, P: gl.constexpr, PQ0: gl.constexpr, NUM_WARPS: gl.constexpr,
    SHARED_A1: gl.constexpr, SHARED_B1: gl.constexpr, SHARED_A2: gl.constexpr, SHARED_B2: gl.constexpr,
    WMMA1: gl.constexpr, OP_A1: gl.constexpr, OP_B1: gl.constexpr, WMMA2: gl.constexpr, OP_A2: gl.constexpr,
    OP_B2: gl.constexpr, STORE1: gl.constexpr, TIMEOUT_NS: gl.constexpr, H_CACHE: gl.constexpr,
    LSC_STRIDE: gl.constexpr, ALIGN_M: gl.constexpr, WMMA_REP: gl.constexpr, HAS_NVR: gl.constexpr,
    RPT: gl.constexpr, MB: gl.constexpr, WPE: gl.constexpr,
    LATE_WAIT: gl.constexpr = 0, XCD_RANKS: gl.constexpr = 0, CNT_SPLIT: gl.constexpr = 1,
):
    KD: gl.constexpr = NGU // 2
    N_GU: gl.constexpr = NGU // ROWS_G
    N_R: gl.constexpr = (NR + ROWS_R - 1) // ROWS_R
    N_D: gl.constexpr = ND // BLOCK_N2
    NQ: gl.constexpr = M * (TOPK // RPT)
    HG: gl.constexpr = ROWS_G // 2  # h columns of a gate/up unit
    NKT2: gl.constexpr = KD // BKT2
    NC2: gl.constexpr = BKT2 // BLOCK_K2
    pid = gl.program_id(axis=0)
    if XCD_RANKS != 0:
        pid = _xcd_pid(pid, P, XCD_RANKS)
        if pid < 0:
            return
    cnt_ptr = ws_ptr + WS_HEADER
    use_ptr = cnt_ptr + WS_MAX_MB
    zi = pid * 0

    # down tile: its Wd slice [BLOCK_N2, KD] goes to LDS before anything else
    d = P - 1 - pid
    is_down = d < N_D
    dd = gl.minimum(d, N_D - 1)
    b2_buf = gl.allocate_shared_memory(wd_ptr.type.element_ty, shape=[NKT2, BLOCK_N2, BKT2], layout=SHARED_B2)
    b2_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=wd_ptr + dd * BLOCK_N2 * KD, shape=(ND - dd * BLOCK_N2, KD), strides=(KD, 1),
        block_shape=(BLOCK_N2, BKT2), layout=SHARED_B2)
    if is_down:
        for k in gl.static_range(NKT2):
            gl.amd.gfx1250.tdm.async_load(b2_desc, [0, k * BKT2], b2_buf.index(k))

    # router unit (router rows nr_ * ROWS_R .., M block rb): logits -> output and the flag scratch
    if pid < N_R * MB:
        rb = pid // N_R
        nr_ = pid - rb * N_R
        _unit_mb(x_ptr, wr_ptr, M, NR_ROWS, K, stride_xm, nr_, lsc_ptr, logits_ptr, LSC_STRIDE, stride_lm,
                 nr_ * ROWS_R + ROWS_R, gl.minimum(nr_ * ROWS_R + ROWS_R, NR), nr_ * ROWS_R,
                 BLOCK_M, BLOCK_N1, BKT1, BLOCK_K1, SHARED_A1, SHARED_B1, WMMA1, OP_A1, OP_B1, ROWS_R, 1, False,
                 STORE1, H_CACHE, rb * BLOCK_M)
    if pid < NQ:
        _route_tile_wide(pid, x_ptr, nvr_ptr, tw_ptr, tid_ptr, rows_ptr, mm_ptr, psum_ptr, pay_ptr, scale_ptr,
                         fws_ptr, ws_ptr, lsc_ptr, M, stride_xm, LSC_STRIDE, ALIGN_M, WMMA_REP, HAS_NVR, NUM_WARPS,
                         RPT, WPE, LATE_WAIT)
    # gate/up unit (weight rows gu * ROWS_G .., M blocks gh * GB ..): h = silu(gate) * up, then one arrival a wave
    GB: gl.constexpr = (MB + 1) // 2
    g = pid - PQ0
    if (g >= 0) & (g < N_GU * 2):
        gh = g // N_GU
        gu = g - gh * N_GU
        _unit_mb(x_ptr, wgu_ptr, M, NGU, K, stride_xm, gu, h_ptr, h_ptr, stride_hm, stride_hm,
                 gu * HG + HG, gu * HG + HG, gu * HG,
                 BLOCK_M, BLOCK_N1, BKT1, BLOCK_K1, SHARED_A1, SHARED_B1, WMMA1, OP_A1, OP_B1, ROWS_G, GB, True,
                 STORE1, H_CACHE, gh * (GB * BLOCK_M))
        if CNT_SPLIT > 1:
            _wave_arrive(cnt_ptr + (g % CNT_SPLIT) * 32, NUM_WARPS)
        else:
            _wave_arrive(cnt_ptr, NUM_WARPS)
    # down tile: wait for every gate/up unit, every M block of h -> LDS, a K chain a block, store
    if is_down:
        if CNT_SPLIT > 1:
            _split_poll(cnt_ptr, N_GU * 2 * NUM_WARPS, ws_ptr, CNT_SPLIT, NUM_WARPS)
        else:
            ok = gl.atomic_poll(cnt_ptr, N_GU * 2 * NUM_WARPS, sem="relaxed", scope="gpu", timeout_ns=TIMEOUT_NS)
            if not ok:
                gl.atomic_add(ws_ptr, 1, sem="relaxed", scope="gpu")
        L: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NUM_WARPS, 1], [1, 0])
        lanes = gl.arange(0, 32, layout=gl.SliceLayout(0, L))
        waves = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(1, L))
        lead = (waves[:, None] + lanes[None, :]) == 0
        zero = waves[:, None] * 0 + lanes[None, :] * 0
        if LATE_WAIT:
            # the poll above was this tile's last read of the counter: count it out now, use the value at the end
            used = gl.atomic_add(use_ptr + _vzero(zero), zero + 1, mask=lead, sem="relaxed", scope="gpu")
        a2_buf = gl.allocate_shared_memory(h_ptr.type.element_ty, shape=[MB * NKT2, BLOCK_M, BKT2],
                                           layout=SHARED_A2)
        a2_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=h_ptr, shape=(M, KD), strides=(stride_hm, 1), block_shape=(BLOCK_M, BKT2), layout=SHARED_A2)
        for mb in gl.static_range(MB):
            for k in gl.static_range(NKT2):
                gl.amd.gfx1250.tdm.async_load(a2_desc, [mb * BLOCK_M, k * BKT2], a2_buf.index(mb * NKT2 + k),
                                              cache_modifier=H_CACHE)
        for mb in gl.static_range(MB):
            acc2 = gl.zeros((BLOCK_M, BLOCK_N2), dtype=gl.float32, layout=WMMA2)
            for k in gl.static_range(NKT2):
                gl.amd.gfx1250.tdm.async_wait((MB - 1 - mb) * NKT2 + NKT2 - 1 - k)
                for c in gl.static_range(NC2):
                    cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                        a2_buf.index(mb * NKT2 + k).slice(c * BLOCK_K2, BLOCK_K2, 1), OP_A2)
                    cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                        b2_buf.index(k).slice(c * BLOCK_K2, BLOCK_K2, 1).permute([1, 0]), OP_B2)
                    acc2 = gl.amd.gfx1250.wmma(cur_a, cur_b, acc2)
            _store_tile(acc2, out_ptr, M, ND, stride_om, zi + mb * BLOCK_M, d * BLOCK_N2, WMMA2, "")
        if LATE_WAIT:
            used = _vkeep(used)
        else:
            used = gl.atomic_add(use_ptr + zero, zero + 1, mask=lead, sem="relaxed", scope="gpu")
        last = lead & (used == N_D - 1)
        gl.atomic_xchg(cnt_ptr + zero, zero, mask=last, sem="relaxed", scope="gpu")
        for j in gl.static_range(1, CNT_SPLIT):
            gl.atomic_xchg(cnt_ptr + j * 32 + zero, zero, mask=last, sem="relaxed", scope="gpu")
        gl.atomic_xchg(use_ptr + zero, zero, mask=last, sem="relaxed", scope="gpu")
