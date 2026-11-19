# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# The MoE router GEMM + bf16 shared expert kernel (`_gemm_a16w16_router_shared_expert_kernel`)
# with `moe_topk_route_quant` (its local decode schedule: fp4, 256 experts, top-8, renorm, no
# fused shared expert) folded in as a fourth tile class, for decode batches of at most 16 rows
# (one M block). One launch replaces two; every output is bitwise that of the pair.
#
# Tiles (one program each, every program resident: the wrapper checks the grid fits):
#   G  gate / up tiles   as the original kernel (h, per-wave arrivals for the down tiles)
#   R  router tiles      as the original kernel, but the logits go out as device-scope stores
#                        followed by one device-scope add per wave to RCNT (fws[1])
#   Q  route tiles       RPT routes of one row each (8 // RPT tiles a row):
#                        1. MXFP4-quantize x[t] (amax, RoundUp e8m0, v_cvt_scalef32_pk8 via
#                           scaled_downcast): needs nothing, runs under the GEMMs
#                        2. spin on RCNT, load row t's 256 routed logits, top-8 as
#                           topkGatingSoftmax<renorm> (u32 keys: bf16 order bits | 255 - col,
#                           ties to the lower expert; w = expf(l - l_max) / sum in k order)
#                        3. publish: atomic_or of bit t into tokmask[e] (fws[256 + e]) for the
#                           tile's routes
#                        4. spin until the token masks hold nvr * 8 bits; count the tile out
#                           (the last one clears RCNT and the masks for the next launch)
#                        5. routing from the masks: masked_m[e] = popc, psum[e] = start + popc,
#                           start[e] = ALIGN_M * (present experts below e) (ALIGN_M >= rows, so a
#                           present expert pads to exactly ALIGN_M), the route's grouped row =
#                           start[e] + popc(mask[e] & ((1 << t) - 1)) (token-major, as native)
#                        6. store the payload row and the WMMA-preshuffled e8m0 dwords to each of
#                           its routes' grouped rows
#   D  down tiles        as the original kernel
# Rows at or past num_valid_rows get ids -1 / weights 0 / rows -1 and claim no grouped row.

import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
    _compute_bound_accumulate,
)
from aiter.ops.triton._gluon_kernels.gfx1250.gemm.fused.fused_gemm_a16w16_router_shared_expert import (
    WS_HEADER,
    WS_MAX_MB,
    _down_tile,
    _store_tile,
    _wave_arrive,
)

TOPK = gl.constexpr(8)
EXPERTS = gl.constexpr(256)
FEAT = gl.constexpr(2048)
# fold workspace (int32): [0] unused, [1] router-wave arrivals, [3] route tiles counted out,
# [256 + e] token mask of expert e
TMOFF = gl.constexpr(256)
SPIN_MAX = gl.constexpr(400000)


@gluon.jit
def _pick(vec, idx, k):
    """Scalar vec[k] of a 1-D tensor (idx: its arange)."""
    return gl.sum(gl.where(idx == k, vec, 0), axis=0)


@gluon.jit
def _bcnt(x):
    return gl.inline_asm_elementwise("v_bcnt_u32_b32 $0, $1, 0", "=v,v", [x], gl.int32, True, 1)


@gluon.jit
def _exp_f32(x):
    """llvm.exp.f32 as the AMDGPU backend expands it with fast FMA (what topkGatingSoftmax's
    expf compiles to), bit for bit, for x <= 0."""
    ph = x * 1.4426950216293335  # 0x3FB8AA3B
    fma0 = gl.fma(x, 1.4426950216293335, -ph)
    pl = gl.fma(x, 1.925963033500011e-08, fma0)  # 0x32A5705F
    e = gl.inline_asm_elementwise("v_rndne_f32 $0, $1", "=v,v", [ph], gl.float32, True, 1)
    a = (ph - e) + pl
    r = gl.inline_asm_elementwise("v_exp_f32 $0, $1", "=v,v", [a], gl.float32, True, 1)
    r = gl.inline_asm_elementwise("v_ldexp_f32 $0, $1, $2", "=v,v,v", [r, e.to(gl.int32)], gl.float32, True, 1)
    return gl.where(x < -103.27892303466797, 0.0, r)  # -0x1.9d1da0p+6: underflow to 0


@gluon.jit
def _count_out(fws_ptr, NT: gl.constexpr, NUM_WARPS: gl.constexpr):
    """Count this route tile out of the shared state; the last one clears it."""
    L: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NUM_WARPS, 1], [1, 0])
    lanes = gl.arange(0, 32, layout=gl.SliceLayout(0, L))
    waves = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(1, L))
    lead = (waves[:, None] + lanes[None, :]) == 0
    zero = waves[:, None] * 0 + lanes[None, :] * 0
    used = gl.atomic_add(fws_ptr + 3 + zero, zero + 1, mask=lead, sem="relaxed", scope="gpu")
    last = lead & (used == NT - 1)
    gl.atomic_xchg(fws_ptr + 1 + zero, zero, mask=last, sem="relaxed", scope="gpu")
    gl.atomic_xchg(fws_ptr + 3 + zero, zero, mask=last, sem="relaxed", scope="gpu")
    if gl.sum(last.to(gl.int32), axis=None) > 0:
        XL: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])
        xj = gl.arange(0, EXPERTS, layout=XL)
        gl.amd.gfx1250.buffer_store(gl.full([EXPERTS], 0, gl.int32, XL), fws_ptr + TMOFF, xj, cache=".cg")


@gluon.jit
def _route_tile(
    u,
    x_ptr,
    logits_ptr,
    nvr_ptr,
    tw_ptr,
    tid_ptr,
    rows_ptr,
    mm_ptr,
    psum_ptr,
    pay_ptr,
    scale_ptr,
    fws_ptr,
    ws_ptr,
    M: gl.constexpr,
    stride_xm,
    stride_lm,
    r_expected,
    ALIGN_M: gl.constexpr,
    WMMA_REP: gl.constexpr,
    HAS_NVR: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    RPT: gl.constexpr,
):
    G: gl.constexpr = TOPK // RPT  # route tiles a row
    NT: gl.constexpr = M * G
    CH: gl.constexpr = (EXPERTS + NT - 1) // NT  # masked_m / psum entries a tile writes
    gl.static_assert(ALIGN_M >= M)
    t = u // G
    g = u - t * G

    # ---- 1. MXFP4 quant of x[t] ---------------------------------------------------------------
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

    # ---- 2. top-8 of row t ------------------------------------------------------------------
    rc = gl.load(fws_ptr + 1, cache_modifier=".cg", volatile=True)
    n_r = 0
    while (rc != r_expected) & (n_r < SPIN_MAX):
        rc = gl.load(fws_ptr + 1, cache_modifier=".cg", volatile=True)
        n_r += 1
    if n_r >= SPIN_MAX:
        gl.atomic_add(ws_ptr, 1, sem="relaxed", scope="gpu")
    KL: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])
    cols = gl.arange(0, EXPERTS, layout=KL)
    lg = gl.amd.gfx1250.buffer_load(logits_ptr + t.to(gl.int64) * stride_lm, cols, cache=".cg")
    b = (lg.to(gl.float32) + 0.0).to(gl.uint32, bitcast=True)  # -0 -> +0
    flip = (b.to(gl.int32, bitcast=True) >> 31).to(gl.uint32, bitcast=True) | 0x80000000
    keys = ((b ^ flip) & 0xFFFF0000) | (255 - cols).to(gl.uint32)
    KO: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])  # 8 in-thread
    wk = gl.arange(0, TOPK, layout=KO)
    wins = gl.zeros([TOPK], gl.uint32, KO)
    bound = gl.full([EXPERTS], 0xFFFFFFFF, gl.uint32, KL)
    for k in gl.static_range(TOPK):
        win = gl.max(gl.where(keys <= bound, keys, 0), axis=0)
        wins = gl.where(wk == k, win, wins)
        bound = bound * 0 + (win - 1)
    top_e = 255 - (wins & 0xFF).to(gl.int32)
    is_pad = t >= nvr
    out_e = gl.where(is_pad, -1, top_e)

    # ---- 3. publish the tile's routes into the token masks ------------------------------------
    LS: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])  # 8 values on lanes 0..7
    wl = gl.arange(0, TOPK, layout=LS)
    e_l = gl.convert_layout(out_e, LS)
    mine = (wl >= g * RPT) & (wl < g * RPT + RPT)
    gl.atomic_or(fws_ptr + TMOFF + gl.where(e_l >= 0, e_l, 0), gl.full([TOPK], 1, gl.int32, LS) << t,
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

    # ---- 4. every valid route landed: nvr * 8 bits in the masks --------------------------------
    EL: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])  # expert e on lane e // 8
    ex = gl.arange(0, EXPERTS, layout=EL)
    tm = gl.load(fws_ptr + TMOFF + ex, cache_modifier=".cg", volatile=True)
    n_it = 0
    while (gl.sum(_bcnt(tm), axis=0) != nvr * TOPK) & (n_it < SPIN_MAX):
        tm = gl.load(fws_ptr + TMOFF + ex, cache_modifier=".cg", volatile=True)
        n_it += 1
    if n_it >= SPIN_MAX:
        gl.atomic_add(ws_ptr, 1, sem="relaxed", scope="gpu")
    _count_out(fws_ptr, NT, NUM_WARPS)  # its round trip overlaps 5. and 6.

    # ---- 5. routing ---------------------------------------------------------------------------
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

    # ---- 6. payload + e8m0 to the grouped rows -----------------------------------------------
    if t < nvr:
        for k in gl.static_range(RPT):
            r = _pick(rows_r, wk, g * RPT + k)
            gl.amd.gfx1250.buffer_store(payload, pay_ptr, r * (FEAT // 8) + poffs)
            tile = r // (16 * WMMA_REP)
            rit = r - tile * (16 * WMMA_REP)
            base = tile * (16 * WMMA_REP) * 16 + (rit // 16) * 16 + (rit % 16)
            gl.amd.gfx1250.buffer_store(sdw, scale_ptr, base + sdi * (WMMA_REP * 16))


@gluon.jit
def _gemm_a16w16_router_shared_expert_topk_kernel(
    x_ptr,
    wr_ptr,
    wgu_ptr,
    wd_ptr,
    logits_ptr,
    h_ptr,
    out_ptr,
    ws_ptr,
    fws_ptr,
    nvr_ptr,
    tw_ptr,
    tid_ptr,
    rows_ptr,
    mm_ptr,
    psum_ptr,
    pay_ptr,
    scale_ptr,
    M: gl.constexpr,
    stride_xm,
    stride_lm,
    stride_hm,
    stride_om,
    K: gl.constexpr,
    NR: gl.constexpr,
    NR_ROWS: gl.constexpr,
    NGU: gl.constexpr,
    ND: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N1: gl.constexpr,
    BLOCK_K1: gl.constexpr,
    NUM_BUFFERS1: gl.constexpr,
    BLOCK_N2: gl.constexpr,
    BLOCK_K2: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    LAYOUT: gl.constexpr,
    SHARED_A1: gl.constexpr,
    SHARED_B1: gl.constexpr,
    SHARED_A2: gl.constexpr,
    SHARED_B2: gl.constexpr,
    WMMA1: gl.constexpr,
    OP_A1: gl.constexpr,
    OP_B1: gl.constexpr,
    WMMA2: gl.constexpr,
    OP_A2: gl.constexpr,
    OP_B2: gl.constexpr,
    STORE1: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
    H_CACHE: gl.constexpr,
    ALIGN_M: gl.constexpr,
    WMMA_REP: gl.constexpr,
    HAS_NVR: gl.constexpr,
    RPT: gl.constexpr,
):
    KD: gl.constexpr = NGU // 2
    N_GU: gl.constexpr = NGU // BLOCK_N1
    N_R: gl.constexpr = (NR + BLOCK_N1 - 1) // BLOCK_N1
    N_D: gl.constexpr = ND // BLOCK_N2
    HALF: gl.constexpr = BLOCK_N1 // 2
    pid = gl.program_id(axis=0)
    t_gu = N_GU
    t_1 = t_gu + N_R
    t_2 = t_1 + M * (TOPK // RPT)
    t_all = t_2 + N_D
    cnt_ptr = ws_ptr + WS_HEADER
    zi = pid * 0

    for t in range(pid, t_all, NUM_PROGRAMS):
        if t < t_gu:
            nb = t
            acc = _compute_bound_accumulate(
                x_ptr, wgu_ptr, M, NGU, K, stride_xm, 1, 1, K, 0, nb,
                BLOCK_M, BLOCK_N1, BLOCK_K1, NUM_BUFFERS1, LAYOUT,
                SHARED_A1, SHARED_B1, WMMA1, OP_A1, OP_B1,
            )
            gate, up = gl.split(gl.reshape(acc, [BLOCK_M, HALF, 2]))
            gate = gate.to(h_ptr.type.element_ty).to(gl.float32)
            up = up.to(h_ptr.type.element_ty).to(gl.float32)
            hv = gate / (1.0 + gl.exp(-gate)) * up
            hv = gl.convert_layout(hv, STORE1)
            _store_tile(hv, h_ptr, M, KD, stride_hm, zi, nb * HALF, STORE1, H_CACHE)
            _wave_arrive(cnt_ptr, NUM_WARPS)
        elif t < t_1:
            nb = t - t_gu
            acc = _compute_bound_accumulate(
                x_ptr, wr_ptr, M, NR_ROWS, K, stride_xm, 1, 1, K, 0, nb,
                BLOCK_M, BLOCK_N1, BLOCK_K1, NUM_BUFFERS1, LAYOUT,
                SHARED_A1, SHARED_B1, WMMA1, OP_A1, OP_B1,
            )
            _store_tile(acc, logits_ptr, M, NR, stride_lm, zi, nb * BLOCK_N1, WMMA1, ".cg")
            _wave_arrive(fws_ptr + 1, NUM_WARPS)
        elif t < t_2:
            _route_tile(
                t - t_1, x_ptr, logits_ptr, nvr_ptr, tw_ptr, tid_ptr, rows_ptr, mm_ptr,
                psum_ptr, pay_ptr, scale_ptr, fws_ptr, ws_ptr, M, stride_xm, stride_lm,
                N_R * NUM_WARPS, ALIGN_M, WMMA_REP, HAS_NVR, NUM_WARPS, RPT,
            )
        else:
            _down_tile(
                h_ptr, wd_ptr, out_ptr, cnt_ptr, cnt_ptr + WS_MAX_MB, ws_ptr, M,
                stride_hm, stride_om, zi, t - t_2, N_GU * NUM_WARPS, KD, ND,
                BLOCK_M, BLOCK_N2, BLOCK_K2, SHARED_A2, SHARED_B2, WMMA2, OP_A2, OP_B2,
                N_D, NUM_WARPS, H_CACHE, TIMEOUT_NS,
            )
