"""Persistent Gluon FP8 PTPC a8w8 GEMM for gfx1250 (k9p, experimental, not wired into the wrapper yet).

y = (x @ w.T) * x_scale * w_scale, x [M, K] fp8, w [N, K] fp8 with the K tail (from K/2) pre-negated
(gemm_a8w8_negate_w_k_tail), fp32 row/column scales, bf16 out. No bias.

Layout: cluster tile BM x BN = 2 x 2 sub-tiles, each its own cluster tensor (TDM descriptor, LDS ring, accumulator).
One wave per SIMD (4 warps); a wave holds 4 quadrant accumulators c00 c01 c10 c11.
K loop: LDS stage BK = 128 in an NB = 4 slot ring, consumed as two K halves of 64 (v_wmma 16x16x64) with two
register sets: half 0 computes on X while loading Y, half 1 computes on Y while loading X of the next stage.
Inside a half, 4 groups of 16 WMMAs; each group loads the other set's slab whose last reader is >= 2 groups back:
    G1 c00 = a0 b0 | a0'   G2 c01 = a0 b1 | b1'   G3 c11 = a1 b1 | b0'   G4 c10 = a1 b0 | a1'
Per stage: tdm wait + CTA barrier in half 0 (RAW of stage i+1; its s_wait_dscnt 0 also makes the warp-0 cluster
arrive safe), cluster arrive after half-1 G3, cluster wait + TDM refill of slot i at the end.
The TDM ring runs across tiles: the last NB stages of a tile load the first NB stages of the next tile.
Truncation-bias fix: the accumulators are negated at stage K/2/BK and the final sign goes into x_scale.
"""

import math

import triton
import triton.experimental.gluon.language as gl
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon

import aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a8w8 as P

KH = gl.constexpr(64)  # K per WMMA (v_wmma_f32_16x16x64_fp8_fp8)
BK = gl.constexpr(128)  # K per LDS stage
NB = gl.constexpr(4)  # LDS ring slots


@gluon.jit
def _ld_a(buf, slot, kh: gl.constexpr, L: gl.constexpr):
    return gl.amd.cdna4.async_copy.load_shared_relaxed(
        buf.index(slot).slice(kh * 64, 64, 1), L
    )


@gluon.jit
def _ld_b(buf, slot, kh: gl.constexpr, L: gl.constexpr):
    return gl.amd.cdna4.async_copy.load_shared_relaxed(
        buf.index(slot).slice(kh * 64, 64, 1).permute([1, 0]), L
    )


@gluon.jit
def _issue(da0, da1, db0, db1, ab0, ab1, bb0, bb1, slot):
    """One fused TDM load per stage: warp w loads slab w (a0 a1 b0 b1)."""
    gl.amd.gfx1250.tdm.async_load_fused(
        [
            (da0, ab0.index(slot), 1),
            (da1, ab1.index(slot), 2),
            (db0, bb0.index(slot), 4),
            (db1, bb1.index(slot), 8),
        ]
    )
    da0 = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        da0, add_offsets=[0, 128], clamp_bounds=False
    )
    da1 = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        da1, add_offsets=[0, 128], clamp_bounds=False
    )
    db0 = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        db0, add_offsets=[0, 128], clamp_bounds=False
    )
    db1 = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        db1, add_offsets=[0, 128], clamp_bounds=False
    )
    return da0, da1, db0, db1


@gluon.jit
def _half(
    c00,
    c01,
    c10,
    c11,
    a0,
    a1,
    b0,
    b1,
    ab0,
    ab1,
    bb0,
    bb1,
    lslot,
    lkh: gl.constexpr,
    OPA: gl.constexpr,
    OPB: gl.constexpr,
    ARRIVE: gl.constexpr,
    NUM_CTAS: gl.constexpr,
    AW: gl.constexpr,
):
    """64 WMMAs on (a0 a1 b0 b1) while loading K half lkh of slot lslot into the other register set.

    sched_barriers after G2 and G4 pin the groups. ARRIVE: cluster arrive after G3.
    AW = 4: tdm wait (stage i+1) after G4. AW = 6: as 4, with G4's load moved into G3. AW = 0: no wait.
    """
    c00 = gl.amd.gfx1250.wmma(a0, b0, c00)
    na0 = _ld_a(ab0, lslot, lkh, OPA)
    c01 = gl.amd.gfx1250.wmma(a0, b1, c01)
    nb1 = _ld_b(bb1, lslot, lkh, OPB)
    gl.amd.hint.sched_barrier()
    c11 = gl.amd.gfx1250.wmma(a1, b1, c11)
    nb0 = _ld_b(bb0, lslot, lkh, OPB)
    if AW == 6:
        na1 = _ld_a(ab1, lslot, lkh, OPA)
        gl.amd.hint.sched_barrier()
    if ARRIVE and NUM_CTAS > 1:
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.hint.sched_barrier()
    c10 = gl.amd.gfx1250.wmma(a1, b0, c10)
    if AW == 6:
        gl.amd.hint.sched_barrier()
        gl.amd.gfx1250.tdm.async_wait(NB - 2)
        gl.amd.hint.sched_barrier()
    else:
        na1 = _ld_a(ab1, lslot, lkh, OPA)
        gl.amd.hint.sched_barrier()
    if AW == 4:
        gl.amd.gfx1250.tdm.async_wait(NB - 2)
        gl.amd.hint.sched_barrier()
    return c00, c01, c10, c11, na0, na1, nb0, nb1


@gluon.jit
def _stage(
    c00,
    c01,
    c10,
    c11,
    x0,
    x1,
    y0,
    y1,
    da0,
    da1,
    db0,
    db1,
    ab0,
    ab1,
    bb0,
    bb1,
    SLOT: gl.constexpr,
    OPA: gl.constexpr,
    OPB: gl.constexpr,
    NUM_CTAS: gl.constexpr,
    AW: gl.constexpr,
):
    """One LDS stage (slot SLOT): half 0 on X, half 1 on Y, then refill slot SLOT with stage i + NB."""
    NSLOT: gl.constexpr = (SLOT + 1) % NB
    c00, c01, c10, c11, p0, p1, q0, q1 = _half(
        c00,
        c01,
        c10,
        c11,
        x0,
        x1,
        y0,
        y1,
        ab0,
        ab1,
        bb0,
        bb1,
        SLOT,
        1,
        OPA,
        OPB,
        0,
        NUM_CTAS,
        AW,
    )
    c00, c01, c10, c11, x0, x1, y0, y1 = _half(
        c00,
        c01,
        c10,
        c11,
        p0,
        p1,
        q0,
        q1,
        ab0,
        ab1,
        bb0,
        bb1,
        NSLOT,
        0,
        OPA,
        OPB,
        1,
        NUM_CTAS,
        0,
    )
    if NUM_CTAS > 1:
        gl.amd.gfx1250.cluster.wait()
    da0, da1, db0, db1 = _issue(da0, da1, db0, db1, ab0, ab1, bb0, bb1, SLOT)
    gl.amd.hint.sched_barrier()
    return c00, c01, c10, c11, x0, x1, y0, y1, da0, da1, db0, db1


@gluon.jit
def _descs(
    a_ptr,
    b_ptr,
    t,
    num_pid_m,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    BM: gl.constexpr,
    BN: gl.constexpr,
    SL_A: gl.constexpr,
    SL_B: gl.constexpr,
):
    """Tile t (M-fastest) -> pid_m, pid_n and the 4 sub-tile TDM descriptors."""
    pid_m = t % num_pid_m
    pid_n = t // num_pid_m
    SM: gl.constexpr = BM // 2
    SN: gl.constexpr = BN // 2
    m0 = pid_m * BM
    n0 = pid_n * BN
    da0 = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + m0.to(gl.int64) * stride_am,
        shape=(M - m0, K),
        strides=(stride_am, 1),
        block_shape=(SM, 128),
        layout=SL_A,
    )
    da1 = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + (m0 + SM).to(gl.int64) * stride_am,
        shape=(gl.maximum(M - m0 - SM, 0), K),
        strides=(stride_am, 1),
        block_shape=(SM, 128),
        layout=SL_A,
    )
    db0 = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + n0.to(gl.int64) * stride_bn,
        shape=(N - n0, K),
        strides=(stride_bn, 1),
        block_shape=(SN, 128),
        layout=SL_B,
    )
    db1 = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + (n0 + SN).to(gl.int64) * stride_bn,
        shape=(gl.maximum(N - n0 - SN, 0), K),
        strides=(stride_bn, 1),
        block_shape=(SN, 128),
        layout=SL_B,
    )
    return pid_m, pid_n, da0, da1, db0, db1


@gluon.jit
def _scales(
    a_scale_ptr,
    b_scale_ptr,
    qm,
    qn,
    M,
    N,
    SM: gl.constexpr,
    SN: gl.constexpr,
    WL: gl.constexpr,
):
    offs_m = qm * SM + gl.arange(0, SM, layout=gl.SliceLayout(1, WL))
    offs_n = qn * SN + gl.arange(0, SN, layout=gl.SliceLayout(0, WL))
    a_s = gl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=0.0)
    b_s = gl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    return a_s, b_s


@gluon.jit
def _store_q(
    acc,
    a_s,
    b_s,
    c_ptr,
    qm,
    qn,
    M,
    N,
    stride_cm,
    stride_cn,
    SM: gl.constexpr,
    SN: gl.constexpr,
    NUM_CTAS: gl.constexpr,
    WL: gl.constexpr,
    STORE_LAYOUT: gl.constexpr,
):
    """Scale, cast and store one quadrant (coalesced store through a layout conversion)."""
    P._scale_and_store(
        None,
        acc,
        a_s[:, None],
        b_s[None, :],
        b_s[None, :],
        c_ptr,
        qm,
        qn,
        M,
        N,
        stride_cm,
        stride_cn,
        SM,
        SN,
        NUM_CTAS,
        WL,
        None,
        2,
        STORE_LAYOUT,
        None,
        False,
    )


@gluon.jit
def _tile_k(
    c00,
    c01,
    c10,
    c11,
    x0,
    x1,
    y0,
    y1,
    da0,
    da1,
    db0,
    db1,
    a_ptr,
    b_ptr,
    nt,
    num_pid_m,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    BM: gl.constexpr,
    BN: gl.constexpr,
    SL_A: gl.constexpr,
    SL_B: gl.constexpr,
    ab0,
    ab1,
    bb0,
    bb1,
    num_first,
    num_main,
    OPA: gl.constexpr,
    OPB: gl.constexpr,
    NUM_CTAS: gl.constexpr,
    AW: gl.constexpr,
):
    """All K stages of one tile (loops unrolled by NB: static slots). The last NB stages refill the ring with the
    first NB stages of tile nt."""
    for it in range(0, num_first // NB):
        for u in gl.static_range(NB):
            c00, c01, c10, c11, x0, x1, y0, y1, da0, da1, db0, db1 = _stage(
                c00,
                c01,
                c10,
                c11,
                x0,
                x1,
                y0,
                y1,
                da0,
                da1,
                db0,
                db1,
                ab0,
                ab1,
                bb0,
                bb1,
                u,
                OPA,
                OPB,
                NUM_CTAS,
                AW,
            )
    # second half of K: the weight is negated there, so negate the accumulators once
    c00 = -c00
    c01 = -c01
    c10 = -c10
    c11 = -c11
    for it in range(num_first // NB, num_main // NB):
        for u in gl.static_range(NB):
            c00, c01, c10, c11, x0, x1, y0, y1, da0, da1, db0, db1 = _stage(
                c00,
                c01,
                c10,
                c11,
                x0,
                x1,
                y0,
                y1,
                da0,
                da1,
                db0,
                db1,
                ab0,
                ab1,
                bb0,
                bb1,
                u,
                OPA,
                OPB,
                NUM_CTAS,
                AW,
            )
    npm, npn, na0, na1, nb0, nb1 = _descs(
        a_ptr, b_ptr, nt, num_pid_m, M, N, K, stride_am, stride_bn, BM, BN, SL_A, SL_B
    )
    for d in gl.static_range(NB):
        c00, c01, c10, c11, x0, x1, y0, y1, na0, na1, nb0, nb1 = _stage(
            c00,
            c01,
            c10,
            c11,
            x0,
            x1,
            y0,
            y1,
            na0,
            na1,
            nb0,
            nb1,
            ab0,
            ab1,
            bb0,
            bb1,
            d,
            OPA,
            OPB,
            NUM_CTAS,
            AW,
        )
    return c00, c01, c10, c11, x0, x1, y0, y1, na0, na1, nb0, nb1, npm, npn


@gluon.jit
def gemm_a8w8_k9p_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M: gl.constexpr,
    N,
    K,
    stride_am: gl.constexpr,
    stride_bn: gl.constexpr,
    stride_cm: gl.constexpr,
    first_stages: gl.constexpr,
    num_tiles: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    SL_A: gl.constexpr,
    SL_B: gl.constexpr,
    WL: gl.constexpr,
    OPA: gl.constexpr,
    OPB: gl.constexpr,
    STORE_LAYOUT: gl.constexpr,
    NUM_CTAS: gl.constexpr,
    B2B: gl.constexpr,
    AW: gl.constexpr,
    GRID: gl.constexpr,
):
    if B2B:
        P._enable_b2b_wmma()
    SM: gl.constexpr = BM // 2
    SN: gl.constexpr = BN // 2
    pid = gl.program_id(0)
    num_pid_m = gl.cdiv(M, BM)

    # ---- prologue: descriptors, LDS ring, first NB stages, scales, register set X of stage 0
    pid_m, pid_n, da0, da1, db0, db1 = _descs(
        a_ptr, b_ptr, pid, num_pid_m, M, N, K, stride_am, stride_bn, BM, BN, SL_A, SL_B
    )
    ab0 = gl.allocate_shared_memory(a_ptr.type.element_ty, [NB, SM, BK], SL_A)
    ab1 = gl.allocate_shared_memory(a_ptr.type.element_ty, [NB, SM, BK], SL_A)
    bb0 = gl.allocate_shared_memory(b_ptr.type.element_ty, [NB, SN, BK], SL_B)
    bb1 = gl.allocate_shared_memory(b_ptr.type.element_ty, [NB, SN, BK], SL_B)
    for s in gl.static_range(NB):
        da0, da1, db0, db1 = _issue(da0, da1, db0, db1, ab0, ab1, bb0, bb1, s)
    as0, bs0 = _scales(a_scale_ptr, b_scale_ptr, pid_m * 2, pid_n * 2, M, N, SM, SN, WL)
    as1, bs1 = _scales(
        a_scale_ptr, b_scale_ptr, pid_m * 2 + 1, pid_n * 2 + 1, M, N, SM, SN, WL
    )
    gl.amd.gfx1250.tdm.async_wait(NB - 1)
    x0 = _ld_a(ab0, 0, 0, OPA)
    x1 = _ld_a(ab1, 0, 0, OPA)
    y0 = _ld_b(bb0, 0, 0, OPB)
    y1 = _ld_b(bb1, 0, 0, OPB)
    num_main = gl.cdiv(K, BK) - NB
    num_first = gl.maximum(gl.minimum(first_stages, num_main), 0)

    for t in range(pid, num_tiles, GRID):
        # ---- main loop over K (the next tile index is clamped so every path issues the same TDM ops)
        nt = gl.minimum(t + GRID, num_tiles - 1)
        c00 = gl.zeros((SM, SN), gl.float32, WL)
        c01 = gl.zeros((SM, SN), gl.float32, WL)
        c10 = gl.zeros((SM, SN), gl.float32, WL)
        c11 = gl.zeros((SM, SN), gl.float32, WL)
        c00, c01, c10, c11, x0, x1, y0, y1, da0, da1, db0, db1, npm, npn = _tile_k(
            c00,
            c01,
            c10,
            c11,
            x0,
            x1,
            y0,
            y1,
            da0,
            da1,
            db0,
            db1,
            a_ptr,
            b_ptr,
            nt,
            num_pid_m,
            M,
            N,
            K,
            stride_am,
            stride_bn,
            BM,
            BN,
            SL_A,
            SL_B,
            ab0,
            ab1,
            bb0,
            bb1,
            num_first,
            num_main,
            OPA,
            OPB,
            NUM_CTAS,
            AW,
        )

        # ---- epilogue: the final sign of the truncation-bias fix goes into the row scale
        _store_q(
            c00,
            -as0,
            bs0,
            c_ptr,
            pid_m * 2,
            pid_n * 2,
            M,
            N,
            stride_cm,
            1,
            SM,
            SN,
            NUM_CTAS,
            WL,
            STORE_LAYOUT,
        )
        _store_q(
            c01,
            -as0,
            bs1,
            c_ptr,
            pid_m * 2,
            pid_n * 2 + 1,
            M,
            N,
            stride_cm,
            1,
            SM,
            SN,
            NUM_CTAS,
            WL,
            STORE_LAYOUT,
        )
        _store_q(
            c10,
            -as1,
            bs0,
            c_ptr,
            pid_m * 2 + 1,
            pid_n * 2,
            M,
            N,
            stride_cm,
            1,
            SM,
            SN,
            NUM_CTAS,
            WL,
            STORE_LAYOUT,
        )
        _store_q(
            c11,
            -as1,
            bs1,
            c_ptr,
            pid_m * 2 + 1,
            pid_n * 2 + 1,
            M,
            N,
            stride_cm,
            1,
            SM,
            SN,
            NUM_CTAS,
            WL,
            STORE_LAYOUT,
        )
        pid_m = npm
        pid_n = npn
        as0, bs0 = _scales(
            a_scale_ptr, b_scale_ptr, pid_m * 2, pid_n * 2, M, N, SM, SN, WL
        )
        as1, bs1 = _scales(
            a_scale_ptr, b_scale_ptr, pid_m * 2 + 1, pid_n * 2 + 1, M, N, SM, SN, WL
        )
    gl.amd.gfx1250.tdm.async_wait(0)


def make_layouts(BM, BN, num_ctas, ctas_m, warps_n=2, num_warps=4, pad=128):
    SM, SN = BM // 2, BN // 2
    cm = ctas_m
    cn = num_ctas // cm
    cga = make_cga_layout([cm, cn], [cm, cn], [0, 1]) if num_ctas > 1 else []
    warp_bases = [(0, 1 << i) for i in range(int(math.log2(warps_n)))]
    warp_bases += [(1 << i, 0) for i in range(int(math.log2(num_warps // warps_n)))]
    wl = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        instr_shape=[16, 16, KH.value],
        cga_layout=cga,
    )
    opa = gl.DotOperandLayout(0, wl, k_width=16)
    opb = gl.DotOperandLayout(1, wl, k_width=16)
    sla = gl.PaddedSharedLayout.with_identity_for(
        [[pad, 16]], [SM, BK.value], [1, 0], opa.cga_layout
    )
    slb = gl.PaddedSharedLayout.with_identity_for(
        [[pad, 16]], [SN, BK.value], [1, 0], [[n, k] for k, n in opb.cga_layout]
    )
    cta_n = SN // cn
    vec = min(8, cta_n)
    threads_n = min(32, cta_n // vec)
    store = gl.BlockedLayout(
        [1, vec], [32 // threads_n, threads_n], [num_warps, 1], [1, 0], cga
    )
    return dict(SL_A=sla, SL_B=slb, WL=wl, OPA=opa, OPB=opb, STORE_LAYOUT=store)


def gemm_a8w8_k9p(x, w, x_scale, w_scale, y, cfg, num_cu=256):
    """x [M, K] fp8, w [N, K] fp8 (K tail pre-negated), x_scale [M, 1], w_scale [1, N] fp32, y [M, N] bf16.

    cfg: BLOCK_M, BLOCK_N (cluster tile), num_ctas, CTAS_M, AW (4 or 6), B2B (0/1).
    """
    M, K = x.shape
    N = w.shape[0]
    BM, BN, nc = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["num_ctas"]
    bk, nb = BK.value, NB.value
    assert (
        K % (2 * bk) == 0 and ((K // 2) // bk) % nb == 0
    ), "K/2 must be a multiple of NB * BK"
    assert y.stride(1) == 1 and x.stride(1) == 1 and w.stride(1) == 1
    lay = make_layouts(BM, BN, nc, cfg["CTAS_M"])
    tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    grid = min(tiles, num_cu // nc)
    return gemm_a8w8_k9p_kernel[(grid,)](
        x,
        w,
        y,
        x_scale,
        w_scale,
        M,
        N,
        K,
        x.stride(0),
        w.stride(0),
        y.stride(0),
        (K // 2) // bk,
        tiles,
        BM=BM,
        BN=BN,
        NUM_CTAS=nc,
        B2B=cfg.get("B2B", 1),
        AW=cfg.get("AW", 6),
        GRID=grid,
        num_warps=4,
        num_ctas=nc,
        **lay
    )


def launch(x, w, xs, ws, y, cfg, *unused):
    """Harness entry point (bench9 / verify9 / compile9)."""
    return gemm_a8w8_k9p(x, w, xs, ws, y, cfg)
