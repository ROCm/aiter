# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode-shaped native-MXFP8 GEMM on gfx1250 (MiniMax-M3 TP4 o_proj, N 6144 x K 2048).

out[M, N] (bf16) = A[M, K] (e4m3, row-major, + row-major e8m0 per 32) x W[N, K]^T
((16, 16)-preshuffled e4m3 + 32x4-shuffled e8m0): the operands and output of
gemm_a8w8_blockscale(isBpreshuffled=True).

On MI455X at decode token counts the GEMM is bound by the activation re-reads
(L2 -> CU) and by LDS, not by HBM, so:

* grid (ks, N / (nb * 16), ceil(M / tile_m)), one workgroup a CU;
* the weight goes straight from global into VGPRs in the WMMA operand order (its
  (16, 16) preshuffle: one wave load reads 512 contiguous bytes), ``wdepth`` K
  steps ahead; the activation tile either takes the same direct path (a lane
  loads row lane16, 16-byte chunk 2 jj + kgrp of the K step) or, with ``atdm``,
  goes to LDS by TDM in 512-byte K chunks (the production gfx1250 GEMMs'
  non-multicast descriptor, padded rows, pipeline_fence), read back as fragments;
* kw K-waves x wn n-waves, every wave holding all tile_m / 16 m-blocks, so no
  weight byte is loaded twice a workgroup;
* the K-waves are summed once, in wave order, through LDS;
* ks > 1: the ks workgroups of an output tile form one cluster (FP32 partials,
  one cluster barrier every workgroup reaches exactly once). A rank reduces
  tile_m / ks rows in split order; its own part of those rows stays in its LDS
  K-wave planes, so only the other ranks' rows (and only real rows) cross memory.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import tdm_ops_gfx1250 as tdm_ops
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import (
    make_lds_copy_ops,
    pipeline_fence,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

WAVE = 32
WMMA_K = 128
P_PAD = 16  # LDS row pad of a K-wave plane
CPOL_SE = 8  # partial slab: the ranks of a cluster share a shader engine


A_KC = 512  # K bytes of one activation TDM chunk (the pad interval must be <= 1024 B)
A_PAD = 16  # LDS row pad of an activation chunk (the production GEMMs' LDS_PAD_A)


def decode_gemm_lds_bytes(tile_m, nb, kw, K=0, ks=1, atdm=0):
    planes = kw * tile_m * (nb * 16 * 4 + P_PAD)
    return planes + (tile_m * (A_KC + A_PAD) * (K // ks // A_KC) if atdm else 0)


def decode_gemm_config_error(N, K, tile_m, nb, ks, kw, wn, wdepth, atdm=0):
    if K % (WMMA_K * ks * kw):
        return f"ks={ks} x kw={kw} must divide {K // WMMA_K} K steps"
    if N % (nb * 16):
        return f"nb={nb} must divide {N // 16} n-blocks"
    if nb % wn:
        return f"wn={wn} must divide nb={nb}"
    if tile_m % 16 or not 16 <= tile_m <= 256:
        return f"tile_m={tile_m}"
    if ks > 1 and (ks > 16 or tile_m % ks):
        return f"ks={ks}: a 2..16 cluster dividing tile_m"
    if kw * wn * WAVE > 1024:
        return "more than 32 waves"
    if atdm and ((K // ks) % A_KC or kw * wn < (K // ks) // A_KC):
        return f"TDM activation: whole {A_KC}-byte K chunks, one wave a chunk"
    if decode_gemm_lds_bytes(tile_m, nb, kw, K, ks, atdm) > 320 * 1024:
        return "LDS over 320 KB"
    if wdepth < 1:
        return "wdepth >= 1"
    return None


def _lds_barrier():
    """Workgroup barrier for LDS hand-offs (waits the LDS counter only)."""
    rocdl.s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


@flyc.jit
def launch_mxfp8_decode_gemm(
    arg_a: fx.Pointer,
    arg_as: fx.Pointer,
    arg_w: fx.Pointer,
    arg_ws: fx.Pointer,
    arg_out: fx.Pointer,
    arg_slab: fx.Pointer,
    i32_m: fx.Int32,
    i32_lda: fx.Int32,
    i32_mpad: fx.Int32,
    stream: fx.Stream,
    N: Constexpr[int],
    K: Constexpr[int],
    tile_m: Constexpr[int],
    nb: Constexpr[int],
    ks: Constexpr[int],
    kw: Constexpr[int],
    wn: Constexpr[int],
    wdepth: Constexpr[int],
    wcpol: Constexpr[int] = 0,
    atdm: Constexpr[int] = 0,
):
    # atdm: the activation tile goes global -> LDS by TDM in the production gfx1250
    # GEMMs' descriptor form (no multicast, pipeline_fence), and the waves read
    # their fragments from LDS; else straight from global into VGPRs.
    err = decode_gemm_config_error(N, K, tile_m, nb, ks, kw, wn, wdepth, atdm)
    if err is not None:
        raise ValueError(f"mxfp8_decode_gemm: {err}")
    steps = K // WMMA_K // ks  # K steps of a workgroup
    steps_w = steps // kw  # K steps of a wave
    cols = nb * 16
    mbt = tile_m // 16
    nbw = nb // wn
    block = kw * wn * WAVE
    wdepth = max(1, min(wdepth, steps_w))
    sa_opsel = mbt % 2 == 0  # one scale dword serves an m-block pair
    sa_loads = mbt // 2 if sa_opsel else mbt
    KB4 = K // 128  # activation-scale dwords of a row
    lds_bytes = decode_gemm_lds_bytes(tile_m, nb, kw, K, ks, atdm)
    PLANES = kw * tile_m * (nb * 16 * 4 + P_PAD)
    A_ROW = A_KC + A_PAD
    A_STAGE = tile_m * A_ROW
    n_chunks = (K // ks) // A_KC
    P_ROW = cols * 4 + P_PAD
    P_PLANE = tile_m * P_ROW
    U_ROW = cols // 4  # float4s of a tile row
    rpr = tile_m // ks  # rows a cluster rank reduces (ks > 1)
    kernel_name = (
        f"mxfp8_decode_gemm_gfx1250_n{N}_k{K}_t{tile_m}x{cols}_ks{ks}_w{kw}x{wn}_d{wdepth}"
        f"{f'_c{wcpol}' if wcpol else ''}{'_at' if atdm else ''}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_decode_gemm(
        arg_a: fx.Pointer,
        arg_as: fx.Pointer,
        arg_w: fx.Pointer,
        arg_ws: fx.Pointer,
        arg_out: fx.Pointer,
        arg_slab: fx.Pointer,
        i32_m: fx.Int32,
        i32_lda: fx.Int32,
        i32_mpad: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx
        kpart = fx.Int32(bid_x)  # cluster rank: the K split
        ntile = fx.Int32(bid_y)
        mgrp = fx.Int32(bid_z)
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        w_k = wave // wn
        w_n = wave % wn
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        m0 = mgrp * tile_m
        ks0 = kpart * steps + w_k * steps_w  # first global K step of the wave

        arena = fx.SharedAllocator(static=False)
        arena.allocate(lds_bytes)
        base_ptr = arena.base_ptr
        lds_idx = fx.Int64(fx.ptrtoint(base_ptr))
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)

        if const_expr(atdm):
            # Wave c < n_chunks moves activation chunk c ([tile_m, 512 B] of the
            # workgroup's K range) into LDS rows of 512 + 16 B; rows past M are
            # zero-filled, not fetched.
            gA_base = fx.recast_iter(fx.Int8, arg_a)
            if wave < n_chunks:
                c_off = fx.Int64(m0) * fx.Int64(i32_lda) + fx.Int64(
                    kpart * (K // ks) + wave * A_KC
                )
                desc = tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=fx.Tensor(
                        fx.make_view(
                            fx.add_offset(gA_base, c_off),
                            fx.make_layout((tile_m, A_KC), (A_KC, 1)),
                        )
                    ),
                    lds_memref=fx.Tensor(
                        fx.make_view(
                            fx.add_offset(base_ptr, PLANES + wave * A_STAGE),
                            fx.make_layout((tile_m, A_KC), (A_ROW, 1)),
                        )
                    ),
                    global_offset=(0, 0),
                    tensor_shape=(tile_m, A_KC),
                    strides=(i32_lda, 1),
                    tile_shape=(tile_m, A_KC),
                    elem_bytes=1,
                    pad_interval=A_KC,
                    pad_amount=A_PAD,
                    num_warps=1,
                    workgroup_mask=0,
                    oob_outer_bound=i32_m - m0,
                )
                tdm_ops.tensor_load_2d(desc)

        # Activation rows past M read zeros (buffer range); their outputs are dropped.
        a_buf = ptr_buf_tensor(
            arg_a, fx.Int32, unit_elems=4, num_records_bytes=i32_m * i32_lda
        )
        sa_buf = ptr_buf_tensor(arg_as, fx.Int32, num_records_bytes=i32_m * (K // 32))
        b_buf = ptr_buf_tensor(arg_w, fx.Int32, unit_elems=4)
        sb_buf = ptr_buf_tensor(arg_ws, fx.Int32)
        lda16 = i32_lda // 16
        a_row = [(m0 + mi * 16 + lane16) * lda16 + kgrp for mi in range_constexpr(mbt)]
        nblk0 = ntile * nb + w_n * nbw
        b_base = [(nblk0 + j) * K + lane for j in range_constexpr(nbw)]
        sb_col = [(nblk0 + j) * 16 + lane16 for j in range_constexpr(nbw)]
        sb_base = [(c // 32) * (K // 4) + c % 32 for c in sb_col]
        sa_row = [
            (
                (m0 + (2 * p) * 16 + lane)
                if const_expr(sa_opsel)
                else (m0 + p * 16 + lane16)
            )
            for p in range_constexpr(sa_loads)
        ]
        sa_base = [r * KB4 for r in sa_row]

        def _cat16(v):
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def load_step(i):
            ksg = ks0 + i
            bfr = [
                [
                    buf_copy_load(
                        b_buf,
                        b_base[j] + ksg * 128 + jj * 32,
                        fx.Int32,
                        4,
                        cache_modifier=wcpol,
                    )
                    for jj in range_constexpr(4)
                ]
                for j in range_constexpr(nbw)
            ]
            sb = [
                buf_copy_load(sb_buf, sb_base[j] + ksg * 32, fx.Int32)
                for j in range_constexpr(nbw)
            ]
            sa = [
                buf_copy_load(sa_buf, sa_base[p] + ksg, fx.Int32)
                for p in range_constexpr(sa_loads)
            ]
            afr = None
            if const_expr(not atdm):
                afr = [
                    [
                        buf_copy_load(a_buf, a_row[mi] + ksg * 8 + jj * 2, fx.Int32, 4)
                        for jj in range_constexpr(4)
                    ]
                    for mi in range_constexpr(mbt)
                ]
            return bfr, sb, sa, afr

        def lds_a_frag(mi, i):
            s_ = w_k * steps_w + i  # K step inside the workgroup range
            base = (
                PLANES
                + (s_ // 4) * A_STAGE
                + (mi * 16 + lane16) * A_ROW
                + (s_ % 4) * 128
                + kgrp * 16
            )
            return [
                Vec(lds_load_b128(lds_idx, base + 32 * jj)) for jj in range_constexpr(4)
            ]

        atoms = [
            fx.make_mma_atom(
                fx.rocdl.WMMAScale(
                    16,
                    16,
                    WMMA_K,
                    fx.Float8E4M3FN,
                    fx.Float8E4M3FN,
                    fx.Float32,
                    opsel_a=0,
                    opsel_b=sel,
                )
            )
            for sel in range_constexpr(2 if sa_opsel else 1)
        ]
        acc = [
            [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(nbw)]
            for _ in range_constexpr(mbt)
        ]
        for row in acc:
            for c in row:
                c.store(fx.full(8, 0.0, fx.Float32))

        def _rmem(v):
            t = fx.make_rmem_tensor(16, fx.Int32)
            t.store(v)
            return t

        def compute_step(st, i):
            bfr, sb, sa, afr = st
            if const_expr(atdm):
                afr = [lds_a_frag(mi, i) for mi in range_constexpr(mbt)]
            bt = [_rmem(_cat16(bfr[j])) for j in range_constexpr(nbw)]
            at = [_rmem(_cat16(afr[mi])) for mi in range_constexpr(mbt)]
            for mi in range_constexpr(mbt):
                for j in range_constexpr(nbw):
                    fx.gemm(
                        atoms[mi % 2 if sa_opsel else 0],
                        acc[mi][j],
                        bt[j],
                        at[mi],
                        acc[mi][j],
                        scale_a=sb[j],
                        scale_b=sa[mi // 2 if sa_opsel else mi],
                    )

        inflight = [load_step(i) for i in range_constexpr(wdepth)]
        rocdl.sched_barrier(0)
        if const_expr(atdm):
            pipeline_fence(
                outstanding=0, use_cluster=False
            )  # the activation tile is in LDS
        for i in range_constexpr(steps_w):
            st = inflight.pop(0)
            if const_expr(i + wdepth < steps_w):
                inflight.append(load_step(i + wdepth))
                rocdl.sched_barrier(0)
            compute_step(st, i)

        # ---- the K-waves' accumulators as LDS planes, summed in wave order ----
        for mi in range_constexpr(mbt):
            for j in range_constexpr(nbw):
                row = mi * 16 + lane16
                col = (w_n * nbw + j) * 16 + kgrp * 8
                v = Vec(acc[mi][j].load())
                off = w_k * P_PLANE + row * P_ROW + col * 4
                lds_store_b128(
                    lds_idx, off, v.shuffle(v, [0, 1, 2, 3]).bitcast(fx.Int32)
                )
                lds_store_b128(
                    lds_idx, off + 16, v.shuffle(v, [4, 5, 6, 7]).bitcast(fx.Int32)
                )
        _lds_barrier()

        def plane_sum(r, c4):
            tot = None
            for k in range_constexpr(kw):
                part = Vec(
                    lds_load_b128(lds_idx, k * P_PLANE + r * P_ROW + c4 * 16)
                ).bitcast(fx.Float32)
                tot = part if tot is None else tot + part
            return tot

        n0 = ntile * cols
        out_buf = ptr_buf_tensor(
            arg_out, fx.Int32, unit_elems=2, num_records_bytes=i32_m * (N * 2)
        )
        p_units = tile_m * U_ROW
        if const_expr(ks == 1):
            for t in range_constexpr((p_units + block - 1) // block):
                u = tid + t * block
                r = u // U_ROW
                c4 = u % U_ROW
                yw = plane_sum(r, c4).to(fx.BFloat16).bitcast(fx.Int32)
                ok = (m0 + r) < i32_m
                if const_expr(p_units % block):
                    ok = ok & (u < p_units)
                if ok:
                    buf_copy_store(
                        out_buf, ((m0 + r) * N + n0) // 4 + c4, yw, fx.Int32, 2
                    )
        else:
            slab_buf = ptr_buf_tensor(
                arg_slab,
                fx.Float32,
                unit_elems=4,
                num_records_bytes=fx.Int64(i32_mpad) * (ks * N * 4),
            )
            # the other ranks' rows out to the slab (real rows only)
            for t in range_constexpr((p_units + block - 1) // block):
                u = tid + t * block
                r = u // U_ROW
                c4 = u % U_ROW
                tot = plane_sum(r, c4)
                ok = ((r // rpr) != kpart) & ((m0 + r) < i32_m)
                if const_expr(p_units % block):
                    ok = ok & (u < p_units)
                if ok:
                    buf_copy_store(
                        slab_buf,
                        ((kpart * i32_mpad + m0 + r) * N + n0) // 4 + c4,
                        tot,
                        fx.Float32,
                        4,
                        cache_modifier=CPOL_SE,
                    )
            rocdl.s_wait_storecnt(0)
            cluster.cluster_barrier()
            r_units = rpr * U_ROW
            for t in range_constexpr((r_units + block - 1) // block):
                u = tid + t * block
                r = kpart * rpr + u // U_ROW
                c4 = u % U_ROW
                own = plane_sum(r, c4)
                parts = []
                for k in range_constexpr(ks):
                    gi = ((k * i32_mpad + m0 + r) * N + n0) // 4 + c4
                    gi = (kpart == k).select(
                        fx.Int32(0x0FFFFFF0), gi
                    )  # own part: out of range, no fetch
                    parts.append(
                        Vec(
                            buf_copy_load(
                                slab_buf, gi, fx.Float32, 4, cache_modifier=CPOL_SE
                            )
                        )
                    )
                tot = None
                for k in range_constexpr(ks):
                    pk = (kpart == k).select(own, parts[k])
                    tot = pk if tot is None else tot + pk
                yw = tot.to(fx.BFloat16).bitcast(fx.Int32)
                ok = (m0 + r) < i32_m
                if const_expr(r_units % block):
                    ok = ok & (u < r_units)
                if ok:
                    buf_copy_store(
                        out_buf, ((m0 + r) * N + n0) // 4 + c4, yw, fx.Int32, 2
                    )

    mgroups = (i32_m + (tile_m - 1)) // tile_m
    kernel_decode_gemm(
        arg_a,
        arg_as,
        arg_w,
        arg_ws,
        arg_out,
        arg_slab,
        i32_m,
        i32_lda,
        i32_mpad,
        value_attrs={"rocdl.cluster_dims": (f"{ks},1,1" if ks > 1 else None)},
    ).launch(
        grid=(ks, N // cols, mgroups),
        block=(block, 1, 1),
        stream=stream,
        cluster=(ks, 1, 1) if ks > 1 else None,
    )


launch_mxfp8_decode_gemm.compile_hints["llvm_options"] = {
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
