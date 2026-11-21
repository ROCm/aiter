# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 TP4 decode QKV chain in one gfx1250 launch.

The native-MXFP8 QKV GEMM (rows [q | k | v | index-q | index-k], 128-wide heads,
K 6144) with the Gemma q/k RMS norm, partial NeoX RoPE (rotary 64), bf16 q /
index-q and the NVFP4 K / V / index-K row inserts fused into its tail.

Grid (ns * ks, heads, ceil(M / tile_m)); the ns * ks workgroups of one (head, row
tile) are one hardware cluster (a cluster sits inside one 16-CU shader engine, so
ns * ks is 4, 8 or 16). Rank r owns columns (r % ns) * 128 / ns of the head and K
range r // ns of ks. Each workgroup

* stages its activation slab [tile_m, K / ks] in LDS (row loads, one wave a
  contiguous run of a row, written in the (16, 16) WMMA fragment order with an XOR
  row swizzle: conflict-free on both sides),
* streams its weight slab [128 / ns, K / ks] straight into VGPRs (the (16, 16)
  preshuffled weight: one wave load reads 512 contiguous bytes); kw x wm x wn
  waves, kw of them splitting the workgroup's K range,
* sums its K-waves in wave order through LDS and stores the FP32 partial tile
  row-contiguously to the slab [ks, Mpad, N] (device scope), then the cluster
  barrier,
* reduces rows [r * rpr, (r + 1) * rpr) of the tile over the ks partial planes in
  split order (FP32, one round to bf16, as the gfx1250 split-K reduce and
  fused_qknorm_idxrqknorm(qkv_partials) do) and runs the head's epilogue, one row a
  wave, 4 dims a lane: the lane map and arithmetic of aiter's
  fused_qknorm_idxrqknorm, whose bytes it reproduces.

The cluster barrier is the only cross-workgroup wait: every workgroup of a cluster
reaches it exactly once (no early exit) and grid x is the cluster size.

Layouts: A [M, K] e4m3 row-major (row stride a multiple of 16 bytes), A scale
[M, K / 32] e8m0 row-major, W [N, K] e4m3 (16, 16) preshuffled, W scale [N, K / 32]
e8m0 32x4 shuffled. K / V caches: NVFP4 row layout, payload [pages, 1, 128, 64] u8
+ E4M3 scale planes [pages, 1, 128, 8]; index cache payload [pages, 128, 64] +
scales [pages, 128, 8]. One KV head (TP4); the head strides are accepted for the
host contract and unused.

Measured on gfx1250 (MI455X, 256 CUs as 16 shader engines of 16): a CU reads
~150 GB/s, global writes cost ~4x reads past ~8 MB, and a cluster stays inside a
shader engine. Hence ns * ks = 8 (two clusters a shader engine, 160 CUs at 20 heads)
and a small ks (the FP32 partial planes are ks * M * N * 4 bytes).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

WAVE = 32
WMMA_K = 128
D = 128
K_DIM = 6144
NQ = 16
NKV = 1
ROT = 64
CPOL_DEV = 16  # device scope: the ranks of a cluster sit on different CUs
A_BATCH = 32  # 16-byte activation units a thread has in flight while staging
P_PAD = 16  # LDS row pad of the partial tile


def chain_heads(niq: int) -> int:
    return NQ + 2 * NKV + niq + 1


def chain_config_error(tile_m, ns, ks, kw, wm, wn, depth):
    steps_total = K_DIM // WMMA_K
    if ns not in (1, 2, 4, 8) or ns * ks > 16 or ns * ks < 2:
        return f"ns={ns} x ks={ks} must be a 2..16 cluster"
    if steps_total % (ks * kw):
        return f"ks={ks} x kw={kw} must divide {steps_total} K steps"
    if tile_m % 16 or not 16 <= tile_m <= 256:
        return f"tile_m={tile_m}"
    if (tile_m // 16) % wm or (8 // ns) % wn:
        return (
            f"wm={wm} / wn={wn} do not divide the {tile_m // 16}x{8 // ns} block grid"
        )
    steps = steps_total // ks
    block = kw * wm * wn * WAVE
    if block > 1024:
        return "more than 32 waves"
    if (tile_m * steps * 8) % block:
        return "staged activation does not divide over the block"
    if depth < 1:
        return "depth >= 1"
    if chain_lds_bytes(tile_m, ns, ks, kw) > 320 * 1024:
        return "LDS over 320 KB"
    return None


def chain_lds_bytes(tile_m, ns, ks, kw):
    a = tile_m * (K_DIM // ks)
    p = kw * tile_m * (D // ns * 4 + P_PAD)
    return max(a, p)


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _bf16x4(v):
    """4 fp32 of a dword pair of bf16 (low half first)."""
    w0, w1 = fx.Int32(v[0]), fx.Int32(v[1])
    return [
        (w0 << 16).bitcast(fx.Float32),
        (w0 & -65536).bitcast(fx.Float32),
        (w1 << 16).bitcast(fx.Float32),
        (w1 & -65536).bitcast(fx.Float32),
    ]


def _xor(x, m):
    return fx.Float32(x).shuffle_xor(fx.Int32(m), fx.Int32(WAVE))


def _rsq(x):
    """v_rsq_f32, as ocml's rsqrtf in a flush-denormal module (x >= eps >= 2^-126)."""
    return fx.Float32(_rocdl_ops.rsq(T.f32, fx.as_ir_value(x)))


def _e4m3_byte(x):
    """e4m3fn byte of x in [0, 448], round to nearest even (aiter nvE4m3Byte)."""
    bits = _bits(x)
    normal = ((bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20) - 960
    # m * 2^-9 below 2^-6, m = RNE(x * 512): the 1.5 * 2^23 trick, read off the bits.
    sub = _bits(fx.math.fma(x, fx.Float32(512.0), fx.Float32(12582912.0))) - 0x4B400000
    return (x >= fx.Float32(0.015625)).select(normal, sub)


def _e4m3_value(byte):
    e = byte >> 3
    m = byte & 7
    normal = (((e + 120) << 23) | (m << 20)).bitcast(fx.Float32)
    sub = fx.Float32(m.to(fx.Float32)) * fx.Float32(0.001953125)
    return (e > 0).select(normal, sub)


def _nvfp4_lane(v, sign_of_scaled):
    """aiter nvRowStore: this lane's 4 values (model-dtype rounded) -> 16-bit codes
    and the e4m3 byte of its 16-element group (lanes l ^ 1, l ^ 2). A code's sign
    comes from x * (1 / scale) where ``sign_of_scaled`` (an i1) holds, else x."""
    a = [fx.math.absf(x) for x in v]
    amax = a[0]
    for i in range_constexpr(1, 4):
        amax = amax.maximumf(a[i])
    amax = amax.maximumf(_xor(amax, 1))
    amax = amax.maximumf(_xor(amax, 2))
    sf = (amax / fx.Float32(6.0)).minimumf(fx.Float32(448.0))
    sbyte = _e4m3_byte(sf)
    sv = _e4m3_value(sbyte)
    inv = (sv > fx.Float32(0.0)).select(fx.Float32(1.0) / sv, fx.Float32(0.0))
    packed = fx.Int32(0)
    for i in range_constexpr(4):
        y = v[i] * inv
        ay = fx.math.absf(y)
        mag = fx.Int32(0)
        for thr, ge in (
            (0.25, False), (0.75, True), (1.25, False), (1.75, True),
            (2.5, False), (3.5, True), (5.0, False),
        ):  # fmt: skip
            hit = (ay >= fx.Float32(thr)) if ge else (ay > fx.Float32(thr))
            mag = mag + hit.select(fx.Int32(1), fx.Int32(0))
        neg = sign_of_scaled.select(y < fx.Float32(0.0), v[i] < fx.Float32(0.0))
        code = mag | neg.select(fx.Int32(8), fx.Int32(0))
        packed = packed | (code << (4 * i))
    return packed, sbyte


def _lds_barrier():
    """Workgroup barrier for LDS hand-offs only: waits the LDS counter, not the
    global loads in flight (gpu.barrier's fence would wait for them all)."""
    rocdl.s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _gptr(addr, ty, align):
    return fx.inttoptr(
        fx.PointerType.get(ty, address_space=fx.AddressSpace.Global, alignment=align),
        fx.Int64(addr),
    )


@flyc.jit
def launch_mxfp8_qkv_norm_rope_nvfp4_insert(
    arg_a: fx.Pointer,
    arg_as: fx.Pointer,
    arg_w: fx.Pointer,
    arg_ws: fx.Pointer,
    arg_slab: fx.Pointer,
    arg_q: fx.Pointer,
    arg_iq: fx.Pointer,
    arg_qkv: fx.Pointer,
    arg_qw: fx.Pointer,
    arg_kw: fx.Pointer,
    arg_iqw: fx.Pointer,
    arg_ikw: fx.Pointer,
    arg_cs: fx.Pointer,
    arg_pos: fx.Pointer,
    arg_slot: fx.Pointer,
    arg_islot: fx.Pointer,
    arg_kp: fx.Pointer,
    arg_vp: fx.Pointer,
    arg_ksc: fx.Pointer,
    arg_vsc: fx.Pointer,
    arg_ip: fx.Pointer,
    arg_isc: fx.Pointer,
    i32_m: fx.Int32,
    i32_lda: fx.Int32,
    i32_mpad: fx.Int32,
    i32_cs_rows: fx.Int32,
    i64_kp_page: fx.Int64,
    i64_kp_head: fx.Int64,
    i64_ks_page: fx.Int64,
    i64_ks_head: fx.Int64,
    i64_ip_page: fx.Int64,
    i64_is_page: fx.Int64,
    i32_eps: fx.Int32,
    stream: fx.Stream,
    tile_m: Constexpr[int],
    ns: Constexpr[int],
    ks: Constexpr[int],
    kw: Constexpr[int],
    wm: Constexpr[int],
    wn: Constexpr[int],
    depth: Constexpr[int],
    niq: Constexpr[int],
    write_qkv: Constexpr[bool],
):
    err = chain_config_error(tile_m, ns, ks, kw, wm, wn, depth)
    if err is not None:
        raise ValueError(f"qkv chain: {err}")
    K = K_DIM
    HEADS = chain_heads(niq)
    N = HEADS * D
    csize = ns * ks
    steps = K // WMMA_K // ks  # K steps of a workgroup
    steps_w = steps // kw  # K steps of a wave
    Kr = steps * WMMA_K
    CPR = Kr // 16  # 16-byte activation units of a staged row
    cols = D // ns  # columns of a workgroup
    nbt = cols // 16  # n-blocks of a workgroup
    mbt = tile_m // 16
    mbw = mbt // wm  # m-blocks of a wave
    nbw = nbt // wn  # n-blocks of a wave
    waves = kw * wm * wn
    block = waves * WAVE
    depth = max(1, min(depth, steps_w))
    sa_opsel = mbw % 2 == 0
    sa_loads = mbw // 2 if sa_opsel else mbw
    KB4 = K // 128  # activation-scale dwords of a row
    per_thr = tile_m * CPR // block
    lds_bytes = chain_lds_bytes(tile_m, ns, ks, kw)
    P_ROW = cols * 4 + P_PAD  # LDS pitch of a partial row
    P_PLANE = tile_m * P_ROW
    p_units = tile_m * cols // 4  # float4s of a partial tile
    U_ROW = cols // 4
    rpr = (tile_m + csize - 1) // csize  # rows a cluster rank reduces
    row_iters = (rpr + waves - 1) // waves
    IQ0 = NQ + 2 * NKV  # first index-q head
    IK = IQ0 + niq  # the index-k head
    cpol = CPOL_DEV
    kernel_name = (
        f"mxfp8_qkv_norm_rope_nvfp4_insert_gfx1250_n{N}_t{tile_m}_ns{ns}_ks{ks}"
        f"_w{kw}x{wm}x{wn}_d{depth}{'_wq' if write_qkv else ''}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_chain(
        arg_a: fx.Pointer,
        arg_as: fx.Pointer,
        arg_w: fx.Pointer,
        arg_ws: fx.Pointer,
        arg_slab: fx.Pointer,
        arg_q: fx.Pointer,
        arg_iq: fx.Pointer,
        arg_qkv: fx.Pointer,
        arg_qw: fx.Pointer,
        arg_kw: fx.Pointer,
        arg_iqw: fx.Pointer,
        arg_ikw: fx.Pointer,
        arg_cs: fx.Pointer,
        arg_pos: fx.Pointer,
        arg_slot: fx.Pointer,
        arg_islot: fx.Pointer,
        arg_kp: fx.Pointer,
        arg_vp: fx.Pointer,
        arg_ksc: fx.Pointer,
        arg_vsc: fx.Pointer,
        arg_ip: fx.Pointer,
        arg_isc: fx.Pointer,
        i32_m: fx.Int32,
        i32_lda: fx.Int32,
        i32_mpad: fx.Int32,
        i32_cs_rows: fx.Int32,
        i64_kp_page: fx.Int64,
        i64_kp_head: fx.Int64,
        i64_ks_page: fx.Int64,
        i64_ks_head: fx.Int64,
        i64_ip_page: fx.Int64,
        i64_is_page: fx.Int64,
        i32_eps: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx
        rank = fx.Int32(bid_x)  # cluster rank
        head = fx.Int32(bid_y)
        mgrp = fx.Int32(bid_z)
        npart = rank % ns
        kpart = rank // ns
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        w_k = wave // (wm * wn)
        w_m = (wave // wn) % wm
        w_n = wave % wn
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        m0 = mgrp * tile_m
        ks0 = kpart * steps  # first global K step of the workgroup

        arena = fx.SharedAllocator(static=False)
        arena.allocate(lds_bytes)
        lds_idx = fx.Int64(fx.ptrtoint(arena.base_ptr))
        lds_f32 = fx.recast_iter(fx.Float32, arena.base_ptr)  # noqa: F841
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)

        # ---- epilogue scalars first: they only wait at the tail ----
        r_base = rank * rpr
        e_rows = [r_base + wave + t * waves for t in range_constexpr(row_iters)]
        e_valid = [
            ((wave + t * waves) < rpr) & (r < tile_m) & (m0 + r < i32_m)
            for t, r in enumerate(e_rows)
        ]
        # int64 tables: the low dword (positions and slots fit 31 bits).
        pos_buf = ptr_buf_tensor(arg_pos, fx.Int32, num_records_bytes=i32_m * 8)
        slot_buf = ptr_buf_tensor(arg_slot, fx.Int32, num_records_bytes=i32_m * 8)
        islot_buf = ptr_buf_tensor(arg_islot, fx.Int32, num_records_bytes=i32_m * 8)
        e_pos = [buf_copy_load(pos_buf, (m0 + r) * 2, fx.Int32) for r in e_rows]
        e_slot = [buf_copy_load(slot_buf, (m0 + r) * 2, fx.Int32) for r in e_rows]
        e_islot = [buf_copy_load(islot_buf, (m0 + r) * 2, fx.Int32) for r in e_rows]

        # ---- GEMM operands ----
        a_buf = ptr_buf_tensor(
            arg_a, fx.Int32, unit_elems=4, num_records_bytes=i32_m * i32_lda
        )
        lda16 = i32_lda // 16
        sa_buf = ptr_buf_tensor(arg_as, fx.Int32, num_records_bytes=i32_m * (K // 32))
        b_buf = ptr_buf_tensor(arg_w, fx.Int32, unit_elems=4)
        sb_buf = ptr_buf_tensor(arg_ws, fx.Int32)

        nblk0 = head * 8 + npart * nbt + w_n * nbw
        b_base = [(nblk0 + j) * K + lane for j in range_constexpr(nbw)]
        sb_col = [(nblk0 + j) * 16 + lane16 for j in range_constexpr(nbw)]
        sb_base = [(c // 32) * (K // 4) + c % 32 for c in sb_col]
        mblk0 = w_m * mbw  # first m-block of the wave inside the tile
        sa_row = [
            (
                (m0 + (mblk0 + 2 * p) * 16 + lane)
                if const_expr(sa_opsel)
                else (m0 + (mblk0 + p) * 16 + lane16)
            )
            for p in range_constexpr(sa_loads)
        ]
        sa_base = [r * KB4 for r in sa_row]
        st0 = w_k * steps_w  # first step of the wave inside the workgroup range

        def _cat16(v):
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def load_b_step(i):
            ksg = ks0 + st0 + i
            bfr = [
                [
                    buf_copy_load(b_buf, b_base[j] + ksg * 128 + jj * 32, fx.Int32, 4)
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
            return [bfr, sb, sa]

        # Activation staging: unit u = (row u // CPR, 16-byte chunk u % CPR of the
        # workgroup's K range) lands at its fragment-order, row-swizzled LDS slot.
        def a_load(u):
            return buf_copy_load(
                a_buf, (m0 + u // CPR) * lda16 + ks0 * 8 + u % CPR, fx.Int32, 4
            )

        def a_lds_off(u):
            r = u // CPR
            c = u % CPR
            return (
                ((r // 16) * steps + c // 8) * 2048
                + (c % 8) * 256
                + ((r % 16) ^ (c % 16)) * 16
            )

        # Fragment read offsets: lane (row lane16, chunk 2 jj + kgrp) of a step of
        # parity par.
        frag_lane = [
            [
                (jj * 2 + kgrp) * 256 + (lane16 ^ (kgrp + (par * 8 + jj * 2))) * 16
                for jj in range_constexpr(4)
            ]
            for par in range_constexpr(2)
        ]

        def load_a_frag(mi, i):
            s = st0 + i  # step inside the workgroup range
            base = ((mblk0 + mi) * steps + s) * 2048
            par = (w_k * steps_w + i) % 2 if const_expr(steps_w % 2) else i % 2
            if const_expr(steps_w % 2):
                return [
                    Vec(
                        lds_load_b128(
                            lds_idx,
                            base
                            + (jj * 2 + kgrp) * 256
                            + (lane16 ^ (kgrp + (par * 8 + jj * 2))) * 16,
                        )
                    )
                    for jj in range_constexpr(4)
                ]
            return [
                Vec(lds_load_b128(lds_idx, base + frag_lane[par][jj]))
                for jj in range_constexpr(4)
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
            for _ in range_constexpr(mbw)
        ]
        for row in acc:
            for c in row:
                c.store(fx.full(8, 0.0, fx.Float32))

        def _rmem(v):
            t = fx.make_rmem_tensor(16, fx.Int32)
            t.store(v)
            return t

        def compute_step(st, afr):
            bfr, sb, sa = st
            bt = [_rmem(_cat16(bfr[j])) for j in range_constexpr(nbw)]
            at = [_rmem(_cat16(afr[mi])) for mi in range_constexpr(mbw)]
            for mi in range_constexpr(mbw):
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

        # Issue order: the activation (in batches of A_BATCH units a thread), the
        # weight window; every load of the first batch and the window is in flight
        # before the first wait.
        batches = [
            [
                tid + (b0 + v) * block
                for v in range_constexpr(min(A_BATCH, per_thr - b0))
            ]
            for b0 in range_constexpr(0, per_thr, A_BATCH)
        ]
        a_vals0 = [a_load(u) for u in batches[0]]
        rocdl.sched_barrier(0)
        inflight = [load_b_step(i) for i in range_constexpr(depth)]
        rocdl.sched_barrier(0)
        for u, val in zip(batches[0], a_vals0):
            lds_store_b128(lds_idx, a_lds_off(u), val)
        for units in batches[1:]:
            vals = [a_load(u) for u in units]
            for u, val in zip(units, vals):
                lds_store_b128(lds_idx, a_lds_off(u), val)
        _lds_barrier()
        rocdl.sched_barrier(0)
        for i in range_constexpr(steps_w):
            st = inflight.pop(0)
            if const_expr(i + depth < steps_w):
                inflight.append(load_b_step(i + depth))
                rocdl.sched_barrier(0)
            afr = [load_a_frag(mi, i) for mi in range_constexpr(mbw)]
            compute_step(st, afr)

        # ---- epilogue inputs that are not partials: in flight across the barrier ----
        is_q = head < NQ
        is_k = head == NQ
        is_v = head == NQ + NKV
        is_iq = (head >= IQ0) & (head < IK)
        is_ik = head == IK
        hl = lane
        dim0 = hl * 4
        first = hl < (ROT // 8)  # dims < rotary / 2
        in_rope = hl < (ROT // 4)  # dims < rotary
        ib = first.select(dim0, dim0 - ROT // 2)
        ib = in_rope.select(ib, fx.Int32(0))
        w_addr = is_q.select(
            fx.Int64(fx.ptrtoint(arg_qw)),
            is_k.select(
                fx.Int64(fx.ptrtoint(arg_kw)),
                is_ik.select(
                    fx.Int64(fx.ptrtoint(arg_ikw)), fx.Int64(fx.ptrtoint(arg_iqw))
                ),
            ),
        )
        nw_buf = ptr_buf_tensor(w_addr, fx.Int32, unit_elems=2, num_records_bytes=D * 2)
        nw = Vec(buf_copy_load(nw_buf, hl, fx.Int32, 2))
        # ---- K-waves summed in wave order through LDS; the partial tile goes out
        # row-contiguously ----
        slab_buf = ptr_buf_tensor(
            arg_slab,
            fx.Float32,
            unit_elems=4,
            num_records_bytes=fx.Int64(i32_mpad) * (ks * N * 4),
        )
        _lds_barrier()  # every wave is done with the activation
        for mi in range_constexpr(mbw):
            for j in range_constexpr(nbw):
                row = (mblk0 + mi) * 16 + lane16
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
        for t in range_constexpr((p_units + block - 1) // block):
            u = tid + t * block
            r = u // U_ROW
            c4 = u % U_ROW
            tot = None
            for k in range_constexpr(kw):
                part = Vec(
                    lds_load_b128(lds_idx, k * P_PLANE + r * P_ROW + c4 * 16)
                ).bitcast(fx.Float32)
                tot = part if tot is None else tot + part
            g = ((kpart * i32_mpad + m0 + r) * N + head * D + npart * cols) // 4 + c4
            if const_expr(p_units % block == 0):
                buf_copy_store(slab_buf, g, tot, fx.Float32, 4, cache_modifier=cpol)
            else:
                if u < p_units:
                    buf_copy_store(slab_buf, g, tot, fx.Float32, 4, cache_modifier=cpol)
        cs_buf = ptr_buf_tensor(
            arg_cs, fx.Int32, unit_elems=2, num_records_bytes=i32_cs_rows * (ROT * 2)
        )
        e_cos = []
        e_sin = []
        for t in range_constexpr(row_iters):
            off = e_pos[t] * (ROT // 4) + ib // 4  # units of 4 bf16
            e_cos.append(Vec(buf_copy_load(cs_buf, off, fx.Int32, 2)))
            e_sin.append(Vec(buf_copy_load(cs_buf, off + ROT // 8, fx.Int32, 2)))

        rocdl.s_wait_storecnt(0)
        cluster.cluster_barrier()

        # ---- reduce this rank's rows in split order, then the head's epilogue ----
        eps = i32_eps.bitcast(fx.Float32)
        parts = []
        for t in range_constexpr(row_iters):
            row = m0 + e_rows[t]
            parts.append(
                [
                    buf_copy_load(
                        slab_buf,
                        ((k * i32_mpad + row) * N + head * D) // 4 + hl,
                        fx.Float32,
                        4,
                        cache_modifier=cpol,
                    )
                    for k in range_constexpr(ks)
                ]
            )
        q_buf = ptr_buf_tensor(
            arg_q, fx.Int32, unit_elems=2, num_records_bytes=i32_m * (NQ * D * 2)
        )
        iq_buf = ptr_buf_tensor(
            arg_iq, fx.Int32, unit_elems=2, num_records_bytes=i32_m * (niq * D * 2)
        )
        qkv_buf = ptr_buf_tensor(
            arg_qkv, fx.Int32, unit_elems=2, num_records_bytes=i32_m * (N * 2)
        )
        w = _bf16x4(nw)
        kp_base = fx.Int64(fx.ptrtoint(arg_kp))
        vp_base = fx.Int64(fx.ptrtoint(arg_vp))
        ksc_base = fx.Int64(fx.ptrtoint(arg_ksc))
        vsc_base = fx.Int64(fx.ptrtoint(arg_vsc))
        ip_base = fx.Int64(fx.ptrtoint(arg_ip))
        isc_base = fx.Int64(fx.ptrtoint(arg_isc))
        for t in range_constexpr(row_iters):
            row = m0 + e_rows[t]
            valid = e_valid[t]
            acc4 = parts[t][0]
            for k in range_constexpr(1, ks):
                acc4 = acc4 + parts[t][k]
            raw = acc4.to(fx.BFloat16)
            raw_w = raw.bitcast(fx.Int32)  # 2 dwords
            e = _bf16x4(raw_w)
            if const_expr(write_qkv):
                if valid:
                    buf_copy_store(
                        qkv_buf, (row * N + head * D) // 4 + hl, raw_w, fx.Int32, 2
                    )
            # Gemma RMS norm + NeoX RoPE (fused_qknorm_idxrqknorm normAndRope).
            ss = e[0] * e[0]
            ss = fx.math.fma(e[1], e[1], ss)
            ss = fx.math.fma(e[2], e[2], ss)
            ss = fx.math.fma(e[3], e[3], ss)
            for msk in (16, 8, 4, 2, 1):
                ss = ss + _xor(ss, msk)
            r = _rsq(fx.math.fma(ss, fx.Float32(0.0078125), eps))
            nrm = [(e[i] * r) * (fx.Float32(1.0) + w[i]) for i in range_constexpr(4)]
            cos = _bf16x4(e_cos[t])
            sin = _bf16x4(e_sin[t])
            y = []
            for i in range_constexpr(4):
                pr = _xor(nrm[i], ROT // 8)
                ps = pr * sin[i]
                roped = fx.math.fma(nrm[i], cos[i], first.select(-ps, ps))
                y.append(in_rope.select(roped, nrm[i]))
            yb = Vec.from_elements(y, fx.Float32).to(fx.BFloat16)
            yw = yb.bitcast(fx.Int32)
            if valid:
                if is_q:
                    buf_copy_store(
                        q_buf, (row * (NQ * D) + head * D) // 4 + hl, yw, fx.Int32, 2
                    )
                if is_iq:
                    buf_copy_store(
                        iq_buf,
                        (row * (niq * D) + (head - IQ0) * D) // 4 + hl,
                        yw,
                        fx.Int32,
                        2,
                    )
            # NVFP4 rows: K / index-K post norm + RoPE, V raw; all bf16-rounded.
            src = is_v.select(raw_w, yw)
            vq = _bf16x4(Vec(src))
            code, sb_ = _nvfp4_lane(vq, is_ik)
            slot = is_ik.select(e_islot[t], e_slot[t])
            page = fx.Int64(slot >> 7)
            pos = fx.Int64(slot & 127)
            kv = is_k | is_v
            pay = kv.select(
                is_k.select(kp_base, vp_base) + page * i64_kp_page + pos * 64,
                ip_base + page * i64_ip_page + pos * 64,
            )
            sca = kv.select(
                is_k.select(ksc_base, vsc_base) + page * i64_ks_page + pos * 8,
                isc_base + page * i64_is_page + pos * 8,
            )
            if valid & (slot >= 0) & (kv | is_ik):
                fx.ptr_store(code.to(fx.Int16), _gptr(pay + fx.Int64(hl * 2), T.i16, 2))
                if hl % 4 == 0:
                    fx.ptr_store(
                        sb_.to(fx.Int8), _gptr(sca + fx.Int64(hl // 4), T.i8, 1)
                    )

    mgroups = (i32_m + (tile_m - 1)) // tile_m
    kernel_chain(
        arg_a, arg_as, arg_w, arg_ws, arg_slab, arg_q, arg_iq, arg_qkv,
        arg_qw, arg_kw, arg_iqw, arg_ikw, arg_cs, arg_pos, arg_slot, arg_islot,
        arg_kp, arg_vp, arg_ksc, arg_vsc, arg_ip, arg_isc,
        i32_m, i32_lda, i32_mpad, i32_cs_rows,
        i64_kp_page, i64_kp_head, i64_ks_page, i64_ks_head, i64_ip_page, i64_is_page,
        i32_eps,
        value_attrs={"rocdl.cluster_dims": f"{csize},1,1"},
    ).launch(
        grid=(csize, HEADS, mgroups),
        block=(block, 1, 1),
        stream=stream,
        cluster=(csize, 1, 1),
    )  # fmt: skip


launch_mxfp8_qkv_norm_rope_nvfp4_insert.compile_hints["llvm_options"] = {
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
