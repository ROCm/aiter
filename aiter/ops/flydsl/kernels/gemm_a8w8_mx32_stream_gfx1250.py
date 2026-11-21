# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode-sized (small M) native MXFP8 GEMM for gfx1250 that streams weights.

The LDS/TDM GEMMs move every operand byte through one TDM queue per
workgroup and launch too few workgroups at decode M to keep the device busy:
a CU then streams its weight slab serially at ~64 B/clk while most CUs idle.
This kernel targets M <= 256:

* Waves load the weight WMMA operand straight from global memory into VGPRs
  with ``buffer_load_dwordx4``. The (16, 16)-preshuffled weight puts the 16
  bytes each lane feeds the WMMA at consecutive addresses, so one load of a
  wave reads 512 contiguous weight bytes; no weight byte is loaded twice.
* A workgroup has ``wn x wk`` waves. The ``wn`` waves split the N tile, the
  ``wk`` waves split the workgroup's K range. The activation either goes
  straight to VGPRs too (``a_lds=0``; for M <= 8, where most A rows are
  out of bounds), is loaded along K and transposed through a wave-private LDS
  slot (``a_lds=2``: a wave load then covers whole 128-byte rows), or is staged
  once per workgroup in LDS and shared by the ``wn`` waves (``a_lds=1``).
  The main loop has no barrier; per-wave K partials are summed through LDS in
  wave order.
* ``split_k`` workgroups share one output tile. Each publishes its FP32 tile;
  the reduction always sums the splits in split order, so results are
  bit-identical across runs and graph replays and there are no output
  atomics. Either the last split to arrive (per-tile arrival counter, no
  waiting, deadlock-free) reduces the tile, or with ``cluster_reduce`` the
  splits form one hardware cluster and every rank reduces a stripe after the
  cluster barrier.

Numerics follow the other gfx1250 MXFP8 kernels: E4M3 x E4M3 WMMA with the two
E8M0 1x32 scales applied in hardware, FP32 accumulation, BF16/FP16 output.

Layouts (as gemm_a8w8_gfx1250 with mxfp8_32):
  A       [M, K] e4m3 row-major (row stride lda, a multiple of 16 bytes)
  B       [N, K] e4m3 (16, 16)-preshuffled
  A scale [M, K // 32] e8m0 row-major
  B scale [N, K // 32] e8m0 32x4 (n32k4) shuffled
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.communication_ops_utils import atomic_add_agent
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import (
    make_lds_copy_ops,
    workgroup_barrier,
)
from aiter.ops.flydsl.kernels.kernels_common import format_kernel_name
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

WAVE = 32
WMMA_K = 128
CPOL_DEVICE = 16  # device scope: a tile's splits may run on different XCDs
A_LDS_PAD = 16


def stream_k_steps_per_wave(K: int, wk: int, split_k: int) -> int:
    """128-wide K steps each wave owns; 0 when K does not divide evenly."""
    steps = K // WMMA_K
    if K % WMMA_K or steps % (wk * split_k):
        return 0
    return steps // (wk * split_k)


A_WAVE_ROW = 128 + 16  # padded LDS row of one wave's [tile_m, 128] A step


def stream_lds_bytes(K, tile_m, n_blocks, wn, wk, split_k, a_lds) -> int:
    """a_lds: 0 direct A loads, 1 workgroup-staged A, 2 per-wave A transpose."""
    tile_n = 16 * n_blocks * wn
    red = wk * tile_m * tile_n * 4 if wn * wk > 1 else 0
    if a_lds == 1:
        # Staged activation [tile_m, K / split_k] plus its scale dwords [tile_m, steps].
        a = tile_m * (K // split_k + A_LDS_PAD) + tile_m * (K // 128 // split_k) * 4
        return max(red, a, 16)
    if a_lds == 2:
        # Each wave's private [tile_m, 128] transpose slot, after the reduction area.
        return red + wn * wk * tile_m * A_WAVE_ROW
    return max(red, 16)


@flyc.jit
def launch_gemm_a8w8_mx32_stream(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    arg_ws: fx.Pointer,
    arg_sem: fx.Pointer,
    i32_m: fx.Int32,
    stream: fx.Stream,
    i32_n: fx.Int32,
    i32_lda: fx.Int32,
    i32_ldc: fx.Int32,
    K: Constexpr[int],
    tile_m: Constexpr[int],
    n_blocks: Constexpr[int],
    wn: Constexpr[int],
    wk: Constexpr[int],
    split_k: Constexpr[int],
    depth: Constexpr[int],
    a_lds: Constexpr[int],
    cluster_reduce: Constexpr[bool],
    out_is_f16: Constexpr[int],
):
    waves = wn * wk
    if tile_m % 16 or not 16 <= tile_m <= 256:
        raise ValueError(f"tile_m must be a multiple of 16 in [16, 256], got {tile_m}")
    if not 1 <= n_blocks <= 8 or not 1 <= waves <= 16:
        raise ValueError(f"unsupported n_blocks={n_blocks} / waves={wn}x{wk}")
    if not 1 <= split_k <= 16:
        raise ValueError(f"split_k must be 1..16, got {split_k}")
    kpw = stream_k_steps_per_wave(K, wk, split_k)
    if kpw < 1:
        raise ValueError(
            f"K={K} must split into whole 128-wide steps over wk*split_k={wk * split_k}"
        )
    depth = max(1, min(depth, kpw))
    mb = tile_m // 16
    wave_n = 16 * n_blocks
    tile_n = wave_n * wn
    block = waves * WAVE
    sa_opsel = mb % 2 == 0
    sa_loads = mb // 2 if sa_opsel else mb
    KB4 = K // 128  # A-scale dwords per row
    k_wg = K // split_k  # K bytes of one workgroup
    a_row = k_wg + A_LDS_PAD  # LDS row pitch of the staged activation
    tile_elems = tile_m * tile_n
    tile_vec = tile_elems // 4
    if tile_vec % block and block % tile_vec:
        raise ValueError(
            f"tile {tile_m}x{tile_n} has {tile_vec} float4s, which neither divides "
            f"nor is a multiple of the {block}-thread block"
        )
    if split_k > 1 and waves == 1:
        raise ValueError("split_k > 1 needs at least two waves")
    if cluster_reduce and (tile_vec % split_k or split_k > 16):
        raise ValueError(f"cluster split_k={split_k} must divide {tile_vec}")
    stripe_vec = tile_vec // split_k
    a_units = tile_m * (k_wg // 16)  # 16-byte A units staged per workgroup
    steps_wg = K // 128 // split_k  # 128-wide K steps of one workgroup
    sa_units = tile_m * steps_wg  # A-scale dwords staged per workgroup
    SA_LDS_OFF = tile_m * a_row  # staged A scales follow the staged A tile
    if a_lds == 1 and a_units % block:
        raise ValueError("the staged A tile must divide over the block")
    lds_bytes = stream_lds_bytes(K, tile_m, n_blocks, wn, wk, split_k, a_lds)
    wave_stage = a_lds == 2
    a_lds = a_lds == 1
    AW_OFF = wk * tile_m * tile_n * 4 if waves > 1 else 0
    out_elem = fx.Float16 if out_is_f16 else fx.BFloat16
    kernel_name = format_kernel_name(
        f"gemm_a8w8_mx32_stream_t{tile_m}x{tile_n}_wn{wn}_wk{wk}_sk{split_k}"
        f"{'c' if cluster_reduce else ''}_d{depth}{'_al' if a_lds else ''}"
        f"{'_aw' if wave_stage else ''}_k{K}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_stream(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        arg_ws: fx.Pointer,
        arg_sem: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldc: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        w_n = wave % wn
        w_k = wave // wn
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        m0 = fx.Int32(bid_x) * tile_m
        n0 = fx.Int32(bid_y) * tile_n
        nw0 = n0 + w_n * wave_n
        split = fx.Int32(bid_z)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(lds_bytes)
        lds_base = arena.base_ptr
        lds_f32 = fx.recast_iter(fx.Float32, lds_base)
        lds_idx = fx.Int64(fx.ptrtoint(lds_base))
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
        lds_load_b32, lds_store_b32 = make_lds_copy_ops(32)

        # A rows past M read as zero through the descriptor bound.
        a_buf = ptr_buf_tensor(
            arg_a, fx.Int32, unit_elems=4, num_records_bytes=i32_m * i32_lda
        )
        b_buf = ptr_buf_tensor(arg_b, fx.Int32, unit_elems=4)
        sa_buf = ptr_buf_tensor(
            arg_scale_a, fx.Int32, num_records_bytes=i32_m * (K // 32)
        )
        sb_buf = ptr_buf_tensor(arg_scale_b, fx.Int32)
        lda16 = i32_lda // 16

        ks_wg = split * (K // 128 // split_k)  # first K step of the workgroup
        ks_w = w_k * kpw  # first K step of the wave inside the workgroup
        ks0 = ks_wg + ks_w
        a_row_g = [m0 + mi * 16 + lane16 for mi in range_constexpr(mb)]
        a_base = [r * lda16 + kgrp for r in a_row_g]
        sa_row = [
            (m0 + p * 32 + lane) if const_expr(sa_opsel) else a_row_g[p]
            for p in range_constexpr(sa_loads)
        ]
        sa_base = [r * KB4 for r in sa_row]
        b_base = [(nw0 // 16 + j) * K + lane for j in range_constexpr(n_blocks)]
        sb_col = [nw0 + j * 16 + lane16 for j in range_constexpr(n_blocks)]
        sb_base = [(c // 32) * (K // 4) + c % 32 for c in sb_col]
        # Byte offset of this lane's first A fragment chunk in the staged tile.
        a_lds_lane = [
            (mi * 16 + lane16) * a_row + kgrp * 16 for mi in range_constexpr(mb)
        ]

        def _cat16(v):
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def load_b_step(i):
            ks = ks0 + i
            bfr = [
                [
                    buf_copy_load(b_buf, b_base[j] + ks * 128 + jj * 32, fx.Int32, 4)
                    for jj in range_constexpr(4)
                ]
                for j in range_constexpr(n_blocks)
            ]
            sb = [
                buf_copy_load(sb_buf, sb_base[j] + ks * 32, fx.Int32)
                for j in range_constexpr(n_blocks)
            ]
            # A staged activation brings its scales through LDS too: a direct
            # A-scale load touches one cache line per lane.
            sa = (
                None
                if const_expr(a_lds)
                else [
                    buf_copy_load(sa_buf, sa_base[p] + ks, fx.Int32)
                    for p in range_constexpr(sa_loads)
                ]
            )
            return bfr, sb, sa

        def load_a_global(i):
            ks = ks0 + i
            return [
                [
                    buf_copy_load(a_buf, a_base[mi] + ks * 8 + jj * 2, fx.Int32, 4)
                    for jj in range_constexpr(4)
                ]
                for mi in range_constexpr(mb)
            ]

        # wave_stage: rows of A are loaded along K (a wave load reads four whole
        # 128-byte rows instead of a 32-byte piece of 16 rows) and transposed
        # into the WMMA fragment layout through a wave-private LDS slot.
        aw_row = [
            [m0 + mi * 16 + q * 4 + lane // 8 for q in range_constexpr(4)]
            for mi in range_constexpr(mb)
        ]
        aw_slot = AW_OFF + wave * (tile_m * A_WAVE_ROW)

        def load_a_rows(i):
            ks = ks0 + i
            return [
                [
                    buf_copy_load(
                        a_buf, aw_row[mi][q] * lda16 + ks * 8 + lane % 8, fx.Int32, 4
                    )
                    for q in range_constexpr(4)
                ]
                for mi in range_constexpr(mb)
            ]

        def transpose_a(rows):
            for mi in range_constexpr(mb):
                for q in range_constexpr(4):
                    lds_store_b128(
                        lds_idx,
                        aw_slot
                        + (mi * 16 + q * 4 + lane // 8) * A_WAVE_ROW
                        + (lane % 8) * 16,
                        rows[mi][q],
                    )
            return [
                [
                    Vec(
                        lds_load_b128(
                            lds_idx,
                            aw_slot
                            + (mi * 16 + lane16) * A_WAVE_ROW
                            + kgrp * 16
                            + jj * 32,
                        )
                    )
                    for jj in range_constexpr(4)
                ]
                for mi in range_constexpr(mb)
            ]

        def load_a_lds(i):
            # K offset of step i inside the workgroup's staged K range.
            kb = (ks_w + i) * 128
            return [
                [
                    Vec(lds_load_b128(lds_idx, a_lds_lane[mi] + kb + jj * 32))
                    for jj in range_constexpr(4)
                ]
                for mi in range_constexpr(mb)
            ]

        sa_lds_row = [
            (p * 32 + lane) if const_expr(sa_opsel) else (p * 16 + lane16)
            for p in range_constexpr(sa_loads)
        ]

        def load_sa_lds(i):
            return [
                lds_load_b32(
                    lds_idx, SA_LDS_OFF + (sa_lds_row[p] * steps_wg + ks_w + i) * 4
                )[0]
                for p in range_constexpr(sa_loads)
            ]

        def load_step(i):
            bfr, sb, sa = load_b_step(i)
            if const_expr(a_lds):
                afr = None
            elif const_expr(wave_stage):
                afr = load_a_rows(i)
            else:
                afr = load_a_global(i)
            return [bfr, afr, sb, sa]

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
            [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_blocks)]
            for _ in range_constexpr(mb)
        ]
        for row in acc:
            for c in row:
                c.store(fx.full(8, 0.0, fx.Float32))

        def _rmem(v):
            t = fx.make_rmem_tensor(16, fx.Int32)
            t.store(v)
            return t

        def compute_step(st):
            bfr, afr, sb, sa = st
            bt = [_rmem(_cat16(bfr[j])) for j in range_constexpr(n_blocks)]
            at = [_rmem(_cat16(afr[mi])) for mi in range_constexpr(mb)]
            for mi in range_constexpr(mb):
                for j in range_constexpr(n_blocks):
                    fx.gemm(
                        atoms[mi % 2 if sa_opsel else 0],
                        acc[mi][j],
                        bt[j],
                        at[mi],
                        acc[mi][j],
                        scale_a=sb[j],
                        scale_b=sa[mi // 2 if sa_opsel else mi],
                    )

        # The activation tile (a_lds) and the weight loads of the prefetch
        # window all go out before any wait: every load is in flight at once.
        # The scheduler would otherwise sink loads next to their first use and
        # serialize their latencies.
        A_BATCH = 32  # 16-byte A units per thread in flight (128 VGPRs)
        per_thr = a_units // block if const_expr(a_lds) else 0
        k16 = k_wg // 16

        def a_stage_loads(b0):
            units = [
                tid + (b0 + u) * block
                for u in range_constexpr(min(A_BATCH, per_thr - b0))
            ]
            vals = [
                buf_copy_load(
                    a_buf, (m0 + u // k16) * lda16 + ks_wg * 8 + u % k16, fx.Int32, 4
                )
                for u in units
            ]
            return units, vals

        def a_stage_stores(units, vals):
            for u, val in zip(units, vals):
                lds_store_b128(lds_idx, (u // k16) * a_row + (u % k16) * 16, val)

        sa_per_thr = (sa_units + block - 1) // block

        def sa_stage():
            # Coalesced A-scale staging: consecutive threads, consecutive dwords.
            units = [tid + v * block for v in range_constexpr(sa_per_thr)]
            vals = [
                buf_copy_load(
                    sa_buf,
                    (m0 + u // steps_wg) * KB4 + ks_wg + u % steps_wg,
                    fx.Int32,
                )
                for u in units
            ]
            return units, vals

        def sa_stage_stores(units, vals):
            for u, val in zip(units, vals):
                if const_expr(sa_units % block == 0):
                    lds_store_b32(
                        lds_idx, SA_LDS_OFF + u * 4, Vec.from_elements([val], fx.Int32)
                    )
                else:
                    if u < sa_units:
                        lds_store_b32(
                            lds_idx,
                            SA_LDS_OFF + u * 4,
                            Vec.from_elements([val], fx.Int32),
                        )

        first = a_stage_loads(0) if const_expr(a_lds) else None
        first_sa = sa_stage() if const_expr(a_lds) else None
        rocdl.sched_barrier(0)
        inflight = [load_step(i) for i in range_constexpr(depth)]
        if const_expr(a_lds):
            rocdl.sched_barrier(0)
            a_stage_stores(*first)
            sa_stage_stores(*first_sa)
            for b0 in range_constexpr(A_BATCH, per_thr, A_BATCH):
                a_stage_stores(*a_stage_loads(b0))
            workgroup_barrier(use_cluster=False)
        rocdl.sched_barrier(0)
        for i in range_constexpr(kpw):
            st = inflight.pop(0)
            if const_expr(i + depth < kpw):
                inflight.append(load_step(i + depth))
                rocdl.sched_barrier(0)
            if const_expr(a_lds):
                st[1] = load_a_lds(i)
                st[3] = load_sa_lds(i)
            if const_expr(wave_stage):
                st[1] = transpose_a(st[1])
            compute_step(st)

        # ---- reduce the wk waves' K ranges through LDS, in wave order ----
        out_ptr_ty = fx.PointerType.get(
            out_elem.ir_type, arg_c.address_space, alignment=8
        )
        out_base = fx.inttoptr(out_ptr_ty, fx.ptrtoint(arg_c))
        ldc64 = fx.Int64(i32_ldc)
        TNV = tile_n // 4

        if const_expr(waves == 1):
            # One wave owns the whole tile and the whole K: its fragments are
            # the result.
            for mi in range_constexpr(mb):
                for j in range_constexpr(n_blocks):
                    grow = m0 + mi * 16 + lane16
                    col = n0 + j * 16 + kgrp * 8
                    if grow < i32_m:
                        fx.ptr_store(
                            Vec(acc[mi][j].load()).to(out_elem),
                            out_base + fx.Int64(grow) * ldc64 + fx.Int64(col),
                        )
            return

        if const_expr(a_lds):
            workgroup_barrier(use_cluster=False)  # all waves done reading A
        for mi in range_constexpr(mb):
            for j in range_constexpr(n_blocks):
                row = mi * 16 + lane16
                col = w_n * wave_n + j * 16 + kgrp * 8
                fx.ptr_store(
                    Vec(acc[mi][j].load()),
                    lds_f32 + (w_k * tile_m + row) * tile_n + col,
                )
        workgroup_barrier(use_cluster=False)

        def wave_sum(e):
            """Sum of the wk waves' partials of float4 ``e`` of the tile, in order."""
            row = e // TNV
            col = (e % TNV) * 4
            total = None
            for w in range_constexpr(wk):
                part = Vec(
                    fx.ptr_load(
                        lds_f32 + (w * tile_m + row) * tile_n + col,
                        result_type=T.vec(4, fx.Float32.ir_type),
                    )
                )
                total = part if total is None else total + part
            return total

        def store_out(e, val):
            row = e // TNV
            col = (e % TNV) * 4
            grow = m0 + row
            if grow < i32_m:
                fx.ptr_store(
                    val.to(out_elem),
                    out_base + fx.Int64(grow) * ldc64 + fx.Int64(n0 + col),
                )

        per_thread = max(1, tile_vec // block)
        partial_block = tile_vec < block

        def for_tile_vecs(fn):
            # Each thread owns float4s tid, tid + block, ...; a tile smaller
            # than the block leaves the upper threads idle.
            for v in range_constexpr(per_thread):
                e = tid + v * block
                if const_expr(partial_block):
                    if e < tile_vec:
                        fn(e)
                else:
                    fn(e)

        if const_expr(split_k == 1):
            for_tile_vecs(lambda e: store_out(e, wave_sum(e)))
            return

        # ---- split-K across workgroups: publish FP32 tiles, reduce in order ----
        flat_tile = fx.Int32(bid_y) * fx.Int32(fx.grid_dim.x) + fx.Int32(bid_x)
        ws = ptr_buf_tensor(
            arg_ws,
            fx.Float32,
            unit_elems=4,
            num_records_bytes=fx.Int64(fx.grid_dim.x * fx.grid_dim.y)
            * (split_k * tile_elems * 4),
        )
        tile_base = flat_tile * (split_k * tile_vec)
        cpol = CPOL_DEVICE

        def publish(e):
            # Device scope: a tile's splits may run on different XCDs.
            buf_copy_store(
                ws,
                tile_base + split * tile_vec + e,
                wave_sum(e),
                fx.Float32,
                4,
                cache_modifier=cpol,
            )

        for_tile_vecs(publish)
        rocdl.s_wait_storecnt(0)

        def reduce_one(e):
            parts = [
                buf_copy_load(
                    ws,
                    tile_base + s * tile_vec + e,
                    fx.Float32,
                    4,
                    cache_modifier=cpol,
                )
                for s in range_constexpr(split_k)
            ]
            total = parts[0]
            for s in range_constexpr(1, split_k):
                total = total + parts[s]
            store_out(e, total)

        if const_expr(cluster_reduce):
            # The splits of a tile are one cluster: after the barrier every
            # rank reduces its own stripe of the tile.
            cluster.cluster_barrier()
            stripe0 = split * stripe_vec
            n_iter = (stripe_vec + block - 1) // block
            for v in range_constexpr(n_iter):
                e_local = tid + v * block
                if const_expr(stripe_vec % block == 0):
                    reduce_one(stripe0 + e_local)
                else:
                    if e_local < stripe_vec:
                        reduce_one(stripe0 + e_local)
            return

        # Arrival counter: the last split of a tile to arrive reduces it. No
        # workgroup ever waits on another, so any grid size is deadlock-free;
        # the reducer hands the counter back zeroed for the next launch.
        workgroup_barrier(use_cluster=False)
        sem_addr = fx.Int64(fx.ptrtoint(arg_sem)) + fx.Int64(flat_tile) * 4
        lds_i32 = fx.recast_iter(fx.Int32, lds_base)
        if tid == 0:
            arrival = fx.Int32(atomic_add_agent(sem_addr, fx.Int32(1)))
            fx.ptr_store(arrival, lds_i32)
        workgroup_barrier(use_cluster=False)
        arrival = fx.Int32(fx.ptr_load(lds_i32))
        if arrival == fx.Int32(split_k - 1):
            for_tile_vecs(reduce_one)
            if tid == 0:
                sem_ptr = fx.inttoptr(
                    fx.PointerType.get(
                        fx.Int32.ir_type,
                        address_space=fx.AddressSpace.Global,
                        alignment=4,
                    ),
                    sem_addr,
                )
                fx.ptr_store(fx.Int32(0), sem_ptr)

    gx = (i32_m + (tile_m - 1)) // tile_m
    gy = i32_n // tile_n
    use_cluster = split_k > 1 and cluster_reduce
    kernel_stream(
        arg_c,
        arg_a,
        arg_b,
        arg_scale_a,
        arg_scale_b,
        arg_ws,
        arg_sem,
        i32_m,
        i32_n,
        i32_lda,
        i32_ldc,
        value_attrs={"rocdl.cluster_dims": (f"1,1,{split_k}" if use_cluster else None)},
    ).launch(
        grid=(gx, gy, split_k),
        block=(block, 1, 1),
        stream=stream,
        cluster=(1, 1, split_k) if use_cluster else None,
    )


launch_gemm_a8w8_mx32_stream.compile_hints["llvm_options"] = {
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
