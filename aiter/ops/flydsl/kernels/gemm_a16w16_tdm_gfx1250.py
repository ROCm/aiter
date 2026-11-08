# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Wave32 TDM BF16/FP16 GEMM: one kernel, two-tile unroll, tunable cluster split-K.

NT uses wave-parity serpentine quadrants to overlap WMMA with independent LDS
reads. The same kernel retains a simple schedule for small or NN tiles.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl, tdm_ops
from flydsl.expr.rocdl.cluster import cluster_barrier
from flydsl.expr.typing import T

from .tdm_gemm_gfx1250_utils import (
    WmmaFragments,
    copy_tdm_operand,
    grouped_cluster_swizzle,
    make_tdm_operand,
)


def validate_config(
    bm,
    bn,
    bk,
    stages,
    mw,
    nw,
    split_k,
    group_m,
    layout,
    interleave,
    split_k_policy="atomic",
):
    """CPU-only launch constraints, shared by the wrapper and compiler."""
    if any(type(x) is not int or x <= 0 for x in (bm, bn, bk, stages, mw, nw, split_k)):
        raise ValueError("tiles, stages, waves and split_k must be positive integers")
    if type(group_m) is not int or group_m < 0:
        raise ValueError("group_m must be a nonnegative integer")
    if split_k_policy not in ("atomic", "workspace"):
        raise ValueError("split_k_policy must be atomic or workspace")
    if layout not in ("nt", "nn"):
        raise ValueError("layout must be nt or nn")
    if bk not in (32, 64, 128, 256, 512) or not 2 <= stages <= 16:
        raise ValueError("K tile must be 32..512 power-of-two; stages must be 2..16")
    waves = mw * nw
    if waves > 32 or not 1 <= split_k <= 9:
        raise ValueError("at most 32 waves and cluster split_k in 1..9")
    if bm % (mw * 16) or bn % (nw * 16) or bm % waves or bn % waves:
        raise ValueError("tile must contain whole wave-owned rows and WMMA atoms")
    ms, ns = bm // (mw * 16), bn // (nw * 16)
    if not (1 <= ms <= 8 and 1 <= ns <= 8):
        raise ValueError("each wave must own 1..8 WMMA tiles per axis")
    if bm * bn % (waves * 32 * 8):
        raise ValueError("tile must support whole-workgroup 128-bit C-shuffle")
    if interleave and (
        layout != "nt"
        or ms % 2
        or ns % 2
        or (ms // 2) * (ns // 2) < max(ms // 2, ns // 2) * 2
    ):
        raise ValueError(
            "interleave needs NT and even quadrants with room for LDS loads"
        )
    b_slots = bn * (bk + 8) if layout == "nt" else bk * (bn + 8)
    c_bytes = 4 if split_k > 1 and split_k_policy == "workspace" else 2
    shared = max(stages * (bm * (bk + 8) + b_slots) * 2, bm * (bn * c_bytes + 16))
    if shared > 320 * 1024:
        raise ValueError("tile exceeds the 320 KiB LDS capacity")
    return shared


@functools.lru_cache(maxsize=256)
def compile_gemm(
    dtype,
    bm,
    bn,
    bk,
    stages,
    mw,
    nw,
    split_k,
    group_m,
    layout,
    interleave,
    has_bias,
    split_k_policy="atomic",
):
    shared_bytes = validate_config(
        bm, bn, bk, stages, mw, nw, split_k, group_m, layout, interleave, split_k_policy
    )
    if dtype not in ("bf16", "fp16"):
        raise ValueError("only BF16/FP16 are supported")
    elem = fx.BFloat16 if dtype == "bf16" else fx.Float16
    waves, threads = mw * nw, mw * nw * 32
    ms, ns = bm // (mw * 16), bn // (nw * 16)
    hm, hn = ms // 2, ns // 2
    ap = bk + 8
    bp = bk + 8 if layout == "nt" else bn + 8
    a_size = bm * ap
    b_size = bn * bp if layout == "nt" else bk * bp
    a_stride = (ap, 1)
    b_stride = (1, bp) if layout == "nt" else (bp, 1)
    workspace_reduce = split_k > 1 and split_k_policy == "workspace"
    c_elem = fx.Float32 if workspace_reduce else elem
    cp = bn + (4 if workspace_reduce else 8)  # 16-byte C-shuffle padding.
    steps = bk // 32

    @fx.struct
    class Shared:
        data: fx.Array[elem, shared_bytes // 2, 16]

    @flyc.kernel
    def kernel(
        a: fx.Pointer,
        b: fx.Pointer,
        c: fx.Pointer,
        bias: fx.Pointer,
        workspace: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        lda: fx.Int32,
        ldb: fx.Int32,
        ldc: fx.Int32,
    ):
        tid = fx.thread_idx.x
        wave, lane = rocdl.wave_id(), tid % 32
        tile_id, part = fx.block_idx.x, fx.block_idx.y
        # Swizzle the logical output-tile id, never the split rank: all cluster
        # members must address the SAME output tile, including ragged groups.
        mt, nt = (m + bm - 1) // bm, (n + bn - 1) // bn
        if const_expr(group_m > 0):
            br, bc = grouped_cluster_swizzle(tile_id, 0, mt, nt, 1, 1, group_m)
        else:
            br, bc = tile_id // nt, tile_id % nt
        row0, col0 = br * bm, bc * bn
        wr, wc = wave // nw * (bm // mw), wave % nw * (bn // nw)
        shared = fx.SharedAllocator(static=True).allocate(Shared).peek().data.ptr
        sa, sb = shared, shared + stages * a_size
        sc = (
            fx.recast_iter(fx.Float32, shared)
            if const_expr(workspace_reduce)
            else shared
        )
        mma = fx.make_mma_atom(rocdl.WMMA(16, 16, 32, elem, fx.Float32))
        fragments = WmmaFragments(elem, 16, 32)
        acc = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(ms * ns)]
        for x in acc:
            x.store(fx.full(8, 0.0, fx.Float32))

        # No scratch partials/counters or separate zero kernel. Each C element
        # has one initializer (cluster rank 0) before ANY packed atomic write.
        if const_expr(split_k > 1 and not workspace_reduce):
            if part == 0:
                for i in range_constexpr(bm * bn // (threads * 8)):
                    off = (i * threads + tid) * 8
                    r, col = off // bn, off % bn
                    dst = c + fx.Int64(row0 + r) * fx.Int64(ldc) + col0 + col
                    if (row0 + bm <= m) and (col0 + bn <= n):
                        fx.make_view(dst, fx.make_layout(8, 1)).store(
                            fx.full(8, 0.0, elem)
                        )
                    else:
                        for j in range_constexpr(4):
                            if (row0 + r < m) and (col0 + col + j * 2 + 1 < n):
                                fx.make_view(dst + j * 2, fx.make_layout(2, 1)).store(
                                    fx.full(2, 0.0, elem)
                                )
            # Wait for global stores, not just LDS, before signaling the cluster.
            fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
            rocdl.s_wait_storecnt(0)
            cluster_barrier()
            fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)

        atom_a, ca = make_tdm_operand(
            a, (m, k), (lda, 1), (bm, bk), a_stride, num_warps=waves
        )
        atom_b, cb = make_tdm_operand(
            b,
            (k, n),
            (1, ldb) if layout == "nt" else (ldb, 1),
            (bk, bn),
            b_stride,
            num_warps=waves,
        )

        def copy_tile(slot, tile):
            copy_tdm_operand(
                atom_b,
                sb,
                cb[None, (None, bc)],
                tile_shape=(bk, bn),
                tile_strides=b_stride,
                stage_size=b_size,
                stages=stages,
                wave=wave,
                num_warps=waves,
                dst_stage=slot,
                src_stage=tile,
            )
            copy_tdm_operand(
                atom_a,
                sa,
                ca[None, (br, None)],
                tile_shape=(bm, bk),
                tile_strides=a_stride,
                stage_size=a_size,
                stages=stages,
                wave=wave,
                num_warps=waves,
                dst_stage=slot,
                src_stage=tile,
            )
            rocdl.sched_barrier(0)

        tiles = k // (split_k * bk)
        first = part * tiles
        last = first + tiles - 1
        # Clamp prologue/tail prefetches to this split. Duplicates are unused,
        # but keep tensorcnt accounting identical for hot loop and drain.
        for slot in range_constexpr(stages):
            copy_tile(fx.Int32(slot), fx.min(first + slot, last))
        tdm_ops.tensor_wait(2 * (stages - 1))
        fx.gpu.barrier()

        if const_expr(interleave):
            seed_a = [fx.make_rmem_tensor(8, fx.Int32) for _ in range_constexpr(hm)]
            seed_b = [fx.make_rmem_tensor(8, fx.Int32) for _ in range_constexpr(hn)]

            def fp(kind, slot, ki):
                ptr, sz, start, pitch = (
                    (sa, a_size, wr, ap) if kind == "a" else (sb, b_size, wc, bp)
                )
                return fragments.pointer(
                    ptr, sz, slot, start, ki * 32, pitch, lane
                ), 16 * pitch

            def seed(slot, ki, parity):
                pa, ra = fp("a", slot, ki)
                pb, rb = fp("b", slot, ki)
                la = [
                    lambda j=j: seed_a[j].store(fragments.load_packed(pa, j * ra))
                    for j in range_constexpr(hm)
                ]
                lb = [
                    lambda j=j: seed_b[j].store(fragments.load_packed(pb, j * rb))
                    for j in range_constexpr(hn)
                ]
                return (la, lb) if parity == 0 else (lb, la)

            def second(kind, slot, ki):
                count = hm if kind == "a" else hn
                p, stride = fp(kind, slot, ki)
                frags, chunks = [None] * count, [[] for _ in range_constexpr(count)]

                def read(j, chunk):
                    chunks[j].append(
                        fragments.load_chunk(p, (count + j) * stride + chunk * 16)
                    )
                    if const_expr(chunk == 1):
                        frags[j] = fragments.register_packed(
                            fragments.join_chunks(chunks[j])
                        )

                loads = {
                    2 * j + t: (lambda j=j, t=t: read(j, t))
                    for j in range(count)
                    for t in range(2)
                }
                return loads, frags

            def quadrant(m0, n0, af, bf, loads, n_fast, before=None):
                if const_expr(before is not None):
                    before()
                for j in range_constexpr(hm * hn):
                    mi, ni = (j // hn, j % hn) if n_fast else (j % hm, j // hm)
                    dst = acc[(m0 + mi) * ns + n0 + ni]
                    # Swapped WMMA operands give row-contiguous accumulator
                    # fragments for the C-shuffle. Both dtypes use this layout.
                    fx.gemm(mma, dst, bf[ni], af[mi], dst)
                    if const_expr(j in loads):
                        loads[j]()
                    if const_expr(j == max(loads, default=-1)):
                        for _ in range_constexpr(len(loads)):
                            rocdl.sched_mfma(1)
                            rocdl.sched_dsrd(1)
                        rocdl.sched_barrier(0)

            def compute(tile, has_next, parity, slot=None):
                slot = tile % stages if slot is None else slot
                for ki in range_constexpr(steps):
                    boundary = ki == steps - 1
                    more = not boundary or has_next
                    next_slot = (
                        ((1 - slot) if stages == 2 else (tile + 1) % stages)
                        if boundary
                        else slot
                    )
                    next_ki = 0 if boundary else ki + 1

                    def signal(boundary=boundary):
                        rocdl.sched_barrier(0)
                        rocdl.s_wait_dscnt(0)
                        if const_expr(boundary and has_next):
                            tdm_ops.tensor_wait(2 * (stages - 2))
                            rocdl.s_barrier_signal(-1)
                        rocdl.sched_barrier(0)

                    def refill(boundary=boundary):
                        if const_expr(boundary and has_next):
                            rocdl.s_barrier_wait(-1)
                            copy_tile(slot, fx.min(first + tile + stages, last))
                        rocdl.sched_barrier(0)

                    early, tail = seed(next_slot, next_ki, parity) if more else ([], [])
                    ready = hm * hn // 4
                    room = hm * hn - ready
                    q2 = {ready - 1: refill}
                    q2.update({ready + j: f for j, f in enumerate(early[:room])})
                    q3 = dict(enumerate(early[room:]))
                    q3.update({hm * hn - len(tail) + j: f for j, f in enumerate(tail)})
                    if parity == 0:
                        loads, b1 = second("b", slot, ki)
                        quadrant(0, 0, seed_a, seed_b, loads, False)
                        loads, a1 = second("a", slot, ki)
                        quadrant(0, hn, seed_a, b1, loads, False)
                        quadrant(hm, hn, a1, b1, q2, True, signal)
                        quadrant(hm, 0, a1, seed_b, q3, True)
                    else:
                        loads, a1 = second("a", slot, ki)
                        quadrant(0, 0, seed_a, seed_b, loads, True)
                        loads, b1 = second("b", slot, ki)
                        quadrant(hm, 0, a1, seed_b, loads, True)
                        quadrant(hm, hn, a1, b1, q2, False, signal)
                        quadrant(0, hn, seed_a, b1, q3, False)

            for group in seed(fx.Int32(0), 0, 0):
                for load in group:
                    load()

            def loop(parity):
                # Explicit unroll2: two consecutive K tiles, not two kernels.
                for pair in range((tiles - 1) // 2):
                    for u in range_constexpr(2):
                        compute(pair * 2 + u, True, parity, u if stages == 2 else None)
                for remainder in range((tiles - 1) % 2):
                    compute(tiles - 2, True, parity)
                compute(tiles - 1, False, parity)

            if fx.Int32(rocdl.readfirstlane(T.i32, wave % 2)) == 0:
                loop(0)
            else:
                loop(1)
        else:

            def compute(tile, has_next):
                slot = tile % stages
                for ki in range_constexpr(steps):
                    bf = [
                        fragments.load(
                            sb,
                            b_size,
                            slot,
                            wc + j * 16,
                            ki * 32,
                            bp,
                            lane,
                            k_contiguous=(layout == "nt"),
                        )
                        for j in range_constexpr(ns)
                    ]
                    for i in range_constexpr(ms):
                        af = fragments.load(
                            sa, a_size, slot, wr + i * 16, ki * 32, ap, lane
                        )
                        for j in range_constexpr(ns):
                            dst = acc[i * ns + j]
                            fx.gemm(mma, dst, bf[j], af, dst)
                if const_expr(has_next):
                    tdm_ops.tensor_wait(2 * (stages - 2))
                    fx.gpu.barrier()
                    copy_tile(slot, fx.min(first + tile + stages, last))

            for pair in range((tiles - 1) // 2):
                for u in range_constexpr(2):
                    compute(pair * 2 + u, True)
            for remainder in range((tiles - 1) % 2):
                compute(tiles - 2, True)
            compute(tiles - 1, False)

        tdm_ops.tensor_wait(0)
        fx.gpu.barrier()
        for i in range_constexpr(ms):
            for j in range_constexpr(ns):
                row = wr + i * 16 + lane % 16
                col = wc + j * 16 + lane // 16 * 8
                values = acc[i * ns + j].load().to(c_elem)
                fx.make_view(sc + row * cp + col, fx.make_layout(8, 1)).store(values)
        fx.gpu.barrier()

        if const_expr(workspace_reduce):
            # One producer per (tile, split, element). Never round partials.
            tile_base = fx.Int64(tile_id) * (split_k * bm * bn)
            for i in range_constexpr(bm * bn // (threads * 4)):
                off = (i * threads + tid) * 4
                row, col = off // bn, off % bn
                values = fx.ptr_load(sc + row * cp + col, result_type=T.vec(4, T.f32))
                fx.make_view(
                    workspace + tile_base + part * (bm * bn) + off, fx.make_layout(4, 1)
                ).store(values)
            fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
            rocdl.s_wait_storecnt(0)
            cluster_barrier()
            fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)
            if part == 0:
                for i in range_constexpr(bm * bn // (threads * 4)):
                    off = (i * threads + tid) * 4
                    row, col = off // bn, off % bn
                    total = fx.ptr_load(
                        workspace + tile_base + off, result_type=T.vec(4, T.f32)
                    )
                    for j in range_constexpr(1, split_k):
                        value = fx.ptr_load(
                            workspace + tile_base + j * (bm * bn) + off,
                            result_type=T.vec(4, T.f32),
                        )
                        total = total + value
                    values = total.to(elem)
                    dest = c + fx.Int64(row0 + row) * fx.Int64(ldc) + col0 + col
                    if (row0 + row < m) and (col0 + col + 3 < n):
                        if const_expr(has_bias):
                            bv = fx.ptr_load(
                                bias + col0 + col, result_type=T.vec(4, elem.ir_type)
                            )
                            values = (values.to(fx.Float32) + bv.to(fx.Float32)).to(
                                elem
                            )
                        fx.make_view(dest, fx.make_layout(4, 1)).store(values)
                    else:
                        for j in range_constexpr(4):
                            if (row0 + row < m) and (col0 + col + j < n):
                                value = values[j]
                                if const_expr(has_bias):
                                    bv = fx.ptr_load(
                                        bias + col0 + col + j, result_type=elem.ir_type
                                    )
                                    value = (
                                        value.to(fx.Float32) + bv.to(fx.Float32)
                                    ).to(elem)
                                fx.ptr_store(value, dest + j)
        elif const_expr(split_k > 1):
            out_buf = rocdl.make_buffer_tensor(
                fx.make_view(c, fx.make_layout((1, 1), (1, 1))), max_size=True
            )
            atomic = fx.make_copy_atom(rocdl.BufferAtomicPkAdd(elem), elem)

            def atomic_pair(values, j, dest):
                frag = fx.make_rmem_tensor(2, elem)
                frag.store(
                    fx.Vector.from_elements([values[j * 2], values[j * 2 + 1]], elem)
                )
                fx.copy(atomic, frag, out_buf[None, dest + j * 2])

            for i in range_constexpr(bm * bn // (threads * 8)):
                off = (i * threads + tid) * 8
                r, col = off // bn, off % bn
                values = fx.ptr_load(
                    sc + r * cp + col, result_type=T.vec(8, elem.ir_type)
                )
                dest = (row0 + r) * ldc + col0 + col
                if (row0 + bm <= m) and (col0 + bn <= n):
                    for j in range_constexpr(4):
                        atomic_pair(values, j, dest)
                else:
                    for j in range_constexpr(4):
                        if (row0 + r < m) and (col0 + col + j * 2 + 1 < n):
                            atomic_pair(values, j, dest)
            if const_expr(has_bias):
                fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
                rocdl.s_wait_storecnt(0)
                cluster_barrier()
                fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)
                if part == 0:
                    for i in range_constexpr(bm * bn // (threads * 2)):
                        off = (i * threads + tid) * 2
                        r, col = off // bn, off % bn
                        if (row0 + r < m) and (col0 + col + 1 < n):
                            dest = c + fx.Int64(row0 + r) * fx.Int64(ldc) + col0 + col
                            values = fx.ptr_load(
                                dest, result_type=T.vec(2, elem.ir_type)
                            )
                            bias_values = fx.ptr_load(
                                bias + col0 + col, result_type=T.vec(2, elem.ir_type)
                            )
                            fx.make_view(dest, fx.make_layout(2, 1)).store(
                                (values.to(fx.Float32) + bias_values.to(fx.Float32)).to(
                                    elem
                                )
                            )
        elif const_expr(not has_bias):
            # C-shuffle is already row-contiguous in LDS. Preserve TDM output
            # stores: the vector-store replacement regressed the 256x256 tile
            # in same-device A/B tests. Mask M/N edges in the global descriptor.
            tile_shape = (bm, cp)
            catom, coords = make_tdm_operand(
                c + fx.Int64(row0) * fx.Int64(ldc) + col0,
                (fx.max(0, fx.min(bm, m - row0)), fx.max(0, fx.min(bn, n - col0))),
                (ldc, 1),
                tile_shape,
                (cp, 1),
                num_warps=waves,
                store=True,
            )
            coords = coords[None, (0, None)]
            cshared = fx.make_view(
                sc,
                fx.make_layout((tile_shape, 1), ((cp, 1), bm * cp)),
            )
            src, dst = rocdl.cdna5.tdm_partition(
                catom,
                wave,
                fx.make_layout(waves, 1),
                cshared,
                coords,
            )
            fx.copy(catom, src[None, 0], dst[None, 0])
            tdm_ops.tensor_wait(0)
        else:
            for i in range_constexpr(bm * bn // (threads * 8)):
                off = (i * threads + tid) * 8
                r, col = off // bn, off % bn
                values = fx.ptr_load(
                    sc + r * cp + col, result_type=T.vec(8, elem.ir_type)
                )
                dest = c + fx.Int64(row0 + r) * fx.Int64(ldc) + col0 + col
                if (row0 + r < m) and (col0 + col + 7 < n):
                    if const_expr(has_bias):
                        bv = fx.ptr_load(
                            bias + col0 + col, result_type=T.vec(8, elem.ir_type)
                        )
                        values = (values.to(fx.Float32) + bv.to(fx.Float32)).to(elem)
                    fx.make_view(dest, fx.make_layout(8, 1)).store(values)
                else:
                    for j in range_constexpr(8):
                        if (row0 + r < m) and (col0 + col + j < n):
                            val = values[j]
                            if const_expr(has_bias):
                                bv = fx.ptr_load(
                                    bias + col0 + col + j, result_type=elem.ir_type
                                )
                                val = (val.to(fx.Float32) + bv.to(fx.Float32)).to(elem)
                            fx.ptr_store(val, dest + j)

    @flyc.jit
    def launch(
        a: fx.Pointer,
        b: fx.Pointer,
        c: fx.Pointer,
        bias: fx.Pointer,
        workspace: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        lda: fx.Int32,
        ldb: fx.Int32,
        ldc: fx.Int32,
        stream: fx.Stream,
    ):
        grid = ((m + bm - 1) // bm * ((n + bn - 1) // bn), split_k, 1)
        if split_k > 1:
            kernel(
                a,
                b,
                c,
                bias,
                workspace,
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                value_attrs={"rocdl.cluster_dims": f"1,{split_k},1"},
            ).launch(
                grid=grid, block=(threads, 1, 1), stream=stream, cluster=(1, split_k, 1)
            )
        else:
            kernel(a, b, c, bias, workspace, m, n, k, lda, ldb, ldc).launch(
                grid=grid, block=(threads, 1, 1), stream=stream
            )

    kernel._func.__name__ = (
        f"tdm_gemm_{dtype}_{layout}_t{bm}x{bn}x{bk}_s{stages}_w{mw}x{nw}"
        f"_sk{split_k}_{split_k_policy}_gm{group_m}_i{int(interleave)}_bias{int(has_bias)}_u2"
    )
    if interleave:
        launch.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
            "amdgpu-sched-strategy": "coexec",
            "enable-post-misched": True,
            "amdgpu-barrier-signal-wait-latency": 32,
        }
    return launch
