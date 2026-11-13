# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experimental BF16 wave-specialized GEMM; no default dispatch uses this path.

Loader waves own disjoint physical LDS stripes. Each ring slot has one ready
sequence per loader and one released sequence per compute wave. The initial
and final barriers involve the entire workgroup; the role loops never do.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl

from .gemm_a16w16_gfx950_utils import make_lds_layout, swizzled_col_idx
from .kernels_common import set_slp_vectorize_once_used
from .tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)


@flyc.jit
def _wait_sequence(ptr: fx.Pointer, want: fx.Int32):
    # Acquire is at workgroup scope: data and flag live in this workgroup's LDS.
    # Polls only same-workgroup resident waves, never an unscheduled peer CTA.
    value = fx.generic_load(
        ptr,
        dtype=fx.Int32,
        memory_order=fx.AtomicOrdering.Acquire,
        syncscope=rocdl.SyncScope.Workgroup,
    )
    while value < want:
        rocdl.s_sleep(1)
        value = fx.generic_load(
            ptr,
            dtype=fx.Int32,
            memory_order=fx.AtomicOrdering.Acquire,
            syncscope=rocdl.SyncScope.Workgroup,
        )


@functools.cache
def build_ws(bm, bn, bk, stages, producer_waves, wm, wn, splits, prefetch):
    """Build a private launcher for one validated geometry, not a tensor cache."""
    producers = producer_waves
    consumers = wm * wn
    threads = (producers + consumers) * 64
    load_threads = producers * 64
    load_a_iters = bm * bk // (load_threads * 8)
    load_b_iters = bn * bk // (load_threads * 8)

    @flyc.kernel
    def ws_kernel(
        op: fx.Pointer,
        ap: fx.Pointer,
        bp: fx.Pointer,
        pp: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        lda: fx.Int32,
        ldb: fx.Int32,
    ):
        tid = fx.thread_idx.x
        wave = tid // 64
        lane = tid % 64
        tiles_n = (n + bn - 1) // bn
        tile_m = fx.block_idx.x // tiles_n
        tile_n = fx.block_idx.x % tiles_n
        split = fx.block_idx.y
        row0 = tile_m * bm
        col0 = tile_n * bn
        partition_k = k // splits
        k0 = split * partition_k
        k_tiles = partition_k // bk

        @fx.struct
        class Shared:
            a: fx.Array[fx.BFloat16, stages * bm * bk, 16]
            b: fx.Array[fx.BFloat16, stages * bn * bk, 16]
            ready: fx.Array[fx.Int32, stages * producers, 4]
            released: fx.Array[fx.Int32, stages * consumers, 4]

        mem = fx.SharedAllocator().allocate(Shared)
        sa, sb = mem.a.peek().ptr, mem.b.peek().ptr
        ready, released = mem.ready.peek().ptr, mem.released.peek().ptr
        for i in range(tid, stages * producers, threads):
            fx.ptr_store(fx.Int32(0), ready + i)
        for i in range(tid, stages * consumers, threads):
            fx.ptr_store(fx.Int32(0), released + i)
        gpu.barrier()

        al = make_lds_layout(bm, bk)
        bl = make_lds_layout(bn, bk)
        a_buffer = fx.rocdl.make_buffer_tensor(
            fx.make_view(ap, fx.make_layout((m, k), (lda, 1))), max_size=True
        )
        b_buffer = fx.rocdl.make_buffer_tensor(
            fx.make_view(bp, fx.make_layout((n, k), (ldb, 1))), max_size=True
        )

        def produce():
            atom = fx.make_copy_atom(rocdl.cdna4.BufferLoadAsyncLDS128b(), 128)
            packet = fx.make_layout(8, 1)

            def issue(tile):
                slot = tile % stages
                generation = tile // stages
                for c in range_constexpr(consumers):
                    _wait_sequence(released + slot * consumers + c, generation)
                # Every lane of this producer wave issues an eight-BF16 DMA.
                # The intrinsic's LDS address is wave-uniform; hardware supplies
                # the lane displacement. Physical stripes cover each tile once.
                for i in range_constexpr(load_a_iters):
                    pos = tid + i * load_threads
                    row = pos // (bk // 8)
                    kk = pos % (bk // 8) * 8
                    gr = row0 + row
                    safe = (gr < m).select(gr, 0)
                    off = (
                        safe * lda + k0 + tile * bk + swizzled_col_idx(row, kk, al, bk)
                    )
                    dst = sa + slot * bm * bk + (wave * 64 + i * load_threads) * 8
                    fx.copy_atom_call(
                        atom,
                        fx.make_view(fx.get_iter(a_buffer) + off, packet),
                        fx.make_view(dst, packet),
                    )
                for i in range_constexpr(load_b_iters):
                    pos = tid + i * load_threads
                    row = pos // (bk // 8)
                    kk = pos % (bk // 8) * 8
                    gc = col0 + row
                    safe = (gc < n).select(gc, 0)
                    off = (
                        safe * ldb + k0 + tile * bk + swizzled_col_idx(row, kk, bl, bk)
                    )
                    dst = sb + slot * bn * bk + (wave * 64 + i * load_threads) * 8
                    fx.copy_atom_call(
                        atom,
                        fx.make_view(fx.get_iter(b_buffer) + off, packet),
                        fx.make_view(dst, packet),
                    )
                rocdl.asyncmark()

            def publish(tile):
                slot = tile % stages
                generation = tile // stages
                # Workgroup release orders completed DMA before ready publication.
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Release,
                    syncscope=rocdl.SyncScope.Workgroup,
                )
                if lane == 0:
                    fx.generic_store(
                        ready + slot * producers + wave,
                        generation + 1,
                        memory_order=fx.AtomicOrdering.Release,
                        syncscope=rocdl.SyncScope.Workgroup,
                    )

            depth = stages - 1
            if fx.const_expr(prefetch):
                if k_tiles >= depth:
                    for tile in range_constexpr(depth):
                        issue(tile)
                    for tile in range(k_tiles - depth):
                        rocdl.wait_asyncmark(depth - 1)
                        publish(tile)
                        issue(tile + depth)
                    for tail in range_constexpr(depth):
                        rocdl.wait_asyncmark(depth - 1 - tail)
                        publish(k_tiles - depth + tail)
                else:
                    for tile in range(k_tiles):
                        issue(tile)
                        rocdl.wait_asyncmark(0)
                        publish(tile)
            else:
                for tile in range(k_tiles):
                    issue(tile)
                    rocdl.wait_asyncmark(0)
                    publish(tile)

        def consume():
            cid = tid - producers * 64
            cwave = wave - producers
            mma = fx.make_tiled_mma(
                fx.make_mma_atom(rocdl.MFMA(16, 16, 32, fx.BFloat16)),
                fx.make_layout((wm, wn, 1), (wn, 1, 0)),
                fx.make_tile(None, None, fx.make_layout((8, 4), (1, 8))),
            )
            thr = mma.thr_slice(cid)
            copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            layout_atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
            ca = fx.make_tiled_copy_A(layout_atom, mma).get_slice(cid)
            cb = fx.make_tiled_copy_B(layout_atom, mma).get_slice(cid)
            fa = thr.make_fragment_A(fx.make_view(sa, al))
            fb = thr.make_fragment_B(fx.make_view(sb, bl))
            coords = fx.make_view(op, fx.make_layout((bm, bn), (bn, 1)))
            fc = thr.make_fragment_C(coords)
            ra, rb = ca.retile(fa), cb.retile(fb)
            fc.fill(0.0)

            def compute_slot(slot):
                va = ca.partition_S(fx.make_view(sa + slot * bm * bk, al))
                vb = cb.partition_S(fx.make_view(sb + slot * bn * bk, bl))
                for kk in range_constexpr(bk // 32):
                    fx.copy(copy_atom, va[None, None, kk], ra[None, None, kk])
                    fx.copy(copy_atom, vb[None, None, kk], rb[None, None, kk])
                    fx.gemm(
                        mma,
                        fc,
                        fa[None, None, kk],
                        fb[None, None, kk],
                        fc,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )

            for tile in range(k_tiles):
                slot = tile % stages
                generation = tile // stages
                for p in range_constexpr(producers):
                    _wait_sequence(ready + slot * producers + p, generation + 1)
                compute_slot(slot)
                # Register-fed MFMA no longer reads LDS once ds reads retire.
                # Release from all lanes before this wave's leader frees a slot.
                rocdl.s_waitcnt(lgkmcnt=0)
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Release,
                    syncscope=rocdl.SyncScope.Workgroup,
                )
                if lane == 0:
                    fx.generic_store(
                        released + slot * consumers + cwave,
                        generation + 1,
                        memory_order=fx.AtomicOrdering.Release,
                        syncscope=rocdl.SyncScope.Workgroup,
                    )
            rows = thr.partition_C(fx.make_view(0, fx.make_layout((bm, bn), (1, 0))))
            cols = thr.partition_C(fx.make_view(0, fx.make_layout((bm, bn), (0, 1))))
            for i in range_constexpr(fx.size(fc.shape).unpack()):
                row = row0 + fx.get_scalar(rows[i])
                col = col0 + fx.get_scalar(cols[i])
                if (row < m) and (col < n):
                    if fx.const_expr(splits == 1):
                        fx.ptr_store(fc[i].to(fx.BFloat16), op + row * n + col)
                    else:
                        fx.ptr_store(fc[i], pp + split * m * n + row * n + col)

        if wave < producers:
            produce()
        else:
            consume()
        gpu.barrier()

    @flyc.kernel(known_block_size=[256, 1, 1])
    def reduce_ws(op: fx.Pointer, pp: fx.Pointer, count: fx.Int32):
        start = (fx.block_idx.x * 256 + fx.thread_idx.x) * 4
        if start < count:
            acc = fx.ptr_load(
                pp + start, result_type=fx.Vector.make_type(4, fx.Float32)
            )
            for s in range_constexpr(1, splits):
                acc = acc + fx.ptr_load(
                    pp + s * count + start,
                    result_type=fx.Vector.make_type(4, fx.Float32),
                )
            fx.ptr_store(acc.to(fx.BFloat16), op + start)

    @flyc.jit
    def launch(
        op: fx.Pointer,
        ap: fx.Pointer,
        bp: fx.Pointer,
        pp: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        lda: fx.Int32,
        ldb: fx.Int32,
        stream: fx.Stream,
    ):
        ws_kernel._known_block_size = [threads, 1, 1]
        ws_kernel._func.__name__ = f"hgemm_ws_v2_{bm}_{bn}_{bk}_s{stages}_p{producers}_c{wm}x{wn}_sk{splits}_pf{int(prefetch)}"
        ws_kernel(op, ap, bp, pp, m, n, k, lda, ldb).launch(
            grid=(((m + bm - 1) // bm) * ((n + bn - 1) // bn), splits, 1),
            block=(threads, 1, 1),
            stream=stream,
        )
        if fx.const_expr(splits > 1):
            reduce_ws(op, pp, m * n).launch(
                grid=((m * n + 1023) // 1024, 1, 1), block=(256, 1, 1), stream=stream
            )

    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        }
    }
    set_slp_vectorize_once_used(launch, False)
    return launch
