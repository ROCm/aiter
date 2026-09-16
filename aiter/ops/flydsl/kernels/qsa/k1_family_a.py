# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Family A FlyDSL QSA K1 (SILOTIGER-1047 2d): paged ReLU-sum + tiled top-512.

Streams compressed index-K from the paged cache in 512-slot tiles, scores
complete causal blocks, and merges a running LDS top-512. Writes
``block_ids [M, 512]``. Scores never land in a global ``[M, n_blocks]`` buffer.

When ``visible <= 512`` the selected set is every complete block: one
workgroup per row writes those ids. Each new tile is scored as a
``4 x 512 x 128`` BF16 GEMM (16x16x16 MFMA, ReLU-sum over heads), then
wave-sorted and merged across eight waves (LDS XOR for strides
``>= 64``, shuffle for 32..1). Decode rows with more than one 512-slot
tile split columns across eight workgroups (idle splits write ``-inf``
heaps) and pair-merge those sorted heaps. Prefill streams tiles in one
workgroup per row.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import BFloat16, Float32, Int32, gpu, range_constexpr

from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from aiter.ops.flydsl.kernels.qsa.shapes import FAMILY_A_INDEXER, FAMILY_A_SCORE_SCALE
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, buf_copy_atom

_BLOCK_THREADS = 512
_TILE = 512
_K = FAMILY_A_INDEXER.block_budget
_CANDIDATES = _K + _TILE
_SPLITS = 8
_MERGE = _SPLITS * _K
_H = FAMILY_A_INDEXER.n_heads
_D = FAMILY_A_INDEXER.head_dim
_R = FAMILY_A_INDEXER.compress_ratio
_VEC = 8
_Q_THREADS = _H * (_D // _VEC)
_WAVE = 64
_WAVES = _BLOCK_THREADS // _WAVE
_MFMA = 16
_N_ROUNDS = _TILE // (_WAVES * _MFMA)
_WAVE_STAGES = tuple(
    (span, stride)
    for span in (2, 4, 8, 16, 32, 64)
    for stride in tuple(1 << shift for shift in range(span.bit_length() - 2, -1, -1))
)
# Reverse-upper + LDS XOR for strides >= wave size. Strides 32..1 stay
# in-register (shuffle); they used to be one workgroup barrier each.
_INTERWAVE_LDS = (
    (128, ()),
    (256, (64,)),
    (512, (128, 64)),
)
_INTRAWAVE_XOR = (32, 16, 8, 4, 2, 1)
_PAIR_MERGE_STRIDES = tuple(1 << shift for shift in range(_K.bit_length() - 1, -1, -1))


def _idiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def _neg_inf():
    return Float32(float("-inf"))


def _mfma_bf16_16x16x16(a_elems, b_elems, acc):
    frag_a = fx.make_rmem_tensor(4, BFloat16)
    frag_b = fx.make_rmem_tensor(4, BFloat16)
    frag_c = fx.make_rmem_tensor(4, Float32)
    frag_a.store(fx.BFloat16x4(a_elems))
    frag_b.store(fx.BFloat16x4(b_elems))
    frag_c.store(acc)
    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, BFloat16))
    fx.gemm(mma, frag_c, frag_a, frag_b, frag_c)
    return fx.Float32x4(frag_c.load())


def build_qsa_k1_family_a_serial(page_size: int):
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if _CANDIDATES % _BLOCK_THREADS:
        raise ValueError("candidate buffer must be a multiple of block threads")
    if _K != _TILE:
        raise ValueError("family A local heap is one tile (k=512)")
    if _D % _VEC:
        raise ValueError("head dimension must be a multiple of vector width")
    if _D % _MFMA or _H > _MFMA or _N_ROUNDS != 4:
        raise ValueError("family A indexer GEMM needs 4x512x128 with 16x16x16 MFMA")
    tile_steps = _TILE // _BLOCK_THREADS
    candidate_steps = _CANDIDATES // _BLOCK_THREADS

    @fx.struct
    class SharedStorage:
        q: fx.Array[BFloat16, _H * _D, 16]
        k: fx.Array[BFloat16, _TILE * _MFMA, 16]
        cand_s: fx.Array[Float32, _CANDIDATES, 16]
        cand_c: fx.Array[Int32, _CANDIDATES, 16]

    @flyc.kernel(
        name="qsa_k1_family_a_serial_"
        + kernel_signature(
            ps=page_size,
            tile=_TILE,
            k=_K,
            h=_H,
            d=_D,
            blk=_BLOCK_THREADS,
            pair=1,
            wav=3,
            pipe=2,
            mma=1,
        ),
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def qsa_k1_family_a_serial(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        block_ids: fx.Tensor,
        n_columns: Int32,
        n_req: Int32,
        score_scale: Float32,
    ):
        row = Int32(gpu.block_id("x"))
        tid = Int32(gpu.thread_id("x"))
        zero = Int32(0)
        one = Int32(1)
        neg_one = Int32(-1)
        page = Int32(page_size)
        n_col = n_columns
        vec_layout = fx.make_layout(_VEC, 1)
        k_copy = buf_copy_atom(16, BFloat16)
        q_load = buf_copy_atom(16, BFloat16)
        q_store = fx.make_copy_atom(fx.UniversalCopy128b(), BFloat16)
        q_buf = fx.rocdl.make_buffer_tensor(q)
        k_buf = fx.rocdl.make_buffer_tensor(k_cache)
        q_tile, q_tv = fx.make_layout_tv(
            fx.make_layout((_H, _D // _VEC), (_D // _VEC, 1)),
            fx.make_layout((1, _VEC), (_VEC, 1)),
        )
        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        smem_q = storage.q.view(fx.make_layout((_H, _D), (_D, 1)))
        smem_k = storage.k.view(fx.make_layout((_TILE, _MFMA), (_MFMA, 1)))
        cand_s = storage.cand_s.view(fx.make_layout(_CANDIDATES, 1))
        cand_c = storage.cand_c.view(fx.make_layout(_CANDIDATES, 1))
        req = token_to_req[row]
        valid_req = (req >= zero) & (req < n_req)
        safe_req = valid_req.select(req, zero)
        qpos = query_positions[row]
        slen = valid_req.select(context_lens[safe_req], zero)
        vis_q = _idiv(qpos + one, Int32(_R))
        vis_s = _idiv(slen, Int32(_R))
        visible = (vis_q < vis_s).select(vis_q, vis_s)
        scored = (visible < n_col).select(visible, n_col)
        n_tiles = fx.ceildiv(scored, Int32(_TILE))
        lane = gpu.lane_id()
        wave = tid // Int32(_WAVE)
        z16 = BFloat16(0)

        def better(s, c, bs, bc):
            return (c >= zero) & ((s > bs) | ((s == bs) & ((bc < zero) | (c < bc))))

        if visible > Int32(_K):
            if tid < Int32(_Q_THREADS):
                q_thr = fx.make_tiled_copy(q_load, q_tv, q_tile).get_slice(tid)
                q_row = fx.slice(q_buf, (row, None, None))
                q_block = fx.slice(fx.zipped_divide(q_row, q_tile), (None, (0, 0)))
                q_src = q_thr.partition_S(q_block)
                q_dst = q_thr.partition_D(smem_q)
                q_frag = fx.make_fragment_like(q_src)
                fx.copy(q_load, q_src, q_frag)
                fx.copy(q_store, q_frag, q_dst)
            for t in range_constexpr(candidate_steps):
                j = tid + Int32(t * _BLOCK_THREADS)
                cand_s[j] = _neg_inf()
                cand_c[j] = neg_one
            gpu.barrier()
            for tile in range(zero, n_tiles, one):
                tile_base = tile * Int32(_TILE)
                acc0 = fx.Float32x4(0.0)
                acc1 = fx.Float32x4(0.0)
                acc2 = fx.Float32x4(0.0)
                acc3 = fx.Float32x4(0.0)
                for kt in range_constexpr(_D // _MFMA):
                    local = tid
                    col = tile_base + local
                    live = (col < n_col) & (col < visible) & valid_req
                    k_dst_chunks = fx.logical_divide(
                        fx.slice(smem_k, (local, None)), vec_layout
                    )
                    if live:
                        logical_page = _idiv(col, page)
                        off = col - logical_page * page
                        phys = page_table[safe_req, logical_page]
                        k_chunks = fx.logical_divide(
                            fx.slice(k_buf, (phys, off, zero, None)), vec_layout
                        )
                        for v in range_constexpr(_MFMA // _VEC):
                            src = fx.slice(k_chunks, (None, kt * (_MFMA // _VEC) + v))
                            dst = fx.slice(k_dst_chunks, (None, v))
                            frag = fx.make_fragment_like(src)
                            fx.copy(k_copy, src, frag)
                            fx.copy(q_store, frag, dst)
                    else:
                        for v in range_constexpr(_MFMA):
                            smem_k[local, Int32(v)] = z16
                    gpu.barrier()
                    m_a = lane % Int32(_MFMA)
                    k0 = (lane // Int32(_MFMA)) * Int32(4)
                    is_q = m_a < Int32(_H)
                    sm = is_q.select(m_a, zero)
                    kd = Int32(kt * _MFMA) + k0
                    a_elems = [
                        is_q.select(smem_q[sm, kd], z16),
                        is_q.select(smem_q[sm, kd + one], z16),
                        is_q.select(smem_q[sm, kd + Int32(2)], z16),
                        is_q.select(smem_q[sm, kd + Int32(3)], z16),
                    ]

                    def k_elems(n_round):
                        n_in = lane % Int32(_MFMA)
                        brow = (
                            Int32(n_round * _WAVES * _MFMA) + wave * Int32(_MFMA) + n_in
                        )
                        return [
                            smem_k[brow, k0],
                            smem_k[brow, k0 + one],
                            smem_k[brow, k0 + Int32(2)],
                            smem_k[brow, k0 + Int32(3)],
                        ]

                    acc0 = _mfma_bf16_16x16x16(a_elems, k_elems(0), acc0)
                    acc1 = _mfma_bf16_16x16x16(a_elems, k_elems(1), acc1)
                    acc2 = _mfma_bf16_16x16x16(a_elems, k_elems(2), acc2)
                    acc3 = _mfma_bf16_16x16x16(a_elems, k_elems(3), acc3)
                    gpu.barrier()
                if lane < Int32(_MFMA):
                    accs = (acc0, acc1, acc2, acc3)
                    n_in = lane
                    for nr in range_constexpr(_N_ROUNDS):
                        acc = accs[nr]
                        total = (
                            acc[0].maximumf(Float32(0.0))
                            + acc[1].maximumf(Float32(0.0))
                            + acc[2].maximumf(Float32(0.0))
                            + acc[3].maximumf(Float32(0.0))
                        )
                        local = Int32(nr * _WAVES * _MFMA) + wave * Int32(_MFMA) + n_in
                        col = tile_base + local
                        live = (col < n_col) & (col < visible) & valid_req
                        cand_s[Int32(_K) + local] = live.select(
                            total * score_scale, _neg_inf()
                        )
                        cand_c[Int32(_K) + local] = live.select(col, neg_one)
                gpu.barrier()
                ws = cand_s[Int32(_K) + tid]
                wc = cand_c[Int32(_K) + tid]
                for span, stride in _WAVE_STAGES:
                    ps = ws.shuffle_xor(stride, _WAVE)
                    pc = wc.shuffle_xor(stride, _WAVE)
                    is_lo = lane < (lane ^ Int32(stride))
                    best_first = (lane & Int32(span)) == zero
                    take_peer = is_lo.select(
                        best_first.select(
                            better(ps, pc, ws, wc),
                            better(ws, wc, ps, pc),
                        ),
                        best_first.select(
                            better(ws, wc, ps, pc),
                            better(ps, pc, ws, wc),
                        ),
                    )
                    ws = take_peer.select(ps, ws)
                    wc = take_peer.select(pc, wc)
                cand_s[Int32(_K) + tid] = ws
                cand_c[Int32(_K) + tid] = wc
                gpu.barrier()
                for win_size, lds_strides in _INTERWAVE_LDS:
                    half = win_size // 2
                    wbase = (tid // Int32(win_size)) * Int32(win_size)
                    local = tid - wbase
                    if local < Int32(half):
                        j = Int32(_K) + tid
                        peer = Int32(_K) + wbase + Int32(win_size - 1) - local
                        s0 = cand_s[j]
                        c0 = cand_c[j]
                        s1 = cand_s[peer]
                        c1 = cand_c[peer]
                        swap = better(s1, c1, s0, c0)
                        cand_s[j] = swap.select(s1, s0)
                        cand_c[j] = swap.select(c1, c0)
                        cand_s[peer] = swap.select(s0, s1)
                        cand_c[peer] = swap.select(c0, c1)
                    gpu.barrier()
                    for stride in lds_strides:
                        peer_local = tid ^ Int32(stride)
                        if tid < peer_local:
                            j = Int32(_K) + tid
                            peer = Int32(_K) + peer_local
                            s0 = cand_s[j]
                            c0 = cand_c[j]
                            s1 = cand_s[peer]
                            c1 = cand_c[peer]
                            swap = better(s1, c1, s0, c0)
                            cand_s[j] = swap.select(s1, s0)
                            cand_c[j] = swap.select(c1, c0)
                            cand_s[peer] = swap.select(s0, s1)
                            cand_c[peer] = swap.select(c0, c1)
                        gpu.barrier()
                    xs = cand_s[Int32(_K) + tid]
                    xc = cand_c[Int32(_K) + tid]
                    for stride in _INTRAWAVE_XOR:
                        ps = xs.shuffle_xor(stride, _WAVE)
                        pc = xc.shuffle_xor(stride, _WAVE)
                        is_lo = lane < (lane ^ Int32(stride))
                        take_peer = is_lo.select(
                            better(ps, pc, xs, xc),
                            better(xs, xc, ps, pc),
                        )
                        xs = take_peer.select(ps, xs)
                        xc = take_peer.select(pc, xc)
                    cand_s[Int32(_K) + tid] = xs
                    cand_c[Int32(_K) + tid] = xc
                    gpu.barrier()
                if tile == zero:
                    for t in range_constexpr(tile_steps):
                        local = tid + Int32(t * _BLOCK_THREADS)
                        cand_s[local] = cand_s[Int32(_K) + local]
                        cand_c[local] = cand_c[Int32(_K) + local]
                else:
                    for t in range_constexpr(tile_steps):
                        local = tid + Int32(t * _BLOCK_THREADS)
                        if local < Int32(_K // 2):
                            a = Int32(_K) + local
                            b = Int32(_CANDIDATES - 1) - local
                            sa = cand_s[a]
                            ca = cand_c[a]
                            sb = cand_s[b]
                            cb = cand_c[b]
                            cand_s[a] = sb
                            cand_c[a] = cb
                            cand_s[b] = sa
                            cand_c[b] = ca
                    gpu.barrier()
                    for stride in _PAIR_MERGE_STRIDES:
                        for t in range_constexpr(candidate_steps):
                            j = tid + Int32(t * _BLOCK_THREADS)
                            peer = j ^ Int32(stride)
                            if j < peer:
                                s0 = cand_s[j]
                                c0 = cand_c[j]
                                s1 = cand_s[peer]
                                c1 = cand_c[peer]
                                swap = better(s1, c1, s0, c0)
                                cand_s[j] = swap.select(s1, s0)
                                cand_c[j] = swap.select(c1, c0)
                                cand_s[peer] = swap.select(s0, s1)
                                cand_c[peer] = swap.select(c0, c1)
                        gpu.barrier()
                gpu.barrier()
            for t in range_constexpr(tile_steps):
                j = tid + Int32(t * _BLOCK_THREADS)
                block_ids[row, j] = cand_c[j]
        else:
            for t in range_constexpr(tile_steps):
                j = tid + Int32(t * _BLOCK_THREADS)
                take = (j < visible) & (j < n_col) & valid_req
                block_ids[row, j] = take.select(j, neg_one)

    @flyc.jit
    def launch_serial(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        block_ids: fx.Tensor,
        n_columns: Int32,
        n_req: Int32,
        score_scale: Float32,
        rows: Int32,
        stream: fx.Stream,
    ):
        qsa_k1_family_a_serial(
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            block_ids,
            n_columns,
            n_req,
            score_scale,
        ).launch(
            grid=(rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_serial


def build_qsa_k1_family_a_split_merge(page_size: int):
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if _CANDIDATES % _BLOCK_THREADS or _MERGE % _BLOCK_THREADS:
        raise ValueError("candidate buffers must be multiples of block threads")
    if _K != _TILE:
        raise ValueError("family A local heap is one tile (k=512)")
    if _D % _VEC:
        raise ValueError("head dimension must be a multiple of vector width")
    if _D % _MFMA or _H > _MFMA or _N_ROUNDS != 4:
        raise ValueError("family A indexer GEMM needs 4x512x128 with 16x16x16 MFMA")
    if _SPLITS & (_SPLITS - 1) or _CANDIDATES != 2 * _K:
        raise ValueError("heap tree merge needs 2^n sorted 512-heaps")
    tile_steps = _TILE // _BLOCK_THREADS
    candidate_steps = _CANDIDATES // _BLOCK_THREADS
    pair_steps = _CANDIDATES // _BLOCK_THREADS
    sig = kernel_signature(
        ps=page_size,
        tile=_TILE,
        k=_K,
        h=_H,
        d=_D,
        blk=_BLOCK_THREADS,
        spl=_SPLITS,
        pair=2,
        wav=3,
        pipe=2,
        liv=1,
        mma=1,
    )

    @fx.struct
    class SplitStorage:
        q: fx.Array[BFloat16, _H * _D, 16]
        k: fx.Array[BFloat16, _TILE * _MFMA, 16]
        cand_s: fx.Array[Float32, _CANDIDATES, 16]
        cand_c: fx.Array[Int32, _CANDIDATES, 16]

    @fx.struct
    class MergeStorage:
        cand_s: fx.Array[Float32, _MERGE, 16]
        cand_c: fx.Array[Int32, _MERGE, 16]

    @flyc.kernel(
        name="qsa_k1_family_a_split_" + sig,
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def qsa_k1_family_a_split(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        heap_s: fx.Tensor,
        heap_c: fx.Tensor,
        n_columns: Int32,
        n_req: Int32,
        score_scale: Float32,
    ):
        row = Int32(gpu.block_id("x"))
        split = Int32(gpu.block_id("y"))
        tid = Int32(gpu.thread_id("x"))
        zero = Int32(0)
        one = Int32(1)
        neg_one = Int32(-1)
        page = Int32(page_size)
        n_col = n_columns
        vec_layout = fx.make_layout(_VEC, 1)
        k_copy = buf_copy_atom(16, BFloat16)
        q_load = buf_copy_atom(16, BFloat16)
        q_store = fx.make_copy_atom(fx.UniversalCopy128b(), BFloat16)
        q_buf = fx.rocdl.make_buffer_tensor(q)
        k_buf = fx.rocdl.make_buffer_tensor(k_cache)
        q_tile, q_tv = fx.make_layout_tv(
            fx.make_layout((_H, _D // _VEC), (_D // _VEC, 1)),
            fx.make_layout((1, _VEC), (_VEC, 1)),
        )
        storage = fx.SharedAllocator().allocate(SplitStorage).peek()
        smem_q = storage.q.view(fx.make_layout((_H, _D), (_D, 1)))
        smem_k = storage.k.view(fx.make_layout((_TILE, _MFMA), (_MFMA, 1)))
        cand_s = storage.cand_s.view(fx.make_layout(_CANDIDATES, 1))
        cand_c = storage.cand_c.view(fx.make_layout(_CANDIDATES, 1))

        req = token_to_req[row]
        valid_req = (req >= zero) & (req < n_req)
        safe_req = valid_req.select(req, zero)
        qpos = query_positions[row]
        slen = valid_req.select(context_lens[safe_req], zero)
        vis_q = _idiv(qpos + one, Int32(_R))
        vis_s = _idiv(slen, Int32(_R))
        visible = (vis_q < vis_s).select(vis_q, vis_s)
        scored = (visible < n_col).select(visible, n_col)
        n_tiles = fx.ceildiv(scored, Int32(_TILE))
        tiles_per = fx.ceildiv(n_tiles, Int32(_SPLITS))
        start = split * tiles_per
        end_raw = start + tiles_per
        end = (end_raw < n_tiles).select(end_raw, n_tiles)
        start = (start < n_tiles).select(start, n_tiles)
        lane = gpu.lane_id()
        wave = tid // Int32(_WAVE)
        z16 = BFloat16(0)

        def better(s, c, bs, bc):
            return (c >= zero) & ((s > bs) | ((s == bs) & ((bc < zero) | (c < bc))))

        if visible > Int32(_K):
            if tid < Int32(_Q_THREADS):
                q_thr = fx.make_tiled_copy(q_load, q_tv, q_tile).get_slice(tid)
                q_row = fx.slice(q_buf, (row, None, None))
                q_block = fx.slice(fx.zipped_divide(q_row, q_tile), (None, (0, 0)))
                q_src = q_thr.partition_S(q_block)
                q_dst = q_thr.partition_D(smem_q)
                q_frag = fx.make_fragment_like(q_src)
                fx.copy(q_load, q_src, q_frag)
                fx.copy(q_store, q_frag, q_dst)
            for t in range_constexpr(candidate_steps):
                j = tid + Int32(t * _BLOCK_THREADS)
                cand_s[j] = _neg_inf()
                cand_c[j] = neg_one
            gpu.barrier()

            for tile in range(start, end, one):
                tile_base = tile * Int32(_TILE)
                acc0 = fx.Float32x4(0.0)
                acc1 = fx.Float32x4(0.0)
                acc2 = fx.Float32x4(0.0)
                acc3 = fx.Float32x4(0.0)
                for kt in range_constexpr(_D // _MFMA):
                    local = tid
                    col = tile_base + local
                    live = (col < n_col) & (col < visible) & valid_req
                    k_dst_chunks = fx.logical_divide(
                        fx.slice(smem_k, (local, None)), vec_layout
                    )
                    if live:
                        logical_page = _idiv(col, page)
                        off = col - logical_page * page
                        phys = page_table[safe_req, logical_page]
                        k_chunks = fx.logical_divide(
                            fx.slice(k_buf, (phys, off, zero, None)), vec_layout
                        )
                        for v in range_constexpr(_MFMA // _VEC):
                            src = fx.slice(k_chunks, (None, kt * (_MFMA // _VEC) + v))
                            dst = fx.slice(k_dst_chunks, (None, v))
                            frag = fx.make_fragment_like(src)
                            fx.copy(k_copy, src, frag)
                            fx.copy(q_store, frag, dst)
                    else:
                        for v in range_constexpr(_MFMA):
                            smem_k[local, Int32(v)] = z16
                    gpu.barrier()
                    m_a = lane % Int32(_MFMA)
                    k0 = (lane // Int32(_MFMA)) * Int32(4)
                    is_q = m_a < Int32(_H)
                    sm = is_q.select(m_a, zero)
                    kd = Int32(kt * _MFMA) + k0
                    a_elems = [
                        is_q.select(smem_q[sm, kd], z16),
                        is_q.select(smem_q[sm, kd + one], z16),
                        is_q.select(smem_q[sm, kd + Int32(2)], z16),
                        is_q.select(smem_q[sm, kd + Int32(3)], z16),
                    ]

                    def k_elems(n_round):
                        n_in = lane % Int32(_MFMA)
                        brow = (
                            Int32(n_round * _WAVES * _MFMA) + wave * Int32(_MFMA) + n_in
                        )
                        return [
                            smem_k[brow, k0],
                            smem_k[brow, k0 + one],
                            smem_k[brow, k0 + Int32(2)],
                            smem_k[brow, k0 + Int32(3)],
                        ]

                    acc0 = _mfma_bf16_16x16x16(a_elems, k_elems(0), acc0)
                    acc1 = _mfma_bf16_16x16x16(a_elems, k_elems(1), acc1)
                    acc2 = _mfma_bf16_16x16x16(a_elems, k_elems(2), acc2)
                    acc3 = _mfma_bf16_16x16x16(a_elems, k_elems(3), acc3)
                    gpu.barrier()
                if lane < Int32(_MFMA):
                    accs = (acc0, acc1, acc2, acc3)
                    n_in = lane
                    for nr in range_constexpr(_N_ROUNDS):
                        acc = accs[nr]
                        total = (
                            acc[0].maximumf(Float32(0.0))
                            + acc[1].maximumf(Float32(0.0))
                            + acc[2].maximumf(Float32(0.0))
                            + acc[3].maximumf(Float32(0.0))
                        )
                        local = Int32(nr * _WAVES * _MFMA) + wave * Int32(_MFMA) + n_in
                        col = tile_base + local
                        live = (col < n_col) & (col < visible) & valid_req
                        cand_s[Int32(_K) + local] = live.select(
                            total * score_scale, _neg_inf()
                        )
                        cand_c[Int32(_K) + local] = live.select(col, neg_one)
                gpu.barrier()
                ws = cand_s[Int32(_K) + tid]
                wc = cand_c[Int32(_K) + tid]
                for span, stride in _WAVE_STAGES:
                    ps = ws.shuffle_xor(stride, _WAVE)
                    pc = wc.shuffle_xor(stride, _WAVE)
                    is_lo = lane < (lane ^ Int32(stride))
                    best_first = (lane & Int32(span)) == zero
                    take_peer = is_lo.select(
                        best_first.select(
                            better(ps, pc, ws, wc),
                            better(ws, wc, ps, pc),
                        ),
                        best_first.select(
                            better(ws, wc, ps, pc),
                            better(ps, pc, ws, wc),
                        ),
                    )
                    ws = take_peer.select(ps, ws)
                    wc = take_peer.select(pc, wc)
                cand_s[Int32(_K) + tid] = ws
                cand_c[Int32(_K) + tid] = wc
                gpu.barrier()
                for win_size, lds_strides in _INTERWAVE_LDS:
                    half = win_size // 2
                    wbase = (tid // Int32(win_size)) * Int32(win_size)
                    local = tid - wbase
                    if local < Int32(half):
                        j = Int32(_K) + tid
                        peer = Int32(_K) + wbase + Int32(win_size - 1) - local
                        s0 = cand_s[j]
                        c0 = cand_c[j]
                        s1 = cand_s[peer]
                        c1 = cand_c[peer]
                        swap = better(s1, c1, s0, c0)
                        cand_s[j] = swap.select(s1, s0)
                        cand_c[j] = swap.select(c1, c0)
                        cand_s[peer] = swap.select(s0, s1)
                        cand_c[peer] = swap.select(c0, c1)
                    gpu.barrier()
                    for stride in lds_strides:
                        peer_local = tid ^ Int32(stride)
                        if tid < peer_local:
                            j = Int32(_K) + tid
                            peer = Int32(_K) + peer_local
                            s0 = cand_s[j]
                            c0 = cand_c[j]
                            s1 = cand_s[peer]
                            c1 = cand_c[peer]
                            swap = better(s1, c1, s0, c0)
                            cand_s[j] = swap.select(s1, s0)
                            cand_c[j] = swap.select(c1, c0)
                            cand_s[peer] = swap.select(s0, s1)
                            cand_c[peer] = swap.select(c0, c1)
                        gpu.barrier()
                    xs = cand_s[Int32(_K) + tid]
                    xc = cand_c[Int32(_K) + tid]
                    for stride in _INTRAWAVE_XOR:
                        ps = xs.shuffle_xor(stride, _WAVE)
                        pc = xc.shuffle_xor(stride, _WAVE)
                        is_lo = lane < (lane ^ Int32(stride))
                        take_peer = is_lo.select(
                            better(ps, pc, xs, xc),
                            better(xs, xc, ps, pc),
                        )
                        xs = take_peer.select(ps, xs)
                        xc = take_peer.select(pc, xc)
                    cand_s[Int32(_K) + tid] = xs
                    cand_c[Int32(_K) + tid] = xc
                    gpu.barrier()
                if tile == start:
                    for t in range_constexpr(tile_steps):
                        local = tid + Int32(t * _BLOCK_THREADS)
                        cand_s[local] = cand_s[Int32(_K) + local]
                        cand_c[local] = cand_c[Int32(_K) + local]
                else:
                    for t in range_constexpr(tile_steps):
                        local = tid + Int32(t * _BLOCK_THREADS)
                        if local < Int32(_K // 2):
                            a = Int32(_K) + local
                            b = Int32(_CANDIDATES - 1) - local
                            sa = cand_s[a]
                            ca = cand_c[a]
                            sb = cand_s[b]
                            cb = cand_c[b]
                            cand_s[a] = sb
                            cand_c[a] = cb
                            cand_s[b] = sa
                            cand_c[b] = ca
                    gpu.barrier()
                    for stride in _PAIR_MERGE_STRIDES:
                        for t in range_constexpr(candidate_steps):
                            j = tid + Int32(t * _BLOCK_THREADS)
                            peer = j ^ Int32(stride)
                            if j < peer:
                                s0 = cand_s[j]
                                c0 = cand_c[j]
                                s1 = cand_s[peer]
                                c1 = cand_c[peer]
                                swap = better(s1, c1, s0, c0)
                                cand_s[j] = swap.select(s1, s0)
                                cand_c[j] = swap.select(c1, c0)
                                cand_s[peer] = swap.select(s0, s1)
                                cand_c[peer] = swap.select(c0, c1)
                        gpu.barrier()
                gpu.barrier()

            for t in range_constexpr(tile_steps):
                j = tid + Int32(t * _BLOCK_THREADS)
                heap_s[row, split, j] = cand_s[j]
                heap_c[row, split, j] = cand_c[j]
        else:
            for t in range_constexpr(tile_steps):
                j = tid + Int32(t * _BLOCK_THREADS)
                heap_s[row, split, j] = _neg_inf()
                heap_c[row, split, j] = neg_one

    @flyc.kernel(
        name="qsa_k1_family_a_merge_" + sig,
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def qsa_k1_family_a_merge(
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        heap_s: fx.Tensor,
        heap_c: fx.Tensor,
        block_ids: fx.Tensor,
        n_columns: Int32,
        n_req: Int32,
    ):
        row = Int32(gpu.block_id("x"))
        tid = Int32(gpu.thread_id("x"))
        zero = Int32(0)
        one = Int32(1)
        neg_one = Int32(-1)
        req = token_to_req[row]
        valid_req = (req >= zero) & (req < n_req)
        safe_req = valid_req.select(req, zero)
        qpos = query_positions[row]
        slen = valid_req.select(context_lens[safe_req], zero)
        vis_q = _idiv(qpos + one, Int32(_R))
        vis_s = _idiv(slen, Int32(_R))
        visible = (vis_q < vis_s).select(vis_q, vis_s)
        scored = (visible < n_columns).select(visible, n_columns)
        n_tiles = fx.ceildiv(scored, Int32(_TILE))
        live = (n_tiles < Int32(_SPLITS)).select(n_tiles, Int32(_SPLITS))
        storage = fx.SharedAllocator().allocate(MergeStorage).peek()
        cand_s = storage.cand_s.view(fx.make_layout(_MERGE, 1))
        cand_c = storage.cand_c.view(fx.make_layout(_MERGE, 1))

        def better(s, c, bs, bc):
            return (c >= zero) & ((s > bs) | ((s == bs) & ((bc < zero) | (c < bc))))

        if visible > Int32(_K):
            for s in range_constexpr(_SPLITS):
                for t in range_constexpr(tile_steps):
                    j = tid + Int32(t * _BLOCK_THREADS)
                    cand_s[Int32(s * _K) + j] = heap_s[row, s, j]
                    cand_c[Int32(s * _K) + j] = heap_c[row, s, j]
            gpu.barrier()
            for n_win in (4, 2, 1):
                take = live > Int32(n_win)
                if take:
                    for w in range_constexpr(n_win):
                        base = Int32(w * _CANDIDATES)
                        for t in range_constexpr(tile_steps):
                            local = tid + Int32(t * _BLOCK_THREADS)
                            if local < Int32(_K // 2):
                                a = base + Int32(_K) + local
                                b = base + Int32(_CANDIDATES - 1) - local
                                sa = cand_s[a]
                                ca = cand_c[a]
                                sb = cand_s[b]
                                cb = cand_c[b]
                                cand_s[a] = sb
                                cand_c[a] = cb
                                cand_s[b] = sa
                                cand_c[b] = ca
                    gpu.barrier()
                    for stride in _PAIR_MERGE_STRIDES:
                        for w in range_constexpr(n_win):
                            base = Int32(w * _CANDIDATES)
                            for t in range_constexpr(pair_steps):
                                local = tid + Int32(t * _BLOCK_THREADS)
                                peer_local = local ^ Int32(stride)
                                if local < peer_local:
                                    j = base + local
                                    peer = base + peer_local
                                    s0 = cand_s[j]
                                    c0 = cand_c[j]
                                    s1 = cand_s[peer]
                                    c1 = cand_c[peer]
                                    swap = better(s1, c1, s0, c0)
                                    cand_s[j] = swap.select(s1, s0)
                                    cand_c[j] = swap.select(c1, c0)
                                    cand_s[peer] = swap.select(s0, s1)
                                    cand_c[peer] = swap.select(c0, c1)
                        gpu.barrier()
                    if n_win == 4:
                        for t in range_constexpr(tile_steps):
                            local = tid + Int32(t * _BLOCK_THREADS)
                            s1 = cand_s[Int32(_CANDIDATES) + local]
                            c1 = cand_c[Int32(_CANDIDATES) + local]
                            s2 = cand_s[Int32(2 * _CANDIDATES) + local]
                            c2 = cand_c[Int32(2 * _CANDIDATES) + local]
                            s3 = cand_s[Int32(3 * _CANDIDATES) + local]
                            c3 = cand_c[Int32(3 * _CANDIDATES) + local]
                            cand_s[Int32(_K) + local] = s1
                            cand_c[Int32(_K) + local] = c1
                            cand_s[Int32(2 * _K) + local] = s2
                            cand_c[Int32(2 * _K) + local] = c2
                            cand_s[Int32(3 * _K) + local] = s3
                            cand_c[Int32(3 * _K) + local] = c3
                        gpu.barrier()
                    if n_win == 2:
                        for t in range_constexpr(tile_steps):
                            local = tid + Int32(t * _BLOCK_THREADS)
                            cand_s[Int32(_K) + local] = cand_s[
                                Int32(_CANDIDATES) + local
                            ]
                            cand_c[Int32(_K) + local] = cand_c[
                                Int32(_CANDIDATES) + local
                            ]
                        gpu.barrier()
            for t in range_constexpr(tile_steps):
                j = tid + Int32(t * _BLOCK_THREADS)
                block_ids[row, j] = cand_c[j]
        else:
            for t in range_constexpr(tile_steps):
                j = tid + Int32(t * _BLOCK_THREADS)
                take = (j < visible) & (j < n_columns) & valid_req
                block_ids[row, j] = take.select(j, neg_one)

    @flyc.jit
    def launch_split_merge(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        heap_s: fx.Tensor,
        heap_c: fx.Tensor,
        block_ids: fx.Tensor,
        n_columns: Int32,
        n_req: Int32,
        score_scale: Float32,
        rows: Int32,
        stream: fx.Stream,
    ):
        qsa_k1_family_a_split(
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            heap_s,
            heap_c,
            n_columns,
            n_req,
            score_scale,
        ).launch(
            grid=(rows, _SPLITS, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        qsa_k1_family_a_merge(
            token_to_req,
            query_positions,
            context_lens,
            heap_s,
            heap_c,
            block_ids,
            n_columns,
            n_req,
        ).launch(
            grid=(rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_split_merge


@lru_cache(maxsize=8)
def _plan_serial(page_size: int):
    return build_qsa_k1_family_a_serial(page_size)


@lru_cache(maxsize=8)
def _plan_split_merge(page_size: int):
    return build_qsa_k1_family_a_split_merge(page_size)


def qsa_k1_family_a_serves(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
) -> str | None:
    """Why this K1 kernel cannot serve these tensors, or None if it can."""
    idx = FAMILY_A_INDEXER
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        return f"q and k_cache must be bfloat16, got {q.dtype} and {k_cache.dtype}"
    if q.dim() != 3 or q.shape[1] != idx.n_heads or q.shape[2] != idx.head_dim:
        return f"q must be [M, {idx.n_heads}, {idx.head_dim}], got {tuple(q.shape)}"
    if k_cache.dim() != 4:
        return f"k_cache must be [pages, page_size, H, D], got {tuple(k_cache.shape)}"
    if k_cache.shape[2] != idx.kv_heads or k_cache.shape[3] != idx.head_dim:
        return (
            f"k_cache KV/D must be ({idx.kv_heads}, {idx.head_dim}), "
            f"got {k_cache.shape[2:]}"
        )
    if page_table.dim() != 2 or page_table.dtype != torch.int32:
        return (
            f"page_table must be int32 [n_req, n_pages], got {tuple(page_table.shape)}"
        )
    return None


def qsa_k1_family_a_block_ids(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor | None = None,
    score_scale: float = FAMILY_A_SCORE_SCALE,
) -> torch.Tensor:
    """Write family A indexer ``block_ids [M, 512]`` from paged compressed K.

    Short rows emit complete-block ids. Decode rows with more than one
    512-slot tile split columns across eight workgroups and pair-merge
    those sorted heaps. Prefill streams tiles in one workgroup per row.
    Does not allocate a score matrix. Expand+tail is still a separate
    launch.
    """
    reason = qsa_k1_family_a_serves(q, k_cache, page_table)
    if reason is not None:
        raise ValueError(f"[FlyDSL qsa_k1_family_a] {reason}")
    m = q.shape[0]
    if token_to_req.shape != (m,) or token_to_req.dtype != torch.int32:
        raise ValueError(f"token_to_req must be int32 [{m}]")
    if query_positions.shape != (m,) or query_positions.dtype != torch.int32:
        raise ValueError(f"query_positions must be int32 [{m}]")
    if context_lens.dim() != 1 or context_lens.dtype != torch.int32:
        raise ValueError("context_lens must be 1-D int32")
    if out is None:
        out = torch.empty(m, _K, dtype=torch.int32, device=q.device)
    elif out.shape != (m, _K) or out.dtype != torch.int32:
        raise ValueError(f"out must be int32 [{m}, {_K}], got {tuple(out.shape)}")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    tensors = (q, k_cache, page_table, token_to_req, query_positions, context_lens, out)
    if any(not t.is_cuda for t in tensors):
        raise ValueError("every tensor must be on the GPU")
    if any(t.device != q.device for t in tensors[1:]):
        raise ValueError("every tensor must be on the same GPU")
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    page_table = page_table.contiguous()
    token_to_req = token_to_req.contiguous()
    query_positions = query_positions.contiguous()
    context_lens = context_lens.contiguous()
    page_size = k_cache.shape[1]
    n_columns = page_table.shape[1] * page_size
    n_req = int(context_lens.shape[0])
    stream = torch.cuda.current_stream(q.device)
    # Decode: split as soon as a row can have more than one tile. Idle
    # splits write -inf heaps; the tree merge is cheap. Prefill already
    # fills the GPU with one workgroup per row, so it stays serial.
    # Short rows still emit inside the serial kernel.
    if n_columns <= _TILE or m > _SPLITS:
        _run_compiled(
            _plan_serial(page_size),
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            out,
            int(n_columns),
            n_req,
            float(score_scale),
            m,
            stream,
        )
        return out
    heap_s = torch.empty(m, _SPLITS, _K, dtype=torch.float32, device=q.device)
    heap_c = torch.empty(m, _SPLITS, _K, dtype=torch.int32, device=q.device)
    _run_compiled(
        _plan_split_merge(page_size),
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        context_lens,
        heap_s,
        heap_c,
        out,
        int(n_columns),
        n_req,
        float(score_scale),
        m,
        stream,
    )
    return out
