# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Family A FlyDSL QSA K2 (SILOTIGER-1047 3d): tiled sparse GQA.

One workgroup per ``(row, kv_head, split)`` gathers ``BLOCK_N`` paged K/V
with 128-bit D-chunks (one page translate per column owner) and uses MFMA
for both QK and PV (group padded to 16). The next K/V tile is prefetched
into registers before current-tile QK and carried across the runtime loop.
Wave 0 runs softmax straight off the reduced QK C fragment, so no ``s`` LDS
tile is needed. gfx950 uses K32 QK; gfx942 keeps K16. Split-K uses BF16
partial outputs, and one merge wave computes the LSE weights. Decode and
prefill share this instantiation. Sigmoid and expand+tail stay unfused.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import BFloat16, Float32, Int32, gpu, range_constexpr
from flydsl.expr import math as fxmath

from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from aiter.ops.flydsl.kernels.qsa.shapes import FAMILY_A_GQA
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, buf_copy_atom

_BLOCK_THREADS = 256
_BLOCK_N = 16
_HQ = FAMILY_A_GQA.n_heads
_HK = FAMILY_A_GQA.kv_heads
_GROUP = FAMILY_A_GQA.group_size
_D = FAMILY_A_GQA.head_dim
_DEFAULT_SCALE = _D**-0.5
_LSE_EMPTY = -1.0e20


def _idiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def _neg_inf():
    return Float32(float("-inf"))


def _choose_splits(rows: int, n_sel: int) -> int:
    """Keep decode occupancy; cap prefill serial tiles at ~4 per split."""
    if n_sel < 1:
        return 1
    tiles = (n_sel + _BLOCK_N - 1) // _BLOCK_N
    base = rows * _HK
    if base <= 8:
        target = 64
    elif base < 32:
        target = 32
    elif base <= 256:
        target = 8
    elif base <= 512:
        target = 4
    else:
        target = max(1, tiles // 4)
        target = min(8, target)
    return max(1, min(target, tiles, n_sel))


def build_qsa_k2_family_a_module(page_size: int, use_k32: bool):
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if _D != _BLOCK_THREADS:
        raise ValueError("decode K2 maps one thread per D element")
    if _HQ != _HK * _GROUP:
        raise ValueError("family A GQA head counts do not form groups")

    _HEAD_PAD = 16
    _QK_K = 32 if use_k32 else 16
    _QK_VEC = _QK_K // 4
    _K_STEPS = 64 // _QK_K
    _N_SUBTILES = _BLOCK_N // 16
    _VEC = 8
    _VEC_CHUNKS = _D // _VEC
    _CHUNK_STRIDE = _VEC_CHUNKS // 2

    @fx.struct
    class SharedStorage:
        q: fx.Array[BFloat16, _HEAD_PAD * _D, 16]
        k: fx.Array[BFloat16, _BLOCK_N * _D, 16]
        v: fx.Array[BFloat16, _D * _BLOCK_N, 16]
        p: fx.Array[BFloat16, _HEAD_PAD * _BLOCK_N, 16]
        live: fx.Array[Int32, _BLOCK_N, 16]
        m: fx.Array[Float32, _HEAD_PAD, 16]
        l: fx.Array[Float32, _HEAD_PAD, 16]
        alpha: fx.Array[Float32, _HEAD_PAD, 16]
        c: fx.Array[Float32, _N_SUBTILES * 4 * 64 * 4, 16]

    @fx.struct
    class MergeStorage:
        weights: fx.Array[Float32, 64, 16]
        denominator: fx.Array[Float32, 1, 16]

    @flyc.kernel(
        name="qsa_k2_family_a_split_"
        + kernel_signature(
            ps=page_size,
            hq=_HQ,
            hk=_HK,
            d=_D,
            blk=_BLOCK_THREADS,
            bn=_BLOCK_N,
            qkk=_QK_K,
            vec=_VEC,
        ),
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def qsa_k2_family_a_split_kernel(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        v_cache: fx.Tensor,
        indices: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        partial_out: fx.Tensor,
        partial_lse: fx.Tensor,
        n_sel: Int32,
        n_req: Int32,
        n_pages: Int32,
        n_splits: Int32,
        softmax_scale: Float32,
    ):
        row = Int32(gpu.block_id("x"))
        kv_h = Int32(gpu.block_id("y"))
        split = Int32(gpu.block_id("z"))
        tid = Int32(gpu.thread_id("x"))
        zero = Int32(0)
        one = Int32(1)
        page = Int32(page_size)
        wave = _idiv(tid, Int32(64))
        lane = tid - wave * Int32(64)
        lane_m = lane % Int32(16)
        lane_kg = _idiv(lane, Int32(16))
        vec_layout = fx.make_layout(_VEC, 1)
        g_copy = buf_copy_atom(16, BFloat16)
        q_buf = fx.rocdl.make_buffer_tensor(q)
        k_buf = fx.rocdl.make_buffer_tensor(k_cache)
        v_buf = fx.rocdl.make_buffer_tensor(v_cache)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        q_lds = storage.q.view(fx.make_layout((_HEAD_PAD, _D), (_D, 1)))
        k_lds = storage.k.view(fx.make_layout((_BLOCK_N, _D), (_D, 1)))
        v_lds = storage.v.view(fx.make_layout((_D, _BLOCK_N), (_BLOCK_N, 1)))
        p_lds = storage.p.view(fx.make_layout((_HEAD_PAD, _BLOCK_N), (_BLOCK_N, 1)))
        live_lds = storage.live.view(fx.make_layout(_BLOCK_N, 1))
        m_lds = storage.m.view(fx.make_layout(_HEAD_PAD, 1))
        l_lds = storage.l.view(fx.make_layout(_HEAD_PAD, 1))
        alpha_lds = storage.alpha.view(fx.make_layout(_HEAD_PAD, 1))
        c_lds = storage.c.view(
            fx.make_layout(
                (_N_SUBTILES, 4, 64, 4),
                (4 * 64 * 4, 64 * 4, 4, 1),
            )
        )
        qk_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, _QK_K, BFloat16))
        qk_a = fx.make_rmem_tensor(fx.make_layout(_QK_VEC, 1), BFloat16)
        qk_b = fx.make_rmem_tensor(fx.make_layout(_QK_VEC, 1), BFloat16)
        qk_c = fx.make_rmem_tensor(fx.make_layout(4, 1), Float32)
        pv_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, BFloat16))
        pv_a = fx.make_rmem_tensor(fx.make_layout(4, 1), BFloat16)
        pv_b = fx.make_rmem_tensor(fx.make_layout(4, 1), BFloat16)
        pv_c = fx.make_rmem_tensor(fx.make_layout(4, 1), Float32)

        def qk_mfma(a_vec, b_vec, c_vec):
            fx.memref_store_vec(a_vec, qk_a)
            fx.memref_store_vec(b_vec, qk_b)
            fx.memref_store_vec(c_vec, qk_c)
            fx.mma_atom_call(qk_mma, qk_c, qk_a, qk_b, qk_c)
            return fx.memref_load_vec(qk_c)

        def pv_mfma(a_vec, b_vec, c_vec):
            fx.memref_store_vec(a_vec, pv_a)
            fx.memref_store_vec(b_vec, pv_b)
            fx.memref_store_vec(c_vec, pv_c)
            fx.mma_atom_call(pv_mma, pv_c, pv_a, pv_b, pv_c)
            return fx.memref_load_vec(pv_c)

        req = token_to_req[row]
        valid_req = (req >= zero) & (req < n_req)
        safe_req = valid_req.select(req, zero)
        col_start = _idiv(split * n_sel, n_splits)
        col_end = _idiv((split + one) * n_sel, n_splits)

        def gather_tile(base):
            col = tid - _idiv(tid, Int32(_BLOCK_N)) * Int32(_BLOCK_N)
            chunk = _idiv(tid, Int32(_BLOCK_N))
            col_i = base + col
            in_col = col_i < col_end
            safe_col = in_col.select(col_i, col_start)
            tok = indices[row, safe_col]
            live = valid_req & in_col & (tok >= zero)
            safe_tok = (tok >= zero).select(tok, zero)
            logical_page = _idiv(safe_tok, page)
            off = safe_tok - logical_page * page
            in_table = logical_page < n_pages
            page_idx = in_table.select(logical_page, zero)
            phys = page_table[safe_req, page_idx]
            k_row = fx.logical_divide(
                fx.slice(k_buf, (phys, off, kv_h, None)), vec_layout
            )
            v_row = fx.logical_divide(
                fx.slice(v_buf, (phys, off, kv_h, None)), vec_layout
            )
            regs = []
            for half in range_constexpr(2):
                d_chunk = chunk + Int32(half * _CHUNK_STRIDE)
                k_src = fx.slice(k_row, (None, d_chunk))
                v_src = fx.slice(v_row, (None, d_chunk))
                k_frag = fx.make_fragment_like(k_src)
                v_frag = fx.make_fragment_like(v_src)
                fx.copy(g_copy, k_src, k_frag)
                fx.copy(g_copy, v_src, v_frag)
                regs.append(fx.Vector(fx.memref_load_vec(k_frag)))
                regs.append(fx.Vector(fx.memref_load_vec(v_frag)))
            return regs + [live.select(one, zero)]

        qh = tid - _idiv(tid, Int32(_HEAD_PAD)) * Int32(_HEAD_PAD)
        q_chunk = _idiv(tid, Int32(_HEAD_PAD))
        q_live = qh < Int32(_GROUP)
        q_head = kv_h * Int32(_GROUP) + qh
        safe_q_head = q_live.select(q_head, kv_h * Int32(_GROUP))
        q_row = fx.logical_divide(fx.slice(q_buf, (row, safe_q_head, None)), vec_layout)
        for half in range_constexpr(2):
            d_chunk = q_chunk + Int32(half * _CHUNK_STRIDE)
            q_src = fx.slice(q_row, (None, d_chunk))
            q_frag = fx.make_fragment_like(q_src)
            fx.copy(g_copy, q_src, q_frag)
            q_vec = fx.Vector(fx.memref_load_vec(q_frag))
            d0 = d_chunk * Int32(_VEC)
            for i in range_constexpr(_VEC):
                qv = q_live.select(q_vec[i].to(Float32), Float32(0.0))
                q_lds[qh, d0 + Int32(i)] = qv.to(BFloat16)
        if tid < Int32(_HEAD_PAD):
            m_lds[tid] = _neg_inf()
            l_lds[tid] = Float32(0.0)
        gpu.barrier()

        zero_acc = [fx.Vector.filled(4, 0.0, Float32) for _c in range(4)]
        current_tile = gather_tile(col_start)
        init_state = zero_acc + current_tile
        span = col_end - col_start
        n_tiles = _idiv(span + Int32(_BLOCK_N - 1), Int32(_BLOCK_N))
        _start = fx.Int64(0)
        _stop = fx.Int64(n_tiles)
        _step = fx.Int64(1)
        for t64, state in range(_start, _stop, _step, init=init_state):
            # The previous tile's PV still reads k/v/p LDS, so this gather
            # cannot start writing them until every wave is past it.
            gpu.barrier()
            base = col_start + Int32(t64) * Int32(_BLOCK_N)
            col = tid - _idiv(tid, Int32(_BLOCK_N)) * Int32(_BLOCK_N)
            chunk = _idiv(tid, Int32(_BLOCK_N))
            current_live = Int32(state[8]) != zero
            for half in range_constexpr(2):
                d_chunk = chunk + Int32(half * _CHUNK_STRIDE)
                k_vec = fx.Vector(state[4 + half * 2])
                v_vec = fx.Vector(state[5 + half * 2])
                d0 = d_chunk * Int32(_VEC)
                for i in range_constexpr(_VEC):
                    kz = current_live.select(k_vec[i].to(Float32), Float32(0.0)).to(
                        BFloat16
                    )
                    vz = current_live.select(v_vec[i].to(Float32), Float32(0.0)).to(
                        BFloat16
                    )
                    k_lds[col, d0 + Int32(i)] = kz
                    v_lds[d0 + Int32(i), col] = vz
            if chunk == zero:
                live_lds[col] = current_live.select(one, zero)
            gpu.barrier()

            # Issue the next paged loads before QK. Their register values are
            # consumed only after current-tile PV and carried to the next loop.
            next_tile = gather_tile(base + Int32(_BLOCK_N))
            for ng in range_constexpr(_N_SUBTILES):
                n_row = Int32(ng * 16) + lane_m
                acc4 = fx.Vector.filled(4, 0.0, Float32)
                for ks in range_constexpr(_K_STEPS):
                    d0 = wave * Int32(64) + Int32(ks * _QK_K) + lane_kg * Int32(_QK_VEC)
                    a_vec = fx.Vector.from_elements(
                        [
                            q_lds[lane_m, d0 + Int32(i)]
                            for i in range_constexpr(_QK_VEC)
                        ],
                        BFloat16,
                    )
                    b_vec = fx.Vector.from_elements(
                        [k_lds[n_row, d0 + Int32(i)] for i in range_constexpr(_QK_VEC)],
                        BFloat16,
                    )
                    acc4 = fx.Vector(qk_mfma(a_vec, b_vec, acc4))
                for i in range_constexpr(4):
                    c_lds[ng, wave, lane, i] = acc4[i]
            gpu.barrier()
            # Wave 0 owns the 16x16 P tile and the running m/l; other waves
            # wait at the next barrier so they cannot race those LDS rows.
            if wave == zero:
                for i in range_constexpr(4):
                    h = lane_kg * Int32(4) + Int32(i)
                    tile_max = _neg_inf()
                    head_scores = []
                    head_lives = []
                    for ng in range_constexpr(_N_SUBTILES):
                        n = Int32(ng * 16) + lane_m
                        sm_live = live_lds[n] != zero
                        s_i = (
                            c_lds[ng, zero, lane, i]
                            + c_lds[ng, one, lane, i]
                            + c_lds[ng, Int32(2), lane, i]
                            + c_lds[ng, Int32(3), lane, i]
                        ) * softmax_scale
                        s_i = sm_live.select(s_i, _neg_inf())
                        head_scores.append(s_i)
                        head_lives.append(sm_live)
                        tile_max = tile_max.maximumf(s_i)
                    for sh in (1, 2, 4, 8):
                        tile_max = tile_max.maximumf(
                            tile_max.shuffle_xor(Int32(sh), Int32(64))
                        )
                    m_prev = m_lds[h]
                    l_prev = l_lds[h]
                    m_new = m_prev.maximumf(tile_max)
                    alpha = fxmath.exp(m_prev - m_new)
                    p_sum = Float32(0.0)
                    for ng in range_constexpr(_N_SUBTILES):
                        n = Int32(ng * 16) + lane_m
                        p = head_lives[ng].select(
                            fxmath.exp(head_scores[ng] - m_new), Float32(0.0)
                        )
                        p_lds[h, n] = p.to(BFloat16)
                        p_sum = p_sum + p
                    for sh in (1, 2, 4, 8):
                        p_sum = p_sum + p_sum.shuffle_xor(Int32(sh), Int32(64))
                    if lane_m == zero:
                        m_lds[h] = m_new
                        l_lds[h] = l_prev * alpha + p_sum
                        alpha_lds[h] = alpha
            gpu.barrier()

            alpha4 = fx.Vector.from_elements(
                [
                    alpha_lds[lane_kg * Int32(4)],
                    alpha_lds[lane_kg * Int32(4) + one],
                    alpha_lds[lane_kg * Int32(4) + Int32(2)],
                    alpha_lds[lane_kg * Int32(4) + Int32(3)],
                ],
                Float32,
            )
            out_acc = []
            for c in range_constexpr(4):
                d = wave * Int32(64) + Int32(c * 16) + lane_m
                acc4 = fx.Vector(state[c]) * alpha4
                for ng in range_constexpr(_N_SUBTILES):
                    n0 = Int32(ng * 16) + lane_kg * Int32(4)
                    p_vec = fx.Vector.from_elements(
                        [
                            p_lds[lane_m, n0],
                            p_lds[lane_m, n0 + one],
                            p_lds[lane_m, n0 + Int32(2)],
                            p_lds[lane_m, n0 + Int32(3)],
                        ],
                        BFloat16,
                    )
                    v_vec = fx.Vector.from_elements(
                        [
                            v_lds[d, n0],
                            v_lds[d, n0 + one],
                            v_lds[d, n0 + Int32(2)],
                            v_lds[d, n0 + Int32(3)],
                        ],
                        BFloat16,
                    )
                    acc4 = fx.Vector(pv_mfma(p_vec, v_vec, acc4))
                out_acc.append(acc4)
            results = yield out_acc + next_tile

        # Rebuild the views outside the runtime loop so cached slice operations
        # cannot retain a loop/if-local defining operation into the epilogue.
        m_final = storage.m.view(fx.make_layout(_HEAD_PAD, 1))
        l_final = storage.l.view(fx.make_layout(_HEAD_PAD, 1))
        for i in range_constexpr(4):
            local_head = lane_kg * Int32(4) + Int32(i)
            if local_head < Int32(_GROUP):
                head = kv_h * Int32(_GROUP) + local_head
                den = l_final[local_head]
                has = den > Float32(0.0)
                for c in range_constexpr(4):
                    d = wave * Int32(64) + Int32(c * 16) + lane_m
                    out_f = has.select(fx.Vector(results[c])[i] / den, Float32(0.0))
                    partial_out[split, row, head, d] = out_f.to(BFloat16)
        if tid < Int32(_GROUP):
            head = kv_h * Int32(_GROUP) + tid
            den = l_final[tid]
            has = den > Float32(0.0)
            lse = has.select(m_final[tid] + fxmath.log(den), Float32(_LSE_EMPTY))
            partial_lse[split, row, head] = lse

    @flyc.kernel(
        name="qsa_k2_family_a_merge_"
        + kernel_signature(hq=_HQ, d=_D, blk=_BLOCK_THREADS),
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def qsa_k2_family_a_merge_kernel(
        partial_out: fx.Tensor,
        partial_lse: fx.Tensor,
        out: fx.Tensor,
        n_splits: Int32,
    ):
        row = Int32(gpu.block_id("x"))
        head = Int32(gpu.block_id("y"))
        tid = Int32(gpu.thread_id("x"))
        lane = tid % Int32(64)
        zero_f = Float32(0.0)
        split_live = lane < n_splits
        safe_split = split_live.select(lane, Int32(0))
        lse = partial_lse[safe_split, row, head]
        lse = split_live.select(lse, Float32(_LSE_EMPTY))
        m_max = lse
        for sh in (32, 16, 8, 4, 2, 1):
            m_max = m_max.maximumf(m_max.shuffle_xor(Int32(sh), Int32(64)))
        live = split_live & (lse > Float32(_LSE_EMPTY))
        w = live.select(fxmath.exp(lse - m_max), zero_f)
        den = w
        for sh in (32, 16, 8, 4, 2, 1):
            den = den + den.shuffle_xor(Int32(sh), Int32(64))

        merge_storage = fx.SharedAllocator().allocate(MergeStorage).peek()
        weights = merge_storage.weights.view(fx.make_layout(64, 1))
        denominator = merge_storage.denominator.view(fx.make_layout(1, 1))
        if tid < Int32(64):
            weights[tid] = w
        if tid == Int32(0):
            denominator[0] = den
        gpu.barrier()

        _start = fx.Int64(0)
        _stop = fx.Int64(n_splits)
        _step = fx.Int64(1)
        for s64, state in range(_start, _stop, _step, init=[zero_f, zero_f]):
            s = Int32(s64)
            acc_in = state[0]
            part = Float32(partial_out[s, row, head, tid])
            results = yield [acc_in + weights[s] * part, zero_f]
        acc_f = Float32(results[0])
        den_f = denominator[0]
        out_f = (den_f > Float32(0.0)).select(acc_f / den_f, Float32(0.0))
        out[row, head, tid] = out_f.to(BFloat16)

    @flyc.jit
    def launch_qsa_k2_family_a(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        v_cache: fx.Tensor,
        indices: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        partial_out: fx.Tensor,
        partial_lse: fx.Tensor,
        out: fx.Tensor,
        n_sel: Int32,
        n_req: Int32,
        n_pages: Int32,
        n_splits: Int32,
        softmax_scale: Float32,
        rows: Int32,
        kv_heads: Int32,
        n_heads: Int32,
        stream: fx.Stream,
    ):
        qsa_k2_family_a_split_kernel(
            q,
            k_cache,
            v_cache,
            indices,
            page_table,
            token_to_req,
            partial_out,
            partial_lse,
            n_sel,
            n_req,
            n_pages,
            n_splits,
            softmax_scale,
        ).launch(
            grid=(rows, kv_heads, n_splits),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        qsa_k2_family_a_merge_kernel(
            partial_out,
            partial_lse,
            out,
            n_splits,
        ).launch(
            grid=(rows, n_heads, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_qsa_k2_family_a.compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
    }
    return launch_qsa_k2_family_a


@lru_cache(maxsize=16)
def _plan(page_size: int, use_k32: bool):
    return build_qsa_k2_family_a_module(page_size, use_k32)


def qsa_k2_family_a_serves(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    page_table: torch.Tensor,
) -> str | None:
    """Why this K2 kernel cannot serve these tensors, or None if it can."""
    gqa = FAMILY_A_GQA
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        return f"q and caches must be bfloat16, got {q.dtype} and {k_cache.dtype}"
    if v_cache.dtype != k_cache.dtype or v_cache.shape != k_cache.shape:
        return "v_cache must match k_cache dtype and shape"
    if q.dim() != 3 or q.shape[1] != gqa.n_heads or q.shape[2] != gqa.head_dim:
        return f"q must be [M, {gqa.n_heads}, {gqa.head_dim}], got {tuple(q.shape)}"
    if k_cache.dim() != 4:
        return f"k_cache must be [pages, page_size, H, D], got {tuple(k_cache.shape)}"
    if k_cache.shape[2] != gqa.kv_heads or k_cache.shape[3] != gqa.head_dim:
        return (
            f"k_cache KV/D must be ({gqa.kv_heads}, {gqa.head_dim}), "
            f"got {k_cache.shape[2:]}"
        )
    if indices.dim() != 2 or indices.shape[0] != q.shape[0]:
        return "indices must be [M, W]"
    if indices.dtype != torch.int32:
        return f"indices must be int32, got {indices.dtype}"
    if page_table.dim() != 2 or page_table.dtype != torch.int32:
        return (
            f"page_table must be int32 [n_req, n_pages], got {tuple(page_table.shape)}"
        )
    return None


def qsa_k2_family_a(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Write family A sparse GQA ``o [M, 24, 256]`` from paged K/V.

    Attends ``indices [M, W]`` (``-1`` padded). Tiled ``BLOCK_N`` MFMA QK/PV
    with 128-bit paged gather, split-K plus LSE merge. Does not apply RoPE
    or the sigmoid gate. Expand+tail is still a separate launch.
    """
    reason = qsa_k2_family_a_serves(q, k_cache, v_cache, indices, page_table)
    if reason is not None:
        raise ValueError(f"[FlyDSL qsa_k2_family_a] {reason}")
    m = q.shape[0]
    if token_to_req.shape != (m,) or token_to_req.dtype != torch.int32:
        raise ValueError(f"token_to_req must be int32 [{m}]")
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype:
        raise ValueError(f"out must match q, got {tuple(out.shape)} {out.dtype}")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    tensors = (q, k_cache, v_cache, indices, page_table, token_to_req, out)
    if any(not t.is_cuda for t in tensors):
        raise ValueError("every tensor must be on the GPU")
    if any(t.device != q.device for t in tensors[1:]):
        raise ValueError("every tensor must be on the same GPU")
    if softmax_scale is None:
        softmax_scale = _DEFAULT_SCALE
    if not m or not indices.shape[1]:
        return out.zero_()
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    indices = indices.contiguous()
    page_table = page_table.contiguous()
    token_to_req = token_to_req.contiguous()
    page_size = k_cache.shape[1]
    use_k32 = torch.cuda.get_device_properties(q.device).gcnArchName.startswith(
        "gfx950"
    )
    n_sel = int(indices.shape[1])
    n_splits = _choose_splits(m, n_sel)
    partial_out = torch.empty(
        (n_splits, m, _HQ, _D), dtype=torch.bfloat16, device=q.device
    )
    partial_lse = torch.empty((n_splits, m, _HQ), dtype=torch.float32, device=q.device)
    _run_compiled(
        _plan(page_size, use_k32),
        q,
        k_cache,
        v_cache,
        indices,
        page_table,
        token_to_req,
        partial_out,
        partial_lse,
        out,
        n_sel,
        int(page_table.shape[0]),
        int(page_table.shape[1]),
        n_splits,
        float(softmax_scale),
        int(m),
        int(_HK),
        int(_HQ),
        torch.cuda.current_stream(q.device),
    )
    return out
