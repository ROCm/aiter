# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Family A FlyDSL QSA K2, shaped after the live AMD Triton kernel.

One workgroup owns ``(row, kv_head, split)``.  Q is staged once, paged K/V
are gathered in ``BLOCK_N`` tiles, QK and PV use BF16 MFMA, and online
softmax is maintained in log2 space.  Host dispatch mirrors live AMD:
small decode uses BLOCK_N=16 / four waves / split-K, while prefill uses
BLOCK_N=64 / two waves / one split and writes output directly.

gfx942 aliases K and V in one LDS tile so BLOCK_N=64 stays under 64 KiB.
gfx950 stores K and V separately and gathers this tile's V after K is
visible so QK can run while V is in flight.  The post-QK barrier still
publishes C for softmax.  Expand, partial RoPE, and the sigmoid output
gate remain outside K2.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import BFloat16, Float32, Int32, const_expr, gpu, range_constexpr
from flydsl.expr import math as fxmath

from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from aiter.ops.flydsl.kernels.qsa.shapes import FAMILY_A_GQA
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, buf_copy_atom

_HQ = FAMILY_A_GQA.n_heads
_HK = FAMILY_A_GQA.kv_heads
_GROUP = FAMILY_A_GQA.group_size
_D = FAMILY_A_GQA.head_dim
# Pad K (and aliased KV) rows so consecutive columns do not share LDS banks
# on 128-bit stores. D=256 makes n*D a multiple of 32 banks otherwise.
_K_STRIDE = _D + 8
_HEAD_PAD = 16
_DEFAULT_SCALE = _D**-0.5
_LSE_EMPTY = -1.0e20
_LOG2E = 1.4426950408889634


def _idiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def _neg_inf():
    return Float32(float("-inf"))


def _exp2(x):
    return Float32(fx.rocdl.exp2(Float32.ir_type, Float32(x).ir_value()))


def _launch_config(rows: int, n_sel: int) -> tuple[int, int, int]:
    """Return ``(BLOCK_N, threads, splits)`` using the tuned AMD-shaped policy."""
    base_programs = rows * _HK
    if base_programs <= 4 or base_programs < 32:
        block_n, target_splits, threads = 16, 32, 256
    elif base_programs <= 256:
        block_n, target_splits, threads = 64, 8, 128
    elif base_programs <= 512:
        block_n, target_splits, threads = 64, 4, 128
    else:
        block_n, target_splits, threads = 64, 1, 128

    tiles = max(1, (n_sel + block_n - 1) // block_n)
    max_useful_splits = 1 << (tiles.bit_length() - 1)
    return block_n, threads, min(max_useful_splits, target_splits)


def build_qsa_k2_family_a_module(
    page_size: int,
    use_k32: bool,
    block_n: int,
    block_threads: int,
    n_splits: int,
):
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if block_n not in (16, 64):
        raise ValueError(f"BLOCK_N must be 16 or 64, got {block_n}")
    if block_threads not in (128, 256):
        raise ValueError(f"block_threads must be 128 or 256, got {block_threads}")
    if block_threads % 64 or block_threads % block_n:
        raise ValueError("thread and column mappings must divide evenly")
    if _HQ != _HK * _GROUP:
        raise ValueError("family A GQA head counts do not form groups")

    num_waves = block_threads // 64
    n_subtiles = block_n // 16
    qk_k = 32 if use_k32 else 16
    qk_vec = qk_k // 4
    qk_steps = _D // (num_waves * qk_k)
    out_chunks = _D // (num_waves * 16)
    vec = 8
    d_chunks = _D // vec
    col_owners = block_threads // block_n
    gather_rounds = d_chunks // col_owners
    gather_span = col_owners * vec
    token_major_v = use_k32 and block_n == 16

    if use_k32:

        @fx.struct
        class SharedStorage:
            k: fx.Array[BFloat16, block_n * _K_STRIDE, 16]
            v: fx.Array[BFloat16, block_n * _D, 16]
            p: fx.Array[BFloat16, _HEAD_PAD * block_n, 16]
            live: fx.Array[Int32, block_n, 16]
            phys: fx.Array[Int32, block_n, 16]
            page_off: fx.Array[Int32, block_n, 16]
            m: fx.Array[Float32, _HEAD_PAD, 16]
            l: fx.Array[Float32, _HEAD_PAD, 16]
            alpha: fx.Array[Float32, _HEAD_PAD, 16]
            c: fx.Array[Float32, n_subtiles * num_waves * 64 * 4, 16]

        _k_field, _v_field = "k", "v"
    else:

        @fx.struct
        class SharedStorage:
            kv: fx.Array[BFloat16, block_n * _K_STRIDE, 16]
            p: fx.Array[BFloat16, _HEAD_PAD * block_n, 16]
            live: fx.Array[Int32, block_n, 16]
            phys: fx.Array[Int32, block_n, 16]
            page_off: fx.Array[Int32, block_n, 16]
            m: fx.Array[Float32, _HEAD_PAD, 16]
            l: fx.Array[Float32, _HEAD_PAD, 16]
            alpha: fx.Array[Float32, _HEAD_PAD, 16]
            c: fx.Array[Float32, n_subtiles * num_waves * 64 * 4, 16]

        _k_field, _v_field = "kv", "kv"

    @fx.struct
    class MergeStorage:
        weights: fx.Array[Float32, 64, 16]
        denominator: fx.Array[Float32, 1, 16]

    @flyc.kernel(
        name="qsa_k2_family_a_port_split_"
        + kernel_signature(
            ps=page_size,
            bn=block_n,
            blk=block_threads,
            ns=n_splits,
            qkk=qk_k,
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def split_kernel(
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
        table_width: Int32,
        n_cache_blocks: Int32,
        softmax_scale_log2: Float32,
    ):
        row = Int32(gpu.block_id("x"))
        kv_h = Int32(gpu.block_id("y"))
        split = Int32(gpu.block_id("z"))
        tid = Int32(gpu.thread_id("x"))
        zero = Int32(0)
        one = Int32(1)
        page = Int32(page_size)
        thread_coord = fx.idx2crd(tid, fx.make_layout((num_waves, 64), (64, 1)))
        wave = Int32(fx.get(thread_coord, 0))
        lane = Int32(fx.get(thread_coord, 1))
        lane_m = lane % Int32(16)
        lane_kg = _idiv(lane, Int32(16))

        vec_layout = fx.make_layout(vec, 1)
        g_copy = buf_copy_atom(16, BFloat16)
        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), BFloat16)
        kv_tile, kv_tv = fx.make_layout_tv(
            fx.make_layout((block_n, col_owners), (1, block_n)),
            fx.make_layout((1, vec), (vec, 1)),
        )
        kv_store = fx.make_tiled_copy(lds_copy, kv_tv, kv_tile).get_slice(tid)
        q_buf = fx.rocdl.make_buffer_tensor(q)
        k_buf = fx.rocdl.make_buffer_tensor(k_cache)
        v_buf = fx.rocdl.make_buffer_tensor(v_cache)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_lds = getattr(storage, _k_field).view(
            fx.make_layout((block_n, _K_STRIDE), (_K_STRIDE, 1))
        )
        v_lds = getattr(storage, _v_field).view(
            fx.make_layout((_D, block_n), (block_n, 1))
            if token_major_v
            else fx.make_layout(
                (block_n, _K_STRIDE if not use_k32 else _D),
                (_K_STRIDE if not use_k32 else _D, 1),
            )
        )
        p_lds = storage.p.view(fx.make_layout((_HEAD_PAD, block_n), (block_n, 1)))
        live_lds = storage.live.view(fx.make_layout(block_n, 1))
        phys_lds = storage.phys.view(fx.make_layout(block_n, 1))
        page_off_lds = storage.page_off.view(fx.make_layout(block_n, 1))
        m_lds = storage.m.view(fx.make_layout(_HEAD_PAD, 1))
        l_lds = storage.l.view(fx.make_layout(_HEAD_PAD, 1))
        alpha_lds = storage.alpha.view(fx.make_layout(_HEAD_PAD, 1))
        c_lds = storage.c.view(
            fx.make_layout(
                (n_subtiles, num_waves, 64, 4),
                (num_waves * 64 * 4, 64 * 4, 4, 1),
            )
        )

        qk_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, qk_k, BFloat16))
        pv_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, BFloat16))
        qk_wave_mma = fx.make_tiled_mma(qk_mma, fx.make_layout((1, 1, 1), (0, 0, 0)))
        qk_b_atom = fx.make_copy_atom(
            fx.UniversalCopy128b() if qk_k == 32 else fx.UniversalCopy64b(),
            BFloat16,
        )
        qk_b_copy = fx.make_tiled_copy_B(qk_b_atom, qk_wave_mma).get_slice(lane)
        pv_wave_mma = fx.make_tiled_mma(pv_mma, fx.make_layout((1, 1, 1), (0, 0, 0)))
        pv_b_atom = fx.make_copy_atom(fx.UniversalCopy64b(), BFloat16)
        pv_b_copy = fx.make_tiled_copy_B(pv_b_atom, pv_wave_mma).get_slice(lane)

        def qk_mfma(a_vec, b_vec, c_vec):
            qk_a = fx.make_rmem_tensor(fx.make_layout(qk_vec, 1), BFloat16)
            qk_b = fx.make_rmem_tensor(fx.make_layout(qk_vec, 1), BFloat16)
            qk_c = fx.make_rmem_tensor(fx.make_layout(4, 1), Float32)
            fx.memref_store_vec(a_vec, qk_a)
            fx.memref_store_vec(b_vec, qk_b)
            fx.memref_store_vec(c_vec, qk_c)
            fx.gemm(qk_mma, qk_c, qk_a, qk_b, qk_c)
            return fx.Vector(fx.memref_load_vec(qk_c))

        def pv_mfma(a_vec, b_vec, c_vec):
            pv_a = fx.make_rmem_tensor(fx.make_layout(4, 1), BFloat16)
            pv_b = fx.make_rmem_tensor(fx.make_layout(4, 1), BFloat16)
            pv_c = fx.make_rmem_tensor(fx.make_layout(4, 1), Float32)
            fx.memref_store_vec(a_vec, pv_a)
            fx.memref_store_vec(b_vec, pv_b)
            fx.memref_store_vec(c_vec, pv_c)
            fx.gemm(pv_mma, pv_c, pv_a, pv_b, pv_c)
            return fx.Vector(fx.memref_load_vec(pv_c))

        req = token_to_req[row]
        valid_req = (req >= zero) & (req < n_req)
        safe_req = valid_req.select(req, zero)
        total_tiles = _idiv(n_sel + Int32(block_n - 1), Int32(block_n))
        tile_start = _idiv(split * total_tiles, Int32(n_splits))
        tile_end = _idiv((split + one) * total_tiles, Int32(n_splits))
        col_start = tile_start * Int32(block_n)
        col_end_unclamped = tile_end * Int32(block_n)
        col_end = (col_end_unclamped < n_sel).select(col_end_unclamped, n_sel)

        # Load the Q fragments once and carry them in registers for every tile.
        q_live = lane_m < Int32(_GROUP)
        q_head = kv_h * Int32(_GROUP) + lane_m
        safe_q_head = q_live.select(q_head, kv_h * Int32(_GROUP))
        q_row = fx.logical_divide(fx.slice(q_buf, (row, safe_q_head, None)), vec_layout)
        q_regs = []
        for ks in range_constexpr(qk_steps):
            q_d0 = (
                wave * Int32(_D // num_waves)
                + Int32(ks * qk_k)
                + lane_kg * Int32(qk_vec)
            )
            d_chunk = _idiv(q_d0, Int32(vec))
            q_off = q_d0 - d_chunk * Int32(vec)
            q_src = fx.slice(q_row, (None, d_chunk))
            q_frag = fx.make_fragment_like(q_src)
            fx.copy(g_copy, q_src, q_frag)
            q_vec = fx.Vector(fx.memref_load_vec(q_frag))
            q_regs.append(
                fx.Vector.from_elements(
                    [
                        q_live.select(
                            q_vec[q_off + Int32(i)].to(Float32), Float32(0.0)
                        ).to(BFloat16)
                        for i in range_constexpr(qk_vec)
                    ],
                    BFloat16,
                )
            )
        if tid < Int32(_HEAD_PAD):
            m_lds[tid] = _neg_inf()
            l_lds[tid] = Float32(0.0)
        gpu.barrier()

        init_acc = [fx.Vector.filled(4, 0.0, Float32) for _ in range(out_chunks)]
        n_tiles = tile_end - tile_start
        start = fx.Int64(0)
        stop = fx.Int64(n_tiles)
        step = fx.Int64(1)
        for tile64, state in range(start, stop, step, init=init_acc):
            gpu.barrier()
            base = col_start + Int32(tile64) * Int32(block_n)
            col = tid % Int32(block_n)
            chunk_owner = _idiv(tid, Int32(block_n))
            col_i = base + col
            in_col = col_i < col_end
            if chunk_owner == zero:
                safe_col = in_col.select(col_i, col_start)
                tok = indices[row, safe_col]
                token_live = valid_req & in_col & (tok >= zero)
                safe_tok = (tok >= zero).select(tok, zero)
                logical_page = _idiv(safe_tok, page)
                page_off_i = safe_tok - logical_page * page
                table_live = logical_page < table_width
                safe_logical_page = table_live.select(logical_page, zero)
                phys = page_table[safe_req, safe_logical_page]
                phys_live = (phys >= zero) & (phys < n_cache_blocks)
                live = token_live & table_live & phys_live
                phys_lds[col] = phys_live.select(phys, zero)
                page_off_lds[col] = page_off_i
                live_lds[col] = live.select(one, zero)
            gpu.barrier()
            safe_phys = phys_lds[col]
            page_off_i = page_off_lds[col]
            live = live_lds[col] != zero

            k_row = fx.logical_divide(
                fx.slice(k_buf, (safe_phys, page_off_i, kv_h, None)), vec_layout
            )
            for gr in range_constexpr(gather_rounds):
                d_chunk = chunk_owner + Int32(gr * col_owners)
                k_src = fx.slice(k_row, (None, d_chunk))
                k_frag = fx.make_fragment_like(k_src)
                fx.copy(g_copy, k_src, k_frag)
                k_vec = fx.Vector(fx.memref_load_vec(k_frag))
                k_vec = fx.Vector.from_elements(
                    [
                        live.select(k_vec[i].to(Float32), Float32(0.0)).to(BFloat16)
                        for i in range_constexpr(vec)
                    ],
                    BFloat16,
                )
                fx.memref_store_vec(k_vec, k_frag)
                k_tile = fx.make_view(
                    fx.get_iter(k_lds) + Int32(gr * gather_span),
                    fx.make_layout((block_n, gather_span), (_K_STRIDE, 1)),
                )
                k_dst = kv_store.partition_D(k_tile)
                k_store_frag = fx.make_fragment_like(k_dst)
                fx.memref_store_vec(k_vec, k_store_frag)
                fx.copy(lds_copy, k_store_frag, k_dst)
            gpu.barrier()

            if const_expr(use_k32):
                v_row = fx.logical_divide(
                    fx.slice(v_buf, (safe_phys, page_off_i, kv_h, None)), vec_layout
                )
                for gr in range_constexpr(gather_rounds):
                    d_chunk = chunk_owner + Int32(gr * col_owners)
                    v_src = fx.slice(v_row, (None, d_chunk))
                    v_frag = fx.make_fragment_like(v_src)
                    fx.copy(g_copy, v_src, v_frag)
                    v_vec = fx.Vector(fx.memref_load_vec(v_frag))
                    v_vec = fx.Vector.from_elements(
                        [
                            live.select(v_vec[i].to(Float32), Float32(0.0)).to(BFloat16)
                            for i in range_constexpr(vec)
                        ],
                        BFloat16,
                    )
                    fx.memref_store_vec(v_vec, v_frag)
                    if const_expr(token_major_v):
                        d0 = d_chunk * Int32(vec)
                        for i in range_constexpr(vec):
                            v_lds[d0 + Int32(i), col] = v_vec[i]
                    else:
                        v_tile = fx.make_view(
                            fx.get_iter(v_lds) + Int32(gr * gather_span),
                            fx.make_layout((block_n, gather_span), (_D, 1)),
                        )
                        v_dst = kv_store.partition_D(v_tile)
                        v_store_frag = fx.make_fragment_like(v_dst)
                        fx.memref_store_vec(v_vec, v_store_frag)
                        fx.copy(lds_copy, v_store_frag, v_dst)

            # QK: waves partition D, then wave 0 reduces their C fragments.
            for ng in range_constexpr(n_subtiles):
                n0 = Int32(ng * 16)
                acc4 = fx.Vector.filled(4, 0.0, Float32)
                for ks in range_constexpr(qk_steps):
                    d_base = wave * Int32(_D // num_waves) + Int32(ks * qk_k)
                    sB = fx.make_view(
                        fx.get_iter(k_lds) + n0 * Int32(_K_STRIDE) + d_base,
                        fx.make_layout((16, qk_k), (_K_STRIDE, 1)),
                    )
                    b_src = qk_b_copy.partition_S(sB)
                    b_frag = fx.make_fragment_like(b_src)
                    fx.copy(qk_b_atom, b_src, b_frag)
                    acc4 = qk_mfma(
                        fx.Vector(q_regs[ks]),
                        fx.Vector(fx.memref_load_vec(b_frag)),
                        acc4,
                    )
                for i in range_constexpr(4):
                    c_lds[ng, wave, lane, i] = acc4[i]
            gpu.barrier()

            if const_expr(not use_k32):
                v_row = fx.logical_divide(
                    fx.slice(v_buf, (safe_phys, page_off_i, kv_h, None)), vec_layout
                )
                for gr in range_constexpr(gather_rounds):
                    d_chunk = chunk_owner + Int32(gr * col_owners)
                    v_src = fx.slice(v_row, (None, d_chunk))
                    v_frag = fx.make_fragment_like(v_src)
                    fx.copy(g_copy, v_src, v_frag)
                    v_vec = fx.Vector(fx.memref_load_vec(v_frag))
                    v_vec = fx.Vector.from_elements(
                        [
                            live.select(v_vec[i].to(Float32), Float32(0.0)).to(BFloat16)
                            for i in range_constexpr(vec)
                        ],
                        BFloat16,
                    )
                    fx.memref_store_vec(v_vec, v_frag)
                    v_tile = fx.make_view(
                        fx.get_iter(v_lds) + Int32(gr * gather_span),
                        fx.make_layout((block_n, gather_span), (_K_STRIDE, 1)),
                    )
                    v_dst = kv_store.partition_D(v_tile)
                    v_store_frag = fx.make_fragment_like(v_dst)
                    fx.memref_store_vec(v_vec, v_store_frag)
                    fx.copy(lds_copy, v_store_frag, v_dst)

            # Softmax reads C and live, not V; the post-QK barrier already
            # published C. V and P meet at the barrier before PV.
            if wave == zero:
                for i in range_constexpr(4):
                    h = lane_kg * Int32(4) + Int32(i)
                    tile_max = _neg_inf()
                    scores = []
                    score_lives = []
                    for ng in range_constexpr(n_subtiles):
                        n = Int32(ng * 16) + lane_m
                        score_live = live_lds[n] != zero
                        score = c_lds[ng, zero, lane, i]
                        for w in range_constexpr(1, num_waves):
                            score = score + c_lds[ng, w, lane, i]
                        score = score * softmax_scale_log2
                        score = score_live.select(score, _neg_inf())
                        scores.append(score)
                        score_lives.append(score_live)
                        tile_max = tile_max.maximumf(score)
                    for sh in (1, 2, 4, 8):
                        tile_max = tile_max.maximumf(
                            tile_max.shuffle_xor(Int32(sh), Int32(64))
                        )
                    m_prev = m_lds[h]
                    l_prev = l_lds[h]
                    m_new = m_prev.maximumf(tile_max)
                    alpha = _exp2(m_prev - m_new)
                    p_sum = Float32(0.0)
                    for ng in range_constexpr(n_subtiles):
                        n = Int32(ng * 16) + lane_m
                        p = score_lives[ng].select(
                            _exp2(scores[ng] - m_new), Float32(0.0)
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
            next_acc = []
            for c in range_constexpr(out_chunks):
                d = wave * Int32(_D // num_waves) + Int32(c * 16) + lane_m
                acc4 = fx.Vector(state[c]) * alpha4
                for ng in range_constexpr(n_subtiles):
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
                    if const_expr(token_major_v):
                        d_base = wave * Int32(_D // num_waves) + Int32(c * 16)
                        sB = fx.make_view(
                            fx.get_iter(v_lds)
                            + d_base * Int32(block_n)
                            + Int32(ng * 16),
                            fx.make_layout((16, 16), (block_n, 1)),
                        )
                        b_src = pv_b_copy.partition_S(sB)
                        b_frag = fx.make_fragment_like(b_src)
                        fx.copy(pv_b_atom, b_src, b_frag)
                        v_vec = fx.Vector(fx.memref_load_vec(b_frag))
                    else:
                        v_vec = fx.Vector.from_elements(
                            [
                                v_lds[n0, d],
                                v_lds[n0 + one, d],
                                v_lds[n0 + Int32(2), d],
                                v_lds[n0 + Int32(3), d],
                            ],
                            BFloat16,
                        )
                    acc4 = pv_mfma(p_vec, v_vec, acc4)
                next_acc.append(acc4)
            results = yield next_acc

        m_final = storage.m.view(fx.make_layout(_HEAD_PAD, 1))
        l_final = storage.l.view(fx.make_layout(_HEAD_PAD, 1))
        for i in range_constexpr(4):
            local_head = lane_kg * Int32(4) + Int32(i)
            if local_head < Int32(_GROUP):
                head = kv_h * Int32(_GROUP) + local_head
                den = l_final[local_head]
                has = den > Float32(0.0)
                for c in range_constexpr(out_chunks):
                    d = wave * Int32(_D // num_waves) + Int32(c * 16) + lane_m
                    value = has.select(fx.Vector(results[c])[i] / den, Float32(0.0))
                    if n_splits == 1:
                        out[row, head, d] = value.to(BFloat16)
                    else:
                        partial_out[split, row, head, d] = value
        if n_splits > 1 and tid < Int32(_GROUP):
            head = kv_h * Int32(_GROUP) + tid
            den = l_final[tid]
            has = den > Float32(0.0)
            lse = has.select(
                m_final[tid] + fxmath.log(den) * Float32(_LOG2E),
                Float32(_LSE_EMPTY),
            )
            partial_lse[split, row, head] = lse

    @flyc.kernel(
        name="qsa_k2_family_a_port_merge_"
        + kernel_signature(ns=n_splits, blk=128, d=_D),
        known_block_size=[128, 1, 1],
    )
    def merge_kernel(
        partial_out: fx.Tensor,
        partial_lse: fx.Tensor,
        out: fx.Tensor,
    ):
        row = Int32(gpu.block_id("x"))
        head = Int32(gpu.block_id("y"))
        tid = Int32(gpu.thread_id("x"))
        lane = tid % Int32(64)
        zero_f = Float32(0.0)
        split_live = lane < Int32(n_splits)
        safe_split = split_live.select(lane, Int32(0))
        lse = partial_lse[safe_split, row, head]
        lse = split_live.select(lse, Float32(_LSE_EMPTY))
        lse_max = lse
        for sh in (32, 16, 8, 4, 2, 1):
            lse_max = lse_max.maximumf(lse_max.shuffle_xor(Int32(sh), Int32(64)))
        live = split_live & (lse > Float32(_LSE_EMPTY))
        weight = live.select(_exp2(lse - lse_max), zero_f)
        den = weight
        for sh in (32, 16, 8, 4, 2, 1):
            den = den + den.shuffle_xor(Int32(sh), Int32(64))

        merge_storage = fx.SharedAllocator().allocate(MergeStorage).peek()
        weights = merge_storage.weights.view(fx.make_layout(64, 1))
        denominator = merge_storage.denominator.view(fx.make_layout(1, 1))
        if tid < Int32(64):
            weights[tid] = weight
        if tid == Int32(0):
            denominator[0] = den
        gpu.barrier()

        for di in range_constexpr(2):
            d = tid + Int32(di * 128)
            start = fx.Int64(0)
            stop = fx.Int64(n_splits)
            step = fx.Int64(1)
            for split64, state in range(start, stop, step, init=[zero_f]):
                s = Int32(split64)
                acc = Float32(state[0])
                part = Float32(partial_out[s, row, head, d])
                merged = yield [acc + weights[s] * part]
            den_f = denominator[0]
            value = (den_f > zero_f).select(Float32(merged) / den_f, zero_f)
            out[row, head, d] = value.to(BFloat16)

    @flyc.jit
    def launch(
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
        table_width: Int32,
        n_cache_blocks: Int32,
        softmax_scale_log2: Float32,
        rows: Int32,
        kv_heads: Int32,
        n_heads: Int32,
        stream: fx.Stream,
    ):
        split_kernel(
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
            n_req,
            table_width,
            n_cache_blocks,
            softmax_scale_log2,
        ).launch(
            grid=(rows, kv_heads, n_splits),
            block=(block_threads, 1, 1),
            stream=stream,
        )
        if n_splits > 1:
            merge_kernel(partial_out, partial_lse, out).launch(
                grid=(rows, n_heads, 1),
                block=(128, 1, 1),
                stream=stream,
            )

    launch.compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
    }
    return launch


@lru_cache(maxsize=32)
def _plan(
    page_size: int,
    use_k32: bool,
    block_n: int,
    block_threads: int,
    n_splits: int,
):
    return build_qsa_k2_family_a_module(
        page_size, use_k32, block_n, block_threads, n_splits
    )


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
    """Write family A sparse GQA ``o [M, 24, 256]`` from paged K/V."""
    reason = qsa_k2_family_a_serves(q, k_cache, v_cache, indices, page_table)
    if reason is not None:
        raise ValueError(f"[FlyDSL qsa_k2_family_a] {reason}")
    rows = q.shape[0]
    if token_to_req.shape != (rows,) or token_to_req.dtype != torch.int32:
        raise ValueError(f"token_to_req must be int32 [{rows}]")
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype:
        raise ValueError(f"out must match q, got {tuple(out.shape)} {out.dtype}")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    tensors = (q, k_cache, v_cache, indices, page_table, token_to_req, out)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("every tensor must be on the GPU")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("every tensor must be on the same GPU")
    if softmax_scale is None:
        softmax_scale = _DEFAULT_SCALE
    if not rows or not indices.shape[1]:
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
    block_n, block_threads, n_splits = _launch_config(rows, n_sel)
    if n_splits == 1:
        partial_out = out
        partial_lse = out
    else:
        partial_out = torch.empty(
            (n_splits, rows, _HQ, _D), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (n_splits, rows, _HQ), dtype=torch.float32, device=q.device
        )

    _run_compiled(
        _plan(page_size, use_k32, block_n, block_threads, n_splits),
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
        int(k_cache.shape[0]),
        float(softmax_scale * _LOG2E),
        int(rows),
        int(_HK),
        int(_HQ),
        torch.cuda.current_stream(q.device),
    )
    return out
