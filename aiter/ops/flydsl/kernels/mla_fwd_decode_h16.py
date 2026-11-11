# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Persistent paged MLA decode forward in FlyDSL (gfx942 / gfx950).

Each workgroup walks its slice of the persistent work list. A work's queries
are viewed as a row-major ``(qo_len * num_heads, 576)`` matrix whose row ``r``
is (token ``r // num_heads``, head ``r % num_heads``). Rows are processed in
blocks of ``BLOCK_M = 48 = lcm(12, 16)`` so a block always holds whole tokens
for 12- and 16-head models; the last block is padded and masked. Each of the
three waves owns 16 rows of a block and computes, per 48-token KV tile,

    S^T = K @ Q^T          (MFMA M = tokens, N = q rows, K = 576)
    O^T += V^T @ P^T       (MFMA M = v columns, N = q rows, K = tokens)

The transposed products make the S^T accumulator layout identical to the P^T
operand layout, and put one q row per lane in both, so the online softmax
needs only lane shuffles within a wave.
"""

import functools

import torch

from aiter.utility import dtypes

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.protocol import dsl_size_of
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import (
    SMEM_CAPACITY_MAP,
    check_smem_capacity,
    get_rocm_arch,
    get_warp_size,
)

QK_HEAD_DIM: int = 576
V_HEAD_DIM: int = 512
NUM_KV_HEADS: int = 1

MMA_TILE: int = 16
WAVE_ROWS: int = MMA_TILE
NUM_WAVES: int = 3
BLOCK_M: int = NUM_WAVES * WAVE_ROWS
BLOCK_N: int = 48
CHUNK_D: int = 64
QK_CHUNKS: int = QK_HEAD_DIM // CHUNK_D
V_CHUNKS: int = V_HEAD_DIM // CHUNK_D
MAX_KV_STAGES: int = 2

# One KV row as seen by one wave: 64 lanes x 8 elements cover the 512 NOPE
# columns, and 64 lanes x 1 element cover the 64 RoPE columns.
ROW_VEC: int = 8
ROW_VECS: int = QK_HEAD_DIM // ROW_VEC
ROPE_COL_GROUP: int = V_HEAD_DIM // CHUNK_D
ROWS_PER_WAVE: int = BLOCK_N // NUM_WAVES
# Padding the LDS row by 16 bytes staggers banks across the rows an MFMA operand reads.
KV_LDS_STRIDE: int = QK_HEAD_DIM + ROW_VEC

# Extent of the "window" mode in row views: mode (rows, start) has stride
# (row_stride, row_stride), so slicing ``start`` selects rows [start, start + rows).
ROW_WINDOW: int = 1 << 20
PAGE_WINDOW: int = 1 << 20

LOG2E: float = 1.4426950408889634
LN2: float = 0.6931471805599453


def _fp8_words_to_f32(words):
    """Decode packed fp8 (i32 words, 4 values each) in the target's native fp8 format."""
    vals = []
    for w in range_constexpr(len(words)):
        for hi in range_constexpr(2):
            pair = fx.Vector(fx.rocdl.cvt_pk_f32_fp8(T.f32x2, words[w], hi == 1))
            vals += [pair[0], pair[1]]
    return vals


def _kv_stage_storage(num_stages):
    return fx.Array[fx.BFloat16, num_stages * BLOCK_N * KV_LDS_STRIDE, 16]


def _num_kv_stages(arch):
    capacity = SMEM_CAPACITY_MAP.get(arch, SMEM_CAPACITY_MAP["gfx942"])
    for num_stages in range(MAX_KV_STAGES, 0, -1):
        if dsl_size_of(_kv_stage_storage(num_stages)) <= capacity:
            return num_stages
    raise RuntimeError(f"MLA decode KV tile does not fit in the {capacity}-byte LDS of {arch}")


@functools.lru_cache(maxsize=None)
def _build_mla_fwd_decode(arch: str):
    wave_size = get_warp_size(arch)
    if wave_size != 64:
        raise ValueError(f"MLA decode requires wave64 MFMA, got wave{wave_size} on {arch}")
    block_threads = NUM_WAVES * wave_size
    num_stages = _num_kv_stages(arch)
    stage_elems = BLOCK_N * KV_LDS_STRIDE

    @fx.struct
    class SharedStorage:
        kv: _kv_stage_storage(num_stages)

    check_smem_capacity(dsl_size_of(SharedStorage), arch)

    @flyc.kernel
    def mla_fwd_decode_kernel(
        query: fx.Tensor,
        kv_buffer: fx.Tensor,
        kv_page_indices: fx.Tensor,
        work_indptr: fx.Tensor,
        work_info_set: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        final_output: fx.Tensor,
        split_output: fx.Tensor,
        split_lse: fx.Tensor,
        sm_scale: fx.Float32,
        num_heads: fx.Constexpr[int],
        page_size: fx.Constexpr[int],
    ):
        tid = fx.thread_idx.x
        cu = fx.block_idx.x
        lane = tid % wave_size
        wave = tid // wave_size

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MMA_TILE, MMA_TILE, MMA_TILE, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(lane)

        # ---- Global views -------------------------------------------------
        q_iter = fx.get_iter(fx.rocdl.make_buffer_tensor(query, max_size=False))
        q_rows = fx.make_view(
            q_iter,
            fx.make_layout(((WAVE_ROWS, ROW_WINDOW), QK_HEAD_DIM), ((QK_HEAD_DIM, QK_HEAD_DIM), 1)),
        )
        # fp8 KV is moved as raw bytes and decoded to bf16 on its way into LDS.
        kv_is_fp8 = kv_buffer.element_type.width == 8
        kv_raw = fx.Int8 if const_expr(kv_is_fp8) else fx.BFloat16
        kv_iter = fx.get_iter(fx.rocdl.make_buffer_tensor(kv_buffer, max_size=False))
        if const_expr(kv_is_fp8):
            kv_iter = fx.recast_iter(kv_raw, kv_iter)
        kv_token = ((page_size, PAGE_WINDOW), (QK_HEAD_DIM * NUM_KV_HEADS, page_size * QK_HEAD_DIM * NUM_KV_HEADS))
        kv_vecs = fx.make_view(
            kv_iter, fx.make_layout((kv_token[0], (ROW_VEC, ROW_VECS)), (kv_token[1], (1, ROW_VEC)))
        )
        kv_cols = fx.make_view(
            kv_iter, fx.make_layout((kv_token[0], (wave_size, ROPE_COL_GROUP + 1)), (kv_token[1], (1, wave_size)))
        )
        token_to_page = fx.make_layout((page_size, PAGE_WINDOW), (1, page_size))
        page_ids = fx.rocdl.make_buffer_tensor(kv_page_indices)
        works = fx.rocdl.make_buffer_tensor(work_indptr)
        work_info = fx.rocdl.make_buffer_tensor(work_info_set)
        last_page_lens = fx.rocdl.make_buffer_tensor(kv_last_page_lens)
        out_rows_layout = fx.make_layout((V_HEAD_DIM, (WAVE_ROWS, ROW_WINDOW)), (1, (V_HEAD_DIM, V_HEAD_DIM)))
        final_rows = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(final_output, max_size=False)), out_rows_layout
        )
        split_rows = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(split_output, max_size=False)), out_rows_layout
        )
        lse_rows = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(split_lse, max_size=False)),
            fx.make_layout((WAVE_ROWS, ROW_WINDOW), (1, 1)),
        )

        # ---- LDS views: one (BLOCK_N, 576) KV tile per stage --------------
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_kv = lds.kv.view(
            fx.make_layout((BLOCK_N, QK_HEAD_DIM, num_stages), (KV_LDS_STRIDE, 1, stage_elems))
        )
        s_kv_vecs = lds.kv.view(
            fx.make_layout((BLOCK_N, (ROW_VEC, ROW_VECS), num_stages), (KV_LDS_STRIDE, (1, ROW_VEC), stage_elems))
        )
        s_kv_cols = lds.kv.view(
            fx.make_layout(
                (BLOCK_N, (wave_size, ROPE_COL_GROUP + 1), num_stages),
                (KV_LDS_STRIDE, (1, wave_size), stage_elems),
            )
        )
        s_vt = lds.kv.view(fx.make_layout((V_HEAD_DIM, BLOCK_N, num_stages), (1, KV_LDS_STRIDE, stage_elems)))

        # ---- Copy atoms and per-lane MMA operand copies -------------------
        g2r_kv = fx.make_copy_atom(fx.rocdl.BufferCopy(ROW_VEC * kv_raw.width), kv_raw)
        r2s_kv = fx.make_copy_atom(fx.UniversalCopy(128), fx.BFloat16)
        g2r_q = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        s2r_k = fx.make_copy_atom(fx.UniversalCopy(64), fx.BFloat16)
        s2r_vt = fx.make_copy_atom(fx.UniversalCopy(16), fx.BFloat16)
        r2g_o32 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        r2g_o16 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        thr_q = fx.make_tiled_copy_B(g2r_q, tiled_mma).get_slice(lane)
        thr_k = fx.make_tiled_copy_A(s2r_k, tiled_mma).get_slice(lane)
        thr_vt = fx.make_tiled_copy_A(s2r_vt, tiled_mma).get_slice(lane)
        thr_o32 = fx.make_tiled_copy_C(r2g_o32, tiled_mma).get_slice(lane)
        thr_o16 = fx.make_tiled_copy_C(r2g_o16, tiled_mma).get_slice(lane)

        # ---- Register fragments -------------------------------------------
        def shaped(m, n):
            return fx.make_view(lds.kv.ptr, fx.make_layout((m, n), (1, m)))

        f_q = [thr_mma.make_fragment_B(shaped(WAVE_ROWS, CHUNK_D)) for _ in range_constexpr(QK_CHUNKS)]
        f_k = thr_mma.make_fragment_A(shaped(BLOCK_N, CHUNK_D))
        f_s = thr_mma.make_fragment_C(shaped(BLOCK_N, WAVE_ROWS))
        f_p = thr_mma.make_fragment_B(shaped(WAVE_ROWS, BLOCK_N))
        f_vt = thr_mma.make_fragment_A(shaped(CHUNK_D, BLOCK_N))
        f_o = [thr_mma.make_fragment_C(shaped(CHUNK_D, WAVE_ROWS)) for _ in range_constexpr(V_CHUNKS)]
        f_o16 = fx.make_fragment_like(f_o[0], fx.BFloat16)
        r_kv_vecs = fx.make_rmem_tensor((ROW_VEC, ROWS_PER_WAVE), kv_raw)
        r_kv_rope = fx.make_rmem_tensor(ROWS_PER_WAVE, kv_raw)
        r_kv_bf16 = fx.make_rmem_tensor(ROW_VEC, fx.BFloat16)

        # Accumulator coordinates of this lane: value (v, m) of f_s is token
        # 16 * m + tok_in_tile[v] of the tile; every value of f_s / f_o is row lane_row.
        c_tv = mma_atom.layout_C_tv
        lane_row = fx.Int32(fx.get_scalar(fx.crd2idx((lane, 0), c_tv))) // MMA_TILE
        tok_in_tile = [fx.Int32(fx.get_scalar(fx.crd2idx((lane, v), c_tv))) % MMA_TILE for v in range_constexpr(4)]
        s_vals = 4 * (BLOCK_N // MMA_TILE)

        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)
        scale_log2 = sm_scale * fx.Float32(LOG2E)

        def wave_allreduce(x, combine):
            for sh in range_constexpr(2):
                x = combine(x, x.shuffle_xor(fx.Int32(MMA_TILE << sh), fx.Int32(wave_size)))
            return x

        def load_kv_tile(tile_start, kv_start, kv_len):
            """Gather this wave's rows of a KV tile into registers (clamped past kv_len)."""
            for k in range_constexpr(ROWS_PER_WAVE):
                tok = fx.min(tile_start + wave + NUM_WAVES * k, kv_len - 1)
                slot_pos = fx.idx2crd(tok, token_to_page)
                slot = fx.Int32(fx.get_scalar(fx.get(slot_pos, 0)))
                page = page_ids[kv_start + fx.Int32(fx.get_scalar(fx.get(slot_pos, 1)))]
                row = (slot, page)
                fx.copy(g2r_kv, kv_vecs[row, (None, lane)], r_kv_vecs[None, k])
                r_kv_rope[k] = kv_cols[row, (lane, ROPE_COL_GROUP)]

        def store_kv_tile(stage):
            if const_expr(kv_is_fp8):
                # Four rows' RoPE bytes share one i32 word for decoding.
                rope = _fp8_words_to_f32(r_kv_rope.load().bitcast(fx.Int32))
            for k in range_constexpr(ROWS_PER_WAVE):
                row = wave + NUM_WAVES * k
                if const_expr(kv_is_fp8):
                    nope = _fp8_words_to_f32(r_kv_vecs[None, k].load().bitcast(fx.Int32))
                    r_kv_bf16.store(fx.Vector.from_elements(nope, fx.Float32).to(fx.BFloat16))
                    fx.copy(r2s_kv, r_kv_bf16, s_kv_vecs[row, (None, lane), stage])
                    s_kv_cols[row, (lane, ROPE_COL_GROUP), stage] = rope[k].to(fx.BFloat16)
                else:
                    fx.copy(r2s_kv, r_kv_vecs[None, k], s_kv_vecs[row, (None, lane), stage])
                    s_kv_cols[row, (lane, ROPE_COL_GROUP), stage] = r_kv_rope[k]

        def attend_tile(stage, tile_start, kv_limit, m_old, l_old):
            """Online-softmax update of (f_o, m, l) with one LDS KV tile (log2 domain)."""
            k_chunks = fx.zipped_divide(s_kv[None, None, stage], fx.make_tile(BLOCK_N, CHUNK_D))
            f_s.fill(0.0)
            for c in range_constexpr(QK_CHUNKS):
                fx.copy(s2r_k, thr_k.partition_S(k_chunks[(None, None), (0, c)]), thr_k.retile(f_k))
                fx.gemm(mma_atom, f_s, f_k, f_q[c], f_s)

            s = f_s.load()
            scores = []
            for i in range_constexpr(s_vals):
                tok = tile_start + MMA_TILE * (i // 4) + tok_in_tile[i % 4]
                scores.append((tok < kv_limit).select(s[i] * scale_log2, neg_inf))
            m_tile = scores[0]
            for i in range_constexpr(1, s_vals):
                m_tile = fx.max(m_tile, scores[i])
            m_new = fx.max(m_old, wave_allreduce(m_tile, fx.max))
            m_safe = (m_new == neg_inf).select(zero, m_new)
            alpha = fx.math.exp2(m_old - m_safe)
            probs = [fx.math.exp2(x - m_safe) for x in scores]
            row_sum = probs[0]
            for i in range_constexpr(1, s_vals):
                row_sum = row_sum + probs[i]
            l_new = l_old * alpha + wave_allreduce(row_sum, lambda a, b: a + b)

            f_p.store(fx.Vector.from_elements(probs, fx.Float32).to(fx.BFloat16))
            vt_chunks = fx.zipped_divide(s_vt[None, None, stage], fx.make_tile(CHUNK_D, BLOCK_N))
            for c in range_constexpr(V_CHUNKS):
                fx.copy(s2r_vt, thr_vt.partition_S(vt_chunks[(None, None), (c, 0)]), thr_vt.retile(f_vt))
                f_o[c].store(f_o[c].load() * alpha)
                fx.gemm(mma_atom, f_o[c], f_vt, f_p, f_o[c])
            return m_new, l_new

        def load_q(q_row0):
            """Load 16 q rows starting at q_row0 as the resident B operand of S^T = K Q^T."""
            q_chunks = fx.zipped_divide(q_rows[(None, q_row0), None], fx.make_tile(WAVE_ROWS, CHUNK_D))
            for c in range_constexpr(QK_CHUNKS):
                fx.copy(g2r_q, thr_q.partition_S(q_chunks[(None, None), (0, c)]), thr_q.retile(f_q[c]))
            for c in range_constexpr(V_CHUNKS):
                f_o[c].fill(0.0)

        def store_split(out_row0, lse):
            out_chunks = fx.zipped_divide(split_rows[None, (None, out_row0)], fx.make_tile(CHUNK_D, WAVE_ROWS))
            for c in range_constexpr(V_CHUNKS):
                fx.copy(r2g_o32, thr_o32.retile(f_o[c]), thr_o32.partition_D(out_chunks[(None, None), (c, 0)]))
            if lane < WAVE_ROWS:
                fx.memref_store(lse, lse_rows, (lane_row, out_row0))

        def store_final(out_row0):
            out_chunks = fx.zipped_divide(final_rows[None, (None, out_row0)], fx.make_tile(CHUNK_D, WAVE_ROWS))
            for c in range_constexpr(V_CHUNKS):
                f_o16.store(f_o[c].load().to(fx.BFloat16))
                fx.copy(r2g_o16, thr_o16.retile(f_o16), thr_o16.partition_D(out_chunks[(None, None), (c, 0)]))

        def run_work(work):
            """Attention of one work: q rows [qo_start, qo_end) x KV pages [kv_start, kv_end)."""
            batch = work_info[work, 0]
            partial_qo_loc = work_info[work, 1]
            qo_start = work_info[work, 2]
            qo_end = work_info[work, 3]
            kv_start = work_info[work, 4]
            kv_end = work_info[work, 5]
            kv_offset = work_info[work, 6]

            qo_len = qo_end - qo_start
            num_pages = kv_end - kv_start
            last_page_len = (kv_offset == 0).select(last_page_lens[batch], fx.Int32(page_size))
            kv_len = (num_pages - 1) * page_size + last_page_len
            num_rows = qo_len * num_heads
            # Query token i sits at kv position kv_len + kv_offset - qo_len + i.
            causal_base = kv_len + kv_offset - qo_len + 1
            num_tiles = fx.ceildiv(kv_len, fx.Int32(BLOCK_N))
            num_blocks = (num_pages > 0).select(fx.ceildiv(num_rows, fx.Int32(BLOCK_M)), fx.Int32(0))
            is_split = partial_qo_loc >= 0
            out_row_base = is_split.select(partial_qo_loc, qo_start) * num_heads

            for block in range(num_blocks):
                wave_row0 = block * BLOCK_M + wave * WAVE_ROWS
                row = wave_row0 + lane_row
                kv_limit = fx.min(kv_len, causal_base + row // num_heads)

                load_q(qo_start * num_heads + wave_row0)
                load_kv_tile(0, kv_start, kv_len)
                store_kv_tile(0)
                fx.gpu.barrier()

                m = neg_inf
                l = zero
                for t in range(num_tiles):
                    load_kv_tile(fx.min(t + 1, num_tiles - 1) * BLOCK_N, kv_start, kv_len)
                    m, l = attend_tile(t % num_stages, t * BLOCK_N, kv_limit, m, l)
                    if const_expr(num_stages == 1):
                        fx.gpu.barrier()
                    store_kv_tile((t + 1) % num_stages)
                    fx.gpu.barrier()

                inv_l = (l > zero).select(fx.Float32(1.0) / l, zero)
                m_safe = (m == neg_inf).select(zero, m)
                lse = (m_safe + fx.math.log2(l)) * fx.Float32(LN2)
                for c in range_constexpr(V_CHUNKS):
                    f_o[c].store(f_o[c].load() * inv_l)

                if row < num_rows:
                    if is_split:
                        store_split(out_row_base + wave_row0, lse)
                    else:
                        store_final(out_row_base + wave_row0)

        for work in range(works[cu], works[cu + 1]):
            run_work(work)

    @flyc.jit
    def launch(
        query: fx.Tensor,
        kv_buffer: fx.Tensor,
        kv_page_indices: fx.Tensor,
        work_indptr: fx.Tensor,
        work_info_set: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        final_output: fx.Tensor,
        split_output: fx.Tensor,
        split_lse: fx.Tensor,
        sm_scale: fx.Float32,
        num_heads: fx.Constexpr[int],
        page_size: fx.Constexpr[int],
        num_cus: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        mla_fwd_decode_kernel(
            query,
            kv_buffer,
            kv_page_indices,
            work_indptr,
            work_info_set,
            kv_last_page_lens,
            final_output,
            split_output,
            split_lse,
            sm_scale,
            num_heads,
            page_size,
        ).launch(grid=(num_cus, 1, 1), block=(block_threads, 1, 1), stream=stream)

    return launch


def mla_fwd_decode_persistent(
    query,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info_set,
    kv_last_page_lens,
    final_output,
    split_output,
    split_lse,
    sm_scale,
):
    """Execute the persistent-decode work list on the GPU.

    Same contract as the work-list reference: unsplit works (partial_qo_loc < 0)
    write final_output[qo_start:qo_end]; split works write split_output /
    split_lse rows [partial_qo_loc, partial_qo_loc + qo_len). Rows with no
    visible key produce out = 0 and lse = -inf.
    """
    num_pages, page_size, nhead_kv, qk_dim = kv_buffer.shape
    num_heads = query.shape[1]
    assert nhead_kv == NUM_KV_HEADS and qk_dim == QK_HEAD_DIM and query.shape[2] == QK_HEAD_DIM
    assert final_output.shape[-1] == V_HEAD_DIM and split_output.shape[-1] == V_HEAD_DIM
    assert query.dtype == final_output.dtype == torch.bfloat16
    # fp8 must be the target's native format: the kernel decodes it with v_cvt_pk_f32_fp8.
    assert kv_buffer.dtype in (torch.bfloat16, dtypes.fp8), kv_buffer.dtype
    assert split_output.dtype == split_lse.dtype == torch.float32
    assert split_output.shape[1] == 1 and split_lse.shape[1] == 1
    for t in (query, kv_buffer, final_output, split_output, split_lse, work_info_set):
        assert t.is_contiguous()

    launch = _build_mla_fwd_decode(get_rocm_arch())
    launch(
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        kv_last_page_lens,
        final_output,
        split_output,
        split_lse,
        float(sm_scale),
        num_heads,
        page_size,
        work_indptr.shape[0] - 1,
    )
    return final_output, split_output, split_lse
