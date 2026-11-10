# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
MLA decode on gfx1250 for 128 query heads, FP8 Q/KV, BF16 output, one KV head, query
length 1, shuffled page-64 KV cache. One program per sequence, 4 warps.

- 128-token tiles of two 64-token pages, a 3-slot LDS ring for KV, page ids read one
  iteration early, and one online-softmax update per tile.
- The LoRA ring is split into two LDS partitions: hidden columns 0-255 in one (A), 256-511
  in the other (B).
- PV layout: each wave owns 128 output columns, so waves 0/2 read V only from A and
  waves 1/3 only from B.
- One fused TDM op per page: warps 0 and 2 load the LoRA plane (one partition each),
  warps 1 and 3 the RoPE plane.
- Lazy rescale: the reference max moves only when a row's max rises by more than
  LAZY_TH (log2 units), and the acc rescale is skipped per lane by a masked region when
  none of the lane's rows moved.
- The row max is taken on the unscaled S; only the max vector is scaled, and exp2 takes
  fma(c, S, -m).

Not bit-exact with the LDS pipeline of _mla_decode_fwd_kernel: the running max is taken
over 128 tokens at a time, so P is rounded to FP8 against different maxima.
LDS: 229,376 bytes.
"""

import torch
import triton.experimental.gluon.language as gl
import triton.language as tl
from triton.experimental import gluon

from aiter.ops.triton._gluon_kernels.gfx1250.attention.mla_decode_lazy_rescale_regions import (
    _any8,
    _rescale512,
)

# S = Q @ K^T: the 4 warps split the 128 query heads.
QK_WMMA: gl.constexpr = gl.amd.AMDWMMALayout(
    version=3,
    transposed=True,
    warp_bases=[(1, 0), (2, 0)],
    reg_bases=[],
    instr_shape=[16, 16, 64],
)
Q_DOT: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=QK_WMMA, k_width=32)
K_DOT: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=QK_WMMA, k_width=32)
PV_WMMA: gl.constexpr = gl.amd.AMDWMMALayout(
    version=3,
    transposed=True,
    warp_bases=[[0, 16], [0, 8]],
    reg_bases=[[0, 1], [0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
    instr_shape=[16, 16, 64],
)
P_DOT: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=PV_WMMA, k_width=32)
V_DOT: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=PV_WMMA, k_width=32)

# Q staging: 8 FP8 per thread, rows padded by 8 bytes in LDS.
Q_LORA_LOAD: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 8], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0]
)
Q_ROPE_LOAD: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[4, 1], order=[1, 0]
)
Q_LORA_SMEM: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
    [[512, 8]], [128, 512], [1, 0]
)
Q_ROPE_SMEM: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
    [[64, 8]], [128, 64], [1, 0]
)
# P staging between the QK and PV layouts.
# fmt: off
P_SMEM: gl.constexpr = gl.SharedLinearLayout(
    [
        [0, 1], [0, 2], [0, 4], [0, 16],
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
        [0, 8], [0, 32], [64, 0],
    ]
)
# fmt: on
O_SMEM3: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
    [[128, 8]], [4, 128, 128], [2, 1, 0]
)

KV_LORA_PART: gl.constexpr = gl.amd.gfx1250.PartitionedSharedLayout(
    num_partitions=2,
    num_groups=1,
    partition_dim=2,
    partition_layout=gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[2, 1, 0]
    ),
)
KV_ROPE_SMEM3: gl.constexpr = gl.SwizzledSharedLayout(
    vec=1, per_phase=1, max_phase=1, order=[2, 1, 0]
)

HINT_L = gl.constexpr(0b0101)
HINT_R = gl.constexpr(0b1010)
# Lazy rescale threshold, log2 units: p = exp2(S - M) stays <= 2**LAZY_TH = 256, which
# fits FP8 e4m3 (max 448).
LAZY_TH = gl.constexpr(8.0)


@gluon.jit
def _load_page(lora_desc, rope_desc, lora_dst, rope_dst, page):
    # One KV page into its ring entries with one fused TDM op.
    gl.amd.gfx1250.tdm.async_load_fused(
        [
            (
                gl.amd.gfx1250.tdm.update_tensor_descriptor(
                    lora_desc, add_offsets=[page, 0, 0], clamp_bounds=True
                ),
                lora_dst,
                HINT_L,
            ),
            (
                gl.amd.gfx1250.tdm.update_tensor_descriptor(
                    rope_desc, add_offsets=[page, 0, 0], clamp_bounds=True
                ),
                rope_dst,
                HINT_R,
            ),
        ],
    )


@gluon.jit
def _mla_decode_h128_fp8_partitioned_lds_kernel(
    output_ptr,  # [num_seqs, 128, 512] bf16
    query_ptr,  # [num_seqs, 128, 576] fp8
    kv_buffer_ptr,  # [num_pages, 1, 64, 576] fp8, shuffled
    block_tables_ptr: tl.const,  # [num_seqs, max_pages] int32
    seq_lens_ptr,  # [num_seqs] int32
    q_scale_ptr,  # [1] fp32
    kv_scale_ptr,  # [1] fp32
    block_tables_stride: gl.int32,
    query_stride_0: gl.int32,
    query_stride_1: gl.int32,
    kv_page_stride: gl.int32,  # kv_buffer.stride(1): bytes per (page, kv head)
    num_pages: gl.int32,
    output_stride_0: gl.int32,
    output_stride_1: gl.int32,
    SCALE: gl.constexpr,
    TDM_STORE: gl.constexpr,  # set by the launcher from the output's alignment
):
    BLOCK_M: gl.constexpr = 128  # query heads per program (all of them)
    PAGE: gl.constexpr = 64  # tokens per KV page
    PAGES: gl.constexpr = 2  # pages per tile
    TILE: gl.constexpr = PAGE * PAGES  # KV tokens per tile
    LORA: gl.constexpr = 512
    ROPE: gl.constexpr = 64
    SHUF_K: gl.constexpr = 16  # shuffle tile width in the KV layout
    NUM_SLOTS: gl.constexpr = 3  # LDS ring depth, in 128-token tiles
    OPS: gl.constexpr = PAGES  # TDM ops per tile (one fused op per page)
    D: gl.constexpr = NUM_SLOTS - 1  # tiles in flight ahead of the one computed
    RCP_LN2: gl.constexpr = 1.4426950408889634

    seq_idx = gl.program_id(0)
    seq_len = gl.load(seq_lens_ptr + seq_idx)
    if seq_len <= 0:
        return
    seq_pages = (seq_len + PAGE - 1) // PAGE
    num_tiles = (seq_len + TILE - 1) // TILE

    q_scale = gl.load(q_scale_ptr)
    kv_scale = gl.load(kv_scale_ptr)
    # S stays unscaled: since qk_factor > 0, max(c*S) == c*max(S), so only the 128-entry
    # row max is scaled, and exp2 takes fma(c, S, -m).
    qk_factor = SCALE * RCP_LN2 * q_scale * kv_scale
    out_factor = kv_scale

    # ---- Q: global -> registers -> LDS -> WMMA A-operand layout ------------
    q_base = query_ptr + seq_idx * query_stride_0
    m_lora = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_LORA_LOAD))
    d_lora = gl.arange(0, LORA, layout=gl.SliceLayout(0, Q_LORA_LOAD))
    q_lora_smem = gl.allocate_shared_memory(
        query_ptr.type.element_ty, [BLOCK_M, LORA], Q_LORA_SMEM
    )
    q_lora_smem.store(
        gl.load(q_base + m_lora[:, None] * query_stride_1 + d_lora[None, :])
    )
    q_lora = q_lora_smem.load(layout=Q_DOT)
    m_rope = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_ROPE_LOAD))
    d_rope = gl.arange(0, ROPE, layout=gl.SliceLayout(0, Q_ROPE_LOAD))
    q_rope_smem = gl.allocate_shared_memory(
        query_ptr.type.element_ty, [BLOCK_M, ROPE], Q_ROPE_SMEM
    )
    q_rope_smem.store(
        gl.load(q_base + m_rope[:, None] * query_stride_1 + (LORA + d_rope)[None, :])
    )
    q_rope = q_rope_smem.load(layout=Q_DOT)

    # ---- KV: TDM descriptors and the ring (slot s = entries 2s and 2s+1) ---
    # LoRA: 3-D so that one descriptor covers all pages (page stride) and the token
    # groups (8192 B stride).
    kv_lora_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=kv_buffer_ptr,
        shape=(num_pages, PAGE // 16, 16 * LORA),
        strides=(kv_page_stride, 16 * LORA, 1),
        block_shape=(1, PAGE // 16, 16 * LORA),
        layout=KV_LORA_PART,
    )
    k_rope_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=kv_buffer_ptr + PAGE * LORA,
        shape=(num_pages, 1, PAGE * ROPE),
        strides=(kv_page_stride, PAGE * ROPE, 1),
        block_shape=(1, 1, PAGE * ROPE),
        layout=KV_ROPE_SMEM3,
    )
    kv_lora_ring = gl.allocate_shared_memory(
        kv_lora_desc.dtype,
        [NUM_SLOTS * PAGES, 1, PAGE // 16, 16 * LORA],
        layout=KV_LORA_PART,
    )
    k_rope_ring = gl.allocate_shared_memory(
        k_rope_desc.dtype, [NUM_SLOTS * PAGES, 1, 1, PAGE * ROPE], layout=KV_ROPE_SMEM3
    )
    table = block_tables_ptr + seq_idx * block_tables_stride

    # Prime tiles 0..D-1. Page ids are clamped to the sequence's last page: a second
    # page past the end reloads that page, and its tokens are masked. Unconditional on
    # the long path so that UpdateAsyncWaitCount sees every load.
    if num_tiles > D:
        for prime in gl.static_range(D):
            for h in gl.static_range(PAGES):
                page = gl.load(table + gl.minimum(prime * PAGES + h, seq_pages - 1)).to(
                    gl.int32
                )
                _load_page(
                    kv_lora_desc,
                    k_rope_desc,
                    kv_lora_ring.index(prime * PAGES + h),
                    k_rope_ring.index(prime * PAGES + h),
                    page,
                )
    else:
        for prime in gl.static_range(D):
            if num_tiles > prime:
                for h in gl.static_range(PAGES):
                    page = gl.load(
                        table + gl.minimum(prime * PAGES + h, seq_pages - 1)
                    ).to(gl.int32)
                    _load_page(
                        kv_lora_desc,
                        k_rope_desc,
                        kv_lora_ring.index(prime * PAGES + h),
                        k_rope_ring.index(prime * PAGES + h),
                        page,
                    )

    M = gl.full([BLOCK_M], float("-inf"), gl.float32, gl.SliceLayout(1, QK_WMMA))
    L = gl.full([BLOCK_M], 1.0, gl.float32, gl.SliceLayout(1, QK_WMMA))
    acc = gl.zeros([BLOCK_M, LORA], gl.float32, PV_WMMA)
    slot: gl.int32 = 0

    # ---- steady state ---------------------------------------------------------
    # Fill the slot D tiles ahead, then wait until only the D newest tiles (OPS*D ops)
    # are outstanding: the current tile has landed. The two page ids for a fill are
    # read one iteration earlier, so the TDM issue does not wait on the block table.
    if num_tiles > D:
        next0 = gl.load(table + gl.minimum(D * PAGES, seq_pages - 1))
        next1 = gl.load(table + gl.minimum(D * PAGES + 1, seq_pages - 1))
        for tile in range(num_tiles - D):
            fill = (slot + D) % NUM_SLOTS
            page0 = next0.to(gl.int32)
            page1 = next1.to(gl.int32)
            # Page ids for the next iteration's fill; clamped to stay in bounds.
            nxt = (tile + D + 1) * PAGES
            next0 = gl.load(table + gl.minimum(nxt, seq_pages - 1))
            next1 = gl.load(table + gl.minimum(nxt + 1, seq_pages - 1))
            _load_page(
                kv_lora_desc,
                k_rope_desc,
                kv_lora_ring.index(fill * PAGES),
                k_rope_ring.index(fill * PAGES),
                page0,
            )
            _load_page(
                kv_lora_desc,
                k_rope_desc,
                kv_lora_ring.index(fill * PAGES + 1),
                k_rope_ring.index(fill * PAGES + 1),
                page1,
            )
            gl.amd.gfx1250.tdm.async_wait(OPS * D)

            # Per page h: un-shuffle in the LDS view, (tok/16, dim/32, 2, 16 tok, 16 dim)
            # -> [64 tok, dim]; K^T is its transpose, V is it as is. S_h = Q @ K_h^T.
            lora0 = (
                kv_lora_ring.index(slot * PAGES)
                .reshape((1, PAGE // 16, LORA // (2 * SHUF_K), 2, 16, SHUF_K))
                .permute((0, 1, 4, 2, 3, 5))
                .reshape((PAGE, LORA))
            )
            rope0 = (
                k_rope_ring.index(slot * PAGES)
                .reshape((1, PAGE // 16, ROPE // (2 * SHUF_K), 2, 16, SHUF_K))
                .permute((0, 1, 4, 2, 3, 5))
                .reshape((PAGE, ROPE))
            )
            lora1 = (
                kv_lora_ring.index(slot * PAGES + 1)
                .reshape((1, PAGE // 16, LORA // (2 * SHUF_K), 2, 16, SHUF_K))
                .permute((0, 1, 4, 2, 3, 5))
                .reshape((PAGE, LORA))
            )
            rope1 = (
                k_rope_ring.index(slot * PAGES + 1)
                .reshape((1, PAGE // 16, ROPE // (2 * SHUF_K), 2, 16, SHUF_K))
                .permute((0, 1, 4, 2, 3, 5))
                .reshape((PAGE, ROPE))
            )
            S0 = gl.zeros([BLOCK_M, PAGE], gl.float32, QK_WMMA)
            S0 = gl.amd.gfx1250.wmma(q_lora, lora0.permute((1, 0)).load(K_DOT), S0)
            S0 = gl.amd.gfx1250.wmma(q_rope, rope0.permute((1, 0)).load(K_DOT), S0)
            S1 = gl.zeros([BLOCK_M, PAGE], gl.float32, QK_WMMA)
            S1 = gl.amd.gfx1250.wmma(q_lora, lora1.permute((1, 0)).load(K_DOT), S1)
            S1 = gl.amd.gfx1250.wmma(q_rope, rope1.permute((1, 0)).load(K_DOT), S1)
            # V is read now, so its LDS latency hides behind the softmax.
            v0 = lora0.load(V_DOT)
            v1 = lora1.load(V_DOT)

            # One online-softmax update and one acc rescale for all 128 tokens.
            m_cand = gl.maximum(gl.max(S0, axis=1), gl.max(S1, axis=1))
            m_cand = m_cand * qk_factor
            # Keep the reference max M unless this tile's max exceeds it by more than
            # LAZY_TH; rows that keep their reference get alpha == 1 exactly.
            m_new = gl.where(m_cand - M > LAZY_TH, m_cand, M)
            p0 = gl.exp2(S0 * qk_factor - m_new[:, None])
            p1 = gl.exp2(S1 * qk_factor - m_new[:, None])
            alpha = gl.exp2(M - m_new)
            M = m_new
            alpha_pv = gl.convert_layout(alpha, gl.SliceLayout(1, PV_WMMA))
            moved = gl.map_elementwise(_any8, alpha_pv, pack=8)[0]
            acc = gl.map_elementwise(
                _rescale512, acc, alpha_pv[:, None], moved[:, None], pack=512
            )[0]
            L = L * alpha + gl.sum(p0, 1) + gl.sum(p1, 1)

            # P_h: fp32 QK layout -> fp8 PV A-operand through LDS, then acc += P_h @ V_h.
            p0_smem = gl.allocate_shared_memory(v0.dtype, (BLOCK_M, PAGE), P_SMEM)
            p0_smem.store(p0.to(v0.dtype))
            acc = gl.amd.gfx1250.wmma(p0_smem.load(P_DOT), v0, acc)
            p1_smem = gl.allocate_shared_memory(v1.dtype, (BLOCK_M, PAGE), P_SMEM)
            p1_smem.store(p1.to(v1.dtype))
            acc = gl.amd.gfx1250.wmma(p1_smem.load(P_DOT), v1, acc)
            slot = (slot + 1) % NUM_SLOTS

    # ---- drain: the last min(D, num_tiles) tiles -----------------------------
    for k in gl.static_range(D):
        tile = num_tiles - D + k
        if tile >= 0:
            gl.amd.gfx1250.tdm.async_wait(OPS * (D - 1 - k))
            lora0 = (
                kv_lora_ring.index(slot * PAGES)
                .reshape((1, PAGE // 16, LORA // (2 * SHUF_K), 2, 16, SHUF_K))
                .permute((0, 1, 4, 2, 3, 5))
                .reshape((PAGE, LORA))
            )
            rope0 = (
                k_rope_ring.index(slot * PAGES)
                .reshape((1, PAGE // 16, ROPE // (2 * SHUF_K), 2, 16, SHUF_K))
                .permute((0, 1, 4, 2, 3, 5))
                .reshape((PAGE, ROPE))
            )
            lora1 = (
                kv_lora_ring.index(slot * PAGES + 1)
                .reshape((1, PAGE // 16, LORA // (2 * SHUF_K), 2, 16, SHUF_K))
                .permute((0, 1, 4, 2, 3, 5))
                .reshape((PAGE, LORA))
            )
            rope1 = (
                k_rope_ring.index(slot * PAGES + 1)
                .reshape((1, PAGE // 16, ROPE // (2 * SHUF_K), 2, 16, SHUF_K))
                .permute((0, 1, 4, 2, 3, 5))
                .reshape((PAGE, ROPE))
            )
            S0 = gl.zeros([BLOCK_M, PAGE], gl.float32, QK_WMMA)
            S0 = gl.amd.gfx1250.wmma(q_lora, lora0.permute((1, 0)).load(K_DOT), S0)
            S0 = gl.amd.gfx1250.wmma(q_rope, rope0.permute((1, 0)).load(K_DOT), S0)
            S1 = gl.zeros([BLOCK_M, PAGE], gl.float32, QK_WMMA)
            S1 = gl.amd.gfx1250.wmma(q_lora, lora1.permute((1, 0)).load(K_DOT), S1)
            S1 = gl.amd.gfx1250.wmma(q_rope, rope1.permute((1, 0)).load(K_DOT), S1)
            if k == D - 1:
                # Last tile: mask tokens past seq_len (partial or reloaded page).
                pos0 = tile * TILE + gl.arange(
                    0, PAGE, layout=gl.SliceLayout(0, QK_WMMA)
                )
                S0 = gl.where(pos0[None, :] < seq_len, S0, float("-inf"))
                S1 = gl.where((pos0 + PAGE)[None, :] < seq_len, S1, float("-inf"))
            v0 = lora0.load(V_DOT)
            v1 = lora1.load(V_DOT)

            m_cand = gl.maximum(gl.max(S0, axis=1), gl.max(S1, axis=1))
            m_cand = m_cand * qk_factor
            m_new = gl.where(m_cand - M > LAZY_TH, m_cand, M)
            p0 = gl.exp2(S0 * qk_factor - m_new[:, None])
            p1 = gl.exp2(S1 * qk_factor - m_new[:, None])
            alpha = gl.exp2(M - m_new)
            M = m_new
            alpha_pv = gl.convert_layout(alpha, gl.SliceLayout(1, PV_WMMA))
            moved = gl.map_elementwise(_any8, alpha_pv, pack=8)[0]
            acc = gl.map_elementwise(
                _rescale512, acc, alpha_pv[:, None], moved[:, None], pack=512
            )[0]
            L = L * alpha + gl.sum(p0, 1) + gl.sum(p1, 1)

            p0_smem = gl.allocate_shared_memory(v0.dtype, (BLOCK_M, PAGE), P_SMEM)
            p0_smem.store(p0.to(v0.dtype))
            acc = gl.amd.gfx1250.wmma(p0_smem.load(P_DOT), v0, acc)
            p1_smem = gl.allocate_shared_memory(v1.dtype, (BLOCK_M, PAGE), P_SMEM)
            p1_smem.store(p1.to(v1.dtype))
            acc = gl.amd.gfx1250.wmma(p1_smem.load(P_DOT), v1, acc)
            slot = (slot + 1) % NUM_SLOTS

    # ---- epilogue: O = acc / L * kv_scale -> bf16 ----------------------------
    out_base = output_ptr + seq_idx * output_stride_0
    factor = gl.convert_layout(1.0 / L[:, None], layout=PV_WMMA) * out_factor
    if TDM_STORE:
        STORE_N: gl.constexpr = 128
        o_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=out_base,
            shape=(LORA // STORE_N, BLOCK_M, STORE_N),
            strides=(STORE_N, output_stride_1, 1),
            block_shape=(LORA // STORE_N, BLOCK_M, STORE_N),
            layout=O_SMEM3,
        )
        o = (acc * factor).to(output_ptr.type.element_ty)
        o = gl.permute(gl.reshape(o, [BLOCK_M, LORA // STORE_N, STORE_N]), [1, 0, 2])
        o_smem = gl.allocate_shared_memory(
            output_ptr.type.element_ty, [LORA // STORE_N, BLOCK_M, STORE_N], O_SMEM3
        )
        o_smem.store(o)
        gl.amd.gfx1250.tdm.async_store(o_desc, [0, 0, 0], o_smem)
        o_smem._keep_alive()
    else:
        m_o = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, PV_WMMA))
        d_o = gl.arange(0, LORA, layout=gl.SliceLayout(0, PV_WMMA))
        gl.store(
            out_base + m_o[:, None] * output_stride_1 + d_o[None, :],
            (acc * factor).to(output_ptr.type.element_ty),
        )


def mla_decode_h128_fp8_partitioned_lds(
    q, kv_buffer, out, seq_lens, block_tables, softmax_scale, q_scale, kv_scale
):
    """Decode attention for q [B, 128, 576] fp8 and a shuffled kv_buffer [P, 1, 64, 576]
    fp8 into out [B, 128, 512] bf16. q_scale and kv_scale are [1] fp32 tensors."""
    fp8 = torch.float8_e4m3fn
    num_seqs, nhead, head_dim = q.shape
    num_pages, num_kv_heads, page, kv_dim = kv_buffer.shape
    assert q.dtype == fp8 and kv_buffer.dtype == fp8 and out.dtype == torch.bfloat16
    assert nhead == 128 and num_kv_heads == 1 and page == 64
    assert head_dim == kv_dim == 576 and out.shape == (num_seqs, 128, 512)
    assert q.stride(2) == 1 and kv_buffer.stride(3) == 1 and out.stride(2) == 1
    assert block_tables.stride(1) == 1 and len(seq_lens) == num_seqs
    tdm_store = (
        out.data_ptr() % 16 == 0 and out.stride(0) % 8 == 0 and out.stride(1) % 8 == 0
    )
    _mla_decode_h128_fp8_partitioned_lds_kernel[(num_seqs,)](
        out,
        q,
        kv_buffer,
        block_tables,
        seq_lens,
        q_scale,
        kv_scale,
        block_tables.stride(0),
        q.stride(0),
        q.stride(1),
        kv_buffer.stride(1),
        num_pages,
        out.stride(0),
        out.stride(1),
        SCALE=softmax_scale,
        TDM_STORE=tdm_store,
        num_warps=4,
        waves_per_eu=1,
    )
    return out
