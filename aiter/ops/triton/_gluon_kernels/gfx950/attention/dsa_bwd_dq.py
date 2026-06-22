"""
Gluon dQ backward kernel for DeepSeek V4 sparse MLA (gfx950 / MI355X).

M1 port of the Triton `_bwd_chunk_dq_store_ds_v4` (V4 chunked_gather dQ) onto the
gluon hardware-control structure of Leon's V3.2 forward.

Per program: 1 query token x BLOCK_H heads x one rank chunk [R_START, R_START+R_CHUNK).
Grid: (total_tokens, cdiv(num_heads, BLOCK_H)).  Dispatch loops chunks externally,
dQ is read-modify-written across chunks (IS_FIRST_CHUNK zero-inits).

Per-tile math (TILE_K wide, looping NUM_TILES = R_CHUNK / TILE_K):
    S       = Q_lora @ K_lora_T + Q_rope @ K_rope_T      # 2 MMAs (mfma_s),  contract over D
    P       = exp(S*scale - lse)                          # lse is sink-inclusive (from fwd)
    dP      = dO @ K_lora_T                               # 1 MMA (mfma_s),  reuses K_lora_T_dot
    dS      = P * (dP - delta) * scale
    dQ_lora += dS @ K_lora                                # 1 MMA (mfma_acc), K_lora = K_lora_T.T view
    dQ_rope += dS @ K_rope                                # 1 MMA (mfma_acc), K_rope = K_rope_T.T view
    store dS, P chunk -> HBM (consumed by dKV-intermediate kernel)

Differences vs Leon's fwd (the structural template):
  * dO added as a second stationary [BH, D_V] operand (async-loaded with Q).
  * No online softmax: single P = exp(S - lse) (lse precomputed by fwd).
  * 5 MMAs/tile vs 3; K_lora read 3 ways (S, dP, dQ_lora), K_rope 2 ways.
  * Per-tile dS/P stores + final dQ store (RMW across chunks) replace the O/LSE write.
  * Sink: d_sink is NOT done here (handled by a torch reduction in the launcher),
    so this kernel needs no atomics. lse from fwd already folds the sink in.

M1 config: BLOCK_H=32, TILE_K=16 (matches Triton TILE_K_DQ and the 16x16x16 MFMA
k-dim). LDS at BH=32/TILE_K=16 ~= 104 KB < 160 KB (gfx950).
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _sparse_mla_bwd_dq_gl_kernel(
    Q_ptr,          # [T, H, D_QK] bf16
    KV_ptr,         # [T, 1, D_QK] bf16
    dO_ptr,         # [T, H, D_V]  bf16
    TopK_ptr,       # [T, TOPK_padded] int32
    LSE_ptr,        # [T, H] fp32 (sink-inclusive)
    Delta_ptr,      # [T, H] fp32
    dQ_ptr,         # [T, H, D_QK] bf16 (read-modify-write across chunks)
    dS_ptr,         # [T, H, R_CHUNK] bf16
    P_ptr,          # [T, H, R_CHUNK] bf16
    stride_q_t: tl.int64, stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64, stride_do_h: tl.int64,
    stride_dq_t: tl.int64, stride_dq_h: tl.int64,
    stride_topk_t: tl.int64,
    stride_ds_t: tl.int64, stride_ds_h: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    R_START: tl.int32,
    R_CHUNK: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    D_V: gl.constexpr,
    D_ROPE: gl.constexpr,
    IS_FIRST_CHUNK: gl.constexpr,
):
    # ===================== constexpr layouts =====================
    mfma_s: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[4, 1],
    )
    mfma_acc: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[4, 1],
    )

    # ---- Blocked layouts for global loads (per Leon's fwd) ----
    _qlora_tpw_k: gl.constexpr = min(64, D_V // 8)
    _qlora_tpw_m: gl.constexpr = 64 // _qlora_tpw_k
    blk_qlora: gl.constexpr = gl.BlockedLayout(          # [BLOCK_H, D_V]  (Q_lora, dO, dQ_lora)
        size_per_thread=[1, 8], threads_per_warp=[_qlora_tpw_m, _qlora_tpw_k],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    blk_qrope: gl.constexpr = gl.BlockedLayout(          # [BLOCK_H, D_ROPE]
        size_per_thread=[1, 8], threads_per_warp=[8, 8],
        warps_per_cta=[4, 1], order=[1, 0],
    )

    _klora_tpw_m: gl.constexpr = min(64, D_V // 8)
    _klora_tpw_n: gl.constexpr = 64 // _klora_tpw_m
    blk_klora: gl.constexpr = gl.BlockedLayout(          # [D_V, TILE_K]
        size_per_thread=[8, 1], threads_per_warp=[_klora_tpw_m, _klora_tpw_n],
        warps_per_cta=[1, 4], order=[0, 1],
    )
    blk_krope: gl.constexpr = gl.BlockedLayout(          # [D_ROPE, TILE_K]
        size_per_thread=[2, 1], threads_per_warp=[32, 2],
        warps_per_cta=[1, 4], order=[0, 1],
    )
    # ---- Shared layouts (only K is staged in LDS) ----
    sh_klora: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [D_V, TILE_K], [0, 1],
    )
    sh_krope: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[0, 1])

    # ---- Dot operand layouts ----
    dot_qlora_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_s, k_width=8)
    dot_qrope_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_s, k_width=8)
    dot_do_a:    gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_s, k_width=8)
    dot_klora_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_s, k_width=8)
    dot_krope_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_s, k_width=8)
    dot_ds_a:      gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_acc, k_width=4)
    dot_klora_v_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_acc, k_width=4)
    dot_krope_v_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_acc, k_width=4)

    # ===================== program ids =====================
    token_idx = gl.program_id(axis=0)
    hg_idx = gl.program_id(axis=1)
    hg_offset = hg_idx * BLOCK_H

    NUM_TILES: gl.constexpr = R_CHUNK // TILE_K

    # ===================== Q / dO offsets =====================
    offs_h_qlora = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_qlora = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_qlora = offs_h_qlora < num_heads
    q_base = token_idx.to(tl.int64) * stride_q_t
    q_offs_lora = q_base + offs_h_qlora[:, None].to(tl.int64) * stride_q_h + offs_v_qlora[None, :].to(tl.int64)
    q_mask_lora = mask_h_qlora[:, None]

    offs_h_qrope = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qrope))
    offs_r_qrope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(0, blk_qrope))
    mask_h_qrope = offs_h_qrope < num_heads
    q_offs_rope = q_base + offs_h_qrope[:, None].to(tl.int64) * stride_q_h + (D_V + offs_r_qrope[None, :]).to(tl.int64)
    q_mask_rope = mask_h_qrope[:, None]

    offs_h_do = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_do = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_do = offs_h_do < num_heads
    do_base = token_idx.to(tl.int64) * stride_do_t
    do_offs = do_base + offs_h_do[:, None].to(tl.int64) * stride_do_h + offs_v_do[None, :].to(tl.int64)
    do_mask = mask_h_do[:, None]

    # ===================== load Q_lora, Q_rope, dO -> registers (no LDS staging) =====================
    # Stationary opIdx-0 operands: load HBM->VGPR (blocked, coalesced) then convert to
    # the dot-operand layout once. The convert's LDS scratch is transient (freed before
    # the K loop), unlike persistent staging, so only K occupies LDS during the loop.
    q_lora_blk = gl.amd.cdna4.buffer_load(ptr=Q_ptr, offsets=q_offs_lora.to(tl.int32), mask=q_mask_lora, other=0.0)
    q_rope_blk = gl.amd.cdna4.buffer_load(ptr=Q_ptr, offsets=q_offs_rope.to(tl.int32), mask=q_mask_rope, other=0.0)
    do_blk = gl.amd.cdna4.buffer_load(ptr=dO_ptr, offsets=do_offs.to(tl.int32), mask=do_mask, other=0.0)
    Q_lora_dot = gl.convert_layout(q_lora_blk, dot_qlora_a)
    Q_rope_dot = gl.convert_layout(q_rope_blk, dot_qrope_a)
    dO_dot = gl.convert_layout(do_blk, dot_do_a)

    # ===================== topk / KV offsets =====================
    topk_base = token_idx.to(tl.int64) * stride_topk_t + R_START

    offs_tile_klora = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_klora))
    offs_tile_krope = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_krope))
    offs_tile_mma = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, mfma_s))
    offs_v_klora = gl.arange(0, D_V, layout=gl.SliceLayout(1, blk_klora))
    offs_r_krope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(1, blk_krope))

    # ===================== shared mem for K loop (double-buffered) =====================
    smem_krope = gl.allocate_shared_memory(KV_ptr.dtype.element_ty, [2, D_ROPE, TILE_K], layout=sh_krope)
    smem_klora = gl.allocate_shared_memory(KV_ptr.dtype.element_ty, [2, D_V, TILE_K], layout=sh_klora)

    # ===================== dQ accumulators =====================
    # Always zero-init; the read-modify-write across chunks is folded in at STORE
    # time in the blocked layout (avoids a big blocked->mfma_acc convert_layout
    # whose LDS scratch would overflow the K/Q/dO buffers).
    dQ_lora = gl.zeros([BLOCK_H, D_V], dtype=gl.float32, layout=mfma_acc)
    dQ_rope = gl.zeros([BLOCK_H, D_ROPE], dtype=gl.float32, layout=mfma_acc)

    # ===================== lse / delta (in mfma_s row-slice layout) =====================
    offs_h_s = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_s))
    mask_h_s = offs_h_s < num_heads
    lse = gl.amd.cdna4.buffer_load(ptr=LSE_ptr, offsets=(token_idx * num_heads + offs_h_s).to(tl.int32), mask=mask_h_s, other=0.0)
    delta = gl.amd.cdna4.buffer_load(ptr=Delta_ptr, offsets=(token_idx * num_heads + offs_h_s).to(tl.int32), mask=mask_h_s, other=0.0)

    # ===================== prologue: K tile 0 (group B, buffer 0) =====================
    topk_pos_klora = gl.amd.cdna4.buffer_load(ptr=TopK_ptr, offsets=(topk_base + offs_tile_klora).to(tl.int32), mask=offs_tile_klora < R_CHUNK, other=-1)
    topk_pos_krope = gl.amd.cdna4.buffer_load(ptr=TopK_ptr, offsets=(topk_base + offs_tile_krope).to(tl.int32), mask=offs_tile_krope < R_CHUNK, other=-1)
    topk_pos_mma = gl.amd.cdna4.buffer_load(ptr=TopK_ptr, offsets=(topk_base + offs_tile_mma).to(tl.int32), mask=offs_tile_mma < R_CHUNK, other=-1)

    valid_klora = topk_pos_klora != -1
    valid_krope = topk_pos_krope != -1
    valid_mma = topk_pos_mma != -1
    safe_klora = gl.where(valid_klora, topk_pos_klora, 0)
    safe_krope = gl.where(valid_krope, topk_pos_krope, 0)

    klora_offs = safe_klora[None, :].to(tl.int64) * stride_kv_t + offs_v_klora[:, None].to(tl.int64)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(dest=smem_klora.index(0), ptr=KV_ptr, offsets=klora_offs.to(tl.int32), mask=valid_klora[None, :])
    krope_offs = safe_krope[None, :].to(tl.int64) * stride_kv_t + (D_V + offs_r_krope[:, None]).to(tl.int64)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(dest=smem_krope.index(0), ptr=KV_ptr, offsets=krope_offs.to(tl.int32), mask=valid_krope[None, :])
    gl.amd.cdna4.async_copy.commit_group()

    # dS / P store offsets in the mfma_s layout (store directly from the compute
    # layout -> no convert_layout/LDS shuffle; less-coalesced HBM write instead).
    ds_base = token_idx.to(tl.int64) * stride_ds_t + hg_idx.to(tl.int64) * BLOCK_H * stride_ds_h
    offs_h_dsp = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_s))
    offs_tile_dsp = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, mfma_s))
    mask_h_dsp = (hg_offset + offs_h_dsp) < num_heads

    # ===================== main loop: prefetch t+1, compute t =====================
    cur_buf = 0
    for t in range(NUM_TILES - 1):
        next_offs_klora = (t + 1) * TILE_K + offs_tile_klora
        next_offs_krope = (t + 1) * TILE_K + offs_tile_krope
        next_offs_mma = (t + 1) * TILE_K + offs_tile_mma

        topk_pos_klora_next = gl.amd.cdna4.buffer_load(ptr=TopK_ptr, offsets=(topk_base + next_offs_klora).to(tl.int32), mask=next_offs_klora < R_CHUNK, other=-1)
        topk_pos_krope_next = gl.amd.cdna4.buffer_load(ptr=TopK_ptr, offsets=(topk_base + next_offs_krope).to(tl.int32), mask=next_offs_krope < R_CHUNK, other=-1)
        topk_pos_mma_next = gl.amd.cdna4.buffer_load(ptr=TopK_ptr, offsets=(topk_base + next_offs_mma).to(tl.int32), mask=next_offs_mma < R_CHUNK, other=-1)

        valid_klora_next = (next_offs_klora < R_CHUNK) & (topk_pos_klora_next != -1)
        valid_krope_next = (next_offs_krope < R_CHUNK) & (topk_pos_krope_next != -1)
        valid_mma_next = (next_offs_mma < R_CHUNK) & (topk_pos_mma_next != -1)
        safe_klora_next = gl.where(valid_klora_next, topk_pos_klora_next, 0)
        safe_krope_next = gl.where(valid_krope_next, topk_pos_krope_next, 0)

        next_buf = 1 - cur_buf
        klora_offs_next = safe_klora_next[None, :].to(tl.int64) * stride_kv_t + offs_v_klora[:, None].to(tl.int64)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(dest=smem_klora.index(next_buf), ptr=KV_ptr, offsets=klora_offs_next.to(tl.int32), mask=valid_klora_next[None, :])
        krope_offs_next = safe_krope_next[None, :].to(tl.int64) * stride_kv_t + (D_V + offs_r_krope[:, None]).to(tl.int64)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(dest=smem_krope.index(next_buf), ptr=KV_ptr, offsets=krope_offs_next.to(tl.int32), mask=valid_krope_next[None, :])
        gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.wait_group(1)

        # ----- read K views from current buffer -----
        klora_smem_cur = smem_klora.index(cur_buf)
        K_lora_T_dot = klora_smem_cur.load(dot_klora_b)              # [D_V, TILE_K]  opIdx1 mfma_s
        K_lora_v_dot = klora_smem_cur.permute([1, 0]).load(dot_klora_v_b)  # [TILE_K, D_V] opIdx1 mfma_acc
        krope_smem_cur = smem_krope.index(cur_buf)
        K_rope_T_dot = krope_smem_cur.load(dot_krope_b)
        K_rope_v_dot = krope_smem_cur.permute([1, 0]).load(dot_krope_v_b)

        # ----- S = Q_lora@K_lora_T + Q_rope@K_rope_T -----
        S = gl.amd.cdna4.mfma(Q_lora_dot, K_lora_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_s))
        S = gl.amd.cdna4.mfma(Q_rope_dot, K_rope_T_dot, S)
        S = S * scale
        offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_s))
        valid_mask = valid_mma[None, :] & (offs_h_mma < num_heads)[:, None]
        S = gl.where(valid_mask, S, float("-inf"))

        # ----- P = exp(S - lse) ; dP = dO@K_lora_T ; dS = P*(dP-delta)*scale -----
        P = gl.exp(S - lse[:, None])
        P = gl.where(valid_mask, P, 0.0)
        dP = gl.amd.cdna4.mfma(dO_dot, K_lora_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_s))
        dS = P * (dP - delta[:, None]) * scale
        dS = gl.where(valid_mask, dS, 0.0)

        # ----- dQ_lora += dS@K_lora ; dQ_rope += dS@K_rope -----
        dS_bf = dS.to(KV_ptr.dtype.element_ty)
        dS_dot = gl.convert_layout(dS_bf, dot_ds_a)
        dQ_lora = gl.amd.cdna4.mfma(dS_dot, K_lora_v_dot, dQ_lora)
        dQ_rope = gl.amd.cdna4.mfma(dS_dot, K_rope_v_dot, dQ_rope)

        # ----- store dS, P chunk -----
        col = t * TILE_K + offs_tile_dsp
        dsp_offs = ds_base + offs_h_dsp[:, None].to(tl.int64) * stride_ds_h + col[None, :].to(tl.int64)
        gl.amd.cdna4.buffer_store(stored_value=dS_bf, ptr=dS_ptr, offsets=dsp_offs.to(tl.int32), mask=mask_h_dsp[:, None])
        gl.amd.cdna4.buffer_store(stored_value=P.to(KV_ptr.dtype.element_ty), ptr=P_ptr, offsets=dsp_offs.to(tl.int32), mask=mask_h_dsp[:, None])

        # promote prefetch -> current
        cur_buf = next_buf
        valid_mma = valid_mma_next

    # ===================== epilogue: last tile =====================
    gl.amd.cdna4.async_copy.wait_group(0)
    t = NUM_TILES - 1
    klora_smem_cur = smem_klora.index(cur_buf)
    K_lora_T_dot = klora_smem_cur.load(dot_klora_b)
    K_lora_v_dot = klora_smem_cur.permute([1, 0]).load(dot_klora_v_b)
    krope_smem_cur = smem_krope.index(cur_buf)
    K_rope_T_dot = krope_smem_cur.load(dot_krope_b)
    K_rope_v_dot = krope_smem_cur.permute([1, 0]).load(dot_krope_v_b)

    S = gl.amd.cdna4.mfma(Q_lora_dot, K_lora_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_s))
    S = gl.amd.cdna4.mfma(Q_rope_dot, K_rope_T_dot, S)
    S = S * scale
    offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_s))
    valid_mask = valid_mma[None, :] & (offs_h_mma < num_heads)[:, None]
    S = gl.where(valid_mask, S, float("-inf"))

    P = gl.exp(S - lse[:, None])
    P = gl.where(valid_mask, P, 0.0)
    dP = gl.amd.cdna4.mfma(dO_dot, K_lora_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_s))
    dS = P * (dP - delta[:, None]) * scale
    dS = gl.where(valid_mask, dS, 0.0)

    dS_bf = dS.to(KV_ptr.dtype.element_ty)
    dS_dot = gl.convert_layout(dS_bf, dot_ds_a)
    dQ_lora = gl.amd.cdna4.mfma(dS_dot, K_lora_v_dot, dQ_lora)
    dQ_rope = gl.amd.cdna4.mfma(dS_dot, K_rope_v_dot, dQ_rope)

    col = t * TILE_K + offs_tile_dsp
    dsp_offs = ds_base + offs_h_dsp[:, None].to(tl.int64) * stride_ds_h + col[None, :].to(tl.int64)
    gl.amd.cdna4.buffer_store(stored_value=dS_bf, ptr=dS_ptr, offsets=dsp_offs.to(tl.int32), mask=mask_h_dsp[:, None])
    gl.amd.cdna4.buffer_store(stored_value=P.to(KV_ptr.dtype.element_ty), ptr=P_ptr, offsets=dsp_offs.to(tl.int32), mask=mask_h_dsp[:, None])

    # ===================== store dQ (lora + rope) =====================
    dq_base = token_idx.to(tl.int64) * stride_dq_t
    offs_h_o = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_o = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_o = offs_h_o < num_heads
    dq_offs_lora = dq_base + offs_h_o[:, None].to(tl.int64) * stride_dq_h + offs_v_o[None, :].to(tl.int64)
    dq_lora_blk = gl.convert_layout(dQ_lora.to(dQ_ptr.dtype.element_ty), blk_qlora)
    if not IS_FIRST_CHUNK:
        prev_lora = gl.amd.cdna4.buffer_load(ptr=dQ_ptr, offsets=dq_offs_lora.to(tl.int32), mask=mask_h_o[:, None], other=0.0)
        dq_lora_blk = (dq_lora_blk.to(gl.float32) + prev_lora.to(gl.float32)).to(dQ_ptr.dtype.element_ty)
    gl.amd.cdna4.buffer_store(stored_value=dq_lora_blk, ptr=dQ_ptr, offsets=dq_offs_lora.to(tl.int32), mask=mask_h_o[:, None])

    offs_h_or = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qrope))
    offs_r_or = gl.arange(0, D_ROPE, layout=gl.SliceLayout(0, blk_qrope))
    mask_h_or = offs_h_or < num_heads
    dq_offs_rope = dq_base + offs_h_or[:, None].to(tl.int64) * stride_dq_h + (D_V + offs_r_or[None, :]).to(tl.int64)
    dq_rope_blk = gl.convert_layout(dQ_rope.to(dQ_ptr.dtype.element_ty), blk_qrope)
    if not IS_FIRST_CHUNK:
        prev_rope = gl.amd.cdna4.buffer_load(ptr=dQ_ptr, offsets=dq_offs_rope.to(tl.int32), mask=mask_h_or[:, None], other=0.0)
        dq_rope_blk = (dq_rope_blk.to(gl.float32) + prev_rope.to(gl.float32)).to(dQ_ptr.dtype.element_ty)
    gl.amd.cdna4.buffer_store(stored_value=dq_rope_blk, ptr=dQ_ptr, offsets=dq_offs_rope.to(tl.int32), mask=mask_h_or[:, None])


# =====================================================================
# Launcher — runs the dQ pass only (chunk loop + RMW), returns dq, chunk dS/P.
# d_sink (if needed) is a torch reduction handled by the caller.
# =====================================================================
def sparse_mla_bwd_dq_gl(q, kv, do, topk_indices_padded, lse, delta,
                         R_CHUNK, topk, kv_lora_rank=512, scale=None,
                         BLOCK_H=64, TILE_K=16):
    """
    Gluon dQ pass. Mirrors the dQ portion of `sparse_mla_bwd_v4`'s chunk loop.

    Returns:
        dq:        [T, H, D_QK] bf16 (fully accumulated across chunks)
        chunk_dS:  [T, H, R_CHUNK] bf16 (LAST chunk's dS — for spot validation)
        chunk_P:   [T, H, R_CHUNK] bf16 (LAST chunk's P)
    """
    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    if scale is None:
        scale = 1.0 / (d_qk ** 0.5)
    assert R_CHUNK % TILE_K == 0, "TILE_K must divide R_CHUNK"

    dq = torch.empty_like(q)
    chunk_dS = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    chunk_P = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)

    num_hg = triton.cdiv(num_heads, BLOCK_H)
    grid = (total_tokens, num_hg)

    for r_start in range(0, topk, R_CHUNK):
        is_first = (r_start == 0)
        _sparse_mla_bwd_dq_gl_kernel[grid](
            q, kv, do, topk_indices_padded, lse, delta,
            dq, chunk_dS, chunk_P,
            q.stride(0), q.stride(1), kv.stride(0),
            do.stride(0), do.stride(1),
            dq.stride(0), dq.stride(1),
            topk_indices_padded.stride(0),
            chunk_dS.stride(0), chunk_dS.stride(1),
            scale, num_heads, r_start,
            R_CHUNK=R_CHUNK, BLOCK_H=BLOCK_H, TILE_K=TILE_K,
            D_V=kv_lora_rank, D_ROPE=rope_rank,
            IS_FIRST_CHUNK=is_first,
            num_warps=4, waves_per_eu=1,
        )
    return dq, chunk_dS, chunk_P
