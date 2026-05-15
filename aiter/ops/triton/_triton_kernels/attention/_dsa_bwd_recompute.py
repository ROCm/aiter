import torch
import triton
import triton.language as tl


# =====================================================================
# Backward method="recompute" — dQ kernel (no intermediate stores)
# =====================================================================
@triton.jit
def _bwd_dq_only(
    Q_ptr, KV_ptr, dO_ptr, TopK_ptr, LSE_ptr, Delta_ptr,
    dQ_ptr,
    stride_q_t: tl.int64, stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64, stride_do_h: tl.int64,
    stride_dq_t: tl.int64, stride_dq_h: tl.int64,
    stride_topk_t: tl.int64,
    scale: tl.float32, num_heads: tl.int32,
    TOPK: tl.constexpr, BLOCK_H: tl.constexpr, TILE_K: tl.constexpr,
    D_V: tl.constexpr, D_ROPE: tl.constexpr,
):
    """Pure dQ kernel -- computes dots 1-5, stores only dQ. No intermediates."""
    token_idx = tl.program_id(0)
    hg_idx = tl.program_id(1)
    offs_h = hg_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    q_base = token_idx * stride_q_t
    Q_lora = tl.load(Q_ptr + q_base + offs_h[:, None] * stride_q_h + offs_v[None, :],
                     mask=mask_h[:, None], other=0.0)
    Q_rope = tl.load(Q_ptr + q_base + offs_h[:, None] * stride_q_h + (D_V + offs_r[None, :]),
                     mask=mask_h[:, None], other=0.0)
    do_base = token_idx * stride_do_t
    dO_val = tl.load(dO_ptr + do_base + offs_h[:, None] * stride_do_h + offs_v[None, :],
                     mask=mask_h[:, None], other=0.0)
    lse = tl.load(LSE_ptr + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)
    delta = tl.load(Delta_ptr + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)

    dQ_lora = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)
    dQ_rope = tl.zeros([BLOCK_H, D_ROPE], dtype=tl.float32)
    NUM_TILES: tl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t
    offs_tile = tl.arange(0, TILE_K)
    topk_pos = tl.load(TopK_ptr + topk_base + offs_tile, mask=offs_tile < TOPK, other=-1)
    topk_pos_next = topk_pos

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        valid = (tile_start + offs_tile) < TOPK
        valid = valid & (topk_pos != -1)
        if t + 1 < NUM_TILES:
            next_offs = (t + 1) * TILE_K + offs_tile
            topk_pos_next = tl.load(TopK_ptr + topk_base + next_offs,
                                    mask=next_offs < TOPK, other=-1)
        safe_pos = tl.where(valid, topk_pos, 0)

        K_lora_T = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None],
                          mask=valid[None, :], other=0.0)
        K_rope_T = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
                          mask=valid[None, :], other=0.0)

        S = tl.dot(Q_lora, K_lora_T)
        S += tl.dot(Q_rope, K_rope_T)
        S *= scale
        S = tl.where(valid[None, :] & mask_h[:, None], S, float("-inf"))
        P = tl.exp(S - lse[:, None])
        P = tl.where(valid[None, :] & mask_h[:, None], P, 0.0)

        dP = tl.dot(dO_val, K_lora_T)
        dS = P * (dP - delta[:, None]) * scale
        dS = tl.where(valid[None, :] & mask_h[:, None], dS, 0.0)

        V_lora = tl.trans(K_lora_T)
        dQ_lora += tl.dot(dS.to(V_lora.dtype), V_lora).to(tl.float32)
        K_rope = tl.trans(K_rope_T)
        dQ_rope += tl.dot(dS.to(K_rope.dtype), K_rope).to(tl.float32)

        if t + 1 < NUM_TILES:
            topk_pos = topk_pos_next

    dq_base = token_idx * stride_dq_t
    tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
             dQ_lora.to(Q_lora.dtype), mask=mask_h[:, None])
    tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
             dQ_rope.to(Q_rope.dtype), mask=mask_h[:, None])


# =====================================================================
# Backward method="recompute" — dKV kernel with full recomputation
# =====================================================================
@triton.jit
def _bwd_dkv_hg_fused_recompute(
    Q_ptr,          # [T, H, D_QK] bf16
    KV_ptr,         # [T, 1, D_QK] bf16
    dO_ptr,         # [T, H, D_V] bf16
    Q_T_ptr,        # [T, D_QK, H] bf16
    dO_T_ptr,       # [T, D_V, H] bf16
    TopK_ptr,       # [T, TOPK] int32
    LSE_ptr,        # [T, H] fp32
    Delta_ptr,      # [T, H] fp32
    dKV_ptr,        # [T, D_QK] fp32
    stride_q_t: tl.int64, stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64, stride_do_h: tl.int64,
    stride_qt_t: tl.int64,
    stride_dot_t: tl.int64,
    stride_topk_t: tl.int64,
    stride_dkv_t: tl.int64,
    scale: tl.float32, num_heads: tl.int32,
    TOPK: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_HG: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """
    dKV kernel with full recomputation -- NO intermediate buffers needed.

    Grid: (total_tokens,) -- ONE program per token.
    Each program: loads Q, dO, lse, delta per HG, recomputes S, P, dS,
    then computes dKV (dots 6-8). Scatters dKV once per tile.
    """
    token_idx = tl.program_id(0)

    NUM_TILES: tl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t
    offs_tile = tl.arange(0, TILE_K)
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs = tile_start + offs_tile
        topk_pos = tl.load(TopK_ptr + topk_base + tile_offs,
                           mask=tile_offs < TOPK, other=-1)
        valid = (tile_offs < TOPK) & (topk_pos != -1)
        safe_pos = tl.where(valid, topk_pos, 0)

        K_lora_T = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None],
                          mask=valid[None, :], other=0.0)
        K_rope_T = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
                          mask=valid[None, :], other=0.0)

        dKV_lora = tl.zeros([D_V, TILE_K], dtype=tl.float32)
        dKV_rope = tl.zeros([D_ROPE, TILE_K], dtype=tl.float32)

        for hg in range(NUM_HG):
            offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
            mask_h = offs_h < num_heads

            q_base = token_idx * stride_q_t
            Q_lora = tl.load(
                Q_ptr + q_base + offs_h[:, None] * stride_q_h + offs_v[None, :],
                mask=mask_h[:, None], other=0.0,
            )
            Q_rope = tl.load(
                Q_ptr + q_base + offs_h[:, None] * stride_q_h + (D_V + offs_r[None, :]),
                mask=mask_h[:, None], other=0.0,
            )
            dO_val = tl.load(
                dO_ptr + token_idx * stride_do_t + offs_h[:, None] * stride_do_h + offs_v[None, :],
                mask=mask_h[:, None], other=0.0,
            )

            lse = tl.load(LSE_ptr + token_idx * num_heads + offs_h,
                         mask=mask_h, other=0.0)
            delta_val = tl.load(Delta_ptr + token_idx * num_heads + offs_h,
                               mask=mask_h, other=0.0)

            # Recompute S, P, dS (dots 1-3)
            S = tl.dot(Q_lora, K_lora_T)
            S += tl.dot(Q_rope, K_rope_T)
            S *= scale
            S = tl.where(valid[None, :] & mask_h[:, None], S, float("-inf"))
            P = tl.exp(S - lse[:, None])
            P = tl.where(valid[None, :] & mask_h[:, None], P, 0.0)

            dP = tl.dot(dO_val, K_lora_T)
            dS_val = P * (dP - delta_val[:, None]) * scale
            dS_val = tl.where(valid[None, :] & mask_h[:, None], dS_val, 0.0)

            # Load Q_T, dO_T for dKV (dots 6-8)
            qt_base = token_idx * stride_qt_t
            Q_lora_T = tl.load(
                Q_T_ptr + qt_base + offs_v[:, None] * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0,
            )
            Q_rope_T = tl.load(
                Q_T_ptr + qt_base + (D_V + offs_r[:, None]) * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0,
            )
            dot_base = token_idx * stride_dot_t
            dO_T = tl.load(
                dO_T_ptr + dot_base + offs_v[:, None] * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0,
            )

            dKV_lora += tl.dot(Q_lora_T, dS_val.to(Q_lora_T.dtype)).to(tl.float32)
            dKV_lora += tl.dot(dO_T, P.to(dO_T.dtype)).to(tl.float32)
            dKV_rope += tl.dot(Q_rope_T, dS_val.to(Q_rope_T.dtype)).to(tl.float32)

        dkv_ptrs_lora = dKV_ptr + safe_pos[None, :] * stride_dkv_t + offs_v[:, None]
        tl.atomic_add(dkv_ptrs_lora, dKV_lora, mask=valid[None, :], sem="relaxed")

        dkv_ptrs_rope = dKV_ptr + safe_pos[None, :] * stride_dkv_t + (D_V + offs_r[:, None])
        tl.atomic_add(dkv_ptrs_rope, dKV_rope, mask=valid[None, :], sem="relaxed")
