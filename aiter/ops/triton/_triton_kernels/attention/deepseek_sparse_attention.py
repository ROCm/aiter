"""
DeepSeek Sparse Attention (DSA) — forward and backward kernels for training.

Sparse MLA attention using TopK token selection with MQA (multi-query attention):
  - Q: [total_tokens, num_heads, d_qk]  (d_qk = kv_lora_rank + rope_rank)
  - KV: [total_tokens, 1, d_qk]         (single KV head, shared across all Q heads)
  - TopK: [total_tokens, topk]           (absolute token indices into KV)

Forward:
  O[t,h] = softmax(Q[t,h] @ KV[topk[t]]^T) @ V[topk[t]]
  Single autotuned kernel with online softmax.

Backward — three strategies:
  1. "fused"              — single fused kernel, 58ms, no extra memory (baseline)
  2. "recompute"          — split dQ + dKV, recomputes S/P/dS, 49ms, 0 extra memory
  3. "split_intermediate" — split dQ + dKV, stores dS/P intermediates, 35ms, 2 GiB extra

Performance measured on MI300X with T=4096 H=128 D=576 TOPK=1024.
"""

import torch
import triton
import triton.language as tl


# =====================================================================
# Utility
# =====================================================================
def _get_lds_limit():
    """Return the per-CU LDS limit in bytes for the current GPU.

    gfx942 (MI300X): 64 KB = 65536 bytes
    gfx950 (MI355X): 160 KB = 163840 bytes
    """
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(prop, "gcnArchName", "")
        if "gfx950" in gcn_arch:
            return 163840
    return 65536


_LDS_LIMIT = _get_lds_limit()


# =====================================================================
# Forward — autotune configs and pruning
# =====================================================================
def _fwd_prune_configs(configs, named_args, **kwargs):
    """Prune autotune configs that would exceed per-CU LDS."""
    D_V = kwargs.get("D_V", named_args.get("D_V"))
    D_ROPE = kwargs.get("D_ROPE", named_args.get("D_ROPE"))
    pruned = []
    for config in configs:
        bh = config.kwargs["BLOCK_H"]
        tk = config.kwargs["TILE_K"]
        ns = config.num_stages
        kv_lds = (D_V + D_ROPE) * tk * 2 * ns
        if kv_lds <= _LDS_LIMIT:
            pruned.append(config)
    if not pruned:
        pruned.append(configs[0])
    return pruned


def _get_fwd_autotune_configs():
    configs = []
    for BLOCK_H in [16, 32, 64]:
        for TILE_K in [16, 32, 64, 128]:
            for num_warps in [4, 8]:
                for num_stages in [1, 2]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_H": BLOCK_H, "TILE_K": TILE_K},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


# =====================================================================
# Forward kernel
# =====================================================================
@triton.autotune(
    configs=_get_fwd_autotune_configs(),
    key=["num_heads", "TOPK", "D_V", "D_ROPE"],
    prune_configs_by={"early_config_prune": _fwd_prune_configs},
)
@triton.jit
def _sparse_mla_fwd_train_kernel(
    Q_ptr,          # [total_tokens, num_heads, D_QK]
    KV_ptr,         # [total_tokens, 1, D_QK]
    TopK_ptr,       # [total_tokens, topk]
    O_ptr,          # [total_tokens, num_heads, D_V]
    LSE_ptr,        # [total_tokens, num_heads]
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    stride_topk_t: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TILE_K: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """
    Sparse MLA forward for training.

    Grid: (total_tokens, cdiv(num_heads, BLOCK_H))
    Each program: 1 query token x BLOCK_H heads.
    """
    token_idx = tl.program_id(0)
    hg_idx = tl.program_id(1)

    offs_h = hg_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    q_base = token_idx * stride_q_t
    Q_lora = tl.load(
        Q_ptr + q_base + offs_h[:, None] * stride_q_h + offs_v[None, :],
        mask=mask_h[:, None], other=0.0,
    )
    Q_rope = tl.load(
        Q_ptr + q_base + offs_h[:, None] * stride_q_h + (D_V + offs_r[None, :]),
        mask=mask_h[:, None], other=0.0,
    )

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)

    NUM_TILES: tl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t
    offs_tile = tl.arange(0, TILE_K)

    topk_pos = tl.load(
        TopK_ptr + topk_base + offs_tile,
        mask=offs_tile < TOPK, other=-1,
    )
    topk_pos_next = topk_pos

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        valid = (tile_start + offs_tile) < TOPK
        valid = valid & (topk_pos != -1)

        if t + 1 < NUM_TILES:
            next_offs = (t + 1) * TILE_K + offs_tile
            topk_pos_next = tl.load(
                TopK_ptr + topk_base + next_offs,
                mask=next_offs < TOPK, other=-1,
            )

        safe_pos = tl.where(valid, topk_pos, 0)

        K_lora = tl.load(
            KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None],
            mask=valid[None, :], other=0.0,
        )
        K_rope = tl.load(
            KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
            mask=valid[None, :], other=0.0,
        )

        S = tl.dot(Q_lora, K_lora)
        S += tl.dot(Q_rope, K_rope)
        S *= scale
        S = tl.where(valid[None, :] & mask_h[:, None], S, float("-inf"))

        m_j = tl.max(S, axis=1)
        m_new = tl.maximum(m_i, m_j)
        m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(S - m_new[:, None])
        l_new = alpha * l_i + tl.sum(P, axis=1)

        acc = acc * alpha[:, None]
        V_lora = tl.trans(K_lora)
        acc += tl.dot(P.to(V_lora.dtype), V_lora)

        m_i = m_new
        l_i = l_new

        if t + 1 < NUM_TILES:
            topk_pos = topk_pos_next

    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    o_base = token_idx * stride_o_t
    tl.store(
        O_ptr + o_base + offs_h[:, None] * stride_o_h + offs_v[None, :],
        acc.to(Q_lora.dtype), mask=mask_h[:, None],
    )
    tl.store(
        LSE_ptr + token_idx * num_heads + offs_h,
        lse, mask=mask_h,
    )


# =====================================================================
# Backward — preprocess kernel (Delta computation)
# =====================================================================
@triton.jit
def _sparse_mla_bwd_preprocess(
    O_ptr,          # [total_tokens, num_heads, D_V]
    dO_ptr,         # [total_tokens, num_heads, D_V]
    Delta_ptr,      # [total_tokens, num_heads]
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    num_heads: tl.int32,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Delta[t, h] = sum_d(O[t, h, d] * dO[t, h, d])

    Grid: (total_tokens, cdiv(num_heads, BLOCK_H))
    """
    token_idx = tl.program_id(0)
    hg_idx = tl.program_id(1)

    offs_h = hg_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_d = tl.arange(0, D_V)

    base = token_idx * stride_o_t

    O = tl.load(
        O_ptr + base + offs_h[:, None] * stride_o_h + offs_d[None, :],
        mask=mask_h[:, None], other=0.0,
    )
    dO = tl.load(
        dO_ptr + base + offs_h[:, None] * stride_o_h + offs_d[None, :],
        mask=mask_h[:, None], other=0.0,
    )

    delta = tl.sum(O.to(tl.float32) * dO.to(tl.float32), axis=1)

    tl.store(
        Delta_ptr + token_idx * num_heads + offs_h,
        delta, mask=mask_h,
    )


# =====================================================================
# Backward — autotune configs (for fused baseline)
# =====================================================================
def _bwd_prune_configs(configs, named_args, **kwargs):
    """Prune autotune configs that would exceed per-CU LDS or hit known bugs."""
    D_V = kwargs.get("D_V", named_args.get("D_V"))
    D_ROPE = kwargs.get("D_ROPE", named_args.get("D_ROPE"))
    pruned = []
    for config in configs:
        bh = config.kwargs["BLOCK_H"]
        tk = config.kwargs["TILE_K"]
        ns = config.num_stages
        kv_lds = (D_V + D_ROPE) * tk * 2 * ns
        if kv_lds > _LDS_LIMIT:
            continue
        # Skip BLOCK_H=64 / TILE_K=16 / num_warps=4 / num_stages=1:
        # produces NaN on AMD CDNA due to compiler bug.
        if bh == 64 and tk == 16 and config.num_warps == 4 and ns == 1:
            continue
        pruned.append(config)
    if not pruned:
        pruned.append(configs[0])
    return pruned


def _get_bwd_autotune_configs():
    configs = []
    for BLOCK_H in [16, 32, 64]:
        for TILE_K in [16, 32, 64, 128]:
            for num_warps in [2, 4, 8, 16]:
                for num_stages in [1, 2, 3, 4]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_H": BLOCK_H, "TILE_K": TILE_K},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


# =====================================================================
# Backward method="fused" — single fused kernel (baseline, 58ms)
# =====================================================================
@triton.autotune(
    configs=_get_bwd_autotune_configs(),
    key=["num_heads", "TOPK", "D_V", "D_ROPE"],
    prune_configs_by={"early_config_prune": _bwd_prune_configs},
    reset_to_zero=["dKV_ptr"],
)
@triton.jit
def _sparse_mla_bwd_kernel(
    Q_ptr,          # [total_tokens, num_heads, D_QK]
    KV_ptr,         # [total_tokens, 1, D_QK]
    dO_ptr,         # [total_tokens, num_heads, D_V]
    TopK_ptr,       # [total_tokens, topk]
    LSE_ptr,        # [total_tokens, num_heads]  float32
    Delta_ptr,      # [total_tokens, num_heads]  float32
    dQ_ptr,         # [total_tokens, num_heads, D_QK]
    dKV_ptr,        # [total_tokens, D_QK]  float32 (atomic target, squeezed)
    Q_T_ptr,        # [total_tokens, D_QK, num_heads]
    dO_T_ptr,       # [total_tokens, D_V, num_heads]
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64,
    stride_do_h: tl.int64,
    stride_dq_t: tl.int64,
    stride_dq_h: tl.int64,
    stride_dkv_t: tl.int64,
    stride_topk_t: tl.int64,
    stride_qt_t: tl.int64,
    stride_dot_t: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TILE_K: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """
    Fused sparse MLA backward: dQ + dKV.

    Grid: (total_tokens, cdiv(num_heads, BLOCK_H))
    Each program: 1 query token x BLOCK_H heads.
    """
    token_idx = tl.program_id(0)
    hg_idx = tl.program_id(1)

    offs_h = hg_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    q_base = token_idx * stride_q_t
    Q_lora = tl.load(
        Q_ptr + q_base + offs_h[:, None] * stride_q_h + offs_v[None, :],
        mask=mask_h[:, None], other=0.0,
    )
    Q_rope = tl.load(
        Q_ptr + q_base + offs_h[:, None] * stride_q_h + (D_V + offs_r[None, :]),
        mask=mask_h[:, None], other=0.0,
    )

    do_base = token_idx * stride_do_t
    dO_val = tl.load(
        dO_ptr + do_base + offs_h[:, None] * stride_do_h + offs_v[None, :],
        mask=mask_h[:, None], other=0.0,
    )

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

    lse = tl.load(
        LSE_ptr + token_idx * num_heads + offs_h,
        mask=mask_h, other=0.0,
    )
    delta = tl.load(
        Delta_ptr + token_idx * num_heads + offs_h,
        mask=mask_h, other=0.0,
    )

    dQ_lora = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)
    dQ_rope = tl.zeros([BLOCK_H, D_ROPE], dtype=tl.float32)

    NUM_TILES: tl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t
    offs_tile = tl.arange(0, TILE_K)

    topk_pos = tl.load(
        TopK_ptr + topk_base + offs_tile,
        mask=offs_tile < TOPK, other=-1,
    )
    topk_pos_next = topk_pos

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        valid = (tile_start + offs_tile) < TOPK
        valid = valid & (topk_pos != -1)

        if t + 1 < NUM_TILES:
            next_offs = (t + 1) * TILE_K + offs_tile
            topk_pos_next = tl.load(
                TopK_ptr + topk_base + next_offs,
                mask=next_offs < TOPK, other=-1,
            )

        safe_pos = tl.where(valid, topk_pos, 0)

        K_lora_T = tl.load(
            KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None],
            mask=valid[None, :], other=0.0,
        )
        K_rope_T = tl.load(
            KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
            mask=valid[None, :], other=0.0,
        )

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

        dKV_lora_T = tl.dot(Q_lora_T, dS.to(Q_lora_T.dtype))
        dKV_lora_T += tl.dot(dO_T, P.to(dO_T.dtype))
        dKV_lora_T = dKV_lora_T.to(tl.float32)

        dKV_rope_T = tl.dot(Q_rope_T, dS.to(Q_rope_T.dtype))
        dKV_rope_T = dKV_rope_T.to(tl.float32)

        dkv_ptrs_lora = (
            dKV_ptr + safe_pos[None, :] * stride_dkv_t + offs_v[:, None]
        )
        tl.atomic_add(dkv_ptrs_lora, dKV_lora_T, mask=valid[None, :], sem="relaxed")

        dkv_ptrs_rope = (
            dKV_ptr + safe_pos[None, :] * stride_dkv_t + (D_V + offs_r[:, None])
        )
        tl.atomic_add(dkv_ptrs_rope, dKV_rope_T, mask=valid[None, :], sem="relaxed")

        if t + 1 < NUM_TILES:
            topk_pos = topk_pos_next

    dq_base = token_idx * stride_dq_t
    tl.store(
        dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
        dQ_lora.to(Q_lora.dtype), mask=mask_h[:, None],
    )
    tl.store(
        dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
        dQ_rope.to(Q_rope.dtype), mask=mask_h[:, None],
    )


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


# =====================================================================
# Backward method="split_intermediate" — dQ kernel (stores dS/P)
# =====================================================================
@triton.jit
def _bwd_dq_store_intermediates(
    Q_ptr, KV_ptr, dO_ptr, TopK_ptr, LSE_ptr, Delta_ptr,
    dQ_ptr,
    dS_ptr,         # [T, H, TOPK] bf16 -- output
    P_ptr,          # [T, H, TOPK] bf16 -- output
    stride_q_t: tl.int64, stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64, stride_do_h: tl.int64,
    stride_dq_t: tl.int64, stride_dq_h: tl.int64,
    stride_topk_t: tl.int64,
    stride_ds_t: tl.int64, stride_ds_h: tl.int64,
    scale: tl.float32, num_heads: tl.int32,
    TOPK: tl.constexpr, BLOCK_H: tl.constexpr, TILE_K: tl.constexpr,
    D_V: tl.constexpr, D_ROPE: tl.constexpr,
):
    """dQ kernel that stores dS and P intermediates for the dKV kernel."""
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

        # Store dS and P intermediates
        ds_base = token_idx * stride_ds_t
        tile_offs = tile_start + offs_tile
        tl.store(dS_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                 dS.to(tl.bfloat16),
                 mask=mask_h[:, None] & (tile_offs[None, :] < TOPK))
        tl.store(P_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                 P.to(tl.bfloat16),
                 mask=mask_h[:, None] & (tile_offs[None, :] < TOPK))

        if t + 1 < NUM_TILES:
            topk_pos = topk_pos_next

    dq_base = token_idx * stride_dq_t
    tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
             dQ_lora.to(Q_lora.dtype), mask=mask_h[:, None])
    tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
             dQ_rope.to(Q_rope.dtype), mask=mask_h[:, None])


# =====================================================================
# Backward method="split_intermediate" — dKV kernel (reads dS/P)
# =====================================================================
@triton.jit
def _bwd_dkv_hg_fused(
    Q_T_ptr,        # [T, D_QK, H] bf16
    dO_T_ptr,       # [T, D_V, H] bf16
    dS_ptr,         # [T, H, TOPK] bf16
    P_ptr,          # [T, H, TOPK] bf16
    TopK_ptr,       # [T, TOPK] int32
    dKV_ptr,        # [T, D_QK] fp32
    stride_qt_t: tl.int64,
    stride_dot_t: tl.int64,
    stride_ds_t: tl.int64,
    stride_ds_h: tl.int64,
    stride_topk_t: tl.int64,
    stride_dkv_t: tl.int64,
    num_heads: tl.int32,
    TOPK: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_HG: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """
    dKV scatter kernel with head-group fusion.

    Grid: (total_tokens,) -- ONE program per token.
    Accumulates dKV across all head groups before scattering -- 2x fewer atomics.
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

        dKV_lora = tl.zeros([D_V, TILE_K], dtype=tl.float32)
        dKV_rope = tl.zeros([D_ROPE, TILE_K], dtype=tl.float32)

        for hg in range(NUM_HG):
            offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
            mask_h = offs_h < num_heads

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

            ds_base = token_idx * stride_ds_t
            dS_val = tl.load(
                dS_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & (tile_offs[None, :] < TOPK),
                other=0.0,
            )
            P_val = tl.load(
                P_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & (tile_offs[None, :] < TOPK),
                other=0.0,
            )

            dKV_lora += tl.dot(Q_lora_T, dS_val.to(Q_lora_T.dtype)).to(tl.float32)
            dKV_lora += tl.dot(dO_T, P_val.to(dO_T.dtype)).to(tl.float32)
            dKV_rope += tl.dot(Q_rope_T, dS_val.to(Q_rope_T.dtype)).to(tl.float32)

        dkv_ptrs_lora = dKV_ptr + safe_pos[None, :] * stride_dkv_t + offs_v[:, None]
        tl.atomic_add(dkv_ptrs_lora, dKV_lora, mask=valid[None, :], sem="relaxed")

        dkv_ptrs_rope = dKV_ptr + safe_pos[None, :] * stride_dkv_t + (D_V + offs_r[:, None])
        tl.atomic_add(dkv_ptrs_rope, dKV_rope, mask=valid[None, :], sem="relaxed")


# =====================================================================
# Python wrappers
# =====================================================================
def sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank=512, scale=None):
    """
    Sparse MLA forward pass for training.

    Args:
        q:             [total_tokens, num_heads, d_qk] bfloat16
        kv:            [total_tokens, 1, d_qk] bfloat16 (or [total_tokens, d_qk])
        topk_indices:  [total_tokens, topk] int32
        kv_lora_rank:  int, default 512
        scale:         float, default 1/sqrt(d_qk)

    Returns:
        o:   [total_tokens, num_heads, kv_lora_rank] same dtype as q
        lse: [total_tokens, num_heads] float32
    """
    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert topk_indices.is_contiguous()

    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]

    if scale is None:
        scale = 1.0 / (d_qk ** 0.5)

    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.shape[0] == total_tokens and kv.shape[-1] == d_qk

    o = torch.empty(total_tokens, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device)
    lse = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    grid = lambda META: (total_tokens, triton.cdiv(num_heads, META["BLOCK_H"]))

    _sparse_mla_fwd_train_kernel[grid](
        Q_ptr=q, KV_ptr=kv, TopK_ptr=topk_indices,
        O_ptr=o, LSE_ptr=lse,
        stride_q_t=q.stride(0), stride_q_h=q.stride(1),
        stride_kv_t=kv.stride(0),
        stride_o_t=o.stride(0), stride_o_h=o.stride(1),
        stride_topk_t=topk_indices.stride(0),
        scale=scale, num_heads=num_heads,
        TOPK=topk, D_V=kv_lora_rank, D_ROPE=rope_rank,
    )

    return o, lse


def sparse_mla_bwd(q, kv, o, do, topk_indices, lse, kv_lora_rank=512, scale=None,
                   method="fused"):
    """
    Sparse MLA backward pass for training.

    Args:
        q:             [total_tokens, num_heads, d_qk] bfloat16
        kv:            [total_tokens, 1, d_qk] bfloat16
        o:             [total_tokens, num_heads, kv_lora_rank] bfloat16
        do:            [total_tokens, num_heads, kv_lora_rank] bfloat16
        topk_indices:  [total_tokens, topk] int32
        lse:           [total_tokens, num_heads] float32
        kv_lora_rank:  int, default 512
        scale:         float, default 1/sqrt(d_qk)
        method:        str, backward strategy:
            "fused"              -- single fused kernel (58ms, no extra memory)
            "recompute"          -- split dQ+dKV, full recomputation (49ms, 0 extra)
            "split_intermediate" -- split dQ+dKV, stores dS/P (35ms, 2 GiB extra)

    Returns:
        dq:  [total_tokens, num_heads, d_qk] same dtype as q
        dkv: [total_tokens, 1, d_qk] same dtype as kv
    """
    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert o.is_contiguous()
    assert do.is_contiguous()
    assert topk_indices.is_contiguous()
    assert lse.is_contiguous()

    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]

    if scale is None:
        scale = 1.0 / (d_qk ** 0.5)

    if kv.dim() == 2:
        kv = kv.unsqueeze(1)

    dq = torch.empty_like(q)
    dkv = torch.zeros(total_tokens, d_qk, dtype=torch.float32, device=q.device)

    delta = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    q_t = q.transpose(1, 2).contiguous()
    do_t = do.transpose(1, 2).contiguous()

    BLOCK_H_PRE = min(64, num_heads)
    BLOCK_H_PRE = triton.next_power_of_2(BLOCK_H_PRE)

    grid_pre = (total_tokens, triton.cdiv(num_heads, BLOCK_H_PRE))
    _sparse_mla_bwd_preprocess[grid_pre](
        O_ptr=o, dO_ptr=do, Delta_ptr=delta,
        stride_o_t=o.stride(0), stride_o_h=o.stride(1),
        num_heads=num_heads, D_V=kv_lora_rank, BLOCK_H=BLOCK_H_PRE,
    )

    if method == "fused":
        grid_bwd = lambda META: (total_tokens, triton.cdiv(num_heads, META["BLOCK_H"]))
        _sparse_mla_bwd_kernel[grid_bwd](
            Q_ptr=q, KV_ptr=kv, dO_ptr=do,
            TopK_ptr=topk_indices, LSE_ptr=lse, Delta_ptr=delta,
            dQ_ptr=dq, dKV_ptr=dkv,
            Q_T_ptr=q_t, dO_T_ptr=do_t,
            stride_q_t=q.stride(0), stride_q_h=q.stride(1),
            stride_kv_t=kv.stride(0),
            stride_do_t=do.stride(0), stride_do_h=do.stride(1),
            stride_dq_t=dq.stride(0), stride_dq_h=dq.stride(1),
            stride_dkv_t=dkv.stride(0),
            stride_topk_t=topk_indices.stride(0),
            stride_qt_t=q_t.stride(0), stride_dot_t=do_t.stride(0),
            scale=scale, num_heads=num_heads,
            TOPK=topk, D_V=kv_lora_rank, D_ROPE=rope_rank,
        )

    elif method == "recompute":
        bh, tk_dq, nw_dq, ns_dq = 64, 16, 4, 2
        tk_dkv, nw_dkv = 32, 2
        num_hg = triton.cdiv(num_heads, bh)

        grid_dq = (total_tokens, num_hg)
        _bwd_dq_only[grid_dq](
            q, kv, do, topk_indices, lse, delta, dq,
            q.stride(0), q.stride(1), kv.stride(0),
            do.stride(0), do.stride(1),
            dq.stride(0), dq.stride(1),
            topk_indices.stride(0),
            scale, num_heads,
            TOPK=topk, BLOCK_H=bh, TILE_K=tk_dq,
            D_V=kv_lora_rank, D_ROPE=rope_rank,
            num_warps=nw_dq, num_stages=ns_dq,
        )

        grid_dkv = (total_tokens,)
        _bwd_dkv_hg_fused_recompute[grid_dkv](
            q, kv, do, q_t, do_t, topk_indices, lse, delta, dkv,
            q.stride(0), q.stride(1), kv.stride(0),
            do.stride(0), do.stride(1),
            q_t.stride(0), do_t.stride(0),
            topk_indices.stride(0), dkv.stride(0),
            scale, num_heads,
            TOPK=topk, TILE_K=tk_dkv, BLOCK_H=bh,
            NUM_HG=num_hg, D_V=kv_lora_rank, D_ROPE=rope_rank,
            num_warps=nw_dkv, num_stages=1,
        )

    elif method == "split_intermediate":
        bh, tk_dq, nw_dq, ns_dq = 64, 16, 4, 2
        tk_dkv, nw_dkv = 64, 4
        num_hg = triton.cdiv(num_heads, bh)

        dS_buf = torch.zeros(total_tokens, num_heads, topk, dtype=torch.bfloat16, device=q.device)
        P_buf = torch.zeros(total_tokens, num_heads, topk, dtype=torch.bfloat16, device=q.device)

        grid_dq = (total_tokens, num_hg)
        _bwd_dq_store_intermediates[grid_dq](
            q, kv, do, topk_indices, lse, delta,
            dq, dS_buf, P_buf,
            q.stride(0), q.stride(1), kv.stride(0),
            do.stride(0), do.stride(1),
            dq.stride(0), dq.stride(1),
            topk_indices.stride(0),
            dS_buf.stride(0), dS_buf.stride(1),
            scale, num_heads,
            TOPK=topk, BLOCK_H=bh, TILE_K=tk_dq,
            D_V=kv_lora_rank, D_ROPE=rope_rank,
            num_warps=nw_dq, num_stages=ns_dq,
        )

        grid_dkv = (total_tokens,)
        _bwd_dkv_hg_fused[grid_dkv](
            q_t, do_t, dS_buf, P_buf, topk_indices, dkv,
            q_t.stride(0), do_t.stride(0),
            dS_buf.stride(0), dS_buf.stride(1),
            topk_indices.stride(0), dkv.stride(0),
            num_heads,
            TOPK=topk, TILE_K=tk_dkv, BLOCK_H=bh,
            NUM_HG=num_hg, D_V=kv_lora_rank, D_ROPE=rope_rank,
            num_warps=nw_dkv, num_stages=1,
        )

    else:
        raise ValueError(f"Unknown backward method: {method!r}. "
                         f"Choose from 'fused', 'recompute', 'split_intermediate'.")

    dkv_out = dkv.unsqueeze(1).to(kv.dtype)
    return dq, dkv_out


class SparseMlaFunc(torch.autograd.Function):
    """Autograd wrapper connecting forward and backward passes."""

    @staticmethod
    def forward(ctx, q, kv, topk_indices, kv_lora_rank, scale, bwd_method):
        o, lse = sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)
        ctx.save_for_backward(q, kv, topk_indices, o, lse)
        ctx.kv_lora_rank = kv_lora_rank
        ctx.scale = scale
        ctx.bwd_method = bwd_method
        return o, lse

    @staticmethod
    def backward(ctx, do, _dlse):
        q, kv, topk_indices, o, lse = ctx.saved_tensors
        dq, dkv = sparse_mla_bwd(
            q, kv, o, do.contiguous(), topk_indices, lse,
            kv_lora_rank=ctx.kv_lora_rank, scale=ctx.scale,
            method=ctx.bwd_method,
        )
        return dq, dkv, None, None, None, None


def sparse_mla_train(q, kv, topk_indices, kv_lora_rank=512, scale=None, bwd_method="fused"):
    """
    Differentiable sparse MLA attention for training.

    Args:
        q:             [total_tokens, num_heads, d_qk] bfloat16
        kv:            [total_tokens, 1, d_qk] bfloat16
        topk_indices:  [total_tokens, topk] int32
        kv_lora_rank:  int, default 512
        scale:         float, default 1/sqrt(d_qk)
        bwd_method:    str, backward strategy (see sparse_mla_bwd)

    Returns:
        o:   [total_tokens, num_heads, kv_lora_rank] same dtype as q
        lse: [total_tokens, num_heads] float32
    """
    return SparseMlaFunc.apply(q, kv, topk_indices, kv_lora_rank, scale, bwd_method)
