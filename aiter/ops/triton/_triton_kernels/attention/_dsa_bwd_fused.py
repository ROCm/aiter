import torch
import triton
import triton.language as tl

from ._dsa_bwd_preprocess import _sparse_mla_bwd_preprocess  # noqa: F401 (re-exported for convenience)


def _get_lds_limit():
    """Return the per-CU LDS limit in bytes for the current GPU."""
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(prop, "gcnArchName", "")
        if "gfx950" in gcn_arch:
            return 163840
    return 65536


_LDS_LIMIT = _get_lds_limit()


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
