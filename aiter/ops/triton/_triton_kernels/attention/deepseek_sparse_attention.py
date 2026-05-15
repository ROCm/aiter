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

from ._dsa_bwd_preprocess import _sparse_mla_bwd_preprocess
from ._dsa_bwd_fused import _sparse_mla_bwd_kernel
from ._dsa_bwd_recompute import _bwd_dq_only, _bwd_dkv_hg_fused_recompute
from ._dsa_bwd_split_intermediate import _bwd_dq_store_intermediates, _bwd_dkv_hg_fused
from ._dsa_bwd_privatized import (
    _bwd_dkv_privatized,
    _bwd_dkv_xcd_local,
    _bwd_dkv_nonatomic_scatter,
    _bwd_dkv_reduce_copies,
)
from ._dsa_bwd_gather import (
    _bwd_compute_dkv_intermediate,
    _bwd_dkv_gather,
    _build_inverted_topk,
    _build_inverted_topk_slice,
    _bwd_chunk_dq_store_ds,
    _bwd_chunk_dq,
    _bwd_chunk_dkv_interm,
    _bwd_dkv_gather_acc,
)
from ._dsa_bwd_persistent import _bwd_persistent_chunk, _bwd_chunk_reduce


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
            "privatized"         -- split dQ+dKV, privatized dKV scatter (experimental)
                                    num_copies=8 reduces atomic serialization depth by 8x
            "xcd_privatized"     -- split dQ+dKV, true XCD-local scatter (MI300X)
                                    routes CTA i to copy (i%304)//38, keeping all atomic
                                    adds L2-local within each XCD (8 copies, 38 CUs/XCD)
            "gather"             -- split dQ+dKV, eliminates all atomics:
                                    stores head-reduced dKV to [T,TOPK,D] bf16 intermediate,
                                    builds CSR inverted topk, gathers with plain bf16 stores
                                    (~6.97 GiB extra: 2.14 GiB dS/P + 4.83 GiB intermediate)

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

    elif method == "privatized":
        num_copies = 8
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

        stride_copies = total_tokens * d_qk
        dkv_copies = torch.zeros(num_copies * stride_copies, dtype=torch.float32, device=q.device)

        grid_dkv = (total_tokens,)
        _bwd_dkv_privatized[grid_dkv](
            q_t, do_t, dS_buf, P_buf, topk_indices, dkv_copies,
            q_t.stride(0), do_t.stride(0),
            dS_buf.stride(0), dS_buf.stride(1),
            topk_indices.stride(0),
            stride_copies, d_qk,
            num_heads,
            TOPK=topk, TILE_K=tk_dkv, BLOCK_H=bh,
            NUM_HG=num_hg, NUM_COPIES=num_copies,
            D_V=kv_lora_rank, D_ROPE=rope_rank,
            num_warps=nw_dkv, num_stages=1,
        )

        total_elems = total_tokens * d_qk
        reduce_block = 1024
        grid_reduce = (triton.cdiv(total_elems, reduce_block),)
        _bwd_dkv_reduce_copies[grid_reduce](
            dkv_copies, dkv,
            stride_copies, total_elems,
            NUM_COPIES=num_copies, BLOCK=reduce_block,
        )

    elif method == "xcd_privatized":
        num_xcd = 8
        cus_per_xcd = 38  # MI300X: 304 CUs total, 38 per XCD
        bh, tk_dq, nw_dq, ns_dq = 64, 16, 4, 2
        tk_dkv, nw_dkv = 64, 4
        num_hg = triton.cdiv(num_heads, bh)

        dS_buf = torch.zeros(total_tokens, num_heads, topk, dtype=torch.bfloat16, device=q.device)
        P_buf  = torch.zeros(total_tokens, num_heads, topk, dtype=torch.bfloat16, device=q.device)

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

        stride_copies = total_tokens * d_qk
        dkv_copies = torch.zeros(num_xcd * stride_copies, dtype=torch.float32, device=q.device)

        grid_dkv = (total_tokens,)
        _bwd_dkv_xcd_local[grid_dkv](
            q_t, do_t, dS_buf, P_buf, topk_indices, dkv_copies,
            q_t.stride(0), do_t.stride(0),
            dS_buf.stride(0), dS_buf.stride(1),
            topk_indices.stride(0), stride_copies, d_qk,
            num_heads,
            TOPK=topk, TILE_K=tk_dkv, BLOCK_H=bh,
            NUM_HG=num_hg, NUM_XCD=num_xcd,
            D_V=kv_lora_rank, D_ROPE=rope_rank,
            num_warps=nw_dkv, num_stages=1,
        )

        total_elems = total_tokens * d_qk
        reduce_block = 1024
        grid_reduce = (triton.cdiv(total_elems, reduce_block),)
        _bwd_dkv_reduce_copies[grid_reduce](
            dkv_copies, dkv,
            stride_copies, total_elems,
            NUM_COPIES=num_xcd, BLOCK=reduce_block,
        )

    elif method == "gather":
        bh, tk_dq, nw_dq, ns_dq = 64, 16, 4, 2
        tk_dkv, nw_dkv = 64, 4
        num_hg = triton.cdiv(num_heads, bh)

        # Phase 1: dQ + store dS/P intermediates (same as split_intermediate)
        dS_buf = torch.zeros(total_tokens, num_heads, topk, dtype=torch.bfloat16, device=q.device)
        P_buf  = torch.zeros(total_tokens, num_heads, topk, dtype=torch.bfloat16, device=q.device)

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

        # Phase 2: compute head-reduced dKV intermediate [T, TOPK, D] bf16 (no atomics)
        interm = torch.empty(total_tokens, topk, d_qk, dtype=torch.bfloat16, device=q.device)
        grid_interm = (total_tokens,)
        _bwd_compute_dkv_intermediate[grid_interm](
            q_t, do_t, dS_buf, P_buf, topk_indices, interm,
            q_t.stride(0), do_t.stride(0),
            dS_buf.stride(0), dS_buf.stride(1),
            topk_indices.stride(0),
            interm.stride(0), interm.stride(1),
            num_heads,
            TOPK=topk, TILE_K=tk_dkv, BLOCK_H=bh,
            NUM_HG=num_hg, D_V=kv_lora_rank, D_ROPE=rope_rank,
            num_warps=nw_dkv, num_stages=1,
        )

        # Phase 3: build CSR inverted topk (Python, ~1ms)
        inv_ptr, inv_data = _build_inverted_topk(topk_indices)

        # Phase 4: gather dKV with plain stores (no atomics)
        dkv_gather = torch.empty(total_tokens, d_qk, dtype=torch.bfloat16, device=q.device)
        grid_gather = (total_tokens,)
        _bwd_dkv_gather[grid_gather](
            interm, inv_ptr, inv_data, dkv_gather,
            interm.stride(1), dkv_gather.stride(0),
            TOPK=topk, D_V=kv_lora_rank, D_ROPE=rope_rank,
            num_warps=nw_dkv,
        )

        dkv_out = dkv_gather.unsqueeze(1)
        return dq, dkv_out

    elif method == "chunked_gather":
        # Chunked gather: process TOPK ranks in R_CHUNK-wide passes.
        # Per pass: store chunk dS/P [T,H,R_CHUNK] → use existing dKV-interm kernel
        # (M=D_V=512 GEMMs) → gather chunk interm into fp32 dkv_acc. No atomics.
        R_CHUNK = min(256, topk)  # for TOPK=1024 → 4 passes
        bh = 64
        num_hg = triton.cdiv(num_heads, bh)
        TILE_K_DQ  = 16
        TILE_K_DKV = 64  # matches original gather dKV-interm for M=D_V=512 GEMMs

        # Chunk dS/P buffers — reused each pass (overwritten)
        chunk_dS = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
        chunk_P  = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
        # fp32 dKV accumulator — persists across all passes
        dkv_acc = torch.zeros(total_tokens, d_qk, dtype=torch.float32, device=q.device)
        # bf16 dKV intermediate — overwritten each pass
        interm = torch.empty(total_tokens, R_CHUNK, d_qk, dtype=torch.bfloat16, device=q.device)

        # Precompute all CSR arrays
        all_csr = []
        for r_start in range(0, topk, R_CHUNK):
            r_end = min(r_start + R_CHUNK, topk)
            topk_slice = topk_indices[:, r_start:r_end]
            if r_end - r_start < R_CHUNK:
                pad = torch.full(
                    (total_tokens, R_CHUNK - (r_end - r_start)),
                    -1, dtype=torch.int32, device=q.device,
                )
                topk_slice = torch.cat([topk_slice, pad], dim=1)
            all_csr.append(_build_inverted_topk_slice(topk_slice, r_start, R_CHUNK))

        for chunk_idx, r_start in enumerate(range(0, topk, R_CHUNK)):
            is_first = (r_start == 0)

            # Kernel 1: dQ accumulation + store chunk dS/P [T, H, R_CHUNK]
            _bwd_chunk_dq_store_ds[(total_tokens, num_hg)](
                q, kv, do, topk_indices, lse, delta, dq, chunk_dS, chunk_P,
                q.stride(0), q.stride(1), kv.stride(0),
                do.stride(0), do.stride(1),
                dq.stride(0), dq.stride(1),
                topk_indices.stride(0),
                chunk_dS.stride(0), chunk_dS.stride(1),
                scale, num_heads,
                R_START=r_start,
                R_CHUNK=R_CHUNK, BLOCK_H=bh, TILE_K=TILE_K_DQ,
                D_V=kv_lora_rank, D_ROPE=rope_rank,
                IS_FIRST_CHUNK=is_first,
                num_warps=4, num_stages=2,
            )

            # Kernel 2: dKV intermediate using stored chunk dS/P (reuses gather kernel)
            # TOPK=R_CHUNK: kernel iterates 0..R_CHUNK-1 ranks of chunk_dS/P and topk_indices[:,r_start:]
            _bwd_compute_dkv_intermediate[(total_tokens,)](
                q_t, do_t, chunk_dS, chunk_P,
                topk_indices[:, r_start:r_start + R_CHUNK].contiguous(),
                interm,
                q_t.stride(0), do_t.stride(0),
                chunk_dS.stride(0), chunk_dS.stride(1),
                # stride_topk_t for the sliced tensor = R_CHUNK (contiguous)
                R_CHUNK,
                interm.stride(0), interm.stride(1),
                num_heads,
                TOPK=R_CHUNK, TILE_K=TILE_K_DKV, BLOCK_H=bh,
                NUM_HG=num_hg, D_V=kv_lora_rank, D_ROPE=rope_rank,
                num_warps=4, num_stages=1,
            )

            # Kernel 3: gather chunk interm into fp32 dkv_acc (no atomics)
            inv_ptr, inv_data = all_csr[chunk_idx]
            _bwd_dkv_gather_acc[(total_tokens,)](
                interm, inv_ptr, inv_data, dkv_acc,
                interm.stride(1), dkv_acc.stride(0),
                D_V=kv_lora_rank, D_ROPE=rope_rank,
                num_warps=4,
            )

        dkv_out = dkv_acc.to(kv.dtype).unsqueeze(1)
        return dq, dkv_out

    elif method == "persistent":
        # Approach A: persistent 304-CTA kernel with L2-local atomics.
        # Each CTA owns ~13-14 query tokens and processes all TOPK ranks.
        # dKV is accumulated into dkv_chunk[NUM_XCD, K_CHUNK, D] fp32 — one copy
        # per XCD. K_CHUNK is chosen so each XCD's copy fits in its 4 MB L2.
        # 3 passes over T tokens for T=4096 with K_CHUNK=ceil(T/3)=1366.
        NUM_CUS = 304
        NUM_XCD = 8
        bh = 64
        num_hg = triton.cdiv(num_heads, bh)

        # K_CHUNK: largest value such that K_CHUNK * d_qk * 4 <= 4 MB per XCD
        # 4 MB = 4*1024*1024 bytes; fp32 = 4 bytes
        max_chunk_bytes = 4 * 1024 * 1024
        K_CHUNK = min(total_tokens, max_chunk_bytes // (d_qk * 4))
        # Round K_CHUNK to produce ~equal passes
        num_passes = triton.cdiv(total_tokens, K_CHUNK)
        K_CHUNK = triton.cdiv(total_tokens, num_passes)  # balanced chunk size

        # Allocate XCD-local chunk buffers and fp32 dKV output
        dkv_chunk = torch.zeros(NUM_XCD, K_CHUNK, d_qk, dtype=torch.float32, device=q.device)
        dkv = torch.empty(total_tokens, d_qk, dtype=torch.float32, device=q.device)

        for k_start in range(0, total_tokens, K_CHUNK):
            k_end = min(k_start + K_CHUNK, total_tokens)
            actual_chunk = k_end - k_start

            # Zero only the used portion of the chunk buffer
            dkv_chunk[:, :actual_chunk, :].zero_()

            tokens_per_cu = triton.cdiv(total_tokens, NUM_CUS)
            _bwd_persistent_chunk[(NUM_CUS,)](
                q, kv, do, topk_indices, lse, delta, dq, dkv_chunk,
                q.stride(0), q.stride(1), kv.stride(0),
                do.stride(0), do.stride(1),
                dq.stride(0), dq.stride(1),
                topk_indices.stride(0),
                dkv_chunk.stride(0), dkv_chunk.stride(1),
                scale, total_tokens, num_heads,
                k_start=k_start, k_end=k_end,
                TOPK=topk, TOKENS_PER_CU=tokens_per_cu,
                BLOCK_H=bh, NUM_HG=num_hg,
                D_V=kv_lora_rank, D_ROPE=rope_rank,
                num_warps=4, num_stages=1,
            )

            # Reduce XCD copies → dKV[k_start:k_end]
            BLOCK_D = 256
            _bwd_chunk_reduce[(actual_chunk, triton.cdiv(d_qk, BLOCK_D))](
                dkv_chunk, dkv,
                dkv_chunk.stride(0), dkv.stride(0),
                k_start=k_start,
                K_CHUNK=K_CHUNK, NUM_XCD=NUM_XCD, D=d_qk, BLOCK_D=BLOCK_D,
                num_warps=4,
            )

        dkv_out = dkv.to(kv.dtype).unsqueeze(1)
        return dq, dkv_out

    else:
        raise ValueError(f"Unknown backward method: {method!r}. "
                         f"Choose from 'fused', 'recompute', 'split_intermediate', "
                         f"'privatized', 'xcd_privatized', 'gather', 'chunked_gather', "
                         f"'persistent'.")

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
