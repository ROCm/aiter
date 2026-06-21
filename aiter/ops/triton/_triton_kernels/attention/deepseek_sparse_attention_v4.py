"""
DeepSeek V4 Sparse Attention — forward kernel for training (CSA / HCA / SWA-only).

Delta vs V3.2 (see deepseek_sparse_attention.py):
  - Optional per-head learnable `attn_sink` [H] fp32 added to the softmax denominator
    as a virtual "sink token" with no K/V contribution. This implements the formula
    from the DeepSeek V4 paper and FlashMLA's `flash_mla_sparse_fwd(attn_sink=...)`.

  - `topk_indices` carries SWA + sparse indices already concatenated by the caller
    (per ATOM #664: `topk_idxs = cat([window_topk_idxs(128), compress_topk_idxs], dim=-1)`).
    No SWA-specific code path is needed inside the kernel — TopK length just grew
    from 512/1024 (V3.2) to 640/1152 (V4) and the same loop handles both.

No structural changes:
  - Same grid `(total_tokens, cdiv(num_heads, BLOCK_H))`
  - Same K-as-V trick (`V_lora = tl.trans(K_lora)`, single global load)
  - Same online softmax with K-loop over TILE_K-wide tiles
  - Same autotune knobs (BLOCK_H, TILE_K, num_warps, num_stages)

Tensor shapes (caller-side):
  Q     [T, H, D_QK=576]      bf16    H=64 (V4-Flash) or H=128 (V4-Pro)
  KV    [T, 1, D_QK=576]      bf16    MQA-style single KV head
  TopK  [T, TOPK]             int32   absolute KV-token indices; -1 invalid
  Sink  [H]                   fp32    per-head learnable sink logit (optional)
  ──
  O     [T, H, D_V=512]       bf16
  LSE   [T, H]                fp32    log-sum-exp INCLUDING the sink term
                                       (so the bwd kernel can recompute
                                        P_j = exp(S_j - LSE) directly,
                                        which then sums to <1 and equals the
                                        contribution share of each gathered token)

Math (per query token t, head h):
  S_j = scale * Q[t,h] . KV[topk[t,j]]             for j in valid topk positions
  m   = max(S_j, sink[h])                           # running max includes sink
  L   = sum_j exp(S_j - m) + exp(sink[h] - m)      # denominator includes sink
  LSE = m + log(L)                                  # returned, includes sink
  O   = sum_j (exp(S_j - LSE) * V[topk[t,j]])      # sink contributes nothing to O
                                                     # (no V_sink term)

Backward (next PR) only needs to add `d_sink[h] = -sum_t exp(sink[h] - LSE[t,h]) * D[t,h]`
where D = rowsum(O * dO) is already computed in the existing preprocess kernel.
The dQ/dKV math is unchanged because P_j = exp(S_j - LSE) using the returned LSE
already accounts for the sink — same formulas as V3.2 backward, just with the
new LSE.
"""

import torch
import triton
import triton.language as tl


# =====================================================================
# Reuse V3.2 LDS limit + autotune machinery
# =====================================================================
from .deepseek_sparse_attention import (
    _get_lds_limit,
    _fwd_prune_configs,
    _get_fwd_autotune_configs,
    _LDS_LIMIT,
)


# =====================================================================
# Forward kernel (V4)
# =====================================================================
@triton.autotune(
    configs=_get_fwd_autotune_configs(),
    key=["num_heads", "TOPK", "D_V", "D_ROPE"],
    prune_configs_by={"early_config_prune": _fwd_prune_configs},
)
@triton.jit
def _sparse_mla_fwd_v4_kernel(
    Q_ptr,          # [total_tokens, num_heads, D_QK]                bf16
    KV_ptr,         # [total_tokens, 1, D_QK]                        bf16
    TopK_ptr,       # [total_tokens, TOPK]                           int32
    Sink_ptr,       # [num_heads]                                    fp32; ignored if HAS_SINK == False
    O_ptr,          # [total_tokens, num_heads, D_V]                 bf16
    LSE_ptr,        # [total_tokens, num_heads]                      fp32 (sink-inclusive)
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
    HAS_SINK: tl.constexpr,
):
    """
    Per-program: 1 query token × BLOCK_H heads × all TOPK tokens.

    Grid: (total_tokens, cdiv(num_heads, BLOCK_H))
    """
    token_idx = tl.program_id(0)
    hg_idx = tl.program_id(1)

    offs_h = hg_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    # ---------- Load Q (lora + rope) once into registers ----------
    q_base = token_idx * stride_q_t
    Q_lora = tl.load(
        Q_ptr + q_base + offs_h[:, None] * stride_q_h + offs_v[None, :],
        mask=mask_h[:, None], other=0.0,
    )
    Q_rope = tl.load(
        Q_ptr + q_base + offs_h[:, None] * stride_q_h + (D_V + offs_r[None, :]),
        mask=mask_h[:, None], other=0.0,
    )

    # ---------- Accumulators ----------
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)

    # ---------- Inner loop over TOPK in TILE_K-wide tiles ----------
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

        # K_lora doubles as V_lora via register-only transpose (MQA trick).
        K_lora = tl.load(
            KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None],
            mask=valid[None, :], other=0.0,
        )
        K_rope = tl.load(
            KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
            mask=valid[None, :], other=0.0,
        )

        # S = Q_lora · K_lora + Q_rope · K_rope
        S = tl.dot(Q_lora, K_lora)
        S += tl.dot(Q_rope, K_rope)
        S *= scale
        S = tl.where(valid[None, :] & mask_h[:, None], S, float("-inf"))

        # Online softmax update
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

    # ---------- Epilogue: fold sink into denominator (V4 delta) ----------
    if HAS_SINK:
        sink = tl.load(Sink_ptr + offs_h, mask=mask_h, other=float("-inf"))
        # m_final = max(m_i, sink); rescale acc and l_i, then add the sink term to l.
        m_final = tl.maximum(m_i, sink)
        alpha_fix = tl.exp(m_i - m_final)
        l_total = l_i * alpha_fix + tl.exp(sink - m_final)
        acc = acc * alpha_fix[:, None]
        lse = m_final + tl.log(l_total)
        acc = acc / l_total[:, None]
    else:
        lse = m_i + tl.log(l_i)
        acc = acc / l_i[:, None]

    # ---------- Store O and LSE ----------
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
# Python wrapper
# =====================================================================
def sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink=None, kv_lora_rank=512, scale=None):
    """
    DeepSeek V4 sparse MLA forward pass (CSA, HCA, and SWA-only layers).

    Args:
        q:             [total_tokens, num_heads, d_qk] bfloat16
                       d_qk = kv_lora_rank + rope_rank (typically 512 + 64 = 576)
        kv:            [total_tokens, 1, d_qk] bfloat16  (or [total_tokens, d_qk])
        topk_indices:  [total_tokens, topk] int32
                       Already includes SWA window indices concatenated with sparse
                       top-k indices, per ATOM #664. `-1` marks invalid slots.
                       Typical sizes: 640 (V4-Flash CSA) or 1152 (V4-Pro CSA).
        attn_sink:     [num_heads] fp32, optional. Per-head learnable sink logit.
                       When None, behaves identically to V3.2 forward.
        kv_lora_rank:  int, default 512
        scale:         float, default 1/sqrt(d_qk)

    Returns:
        o:   [total_tokens, num_heads, kv_lora_rank] same dtype as q
        lse: [total_tokens, num_heads] float32, INCLUDES the sink term in its
             denominator (so backward can recompute P_j = exp(S_j - lse) directly).
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

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.is_contiguous()
        assert attn_sink.dtype == torch.float32
        assert attn_sink.shape == (num_heads,)
        sink_ptr = attn_sink
    else:
        # Placeholder pointer; kernel guards with HAS_SINK constexpr so it's never read
        sink_ptr = torch.empty(1, dtype=torch.float32, device=q.device)

    o = torch.empty(total_tokens, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device)
    lse = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    grid = lambda META: (total_tokens, triton.cdiv(num_heads, META["BLOCK_H"]))

    _sparse_mla_fwd_v4_kernel[grid](
        Q_ptr=q, KV_ptr=kv, TopK_ptr=topk_indices, Sink_ptr=sink_ptr,
        O_ptr=o, LSE_ptr=lse,
        stride_q_t=q.stride(0), stride_q_h=q.stride(1),
        stride_kv_t=kv.stride(0),
        stride_o_t=o.stride(0), stride_o_h=o.stride(1),
        stride_topk_t=topk_indices.stride(0),
        scale=scale, num_heads=num_heads,
        TOPK=topk, D_V=kv_lora_rank, D_ROPE=rope_rank,
        HAS_SINK=has_sink,
    )

    return o, lse


# =====================================================================
# Backward — chunked_gather (only method ported to V4 for now)
# =====================================================================
def sparse_mla_bwd_v4(q, kv, o, do, topk_indices, lse, attn_sink=None,
                      kv_lora_rank=512, scale=None):
    """
    DeepSeek V4 sparse MLA backward (chunked_gather, atomic-free dKV).

    Computes dQ, dKV, and dSink (if attn_sink was provided).

    Reuses V3.2 preprocess, dKV-intermediate, and dKV-gather kernels unchanged.
    Only the dQ kernel is V4-specific (sink-inclusive LSE handling +
    d_sink reduction).

    Args:
        q:             [T, H, d_qk] bf16
        kv:            [T, 1, d_qk] bf16
        o:             [T, H, d_v] bf16 (from V4 fwd)
        do:            [T, H, d_v] bf16
        topk_indices:  [T, TOPK] int32  (SWA + sparse, pre-concatenated)
        lse:           [T, H] fp32 (sink-inclusive, from V4 fwd)
        attn_sink:     [H] fp32 or None — must match what fwd was called with
        kv_lora_rank:  int, default 512
        scale:         float, default 1/sqrt(d_qk)

    Returns:
        dq:     [T, H, d_qk] same dtype as q
        dkv:    [T, 1, d_qk] same dtype as kv
        d_sink: [H] fp32, or None if attn_sink was None
    """
    import torch as _torch  # local alias for clarity inside the wrapper
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

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.dtype == _torch.float32
        assert attn_sink.shape == (num_heads,)
        assert attn_sink.is_contiguous()
        sink_ptr = attn_sink
        d_sink = _torch.zeros(num_heads, dtype=_torch.float32, device=q.device)
    else:
        sink_ptr = _torch.empty(1, dtype=_torch.float32, device=q.device)
        d_sink = _torch.empty(1, dtype=_torch.float32, device=q.device)

    # Reuse V3.2 preprocess + dKV kernels — no V4 changes needed there.
    from ._dsa_bwd_preprocess import _sparse_mla_bwd_preprocess
    from ._dsa_bwd_gather import (
        _bwd_compute_dkv_intermediate,
        _build_inverted_topk_slice,
        _bwd_dkv_gather_acc,
    )
    from ._dsa_bwd_v4_chunked_gather import _bwd_chunk_dq_store_ds_v4

    dq = _torch.empty_like(q)
    delta = _torch.empty(total_tokens, num_heads, dtype=_torch.float32, device=q.device)

    # Delta = rowsum(O * dO). Unchanged from V3.2 — sink doesn't enter D.
    BLOCK_H_PRE = min(64, num_heads)
    BLOCK_H_PRE = triton.next_power_of_2(BLOCK_H_PRE)
    _sparse_mla_bwd_preprocess[(total_tokens, triton.cdiv(num_heads, BLOCK_H_PRE))](
        O_ptr=o, dO_ptr=do, Delta_ptr=delta,
        stride_o_t=o.stride(0), stride_o_h=o.stride(1),
        num_heads=num_heads, D_V=kv_lora_rank, BLOCK_H=BLOCK_H_PRE,
    )

    q_t  = q.transpose(1, 2).contiguous()
    do_t = do.transpose(1, 2).contiguous()

    # Chunked gather — same R_CHUNK / BLOCK_H / TILE_K as V3.2.
    R_CHUNK = min(256, topk)
    bh = 64
    num_hg = triton.cdiv(num_heads, bh)
    TILE_K_DQ  = 16
    TILE_K_DKV = 64

    chunk_dS = _torch.empty(total_tokens, num_heads, R_CHUNK, dtype=_torch.bfloat16, device=q.device)
    chunk_P  = _torch.empty(total_tokens, num_heads, R_CHUNK, dtype=_torch.bfloat16, device=q.device)
    dkv_acc  = _torch.zeros(total_tokens, d_qk, dtype=_torch.float32, device=q.device)
    interm   = _torch.empty(total_tokens, R_CHUNK, d_qk, dtype=_torch.bfloat16, device=q.device)

    # Pad topk_indices to a multiple of R_CHUNK so the dQ kernel never reads
    # past the end of the row. The dQ kernel reads R_CHUNK indices starting at
    # R_START and uses topk_pos != -1 as its validity mask. Padding with -1
    # cleanly disables the padded slots.
    topk_padded_len = ((topk + R_CHUNK - 1) // R_CHUNK) * R_CHUNK
    if topk_padded_len != topk:
        pad = _torch.full(
            (total_tokens, topk_padded_len - topk),
            -1, dtype=_torch.int32, device=q.device,
        )
        topk_indices_padded = _torch.cat([topk_indices, pad], dim=1).contiguous()
    else:
        topk_indices_padded = topk_indices

    # Precompute all CSR arrays for dKV gather phase
    all_csr = []
    for r_start in range(0, topk, R_CHUNK):
        topk_slice = topk_indices_padded[:, r_start:r_start + R_CHUNK]
        all_csr.append(_build_inverted_topk_slice(topk_slice, r_start, R_CHUNK))

    for chunk_idx, r_start in enumerate(range(0, topk, R_CHUNK)):
        is_first = (r_start == 0)

        # V4 dQ kernel (sink-aware). Sink reduction happens only in is_first chunk.
        # Use padded topk_indices so the kernel can safely read R_CHUNK entries
        # starting at any R_START — padding cells are -1 and get masked out.
        _bwd_chunk_dq_store_ds_v4[(total_tokens, num_hg)](
            q, kv, do, topk_indices_padded, lse, delta, sink_ptr,
            dq, chunk_dS, chunk_P, d_sink,
            q.stride(0), q.stride(1), kv.stride(0),
            do.stride(0), do.stride(1),
            dq.stride(0), dq.stride(1),
            topk_indices_padded.stride(0),
            chunk_dS.stride(0), chunk_dS.stride(1),
            scale, num_heads,
            R_START=r_start,
            R_CHUNK=R_CHUNK, BLOCK_H=bh, TILE_K=TILE_K_DQ,
            D_V=kv_lora_rank, D_ROPE=rope_rank,
            IS_FIRST_CHUNK=is_first,
            HAS_SINK=has_sink,
            num_warps=4, num_stages=2,
        )

        # dKV intermediate kernel (V3.2, unchanged). Uses padded topk slice for
        # consistency, though this kernel doesn't actually dereference TopK_ptr.
        chunk_topk = topk_indices_padded[:, r_start:r_start + R_CHUNK].contiguous()

        _bwd_compute_dkv_intermediate[(total_tokens,)](
            q_t, do_t, chunk_dS, chunk_P,
            chunk_topk,
            interm,
            q_t.stride(0), do_t.stride(0),
            chunk_dS.stride(0), chunk_dS.stride(1),
            R_CHUNK,
            interm.stride(0), interm.stride(1),
            num_heads,
            TOPK=R_CHUNK, TILE_K=TILE_K_DKV, BLOCK_H=bh,
            NUM_HG=num_hg, D_V=kv_lora_rank, D_ROPE=rope_rank,
            num_warps=4, num_stages=1,
        )

        inv_ptr, inv_data = all_csr[chunk_idx]
        _bwd_dkv_gather_acc[(total_tokens,)](
            interm, inv_ptr, inv_data, dkv_acc,
            interm.stride(1), dkv_acc.stride(0),
            D_V=kv_lora_rank, D_ROPE=rope_rank,
            num_warps=4,
        )

    dkv_out = dkv_acc.to(kv.dtype).unsqueeze(1)
    return dq, dkv_out, (d_sink if has_sink else None)
