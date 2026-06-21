import torch
import triton
import triton.language as tl


# =====================================================================
# Backward method="gather" — intermediate storage + inverted topk gather
# =====================================================================
@triton.jit
def _bwd_compute_dkv_intermediate(
    Q_T_ptr,        # [T, D_QK, H] bf16  (q transposed: stride_qt_t = D_QK * H)
    dO_T_ptr,       # [T, D_V,  H] bf16
    dS_ptr,         # [T, H, TOPK] bf16
    P_ptr,          # [T, H, TOPK] bf16
    TopK_ptr,       # [T, TOPK] int32
    Interm_ptr,     # [T, TOPK, D_QK] bf16 — output, one writer per (q, topk_rank)
    stride_qt_t: tl.int64,
    stride_dot_t: tl.int64,
    stride_ds_t: tl.int64,
    stride_ds_h: tl.int64,
    stride_topk_t: tl.int64,
    stride_interm_t: tl.int64,   # TOPK * D_QK
    stride_interm_k: tl.int64,   # D_QK
    num_heads: tl.int32,
    TOPK: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_HG: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """
    Same compute as _bwd_dkv_hg_fused but writes to a private intermediate
    [T, TOPK, D] bf16 instead of atomic_add to shared dKV — no atomics needed.

    Grid: (total_tokens,) -- one program per query token q.
    For each tile of TOPK, stores dKV_lora/rope for that (q, tile) block.
    """
    token_idx = tl.program_id(0)

    NUM_TILES: tl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t
    offs_tile = tl.arange(0, TILE_K)
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    interm_base_t = token_idx * stride_interm_t  # base for this query token

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs = tile_start + offs_tile
        valid = tile_offs < TOPK

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
                mask=mask_h[:, None] & valid[None, :], other=0.0,
            )
            P_val = tl.load(
                P_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & valid[None, :], other=0.0,
            )

            dKV_lora += tl.dot(Q_lora_T, dS_val.to(Q_lora_T.dtype)).to(tl.float32)
            dKV_lora += tl.dot(dO_T, P_val.to(dO_T.dtype)).to(tl.float32)
            dKV_rope += tl.dot(Q_rope_T, dS_val.to(Q_rope_T.dtype)).to(tl.float32)

        # Store to intermediate: Interm[token_idx, tile_start:tile_start+TILE_K, 0:D_V]
        # Layout: [T, TOPK, D] so pointer = interm_base_t + tile_offs[None,:]*D + offs_v[:,None]
        interm_lora_ptrs = (Interm_ptr + interm_base_t
                            + tile_offs[None, :] * stride_interm_k
                            + offs_v[:, None])
        tl.store(interm_lora_ptrs, dKV_lora.to(tl.bfloat16), mask=valid[None, :])

        interm_rope_ptrs = (Interm_ptr + interm_base_t
                            + tile_offs[None, :] * stride_interm_k
                            + D_V + offs_r[:, None])
        tl.store(interm_rope_ptrs, dKV_rope.to(tl.bfloat16), mask=valid[None, :])


@triton.jit
def _bwd_dkv_gather(
    Interm_ptr,     # [T, TOPK, D] bf16, flattened as [T*TOPK, D]
    InvPtr_ptr,     # [T+1] int32 — CSR row pointers (kv_token -> range in inv_data)
    InvData_ptr,    # [T*TOPK] int32 — encoded as q*TOPK+r, sorted by KV token
    dKV_ptr,        # [T, D] bf16 — output
    stride_interm_k: tl.int64,   # D_V + D_ROPE
    stride_dkv_t: tl.int64,
    TOPK: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """
    Gather dKV from intermediate buffer using CSR-style inverted topk index.

    Grid: (total_tokens,) -- one CTA per KV token k.
    Accumulates in fp32, stores bf16. No atomics.
    """
    k = tl.program_id(0)
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    start = tl.load(InvPtr_ptr + k)
    end = tl.load(InvPtr_ptr + k + 1)

    dkv_acc_lora = tl.zeros([D_V], dtype=tl.float32)
    dkv_acc_rope = tl.zeros([D_ROPE], dtype=tl.float32)

    n_entries = end - start
    for i in range(n_entries):
        # entry = q*TOPK + r, used directly as flat index into [T*TOPK, D] intermediate
        entry = tl.load(InvData_ptr + start + i).to(tl.int64)
        base = entry * stride_interm_k
        lora_val = tl.load(Interm_ptr + base + offs_v)
        rope_val = tl.load(Interm_ptr + base + D_V + offs_r)
        dkv_acc_lora += lora_val.to(tl.float32)
        dkv_acc_rope += rope_val.to(tl.float32)

    dkv_base = k.to(tl.int64) * stride_dkv_t
    tl.store(dKV_ptr + dkv_base + offs_v, dkv_acc_lora.to(tl.bfloat16))
    tl.store(dKV_ptr + dkv_base + D_V + offs_r, dkv_acc_rope.to(tl.bfloat16))


def _build_inverted_topk(topk_indices):
    """
    Build CSR-style inverted index from topk_indices [T, TOPK] int32.

    Returns:
        inv_ptr:  [T+1] int32 — row pointers (kv_token -> range in inv_data)
        inv_data: [T*TOPK] int32 — encoded (q*TOPK+r) values, sorted by KV token
    """
    T, TOPK = topk_indices.shape
    device = topk_indices.device

    flat_kv = topk_indices.reshape(-1).long()   # [T*TOPK] KV token indices
    # argsort by KV token to get the flat indices (q*TOPK+r) in sorted order
    order = torch.argsort(flat_kv, stable=True)
    inv_data = order.to(torch.int32)  # [T*TOPK]

    counts = torch.zeros(T, dtype=torch.int32, device=device)
    counts.scatter_add_(0, flat_kv, torch.ones(T * TOPK, dtype=torch.int32, device=device))

    inv_ptr = torch.zeros(T + 1, dtype=torch.int32, device=device)
    torch.cumsum(counts, dim=0, out=inv_ptr[1:])

    return inv_ptr, inv_data


def _build_inverted_topk_slice(topk_indices_slice, r_start, R_CHUNK):
    """
    Build CSR-style inverted index for a topk slice, excluding invalid (-1) entries.

    Args:
        topk_indices_slice: [T, R_CHUNK] int32 — topk_indices[:, r_start:r_start+R_CHUNK]
          May contain -1 for padding (when actual chunk < R_CHUNK at the last chunk).
        r_start:  int — first rank index in this slice (unused, for documentation)
        R_CHUNK:  int — number of ranks in this slice (constexpr width)

    Returns:
        inv_ptr:  [T+1] int32 — row pointers (kv_token -> range in inv_data)
        inv_data: [valid_entries] int32 — flat indices q*R_CHUNK+local_r, sorted by KV token
    """
    T, RC = topk_indices_slice.shape
    device = topk_indices_slice.device

    flat_idx = torch.arange(T * RC, dtype=torch.int32, device=device)  # q*RC + local_r
    flat_kv = topk_indices_slice.reshape(-1).long()   # [T*R_CHUNK] KV token indices

    # Exclude invalid entries (kv_token == -1)
    valid_mask = flat_kv >= 0
    valid_flat_kv = flat_kv[valid_mask]
    valid_flat_idx = flat_idx[valid_mask]

    # Sort valid entries by KV token
    order = torch.argsort(valid_flat_kv, stable=True)
    inv_data = valid_flat_idx[order].to(torch.int32)

    counts = torch.zeros(T, dtype=torch.int32, device=device)
    counts.scatter_add_(0, valid_flat_kv,
                        torch.ones(valid_flat_kv.shape[0], dtype=torch.int32, device=device))

    inv_ptr = torch.zeros(T + 1, dtype=torch.int32, device=device)
    torch.cumsum(counts, dim=0, out=inv_ptr[1:])

    return inv_ptr, inv_data


# =====================================================================
# Backward method="chunked_gather" — three kernels per chunk (no atomics)
# =====================================================================
@triton.jit
def _bwd_chunk_dq_store_ds(
    Q_ptr,          # [T, H, D] bf16
    KV_ptr,         # [T, 1, D] bf16
    dO_ptr,         # [T, H, D_V] bf16
    TopK_ptr,       # [T, TOPK] int32
    LSE_ptr,        # [T, H] fp32
    Delta_ptr,      # [T, H] fp32
    dQ_ptr,         # [T, H, D] bf16 — read-modify-write across chunks
    dS_ptr,         # [T, H, R_CHUNK] bf16 — output chunk dS
    P_ptr,          # [T, H, R_CHUNK] bf16 — output chunk P
    stride_q_t: tl.int64, stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64, stride_do_h: tl.int64,
    stride_dq_t: tl.int64, stride_dq_h: tl.int64,
    stride_topk_t: tl.int64,
    stride_ds_t: tl.int64, stride_ds_h: tl.int64,
    scale: tl.float32, num_heads: tl.int32,
    R_START: tl.int32,
    R_CHUNK: tl.constexpr,
    BLOCK_H: tl.constexpr, TILE_K: tl.constexpr,
    D_V: tl.constexpr, D_ROPE: tl.constexpr,
    IS_FIRST_CHUNK: tl.constexpr,
):
    """
    dQ accumulation for rank chunk [R_START, R_START+R_CHUNK), plus stores
    chunk dS and P to [T, H, R_CHUNK] buffers for use by _bwd_compute_dkv_intermediate.
    Grid: (total_tokens, num_hg).
    """
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
    lse   = tl.load(LSE_ptr   + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)
    delta = tl.load(Delta_ptr + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)

    dq_base = token_idx * stride_dq_t
    if IS_FIRST_CHUNK:
        dQ_lora = tl.zeros([BLOCK_H, D_V],   dtype=tl.float32)
        dQ_rope = tl.zeros([BLOCK_H, D_ROPE], dtype=tl.float32)
    else:
        dQ_lora = tl.load(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
                          mask=mask_h[:, None], other=0.0).to(tl.float32)
        dQ_rope = tl.load(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
                          mask=mask_h[:, None], other=0.0).to(tl.float32)

    NUM_TILES: tl.constexpr = (R_CHUNK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t + R_START
    offs_tile = tl.arange(0, TILE_K)
    ds_base = token_idx * stride_ds_t + hg_idx * BLOCK_H * stride_ds_h

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs  = tile_start + offs_tile
        valid = tile_offs < R_CHUNK
        topk_pos = tl.load(TopK_ptr + topk_base + tile_offs, mask=valid, other=-1)
        valid = valid & (topk_pos != -1)
        safe_pos = tl.where(valid, topk_pos, 0)

        K_lora_T = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None],
                           mask=valid[None, :], other=0.0)
        K_rope_T = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
                           mask=valid[None, :], other=0.0)

        S  = tl.dot(Q_lora, K_lora_T) + tl.dot(Q_rope, K_rope_T)
        S  = tl.where(valid[None, :] & mask_h[:, None], S * scale, float("-inf"))
        P  = tl.exp(S - lse[:, None])
        P  = tl.where(valid[None, :] & mask_h[:, None], P, 0.0)
        dP = tl.dot(dO_val, K_lora_T)
        dS = P * (dP - delta[:, None]) * scale
        dS = tl.where(valid[None, :] & mask_h[:, None], dS, 0.0)

        dQ_lora += tl.dot(dS.to(tl.bfloat16), tl.trans(K_lora_T)).to(tl.float32)
        dQ_rope += tl.dot(dS.to(tl.bfloat16), tl.trans(K_rope_T)).to(tl.float32)

        # Store chunk dS and P for this tile — use local head offsets (0..BLOCK_H-1)
        # since ds_base already encodes hg_idx*BLOCK_H*stride_ds_h
        local_h = tl.arange(0, BLOCK_H)
        tl.store(dS_ptr + ds_base + local_h[:, None] * stride_ds_h + tile_offs[None, :],
                 dS.to(tl.bfloat16), mask=mask_h[:, None] & valid[None, :])
        tl.store(P_ptr  + ds_base + local_h[:, None] * stride_ds_h + tile_offs[None, :],
                 P.to(tl.bfloat16),  mask=mask_h[:, None] & valid[None, :])

    tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
             dQ_lora.to(Q_lora.dtype), mask=mask_h[:, None])
    tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
             dQ_rope.to(Q_rope.dtype), mask=mask_h[:, None])


@triton.jit
def _bwd_chunk_dq(
    Q_ptr,          # [T, H, D] bf16
    KV_ptr,         # [T, 1, D] bf16
    dO_ptr,         # [T, H, D_V] bf16
    TopK_ptr,       # [T, TOPK] int32
    LSE_ptr,        # [T, H] fp32
    Delta_ptr,      # [T, H] fp32
    dQ_ptr,         # [T, H, D] bf16 — read-modify-write across chunks
    stride_q_t: tl.int64, stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64, stride_do_h: tl.int64,
    stride_dq_t: tl.int64, stride_dq_h: tl.int64,
    stride_topk_t: tl.int64,
    scale: tl.float32, num_heads: tl.int32,
    R_START: tl.int32,
    R_CHUNK: tl.constexpr,
    BLOCK_H: tl.constexpr, TILE_K: tl.constexpr,
    D_V: tl.constexpr, D_ROPE: tl.constexpr,
    IS_FIRST_CHUNK: tl.constexpr,
):
    """
    dQ accumulation for one rank chunk [R_START, R_START+R_CHUNK).
    Grid: (total_tokens, num_hg). No writes to intermediate buffer.
    IS_FIRST_CHUNK=True: initialises dQ to zero (avoids a global memory read).
    """
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
    lse   = tl.load(LSE_ptr   + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)
    delta = tl.load(Delta_ptr + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)

    dq_base = token_idx * stride_dq_t
    if IS_FIRST_CHUNK:
        dQ_lora = tl.zeros([BLOCK_H, D_V],   dtype=tl.float32)
        dQ_rope = tl.zeros([BLOCK_H, D_ROPE], dtype=tl.float32)
    else:
        dQ_lora = tl.load(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
                          mask=mask_h[:, None], other=0.0).to(tl.float32)
        dQ_rope = tl.load(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
                          mask=mask_h[:, None], other=0.0).to(tl.float32)

    NUM_TILES: tl.constexpr = (R_CHUNK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t + R_START
    offs_tile = tl.arange(0, TILE_K)
    topk_pos = tl.load(TopK_ptr + topk_base + offs_tile, mask=offs_tile < R_CHUNK, other=-1)
    topk_pos_next = topk_pos

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        valid = (tile_start + offs_tile) < R_CHUNK
        valid = valid & (topk_pos != -1)
        if t + 1 < NUM_TILES:
            next_offs = (t + 1) * TILE_K + offs_tile
            topk_pos_next = tl.load(TopK_ptr + topk_base + next_offs,
                                    mask=next_offs < R_CHUNK, other=-1)
        safe_pos = tl.where(valid, topk_pos, 0)

        K_lora_T = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None],
                           mask=valid[None, :], other=0.0)
        K_rope_T = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
                           mask=valid[None, :], other=0.0)

        S  = tl.dot(Q_lora, K_lora_T) + tl.dot(Q_rope, K_rope_T)
        S  = tl.where(valid[None, :] & mask_h[:, None], S * scale, float("-inf"))
        P  = tl.exp(S - lse[:, None])
        P  = tl.where(valid[None, :] & mask_h[:, None], P, 0.0)
        dP = tl.dot(dO_val, K_lora_T)
        dS = P * (dP - delta[:, None]) * scale
        dS = tl.where(valid[None, :] & mask_h[:, None], dS, 0.0)

        dQ_lora += tl.dot(dS.to(tl.bfloat16), tl.trans(K_lora_T)).to(tl.float32)
        dQ_rope += tl.dot(dS.to(tl.bfloat16), tl.trans(K_rope_T)).to(tl.float32)

        if t + 1 < NUM_TILES:
            topk_pos = topk_pos_next

    tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
             dQ_lora.to(Q_lora.dtype), mask=mask_h[:, None])
    tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
             dQ_rope.to(Q_rope.dtype), mask=mask_h[:, None])


@triton.jit
def _bwd_chunk_dkv_interm(
    Q_T_ptr,        # [T, D_QK, H] bf16  (transposed: stride = D_QK * H)
    dO_T_ptr,       # [T, D_V,  H] bf16
    TopK_ptr,       # [T, TOPK] int32
    LSE_ptr,        # [T, H] fp32
    Delta_ptr,      # [T, H] fp32
    KV_ptr,         # [T, 1, D] bf16
    Interm_ptr,     # [T, R_CHUNK, D] bf16 — output (plain store, one writer per slot)
    stride_qt_t: tl.int64,
    stride_dot_t: tl.int64,
    stride_topk_t: tl.int64,
    stride_kv_t: tl.int64,
    stride_interm_t: tl.int64,   # R_CHUNK * D
    stride_interm_r: tl.int64,   # D
    scale: tl.float32, num_heads: tl.int32,
    R_START: tl.int32,
    R_CHUNK: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_HG: tl.constexpr,
    D_V: tl.constexpr, D_ROPE: tl.constexpr,
):
    """
    dKV intermediate for one rank chunk [R_START, R_START+R_CHUNK).
    Grid: (total_tokens,) — one CTA per query token, inner loop over head groups.
    Recomputes S/P/dS on-the-fly. Plain stores to bf16 interm — no atomics.
    """
    token_idx = tl.program_id(0)
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)
    offs_tile = tl.arange(0, TILE_K)

    NUM_TILES: tl.constexpr = (R_CHUNK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t + R_START
    interm_base_t = token_idx * stride_interm_t

    qt_base  = token_idx * stride_qt_t
    dot_base = token_idx * stride_dot_t

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs  = tile_start + offs_tile
        valid_tile = tile_offs < R_CHUNK

        topk_pos = tl.load(TopK_ptr + topk_base + tile_start + offs_tile,
                           mask=valid_tile, other=-1)
        valid    = valid_tile & (topk_pos != -1)
        safe_pos = tl.where(valid, topk_pos, 0)

        K_lora = tl.load(KV_ptr + safe_pos[:, None] * stride_kv_t + offs_v[None, :],
                         mask=valid[:, None], other=0.0)   # [TILE_K, D_V]
        K_rope = tl.load(KV_ptr + safe_pos[:, None] * stride_kv_t + (D_V + offs_r[None, :]),
                         mask=valid[:, None], other=0.0)   # [TILE_K, D_ROPE]

        dKV_lora = tl.zeros([TILE_K, D_V],   dtype=tl.float32)
        dKV_rope = tl.zeros([TILE_K, D_ROPE], dtype=tl.float32)

        for hg in range(NUM_HG):
            offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
            mask_h = offs_h < num_heads

            Q_lora_T = tl.load(Q_T_ptr + qt_base  + offs_v[:, None] * num_heads + offs_h[None, :],
                                mask=mask_h[None, :], other=0.0)          # [D_V, BLOCK_H]
            Q_rope_T = tl.load(Q_T_ptr + qt_base  + (D_V + offs_r[:, None]) * num_heads + offs_h[None, :],
                                mask=mask_h[None, :], other=0.0)          # [D_ROPE, BLOCK_H]
            dO_T     = tl.load(dO_T_ptr + dot_base + offs_v[:, None] * num_heads + offs_h[None, :],
                                mask=mask_h[None, :], other=0.0)          # [D_V, BLOCK_H]
            lse_h    = tl.load(LSE_ptr   + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)
            delta_h  = tl.load(Delta_ptr + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)

            # S = K @ Q^T  [TILE_K, BLOCK_H]
            S  = tl.dot(K_lora, Q_lora_T) + tl.dot(K_rope, Q_rope_T)
            S  = tl.where(valid[:, None] & mask_h[None, :], S * scale, float("-inf"))
            P  = tl.exp(S - lse_h[None, :])
            P  = tl.where(valid[:, None] & mask_h[None, :], P, 0.0)
            dP = tl.dot(K_lora, dO_T)   # [TILE_K, BLOCK_H]
            dS = P * (dP - delta_h[None, :]) * scale
            dS = tl.where(valid[:, None] & mask_h[None, :], dS, 0.0)

            dKV_lora += tl.dot(dS.to(tl.bfloat16), tl.trans(Q_lora_T)).to(tl.float32)
            dKV_lora += tl.dot(P.to(tl.bfloat16),  tl.trans(dO_T)).to(tl.float32)
            dKV_rope += tl.dot(dS.to(tl.bfloat16), tl.trans(Q_rope_T)).to(tl.float32)

        # Plain store to bf16 interm — one writer per (token, local_r) slot
        interm_lora_ptrs = (Interm_ptr + interm_base_t
                            + tile_offs[:, None] * stride_interm_r + offs_v[None, :])
        tl.store(interm_lora_ptrs, dKV_lora.to(tl.bfloat16), mask=valid[:, None])

        interm_rope_ptrs = (Interm_ptr + interm_base_t
                            + tile_offs[:, None] * stride_interm_r + (D_V + offs_r[None, :]))
        tl.store(interm_rope_ptrs, dKV_rope.to(tl.bfloat16), mask=valid[:, None])


@triton.jit
def _bwd_dkv_gather_acc(
    Interm_ptr,     # [T, R_CHUNK, D] bf16 — chunk intermediate
    InvPtr_ptr,     # [T+1] int32 — CSR row pointers
    InvData_ptr,    # [T*R_CHUNK] int32 — encoded as q*R_CHUNK + local_r
    dKV_acc_ptr,    # [T, D] fp32 — accumulator (read-modify-write across chunks)
    stride_interm_r: tl.int64,   # D
    stride_acc_t: tl.int64,      # D
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """
    Gather one chunk's bf16 intermediate into the fp32 dKV accumulator.
    Grid: (total_tokens,) — one CTA per KV token k, no atomics.
    """
    k = tl.program_id(0)
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    start = tl.load(InvPtr_ptr + k)
    end   = tl.load(InvPtr_ptr + k + 1)

    acc_base = k.to(tl.int64) * stride_acc_t
    dkv_acc_lora = tl.load(dKV_acc_ptr + acc_base + offs_v).to(tl.float32)
    dkv_acc_rope = tl.load(dKV_acc_ptr + acc_base + D_V + offs_r).to(tl.float32)

    n_entries = end - start
    for i in range(n_entries):
        entry = tl.load(InvData_ptr + start + i).to(tl.int64)
        base  = entry * stride_interm_r
        dkv_acc_lora += tl.load(Interm_ptr + base + offs_v).to(tl.float32)
        dkv_acc_rope += tl.load(Interm_ptr + base + D_V + offs_r).to(tl.float32)

    tl.store(dKV_acc_ptr + acc_base + offs_v,       dkv_acc_lora)
    tl.store(dKV_acc_ptr + acc_base + D_V + offs_r, dkv_acc_rope)
