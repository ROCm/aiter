"""
DeepSeek V4 backward — chunked_gather variant.

Algorithmically identical to V3.2 chunked_gather, with two surgical additions:

1. dQ kernel takes sink-inclusive LSE produced by V4 fwd. Same `dS = P * (dP - D) * scale`
   formula because P = exp(S - LSE_sink_inclusive) sums to <1 (sink absorbs the rest)
   but the dS algebra is unchanged — sink has no V-vector to back-propagate through.

2. The dQ kernel additionally accumulates `d_sink[h] = -sum_t α_sink(t,h) * D[t,h]`
   into a [H] fp32 buffer via atomic_add. This is computed once per query
   (in IS_FIRST_CHUNK branch only), so total atomics = T * H ≈ 500K — fits in
   L2 for a 128-fp32 tensor, so atomic contention is negligible.

The dKV path (build_inverted_topk_slice, _bwd_compute_dkv_intermediate,
_bwd_dkv_gather_acc) is bit-identical to V3.2 and re-used unchanged.

Inputs that change vs V3.2 dQ kernel:
  + Sink_ptr   [H] fp32, ignored when HAS_SINK == False
  + dSink_ptr  [H] fp32, accumulator (caller pre-zeros)
  + HAS_SINK   constexpr boolean
"""

import triton
import triton.language as tl


@triton.jit
def _bwd_chunk_dq_store_ds_v4(
    Q_ptr,          # [T, H, D] bf16
    KV_ptr,         # [T, 1, D] bf16
    dO_ptr,         # [T, H, D_V] bf16
    TopK_ptr,       # [T, TOPK] int32
    LSE_ptr,        # [T, H] fp32 — sink-inclusive (from V4 fwd)
    Delta_ptr,      # [T, H] fp32 — rowsum(O * dO), from preprocess (unchanged from V3.2)
    Sink_ptr,       # [H] fp32 — per-head learnable sink (ignored if HAS_SINK == False)
    dQ_ptr,         # [T, H, D] bf16 — read-modify-write across chunks
    dS_ptr,         # [T, H, R_CHUNK] bf16 — chunk dS (consumed by dKV-interm kernel)
    P_ptr,          # [T, H, R_CHUNK] bf16 — chunk P
    dSink_ptr,      # [H] fp32 — accumulator for sink gradient
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
    HAS_SINK: tl.constexpr,
):
    """
    Per-program: 1 query token × BLOCK_H heads × one rank chunk [R_START, R_START+R_CHUNK).

    Grid: (total_tokens, num_hg).

    IS_FIRST_CHUNK=True: zero-initialize dQ and contribute to d_sink. Must be called
                        exactly once per (token, head_group) across all chunks.
    """
    token_idx = tl.program_id(0)
    hg_idx = tl.program_id(1)
    offs_h = hg_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    # ---------- Load Q, dO, LSE, Delta ----------
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

    # ---------- V4 ADDITION: contribute to d_sink (only once per query) ----------
    if HAS_SINK and IS_FIRST_CHUNK:
        # α_sink(t,h) = exp(sink[h] - LSE[t,h])   (because LSE is sink-inclusive)
        # d_sink[h] = -sum_t α_sink(t,h) * D[t,h]
        sink = tl.load(Sink_ptr + offs_h, mask=mask_h, other=0.0)
        alpha_sink = tl.exp(sink - lse)
        d_sink_partial = -alpha_sink * delta
        # mask out invalid heads (zero contribution from head-group padding)
        d_sink_partial = tl.where(mask_h, d_sink_partial, 0.0)
        tl.atomic_add(dSink_ptr + offs_h, d_sink_partial, mask=mask_h)

    # ---------- dQ accumulators ----------
    dq_base = token_idx * stride_dq_t
    if IS_FIRST_CHUNK:
        dQ_lora = tl.zeros([BLOCK_H, D_V],   dtype=tl.float32)
        dQ_rope = tl.zeros([BLOCK_H, D_ROPE], dtype=tl.float32)
    else:
        dQ_lora = tl.load(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
                          mask=mask_h[:, None], other=0.0).to(tl.float32)
        dQ_rope = tl.load(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
                          mask=mask_h[:, None], other=0.0).to(tl.float32)

    # ---------- Inner loop (identical to V3.2 _bwd_chunk_dq_store_ds) ----------
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
        P  = tl.exp(S - lse[:, None])                        # sink-inclusive LSE → sum(P) < 1
        P  = tl.where(valid[None, :] & mask_h[:, None], P, 0.0)
        dP = tl.dot(dO_val, K_lora_T)
        dS = P * (dP - delta[:, None]) * scale               # same formula as V3.2
        dS = tl.where(valid[None, :] & mask_h[:, None], dS, 0.0)

        dQ_lora += tl.dot(dS.to(tl.bfloat16), tl.trans(K_lora_T)).to(tl.float32)
        dQ_rope += tl.dot(dS.to(tl.bfloat16), tl.trans(K_rope_T)).to(tl.float32)

        local_h = tl.arange(0, BLOCK_H)
        tl.store(dS_ptr + ds_base + local_h[:, None] * stride_ds_h + tile_offs[None, :],
                 dS.to(tl.bfloat16), mask=mask_h[:, None] & valid[None, :])
        tl.store(P_ptr  + ds_base + local_h[:, None] * stride_ds_h + tile_offs[None, :],
                 P.to(tl.bfloat16),  mask=mask_h[:, None] & valid[None, :])

    tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
             dQ_lora.to(Q_lora.dtype), mask=mask_h[:, None])
    tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
             dQ_rope.to(Q_rope.dtype), mask=mask_h[:, None])
