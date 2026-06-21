import torch
import triton
import triton.language as tl


# =====================================================================
# Backward method="persistent" — Approach A: persistent 304-CTA kernel
# =====================================================================
@triton.jit
def _bwd_persistent_chunk(
    Q_ptr,          # [T, H, D] bf16
    KV_ptr,         # [T, 1, D] bf16
    dO_ptr,         # [T, H, D_V] bf16
    TopK_ptr,       # [T, TOPK] int32
    LSE_ptr,        # [T, H] fp32
    Delta_ptr,      # [T, H] fp32
    dQ_ptr,         # [T, H, D] bf16 — output
    dKV_chunk_ptr,  # [NUM_XCD, K_CHUNK, D] fp32 — XCD-local accumulator
    stride_q_t: tl.int64, stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64, stride_do_h: tl.int64,
    stride_dq_t: tl.int64, stride_dq_h: tl.int64,
    stride_topk_t: tl.int64,
    stride_chunk_xcd: tl.int64,   # K_CHUNK * D
    stride_chunk_k: tl.int64,     # D
    scale: tl.float32,
    total_tokens: tl.int32,
    num_heads: tl.int32,
    k_start: tl.int32,            # first KV token index in this chunk
    k_end: tl.int32,              # one past last KV token index in this chunk
    TOPK: tl.int32,               # NOT constexpr — avoids unrolling 1024 iterations
    TOKENS_PER_CU: tl.int32,     # NOT constexpr — avoids unrolling 14 token iterations
    BLOCK_H: tl.constexpr,
    NUM_HG: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """
    Persistent backward kernel for Approach A.

    grid=(NUM_CUS,) — one CTA per physical CU on MI300X.
    Each CTA:
      1. Reads hw_id to obtain its XCD index (0-7) for L2-local atomics.
      2. Owns tokens [pid*(T//NUM_CUS), (pid+1)*(T//NUM_CUS)).
      3. For each owned token: loops all TOPK ranks, accumulates dQ in fp32
         registers, atomic_adds dKV contributions into dKV_chunk[xcd, k-k_start, :]
         (only for KV tokens in [k_start, k_end)).
      4. Writes dQ[q] once per token.

    dKV_chunk[xcd] is K_CHUNK*D fp32 = 3.14 MB for K_CHUNK=1365, D=576.
    This fits in the 4 MB L2 per XCD so atomics stay L2-local.
    """
    pid = tl.program_id(0)

    # XCD index: use pid % 8 as a uniform proxy for XCD assignment.
    # (Reading hw_id via inline_asm_elementwise returns a vgpr/per-lane tensor
    #  which causes Triton to generate per-lane scatter addressing, blowing up
    #  compilation. The static formula is sufficient since the kernel's benefit
    #  comes from K_CHUNK fitting in L2, not from precise CTA→XCD routing.)
    xcd = pid % 8

    # Token range owned by this CTA — last CTA may own fewer tokens (guarded by valid_q)
    q_start = pid * TOKENS_PER_CU

    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)
    chunk_base = xcd.to(tl.int64) * stride_chunk_xcd

    for q_off in range(TOKENS_PER_CU):
        q = q_start + q_off
        valid_q = q < total_tokens  # guard for last CTA which may own fewer tokens

        q_base  = q * stride_q_t
        do_base = q * stride_do_t
        topk_base = q * stride_topk_t

        for hg in range(NUM_HG):
            offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
            mask_h = (offs_h < num_heads) & valid_q

            Q_lora = tl.load(Q_ptr + q_base + offs_h[:, None] * stride_q_h + offs_v[None, :],
                             mask=mask_h[:, None], other=0.0)   # [BLOCK_H, D_V]
            Q_rope = tl.load(Q_ptr + q_base + offs_h[:, None] * stride_q_h + (D_V + offs_r[None, :]),
                             mask=mask_h[:, None], other=0.0)   # [BLOCK_H, D_ROPE]
            dO_val = tl.load(dO_ptr + do_base + offs_h[:, None] * stride_do_h + offs_v[None, :],
                             mask=mask_h[:, None], other=0.0)   # [BLOCK_H, D_V]
            lse   = tl.load(LSE_ptr   + q * num_heads + offs_h, mask=mask_h, other=0.0)
            delta = tl.load(Delta_ptr + q * num_heads + offs_h, mask=mask_h, other=0.0)

            dQ_lora = tl.zeros([BLOCK_H, D_V],   dtype=tl.float32)
            dQ_rope = tl.zeros([BLOCK_H, D_ROPE], dtype=tl.float32)

            for r in range(TOPK):
                kv_pos = tl.load(TopK_ptr + topk_base + r, mask=valid_q, other=0).to(tl.int32)

                # Load KV token and promote to fp32 for accurate dot products
                K_lora_T = tl.load(KV_ptr + kv_pos * stride_kv_t + offs_v).to(tl.float32)    # [D_V]
                K_rope_T = tl.load(KV_ptr + kv_pos * stride_kv_t + D_V + offs_r).to(tl.float32)  # [D_ROPE]

                # Attention scores S = Q @ K^T → [BLOCK_H] (fp32 accumulation)
                S = tl.sum(Q_lora.to(tl.float32) * K_lora_T[None, :], axis=1) \
                  + tl.sum(Q_rope.to(tl.float32) * K_rope_T[None, :], axis=1)
                S = tl.where(mask_h, S * scale, float("-inf"))
                P = tl.exp(S - lse)
                P = tl.where(mask_h, P, 0.0)

                # dS = P * (dO @ K^T - delta) * scale → [BLOCK_H]
                dP = tl.sum(dO_val.to(tl.float32) * K_lora_T[None, :], axis=1)
                dS = P * (dP - delta) * scale
                dS = tl.where(mask_h, dS, 0.0)

                # Accumulate dQ (fp32)
                dQ_lora += dS[:, None] * K_lora_T[None, :]
                dQ_rope += dS[:, None] * K_rope_T[None, :]

                # Accumulate dKV for this chunk — only for KV tokens in [k_start, k_end)
                in_chunk = (kv_pos >= k_start) & (kv_pos < k_end)
                local_k = tl.where(in_chunk, kv_pos - k_start, 0)
                dkv_ptr_lora = dKV_chunk_ptr + chunk_base + local_k * stride_chunk_k + offs_v
                dkv_ptr_rope = dKV_chunk_ptr + chunk_base + local_k * stride_chunk_k + D_V + offs_r
                # dKV_lora += dS * Q_lora + P * dO  (summed over BLOCK_H heads, fp32)
                dkv_contrib_lora = tl.sum(dS[:, None] * Q_lora.to(tl.float32), axis=0) \
                                 + tl.sum(P[:, None]  * dO_val.to(tl.float32), axis=0)
                dkv_contrib_rope = tl.sum(dS[:, None] * Q_rope.to(tl.float32), axis=0)
                # Broadcast in_chunk scalar to vector mask — suppresses add for out-of-chunk tokens
                mask_v = in_chunk & (offs_v < D_V)
                mask_r = in_chunk & (offs_r < D_ROPE)
                tl.atomic_add(dkv_ptr_lora, dkv_contrib_lora, mask=mask_v)
                tl.atomic_add(dkv_ptr_rope, dkv_contrib_rope, mask=mask_r)

            # Write dQ for this head group
            dq_base = q * stride_dq_t
            tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
                     dQ_lora.to(Q_lora.dtype), mask=mask_h[:, None])
            tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
                     dQ_rope.to(Q_rope.dtype), mask=mask_h[:, None])


@triton.jit
def _bwd_chunk_reduce(
    dKV_chunk_ptr,  # [NUM_XCD, K_CHUNK, D] fp32
    dKV_ptr,        # [T, D] fp32 — output
    stride_chunk_xcd: tl.int64,
    stride_dkv_t: tl.int64,
    k_start: tl.int32,
    K_CHUNK: tl.constexpr,
    NUM_XCD: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Reduce the NUM_XCD dkv_chunk copies for one K_CHUNK into dKV[k_start:k_end].
    Grid: (K_CHUNK, D // BLOCK_D)
    """
    k_local = tl.program_id(0)
    d_block = tl.program_id(1)
    offs_d  = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d  = offs_d < D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for xcd in tl.static_range(NUM_XCD):
        ptr = dKV_chunk_ptr + xcd * stride_chunk_xcd + k_local * D + offs_d
        acc += tl.load(ptr, mask=mask_d, other=0.0)

    kv_token = k_start + k_local
    tl.store(dKV_ptr + kv_token * stride_dkv_t + offs_d, acc, mask=mask_d)
