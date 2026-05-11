# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Side-by-side example: feed the SAME logical (Q, K, V) batch to

    1. CK Unified Attention   -> aiter.ops.unified_attention.unified_attention_fwd
    2. CK FMHA  (4D batched)  -> aiter.ops.mha.mha_fwd

The point of this script is to make the *signature* differences obvious; to
keep it runnable on stock builds we use a uniform batch (every seq has the
same q_len / kv_len) so FMHA's simple 4D path is happy and we don't depend
on FMHA's paged splitkv / batch_prefill instances (some of which can be
broken on certain JIT builds).

Run with:

    python op_tests/compare_unified_vs_fmha_signatures.py
"""

import torch

from aiter.ops.unified_attention import unified_attention_fwd
from aiter.ops.mha import mha_fwd


def main():
    device = "cuda"
    dtype = torch.bfloat16

    # ---------------------------------------------------------------
    # 1. Common problem shape
    #
    # The CK unified-attention dispatch
    # (3rdparty/.../42_unified_attention/unified_attention.cpp) only ships
    # compiled instances for two shape families:
    #   * hdim=128, num_queries_per_kv=1   (MHA  -- this script)
    #   * hdim=64,  num_queries_per_kv=8   (GQA-8)
    # Anything else returns "no matching kernel instance".
    # ---------------------------------------------------------------
    batch          = 2
    seqlen_q       = 128
    seqlen_k       = 128                     # uniform: every seq has same q_len/kv_len
    num_kv_heads   = 8
    num_q_per_kv   = 1                       # MHA
    num_q_heads    = num_kv_heads * num_q_per_kv
    head_size      = 128
    page_blk_size  = 64
    scale_s        = head_size ** -0.5

    # ---------------------------------------------------------------
    # 2. Generate the shared logical (Q, K, V) data once, in a 4D layout.
    #    We then derive both the unified-attention layout and the FMHA
    #    layout from this same data so both kernels compute on equivalent
    #    inputs.
    # ---------------------------------------------------------------
    torch.manual_seed(0)

    # FMHA-friendly layout: [batch, seqlen, num_heads, head_size]
    q_4d = torch.randn(batch, seqlen_q, num_q_heads,  head_size, dtype=dtype, device=device)
    k_4d = torch.randn(batch, seqlen_k, num_kv_heads, head_size, dtype=dtype, device=device)
    v_4d = torch.randn(batch, seqlen_k, num_kv_heads, head_size, dtype=dtype, device=device)

    # ---------------------------------------------------------------
    # 3. Unified-attention layout (varlen + paged KV cache)
    # ---------------------------------------------------------------
    total_q_tokens   = batch * seqlen_q
    q_packed         = q_4d.reshape(total_q_tokens, num_q_heads, head_size).contiguous()
    out_ua           = torch.empty_like(q_packed)

    # Paged KV cache, [num_pages, page_blk_size, num_kv_heads, head_size]
    pages_per_seq    = (seqlen_k + page_blk_size - 1) // page_blk_size
    total_pages      = batch * pages_per_seq
    key_cache        = torch.zeros(total_pages, page_blk_size, num_kv_heads,
                                   head_size, dtype=dtype, device=device)
    value_cache      = torch.zeros_like(key_cache)
    block_tables     = torch.arange(total_pages, dtype=torch.int32,
                                    device=device).reshape(batch, pages_per_seq)

    # Scatter k_4d/v_4d into the paged buffer using block_tables
    for b in range(batch):
        for p in range(pages_per_seq):
            phys = int(block_tables[b, p].item())
            tok_start = p * page_blk_size
            tok_end   = min(tok_start + page_blk_size, seqlen_k)
            n = tok_end - tok_start
            key_cache[phys, :n]   = k_4d[b, tok_start:tok_end]
            value_cache[phys, :n] = v_4d[b, tok_start:tok_end]

    seq_lens_t          = torch.full((batch,), seqlen_k, dtype=torch.int32, device=device)
    query_start_len     = torch.arange(0, total_q_tokens + 1, seqlen_q,
                                       dtype=torch.int32, device=device)
    cu_seqlens_q        = query_start_len.clone()           # FMHA name for the same thing
    cu_seqlens_k        = torch.arange(0, batch * seqlen_k + 1, seqlen_k,
                                       dtype=torch.int32, device=device)

    # ---------------------------------------------------------------
    # 4. CK Unified Attention call
    #    output is written in-place; nothing returned.
    # ---------------------------------------------------------------
    unified_attention_fwd(
        out_ua,                      # [total_q_tokens, num_q_heads, head_size]
        q_packed,                    # same shape as output
        key_cache,                   # [num_pages, page_blk_size, num_kv_heads, head_size]
        value_cache,
        block_tables,                # [batch, pages_per_seq]
        seq_lens_t,                  # [batch]    full context len per seq
        query_start_len,             # [batch+1]  cumulative query tokens
        mask_type=2,                 # 0 = no mask, 2 = causal  (only options)
        scale_s=scale_s,
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,  # FP8 quant scales
    )
    out_ua_4d = out_ua.view(batch, seqlen_q, num_q_heads, head_size)

    # ---------------------------------------------------------------
    # 5. CK FMHA  -- mha_fwd  (batched, non-paged)
    #    Returns (out, lse, P, rng_state).
    # ---------------------------------------------------------------
    out_fmha, lse, _, _ = mha_fwd(
        q_4d,                        # [batch, seqlen_q, num_q_heads, head_size]
        k_4d,                        # [batch, seqlen_k, num_kv_heads, head_size]
        v_4d,
        0.0,                         # dropout_p           (not in unified attention)
        scale_s,                     # softmax_scale
        True,                        # is_causal           (vs. unified's mask_type=2)
        -1,                          # window_size_left    (sliding window, not in unified)
        -1,                          # window_size_right
        0,                           # sink_size           (not in unified attention)
        False,                       # return_softmax_lse  (not in unified attention)
        False,                       # return_dropout_randval
        # below are kw-only optional tensors:
        cu_seqlens_q=None,           # set for varlen path (not used here)
        cu_seqlens_kv=None,
        out=None,
        bias=None,                   # custom bias       (not in unified attention)
        alibi_slopes=None,           # ALiBi             (not in unified attention)
        q_descale=None,              # FP8 descale       (unified uses scale_s/scale_k/scale_v/scale_out)
        k_descale=None,
        v_descale=None,
        sink_ptr=None,
        gen=None,
    )

    torch.cuda.synchronize()

    # ---------------------------------------------------------------
    # 6. Report
    # ---------------------------------------------------------------
    print("=" * 68)
    print("INPUTS")
    print("=" * 68)
    print(f"  batch={batch}  seqlen_q={seqlen_q}  seqlen_k={seqlen_k}")
    print(f"  num_q_heads={num_q_heads}  num_kv_heads={num_kv_heads}  "
          f"num_q_per_kv={num_q_per_kv}  head_size={head_size}")
    print(f"  page_blk_size={page_blk_size}  total_pages={total_pages}")
    print()
    print(f"  unified-attention layout:")
    print(f"    q_packed     {tuple(q_packed.shape)}      {q_packed.dtype}")
    print(f"    key_cache    {tuple(key_cache.shape)}  paged")
    print(f"    block_tables {tuple(block_tables.shape)}  {block_tables.dtype}")
    print(f"    seq_lens     {seq_lens_t.tolist()}")
    print(f"    q_start_len  {query_start_len.tolist()}")
    print()
    print(f"  FMHA mha_fwd layout:")
    print(f"    q_4d         {tuple(q_4d.shape)}    {q_4d.dtype}")
    print(f"    k_4d         {tuple(k_4d.shape)}    contiguous (non-paged)")
    print()

    print("=" * 68)
    print("OUTPUTS")
    print("=" * 68)
    print(f"  unified out  {tuple(out_ua_4d.shape)}  {out_ua_4d.dtype}  "
          f"abs_mean={out_ua_4d.abs().mean().item():.6f}")
    print(f"  fmha    out  {tuple(out_fmha.shape)}  {out_fmha.dtype}  "
          f"abs_mean={out_fmha.abs().mean().item():.6f}")
    diff = (out_ua_4d.float() - out_fmha.float()).abs()
    print(f"  max abs diff  : {diff.max().item():.4e}")
    print(f"  mean abs diff : {diff.mean().item():.4e}")


if __name__ == "__main__":
    main()
