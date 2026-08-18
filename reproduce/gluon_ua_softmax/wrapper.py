"""Minimal launcher for the gfx950 Gluon unified-attention kernel.

Trimmed from aiter/ops/triton/attention/unified_attention.py: prefill config only
(num_splits == 1), so no split-reduce kernel is needed.
"""

import torch

from ua_kernel import _unified_attention_gluon_kernel


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    max_seqlen_q,
    softmax_scale,
    causal,
    sinks,
    block_table,
    window_size=(-1, -1),
):
    NUM_SEQS = len(seqused_k)
    NUM_Q_HEADS = q.shape[1]
    HEAD_SIZE = q.shape[2]
    num_blocks, BLOCK_SIZE, NUM_KV_HEADS = k.shape[0], k.shape[1], k.shape[2]
    ALL_DECODE = max_seqlen_q == 1
    assert not ALL_DECODE, "prefill config only"
    SLIDING_WINDOW = 1 + window_size[0]
    NUM_QUERIES_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
    Q_FP8 = q.element_size() == 1

    if SLIDING_WINDOW > 0 and HEAD_SIZE <= 128:
        waves_per_eu = 0
    elif HEAD_SIZE < 128:
        waves_per_eu = 3
    elif HEAD_SIZE >= 256:
        waves_per_eu = 2 if Q_FP8 else 1
    else:
        waves_per_eu = 2

    TILE_SIZE = 64
    num_warps, BLOCK_M, mfma_dim, num_buffers = 4, 128, 32, 2
    if HEAD_SIZE >= 256 and Q_FP8:
        num_buffers = 1
    elif HEAD_SIZE >= 256 and BLOCK_SIZE >= 32:
        num_warps, BLOCK_M, TILE_SIZE = 2, 64, 32
    if 4 * TILE_SIZE * HEAD_SIZE * k.element_size() > 160 * 1024:
        num_buffers = 1

    BLOCK_Q = BLOCK_M // NUM_QUERIES_PER_KV
    total_query_blocks = q.shape[0] // BLOCK_Q + NUM_SEQS
    MAX_INT32 = 2**31 - 1

    _unified_attention_gluon_kernel[(NUM_KV_HEADS, total_query_blocks)](
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        output_ptr=out,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        query_start_len_ptr=cu_seqlens_q,
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        k_descale_ptr=None,
        v_descale_ptr=None,
        q_descale_ptr=None,
        out_scale_ptr=None,
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=SLIDING_WINDOW,
        num_blocks=num_blocks,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        block_table_stride=block_table.stride(0),
        num_seqs=NUM_SEQS,
        SCALE=softmax_scale,
        NUM_QUERY_HEADS=NUM_Q_HEADS,
        NUM_KV_HEADS=NUM_KV_HEADS,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        MFMA_DIM=mfma_dim,
        ARCH_NAME="gfx950",
        waves_per_eu=waves_per_eu,
        USE_LOAD_BUFFER_OP=k.nelement() * k.element_size() <= MAX_INT32,
        USE_STORE_BUFFER_OP=out.nelement() * out.element_size() <= MAX_INT32,
        num_warps=num_warps,
        ALL_DECODE=ALL_DECODE,
        CAUSAL=causal,
        REMOVE_INDIRECT_ACCESS=False,
        NUM_BUFFERS=num_buffers,
        NUM_SPLITS=1,
        partial_m_ptr=None,
        partial_l_ptr=None,
        partial_acc_ptr=None,
    )
    return out
