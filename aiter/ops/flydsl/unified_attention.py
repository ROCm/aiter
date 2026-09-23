"""FlyDSL unified-attention entrypoint."""

import torch


def unified_attention_flydsl(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    q_scales=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    shuffled_kv_cache: bool = False,
    skip_reduce: bool = False,
    # backend
    backend: str | None = None,  # "triton" | "gluon" | "flydsl"
) -> torch.Tensor | None:
    try:
        from aiter.ops.flydsl.unified_attention_kernels import is_flydsl_available
    except ImportError:
        return None

    q_device_index = q.device.index
    if q_device_index is None:
        q_device_index = torch.cuda.current_device()
    with torch.cuda.device(q_device_index):
        if not is_flydsl_available(q_device_index):
            return None

    try:
        from aiter.ops.flydsl.unified_attention_kernels import flydsl_unified_attention
    except ImportError:
        return None

    _num_tokens, num_query_heads, _head_size = q.shape
    kv_cache_dtype = k.dtype
    if k.dim() == 5:
        _num_blocks, num_kv_heads, _, block_size, _K_WIDTH = k.shape
        shuffled_kv_cache = True
    elif k.dim() == 4:
        if shuffled_kv_cache and kv_cache_dtype == torch.uint8:
            _num_blocks, num_kv_heads, block_size, _ = k.shape
        else:
            _num_blocks, block_size, num_kv_heads, _ = k.shape
            shuffled_kv_cache = False
    else:
        return None

    num_seqs = len(seqused_k)
    num_queries_per_kv = num_query_heads // num_kv_heads

    return flydsl_unified_attention(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        num_queries_per_kv=num_queries_per_kv,
        num_seqs=num_seqs,
        q_scales=q_scales,
        alibi_slopes=alibi_slopes,
        output_scale=output_scale,
        qq_bias=qq_bias,
        sinks=sinks,
        shuffled_kv_cache=shuffled_kv_cache,
        skip_reduce=skip_reduce,
    )
