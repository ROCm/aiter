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
) -> torch.Tensor | None:
    try:
        from aiter.ops.flydsl.unified_attention_kernels import (
            flydsl_unified_attention,
            is_flydsl_available,
        )
    except ModuleNotFoundError as e:
        # Only an absent FlyDSL package means "no backend"; a broken internal
        # import must surface with its own traceback.
        if e.name == "flydsl":
            return None
        raise

    if not q.is_cuda:
        return None

    q_device_index = q.device.index
    if q_device_index is None:
        q_device_index = torch.cuda.current_device()
    with torch.cuda.device(q_device_index):
        if not is_flydsl_available(q_device_index):
            return None

    # shuffled_kv_cache is already normalized by the router: rank 5 <=> shuffled.
    if k.dim() == 5:
        num_kv_heads, block_size = k.shape[1], k.shape[3]
    elif k.dim() == 4:
        # A shuffled flag on a rank-4 cache is declined by the stride gate.
        block_size, num_kv_heads = k.shape[1], k.shape[2]
    else:
        return None

    num_seqs = len(seqused_k)

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
        num_seqs=num_seqs,
        q_scales=q_scales,
        alibi_slopes=alibi_slopes,
        output_scale=output_scale,
        qq_bias=qq_bias,
        sinks=sinks,
        shuffled_kv_cache=shuffled_kv_cache,
        skip_reduce=skip_reduce,
    )
