"""Route unified attention between FlyDSL and Triton/Gluon backends."""


def unified_attention(
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
):
    if backend not in (None, "flydsl", "triton", "gluon"):
        raise ValueError(
            f"Unknown backend '{backend}', must be None, 'triton', 'gluon' or 'flydsl'"
        )

    if backend in (None, "flydsl"):
        from aiter.ops.flydsl.unified_attention import unified_attention_flydsl

        flydsl_out = unified_attention_flydsl(
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
            q_scales=q_scales,
            alibi_slopes=alibi_slopes,
            output_scale=output_scale,
            qq_bias=qq_bias,
            sinks=sinks,
            shuffled_kv_cache=shuffled_kv_cache,
            skip_reduce=skip_reduce,
            backend=backend,
        )
        if flydsl_out is not None:
            return flydsl_out
        if backend == "flydsl":
            raise RuntimeError(
                "FlyDSL unified_attention backend is unavailable or does not support this configuration"
            )

    from aiter.ops.triton.attention.unified_attention import (
        unified_attention as _triton_unified_attention,
    )

    result = _triton_unified_attention(
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
        q_scales=q_scales,
        alibi_slopes=alibi_slopes,
        output_scale=output_scale,
        qq_bias=qq_bias,
        sinks=sinks,
        shuffled_kv_cache=shuffled_kv_cache,
        skip_reduce=skip_reduce,
        backend=backend,
    )
    if backend is None and max_seqlen_q == 1 and k.dim() == 4 and not shuffled_kv_cache:
        import torch

        # Triton yields NaN for empty KV rows; the FlyDSL path returns zero.
        tokens = torch.arange(q.shape[0], device=q.device, dtype=cu_seqlens_q.dtype)
        seq_ids = torch.searchsorted(cu_seqlens_q[1:], tokens, right=True)
        empty_rows = seqused_k[seq_ids] == 0
        out.masked_fill_(empty_rows[:, None, None], 0)
    return result
