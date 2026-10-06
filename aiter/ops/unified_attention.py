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
    shuffled_kv_cache: bool | None = None,
    skip_reduce: bool = False,
    # backend
    backend: str | None = None,  # "triton" | "gluon" | "flydsl"
):
    """Compute paged variable-length scaled dot-product attention into ``out``.

    ``q``/``out`` are [tokens, query_heads, head_dim]. Linear K/V are
    [blocks, page_size, kv_heads, head_dim]; shuffled K/V use the vectorized
    rank-5 layouts. ``cu_seqlens_q`` contains cumulative query offsets,
    ``seqused_k`` contains KV lengths, and ``block_table`` maps logical pages.
    ``max_seqlen_q``/``max_seqlen_k`` bound lengths; ``softmax_scale`` scales
    QK logits, and ``q_descale``/``k_descale``/``v_descale`` dequantize FP8.

    ``causal`` selects bottom-right masking; ``window_size`` selects the
    attention window. Optional scales, biases, softcap, sinks, and
    ``skip_reduce`` are backend-dependent. ``shuffled_kv_cache=None`` infers
    the layout; False with rank-5 K raises ValueError.

    ``backend=None`` tries FlyDSL then Triton/Gluon; explicit ``"flydsl"``
    raises RuntimeError when declined. ``"triton"``/``"gluon"`` bypass FlyDSL.
    The Triton/Gluon fallback supports causal attention only. Returns ``out``
    written in place; with ``skip_reduce`` the fallback may instead return
    unreduced segment buffers.
    """
    if backend is not None:
        backend = backend.lower()
    if backend not in (None, "flydsl", "triton", "gluon"):
        raise ValueError(
            f"Unknown backend '{backend}', must be None, 'triton', 'gluon' or 'flydsl'"
        )

    # Normalize the cache layout once so every backend sees the same flag. A
    # rank-5 cache is the shuffled layout; an explicit False contradicts it.
    if k.dim() == 5:
        if shuffled_kv_cache is False:
            raise ValueError("shuffled_kv_cache=False contradicts a rank-5 KV cache")
        shuffled_kv_cache = True
    else:
        shuffled_kv_cache = bool(shuffled_kv_cache)

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
    return result
