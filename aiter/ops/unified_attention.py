# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Route unified attention between FlyDSL and Triton/Gluon backends."""

import os


def _pa_decode_env() -> bool | None:
    """Parse AITER_UNIFIED_ATTN_PA_DECODE once for every consumer.

    True for "1", False for any other value, None when unset. The router sends
    auto causal decode to FlyDSL only on True; the FlyDSL decode route is disabled
    on False.
    """
    raw = os.environ.get("AITER_UNIFIED_ATTN_PA_DECODE")
    return None if raw is None else raw == "1"


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

    ``backend=None`` tries FlyDSL then Triton/Gluon, except that causal decode
    (max_seqlen_q == 1) goes straight to Triton/Gluon unless
    AITER_UNIFIED_ATTN_PA_DECODE=1; explicit ``"flydsl"`` raises RuntimeError when
    declined. ``"triton"``/``"gluon"`` bypass FlyDSL. The Triton/Gluon fallback
    supports causal attention only. Returns ``out`` written in place; with
    ``skip_reduce`` the fallback may instead return unreduced segment buffers.

    FlyDSL contract: q/k/v are float8_e4m3fn, out is bf16/f16, and the three
    descales are float32 with one element (0-dim is widened to [1]). Q/out are
    contiguous with equal row stride; the block table is int32 with stride-1
    last dim; K/V use the exact linear or shuffled layout. AOT covers bf16 and
    f16 outputs for Hkv=4 with GQA group 4, 8 and 16 (both cache layouts, causal
    and non-causal; decode calls share the prefill keys). Other shapes JIT-compile
    on first call, and under FLYDSL_RUNTIME_RUN_ONLY=1 raise on a cache miss.

    Decode (max_seqlen_q == 1) runs on the fmha kernel as 1-token rows, or on
    FlyDSL pa_decode once ``pa_decode`` sets ``SUPPORTS_FP8_QUERY``. In that
    pa_decode route a captured call requires workspace capacity prepared outside
    capture on the same stream and head geometry; otherwise it falls back to
    Triton/Gluon with a one-time warning (``backend="flydsl"`` raises instead).
    Each captured call records a planner refresh, so sequence lengths may change
    between replays. Calls and graph replays sharing workspace must not overlap.
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

    # Auto causal decode goes to the Triton/Gluon kernel: paired A/B showed FlyDSL
    # pa_decode slower on 33/35 uniform cells (up to 1.98x), and no sync-free host
    # quantity separates the skew batches where it wins (max_seqlen_k, num_seqs and
    # table width are identical to the uniform cells). Non-causal decode has no
    # Triton path, so it stays on FlyDSL. backend="flydsl" or
    # AITER_UNIFIED_ATTN_PA_DECODE=1 sends causal decode to FlyDSL as well.
    auto_decode_to_triton = (
        backend is None
        and max_seqlen_q == 1
        and causal
        and _pa_decode_env() is not True
    )

    unified_attention_flydsl = None
    if backend == "flydsl" or (backend is None and not auto_decode_to_triton):
        # FlyDSL is absent in AITER_TRITON_ONLY and Windows installs, where this
        # op ran Triton; only an explicit backend="flydsl" may fail on it.
        try:
            from aiter.ops.flydsl.unified_attention import unified_attention_flydsl
        except ModuleNotFoundError as e:
            if e.name != "flydsl" or backend == "flydsl":
                raise
    if unified_attention_flydsl is not None:
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
