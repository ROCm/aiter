# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Route unified attention between FlyDSL and Triton/Gluon backends."""

import torch


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
    # Triton/Gluon-only arguments, appended so existing positional calls keep working.
    equal_seqlen_q: bool = False,
    attn_work: int | None = None,
    force_fp8_pv_on_fp4_kv_cache: bool = False,
    output_gate: torch.Tensor | None = None,
    output_rows: torch.Tensor | None = None,
):
    """Compute paged variable-length scaled dot-product attention into ``out``.

    ``q``/``out`` are [tokens, query_heads, head_dim]. Linear K/V are
    [blocks, page_size, kv_heads, head_dim]; shuffled K/V use the vectorized
    rank-5 layouts. ``cu_seqlens_q`` contains cumulative query offsets,
    ``seqused_k`` contains KV lengths, and ``block_table`` maps logical pages.
    ``max_seqlen_q``/``max_seqlen_k`` bound lengths; ``softmax_scale`` scales
    QK logits, and ``q_descale``/``k_descale``/``v_descale`` dequantize FP8.
    A FlyDSL CUDA graph may replay only while the tensor-held lengths remain
    within the host bounds captured in that graph. In particular, exceeding
    the captured ``max_seqlen_k`` can silently omit reducer partitions.

    ``causal`` selects bottom-right masking; ``window_size`` selects the
    attention window. Optional scales, biases, softcap, sinks, and
    ``skip_reduce`` are backend-dependent. ``shuffled_kv_cache=None`` infers
    the layout; False with rank-5 K raises ValueError.

    This is the entry point that can pick FlyDSL. ``aiter.ops.triton.unified_attention``
    and ``aiter.ops.triton.attention.unified_attention`` stay Triton/Gluon only, so
    their callers keep today's kernels; a framework gets the FlyDSL routes by
    importing ``unified_attention`` from this module instead.

    ``backend=None`` tries FlyDSL then Triton/Gluon; explicit ``"flydsl"``
    raises RuntimeError when declined. ``"triton"``/``"gluon"`` bypass FlyDSL.
    The Triton/Gluon fallback supports causal attention only. Returns ``out``
    written in place; with ``skip_reduce`` the fallback may instead return
    unreduced segment buffers.

    FlyDSL BF16 contract: the configurations validated in
    ``aiter.ops.flydsl.unified_attention_bf16`` (gfx950, BF16 linear KV, the
    MiMo V2.6 geometry); ``backend="flydsl"`` names the declined property.
    Before graph capture, warm up uncaptured on the capture stream with the
    same route-selecting host arguments; a larger workspace does not imply that
    every kernel specialization is compiled.
    When q holds ``num_seqs * max_seqlen_q`` tokens with ``max_seqlen_q <= 8``,
    they require ``cu_seqlens_q == arange(num_seqs + 1) * max_seqlen_q``, which
    is not checked on the GPU. ``AITER_UNIFIED_ATTN_FLYDSL=0`` disables
    automatic use.

    Triton/Gluon-only arguments: ``equal_seqlen_q`` (every sequence has query
    length ``max_seqlen_q``), ``attn_work`` (host-computed causal query-key pair
    count, used only by the gfx1250 router), ``force_fp8_pv_on_fp4_kv_cache``
    (NVFP4 KV cache: run PV as an FP8 WMMA), ``output_gate`` (multiply ``out`` by
    ``sigmoid(output_gate)``, [tokens, query_heads * head_dim_v]) and
    ``output_rows`` ([R] int32/int64 token indices; ``out`` has R rows). All five
    are forwarded to Triton/Gluon. FlyDSL implements none of the last three, so
    when any is set auto routing uses Triton/Gluon and ``backend="flydsl"``
    raises RuntimeError. ``equal_seqlen_q`` and ``attn_work`` do not affect routing.
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

    # FlyDSL has no gate/row-select epilogue and no FP8-PV-on-FP4 path.
    flydsl_unsupported = [
        name
        for name, used in (
            ("output_gate", output_gate is not None),
            ("output_rows", output_rows is not None),
            ("force_fp8_pv_on_fp4_kv_cache", force_fp8_pv_on_fp4_kv_cache),
        )
        if used
    ]
    if backend == "flydsl" and flydsl_unsupported:
        raise RuntimeError(
            "FlyDSL unified_attention backend does not support "
            + ", ".join(flydsl_unsupported)
        )

    unified_attention_flydsl = None
    if backend in (None, "flydsl") and not flydsl_unsupported:
        # FlyDSL is absent in AITER_TRITON_ONLY and Windows installs, where this
        # op ran Triton; only an explicit backend="flydsl" may fail on it.
        try:
            from aiter.ops.flydsl.unified_attention_bf16 import (
                unified_attention_flydsl,
            )
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
            explicit=backend == "flydsl",
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
        equal_seqlen_q=equal_seqlen_q,
        attn_work=attn_work,
        force_fp8_pv_on_fp4_kv_cache=force_fp8_pv_on_fp4_kv_cache,
        output_gate=output_gate,
        output_rows=output_rows,
    )
    return result


def release_flydsl_resources():
    """Free FlyDSL unified attention's workspaces and the kernels only it compiles.

    Only once no captured CUDA graph containing a FlyDSL call will replay, and
    never concurrently with a call; see
    ``aiter.ops.flydsl.unified_attention_bf16.release_flydsl_resources``.
    """
    import sys

    flydsl_ua = sys.modules.get("aiter.ops.flydsl.unified_attention_bf16")
    if flydsl_ua is not None:
        flydsl_ua.release_flydsl_resources()
