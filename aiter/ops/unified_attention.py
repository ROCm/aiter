# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

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
    """Compute paged variable-length attention into ``out``; see the Triton op.

    ``backend=None`` runs FlyDSL for validated configurations and Triton/Gluon
    otherwise. ``"flydsl"`` raises RuntimeError naming the unsupported property
    instead of falling back. ``"triton"``/``"gluon"`` bypass FlyDSL. The
    validated FlyDSL configurations are listed in
    ``aiter.ops.flydsl.unified_attention``. When q holds
    ``num_seqs * max_seqlen_q`` tokens with ``max_seqlen_q <= 8``, they require
    ``cu_seqlens_q == arange(num_seqs + 1) * max_seqlen_q``, which is not
    checked on the GPU. ``AITER_UNIFIED_ATTN_FLYDSL=0`` disables automatic use.

    ``shuffled_kv_cache=None`` infers the layout from the K rank (5D shuffled,
    4D linear); False with a 5D cache raises ValueError.
    """
    if backend is not None:
        backend = backend.lower()
    if backend not in (None, "flydsl", "triton", "gluon"):
        raise ValueError(
            f"Unknown backend '{backend}', must be None, 'triton', 'gluon' or 'flydsl'"
        )
    if k.dim() == 5:
        if shuffled_kv_cache is False:
            raise ValueError("shuffled_kv_cache=False contradicts a rank-5 KV cache")
        shuffled_kv_cache = True
    else:
        shuffled_kv_cache = bool(shuffled_kv_cache)

    args = (
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
    )
    kwargs = {
        "q_scales": q_scales,
        "alibi_slopes": alibi_slopes,
        "output_scale": output_scale,
        "qq_bias": qq_bias,
        "sinks": sinks,
        "shuffled_kv_cache": shuffled_kv_cache,
        "skip_reduce": skip_reduce,
    }
    if backend in (None, "flydsl"):
        flydsl = None
        try:
            from aiter.ops.flydsl import unified_attention as flydsl
        except ModuleNotFoundError as e:
            # FlyDSL is absent in AITER_TRITON_ONLY and Windows installs.
            if e.name != "flydsl" or backend == "flydsl":
                raise
        if flydsl is not None:
            result = flydsl.unified_attention_flydsl(
                *args, **kwargs, explicit=backend == "flydsl"
            )
            if result is not None:
                return result
            if backend == "flydsl":
                reason = flydsl.flydsl_decline_reason(*args, **kwargs)
                raise RuntimeError(
                    "FlyDSL unified_attention does not support this configuration: "
                    f"{reason}"
                )

    from aiter.ops.triton.attention.unified_attention import (
        unified_attention as _triton_unified_attention,
    )

    return _triton_unified_attention(*args, **kwargs, backend=backend)
