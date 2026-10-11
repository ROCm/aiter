"""Route unified attention between FlyDSL and Triton/Gluon backends."""

import os

_BACKEND_OVERRIDE = os.environ.get("AITER_UNIFIED_ATTENTION_BACKEND", "auto").lower()
if _BACKEND_OVERRIDE not in ("auto", "flydsl", "triton", "gluon"):
    raise ValueError(
        "AITER_UNIFIED_ATTENTION_BACKEND must be auto, flydsl, triton or gluon"
    )


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
    """Paged variable-length attention, on FlyDSL when supported.

    ``backend=None`` tries FlyDSL, then Triton/Gluon; ``"flydsl"`` raises if
    the config is declined; ``"triton"`` and ``"gluon"`` skip FlyDSL.
    AITER_UNIFIED_ATTENTION_BACKEND=flydsl|triton|gluon replaces
    ``backend=None`` (default: auto).
    """
    if backend is None and _BACKEND_OVERRIDE != "auto":
        backend = _BACKEND_OVERRIDE
    if backend is not None:
        backend = backend.lower()
    if backend not in (None, "flydsl", "triton", "gluon"):
        raise ValueError(
            f"Unknown backend '{backend}', must be None, 'triton', 'gluon' or 'flydsl'"
        )

    # Normalize once so FlyDSL and the fallback agree. Rank 5 is the shuffled
    # layout; vLLM's plain K/V views are rank 4 despite non-contiguous strides.
    if k.dim() == 5:
        if shuffled_kv_cache is False:
            raise ValueError("shuffled_kv_cache=False contradicts a rank-5 KV cache")
        shuffled_kv_cache = True
    else:
        shuffled_kv_cache = bool(shuffled_kv_cache)

    if backend in (None, "flydsl"):
        try:
            from aiter.ops.flydsl.unified_attention_kernels import (
                flydsl_unified_attention,
            )
        except ModuleNotFoundError as error:
            # Missing FlyDSL means the backend is unavailable; any other import
            # error keeps its traceback.
            if error.name != "flydsl":
                raise
            flydsl_out = None
        else:
            flydsl_out = flydsl_unified_attention(
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

    return _triton_unified_attention(
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
