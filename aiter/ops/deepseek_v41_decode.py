# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1 decode layer, as two routed ops (contract v1).

One decoder layer of a DeepSeek-V4.1 decode step (up to 48 rows on one TP rank)
is two calls, in this order, on one stream:

``deepseek_v41_decode_attn_pre``
    the attention seam (mHC fold of the previous sublayer's output, the mix
    projection, Sinkhorn gates, collapse, attn_norm), ``wqkv_a``, q/kv norms,
    ``wq_b``, RoPE, the token's window KV row written into the KV cache, and
    (FULL / REINDEX layers) the indexer's query and head weights.

``deepseek_v41_decode_layer_post``
    sparse decode attention over the given keys (selected compressed rows, then
    the window ring), inverse RoPE, ``wo_a`` / ``wo_b`` and their TP all-reduce,
    the FFN seam and ``ffn_norm``, then the MoE (router, routed MXFP4 experts,
    shared FP8 expert) and its TP all-reduce.

AITER owns these contracts. The full sheet -- shapes, dtypes, weight layouts,
mutation, tolerance, communication and capture rules -- is
``deepseek_v41_decode.md`` beside this file. The native implementation is the
pure-torch reference in ``deepseek_v41_decode_ref.py``: correct, not fast. A
provider (e.g. a fused monokernel) may serve any call and must decline any call
it cannot serve exactly as specified.

Communication follows ``sp_qkv_fused``: ``group`` is the caller's TP group and is
execution context, ``stream`` the caller's stream. A provider owns whatever
symmetric/peer memory it needs for the in-op all-reduces; the op never allocates
caller-visible workspace.
"""

from __future__ import annotations

from ..plugin import route as _route

__all__ = [
    "deepseek_v41_decode_attn_pre",
    "deepseek_v41_decode_layer_post",
]


def _reference():
    # Imported on first call: the reference imports only torch, but keeping it
    # lazy keeps this module (and the contract) importable everywhere.
    from . import deepseek_v41_decode_ref

    return deepseek_v41_decode_ref


@_route(op="deepseek_v41_decode_attn_pre", version=1)
def deepseek_v41_decode_attn_pre(
    residual,
    pre_mix,
    hc_fn,
    hc_scale,
    hc_base,
    attn_norm_weight,
    wqkv_a,
    wqkv_a_scale,
    q_norm_weight,
    kv_norm_weight,
    wq_b,
    wq_b_scale,
    cos_cache,
    sin_cache,
    positions,
    kv_cache,
    kv_slots,
    pre_out,
    post_out,
    comb_out,
    q_out,
    *,
    layer_index,
    sublayer_output=None,
    post_mix=None,
    comb_mix=None,
    residual_out=None,
    normed_out=None,
    q_latent_out=None,
    q_latent_scale_out=None,
    index_wq_b=None,
    index_wq_b_scale=None,
    index_weights_proj=None,
    index_q_out=None,
    index_q_scale_out=None,
    index_weights_out=None,
    aux_out=None,
    eps=1e-20,
    hc_eps=1e-6,
    group=None,
    stream=None,
) -> None:
    """K1 of a DeepSeek-V4.1 decode layer: the attention seam to the rotated query.

    Writes ``pre_out``/``post_out``/``comb_out`` (this seam's gates), ``q_out``,
    the window row of every token whose ``kv_slots`` entry is not -1 into
    ``kv_cache``, and the optional outputs that are given. Folding
    (``sublayer_output``/``post_mix``/``comb_mix``, all or none) also writes
    ``residual_out``; without a fold the seam's residual is ``residual`` itself.
    No TP communication happens here; ``group`` names the TP group the shard
    belongs to (a provider may communicate inside the op). Returns ``None``.
    See ``deepseek_v41_decode.md`` sections 2-4.
    """

    _reference().attn_pre(
        residual,
        pre_mix,
        hc_fn,
        hc_scale,
        hc_base,
        attn_norm_weight,
        wqkv_a,
        wqkv_a_scale,
        q_norm_weight,
        kv_norm_weight,
        wq_b,
        wq_b_scale,
        cos_cache,
        sin_cache,
        positions,
        kv_cache,
        kv_slots,
        pre_out,
        post_out,
        comb_out,
        q_out,
        layer_index=layer_index,
        sublayer_output=sublayer_output,
        post_mix=post_mix,
        comb_mix=comb_mix,
        residual_out=residual_out,
        normed_out=normed_out,
        q_latent_out=q_latent_out,
        q_latent_scale_out=q_latent_scale_out,
        index_wq_b=index_wq_b,
        index_wq_b_scale=index_wq_b_scale,
        index_weights_proj=index_weights_proj,
        index_q_out=index_q_out,
        index_q_scale_out=index_q_scale_out,
        index_weights_out=index_weights_out,
        aux_out=aux_out,
        eps=eps,
        hc_eps=hc_eps,
        group=group,
        stream=stream,
    )


@_route(op="deepseek_v41_decode_layer_post", version=1)
def deepseek_v41_decode_layer_post(
    q,
    kv_cache,
    topk_rows,
    topk_count,
    window_meta,
    attn_sink,
    positions,
    cos_cache,
    sin_cache,
    wo_a,
    wo_a_scale,
    wo_b,
    wo_b_scale,
    residual,
    pre_mix,
    post_mix,
    comb_mix,
    hc_fn,
    hc_scale,
    hc_base,
    ffn_norm_weight,
    gate_weight,
    gate_bias,
    w13,
    w13_scale,
    w2,
    w2_scale,
    shared_w13,
    shared_w13_scale,
    shared_w2,
    shared_w2_scale,
    residual_out,
    pre_out,
    post_out,
    comb_out,
    ffn_out,
    *,
    softmax_scale,
    ring_slots,
    layer_index,
    ffn_normed_out=None,
    top_k=6,
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    eps=1e-20,
    hc_eps=1e-6,
    group=None,
    stream=None,
) -> None:
    """K2 of a DeepSeek-V4.1 decode layer: sparse attention to the reduced MoE output.

    Reads K1's ``q`` and gates, the KV cache rows named by ``topk_rows`` /
    ``topk_count`` and ``window_meta``, and writes the FFN seam's
    ``residual_out``/``pre_out``/``post_out``/``comb_out``, ``ffn_out`` (the MoE
    output summed over ``group``: the next layer's ``sublayer_output``) and, when
    given, ``ffn_normed_out``. Both TP all-reduces (after ``wo_b`` and after the
    MoE) happen inside the op, over ``group``. Returns ``None``. See
    ``deepseek_v41_decode.md`` sections 2, 5 and 6.
    """

    _reference().layer_post(
        q,
        kv_cache,
        topk_rows,
        topk_count,
        window_meta,
        attn_sink,
        positions,
        cos_cache,
        sin_cache,
        wo_a,
        wo_a_scale,
        wo_b,
        wo_b_scale,
        residual,
        pre_mix,
        post_mix,
        comb_mix,
        hc_fn,
        hc_scale,
        hc_base,
        ffn_norm_weight,
        gate_weight,
        gate_bias,
        w13,
        w13_scale,
        w2,
        w2_scale,
        shared_w13,
        shared_w13_scale,
        shared_w2,
        shared_w2_scale,
        residual_out,
        pre_out,
        post_out,
        comb_out,
        ffn_out,
        softmax_scale=softmax_scale,
        ring_slots=ring_slots,
        layer_index=layer_index,
        ffn_normed_out=ffn_normed_out,
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
        swiglu_limit=swiglu_limit,
        eps=eps,
        hc_eps=hc_eps,
        group=group,
        stream=stream,
    )
