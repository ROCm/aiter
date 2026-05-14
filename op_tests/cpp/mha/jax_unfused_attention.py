# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Unfused JAX MHA: `gen_data`, `jax_attention`, and helpers aligned with ROCm TransformerEngine
# `benchmark_extended_small_seq/utils.py` + `fa_profiling.py` (JAX path only).
#   https://github.com/ROCm/TransformerEngine/blob/veergopu/extended_small_seq_benchmarking/benchmark_extended_small_seq/utils.py
#   https://github.com/ROCm/TransformerEngine/blob/veergopu/extended_small_seq_benchmarking/benchmark_extended_small_seq/fa_profiling.py
#
# Layout: BSHD (`utils.gen_data`), same B / s_q / s_kv / h / d as the CK cross-attn BHSD bench (`iperm=1`).

from __future__ import annotations

import jax
import jax.numpy as jnp
from einops import rearrange


def gen_data(
    dtype,
    n: int,
    seqlen_q: int,
    seqlen_kv: int,
    h: int,
    d: int,
    gqa_ratio: int = 1,
    nr_segments: int = 1,
):
    """Same tensor and segment-id construction as TE `utils.gen_data`."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (n, seqlen_q, h, d), dtype=dtype) / 8
    k = jax.random.normal(key, (n, seqlen_kv, h // gqa_ratio, d), dtype=dtype) / 8
    v = jax.random.normal(key, (n, seqlen_kv, h // gqa_ratio, d), dtype=dtype) / 8
    do = jax.random.normal(key, (n, seqlen_q, h, d), dtype=dtype) / 8
    segment_ids_q = (
        jnp.array([range(0, nr_segments)], dtype=jnp.int32)
        .repeat(seqlen_q // nr_segments, axis=1)
        .repeat(n, axis=0)
    )
    segment_ids_kv = (
        jnp.array([range(0, nr_segments)], dtype=jnp.int32)
        .repeat(seqlen_kv // nr_segments, axis=1)
        .repeat(n, axis=0)
    )
    return q, k, v, do, segment_ids_q, segment_ids_kv


def segment_ids_to_cu_seqlens(segment_ids: jnp.ndarray, max_segments: int) -> jnp.ndarray:
    """Same as TE `utils.segment_ids_to_cu_seqlens` (used by THD / Triton paths)."""
    sid = segment_ids.astype(jnp.int32)
    counts = jax.vmap(lambda x: jnp.bincount(x, length=max_segments))(sid)
    lengths = counts.reshape(-1).astype(jnp.int32)
    return jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(lengths)], axis=0)


def jax_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    segment_ids_q: jax.Array,
    segment_ids_kv: jax.Array,
    softmax_scale: float,
    is_causal: bool,
    window_size: tuple[int, int],
    layout: str = "bshd",
    nr_segments: int = 1,
) -> jax.Array:
    """Unfused attention forward (same math as TE `utils.jax_attention`). `layout` / `nr_segments` are unused (jit static slots matching `fa_profiling.py`)."""
    del layout, nr_segments
    kv_heads = key.shape[-2]
    q_heads = query.shape[-2]
    if kv_heads != q_heads:
        key = jnp.repeat(key, q_heads // kv_heads, axis=-2)
        value = jnp.repeat(value, q_heads // kv_heads, axis=-2)

    query = jnp.einsum("b n h d -> b h n d", query)
    key = jnp.einsum("b n h d -> b h n d", key)
    scores = jnp.einsum("b h n d, b h s d -> b h n s", query, key)
    scores = (scores * softmax_scale).astype(query.dtype)

    mask_causal = (
        jnp.tril(jnp.ones((scores.shape[-1], scores.shape[-1])))
        if is_causal
        else jnp.ones((scores.shape[-2], scores.shape[-1]))
    )
    mask_seq = segment_ids_q[:, :, None] == segment_ids_kv[:, None, :]
    mask = jnp.logical_and(mask_causal, mask_seq)
    if window_size[0] != -1:
        mask_window = jnp.ones((scores.shape[-2], scores.shape[-1])) - jnp.tril(
            jnp.ones((scores.shape[-2], scores.shape[-1])), k=-window_size[0]
        )
    else:
        mask_window = jnp.ones((scores.shape[-2], scores.shape[-1]))

    mask = jnp.logical_and(mask, mask_window)

    scores = jnp.where(mask[:, None, :], scores, jnp.finfo(scores.dtype).min)
    attention_weights = jax.nn.softmax(
        jnp.asarray(scores, dtype=jnp.float32), axis=-1
    )
    attention_weights = attention_weights.astype(value.dtype)

    out = jnp.einsum("b h s S, b S h d -> b h s d", attention_weights, value)
    out = rearrange(out, "b h n d -> b n h d")
    return out


def jax_attention_te_entry(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids_q: jax.Array,
    segment_ids_kv: jax.Array,
    sm_scale: float,
    causal: bool,
    window_size: int | None,
    layout: str,
    nr_segments: int,
) -> jax.Array:
    """Same window handling as TE `fa_profiling.jax_attention` before calling `utils.jax_attention`."""
    window = (-1, -1) if (window_size is None or window_size == -1) else (window_size, 0)
    return jax_attention(
        q, k, v, segment_ids_q, segment_ids_kv, sm_scale, causal, window, layout, nr_segments
    )


jax_unfused_attention = jax_attention
