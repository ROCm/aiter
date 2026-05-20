# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# JAX unfused reference: THD + cu_seqlens (scenarios 1–2 fwd, matches CK group mode).

from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def block_until_ready_tree(x) -> None:
    jax.tree_util.tree_map(lambda y: y.block_until_ready(), x)


def sm_scale_ck(hdim: int) -> float:
    return float(1.0 / jnp.sqrt(jnp.array(hdim, dtype=jnp.float32)))


def logical_lens_to_cu_seqlens(logical_lens: list[int]) -> jnp.ndarray:
    return jnp.asarray([0, *np.cumsum(logical_lens, dtype=np.int32)], dtype=jnp.int32)


def jax_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    segment_ids_q: jax.Array,
    segment_ids_kv: jax.Array,
    softmax_scale: float,
    is_causal: bool,
    seqlens_q: jax.Array | None = None,
    seqlens_k: jax.Array | None = None,
) -> jax.Array:
    """Unfused BSHD attention; optional per-row seqlens mask padded tiles."""
    kv_heads = key.shape[-2]
    q_heads = query.shape[-2]
    if kv_heads != q_heads:
        key = jnp.repeat(key, q_heads // kv_heads, axis=-2)
        value = jnp.repeat(value, q_heads // kv_heads, axis=-2)

    query = jnp.einsum("b n h d -> b h n d", query)
    key = jnp.einsum("b n h d -> b h n d", key)
    scores = jnp.einsum("b h n d, b h s d -> b h n s", query, key)
    scores = (scores * softmax_scale).astype(query.dtype)

    if is_causal:
        mask = jnp.tril(jnp.ones((scores.shape[-2], scores.shape[-1]), dtype=jnp.bool_))
    else:
        mask = jnp.ones((scores.shape[-2], scores.shape[-1]), dtype=jnp.bool_)

    mask = jnp.logical_and(mask, segment_ids_q[:, :, None] == segment_ids_kv[:, None, :])
    if seqlens_q is not None:
        n_q, n_k = scores.shape[-2], scores.shape[-1]
        mask = jnp.logical_and(mask, jnp.arange(n_q)[None, :, None] < seqlens_q[:, None, None])
        mask = jnp.logical_and(mask, jnp.arange(n_k)[None, None, :] < seqlens_k[:, None, None])

    # mask (B, n_q, n_k) → (B, 1, n_q, n_k) for scores (B, H, n_q, n_k)
    scores = jnp.where(mask[:, None, :, :], scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(jnp.asarray(scores, dtype=jnp.float32), axis=-1).astype(value.dtype)
    out = jnp.einsum("b h s S, b S h d -> b h s d", weights, value)
    return jnp.transpose(out, (0, 2, 1, 3))


def _thd_to_bshd(x_thd: jax.Array, cu_seqlens: jax.Array, max_seqlen: int) -> jax.Array:
    nbatch = cu_seqlens.shape[0] - 1

    def one(i: jax.Array) -> jax.Array:
        return jax.lax.dynamic_slice_in_dim(x_thd, cu_seqlens[i], max_seqlen, axis=0)

    return jax.vmap(one)(jnp.arange(nbatch, dtype=jnp.int32))


@partial(jax.jit, static_argnames=("max_sq", "max_sk", "softmax_scale", "is_causal"))
def jax_attention_thd(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    cu_seqlens_q: jax.Array,
    cu_seqlens_k: jax.Array,
    max_sq: int,
    max_sk: int,
    softmax_scale: float,
    is_causal: bool,
) -> jax.Array:
    sq_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    sk_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    nbatch = cu_seqlens_q.shape[0] - 1
    q_b = _thd_to_bshd(query, cu_seqlens_q, max_sq)
    k_b = _thd_to_bshd(key, cu_seqlens_k, max_sk)
    v_b = _thd_to_bshd(value, cu_seqlens_k, max_sk)
    seg = jnp.arange(nbatch, dtype=jnp.int32)[:, None]
    return jax_attention(
        q_b,
        k_b,
        v_b,
        jnp.repeat(seg, max_sq, axis=1),
        jnp.repeat(seg, max_sk, axis=1),
        softmax_scale,
        is_causal,
        sq_lens,
        sk_lens,
    )


@partial(jax.jit, static_argnames=("max_sk", "softmax_scale", "is_causal"))
def jax_cross_attention_thd(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    cu_seqlens_k: jax.Array,
    max_sk: int,
    softmax_scale: float,
    is_causal: bool,
) -> jax.Array:
    sk_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    nbatch = query.shape[0]
    k_b = _thd_to_bshd(key, cu_seqlens_k, max_sk)
    v_b = _thd_to_bshd(value, cu_seqlens_k, max_sk)
    seg_q = jnp.arange(nbatch, dtype=jnp.int32)[:, None]
    seg_k = jnp.arange(nbatch, dtype=jnp.int32)[:, None].repeat(max_sk, axis=1)
    return jax_attention(
        query,
        k_b,
        v_b,
        seg_q,
        seg_k,
        softmax_scale,
        is_causal,
        jnp.ones((nbatch,), dtype=jnp.int32),
        sk_lens,
    )


def gen_thd_varlen(dtype, logical_sq, logical_sk, nheads, hdim, gqa_ratio=1):
    key = jax.random.PRNGKey(0)
    tq, tk = sum(logical_sq), sum(logical_sk)
    q = jax.random.normal(key, (tq, nheads, hdim), dtype=dtype) / 8
    k = jax.random.normal(key, (tk, nheads // gqa_ratio, hdim), dtype=dtype) / 8
    v = jax.random.normal(key, (tk, nheads // gqa_ratio, hdim), dtype=dtype) / 8
    return q, k, v, logical_lens_to_cu_seqlens(logical_sq), logical_lens_to_cu_seqlens(logical_sk)


def gen_thd_cross_attn(dtype, batch, logical_sk, nheads, hdim, gqa_ratio=1):
    key = jax.random.PRNGKey(0)
    tk = sum(logical_sk)
    q = jax.random.normal(key, (batch, 1, nheads, hdim), dtype=dtype) / 8
    k = jax.random.normal(key, (tk, nheads // gqa_ratio, hdim), dtype=dtype) / 8
    v = jax.random.normal(key, (tk, nheads // gqa_ratio, hdim), dtype=dtype) / 8
    return q, k, v, logical_lens_to_cu_seqlens(logical_sk)


def gen_dense_self_attn(dtype, batch, seqlen, nheads, hdim, gqa_ratio=1):
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (batch, seqlen, nheads, hdim), dtype=dtype) / 8
    k = jax.random.normal(key, (batch, seqlen, nheads // gqa_ratio, hdim), dtype=dtype) / 8
    v = jax.random.normal(key, (batch, seqlen, nheads // gqa_ratio, hdim), dtype=dtype) / 8
    seg = jnp.arange(batch, dtype=jnp.int32)[:, None].repeat(seqlen, axis=1)
    return q, k, v, seg, seg


def gen_dense_batch_uniform(dtype, batch, sq_pad, sk_pad, nheads, hdim, gqa_ratio=1):
    """Scenarios 1–2 bwd: dense batch shapes (matches CK bwd)."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (batch, sq_pad, nheads, hdim), dtype=dtype) / 8
    k = jax.random.normal(key, (batch, sk_pad, nheads // gqa_ratio, hdim), dtype=dtype) / 8
    v = jax.random.normal(key, (batch, sk_pad, nheads // gqa_ratio, hdim), dtype=dtype) / 8
    seg = jnp.arange(batch, dtype=jnp.int32)[:, None]
    return q, k, v, seg.repeat(sq_pad, axis=1), seg.repeat(sk_pad, axis=1)


def bench_mean_ms(jit_fn, args: tuple, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        block_until_ready_tree(jit_fn(*args))
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        block_until_ready_tree(jit_fn(*args))
        times.append((time.perf_counter_ns() - t0) / 1e6)
    return sum(times) / len(times)
