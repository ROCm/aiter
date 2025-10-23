# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Dict
import torch
import triton
import triton.language as tl

import jax
import jax.numpy as jnp
import jax_triton as jt

from utils.logger import AiterTritonLogger
from _triton_kernels.mha_fused_bwd import (
    _bwd_preprocess,
    _bwd_kernel_dkdvdq_causal,
    _bwd_kernel_dkdvdq_noncausal,
    _get_config,
)


_LOGGER = AiterTritonLogger()


def flash_attn_fused_backward(
    # Input tensors
    do: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    o: jnp.ndarray,
    softmax_lse: jnp.ndarray,
    # Output tensors
    dq: jnp.ndarray,
    dk: jnp.ndarray,
    dv: jnp.ndarray,
    dbias: jnp.ndarray,
    # Meta-parameters
    sm_scale: float,
    alibi_slopes: Optional[jnp.ndarray],
    causal: bool,
    cu_seqlens_q: Optional[jnp.ndarray],
    cu_seqlens_k: Optional[jnp.ndarray],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    philox_seed: Optional[int] = 0,
    philox_offset: Optional[int] = 0,
    USE_INT64_STRIDES: Optional[bool] = False,
    config: Optional[Dict[str, any]] = None,
):
    _LOGGER.info(
        f"FLASH_ATTN_FUSED_BKWD: do={tuple(do.shape)} q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)} "
        + f"dq={tuple(dq.shape)}  dk={tuple(dk.shape)}  dv={tuple(dv.shape)}"
    )
    if dbias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")

    IS_VARLEN = True if cu_seqlens_q is not None else False

    q_strides_in = jt.strides_from_shape(q.shape)
    k_strides_in = jt.strides_from_shape(k.shape)
    v_strides_in = jt.strides_from_shape(v.shape)
    o_strides_in = jt.strides_from_shape(o.shape)
    dq_strides_in = jt.strides_from_shape(dq.shape)
    dk_strides_in = jt.strides_from_shape(dk.shape)
    do_strides_in = jt.strides_from_shape(do.shape)

    # get strides and shape
    if IS_VARLEN:
        # Layout for q,k,v is thd ie [total tokens, num_head, head_dim]
        batch, seqlen_q, num_q_heads, head_sz = (
            len(cu_seqlens_q) - 1,
            max_seqlen_q,
            q.shape[1],
            q.shape[2],
        )
        num_k_heads = k.shape[1]
        q_strides = (0, q_strides_in[1], q_strides_in[0], q_strides_in[2])
        k_strides = (0, k_strides_in[1], k_strides_in[0], k_strides_in[2])
        v_strides = (0, v_strides_in[1], v_strides_in[0], v_strides_in[2])
        o_strides = (0, o_strides_in[1], o_strides_in[0], o_strides_in[2])
        dq_strides = (0, dq_strides_in[1], dq_strides_in[0], dq_strides_in[2])
        dk_strides = (0, dk_strides_in[1], dk_strides_in[0], dk_strides_in[2])
        do_strides = (0, do_strides_in[1], do_strides_in[0], do_strides_in[2])
    else:
        # Layout for q,k,v is bshd ie [batch, seq_len, num_head, head_dim]
        batch, seqlen_q, num_q_heads, head_sz = q.shape
        num_k_heads = k.shape[2]
        q_strides = (q_strides_in[0], q_strides_in[2], q_strides_in[1], q_strides_in[3])
        k_strides = (k_strides_in[0], k_strides_in[2], k_strides_in[1], k_strides_in[3])
        v_strides = (v_strides_in[0], v_strides_in[2], v_strides_in[1], v_strides_in[3])
        o_strides = (o_strides_in[0], o_strides_in[2], o_strides_in[1], o_strides_in[3])
        dq_strides = (dq_strides_in[0], dq_strides_in[2], dq_strides_in[1], dq_strides_in[3])
        dk_strides = (dk_strides_in[0], dk_strides_in[2], dk_strides_in[1], dk_strides_in[3])
        do_strides = (do_strides_in[0], do_strides_in[2], do_strides_in[1], do_strides_in[3])

    # BLOCK_D_MODEL, BLOCK_D_MODEL_POW2
    # padding for head_dim. Power of 2 or 16
    BLOCK_D_MODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_D_MODEL_POW2 = max(BLOCK_D_MODEL_POW2, 16)

    # init delta
    delta = jnp.zeros_like(softmax_lse)
    delta_strides_in = jt.strides_from_shape(delta.shape)
    if IS_VARLEN:
        # [total_tokens, num_q_heads, seqlen_q]
        delta_strides = (0, delta_strides_in[1], delta_strides_in[0])
    else:
        # [batch, num_q_heads, seqlen_q]
        delta_strides = delta_strides_in

    # preprocess
    # compute D(delta) = rowsum(dO*O). Note, multiplication is element-wise.
    if config is None:
        config = _get_config()

    pre_grid = (
        triton.cdiv(max_seqlen_q, config["preprocess_kernel"]["PRE_BLOCK"]),
        batch,
        num_q_heads,
    )
    out_shape = jax.ShapeDtypeStruct(shape=delta.shape, dtype=delta.dtype)

    metaparams_pre = dict(
        BLOCK_M=config["preprocess_kernel"]["PRE_BLOCK"],
        BLOCK_D_MODEL=head_sz,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
        IS_VARLEN=IS_VARLEN,
    )
    print(o_strides)
    print(delta_strides)

    delta = jt.triton_call(
        o,
        do,
        delta,
        *o_strides,
        *delta_strides,
        cu_seqlens_q,
        max_seqlen_q,
        kernel=_bwd_preprocess,
        grid=pre_grid,
        out_shape=out_shape,
        **metaparams_pre
    )

    # dropout_mask
    use_dropout = dropout_p > 0.0
    if use_dropout:
        dropout_mask = jnp.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            dtype=jnp.float32,
        )
        dropout_strides = jt.strides_from_shape(dropout_mask.shape)
    else:
        dropout_mask = None
        dropout_strides = (0, 0, 0, 0)

    # Fuses dk,dv and dq computations into one kernel using atomics
    if BLOCK_D_MODEL_POW2 > 160 or q.dtype == jnp.float32:
        config_dkdvdq = config["dkdvdq_kernel_N64"]
    else:
        config_dkdvdq = config["dkdvdq_kernel_N128"]

    num_k_pids = (max_seqlen_k + config_dkdvdq["BLOCK_N"] - 1) // config_dkdvdq["BLOCK_N"]

    metaparams = dict(
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        BATCH=batch,
        NUM_K_PIDS=num_k_pids,
        BLOCK_D_MODEL=head_sz,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2,
        ENABLE_DROPOUT=use_dropout,
        IS_VARLEN=IS_VARLEN,
        USE_INT64_STRIDES=USE_INT64_STRIDES,
        NUM_XCD=8,
    )

    if causal:
        kernel = _bwd_kernel_dkdvdq_causal
        grid_dkdvdq = (batch * num_q_heads * num_k_pids,)
    else:
        # in non causal inner loop over grouped q heads
        kernel = _bwd_kernel_dkdvdq_noncausal
        grid_dkdvdq = (batch * num_k_heads * num_k_pids,)

    out_shape = [
        jax.ShapeDtypeStruct(shape=dk.shape, dtype=dk.dtype),
        jax.ShapeDtypeStruct(shape=dv.shape, dtype=dv.dtype),
        jax.ShapeDtypeStruct(shape=dq.shape, dtype=dq.dtype),        
    ]

    dq, dk, dv = jt.triton_call(
        # Input tensors
        q, k, v,
        do,
        softmax_lse, 
        delta,
        # Output tensors
        dq, dk, dv,
        # Strides
        *q_strides,
        *k_strides,
        *v_strides,
        *dk_strides,
        *dq_strides,
        *delta_strides,
        *do_strides,
        *dropout_strides,
        # Meta-parameters
        sm_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_mask,
        dropout_p,
        philox_seed,
        philox_offset,
        kernel=kernel,
        out_shape=out_shape,
        grid=grid_dkdvdq,
        **metaparams,
        **config_dkdvdq
    )

    return dq, dk, dv


def mha_fwd_reference(q, k, v, causal=True, sm_scale=None):
    """Reference forward using JAX numpy to produce out and softmax_lse.

    Returns:
        out: [B, Lq, H, D]  (same layout as q/k/v in your code: b, s, h, d)
        softmax_lse: [B, H, Lq]  (logsumexp per query position)
    """
    B, Lq, H, D = q.shape
    max_seqlen_q = SEQ_LEN
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # compute logits: [B, H, Lq, Lk]
    # q: [B, Lq, H, D]  -> transpose to [B, H, Lq, D]
    qT = jnp.transpose(q, (0, 2, 1, 3))
    kT = jnp.transpose(k, (0, 2, 1, 3))

    # einsum for clarity: [B, H, Lq, D] @ [B, H, D, Lk] -> [B, H, Lq, Lk]
    logits = jnp.einsum("bhqd,bhkd->bhqk", qT, kT) * sm_scale

    # causal mask: allow j <= i
    if causal:
        # mask shape [Lq, Lk]
        mask = jnp.tril(jnp.ones((Lq, max_seqlen_q), dtype=logits.dtype))
        logits = jnp.where(mask[None, None, :, :] == 1, logits, -jnp.inf)

    # compute softmax weights and out
    # jax.nn.softmax handles numeric stability internally (uses logsumexp)
    weights = jax.nn.softmax(logits, axis=-1)  # shape [B, H, Lq Lk]
    vT = jnp.transpose(v, (0, 2, 1, 3))  # [B, H, Lk, D]
    outT = jnp.einsum("bhqk,bhkd->bhqd", weights, vT)  # [B, H, Lq, D]

    # transpose back to [B, Lq, H, D] to match your function's expected layout
    out = jnp.transpose(outT, (0, 2, 1, 3))

    # softmax_lse = logsumexp(logits, axis=-1), shape [B, H, Lq]
    softmax_lse = jax.scipy.special.logsumexp(logits, axis=-1)

    return out, softmax_lse


# MHA shape
BATCH_SIZE: int = 2
SEQ_LEN: int = 1024
NUM_HEADS: int = 64
HEAD_SIZE: int = 128

MHA_SHAPE: tuple[int, int, int, int] = (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_SIZE)
assert all(dim > 0 for dim in MHA_SHAPE)

# MHA dtype
MHA_DTYPE = jnp.float32

RNG_SEED = 42


def main(unused_argv):
    # generate random key
    key = jax.random.PRNGKey(RNG_SEED)
    q_key, k_key, v_key, do_key = jax.random.split(key, 4)

    # fwd inputs
    q = jax.random.normal(q_key, MHA_SHAPE, dtype=MHA_DTYPE)
    k = jax.random.normal(k_key, MHA_SHAPE, dtype=MHA_DTYPE)
    v = jax.random.normal(v_key, MHA_SHAPE, dtype=MHA_DTYPE)

    # metaparams
    sm_scale = HEAD_SIZE ** -0.5
    causal = True
    alibi_slopes = None
    cu_seqlens_q = cu_seqlens_k = None
    max_seqlen_q = max_seqlen_k = SEQ_LEN
    dropout_p = 0.2

    # save fwd outputs for bwd
    o, softmax_lse = mha_fwd_reference(q, k, v, sm_scale=sm_scale, causal=causal)
    do = jax.random.normal(do_key, o.shape, dtype=MHA_DTYPE)

    # bwd outputs
    dq = jnp.zeros_like(q)
    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    # jax-triton mha fused bwd
    dq, dk, dv = flash_attn_fused_backward(
        # Input tensors
        do=do,
        q=q,
        k=k,
        v=v,
        o=o,
        softmax_lse=softmax_lse,
        # Output tensors
        dq=dq,
        dk=dk,
        dv=dv,
        dbias=None,
        # Meta-parameters
        sm_scale=sm_scale,
        causal=causal,
        alibi_slopes=alibi_slopes,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        # philox_seed=philox_seed,
        # philox_offset=philox_offset,
        # USE_INT64_STRIDES=False,
        # config=config,
    )


if __name__ == "__main__":
    from absl import app
    app.run(main)
