# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Dict
import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

import jax
import jax.numpy as jnp
import jax_triton as jt

from utils.logger import AiterTritonLogger
from _triton_kernels.mha_onekernel_bwd_kernel import (
    _bwd_preprocess,
    bwd_kernel_causal,
    bwd_kernel_noncausal,
    _get_config,
)


_LOGGER = AiterTritonLogger()


def safe_tensor(x):
    if x is None:
        return jnp.zeros((1,), dtype=jnp.int32)
    return x


def flash_attn_onekernel_backward(
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
    # Configurations
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
        f"FLASH_ATTN_ONEKERNEL_BKWD: do={tuple(do.shape)} q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)} "
        + f"dq={tuple(dq.shape)}  dk={tuple(dk.shape)}  dv={tuple(dv.shape)}"
    )
    if dbias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")

    use_alibi, (stride_az, stride_ah) = (
        (True, alibi_slopes.stride()) if alibi_slopes is not None else (False, (0, 0))
    )

    IS_VARLEN = True if cu_seqlens_q is not None else False

    q_strides_in = jt.strides_from_shape(q.shape)
    k_strides_in = jt.strides_from_shape(k.shape)
    v_strides_in = jt.strides_from_shape(v.shape)
    o_strides_in = jt.strides_from_shape(o.shape)
    dq_strides_in = jt.strides_from_shape(dq.shape)
    dk_strides_in = jt.strides_from_shape(dk.shape)
    dv_strides_in = jt.strides_from_shape(dv.shape)
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
        _, num_k_heads = max_seqlen_k, k.shape[1]
        num_k_heads = k.shape[1]
        q_strides = (0, q_strides_in[1], q_strides_in[0], q_strides_in[2])
        k_strides = (0, k_strides_in[1], k_strides_in[0], k_strides_in[2])
        v_strides = (0, v_strides_in[1], v_strides_in[0], v_strides_in[2])
        o_strides = (0, o_strides_in[1], o_strides_in[0], o_strides_in[2])
        dq_strides = (0, dq_strides_in[1], dq_strides_in[0], dq_strides_in[2])
        dk_strides = (0, dk_strides_in[1], dk_strides_in[0], dk_strides_in[2])
        dv_strides = (0, dv_strides_in[1], dv_strides_in[0], dv_strides_in[2])
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
        dv_strides = (dv_strides_in[0], dv_strides_in[2], dv_strides_in[1], dv_strides_in[3])
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

    # Configs
    if config is None:
        config = _get_config()

    # preprocess
    # compute D(delta) = rowsum(dO*O). Note, multiplication is element-wise.
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

    delta = jt.triton_call(
        o,
        do,
        delta,
        *o_strides,
        *delta_strides,
        safe_tensor(cu_seqlens_q),
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

    seqlen = max(max_seqlen_q, max_seqlen_k)

    config_onekernel = config["onekernel"]

    metaparams = dict(
        HEAD_DIM=head_sz,
        ACTUAL_HEAD_DIM=BLOCK_D_MODEL_POW2,
        ENABLE_DROPOUT=use_dropout,
        IS_VARLEN=IS_VARLEN,
        USE_ALIBI=use_alibi,
        USE_EXP2=True,
        DEBUG_TRITON=False,
        DEBUG_TRITON_DETAIL=False,
        USE_INT64_STRIDES=USE_INT64_STRIDES,
    )

    grid = (
        num_k_heads,
        triton.cdiv(seqlen, config_onekernel["BLOCK_N1"]),
        batch,
    )

    if causal:
        kernel = bwd_kernel_causal
    else:
        kernel = bwd_kernel_noncausal

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
        *dq_strides,
        *dk_strides,
        *dv_strides,
        *delta_strides,
        *do_strides,
        *dropout_strides,
        stride_az,
        stride_ah,
        # Configurations
        sm_scale,
        num_q_heads,
        num_k_heads,
        safe_tensor(cu_seqlens_q),
        safe_tensor(cu_seqlens_k),
        max_seqlen_q,
        max_seqlen_k,
        dropout_mask,
        dropout_p,
        philox_seed,
        philox_offset,
        alibi_slopes,
        kernel=kernel,
        out_shape=out_shape,
        grid=grid,
        **metaparams,
        **config_onekernel
    )

    return dq, dk, dv


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

    # configurations
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
    dq, dk, dv = flash_attn_onekernel_backward(
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
        # Configurations
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

    dq_ref, dk_ref, dv_ref = jax_attn_bwd_reference(q,k,v)

    # numeric check (allow fp16 tolerance)
    atol = 1e-2 if MHA_DTYPE in (jnp.float16, jnp.bfloat16) else 1e-3
    max_diff = jnp.max(jnp.abs(dq - dq_ref))
        
    assert jnp.allclose(dq, dq_ref, atol=atol), \
        f"Max diff ({max_diff}) exceeds tolerance (atol={atol})"


if __name__ == "__main__":
    from absl import app
    app.run(main)
