"""CK adapter: translates flash_attn_2_cuda calling convention to aiter.ops.mha."""

import torch
from typing import Optional, Union

from aiter.ops.mha import mha_fwd, mha_varlen_fwd, mha_bwd, mha_varlen_bwd


def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    return_softmax: bool,
    gen_: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    # aiter CK does not support softcap
    if softcap != 0.0:
        raise NotImplementedError(
            "softcap is not supported in the CK adapter (expected 0.0)."
        )

    out_tensor, softmax_lse, S_dmask, rng_state = mha_fwd(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        is_causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        sink_size=0,
        return_softmax_lse=True,
        return_dropout_randval=(dropout_p > 0.0 and return_softmax),
        out=out,
        alibi_slopes=alibi_slopes,
        gen=gen_,
    )

    # CK returns S_dmask as empty tensor when not needed; FA expects None
    if S_dmask.numel() == 0:
        S_dmask = None

    return out_tensor, softmax_lse, S_dmask, rng_state


def bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    deterministic: bool,
    gen_: Optional[torch.Tensor] = None,
    rng_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if softcap != 0.0:
        raise NotImplementedError(
            "softcap is not supported in the CK adapter (expected 0.0)."
        )

    return mha_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dropout_p,
        softmax_scale,
        is_causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        deterministic=deterministic,
        dq=dq,
        dk=dk,
        dv=dv,
        alibi_slopes=alibi_slopes,
        rng_state=rng_state,
        gen=gen_,
    )


def varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: Optional[torch.Tensor],
    leftpad_k: Optional[torch.Tensor],
    block_table_: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    return_softmax: bool,
    gen_: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if softcap != 0.0:
        raise NotImplementedError(
            "softcap is not supported in the CK adapter (expected 0.0)."
        )
    if leftpad_k is not None:
        raise NotImplementedError("leftpad_k is not supported in the CK adapter.")
    if seqused_k is not None:
        raise NotImplementedError("seqused_k is not supported in the CK adapter.")

    out_tensor, softmax_lse, S_dmask, rng_state = mha_varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q=0,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        logits_soft_cap=0.0,
        zero_tensors=zero_tensors,
        is_causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        sink_size=0,
        return_softmax_lse=True,
        return_dropout_randval=(dropout_p > 0.0 and return_softmax),
        out=out,
        block_table=block_table_,
        alibi_slopes=alibi_slopes,
        gen=gen_,
    )

    if S_dmask.numel() == 0:
        S_dmask = None

    return out_tensor, softmax_lse, S_dmask, rng_state


def varlen_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    deterministic: bool,
    gen_: Optional[torch.Tensor] = None,
    rng_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if softcap != 0.0:
        raise NotImplementedError(
            "softcap is not supported in the CK adapter (expected 0.0)."
        )

    return mha_varlen_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors=zero_tensors,
        is_causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        deterministic=deterministic,
        dq=dq,
        dk=dk,
        dv=dv,
        alibi_slopes=alibi_slopes,
        rng_state=rng_state,
        gen=gen_,
    )


def fwd_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    cache_seqlens: Optional[Union[int, torch.Tensor]],
    rotary_cos: Optional[torch.Tensor],
    rotary_sin: Optional[torch.Tensor],
    cache_batch_idx: Optional[torch.Tensor],
    cache_leftpad: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    rotary_interleaved: bool,
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported in the CK adapter.")
    if k is not None or v is not None:
        raise NotImplementedError(
            "Appending new K/V is not supported in the CK adapter."
        )
    if rotary_cos is not None or rotary_sin is not None:
        raise NotImplementedError(
            "Rotary embeddings are not supported in the CK adapter."
        )
    if cache_batch_idx is not None:
        raise NotImplementedError("cache_batch_idx is not supported in the CK adapter.")
    if cache_leftpad is not None:
        raise NotImplementedError("cache_leftpad is not supported in the CK adapter.")

    if block_table is not None:
        raise NotImplementedError(
            "Paged KV (block_table) is not supported in the CK adapter."
        )

    batch = q.shape[0]

    # Convert cache_seqlens to cu_seqlens_kv
    if cache_seqlens is None:
        cu_seqlens_kv = None
    elif isinstance(cache_seqlens, int):
        cu_seqlens_kv = torch.arange(
            0,
            (batch + 1) * cache_seqlens,
            cache_seqlens,
            dtype=torch.int32,
            device=q.device,
        )
    else:
        cu_seqlens_kv = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
        cu_seqlens_kv[1:] = cache_seqlens.to(torch.int32).cumsum(0)

    out_tensor, softmax_lse, _, _ = mha_fwd(
        q,
        k_cache,
        v_cache,
        0.0,
        softmax_scale,
        is_causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        sink_size=0,
        return_softmax_lse=True,
        return_dropout_randval=False,
        cu_seqlens_kv=cu_seqlens_kv,
        out=out,
        alibi_slopes=alibi_slopes,
    )

    return out_tensor, softmax_lse
