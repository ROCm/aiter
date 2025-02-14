# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import math
from .bert_padding import pad_input, unpad_input
from einops import rearrange, repeat
import torch
import torch.nn.functional as F


def ck_randval_to_dropout_mask(randval, p):
    # If p = 0.3, randval in 255 * (0.7, 1.0] will be dropout
    # randval in 255 * [0, 0.7] will be kept
    # If return dropout_mask >=0, value will be kept
    return math.floor(255.0 * (1 - p)) - randval.to(torch.float32)


def convert_flash_attn_S_to_softmax(
    S,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    head_dim,
    is_dropout,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q_rounded)
        key_padding_mask: (batch_size, seqlen_k_rounded)
    """
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
    S_converted = S
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            S.device,
        )
        local_mask = F.pad(
            local_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        S_converted = S_converted.masked_fill(local_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = (
        query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
        S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
    S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
    return S_converted[:, :, :seqlen_q, :seqlen_k]


def pad_rearrange_dropout_mask_hts_to_bhss(S_dmask, cu_seqlens_q, seqlen_q_rounded, seqlen_k_rounded):
    """ pad + rearrange [nheads, total_q, max_seqlen_k] into [b, nheads, seqlen_q_rounded, seqlen_k_rounded]
    Arguments:
        S_dmask: (nheads, total_q, max_seqlen_k)
        cu_seqlens_q: (b + 1)
    Output:
        S_dmask: (b, nheads, seqlen_q_rounded, seqlen_k_rounded)
    """
    batch_size = cu_seqlens_q.numel() - 1
    seqlens_q = torch.roll(cu_seqlens_q, shifts = -1) - cu_seqlens_q
    seqlens_q = seqlens_q[0:batch_size].tolist()
    S_dmask = torch.split(S_dmask, seqlens_q, dim=1)
    # [(nheads, seqlen_q0, max_seqlen_k), (nheads, seqlen_q1, max_seqlen_k), ..., (nheads, seqlen_qb, max_seqlen_k)]
    masks = ()
    for mask in S_dmask:
        # (nheads, seqlen_qi, max_seqlen_k) -> (nheads, seqlen_q_rounded, seqlen_k_rounded)
        mask = F.pad(mask, (0, seqlen_k_rounded - mask.shape[2], 0, seqlen_q_rounded - mask.shape[1], 0, 0)).unsqueeze(1)
        masks = masks + (mask, )
    S_dmask = torch.cat(masks, dim=1)

    S_dmask = S_dmask.transpose(0, 1)
    return S_dmask


def attn_bias_from_alibi_slopes(
    slopes, seqlen_q, seqlen_k, query_padding_mask=None, key_padding_mask=None, causal=False, key_leftpad=None
):
    batch, nheads = slopes.shape
    device = slopes.device
    slopes = rearrange(slopes, "b h -> b h 1 1")
    if causal:
        return torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32) * slopes
    else:
        row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        if key_leftpad is not None:
            key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
            col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
            col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        relative_pos = torch.abs(row_idx + sk - sq - col_idx)
        return -slopes * relative_pos.to(dtype=slopes.dtype)


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask


def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)
