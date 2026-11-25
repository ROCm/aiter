# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Float references of what aiter.ops.rdna_attn computes, for its tests and
tuning tools: attention with every product and the softmax in fp32."""

from collections.abc import Sequence

import torch


def paged_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    seq_len: int,
    window: int = 0,
    mm_ranges: Sequence[tuple[int, int]] = (),
    block_table: torch.Tensor | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Attention of the last tokens of one sequence over its paged KV.

    Args:
        q: (M, Hq, D), the last M tokens of the sequence.
        kv: The packed HND cache (pages, Hkv, page_size, 2 * D).
        seq_len: Sequence length, the M tokens included.
        window: Sliding window in keys, 0 for full attention.
        mm_ranges: Multimodal ranges [start, end] (absolute positions,
            inclusive): a token in one also sees the range's later keys. The
            window still bounds its earlier keys.
        block_table: The sequence's pages, in order; by default the cache's
            own order.
        causal: False for every token to see every key.

    Returns:
        (M, Hq, D) in fp32.

    """
    m, hq, hkv, hd = q.shape[0], q.shape[1], kv.shape[1], q.shape[2]
    pages = kv if block_table is None else kv[block_table.long()]
    flat = pages.transpose(1, 2).reshape(-1, hkv, 2 * hd)[:seq_len].float()
    gqa = hq // hkv
    pos = torch.arange(seq_len, device=q.device).view(1, seq_len)
    lim = (seq_len - m + torch.arange(m, device=q.device)).view(m, 1)
    masked = pos > lim if causal else torch.zeros_like(pos > lim)
    for start, end in mm_ranges:
        masked &= ~((lim >= start) & (lim <= end) & (pos >= start) & (pos <= end))
    if window:
        masked |= pos < lim - (window - 1)
    out = torch.empty(m, hq, hd, device=q.device)
    # A few heads at a time: the scores of every head may not fit.
    step = max(1, min(hq, (1 << 28) // (m * seq_len)))
    for h0 in range(0, hq, step):
        heads = range(h0, min(hq, h0 + step))
        kvh = [h // gqa for h in heads]
        qf = q[:, h0 : h0 + len(heads)].float().permute(1, 0, 2)
        k = flat[:, kvh, :hd].permute(1, 0, 2)
        v = flat[:, kvh, hd:].permute(1, 0, 2)
        scores = torch.bmm(qf, k.transpose(1, 2)) * (hd**-0.5)
        scores = scores.masked_fill(masked.view(1, m, seq_len), float("-inf"))
        out[:, h0 : h0 + len(heads)] = torch.bmm(torch.softmax(scores, -1), v).permute(
            1, 0, 2
        )
    return out


def full_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention of one sequence whose tokens see every key.

    Args:
        q: (tokens, Hq, D).
        k: (tokens, Hkv, D).
        v: (tokens, Hkv, D).

    Returns:
        (tokens, Hq, D) in fp32.

    """
    m, hq, hd = q.shape
    g = hq // k.shape[1]
    # A few heads at a time: all 64 heads of a long sequence take 16 GiB.
    step = max(1, (1 << 28) // (m * m))
    return torch.cat(
        [
            torch.nn.functional.scaled_dot_product_attention(
                *(
                    x[:, h : h + step].float().transpose(0, 1)
                    for x in (q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1))
                ),
                scale=hd**-0.5,
            ).transpose(0, 1)
            for h in range(0, hq, step)
        ],
        dim=1,
    )


def row_scale(ref: torch.Tensor) -> torch.Tensor:
    """What an output element's error is judged against: its own magnitude,
    or its row's RMS if larger.

    A kernel that takes P or V in 16 bits errs in proportion to what a row
    (one token and head) adds up, not to each element: an element near zero is
    as far off as its neighbours. And a row over many keys is small, about
    1 / sqrt(keys) of one over few, so a bound that is absolute, or scaled by
    the largest output, lets a wrong row through.
    """
    return torch.maximum(ref.abs(), ref.pow(2).mean(-1, keepdim=True).sqrt())
