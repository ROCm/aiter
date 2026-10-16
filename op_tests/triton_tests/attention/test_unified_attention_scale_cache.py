# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Paged FP8 scalar/per-token reads vs independent dense causal attention.

Can run as a standalone CI preflight before loading a model; no pytest or model
runtime patching is needed. Includes both 2-D prefill and 3-D split-K launches.
"""

import math

import torch

from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils.types import e4m3_dtype


def check_case(
    layout,
    mode,
    split,
    *,
    page=16,
    dim=128,
    heads=16,
    kv_heads=None,
    q_lens=None,
    k_lens=None,
):
    torch.manual_seed(125)
    device = "cuda"
    q_lens = q_lens or ([1, 1] if split else [2, 7, 19])
    k_lens = k_lens or ([2049, 2071] if split else [39, 145, 273])
    kv_heads = (1 if split else 16) if kv_heads is None else kv_heads
    pages_per_request = [math.ceil(n / page) for n in k_lens]
    num_pages = sum(pages_per_request) + 4
    kv_dtype = torch.bfloat16 if mode == "bf16" else e4m3_dtype
    dense_k = torch.randn(num_pages, page, kv_heads, dim, device=device).to(kv_dtype)
    dense_v = torch.randn_like(dense_k, dtype=torch.float32).to(kv_dtype)
    # Use physical page IDs unrelated to sequence order and share a prefix page.
    perm = torch.randperm(num_pages, device=device)
    table = torch.zeros(
        len(q_lens), max(pages_per_request) + 4, dtype=torch.int32, device=device
    )
    offset = 0
    for i, count in enumerate(pages_per_request):
        table[i, :count] = perm[offset : offset + count]
        offset += count
    table[1, 0] = table[0, 0]
    # Poison invalid final-page V slots: the kernel must mask them before PV.
    for i, length in enumerate(k_lens):
        if length % page:
            dense_v[table[i, pages_per_request[i] - 1].long(), length % page :] = float(
                "nan"
            )
    if mode == "token":
        # Strided scales exercise the operator's scale strides, not flat offsets.
        ks = (torch.rand(num_pages, kv_heads, page * 2, device=device) + 0.1)[..., ::2]
        vs = (torch.rand(num_pages, kv_heads, page * 2, device=device) + 0.2)[..., ::2]
    elif mode == "scalar":
        ks = torch.tensor([0.7], device=device)
        vs = torch.tensor([1.3], device=device)
    else:
        ks = vs = None
    pack = 16 // dense_k.element_size()
    if layout == "flash":
        kc, vc = dense_k, dense_v
    else:
        kc = (
            dense_k.view(num_pages, page, kv_heads, dim // pack, pack)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
        )
        if layout == "shuffle":
            vc = (
                dense_v.view(num_pages, page // pack, pack, kv_heads, dim)
                .permute(0, 3, 1, 4, 2)
                .contiguous()
            )
        else:
            vc = dense_v.permute(0, 2, 3, 1).contiguous()
    q = torch.randn(sum(q_lens), heads, dim, dtype=torch.bfloat16, device=device)
    cu = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    lengths = torch.tensor(k_lens, dtype=torch.int32, device=device)
    expected = []
    begin = 0
    for i, (qlen, klen) in enumerate(zip(q_lens, k_lens)):
        positions = torch.arange(klen, device=device)
        blocks, slots = table[i, positions // page].long(), positions % page
        keys = dense_k.float()[blocks, slots]
        values = dense_v.float()[blocks, slots]
        if mode == "token":
            keys *= ks[blocks, :, slots].unsqueeze(-1)
            values *= vs[blocks, :, slots].unsqueeze(-1)
            # Match the previous gather -> BF16 attention input precision.
            keys, values = keys.bfloat16().float(), values.bfloat16().float()
        elif mode == "scalar":
            keys, values = keys * ks, values * vs
        keys = keys.repeat_interleave(heads // kv_heads, dim=1)
        values = values.repeat_interleave(heads // kv_heads, dim=1)
        scores = torch.einsum(
            "qhd,khd->hqk", q[begin : begin + qlen].float(), keys
        ) / math.sqrt(dim)
        mask = (
            positions[None, :]
            > (torch.arange(qlen, device=device) + klen - qlen)[:, None]
        )
        scores.masked_fill_(mask, -float("inf"))
        expected.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), values))
        begin += qlen
    expected = torch.cat(expected).bfloat16()
    out = torch.empty_like(q)
    kwargs = {"k_scale_cache": ks, "v_scale_cache": vs} if mode == "token" else {}
    unified_attention(
        q,
        kc,
        vc,
        out,
        cu,
        max(q_lens),
        lengths,
        max(k_lens),
        1 / math.sqrt(dim),
        True,
        (-1, -1),
        table,
        0,
        None,
        ks if mode == "scalar" else None,
        vs if mode == "scalar" else None,
        shuffled_kv_cache=layout != "flash",
        backend="triton",
        **kwargs,
    )
    torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
    print(
        f"[paged-prefill-check] layout={layout} scale={mode} split_k={split} passed",
        flush=True,
    )


def test_unified_attention_scale_cache():
    for layout in ("shuffle", "nhd", "flash"):
        for mode in ("scalar", "token", "bf16"):
            for split in (False, True):
                check_case(layout, mode, split)


if __name__ == "__main__":
    test_unified_attention_scale_cache()
    # Mixed request lengths and the 255/256 query tile boundary. The broader
    # layouts/scales retain their original paths; no model-specific dispatch.
    for layout, mode, page, dim, kv_heads in (
        ("shuffle", "token", 128, 128, 1),
        ("shuffle", "token", 128, 128, 16),
        ("shuffle", "token", 64, 64, 4),
        ("shuffle", "token", 256, 256, 4),
        ("nhd", "token", 128, 128, 1),
        ("flash", "token", 16, 96, 4),
        ("shuffle", "bf16", 128, 128, 1),
        ("shuffle", "scalar", 128, 128, 16),
    ):
        check_case(
            layout,
            mode,
            False,
            page=page,
            dim=dim,
            kv_heads=kv_heads,
            q_lens=[1, 17, 255, 256, 513],
            k_lens=[257, 273, 1023, 1037, 2049],
        )
