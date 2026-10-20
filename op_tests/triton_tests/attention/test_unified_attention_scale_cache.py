# SPDX-License-Identifier: MIT
"""Check native FP8 scale caches through the public Gluon wrapper."""

import math

import pytest
import torch


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
    backend="triton",
    device="cuda",
    reference_only=False,
):
    torch.manual_seed(125)
    q_lens = q_lens or ([1, 1] if split else [2, 7, 19])
    k_lens = k_lens or ([2049, 2071] if split else [39, 145, 273])
    kv_heads = (1 if split else 16) if kv_heads is None else kv_heads
    pages_per_request = [math.ceil(n / page) for n in k_lens]
    num_pages = sum(pages_per_request) + 4
    if str(device) == "cpu":
        # Match the target gfx950 encoding without importing GPU libraries.
        e4m3_dtype = torch.float8_e4m3fn
    else:
        from aiter.ops.triton.utils.types import e4m3_dtype

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
    # Only a full prefix page may be shared. Poisoning a partial page for
    # a shorter request would overwrite another request's valid values.
    if len(q_lens) > 1 and min(k_lens[:2]) >= page:
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
            # Use the BF16 operands consumed by the scaled kernel.
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
    assert (
        torch.isfinite(expected).all().item()
    ), "Invalid fixture: reference contains NaN/Inf before the attention launch"
    if reference_only:
        return expected, table, dense_v
    from op_tests.triton_tests.attention.test_unified_attention import shuffle_kv_cache

    assert layout == "shuffle"
    kc, vc = shuffle_kv_cache(dense_k, dense_v)
    # Guard the padded-head span so a wrong single-segment output pitch is
    # detected without corrupting the query, cache, or subsequent diagnostics.
    padded_dim = 1 << (dim - 1).bit_length()
    output_storage = torch.full(
        (q.shape[0] * heads * padded_dim + 256,),
        123.0,
        dtype=q.dtype,
        device=q.device,
    )
    out = output_storage[: q.numel()].view_as(q)
    out.fill_(float("nan"))
    kwargs = {"k_scale_cache": ks, "v_scale_cache": vs} if mode == "token" else {}
    from aiter.ops.triton.attention.unified_attention import unified_attention

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
        backend=backend,
        **kwargs,
    )
    torch.testing.assert_close(
        output_storage[q.numel() :],
        torch.full_like(output_storage[q.numel() :], 123.0),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("kv_heads", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize(
    "q_lens,k_lens", [([1], [16385]), ([1, 4, 19], [257, 2049, 3073])]
)
def test_gfx950_native_scale_cache(kv_heads, q_lens, k_lens):
    check_case(
        "shuffle",
        "token",
        False,
        page=128,
        heads=64,
        kv_heads=kv_heads,
        q_lens=q_lens,
        k_lens=k_lens,
        backend="gluon",
    )
