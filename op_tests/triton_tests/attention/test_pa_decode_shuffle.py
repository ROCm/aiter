# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.attention.pa_decode_shuffle import (
    paged_attention_decode_shuffle_swa,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm GPU"
)

FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e4m3fnuz]
X = 16  # 16 // fp8 elem size


def make_shuffle_inputs(
    seq_lens,
    num_q_heads,
    num_kv_heads,
    head,
    block_size,
    fp8_dtype,
    per_token_scale,
    device="cuda",
    seed=0,
):
    """Shuffled fp8 KV cache for the given seq_lens. Unused slots and their
    per-token scales hold NaN. Shared with the benchmark.

    Returns (query, key_cache, value_cache, block_tables, seq_lens, k_scale,
    v_scale, k_deq, v_deq), k_deq / v_deq are the dequantized fp32 tokens per
    sequence for the reference."""
    gen = torch.Generator(device=device).manual_seed(seed)
    num_seqs = len(seq_lens)
    blocks = [(n + block_size - 1) // block_size for n in seq_lens]
    max_blocks = max(1, max(blocks))
    num_blocks = sum(blocks) + 2
    perm = torch.randperm(num_blocks, generator=gen, device=device).to(torch.int32)
    block_tables = torch.zeros((num_seqs, max_blocks), dtype=torch.int32, device=device)
    nb = 0
    for i, n in enumerate(blocks):
        block_tables[i, :n] = perm[nb : nb + n]
        nb += n

    key_cache = torch.full(
        (num_blocks, num_kv_heads, head // X, block_size, X),
        float("nan"),
        dtype=fp8_dtype,
        device=device,
    )
    value_cache = torch.full(
        (num_blocks, num_kv_heads, block_size // X, head, X),
        float("nan"),
        dtype=fp8_dtype,
        device=device,
    )
    if per_token_scale:
        # V scale is a strided view so K and V scales have different strides
        k_scale = torch.full(
            (num_blocks, num_kv_heads, block_size), float("nan"), device=device
        )
        v_scale = torch.full(
            (num_blocks, num_kv_heads, block_size, 2), float("nan"), device=device
        )[..., 1]
    else:
        k_scale = torch.tensor([0.7], device=device)
        v_scale = torch.tensor([1.3], device=device)

    fp8_max = torch.finfo(fp8_dtype).max
    k_deq, v_deq = [], []
    for i, n in enumerate(seq_lens):
        pos = torch.arange(n, device=device)
        blk = block_tables[i, pos // block_size].long()
        slot = pos % block_size
        k = 2 * torch.randn((n, num_kv_heads, head), generator=gen, device=device)
        v = 2 * torch.randn((n, num_kv_heads, head), generator=gen, device=device)
        if per_token_scale:
            ks = 0.5 + torch.rand((n, num_kv_heads), generator=gen, device=device)
            vs = 0.5 + torch.rand((n, num_kv_heads), generator=gen, device=device)
            k_scale[blk, :, slot] = ks
            v_scale[blk, :, slot] = vs
            ks, vs = ks[..., None], vs[..., None]
        else:
            ks, vs = k_scale, v_scale
        kq = (k / ks).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        vq = (v / vs).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        key_cache[blk, :, :, slot, :] = kq.view(n, num_kv_heads, head // X, X)
        value_cache[blk, :, slot // X, :, slot % X] = vq
        k_deq.append(kq.float() * ks)
        v_deq.append(vq.float() * vs)

    query = torch.randn((num_seqs, num_q_heads, head), generator=gen, device=device).to(
        torch.bfloat16
    )
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return (
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens_t,
        k_scale,
        v_scale,
        k_deq,
        v_deq,
    )


def ref_decode(query, k_deq, v_deq, seq_lens, scale, sliding_window):
    num_seqs, num_q_heads, head = query.shape
    out = torch.zeros((num_seqs, num_q_heads, head), device=query.device)
    for i, n in enumerate(seq_lens):
        if n == 0:
            continue
        lo = 0 if sliding_window <= 0 else max(0, n - sliding_window)
        grp = num_q_heads // k_deq[i].shape[1]
        k = k_deq[i][lo:n].repeat_interleave(grp, dim=1)
        v = v_deq[i][lo:n].repeat_interleave(grp, dim=1)
        p = torch.softmax(torch.einsum("hd,khd->hk", query[i].float(), k) * scale, -1)
        out[i] = torch.einsum("hk,khd->hd", p, v)
    return out


@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("per_token_scale", [False, True])
@pytest.mark.parametrize(
    "seq_lens, num_q_heads, num_kv_heads, block_size, window",
    [
        ([300, 130, 600, 65], 32, 8, 16, 256),
        ([300, 130, 600, 65], 32, 8, 16, 0),
        ([500, 200, 90], 32, 8, 32, 256),
        # one valid slot in a 16-slot page, window edges inside a page
        ([1, 17, 50, 200], 32, 8, 16, 64),
        ([0, 31, 255, 1000], 16, 1, 16, 100),
        ([128, 256], 8, 8, 16, 128),
    ],
)
def test_pa_decode_shuffle(
    seq_lens, num_q_heads, num_kv_heads, block_size, window, per_token_scale, fp8_dtype
):
    head = 128
    q, kc, vc, bt, sl, ks, vs, k_deq, v_deq = make_shuffle_inputs(
        seq_lens,
        num_q_heads,
        num_kv_heads,
        head,
        block_size,
        fp8_dtype,
        per_token_scale,
    )
    scale = head**-0.5
    out = torch.full_like(q, float("nan"))
    paged_attention_decode_shuffle_swa(out, q, kc, vc, bt, sl, scale, ks, vs, window)
    ref = ref_decode(q, k_deq, v_deq, seq_lens, scale, window)
    assert torch.isfinite(out).all(), "NaN or inf in the output"
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
