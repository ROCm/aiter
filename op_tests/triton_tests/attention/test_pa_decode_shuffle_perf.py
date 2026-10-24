# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter.ops.triton.attention.pa_decode_shuffle_perf as pa_perf
from aiter.ops.triton.attention.pa_decode_shuffle_perf import (
    paged_attention_decode_shuffle_swa_perf,
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


def run_and_check(
    seq_lens,
    num_q_heads,
    num_kv_heads,
    head,
    block_size,
    window,
    fp8_dtype,
    per_token_scale,
    force_split,
    workspace=None,
    partition_size=256,
):
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
    out = torch.empty_like(q)
    paged_attention_decode_shuffle_swa_perf(
        out,
        q,
        kc,
        vc,
        bt,
        sl,
        scale,
        ks,
        vs,
        window,
        partition_size=partition_size,
        max_seq_len=max(seq_lens),
        workspace=workspace,
        force_split=force_split,
    )
    ref = ref_decode(q, k_deq, v_deq, seq_lens, scale, window)
    assert torch.isfinite(out).all(), "NaN or inf in the output"
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.fixture
def auto_split(monkeypatch):
    # make the auto path deterministic: split allowed, 256 CUs
    monkeypatch.setattr(pa_perf, "_arch_allows_split", lambda: True)
    monkeypatch.setattr(pa_perf, "get_num_sms", lambda: 256)


@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("per_token_scale", [False, True])
@pytest.mark.parametrize("force_split", [False, True])
@pytest.mark.parametrize(
    "seq_lens, window, block_size",
    [
        ([300, 130, 600, 65], 256, 16),
        ([300, 130, 600, 65], 0, 16),
        ([500, 200, 90], 256, 32),
        ([1, 17, 50, 200], 64, 16),
        ([0, 33, 4096, 3000], 1024, 16),
    ],
)
def test_pa_decode_shuffle_perf(
    seq_lens, window, block_size, force_split, per_token_scale, fp8_dtype
):
    run_and_check(
        seq_lens,
        32,
        8,
        128,
        block_size,
        window,
        fp8_dtype,
        per_token_scale,
        force_split,
    )


@pytest.mark.parametrize(
    "num_seqs, seq_len, window",
    [
        (16, 4096, 0),  # base grid 128 caps n_part at 4: each partition 1024 keys
        (16, 6000, 4096),
        (2, 20000, 0),
        (64, 2048, 0),  # full grid, long enough walk: one partition per 256 keys
    ],
)
def test_pa_decode_shuffle_perf_auto(auto_split, num_seqs, seq_len, window):
    run_and_check(
        [seq_len] * (num_seqs - 1) + [seq_len // 3],
        32,
        8,
        128,
        16,
        window,
        torch.float8_e4m3fn,
        False,
        None,
    )


@pytest.mark.parametrize("force_split", [True, None])
def test_pa_decode_shuffle_perf_empty_rows(auto_split, force_split):
    # zero-length (cudagraph padding) rows and partitions past seq_len, with
    # NaN in the scratch the reduce would skip
    ws = {}
    run_and_check(
        [0, 256, 4096, 0], 32, 8, 128, 16, 0, torch.float8_e4m3fn, True, force_split, ws
    )
    for t in ws["bufs"]:
        t.fill_(float("nan"))
    run_and_check(
        [0, 256, 4096, 0], 32, 8, 128, 16, 0, torch.float8_e4m3fn, True, force_split, ws
    )


def test_pa_decode_shuffle_perf_workspace_reuse():
    # one workspace across head sizes and partition counts, it has to grow
    # instead of being reused undersized
    ws = {}
    for head, seq_len in (
        (64, 4096),
        (128, 4096),
        (128, 1100),
        (64, 9000),
        (128, 4096),
    ):
        run_and_check(
            [seq_len, seq_len // 2],
            32,
            8,
            head,
            16,
            0,
            torch.float8_e4m3fn,
            False,
            True,
            ws,
        )
    ptr = ws["bufs"][2].data_ptr()
    run_and_check([4096, 2048], 32, 8, 128, 16, 0, torch.float8_e4m3fn, False, True, ws)
    assert ws["bufs"][2].data_ptr() == ptr, "scratch was reallocated for a smaller call"


def test_pa_decode_shuffle_perf_two_streams():
    # internal scratch is per stream: two split calls in flight on two streams
    args = [
        make_shuffle_inputs(
            [5000, 3000], 32, 8, 128, 16, torch.float8_e4m3fn, False, seed=s
        )
        for s in (1, 2)
    ]
    outs = [torch.empty_like(a[0]) for a in args]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.cuda.synchronize()
    for _ in range(4):
        for st, a, o in zip(streams, args, outs):
            with torch.cuda.stream(st):
                q, kc, vc, bt, sl, ks, vs, _, _ = a
                paged_attention_decode_shuffle_swa_perf(
                    o,
                    q,
                    kc,
                    vc,
                    bt,
                    sl,
                    128**-0.5,
                    ks,
                    vs,
                    0,
                    max_seq_len=5000,
                    force_split=True,
                )
    torch.cuda.synchronize()
    for a, o in zip(args, outs):
        ref = ref_decode(a[0], a[7], a[8], [5000, 3000], 128**-0.5, 0)
        torch.testing.assert_close(o.float(), ref, atol=2e-2, rtol=2e-2)
