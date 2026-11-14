# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 IOD-local placement (aiter.ops.kv_iod, aiter.ops.lmhead_iod) against the stock layouts.

The cache is filled by the stock fused norm + MRoPE + A8K4V8 write, then copied into one layer's placed K / V
regions. Placed writes, placed prefill attention and the placed LM head are bitwise equal to stock. Placed decode
rounds the token split's T to the placement chunk, so it matches stock up to the FP8-P rounding of another
segmentation; its three schedules are bitwise equal to each other.
"""

import numpy as np
import pytest
import torch

from aiter.ops.triton.utils._triton import arch_info

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_info.get_arch() != "gfx1250",
    reason="gfx1250 only",
)

HD, QH, KH, BLOCK, MAX_POS = 256, 16, 2, 128, 65536


def _fp8():
    import aiter

    return aiter.dtypes.fp8


def _layer(nblk):
    from aiter.ops import kv_iod

    k_reg = torch.empty(
        (nblk // 28 + 4) * kv_iod.PAGE, dtype=torch.uint8, device="cuda"
    )
    v_reg = torch.empty(
        (nblk // 16 + 4) * kv_iod.PAGE, dtype=torch.uint8, device="cuda"
    )
    layer = kv_iod.KvIodLayer(k_reg, v_reg)
    layer._regions = (k_reg, v_reg)
    return layer


def _write_args(T, slots, seed):
    torch.manual_seed(seed)
    qkv = torch.randn((T, (QH + 2 * KH) * HD), dtype=torch.bfloat16, device="cuda")
    q, k, v = torch.split(qkv, [QH * HD, KH * HD, KH * HD], dim=-1)
    pos = torch.randint(0, MAX_POS - 16, (T,), device="cuda")
    return {
        "q": q,
        "k": k,
        "v": v,
        "q_weight": torch.randn(HD, dtype=torch.bfloat16, device="cuda") * 0.2,
        "k_weight": torch.randn(HD, dtype=torch.bfloat16, device="cuda") * 0.2,
        "eps": 1e-6,
        "positions": torch.stack([pos, pos + 3, pos + 11]),
        "cos": torch.rand((MAX_POS, 32), dtype=torch.float32, device="cuda"),
        "sin": torch.rand((MAX_POS, 32), dtype=torch.float32, device="cuda"),
        "mrope_section": [11, 11, 10],
        "slot_mapping": slots,
        "kv_scale": 0.5,
    }


def _caches(codes, zero=False):
    if (
        zero
    ):  # attention reads unwritten tail positions with weight 0, and 0 * NaN is NaN
        k = torch.zeros(
            (codes, KH, BLOCK, HD // 2 + HD // 16), dtype=torch.uint8, device="cuda"
        )
        return k, torch.zeros(
            (codes, KH, BLOCK // 16, HD, 16), dtype=torch.uint8, device="cuda"
        ).view(_fp8())
    k = torch.randint(
        0, 256, (codes, KH, BLOCK, HD // 2 + HD // 16), dtype=torch.uint8, device="cuda"
    )
    v = torch.randint(
        0, 256, (codes, KH, BLOCK // 16, HD, 16), dtype=torch.uint8, device="cuda"
    ).view(_fp8())
    return k, v


@pytest.mark.parametrize("T", [7, 128, 4096])
def test_kv_iod_write(T):
    """Placed write (row kernel below 4096 tokens, page kernel from 4096) == stock write, byte for byte."""
    from aiter.ops import kv_iod
    from aiter.ops.triton.fusions.fused_qk_norm_mrope import fused_qk_norm_mrope_cache

    layer = _layer(96 if T < 4096 else 160)
    codes = layer.codes
    x = layer.x_table(codes, "cuda")
    g = torch.Generator().manual_seed(T)
    if T < 4096:
        blocks = torch.randperm(codes, generator=g)[: max(2, T // 32)]
        cand = (blocks[:, None] * BLOCK + torch.arange(BLOCK)[None, :]).reshape(-1)
        slots = cand[torch.randperm(cand.numel(), generator=g)[:T]].clone()
    else:  # a prefill: consecutive slots over random blocks of both parities
        blocks = torch.randperm(codes, generator=g)[: T // BLOCK + 1]
        slots = (
            (blocks[:, None] * BLOCK + torch.arange(BLOCK)[None, :])
            .reshape(-1)[:T]
            .clone()
        )
    slots[3::11] = -1
    args = _write_args(T, slots.cuda(), T)
    k_stock, v_stock = _caches(codes)
    kv_iod.stock_to_placed(k_stock, v_stock, layer, np.arange(codes), x)
    q_s, k_s = fused_qk_norm_mrope_cache(**args, k_cache=k_stock, v_cache=v_stock)
    q_p, k_p = fused_qk_norm_mrope_cache(
        **args,
        k_cache=k_stock.clone(),
        v_cache=v_stock.clone(),
        kv_loc=(layer.k, layer.v.view(_fp8()), x)
    )
    kb, vb = kv_iod.placed_to_stock(layer, np.arange(codes), x, None, None)
    assert torch.equal(kb, k_stock.view(codes, KH, -1))
    assert torch.equal(vb, v_stock.view(torch.uint8).view(codes, KH, -1))
    assert torch.equal(q_s.view(torch.uint8), q_p.view(torch.uint8)) and torch.equal(
        k_s, k_p
    )


def _attention_setup(ctx, seed):
    """Stock and placed caches of sequences of `ctx` tokens, filled by the stock write; block ids are codes
    placed in 16-block chunks alternating domains (kv_iod's allocation rule)."""
    from aiter.ops import kv_iod
    from aiter.ops.triton.fusions.fused_qk_norm_mrope import fused_qk_norm_mrope_cache

    ntile = [(c + BLOCK - 1) // BLOCK for c in ctx]
    layer = _layer(2 * sum(ntile) + 64)
    codes = layer.codes
    x = layer.x_table(codes, "cuda")
    nxt = [0, 0]
    rows = []
    for s, n in enumerate(ntile):
        row = []
        for p in range(n):
            d = ((p // 16) + s) & 1
            row.append(2 * nxt[d] + d)
            nxt[d] += 1
        rows.append(row)
    width = max(ntile) + 8
    bt = torch.tensor(
        [r + [r[0]] * (width - len(r)) for r in rows], dtype=torch.int32, device="cuda"
    )
    slots = torch.cat(
        [
            bt[s, : ntile[s]].long()[:, None] * BLOCK
            + torch.arange(BLOCK, device="cuda")[None, :]
            for s in range(len(ctx))
        ],
        0,
    ).reshape(len(ctx), -1)
    slots = torch.cat([slots[s, : ctx[s]] for s in range(len(ctx))])
    k_stock, v_stock = _caches(codes, zero=True)
    fused_qk_norm_mrope_cache(
        **_write_args(slots.numel(), slots, seed), k_cache=k_stock, v_cache=v_stock
    )
    kv_iod.stock_to_placed(k_stock, v_stock, layer, np.arange(codes), x)
    return layer, x, bt, k_stock, v_stock


def _q(rows, seed):
    torch.manual_seed(seed)
    return (
        (torch.randn((rows, QH, HD), dtype=torch.bfloat16, device="cuda") * 2)
        .clamp(-448, 448)
        .to(_fp8())
    )


def test_kv_iod_prefill():
    """Prefill (unified attention 2D) on the placed cache: with and without a resolved table bitwise equal, and
    stock up to the accumulation order of another KV tiling."""
    from aiter.ops import kv_iod
    from aiter.ops.triton.attention.unified_attention import unified_attention

    ctx, new = [3000, 4500], [512, 700]
    layer, x, bt, k_stock, v_stock = _attention_setup(ctx, 1)
    q = _q(sum(new), 2)
    cu = torch.tensor([0, new[0], new[0] + new[1]], dtype=torch.int32, device="cuda")
    args = {
        "q": q,
        "k": k_stock,
        "v": v_stock,
        "cu_seqlens_q": cu,
        "max_seqlen_q": max(new),
        "seqused_k": torch.tensor(ctx, dtype=torch.int32, device="cuda"),
        "max_seqlen_k": max(ctx),
        "softmax_scale": HD**-0.5,
        "causal": True,
        "window_size": (-1, -1),
        "softcap": 0,
        "q_descale": None,
        "k_descale": None,
        "v_descale": torch.tensor(2.0, device="cuda"),
        "shuffled_kv_cache": True,
    }
    ref = torch.empty(q.shape, dtype=torch.bfloat16, device="cuda")
    unified_attention(**args, out=ref, block_table=bt)
    kv = (x, layer.k, layer.v.view(_fp8()), layer.label_xor, 16, 2)
    outs = []
    for table, loc in (
        (bt, kv),
        (kv_iod.resolve_block_tables(bt, x[None])[0].contiguous(), kv + (True,)),
    ):
        outs.append(torch.empty_like(ref))
        unified_attention(**args, out=outs[-1], block_table=table, kv_loc=loc)
    # the two placed calls are the same computation; stock may tile the KV differently (several pages a tile)
    assert torch.equal(outs[0], outs[1])
    assert not torch.isnan(ref).any()
    d = (outs[0].float() - ref.float()).abs()
    assert float(d.mean()) < 5e-3 and float(d.max()) < 0.25, (
        float(d.mean()),
        float(d.max()),
    )


# the reduce merging the segments: Triton reduce_segments, gfx1250 split reduce, its one-shot variant
@pytest.mark.parametrize("reduce", ["triton", "split", "oneshot"])
def test_kv_iod_decode(monkeypatch, reduce):
    """4-token verify decode (unified attention 3D, token split) on the placed cache: modes 1-3 bitwise equal to
    each other, and stock up to FP8-P segmentation rounding."""
    import aiter.ops.triton.attention.unified_attention as U
    from aiter.ops import kv_iod

    monkeypatch.setattr(U, "_GFX12_TOKEN_SPLIT", True)
    ctx = [20000, 9000, 33000, 15000, 4100, 27000, 12000, 8000]
    layer, x, bt, k_stock, v_stock = _attention_setup(ctx, 3)
    B = len(ctx)
    q = _q(4 * B, 4)
    args = {
        "q": q,
        "k": k_stock,
        "v": v_stock,
        "cu_seqlens_q": torch.arange(0, 4 * B + 1, 4, dtype=torch.int32, device="cuda"),
        "max_seqlen_q": 4,
        "seqused_k": torch.tensor(ctx, dtype=torch.int32, device="cuda"),
        "max_seqlen_k": max(ctx),
        "softmax_scale": HD**-0.5,
        "causal": True,
        "window_size": (-1, -1),
        "softcap": 0,
        "q_descale": None,
        "k_descale": None,
        "v_descale": torch.tensor(2.0, device="cuda"),
        "shuffled_kv_cache": True,
        "equal_seqlen_q": True,
    }
    args["split_reduce"] = reduce != "triton"
    args["split_reduce_oneshot"] = reduce == "oneshot"
    ref = torch.empty(q.shape, dtype=torch.bfloat16, device="cuda")
    U.unified_attention(**args, out=ref, block_table=bt)
    btr = kv_iod.resolve_block_tables(bt, x[None])[0].contiguous()
    outs = {}
    for mode in (1, 2, 3):
        outs[mode] = torch.empty_like(ref)
        kv = (x, layer.k, layer.v.view(_fp8()), layer.label_xor, 16, mode, True)
        U.unified_attention(**args, out=outs[mode], block_table=btr, kv_loc=kv)
    assert torch.equal(outs[1], outs[2]) and torch.equal(outs[2], outs[3])
    d = (outs[2].float() - ref.float()).abs()
    assert float(d.mean()) < 5e-3 and float(d.max()) < 0.1, (
        float(d.mean()),
        float(d.max()),
    )


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_lmhead_iod(monkeypatch, mode):
    """IOD-placed bf16 [N, K] weight through gemm_a16w16 (compute_bound, 128 x 128 tiles) == stock."""
    from aiter.ops import lmhead_iod
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    torch.manual_seed(0)
    w = (
        torch.randn(8192, 2048, dtype=torch.bfloat16, device="cuda") * 0.02
    ).contiguous()
    try:
        lmhead_iod.place(w)
    except (
        RuntimeError
    ) as e:  # the home probe can fail; the caller then keeps the stock weight
        pytest.skip(str(e))
    x = torch.randn(8, 2048, dtype=torch.bfloat16, device="cuda")
    monkeypatch.setenv("AITER_LMHEAD_LOC", "0")
    ref = gemm_a16w16(x, w, kernel_type="compute_bound")
    monkeypatch.setenv("AITER_LMHEAD_LOC", str(mode))
    out = gemm_a16w16(x, w, kernel_type="compute_bound")
    assert torch.equal(out, ref)
