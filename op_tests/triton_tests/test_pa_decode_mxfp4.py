# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy test: gluon MXFP4 sparse PA decode vs fp32 attention over the
exactly dequantized MXFP4 pages (gfx942)."""

import importlib.util
import os

import pytest
import torch

from aiter.ops.triton.utils._triton import arch_info

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
if arch_info.get_arch() != "gfx942":
    pytest.skip("gfx942 (fp8 e4m3fnuz) kernel", allow_module_level=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNEL = os.path.join(
    _HERE, "..", "..", "aiter", "ops", "triton", "gluon", "pa_decode_mxfp4_gluon.py"
)
if os.path.exists(_KERNEL):  # prefer this checkout over an installed aiter
    spec = importlib.util.spec_from_file_location("pa_decode_mxfp4_gluon", _KERNEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
else:
    from aiter.ops.triton.gluon import pa_decode_mxfp4_gluon as mod

D = 128
BLOCK = 128
TOPK = 16
PAGE = mod.PAGE_BYTES
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant(pages):  # uint8 [..., 8704] -> fp32 [..., 128, 128]
    data = pages[..., : BLOCK * D // 2].unflatten(-1, (BLOCK, D // 2)).int()
    scale = pages[..., BLOCK * D // 2 :].unflatten(-1, (BLOCK, D // 32))
    code = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2)
    grid = torch.tensor(E2M1, device=pages.device)
    value = grid[code & 7] * torch.where(code & 8 != 0, -1.0, 1.0)
    step = torch.exp2(scale.float() - 127.0)[..., None]
    return (value.unflatten(-1, (4, 32)) * step).flatten(-2)


def make_case(batch, qlen, ctx, H, seed, spread):
    g = torch.Generator(device="cuda").manual_seed(seed)
    nblk = (ctx + BLOCK - 1) // BLOCK
    rows = batch * qlen
    pool = batch * max(nblk, TOPK)
    perm = torch.randperm(pool, device="cuda", generator=g).view(batch, -1)
    cache = torch.randint(
        0, 256, (pool, 2 * H, PAGE), dtype=torch.uint8, device="cuda", generator=g
    )
    cache[..., BLOCK * D // 2 :] = torch.randint(
        127 - spread,
        128,
        (pool, 2 * H, BLOCK * 4),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    r = torch.arange(rows, device="cuda")
    req, qi = r // qlen, r % qlen
    causal = ctx - qlen + qi + 1  # tokens visible to the row
    tail = (causal + BLOCK - 1) // BLOCK - 1
    sel = torch.empty(rows, H, TOPK, dtype=torch.int64, device="cuda")
    ctx_lens = torch.empty(rows, H, dtype=torch.int32, device="cuda")
    for i in range(rows):
        for h in range(H):
            n_prev = min(TOPK - 1, int(tail[i]))
            prev = torch.randperm(int(tail[i]), device="cuda", generator=g)[:n_prev]
            logical = torch.cat([prev.sort().values, tail[i : i + 1]])
            phys = perm[req[i], logical]
            sel[i, h, : n_prev + 1] = phys
            sel[i, h, n_prev + 1 :] = 0
            ctx_lens[i, h] = n_prev * BLOCK + int(causal[i] - tail[i] * BLOCK)
    q = torch.randn(rows, H * 16, D, device="cuda", generator=g, dtype=torch.bfloat16)
    j8 = torch.arange(8, device="cuda")
    hh = torch.arange(H, device="cuda")[None, :, None, None]
    bt16 = ((sel[..., None] * 8 + j8) * H + hh).view(rows * H, TOPK * 8).to(torch.int32)
    return cache, sel, ctx_lens, q, bt16


def reference(q, cache, sel, ctx_lens, H, scale):
    rows = q.shape[0]
    out = torch.empty(rows, q.shape[1], D, device="cuda")
    for r in range(rows):
        for h in range(H):
            n = int(ctx_lens[r, h])
            blk = sel[r, h, : (n + BLOCK - 1) // BLOCK]
            k = dequant(cache[blk, h]).reshape(-1, D)[:n]
            v = dequant(cache[blk, H + h]).reshape(-1, D)[:n]
            qq = q[r, h * 16 : (h + 1) * 16].float()
            p = torch.softmax(qq @ k.T * scale, -1)
            out[r, h * 16 : (h + 1) * 16] = p @ v
    return out


@pytest.mark.parametrize("ctx", [17, 255, 256, 257, 1000, 8192, 32768])
@pytest.mark.parametrize("qlen", [1, 4])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("splits", [None, 1, 3])
def test_pa_decode_mxfp4(ctx, qlen, seed, splits):
    H, batch = 2, 2
    cache, sel, ctx_lens, q, bt16 = make_case(batch, qlen, ctx, H, seed, spread=8)
    scale = D**-0.5
    out = torch.full_like(q, float("nan"))
    mod.pa_decode_mxfp4_gluon(
        out,
        q,
        cache.view(-1, 2 * H, BLOCK, 68),
        ctx_lens.view(-1),
        bt16,
        scale,
        H,
        max_context_partition_num=splits,
    )
    ref = reference(q, cache, sel, ctx_lens, H, scale)
    o = out.float()
    assert not o.isnan().any()
    rel = ((o - ref).norm() / ref.norm()).item()
    # fp8 (e4m3fnuz) Q / P / decoded K,V compute, as in the fp8 KV gluon path.
    assert rel < 8e-2, rel


@pytest.mark.parametrize("ctx", [17, 31, 32, 33, 255, 1000, 8192, 32768])
@pytest.mark.parametrize("qlen", [1, 4])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    "cfg", [None, (64, 32, 1), (32, 64, 2), (5, 32, 1), (3, 64, 2)]
)
def test_pa_decode_mxfp4_small_grid(ctx, qlen, seed, cfg):
    """Small-grid configs: 1 warp x 32-token / 2 warps x 64-token partitions
    (cfg = splits, partition_size, num_warps; None = the default heuristic at
    batch 1, which picks them for rows * H <= 8)."""
    H, batch = 2, 1
    cache, sel, ctx_lens, q, bt16 = make_case(batch, qlen, ctx, H, seed, spread=8)
    scale = D**-0.5
    out = torch.full_like(q, float("nan"))
    kw = {}
    if cfg is not None:
        kw = {
            "max_context_partition_num": cfg[0],
            "partition_size": cfg[1],
            "num_warps": cfg[2],
        }
    mod.pa_decode_mxfp4_gluon(
        out,
        q,
        cache.view(-1, 2 * H, BLOCK, 68),
        ctx_lens.view(-1),
        bt16,
        scale,
        H,
        **kw
    )
    ref = reference(q, cache, sel, ctx_lens, H, scale)
    o = out.float()
    assert not o.isnan().any()
    rel = ((o - ref).norm() / ref.norm()).item()
    assert rel < 8e-2, rel
