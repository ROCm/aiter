# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.attn_res import (
    attn_res_fwd,
    attn_res_gate,
    attn_res_inplace,
    attn_res_schedule,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

H = 7168
FP8 = torch.float8_e4m3fn
TOL = {"atol": 2e-2, "rtol": 2e-2}
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_arch() != "gfx1250", reason="gfx1250 only"
)


def _ref(p, blocks, nw, qkw, onw, n, widx, eps, oeps, quant_dtype):
    """fp32 reference of one call on p = bf16(prefix + delta); writes the block row of blocks in place."""
    if widx >= 0:
        blocks[:, widx] = p
    mixed = p.float()
    if n:
        src = torch.cat([blocks[:, :n].float(), p.float()[:, None]], dim=1)
        logits = (src * (nw.float() * qkw.float())).sum(-1) * torch.rsqrt(
            src.pow(2).mean(-1) + eps
        )
        mixed = (torch.softmax(logits, dim=-1)[..., None] * src).sum(1)
    out = (
        mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + oeps) * onw.float()
    ).to(torch.bfloat16)
    if quant_dtype is None:
        return out, None
    qmax = torch.finfo(quant_dtype).max
    scale = out.float().abs().amax(-1, keepdim=True).clamp(min=1e-10) / qmax
    return (out.float() / scale).clamp(-qmax, qmax).to(quant_dtype), scale


def _check(out, ref):
    ry, rs = ref
    if rs is None:
        torch.testing.assert_close(out.float(), ry.float(), **TOL)
        return
    y, s = out
    torch.testing.assert_close(s, rs, atol=1e-6, rtol=1e-2)
    torch.testing.assert_close(
        y.float() * s, ry.float() * rs, atol=2 * rs.max().item(), rtol=0.125
    )


def _model(T, seed):
    """A 187-call forward (93 layers x attention and MLP, then the final call) and its inputs."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    full = {
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
        36,
        40,
        44,
        48,
        52,
        56,
        60,
        64,
        68,
        72,
        76,
        80,
        84,
        88,
        92,
        93,
    }
    calls = []
    for L in range(93):
        n, wr = -(-L // 12), L % 12 == 0
        q = None if (L + 1) in full else FP8
        calls.append(
            {
                "n": n,
                "block_write_idx": L // 12 if wr else -1,
                "store_prefix": not wr,
                "quant_dtype": q,
            }
        )
        calls.append(
            {
                "n": n + wr,
                "block_write_idx": -1,
                "store_prefix": True,
                "quant_dtype": FP8 if L == 0 else None,
            }
        )
    calls.append(
        {"n": 8, "block_write_idx": -1, "store_prefix": False, "quant_dtype": None}
    )
    w = torch.randn((len(calls), 3, H), generator=g, device="cuda", dtype=torch.float32)
    for c, (nw, qkw, onw) in zip(calls, w):
        c.update(
            norm_weight=(1 + 0.1 * nw).to(torch.bfloat16),
            qk_weight=(0.05 * qkw).to(torch.bfloat16),
        )
        c.update(
            output_norm_weight=(1 + 0.1 * onw).to(torch.bfloat16),
            eps=1e-6,
            output_norm_eps=1e-6,
        )
    x = torch.randn((T, 11, H), generator=g, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    deltas = [None] + [x[:, 1 + i % 2] for i in range(1, len(calls))]
    return calls, deltas, 4 * x[:, 0].contiguous(), 4 * x[:, 3:].contiguous()


def _forward(calls, deltas, x0, blocks, sched, residual_added=False):
    prefix, blocks, outs = x0.clone(), blocks.clone(), []
    for i, c in enumerate(calls):
        delta = deltas[i]
        if residual_added and delta is not None:
            prefix.add_(delta)
            delta = None
        args = (
            c["norm_weight"],
            c["qk_weight"],
            c["output_norm_weight"],
            c["n"],
            c["block_write_idx"],
            c["eps"],
        )
        out = attn_res_inplace(
            prefix,
            delta,
            blocks,
            *args,
            c["output_norm_eps"],
            quant_dtype=c["quant_dtype"],
            store_prefix=c["store_prefix"],
            residual_added=residual_added,
            schedule=sched,
            call_idx=i,
        )
        outs.append(
            tuple(t.clone() for t in out) if isinstance(out, tuple) else out.clone()
        )
    return outs


@pytest.mark.parametrize("T", [8, 1728])
@pytest.mark.parametrize("mode", [{}, {"residual_added": True}])
def test_attn_res_inplace_forward(T, mode):
    calls, deltas, x0, blocks = _model(T, seed=T)
    residual_added = mode.get("residual_added", False)
    outs = _forward(
        calls, deltas, x0, blocks, attn_res_schedule(calls, **mode), residual_added
    )
    prefix, blocks = x0.clone(), blocks.clone()
    for i, c in enumerate(calls):
        p = (
            prefix
            if deltas[i] is None
            else (prefix.float() + deltas[i].float()).to(torch.bfloat16)
        )
        args = (
            c["norm_weight"],
            c["qk_weight"],
            c["output_norm_weight"],
            c["n"],
            c["block_write_idx"],
            c["eps"],
        )
        _check(outs[i], _ref(p, blocks, *args, c["output_norm_eps"], c["quant_dtype"]))
        if c["store_prefix"] or residual_added:
            prefix = p


@pytest.mark.parametrize("L", [2, 9])
def test_attn_res_fwd_and_gate_gluon(L):
    g = torch.Generator(device="cuda").manual_seed(L)
    x = torch.randn(
        (320, L + 3, H), generator=g, device="cuda", dtype=torch.float32
    ).to(torch.bfloat16)
    res, query, w, ow = x[:, :L].contiguous(), x[0, L], x[1, L], x[2, L]
    ref = attn_res_fwd(query, res, w, ow, layout="packed")
    out = attn_res_fwd(query, res, w, ow, layout="packed", backend="gluon")
    torch.testing.assert_close(out.float(), ref.float(), **TOL)
    sw = (query.float() * w.float()).contiguous()
    prefix, blocks = res[:, -1].contiguous(), res[:, :-1].contiguous()
    y_ref, _ = attn_res_gate(prefix, blocks, sw, output_rms_weight=ow)
    y, _ = attn_res_gate(prefix, blocks, sw, output_rms_weight=ow, backend="gluon")
    torch.testing.assert_close(y.float(), y_ref.float(), **TOL)
