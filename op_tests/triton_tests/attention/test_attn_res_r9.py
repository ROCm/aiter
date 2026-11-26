# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# gfx1250 Kimi-K3 AttnRes (aiter.ops.triton.gluon.attn_res_r9, r9 tuned tables) against the torch fp32 reference and tolerances of
# attn_res_ref.py: the plain drop-in (n = 0..8 x fp8/bf16 out x delta/no delta x block write), mix-ahead
# groups (producer + consumers, the last one writing the next block) and the real 187-call K3.1 order.
import importlib.util
import os

import pytest
import torch

from aiter.ops.triton.gluon.attn_res_r9 import AttnResSchedule, attn_res, attn_res_consume, attn_res_produce

_spec = importlib.util.spec_from_file_location(
    "attn_res_ref", os.path.join(os.path.dirname(__file__), "attn_res_ref.py"))
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)
H = ref.H
FULL = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93}


@pytest.mark.parametrize("T", [1, 7, 160, 5100, 6912])
@pytest.mark.parametrize("n", range(9))
@pytest.mark.parametrize("quant", [True, False])
@pytest.mark.parametrize("has_delta", [True, False])
@pytest.mark.parametrize("write", [False, True])
def test_attn_res_plain(T, n, quant, has_delta, write):
    if write and n == 8:
        pytest.skip("8 blocks: no free slot to write")
    qd = ref.FP8 if quant else None
    widx = n if write else -1
    prefix, delta, blocks, nw, qkw, onw = ref.make_inputs(T, n, has_delta, True)
    rp, rb, rout, rscale = ref.ref_attn_res(prefix.clone(), delta, blocks.clone(), nw, qkw, onw, n, widx, 1e-5, 1e-5, qd)
    p, b = prefix.clone(), blocks.clone()
    res = attn_res(p, delta, b, nw, qkw, onw, n, widx, 1e-5, 1e-5, quant_dtype=qd)
    out, scale = res if qd is not None else (res, None)
    torch.testing.assert_close(p, rp, rtol=0, atol=0, msg="prefix")
    torch.testing.assert_close(b, rb, rtol=0, atol=0, msg="blocks")
    ref._check(out, scale, rout, rscale)


@pytest.mark.parametrize("T", [1, 7, 160, 6912])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("has_delta,has_onorm,quant", [(True, True, True), (True, True, False), (False, False, False), (False, True, True)])
def test_attn_res_group(T, n, has_delta, has_onorm, quant):
    K = 3
    qd = ref.FP8 if quant else None
    prefix, delta, blocks, nw, qkw0, onw = ref.make_inputs(T, n, has_delta, has_onorm)
    g = torch.Generator(device="cuda").manual_seed(n)
    qkws = [(torch.randn(H, generator=g, device="cuda") * 0.05).to(torch.bfloat16) for _ in range(K)]
    w = torch.stack([nw.float() * q.float() for q in qkws]).contiguous()
    rpref, rblk = prefix.clone(), blocks.clone()
    rp, rb, rout, rscale = ref.ref_attn_res(rpref, delta, rblk, nw, qkw0, onw, n, -1, 1e-5, 1e-5, qd)
    res = attn_res_produce(prefix, delta, blocks, nw, qkw0, onw, n, -1, 1e-5, 1e-5, w_next=w, quant_dtype=qd)
    out, scale = (res[0], res[1]) if qd is not None else (res[0], None)
    mix, stats = res[-2], res[-1]
    torch.testing.assert_close(prefix, rp, rtol=0, atol=0, msg="prefix")
    assert torch.equal(blocks, rb)
    ref._check(out, scale, rout, rscale)
    rpref = rp
    for c in range(K):
        widx = n if (c == K - 1 and n < 8) else -1  # the group's last call writes the next block
        rp, rb, rout, rscale = ref.ref_attn_res(rpref, delta, rblk, nw, qkws[c], onw, n, widx, 1e-5, 1e-5, qd)
        res = attn_res_consume(prefix, delta, blocks, mix[:, c], stats[:, c], nw, qkws[c], onw, widx, 1e-5, 1e-5, quant_dtype=qd)
        out, scale = res if qd is not None else (res, None)
        torch.testing.assert_close(prefix, rp, rtol=0, atol=0, msg="prefix")
        assert torch.equal(blocks, rb), "blocks"
        ref._check(out, scale, rout, rscale)
        rpref, rblk = rp, rb


def _calls(g):
    rnd = lambda sc=1.0: (torch.randn(H, generator=g, device="cuda") * sc).bfloat16()
    calls = []
    for L in range(93):
        n = -(-L // 12); wr = L % 12 == 0
        for part in ("attn", "mlp"):
            calls.append(dict(
                n=n if part == "attn" else n + wr,
                block_write_idx=L // 12 if (wr and part == "attn") else -1,
                quant_dtype=(None if (L + 1) in FULL else ref.FP8) if part == "attn" else (ref.FP8 if L == 0 else None),
                store_prefix=not (wr and part == "attn"),
                delta=("none" if L == 0 else "mlp") if part == "attn" else ("none" if wr else "attn"),
                new_prefix=part == "mlp" and wr,
                norm_weight=(1 + 0.1 * rnd()).bfloat16(), qk_weight=rnd(0.05), output_norm_weight=(1 + 0.1 * rnd()).bfloat16(),
                eps=1e-6, output_norm_eps=1e-6))
    calls.append(dict(n=8, block_write_idx=-1, quant_dtype=None, store_prefix=False, delta="mlp", new_prefix=False,
                      norm_weight=(1 + 0.1 * rnd()).bfloat16(), qk_weight=rnd(0.05),
                      output_norm_weight=(1 + 0.1 * rnd()).bfloat16(), eps=1e-6, output_norm_eps=1e-6))
    return calls


@pytest.mark.parametrize("use_ahead", [False, True])
@pytest.mark.parametrize("T", [7, 160, 1728])
def test_attn_res_schedule(T, use_ahead):
    g = torch.Generator(device="cuda").manual_seed(T)
    calls = _calls(g)
    assert len(calls) == 187 and sum(c["quant_dtype"] is not None for c in calls) == 70
    sched = AttnResSchedule([{k: v for k, v in c.items() if k not in ("delta", "new_prefix")} for c in calls], use_ahead=use_ahead)
    if use_ahead:
        assert sched.role.count("produce") == 4 and sched.role.count("consume") == 86  # one group per n >= 5
    x0 = (torch.randn(T, H, generator=g, device="cuda") * 4).bfloat16()
    attn_o = torch.randn(T, H, generator=g, device="cuda").bfloat16()
    mlp_o = torch.randn(T, H, generator=g, device="cuda").bfloat16()
    blocks = torch.zeros(T, 8, H, device="cuda", dtype=torch.bfloat16)
    prefix, rprefix, rblocks = x0.clone(), x0.clone(), blocks.clone()
    for i, c in enumerate(calls):
        if c["new_prefix"]:
            prefix.copy_(attn_o); rprefix = attn_o.clone()
        d = None if c["delta"] == "none" else (attn_o if c["delta"] == "attn" else mlp_o)
        qd = c["quant_dtype"]
        rp, rb, rout, rscale = ref.ref_attn_res(rprefix, d, rblocks, c["norm_weight"], c["qk_weight"], c["output_norm_weight"],
                                                c["n"], c["block_write_idx"], c["eps"], c["output_norm_eps"], qd)
        res = sched(i, prefix, d, blocks)
        out, scale = res if qd is not None else (res, None)
        assert torch.equal(blocks, rb), f"blocks after call {i}"
        if c["store_prefix"]:
            torch.testing.assert_close(prefix, rp, rtol=0, atol=0, msg=f"prefix after call {i}")
        ref._check(out, scale, rout, rscale)
        rprefix, rblocks = (rp if c["store_prefix"] else rprefix), rb
