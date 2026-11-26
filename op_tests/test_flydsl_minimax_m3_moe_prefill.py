# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 FlyDSL W4A4 MXFP4 prefill MoE chain (gfx950).

`a4w4_prefill_moe` = sort -> gemm1 (gate/up, SwiGLU-OAI, MXFP4 requant) -> gemm2
(routing-weighted partials) -> bf16 reduce, at the TP4 shapes of MiniMax-M3
(hidden 6144, intermediate 768, 128 experts, topk 4).

Reference: a torch MoE on the dequantized values, nothing from the kernel under
test. Weights and activations are quantized by aiter's `per_1x32_f4_quant`
(RoundUp) and dequantized with `fp4_utils`; per expert the reference runs
x @ W13^T -> SwiGLU-OAI (alpha 1.702, limit 7) -> bf16 -> MXFP4 round trip (the
kernel requantizes the intermediate) -> @ W2^T, weighted by the routing weight.

Run:
    pytest op_tests/test_flydsl_minimax_m3_moe_prefill.py
"""

import subprocess
import sys

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="MXFP4 MFMA prefill kernels need gfx950"
)

HIDDEN, INTER, EXPERTS, TOPK = 6144, 768, 128, 4
ALPHA, LIMIT = 1.702, 7.0


def _quant(x):
    """bf16 [..., K] -> (fp4 bytes [..., K/2], e8m0 bytes [..., K/32])."""
    from aiter.ops.quant import per_1x32_f4_quant

    q, s = per_1x32_f4_quant(x.reshape(-1, x.shape[-1]))
    k = x.shape[-1]
    return (
        q.view(torch.uint8).reshape(*x.shape[:-1], k // 2),
        s.view(torch.uint8).reshape(*x.shape[:-1], k // 32),
    )


def _dequant(q, s):
    """Inverse of `_quant` -> fp32 [..., K]."""
    k = q.shape[-1] * 2
    f = fp4_utils.mxfp4_to_f32(q.reshape(-1, k // 2).clone())
    e = fp4_utils.e8m0_to_f32(s.reshape(-1, k // 32)).repeat_interleave(32, dim=-1)
    return (f * e).reshape(*q.shape[:-1], k)


def _swiglu_oai(g, u):
    g = g.clamp(max=LIMIT)
    u = u.clamp(-LIMIT, LIMIT)
    return g * torch.sigmoid(g * ALPHA) * (u + 1.0)


def _reference(x_q, x_s, w13, w2, topk_ids, topk_weights):
    """w13 / w2: (fp4, e8m0) of the raw [E, 2I, H] / [E, H, I] weights."""
    x = _dequant(x_q, x_s)
    out = torch.zeros_like(x)
    for e in range(EXPERTS):
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        gu = x[tok] @ _dequant(w13[0][e], w13[1][e]).T
        h = _swiglu_oai(gu[:, :INTER], gu[:, INTER:]).to(torch.bfloat16)
        h = _dequant(*_quant(h))
        y = h @ _dequant(w2[0][e], w2[1][e]).T
        out.index_add_(0, tok, y * topk_weights[tok, slot, None])
    return out


def _make_inputs(n_tokens, seed=0, device="cuda"):
    from aiter.ops.shuffle import shuffle_scale

    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, scale):
        return (torch.randn(*shape, device=device, generator=g) * scale).to(
            torch.bfloat16
        )

    x = randn(n_tokens, HIDDEN, scale=1.0)
    w13 = _quant(randn(EXPERTS, 2 * INTER, HIDDEN, scale=HIDDEN**-0.5))
    w2 = _quant(randn(EXPERTS, HIDDEN, INTER, scale=INTER**-0.5))
    logits = torch.rand(n_tokens, EXPERTS, device=device, generator=g)
    topk_ids = logits.topk(TOPK, dim=1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.rand(n_tokens, TOPK, device=device, generator=g), dim=1
    )

    def kernel_form(w):  # the layouts of the kernel docstrings
        q, s = w
        return (
            shuffle_weight(q.view(dtypes.fp4x2), layout=(16, 16)).view(torch.uint8),
            shuffle_scale(s.reshape(-1, s.shape[-1])),
        )

    return x, w13, w2, topk_ids, topk_weights, kernel_form(w13), kernel_form(w2)


def _check(n_tokens, device="cuda"):
    from aiter.ops.flydsl.kernels.minimax_m3_moe_prefill.moe_mxfp4.prefill import (
        a4w4_prefill_moe,
    )

    x, w13, w2, ids, wts, (w13_k, w13_ks), (w2_k, w2_ks) = _make_inputs(
        n_tokens, device=device
    )
    x_q, x_s = _quant(x)
    got = a4w4_prefill_moe(
        (x_q, x_s),
        w13_k,
        w13_ks,
        w2_k,
        w2_ks,
        wts,
        ids,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_experts=EXPERTS,
    )
    torch.cuda.synchronize(device)
    # the kernel returns bf16: round the reference the same way before comparing
    ref = _reference(x_q, x_s, w13, w2, ids, wts).to(torch.bfloat16).float()

    assert got.shape == (n_tokens, HIDDEN) and got.dtype == torch.bfloat16
    assert got.device == x.device
    got = got.float()
    rel = ((got - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
    print(f"\n{device} n_tokens {n_tokens}: rel L2 err {rel:.5f}, cos {cos.item():.6f}")
    assert rel < 0.03, rel
    assert cos > 0.999, cos


@pytest.mark.parametrize("n_tokens", [3072, 5000, 16384, 65536])  # 16384: block_m 256
def test_a4w4_prefill_moe(n_tokens):
    _check(n_tokens)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_a4w4_prefill_moe_non_current_device():
    # tensors on cuda:1 while cuda:0 is current: a fresh process, as FlyDSL binds a
    # compiled kernel to the first device that runs it
    code = (
        "import sys; sys.path.insert(0, 'op_tests'); "
        "import test_flydsl_minimax_m3_moe_prefill as t; t._check(3072, 'cuda:1')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a4w4_prefill_moe_rejects_bad_inputs():
    from aiter.ops.flydsl.kernels.minimax_m3_moe_prefill.moe_mxfp4.prefill import (
        MIN_PREFILL_TOKENS,
        a4w4_prefill_moe,
    )

    n = MIN_PREFILL_TOKENS
    dev = "cuda"
    q = torch.zeros(n, HIDDEN // 2, dtype=torch.uint8, device=dev)
    s = torch.zeros(n, HIDDEN // 32, dtype=torch.uint8, device=dev)
    w13 = torch.zeros(EXPERTS, 2 * INTER, HIDDEN // 2, dtype=torch.uint8, device=dev)
    w2 = torch.zeros(EXPERTS, HIDDEN, INTER // 2, dtype=torch.uint8, device=dev)
    sc = torch.zeros(1, dtype=torch.uint8, device=dev)
    ids = torch.zeros(n, TOPK, dtype=torch.int32, device=dev)
    wts = torch.zeros(n, TOPK, dtype=torch.float32, device=dev)
    kw = {"hidden_size": HIDDEN, "intermediate_size": INTER, "num_experts": EXPERTS}
    args = ((q, s), w13, sc, w2, sc, wts, ids)

    short_out = torch.empty(n - 1, HIDDEN, dtype=torch.bfloat16, device=dev)
    with pytest.raises(AssertionError):
        a4w4_prefill_moe(*args, **kw, out=short_out)
    with pytest.raises(AssertionError):
        a4w4_prefill_moe(
            *args, **kw, out=torch.empty(n, HIDDEN, dtype=torch.float32, device=dev)
        )
    with pytest.raises(AssertionError):
        a4w4_prefill_moe(*args[:3], w2[:-1], *args[4:], **kw)
