# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Inputs and FP32 references shared by HK MoE tests and benchmarks."""

import torch

FP8 = torch.float8_e4m3fn


def requant_pow2(w):
    """[E, n, k] float -> fp8 with power-of-two 128x128 block scales [E, n/128, k/128]."""
    E, n, k = w.shape
    b = w.float().view(E, n // 128, 128, k // 128, 128)
    s = torch.exp2(
        torch.ceil(
            torch.log2(b.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12) / 448.0)
        )
    )
    return (b / s).to(FP8).view(E, n, k), s.view(E, n // 128, k // 128).contiguous()


def dequant(q, s):
    E, n, k = q.shape
    b = q.float().view(E, n // 128, 128, k // 128, 128)
    return (b * s.view(E, n // 128, 1, k // 128, 1)).view(E, n, k)


def activation_roundtrip(x):
    """FP32 values after the kernel's power-of-two 1x128 E4M3 quantization.

    Both ``quant_tok16_block`` and ``epilogue_silu_quant`` choose
    ``2**ceil(log2(max(amax, 1e-10) / 448))`` independently for each row/block.
    PyTorch's E4M3 cast provides round-to-nearest-even, including subnormals.
    This oracle uses no HK kernel, packed scale format, or shuffled weights.
    """
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // 128, 128)
    scale = torch.exp2(
        torch.ceil(
            torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0)
        )
    )
    return ((blocks / scale).to(FP8).float() * scale).reshape(x.shape)


def reference(x, W13, W2, tw, ids, limit, *, quantized=False):
    """FP32 MoE oracle with optional HK activation quantization/storage rounds.

    The default is the original ideal-activation FP32 reference. Benchmarks
    retain that common reference when comparing different quantization schemes.
    ``quantized=True`` independently models the HK kernel's two FP8 activation
    rounds, BF16 expert outputs, and BF16 final output. GEMMs, SwiGLU, and routed
    accumulation still use PyTorch FP32 arithmetic, exposing kernel arithmetic,
    routing, and clamp defects separately from expected quantization error.
    """
    out = torch.zeros(x.shape[0], W2.shape[1], device=x.device)
    inputs = activation_roundtrip(x) if quantized else x.float()
    for e in torch.unique(ids).tolist():
        if e < 0 or e >= W13.shape[0]:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        g, u = (inputs[tok] @ W13[e].t()).chunk(2, dim=-1)
        if limit > 0:
            g, u = g.clamp(max=limit), u.clamp(-limit, limit)
        act = torch.nn.functional.silu(g) * u
        if quantized:
            act = activation_roundtrip(act)
        expert_out = act @ W2[e].t()
        if quantized:
            expert_out = expert_out.to(torch.bfloat16).float()
        out.index_add_(0, tok, expert_out * tw[tok, slot, None])
    return out.to(torch.bfloat16).float() if quantized else out


def make_weights(E):
    from aiter.benchmark_data_init import make_generator
    from aiter.ops.hk_fp8_moe import HIDDEN, INTER

    gen = make_generator(0)
    w13, s13 = requant_pow2(
        torch.randn(E, 2 * INTER, HIDDEN, device="cuda", generator=gen) * 0.02
    )
    w2, s2 = requant_pow2(
        torch.randn(E, HIDDEN, INTER, device="cuda", generator=gen) * 0.02
    )
    return w13, s13, w2, s2


def make_routing(T, E, topk=None):
    from aiter.benchmark_data_init import make_generator
    from aiter.ops.hk_fp8_moe import HIDDEN

    gen = make_generator(T)
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen)
    shared = topk is None and E == 289
    routed_experts, routed_topk = E - int(shared), topk or 8
    tw, ids = torch.topk(
        torch.randn(
            T, max(routed_experts, routed_topk), device="cuda", generator=gen
        ).softmax(-1),
        routed_topk,
        dim=-1,
    )
    ids %= routed_experts  # Exercise repeated selections when E < top-k.
    if shared:
        ids = torch.cat([ids, torch.full_like(ids[:, :1], E - 1)], dim=1)
        tw = torch.cat([tw, torch.ones_like(tw[:, :1])], dim=1)
    return x, ids.to(torch.int32), tw
