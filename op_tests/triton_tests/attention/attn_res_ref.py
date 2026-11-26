# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Torch fp32 reference for the gfx1250 Gluon AttnRes tests (bf16 rounding points of the vLLM Triton op).
import torch


H = 7168
FP8 = torch.float8_e4m3fn


def ref_attn_res(prefix, delta, blocks, nw, qkw, onw, n, widx, eps, oeps, quant_dtype=None):
    """Returns (prefix_after, blocks_after, out, scale) without touching the inputs."""
    prefix = prefix.clone()
    blocks = blocks.clone()
    p = prefix.float()
    if delta is not None:
        p = (p + delta.float()).to(prefix.dtype).float()
        prefix.copy_(p.to(prefix.dtype))
    if widx >= 0:
        blocks[:, widx] = p.to(blocks.dtype)
    if n == 0:
        mixed = p
    else:
        src = torch.cat([blocks[:, :n].float(), p[:, None]], dim=1)  # [T, n+1, H]
        w = nw.float() * qkw.float()
        rstd = torch.rsqrt(src.pow(2).mean(-1) + eps)
        logits = (src * w).sum(-1) * rstd
        mixed = (torch.softmax(logits, dim=-1)[..., None] * src).sum(1)
    out = mixed
    if onw is not None:
        out = mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + oeps) * onw.float()
    scale = None
    if quant_dtype is not None:
        qmax = torch.finfo(quant_dtype).max
        out = out.to(prefix.dtype).float()
        scale = out.abs().amax(-1, keepdim=True).clamp(min=1e-10) / qmax
        out = (out / scale).clamp(-qmax, qmax).to(quant_dtype)
    else:
        out = out.to(prefix.dtype)
    return prefix, blocks, out, scale


def make_inputs(T, n, has_delta, has_onorm, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    def r(*s, scale=1.0):
        return (torch.randn(*s, generator=g, device="cuda") * scale).to(torch.bfloat16)
    prefix = r(T, H, scale=4.0)
    delta = r(T, H) if has_delta else None
    blocks = r(T, 8, H, scale=4.0)
    nw = (1.0 + 0.1 * torch.randn(H, generator=g, device="cuda")).to(torch.bfloat16)
    qkw = r(H, scale=0.05)
    onw = (1.0 + 0.1 * torch.randn(H, generator=g, device="cuda")).to(torch.bfloat16) if has_onorm else None
    return prefix, delta, blocks, nw, qkw, onw


def _check(out, scale, ref_out, ref_scale):
    if ref_scale is not None:
        torch.testing.assert_close(scale, ref_scale, rtol=1e-2, atol=1e-6)
        # dequantized comparison: allow one fp8 ulp at the row's scale
        torch.testing.assert_close((out.float() * scale), (ref_out.float() * ref_scale),
                                   rtol=0.125, atol=2 * ref_scale.max().item())
    else:
        torch.testing.assert_close(out.float(), ref_out.float(), rtol=2e-2, atol=2e-2)


