# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Module-level forward / backward parity for `gated_mla_train` (Kimi-K3 gated MLA).

The reference is an eager copy of `KimiMLAAttention.forward` from
`modeling_kimi_linear.py` (Qinghao/Kimi-K3-Tiny-L13), run in fp32. The BF16 module is
the same composition with the attention core replaced by `gated_mla_train`. The output,
the input gradient and every weight gradient are compared with the fp32 reference; the
BF16 error must stay within twice that of the same eager composition run in BF16.

Weights are random at the K3 widths by default. Set AITER_K3_TINY_L13 to a Tiny-L13
checkpoint directory (`model.safetensors.index.json` plus shards 4, 8 and 12) to run the
real layers 3, 7 and 11.
"""

import itertools
import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.gated_mla_train import gated_mla_train

HIDDEN = 7168
HEADS = 96
Q_LORA = 1536
KV_LORA = 512
D_NOPE = 128
D_ROT = 64
D_V = 128
# KimiRMSNorm's default; the two MLA norms do not take config.rms_norm_eps.
NORM_EPS = 1e-6

_WEIGHT_SHAPES = {
    "q_a_proj": (Q_LORA, HIDDEN),
    "q_a_layernorm": (Q_LORA,),
    "q_b_proj": (HEADS * (D_NOPE + D_ROT), Q_LORA),
    "kv_a_proj_with_mqa": (KV_LORA + D_ROT, HIDDEN),
    "kv_a_layernorm": (KV_LORA,),
    "kv_b_proj": (HEADS * (D_NOPE + D_V), KV_LORA),
    "g_proj": (HEADS * D_V, HIDDEN),
    "o_proj": (HIDDEN, HEADS * D_V),
}

_CKPT = os.environ.get("AITER_K3_TINY_L13")


def _is_gfx950():
    try:
        return get_gfx() == "gfx950"
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _is_gfx950(), reason="gfx950 only")


def _rms_norm(x, weight):
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + NORM_EPS)
    return weight * x.to(dtype)


def _eager_core(q, kv_latent, k_rot, kv_b_weight, gate, cu_seqlens, max_seqlen):
    """Expansion, `eager_attention_forward` (causal) and the output gate, per sequence."""
    t = q.shape[0]
    kv = F.linear(kv_latent, kv_b_weight).view(t, HEADS, D_NOPE + D_V)
    k_nope, v = kv.split([D_NOPE, D_V], dim=-1)
    k = torch.cat([k_nope, k_rot.unsqueeze(1).expand(t, HEADS, D_ROT)], dim=-1)
    scale = (D_NOPE + D_ROT) ** -0.5
    outs = []
    for b in range(cu_seqlens.numel() - 1):
        rows = slice(int(cu_seqlens[b]), int(cu_seqlens[b + 1]))
        qs, ks, vs = (x[rows].transpose(0, 1) for x in (q, k, v))
        s = qs.shape[1]
        scores = torch.einsum("hqd,hkd->hqk", qs, ks) * scale
        mask = torch.full((s, s), float("-inf"), dtype=q.dtype, device=q.device)
        mask = mask.triu(1)
        probs = F.softmax(scores + mask, dim=-1, dtype=torch.float32).to(q.dtype)
        outs.append(torch.einsum("hqk,hkd->qhd", probs, vs))
    o = torch.cat(outs).reshape(t, HEADS * D_V)
    return o * gate.sigmoid()


def _mla_module(x, w, cu_seqlens, max_seqlen, core):
    """`KimiMLAAttention.forward` on packed tokens with a swappable attention core."""
    t = x.shape[0]
    q = F.linear(
        _rms_norm(F.linear(x, w["q_a_proj"]), w["q_a_layernorm"]), w["q_b_proj"]
    ).view(t, HEADS, D_NOPE + D_ROT)
    kv_latent, k_rot = F.linear(x, w["kv_a_proj_with_mqa"]).split(
        [KV_LORA, D_ROT], dim=-1
    )
    kv_latent = _rms_norm(kv_latent, w["kv_a_layernorm"])
    gate = F.linear(x, w["g_proj"])
    attn = core(q, kv_latent, k_rot, w["kv_b_proj"], gate, cu_seqlens, max_seqlen)
    return F.linear(attn, w["o_proj"])


def _random_weights(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = {}
    for name, shape in _WEIGHT_SHAPES.items():
        r = torch.randn(shape, generator=g, device="cuda")
        w[name] = 1.0 + 0.1 * r if len(shape) == 1 else r * shape[1] ** -0.5
    return w


def _tiny_l13_weights(layer):
    from safetensors import safe_open

    with open(os.path.join(_CKPT, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"language_model.model.layers.{layer}.self_attn."
    w = {}
    for name, shape in _WEIGHT_SHAPES.items():
        key = f"{prefix}{name}.weight"
        with safe_open(os.path.join(_CKPT, weight_map[key]), "pt", device="cuda") as f:
            w[name] = f.get_tensor(key).float()
        assert w[name].shape == shape, (key, tuple(w[name].shape))
    return w


def _rel_err(a, ref):
    ref = ref.float()
    return ((a.float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def _run_parity(w, seqlens, seed=0):
    torch.manual_seed(seed)
    cu = torch.tensor(
        [0, *itertools.accumulate(seqlens)], dtype=torch.int32, device="cuda"
    )
    t, max_seqlen = int(cu[-1]), max(seqlens)
    x = torch.randn(t, HIDDEN, device="cuda")
    dy = torch.randn(t, HIDDEN, device="cuda")

    def fwd_bwd(dtype, core):
        ws = {k: v.detach().to(dtype).requires_grad_() for k, v in w.items()}
        xs = x.detach().to(dtype).requires_grad_()
        y = _mla_module(xs, ws, cu, max_seqlen, core)
        y.backward(dy.to(dtype))
        return {"out": y, "dx": xs.grad, **{f"d_{k}": v.grad for k, v in ws.items()}}

    ref = fwd_bwd(torch.float32, _eager_core)
    eager = fwd_bwd(torch.bfloat16, _eager_core)
    ours = fwd_bwd(torch.bfloat16, gated_mla_train)

    failed = []
    for name, r in ref.items():
        err, eager_err = _rel_err(ours[name], r), _rel_err(eager[name], r)
        print(f"  {name:22s} rel_err {err:.3e}  bf16 eager {eager_err:.3e}")
        if err > 2 * eager_err + 1e-6:
            failed.append(name)
    assert not failed, f"error above 2x the bf16 eager composition: {failed}"


_SEQLENS = [[2048], [1000, 3000, 513]]


@pytest.mark.parametrize(
    "seqlens", _SEQLENS, ids=[f"b{len(s)}_max{max(s)}" for s in _SEQLENS]
)
def test_gated_mla_train_random_weights(seqlens):
    _run_parity(_random_weights(), seqlens)


@pytest.mark.skipif(_CKPT is None, reason="AITER_K3_TINY_L13 not set")
@pytest.mark.parametrize("layer", [3, 7, 11])
def test_gated_mla_train_tiny_l13(layer):
    _run_parity(_tiny_l13_weights(layer), [1000, 3000, 513])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
