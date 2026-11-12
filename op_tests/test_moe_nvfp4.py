# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""NVFP4 MoE (aiter.ops.moe_nvfp4) vs a dequantized fp32 torch reference.

Covers moe_nvfp4 and the fused_moe path (w1/w2_global_scale) on the MiniMax-M3.1
shapes (TP4 / TP8, E 128, topk 4), the main path and the fallback (T 2048), bf16
and f16 activations, expert contributions that cancel beyond the f16 range, a
dispatch CSV (AITER_CONFIG_MOE_NVFP4) naming a kernel outside the prebuilt module
(per-kernel JIT) and the fused_moe errors.
"""

import os
import subprocess
import sys

import pytest
import torch

from aiter import ActivationType, NvFp4MoeWeights, dtypes, moe_nvfp4, uses_fallback
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.shuffle import shuffle_scale_nvfp4_moe, shuffle_weight
from aiter.utility.fp4_utils import mxfp4_to_f32

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm device"),
    pytest.mark.skipif(get_gfx() != "gfx950", reason="NVFP4 MoE kernels need gfx950"),
]

H, E, TOPK = 6144, 128, 4
SHAPES = {"tp4": 768, "tp8": 384}
TOKENS = [1, 8, 64, 1024, 2048]
DTYPES = [torch.bfloat16, torch.float16]
TOL = 1e-2
# Shipped TP4 main-path stage-1 kernel without _xcdremap: not in the prebuilt module
JIT_S1 = "nvfp4_moe_s1_32x256_xdma_waves8_wbuf2x128_xbuf3_gs6"
JIT_S2 = (
    "nvfp4_moe_s2_32x256_xres_waves4_wbuf3x256_kdim768_gs6_upfront_occ3_staged_outbf16"
)


def _weights(I, seed=0):
    """Raw NVFP4 weights: packed E2M1 [E, N, K/2], E4M3 scale bits [E, N, K/16]
    (1.875 .. 448), fp32 1 / global scale [E, 2] and [E]."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def u8(lo, hi, *s):
        t = torch.randint(lo, hi, s, generator=g, device="cuda", dtype=torch.int32)
        return t.to(torch.uint8)

    w13, sc13 = u8(0, 256, E, 2 * I, H // 2), u8(0x3F, 0x7F, E, 2 * I, H // 16)
    w2, sc2 = u8(0, 256, E, H, I // 2), u8(0x3F, 0x7F, E, H, I // 16)
    g13 = (torch.rand(E, 2, generator=g, device="cuda") * 0.5 + 0.5) * 2e-4
    g2 = (torch.rand(E, generator=g, device="cuda") * 0.5 + 0.5) * 2e-4
    return w13, sc13, w2, sc2, g13, g2


@pytest.fixture(scope="module", params=list(SHAPES))
def shape(request):
    I = SHAPES[request.param]
    raw = _weights(I)
    w13, sc13, w2, sc2, g13, g2 = raw
    f8 = torch.float8_e4m3fn
    fused = {
        "w1": shuffle_weight(w13.view(dtypes.fp4x2), (16, 32)),
        "w2": shuffle_weight(w2.view(dtypes.fp4x2), (16, 32)),
        "w1_scale": shuffle_scale_nvfp4_moe(sc13.view(f8)),
        "w2_scale": shuffle_scale_nvfp4_moe(sc2.view(f8)),
        "w1_global_scale": g13,
        "w2_global_scale": g2,
    }
    return I, raw, NvFp4MoeWeights(*raw), fused


def _dequant(w, sc, gm):
    """[N, K/2] packed, [N, K/16] scale bits, scalar multiplier -> fp32 [N, K]."""
    v = mxfp4_to_f32(w)
    N, K = v.shape
    s = sc.view(torch.float8_e4m3fn).float().view(N, K // 16, 1)
    return (v.view(N, K // 16, 16) * s).view(N, K) * gm


def _reference(x, raw, tid, tw):
    """fp32 MoE on the f16-rounded input and f16-rounded stage-1 output (the
    kernels' activation dtype), SwiGLU-OAI (alpha 1.702, limit 7)."""
    w13, sc13, w2, sc2, g13, g2 = raw
    I = w13.shape[1] // 2
    xf = x.half().float()
    out = torch.zeros(x.shape[0], H, device=x.device)
    ids, wts = tid.view(-1), tw.view(-1).float()
    for e in ids.unique().tolist():
        rows = (ids == e).nonzero().view(-1)
        w = torch.cat([_dequant(w13[e, :I], sc13[e, :I], g13[e, 0]),
                       _dequant(w13[e, I:], sc13[e, I:], g13[e, 1])])  # fmt: skip
        h = xf[rows // TOPK] @ w.t()
        gate, up = h[:, :I].clamp(max=7.0), h[:, I:].clamp(-7.0, 7.0)
        a2 = (gate * torch.sigmoid(1.702 * gate) * (up + 1)).half().float()
        y = a2 @ _dequant(w2[e], sc2[e], g2[e]).t()
        out.index_add_(0, rows // TOPK, y * wts[rows].view(-1, 1))
    return out


def _inputs(T, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, H, generator=g, device="cuda").to(dtype)
    scores = torch.randn(T, E, generator=g, device="cuda").softmax(-1)
    tw, tid = torch.topk(scores, TOPK, dim=-1)
    return x, tid.to(torch.int32), tw.float()


def _relf(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("T", TOKENS)
def test_moe_nvfp4(shape, T, dtype):
    I, raw, W, fused = shape
    x, tid, tw = _inputs(T, dtype, seed=T)
    ref = _reference(x, raw, tid, tw)
    assert uses_fallback(T, TOPK, E) == (T == 2048)
    for name, out in (
        ("moe_nvfp4", moe_nvfp4(W, x, tid, tw)),
        ("fused_moe", fused_moe(x, topk_weight=tw, topk_ids=tid,
                                activation=ActivationType.Swiglu, **fused)),
    ):  # fmt: skip
        assert out.dtype == dtype and out.shape == (T, H), name
        assert torch.isfinite(out).all(), name
        err = _relf(out, ref)
        assert err < TOL, f"{name} I={I} T={T} {dtype}: relF {err:.2e}"


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_cancellation(dtype, fallback):
    """Two experts with opposite down weights: each contribution (+-86016) is above
    the f16 range, their sum is 0 (f16 x: no inf - inf in the output atomics)."""
    I, E2 = SHAPES["tp8"], 2

    def full(v, *s):
        return torch.full(s, v, dtype=torch.uint8, device="cuda")

    # fp4 0x2 = 1, 0xA = -1, E4M3 0x38 = 1: stage 1 gives the SwiGLU-OAI bound 56
    w2 = torch.cat([full(0x22, 1, H, I // 2), full(0xAA, 1, H, I // 2)])
    W = NvFp4MoeWeights(
        full(0x22, E2, 2 * I, H // 2), full(0x38, E2, 2 * I, H // 16),
        w2, full(0x38, E2, H, I // 16),
        torch.ones(E2, 2, device="cuda"), torch.full((E2,), 8.0, device="cuda"),
    )  # fmt: skip
    x = torch.ones(1, H, dtype=dtype, device="cuda")
    tid = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    tw = torch.full((1, 2), 0.5, device="cuda")
    out = moe_nvfp4(W, x, tid, tw, fallback=fallback)
    assert out.dtype == dtype and (out == 0).all(), out.float().abs().max()


def _check_jit_csv():
    """Run in a subprocess with AITER_CONFIG_MOE_NVFP4 set (read at import)."""
    from aiter.ops import moe_nvfp4 as op

    raw = _weights(SHAPES["tp4"])
    W = NvFp4MoeWeights(*raw)
    assert JIT_S1 not in op._KERNELS
    kernel1, kernel2 = op._pick(W, 1000, TOPK, "bf16", False)
    assert kernel1["name"] == JIT_S1 and kernel1["build"] is not None, kernel1["name"]
    assert kernel2["name"] == JIT_S2, kernel2["name"]
    x, tid, tw = _inputs(1000, torch.bfloat16, seed=1)
    err = _relf(moe_nvfp4(W, x, tid, tw), _reference(x, raw, tid, tw))
    assert err < TOL, f"JIT CSV kernel: relF {err:.2e}"


def test_jit_csv(tmp_path):
    path = tmp_path / "moe_nvfp4_dispatch.csv"
    path.write_text(
        "cu_num,H,I,E,topk,x_dtype,out_dtype,T,stage1,stage2\n"
        f"{get_cu_num()},{H},768,{E},{TOPK},bf16,bf16,1024,{JIT_S1},{JIT_S2}\n"
    )
    env = dict(os.environ, AITER_CONFIG_MOE_NVFP4=str(path))
    r = subprocess.run([sys.executable, __file__, "--jit-csv"], env=env, check=False,
                       capture_output=True, text=True, timeout=1800)  # fmt: skip
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]


# fused_moe arguments the NVFP4 path rejects (8 tokens), from its fused_moe inputs
ERRORS = {
    "expert_mask": lambda f: {"expert_mask": torch.ones(E, device="cuda").int()},
    "one global scale": lambda f: {"w2_global_scale": None},
    "unshuffled w1": lambda f: {"w1": f["w1"].view(torch.uint8).view(dtypes.fp4x2)},
    "scale layout": lambda f: {"w1_scale": f["w1_scale"].reshape(-1)[:1024]},
    "scale dtype": lambda f: {"w2_scale": f["w2_scale"].view(torch.int16)},
    "herd": lambda f: {"herd_topk": 3},
    "bias": lambda f: {"bias2": torch.zeros(E, H, device="cuda")},
    "doweight_stage1": lambda f: {"doweight_stage1": True},
    "a1_scale": lambda f: {"a1_scale": torch.ones(8, H // 32, device="cuda")},
    "num_local_tokens": lambda f: {
        "num_local_tokens": torch.full((1,), 8, dtype=torch.int32, device="cuda")
    },
    "padding": lambda f: {"hidden_pad": 128},
    "gate_mode": lambda f: {"gate_mode": "interleave"},
    "activation": lambda f: {"activation": ActivationType.Silu},
    "swiglu_limit": lambda f: {"swiglu_limit": 10.0},
    "dtype": lambda f: {"dtype": torch.float16},
}


@pytest.mark.parametrize("case", list(ERRORS))
def test_fused_moe_errors(shape, case):
    fused = shape[3]
    x, tid, tw = _inputs(8, torch.bfloat16, seed=0)
    kw = dict(fused, activation=ActivationType.Swiglu)
    kw.update(ERRORS[case](fused))
    w1, w2 = kw.pop("w1"), kw.pop("w2")
    with pytest.raises(ValueError):
        fused_moe(x, w1, w2, tw, tid, **kw)


if __name__ == "__main__":
    if sys.argv[1:] == ["--jit-csv"]:
        _check_jit_csv()
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
