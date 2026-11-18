# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL a16w4 (bf16 A x MXFP4 W) SwiGLU MoE: the fused BM16 sort path.

Decode-sized batches whose stage1 is a BM16 FlyDSL kernel take their routes from the
fused sort that also zeroes the atomic stage2 output, instead of the generic sort.
The result must match the generic-sort result (up to the bf16 atomic-add order) and
a torch reference.

    python -m pytest op_tests/test_flydsl_moe_a16w4_fused_sort.py -q
"""

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_mod
from aiter import ActivationType, QuantType
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="needs gfx950",
)

HIDDEN, INTER, EXPERTS, TOPK = 6144, 768, 129, 5
SWIGLU_LIMIT = 7.0
FP4_LUT = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _dequant(w, scale, k):
    lut = torch.tensor(FP4_LUT, device=w.device)
    f = torch.stack([lut[(w & 15).long()], lut[(w >> 4).long()]], -1)
    f = f.reshape(*w.shape[:-1], k // 32, 32)
    sc = torch.exp2(scale.float().reshape(*w.shape[:-1], k // 32) - 127)
    return (f * sc.unsqueeze(-1)).reshape(*w.shape[:-1], k)


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator(device="cuda").manual_seed(5)
    E = EXPERTS
    w1 = torch.randint(
        0,
        256,
        (E, 2 * INTER, HIDDEN // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w2 = torch.randint(
        0, 256, (E, HIDDEN, INTER // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    s1 = torch.randint(
        118,
        124,
        (E * 2 * INTER, HIDDEN // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    s2 = torch.randint(
        118,
        124,
        (E * HIDDEN, INTER // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    ref = (_dequant(w1, s1, HIDDEN), _dequant(w2, s2, INTER))
    kernel = (
        moe_shuffle_weight(
            w1.view(torch.float4_e2m1fn_x2),
            experts_cnt=E,
            is_guinterleave=False,
            gate_up=True,
        ),
        moe_shuffle_weight(
            w2.view(torch.float4_e2m1fn_x2),
            experts_cnt=E,
            is_guinterleave=False,
            gate_up=False,
        ),
        moe_shuffle_scale(
            s1.view(torch.float8_e8m0fnu), E, is_guinterleave=False, gate_up=True
        ),
        moe_shuffle_scale(
            s2.view(torch.float8_e8m0fnu), E, is_guinterleave=False, gate_up=False
        ),
    )
    return kernel, ref


def _run(x, topk_w, topk_ids, kernel):
    w1, w2, s1, s2 = kernel
    return fused_moe_mod.fused_moe(
        x,
        w1,
        w2,
        topk_w,
        topk_ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=s1,
        w2_scale=s2,
        swiglu_limit=SWIGLU_LIMIT,
        gate_mode="separated",
    )


def _torch_ref(x, topk_w, topk_ids, w1, w2):
    out = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    for t in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            gu = w1[e] @ x[t].float()
            g, u = gu[:INTER].clamp(max=SWIGLU_LIMIT), gu[INTER:].clamp(
                -SWIGLU_LIMIT, SWIGLU_LIMIT
            )
            h = ((u + 1) * g * torch.sigmoid(1.702 * g)).bfloat16().float()
            out[t] += topk_w[t, j] * (w2[e] @ h)
    return out


@pytest.mark.parametrize("tokens", [9, 16, 17, 32, 64, 100])
@pytest.mark.parametrize("skewed", [False, True])
def test_fused_bm16_sort_matches_generic_sort(weights, monkeypatch, tokens, skewed):
    kernel, (ref_w1, ref_w2) = weights
    g = torch.Generator(device="cuda").manual_seed(tokens)
    x = (torch.randn(tokens, HIDDEN, device="cuda", generator=g) * 0.5).bfloat16()
    topk_w = torch.rand(tokens, TOPK, device="cuda", generator=g)
    pool = 6 if skewed else 128
    routed = torch.stack(
        [
            torch.randperm(pool, device="cuda", generator=g)[: TOPK - 1]
            for _ in range(tokens)
        ]
    )
    shared = torch.full((tokens, 1), EXPERTS - 1, device="cuda")
    topk_ids = torch.cat([routed, shared], 1).int()

    fused = _run(x, topk_w, topk_ids, kernel).float()
    monkeypatch.setattr(fused_moe_mod, "_FUSED_BM16_SORT_MAX_ROUTED_ROWS", 0)
    generic = _run(x, topk_w, topk_ids, kernel).float()

    err = (fused - generic).norm() / generic.norm()
    assert err < 5e-3, f"fused vs generic sort: rel L2 {err:.2e}"
    ref = _torch_ref(x, topk_w, topk_ids, ref_w1, ref_w2)
    err = (fused - ref).norm() / ref.norm()
    assert err < 2e-2, f"fused vs torch reference: rel L2 {err:.2e}"
