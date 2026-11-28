# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3.1 TP8 A4W4 MoE prefill (3584 / 384, 896 experts, top-16, per_1x32
MXFP4, SiTUv2 beta 4 / linear_beta 25; ``AITER_SITUV2_A4W4=1`` as the model's
recipe sets it), random MXFP4 weights:

  - a row-strided activation (a column slice of a wider GEMM output) gives the
    dense input's output bit for bit (``AITER_MOE_ROW_STRIDED_QUANT_INPUT``);
  - 2049..3584 tokens on the 3584-token tier row give the 4096-token row's
    output bit for bit (the tier only re-schedules the same arithmetic);
  - the shared-exponential SiTUv2 epilogue (``AITER_MOE_SITUV2_FAST``) stays
    within a small bound of the exact epilogue.

    pytest op_tests/test_moe_kimik3_a4w4_prefill.py
"""

import math

import pytest
import torch

E, TOPK, D, INTER = 896, 16, 3584, 384
BETA, LBETA = 4.0, 25.0

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _gfx950():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx950"


@pytest.fixture(scope="module")
def weights():
    if not _gfx950():
        pytest.skip("Kimi-K3.1 A4W4 rows are tuned for gfx950")
    import aiter
    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

    g = torch.Generator(device="cuda").manual_seed(0)

    def make(rows, cols, gate_up):
        quant = aiter.get_torch_quant(QuantType.per_1x32)
        qs, ss = [], []
        for _ in range(E // 32):
            w = torch.randn(
                32, rows, cols, dtype=torch.bfloat16, device="cuda", generator=g
            )
            q, s = quant(w.mul_(1.0 / math.sqrt(cols)), quant_dtype=dtypes.fp4x2)
            qs.append(q.view(32, rows, cols // 2))
            ss.append(s.view(32 * rows, -1))
        q, s = torch.cat(qs), torch.cat(ss)
        w = moe_shuffle_weight(
            q.view(dtypes.fp4x2), experts_cnt=E, is_guinterleave=False, gate_up=gate_up
        )
        s = moe_shuffle_scale(s, E, is_guinterleave=False, gate_up=gate_up)
        w.is_shuffled = True
        return w, s

    w1, s1 = make(2 * INTER, D, True)
    w2, s2 = make(D, INTER, False)
    return w1, w2, s1, s2


def _call(x, M, W, seed=1):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe, fused_topk

    g = torch.Generator(device="cuda").manual_seed(seed * 1000 + M)
    score = torch.randn(M, E, dtype=torch.bfloat16, device="cuda", generator=g)
    score[:, :8] += 2.0
    tw, ti = fused_topk(
        torch.empty(M, D, dtype=torch.bfloat16, device="cuda"), score, TOPK, True
    )
    w1, w2, s1, s2 = W
    out = fused_moe(
        x,
        w1,
        w2,
        tw.float().contiguous(),
        ti.to(torch.int32).contiguous(),
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        w1_scale=s1,
        w2_scale=s2,
        doweight_stage1=False,
        gate_mode="separated",
        swiglu_limit=0.0,
        beta=BETA,
        linear_beta=LBETA,
    )
    torch.cuda.synchronize()
    return out


def _x(M, lead):
    g = torch.Generator(device="cuda").manual_seed(7 + M)
    wide = torch.randn(M, lead + D, dtype=torch.bfloat16, device="cuda", generator=g)
    return wide[:, lead:]


@pytest.fixture(autouse=True)
def _recipe_env(monkeypatch):
    monkeypatch.setenv("AITER_SITUV2_A4W4", "1")


@pytest.mark.parametrize("M", [2304, 8193])
def test_row_strided_input_is_bitwise_dense(weights, monkeypatch, M):
    import aiter.fused_moe as fm

    x = _x(M, 896 if M > 4096 else 1536 + 896)
    assert not x.is_contiguous()
    monkeypatch.setattr(fm, "_MOE_ROW_STRIDED_QUANT_INPUT", True)
    strided = _call(x, M, weights)
    dense = _call(x.contiguous(), M, weights)
    assert torch.equal(strided, dense)


@pytest.mark.parametrize("M", [2049, 3000, 3584])
def test_tier_row_is_bitwise_the_bucket_row(weights, monkeypatch, M):
    import aiter.fused_moe as fm

    x = _x(M, 0).contiguous()
    assert fm._tuned_M(M) == 3584
    tier = _call(x, M, weights)
    monkeypatch.setattr(fm, "_tuned_M", fm.get_padded_M)
    fm.get_2stage_cfgs.cache_clear()
    try:
        bucket = _call(x, M, weights)
    finally:
        fm.get_2stage_cfgs.cache_clear()
    assert torch.equal(tier, bucket)


@pytest.mark.parametrize("M", [2304, 16384])
def test_situv2_fast_epilogue_error_is_bounded(weights, monkeypatch, M):
    import aiter.ops.flydsl.mxfp4_gemm1_kernels as g1

    x = _x(M, 0).contiguous()
    monkeypatch.setattr(g1, "_SITUV2_FAST", True)
    fast = _call(x, M, weights).float()
    monkeypatch.setattr(g1, "_SITUV2_FAST", False)
    exact = _call(x, M, weights).float()
    rel = float((fast - exact).norm() / exact.norm())
    assert float((fast - exact).abs().max()) < 0.05
    assert rel < 1e-2, rel


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
