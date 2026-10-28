# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL-side coverage for the a16w4 (bf16 A x MXFP4 W) SiTUv2 MoE kernel.

Exercises the NEW shared a16w-mix a16w4 kernel
(``aiter/ops/flydsl/kernels/moe_2stage_a16wmix``) through the production
``fused_moe`` a16w4 path (the standard 2-stage FlyDSL dispatch:
``get_2stage_cfgs`` -> ``fused_moe_2stages`` -> the a16w4 branches of
``_flydsl_moe_stage{1,2}_impl`` -> ``flydsl_a16w4_gemm{1,2}``, built via
``compile_flydsl_moe_stage{1,2}``), in the SiTUv2 / SEPARATED bf16-activation x
mxfp4-weight configuration, against a bf16 SiTUv2 torch reference with a strict
cos/logits_diff gate.

This is the explicit FlyDSL-side test complementing the routed a16w4 rows of
``op_tests/test_moe_2stage.py``. It goes through ``fused_moe`` (or
``flydsl_a16w4_gemm1/2``), NOT the removed low-level ``compile_mixed_moe_gemm1_a16w4``
API.

The ``*_fp8_mfma*`` tests cover the opt-in gfx942 mode
``AITER_FLYDSL_A16W4_FP8_MFMA=1``, which quantizes the activation and the stage-1
intermediate per row to fp8 and runs both GEMMs on fp8 MFMA.

Run:
    pytest op_tests/flydsl_tests/test_flydsl_moe_a16wfp4.py -q
"""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

_SKIP = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="CDNA (gfx942/gfx950) + FlyDSL required for a16w4 SiTUv2",
)
_SKIP_FP8_MFMA = pytest.mark.skipif(
    get_gfx() != "gfx942", reason="the a16w4 fp8 MFMA mode is gfx942-only"
)


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom)


def _situv2_problem(model_dim, inter_dim, token):
    """Random E=896 / topk=16 a16w4 problem: unshuffled MXFP4 for the reference and
    the standard GGUU preshuffle that ``fused_moe`` consumes."""
    E, topk = 896, 16
    dtype = dtypes.bf16
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, E), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = tq(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = tq(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    return {
        "E": E,
        "topk": topk,
        "inp": inp,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "w1_qt": w1_qt,
        "w2_qt": w2_qt,
        "w1_scale_e": w1_scale.view(E, inter_dim * 2, model_dim // 32),
        "w2_scale_e": w2_scale.view(E, model_dim, inter_dim // 32),
        # caller contract: standard GGUU (separated gate/up) W1 layout, matching main
        "w1_gui": shuffle_weight_a16w4(w1_qt, 16, False),
        "w2_gui": shuffle_weight_a16w4(w2_qt, 16, False),
        "w1_scale_gui": shuffle_scale_a16w4(w1_scale, E, False),
        "w2_scale_gui": shuffle_scale_a16w4(w2_scale, E, False),
    }


def _situv2_ref(p, hidden, situ_beta, situ_linear_beta, quant_inter=None):
    """bf16 SiTUv2 SEPARATED reference; ``quant_inter`` rewrites the stage-1 output."""
    token = p["inp"].shape[0]
    inter_dim = p["w2_qt"].shape[-1] * 2
    o1 = torch_moe_stage1(
        hidden,
        p["w1_qt"].view(dtypes.fp4x2),
        p["w2_qt"].view(dtypes.fp4x2),
        p["topk_weights"],
        p["topk_ids"],
        dtype=dtypes.bf16,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=p["w1_scale_e"],
        doweight=False,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    o1 = o1.view(token, p["topk"], inter_dim)
    if quant_inter is not None:
        o1 = quant_inter(o1)
    return torch_moe_stage2(
        o1,
        p["w1_qt"].view(dtypes.fp4x2),
        p["w2_qt"].view(dtypes.fp4x2),
        p["topk_weights"],
        p["topk_ids"],
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=p["w2_scale_e"],
        a2_scale=None,
        doweight=True,
    )


def _situv2_fused_moe(p, situ_beta, situ_linear_beta):
    return fused_moe(
        p["inp"],
        p["w1_gui"],
        p["w2_gui"],
        p["topk_weights"],
        p["topk_ids"],
        w1_scale=p["w1_scale_gui"],
        w2_scale=p["w2_scale_gui"],
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Situv2,
        doweight_stage1=False,
        gate_mode=GateMode.SEPARATED.value,
        beta=situ_beta,
        linear_beta=situ_linear_beta,
    )


def _fp8_rows(x):
    """Per-row e4m3fnuz quantize -> f32 dequantize, with the torch quantizer."""
    q, s = aiter.get_torch_quant(QuantType.per_Token)(
        x.reshape(-1, x.shape[-1]), quant_dtype=dtypes.fp8
    )
    return (q.float() * s).view(x.shape)


@_SKIP
@pytest.mark.parametrize("model_dim,inter_dim", [(3584, 512), (3584, 384)])
@pytest.mark.parametrize("token", [1, 16, 128])
# (1,1) = plain tanh; (4,25) = kimi-k3 production betas, which exercise the
# runtime situ_beta/situ_linear_beta f32 args (beta is NOT a compile key, so this
# adds no extra kernel compiles).
@pytest.mark.parametrize("situ_beta,situ_linear_beta", [(1.0, 1.0), (4.0, 25.0)])
def test_flydsl_a16wfp4_situv2_e2e(
    model_dim, inter_dim, token, situ_beta, situ_linear_beta
):
    """a16w4 SiTUv2 SEPARATED end-to-end through fused_moe vs a bf16 SiTUv2 ref."""
    p = _situv2_problem(model_dim, inter_dim, token)
    ref = _situv2_ref(p, p["inp"], situ_beta, situ_linear_beta)
    out = _situv2_fused_moe(p, situ_beta, situ_linear_beta)

    assert not out.isnan().any().item(), "a16w4 SiTUv2 output contains NaN"
    ld = _cos_diff(ref.float(), out.float())
    assert ld < 1e-2, f"a16w4 SiTUv2 cos/logits_diff too large: {ld:.3e}"


@_SKIP_FP8_MFMA
@pytest.mark.parametrize("model_dim,inter_dim", [(3584, 512), (3584, 384)])
@pytest.mark.parametrize("token", [1, 16, 128])
def test_flydsl_a16wfp4_situv2_fp8_mfma(model_dim, inter_dim, token, monkeypatch):
    """Opt-in fp8 MFMA mode end-to-end through fused_moe.

    The mode is not a16 numerics, so it is held to a reference that quantizes at the
    same two points (activation per token, bf16 stage-1 output per route) with the
    torch quantizer; the bf16 reference above is not its target.
    """
    from aiter.ops.flydsl.kernels import moe_2stage_a16wmix

    situ_beta, situ_linear_beta = 4.0, 25.0
    p = _situv2_problem(model_dim, inter_dim, token)
    monkeypatch.delenv("AITER_FLYDSL_A16W4_FP8_MFMA", raising=False)
    out_bf16 = _situv2_fused_moe(p, situ_beta, situ_linear_beta)

    ran = []

    def _spy(name):
        fn = getattr(moe_2stage_a16wmix, name)

        def spy(**kw):
            ran.append((name, kw.get("a_dtype", "bf16")))
            return fn(**kw)

        return spy

    for name in ("flydsl_a16w4_gemm1", "flydsl_a16w4_gemm2"):
        monkeypatch.setattr(moe_2stage_a16wmix, name, _spy(name))
    monkeypatch.setenv("AITER_FLYDSL_A16W4_FP8_MFMA", "1")
    out = _situv2_fused_moe(p, situ_beta, situ_linear_beta)

    assert ran == [("flydsl_a16w4_gemm1", "fp8"), ("flydsl_a16w4_gemm2", "fp8")], ran
    assert not out.isnan().any().item(), "a16w4 fp8 MFMA output contains NaN"
    assert not torch.equal(out, out_bf16), "fp8 MFMA mode returned the bf16 result"
    ref = _situv2_ref(
        p, _fp8_rows(p["inp"]), situ_beta, situ_linear_beta, quant_inter=_fp8_rows
    )
    ld = _cos_diff(ref.float(), out.float())
    assert ld < 1e-3, f"a16w4 fp8 MFMA vs fp8-aware reference: {ld:.3e}"


@_SKIP_FP8_MFMA
@pytest.mark.parametrize("tile_m,tile_n", [(16, 64), (32, 128)])
def test_flydsl_a16wfp4_fp8_mfma_gemm1_rows(tile_m, tile_n):
    """fp8 gemm1 writes exactly the live sorted rows and matches its fp8 reference.

    Padding rows inside the sorted buffer (sentinel token ids) and rows past
    num_valid_ids must keep their prior contents.
    """
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
        flydsl_a16w4_gemm1,
        mxfp4_scale_ref_exponent,
    )

    model_dim, inter_dim, token = 3584, 384, 128
    p = _situv2_problem(model_dim, inter_dim, token)
    E, topk = p["E"], p["topk"]
    sorted_ids, _, sorted_eids, num_valid_ids, _ = moe_sorting(
        p["topk_ids"], p["topk_weights"], E, model_dim, dtypes.bf16, block_size=tile_m
    )
    a_q, a_s = aiter.get_torch_quant(QuantType.per_Token)(
        p["inp"], quant_dtype=dtypes.fp8
    )
    sref, max_res = mxfp4_scale_ref_exponent(
        p["w1_scale_gui"].view(torch.uint8), E * 2 * inter_dim, model_dim // 32
    )
    sentinel = 0x1357
    inter = torch.empty(
        (sorted_eids.numel() * tile_m, inter_dim), dtype=torch.bfloat16, device="cuda"
    )
    inter.view(torch.int16).fill_(sentinel)
    flydsl_a16w4_gemm1(
        a_bf16=a_q,
        w1_u8=p["w1_gui"].view(torch.uint8),
        w1_scale_u8=p["w1_scale_gui"].view(torch.uint8).view(-1),
        sorted_expert_ids=sorted_eids,
        cumsum_tensor=num_valid_ids,
        m_indices=sorted_ids,
        inter_sorted_bf16=inter,
        n_tokens=token,
        NE=E,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=128,
        act="situv2",
        situ_beta=4.0,
        situ_linear_beta=25.0,
        a_dtype="fp8",
        a_scale=a_s,
        w1_sref=sref.view(E, -1),
    )
    ref = torch_moe_stage1(
        a_q.float() * a_s,
        p["w1_qt"].view(dtypes.fp4x2),
        p["w2_qt"].view(dtypes.fp4x2),
        p["topk_weights"],
        p["topk_ids"],
        dtype=dtypes.fp32,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        w1_scale=p["w1_scale_e"],
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    nv = int(num_valid_ids[0].item())
    sid = sorted_ids[:nv].long()
    tok, slot = sid & 0xFFFFFF, sid >> 24
    live = tok < token
    bits = inter.view(torch.int16)
    assert max_res <= 6, max_res
    assert bool((bits[:nv][~live] == sentinel).all()), "a padding row was written"
    assert bool((bits[nv:] == sentinel).all()), "a row past num_valid_ids was written"
    got, want = inter[:nv][live].double(), ref[tok[live], slot[live]].double()
    rel = float((got - want).norm() / want.norm())
    assert rel < 5e-3, f"fp8 gemm1 vs fp8 stage-1 reference: rel_l2 {rel:.3e}"


def test_mxfp4_scale_ref_exponent():
    """The fp8 mode's per-row reference exponent, read from the preshuffled scale."""
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import mxfp4_scale_ref_exponent

    E, N, K = 4, 64, 384  # K/32 = 12 groups, padded to 16 by the shuffle
    torch.manual_seed(0)
    scale = torch.randint(110, 130, (E * N, K // 32), dtype=torch.uint8)
    shuffled = shuffle_scale_a16w4(scale, E, False)
    sref, max_res = mxfp4_scale_ref_exponent(shuffled, E * N, K // 32)
    want = scale.amax(dim=1)
    assert torch.equal(sref, want)
    assert max_res == int((want[:, None].int() - scale.int()).max())
