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

Run:
    pytest op_tests/flydsl_tests/test_flydsl_moe_a16wfp4.py -q
"""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

_SKIP = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950") or not is_flydsl_available(),
    reason="CDNA (gfx942/gfx950) + FlyDSL required for a16w4 SiTUv2",
)


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom)


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
    w1_scale_e = w1_scale.view(E, inter_dim * 2, model_dim // 32)
    w2_scale_e = w2_scale.view(E, model_dim, inter_dim // 32)

    # bf16 SiTUv2 SEPARATED reference
    o1 = torch_moe_stage1(
        inp.to(dtype),
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w1_scale_e,
        doweight=False,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    ref = torch_moe_stage2(
        o1.view(token, topk, inter_dim),
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale_e,
        a2_scale=None,
        doweight=True,
    )

    # caller contract: standard GGUU (separated gate/up) W1 layout, matching main
    w1_gui = shuffle_weight_a16w4(w1_qt, 16, False)
    w2_gui = shuffle_weight_a16w4(w2_qt, 16, False)
    w1_scale_gui = shuffle_scale_a16w4(w1_scale, E, False)
    w2_scale_gui = shuffle_scale_a16w4(w2_scale, E, False)

    out = fused_moe(
        inp,
        w1_gui,
        w2_gui,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale_gui,
        w2_scale=w2_scale_gui,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Situv2,
        doweight_stage1=False,
        gate_mode=GateMode.SEPARATED.value,
        beta=situ_beta,
        linear_beta=situ_linear_beta,
    )

    assert not out.isnan().any().item(), "a16w4 SiTUv2 output contains NaN"
    ld = _cos_diff(ref.float(), out.float())
    assert ld < 1e-2, f"a16w4 SiTUv2 cos/logits_diff too large: {ld:.3e}"


def test_a16w4_lds_budget_math():
    """a16w-mix LDS / num_acc_n gates, independent of the live GPU."""
    from aiter.ops.flydsl.moe_kernels import a16w4_config_fits_device

    # Decode + prefill tiles used by the gfx942 heuristic: exactly 64KiB or less.
    assert a16w4_config_fits_device(1, 32, 64, 256, 2, gfx="gfx942")
    assert a16w4_config_fits_device(1, 32, 128, 256, 1, gfx="gfx942")
    assert a16w4_config_fits_device(1, 64, 128, 256, 1, gfx="gfx942")
    assert a16w4_config_fits_device(2, 32, 128, 256, gfx="gfx942")
    assert a16w4_config_fits_device(2, 64, 128, 256, gfx="gfx942")
    # tile_m=16 is a legal a16w4 BM (same port as a16wi4 decode).
    assert a16w4_config_fits_device(1, 16, 16, 128, 4, gfx="gfx942")
    assert a16w4_config_fits_device(1, 16, 32, 256, 2, gfx="gfx942")
    assert a16w4_config_fits_device(1, 16, 128, 128, 1, gfx="gfx942")
    assert a16w4_config_fits_device(2, 16, 128, 128, gfx="gfx942")
    assert not a16w4_config_fits_device(1, 16, 16, 256, 1, gfx="gfx942")
    # tile_m=128 at tile_k=256 is 128KiB — illegal on gfx942, legal on gfx950.
    assert not a16w4_config_fits_device(1, 128, 128, 256, 1, gfx="gfx942")
    assert not a16w4_config_fits_device(2, 128, 128, 256, gfx="gfx942")
    assert a16w4_config_fits_device(1, 128, 128, 256, 1, gfx="gfx950")
    # tile_n=32 + k_wave=1 => num_acc_n=0 (silent zeros).
    assert not a16w4_config_fits_device(1, 32, 32, 256, 1, gfx="gfx942")
    assert a16w4_config_fits_device(1, 32, 32, 256, 2, gfx="gfx942")


@_SKIP
def test_gfx942_a16w4_registry_and_heuristic_lds():
    """On gfx942 the registry and no-CSV heuristic must stay within 64KiB."""
    if get_gfx() != "gfx942":
        pytest.skip("gfx942 a16w4 dispatch only")

    from aiter.fused_moe import get_2stage_cfgs, get_padded_M
    from aiter.ops.flydsl.moe_kernels import (
        a16w4_config_fits_device,
        get_flydsl_kernel_params,
    )

    assert (
        get_flydsl_kernel_params("flydsl_moe1_abf16_wfp4_bf16_t128x128x256") is None
    ), "tile_m=128 a16w4 gemm1 must not be registered on gfx942"
    assert get_flydsl_kernel_params("flydsl_moe1_abf16_wfp4_bf16_t32x64x256_kw2") is not None
    assert get_flydsl_kernel_params("flydsl_moe1_abf16_wfp4_bf16_t64x128x256") is not None
    assert get_flydsl_kernel_params("flydsl_moe1_abf16_wfp4_bf16_t16x32x256_kw2") is not None
    assert get_flydsl_kernel_params("flydsl_moe1_abf16_wfp4_bf16_t16x16x128_kw4") is not None

    for token in (1, 32, 2048, 4096, 8192, 16384):
        md = get_2stage_cfgs(
            get_padded_M(token),
            3584,
            512,
            896,
            16,
            dtypes.bf16,
            dtypes.bf16,
            dtypes.fp4x2,
            QuantType.per_1x32,
            True,
            ActivationType.Situv2,
            False,
            0,
            0,
            True,
            GateMode.SEPARATED.value,
        )
        kn1 = md.stage1.keywords["kernelName"]
        kn2 = md.stage2.keywords["kernelName"]
        p1 = get_flydsl_kernel_params(kn1)
        p2 = get_flydsl_kernel_params(kn2)
        assert p1 is not None, f"unregistered heuristic kn1={kn1} token={token}"
        assert p2 is not None, f"unregistered heuristic kn2={kn2} token={token}"
        assert md.block_m in (32, 64), f"tile_m={md.block_m} token={token}"
        assert a16w4_config_fits_device(
            1,
            p1["tile_m"],
            p1["tile_n"],
            p1["tile_k"],
            p1.get("k_wave", 1),
            gfx="gfx942",
        ), kn1
        assert a16w4_config_fits_device(
            2, p2["tile_m"], p2["tile_n"], p2["tile_k"], gfx="gfx942"
        ), kn2


@_SKIP
@pytest.mark.parametrize("model_dim,inter_dim", [(3584, 512), (3584, 384)])
@pytest.mark.parametrize("token", [1, 4096, 8192])
def test_flydsl_a16wfp4_situv2_prefill_gfx942(model_dim, inter_dim, token):
    """Prefill tokens that used to pick tile_m=128 (128KiB LDS) on the gfx950 heuristic."""
    if get_gfx() != "gfx942":
        pytest.skip("gfx942 a16w4 prefill LDS path")

    E, topk = 128, 4
    dtype = dtypes.bf16
    situ_beta, situ_linear_beta = 4.0, 25.0
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
    w1_scale_e = w1_scale.view(E, inter_dim * 2, model_dim // 32)
    w2_scale_e = w2_scale.view(E, model_dim, inter_dim // 32)

    o1 = torch_moe_stage1(
        inp.to(dtype),
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w1_scale_e,
        doweight=False,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    ref = torch_moe_stage2(
        o1.view(token, topk, inter_dim),
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale_e,
        a2_scale=None,
        doweight=True,
    )

    w1_gui = shuffle_weight_a16w4(w1_qt, 16, False)
    w2_gui = shuffle_weight_a16w4(w2_qt, 16, False)
    w1_scale_gui = shuffle_scale_a16w4(w1_scale, E, False)
    w2_scale_gui = shuffle_scale_a16w4(w2_scale, E, False)

    out = fused_moe(
        inp,
        w1_gui,
        w2_gui,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale_gui,
        w2_scale=w2_scale_gui,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Situv2,
        doweight_stage1=False,
        gate_mode=GateMode.SEPARATED.value,
        beta=situ_beta,
        linear_beta=situ_linear_beta,
    )

    assert not out.isnan().any().item(), "a16w4 SiTUv2 prefill output contains NaN"
    ld = _cos_diff(ref.float(), out.float())
    assert ld < 1e-2, f"a16w4 SiTUv2 prefill cos/logits_diff too large: {ld:.3e}"


@_SKIP
def test_gfx942_a16w4_tuner_enumerates_candidates():
    """Tuner must search LDS-legal a16w4 tiles on gfx942 (not the gfx950-only skip)."""
    if get_gfx() != "gfx942":
        pytest.skip("gfx942 a16w4 tuner only")

    import sys

    from aiter.jit.core import AITER_CSRC_DIR
    from aiter.ops.flydsl.moe_kernels import (
        a16w4_config_fits_device,
        get_flydsl_kernel_params,
    )

    sys.path.insert(0, f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/")
    from gemm_moe_tune import FmoeTuner, _is_a16w4_per1x32

    assert _is_a16w4_per1x32(dtypes.bf16, dtypes.fp4x2, QuantType.per_1x32)
    assert not _is_a16w4_per1x32(dtypes.fp8, dtypes.fp4x2, QuantType.per_1x32)

    info = (
        "gfx942",
        304,
        1,
        3584,
        512,
        128,
        4,
        ActivationType.Situv2,
        dtypes.bf16,
        dtypes.bf16,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        False,
    )
    tasks = FmoeTuner.gen_flydsl_2stages_task(
        FmoeTuner.__new__(FmoeTuner), info, [16, 32, 64, 128]
    )
    s1 = [t[0][2] for t in tasks if t[0][1] == "stage1"]
    s2 = [t[0][2] for t in tasks if t[0][1] == "stage2"]
    assert s1, "gfx942 a16w4 tuner must enumerate stage1 kernels"
    assert s2, "gfx942 a16w4 tuner must enumerate stage2 kernels"
    assert any("t32x64x256" in n and "kw2" in n for n in s1)
    assert any("t16x" in n for n in s1), "tile_m=16 must be in the a16w4 search"
    assert not any("t128x" in n for n in s1 + s2)
    for name in s1:
        p = get_flydsl_kernel_params(name)
        assert p is not None, name
        assert p.get("k_batch", 1) == 1, name
        assert a16w4_config_fits_device(
            1,
            p["tile_m"],
            p["tile_n"],
            p["tile_k"],
            p.get("k_wave", 1),
            gfx="gfx942",
        ), name
    for name in s2:
        p = get_flydsl_kernel_params(name.split("_sbm")[0])
        assert p is not None, name
        assert a16w4_config_fits_device(
            2, p["tile_m"], p["tile_n"], p["tile_k"], gfx="gfx942"
        ), name
