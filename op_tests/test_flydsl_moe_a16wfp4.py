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

import argparse
import functools
import itertools

import pandas as pd
import pytest
import torch
import torch.nn.functional as functional

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
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
    flydsl_a16w4_gemm1,
    flydsl_a16w4_gemm2,
)
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

_SKIP = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"),
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


def _stage1_reference(hidden, weight, expert_ids, limit, activation_bf16):
    projected = torch.empty(
        (hidden.shape[0], expert_ids.shape[1], weight.shape[1]),
        dtype=torch.float32,
        device=hidden.device,
    )
    for expert in range(weight.shape[0]):
        token_ids, slots = torch.where(expert_ids == expert)
        projected[token_ids, slots] = (
            hidden[token_ids].float() @ weight[expert].float().T
        )
    if activation_bf16:
        projected = projected.to(torch.bfloat16).float()
    gate, up = projected.chunk(2, dim=-1)
    return (functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)).to(
        torch.bfloat16
    )


@benchmark()
def benchmark_stage1_clamped_silu(
    token,
    model_dim,
    inter_dim,
    experts,
    topk,
    weight_dtype,
    activation_bf16,
    logical_inter=0,
    dtype=dtypes.bf16,
):
    """Measure useful FLOPs and unique routed-expert bytes, not DRAM counters.

    DeepSeek-V4.1-Flash TP8: H=5120, I=2304/8=288 padded to 384,
    E=384, topk=6. Zero-pad gate and up separately before packing weights.

    Compare using two bf16 relative ULPs plus 0.002 row RMS for cancellation.
    Different f32 GEMM reduction orders can straddle a bf16 rounding midpoint.
    Rounding before activation and on output can each contribute one ULP.
    """
    assert dtype == dtypes.bf16
    assert 0 <= logical_inter <= inter_dim
    assert 0 < topk <= experts
    torch.manual_seed(17)
    hidden = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    weight = torch.randn(
        (experts, 2 * inter_dim, model_dim),
        dtype=dtype,
        device=hidden.device,
    )
    if logical_inter:
        weight[:, logical_inter:inter_dim] = 0
        weight[:, inter_dim + logical_inter :] = 0
    scores = torch.randn((token, experts), device=hidden.device)
    routing_ids = scores.topk(topk, dim=-1).indices.to(torch.int32)
    routing_weights = torch.rand((token, topk), device=hidden.device)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    if weight_dtype == "fp4":
        quantize = aiter.get_torch_quant(QuantType.per_1x32)
        packed, scales = quantize(weight, quant_dtype=dtypes.fp4x2)
        packed = packed.view(experts, 2 * inter_dim, model_dim // 2)
        scales = scales.view(experts, 2 * inter_dim, model_dim // 32)
        reference_weight = (
            (
                mxfp4_to_f32(packed).view(experts, 2 * inter_dim, -1, 32)
                * e8m0_to_f32(scales).unsqueeze(-1)
            )
            .reshape_as(weight)
            .to(torch.bfloat16)
        )
        shuffled_weight = shuffle_weight_a16w4(packed, 16, False)
        shuffled_scale = shuffle_scale_a16w4(
            scales.reshape(-1, scales.shape[-1]), experts, False
        )
    else:
        from aiter.ops.shuffle import shuffle_weight

        reference_weight = weight
        shuffled_weight = shuffle_weight(weight, (16, 16))
        shuffled_scale = torch.empty(0, dtype=torch.uint8, device=hidden.device)
    block_m = 32
    sorted_ids, _, sorted_experts, cumsum, _ = moe_sorting(
        routing_ids,
        routing_weights,
        experts,
        model_dim,
        dtypes.bf16,
        block_m,
    )
    sorted_size = sorted_experts.numel() * block_m
    guard = 64
    canary = -123.0
    storage = torch.full(
        (sorted_size * inter_dim + 2 * guard,),
        canary,
        dtype=torch.bfloat16,
        device=hidden.device,
    )
    output = storage[guard:-guard].view(sorted_size, inter_dim)
    valid_size = int(cumsum[0])
    packed_ids = sorted_ids[:valid_size].long()
    token_ids = packed_ids & 0x00FFFFFF
    slots = packed_ids >> 24
    valid = token_ids < token
    valid_positions = torch.arange(valid_size, device=hidden.device)[valid]
    padded = torch.ones(sorted_size, device=hidden.device, dtype=torch.bool)
    padded[valid_positions] = False
    launch = functools.partial(
        flydsl_a16w4_gemm1,
        a_bf16=hidden,
        w1_u8=shuffled_weight.view(torch.uint8),
        w1_scale_u8=shuffled_scale.view(torch.uint8),
        sorted_expert_ids=sorted_experts,
        cumsum_tensor=cumsum,
        m_indices=sorted_ids,
        inter_sorted_bf16=output,
        n_tokens=token,
        NE=experts,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        tile_m=block_m,
        tile_n=128,
        tile_k=256,
        act="silu_clamped",
        activation_bf16=activation_bf16,
        w_dtype=weight_dtype,
    )
    candidates = {"flydsl": launch}
    result = {"gfx": get_gfx()}
    flops = 4 * token * topk * model_dim * inter_dim
    active_experts = routing_ids.unique().numel()
    nbytes = (
        hidden.nbytes
        + active_experts * (shuffled_weight.nbytes + shuffled_scale.nbytes) // experts
        + token * topk * inter_dim * output.element_size()
    )
    for name, candidate in candidates.items():
        max_normalized_error = 0.0
        for limit in (10.0, 2.0, float("inf"), 10.0):
            reference = _stage1_reference(
                hidden, reference_weight, routing_ids, limit, activation_bf16
            )
            candidate(swiglu_limit=limit)
            torch.cuda.synchronize()
            actual = output[valid_positions]
            expected = reference[token_ids[valid], slots[valid]]
            assert torch.isfinite(actual).all().item()
            rms = (
                expected.float()
                .square()
                .mean(dim=-1, keepdim=True)
                .sqrt()
                .clamp_min(1e-6)
            )
            error = checkAllclose(
                expected.float() / rms,
                actual.float() / rms,
                rtol=2 * torch.finfo(dtype).eps,
                atol=0.002,
                msg=f"{name}: clamped SiLU limit={limit}",
            )
            max_normalized_error = max(
                max_normalized_error,
                ((expected.float() - actual.float()).abs() / rms).max().item(),
            )
            assert error == 0, error
            assert (storage[:guard] == canary).all().item()
            assert (storage[-guard:] == canary).all().item()
            assert (output[padded] == canary).all().item()
            if logical_inter:
                assert (actual[:, logical_inter:] == 0).all().item()
            if limit == float("inf") and not activation_bf16:
                candidate(act="silu", swiglu_limit=10.0)
                torch.testing.assert_close(
                    output[valid_positions], actual, rtol=0, atol=0
                )
        for invalid_limit in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError, match="swiglu_limit > 0"):
                candidate(swiglu_limit=invalid_limit)
        _, latency = run_perftest(candidate, swiglu_limit=10.0)
        result[f"{name} us"] = latency
        result[f"{name} TFLOPS"] = flops / latency / 1e6
        result[f"{name} TB/s"] = nbytes / latency / 1e6
        result[f"{name} err"] = error
        result[f"{name} max normalized err"] = max_normalized_error
    return result


@_SKIP
@pytest.mark.parametrize("token", [1, 8, 31, 32, 33, 65])
@pytest.mark.parametrize("activation_bf16", [False, True])
def test_stage1_clamped_silu(token, activation_bf16):
    result = benchmark_stage1_clamped_silu(
        token, 256, 256, 8, 6, "fp4", activation_bf16, logical_inter=192
    )
    aiter.logger.info(
        "Clamped SiLU stage1 summary (markdown):\n%s",
        pd.DataFrame([result]).to_markdown(index=False),
    )


def _pack_mxfp4_reference(weight):
    experts, rows, columns = weight.shape
    packed, scales = aiter.get_torch_quant(QuantType.per_1x32)(
        weight, quant_dtype=dtypes.fp4x2
    )
    packed = packed.view(experts, rows, columns // 2)
    scales = scales.view(experts, rows, columns // 32)
    dequantized = (
        (
            mxfp4_to_f32(packed).view(experts, rows, -1, 32)
            * e8m0_to_f32(scales).unsqueeze(-1)
        )
        .reshape_as(weight)
        .to(torch.bfloat16)
    )
    return (
        shuffle_weight_a16w4(packed, 16, False),
        shuffle_scale_a16w4(scales.flatten(0, 1), experts, False),
        dequantized,
    )


def _moe_reference(hidden, weight1, weight2, routing_ids, routing_weights, rounding):
    intermediate = _stage1_reference(hidden, weight1, routing_ids, 10.0, rounding)
    contributions = torch.empty(
        (*routing_ids.shape, hidden.shape[1]),
        dtype=torch.float32,
        device=hidden.device,
    )
    for expert in range(weight2.shape[0]):
        token_ids, slots = torch.where(routing_ids == expert)
        contributions[token_ids, slots] = (
            intermediate[token_ids, slots].float() @ weight2[expert].float().T
        ) * routing_weights[token_ids, slots, None]
    return contributions.sum(dim=1).to(hidden.dtype)


@benchmark()
def benchmark_moe_clamped_silu(
    token,
    model_dim,
    inter_dim,
    experts,
    topk,
    activation_bf16,
    logical_inter=0,
    dtype=dtypes.bf16,
    tile_m=32,
    tile_n=128,
    tile_k=256,
):
    """Full routed TP-local MoE, including sorting, both GEMMs and output zeroing.

    Inputs to the routing interface are top-k IDs and weights. The upstream
    router scoring and the TP all-reduce are outside this operator measurement.
    """
    assert dtype == dtypes.bf16
    assert 0 <= logical_inter <= inter_dim
    assert 0 < topk <= experts
    torch.manual_seed(23)
    hidden = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    weight1 = torch.randn(
        (experts, 2 * inter_dim, model_dim), dtype=dtype, device=hidden.device
    )
    weight2 = torch.randn(
        (experts, model_dim, inter_dim), dtype=dtype, device=hidden.device
    )
    if logical_inter:
        weight1[:, logical_inter:inter_dim] = 0
        weight1[:, inter_dim + logical_inter :] = 0
        weight2[:, :, logical_inter:] = 0
    packed1, scales1, reference1 = _pack_mxfp4_reference(weight1)
    packed2, scales2, reference2 = _pack_mxfp4_reference(weight2)
    scores = torch.randn((token, experts), device=hidden.device)
    routing_ids = scores.topk(topk, dim=-1).indices.to(torch.int32)
    routing_weights = torch.rand((token, topk), device=hidden.device)
    routing_weights *= 1.5 / routing_weights.sum(dim=-1, keepdim=True)
    reference = _moe_reference(
        hidden, reference1, reference2, routing_ids, routing_weights, activation_bf16
    )
    metadata = moe_sorting(
        routing_ids, routing_weights, experts, model_dim, dtype, tile_m
    )
    sorted_size = metadata[2].numel() * tile_m
    intermediate = torch.empty(
        (sorted_size, inter_dim), dtype=dtype, device=hidden.device
    )
    guard = 64
    canary = -123.0
    storage = torch.full(
        (token * model_dim + 2 * guard,), canary, dtype=dtype, device=hidden.device
    )
    output = storage[guard:-guard].view(token, model_dim)

    def run_flydsl():
        sorted_ids, sorted_weights, sorted_experts, cumsum, _ = moe_sorting(
            routing_ids, routing_weights, experts, model_dim, dtype, tile_m
        )
        output.zero_()
        flydsl_a16w4_gemm1(
            a_bf16=hidden,
            w1_u8=packed1.view(torch.uint8),
            w1_scale_u8=scales1.view(torch.uint8),
            sorted_expert_ids=sorted_experts,
            cumsum_tensor=cumsum,
            m_indices=sorted_ids,
            inter_sorted_bf16=intermediate,
            n_tokens=token,
            NE=experts,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            act="silu_clamped",
            swiglu_limit=10.0,
            activation_bf16=activation_bf16,
        )
        return flydsl_a16w4_gemm2(
            inter_sorted_bf16=intermediate,
            w2_u8=packed2.view(torch.uint8),
            w2_scale_u8=scales2.view(torch.uint8),
            sorted_expert_ids=sorted_experts,
            cumsum_tensor=cumsum,
            sorted_token_ids=sorted_ids,
            sorted_weights=sorted_weights,
            flat_out=output,
            M_logical=token,
            max_sorted=sorted_size,
            NE=experts,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            topk=topk,
            tile_m=tile_m,
            tile_n=256,
            tile_k=128,
        )

    candidates = {"flydsl_moe": run_flydsl}
    result = {"gfx": get_gfx()}
    flops = 6 * token * topk * model_dim * inter_dim
    active_experts = routing_ids.unique().numel()
    nbytes = (
        hidden.nbytes
        + output.nbytes
        + 2 * token * topk * inter_dim * hidden.element_size()
        + active_experts
        * (packed1.nbytes + scales1.nbytes + packed2.nbytes + scales2.nbytes)
        // experts
    )
    rms = reference.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    for name, candidate in candidates.items():
        for initial in (0.0, 17.0):
            output.fill_(initial)
            actual = candidate()
            torch.cuda.synchronize()
            error = checkAllclose(
                reference.float() / rms,
                actual.float() / rms,
                rtol=0.03,
                atol=0.03,
                msg=f"{name}: routed clamped SiLU",
            )
            assert error == 0
            assert torch.isfinite(actual).all().item()
            assert (storage[:guard] == canary).all().item()
            assert (storage[-guard:] == canary).all().item()
        _, latency = run_perftest(candidate, num_rotate_args=1)
        result[f"{name} us"] = latency
        result[f"{name} TFLOPS"] = flops / latency / 1e6
        result[f"{name} TB/s"] = nbytes / latency / 1e6
        result[f"{name} err"] = error
    return result


@_SKIP
@pytest.mark.parametrize("token", [1, 8, 33])
def test_moe_clamped_silu(token):
    result = benchmark_moe_clamped_silu(token, 256, 384, 8, 6, True, logical_inter=288)
    aiter.logger.info(
        "Clamped SiLU full MoE summary (markdown):\n%s",
        pd.DataFrame([result]).to_markdown(index=False),
    )


def main():
    if get_gfx() not in ("gfx942", "gfx950"):
        aiter.logger.warning("FlyDSL a16w-mix unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(description="Clamped SiLU a16w-mix stage1")
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="+",
        default=[dtypes.bf16],
        choices=[dtypes.bf16],
    )
    parser.add_argument("-b", "--batch", type=int, nargs="+", default=[1, 8, 33, 128])
    parser.add_argument(
        "-s", "--mnk", type=dtypes.str2tuple, nargs="+", default=[(5120, 384)]
    )
    parser.add_argument("--experts", type=int, nargs="+", default=[8, 384])
    parser.add_argument("--topk", type=int, nargs="+", default=[6])
    parser.add_argument(
        "--weight-dtype", choices=["fp4", "bf16"], nargs="+", default=["fp4"]
    )
    parser.add_argument(
        "--activation-bf16", type=int, choices=[0, 1], nargs="+", default=[0, 1]
    )
    parser.add_argument("--logical-inter", type=int, nargs="+", default=[0])
    parser.add_argument("--stages", type=int, choices=[1, 2], nargs="+", default=[1])
    parser.add_argument("--tile-m", type=int, nargs="+", default=[32])
    parser.add_argument("--tile-n", type=int, nargs="+", default=[128])
    parser.add_argument("--tile-k", type=int, nargs="+", default=[256])
    args = parser.parse_args()
    if 2 in args.stages:
        assert args.weight_dtype == ["fp4"], "full MoE sweep requires fp4 weights"
        rows = [
            benchmark_moe_clamped_silu(
                token,
                model_dim,
                inter_dim,
                experts,
                topk,
                bool(rounding),
                logical_inter,
                dtype,
                tile_m,
                tile_n,
                tile_k,
            )
            for token, (
                model_dim,
                inter_dim,
            ), experts, topk, rounding, logical_inter, dtype, tile_m, tile_n, tile_k in itertools.product(
                args.batch,
                args.mnk,
                args.experts,
                args.topk,
                args.activation_bf16,
                args.logical_inter,
                args.dtype,
                args.tile_m,
                args.tile_n,
                args.tile_k,
            )
        ]
        aiter.logger.info(
            "Clamped SiLU full MoE summary (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )
    if 1 not in args.stages:
        return
    rows = [
        benchmark_stage1_clamped_silu(
            token,
            model_dim,
            inter_dim,
            experts,
            topk,
            weight_dtype,
            bool(rounding),
            logical_inter,
            dtype,
        )
        for token, (
            model_dim,
            inter_dim,
        ), experts, topk, weight_dtype, rounding, logical_inter, dtype in itertools.product(
            args.batch,
            args.mnk,
            args.experts,
            args.topk,
            args.weight_dtype,
            args.activation_bf16,
            args.logical_inter,
            args.dtype,
        )
    ]
    aiter.logger.info(
        "Clamped SiLU stage1 summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
