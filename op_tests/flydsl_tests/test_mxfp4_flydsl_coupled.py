# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Validate the same coupled MXMOE candidate boundary used by the tuner."""

import argparse
import functools
import itertools
import math
import os

import pandas as pd
import pytest
import torch

import aiter
from aiter import ActivationType, dtypes
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils
from aiter.utility.dtypes import str2Dtype
from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import (
    Mxfp4FlydslTuner,
    cosine_diff_compare,
)

SUPPORTED_GFX = ("gfx950",)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX,
    reason="Coupled MXMOE requires gfx950",
)


def run_torch(data, topk, dtype, activation, swiglu_limit=None):
    return Mxfp4FlydslTuner._torch_ref(
        data, topk, dtype, activation, swiglu_limit=swiglu_limit
    )


def check_mxfp8_input(data, token, model_dim, expert, topk, block_m, dtype):
    """Check payload and valid sorted scales against the public torch quantizer."""
    ref_q, ref_s = per_1x32_f8_scale_f8_quant(
        data["input"], scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    q = data["a1_qt"]
    scale = data["a1_scale"]
    assert q.dtype == dtypes.fp8
    assert scale.dtype == dtypes.fp8_e8m0
    assert q.shape == (token, model_dim)
    assert torch.equal(ref_q.view(torch.uint8), q.view(torch.uint8))
    nonzero_group = data["input"].reshape(token, -1, 32).abs().amax(-1) > 0
    # HIP floors amax at 1e-10; the torch helper keeps exponent 0 for zero
    # groups. Require exact positive-group scales and joint decoded values.
    assert torch.equal(
        ref_s.view(torch.uint8)[nonzero_group],
        scale.view(torch.uint8)[nonzero_group],
    )
    scale_f32 = fp4_utils.e8m0_to_f32(scale)
    assert torch.isfinite(scale_f32).all()
    assert (scale_f32 > 0).all()
    reconstructed = q.float() * scale_f32.repeat_interleave(32, 1)
    ref_reconstructed = ref_q.float() * fp4_utils.e8m0_to_f32(ref_s).repeat_interleave(
        32, 1
    )
    assert torch.equal(reconstructed, ref_reconstructed)
    assert torch.isfinite(reconstructed).all()
    assert torch.equal(
        reconstructed.reshape(token, -1, 32)[~nonzero_group],
        torch.zeros_like(reconstructed.reshape(token, -1, 32)[~nonzero_group]),
    )
    error = checkAllclose(
        data["input"].float(),
        reconstructed,
        # E4M3 rounds by up to half one 3-bit mantissa step (1/16).
        rtol=0.0625,
        atol=0.01,
        tol_err_ratio=0,
        msg="MXFP8 A payload + E8M0 scales",
    )
    assert error == 0
    input_error = cosine_diff_compare(
        data["input"], reconstructed, msg="MXFP8 input quantization"
    )
    assert math.isfinite(input_error) and input_error < 0.005
    assert torch.unique(scale.view(torch.uint8)).numel() > 1

    sti, sw, _, nvi, _, _, _ = moe_sorting(
        data["topk_ids"],
        data["topk_weights"],
        expert,
        model_dim,
        dtype,
        block_size=block_m,
        accumulate=False,
        output_aux="opus",
    )
    sorted_q, sorted_scale = aiter.fused_dynamic_mxfp8_quant_moe_sort(
        input=data["input"],
        sorted_ids=sti,
        num_valid_ids=nvi,
        token_num=token,
        topk=topk,
        block_size=block_m,
        sorted_weights=sw,
        num_experts_upper_bound=expert,
    )
    assert torch.equal(q.view(torch.uint8), sorted_q.view(torch.uint8))
    # Match only active scale addresses. Padding bytes have no consumer contract.
    valid_count = int(nvi[0].item())
    row = torch.arange(valid_count, device=sti.device)
    group = torch.arange(model_dim // 32, device=sti.device)
    token_id = sti[:valid_count].to(torch.int64) & 0xFFFFFF
    valid = token_id < token
    r = row[valid, None]
    g = group[None, :]
    cols_pad = sorted_scale.shape[1]
    offset = (
        (r // 32) * 32 * cols_pad
        + (g // 8) * 256
        + (g % 4) * 64
        + (r % 16) * 4
        + ((g % 8) // 4) * 2
        + (r % 32) // 16
    )
    actual = sorted_scale.view(torch.uint8).flatten()[offset]
    expected = scale.view(torch.uint8)[token_id[valid, None], g]
    assert torch.equal(actual, expected)


@benchmark()
def test_coupled_mxmoe(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    dtype,
    precision,
    activation,
    block_m,
    tile_k,
):
    a_dtype = "fp8" if precision == "A8W4" else "fp4"
    data = Mxfp4FlydslTuner._prepare_case(
        token, model_dim, inter_dim, expert, topk, dtype, a_dtype=a_dtype
    )
    if precision == "A8W4":
        # Different group amplitudes and zero groups expose scale omissions;
        # regenerating the quantized reference is outside candidate timing.
        amplitude = torch.ones_like(data["input"])
        amplitude[: token // 2, :32] = 0
        amplitude[token // 2 :, 32:64] = 16
        data["input"].mul_(amplitude)
        from aiter.ops.quant import per_1x32_mx_quant_hip

        data["a1_qt"], data["a1_scale"] = per_1x32_mx_quant_hip(
            data["input"],
            quant_dtype=dtypes.fp8,
            scale_type=dtypes.fp8_e8m0,
            shuffle=False,
        )
        check_mxfp8_input(data, token, model_dim, expert, topk, block_m, dtype)

    act = {"Silu": "silu", "Situv2": "situv2", "Swiglu": "swiglu"}[activation]
    act_type = getattr(ActivationType, activation)
    limit_env = os.environ.get("AITER_MXFP4_TUNE_SWIGLU_LIMIT")
    limit = None if limit_env in (None, "") else float(limit_env)
    reference = run_torch(data, topk, dtype, act_type, swiglu_limit=limit)
    g1 = Mxfp4FlydslTuner._g1_kname(
        block_m, False, False, act=act, a_dtype=a_dtype, out_dtype=a_dtype
    )
    modes = ("atomic", "reduce", "scatter") if block_m == 128 else ("atomic", "reduce")
    candidates = {}
    for mode in modes:
        g2 = (
            f"flydsl_moe2_layout_a{a_dtype}_wfp4_bf16_"
            f"t{block_m}x128x{tile_k}_{mode}_sbm{block_m}"
        )
        candidates[mode] = functools.partial(
            Mxfp4FlydslTuner._port_e2e,
            data,
            g1,
            g2,
            topk,
            expert,
            model_dim,
            dtype,
            swiglu_limit=limit,
        )
    flops = 6 * token * topk * model_dim * inter_dim
    nbytes = (
        sum(
            data[key].numel() * data[key].element_size()
            for key in (
                "input",
                "w1_a16",
                "w1s_a16",
                "w2_a16",
                "w2s_a16",
            )
        )
        + token * model_dim * torch.empty((), dtype=dtype).element_size()
    )
    ret = {"gfx": get_gfx(), "GEMM1": g1}
    for name, candidate in candidates.items():
        output, us = run_perftest(candidate, num_warmup=2, num_iters=5)
        assert math.isfinite(us) and us > 0
        assert torch.isfinite(output).all()
        error = cosine_diff_compare(reference, output, msg=f"{precision} {name}")
        assert math.isfinite(error) and error < 0.1
        # Also retain a position-sensitive check against the unquantized
        # intermediate reference; the tuner gate remains the amplitude metric.
        checkAllclose(
            reference.float(),
            output.float(),
            rtol=0.2,
            atol=0.2,
            msg=f"{precision} {name}",
            catastrophic_check=True,
        )
        first = output.clone()
        repeated = candidate()
        repeat_error = checkAllclose(
            first.float(),
            repeated.float(),
            rtol=0.02,
            atol=0.02,
            msg=f"{precision} {name} repeat",
        )
        assert repeat_error == 0
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = error
    return ret


test_coupled_mxmoe.__test__ = False  # The CLI owns the perf sweep axes.


@pytest.mark.parametrize("precision", ["A4W4", "A8W4"])
@pytest.mark.parametrize("activation", ["Silu", "Situv2", "Swiglu"])
@pytest.mark.parametrize("block_m", [32, 64, 128])
@pytest.mark.parametrize("inter_dim,tile_k", [(384, 128), (512, 256)])
def test_coupled_candidate_numerics(precision, activation, block_m, inter_dim, tile_k):
    with torch.device("cuda"):
        result = test_coupled_mxmoe(
            64,
            1024,
            inter_dim,
            4,
            2,
            dtypes.bf16,
            precision,
            activation,
            block_m,
            tile_k,
        )
    aiter.logger.info(
        "coupled MXMOE summary (markdown):\n%s",
        pd.DataFrame([result]).to_markdown(index=False),
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("Coupled MXMOE is unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--dtype", type=str2Dtype, nargs="*", default=[dtypes.bf16]
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[64])
    # This Opus-routed workload exercises multiple experts and scale-N padding;
    # it does not require adding a test-only generated auxiliary model key.
    parser.add_argument(
        "-s", "--mnk", type=dtypes.str2tuple, nargs="*", default=[(1024, 384)]
    )
    parser.add_argument("--expert", type=int, nargs="*", default=[4])
    parser.add_argument("--topk", type=int, nargs="*", default=[2])
    parser.add_argument(
        "--precision", choices=("A4W4", "A8W4"), nargs="*", default=["A4W4", "A8W4"]
    )
    parser.add_argument(
        "--activation",
        choices=("Silu", "Situv2", "Swiglu"),
        nargs="*",
        default=["Silu", "Situv2", "Swiglu"],
    )
    parser.add_argument(
        "--block-m", type=int, choices=(32, 64, 128), nargs="*", default=[32, 64, 128]
    )
    parser.add_argument(
        "--tile-k", type=int, choices=(128, 256), nargs="*", default=[128, 256]
    )
    args = parser.parse_args()
    torch.set_default_device("cuda")
    results = []
    for (
        dtype,
        token,
        shape,
        expert,
        topk,
        precision,
        activation,
        bm,
        bk,
    ) in itertools.product(
        args.dtype,
        args.batch,
        args.mnk,
        args.expert,
        args.topk,
        args.precision,
        args.activation,
        args.block_m,
        args.tile_k,
    ):
        h, inter = shape
        if inter % bk or dtype != dtypes.bf16 or topk > expert:
            continue
        results.append(
            test_coupled_mxmoe(
                token, h, inter, expert, topk, dtype, precision, activation, bm, bk
            )
        )
    aiter.logger.info(
        "coupled MXMOE summary (markdown):\n%s",
        pd.DataFrame(results).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
