# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 1-stage asm MXFP4 x MXFP8 SiTUv2 MoE through fused_moe (MI300X / MI325X)."""

import argparse
import itertools

import pandas as pd
import torch

import aiter
import aiter.fused_moe as moe
from aiter import ActivationType, QuantType, dtypes
from aiter.aot.flydsl.common import override_env
from aiter.fused_moe import fused_moe, fused_topk, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import (
    shuffle_scale_a8w4_gfx942,
    shuffle_scale_a16w4,
    shuffle_weight_a8w4_gfx942,
    shuffle_weight_a16w4,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

EXPERT, TOPK = 896, 16
BETA, LINEAR_BETA = 4.0, 25.0
ASM_KERNEL = "pertokenMXfp4Fp8_g1u1_novs_situv2"
TUNED_TOKENS = (1024, 2048, 4096, 8192, 16384, 32768)


def cos_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


def check_dispatch(model_dim, inter_dim, dtype=dtypes.bf16):
    kernel = f"_ZN5aiter50fmoe_bf16_{ASM_KERNEL}_32x{inter_dim // 2}E"
    for token in TUNED_TOKENS:
        metadata = moe.get_2stage_cfgs(
            moe.get_padded_M(token),
            model_dim,
            inter_dim,
            EXPERT,
            TOPK,
            dtype,
            dtypes.fp8,
            dtypes.fp4x2,
            QuantType.per_1x32,
            True,
            ActivationType.Situv2,
            False,
            0,
            0,
            is_shuffled=True,
            opus_weights_shuffled=True,
            gate_mode=GateMode.SEPARATED.value,
            gate_up_interleave=False,
            situ_beta=BETA,
            situ_linear_beta=LINEAR_BETA,
        )
        assert metadata.run_1stage, token
        assert metadata.stage1.func is moe.fused_moe_1stage, token
        assert metadata.stage1.keywords["kernelName"] == kernel, (
            token,
            metadata.stage1.keywords["kernelName"],
        )


@benchmark()
def test_fmoe_asm_mxfp4fp8(token, model_dim, inter_dim, dtype=dtypes.bf16):
    E = EXPERT
    torch.manual_seed(0)
    x = torch.randn((token, model_dim), dtype=dtype)
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype)
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype)
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(x, score, TOPK, True)

    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = tq(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = tq(w2, quant_dtype=dtypes.fp4x2)
    del w1, w2
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)

    a1_qt, a1_scale = per_1x32_f8_scale_f8_quant(
        x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    o1 = torch_moe_stage1(
        a1_qt,
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=a1_scale,
        w1_scale=w1_scale.view(E, inter_dim * 2, model_dim // 32),
        doweight=False,
        situ_beta=BETA,
        situ_linear_beta=LINEAR_BETA,
    )
    ref = torch_moe_stage2(
        o1.view(token, TOPK, inter_dim),
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale.view(E, model_dim, inter_dim // 32),
        a2_scale=None,
        doweight=True,
    )

    w1_s = shuffle_weight_a16w4(w1_qt, 16, False)
    w2_s = shuffle_weight_a16w4(w2_qt, 16, False)
    w1_sc = shuffle_scale_a16w4(w1_scale, E, False)
    w2_sc = shuffle_scale_a16w4(w2_scale, E, False)
    w1_s8 = shuffle_weight_a8w4_gfx942(w1_s)
    w1_sc8 = shuffle_scale_a8w4_gfx942(w1_sc)

    def run(x, w1, w1_scale, a1_scale=None, a8w4="1"):
        # fused_moe reads the flag on each call
        with override_env("AITER_SITUV2_A8W4", a8w4):
            return fused_moe(
                x,
                w1,
                w2_s,
                topk_weights,
                topk_ids,
                w1_scale=w1_scale,
                w2_scale=w2_sc,
                quant_type=QuantType.per_1x32,
                activation=ActivationType.Situv2,
                doweight_stage1=False,
                gate_mode=GateMode.SEPARATED.value,
                beta=BETA,
                linear_beta=LINEAR_BETA,
                a1_scale=a1_scale,
                dtype=dtype,
            )

    out, us = run_perftest(run, x, w1_s8, w1_sc8, num_iters=5, num_warmup=2)
    assert not out.isnan().any().item(), "NaN in output"
    err = checkAllclose(
        ref.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=2e-2 * ref.abs().max().item(),
        msg="asm: fused_moe vs torch",
    )
    # asm requantizes the intermediate to fp8 per 128 columns; the torch reference
    # keeps it in bf16, which puts single elements past atol
    assert err <= 1e-6, err
    ret = {"gfx": get_gfx(), "us": us, "err": err, "cos": cos_diff(ref, out)}

    out_q = run(a1_qt, w1_s8, w1_sc8, a1_scale=a1_scale)
    ret["fp8 x cos"] = cos_diff(out, out_q)
    assert ret["fp8 x cos"] < 1e-4, ret

    # W1 in the a16w4 layout has the same shape and dtype; only its tag is missing
    for bad_w1, bad_w1_sc in ((w1_s, w1_sc), (w1_s8, w1_sc), (w1_s, w1_sc8)):
        try:
            run(x, bad_w1, bad_w1_sc)
        except ValueError as e:
            assert "a8w4_gfx942_layout" in str(e), e
        else:
            raise AssertionError("asm kernel ran on a W1 that is not relaid")
    try:
        run(x, w1_s8, w1_sc8, a8w4="0")
    except ValueError as e:
        assert "AITER_SITUV2_A8W4" in str(e), e
    else:
        raise AssertionError("a16w4 kernel ran on a relaid W1")
    return ret


def main():
    if get_gfx() != "gfx942" or get_cu_num() != 304:
        aiter.logger.warning(
            "gfx942 asm MXFP4 x MXFP8 MoE needs gfx942 with 304 CUs (MI300X / MI325X), "
            "skipping on %s / %d CUs",
            get_gfx(),
            get_cu_num(),
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="gfx942 1-stage asm MXFP4 x MXFP8 SiTUv2 MoE (Kimi-K3 shapes)",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=int,
        nargs="*",
        default=[1024, 4096],
        help="number of tokens (the tuned asm rows start at 1024)",
    )
    parser.add_argument(
        "-dim",
        "--dim",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(3584, 512), (3584, 384)],
        help="model_dim,inter_dim",
    )
    args = parser.parse_args()

    for model_dim, inter_dim in args.dim:
        check_dispatch(model_dim, inter_dim)
    df = pd.DataFrame(
        [
            test_fmoe_asm_mxfp4fp8(token, model_dim, inter_dim)
            for (model_dim, inter_dim), token in itertools.product(args.dim, args.token)
        ]
    )
    aiter.logger.info(
        "gfx942 asm MXFP4 x MXFP8 MoE summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
