# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 a8w4 (MXFP8 A x MXFP4 W) SiTUv2 MoE through the production fused_moe path.

``AITER_SITUV2_A8W4=1`` makes fused_moe quantise x to MXFP8 (E4M3FNUZ) and dispatch the
``_g942`` stage-1 kernels (``moe_2stage_a16wmix/gemm1_a8w4_gfx942.py``) with the a16w4
stage 2, using the gfx942 rows of ``kimik3_a8w4_tuned_fmoe.csv``. W1 / W1-scale are the
a16w4 preshuffle relaid by ``shuffle_{weight,scale}_a8w4_gfx942``.

Candidates (both through fused_moe):
  a16w4  bf16 x (the default path)       -> checked against the bf16 torch reference
  a8w4   MXFP8 x (AITER_SITUV2_A8W4=1)   -> checked against the torch reference on x rounded
                                            to MXFP8 (same maths, so near-exact) and, as
                                            ``bf16 err``, against the bf16 reference
"""

import argparse
import itertools
import os

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.gemm1_a8w4_gfx942 import (
    mxfp8_quant_a8w4_gfx942_ref,
    shuffle_scale_a8w4_gfx942,
    shuffle_weight_a8w4_gfx942,
)
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]  # the a8w4 port decodes FP4 -> FP8 for the gfx942 fp8 MFMA
EXPERT, TOPK = 896, 16  # Kimi-K3 routed experts
BETA, LINEAR_BETA = 4.0, 25.0  # Kimi-K3 SiTUv2


def mxfp8_round(x):
    """x rounded to MXFP8 E4M3FNUZ and back (exact in bf16): the a8w4 kernel's input."""
    M, K = x.shape
    q, e8m0 = mxfp8_quant_a8w4_gfx942_ref(x)
    # undo the a8w4 byte order (0,2,4,6,1,3,5,7) within each 8
    q = q.view(torch.float8_e4m3fnuz).view(M, K // 8, 2, 4).transpose(-1, -2)
    scale = torch.exp2(e8m0.float() - 127).repeat_interleave(32, dim=1)
    return (q.reshape(M, K).float() * scale).to(x.dtype)


def run_torch(x, w1_qt, w2_qt, w1_scale, w2_scale, topk_weights, topk_ids, inter_dim):
    """bf16 SiTUv2 SEPARATED MoE reference. Not timed."""
    token, dtype = x.shape[0], x.dtype
    o1 = torch_moe_stage1(
        x,
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w1_scale,
        doweight=False,
        situ_beta=BETA,
        situ_linear_beta=LINEAR_BETA,
    )
    return torch_moe_stage2(
        o1.view(token, TOPK, inter_dim),
        w1_qt.view(dtypes.fp4x2),
        w2_qt.view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=None,
        doweight=True,
    )


def cos_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


@benchmark()
def test_moe_a8w4_gfx942(token, model_dim, inter_dim, dtype):
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
    w1_scale_e = w1_scale.view(E, inter_dim * 2, model_dim // 32)
    w2_scale_e = w2_scale.view(E, model_dim, inter_dim // 32)

    ref_args = (w1_qt, w2_qt, w1_scale_e, w2_scale_e, topk_weights, topk_ids, inter_dim)
    ref_bf16 = run_torch(x, *ref_args)
    ref_mxfp8 = run_torch(mxfp8_round(x), *ref_args)

    # caller contract: a16w4 GGUU preshuffle; a8w4 relays W1 / W1-scale once more
    w1_s = shuffle_weight_a16w4(w1_qt, 16, False)
    w2_s = shuffle_weight_a16w4(w2_qt, 16, False)
    w1_sc = shuffle_scale_a16w4(w1_scale, E, False)
    w2_sc = shuffle_scale_a16w4(w2_scale, E, False)
    w1_s8 = shuffle_weight_a8w4_gfx942(w1_s).view(w1_s.dtype)
    w1_sc8 = shuffle_scale_a8w4_gfx942(w1_sc).view(w1_sc.dtype)

    def moe(a8w4):
        def f():
            # fused_moe reads the opt-in flag per call
            if a8w4:
                os.environ["AITER_SITUV2_A8W4"] = "1"
            try:
                return fused_moe(
                    x,
                    w1_s8 if a8w4 else w1_s,
                    w2_s,
                    topk_weights,
                    topk_ids,
                    w1_scale=w1_sc8 if a8w4 else w1_sc,
                    w2_scale=w2_sc,
                    quant_type=QuantType.per_1x32,
                    activation=ActivationType.Situv2,
                    doweight_stage1=False,
                    gate_mode=GateMode.SEPARATED.value,
                    beta=BETA,
                    linear_beta=LINEAR_BETA,
                )
            finally:
                os.environ.pop("AITER_SITUV2_A8W4", None)

        return f

    candidates = {"a16w4": (moe(False), ref_bf16), "a8w4": (moe(True), ref_mxfp8)}

    flops = (
        2 * token * TOPK * 3 * inter_dim * model_dim
    )  # gate+up (2I x D) + down (D x I)
    n_exp = min(E, token * TOPK)  # experts touched
    w_bytes = n_exp * 3 * inter_dim * model_dim // 2 * (1 + 1 / 16)  # fp4 + e8m0 / 32
    nbytes = w_bytes + 2 * token * model_dim * x.element_size()

    ret = {"gfx": get_gfx()}
    outs = {}
    for name, (fn, ref) in candidates.items():
        out, us = run_perftest(fn)
        outs[name] = out
        assert not out.isnan().any().item(), f"{name}: NaN in output"
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: fused_moe vs torch",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        ret[f"{name} cos"] = cos_diff(ref, out)
    # the precision cost of the opt-in path: a8w4 vs the bf16-activation reference
    ret["a8w4 bf16 cos"] = cos_diff(ref_bf16, outs["a8w4"])
    assert ret["a16w4 cos"] < 1e-2, ret
    assert ret["a8w4 cos"] < 1e-3, ret
    assert ret["a8w4 bf16 cos"] < 1e-2, ret
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("a8w4 gfx942 MoE unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="gfx942 a8w4 SiTUv2 MoE (Kimi-K3 shapes) through fused_moe",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="activation / output dtype",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=int,
        nargs="*",
        default=[1, 2, 16, 128],
        help="number of tokens",
    )
    parser.add_argument(
        "-dim",
        "--dim",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(3584, 512), (3584, 384)],
        help="model_dim,inter_dim (Kimi-K3 latent MoE; inter 512 = TP6, 384 = TP8)",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = [
            test_moe_a8w4_gfx942(token, model_dim, inter_dim, dtype)
            for (model_dim, inter_dim), token in itertools.product(args.dim, args.token)
        ]
        df = pd.DataFrame(df)
        aiter.logger.info(
            "a8w4 gfx942 MoE summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
