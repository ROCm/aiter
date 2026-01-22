# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import torch
import itertools
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter.int4_utils import *
from aiter.utility import fp4_utils
from aiter.jit.utils.chip_info import get_gfx
import argparse
import pandas as pd
import logging

from aiter.fused_moe import (
    fused_topk,
    fused_moe,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)


from aiter.ops.shuffle import (
    shuffle_weight,
    shuffle_scale_a16w4,
    shuffle_weight_a16w4,
)

torch.int4 = getattr(torch, "int4", torch.uint32)
torch.set_default_device("cuda")


@benchmark()
def test_fmoe(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    actType,
    qType,
    AQDType,
    WQDType,
    use_g1u1=False,
    doweight_stage1=False,
    hidden_pad=0,
    intermediate_pad=0,
    preshuffle=False,
):
    num_iters = int(os.environ.get("AITER_MOE_NUM_ITERS", "5"))
    num_warmup = int(os.environ.get("AITER_MOE_NUM_WARMUP", "2"))
    # NOTE: For fp16 configs, the random init range can easily cause overflow (+/-inf) in
    # fp16 accumulation paths (both CK and FLIR), which then makes checkAllclose meaningless.
    # Allow scaling the random init down to keep values finite for correctness checks.
    # - Default: 1.0 for bf16, 0.1 for fp16 (can override via env).
    _scale_env = os.environ.get("AITER_MOE_INIT_SCALE", "").strip()
    if _scale_env:
        init_scale = float(_scale_env)
    else:
        init_scale = 0.1 if dtype == torch.float16 else 1.0
    if get_gfx() not in ["gfx950"] and qType == aiter.QuantType.per_1x32:
        return
    torch_quant = aiter.get_torch_quant(qType)
    input = torch.randn((token, model_dim), dtype=dtype) * init_scale
    if use_g1u1:
        w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) * init_scale
        if hidden_pad != 0 and intermediate_pad != 0:
            w1[:, :, -hidden_pad:] = 0
            w1[:, -intermediate_pad:, :] = 0
            w1[:, inter_dim - intermediate_pad : inter_dim, :] = 0
        exp_bias1 = torch.clamp(torch.randn((E, inter_dim * 2), dtype=dtype), -1.0, 1.0)
    else:
        w1 = torch.randn((E, inter_dim, model_dim), dtype=dtype) * init_scale
        exp_bias1 = torch.clamp(torch.randn((E * inter_dim), dtype=dtype), -1.0, 1.0)
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) * init_scale
    if hidden_pad != 0 and intermediate_pad != 0:
        w2[:, :, -intermediate_pad:] = 0
        w2[:, -hidden_pad:, :] = 0
    exp_bias2 = torch.clamp(torch.randn((E, model_dim), dtype=dtype), -1.0, 1.0)
    score = torch.randn((token, E), dtype=dtype) * init_scale
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    if qType == aiter.QuantType.per_Tensor:
        w1_qt, w1_scale = aiter.pertoken_quant(w1.view(E, -1), quant_dtype=WQDType)
        w2_qt, w2_scale = aiter.pertoken_quant(w2.view(E, -1), quant_dtype=WQDType)
        w1_qt = w1_qt.view(w1.shape)
        w2_qt = w2_qt.view(w2.shape)
    elif qType == aiter.QuantType.per_Token and WQDType == torch.int4:  # int4 w quant
        w1_qt, w1_scale = aiter.pertoken_quant(w1, quant_dtype=dtypes.i8, dtypeMax=7)
        w2_qt, w2_scale = aiter.pertoken_quant(w2, quant_dtype=dtypes.i8, dtypeMax=7)
    elif qType == aiter.QuantType.per_128x128:

        def weight_per_128x128_quant(weight, quant_dtype):
            E, dim1, dim2 = weight.shape
            weight_blocks = weight.view(
                E, dim1 // 128, 128, dim2 // 128, 128
            )  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
            weight_blocks = weight_blocks.permute(
                0, 1, 3, 2, 4
            ).contiguous()  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
            weight_blocks = weight_blocks.view(
                E, -1, 128 * 128
            )  # [E, num_blocks, 128*128]
            weight_qt, weight_scale = aiter.pertoken_quant(
                weight_blocks, quant_dtype=quant_dtype
            )
            weight_qt = weight_qt.view(
                E, dim1 // 128, dim2 // 128, 128, 128
            )  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
            weight_qt = weight_qt.permute(
                0, 1, 3, 2, 4
            ).contiguous()  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
            weight_qt = weight_qt.view(E, dim1, dim2)  # [E, dim1, dim2]
            weight_scale = weight_scale.view(
                E, dim1 // 128, dim2 // 128
            )  # [E, num_blocks_dim1, num_blocks_dim2]
            return weight_qt, weight_scale

        w1_qt, w1_scale = weight_per_128x128_quant(w1, quant_dtype=WQDType)
        w2_qt, w2_scale = weight_per_128x128_quant(w2, quant_dtype=WQDType)
    else:
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=WQDType)
        w2_qt, w2_scale = torch_quant(w2, quant_dtype=WQDType)

    if qType != aiter.QuantType.per_1x32:
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape)
    else:
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    # Quant-ing a
    if qType == aiter.QuantType.per_128x128:
        a1_qt, a1_scale = aiter.pertoken_quant(
            input.view(token, -1, 128), quant_dtype=AQDType
        )
        a1_qt = a1_qt.view(token, model_dim)
        a1_scale = a1_scale.squeeze(-1)
    elif (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
        and WQDType == dtypes.fp4x2
    ):  # a16w4 & a8w4
        a1_qt = input.to(dtypes.bf16)
        a1_scale = None
    else:
        a1_qt, a1_scale = torch_quant(input, quant_dtype=AQDType)

    # bias dtype convert
    if (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
        and (WQDType == dtypes.fp4x2)
    ):  # a16w4
        exp_bias1_aiter = exp_bias1.to(dtypes.fp32)
        exp_bias2_aiter = exp_bias2.to(dtypes.fp32)
    else:
        exp_bias1_aiter = exp_bias1 = None
        exp_bias2_aiter = exp_bias2 = None

    # pre-shuffle
    w1_scale_aiter = w1_scale
    w2_scale_aiter = w2_scale
    if WQDType == torch.int4:  # int4 w quant
        w1_qt_aiter = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(
                shuffle_weight(w1_qt_aiter, (16, 16), use_int4=True)
            )
        )
        w2_qt_aiter = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(
                shuffle_weight(w2_qt_aiter, (16, 16), use_int4=True)
            )
        )
        w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)
    elif (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
        and (WQDType == dtypes.fp4x2)
    ):  # a16w4
        w1_qt_aiter = shuffle_weight_a16w4(w1_qt_aiter, 16, True)
        w1_scale_aiter = shuffle_scale_a16w4(w1_scale, E, True)
        w2_qt_aiter = shuffle_weight_a16w4(w2_qt_aiter, 16, False)
        w2_scale_aiter = shuffle_scale_a16w4(w2_scale, E, False)
    elif WQDType != dtypes.fp4x2 or preshuffle:
        w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(16, 16))
        w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)
    else:
        w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)

    # # ######################## stage 1 start ###########
    out1_ref = torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=actType,
        quant_type=qType,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        w1_bias=exp_bias1,
        doweight=doweight_stage1,
    )

    # ######################## stage 2 start ###########
    if qType == aiter.QuantType.per_128x128:
        a2_qt, a2_scale = aiter.pertoken_quant(
            out1_ref.view(token, -1, 128), quant_dtype=AQDType
        )
        a2_scale = a2_scale.view(token, topk, -1)
    elif (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
        and (WQDType == dtypes.fp4x2)
    ):  # a16w4 & a8w4
        a2_qt = out1_ref
        a2_scale = None
    else:
        a2_qt, a2_scale = torch_quant(out1_ref, quant_dtype=AQDType)
    a2_qt = a2_qt.view(token, topk, -1)

    out2_ref = torch_moe_stage2(
        a2_qt,
        w1_qt,  # E, inter_dim*2, model_dim
        w2_qt,  # E, model_dim, inter_dim
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=qType,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        w2_bias=exp_bias2,
        doweight=not doweight_stage1,
    )

    # ######################## stage 2 end ###########
    # Only swap CK->FlyDSL for FP8 per-token case (as requested).
    # NOTE: current FlyDSL MoE 2-stage kernels implement the g1u1 (SwiGLU) path, i.e.
    # W1 shape must be [E, 2*inter_dim, model_dim]. We only enable the swap when `use_g1u1=True`.
    use_flir = (
        (qType == aiter.QuantType.per_Token)
        and (AQDType == dtypes.fp8)
        and (WQDType == dtypes.fp8)
        and bool(use_g1u1)
    )

    # Keep git-head behavior by default; allow disabling the swap to measure CK baseline:
    #   AITER_USE_FLIR_MOE=0 python3 op_tests/test_moe_2stage.py -q 2 ...
    if os.environ.get("AITER_USE_FLIR_MOE", "1") not in ("1", "true", "True", "YES", "yes"):
        use_flir = False
    if use_flir:
        import sys

        DSL2_ROOT = os.environ.get("DSL2_ROOT", None)
        if not DSL2_ROOT:
            raise RuntimeError(
                "FlyDSL path not found. Please set environment variable, e.g. "
                "`export DSL2_ROOT=/path/to/FlyDSL`"
            )
        if DSL2_ROOT not in sys.path:
            sys.path.insert(0, DSL2_ROOT)

        from kernels.moe_gemm_2stage import compile_moe_gemm1, compile_moe_gemm2  # type: ignore
        from tests.utils import shuffle_weight as dsl2_shuffle_weight  # type: ignore

        # FlyDSL tiling defaults: keep consistent with dsl2/tests/kernels/test_moe_gemm.py.
        tile_m = 64
        tile_n1 = 128
        tile_k1 = 128
        tile_n2 = 256
        tile_k2 = 128

        # Routing buffers (Aiter moe_sorting, then trim+pad like dsl2 test).
        topk_ids_i32 = topk_ids.to(torch.int32)
        topk_w_f32 = topk_weights.to(torch.float32)
        sorted_ids, sorted_w, sorted_eids, num_valid_ids, _moe_buf = moe_sorting(
            topk_ids_i32,
            topk_w_f32,
            E,
            model_dim,
            torch.float16,
            tile_m,
        )
        size_expert_ids_total = int(sorted_eids.numel())
        # Align with CK launch rules: keep routing buffers as produced by `moe_sorting` (no host trim/pad),
        # and launch full expert-block range. Per-block early-exit is controlled by `num_valid_ids`.
        blocks = int(size_expert_ids_total)
        sorted_ids = sorted_ids.contiguous()
        sorted_w = sorted_w.contiguous()
        sorted_eids = sorted_eids.contiguous()
        sorted_size = int(sorted_ids.numel())
        # Flatten/reshape inputs to match FlyDSL kernel ABI.
        # - X: fp8, scale_x: [token]
        x_q = a1_qt
        scale_x_1d = a1_scale.view(-1).contiguous() if a1_scale is not None else None
        if scale_x_1d is None:
            raise RuntimeError("FP8 per-token swap requires a1_scale (got None)")
        # Keep X as-is (no extra padding).
        x_q = x_q.contiguous().view(token, model_dim)

        # - W1/W2: use dsl2 preshuffle (layout must match FlyDSL kernel expectation)
        w1_shuf = dsl2_shuffle_weight(w1_qt)
        w2_shuf = dsl2_shuffle_weight(w2_qt)
        w1_flat = w1_shuf.contiguous().view(E * (2 * inter_dim), model_dim)
        # FlyDSL expects scales in logical row order (unshuffled), like dsl2 tests.
        w1_scale_1d = w1_scale.view(-1).contiguous()
        w1_flat = w1_flat.contiguous()

        # - sorted_weights: [sorted_size] f32
        sorted_w_1d = sorted_w.view(-1).contiguous()

        # Stage1 out: bf16 is supported in FlyDSL now; prefer bf16 to avoid fp16 overflow
        # before stage2 quantization.
        out1_flir = torch.empty((token, topk, inter_dim), device="cuda", dtype=dtype)
        exe1 = compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=E,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n1,
            tile_k=tile_k1,
            doweight_stage1=bool(doweight_stage1),
            in_dtype="fp8",
            out_dtype="bf16" if dtype == torch.bfloat16 else "f16",
            use_cshuffle_epilog=False,
        )

        # Stage2 compile once (compile inside perftest can be unstable + distorts perf).
        exe2 = compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=E,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n2,
            tile_k=tile_k2,
            doweight_stage2=bool(not doweight_stage1),
            in_dtype="fp8",
            # For bf16 test configs, fp16 atomic accumulation can overflow to +/-inf.
            # Use fp32 atomics for correctness; cast to `dtype` for comparison below.
            out_dtype="f32" if dtype == torch.bfloat16 else "f16",
        )
        w2_flat = w2_shuf.contiguous().view(E * model_dim, inter_dim).contiguous()
        w2_scale_1d = w2_scale.view(-1).contiguous()
        # FlyDSL stage2 output dtype follows `out_dtype` above.
        out2 = torch.empty(
            (token, model_dim),
            device="cuda",
            dtype=(torch.float32 if dtype == torch.bfloat16 else torch.float16),
        )

        # Precompute stage2 inputs once; timing should include only GEMM kernels.
        def launch_stage1():
            exe1(
                out1_flir,
                x_q,
                w1_flat,
                scale_x_1d,
                w1_scale_1d,
                sorted_ids,
                sorted_eids,
                sorted_w_1d,
                num_valid_ids,
                token,
                inter_dim,
                model_dim,
                int(blocks),
            )
            return out1_flir

        _, us1 = run_perftest(launch_stage1, num_iters=num_iters, num_warmup=num_warmup)

        # Quantize stage1->stage2 input exactly like git-head reference (outside timing).
        a2_qt_flir, a2_scale_flir = torch_quant(out1_flir.to(dtype), quant_dtype=AQDType)
        a2_qt_flir = a2_qt_flir.view(token, topk, -1).contiguous()

        a2_qt_flat = a2_qt_flir.contiguous().view(-1)
        a2_scale_1d = a2_scale_flir.view(-1).contiguous()

        def launch_stage2():
            out2.zero_()
            exe2(
                out2,
                a2_qt_flat,
                w2_flat.view(-1),
                a2_scale_1d,
                w2_scale_1d,
                sorted_ids,
                sorted_eids,
                sorted_w_1d,
                num_valid_ids,
                token,
                model_dim,
                inter_dim,
                int(blocks),
            )
            return out2

        out2_ck, us2 = run_perftest(launch_stage2, num_iters=num_iters, num_warmup=num_warmup)
        us2 = us1 + us2
        print(f"[aiter] flir_moe_gemm1: {us1:8.2f} us, flir_moe_gemm2: {us2 - us1:8.2f} us")
        # Keep git-head test behavior: compare in the original `dtype` (often bf16).
        out2_ck = out2_ck.to(dtype)
    else:
        # Optional CK kernel-only timing for FP8 per-token g1u1.
        _ck_kernel_only = (
            (qType == aiter.QuantType.per_Token)
            and (AQDType == dtypes.fp8)
            and (WQDType == dtypes.fp8)
            and bool(use_g1u1)
        )
        if _ck_kernel_only:
            # Use the same routing buffers as the FLIR path for fair per-kernel timing.
            block_m = 64
            topk_ids_i32 = topk_ids.to(torch.int32)
            topk_w_f32 = topk_weights.to(torch.float32)
            sorted_ids, sorted_w, sorted_eids, num_valid_ids, _moe_buf = moe_sorting(
                topk_ids_i32,
                topk_w_f32,
                E,
                model_dim,
                torch.float16,
                block_m,
            )
            # Keep routing buffers as-is (no host trim/pad). Launch full expert-block range.
            blocks = int(sorted_eids.numel())
            sorted_ids = sorted_ids.contiguous()
            sorted_w = sorted_w.contiguous()
            sorted_eids = sorted_eids.contiguous()

            out1_ck = torch.empty((token, topk, inter_dim), device="cuda", dtype=dtype)

            def launch_ck_stage1():
                aiter.ck_moe_stage1_fwd(
                    a1_qt,
                    w1_qt_aiter,
                    w2_qt_aiter,
                    sorted_ids,
                    sorted_eids,
                    num_valid_ids,
                    out1_ck,
                    topk,
                    "",
                    w1_scale_aiter,
                    a1_scale,
                    block_m,
                    sorted_w,
                    qType,
                    actType,
                    1,
                    dtype,
                )
                return out1_ck

            _, us1 = run_perftest(launch_ck_stage1, num_iters=num_iters, num_warmup=num_warmup)

            # Prepare stage2 inputs from the same reference stage1 output as git-head path,
            # so stage2 correctness is comparable and independent of stage1 behavior.
            a2_qt_ck, a2_scale_ck = torch_quant(out1_ref.to(dtype), quant_dtype=AQDType)
            a2_qt_ck = a2_qt_ck.view(token, topk, -1).contiguous()

            # Stage2 accumulates into `out`, so ensure we start from zero for correctness.
            out2_ck = torch.zeros((token, model_dim), device="cuda", dtype=dtype)

            def launch_ck_stage2():
                out2_ck.zero_()
                aiter.ck_moe_stage2_fwd(
                    a2_qt_ck,
                    w1_qt_aiter,
                    w2_qt_aiter,
                    sorted_ids,
                    sorted_eids,
                    num_valid_ids,
                    out2_ck,
                    topk,
                    "",
                    w2_scale_aiter,
                    a2_scale_ck,
                    block_m,
                    sorted_w,
                    qType,
                    actType,
                )
                return out2_ck

            _, us2 = run_perftest(launch_ck_stage2, num_iters=num_iters, num_warmup=num_warmup)
            us2 = us1 + us2
            print(f"[aiter] ck_moe_gemm1: {us1:8.2f} us, ck_moe_gemm2: {us2 - us1:8.2f} us")
        else:
            out2_ck, us2 = run_perftest(
                fused_moe,
                input,
                w1_qt_aiter,
                w2_qt_aiter,
                topk_weights,
                topk_ids,
                w1_scale=w1_scale_aiter,
                w2_scale=w2_scale_aiter,
                quant_type=qType,
                activation=actType,
                doweight_stage1=doweight_stage1,
                intermediate_pad=intermediate_pad,
                hidden_pad=hidden_pad,
                bias1=exp_bias1_aiter,
                bias2=exp_bias2_aiter,
                num_iters=num_iters,
                num_warmup=num_warmup,
            )

    err = checkAllclose(
        out2_ref,
        out2_ck,
        msg=f"ck_moe_2stages:{us2:>8.2f} us, {token*model_dim*inter_dim*3*topk*2/us2/1000/1000:>8.2f} tflops......(quant:{AQDType})",
    )

    def calc_diff(x: torch.Tensor, y: torch.Tensor):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        sim = 2 * (x * y).sum() / denominator
        return 1 - sim

    # Optional stage1 localization logs removed (debug-only).

    logits_diff = calc_diff(out2_ref, out2_ck)
    if logits_diff > 1e-3:
        logging.warning(
            f"logits_diff: {logits_diff} is too large, please check the implementation"
        )

    return {"us": us2, "err": err}


# Keep both bf16/fp16 available via CLI so we can run FLIR swap (which expects fp16 outputs).
l_dtype = ["bf16", "fp16"]
# l_dim = [(6144, 4096)]
l_dim = [(7168, 256)]
# l_dim = [(3072, 3072)]
l_tokenNum = [
    1,
    3,
    5,
    16,
    32,
    64,
    128,
    256,
    1024,
    4096,
    163840,
]
l_quant = [
    (aiter.QuantType.No, None, None),  # a16w16
    (aiter.QuantType.per_Tensor, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_Token, dtypes.fp8, torch.int4),  # a8w4
    (aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2),  # a4w4
    (aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_1x32, dtypes.bf16, dtypes.fp4x2),  # a16w4
    (aiter.QuantType.per_1x32, dtypes.fp8, dtypes.fp4x2),  # a8w4
]
l_act = [aiter.ActivationType.Silu, aiter.ActivationType.Gelu][:1]
l_doweight_stage1 = [False, True][:1]
# l_hidden_intermediate_pad = [(0, 0), (65, 65), (129, 191)][1:2]
l_hidden_intermediate_pad = [(0, 0), (192, 128), (129, 191)][1:2]
l_preshuffle = [False, True]


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)

parser.add_argument(
    "-dim",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""Model dimension.
    e.g.: -dim 6144,4096""",
)

parser.add_argument(
    "-t",
    "--tokenNum",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="""Number of tokens.
    e.g.: -t 1024""",
)

parser.add_argument(
    "-q",
    "--quant",
    type=int,
    choices=range(len(l_quant)),
    help="""select quantization type:
    0 : aiter.QuantType.No, None, None),  # a16w16
    1: aiter.QuantType.per_Tensor, dtypes.fp8, dtypes.fp8  # a8w8
    2: aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8  # a8w8
    3: aiter.QuantType.per_Token, dtypes.fp8, torch.int4  # a8w4
    4: aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2  # a4w4
    5: aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8,  # a8w8,
    6: aiter.QuantType.per_1x32, dtypes.bf16, dtypes.fp4x2,  # a16w4,
    7: aiter.QuantType.per_1x32, dtypes.fp8, dtypes.fp4x2,  # a8w4,""",
)

parser.add_argument(
    "-a",
    "--act",
    type=str,
    choices=["silu", "gelu"],
    default=None,
    help="""Select activation type.
    e.g.: -a silu""",
)

parser.add_argument(
    "-s",
    "--doweight_stage1",
    type=dtypes.str2bool,
    nargs="?",
    const=None,
    default=None,
    help="""Whether to do weight in stage 1. Default is [False, True].
    -s f    # False.
    -s t    # True.""",
)

parser.add_argument(
    "-e",
    "--expert",
    type=int,
    default=8,
    help="""Number of experts.
    e.g.: -e 8""",
)

parser.add_argument(
    "-k",
    "--topk",
    type=int,
    default=2,
    help="""Number of top experts.
    e.g.: -k 2""",
)

parser.add_argument(
    "-p",
    "--preshuffle",
    type=dtypes.str2bool,
    nargs="?",
    const=None,
    default=None,
    help="""Whether to use pre-shuffle weight mode. Default is [False, True].
    -p f    # False.
    -p t    # True.""",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]

if args.dim is not None:
    l_dim = [args.dim]

if args.tokenNum is not None:
    l_tokenNum = [args.tokenNum]

l_quant = [l_quant[args.quant]] if args.quant is not None else l_quant

if args.act is not None:
    l_act = [getattr(aiter.ActivationType, args.act.capitalize())]

if args.doweight_stage1 is not None:
    l_doweight_stage1 = [args.doweight_stage1]

if args.preshuffle is not None:
    l_preshuffle = [args.preshuffle]

df = []
for (
    dtype,
    (quant_type, aq_dtype, wq_dtype),
    (model_dim, inter_dim),
    doweight_stage1,
) in itertools.product(l_dtype, l_quant, l_dim, l_doweight_stage1):
    if (quant_type, aq_dtype, wq_dtype) == (
        aiter.QuantType.per_1x32,
        dtypes.bf16,
        dtypes.fp4x2,
    ):
        for hidden_pad, intermediate_pad in l_hidden_intermediate_pad:
            for m in l_tokenNum:
                ret = test_fmoe(
                    dtype,
                    m,
                    model_dim,
                    inter_dim,
                    args.expert,
                    args.topk,
                    aiter.ActivationType.Swiglu,
                    quant_type,
                    aq_dtype,
                    wq_dtype,
                    use_g1u1=True,
                    doweight_stage1=doweight_stage1,
                    hidden_pad=hidden_pad,
                    intermediate_pad=intermediate_pad,
                )
                df.append(ret)
    elif (quant_type, aq_dtype, wq_dtype) == (
        aiter.QuantType.per_1x32,
        dtypes.fp8,
        dtypes.fp4x2,
    ):
        for hidden_pad, intermediate_pad in l_hidden_intermediate_pad:
            for m in l_tokenNum:
                ret = test_fmoe(
                    dtype,
                    m,
                    model_dim,
                    inter_dim,
                    args.expert,
                    args.topk,
                    aiter.ActivationType.Swiglu,
                    quant_type,
                    aq_dtype,
                    wq_dtype,
                    use_g1u1=True,
                    doweight_stage1=doweight_stage1,
                    hidden_pad=hidden_pad,
                    intermediate_pad=intermediate_pad,
                )
                df.append(ret)
    elif (quant_type, aq_dtype, wq_dtype) == (
        aiter.QuantType.per_1x32,
        dtypes.fp4x2,
        dtypes.fp4x2,
    ):
        for preshuffle in l_preshuffle:
            for act_type in l_act:
                for m in l_tokenNum:
                    ret = test_fmoe(
                        dtype,
                        m,
                        model_dim,
                        inter_dim,
                        args.expert,
                        args.topk,
                        act_type,
                        quant_type,
                        aq_dtype,
                        wq_dtype,
                        use_g1u1=True,
                        doweight_stage1=doweight_stage1,
                        preshuffle=preshuffle,
                    )
                    df.append(ret)
    else:
        for act_type in l_act:
            for m in l_tokenNum:
                ret = test_fmoe(
                    dtype,
                    m,
                    model_dim,
                    inter_dim,
                    args.expert,
                    args.topk,
                    act_type,
                    quant_type,
                    aq_dtype,
                    wq_dtype,
                    use_g1u1=True,
                    doweight_stage1=doweight_stage1,
                )
                df.append(ret)
df = pd.DataFrame(df)
try:
    df_md = df.to_markdown(index=False)
    aiter.logger.info("moe_2stage summary (markdown):\n%s", df_md)
except ImportError:
    # `tabulate` is optional for pandas markdown output; keep the test runnable without it.
    aiter.logger.info("moe_2stage summary:\n%s", df.to_string(index=False))
