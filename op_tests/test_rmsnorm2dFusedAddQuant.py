# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: N999  camelCase module name kept: renaming would break the test
# selection lists and FILE_TIMES bookkeeping that reference this path.

import argparse
import itertools
from functools import partial

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import QuantType, dtypes, get_gfx, get_torch_quant
from aiter.test_common import benchmark, checkAllclose, perftest, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

_FP4_OUTPUT_GUARD_VALUE = 0xA5


@perftest(num_warmup=0, num_iters=10)
def run_torch(
    input,
    weight,
    eps,
    residual=None,
    x_scale=None,
    q_dtype=None,
    quant_type=QuantType.per_Token,
):
    quant_func = get_torch_quant(quant_type)
    if residual is None:
        residual_out = None
        output = F.rms_norm(
            input=input, normalized_shape=(input.shape[-1],), weight=weight, eps=eps
        )
    else:
        residual_out = input + residual
        output = F.rms_norm(
            input=residual_out,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            eps=eps,
        )
    if q_dtype is None:
        y_scale = None
        output_q = output
    else:
        if x_scale is None:
            output_q, y_scale = quant_func(output, quant_dtype=q_dtype)
        else:
            output_q, y_scale = quant_func(output, x_scale=x_scale, quant_dtype=q_dtype)
    return output_q, residual_out, y_scale, output


@perftest()
def run_ck(
    input,
    weight,
    eps,
    residual=None,
    x_scale=None,
    q_dtype=None,
    quant_type=QuantType.No,
    model_sensitive=0,
):
    out_before_quant = None
    if quant_type == QuantType.No:
        y_scale = None
        if residual is None:
            residual_out = None
            output = aiter.rmsnorm2d_fwd(input, weight, eps)
        elif residual is not None:
            residual_out = torch.empty_like(input)
            output = torch.empty_like(input)
            aiter.rmsnorm2d_fwd_with_add(
                output, input, residual, residual_out, weight, eps
            )
    elif x_scale is None:
        y_scale = torch.empty(input.shape[0], 1, dtype=dtypes.fp32)
        output = torch.empty(input.shape, dtype=q_dtype)
        if residual is None:
            residual_out = None
            aiter.rmsnorm2d_fwd_with_dynamicquant(
                output, input, y_scale, weight, eps, model_sensitive
            )
        elif residual is not None:
            residual_out = torch.empty_like(input)
            aiter.rmsnorm2d_fwd_with_add_dynamicquant(
                output,
                input,
                residual,
                residual_out,
                y_scale,
                weight,
                eps,
                model_sensitive,
            )
    else:
        y_scale = torch.empty(input.shape[0], 1, dtype=dtypes.fp32)
        output = torch.empty(input.shape, dtype=q_dtype)
        if residual is None:
            residual_out = None
            aiter.rmsnorm2d_fwd_with_smoothquant(
                output, input, x_scale, y_scale, weight, eps, model_sensitive
            )
        elif residual is not None:
            residual_out = torch.empty_like(input)
            out_before_quant = torch.empty_like(input)
            aiter.rmsnorm2d_fwd_with_add_smoothquant(
                output,
                input,
                residual,
                residual_out,
                x_scale,
                y_scale,
                weight,
                eps,
                out_before_quant=out_before_quant,
            )

    return output, residual_out, y_scale, out_before_quant


@perftest()
def run_hip(
    input,
    weight,
    eps,
    residual,
    q_dtype=None,
    quant_type=QuantType.No,
):
    if quant_type == QuantType.No:
        group_size = 0
    elif quant_type == QuantType.per_Token:
        group_size = 0
        scale_shape = (input.shape[0], 1)
    elif quant_type == QuantType.per_1x32:
        group_size = 32
    elif quant_type == QuantType.per_1x128:
        group_size = 128
    else:
        raise ValueError(f"Unsupported quant type: {quant_type}")
    output_guard = None
    if quant_type in [QuantType.per_1x32, QuantType.per_1x128]:
        group_per_row = (input.shape[1] + group_size - 1) // group_size
        if q_dtype == dtypes.fp4x2:
            scale_per_row = (group_per_row + 7) // 8 * 8 // 4
        else:
            scale_per_row = group_per_row
        scale_shape = (input.shape[0], scale_per_row)
    residual_out = torch.empty_like(input)
    if quant_type == QuantType.No:
        scale = None
        output = torch.empty_like(input)
        if residual is None:
            residual_out = None
            aiter.rmsnorm(output, input, weight, eps)
        else:
            residual_out = torch.empty_like(input)
            aiter.add_rmsnorm(output, input, residual, residual_out, weight, eps)
    else:
        if q_dtype == dtypes.fp4x2:
            output_storage = torch.empty(
                (input.shape[0] + 1, input.shape[1] // 2), dtype=q_dtype
            )
            output = output_storage[:-1]
            output_guard = output_storage[-1:].view(torch.uint8)
            output_guard.fill_(_FP4_OUTPUT_GUARD_VALUE)
        else:
            output = torch.empty(input.shape, dtype=q_dtype)
        scale = torch.empty(scale_shape, dtype=dtypes.fp32)
        if residual is None:
            residual_out = None
            aiter.rmsnorm_quant(output, input, scale, weight, eps, group_size)
        else:
            residual_out = torch.empty_like(input)
            aiter.add_rmsnorm_quant(
                output, input, residual, residual_out, scale, weight, eps, group_size
            )
    return output, residual_out, scale, output_guard


@benchmark()
def test_rmsnorm(
    m,
    n,
    dtype=torch.bfloat16,
    add_residual=False,
    smoothquant=False,
    quant_dtype=None,
    quant_type=QuantType.No,
):
    if quant_dtype is dtypes.fp4x2 and quant_type == QuantType.per_Token:
        print("fp4x2 per token is not supported")
        return {}
    elif quant_type == QuantType.per_1x32 and (
        quant_dtype is not dtypes.fp4x2 or get_gfx() not in ["gfx950"]
    ):
        print("per_1x32 is only supported for fp4x2 on gfx950")
        return {}
    dim = (m, n)
    scale_type = dtypes.fp32
    input = torch.randn(dim, dtype=dtype)
    weight = torch.randn(n, dtype=dtype)
    res = torch.randn(dim, dtype=dtype) if add_residual else None
    xscale = torch.randn(n, dtype=scale_type) if smoothquant else None

    def calculateTensorsSize(*args):
        num_btype = 0
        for el in args:
            if isinstance(el, torch.Tensor):
                num_btype += el.element_size() * el.numel()
        return num_btype

    read_datasize = calculateTensorsSize(input, weight, res, xscale)

    atol = 1 if quant_dtype == dtypes.i8 else 1e-2
    ret = {}
    (a, res_a, yscale_a, _), avg_a = run_torch(
        input,
        weight,
        1e-5,
        residual=res,
        x_scale=xscale,
        q_dtype=quant_dtype,
        quant_type=quant_type,
    )
    write_datasize = calculateTensorsSize(a, res_a, yscale_a)
    ret["torch us"] = avg_a
    if quant_type in [QuantType.per_Token, QuantType.No] and quant_dtype in [
        None,
        dtypes.fp8,
        dtypes.i8,
    ]:
        (b, res_b, yscale_b, _), avg_b = run_ck(
            input,
            weight,
            1e-5,
            residual=res,
            x_scale=xscale,
            q_dtype=quant_dtype,
            quant_type=quant_type,
        )
        err_ck = checkAllclose(
            a.to(dtypes.fp32), b.to(dtypes.fp32), rtol=0, atol=atol, msg="check ck out"
        )
        if add_residual:
            checkAllclose(res_a, res_b, msg="check ck res")
        if quant_type != QuantType.No:
            checkAllclose(yscale_a, yscale_b, msg="check ck scale")
        ret["ck us"] = avg_b
        ret["ck err"] = err_ck
        ret["ck bw(GB/s)"] = (
            (read_datasize + write_datasize) / avg_b / 1024 / 1024 / 1024 * 1e6
        )
    if not smoothquant and n <= 8192:
        (c, res_c, yscale_c, output_guard), avg_c = run_hip(
            input, weight, 1e-5, res, q_dtype=quant_dtype, quant_type=quant_type
        )
        if output_guard is not None:
            changed_bytes = torch.count_nonzero(
                output_guard != _FP4_OUTPUT_GUARD_VALUE
            ).item()
            assert changed_bytes == 0, (
                f"{'add_' if add_residual else ''}rmsnorm_quant wrote "
                f"{changed_bytes} bytes past an FP4 output row"
            )
        if quant_dtype == dtypes.fp4x2:
            a = fp4_utils.mxfp4_to_f32(a)
            c = fp4_utils.mxfp4_to_f32(c)
        err_hip = checkAllclose(
            a.to(dtypes.fp32), c.to(dtypes.fp32), rtol=0, atol=atol, msg="check hip out"
        )
        if add_residual:
            checkAllclose(res_a, res_c, msg="check hip res")
        if quant_type != QuantType.No:
            checkAllclose(
                yscale_a.view(torch.float32),
                yscale_c.view(torch.float32),
                msg="check hip scale",
            )
        ret["hip us"] = avg_c
        ret["hip err"] = err_hip
        ret["hip bw(GB/s)"] = (
            (read_datasize + write_datasize) / avg_c / 1024 / 1024 / 1024 * 1e6
        )

    return ret


@benchmark()
def _round_before_quant_case(m, n, dtype, gemma, pattern):
    """Match the T-rounded split path, not direct FP32 quantization."""
    torch.manual_seed(2026)
    x = torch.randn((m, n), dtype=dtype, device="cuda") * 0.2
    residual = torch.randn_like(x)
    weight = torch.randn(n, dtype=dtype, device="cuda") * 0.1
    if pattern == "zero":
        x.zero_()
        residual.zero_()
    elif pattern == "cancel":
        residual = -x
    elif pattern == "small":
        x *= 1e-5
        residual *= 1e-5
    norm = torch.empty_like(x)
    q = torch.empty_like(x, dtype=dtypes.fp8)
    scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    res_out = torch.empty_like(x)

    def split():
        aiter.rmsnorm2d_fwd_with_add(norm, x, residual, res_out, weight, 1e-6, gemma)
        aiter.dynamic_per_token_scaled_quant(q, norm, scale)

    def fused():
        aiter.add_rmsnorm_quant(
            q,
            x,
            residual,
            res_out,
            scale,
            weight,
            1e-6,
            gemma_norm=gemma,
            round_before_quant=True,
        )

    split()
    expected = [v.clone() for v in (q, scale, res_out)]
    # Independent FP32 reference for norm and residual.
    value = x.float() + residual.float()
    ref = (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        * (weight.float() + int(gemma))
    ).to(dtype)
    checkAllclose(ref.float(), norm.float(), rtol=1e-2, atol=1e-2, tol_err_ratio=0)
    assert torch.equal(res_out, value.to(dtype))
    ret = {"gfx": get_gfx()}
    candidates = {"split": split, "fused": fused}
    for name, fn in candidates.items():
        fn()
        for actual, want in zip((q, scale, res_out), expected):
            assert torch.equal(actual.view(torch.uint8), want.view(torch.uint8))
        err = checkAllclose(
            expected[0].float() * expected[1],
            q.float() * scale,
            rtol=0,
            atol=0,
            tol_err_ratio=0,
            msg=f"{name}: match split quant",
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graph.replay()
        _, us = run_perftest(graph.replay, num_iters=7, num_warmup=2, num_rotate_args=1)
        for actual, want in zip((q, scale, res_out), expected):
            assert torch.equal(actual.view(torch.uint8), want.view(torch.uint8))
        # Logical local arithmetic and memory traffic; reduction overhead excluded.
        flops = (10 + int(gemma)) * m * n + 2 * m
        nbytes = (3 * m * n + n) * x.element_size() + m * n + 4 * m
        if name == "split":
            nbytes += 2 * m * n * x.element_size()
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def test_round_before_quant():
    import pytest

    if not torch.cuda.is_available() or get_gfx() != "gfx1250":
        pytest.skip("requires gfx1250")
    # 96 catches the standalone quantizer's non-floored narrow-row path.
    rows = [
        _round_before_quant_case(m, n, dtype, gemma, pattern)
        for (m, n), dtype, gemma, pattern in itertools.product(
            [
                (3, 32),
                (3, 64),
                (3, 96),
                (3, 128),
                (3, 160),
                (12, 512),
                (12, 4096),
                (12, 6144),
                (33, 6176),
                (33, 8192),
            ],
            [dtypes.bf16, dtypes.fp16],
            [False, True],
            ["random", "zero", "cancel", "small"],
        )
    ]
    aiter.logger.info(
        "round-before-quant summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


def test_round_before_quant_rejects_unsupported():
    import pytest

    if not torch.cuda.is_available() or get_gfx() != "gfx1250":
        pytest.skip("requires gfx1250")
    # Rejected configurations are tested, not silently excluded from the sweep.
    for n, out_dtype, group_size, shuffle, layout, scale_dtype in [
        (0, dtypes.fp8, 0, False, False, torch.float32),
        (128, dtypes.fp8, 0, False, False, torch.uint8),
        (136, dtypes.fp8, 0, False, False, torch.float32),
        (2056, dtypes.fp8, 0, False, False, torch.float32),
        (8320, dtypes.fp8, 0, False, False, torch.float32),
        (128, dtypes.i8, 0, False, False, torch.float32),
        (128, dtypes.fp8, 32, False, False, torch.float32),
        (128, dtypes.fp8, 0, True, False, torch.float32),
        (128, dtypes.fp8, 32, False, True, torch.uint8),
    ]:
        x = torch.zeros((2, n), dtype=dtypes.bf16, device="cuda")
        out = torch.empty_like(x, dtype=out_dtype)
        scale = torch.empty((2, n), dtype=scale_dtype, device=x.device)
        with pytest.raises(RuntimeError, match="round_before_quant requires"):
            aiter.add_rmsnorm_quant(
                out,
                x,
                x,
                torch.empty_like(x),
                scale,
                torch.ones(n, dtype=x.dtype, device=x.device),
                1e-6,
                group_size=group_size,
                shuffle_scale=shuffle,
                scale_layout_m32k4=layout,
                round_before_quant=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="test_rmsnorm2dFusedSQuant",
        description="Test ck rmsnorm2d Fused add and SmoothQuant",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="1: test_rmsnorm2d, \n2:test_rmsnorm2d_fuseAdd, \n"
        + "3:test_rmsnorm2d_fuseSmoothquant, \n4:test_rmsnorm2d_fuseAdd_Smoothquant"
        + "5:test_rmsnorm2d_fuseDynamicquant_per_Token, \n6:test_rmsnorm2d_fuseAdd_Dynamicquant_per_Token"
        + "7:test_rmsnorm2d_fuseAdd_fuseDynamicquant_per_1x128, \n8:test_rmsnorm2d_fuseAdd_fuseDynamicquant_per_1x32",
        default=1,
    )
    parser.add_argument(
        "-q",
        "--quant_dtype",
        type=dtypes.str2Dtype,
        default=dtypes.d_dtypes["i8"],
        # nargs="*",
        choices=[
            dtypes.d_dtypes["i8"],
            dtypes.d_dtypes["fp8"],
            dtypes.d_dtypes["fp4x2"],
        ],
        help="""Quantization data types.
    e.g.: --quant_dtype i8 fp8 fp4x2""",
    )
    parser.add_argument(
        "-m",
        type=int,
        default=[8, 256, 256 * 8, 256 * 10, 32768],
        nargs="*",
        help="""M of mnk.
    e.g.: -m 32""",
    )
    parser.add_argument(
        "-n",
        type=int,
        # 5120 and 6144 cover the n in (4096, 6144] band, which dispatches to
        # thread_data_size=24; without them no shape here exercises a group quant
        # whose group_size is not a multiple of the per-thread chunk.
        default=[1024, 2048, 3584, 4096, 5120, 6144, 8192],
        nargs="*",
        help="""N of mnk.
    e.g.: -n 1024""",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        default=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
    )
    args = parser.parse_args()
    if args.mode == 1:
        test_rmsnorm_func = partial(
            test_rmsnorm, quant_type=QuantType.No, add_residual=False
        )
    elif args.mode == 2:
        test_rmsnorm_func = partial(
            test_rmsnorm, quant_type=QuantType.No, add_residual=True
        )
    elif args.mode == 3:
        test_rmsnorm_func = partial(
            test_rmsnorm, quant_type=QuantType.No, add_residual=False, smoothquant=True
        )
    elif args.mode == 4:
        test_rmsnorm_func = partial(
            test_rmsnorm, quant_type=QuantType.No, add_residual=True, smoothquant=True
        )
    elif args.mode == 5:
        test_rmsnorm_func = partial(
            test_rmsnorm, quant_type=QuantType.per_Token, add_residual=False
        )
    elif args.mode == 6:
        test_rmsnorm_func = partial(
            test_rmsnorm, quant_type=QuantType.per_Token, add_residual=True
        )
    elif args.mode == 7:
        test_rmsnorm_func = partial(
            test_rmsnorm, quant_type=QuantType.per_1x128, add_residual=True
        )
    elif args.mode == 8:
        test_rmsnorm_func = partial(
            test_rmsnorm, quant_type=QuantType.per_1x32, add_residual=True
        )

    df = []
    for n in args.n:
        for m in args.m:
            for dtype in args.dtype:
                ret = test_rmsnorm_func(
                    m,
                    n,
                    dtype=dtype,
                    quant_dtype=args.quant_dtype if args.mode not in [1, 2] else None,
                )
            df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("rmsnorm2d summary (markdown):\n%s", df_md)
