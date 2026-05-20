import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
import aiter
from aiter.test_common import run_perftest, checkAllclose, benchmark
from aiter import dtypes
import functools
import pandas as pd
import argparse
from aiter.utility import fp4_utils


def torch_silu_and_mul(input: torch.Tensor, limit: float = 0.0) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    if limit > 0:
        x = torch.clamp(x, max=limit)
        y = torch.clamp(y, min=-limit, max=limit)
    out = F.silu(x) * y
    return out


@benchmark()
def test_scaled_silu_and_mul(m, n, dtype, output_dtype=None):
    """
    Test scaled_silu_and_mul with flexible input/output types.
    If output_dtype is None, defaults to fp8 for quantization.
    """
    ret = {}
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    scale = torch.max(input).to(torch.float32)
    out_dtype = output_dtype if output_dtype is not None else dtypes.fp8
    out = torch.empty((m, n // 2), dtype=out_dtype, device="cuda")

    # Reference: compute, scale, convert to output dtype
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    ref = (F.silu(x) * y / scale).to(out_dtype)

    _, us_aiter = run_perftest(
        aiter.scaled_silu_and_mul,
        out,
        input,
        scale,
    )

    # Check if the results are close
    err = checkAllclose(ref.to(torch.float), out.to(torch.float))

    # Record input/output types for clarity
    dtype_map = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        dtypes.fp8: "fp8",
    }
    ret["input_dtype"] = dtype_map.get(dtype, str(dtype))
    ret["output_dtype"] = dtype_map.get(out_dtype, str(out_dtype))
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


@benchmark()
def test_silu_and_mul(m, n, dtype, output_dtype=None, limit=0.0):
    """
    Test silu_and_mul with flexible input/output types.
    If output_dtype is None, output matches input dtype.
    """
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    out_dtype = output_dtype if output_dtype is not None else dtype
    out = torch.empty((m, n // 2), dtype=out_dtype, device="cuda")

    # Reference: compute in input dtype, convert to output dtype if needed
    ref = torch_silu_and_mul(input, limit=limit)
    if output_dtype is not None:
        ref = ref.to(output_dtype)

    _, us_aiter = run_perftest(
        aiter.silu_and_mul,
        out,
        input,
        limit,
    )

    # Check if the results are close
    err = checkAllclose(ref, out)

    # Record input/output types for clarity
    dtype_map = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
    ret = {}
    ret["input_dtype"] = dtype_map.get(dtype, str(dtype))
    ret["output_dtype"] = dtype_map.get(out_dtype, str(out_dtype))
    ret["limit"] = limit
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


class GELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://huggingface.co/papers/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self, use_gelu_tanh_python: bool = False):
        super().__init__()
        if use_gelu_tanh_python:
            self.act = self._gelu_tanh_python
        else:
            self.act = functools.partial(nn.functional.gelu, approximate="tanh")

    def _gelu_tanh_python(self, input: Tensor) -> Tensor:
        return (
            input
            * 0.5
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


def torch_gelu_ref(x: torch.Tensor) -> torch.Tensor:
    out = GELUTanh()(x)
    return out


def gelu_fast_wrapper(input: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(input)
    aiter.gelu_fast(out, input)
    return out


@benchmark()
def test_gelu_fast(m, n, dtype, output_dtype=None):
    ret = {}
    input = torch.randn(m, 1, n, dtype=dtype, device="cuda")
    out_dtype = output_dtype if output_dtype is not None else dtype

    out, us_aiter = run_perftest(gelu_fast_wrapper, input)
    ref, us_torch = run_perftest(torch_gelu_ref, input)

    if output_dtype is not None:
        ref = ref.to(output_dtype)

    # Check if the results are close
    err = checkAllclose(ref, out)

    # Record input/output types for clarity
    dtype_map = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
    ret = {}
    ret["input_dtype"] = dtype_map.get(dtype, str(dtype))
    ret["output_dtype"] = dtype_map.get(out_dtype, str(out_dtype))
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["torch_us"] = us_torch
    ret["speedup_vs_torch"] = us_torch / us_aiter
    ret["perf_gain_vs_torch_pct"] = (us_torch - us_aiter) / us_torch * 100.0
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


def _dequant_fp8_group(q, s, group_size):
    m, n = q.shape
    return (
        q.float().view(m, n // group_size, group_size) * s.float().view(m, -1, 1)
    ).view(m, n)


def _dequant_fp4_group(q, s, group_size):
    from aiter.utility import fp4_utils

    m = q.shape[0]
    n = s.shape[1] * group_size
    packed_cols = n // 2
    q_u8 = q.view(torch.uint8) if q.dtype == dtypes.fp4x2 else q
    if q_u8.shape[1] > packed_cols:
        q_u8 = q_u8[:, :packed_cols]
    q_f32 = fp4_utils.mxfp4_to_f32(q_u8).view(m, n // group_size, group_size)
    s_f32 = fp4_utils.e8m0_to_f32(s).float().view(m, -1, 1)
    return (q_f32 * s_f32).view(m, n)


def _ref_group_scales_fp8(x: torch.Tensor, group_size: int, out_dtype) -> torch.Tensor:
    m, n = x.shape
    xg = x.view(m, n // group_size, group_size).float()
    dmax = torch.finfo(out_dtype).max
    x_max = torch.amax(torch.abs(xg), dim=-1)
    x_max = torch.maximum(x_max, torch.full_like(x_max, 1e-10))
    return x_max / dmax


def _ref_group_scales_fp4(x: torch.Tensor, group_size: int) -> torch.Tensor:
    m, n = x.shape
    xg = x.view(m, n // group_size, group_size).float()
    x_max = torch.amax(torch.abs(xg), dim=-1)
    x_max = torch.maximum(x_max, torch.full_like(x_max, 1e-10))
    # NV ROUND_UP / DSv4 / FlashInfer default: scale = ceil_pow2(amax / 6)
    # (matches aiter::fp4_f32_to_e8m0_scale and silu_and_mul_quant HIP kernel).
    scale_e8m0 = fp4_utils.f32_to_e8m0_ceil(x_max / 6.0)
    return scale_e8m0.view(torch.uint8)


@benchmark()
def test_silu_and_mul_quant(m, n, dtype, group_size, output_dtype=None, limit=0.0):
    """
    Test silu_and_mul_quant with per-group quantization to fp8 or fp4.
    Benchmarks HIP kernel and validates against PyTorch reference.
    """
    ret = {}
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    d = n // 2
    out_dtype = output_dtype if output_dtype is not None else dtypes.fp8
    num_groups = d // group_size

    is_fp4 = out_dtype == dtypes.fp4x2
    if is_fp4:
        out = torch.empty((m, d // 2), dtype=out_dtype, device="cuda")
        scale = torch.empty((m, num_groups), dtype=torch.uint8, device="cuda")
    else:
        out = torch.empty((m, d), dtype=out_dtype, device="cuda")
        scale = torch.empty((m, num_groups), dtype=torch.float32, device="cuda")

    _, us_aiter = run_perftest(
        aiter.silu_and_mul_quant,
        out,
        input,
        scale,
        group_size,
        limit,
    )

    # Accuracy validation
    ref = torch_silu_and_mul(input, limit=limit).float()

    if is_fp4:
        q_atol, q_rtol = 0.5, 0.5
        hip_deq = _dequant_fp4_group(out, scale, group_size)
        ref_scale = _ref_group_scales_fp4(ref, group_size)
        scale_diff = (scale.to(torch.int16) - ref_scale.to(torch.int16)).abs()
        scale_max_abs_diff = scale_diff.max().item()
        scale_mismatch_ratio = (scale != ref_scale).float().mean().item()
        err_scale = checkAllclose(
            scale.float(),
            ref_scale.float(),
            rtol=0.0,
            atol=0.0,
            msg=f"HIP scale vs ref (M={m}, N={n}, gs={group_size}): ",
        )
    else:
        q_atol, q_rtol = 0.05, 0.05
        hip_deq = _dequant_fp8_group(out, scale, group_size)
        ref_scale = _ref_group_scales_fp8(ref, group_size, out_dtype)
        scale_diff = (scale.float() - ref_scale.float()).abs()
        scale_max_abs_diff = scale_diff.max().item()
        scale_mismatch_ratio = (
            (~torch.isclose(scale.float(), ref_scale.float(), rtol=1e-3, atol=1e-3))
            .float()
            .mean()
            .item()
        )
        err_scale = checkAllclose(
            scale.float(),
            ref_scale.float(),
            rtol=1e-3,
            atol=1e-3,
            msg=f"HIP scale vs ref (M={m}, N={n}, gs={group_size}): ",
        )

    err_hip = checkAllclose(
        ref,
        hip_deq,
        rtol=q_rtol,
        atol=q_atol,
        msg=f"HIP vs ref (M={m}, N={n}, gs={group_size}): ",
    )

    dtype_map = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        dtypes.fp8: "fp8",
        dtypes.fp4x2: "fp4",
    }
    ret["input_dtype"] = dtype_map.get(dtype, str(dtype))
    ret["output_dtype"] = dtype_map.get(out_dtype, str(out_dtype))
    ret["limit"] = limit
    ret["group_size"] = group_size
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes + scale.nbytes) / us_aiter / 1e6
    ret["err_hip"] = err_hip
    ret["err_scale"] = err_scale
    ret["scale_max_abs_diff"] = scale_max_abs_diff
    ret["scale_mismatch_ratio"] = scale_mismatch_ratio
    return ret


@benchmark()
def test_scaled_silu_and_mul_mixed_dtype(m, n, input_dtype, output_dtype):
    """Test fp32 input with fp16/bf16 output for scaled activation"""
    input = torch.randn(m, n, dtype=input_dtype, device="cuda")
    scale = torch.max(input).to(torch.float32)
    out = torch.empty((m, n // 2), dtype=output_dtype, device="cuda")

    # Reference: compute in fp32, scale, convert to output dtype
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    ref = (F.silu(x) * y / scale).to(output_dtype)

    _, us_aiter = run_perftest(
        aiter.scaled_silu_and_mul,
        out,
        input,
        scale,
    )

    err = checkAllclose(ref.to(torch.float), out.to(torch.float))
    dtype_map = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        dtypes.fp8: "fp8",
    }
    ret = {}
    ret["input_dtype"] = dtype_map.get(input_dtype, str(input_dtype))
    ret["output_dtype"] = dtype_map.get(output_dtype, str(output_dtype))
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["fp16"], dtypes.d_dtypes["bf16"]],
    nargs="*",
    metavar="{fp16, bf16}",
    default="fp16, bf16",
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    choices=[1, 32, 64, 128, 256, 512, 1024, 4096, 8192, 163840],
    default=[1, 32, 64, 128, 256, 512, 1024, 4096, 8192, 163840],
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-n",
    type=int,
    nargs="*",
    choices=[1024, 4096, 6400, 8192],
    default=[1024, 4096, 6400, 8192],
    help="""N of mnk.
    e.g.: -n 1024""",
)

args = parser.parse_args()

df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            ret = test_scaled_silu_and_mul(m, n, dtype)
            df.append(ret)
df = pd.DataFrame(df)
df = df[
    ["M", "N", "input_dtype", "output_dtype", "us", "TB/s", "RD TB/s", "WR TB/s", "err"]
]
df_md = df.to_markdown(index=False)
aiter.logger.info("scaled_silu_and_mul summary (markdown):\n%s", df_md)

df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            ret = test_silu_and_mul(m, n, dtype)
            df.append(ret)
# Add fp32 input with fp16/bf16 output (bandwidth optimization)
for output_dtype in [torch.float16, torch.bfloat16]:
    for m in args.m:
        for n in args.n:
            ret = test_silu_and_mul(m, n, torch.float32, output_dtype=output_dtype)
            df.append(ret)
df = pd.DataFrame(df)
df = df[
    ["M", "N", "input_dtype", "output_dtype", "us", "TB/s", "RD TB/s", "WR TB/s", "err"]
]

df_md = df.to_markdown(index=False)
aiter.logger.info("silu_and_mul summary (markdown):\n%s", df_md)

df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            ret = test_silu_and_mul(m, n, dtype, limit=10.0)
            df.append(ret)
df = pd.DataFrame(df)
df = df[
    [
        "M",
        "N",
        "input_dtype",
        "output_dtype",
        "limit",
        "us",
        "TB/s",
        "RD TB/s",
        "WR TB/s",
        "err",
    ]
]
df_md = df.to_markdown(index=False)
aiter.logger.info("silu_and_mul with limit=10.0 summary (markdown):\n%s", df_md)

quant_cols = [
    "M",
    "N",
    "input_dtype",
    "output_dtype",
    "group_size",
    "us",
    "TB/s",
    "RD TB/s",
    "WR TB/s",
    "err_hip",
    "err_scale",
    "scale_max_abs_diff",
    "scale_mismatch_ratio",
]

# silu_and_mul_quant with fp8 (group_size=64, 128)
df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            for gs in [64, 128]:
                d = n // 2
                if d >= gs and d % gs == 0:
                    ret = test_silu_and_mul_quant(m, n, dtype, group_size=gs)
                    df.append(ret)
if df:
    df = pd.DataFrame(df)
    df = df[quant_cols]
    df_md = df.to_markdown(index=False)
    aiter.logger.info("silu_and_mul_quant (fp8) summary (markdown):\n%s", df_md)

# silu_and_mul_quant with fp4 (group_size=32)
df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            d = n // 2
            gs = 32
            if d >= gs and d % gs == 0:
                ret = test_silu_and_mul_quant(
                    m, n, dtype, group_size=gs, output_dtype=dtypes.fp4x2
                )
                df.append(ret)
if df:
    df = pd.DataFrame(df)
    df = df[quant_cols]
    df_md = df.to_markdown(index=False)
    aiter.logger.info("silu_and_mul_quant (fp4) summary (markdown):\n%s", df_md)

# silu_and_mul_quant with fp8 + limit=10
df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            d = n // 2
            gs = 128
            if d >= gs and d % gs == 0:
                ret = test_silu_and_mul_quant(m, n, dtype, group_size=gs, limit=10.0)
                df.append(ret)
if df:
    df = pd.DataFrame(df)
    df = df[quant_cols + ["limit"]]
    df_md = df.to_markdown(index=False)
    aiter.logger.info(
        "silu_and_mul_quant (fp8, limit=10) summary (markdown):\n%s", df_md
    )

df = []
for dtype in args.dtype:
    for m in args.m:
        for n in args.n:
            ret = test_gelu_fast(m, n, dtype)
            df.append(ret)
df = pd.DataFrame(df)
df = df[
    [
        "M",
        "N",
        "input_dtype",
        "output_dtype",
        "us",
        "torch_us",
        "speedup_vs_torch",
        "perf_gain_vs_torch_pct",
        "TB/s",
        "RD TB/s",
        "WR TB/s",
        "err",
    ]
]
df_md = df.to_markdown(index=False)
aiter.logger.info("gelu_fast summary (markdown):\n%s", df_md)
