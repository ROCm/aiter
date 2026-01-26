import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import run_perftest, checkAllclose, benchmark
from aiter import dtypes
import pandas as pd
import argparse


def torch_silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    out = F.silu(x) * y
    return out


def triton_silu_mul_fp8_quantization_fuse(x, x_scale, rocm_fp8_dtype):
    quant_out = aiter.ops.triton.quant.fused_fp8_quant.fused_silu_mul_fp8_per_tensor_static_quant(
        x, x_scale, dtype_quant=rocm_fp8_dtype, silu_convert_to_inp_type=True
    )
    return quant_out


def aiter_scaled_silu_mul(m, n, x, x_scale, dtype_quant=aiter.dtypes.fp8):
    silu_out = torch.empty((m, n // 2), dtype=dtype_quant, device=x.device)
    aiter.scaled_silu_and_mul(silu_out, x, x_scale)
    return silu_out

def aiter_silu_and_mul(m, n, x, out_type):
    silu_out = torch.empty((m, n // 2), dtype=out_type, device=x.device)
    aiter.silu_and_mul(silu_out, x)
    return silu_out

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

    # Reference: compute, scale, convert to output dtype
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    ref = (F.silu(x) * y / scale).to(out_dtype)

    out, us_aiter = run_perftest(
        aiter_scaled_silu_mul, m, n, input, scale, out_dtype
    )
    fp8_x, us_triton = run_perftest(
        triton_silu_mul_fp8_quantization_fuse, input, scale, aiter.dtypes.fp8
    )

    err = checkAllclose(ref.to(torch.float), out.to(torch.float))
    err_triton = checkAllclose(ref.to(torch.float), fp8_x.to(torch.float))
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
    ret["hip us"] = us_aiter
    ret["triton us"] = us_triton
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    ret["err_triton"] = err_triton
    return ret


@benchmark()
def test_silu_and_mul(m, n, dtype, output_dtype=None):
    """
    Test silu_and_mul with flexible input/output types.
    If output_dtype is None, output matches input dtype.
    """
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    out_dtype = output_dtype if output_dtype is not None else dtype

    # Reference: compute in input dtype, convert to output dtype if needed
    ref = torch_silu_and_mul(input)
    if output_dtype is not None:
        ref = ref.to(output_dtype)

    out, us_aiter = run_perftest(
        aiter_silu_and_mul, m, n, input, out_dtype
    )
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
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


@benchmark()
def test_scaled_silu_and_mul_mixed_dtype(m, n, input_dtype, output_dtype):
    """Test fp32 input with fp16/bf16 output for scaled activation"""
    input = torch.randn(m, n, dtype=input_dtype, device="cuda")
    scale = torch.max(input).to(torch.float32)
    #out = torch.empty((m, n // 2), dtype=output_dtype, device="cuda")

    # Reference: compute in fp32, scale, convert to output dtype
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    ref = (F.silu(x) * y / scale).to(output_dtype)

    out, us_aiter = run_perftest(
        aiter_scaled_silu_mul, m, n, input, scale, output_dtype
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
    [
        "M",
        "N",
        "input_dtype",
        "output_dtype",
        "hip us",
        "triton us",
        "TB/s",
        "RD TB/s",
        "WR TB/s",
        "err",
        "err_triton",
    ]
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
