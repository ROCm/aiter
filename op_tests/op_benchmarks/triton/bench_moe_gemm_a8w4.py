# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py

from itertools import chain
from pathlib import Path
import triton.profiler as proton
import torch
import argparse
from aiter.ops.triton.moe_routing.routing import routing
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.moe_op_gemm_a8w4 import (
    moe_gemm_a8w4,
    swizzle_scales,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
import tempfile
from aiter.ops.triton.quant_moe import downcast_to_static_fp8, downcast_to_mxfp
import inspect


def parse_profile(profile_path, useful_op_regex, reps):
    """
    construct a PerfRecord from a (proton) profile path and a regex for useful operations
    """
    from triton.profiler import viewer

    gf, _, _, _ = viewer.read(profile_path)
    # aggregate "useful" flops + bytes
    useful = gf.filter(
        f"MATCH ('*', c) WHERE c.'name' =~ '{useful_op_regex}' AND c IS LEAF"
    ).dataframe
    bytes = int(useful["bytes"].sum())
    flops = int(
        sum(useful[[c for c in ["flops8", "flops16"] if c in useful.columns]].sum())
    )
    # take all ops (incl. "not useful" ones) when computing total time
    allops = gf.filter("MATCH ('*', c) WHERE c IS LEAF").dataframe
    total_time_ns = allops["time (ns)"].sum()
    kernel_time_ns = useful["time (ns)"].sum()
    return {
        "total_time_ns": total_time_ns,
        "kernel_time_ns": kernel_time_ns,
        "flops": flops,
        "bytes": bytes,
        "reps": reps,
    }


def compute_roofline(
    *args, bench_fn, intensity_proxy_name, intensity_proxy_values, out_path, **kwargs
):
    # validate input args
    if not isinstance(intensity_proxy_name, str):
        raise TypeError(
            "intensity_proxy must be a string naming a parameter in target_fn"
        )
    # determine position of intensity_proxy in target_fn signature
    sig = inspect.signature(bench_fn)
    params = list(sig.parameters.values())
    if intensity_proxy_name not in sig.parameters:
        raise ValueError(
            f"Parameter '{intensity_proxy_name}' not found in {bench_fn.__name__} signature"
        )
    pos_index = [p.name for p in params].index(intensity_proxy_name)

    # wrapper to inject intensity proxy into target_fn and call it
    def inject_proxy_and_call(val, args, kwargs):
        args_list = list(args)
        args_list.insert(pos_index, val)
        return bench_fn(*args_list, **kwargs)

    # collect performance data
    perfs = []
    print("=========================================")
    print(f"{out_path   }...")
    print("=========================================")
    for val in intensity_proxy_values:
        perf = inject_proxy_and_call(val, args, kwargs)
        perfs.append(perf)
        tflops = perfs[-1]["flops"] / perfs[-1]["kernel_time_ns"] * 1e-3
        tbps = perfs[-1]["bytes"] / perfs[-1]["kernel_time_ns"] * 1e-3
        total_latency = perfs[-1]["total_time_ns"] / 1e3 / perfs[-1]["reps"]
        kernel_latency = perfs[-1]["kernel_time_ns"] / 1e3 / perfs[-1]["reps"]
        print(
            f"{intensity_proxy_name}: {val:5d} | Total latency (us): {total_latency:.2f} | Kernel latency (us): {kernel_latency:.2f} | TFLOPS: {tflops:#.4g} | TBPS: {tbps:.2f}"
        )


def check_and_swizzle_scales(scale, N, K):
    if N % 32 == 0 and K % (32 * 8) == 0:
        scale = swizzle_scales(scale)
        return scale, "CDNA4_SCALE"
    else:
        return scale, None


def quantize(x, dtype):
    if dtype == "bf16":
        x = x.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return x, None
    elif dtype == "fp8":
        scale = x.abs().max().item() / 448.0
        fp8e4_dtype = (
            torch.float8_e4m3fn if get_arch() != "gfx942" else torch.float8_e4m3fnuz
        )
        x = x.to(fp8e4_dtype)
        return x, scale
    elif dtype == "mx8":
        fp8e4_dtype = (
            torch.float8_e4m3fn if get_arch() != "gfx942" else torch.float8_e4m3fnuz
        )
        x, scale = downcast_to_mxfp(x, fp8e4_dtype, axis=1)
        return x, scale
    else:
        assert dtype == "mx4", f"{dtype=}"
        x, scale = downcast_to_mxfp(x.to(torch.bfloat16), torch.uint8, axis=1)
        return x, scale


def bench_mlp(
    batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, op_regex
):
    rank = 0
    dev = f"cuda:{rank}"

    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"

    # -- init data --
    # weights
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    w1 = torch.randn((n_expts_tot, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot, dim2 // TP // 2, dim1), device=dev)
    # biases
    bg = torch.randn((n_expts_tot,), device=dev)
    b1 = torch.randn((n_expts_tot, dim2 // TP), device=dev)
    b2 = torch.randn((n_expts_tot, dim1), device=dev)

    # -- numerics --
    wg, _ = quantize(wg, "bf16")
    w1, w1_scale = quantize(w1, w_dtype)
    w2, w2_scale = quantize(w2, w_dtype)
    w1_scale, swizzle_mx_scale1 = check_and_swizzle_scales(w1_scale, dim2 // TP, dim1)
    w2_scale, swizzle_mx_scale2 = check_and_swizzle_scales(
        w2_scale, dim1, dim2 // TP // 2
    )

    # -- benchmark --
    x_dtype_str = x_dtype
    x_dtype = torch.float8_e4m3fn
    # special treatment of fp8_e4m3 on AMD CDNA3 because it uses fp8_e4m3fnuz
    if x_dtype == torch.float8_e4m3fn and get_arch() == "gfx942":
        x_dtype = torch.float8_e4m3fnuz

    reps = 100
    x = torch.randn((batch, dim1), dtype=torch.bfloat16, device=dev)
    xg = x
    if x_dtype_str == "fp8":
        static_scale = torch.tensor(1e-4, device=dev)
    # run layer
    fpath = Path(tempfile.mktemp())
    proton.start(str(fpath), hook="triton")
    for i in range(reps):
        logits = gemm_a16w16(xg, wg.T, bg)
        rdata, gather_indx, scatter_indx = routing(logits, n_expts_act)
        if x_dtype_str == "fp8":
            x = downcast_to_static_fp8(x, static_scale)
            x = moe_gemm_a8w4(
                x,
                w1,
                None,
                w1_scale,
                static_scale,
                static_scale,
                b1,
                rdata,
                gather_indx=gather_indx,
                swizzle_mx_scale=swizzle_mx_scale1,
                out_dtype=x_dtype,
                apply_swiglu=True,
            )
            x = moe_gemm_a8w4(
                x,
                w2,
                None,
                w2_scale,
                static_scale,
                None,
                b2,
                rdata,
                scatter_indx=scatter_indx,
                swizzle_mx_scale=swizzle_mx_scale2,
            )
        else:
            assert x_dtype_str == "mx8"
            x, _, x_scale = quantize(x, x_dtype_str)
            x = moe_gemm_a8w4(
                x,
                w1,
                x_scale,
                w1_scale,
                None,
                None,
                b1,
                rdata,
                gather_indx=gather_indx,
                swizzle_mx_scale="CDNA4_SCALE",
                apply_swiglu=True,
            )
            x, _, x_scale = quantize(x, x_dtype_str)
            x = moe_gemm_a8w4(
                x,
                w2,
                x_scale,
                w2_scale,
                None,
                None,
                b2,
                rdata,
                scatter_indx=scatter_indx,
                swizzle_mx_scale="CDNA4_SCALE",
            )
    proton.finalize()
    return parse_profile(
        fpath.with_suffix(".hatchet"), useful_op_regex=op_regex, reps=reps
    )


def roofline_mlp(
    batch_sizes,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    TP,
    op_regex,
    name="",
):
    out_path = Path(f"logs/{name}/{x_dtype}x-{w_dtype}w-TP{TP}/")
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = compute_roofline(
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        x_dtype,
        w_dtype,
        TP,
        op_regex,  # fixed args
        bench_fn=bench_mlp,  # function to benchmark
        intensity_proxy_name="batch",  # intensity proxy name
        intensity_proxy_values=batch_sizes,  # intensity proxy values to sweep
        out_path=out_path.with_suffix(".csv"),
    )  # output path


def parse_args():
    parser = argparse.ArgumentParser(prog="Benchmark MoE")
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        metavar=("DIM"),
        help="Input feature dimensions of MoE layers. Must be two integers.",
    )
    parser.add_argument(
        "--experts",
        type=int,
        nargs="+",
        metavar=("DIM"),
        help="Number of total and active experts in [total experts, active experts] order.",
    )
    parser.add_argument(
        "--op-regex",
        type=str,
        default=".*moe_gemm.*",
        help="Regex to find perf for specific operation by its kernel name.",
    )
    parser.add_argument(
        "--act-dtype",
        type=str,
        default="fp8",
        help="Activation dtype, fp8 or mx8.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    dim1, dim2 = args.shape
    total_experts, active_experts = args.experts
    batch_ranges_moe = [
        (1, 2, 1),
        (2, 5, 2),
        (8, 18, 8),
        (32, 65, 32),
        (128, 257, 128),
        (1024, 1200, 200),
        (4096, 8200, 4096),
    ]
    batch_sizes_moe = list(chain(*[range(*r) for r in batch_ranges_moe]))
    quantized_dtypes = [args.act_dtype, "mx4"]

    roofline_mlp(
        batch_sizes_moe,
        dim1,
        dim2,
        total_experts,
        active_experts,
        quantized_dtypes[0],
        quantized_dtypes[1],
        TP=1,
        op_regex=args.op_regex,
        name="gpt-oss-x2",
    )
