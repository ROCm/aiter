# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py

from itertools import chain
from pathlib import Path
import triton.profiler as proton
import torch
import argparse
from aiter.ops.triton.moe.moe_op_gemm_int8_smoothquant import (
    moe_gemm_int8_smoothquant,
    preshuffle_weights,
)
from aiter.ops.triton.moe.quant_moe import (
    smoothquant_quantize,
    quantize_weights_int8,
)
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
import tempfile
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


def bench_mlp_single_weight_init(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    preshuffled,
    x_dtype,
    w_dtype,
    TP,
    op_regex,
):
    rank = 0
    dev = f"cuda:{rank}"

    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"

    # -- init data --
    # weights
    wg = torch.randn((dim1, n_expts_tot), device=dev, dtype=torch.bfloat16)
    w1 = torch.randn((n_expts_tot, dim1, dim2 // TP), device=dev, dtype=torch.bfloat16)
    w2 = torch.randn(
        (n_expts_tot, dim2 // TP // 2, dim1), device=dev, dtype=torch.bfloat16
    )
    # biases
    bg = torch.randn((n_expts_tot,), device=dev, dtype=torch.bfloat16)

    # smoothQuant scales
    fc1_smooth_scale = torch.randn((dim1,), device=dev, dtype=torch.float32).abs() + 0.1
    fc2_smooth_scale = (
        torch.randn((dim2 // TP // 2,), device=dev, dtype=torch.float32).abs() + 0.1
    )

    # -- numerics --
    w1_int8, w1_scale = quantize_weights_int8(w1)
    w2_int8, w2_scale = quantize_weights_int8(w2)
    w1_int8 = w1_int8.transpose(1, 2).contiguous().transpose(1, 2)
    w2_int8 = w2_int8.transpose(1, 2).contiguous().transpose(1, 2)
    if preshuffled:
        w1_int8 = preshuffle_weights(w1_int8)
        w2_int8 = preshuffle_weights(w2_int8)
    # -- benchmark --
    reps = 100
    x = torch.randn((batch, dim1), dtype=torch.bfloat16, device=dev)
    xg = x

    fpath = Path(tempfile.mktemp())
    proton.start(str(fpath), hook="triton")

    for i in range(reps):
        logits = gemm_a16w16(xg, wg.T, bg)
        rdata, gather_indx, scatter_indx = routing(logits, n_expts_act)

        x, x_scale = smoothquant_quantize(x, fc1_smooth_scale)
        x = moe_gemm_int8_smoothquant(
            x,
            w1_int8,
            x_scale,
            w1_scale,
            None,
            rdata,
            gather_indx=gather_indx,
            scatter_indx=None,
            preshuffled=preshuffled,
            out_dtype=torch.float32,
            apply_activation=True,
            limit=None,
            add_residual=False,
        )
        x, x_scale = smoothquant_quantize(x, fc2_smooth_scale)
        x = moe_gemm_int8_smoothquant(
            x,
            w2_int8,
            x_scale,
            w2_scale,
            None,
            rdata,
            gather_indx=None,
            scatter_indx=scatter_indx,
            preshuffled=preshuffled,
            out_dtype=torch.bfloat16,
            apply_activation=False,
            add_residual=False,
        )
    proton.finalize()
    return parse_profile(
        fpath.with_suffix(".hatchet"), useful_op_regex=op_regex, reps=reps
    )


def bench_mlp(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    preshuffled,
    x_dtype,
    w_dtype,
    TP,
    op_regex,
    num_weight_inits=1,
):
    all_results = []
    for i in range(num_weight_inits):
        result = bench_mlp_single_weight_init(
            batch,
            dim1,
            dim2,
            n_expts_tot,
            n_expts_act,
            preshuffled,
            x_dtype,
            w_dtype,
            TP,
            op_regex,
        )
        all_results.append(result)

    num_runs = len(all_results)
    aggregated = {
        "total_time_ns": sum(r["total_time_ns"] for r in all_results) / num_runs,
        "kernel_time_ns": sum(r["kernel_time_ns"] for r in all_results) / num_runs,
        "flops": sum(r["flops"] for r in all_results) / num_runs,
        "bytes": sum(r["bytes"] for r in all_results) / num_runs,
        "reps": all_results[0]["reps"],
    }

    return aggregated


def roofline_mlp(
    batch_sizes,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    preshuffled,
    x_dtype,
    w_dtype,
    TP,
    op_regex,
    num_weight_inits=1,
    name="",
):
    out_path = Path(f"logs/{name}/{x_dtype}x-{w_dtype}w-TP{TP}/")
    out_path.mkdir(parents=True, exist_ok=True)
    compute_roofline(
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        preshuffled,
        x_dtype,
        w_dtype,
        TP,
        op_regex,  # fixed args
        num_weight_inits,
        bench_fn=bench_mlp,  # function to benchmark
        intensity_proxy_name="batch",  # intensity proxy name
        intensity_proxy_values=batch_sizes,  # intensity proxy values to sweep
        out_path=out_path.with_suffix(".csv"),
    )  # output path


def parse_args():
    parser = argparse.ArgumentParser(prog="Benchmark MoE Int8 SmoothQuant GEMM")
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
        "--preshuffled",
        action="store_true",
        help="Preshuffle weights.",
    )
    parser.add_argument(
        "--op-regex",
        type=str,
        default=".*moe_gemm.*",
        help="Regex to find perf for specific operation by its kernel name.",
    )
    parser.add_argument(
        "--num-weight-inits",
        type=int,
        default=1,
        help="Number of different weight initializations to run for more stable results (default: 1). "
        "Each initialization runs 100 iterations. Use higher values (e.g., 10) for more stable benchmarks.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    dim1, dim2 = args.shape
    total_experts, active_experts = args.experts
    preshuffled = args.preshuffled
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
    quantized_dtypes = [torch.int8, torch.int8]

    roofline_mlp(
        batch_sizes_moe,
        dim1,
        dim2,
        total_experts,
        active_experts,
        preshuffled,
        quantized_dtypes[0],
        quantized_dtypes[1],
        TP=1,
        op_regex=args.op_regex,
        num_weight_inits=args.num_weight_inits,
        name="int8-smoothquant",
    )
