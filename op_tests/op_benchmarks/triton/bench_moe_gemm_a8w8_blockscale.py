# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py

from itertools import chain
from pathlib import Path
import triton.profiler as proton
import torch
import argparse
import csv
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.moe.moe_op_gemm_a8w8_blockscale import (
    moe_gemm_a8w8_blockscale,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (
    _get_config,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
import tempfile
import inspect

# Default group_m, group_n, group_k
group_shape = (128, 128, 128)


def parse_profile(profile_path, useful_op_regex, reps):
    """
    construct a PerfRecord from a (proton) profile path and a regex for useful operations
    """
    from triton.profiler import viewer

    gf, _, _, _ = viewer.read(profile_path)
    useful = gf.filter(
        f"MATCH ('*', c) WHERE c.'name' =~ '{useful_op_regex}' AND c IS LEAF"
    ).dataframe
    bytes_ = int(useful["bytes"].sum())
    flops = int(
        sum(useful[[c for c in ["flops8", "flops16"] if c in useful.columns]].sum())
    )
    allops = gf.filter("MATCH ('*', c) WHERE c IS LEAF").dataframe
    total_time_ns = allops["time (ns)"].sum()
    kernel_time_ns = useful["time (ns)"].sum()
    return {
        "total_time_ns": total_time_ns,
        "kernel_time_ns": kernel_time_ns,
        "flops": flops,
        "bytes": bytes_,
        "reps": reps,
    }


def compute_roofline(
    *args, bench_fn, intensity_proxy_name, intensity_proxy_values, out_path, **kwargs
):
    """
    Sweeps intensity_proxy_values by injecting them into bench_fn, prints summary, and writes a CSV to out_path.
    """
    if not isinstance(intensity_proxy_name, str):
        raise TypeError(
            "intensity_proxy must be a string naming a parameter in target_fn"
        )

    sig = inspect.signature(bench_fn)
    params = list(sig.parameters.values())
    if intensity_proxy_name not in sig.parameters:
        raise ValueError(
            f"Parameter '{intensity_proxy_name}' not found in {bench_fn.__name__} signature"
        )
    pos_index = [p.name for p in params].index(intensity_proxy_name)

    def inject_proxy_and_call(val, args_, kwargs_):
        args_list = list(args_)
        args_list.insert(pos_index, val)
        return bench_fn(*args_list, **kwargs_)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []  # list[(proxy_val, perf_dict)]
    print("=========================================")
    print(f"{out_path}...")
    print("=========================================")
    for val in intensity_proxy_values:
        perf = inject_proxy_and_call(val, args, kwargs)
        results.append((val, perf))

        tflops = perf["flops"] / perf["kernel_time_ns"] * 1e-3
        tbps = perf["bytes"] / perf["kernel_time_ns"] * 1e-3
        total_latency_us = perf["total_time_ns"] / 1e3 / perf["reps"]
        kernel_latency_us = perf["kernel_time_ns"] / 1e3 / perf["reps"]
        print(
            f"{intensity_proxy_name}: {val:5d} | "
            f"Total latency (us): {total_latency_us:.2f} | "
            f"Kernel latency (us): {kernel_latency_us:.2f} | "
            f"TFLOPS: {tflops:#.4g} | "
            f"TBPS: {tbps:.2f}"
        )

    # Write CSV (missing in original code)
    fieldnames = [
        intensity_proxy_name,  # e.g. "batch"
        "total_latency_us",
        "kernel_latency_us",
        "tflops",
        "tbps",
        "total_time_ns",
        "kernel_time_ns",
        "flops",
        "bytes",
        "reps",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for val, perf in results:
            w.writerow(
                {
                    intensity_proxy_name: val,
                    "total_latency_us": perf["total_time_ns"] / 1e3 / perf["reps"],
                    "kernel_latency_us": perf["kernel_time_ns"] / 1e3 / perf["reps"],
                    "tflops": perf["flops"] / perf["kernel_time_ns"] * 1e-3,
                    "tbps": perf["bytes"] / perf["kernel_time_ns"] * 1e-3,
                    "total_time_ns": perf["total_time_ns"],
                    "kernel_time_ns": perf["kernel_time_ns"],
                    "flops": perf["flops"],
                    "bytes": perf["bytes"],
                    "reps": perf["reps"],
                }
            )


def bench_mlp_single_weight_init(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    per_row_act_quant,
    TP,
    op_regex,
):
    rank = 0
    dev = f"cuda:{rank}"

    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"

    group_shape_m, group_shape_n, group_shape_k = group_shape

    wg = torch.randn((dim1, n_expts_tot), device=dev)
    w1 = (
        torch.randn((n_expts_tot, dim1, dim2 // TP), dtype=torch.bfloat16, device=dev)
        / 10
    ).to(torch.float8_e4m3fn)
    w2 = (
        torch.randn(
            (n_expts_tot, dim2 // TP // 2, dim1), dtype=torch.bfloat16, device=dev
        )
        / 10
    ).to(torch.float8_e4m3fn)

    bg = torch.randn((n_expts_tot,), device=dev)
    b1 = torch.randn((n_expts_tot, dim2 // TP), dtype=torch.float32, device=dev)
    b2 = torch.randn((n_expts_tot, dim1), dtype=torch.float32, device=dev)

    x_dtype_str = x_dtype
    w_dtype_str = w_dtype
    x_dtype = torch.float8_e4m3fn
    w_dtype = torch.float8_e4m3fn
    if x_dtype == torch.float8_e4m3fn and get_arch() == "gfx942":
        x_dtype = torch.float8_e4m3fnuz
    if w_dtype == torch.float8_e4m3fn and get_arch() == "gfx942":
        w_dtype = torch.float8_e4m3fnuz

    reps = 100
    x = (torch.randn((batch, dim1), dtype=torch.bfloat16, device=dev) / 10).to(
        torch.float8_e4m3fn
    )
    xg = x.to(torch.float32)

    def num_blocks(length, block):
        return (length + block - 1) // block

    # scales
    if x_dtype_str == "fp8":
        x_static_scale = torch.tensor(1e-4, device=dev)
    else:
        k_blocks_x = num_blocks(dim1, group_shape_k)
        if per_row_act_quant == "True":
            x_scale = torch.rand((batch, k_blocks_x), dtype=torch.float32, device=dev)
        else:
            m_blocks = num_blocks(batch, group_shape_m)
            x_scale = torch.rand((m_blocks, k_blocks_x), dtype=torch.float32, device=dev)

    if w_dtype_str == "fp8":
        w_static_scale = torch.tensor(1e-4, device=dev)
    else:
        k_blocks_w1 = num_blocks(dim1, group_shape_k)
        n_blocks_w1 = num_blocks(dim2 // TP, group_shape_n)
        k_blocks_w2 = num_blocks(dim2 // TP // 2, group_shape_k)
        n_blocks_w2 = num_blocks(dim1, group_shape_n)

        w1_scale = torch.rand(
            (n_expts_tot, k_blocks_w1, n_blocks_w1), dtype=torch.float32, device=dev
        )
        w2_scale = torch.rand(
            (n_expts_tot, k_blocks_w2, n_blocks_w2), dtype=torch.float32, device=dev
        )

    fpath = Path(tempfile.mktemp())
    M, K = xg.shape
    K, N = wg.shape
    config, _ = _get_config(M, N, K)
    config["BLOCK_SIZE_M"] = 128 if config["BLOCK_SIZE_M"] > 128 else config["BLOCK_SIZE_M"]
    config["BLOCK_SIZE_N"] = 128 if config["BLOCK_SIZE_N"] > 128 else config["BLOCK_SIZE_N"]
    config["BLOCK_SIZE_K"] = 128 if config["BLOCK_SIZE_K"] > 128 else config["BLOCK_SIZE_K"]

    proton.start(str(fpath), hook="triton")
    for _ in range(reps):
        logits = gemm_a16w16(xg, wg.T, bg, config=config)
        rdata, gather_indx, scatter_indx = routing(logits, n_expts_act)

        if x_dtype_str == "fp8" and w_dtype_str == "fp8":
            x = moe_gemm_a8w8_blockscale(
                x, w1, None, None, x_static_scale, w_static_scale, None, b1, rdata,
                gather_indx=gather_indx, out_dtype=x_dtype, apply_swiglu=True,
            )
            x = moe_gemm_a8w8_blockscale(
                x, w2, None, None, x_static_scale, w_static_scale, None, b2, rdata,
                out_dtype=x_dtype, scatter_indx=scatter_indx,
            )
        elif x_dtype_str == "fp8" and w_dtype_str == "bs8":
            x = moe_gemm_a8w8_blockscale(
                x, w1, None, w1_scale, x_static_scale, None, None, b1, rdata,
                gather_indx=gather_indx, out_dtype=x_dtype, apply_swiglu=True,
            )
            x = moe_gemm_a8w8_blockscale(
                x, w2, None, w2_scale, x_static_scale, None, None, b2, rdata,
                out_dtype=x_dtype, scatter_indx=scatter_indx,
            )
        elif x_dtype_str == "bs8" and w_dtype_str == "fp8":
            x = moe_gemm_a8w8_blockscale(
                x, w1, x_scale, None, None, w_static_scale, None, b1, rdata,
                gather_indx=gather_indx, out_dtype=x_dtype, apply_swiglu=True,
            )
            x = moe_gemm_a8w8_blockscale(
                x, w2, x_scale, None, None, w_static_scale, None, b2, rdata,
                out_dtype=x_dtype, scatter_indx=scatter_indx,
            )
        else:
            x = moe_gemm_a8w8_blockscale(
                x, w1, x_scale, w1_scale, None, None, None, b1, rdata,
                out_dtype=x_dtype, gather_indx=gather_indx, apply_swiglu=True,
            )
            x = moe_gemm_a8w8_blockscale(
                x, w2, x_scale, w2_scale, None, None, None, b2, rdata,
                out_dtype=x_dtype, scatter_indx=scatter_indx,
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
    x_dtype,
    w_dtype,
    per_row_act_quant,
    TP,
    op_regex,
    num_weight_inits=1,
):
    all_results = []
    for _ in range(num_weight_inits):
        result = bench_mlp_single_weight_init(
            batch,
            dim1,
            dim2,
            n_expts_tot,
            n_expts_act,
            x_dtype,
            w_dtype,
            per_row_act_quant,
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
    x_dtype,
    w_dtype,
    per_row_act_quant,
    TP,
    op_regex,
    num_weight_inits=1,
    name="",
):
    # Avoid creating an empty directory named like the output CSV stem.
    out_dir = Path("logs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"{x_dtype}x-{w_dtype}w-TP{TP}.csv"

    compute_roofline(
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        x_dtype,
        w_dtype,
        per_row_act_quant,
        TP,
        op_regex,
        num_weight_inits,
        bench_fn=bench_mlp,
        intensity_proxy_name="batch",
        intensity_proxy_values=batch_sizes,
        out_path=out_csv,
    )


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="Benchmark MoE")

    parser.add_argument(
        "--M",
        type=int,
        nargs="+",
        default=None,
        help="MoE batch sizes M (one or more integers). "
        "If not set, a predermined list of values will be used.",
    )
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
        help="Activation dtype, fp8 or bs8.",
    )
    parser.add_argument(
        "--w-dtype",
        type=str,
        default="fp8",
        help="Weight dtype, fp8 or bs8.",
    )
    parser.add_argument(
        "--act-per-row-bs",
        type=str,
        default="False",
        help="Use per-row blockscale (True) or per-M-block (False) if act-dtype is bs8.",
    )
    parser.add_argument(
        "--num-weight-inits",
        type=int,
        default=1,
        help="Number of different weight initializations to run for more stable results (default: 1). "
        "Each initialization runs 100 iterations. Use higher values (e.g., 10) for more stable benchmarks.",
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)

    dim1, dim2 = parsed_args.shape
    total_experts, active_experts = parsed_args.experts
    if parsed_args.M is None:
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
    else:
        batch_sizes_moe = parsed_args.M

    quantized_dtypes = [parsed_args.act_dtype, parsed_args.w_dtype]
    per_row_act_quant = parsed_args.act_per_row_bs

    roofline_mlp(
        batch_sizes_moe,
        dim1,
        dim2,
        total_experts,
        active_experts,
        quantized_dtypes[0],
        quantized_dtypes[1],
        per_row_act_quant,
        TP=1,
        op_regex=parsed_args.op_regex,
        num_weight_inits=parsed_args.num_weight_inits,
        name="gpt-oss-x2",
    )


if __name__ == "__main__":
    main()