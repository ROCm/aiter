# Benchmark for Int8 SmoothQuant MoE GEMM
# Based on bench_moe_gemm_a8w8.py

from itertools import chain
from pathlib import Path
import triton.profiler as proton
import torch
import argparse
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.moe.moe_op_gemm_int8_smoothquant import (
    smoothquant_quantize,
    quantize_weights_int8,
    moe_gemm_int8_smoothquant,
    smoothquant_moe_mlp,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
import tempfile
import inspect


def parse_profile(profile_path, useful_op_regex, reps):
    """
    Construct performance metrics from a (proton) profile path.
    """
    from triton.profiler import viewer

    gf, _, _, _ = viewer.read(profile_path)
    
    # Aggregate "useful" flops + bytes
    useful = gf.filter(
        f"MATCH ('*', c) WHERE c.'name' =~ '{useful_op_regex}' AND c IS LEAF"
    ).dataframe
    bytes = int(useful["bytes"].sum()) if "bytes" in useful.columns else 0
    flops = int(
        sum(useful[[c for c in ["flops8", "flops16", "flops32"] if c in useful.columns]].sum())
    )
    
    # All ops for total time
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
    """Run benchmarks across different batch sizes and print results."""
    # Determine position of intensity_proxy in target_fn signature
    sig = inspect.signature(bench_fn)
    params = list(sig.parameters.values())
    if intensity_proxy_name not in sig.parameters:
        raise ValueError(
            f"Parameter '{intensity_proxy_name}' not found in {bench_fn.__name__} signature"
        )
    pos_index = [p.name for p in params].index(intensity_proxy_name)

    def inject_proxy_and_call(val, args, kwargs):
        args_list = list(args)
        args_list.insert(pos_index, val)
        return bench_fn(*args_list, **kwargs)

    # Collect performance data
    perfs = []
    print("=========================================")
    print(f"{out_path}...")
    print("=========================================")
    for val in intensity_proxy_values:
        perf = inject_proxy_and_call(val, args, kwargs)
        perfs.append(perf)
        tflops = perfs[-1]["flops"] / perfs[-1]["kernel_time_ns"] * 1e-3 if perfs[-1]["kernel_time_ns"] > 0 else 0
        tbps = perfs[-1]["bytes"] / perfs[-1]["kernel_time_ns"] * 1e-3 if perfs[-1]["kernel_time_ns"] > 0 else 0
        total_latency = perfs[-1]["total_time_ns"] / 1e3 / perfs[-1]["reps"]
        kernel_latency = perfs[-1]["kernel_time_ns"] / 1e3 / perfs[-1]["reps"]
        print(
            f"{intensity_proxy_name}: {val:5d} | Total latency (us): {total_latency:.2f} | "
            f"Kernel latency (us): {kernel_latency:.2f} | TFLOPS: {tflops:#.4g} | TBPS: {tbps:.2f}"
        )


def bench_moe_int8_smoothquant_single_init(
    batch, dim1, dim2, n_expts_tot, n_expts_act, TP, op_regex
):
    """Benchmark single weight initialization."""
    rank = 0
    dev = f"cuda:{rank}"

    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"

    # -- Initialize weights --
    # Gate weights (bf16, not quantized)
    wg = torch.randn((dim1, n_expts_tot), device=dev, dtype=torch.bfloat16)
    
    # FC1: dim1 -> dim2 // TP (double-width for SwiGLU gating)
    w1 = torch.randn((n_expts_tot, dim1, dim2 // TP), device=dev, dtype=torch.bfloat16)
    
    # FC2: dim2 // TP // 2 -> dim1 (SwiGLU halves the output dimension)
    w2 = torch.randn((n_expts_tot, dim2 // TP // 2, dim1), device=dev, dtype=torch.bfloat16)
    
    # Biases
    bg = torch.randn((n_expts_tot,), device=dev, dtype=torch.bfloat16)
    b1 = torch.randn((n_expts_tot, dim2 // TP), device=dev, dtype=torch.float32)
    b2 = torch.randn((n_expts_tot, dim1), device=dev, dtype=torch.float32)
    
    # Smooth scales
    fc1_smooth_scale = torch.randn((dim1,), device=dev, dtype=torch.float32).abs() + 0.1
    fc2_smooth_scale = torch.randn((dim2 // TP // 2,), device=dev, dtype=torch.float32).abs() + 0.1

    # -- Quantize weights --
    w1_int8, w1_scale = quantize_weights_int8(w1)
    w2_int8, w2_scale = quantize_weights_int8(w2)

    # -- Benchmark --
    reps = 100
    x = torch.randn((batch, dim1), dtype=torch.bfloat16, device=dev)
    xg = x

    fpath = Path(tempfile.mktemp())
    proton.start(str(fpath), hook="triton")
    
    for i in range(reps):
        # Gate computation (unchanged - a16w16)
        logits = gemm_a16w16(xg, wg.T, bg)
        rdata, gather_indx, scatter_indx = routing(logits, n_expts_act)
        
        # SmoothQuant FC1
        x_int8, x_scale = smoothquant_quantize(x, fc1_smooth_scale)
        
        # MoE FC1 with SwiGLU (halves output dimension)
        intermediate = moe_gemm_int8_smoothquant(
            x_int8,
            x_scale,
            w1_int8,
            w1_scale,
            b1,
            rdata,
            gather_indx=gather_indx,
            scatter_indx=None,
            out_dtype=torch.float32,
            apply_swiglu=True,
        )
        
        # SmoothQuant FC2
        y_int8, y_scale = smoothquant_quantize(intermediate, fc2_smooth_scale)
        
        # MoE FC2
        x = moe_gemm_int8_smoothquant(
            y_int8,
            y_scale,
            w2_int8,
            w2_scale,
            b2,
            rdata,
            gather_indx=None,
            scatter_indx=scatter_indx,
            out_dtype=torch.bfloat16,
        )
    
    proton.finalize()
    return parse_profile(
        fpath.with_suffix(".hatchet"), useful_op_regex=op_regex, reps=reps
    )


def bench_moe_int8_smoothquant(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    TP,
    op_regex,
    num_weight_inits=1,
):
    """Run benchmark with multiple weight initializations for stability."""
    all_results = []
    for i in range(num_weight_inits):
        result = bench_moe_int8_smoothquant_single_init(
            batch, dim1, dim2, n_expts_tot, n_expts_act, TP, op_regex
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


def roofline_moe(
    batch_sizes,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    TP,
    op_regex,
    num_weight_inits=1,
    name="",
):
    """Generate roofline data across batch sizes."""
    out_path = Path(f"logs/{name}/int8-smoothquant-TP{TP}/")
    out_path.mkdir(parents=True, exist_ok=True)
    compute_roofline(
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        TP,
        op_regex,
        num_weight_inits,
        bench_fn=bench_moe_int8_smoothquant,
        intensity_proxy_name="batch",
        intensity_proxy_values=batch_sizes,
        out_path=out_path.with_suffix(".csv"),
    )


def bench_smoothquant_only(batch_sizes, dim, device="cuda"):
    """Benchmark just the smoothquant quantization kernel."""
    import time
    
    print("\n=========================================")
    print("SmoothQuant Quantization Benchmark")
    print("=========================================")
    
    smooth_scale = torch.randn((dim,), device=device, dtype=torch.float32).abs() + 0.1
    
    for batch in batch_sizes:
        x = torch.randn((batch, dim), device=device, dtype=torch.bfloat16)
        
        # Warmup
        for _ in range(10):
            x_int8, x_scale = smoothquant_quantize(x, smooth_scale)
        
        torch.cuda.synchronize()
        
        # Benchmark
        reps = 100
        start = time.perf_counter()
        for _ in range(reps):
            x_int8, x_scale = smoothquant_quantize(x, smooth_scale)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        latency_us = (end - start) / reps * 1e6
        throughput_gb = batch * dim * 2 / latency_us * 1e-3  # GB/s (bf16 input)
        
        print(f"Batch: {batch:5d} | Dim: {dim:5d} | Latency: {latency_us:.2f} us | Throughput: {throughput_gb:.2f} GB/s")


def parse_args():
    parser = argparse.ArgumentParser(prog="Benchmark Int8 SmoothQuant MoE")
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        metavar=("DIM"),
        default=[7168, 4096],
        help="Input feature dimensions of MoE layers [dim1, dim2].",
    )
    parser.add_argument(
        "--experts",
        type=int,
        nargs="+",
        metavar=("DIM"),
        default=[256, 8],
        help="Number of total and active experts [total, active].",
    )
    parser.add_argument(
        "--op-regex",
        type=str,
        default=".*moe_gemm.*|.*smoothquant.*",
        help="Regex to find perf for specific operation by its kernel name.",
    )
    parser.add_argument(
        "--num-weight-inits",
        type=int,
        default=1,
        help="Number of different weight initializations for stable results.",
    )
    parser.add_argument(
        "--bench-quant-only",
        action="store_true",
        help="Only benchmark the smoothquant quantization kernel.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism degree.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.bench_quant_only:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        dim = args.shape[0] if args.shape else 7168
        bench_smoothquant_only(batch_sizes, dim)
    else:
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

        roofline_moe(
            batch_sizes_moe,
            dim1,
            dim2,
            total_experts,
            active_experts,
            TP=args.tp,
            op_regex=args.op_regex,
            num_weight_inits=args.num_weight_inits,
            name="int8-smoothquant-moe",
        )
