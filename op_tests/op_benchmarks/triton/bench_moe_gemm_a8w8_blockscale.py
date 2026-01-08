# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py

from itertools import chain
from pathlib import Path
import triton.profiler as proton
import torch
import argparse
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
    print(f"{out_path}...")
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
):
    rank = 0
    dev = f"cuda:{rank}"

    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"

    # -- init data --
    # Assume that x and w are quantized shapes if bs8, else assume they are fp8
    group_shape_m, group_shape_n, group_shape_k = group_shape

    # weights
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
    # biases
    bg = torch.randn((n_expts_tot,), device=dev)
    b1 = torch.randn((n_expts_tot, dim2 // TP), dtype=torch.float32, device=dev)
    b2 = torch.randn((n_expts_tot, dim1), dtype=torch.float32, device=dev)

    # -- benchmark --
    x_dtype_str = x_dtype
    w_dtype_str = w_dtype
    x_dtype = torch.float8_e4m3fn
    w_dtype = torch.float8_e4m3fn
    # special treatment of fp8_e4m3 on AMD CDNA3 because it uses fp8_e4m3fnuz
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
            # [M, K_blocks]
            x_scale = torch.rand((batch, k_blocks_x), dtype=torch.float32, device=dev)
        else:
            # [M_blocks, K_blocks]
            m_blocks = num_blocks(batch, group_shape_m)
            x_scale = torch.rand(
                (m_blocks, k_blocks_x), dtype=torch.float32, device=dev
            )
    if w_dtype_str == "fp8":
        w_static_scale = torch.tensor(1e-4, device=dev)
    else:
        # [n_expts_tot, dim1, dim2 // TP]
        k_blocks_w1 = num_blocks(dim1, group_shape_k)
        n_blocks_w1 = num_blocks(dim2 // TP, group_shape_n)
        # [n_expts_tot, dim2 // TP // 2, dim1]
        k_blocks_w2 = num_blocks(dim2 // TP // 2, group_shape_k)
        n_blocks_w2 = num_blocks(dim1, group_shape_n)

        w1_scale = torch.rand(
            (n_expts_tot, k_blocks_w1, n_blocks_w1), dtype=torch.float32, device=dev
        )
        w2_scale = torch.rand(
            (n_expts_tot, k_blocks_w2, n_blocks_w2), dtype=torch.float32, device=dev
        )

    # run layer
    fpath = Path(tempfile.mktemp())
    M, K = xg.shape
    K, N = wg.shape
    # Reduce blocksize to prevent LDS out of resource limits
    config, _ = _get_config(M, N, K)
    config["BLOCK_SIZE_M"] = (
        128 if config["BLOCK_SIZE_M"] > 128 else config["BLOCK_SIZE_M"]
    )
    config["BLOCK_SIZE_N"] = (
        128 if config["BLOCK_SIZE_N"] > 128 else config["BLOCK_SIZE_N"]
    )
    config["BLOCK_SIZE_K"] = (
        128 if config["BLOCK_SIZE_K"] > 128 else config["BLOCK_SIZE_K"]
    )
    proton.start(str(fpath), hook="triton")
    for i in range(reps):
        logits = gemm_a16w16(xg, wg.T, bg, config=config)
        rdata, gather_indx, scatter_indx = routing(logits, n_expts_act)
        if x_dtype_str == "fp8" and w_dtype_str == "fp8":
            x = moe_gemm_a8w8_blockscale(
                x,
                w1,
                None,
                None,
                x_static_scale,
                w_static_scale,
                None,
                b1,
                rdata,
                gather_indx=gather_indx,
                out_dtype=x_dtype,
                apply_swiglu=True,
            )
            x = moe_gemm_a8w8_blockscale(
                x,
                w2,
                None,
                None,
                x_static_scale,
                w_static_scale,
                None,
                b2,
                rdata,
                out_dtype=x_dtype,
                scatter_indx=scatter_indx,
            )
        elif x_dtype_str == "fp8" and w_dtype_str == "bs8":
            x = moe_gemm_a8w8_blockscale(
                x,
                w1,
                None,
                w1_scale,
                x_static_scale,
                None,
                None,
                b1,
                rdata,
                gather_indx=gather_indx,
                out_dtype=x_dtype,
                apply_swiglu=True,
            )
            x = moe_gemm_a8w8_blockscale(
                x,
                w2,
                None,
                w2_scale,
                x_static_scale,
                None,
                None,
                b2,
                rdata,
                out_dtype=x_dtype,
                scatter_indx=scatter_indx,
            )

        elif x_dtype_str == "bs8" and w_dtype_str == "fp8":
            x = moe_gemm_a8w8_blockscale(
                x,
                w1,
                x_scale,
                None,
                None,
                w_static_scale,
                None,
                b1,
                rdata,
                gather_indx=gather_indx,
                out_dtype=x_dtype,
                apply_swiglu=True,
            )
            x = moe_gemm_a8w8_blockscale(
                x,
                w2,
                x_scale,
                None,
                None,
                w_static_scale,
                None,
                b2,
                rdata,
                out_dtype=x_dtype,
                scatter_indx=scatter_indx,
            )
        else:
            x = moe_gemm_a8w8_blockscale(
                x,
                w1,
                x_scale,
                w1_scale,
                None,
                None,
                None,
                b1,
                rdata,
                out_dtype=x_dtype,
                gather_indx=gather_indx,
                apply_swiglu=True,
            )
            x = moe_gemm_a8w8_blockscale(
                x,
                w2,
                x_scale,
                w2_scale,
                None,
                None,
                None,
                b2,
                rdata,
                out_dtype=x_dtype,
                scatter_indx=scatter_indx,
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
    per_row_act_quant,
    TP,
    op_regex,
    name="",
):
    out_path = Path(f"logs/{name}/{x_dtype}x-{w_dtype}w-TP{TP}/")
    out_path.mkdir(parents=True, exist_ok=True)
    compute_roofline(
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        x_dtype,
        w_dtype,
        per_row_act_quant,
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
    quantized_dtypes = [args.act_dtype, args.w_dtype]
    per_row_act_quant = args.act_per_row_bs

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
        op_regex=args.op_regex,
        name="gpt-oss-x2",
    )
