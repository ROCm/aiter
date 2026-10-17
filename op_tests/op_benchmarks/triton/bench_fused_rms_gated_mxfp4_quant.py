import argparse
import sys

import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.quant import dynamic_mxfp4_quant, fused_rms_gated_mxfp4_quant
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

EPS = 1e-6


def get_default_shapes() -> list[tuple[int, int, int]]:
    # Qwen3.8-Flash-Next GDN out_proj input per rank: 48 heads x 128 at TP1,
    # 24 at TP2; decode M up to 64, plus one prefill-sized M.
    return [(M, N, 128) for N in (6144, 3072) for M in (1, 4, 8, 16, 31, 32, 64, 1024)]


def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def torch_rms_gated(x, weight, z, group_size, norm_before_gate, activation):
    act = F.sigmoid if activation == "sigmoid" else F.silu
    xf, zf = x.float(), z.float()
    if not norm_before_gate:
        xf = xf * act(zf)
    xg = xf.view(xf.shape[0], -1, group_size)
    y = xg * torch.rsqrt(xg.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = y.view_as(xf) * weight.float().repeat(xf.shape[1] // group_size)
    if norm_before_gate:
        y = y * act(zf)
    return y.to(x.dtype)


def run_benchmark(args):
    if args.shape is not None:
        M, N, G = args.shape
        x_vals = [(M, N, G)]
    else:
        x_vals = get_default_shapes()
    providers = args.provider.split(",")

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    benchmark = triton.testing.Benchmark(
        x_names=["M", "N", "group_size"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={
            "metric": args.metric,
            "dtype": args.dtype,
            "activation": args.activation,
            "norm_before_gate": not args.norm_after_gate,
        },
    )

    @triton.testing.perf_report([benchmark])
    def bench_fused_rms_gated_mxfp4_quant(
        M, N, group_size, metric, provider, dtype, activation, norm_before_gate
    ):
        dtype = get_dtype(dtype)
        x = torch.randn((M, N), dtype=dtype, device="cuda")
        z = torch.randn((M, N), dtype=dtype, device="cuda")
        w = torch.randn(group_size, dtype=dtype, device="cuda")

        if provider == "fused":

            def fn():
                fused_rms_gated_mxfp4_quant(
                    x,
                    w,
                    z,
                    EPS,
                    norm_before_gate=norm_before_gate,
                    activation=activation,
                    group_size=group_size,
                )

        elif provider == "quant_only":
            # The quant alone on an already normalized activation: what the
            # fused kernel costs on top of it.
            y = torch_rms_gated(x, w, z, group_size, norm_before_gate, activation)

            def fn():
                dynamic_mxfp4_quant(y)

        elif provider == "torch":

            def fn():
                dynamic_mxfp4_quant(
                    torch_rms_gated(x, w, z, group_size, norm_before_gate, activation)
                )

        else:
            raise ValueError(f"Unknown provider: {provider}")

        ms = triton.testing.do_bench_cudagraph(fn, rep=100)

        # Read x, z and weight; write fp4 output and e8m0 scales.
        total_bytes = 2 * x.numel() * x.element_size() + M * (N // 2) + M * (N // 32)
        if metric == "time":
            return ms
        return total_bytes / (ms * 1e-3) * 1e-9

    bench_fused_rms_gated_mxfp4_quant.run(
        save_path="." if args.o else None, print_data=True
    )


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="Benchmark fused gated RMSNorm + MXFP4 quant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("M", "N", "GROUP_SIZE"),
        help="Single shape to benchmark.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="fused,quant_only,torch",
        help="Comma-separated providers from: fused, quant_only, torch.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Input dtype.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["silu", "sigmoid"],
        default="sigmoid",
        help="Gate activation.",
    )
    parser.add_argument(
        "--norm-after-gate",
        action="store_true",
        help="Gate before the norm (norm_before_gate=False).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "bandwidth"],
        default="time",
        help="Metric to plot.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file."
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None):
    run_benchmark(parse_args(args))


if __name__ == "__main__":
    sys.exit(main())
