import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import run_perftest, checkAllclose, benchmark
from aiter import dtypes
from aiter.ops.activation import fused_silu_mul_per_token_quant, fused_gelu_mul_per_token_quant, fused_gelu_tanh_mul_per_token_quant
from aiter.ops.quant import dynamic_per_token_scaled_quant
from aiter.ops.activation import silu_and_mul, gelu_and_mul, gelu_tanh_and_mul
import pandas as pd
import argparse
from functools import partial


def torch_act_and_mul_quant(torch_act_fn, input, quant_dtype):
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)

    out_fp = torch_act_fn(x) * y

    shape = out_fp.shape
    out_fp_2d = out_fp.view(-1, shape[-1])
    num_tokens = out_fp_2d.shape[0]

    absmax = torch.max(torch.abs(out_fp_2d), dim=-1, keepdim=True)[0]

    if quant_dtype == dtypes.fp8:
        dtype_max = torch.finfo(dtypes.fp8).max
    elif quant_dtype == torch.int8:
        dtype_max = 127.0
    else:
        raise ValueError(f"Unsupported quant dtype: {quant_dtype}")

    scale = absmax / dtype_max
    scale[scale == 0] = 1.0

    out_quant = (out_fp_2d / scale).to(quant_dtype)
    out_quant = out_quant.view(shape)
    scale = scale.view(num_tokens, 1).to(torch.float32)  # Ensure scale is float32

    return out_quant, scale


def separate_kernel_baseline(input: torch.Tensor, quant_dtype: torch.dtype, activation_fn) -> tuple:
    d = input.shape[-1] // 2
    shape = input.shape
    num_tokens = input.numel() // input.shape[-1]

    out_act = torch.empty((num_tokens, d), dtype=input.dtype, device=input.device)
    activation_fn(out_act, input)

    out_quant = torch.empty((num_tokens, d), dtype=quant_dtype, device=input.device)
    scales = torch.empty((num_tokens, 1), dtype=torch.float32, device=input.device)
    dynamic_per_token_scaled_quant(out_quant, out_act, scales)

    return out_quant, scales


@benchmark()
def test_fused_act_mul_quant(torch_act_fn, fused_fn, m, n, dtype, quant_dtype):
    """Test fused ACT+mul+quant kernel"""
    ret = {}
    input_data = torch.randn(m, n, dtype=dtype, device="cuda")
    out = torch.empty((m, n // 2), dtype=quant_dtype, device="cuda")
    scales = torch.empty((m, 1), dtype=torch.float32, device="cuda")

    ref_out, ref_scales = torch_act_and_mul_quant(torch_act_fn, input_data, quant_dtype)

    _, us_fused = run_perftest(
        fused_fn,
        out,
        scales,
        input_data,
    )

    ref_dequant = ref_out.to(torch.float) * ref_scales
    fused_dequant = out.to(torch.float) * scales
    err_out = checkAllclose(ref_dequant, fused_dequant, rtol=5e-2, atol=5e-2)
    err_scale = checkAllclose(ref_scales, scales, rtol=1e-2, atol=1e-2)

    ret["us"] = us_fused
    ret["TB/s"] = (input_data.nbytes + out.nbytes + scales.nbytes) / us_fused / 1e6
    ret["err_out"] = err_out
    ret["err_scale"] = err_scale
    ret["dtype"] = str(dtype)
    ret["quant_dtype"] = str(quant_dtype)

    return ret


@benchmark()
def benchmark_fused_vs_separate(m, n, dtype, quant_dtype, activation="silu"):
    ret = {}
    input_data = torch.randn(m, n, dtype=dtype, device="cuda")

    if activation == "silu":
        fused_fn = fused_silu_mul_per_token_quant
        separate_act_fn = silu_and_mul
    elif activation == "gelu":
        fused_fn = fused_gelu_mul_per_token_quant
        separate_act_fn = gelu_and_mul
    elif activation == "gelu_tanh":
        fused_fn = fused_gelu_tanh_mul_per_token_quant
        separate_act_fn = gelu_tanh_and_mul
    else:
        raise ValueError(f"Unknown activation: {activation}")

    out_fused = torch.empty((m, n // 2), dtype=quant_dtype, device="cuda")
    scales_fused = torch.empty((m, 1), dtype=torch.float32, device="cuda")
    _, us_fused = run_perftest(
        fused_fn,
        out_fused,
        scales_fused,
        input_data,
    )

    out_separate, scales_separate = separate_kernel_baseline(input_data, quant_dtype, separate_act_fn)
    _, us_separate = run_perftest(
        lambda: separate_kernel_baseline(input_data, quant_dtype, separate_act_fn),
    )

    fused_dequant = out_fused.to(torch.float) * scales_fused
    sep_dequant = out_separate.to(torch.float) * scales_separate
    err_out = checkAllclose(fused_dequant, sep_dequant, rtol=5e-2, atol=5e-2)
    err_scale = checkAllclose(scales_fused, scales_separate, rtol=1e-2, atol=1e-2)

    ret["activation"] = activation
    ret["m"] = m
    ret["n"] = n
    ret["dtype"] = str(dtype)
    ret["quant_dtype"] = str(quant_dtype)
    ret["us_fused"] = us_fused
    ret["us_separate"] = us_separate
    ret["speedup"] = us_separate / us_fused
    ret["TB/s_fused"] = (input_data.nbytes + out_fused.nbytes + scales_fused.nbytes) / us_fused / 1e6
    ret["TB/s_separate"] = (input_data.nbytes + out_separate.nbytes + scales_separate.nbytes) / us_separate / 1e6
    ret["err_out"] = err_out
    ret["err_scale"] = err_scale

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                       choices=["correctness", "benchmark", "all"],
                       help="Which tests to run")
    parser.add_argument("--save-csv", action="store_true", default=False,
                        help="Save results to CSV")
    args = parser.parse_args()

    print("=" * 80)
    print("Testing Fused Activation + Quantization Kernels")
    print("=" * 80)

    if args.test in ["correctness", "all"]:
        print("\n" + "=" * 80)
        print("Correctness Tests")
        print("=" * 80)

        test_configs = [
            (32, 4096, dtypes.bf16, dtypes.fp8),
            (128, 4096, dtypes.bf16, dtypes.fp8),
            (1024, 4096, dtypes.bf16, dtypes.fp8),
            (32, 11008, dtypes.bf16, dtypes.fp8),
            (128, 11008, dtypes.bf16, dtypes.fp8),
            (1024, 11008, dtypes.bf16, dtypes.fp8),
            (32, 14784, dtypes.bf16, dtypes.fp8),
            (128, 14784, dtypes.bf16, dtypes.fp8),
            (1024, 14784, dtypes.bf16, dtypes.fp8),
            (32, 29568, dtypes.bf16, dtypes.fp8),
            (128, 29568, dtypes.bf16, dtypes.fp8),
            (1024, 29568, dtypes.bf16, dtypes.fp8),
            (32, 4096, dtypes.fp16, torch.int8),
            (128, 4096, dtypes.fp16, torch.int8),
        ]
        
        for module in [
            ("silu", fused_silu_mul_per_token_quant, F.silu),
            ("gelu", fused_gelu_mul_per_token_quant, F.gelu),
            ("gelu_tanh", fused_gelu_tanh_mul_per_token_quant, partial(F.gelu, approximate="tanh")),
        ]:
            name, fused_fn, torch_act_fn = module
            print(f"\nTesting {name.upper()} + MUL + Token QUANT Fusion:")
            results = []
            for m, n, dtype, quant_dtype in test_configs:
                result = test_fused_act_mul_quant(torch_act_fn, fused_fn, m, n, dtype, quant_dtype)
                results.append(result)
                print(f"  [{m:4d} x {n:5d}] {dtype} -> {quant_dtype}: "
                    f"err_out={result['err_out']:.2e}, err_scale={result['err_scale']:.2e}, "
                    f"time={result['us']:.2f}us, BW={result['TB/s']:.2f}TB/s")

    if args.test in ["benchmark", "all"]:
        print("\n" + "=" * 80)
        print("Performance Benchmarks: Fused vs Separate Kernels")
        print("=" * 80)

        benchmark_configs = [
            # (num_tokens, intermediate_size)
            (32, 4096),
            (64, 4096),
            (128, 4096),
            (256, 4096),
            (512, 4096),
            (1024, 4096),
            (2048, 4096),
            (32, 11008),
            (128, 11008),
            (1024, 11008),
            (32, 14784),
            (128, 14784),
            (1024, 14784),
            (32, 29568),
            (128, 29568),
            (1024, 29568),
        ]

        activations = ["silu", "gelu", "gelu_tanh"]

        all_results = []
        for activation in activations:
            print(f"\n{activation.upper()} Activation:")
            print("-" * 80)
            print(f"{'m':>6} {'n':>6} {'Fused(us)':>12} {'Separate(us)':>14} {'Speedup':>10} {'Fused BW':>12} {'Sep BW':>12}")
            print("-" * 80)

            for m, n in benchmark_configs:
                result = benchmark_fused_vs_separate(m, n, dtypes.bf16, dtypes.fp8, activation)
                all_results.append(result)
                print(f"{m:6d} {n:6d} {result['us_fused']:12.2f} {result['us_separate']:14.2f} "
                      f"{result['speedup']:10.2f}x {result['TB/s_fused']:11.2f}T {result['TB/s_separate']:11.2f}T")

        df = pd.DataFrame(all_results)
        if args.save_csv:
            csv_path = "fused_act_quant_benchmark.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nBenchmark results saved to {csv_path}")

        print("\n" + "=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        for activation in activations:
            df_act = df[df['activation'] == activation]
            avg_speedup = df_act['speedup'].mean()
            max_speedup = df_act['speedup'].max()
            min_speedup = df_act['speedup'].min()
            print(f"{activation.upper():12s}: Avg speedup={avg_speedup:.2f}x, "
                  f"Max={max_speedup:.2f}x, Min={min_speedup:.2f}x")
