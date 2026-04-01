from __future__ import annotations
from typing import List, Dict, Any
import csv
import torch
import os
import glob
import sys
import argparse
import triton
import logging
import aiter

from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
    fav3_sage_mxfp4_wrapper,
    get_sage_fwd_configs_mxfp4,
    fav3_sage_mxfp4_func,
)

from aiter.ops.triton.attention.fav3_sage import (
    block_attn_mask_to_ragged_lut,
)

from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    sage_quant_mxfp4,
)

from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    create_hadamard_matrix,
)

from aiter.test_mha_common import (
    attention_ref,
    attention_ref_block_sparse,
)
from op_tests.triton_tests.attention.test_fav3_sage import (
    compare_accuracy,
    check_attention_outputs,
    input_helper,
)
from op_tests.op_benchmarks.triton.bench_fav3_sage import fav2_forward_func, sparse_flops_from_lut
from op_tests.op_benchmarks.triton.utils.benchmark_utils import print_vgpr, get_caller_name_no_ext

# Configuration
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def layout_preprocess(q, k, v, layout: str, target_layout: str = "bshd"):
    if layout != target_layout:
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    return q, k, v


def primary_output(result):
    """Return the main tensor output produced by a kernel."""
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (list, tuple)) and len(result) > 0:
        return result[0]
    return result


def bench_kernel(q, k, v, args, provider, block_lut=None, block_attn_mask=None):
    """Main benchmarking logic for a single configuration."""
    if args.layout == "bshd":
        BATCH, N_CTX_Q, HQ, D_HEAD = q.shape
        _, N_CTX_K, HK, D_HEAD_V = v.shape
    else:
        BATCH, HQ, N_CTX_Q, D_HEAD = q.shape
        _, HK, N_CTX_K, D_HEAD_V = v.shape

    BLOCK_R = args.BLOCK_R
    R = create_hadamard_matrix(BLOCK_R, device=q.device, dtype=q.dtype) / (BLOCK_R**0.5)

    if args.include_quant_overhead:

        def fn():
            return fav3_sage_mxfp4_wrapper(
                q,
                k,
                v,
                causal=args.causal,
                layout=args.layout,
                q_smooth=args.qsmooth,
                hadamard_rotation=args.hadamard_rotate,
                R=R,
                block_lut=block_lut,
            )

    else:
        config = get_sage_fwd_configs_mxfp4()

        FP8_TYPE = aiter.dtypes.fp8
        FP8_MAX = torch.finfo(FP8_TYPE).max
        (
            q_quantized,
            q_descale,
            k_quantized,
            k_descale,
            v_quantized,
            v_descale,
            delta_s,
        ) = sage_quant_mxfp4(
            q,
            k,
            v,
            FP8_TYPE,
            FP8_MAX,
            BLKQ=config["BLOCK_M"],
            BLKK=64,
            layout=args.layout,
            R=R,
            BLOCK_R=BLOCK_R,
            q_smoothing=args.qsmooth,
        )

        if block_lut is not None:
            kv_block_indices, lut_start_t, lut_count_t = block_lut
            use_block_sparse = True
        else:
            kv_block_indices = lut_start_t = lut_count_t = None
            use_block_sparse = False

        def fn():
            return fav3_sage_mxfp4_func(
                q=q_quantized,
                k=k_quantized,
                v=v_quantized,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                bias=delta_s,
                causal=args.causal,
                layout=args.layout,
                config=config,
                kv_block_indices=kv_block_indices,
                lut_start=lut_start_t,
                lut_count=lut_count_t,
                use_block_sparse=use_block_sparse,
            )

    rep = getattr(args, "rep", 100)
    warmup = getattr(args, "warmup", 25)
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    if getattr(args, "compare_to_ref", False):
        current_output = fn()
        assert current_output is not None
        current_primary = primary_output(current_output)

        if args.layout == "bhsd":
            current_primary = current_primary.permute(0, 2, 1, 3)

        ref_name = getattr(args, "ref", None) or "torch"
        if block_attn_mask is not None:
            if ref_name != "torch":
                raise ValueError(f"Reference kernel {ref_name} not supported for block sparsity")

            q_bshd, k_bshd, v_bshd = layout_preprocess(
                q, k, v, layout=args.layout, target_layout="bshd"
            )
            config = get_sage_fwd_configs_mxfp4()
            ref_out = attention_ref_block_sparse(
                q_bshd,
                k_bshd,
                v_bshd,
                block_attn_mask,
                config["BLOCK_M"],
                config["BLOCK_N"],
                dropout_p=0.0,
                dropout_mask=None,
                upcast=True,
            )
            reference_primary = ref_out[0]
        else:
            q_ref, k_ref, v_ref = layout_preprocess(
                q, k, v, layout=args.layout, target_layout="bshd"
            )
            sm_scale = D_HEAD**-0.5

            if ref_name == "fav2":
                ref_out = fav2_forward_func(
                    q_ref, k_ref, v_ref,
                    dropout_p=0.0, softmax_scale=sm_scale,
                    causal=False, return_lse=False, return_attn_probs=False,
                )()
            else:
                ref_out = attention_ref(
                    q_ref, k_ref, v_ref,
                    dropout_p=0.0, dropout_mask=None, causal=False,
                )

            reference_primary = primary_output(ref_out)
        compare_accuracy(current_primary, reference_primary)
        check_attention_outputs(current_primary, reference_primary, fp8=False)

    total_flops = 2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * (D_HEAD + D_HEAD_V)

    if block_lut is not None:
        sparse_flops, _ = sparse_flops_from_lut(block_lut, BATCH, N_CTX_Q, N_CTX_K, HQ, D_HEAD, D_HEAD_V)
    else:
        sparse_flops = 0

    q_element_size = q.element_size()
    k_element_size = k.element_size()
    v_element_size = v.element_size()

    total_num_tokens_q = BATCH * N_CTX_Q
    total_num_tokens_k = BATCH * N_CTX_K
    q_size = total_num_tokens_q * HQ * D_HEAD * q_element_size
    k_size = total_num_tokens_k * HK * D_HEAD * k_element_size
    v_size = total_num_tokens_k * HK * D_HEAD_V * v_element_size
    o_size = total_num_tokens_q * HQ * D_HEAD_V * q_element_size
    mem = q_size + k_size + v_size + o_size

    if "ms" in provider:
        return ms
    elif "throughput_sparse(TFLOPS)" in provider:
        return sparse_flops / ms * 1e-9
    elif "TFLOPS" in provider:
        return total_flops / ms * 1e-9
    elif "GB/s" in provider:
        return mem / ms * 1e-6
    return ms

def create_benchmark_configs(args):
    dtype = arg_to_torch_dtype[args.dtype]
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    head_size = 128 if not args.d else args.d
    head_size_v = head_size if not args.dv else args.dv
    layout = args.layout if args.layout else "bshd"
    x_names = ["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K"]
    causal = False

    configs = []
    plot_name = get_caller_name_no_ext()
    extra_args = {
        "D_HEAD": head_size,
        "D_HEAD_V": head_size_v,
        "dtype": dtype,
        "layout": layout,
        "causal": causal,
    }
    x_vals_list = [(args.b, args.hq, hk, args.sq, sk)]
    unit = ""
    line_vals = [
        "time(ms)",
        "throughput(TFLOPS)",
        "bandwidth(GB/s)",
    ]
    if getattr(args, "block_sparsity", None) is not None or getattr(
        args, "block_mask_file", None
    ):
        line_vals.append("throughput_sparse(TFLOPS)")

    # if comparing to reference, or specific metric provided, adjust line_vals accordingly
    if args.compare_to_ref or (args.metric and args.metric != "all"):
        if args.compare_to_ref:
            line_vals = [
                "time(ms)"
            ]  # avoid redundant runs of other metrics when comparing to reference. default to time only.
        else:
            metric_map = {
                "time": "time(ms)",
                "throughput": "throughput(TFLOPS)",
                "bandwidth": "bandwidth(GB/s)",
                "throughput_sparse": "throughput_sparse(TFLOPS)",
            }
            line_vals = [metric_map[args.metric]]

    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            styles=[("red", "-"), ("green", "-"), ("yellow", "-"), ("blue", "-"), ("cyan", "-")],
            ylabel=unit,
            plot_name=plot_name,
            args=extra_args,
        )
    )
    return configs

def run_benchmark(args):
    torch.manual_seed(20)

    @triton.testing.perf_report(create_benchmark_configs(args))
    def bench_mha(
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        D_HEAD_V,
        dtype,
        layout,
        causal,
        provider,
        device="cuda",
    ):
        q = torch.randn((BATCH, HQ, N_CTX_Q, D_HEAD), device=device, dtype=dtype)
        k = torch.randn((BATCH, HK, N_CTX_K, D_HEAD), device=device, dtype=dtype)
        v = torch.randn((BATCH, HK, N_CTX_K, D_HEAD_V), device=device, dtype=dtype)

        q, k, v = layout_preprocess(q, k, v, layout="bhsd", target_layout=layout)

        block_lut = None
        block_attn_mask = None
        if getattr(args, "block_sparsity", None) is not None:
            config = get_sage_fwd_configs_mxfp4()
            BLOCK_M, BLOCK_N = config["BLOCK_M"], config["BLOCK_N"]
            num_q_blocks = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
            num_kv_blocks = (N_CTX_K + BLOCK_N - 1) // BLOCK_N
            block_attn_mask = (
                torch.rand(BATCH, HQ, num_q_blocks, num_kv_blocks, device=device)
                > args.block_sparsity
            ).to(torch.bool)
            block_lut = block_attn_mask_to_ragged_lut(block_attn_mask)

        return bench_kernel(q, k, v, args, provider, block_lut=block_lut, block_attn_mask=block_attn_mask)

    bench_mha.run(save_path="." if getattr(args, "o", False) else None, print_data=True)


def run_benchmark_block_sparse_repetitions(args):
    """
    When -block_sparsity and --n_repetitions are set: run n_repetitions times with
    a new random block mask each time, report throughput statistics (median, Q1, Q3, p10, p90).
    """
    torch.manual_seed(20)
    device = "cuda"
    dtype = arg_to_torch_dtype[args.dtype]
    layout = args.layout
    hk = args.hk
    BATCH, HQ, N_CTX_Q, N_CTX_K = args.b, args.hq, args.sq, args.sk
    D_HEAD = args.d
    D_HEAD_V = args.dv

    q = torch.randn((BATCH, HQ, N_CTX_Q, D_HEAD), device=device, dtype=dtype)
    k = torch.randn((BATCH, hk, N_CTX_K, D_HEAD), device=device, dtype=dtype)
    v = torch.randn((BATCH, hk, N_CTX_K, D_HEAD_V), device=device, dtype=dtype)
    q.requires_grad = False
    k.requires_grad = False
    v.requires_grad = False
    q, k, v = layout_preprocess(q, k, v, layout="bhsd", target_layout=layout)

    config = get_sage_fwd_configs_mxfp4()
    BLOCK_M, BLOCK_N = config["BLOCK_M"], config["BLOCK_N"]
    num_q_blocks = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (N_CTX_K + BLOCK_N - 1) // BLOCK_N

    # JIT warmup: compile kernel before timed runs so reported ms is not inflated.
    _warmup_mask = (
        torch.rand(BATCH, HQ, num_q_blocks, num_kv_blocks, device=device) > args.block_sparsity
    ).to(torch.bool)
    _warmup_lut = block_attn_mask_to_ragged_lut(_warmup_mask)
    bench_kernel(
        q, k, v, args, "time(ms)", block_lut=_warmup_lut, block_attn_mask=_warmup_mask
    )

    n_rep = args.n_repetitions
    throughputs_tflops = []
    latencies_ms = []
    effective_tflops_list = []
    for _ in range(n_rep):
        block_attn_mask = (
            torch.rand(BATCH, HQ, num_q_blocks, num_kv_blocks, device=device) > args.block_sparsity
        ).to(torch.bool)
        block_lut = block_attn_mask_to_ragged_lut(block_attn_mask)
        ms = bench_kernel(
            q, k, v, args, "time(ms)", block_lut=block_lut, block_attn_mask=block_attn_mask
        )
        sparse_flops, total_flops = sparse_flops_from_lut(block_lut, BATCH, N_CTX_Q, N_CTX_K, HQ, D_HEAD, D_HEAD_V)
        latencies_ms.append(ms)
        ops_per_sec = total_flops / (ms * 1e-3)
        tflops = ops_per_sec / 1e12
        throughputs_tflops.append(tflops)
        effective_tflops = (sparse_flops / (ms * 1e-3)) / 1e12
        effective_tflops_list.append(effective_tflops)

    t = torch.tensor(throughputs_tflops)
    median_tflops = torch.quantile(t, 0.5).item()
    q1_tflops = torch.quantile(t, 0.25).item()
    q3_tflops = torch.quantile(t, 0.75).item()
    p10_tflops = torch.quantile(t, 0.1).item()
    p90_tflops = torch.quantile(t, 0.9).item()

    t_lat = torch.tensor(latencies_ms)
    median_latency_ms = torch.quantile(t_lat, 0.5).item()
    q1_latency_ms = torch.quantile(t_lat, 0.25).item()
    q3_latency_ms = torch.quantile(t_lat, 0.75).item()
    p10_latency_ms = torch.quantile(t_lat, 0.1).item()
    p90_latency_ms = torch.quantile(t_lat, 0.9).item()

    t_eff = torch.tensor(effective_tflops_list)
    median_effective_tflops = torch.quantile(t_eff, 0.5).item()
    q1_effective_tflops = torch.quantile(t_eff, 0.25).item()
    q3_effective_tflops = torch.quantile(t_eff, 0.75).item()
    p10_effective_tflops = torch.quantile(t_eff, 0.1).item()
    p90_effective_tflops = torch.quantile(t_eff, 0.9).item()

    summary = (
        f"block_sparsity={args.block_sparsity}, n_repetitions={n_rep}: "
        f"median_TFLOPS={median_tflops:.4f}, Q1={q1_tflops:.4f}, Q3={q3_tflops:.4f}, "
        f"p10={p10_tflops:.4f}, p90={p90_tflops:.4f} | "
        f"median_latency_ms={median_latency_ms:.4f}, Q1={q1_latency_ms:.4f}, Q3={q3_latency_ms:.4f}, "
        f"p10={p10_latency_ms:.4f}, p90={p90_latency_ms:.4f} | "
        f"median_effective_TFLOPS={median_effective_tflops:.4f}, Q1={q1_effective_tflops:.4f}, "
        f"Q3={q3_effective_tflops:.4f}, p10={p10_effective_tflops:.4f}, p90={p90_effective_tflops:.4f}"
    )
    logger.info(summary)
    print(summary)

    if getattr(args, "o", False):
        csv_path = "bench_fav3_sage_mxfp4_block_sparse_repetitions.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [
                BATCH, HQ, N_CTX_Q, N_CTX_K, D_HEAD, D_HEAD_V,
                args.block_sparsity, n_rep,
                median_tflops, q1_tflops, q3_tflops, p10_tflops, p90_tflops,
                median_latency_ms, q1_latency_ms, q3_latency_ms, p10_latency_ms, p90_latency_ms,
                median_effective_tflops, q1_effective_tflops, q3_effective_tflops,
                p10_effective_tflops, p90_effective_tflops,
            ]
            if not file_exists:
                writer.writerow([
                    "BATCH", "HQ", "N_CTX_Q", "N_CTX_K", "D_HEAD", "D_HEAD_V",
                    "block_sparsity", "n_repetitions",
                    "median_TFLOPS", "q1_TFLOPS", "q3_TFLOPS", "p10_TFLOPS", "p90_TFLOPS",
                    "median_latency_ms", "q1_latency_ms", "q3_latency_ms", "p10_latency_ms", "p90_latency_ms",
                    "median_effective_TFLOPS", "q1_effective_TFLOPS", "q3_effective_TFLOPS",
                    "p10_effective_TFLOPS", "p90_effective_TFLOPS",
                ])
            writer.writerow(row)
        logger.info(f"Wrote CSV row to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simplified MXFP4 Attention Benchmark")
    parser.add_argument("-b", type=int, required=True, help="Batch size")
    parser.add_argument("-hq", type=int, required=True, help="Number of Q heads")
    parser.add_argument("-hk", type=int, default=0, help="Number of K heads (GQA)")
    parser.add_argument("-sq", type=int, required=True, help="Q Sequence length")
    parser.add_argument("-sk", type=int, default=0, help="K Sequence length")
    parser.add_argument("-d", type=int, required=True, help="Head dimension")
    parser.add_argument("-dv", type=int, default=0, help="V head dimension")
    parser.add_argument("-dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("-layout", type=str, default="bhsd", choices=["bshd", "bhsd"])
    parser.add_argument(
        "-captured_dir",
        type=str,
        default=None,
        help="Provide dir for captured inputs, for accuracy comparison.",
    )
    parser.add_argument(
        "-hadamard_rotate",
        type=lambda v: bool(int(v)),
        default=True,
        help="whether to apply hadamard rotate (1) or not (0). Default 1.",
    )

    parser.add_argument(
        "-BLOCK_R",
        type=int,
        default=128,
        help="Hadamard matrix size. Should be <= d",
    )
    parser.add_argument(
        "-qsmooth",
        action="store_true",
        help="Do q smoothing (Warning! Smoothing Q requires bias addition which drops the perf as of now!)",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        help="Print VGPR usage of the called Triton kernels",
    )
    parser.add_argument(
        "-causal",
        action="store_true",
        help="Causal masking",
    )
    parser.add_argument(
        "-test",
        action="store_true",
        help="Test benchmark shape correctness.",
    )
    parser.add_argument(
        "-include_quant_overhead",
        action="store_true",
        help="Include quantization overhead to bench.",
    )
    parser.add_argument(
        "--block_sparsity",
        type=float,
        default=None,
        help="Fraction of (q_block, kv_block) pairs disallowed (0=dense, 0.5=50%% masked). Uses random mask.",
    )
    parser.add_argument(
        "--n_repetitions",
        type=int,
        default=None,
        help="When -block_sparsity is set: run this many times with new random mask each time; report throughput stats.",
    )
    parser.add_argument(
        "--compare_to_ref",
        action="store_true",
        help="Execute the reference kernel (-ref) and assert outputs match.",
    )
    parser.add_argument(
        "-ref",
        type=str,
        default=None,
        help="Reference kernel for --compare_to_ref: torch (default) or fav2.",
    )
    parser.add_argument(
        "-metric",
        nargs="?",
        const="throughput",
        choices=["all", "time", "throughput", "bandwidth"],
        default=None,
        help="Metrics for the kernel benchmark.",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        help="Write performance results to CSV file",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Repetition time in ms for triton.testing.do_bench.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup time in ms for triton.testing.do_bench.",
    )

    return parser.parse_args()


def load_captured_inputs(input_dir: str) -> List[Dict[str, Any]]:
    """
    Load captured input tensors from disk.
    Args:
        input_dir: Directory containing captured .pt files

    Returns:
        List of dictionaries containing q, k, v tensors and metadata
    """
    input_files = sorted(glob.glob(os.path.join(input_dir, "*_input_*.pt")))
    if not input_files:
        raise FileNotFoundError(f"No captured input files found in {input_dir}")

    inputs = []
    for f in input_files:
        data = torch.load(f, weights_only=False)
        inputs.append(data)

    return inputs


def test_accuracy(q, k, v, args):

    BLOCK_R = args.BLOCK_R
    R = create_hadamard_matrix(BLOCK_R, device=q.device, dtype=q.dtype) / (BLOCK_R**0.5)

    triton_out = fav3_sage_mxfp4_wrapper(
        q,
        k,
        v,
        causal=args.causal,
        layout=args.layout,
        q_smooth=args.qsmooth,
        hadamard_rotation=args.hadamard_rotate,
        R=R,
    )
    # permute because FAv2 assumes bshd
    if args.layout == "bhsd":
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

    print("Using as ref: Triton FAv2")
    sm_scale = q.shape[-1] ** -0.5
    ref_out = fav2_forward_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=False,
        return_lse=False,
        return_attn_probs=False,
    )()
    if args.layout == "bhsd":
        ref_out = ref_out.permute(0, 2, 1, 3)

    assert ref_out.shape == triton_out.shape
    compare_accuracy(triton_out, ref_out)


def test_accuracy_with_captured_inputs(args):
    input_dir = args.captured_dir
    inputs = load_captured_inputs(input_dir)
    n_ = len(inputs)

    for input_i in range(n_):
        # Get the input tensors for this configuration
        inp = inputs[input_i]
        q = inp["q"].to("cuda")
        k = inp["k"].to("cuda")
        v = inp["v"].to("cuda")
        print("Testing accuracy on captured input:")
        print("q.shape: ", q.shape)
        print("k.shape: ", k.shape)
        print("v.shape: ", v.shape)
        test_accuracy(q, k, v, args)


def test_accuracy_with_shape(
    args,
    dtype=torch.bfloat16,
):
    torch.cuda.empty_cache()
    torch.manual_seed(20)
    q, k, v = input_helper(
        args.b,
        args.hq,
        args.hk,
        args.sq,
        args.sk,
        args.d,
        args.dv,
        dtype,
        args.layout,
    )
    print("Testing accuracy on shape:")
    print("q.shape: ", q.shape)
    print("k.shape: ", k.shape)
    print("v.shape: ", v.shape)
    test_accuracy(q, k, v, args)


def main():
    args = parse_args()
    if not args.dv:
        args.dv = args.d
    if not args.sk:
        args.sk = args.sq
    if not args.hk:
        args.hk = args.hq

    assert args.BLOCK_R <= args.d, "Rotation block size should be <= d"

    if getattr(args, "block_sparsity", None) is not None:
        if not (0 <= args.block_sparsity <= 1):
            raise ValueError(
                f"-block_sparsity must be in [0, 1], got {args.block_sparsity}"
            )

    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), "MXFP4_Attention_Performance")
        return 0

    if args.test:
        if args.captured_dir is not None:
            test_accuracy_with_captured_inputs(args)
        else:
            test_accuracy_with_shape(args)

    # Block-sparsity with n_repetitions: throughput stats path
    if (
        args.block_sparsity is not None
        and getattr(args, "n_repetitions", None) is not None
    ):
        run_benchmark_block_sparse_repetitions(args)
        return 0

    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
