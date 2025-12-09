from __future__ import annotations
from typing import Optional, Tuple, Union
import torch

import sys
import warnings
import argparse
import itertools

import triton

from aiter.ops.triton.mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    mha_set_use_fused_bwd_kernel,
)
from aiter.ops.triton.mha_v3 import (
    flash_attn_fp8_func,
    flash_attn_varlen_fp8_func,
)
from aiter.ops.triton.attn_qk_int8_per_block import (
    attn_qk_int8_per_block,
    per_block_int8,
    _get_config,
)
from aiter.test_mha_common import (
    generate_random_padding_mask,
    generate_qkv,
)
from compare_outputs import save_benchmark_output
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    print_vgpr,
    get_caller_name_no_ext,
)
from op_tests.triton_tests.test_mha import check_attention_outputs
from op_tests.op_benchmarks.triton.mha_correctness_utils import (
    primary_output,
    restore_tensor_layout,
    print_output_comparison_stats,
    run_sage_reference,
)



from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3
from aiter.ops.triton.mha_v3 import _quantize_bshd


# test_mha.py configures root logging to DEBUG on import; reset to INFO to avoid noisy deps
import logging
logging.getLogger().setLevel(logging.INFO)


def fav3_fp8_forward_func(
        q: torch.Tensor,  # High precision (BF16/FP32)
        k: torch.Tensor,  # High precision (BF16/FP32)
        v: torch.Tensor,  # High precision (BF16/FP32)
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        attention_chunk: int,
        softcap: float,
        deterministic: bool,
        sm_margin: int,
    ):
        batch, seqlen, num_q_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k.shape

        # Quantize inputs to FP8
        fp8_dtype = torch.float8_e4m3fnuz

        # For GQA/MQA: quantize query with grouped scaling
        group_size = (
            num_q_heads // num_kv_heads if num_q_heads != num_kv_heads else None
        )
        q_fp8, q_descale = _quantize_bshd(q, fp8_dtype, group_size=group_size)
        k_fp8, k_descale = _quantize_bshd(k, fp8_dtype)
        v_fp8, v_descale = _quantize_bshd(v, fp8_dtype)

        # Verify descale shapes for GQA/MQA
        assert q_descale.shape == (
            batch,
            num_kv_heads,
        ), f"q_descale shape {q_descale.shape} != expected {(batch, num_kv_heads)}"
        assert k_descale.shape == (
            batch,
            num_kv_heads,
        ), f"k_descale shape {k_descale.shape} != expected {(batch, num_kv_heads)}"
        assert v_descale.shape == (
            batch,
            num_kv_heads,
        ), f"v_descale shape {v_descale.shape} != expected {(batch, num_kv_heads)}"

        # Derive softmax scale if not provided
        if softmax_scale is None:
            softmax_scale = head_dim ** (-0.5)

        # Validate unsupported features
        if attention_chunk not in (0, 1):
            raise NotImplementedError("attention_chunk > 1 not supported (0 or 1 only)")
        if softcap != 0.0:
            raise NotImplementedError(
                "softcap not implemented in FP8 high-precision API"
            )
        if sm_margin != 0:
            raise NotImplementedError(
                "sm_margin != 0 not supported in FP8 high-precision API"
            )

        # Call flash attention forward
        return lambda: flash_attn_3.fwd(
            q_fp8,
            k_fp8,
            v_fp8,
            None,
            None,
            None,
            None,  # k_new, v_new, qv, out
            None,
            None,
            None,  # cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new
            None,
            None,
            None,
            None,  # seqused_q, seqused_k, max_seqlen_q, max_seqlen_k
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k
            None,
            None,
            None,  # rotary_cos, rotary_sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal,
            int(window_size[0]),
            int(window_size[1]),
            attention_chunk,
            softcap,
            False,  # rotary_interleaved
            None,
            1,
            None,
            sm_margin,  # scheduler_metadata, num_splits, pack_gqa, sm_margin
        )

def nonvarlen_benchmark_configs():
    batch_sizes = [1, 4, 16]
    N_HEADS = [16, 48]
    seq_len_q = [1, 1024, 4096]
    seq_len_k = [163, 8192]
    configs = list(itertools.product(batch_sizes, N_HEADS, seq_len_q, seq_len_k))
    configs = [
        (batch_size, N_HEAD, N_HEAD, seq_len_q, seq_len_k)
        for batch_size, N_HEAD, seq_len_q, seq_len_k in configs
    ]
    return configs


def create_benchmark_configs(custom, args):
    dtype = arg_to_torch_dtype[args.dtype]
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    head_size = 128 if not args.d else args.d
    head_size_v = head_size if not args.dv else args.dv
    mode = args.mode
    x_names = ["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K"]
    causal = args.causal

    configs = []
    plot_name = get_caller_name_no_ext()
    extra_args = {
        "D_HEAD": head_size,
        "D_HEAD_V": head_size_v,
        "dtype": dtype,
        "causal": causal,
        "mode": mode,
    }
    x_vals_list = [(args.b, args.hq, hk, args.sq, sk)]
    unit = ""
    line_vals = ["time(ms)", "throughput(TFLOPS)", "bandwidth(GB/s)", "arithmetic_intensity(FLOP/byte)"]

    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            styles=[("red", "-"), ("green", "-"), ("yellow", "-"), ("blue", "-")],
            ylabel=unit,
            plot_name=plot_name,
            args=extra_args,
        )
    )
    return configs


def run_benchmark(custom, args):
    torch.manual_seed(20)
    saved_output_keys = set()

    @triton.testing.perf_report(create_benchmark_configs(custom, args))
    def bench_mha(
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        D_HEAD_V,
        dtype,
        causal,
        mode,
        provider,
        dropout=0.0,
        model=None,
        sm_scale=None,
        device="cuda",
    ):
        """
        Benchmark or test function for multi-head attention backward pass.
        In test_mode, verifies output matching with non-varlen inputs.
        """
        assert dropout <= 0.0, "Dropout not supported in this benchmark."
        requires_grad = mode == "bwd" or args.test_mode
        return_lse = True
        return_attn_probs = False
        varlen = args.layout == "thd"
        has_pe = D_HEAD > D_HEAD_V
        assert not (
            args.fp8 and has_pe
        ), "Positional Encoding (PE) doesn't support FP8 data type."
        assert not (
            has_pe and "fused-bwd" in provider
        ), "'Fused' backward implementation doesn't support Positional Encoding (PE)."


        # Default softmax scale to match standard attention
        if sm_scale is None:
            sm_scale = 1.0 / (D_HEAD**0.5)

        # Generate base inputs
        q = torch.randn((BATCH, N_CTX_Q, HQ, D_HEAD), device=device, dtype=dtype)
        k = torch.randn((BATCH, N_CTX_K, HK, D_HEAD), device=device, dtype=dtype)
        v = torch.randn((BATCH, N_CTX_K, HK, D_HEAD_V), device=device, dtype=dtype)
        q.requires_grad = requires_grad
        k.requires_grad = requires_grad
        v.requires_grad = requires_grad

        # FLOPS calculation variables
        total_flops = 0.0

        # Input preparation
        q_input, k_input, v_input = q, k, v
        total_flops += (
            2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * (D_HEAD + D_HEAD_V)
        )
        if args.fp8: #  fav3 fp8
            fn = fav3_fp8_forward_func(
                q,
                k,
                v,
                softmax_scale=sm_scale,
                causal=False,
                window_size=(-1, -1),
                attention_chunk=0,
                softcap=0.0,
                deterministic=False,
                sm_margin=0,
            )
        elif args.qk_int8: # sage v1
            assert args.layout == "bshd", "int8 quantization only supports bshd layout."
            # tensor_layout for attn_qk_int8_per_block kernel: "HND" (batch, heads, seq, dim) or "NHD" (batch, seq, heads, dim)
            tensor_layout = args.qk_int8_layout
            config = _get_config()
            # Original tensors are in NHD format (batch, seq, heads, dim)
            if tensor_layout == "HND":
                # Convert from NHD to HND for better memory access
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous().to(torch.float16)
            else:  # NHD
                # Keep original layout
                v = v.to(torch.float16)
            k_mean = None
            sm_scale = D_HEAD**-0.5
            q, q_scale, k, k_scale = per_block_int8(
                q, k, km=k_mean, BLKQ=config["BLOCK_SIZE_M"], BLKK=config["BLOCK_SIZE_N"], sm_scale=sm_scale, tensor_layout=tensor_layout
            )
            q_scale = q_scale.to(torch.float32).unsqueeze(-1).contiguous()
            k_scale = k_scale.to(torch.float32).unsqueeze(-1).contiguous()
            fn = lambda: attn_qk_int8_per_block(q, k, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=torch.bfloat16, config=config)
        else: # fav2 (no quantization)
            def fn():
                return flash_attn_func(
                    q_input,
                    k_input,
                    v_input,
                    dropout_p=dropout,
                    softmax_scale=sm_scale,
                    causal=causal,
                    return_lse=return_lse,
                    return_attn_probs=return_attn_probs,
                )

        metric_choice = args.metric or "all"
        metric_is_all = metric_choice == "all"
        primary_provider = "time(ms)" if metric_is_all else provider
        run_correctness_check = (
            args.check_correctness_FAv3FP8_SageAttnV1
            and args.fp8
            and not varlen
            and (
                not metric_is_all
                or provider == primary_provider
            )
        )
        if args.check_correctness_FAv3FP8_SageAttnV1 and args.fp8 and varlen:
            raise ValueError("--check_correctness_FAv3FP8_SageAttnV1 currently supports only dense (bshd) layout.")

        ms = triton.testing.do_bench(fn)

        needs_output_value = args.save_output or run_correctness_check
        cached_output = fn() if needs_output_value else None

        if args.save_output:
            assert cached_output is not None
            save_benchmark_output(
                args,
                lambda output=cached_output: output,
                saved_output_keys,
                model or "",
                BATCH,
                HQ,
                HK,
                N_CTX_Q,
                N_CTX_K,
                D_HEAD,
                D_HEAD_V,
                dtype,
                causal,
                mode,
                varlen,
            )

        if run_correctness_check:
            assert cached_output is not None
            current_primary = primary_output(cached_output).contiguous()
            reference_primary = primary_output(
                run_sage_reference(
                    q_input,
                    k_input,
                    v_input,
                    args,
                    D_HEAD,
                    D_HEAD_V,
                )
            )
            reference_primary = restore_tensor_layout(
                reference_primary,
                args.qk_int8_layout,
            )
            print("Asserting FP8 output matches SageAttnV1 reference...")
            print_output_comparison_stats(current_primary, reference_primary)
            check_attention_outputs(current_primary, reference_primary, fp8=True)

        total_num_tokens_q = BATCH * N_CTX_Q
        total_num_tokens_k = BATCH * N_CTX_K
        q_size = total_num_tokens_q * HQ * D_HEAD * q.element_size()
        k_size = total_num_tokens_k * HK * D_HEAD * k.element_size()
        v_size = total_num_tokens_k * HK * D_HEAD_V * v.element_size()
        o_size = total_num_tokens_q * HQ * D_HEAD_V * q.element_size()

        # read q, k, v
        mem_read = q_size + k_size + v_size
        # write o
        mem_write = o_size
        mem = mem_read + mem_write

        # return ms
        if "ms" in provider:
            return ms
        elif "TFLOPS" in provider:
            return total_flops / ms * 1e-9
        elif "GB/s" in provider:  # GB/s
            return mem / ms * 1e-6
        elif "arithmetic_intensity" in provider:
            return total_flops / mem


    bench_mha.run(save_path="." if args.o else None, print_data=True)


def supported_layouts():
    layouts = (
        "bshd: Q, K, V are individual tensors of [batch, seqlen_q/k, num_heads, head_size]. "
        "thd: Q, K, V are individual tensors of [total_q/k, num_heads, head_size]. "
    )
    return layouts


# argparse lacks support for boolean argument type (sigh...)
def str2bool(v):
    if isinstance(v, bool) or v is None:
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = get_parser(kernel_name="FlashAttention")
    parser.add_argument(
        "-mode", type=str, default="fwd", help="fwd:forward kernel, bwd:backward kernel"
    )
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-sk", type=int, default=0)
    parser.add_argument(
        "-equal_seqlens",
        action="store_true",
        default=False,
        help="If specified, uses equal sequence lengths with thd layout, i.e t = b * sq",
    )
    parser.add_argument(
        "-d",
        type=int,
        default=0,
        help="Q and K head size, if -dv is absent then -d specifies V head size too",
    )
    parser.add_argument("-dv", type=int, default=0, help="optional V head size")
    parser.add_argument("-causal", type=str2bool, default=None)
    parser.add_argument("-fp8", action="store_true", default=False)
    parser.add_argument("-qk_int8", action="store_true", default=False)
    parser.add_argument("-qk_int8_layout", type=str, default="HND", choices=["HND", "NHD"],
        help="Tensor layout for qk_int8: HND (batch, heads, seq, dim) or NHD (batch, seq, heads, dim). Default: HND.")
    parser.add_argument("-quantize_p", action="store_true", default=False)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("-bench_torch", action="store_true", default=False)
    parser.add_argument("-fused_bwd", action="store_true", default=False)
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument(
        "-test_mode",
        action="store_true",
        default=False,
        help="Tests correctness of the Triton provider comparing the output to the Torch sdpa.",
    )

    parser.add_argument("--layout", type=str, default=None, help=supported_layouts())
    parser.add_argument(
        "-metric",
        nargs="?",
        const="throughput",
        choices=["all", "time", "throughput", "bandwidth"],
        default=None,
        help="Metrics for the kernel benchmark.",
    )
    parser.add_argument(
        "-persistent",
        nargs="?",
        const="fixed",
        choices=["fixed", "dynamic"],
        default=None,
        help="Enable persistent kernels. Use '-persistent dynamic' for dynamic scheduling of the tiles.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    parser.add_argument(
        "--save_output",
        action="store_true",
        default=False,
        help="Store one representative tensor per benchmark configuration for later comparisons.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to store tensors when --save_output is used.",
    )
    parser.add_argument(
        "--check_correctness_FAv3FP8_SageAttnV1",
        action="store_true",
        help="For -fp8 runs, also execute the SageAttn reference kernel and assert outputs match.",
    )
    return parser.parse_args()


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def main():
    args = parse_args()

    if args.model:
        if args.causal is None:  # User didn't specify -causal
            args.causal = True
        if args.layout is None:  # User didn't specify -layout
            args.layout = "thd"
        print(
            f"Note: using -model config defaults: causal={True}, layout={'thd'}. This is the most common real life scenario, but can be overridden with -causal and -layout flags."
        )
    else:
        # the defaults for causal and varlen when not using the -model
        if args.causal is None:  # User didn't specify -causal
            args.causal = False
        if args.layout is None:  # User didn't specify -layout
            args.layout = "bshd"

    custom_config = False

    assert (
        args.layout == "thd" or not args.equal_seqlens or args.model
    ), "Equal sequence lengths arg must be used with the thd layout or a model config."
    # if args.hq or args.hk or args.d or args.dv:
    custom_config = True
    if not args.dv:
        args.dv = args.d
    assert (
        args.b and args.hq and args.sq and args.d and args.dv
    ), "If custom config is specified, please provide \
            all of batch, number of Q heads, Q sequence length \
            and head size."

    if args.model:
        assert not (
            args.hq or args.hk or args.d or args.dv
        ), "Specifying model fixes hq, hk and d already. Do not provide them!"

    assert (
        args.dtype in arg_to_torch_dtype
    ), "Only fp16, bf16 and f32 types currently supported."

    assert (
        args.layout in supported_layouts()
    ), f"{args.layout} is not in supported layouts: {supported_layouts()}."

    if args.layout == "thd" and args.equal_seqlens:
        warnings.warn(
            "Using 'thd' layout with equal_seqlen=True incurs an extra sequence length lookup cost "
            "compared to 'bshd' layout. Consider using 'bshd' for better performance.",
            category=RuntimeWarning,
        )

    if args.print_vgpr:
        assert not args.bench_torch, "Do not use -bench_torch with -print_vgpr."
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(custom_config, args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0

    run_benchmark(custom_config, args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
