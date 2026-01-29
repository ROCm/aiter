from __future__ import annotations
from typing import Literal, Optional, Tuple, List, Dict, Any
import torch
import os
import glob

import sys
import argparse
import aiter
import triton
import logging

from aiter.ops.triton.mha import (
    flash_attn_func,
)

from aiter.test_mha_common import (
    attention_ref,
)
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    print_vgpr,
    get_caller_name_no_ext,
)
from op_tests.triton_tests.attention.test_sage_attn import check_attention_outputs
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3
from aiter.ops.triton.attention.mha_v3 import _quantize_bshd

from aiter.ops.triton.attention.fav3_sage import (
    fav3_sage_func,
)
from aiter.ops.triton._triton_kernels.sage_attn_triton_amd import (
    fav3_sage,
    get_sage_fwd_configs,
    sage_quant,
    sage_quant_v2,
    sage_quant_v3
)
from op_tests.triton_tests.utils.accuracy_analysis import compare_accuracy

CAUSAL = False
layout_converter = {
    "bshd": "NHD",
    "bhsd": "HND",
}

reversed_layout_converter = {v: k for k, v in layout_converter.items()}

# test_mha.py configures root logging to DEBUG on import; reset to INFO to avoid noisy deps
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def get_arch():
    return triton.runtime.driver.active.get_current_target().arch


def layout_preprocess(
    q,
    k,
    v,
    layout: Literal["bshd", "bhsd"],
    target_layout: Literal["bshd", "bhsd"] = "bshd",
):
    """
    Preprocess input tensors to the target layout.

    Args:
        q, k, v: Input tensors
        layout: Current layout of the tensors
        target_layout: Desired layout of the tensors

    Returns:
        q, k, v tensors in the target layout
    """
    if layout != target_layout:
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

    return q, k, v


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
    for i, f in enumerate(input_files):
        data = torch.load(f, weights_only=False)
        inputs.append(data)
        # logger.info(f"Loaded [{i}] {os.path.basename(f)}: q={tuple(data['q_shape'])}")

    logger.info(f"Loaded {len(inputs)} captured inputs for benchmarking")
    return inputs


def fp8_quantize(q, k, v, scale=None):
    quant_dtype = aiter.dtypes.fp8
    # Computing "dynamic" scale before quantization improves thpt a small amount (~1-2%) for the (1, 75352, 5, 128) shape
    quant_q, q_descale = aiter.per_tensor_quant(
        q,
        scale=torch.abs(q).max() if scale is None else scale,
        quant_dtype=quant_dtype,
        dtypeMax=torch.finfo(quant_dtype).max,
    )
    quant_k, k_descale = aiter.per_tensor_quant(
        k,
        scale=torch.abs(k).max() if scale is None else scale,
        quant_dtype=quant_dtype,
        dtypeMax=torch.finfo(quant_dtype).max,
    )
    quant_v, v_descale = aiter.per_tensor_quant(
        v,
        scale=torch.abs(v).max() if scale is None else scale,
        quant_dtype=quant_dtype,
        dtypeMax=torch.finfo(quant_dtype).max,
    )
    return quant_q, quant_k, quant_v, q_descale, k_descale, v_descale


def run_aiter_fp8_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    has_descale: bool = False,
    scale: Optional[torch.Tensor] = None,
):
    scale = scale
    q, k, v, q_descale, k_descale, v_descale = fp8_quantize(q, k, v, scale=scale)
    attn_kwargs = {}
    if has_descale:
        attn_kwargs = {
            "q_descale": q_descale,
            "k_descale": k_descale,
            "v_descale": v_descale,
        }

    def fn():
        return aiter.flash_attn_fp8_pertensor_func(
            q,
            k,
            v,
            **attn_kwargs,
        )

    return fn
    # torch.cuda.synchronize()
    # return output


def run_aiter_flash_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, has_round_mode: bool = False
):
    # Note this will JIT compile on first invocation
    # [aiter] start build [module_fmha_v3_fwd] under /opt/aiter/aiter/jit/build/module_fmha_v3_fwd
    # Successfully preprocessed all matching files.
    # [aiter] finish build [module_fmha_v3_fwd], cost 53.76911977s
    # [aiter] type hints mismatch, override to --> fmha_v3_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float, softmax_scale: float, is_causal: bool, window_size_left: int, window_size_right: int, return_softmax_lse: bool, return_dropout_randval: bool, out: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None, alibi_slopes: Optional[torch.Tensor] = None, gen: Optional[torch.Generator] = None) -> list[torch.Tensor]

    if has_round_mode:

        def fn():
            return aiter.ops.mha.flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                causal=False,
                return_attn_probs=False,
                how_v3_bf16_cvt=2,
            )

    else:

        def fn():
            return aiter.ops.mha.flash_attn_func(
                q, k, v, dropout_p=0.0, causal=False, return_attn_probs=False
            )

    return fn


# taken from mha_v3.py
def fav3_fp8_forward_func(
    q: torch.Tensor,  # High precision (BF16/FP32)
    k: torch.Tensor,  # High precision (BF16/FP32)
    v: torch.Tensor,  # High precision (BF16/FP32)
    softmax_scale: Optional[float],
    causal: bool,
    window_size: Tuple[int, int],
    attention_chunk: int,
    softcap: float,
    sm_margin: int,
):
    batch, seqlen, num_q_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape

    # Quantize inputs to FP8
    fp8_dtype = aiter.dtypes.fp8
    # For GQA/MQA: quantize query with grouped scaling
    group_size = num_q_heads // num_kv_heads if num_q_heads != num_kv_heads else None
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
        raise NotImplementedError("softcap not implemented in FP8 high-precision API")
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


def fav2_forward_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: Optional[float],
    causal: bool,
    return_lse: bool,
    return_attn_probs: bool,
):
    return lambda: flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=return_lse,
        return_attn_probs=return_attn_probs,
    )


def fav3_sage_forward_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    inference_mode: bool,  # not return softmax_lse
    layout: Literal["bshd", "bhsd"],
    sage_version=fav3_sage.Sage_version.V1
):
    config = get_sage_fwd_configs(sage_version)
    BLKQ = config["BLOCK_M"]
    BLKK = config["BLOCK_N"]

    head_dim = q.shape[-1]
    softmax_scale = head_dim**-0.5

    fp8_dtype = aiter.dtypes.fp8
    FP8_MAX = torch.finfo(fp8_dtype).max

    sage_quant_func = sage_quant
    if sage_version == fav3_sage.Sage_version.V2:
        sage_quant_func = sage_quant_v2
    if sage_version == fav3_sage.Sage_version.V3:
        sage_quant_func = sage_quant_v3

    q_quantized, q_descale, k_quantized, k_descale, v_quantized, v_descale = sage_quant_func(
        q,
        k,
        v,
        fp8_dtype,
        FP8_MAX,
        sm_scale=softmax_scale,
        BLKQ=BLKQ,
        BLKK=BLKK,
        layout=layout,
    )

    return lambda: fav3_sage_func(
        q_quantized,
        k_quantized,
        v_quantized,
        q_descale,
        k_descale,
        v_descale,
        FP8_MAX,
        causal=causal,
        inference_mode=inference_mode,
        layout=layout,
        sage_version=sage_version
    )

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
        "arithmetic_intensity(FLOP/byte)",
    ]

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
                "arithmetic_intensity": "arithmetic_intensity(FLOP/byte)",
            }
            line_vals = [metric_map[args.metric]]

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


def create_benchmark_configs_from_captured(inputs: List[Dict[str, Any]], args):
    """
    Create triton.testing.Benchmark configurations from captured inputs.

    Captured inputs are in BHSD format (batch, heads, seqlen, dim).
    """
    # Extract x_vals from loaded inputs
    x_vals_list = []
    for i, inp in enumerate(inputs):
        # Shape from BSHD format: (batch, seqlen, heads, dim)
        x_vals_list.append((i))

    x_names = [
        "INPUT_IDX",
    ]

    # Determine line_vals based on metric
    if args.metric == "all" or args.metric is None:
        line_vals = [
            "time(ms)",
            "throughput(TFLOPS)",
            "bandwidth(GB/s)",
            "arithmetic_intensity(FLOP/byte)",
        ]
    else:
        metric_map = {
            "time": "time(ms)",
            "throughput": "throughput(TFLOPS)",
            "bandwidth": "bandwidth(GB/s)",
            "arithmetic_intensity": "arithmetic_intensity(FLOP/byte)",
        }
        line_vals = [metric_map.get(args.metric, "throughput(TFLOPS)")]

    plot_name = "bench_diffusion_attention_captured"

    configs = [
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            styles=[("red", "-"), ("green", "-"), ("yellow", "-"), ("blue", "-")],
            ylabel="",
            plot_name=plot_name,
            args={
                "inputs": inputs,
            },
        )
    ]
    return configs


def primary_output(result):
    """Return the main tensor output produced by a Triton kernel."""
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (list, tuple)) and len(result) > 0:
        return result[0]
    return result


def attn_forward_func(q, k, v, func_name, softmax_scale, k_smooth, layout, dtype, sage_version=fav3_sage.Sage_version.V1):
    if func_name == "fav3_sage":  # fav3 sage hybrid
        fn = fav3_sage_forward_func(
            q,
            k,
            v,
            causal=False,
            inference_mode=True,
            layout=layout,
            sage_version=sage_version
        )
    else:
        q, k, v = layout_preprocess(q, k, v, layout=layout, target_layout="bshd")
        if func_name == "aiter_bf16":
            fn = run_aiter_flash_attn(q, k, v)
        elif func_name == "aiter_fp8":
            fn = run_aiter_fp8_flash_attn(q, k, v, has_descale=True)
        elif func_name == "fav2":  # fav2 (no quantization)
            fn = fav2_forward_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=False,
                return_lse=False,
                return_attn_probs=False,
            )
        elif func_name == "fav3_fp8":  #  fav3 fp8
            fn = fav3_fp8_forward_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=False,
                window_size=(-1, -1),
                attention_chunk=0,
                softcap=0.0,
                sm_margin=0,
            )
        else:

            def fn():
                return attention_ref(
                    q, k, v, dropout_p=0.0, dropout_mask=None, causal=False
                )

    return fn


def bench_kernel(q, k, v, args, provider):
    # Default softmax scale
    if args.layout == "bshd":
        BATCH, N_CTX_Q, HQ, D_HEAD = q.shape
        _, N_CTX_K, HK, D_HEAD_V = v.shape
    else:  # bhsd
        BATCH, HQ, N_CTX_Q, D_HEAD = q.shape
        _, HK, N_CTX_K, D_HEAD_V = v.shape
    assert args.sage_version in [1, 2, 3], f"Unsupported sage version {args.sage_version}"

    sage_version = fav3_sage.Sage_version.V1
    if args.sage_version == 2:
        sage_version = fav3_sage.Sage_version.V2
    if args.sage_version == 3:
        sage_version = fav3_sage.Sage_version.V3

    softmax_scale = 1.0 / (D_HEAD**0.5)
    k_smooth = args.k_smooth

    # FLOPS calculation variables
    total_flops = 0.0
    total_flops += 2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * (D_HEAD + D_HEAD_V)
    bench_func_name = ""
    if args.fav3_fp8:
        bench_func_name = "fav3_fp8"
    elif args.aiter_fp8:
        bench_func_name = "aiter_fp8"
    elif args.aiter_bf16:
        bench_func_name = "aiter_bf16"
    else:
        bench_func_name = "fav3_sage"

    fn = attn_forward_func(
        q,
        k,
        v,
        func_name=bench_func_name,
        softmax_scale=softmax_scale,
        k_smooth=k_smooth,
        layout=args.layout,
        dtype=arg_to_torch_dtype[args.dtype],
        sage_version=sage_version
    )
    ms = triton.testing.do_bench(fn)

    if args.compare_to_ref:
        current_output = fn()
        assert current_output is not None
        current_primary = primary_output(current_output)

        if args.fav3_sage and args.layout == "bhsd":
            current_primary = current_primary.permute(
                0, 2, 1, 3
            )  # we do comparison in BSHD

        reference_primary = primary_output(
            attn_forward_func(
                q,
                k,
                v,
                func_name=args.ref,
                softmax_scale=softmax_scale,
                k_smooth=k_smooth,
                layout=args.layout,
                dtype=arg_to_torch_dtype[args.dtype],
            )()
        )

        if args.ref == "fav3_sage" and args.layout == "bhsd":
            reference_primary = reference_primary.permute(
                0, 2, 1, 3
            )  # we do comparison in BSHD
        
        compare_accuracy(current_primary, reference_primary)
        check_attention_outputs(current_primary, reference_primary, fp8=True)

    q_element_size = 1 if args.fav3_fp8 or args.fav3_sage else q.element_size()
    k_element_size = 1 if args.fav3_fp8 or args.fav3_sage else k.element_size()
    v_element_size = 1 if args.fav3_fp8 else v.element_size()

    total_num_tokens_q = BATCH * N_CTX_Q
    total_num_tokens_k = BATCH * N_CTX_K
    q_size = total_num_tokens_q * HQ * D_HEAD * q_element_size
    k_size = total_num_tokens_k * HK * D_HEAD * k_element_size
    v_size = total_num_tokens_k * HK * D_HEAD_V * v_element_size
    o_size = total_num_tokens_q * HQ * D_HEAD_V * q_element_size

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
    return ms


def run_benchmark_captured(args):
    """
    Run benchmark using captured inputs from disk.
    Captured inputs are in BHSD format and need to be transposed to BSHD for kernels.
    """
    torch.manual_seed(20)

    # Load captured inputs
    inputs = load_captured_inputs(args.captured_dir)
    # logger.info(f"Loaded {len(inputs)} captured inputs for benchmarking")

    @triton.testing.perf_report(create_benchmark_configs_from_captured(inputs, args))
    def bench_mha_captured(
        INPUT_IDX,
        inputs,
        provider,
        device="cuda",
    ):
        """
        Benchmark function for attention kernels using captured inputs.
        INPUT_IDX: Index in the loaded inputs list
        """
        # Get the input tensors for this configuration
        inp = inputs[INPUT_IDX]

        # Load tensors to GPU - captured inputs are in BHSD format (batch, heads, seq, dim). Permute it to the BSHD which bench kernel expects.
        # Permute shouldnt move data, so contiguousness of dimensions should stay intact.
        q = inp["q"].to(device)
        k = inp["k"].to(device)
        v = inp["v"].to(device)

        return bench_kernel(q, k, v, args, provider)

    args.layout = "bhsd"  # captured inputs are in BHSD format
    logger.info(
        "Captured inpputs are in BHSD format. Setting args.layout to bhsd for benchmark."
    )
    bench_mha_captured.run(save_path="." if args.o else None, print_data=True)


saved_output_keys = set()


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
        dropout=0.0,
        device="cuda",
    ):
        """
        Benchmark function for attention kernels with generated random inputs.
        """
        assert dropout <= 0.0, "Dropout not supported in this benchmark."
        assert not causal, "Causal not supported in this benchmark."

        # Generate base inputs in BHSD layout which is the layout used in wan model.
        q = torch.randn((BATCH, HQ, N_CTX_Q, D_HEAD), device=device, dtype=dtype)
        k = torch.randn((BATCH, HK, N_CTX_K, D_HEAD), device=device, dtype=dtype)
        v = torch.randn((BATCH, HK, N_CTX_K, D_HEAD_V), device=device, dtype=dtype)
        q.requires_grad = False
        k.requires_grad = False
        v.requires_grad = False

        # permute to the BSHD layout which is expected by the bench_kernel
        # permute does not move data so this doesnt affect the performance
        q, k, v = layout_preprocess(q, k, v, layout="bhsd", target_layout=layout)

        return bench_kernel(q, k, v, args, provider)

    bench_mha.run(save_path="." if args.o else None, print_data=True)


def supported_layouts():
    layouts = (
        "bshd: Q, K, V are individual tensors of [batch, seqlen_q/k, num_heads, head_size]. "
        "bhsd: Q, K, V are individual tensors of [batch, num_heads, seqlen_q/k, head_size]. "
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
    parser.add_argument("--sage_version", type=int, default=1)
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-sk", type=int, default=0)
    parser.add_argument(
        "-d",
        type=int,
        default=0,
        help="Q and K head size, if -dv is absent then -d specifies V head size too",
    )
    parser.add_argument("-dv", type=int, default=0, help="optional V head size")
    parser.add_argument(
        "-fav3_fp8",
        action="store_true",
        default=False,
        help="Use fav3 fp8 kernel (instead of default fav3_sage): per tensor quantization, QK and PV in fp8, accumulation in fp32",
    )
    parser.add_argument(
        "-aiter_fp8",
        action="store_true",
        default=False,
        help="Use ck tile fmhav2 fp8 kernel (instead of default fav3_sage)",
    )
    parser.add_argument(
        "-aiter_bf16",
        action="store_true",
        default=False,
        help="Use asm fav3 bf16 kernel (instead of default fav3_sage)",
    )
    parser.add_argument("-k_smooth", action="store_true", default=True)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument("--layout", type=str, default="bshd", help=supported_layouts())
    parser.add_argument(
        "-ref",
        type=str,
        default=None,
        help="fp8, qk_int8, fav2 or torch ref (default).",
    )
    parser.add_argument(
        "-metric",
        nargs="?",
        const="throughput",
        choices=["all", "time", "throughput", "bandwidth", "arithmetic_intensity"],
        default=None,
        help="Metrics for the kernel benchmark.",
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
        "--compare_to_ref",
        action="store_true",
        help="also execute the reference kernel (-ref) and assert outputs match.",
    )
    # Captured input loading
    parser.add_argument(
        "--load_captured",
        action="store_true",
        help="Load captured inputs from disk instead of generating random tensors",
    )
    parser.add_argument(
        "--captured_dir",
        type=str,
        default="./captured_inputs",
        help="Directory containing captured input .pt files",
    )
    return parser.parse_args()


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def main():
    args = parse_args()

    args.fav3_sage = not args.fav3_fp8 and not args.aiter_fp8 and not args.aiter_bf16

    # Handle captured input mode separately
    if args.load_captured:
        logger.info(f"Running benchmark with captured inputs from: {args.captured_dir}")
        run_benchmark_captured(args)
        return 0

    if not args.dv:
        args.dv = args.d
    assert (
        args.b and args.hq and args.sq and args.d and args.dv
    ), "If not running on captured (--load_captured) please provide \
            all of batch, number of Q heads, Q sequence length \
            and head size."

    assert (
        args.dtype in arg_to_torch_dtype
    ), "Only fp16, bf16 and f32 types currently supported."

    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")

        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0

    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
