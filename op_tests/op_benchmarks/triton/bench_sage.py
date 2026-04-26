from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import argparse
import csv
import glob
import json
import logging
import os
import sys

import torch
import triton

import aiter
from aiter.ops.triton.mha import flash_attn_func
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3
from aiter.ops.triton.attention.mha_v3 import _quantize_bshd
from aiter.ops.triton.attention.fav3_sage import (
    fav3_sage_wrapper_func,
    get_sage_fwd_configs,
)
from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
    fav3_sage_mxfp4_func,
    fav3_sage_mxfp4_wrapper,
    get_sage_fwd_configs_mxfp4,
)
from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    create_hadamard_matrix,
    sage_quant_mxfp4,
)
from aiter.test_mha_common import attention_ref, attention_ref_block_sparse

from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.attention.test_fav3_sage import (
    check_attention_outputs,
    compare_accuracy,
)


logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


KernelName = Literal[
    "sage",
    "sage_mxfp4",
    "fav3_fp8",
    "aiter_fp8",
    "aiter_bf16",
]


@dataclass
class ShapeSpec:
    batch: int
    hq: int
    hk: int
    n_ctx_q: int
    n_ctx_k: int
    d_head: int
    d_head_v: int


@dataclass
class LoadedMask:
    mask: torch.Tensor
    batch: int
    num_q_blocks: int
    num_kv_blocks: int


def layout_preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: Literal["bshd", "bhsd"],
    target_layout: Literal["bshd", "bhsd"] = "bshd",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if layout != target_layout:
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    return q, k, v


def primary_output(result: Any) -> Any:
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (tuple, list)) and len(result) > 0:
        return result[0]
    return result


def infer_shape_spec(
    q: torch.Tensor,
    v: torch.Tensor,
    layout: Literal["bshd", "bhsd"],
) -> ShapeSpec:
    if layout == "bshd":
        batch, n_ctx_q, hq, d_head = q.shape
        _, n_ctx_k, hk, d_head_v = v.shape
    else:
        batch, hq, n_ctx_q, d_head = q.shape
        _, hk, n_ctx_k, d_head_v = v.shape
    return ShapeSpec(
        batch=batch,
        hq=hq,
        hk=hk,
        n_ctx_q=n_ctx_q,
        n_ctx_k=n_ctx_k,
        d_head=d_head,
        d_head_v=d_head_v,
    )


def _array_ndim(arr: Any) -> int:
    if not isinstance(arr, list):
        return 0
    if not arr:
        return 1
    return 1 + _array_ndim(arr[0])


def _mask_array_to_tensor(
    mask_arr: List[Any],
    device: torch.device,
) -> LoadedMask:
    if not mask_arr:
        raise ValueError("mask array is empty")

    depth = _array_ndim(mask_arr)
    if depth == 2:
        mask = torch.tensor(mask_arr, dtype=torch.bool, device=device)
        num_q_blocks, num_kv_blocks = mask.shape
        mask = mask.unsqueeze(0)
        return LoadedMask(mask, 1, num_q_blocks, num_kv_blocks)

    if depth == 3:
        mask = torch.tensor(mask_arr, dtype=torch.bool, device=device)
        batch, num_q_blocks, num_kv_blocks = mask.shape
        return LoadedMask(mask, batch, num_q_blocks, num_kv_blocks)

    raise ValueError(f"mask must be 2D or 3D, got {depth}D")


def load_block_mask_from_json(
    path: Optional[str],
    device: torch.device,
) -> Optional[Union[LoadedMask, List[LoadedMask]]]:
    if not path or not path.strip():
        return None

    path = path.strip()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Block mask file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if not data:
        return None

    if "masks" in data:
        loaded = []
        for item in data["masks"]:
            if "mask" not in item:
                raise ValueError("Each element in 'masks' must include key 'mask'")
            m = _mask_array_to_tensor(item["mask"], device)
            if "num_q_blocks" in item and item["num_q_blocks"] != m.num_q_blocks:
                raise ValueError(
                    f"num_q_blocks mismatch: inferred {m.num_q_blocks}, got {item['num_q_blocks']}"
                )
            if (
                "num_kv_blocks" in item
                and item["num_kv_blocks"] != m.num_kv_blocks
            ):
                raise ValueError(
                    f"num_kv_blocks mismatch: inferred {m.num_kv_blocks}, got {item['num_kv_blocks']}"
                )
            loaded.append(m)
        return loaded

    if "mask" in data:
        m = _mask_array_to_tensor(data["mask"], device)
        if "num_q_blocks" in data and data["num_q_blocks"] != m.num_q_blocks:
            raise ValueError(
                f"num_q_blocks mismatch: inferred {m.num_q_blocks}, got {data['num_q_blocks']}"
            )
        if "num_kv_blocks" in data and data["num_kv_blocks"] != m.num_kv_blocks:
            raise ValueError(
                f"num_kv_blocks mismatch: inferred {m.num_kv_blocks}, got {data['num_kv_blocks']}"
            )
        return m

    return None


def kernel_block_sizes(kernel: KernelName) -> Tuple[int, int]:
    if kernel == "sage_mxfp4":
        cfg = get_sage_fwd_configs_mxfp4()
    else:
        cfg = get_sage_fwd_configs()
    return cfg["BLOCK_M"], cfg["BLOCK_N"]


def maybe_expand_mask(
    mask: LoadedMask,
    batch: int,
    hq: int,
) -> torch.Tensor:
    out = mask.mask
    if mask.batch != batch:
        if mask.batch == 1:
            out = out.expand(batch, -1, -1).clone()
        else:
            raise ValueError(
                f"Mask batch ({mask.batch}) does not match benchmark batch ({batch})"
            )

    if out.dim() == 3:
        out = out.unsqueeze(1).expand(batch, hq, mask.num_q_blocks, mask.num_kv_blocks)
    return out.clone()


def build_block_mask(
    args: argparse.Namespace,
    shape: ShapeSpec,
    device: torch.device,
    loaded_single_mask: Optional[LoadedMask],
) -> Optional[torch.Tensor]:
    if loaded_single_mask is not None:
        block_m, block_n = kernel_block_sizes(args.kernel)
        expected_q_blocks = (shape.n_ctx_q + block_m - 1) // block_m
        expected_kv_blocks = (shape.n_ctx_k + block_n - 1) // block_n

        if loaded_single_mask.num_q_blocks != expected_q_blocks:
            raise ValueError(
                f"Mask q blocks mismatch: expected {expected_q_blocks}, got {loaded_single_mask.num_q_blocks}"
            )
        if loaded_single_mask.num_kv_blocks != expected_kv_blocks:
            raise ValueError(
                f"Mask kv blocks mismatch: expected {expected_kv_blocks}, got {loaded_single_mask.num_kv_blocks}"
            )

        return maybe_expand_mask(loaded_single_mask, shape.batch, shape.hq)

    if args.block_sparsity is None:
        return None

    block_m, block_n = kernel_block_sizes(args.kernel)
    num_q_blocks = (shape.n_ctx_q + block_m - 1) // block_m
    num_kv_blocks = (shape.n_ctx_k + block_n - 1) // block_n

    return (
        torch.rand(
            shape.batch,
            shape.hq,
            num_q_blocks,
            num_kv_blocks,
            device=device,
        )
        > args.block_sparsity
    ).to(torch.bool)


def sparse_flops_from_lut(
    kernel: KernelName,
    block_lut: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    shape: ShapeSpec,
) -> Tuple[float, float]:
    _, _, lut_count = block_lut
    num_sparse_pairs = lut_count.sum().item()

    block_m, block_n = kernel_block_sizes(kernel)
    num_q_blocks = (shape.n_ctx_q + block_m - 1) // block_m
    num_kv_blocks = (shape.n_ctx_k + block_n - 1) // block_n
    num_dense_pairs = shape.batch * shape.hq * num_q_blocks * num_kv_blocks

    total_dense_flops = (
        2.0
        * shape.batch
        * shape.hq
        * shape.n_ctx_q
        * shape.n_ctx_k
        * (shape.d_head + shape.d_head_v)
    )

    if num_dense_pairs == 0:
        return 0.0, total_dense_flops

    sparse_flops = total_dense_flops * (num_sparse_pairs / num_dense_pairs)
    return sparse_flops, total_dense_flops


def fp8_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    quant_dtype = aiter.dtypes.fp8
    q_quant, q_descale = aiter.per_tensor_quant(
        q,
        scale=torch.abs(q).max() if scale is None else scale,
        quant_dtype=quant_dtype,
        dtypeMax=torch.finfo(quant_dtype).max,
    )
    k_quant, k_descale = aiter.per_tensor_quant(
        k,
        scale=torch.abs(k).max() if scale is None else scale,
        quant_dtype=quant_dtype,
        dtypeMax=torch.finfo(quant_dtype).max,
    )
    v_quant, v_descale = aiter.per_tensor_quant(
        v,
        scale=torch.abs(v).max() if scale is None else scale,
        quant_dtype=quant_dtype,
        dtypeMax=torch.finfo(quant_dtype).max,
    )
    return q_quant, k_quant, v_quant, q_descale, k_descale, v_descale


def make_fav3_fp8_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float],
    causal: bool,
) -> Any:
    batch, _, num_q_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape

    fp8_dtype = aiter.dtypes.fp8
    group_size = num_q_heads // num_kv_heads if num_q_heads != num_kv_heads else None

    q_fp8, q_descale = _quantize_bshd(q, fp8_dtype, group_size=group_size)
    k_fp8, k_descale = _quantize_bshd(k, fp8_dtype)
    v_fp8, v_descale = _quantize_bshd(v, fp8_dtype)

    assert q_descale.shape == (batch, num_kv_heads)
    assert k_descale.shape == (batch, num_kv_heads)
    assert v_descale.shape == (batch, num_kv_heads)

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    return lambda: flash_attn_3.fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        -1,
        -1,
        0,
        0.0,
        False,
        None,
        1,
        None,
        0,
    )


def make_fav2_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float],
    causal: bool,
) -> Any:
    return lambda: flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=False,
        return_attn_probs=False,
    )


def make_torch_ref_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> Any:
    return lambda: attention_ref(q, k, v, dropout_p=0.0, dropout_mask=None, causal=causal)


def make_kernel_runner(
    args: argparse.Namespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Any:
    q_bshd, k_bshd, v_bshd = layout_preprocess(q, k, v, layout=args.layout, target_layout="bshd")
    head_dim = q_bshd.shape[-1]
    softmax_scale = head_dim**-0.5

    if args.kernel == "sage":
        return lambda: fav3_sage_wrapper_func(
            q,
            k,
            v,
            softmax_scale,
            causal=args.causal,
            return_lse=False,
            layout=args.layout,
            block_lut=block_lut,
        )

    if args.kernel == "sage_mxfp4":
        block_r = args.block_r
        if block_r > q.shape[-1]:
            raise ValueError(f"block_r ({block_r}) must be <= head dim ({q.shape[-1]})")

        r = create_hadamard_matrix(block_r, device=q.device, dtype=q.dtype) / (block_r**0.5)

        if args.include_quant_overhead:
            return lambda: fav3_sage_mxfp4_wrapper(
                q,
                k,
                v,
                causal=args.causal,
                layout=args.layout,
                q_smooth=args.qsmooth,
                hadamard_rotation=args.hadamard_rotate,
                R=r,
                block_lut=block_lut,
            )

        cfg = get_sage_fwd_configs_mxfp4()
        fp8_type = aiter.dtypes.fp8
        fp8_max = torch.finfo(fp8_type).max

        (
            q_quant,
            q_descale,
            k_quant,
            k_descale,
            v_quant,
            v_descale,
            delta_s,
        ) = sage_quant_mxfp4(
            q,
            k,
            v,
            fp8_type,
            fp8_max,
            BLKQ=cfg["BLOCK_M"],
            BLKK=64,
            layout=args.layout,
            R=r,
            BLOCK_R=block_r,
            q_smoothing=args.qsmooth,
        )

        if block_lut is not None:
            kv_block_indices, lut_start, lut_count = block_lut
            use_block_sparse = True
        else:
            kv_block_indices = lut_start = lut_count = None
            use_block_sparse = False

        return lambda: fav3_sage_mxfp4_func(
            q=q_quant,
            k=k_quant,
            v=v_quant,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            bias=delta_s,
            causal=args.causal,
            layout=args.layout,
            config=cfg,
            kv_block_indices=kv_block_indices,
            lut_start=lut_start,
            lut_count=lut_count,
            use_block_sparse=use_block_sparse,
        )

    if args.kernel == "aiter_bf16":
        return lambda: aiter.ops.mha.flash_attn_func(
            q_bshd,
            k_bshd,
            v_bshd,
            dropout_p=0.0,
            causal=args.causal,
            return_attn_probs=False,
        )

    if args.kernel == "aiter_fp8":
        q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale = fp8_quantize(q_bshd, k_bshd, v_bshd)

        return lambda: aiter.flash_attn_fp8_pertensor_func(
            q_fp8,
            k_fp8,
            v_fp8,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    if args.kernel == "fav3_fp8":
        return make_fav3_fp8_runner(
            q_bshd,
            k_bshd,
            v_bshd,
            softmax_scale=softmax_scale,
            causal=args.causal,
        )

    raise ValueError(f"Unsupported kernel: {args.kernel}")


def to_bshd_output_if_needed(
    out: torch.Tensor,
    layout: Literal["bshd", "bhsd"],
) -> torch.Tensor:
    if layout == "bhsd":
        return out.permute(0, 2, 1, 3).contiguous()
    return out


def make_reference_output(
    args: argparse.Namespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_attn_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    q_bshd, k_bshd, v_bshd = layout_preprocess(q, k, v, layout=args.layout, target_layout="bshd")
    softmax_scale = q_bshd.shape[-1] ** -0.5

    ref = args.ref or "torch"

    if block_attn_mask is not None:
        if ref != "torch":
            raise ValueError("Block sparse comparison currently supports --ref=torch only")
        block_m, block_n = kernel_block_sizes(args.kernel)
        ref_out = attention_ref_block_sparse(
            q_bshd,
            k_bshd,
            v_bshd,
            block_attn_mask,
            block_m,
            block_n,
            dropout_p=0.0,
            dropout_mask=None,
            upcast=True,
        )
        return primary_output(ref_out)

    if ref == "fav2":
        return primary_output(make_fav2_runner(q_bshd, k_bshd, v_bshd, softmax_scale, args.causal)())

    if ref == "sage":
        return primary_output(
            fav3_sage_wrapper_func(
                q_bshd,
                k_bshd,
                v_bshd,
                softmax_scale,
                causal=args.causal,
                return_lse=False,
                layout="bshd",
                block_lut=None,
            )
        )

    return primary_output(make_torch_ref_runner(q_bshd, k_bshd, v_bshd, args.causal)())


def compute_memory_bytes(
    shape: ShapeSpec,
    q_element_size: int,
    k_element_size: int,
    v_element_size: int,
) -> float:
    total_num_tokens_q = shape.batch * shape.n_ctx_q
    total_num_tokens_k = shape.batch * shape.n_ctx_k

    q_size = total_num_tokens_q * shape.hq * shape.d_head * q_element_size
    k_size = total_num_tokens_k * shape.hk * shape.d_head * k_element_size
    v_size = total_num_tokens_k * shape.hk * shape.d_head_v * v_element_size
    o_size = total_num_tokens_q * shape.hq * shape.d_head_v * q_element_size
    return q_size + k_size + v_size + o_size


def benchmark_single_case(
    args: argparse.Namespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    provider: str,
    loaded_single_mask: Optional[LoadedMask],
    explicit_block_attn_mask: Optional[torch.Tensor] = None,
) -> float:
    shape = infer_shape_spec(q, v, args.layout)
    block_attn_mask = (
        explicit_block_attn_mask
        if explicit_block_attn_mask is not None
        else build_block_mask(args, shape, q.device, loaded_single_mask)
    )
    block_lut = (
        block_attn_mask_to_ragged_lut(block_attn_mask, return_none_if_dense=True)
        if block_attn_mask is not None
        else None
    )

    fn = make_kernel_runner(args, q, k, v, block_lut=block_lut)
    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)

    if args.compare_to_ref:
        current_primary = primary_output(fn())
        current_primary = to_bshd_output_if_needed(current_primary, args.layout)
        ref_primary = make_reference_output(args, q, k, v, block_attn_mask)
        compare_accuracy(current_primary, ref_primary)
        check_attention_outputs(current_primary, ref_primary, fp8=False)

    total_flops = (
        2.0
        * shape.batch
        * shape.hq
        * shape.n_ctx_q
        * shape.n_ctx_k
        * (shape.d_head + shape.d_head_v)
    )

    if args.kernel in ("fav3_fp8", "aiter_fp8", "sage", "sage_mxfp4"):
        q_elem_size = 1
        k_elem_size = 1
    else:
        q_elem_size = q.element_size()
        k_elem_size = k.element_size()

    v_elem_size = 1 if args.kernel in ("fav3_fp8", "aiter_fp8") else v.element_size()
    mem = compute_memory_bytes(shape, q_elem_size, k_elem_size, v_elem_size)

    sparse_flops = None
    if block_lut is not None:
        sparse_flops, _ = sparse_flops_from_lut(args.kernel, block_lut, shape)

    if "time(ms)" in provider:
        return ms
    if "throughput_sparse(TFLOPS)" in provider:
        flops = sparse_flops if sparse_flops is not None else total_flops
        return flops / ms * 1e-9
    if "throughput(TFLOPS)" in provider:
        return total_flops / ms * 1e-9
    if "bandwidth(GB/s)" in provider:
        return mem / ms * 1e-6
    if "arithmetic_intensity(FLOP/byte)" in provider:
        return total_flops / mem
    return ms


def metric_lines(args: argparse.Namespace, include_sparse_metric: bool) -> List[str]:
    all_metrics = [
        "time(ms)",
        "throughput(TFLOPS)",
        "bandwidth(GB/s)",
        "arithmetic_intensity(FLOP/byte)",
    ]
    if include_sparse_metric:
        all_metrics.append("throughput_sparse(TFLOPS)")

    if args.compare_to_ref:
        return ["time(ms)"]

    if args.metric == "all":
        return all_metrics

    metric_map = {
        "time": "time(ms)",
        "throughput": "throughput(TFLOPS)",
        "bandwidth": "bandwidth(GB/s)",
        "arithmetic_intensity": "arithmetic_intensity(FLOP/byte)",
        "throughput_sparse": "throughput_sparse(TFLOPS)",
    }

    if args.metric == "throughput_sparse" and not include_sparse_metric:
        raise ValueError("throughput_sparse requires --block-sparsity or --block-mask-file")

    return [metric_map[args.metric]]


def make_styles(num_lines: int) -> List[Tuple[str, str]]:
    palette = ["red", "green", "yellow", "blue", "cyan", "magenta"]
    return [(palette[i % len(palette)], "-") for i in range(num_lines)]


def create_single_shape_config(args: argparse.Namespace) -> List[Any]:
    hk = args.hk if args.hk else args.hq
    sk = args.sk if args.sk else args.sq
    d_head = args.d if args.d else 128
    d_head_v = args.dv if args.dv else d_head

    include_sparse_metric = (
        args.block_sparsity is not None or args.block_mask_file is not None
    )
    lines = metric_lines(args, include_sparse_metric)

    return [
        triton.testing.Benchmark(
            x_names=["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K"],
            x_vals=[(args.b, args.hq, hk, args.sq, sk)],
            line_arg="provider",
            line_vals=lines,
            line_names=lines,
            styles=make_styles(len(lines)),
            ylabel="",
            plot_name=get_caller_name_no_ext(),
            args={
                "D_HEAD": d_head,
                "D_HEAD_V": d_head_v,
                "dtype": arg_to_torch_dtype[args.dtype],
                "layout": args.layout,
                "causal": args.causal,
            },
        )
    ]


def create_captured_config(
    args: argparse.Namespace,
    inputs: List[Dict[str, Any]],
) -> List[Any]:
    include_sparse_metric = (
        args.block_sparsity is not None or args.block_mask_file is not None
    )
    lines = metric_lines(args, include_sparse_metric)

    return [
        triton.testing.Benchmark(
            x_names=["INPUT_IDX"],
            x_vals=[(i,) for i in range(len(inputs))],
            line_arg="provider",
            line_vals=lines,
            line_names=lines,
            styles=make_styles(len(lines)),
            ylabel="",
            plot_name="bench_sage_captured",
            args={"inputs": inputs},
        )
    ]


def create_mask_list_config(
    args: argparse.Namespace,
    masks: List[LoadedMask],
) -> List[Any]:
    lines = metric_lines(args, include_sparse_metric=True)
    hk = args.hk if args.hk else args.hq

    return [
        triton.testing.Benchmark(
            x_names=["MASK_IDX"],
            x_vals=[(i,) for i in range(len(masks))],
            line_arg="provider",
            line_vals=lines,
            line_names=lines,
            styles=make_styles(len(lines)),
            ylabel="",
            plot_name=get_caller_name_no_ext() + "_masks",
            args={
                "masks": masks,
                "D_HEAD": args.d,
                "D_HEAD_V": args.dv,
                "dtype": arg_to_torch_dtype[args.dtype],
                "layout": args.layout,
                "causal": args.causal,
                "args": args,
                "HQ": args.hq,
                "HK": hk,
            },
        )
    ]


def load_captured_inputs(input_dir: str) -> List[Dict[str, Any]]:
    input_files = sorted(glob.glob(os.path.join(input_dir, "*_input_*.pt")))
    if not input_files:
        raise FileNotFoundError(f"No captured input files found in {input_dir}")

    inputs = []
    for file_path in input_files:
        inputs.append(torch.load(file_path, weights_only=False))

    logger.info("Loaded %d captured inputs", len(inputs))
    return inputs


def validate_args(args: argparse.Namespace) -> None:
    if not args.load_captured:
        required = [args.b, args.hq, args.sq, args.d]
        if any(v <= 0 for v in required):
            raise ValueError("For generated inputs provide positive --b --hq --sq --d")

    if args.dv <= 0:
        args.dv = args.d
    if args.hk <= 0:
        args.hk = args.hq
    if args.sk <= 0:
        args.sk = args.sq

    if args.block_sparsity is not None and not (0.0 <= args.block_sparsity <= 1.0):
        raise ValueError(f"--block-sparsity must be in [0,1], got {args.block_sparsity}")

    if args.block_sparsity is not None and args.block_mask_file:
        logger.info("Using --block-mask-file; ignoring --block-sparsity")

    if args.compare_to_ref and args.ref not in ("torch", "fav2", "sage"):
        raise ValueError("--ref must be one of: torch, fav2, sage")

    if args.kernel != "sage_mxfp4" and (
        args.include_quant_overhead or args.qsmooth or args.hadamard_rotate is False
    ):
        logger.warning(
            "MXFP4-specific flags are ignored unless --kernel=sage_mxfp4"
        )


def run_benchmark_generated(
    args: argparse.Namespace,
    loaded_single_mask: Optional[LoadedMask],
) -> None:
    @triton.testing.perf_report(create_single_shape_config(args))
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

        q.requires_grad = False
        k.requires_grad = False
        v.requires_grad = False

        q, k, v = layout_preprocess(q, k, v, layout="bhsd", target_layout=layout)

        return benchmark_single_case(
            args,
            q,
            k,
            v,
            provider,
            loaded_single_mask=loaded_single_mask,
        )

    bench_mha.run(save_path="." if args.o else None, print_data=True)


def run_benchmark_captured(
    args: argparse.Namespace,
    loaded_single_mask: Optional[LoadedMask],
) -> None:
    inputs = load_captured_inputs(args.captured_dir)

    @triton.testing.perf_report(create_captured_config(args, inputs))
    def bench_mha_captured(INPUT_IDX, inputs, provider, device="cuda"):
        inp = inputs[INPUT_IDX]
        q = inp["q"].to(device)
        k = inp["k"].to(device)
        v = inp["v"].to(device)

        return benchmark_single_case(
            args,
            q,
            k,
            v,
            provider,
            loaded_single_mask=loaded_single_mask,
        )

    bench_mha_captured.run(save_path="." if args.o else None, print_data=True)


def run_benchmark_mask_list(args: argparse.Namespace, masks: List[LoadedMask]) -> None:
    block_m, block_n = kernel_block_sizes(args.kernel)

    @triton.testing.perf_report(create_mask_list_config(args, masks))
    def bench_mha_masks(
        MASK_IDX,
        masks,
        D_HEAD,
        D_HEAD_V,
        dtype,
        layout,
        causal,
        args,
        HQ,
        HK,
        provider,
        device="cuda",
    ):
        loaded = masks[MASK_IDX]
        mask = maybe_expand_mask(loaded, loaded.batch, HQ)

        n_ctx_q = loaded.num_q_blocks * block_m
        n_ctx_k = loaded.num_kv_blocks * block_n

        q = torch.randn((loaded.batch, HQ, n_ctx_q, D_HEAD), device=device, dtype=dtype)
        k = torch.randn((loaded.batch, HK, n_ctx_k, D_HEAD), device=device, dtype=dtype)
        v = torch.randn((loaded.batch, HK, n_ctx_k, D_HEAD_V), device=device, dtype=dtype)
        q.requires_grad = False
        k.requires_grad = False
        v.requires_grad = False

        q, k, v = layout_preprocess(q, k, v, layout="bhsd", target_layout=layout)
        return benchmark_single_case(
            args,
            q,
            k,
            v,
            provider,
            loaded_single_mask=None,
            explicit_block_attn_mask=mask,
        )

    bench_mha_masks.run(save_path="." if args.o else None, print_data=True)


def run_block_sparse_repetitions(
    args: argparse.Namespace,
    loaded_single_mask: Optional[LoadedMask],
) -> None:
    if loaded_single_mask is not None:
        raise ValueError("--n-repetitions is only supported with random --block-sparsity")

    if args.load_captured:
        raise ValueError("--n-repetitions is supported only with generated random inputs")

    dtype = arg_to_torch_dtype[args.dtype]
    device = "cuda"

    q = torch.randn((args.b, args.hq, args.sq, args.d), device=device, dtype=dtype)
    k = torch.randn((args.b, args.hk, args.sk, args.d), device=device, dtype=dtype)
    v = torch.randn((args.b, args.hk, args.sk, args.dv), device=device, dtype=dtype)
    q.requires_grad = False
    k.requires_grad = False
    v.requires_grad = False
    q, k, v = layout_preprocess(q, k, v, layout="bhsd", target_layout=args.layout)

    shape = infer_shape_spec(q, v, args.layout)
    block_m, block_n = kernel_block_sizes(args.kernel)
    num_q_blocks = (shape.n_ctx_q + block_m - 1) // block_m
    num_kv_blocks = (shape.n_ctx_k + block_n - 1) // block_n

    warmup_mask = (
        torch.rand(shape.batch, shape.hq, num_q_blocks, num_kv_blocks, device=device)
        > args.block_sparsity
    ).to(torch.bool)
    warmup_lut = block_attn_mask_to_ragged_lut(warmup_mask, return_none_if_dense=True)
    fn_warmup = make_kernel_runner(args, q, k, v, block_lut=warmup_lut)
    triton.testing.do_bench(fn_warmup, warmup=args.warmup, rep=args.rep)

    total_flops = (
        2.0
        * shape.batch
        * shape.hq
        * shape.n_ctx_q
        * shape.n_ctx_k
        * (shape.d_head + shape.d_head_v)
    )

    latencies_ms: List[float] = []
    tflops_dense: List[float] = []
    tflops_effective: List[float] = []

    for _ in range(args.n_repetitions):
        mask = (
            torch.rand(shape.batch, shape.hq, num_q_blocks, num_kv_blocks, device=device)
            > args.block_sparsity
        ).to(torch.bool)
        lut = block_attn_mask_to_ragged_lut(mask, return_none_if_dense=True)

        fn = make_kernel_runner(args, q, k, v, block_lut=lut)
        ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
        latencies_ms.append(ms)

        dense_tflops = (total_flops / (ms * 1e-3)) / 1e12
        tflops_dense.append(dense_tflops)

        sparse_flops, _ = sparse_flops_from_lut(args.kernel, lut, shape)
        effective_tflops = (sparse_flops / (ms * 1e-3)) / 1e12
        tflops_effective.append(effective_tflops)

    def stats(x: List[float]) -> Dict[str, float]:
        t = torch.tensor(x)
        return {
            "median": torch.quantile(t, 0.5).item(),
            "q1": torch.quantile(t, 0.25).item(),
            "q3": torch.quantile(t, 0.75).item(),
            "p10": torch.quantile(t, 0.1).item(),
            "p90": torch.quantile(t, 0.9).item(),
        }

    st_dense = stats(tflops_dense)
    st_lat = stats(latencies_ms)
    st_eff = stats(tflops_effective)

    summary = (
        f"kernel={args.kernel}, block_sparsity={args.block_sparsity}, n_repetitions={args.n_repetitions}: "
        f"median_TFLOPS={st_dense['median']:.4f}, Q1={st_dense['q1']:.4f}, Q3={st_dense['q3']:.4f}, "
        f"p10={st_dense['p10']:.4f}, p90={st_dense['p90']:.4f} | "
        f"median_latency_ms={st_lat['median']:.4f}, Q1={st_lat['q1']:.4f}, Q3={st_lat['q3']:.4f}, "
        f"p10={st_lat['p10']:.4f}, p90={st_lat['p90']:.4f} | "
        f"median_effective_TFLOPS={st_eff['median']:.4f}, Q1={st_eff['q1']:.4f}, "
        f"Q3={st_eff['q3']:.4f}, p10={st_eff['p10']:.4f}, p90={st_eff['p90']:.4f}"
    )
    logger.info(summary)
    print(summary)

    if args.o:
        csv_path = "bench_sage_block_sparse_repetitions.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "kernel",
                        "BATCH",
                        "HQ",
                        "N_CTX_Q",
                        "N_CTX_K",
                        "D_HEAD",
                        "D_HEAD_V",
                        "block_sparsity",
                        "n_repetitions",
                        "median_TFLOPS",
                        "q1_TFLOPS",
                        "q3_TFLOPS",
                        "p10_TFLOPS",
                        "p90_TFLOPS",
                        "median_latency_ms",
                        "q1_latency_ms",
                        "q3_latency_ms",
                        "p10_latency_ms",
                        "p90_latency_ms",
                        "median_effective_TFLOPS",
                        "q1_effective_TFLOPS",
                        "q3_effective_TFLOPS",
                        "p10_effective_TFLOPS",
                        "p90_effective_TFLOPS",
                    ]
                )
            writer.writerow(
                [
                    args.kernel,
                    shape.batch,
                    shape.hq,
                    shape.n_ctx_q,
                    shape.n_ctx_k,
                    shape.d_head,
                    shape.d_head_v,
                    args.block_sparsity,
                    args.n_repetitions,
                    st_dense["median"],
                    st_dense["q1"],
                    st_dense["q3"],
                    st_dense["p10"],
                    st_dense["p90"],
                    st_lat["median"],
                    st_lat["q1"],
                    st_lat["q3"],
                    st_lat["p10"],
                    st_lat["p90"],
                    st_eff["median"],
                    st_eff["q1"],
                    st_eff["q3"],
                    st_eff["p10"],
                    st_eff["p90"],
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified SAGE attention benchmark (FAv3, MXFP4, AITER, FP8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--kernel",
        type=str,
        default="sage",
        choices=["sage", "sage_mxfp4", "fav3_fp8", "aiter_fp8", "aiter_bf16"],
        help="Kernel implementation to benchmark.",
    )

    parser.add_argument("--b", type=int, default=0, help="Batch size")
    parser.add_argument("--hq", type=int, default=0, help="Number of Q heads")
    parser.add_argument("--hk", type=int, default=0, help="Number of KV heads")
    parser.add_argument("--sq", type=int, default=0, help="Query sequence length")
    parser.add_argument("--sk", type=int, default=0, help="KV sequence length")
    parser.add_argument("--d", type=int, default=0, help="Q/K head dimension")
    parser.add_argument("--dv", type=int, default=0, help="V head dimension")

    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--layout", type=str, default="bshd", choices=["bshd", "bhsd"])
    parser.add_argument("--causal", action="store_true", help="Enable causal attention")

    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=[
            "all",
            "time",
            "throughput",
            "bandwidth",
            "arithmetic_intensity",
            "throughput_sparse",
        ],
        help="Metric(s) to report",
    )

    parser.add_argument("-o", action="store_true", help="Write Triton output CSV")
    parser.add_argument("--print-vgpr", action="store_true", help="Print kernel VGPR usage")

    parser.add_argument("--compare-to-ref", action="store_true", help="Compare against reference")
    parser.add_argument(
        "--ref",
        type=str,
        default="torch",
        choices=["torch", "fav2", "sage"],
        help="Reference kernel for --compare-to-ref",
    )

    parser.add_argument(
        "--load-captured",
        action="store_true",
        help="Use captured tensors from disk instead of random generation",
    )
    parser.add_argument(
        "--captured-dir",
        type=str,
        default="./captured_inputs",
        help="Directory containing *_input_*.pt files",
    )

    parser.add_argument(
        "--block-sparsity",
        type=float,
        default=None,
        help="Random block sparsity ratio in [0,1]",
    )
    parser.add_argument(
        "--block-mask-file",
        type=str,
        default=None,
        help="JSON file with block masks; takes precedence over --block-sparsity",
    )
    parser.add_argument(
        "--n-repetitions",
        type=int,
        default=None,
        help="With random block sparsity: run repeated masks and report quantiles",
    )

    parser.add_argument(
        "--include-quant-overhead",
        action="store_true",
        help="(MXFP4 only) Include quantization overhead in benchmark",
    )
    parser.add_argument(
        "--hadamard-rotate",
        type=lambda v: bool(int(v)),
        default=True,
        help="(MXFP4 only) Apply Hadamard rotation: 1/0",
    )
    parser.add_argument(
        "--block-r",
        type=int,
        default=128,
        help="(MXFP4 only) Hadamard block size, must be <= head dim",
    )
    parser.add_argument(
        "--qsmooth",
        action="store_true",
        help="(MXFP4 only) Enable Q smoothing",
    )

    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="do_bench rep time in ms",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="do_bench warmup time in ms",
    )

    return parser.parse_args()


def run_with_optional_vgpr(args: argparse.Namespace, runner: Any) -> int:
    if args.print_vgpr:
        print_vgpr(runner, get_caller_name_no_ext())
    else:
        runner()
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)

    loaded_masks = load_block_mask_from_json(args.block_mask_file, torch.device("cuda"))
    loaded_single_mask: Optional[LoadedMask] = None

    if isinstance(loaded_masks, list):
        if args.load_captured:
            raise ValueError("List mask mode and --load-captured cannot be combined")
        if args.hq <= 0 or args.d <= 0:
            raise ValueError("For list mask mode, provide positive --hq and --d")
        if args.dv <= 0:
            args.dv = args.d
        if args.hk <= 0:
            args.hk = args.hq
        return run_with_optional_vgpr(
            args,
            lambda: run_benchmark_mask_list(args, loaded_masks),
        )

    if isinstance(loaded_masks, LoadedMask):
        loaded_single_mask = loaded_masks

    if (
        args.block_sparsity is not None
        and args.n_repetitions is not None
        and args.block_mask_file is None
    ):
        return run_with_optional_vgpr(
            args,
            lambda: run_block_sparse_repetitions(args, loaded_single_mask),
        )

    default_runner = (
        lambda: run_benchmark_captured(args, loaded_single_mask)
        if args.load_captured
        else lambda: run_benchmark_generated(args, loaded_single_mask)
    )
    return run_with_optional_vgpr(args, default_runner)


if __name__ == "__main__":
    sys.exit(main())
