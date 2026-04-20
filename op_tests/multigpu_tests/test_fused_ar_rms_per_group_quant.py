# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Unit test for fused AllReduce + RMSNorm + per-1×128-group FP8 quantization.

This test validates that the fused kernel produces results matching the
three-step reference: all-reduce → RMSNorm → per-group FP8 quant.

Usage:
    python test_fused_ar_rms_per_group_quant.py -t 8 -s 64,4096
    python test_fused_ar_rms_per_group_quant.py -t 4 --group-size 128
"""

import os
from typing import Optional
import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import itertools
import pandas as pd
from aiter import dtypes

from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
    get_tp_group,
    destroy_model_parallel,
    destroy_distributed_environment,
)
from aiter.dist.utils import get_open_port, get_distributed_init_method, get_ip
from aiter.test_common import (
    checkAllclose,
    perftest,
    benchmark,
)
from multiprocessing import set_start_method, Pool, freeze_support
import logging

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max


def _per_group_quant_ref(x_bf16: torch.Tensor, group_size: int = 128):
    """Reference per-group FP8 quantization on CPU/GPU (bf16 → fp8 + f32 scales)."""
    M, K = x_bf16.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    num_groups = K // group_size

    x_groups = x_bf16.float().reshape(M, num_groups, group_size)
    amax = x_groups.abs().amax(dim=-1)  # (M, num_groups)
    scale = amax / FP8_MAX
    scale = scale.clamp(min=1e-12)

    x_scaled = x_groups / scale.unsqueeze(-1)
    x_fp8 = x_scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fnuz)
    x_fp8 = x_fp8.reshape(M, K)

    return x_fp8, scale  # (M, K) fp8, (M, num_groups) f32


def fused_ar_rmsnorm_per_group_quant(
    tp_size,
    pp_size,
    rankID,
    x,
    weight,
    eps,
    group_size=128,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
    emit_bf16: bool = False,
):
    """Run fused AR+RMSNorm+per-group-quant on a single rank.

    When ``emit_bf16=True`` the kernel ALSO writes the pre-quantization
    bf16/fp16 normed output; we cross-check that bf16 output against the
    fp8+scale dequant to verify the two outputs agree to FP8 precision.
    """
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)
    weight = weight.to(device)

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group,
    )

    @perftest()
    def run_fused(x):
        res = tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(
            x, x, weight, eps, group_size=group_size, emit_bf16=emit_bf16
        )
        if emit_bf16:
            out, res_out, scale_out, bf16_out = res
            return out, scale_out, res_out, bf16_out
        out, res_out, scale_out = res
        return out, scale_out, res_out

    result, us = run_fused(x)
    if emit_bf16:
        out_fp8, scale_out, res_out, bf16_out = result
    else:
        out_fp8, scale_out, res_out = result
        bf16_out = None
    dequant = out_fp8.float() * scale_out.repeat_interleave(group_size, dim=-1)

    # When requesting bf16 output, verify it matches the fp8+scale dequant
    # at FP8 precision: both are produced by the same fused kernel from the
    # same internal fp32 normed value, so they should differ only by the
    # post-quant rounding.
    bf16_vs_fp8_diff = None
    if bf16_out is not None:
        bf16_vs_fp8_diff = (bf16_out.float() - dequant).abs().max().item()

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    return dequant.to(x.dtype), us, scale_out.shape, bf16_vs_fp8_diff


@benchmark()
def test_fused_ar_rmsnorm_per_group_quant(
    tp_size,
    pp_size,
    shape,
    dtype,
    group_size=128,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
    emit_bf16: bool = False,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    n = shape[1]
    eps = 1e-6
    weight = torch.randn((n,), dtype=dtype)
    x = torch.randn(shape, dtype=dtype)
    ref = x * tp_size

    rets = []
    cpu_rslt = []
    for i in range(tp_size):
        rets.append(
            pool.apply_async(
                fused_ar_rmsnorm_per_group_quant,
                args=(
                    tp_size,
                    pp_size,
                    i,
                    x,
                    weight,
                    eps,
                    group_size,
                    withGraph,
                    distributed_init_method,
                    emit_bf16,
                ),
            )
        )
    pool.close()
    pool.join()

    for i in range(tp_size):
        host_normed = F.rms_norm(
            input=(ref + x),
            normalized_shape=(ref.shape[-1],),
            weight=weight,
            eps=eps,
        )
        cpu_rslt.append(host_normed)

    rets = [el.get() for el in rets]
    all_us = [us for _, us, _, _ in rets]
    scale_shapes = [ss for _, _, ss, _ in rets]
    bf16_diffs = [bd for _, _, _, bd in rets if bd is not None]

    M, K = shape
    expected_scale_shape = (M, K // group_size)
    for ss in scale_shapes:
        assert (
            ss == expected_scale_shape
        ), f"Scale shape mismatch: got {ss}, expected {expected_scale_shape}"

    atol = 5e-2
    rtol = 5e-2
    max_err = 0.0
    for dequant_out, us, _, _ in rets:
        msg = (
            f"test_fused_ar_rmsnorm_per_group_quant: "
            f"{shape=} {dtype=} {group_size=} {withGraph=} "
            f"{emit_bf16=} {us:>8.2f}"
        )
        err = checkAllclose(
            cpu_rslt[dequant_out.device.index],
            dequant_out.to(ref),
            msg=msg,
            atol=atol,
            rtol=rtol,
        )
        max_err = max(max_err, err)

    # bf16 side-output correctness: should agree with fp8+scale dequant to
    # within at most one FP8 quantization step (~3% relative).
    max_bf16_vs_fp8 = max(bf16_diffs) if bf16_diffs else 0.0
    if emit_bf16:
        assert max_bf16_vs_fp8 < 1.0, (
            f"bf16 side-output disagrees with fp8 dequant by "
            f"{max_bf16_vs_fp8}, expected <1.0"
        )

    return {
        "emit_bf16": emit_bf16,
        "per_group_min_us": min(all_us),
        "per_group_max_us": max(all_us),
        "per_group_err": max_err,
        "bf16_vs_fp8": max_bf16_vs_fp8,
    }


l_dtype = ["bf16"]
# Default matrix covers:
#   * Unaligned token counts (13, 17) that exercise the thread-padding path.
#   * Decode-scale batches (1, 32, 128, 512).
#   * Prefill-scale batches (1024, 2048) that straddle the 1-stage (<=128KB),
#     2-stage (<=512KB), and split (>512KB) kernel dispatch boundaries.
#   * Hidden sizes covering the common FP8/MoE model families:
#       4096   Qwen3.5-FP8, Qwen3-MoE, Mixtral 8x7B, Llama 3 8B
#       6144   Mixtral 8x22B, some hybrid configs
#       7168   DeepSeek-V2/V3
#       8192   Llama 3/3.1 70B, GLM-4
l_shape = [
    # hidden = 4096 (Qwen3.5-FP8 / Qwen3-MoE / Mixtral 8x7B / Llama 3 8B)
    (13, 4096),
    (17, 4096),
    (1, 4096),
    (32, 4096),
    (128, 4096),
    (512, 4096),
    (1024, 4096),
    (2048, 4096),
    # hidden = 6144 (Mixtral 8x22B)
    (1, 6144),
    (32, 6144),
    (128, 6144),
    (512, 6144),
    # hidden = 7168 (DeepSeek-V2/V3)
    (1, 7168),
    (32, 7168),
    (128, 7168),
    (512, 7168),
    # hidden = 8192 (Llama 3/3.1 70B / GLM-4)
    (1, 8192),
    (32, 8192),
    (128, 8192),
    (512, 8192),
]
l_tp = [8]
l_pp = [1]
l_graph = [False]
l_group_size = [128]
# Cover both the fp8-only output (keep_bf16=False, std-attention layers)
# and the fp8+bf16 dual-output (keep_bf16=True, GDN-style layers).
l_emit_bf16 = [False, True]

parser = argparse.ArgumentParser(
    description="Test fused AR+RMSNorm+per-group FP8 quant"
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=["fp16", "bf16"],
    nargs="?",
    const=None,
    default=None,
)
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    nargs="*",
    default=None,
    help="shape(s). e.g. -s 128,4096 64,4096",
)
parser.add_argument(
    "-t",
    "--tp",
    type=int,
    nargs="?",
    const=None,
    default=None,
)
parser.add_argument(
    "-p",
    "--pp",
    type=int,
    nargs="?",
    const=None,
    default=None,
)
parser.add_argument(
    "-g",
    "--graphon",
    type=int,
    nargs="?",
    const=None,
    default=None,
)
parser.add_argument(
    "--group-size",
    type=int,
    nargs="?",
    const=None,
    default=None,
)

if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.shape is not None:
        l_shape = args.shape
    if args.tp is not None:
        l_tp = [args.tp]
    if args.pp is not None:
        l_pp = [args.pp]
    if args.graphon is not None:
        l_graph = [args.graphon]
    if args.group_size is not None:
        l_group_size = [args.group_size]

    df = []
    for dtype, shape, tp, pp, graph_on, gs, emit_bf16 in itertools.product(
        l_dtype, l_shape, l_tp, l_pp, l_graph, l_group_size, l_emit_bf16
    ):
        ret = test_fused_ar_rmsnorm_per_group_quant(
            tp,
            pp,
            shape,
            dtype,
            group_size=gs,
            withGraph=graph_on,
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
            emit_bf16=emit_bf16,
        )
        df.append(ret)

    df = pd.DataFrame(df)
    show_cols = [
        "tp_size",
        "shape",
        "dtype",
        "withGraph",
        "emit_bf16",
        "per_group_min_us",
        "per_group_max_us",
        "per_group_err",
        "bf16_vs_fp8",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    logger.info(
        "fused AR+RMSNorm+per-group-quant summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
