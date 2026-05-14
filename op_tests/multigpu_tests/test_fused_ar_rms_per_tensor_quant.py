# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Test: fused AllReduce + RMSNorm + per-tensor static FP8 quantization
# Usage:
#   python test_fused_ar_rms_per_tensor_quant.py
#   python test_fused_ar_rms_per_tensor_quant.py -t 8 -d bf16
#   python test_fused_ar_rms_per_tensor_quant.py -t 8 -g 1   # with CUDA graph

import os
from typing import Optional
import aiter
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
    graph_capture,
    destroy_model_parallel,
    destroy_distributed_environment,
)
from aiter.dist.utils import get_open_port, get_distributed_init_method, get_ip
from aiter.dist.communication_op import (
    tensor_model_parallel_fused_allreduce_rmsnorm_per_tensor_quant,
)
from aiter.test_common import (
    checkAllclose,
    perftest,
    benchmark,
)
from multiprocessing import set_start_method, Pool, freeze_support
import logging

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def fused_ar_rmsnorm_per_tensor_quant(
    tp_size,
    pp_size,
    rankID,
    x,
    weight,
    eps,
    scale_factor,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    """Worker function executed in each GPU process."""
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)

    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)
    weight = weight.to(device)
    scale = torch.tensor([scale_factor], dtype=torch.float32, device=device)

    # Warm-up barrier
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                out_fp8, res_out = (
                    tensor_model_parallel_fused_allreduce_rmsnorm_per_tensor_quant(
                        x, x, weight, eps, scale
                    )
                )
        out_fp8.fill_(0)
        res_out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        # Dequant to float for comparison
        out = (out_fp8.float() * scale_factor, us)
    else:

        @perftest()
        def run_ca(x):
            out_fp8, res_out = (
                tensor_model_parallel_fused_allreduce_rmsnorm_per_tensor_quant(
                    x, x, weight, eps, scale
                )
            )
            return out_fp8, res_out

        result = run_ca(x)
        out_fp8_val, us = result[0][0], result[1]
        out = (out_fp8_val.float() * scale_factor, us)

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


@benchmark()
def test_fused_ar_rmsnorm_per_tensor_quant(
    tp_size,
    pp_size,
    shape,
    dtype,
    scale_factor=1.0,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    """
    Reference: allreduce x -> add residual x -> RMSNorm -> scale/scale_factor -> clamp -> fp8
    The test dequantizes the fp8 output (mul scale_factor) before comparing with the float ref.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49375"
    pool = Pool(processes=tp_size)
    n = shape[1]
    eps = 1e-6
    weight = torch.randn((n,), dtype=dtype)
    x = torch.randn(shape, dtype=dtype)

    # Each rank contributes `x`; allreduce result = x * tp_size
    ref = x * tp_size

    rets = []
    cpu_rslt = []
    weight_list = []
    res_inp_list = []

    for i in range(tp_size):
        res_inp_list.append(x)
        weight_list.append(weight)
        rets.append(
            pool.apply_async(
                fused_ar_rmsnorm_per_tensor_quant,
                args=(
                    tp_size,
                    pp_size,
                    i,
                    x,
                    weight,
                    eps,
                    scale_factor,
                    withGraph,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    # Collect results with per-worker timeout; terminate pool on any failure so
    # worker processes are reaped and do not accumulate as zombies.
    try:
        rets = [r.get(timeout=120) for r in rets]
    except Exception:
        pool.terminate()
        raise
    finally:
        pool.join()

    # CPU reference: RMSNorm(ar_result + residual) quantized with static scale, then dequant
    fp8_max = 448.0  # e4m3fnuz max
    inv_scale = 1.0 / scale_factor
    for i in range(tp_size):
        rms_ref = F.rms_norm(
            input=(ref + res_inp_list[i]),
            normalized_shape=(ref.shape[-1],),
            weight=weight_list[i],
            eps=eps,
        )
        # Simulate per-tensor quant + dequant
        quantized = (rms_ref.float() * inv_scale).clamp(-fp8_max, fp8_max)
        dequant = quantized * scale_factor
        cpu_rslt.append(dequant)
    all_us = [us for _, us in rets]

    # Allow larger tolerance due to FP8 quantization error
    atol = 5e-2
    rtol = 5e-2
    max_err = 0.0
    for out, us in rets:
        msg = (
            f"test_fused_ar_rmsnorm_per_tensor_quant: "
            f"{shape=} {dtype=} {withGraph=} scale={scale_factor:.4f} {us:>8.2f}us"
        )
        err = checkAllclose(
            cpu_rslt[out.device.index].to(out.device),
            out,
            msg=msg,
            atol=atol,
            rtol=rtol,
        )
        max_err = max(max_err, err)

    return {
        "pt_quant_min_us": min(all_us),
        "pt_quant_max_us": max(all_us),
        "pt_quant_err": max_err,
    }


# ── Test configuration ──
l_dtype = ["fp16", "bf16"]
l_shape = [
    (13, 512),
    (13, 1024),
    (13, 2048),
    (17, 4096),
    (19, 8192),
]
l_tp = [8]
l_pp = [1]
l_graph = [False, True]
# Use a realistic calibrated scale; any positive float is valid
l_scale = [0.5]

parser = argparse.ArgumentParser(description="test fused AR+RMSNorm+per-tensor-quant")
parser.add_argument("-d", "--dtype", type=str, choices=l_dtype, default=None)
parser.add_argument("-s", "--shape", type=dtypes.str2tuple, nargs="*", default=None)
parser.add_argument("-t", "--tp", type=int, default=None)
parser.add_argument("-p", "--pp", type=int, default=None)
parser.add_argument("-g", "--graphon", type=int, default=None)
parser.add_argument("--scale", type=float, default=None, help="static scale factor, e.g. 0.5")

if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.dtype is not None:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    else:
        l_dtype = [dtypes.d_dtypes[k] for k in l_dtype]
    if args.shape is not None:
        l_shape = args.shape
    if args.tp is not None:
        l_tp = [args.tp]
    if args.pp is not None:
        l_pp = [args.pp]
    if args.graphon is not None:
        l_graph = [bool(args.graphon)]
    if args.scale is not None:
        l_scale = [args.scale]

    df = []
    for dtype, shape, tp, pp, graph_on, scale in itertools.product(
        l_dtype, l_shape, l_tp, l_pp, l_graph, l_scale
    ):
        ret = test_fused_ar_rmsnorm_per_tensor_quant(
            tp,
            pp,
            shape,
            dtype,
            scale_factor=scale,
            withGraph=graph_on,
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
        )
        df.append(ret)

    df = pd.DataFrame(df)
    show_cols = [c for c in [
        "tp_size", "shape", "dtype", "withGraph",
        "pt_quant_min_us", "pt_quant_max_us", "pt_quant_err",
    ] if c in df.columns]
    logger.info(
        "fused AR+RMSNorm+per-tensor-quant summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
