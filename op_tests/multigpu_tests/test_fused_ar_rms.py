# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

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
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
    tensor_model_parallel_fused_allreduce_rmsnorm_quant,
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


def fused_ar_rmsnorm(
    tp_size,
    pp_size,
    rankID,
    x,
    weight,
    eps,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
    post_per_token_quant: bool = False,
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
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
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                if not post_per_token_quant:
                    out, res_out = tensor_model_parallel_fused_allreduce_rmsnorm(
                        x, x, weight, eps
                    )
                else:
                    out, res_out, scale_out = (
                        tensor_model_parallel_fused_allreduce_rmsnorm_quant(
                            x, x, weight, eps
                        )
                    )
        out.fill_(0)
        res_out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        if not post_per_token_quant:
            out = (out, us)
        else:
            out = (out.float() * scale_out, us)
    else:

        @perftest()
        def run_ca(x):
            if not post_per_token_quant:
                out, res_out = tensor_model_parallel_fused_allreduce_rmsnorm(
                    x, x, weight, eps
                )
                return out
            else:
                out, res_out, scale_out = (
                    tensor_model_parallel_fused_allreduce_rmsnorm_quant(
                        x, x, weight, eps
                    )
                )
                return out, scale_out

        if not post_per_token_quant:
            out = run_ca(x)
        else:
            out = run_ca(x)
            out = (out[0][0].float() * out[0][1], out[1])

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def get_acc_value_with_cudagraph(
    tp_size,
    pp_size,
    rankID,
    x,
    weight,
    eps,
    loop_time=1,
    distributed_init_method: Optional[str] = None,
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
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
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    # out = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc:
        with torch.cuda.graph(graph, stream=gc.stream):
            # out = torch.empty_like(x)
            out, res_out = tensor_model_parallel_fused_allreduce_rmsnorm(
                x, x, weight, eps
            )
    out.fill_(0)

    def run_ca():
        graph.replay()
        rslt = out.clone()
        out.fill_(0)
        return rslt

    for i in range(loop_time):
        out = run_ca()

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def get_acc_value_only(
    tp_size,
    pp_size,
    rankID,
    x,
    weight,
    eps,
    loop_time=1,
    distributed_init_method: Optional[str] = None,
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
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
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    torch.cuda.synchronize()

    for i in range(loop_time):
        out, res = tensor_model_parallel_fused_allreduce_rmsnorm(x, x, weight, eps)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def split_ar_rmsnorm(
    tp_size,
    pp_size,
    rankID,
    x,
    weight,
    eps,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
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
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                ar_out = tensor_model_parallel_all_reduce(x)
                # out = aiter.rms_norm(ar_out, weight, eps, 0)
                out = torch.empty_like(ar_out)
                residual_out = torch.empty_like(ar_out)
                aiter.rmsnorm2d_fwd_with_add(
                    out,
                    ar_out,
                    x,
                    residual_out,
                    weight,
                    eps,
                    0,
                )
        out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        out = (out, us)
    else:

        @perftest()
        def run_ca(x):
            ar_out = tensor_model_parallel_all_reduce(x)
            out = torch.empty_like(ar_out)
            residual_out = torch.empty_like(ar_out)
            aiter.rmsnorm2d_fwd_with_add(
                out,
                ar_out,
                x,
                residual_out,
                weight,
                eps,
                0,
            )
            return out

        out = run_ca(x)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


@benchmark()
def test_fused_ar_rmsnorm(
    tp_size,
    pp_size,
    shape,
    dtype,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
    post_per_token_quant: bool = False,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    cpu_rslt = []
    weight_list = []
    res_inp = []
    # print(type(shape[0]), shape[1], ref.device)
    n = shape[1]
    eps = 1e-6
    weight = torch.randn((n,), dtype=dtype)
    x = torch.randn(shape, dtype=dtype)
    ref = x * tp_size
    for i in range(tp_size):
        res_inp.append(x)
        weight_list.append(weight)
        rets.append(
            pool.apply_async(
                fused_ar_rmsnorm,
                args=(
                    tp_size,
                    pp_size,
                    i,
                    x,
                    weight,
                    eps,
                    withGraph,
                    distributed_init_method,
                    post_per_token_quant,
                ),
            )
        )
    pool.close()
    pool.join()
    print(f"rslt[0][0] = {ref[0][0]}")

    for i in range(tp_size):
        host_rslt = F.rms_norm(
            input=(ref + res_inp[i]),
            normalized_shape=(ref.shape[-1],),
            weight=weight_list[i],
            eps=eps,
        )
        # host_rslt = ref + res_inp[i]
        cpu_rslt.append(host_rslt)

    rets = [el.get() for el in rets]
    all_us = [us for _, us in rets]
    atol = 5e-2 if post_per_token_quant else 1e-2
    rtol = atol
    max_err = 0.0
    for out, us in rets:
        msg = f"test_fused_ar_rmsnorm: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        # print(cpu_rslt[out.device.index])
        err = checkAllclose(
            cpu_rslt[out.device.index], out.to(ref), msg=msg, atol=atol, rtol=rtol
        )
        max_err = max(max_err, err)
        # checkAllclose(ref, out.to(ref), msg=msg)
    suffix = "quant" if post_per_token_quant else "fused"
    return {
        f"{suffix}_min_us": min(all_us),
        f"{suffix}_max_us": max(all_us),
        f"{suffix}_err": max_err,
    }


def _make_strided_lastdim_view(x: torch.Tensor, extra_cols: int = 32) -> torch.Tensor:
    *prefix, n = x.shape
    base = torch.empty((*prefix, n + extra_cols), dtype=x.dtype, device=x.device)
    base[..., :n].copy_(x)
    return base[..., :n]


def fused_ar_rmsnorm_pad_stride(
    tp_size,
    pp_size,
    rankID,
    x,
    residual,
    weight,
    eps,
    x_pad_to_multiple=0,
    input_strided=False,
    residual_strided=False,
    distributed_init_method: Optional[str] = None,
):
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
    residual = residual.to(device)
    weight = weight.to(device)
    if input_strided:
        x = _make_strided_lastdim_view(x)
    if residual_strided:
        residual = _make_strided_lastdim_view(residual)

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    out, res_out = tensor_model_parallel_fused_allreduce_rmsnorm(
        x,
        residual,
        weight,
        eps,
        x_pad_to_multiple=x_pad_to_multiple,
    )

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out, res_out


def fused_ar_rmsnorm_padded_input(
    tp_size,
    pp_size,
    rankID,
    x,
    residual,
    weight,
    eps,
    input_storage_width,
    distributed_init_method: Optional[str] = None,
):
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
    residual = residual.to(device)
    weight = weight.to(device)

    if input_storage_width > x.shape[-1]:
        padded_x = torch.zeros(
            x.shape[:-1] + (input_storage_width,), dtype=x.dtype, device=x.device
        )
        padded_x[..., : x.shape[-1]].copy_(x)
        x = padded_x

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    out, res_out = tensor_model_parallel_fused_allreduce_rmsnorm(
        x,
        residual,
        weight,
        eps,
    )

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out, res_out


def test_fused_ar_rmsnorm_pad_stride_case(
    tp_size,
    pp_size,
    shape,
    dtype,
    *,
    x_pad_to_multiple=0,
    input_strided=False,
    residual_strided=False,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    residual = torch.randn(shape, dtype=dtype)
    weight = torch.randn((shape[-1],), dtype=dtype)
    rets = []
    eps = 1e-6
    for rank in range(tp_size):
        rets.append(
            pool.apply_async(
                fused_ar_rmsnorm_pad_stride,
                args=(
                    tp_size,
                    pp_size,
                    rank,
                    x,
                    residual,
                    weight,
                    eps,
                    x_pad_to_multiple,
                    input_strided,
                    residual_strided,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()

    ref = x * tp_size
    ref_residual = ref + residual
    ref_out = F.rms_norm(
        input=ref_residual,
        normalized_shape=(shape[-1],),
        weight=weight,
        eps=eps,
    )
    if x_pad_to_multiple > 0:
        n = shape[-1]
        n_out = ((n + x_pad_to_multiple - 1) // x_pad_to_multiple) * x_pad_to_multiple
        ref_out = F.pad(ref_out, (0, n_out - n), "constant", 0.0)
    rets = [ret.get() for ret in rets]

    atol = 1e-2 if dtype != torch.bfloat16 else 5e-2
    rtol = atol
    max_err = 0.0
    max_res_err = 0.0
    expected_n = ref_out.shape[-1]
    for out, res_out in rets:
        assert out.shape == shape[:-1] + (expected_n,)
        assert res_out.shape == shape
        err = checkAllclose(
            ref_out,
            out.to(ref_out),
            msg=(
                "test_fused_ar_rmsnorm_pad_stride_case: "
                f"{shape=} {dtype=} {x_pad_to_multiple=} "
                f"{input_strided=} {residual_strided=}"
            ),
            atol=atol,
            rtol=rtol,
        )
        res_err = checkAllclose(
            ref_residual,
            res_out.to(ref_residual),
            msg=(
                "test_fused_ar_rmsnorm_pad_stride_case residual: "
                f"{shape=} {dtype=} {x_pad_to_multiple=} "
                f"{input_strided=} {residual_strided=}"
            ),
            atol=atol,
            rtol=rtol,
        )
        max_err = max(max_err, err)
        max_res_err = max(max_res_err, res_err)
    return {
        "shape": shape,
        "dtype": str(dtype),
        "x_pad_to_multiple": x_pad_to_multiple,
        "input_strided": input_strided,
        "residual_strided": residual_strided,
        "err": max_err,
        "res_err": max_res_err,
    }


def test_fused_ar_rmsnorm_padded_input_case(
    tp_size,
    pp_size,
    shape,
    dtype,
    *,
    input_storage_width,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    residual = torch.randn(shape, dtype=dtype)
    weight = torch.randn((shape[-1],), dtype=dtype)
    rets = []
    eps = 1e-6
    for rank in range(tp_size):
        rets.append(
            pool.apply_async(
                fused_ar_rmsnorm_padded_input,
                args=(
                    tp_size,
                    pp_size,
                    rank,
                    x,
                    residual,
                    weight,
                    eps,
                    input_storage_width,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()

    ref = x * tp_size
    ref_residual = ref + residual
    ref_out = F.rms_norm(
        input=ref_residual,
        normalized_shape=(shape[-1],),
        weight=weight,
        eps=eps,
    )
    rets = [ret.get() for ret in rets]

    atol = 1e-2 if dtype != torch.bfloat16 else 5e-2
    rtol = atol
    max_err = 0.0
    max_res_err = 0.0
    for out, res_out in rets:
        assert out.shape == shape
        assert res_out.shape == shape
        err = checkAllclose(
            ref_out,
            out.to(ref_out),
            msg=(
                "test_fused_ar_rmsnorm_padded_input_case: "
                f"{shape=} {dtype=} {input_storage_width=}"
            ),
            atol=atol,
            rtol=rtol,
        )
        res_err = checkAllclose(
            ref_residual,
            res_out.to(ref_residual),
            msg=(
                "test_fused_ar_rmsnorm_padded_input_case residual: "
                f"{shape=} {dtype=} {input_storage_width=}"
            ),
            atol=atol,
            rtol=rtol,
        )
        max_err = max(max_err, err)
        max_res_err = max(max_res_err, res_err)
    return {
        "shape": shape,
        "dtype": str(dtype),
        "input_storage_width": input_storage_width,
        "err": max_err,
        "res_err": max_res_err,
    }


def fused_ar_rmsnorm_gemma(
    tp_size,
    pp_size,
    rankID,
    x,
    residual,
    weight,
    eps,
    distributed_init_method: Optional[str] = None,
):
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
    residual = residual.to(device)
    weight = weight.to(device)

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    out, res_out = tensor_model_parallel_fused_allreduce_rmsnorm(
        x,
        residual,
        weight,
        eps,
        gemma_norm=True,
    )

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out, res_out


def test_fused_ar_gemma_rmsnorm_case(
    tp_size,
    pp_size,
    shape,
    dtype,
    *,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    torch.manual_seed(123)
    x = torch.randn(shape, dtype=dtype)
    residual = torch.randn(shape, dtype=dtype)
    # GemmaRMSNorm stores zero-centered weights and applies (1 + weight).
    weight = torch.randn((shape[-1],), dtype=dtype) * 0.1
    eps = 1e-6
    rets = []
    for rank in range(tp_size):
        rets.append(
            pool.apply_async(
                fused_ar_rmsnorm_gemma,
                args=(
                    tp_size,
                    pp_size,
                    rank,
                    x,
                    residual,
                    weight,
                    eps,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()

    ref_residual = x * tp_size + residual
    ref_out = F.rms_norm(
        input=ref_residual,
        normalized_shape=(shape[-1],),
        weight=weight + 1.0,
        eps=eps,
    )
    rets = [ret.get() for ret in rets]

    atol = 1e-2 if dtype != torch.bfloat16 else 5e-2
    rtol = atol
    max_err = 0.0
    max_res_err = 0.0
    for out, res_out in rets:
        assert out.shape == shape
        assert res_out.shape == shape
        err = checkAllclose(
            ref_out,
            out.to(ref_out),
            msg=f"test_fused_ar_gemma_rmsnorm_case: {shape=} {dtype=}",
            atol=atol,
            rtol=rtol,
        )
        res_err = checkAllclose(
            ref_residual,
            res_out.to(ref_residual),
            msg=f"test_fused_ar_gemma_rmsnorm_case residual: {shape=} {dtype=}",
            atol=atol,
            rtol=rtol,
        )
        max_err = max(max_err, err)
        max_res_err = max(max_res_err, res_err)
    return {
        "shape": shape,
        "dtype": str(dtype),
        "err": max_err,
        "res_err": max_res_err,
    }


try:
    import pytest

    @pytest.mark.parametrize(
        "x_pad_to_multiple,input_strided,residual_strided",
        [
            (0, True, False),
            (0, False, True),
            (256, True, False),
            (256, True, True),
        ],
    )
    def test_pad_and_stride_tp2(
        x_pad_to_multiple: int, input_strided: bool, residual_strided: bool
    ):
        if torch.cuda.device_count() < 2:
            pytest.skip(f"requires >= 2 GPUs (have {torch.cuda.device_count()})")
        ret = test_fused_ar_rmsnorm_pad_stride_case(
            tp_size=2,
            pp_size=1,
            shape=(13, 2880),
            dtype=dtypes.d_dtypes["bf16"],
            x_pad_to_multiple=x_pad_to_multiple,
            input_strided=input_strided,
            residual_strided=residual_strided,
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
        )
        assert ret["err"] < 5e-2, f"fused_ar_rmsnorm err={ret['err']} config={ret}"
        assert (
            ret["res_err"] < 5e-2
        ), f"fused_ar_rmsnorm residual err={ret['res_err']} config={ret}"

    def test_padded_input_tp2():
        if torch.cuda.device_count() < 2:
            pytest.skip(f"requires >= 2 GPUs (have {torch.cuda.device_count()})")
        ret = test_fused_ar_rmsnorm_padded_input_case(
            tp_size=2,
            pp_size=1,
            shape=(13, 2880),
            dtype=dtypes.d_dtypes["bf16"],
            input_storage_width=3072,
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
        )
        assert ret["err"] < 5e-2, f"fused_ar_rmsnorm err={ret['err']} config={ret}"
        assert (
            ret["res_err"] < 5e-2
        ), f"fused_ar_rmsnorm residual err={ret['res_err']} config={ret}"

    def test_large_padded_output_tp2():
        if torch.cuda.device_count() < 2:
            pytest.skip(f"requires >= 2 GPUs (have {torch.cuda.device_count()})")
        ret = test_fused_ar_rmsnorm_pad_stride_case(
            tp_size=2,
            pp_size=1,
            shape=(13, 4096),
            dtype=dtypes.d_dtypes["bf16"],
            x_pad_to_multiple=16384,
            input_strided=False,
            residual_strided=False,
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
        )
        assert ret["err"] < 5e-2, f"fused_ar_rmsnorm err={ret['err']} config={ret}"
        assert (
            ret["res_err"] < 5e-2
        ), f"fused_ar_rmsnorm residual err={ret['res_err']} config={ret}"

    def test_strided_nd_input_tp2():
        if torch.cuda.device_count() < 2:
            pytest.skip(f"requires >= 2 GPUs (have {torch.cuda.device_count()})")
        ret = test_fused_ar_rmsnorm_pad_stride_case(
            tp_size=2,
            pp_size=1,
            shape=(3, 13, 2880),
            dtype=dtypes.d_dtypes["bf16"],
            x_pad_to_multiple=0,
            input_strided=True,
            residual_strided=False,
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
        )
        assert ret["err"] < 5e-2, f"fused_ar_rmsnorm err={ret['err']} config={ret}"
        assert (
            ret["res_err"] < 5e-2
        ), f"fused_ar_rmsnorm residual err={ret['res_err']} config={ret}"

    def test_gemma_norm_tp2():
        if torch.cuda.device_count() < 2:
            pytest.skip(f"requires >= 2 GPUs (have {torch.cuda.device_count()})")
        ret = test_fused_ar_gemma_rmsnorm_case(
            tp_size=2,
            pp_size=1,
            shape=(17, 2048),
            dtype=dtypes.d_dtypes["bf16"],
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
        )
        assert (
            ret["err"] < 5e-2
        ), f"fused_ar_gemma_rmsnorm err={ret['err']} config={ret}"
        assert (
            ret["res_err"] < 5e-2
        ), f"fused_ar_gemma_rmsnorm residual err={ret['res_err']} config={ret}"

except ImportError:
    pass


l_dtype = ["fp16", "bf16"]
# (13, 2880): GPT-OSS-120B / GPT-OSS-20B hidden_size (n_bytes=5760, 4096 < 5760 < 8192)
l_shape = [
    (13, 512),
    (13, 1024),
    (13, 2048),
    (13, 2880),
    (17, 4096),
    (17, 7168),
    (19, 8192),
]
l_tp = [8]
l_pp = [1]
l_graph = [False, True]

parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="data type",
)
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    nargs="*",
    default=None,
    help="shape(s). e.g. -s 128,8192 256,7168",
)

parser.add_argument(
    "-t",
    "--tp",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="tp num. e.g. -t 8",
)

parser.add_argument(
    "-p",
    "--pp",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="tp num. e.g. -p 1",
)

parser.add_argument(
    "-g",
    "--graphon",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="open cudagraph. e.g. -g 1",
)

l_test_types = ["fused", "quant"]
parser.add_argument(
    "--test",
    type=str,
    choices=l_test_types,
    nargs="*",
    default=None,
    help="test type(s) to run. e.g. --test fused quant",
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
        print(args.graphon)
        l_graph = [args.graphon]
    run_tests = args.test if args.test else l_test_types
    df = []
    for dtype, shape, tp, pp, graph_on in itertools.product(
        l_dtype, l_shape, l_tp, l_pp, l_graph
    ):
        row = {}
        if "fused" in run_tests:
            ret = test_fused_ar_rmsnorm(
                tp,
                pp,
                shape,
                dtype,
                withGraph=graph_on,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
                post_per_token_quant=False,
            )
            row.update(ret)
        if "quant" in run_tests:
            ret = test_fused_ar_rmsnorm(
                tp,
                pp,
                shape,
                dtype,
                withGraph=graph_on,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
                post_per_token_quant=True,
            )
            row.update(ret)
        df.append(row)
    df = pd.DataFrame(df)
    show_cols = [
        "tp_size",
        "shape",
        "dtype",
        "withGraph",
        "fused_min_us",
        "fused_max_us",
        "fused_err",
        "quant_min_us",
        "quant_max_us",
        "quant_err",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    logger.info(
        "fused allreduce rmsnorm summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
