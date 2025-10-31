# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import aiter
import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import itertools
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


def fused_ar_rmsnorm(tp_size, pp_size, rankID, x, weight, eps, withGraph=False):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
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
                out = tensor_model_parallel_fused_allreduce_rmsnorm(x, weight, eps)
        out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        out = (out, us)
    else:

        @perftest()
        def run_ca(x):
            return tensor_model_parallel_fused_allreduce_rmsnorm(x, weight, eps)

        out = run_ca(x)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def get_acc_value_with_cudagraph(tp_size, pp_size, rankID, x, weight, eps, loop_time=1):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
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
            out = tensor_model_parallel_fused_allreduce_rmsnorm(x, weight, eps)
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


def get_acc_value_only(tp_size, pp_size, rankID, x, weight, eps, loop_time=1):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)
    weight = weight.to(device)
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    torch.cuda.synchronize()

    for i in range(loop_time):
        out = tensor_model_parallel_fused_allreduce_rmsnorm(x, weight, eps)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def split_ar_rmsnorm(tp_size, pp_size, rankID, x, weight, eps, withGraph=False):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
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


def run_cu(input, weight, eps, device_id):
    device = f"cuda:{device_id}"
    input = input.to(device)
    weight = weight.to(device)

    @perftest()
    def compute():
        output = torch.empty_like(input)
        aiter.rms_norm_cu(output, input, weight, eps)

    return compute()


@benchmark()
def test_split_ar_rmsnorm(tp_size, pp_size, shape, dtype, withGraph=False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    cpu_rslt = []
    weight_list = []
    res_inp = []
    # print(type(shape[0]), shape[1], ref.device)
    m = shape[0]
    n = shape[1]
    eps = 1e-6
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        res_inp.append(x)
        ref += x
        weight = torch.randn((n,), dtype=dtype)
        weight_list.append(weight)
        rets.append(
            pool.apply_async(
                split_ar_rmsnorm, args=(tp_size, pp_size, i, x, weight, eps, withGraph)
            )
            # pool.apply_async(run_cu, args=(x, weight, eps, i))
        )
    pool.close()
    pool.join()
    for i in range(tp_size):
        host_rslt = F.rms_norm(
            input=(ref + res_inp[i]),
            normalized_shape=(ref.shape[-1],),
            weight=weight_list[i],
            eps=eps,
        )
        cpu_rslt.append(host_rslt)
    rets = [el.get() for el in rets]
    for out, us in rets:
        msg = f"test_split_ar_rmsnorm: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        # print(cpu_rslt[out.device.index])
        checkAllclose(cpu_rslt[out.device.index], out.to(ref), msg=msg)


@benchmark()
def test_fused_ar_rmsnorm(tp_size, pp_size, shape, dtype, withGraph=False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    cpu_rslt = []
    weight_list = []
    res_inp = []
    # print(type(shape[0]), shape[1], ref.device)
    m = shape[0]
    n = shape[1]
    eps = 1e-6
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        # x = torch.ones(shape, dtype=dtype)
        res_inp.append(x)
        # print(f"device {i}, x[0][0] = {x[0][0]}")
        ref += x
        weight = torch.randn((n,), dtype=dtype)
        weight_list.append(weight)
        rets.append(
            pool.apply_async(
                fused_ar_rmsnorm, args=(tp_size, pp_size, i, x, weight, eps, withGraph)
            )
            # pool.apply_async(run_cu, args=(x, weight, eps, i))
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
    for out, us in rets:
        msg = f"test_fused_ar_rmsnorm: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        # print(cpu_rslt[out.device.index])
        checkAllclose(cpu_rslt[out.device.index], out.to(ref), msg=msg)
        # checkAllclose(ref, out.to(ref), msg=msg)


def acc_test(tp_size, pp_size, shape, dtype):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    cpu_rslt = []
    weight_list = []
    # print(type(shape[0]), shape[1], ref.device)
    m = shape[0]
    n = shape[1]
    eps = 1e-6
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        ref += x
        weight = torch.randn((n,), dtype=dtype)
        weight_list.append(weight)
        rets.append(
            pool.apply_async(
                get_acc_value_only, args=(tp_size, pp_size, i, x, weight, eps, 1)
            )
        )
    pool.close()
    pool.join()

    ar_rslt = []
    for i, ret in enumerate(rets):
        rslt = ret.get()
        ar_rslt.append(rslt)
    for i in ar_rslt:
        checkAllclose(ref, i.to(ref))


def acc_test_cudagraph_on(tp_size, pp_size, shape, dtype, loop_time=1):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    cpu_rslt = []
    weight_list = []
    # print(type(shape[0]), shape[1], ref.device)
    m = shape[0]
    n = shape[1]
    eps = 1e-6
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        ref += x
        weight = torch.randn((n,), dtype=dtype)
        weight_list.append(weight)
        rets.append(
            pool.apply_async(
                get_acc_value_with_cudagraph,
                args=(tp_size, pp_size, i, x, weight, eps, loop_time),
            )
        )
    pool.close()
    pool.join()

    ar_rslt = []
    for i, ret in enumerate(rets):
        rslt = ret.get()
        ar_rslt.append(rslt)
    for i in ar_rslt:
        checkAllclose(ref, i.to(ref))


# def acc_test(tp_size, pp_size, shape, dtype):
#     os.environ["MASTER_ADDR"] = "127.0.0.1"
#     os.environ["MASTER_PORT"] = "49373"
#     pool = Pool(processes=tp_size)
#     ref = torch.zeros(shape, dtype=dtype)
#     rets = []
#     cpu_rslt = []
#     weight_list = []
#     # print(type(shape[0]), shape[1], ref.device)
#     m = shape[0]
#     n = shape[1]
#     eps = 1e-6
#     for i in range(tp_size):
#         x = torch.randn(shape, dtype=dtype)
#         print(f"device {i}, x[0][0] = {x[0][0]}")
#         ref += x
#         weight = torch.randn((n,), dtype=dtype)
#         weight_list.append(weight)
#         rets.append(
#             pool.apply_async(get_acc_value_only, args=(tp_size, pp_size, i, x, weight, eps))
#         )
#     pool.close()
#     pool.join()
#     for i in range(tp_size):
#         host_rslt = F.rms_norm(
#             input=ref, normalized_shape=(ref.shape[-1],), weight=weight_list[i], eps=eps
#         )
#         cpu_rslt.append(host_rslt)
#
#     ar_rslt = []
#     for i, ret in enumerate(rets):
#         rslt = ret.get()
#         ar_rslt.append(rslt)
#     for i in range(len(ar_rslt)):
#         checkAllclose(cpu_rslt[i], ar_rslt[i].to(ref))

l_dtype = ["bf16"]
l_shape = [
    # (4096, 2048)
    (64, 7168)
    # (64, 512 * 99)
    # (16, 512)
]
l_tp = [8]
l_pp = [1]
l_graph = [True, False]

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
    nargs="?",
    const=None,
    default=None,
    help="shape. e.g. -s 128,8192",
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


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.shape is not None:
        l_shape = [args.shape]
    if args.tp is not None:
        l_tp = [args.tp]
    if args.pp is not None:
        l_pp = [args.pp]
    if args.graphon is not None:
        print(args.graphon)
        l_graph = [args.graphon]
    for dtype, shape, tp, pp, graph_on in itertools.product(
        l_dtype, l_shape, l_tp, l_pp, l_graph
    ):
        test_split_ar_rmsnorm(tp, pp, shape, dtype, withGraph=graph_on)
        test_fused_ar_rmsnorm(tp, pp, shape, dtype, withGraph=graph_on)
