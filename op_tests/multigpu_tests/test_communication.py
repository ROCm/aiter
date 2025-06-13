# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import torch.nn.functional as F
import os
import aiter
from aiter.test_common import checkAllclose, perftest
from aiter.dist.parallel_state import graph_capture
from aiter import dtypes
import sys
import traceback
import logging
import multiprocessing as mp
import argparse

logger = logging.getLogger("aiter")

NORM = 0
DUMP = 1
VERIFY = 2
# debug_mode = DUMP
# debug_mode = VERIFY
debug_mode = NORM


def run_commun_fwd(tp_size, pp_size, gpuID, input, withGraph=False):
    try:
        device = torch.device(f"cuda:{gpuID}")
        torch.cuda.set_device(device)
        aiter.init_dist_env(tp_size, gpuID)

        input = input.to(device)
        x = torch.empty_like(input)
        torch.cuda.synchronize()
        dist.barrier()

        if withGraph:
            graph = torch.cuda.CUDAGraph()
            with graph_capture() as gc:
                with torch.cuda.graph(graph, stream=gc.stream):
                    # run inplace here, to test accuracy, we need this
                    x.copy_(input)
                    out = aiter.all_reduce_asm(x)
            torch.cuda.synchronize()
            out.fill_(0)
            dist.barrier()

            @perftest()
            def run_ca():
                return graph.replay()

            _, us = run_ca()
        else:

            @perftest()
            def run_ca(x):
                return aiter.all_reduce_asm(x)

            out, us = run_ca(x)
        torch.cuda.synchronize()
        print(gpuID, "finished")
        out = out.cpu()
    except Exception:
        logger.error(
            "\n-->[History]: {}".format(
                "".join(traceback.format_exception(*sys.exc_info()))
            )
        )
    finally:
        aiter.destroy_dist_env()
        return out, us


def test_communication(tp_size, shape, dtype, withGraph=False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    xs = []
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        xs.append(x)
        ref += x
        rets.append(
            pool.apply_async(run_commun_fwd, args=(tp_size, 1, i, x, withGraph))
        )
    pool.close()
    pool.join()

    rets = [el.get() for el in rets]
    for out, us in rets:
        msg = f"test_allreduce_custom: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        checkAllclose(ref, out.to(ref), msg=msg)
        break
    print(
        f"finished test_allreduce_custom: {tp_size=}, {shape=}, {dtype=}, {withGraph=}\n"
    )


def run_all_reduce_rmsnorm(
    tp_size, pp_size, gpuID, input, residual_in, weight, bias, epsilon, withGraph=False
):
    try:
        device = torch.device(f"cuda:{gpuID}")
        torch.cuda.set_device(device)
        aiter.init_dist_env(tp_size, gpuID)

        input = input.to(device)
        residual_in = residual_in.to(device)
        weight = weight.to(device)
        bias = bias.to(device)

        if withGraph:

            @perftest()
            def run_ca(graph):
                return graph.replay()

            graph = torch.cuda.CUDAGraph()
            with graph_capture() as gc:
                with torch.cuda.graph(graph, stream=gc.stream):
                    out, residual_out = aiter.all_reduce_rmsnorm(
                        input, residual_in, weight, bias, epsilon
                    )
            torch.cuda.synchronize()
            out.fill_(0)
            residual_out.fill_(0)

            _, us = run_ca(graph)
        else:

            @perftest()
            def run_ca(*args):
                return aiter.all_reduce_rmsnorm(*args)

            (out, residual_out), us = run_ca(input, residual_in, weight, bias, epsilon)
        torch.cuda.synchronize()
        print(f"{gpuID=} finished")
        out = out.cpu()
        residual_out = residual_out.cpu()
    except Exception:
        logger.error(
            "\n-->[History]: {}".format(
                "".join(traceback.format_exception(*sys.exc_info()))
            )
        )
    finally:
        aiter.destroy_dist_env()
        return (out, residual_out), us


def run_all_reduce_rmsnorm_quant(
    tp_size,
    pp_size,
    gpuID,
    input,
    residual_in,
    xscale,
    weight,
    bias,
    epsilon,
    withGraph=False,
):
    try:
        device = torch.device(f"cuda:{gpuID}")
        torch.cuda.set_device(device)
        aiter.init_dist_env(tp_size, gpuID)

        input = input.to(device)
        residual_in = residual_in.to(device)
        xscale = xscale.to(device)
        weight = weight.to(device)
        bias = bias.to(device)

        if withGraph:

            @perftest()
            def run_ca(graph):
                return graph.replay()

            graph = torch.cuda.CUDAGraph()
            with graph_capture() as gc:
                with torch.cuda.graph(graph, stream=gc.stream):
                    out, residual_out, ysacle = aiter.all_reduce_rmsnorm_quant(
                        input, residual_in, xscale, weight, bias, epsilon
                    )
            torch.cuda.synchronize()
            out.fill_(0)
            residual_out.fill_(0)

            _, us = run_ca(graph)
        else:

            @perftest()
            def run_ca(*args):
                return aiter.all_reduce_rmsnorm_quant(*args)

            (out, residual_out, ysacle), us = run_ca(
                input, residual_in, xscale, weight, bias, epsilon
            )
        torch.cuda.synchronize()
        print(f"{gpuID=} finished")
        out = out.cpu()
        residual_out = residual_out.cpu()
        ysacle = ysacle.cpu()
    except Exception:
        logger.error(
            "\n-->[History]: {}".format(
                "".join(traceback.format_exception(*sys.exc_info()))
            )
        )
    finally:
        aiter.destroy_dist_env()
        return (out, residual_out, ysacle), us


def test_all_reduce_rmsnorm(tp_size, shape, dtype, withGraph=False, perTKQuant=False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    mp.set_start_method("spawn", force=True)

    res_in = torch.randn(shape, dtype=dtype)
    weight = torch.randn(shape[-1], dtype=dtype)
    xscale = torch.randn(shape[-1], dtype=dtypes.fp32)
    xscale.fill_(1.0)
    bias = torch.randn(shape[-1], dtype=dtype)
    epsilon = 1e-5

    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    xs = []
    pool = mp.Pool(processes=tp_size)
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        xs.append(x)
        ref += x
        if perTKQuant:
            rets.append(
                pool.apply_async(
                    run_all_reduce_rmsnorm_quant,
                    args=(
                        tp_size,
                        1,
                        i,
                        x,
                        res_in,
                        xscale,
                        weight,
                        bias,
                        epsilon,
                        withGraph,
                    ),
                )
            )
        else:
            rets.append(
                pool.apply_async(
                    run_all_reduce_rmsnorm,
                    args=(tp_size, 1, i, x, res_in, weight, bias, epsilon, withGraph),
                )
            )
    pool.close()
    pool.join()

    ref_res = ref + res_in
    ref_out = (
        F.rms_norm(
            input=ref_res,
            normalized_shape=(ref_res.shape[-1],),
            weight=weight,
            # bias=bias,
            eps=epsilon,
        )
        + bias
    )
    yscale = None
    if perTKQuant:
        ref_out, yscale = aiter.pertoken_quant(ref_out, x_scale=xscale)

    rets = [el.get() for el in rets]
    refs = [ref_out, ref_res, yscale]
    names = ["norm out", "residual out", "y scale"]
    for ret, us in rets:
        print(f"test_all_reduce_rmsnorm: {shape=} {dtype=} {withGraph=} {us=:>8.2f}")
        for i, el in enumerate(ret):
            checkAllclose(refs[i], el, msg=names[i])
        break
    # if debug_mode == DUMP:
    #     for i, el in enumerate(xs):
    #         tensor_dump(el, f'input{i}', dir='./debug')
    #     tensor_dump(res_in, f'res_in', dir='./debug')
    #     tensor_dump(weight, f'weight', dir='./debug')
    #     tensor_dump(bias, f'bias', dir='./debug')
    #     tensor_dump(xscale, f'xscale', dir='./debug')
    #     tensor_dump(yscale, f'yscale', dir='./debug')
    #     tensor_dump(ref_res, f'res_out', dir='./debug')
    #     tensor_dump(ref_out, f'output', dir='./debug')
    print(
        f"finished test_all_reduce_rmsnorm: {tp_size=}, {shape=}, {dtype=}, {withGraph=}, {perTKQuant=}\n"
    )


l_dtype = ["bf16"]
l_shape = [(128, 8192)]

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
    choices=l_shape,
    nargs="?",
    const=None,
    default=None,
    help="shape",
)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.shape is not None:
        l_shape = [args.shape]

    mp.freeze_support()
    # for dtype in [dtypes.bf16]:
    #     for shape in [(128, 8192)]:
    #         # test_communication(8, shape, dtype, withGraph=False)
    #         test_communication(8, shape, dtype, withGraph=True)

    print("start test test_communication\n")
    for dtype in l_dtype:
        for shape in l_shape:
            # test_all_reduce_rmsnorm(8, shape, dtype, withGraph=False)
            test_all_reduce_rmsnorm(8, shape, dtype, withGraph=False, perTKQuant=True)
