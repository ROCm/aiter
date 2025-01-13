import torch
import torch.distributed as dist
import torch.nn.functional as F
import os
import ater
from ater.test_common import checkAllclose, perftest, tensor_dump, tensor_load
from ater.dist.parallel_state import (graph_capture)
import sys
import traceback
import logging
import multiprocessing as mp
logger = logging.getLogger("ater")

NORM = 0
DUMP = 1
VERIFY = 2
# debug_mode = DUMP
# debug_mode = VERIFY
debug_mode = NORM


def run_commun_fwd(tp_size, pp_size,  gpuID, input, withGraph=False):
    try:
        device = torch.device(f"cuda:{gpuID}")
        torch.cuda.set_device(device)
        ater.init_dist_env(tp_size, gpuID)

        input = input.to(device)
        x = torch.empty_like(input)
        torch.cuda.synchronize()
        dist.barrier()

        if withGraph:
            @perftest()
            def run_ca(graph):
                return graph.replay()

            graph = torch.cuda.CUDAGraph()
            with graph_capture() as gc:
                with torch.cuda.graph(graph, stream=gc.stream):
                    # run inplace here, to test accuracy, we need this
                    x.copy_(input)
                    out = ater.all_reduce_asm(x)
            torch.cuda.synchronize()
            out.fill_(0)
            dist.barrier()

            _, us = run_ca(graph)
        else:
            @perftest()
            def run_ca(x):
                return ater.all_reduce_asm(x)
            out, us = run_ca(x)
        torch.cuda.synchronize()
        print(gpuID, 'finished')
        out = out.cpu()
    except Exception as e:
        logger.error('\n-->[History]: {}'.format(
            ''.join(traceback.format_exception(*sys.exc_info()))
        ))
    finally:
        ater.destroy_dist_env()
        return out, us


def test_communication(tp_size, shape, dtype,  withGraph=False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    xs = []
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        xs.append(x)
        ref += x
        rets.append(pool.apply_async(run_commun_fwd,
                                     args=(tp_size, 1, i, x, withGraph)))
    pool.close()
    pool.join()

    rets = [el.get() for el in rets]
    for out, us in rets:
        msg = f'test_allreduce_custom: {shape=} {dtype=} {withGraph=} {us:.2f}'
        checkAllclose(ref, out.to(ref), msg=msg)
        break


def run_all_reduce_rmsnorm(tp_size, pp_size,  gpuID, input, residual_in, weight, bias, epsilon, withGraph=False):
    try:
        device = torch.device(f"cuda:{gpuID}")
        torch.cuda.set_device(device)
        ater.init_dist_env(tp_size, gpuID)

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
                    out, residual_out = ater.all_reduce_rmsnorm(
                        input, residual_in, weight, bias, epsilon)
            torch.cuda.synchronize()
            out.fill_(0)
            residual_out.fill_(0)

            _, us = run_ca(graph)
        else:
            @perftest()
            def run_ca(*args):
                return ater.all_reduce_rmsnorm(*args)
            (out, residual_out), us = run_ca(
                input, residual_in, weight, bias, epsilon)
        torch.cuda.synchronize()
        print(f'{gpuID=} finished')
        out = out.cpu()
        residual_out = residual_out.cpu()
    except Exception as e:
        logger.error('\n-->[History]: {}'.format(
            ''.join(traceback.format_exception(*sys.exc_info()))
        ))
    finally:
        ater.destroy_dist_env()
        return (out, residual_out), us


def test_all_reduce_rmsnorm(tp_size, shape, dtype,  withGraph=False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    mp.set_start_method('spawn', force=True)

    res_in = torch.randn(shape, dtype=dtype)
    weight = torch.randn(shape[-1], dtype=dtype)
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
        rets.append(pool.apply_async(run_all_reduce_rmsnorm,
                                     args=(tp_size, 1, i, x, res_in, weight, bias, epsilon, withGraph)))
    pool.close()
    pool.join()

    ref_res = ref+res_in
    ref_out = F.rms_norm(
        input=ref_res,
        normalized_shape=(ref_res.shape[-1],),
        weight=weight,
        # bias=bias,
        eps=epsilon
    )+bias

    rets = [el.get() for el in rets]
    for (out, residual_out), us in rets:
        msg = f'test_all_reduce_rmsnorm: {shape=} {dtype=} {withGraph=} {us:.2f}'
        print(msg)
        checkAllclose(ref_res, residual_out, msg='residual out')
        checkAllclose(ref_out, out, msg='norm out')
        break
    if debug_mode == DUMP:
        for i, el in enumerate(xs):
            tensor_dump(el, f'input{i}')
        tensor_dump(res_in, f'res_in')
        tensor_dump(weight, f'weight')
        tensor_dump(bias, f'bias')
        tensor_dump(ref_res, f'res_out')
        tensor_dump(ref_out, f'output')


if __name__ == '__main__':
    mp.freeze_support()
    for dtype in [torch.bfloat16]:
        for shape in [(128, 8192)]:
            # test_communication(8, shape, dtype, withGraph=False)
            test_communication(8, shape, dtype, withGraph=True)

    print('start test test_communication\n')
    for dtype in [torch.bfloat16]:
        for shape in [(128, 8192)]:
            test_all_reduce_rmsnorm(8, shape, dtype, withGraph=False)
            # test_all_reduce_rmsnorm(8, shape, dtype, withGraph=True)
