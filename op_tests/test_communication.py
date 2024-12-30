import torch
import torch.distributed as dist
import os
import ater
from ater.test_common import checkAllclose, perftest, tensor_dump, tensor_load
from ater.dist.parallel_state import (graph_capture)
import sys
import traceback
import logging
import multiprocessing as mp
logger = logging.getLogger("ater")

MEMOP_S_READ = 0
MEMOP_S_WRITE = 1
MEMOP_ATOMIC_S_INC = 2
MEMOP_ATOMIC_V_INC = 3
MEMOP_WAIT_VALUE = 4
MEMOP_V_READ = 5
MEMOP_V_WRITE = 6


def run_commun_fwd(tp_size, pp_size,  gpuID, x, withGraph=False):
    try:
        device = torch.device(f"cuda:{gpuID}")
        torch.cuda.set_device(device)
        x = x.to(device)

        ater.init_dist_env_asm(tp_size, gpuID)

        if withGraph:
            @perftest()
            def run_ca(graph):
                return graph.replay()

            graph = torch.cuda.CUDAGraph()
            with graph_capture() as gc:
                with torch.cuda.graph(graph, stream=gc.stream):
                    # run inplace here, to test accuracy, we need this
                    b = torch.empty_like(x).copy_(x)
                    out = ater.call_all_reduce_asm(b)
            out.fill_(0)
            torch.cuda.synchronize()

            _, us = run_ca(graph)
        else:
            @perftest()
            def run_ca(x):
                return ater.call_all_reduce_asm(x)
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
    # for i, (out, us) in enumerate(rets):
    #     ref = xs[i]
    #     msg = f'test_allreduce_custom: {shape=} {dtype=} {withGraph=}'
    #     checkAllclose(ref, out.to(ref), msg=msg)


if __name__ == '__main__':
    mp.freeze_support()
    for dtype in [torch.bfloat16]:
        for shape in [(128, 8192)]:
            test_communication(8, shape, dtype, withGraph=True)
            # test_communication(8, shape, dtype, withGraph=False)
