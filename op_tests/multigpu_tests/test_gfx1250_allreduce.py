# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Test for gfx1250 (MI450) dedicated allreduce kernel (ar_gfx1250_naive_unroll4).
# Only tests tp2 and tp4 configurations.
#
# Known MI450 issues:
#   - hipExtMallocWithFlags with hipDeviceMallocUncached may fail or produce
#     buffers that cannot be shared via hipIpcGetMemHandle.
#   - Multi-GPU IPC handle broadcast may hang if the above allocation fails
#     silently (returns success but produces an unusable handle).
#
# This test wraps initialization in a timeout and provides clear diagnostics.

import argparse
import os
import sys
from multiprocessing import Pool, TimeoutError as MpTimeoutError, freeze_support, set_start_method

import torch
import torch.distributed as dist
import pandas as pd

from aiter import dtypes
from aiter import logger
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import benchmark, checkAllclose, perftest

set_start_method("spawn", force=True)

_INIT_TIMEOUT_SEC = 120


def _get_gpu_arch(device_idx: int = None) -> str:
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    return getattr(props, "gcnArchName", "")


def _is_gfx1250(device_idx: int = 0) -> bool:
    return "gfx1250" in _get_gpu_arch(device_idx)


def _worker(
    tp_size: int,
    rank: int,
    tensor_on_cpu: torch.Tensor,
    distributed_init_method: str,
    with_graph: bool = False,
):
    """Per-rank worker: init custom allreduce and run the test."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        set_custom_all_reduce,
    )
    from aiter.dist.communication_op import tensor_model_parallel_all_reduce

    arch = _get_gpu_arch()
    logger.info("RANK %d: arch=%s, tp_size=%d, init...", rank, arch, tp_size)

    set_custom_all_reduce(True)

    try:
        init_distributed_environment(
            world_size=tp_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
        )
        ensure_model_parallel_initialized(tp_size, 1)
    except RuntimeError as e:
        err_msg = str(e)
        if "hipExtMallocWithFlags" in err_msg or "hipIpc" in err_msg:
            logger.error(
                "RANK %d: IPC initialization failed (likely hipExtMallocWithFlags "
                "or hipIpcGetMemHandle issue on this GPU arch). Error: %s",
                rank, err_msg,
            )
        raise

    x = tensor_on_cpu.to(device)

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    logger.info("RANK %d: initialization complete, running allreduce...", rank)

    if with_graph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                out = tensor_model_parallel_all_reduce(x)
        out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        out = (out, us)
    else:

        @perftest()
        def run_ca(x):
            return tensor_model_parallel_all_reduce(x)

        out = run_ca(x)

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    return out


@benchmark()
def test_gfx1250_allreduce(
    tp_size: int,
    shape: tuple,
    dtype: torch.dtype,
    with_graph: bool = False,
    distributed_init_method: str = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        ref += x
        rets.append(
            pool.apply_async(
                _worker,
                args=(tp_size, i, x, distributed_init_method, with_graph),
            )
        )
    pool.close()

    # Collect results with a per-worker timeout to detect IPC hangs
    results = []
    try:
        for i, r in enumerate(rets):
            results.append(
                r.get(timeout=_INIT_TIMEOUT_SEC)
            )
    except Exception as e:
        pool.terminate()
        pool.join()
        err_msg = str(e)
        if isinstance(e, (MpTimeoutError, TimeoutError)):
            raise RuntimeError(
                f"Worker timed out after {_INIT_TIMEOUT_SEC}s — likely hung "
                f"in IPC handle exchange (hipIpcGetMemHandle/"
                f"hipIpcOpenMemHandle). On MI450, "
                f"hipExtMallocWithFlags(hipDeviceMallocUncached) may produce "
                f"buffers incompatible with hipIpc*. Consider using regular "
                f"torch allocations for the meta/signal buffer."
            ) from e
        raise
    pool.join()

    rets = results
    all_us = [us for _, us in rets]
    max_err = 0.0
    for out, us in rets:
        msg = (
            f"test_gfx1250_allreduce: tp={tp_size} {shape=} {dtype=} "
            f"{with_graph=} {us:>8.2f}"
        )
        err = checkAllclose(ref, out.to(ref), msg=msg)
        max_err = max(max_err, err)
    return {
        "min_us": min(all_us),
        "max_us": max(all_us),
        "err": max_err,
    }


l_dtype = ["fp16", "bf16"]
l_shape = [
    (1, 7168),
    (2, 7168),
    (1, 8192),
    (128, 8192),
    (512, 8192),
]
l_tp_size = [2, 4]

parser = argparse.ArgumentParser(
    description="Test gfx1250 (MI450) dedicated allreduce kernel"
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=["fp16", "bf16"],
    default=None,
    help="data type (default: test both fp16 and bf16)",
)
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    default=None,
    help="shape, e.g. -s 128,8192",
)
parser.add_argument(
    "-t",
    "--tp-size",
    type=int,
    choices=[2, 4],
    default=None,
    help="tensor parallel size (default: test both 2 and 4)",
)
parser.add_argument(
    "-g",
    "--with-graph",
    type=lambda x: str(x).lower() in ["true", "1", "yes"],
    default=False,
    help="use CUDA graph (default: False)",
)
parser.add_argument(
    "--check-arch",
    action="store_true",
    help="check if running on gfx1250 and warn if not",
)

if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()

    if args.check_arch:
        torch.cuda.set_device(0)
        if not _is_gfx1250():
            arch = _get_gpu_arch(0)
            logger.warning(
                "Not running on gfx1250 (detected: %s). "
                "The kernel will NOT dispatch to ar_gfx1250_naive_unroll4 — "
                "test will exercise the generic allreduce path instead.",
                arch,
            )

    num_gpus = torch.cuda.device_count()
    if args.dtype is None:
        test_dtypes = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        test_dtypes = [dtypes.d_dtypes[args.dtype]]
    if args.shape is not None:
        test_shapes = [args.shape]
    else:
        test_shapes = l_shape
    if args.tp_size is not None:
        test_tp_sizes = [args.tp_size]
    else:
        test_tp_sizes = [tp for tp in l_tp_size if tp <= num_gpus]

    if not test_tp_sizes:
        logger.error("Not enough GPUs: need at least 2, have %d", num_gpus)
        sys.exit(1)

    df = []
    for tp_size in test_tp_sizes:
        if tp_size > num_gpus:
            logger.warning(
                "Skipping tp=%d: only %d GPUs available", tp_size, num_gpus
            )
            continue
        for dtype in test_dtypes:
            for shape in test_shapes:
                ret = test_gfx1250_allreduce(
                    tp_size,
                    shape,
                    dtype,
                    with_graph=args.with_graph,
                    distributed_init_method=get_distributed_init_method(
                        get_ip(), get_open_port()
                    ),
                )
                df.append(ret)

    df = pd.DataFrame(df)
    show_cols = [
        "tp_size",
        "shape",
        "dtype",
        "withGraph",
        "min_us",
        "max_us",
        "err",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    logger.info(
        "gfx1250 allreduce summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
