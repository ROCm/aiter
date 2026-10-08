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

# gfx1250 has no CK support: the base custom_all_reduce module won't compile with
# CK enabled (ck_tile warp_size is not constexpr on this arch). Force the CK-free
# build here so the JIT build succeeds without a manual `ENABLE_CK=0` prefix.
# Must run before importing torch/aiter — ENABLE_CK is read at aiter import time
# (aiter/jit/core.py). setdefault keeps an explicit `ENABLE_CK=1` override working.
os.environ.setdefault("ENABLE_CK", "0")

from multiprocessing import (
    Pool,
    freeze_support,
    set_start_method,
)
from multiprocessing import (
    TimeoutError as MpTimeoutError,
)

import pandas as pd
import torch
import torch.distributed as dist

from aiter import dtypes, logger
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import checkAllclose, perftest

set_start_method("spawn", force=True)

_INIT_TIMEOUT_SEC = 120
_CASE_TIMEOUT_SEC = 10


def _get_gpu_arch(device_idx: int | None = None) -> str:
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    return getattr(props, "gcnArchName", "")


def _is_gfx1250(device_idx: int = 0) -> bool:
    return "gfx1250" in _get_gpu_arch(device_idx)


def _run_allreduce_case(
    tp_size: int,
    rank: int,
    case_idx: int,
    shape: tuple,
    dtype: torch.dtype,
    with_graph: bool,
    graphs: list,
):
    """All-reduce one ``(shape, dtype)`` on an initialized rank and check it
    against the sum of every rank's input.

    Each rank regenerates all ``tp_size`` inputs from the same ``case_idx``
    seed and keeps its own, so the reference is available locally and only
    scalars go back to the parent. In graph mode the captured graph and its
    buffers are appended to ``graphs`` and must outlive the whole sweep:
    custom all-reduce caches the peer IPC address of every captured buffer by
    its local pointer, so freeing one and capturing a later case at the same
    address would replay with stale peer pointers.
    """
    from aiter.dist.communication_op import tensor_model_parallel_all_reduce
    from aiter.dist.parallel_state import graph_capture

    gen = torch.Generator(device="cuda").manual_seed(case_idx)
    ref = torch.zeros(shape, dtype=dtype, device="cuda")
    for r in range(tp_size):
        xr = torch.randn(shape, dtype=dtype, device="cuda", generator=gen)
        ref += xr
        if r == rank:
            x = xr

    if with_graph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            out = tensor_model_parallel_all_reduce(x)
        out.fill_(0)
        graphs.append((graph, x, out))

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
    else:

        @perftest()
        def run_ca(x):
            return tensor_model_parallel_all_reduce(x)

        out, us = run_ca(x)

    msg = (
        f"test_gfx1250_allreduce: rank={rank} tp={tp_size} {shape=} {dtype=} "
        f"{with_graph=} {us:>8.2f}"
    )
    return {"us": us, "err": checkAllclose(ref, out, msg=msg)}


def _allreduce_sweep(
    tp_size: int,
    rank: int,
    cases: list,
    distributed_init_method: str,
    with_graph: bool = False,
):
    """Per-rank worker: init custom allreduce once and run every
    ``(shape, dtype)`` case.

    Setting up the TP group costs seconds per init vs. microseconds of kernel
    time, so the group is created and torn down once.
    """
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        set_custom_all_reduce,
    )

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
                rank,
                err_msg,
            )
        raise

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    logger.info("RANK %d: initialization complete, running allreduce...", rank)

    graphs = []
    results = [
        _run_allreduce_case(tp_size, rank, case_idx, shape, dtype, with_graph, graphs)
        for case_idx, (shape, dtype) in enumerate(cases)
    ]

    if dist.is_initialized():
        # Drain and align all ranks before freeing IPC buffers / groups.
        torch.cuda.synchronize()
        get_tp_group().barrier()
        torch.cuda.synchronize()
        destroy_model_parallel()
        destroy_distributed_environment()
        graphs.clear()
        torch.cuda.empty_cache()

    return results


def test_gfx1250_allreduce(
    tp_size: int,
    cases: list,
    with_graph: bool = False,
):
    """Sweep ``(shape, dtype)`` cases on one TP group and return one summary
    row per case."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    pool = Pool(processes=tp_size)
    rets = [
        pool.apply_async(
            _allreduce_sweep,
            args=(tp_size, rank, cases, init_method, with_graph),
        )
        for rank in range(tp_size)
    ]
    pool.close()

    # Collect results with a per-worker timeout to detect IPC hangs
    timeout = _INIT_TIMEOUT_SEC + _CASE_TIMEOUT_SEC * len(cases)
    per_rank = []
    try:
        for r in rets:
            per_rank.append(r.get(timeout=timeout))
    except Exception as e:
        pool.terminate()
        pool.join()
        if isinstance(e, (MpTimeoutError, TimeoutError)):
            raise RuntimeError(  # noqa: TRY004
                f"Worker timed out after {timeout}s — likely hung "
                f"in IPC handle exchange (hipIpcGetMemHandle/"
                f"hipIpcOpenMemHandle). On MI450, "
                f"hipExtMallocWithFlags(hipDeviceMallocUncached) may produce "
                f"buffers incompatible with hipIpc*. Consider using regular "
                f"torch allocations for the meta/signal buffer."
            ) from e
        raise
    pool.join()

    rows = []
    for i, (shape, dtype) in enumerate(cases):
        all_us = [results[i]["us"] for results in per_rank]
        rows.append(
            {
                "tp_size": tp_size,
                "shape": shape,
                "dtype": dtype,
                "withGraph": with_graph,
                "min_us": min(all_us),
                "max_us": max(all_us),
                "err": max(results[i]["err"] for results in per_rank),
            }
        )
    return rows


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
    action="append",
    default=None,
    help="shape (repeatable), e.g. -s 16,7168 -s 32,7168",
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
        test_shapes = args.shape
    else:
        test_shapes = l_shape
    if args.tp_size is not None:
        test_tp_sizes = [args.tp_size]
    else:
        test_tp_sizes = [tp for tp in l_tp_size if tp <= num_gpus]

    if not test_tp_sizes:
        logger.error("Not enough GPUs: need at least 2, have %d", num_gpus)
        sys.exit(1)

    cases = [(shape, dtype) for dtype in test_dtypes for shape in test_shapes]
    df = []
    for tp_size in test_tp_sizes:
        if tp_size > num_gpus:
            logger.warning("Skipping tp=%d: only %d GPUs available", tp_size, num_gpus)
            continue
        df.extend(test_gfx1250_allreduce(tp_size, cases, with_graph=args.with_graph))

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
