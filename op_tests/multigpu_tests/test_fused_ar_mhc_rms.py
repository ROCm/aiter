# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method
from statistics import median
from typing import Optional

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_mhc_fused_post_pre_rmsnorm,
)
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import benchmark, checkAllclose, perftest

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

JIT_WARMUP_ITERS = 5
BENCH_WARMUP = 2
BENCH_ITERS = 101

EXTRA_ARGS = {
    "rms_eps": 1e-6,
    "hc_pre_eps": 1e-6,
    "hc_sinkhorn_eps": 1e-6,
    "hc_post_mult_value": 2.0,
    "sinkhorn_repeat": 20,
}


def _make_inputs(
    m: int,
    hidden_size: int,
    rank: int,
    device: torch.device,
    *,
    hc_mult: int = 4,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260611)
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    base_layer_input = torch.randn(m, hidden_size, dtype=dtypes.bf16, device=device)
    return {
        "layer_input": base_layer_input * float(rank + 1),
        "residual_in": torch.randn(
            m, hc_mult, hidden_size, dtype=dtypes.bf16, device=device
        ),
        "post_layer_mix": torch.randn(m, hc_mult, 1, dtype=dtypes.fp32, device=device),
        "comb_res_mix": torch.randn(
            m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device
        ),
        "fn": torch.randn(
            hc_mult3, hc_mult * hidden_size, dtype=dtypes.fp32, device=device
        ),
        "hc_scale": torch.randn((3,), dtype=dtypes.fp32, device=device) * 0.1,
        "hc_base": torch.randn((hc_mult3,), dtype=dtypes.fp32, device=device) * 0.1,
        "norm_weight": torch.randn(hidden_size, dtype=dtypes.bf16, device=device),
    }


def _run_split_mhc(tensors: dict[str, torch.Tensor], *, force_fused: bool):
    reduced = tensor_model_parallel_all_reduce(tensors["layer_input"])
    return aiter.mhc_fused_post_pre(
        reduced,
        tensors["residual_in"],
        tensors["post_layer_mix"],
        tensors["comb_res_mix"],
        tensors["fn"],
        tensors["hc_scale"],
        tensors["hc_base"],
        norm_weight=tensors["norm_weight"],
        norm_eps=1e-6,
        force_fused=force_fused,
        **EXTRA_ARGS,
    )


def _run_fused_mhc(tensors: dict[str, torch.Tensor], *, force_fused: bool):
    return tensor_model_parallel_fused_allreduce_mhc_fused_post_pre_rmsnorm(
        tensors["layer_input"],
        tensors["residual_in"],
        tensors["post_layer_mix"],
        tensors["comb_res_mix"],
        tensors["fn"],
        tensors["hc_scale"],
        tensors["hc_base"],
        tensors["norm_weight"],
        norm_eps=1e-6,
        force_fused=force_fused,
        **EXTRA_ARGS,
    )


def _max_mhc_err(ref, out, *, m: int, rank: int, tag: str) -> float:
    return max(
        checkAllclose(ref[0], out[0], msg=f"{tag}/post_mix m={m} rank={rank}"),
        checkAllclose(ref[1], out[1], msg=f"{tag}/comb_mix m={m} rank={rank}"),
        checkAllclose(ref[2], out[2], msg=f"{tag}/layer_input m={m} rank={rank}"),
        checkAllclose(ref[3], out[3], msg=f"{tag}/next_residual m={m} rank={rank}"),
    )


def _jit_warmup(
    tensors: dict[str, torch.Tensor],
    *,
    force_fused: bool,
    run_split: bool,
    run_fused: bool,
    jit_warmup_iters: int,
):
    for _ in range(jit_warmup_iters):
        if run_split:
            _run_split_mhc(tensors, force_fused=force_fused)
        if run_fused:
            _run_fused_mhc(tensors, force_fused=force_fused)
    torch.cuda.synchronize()


def _bench_path(fn, *, with_graph: bool, graph_holder: dict):
    if with_graph:
        if "graph" not in graph_holder:
            graph = torch.cuda.CUDAGraph()
            with graph_capture() as gc:
                with torch.cuda.graph(graph, stream=gc.stream):
                    fn()
            graph_holder["graph"] = graph

        graph = graph_holder["graph"]

        @perftest(num_warmup=BENCH_WARMUP, num_iters=BENCH_ITERS)
        def run_replay():
            graph.replay()

        _, us = run_replay()
        return us

    @perftest(num_warmup=BENCH_WARMUP, num_iters=BENCH_ITERS)
    def run_eager():
        return fn()

    _, us = run_eager()
    return us


def ar_mhc_rmsnorm_worker(
    tp_size,
    pp_size,
    rankID,
    shape,
    force_fused,
    withGraph,
    distributed_init_method,
    run_split,
    run_fused,
    jit_warmup_iters,
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    logger.info("RANK: %s tp=%s init_process_group...", rankID, tp_size)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    m, hidden_size = shape
    tensors = _make_inputs(m, hidden_size, rankID, device=device)

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    _jit_warmup(
        tensors,
        force_fused=force_fused,
        run_split=run_split,
        run_fused=run_fused,
        jit_warmup_iters=jit_warmup_iters,
    )

    err = 0.0
    if run_fused:
        ref = _run_split_mhc(tensors, force_fused=force_fused)
        fused_out = _run_fused_mhc(tensors, force_fused=force_fused)
        err = _max_mhc_err(ref, fused_out, m=m, rank=rankID, tag="fused_ar_mhc")

    split_us = None
    fused_us = None
    split_graph_holder: dict = {}
    fused_graph_holder: dict = {}

    if run_split:
        split_us = _bench_path(
            lambda: _run_split_mhc(tensors, force_fused=force_fused),
            with_graph=withGraph,
            graph_holder=split_graph_holder,
        )

    if run_fused:
        fused_us = _bench_path(
            lambda: _run_fused_mhc(tensors, force_fused=force_fused),
            with_graph=withGraph,
            graph_holder=fused_graph_holder,
        )

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    return {
        "split_us": split_us,
        "fused_us": fused_us,
        "err": err,
    }


def _aggregate_us(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {}
    return {
        f"{prefix}_median_us": median(values),
        f"{prefix}_min_us": min(values),
        f"{prefix}_max_us": max(values),
    }


@benchmark()
def test_ar_mhc_rmsnorm(
    tp_size,
    pp_size,
    shape,
    force_fused,
    withGraph=False,
    distributed_init_method: Optional[str] = None,
    run_tests=None,
    jit_warmup_iters: int = JIT_WARMUP_ITERS,
):
    if run_tests is None:
        run_tests = ("split", "fused")
    run_split = "split" in run_tests
    run_fused = "fused" in run_tests

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    rets = []
    for rank in range(tp_size):
        rets.append(
            pool.apply_async(
                ar_mhc_rmsnorm_worker,
                args=(
                    tp_size,
                    pp_size,
                    rank,
                    shape,
                    force_fused,
                    withGraph,
                    distributed_init_method,
                    run_split,
                    run_fused,
                    jit_warmup_iters,
                ),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]

    m, hidden_size = shape
    out: dict = {}
    if run_split:
        split_us = [r["split_us"] for r in rets if r["split_us"] is not None]
        out.update(_aggregate_us(split_us, "split"))
    if run_fused:
        fused_us = [r["fused_us"] for r in rets if r["fused_us"] is not None]
        out.update(_aggregate_us(fused_us, "fused"))
        out["fused_err"] = max(r["err"] for r in rets)

    for rank, r in enumerate(rets):
        msg = (
            f"test_ar_mhc_rmsnorm: shape={(m, hidden_size)} tp={tp_size} "
            f"force_fused={force_fused} withGraph={withGraph} rank={rank} "
            f"split_us={r['split_us']} fused_us={r['fused_us']}"
        )
        if run_fused:
            logger.info("%s err=%s", msg, r["err"])
        else:
            logger.info("%s", msg)

    if run_split and run_fused:
        split_med = out.get("split_median_us")
        fused_med = out.get("fused_median_us")
        if split_med and fused_med and split_med > 0:
            out["speedup_pct"] = (split_med - fused_med) / split_med * 100.0

    return out


try:
    import pytest

    def test_fused_ar_mhc_rms_tp2_smoke():
        if torch.cuda.device_count() < 2:
            pytest.skip(f"requires >=2 GPUs, got {torch.cuda.device_count()}")
        ret = test_ar_mhc_rmsnorm(
            tp_size=2,
            pp_size=1,
            shape=(16, 4096),
            force_fused=True,
            withGraph=True,
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
            run_tests=("split", "fused"),
        )
        assert ret["fused_err"] == 0

except ImportError:
    pass


l_shape = [
    (1, 4096),
    (2, 4096),
    (4, 4096),
    (16, 4096),
    (32, 4096),
    (128, 4096),
    (1024, 4096),
    (2048, 4096),
    (8192, 4096),
]
l_tp = [2]
l_pp = [1]
l_graph = [True]
l_force_fused = [True]

parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    nargs="*",
    default=None,
    help="shape(s) as M,hidden_size. e.g. -s 16,4096 128,4096",
)
parser.add_argument(
    "-t",
    "--tp",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="tp size (single). e.g. -t 2. Ignored if --tp-sizes is set.",
)
parser.add_argument(
    "--tp-sizes",
    type=lambda s: [int(x) for x in s.split(",") if x.strip()],
    default=None,
    help="comma-separated TP sizes from {2, 4, 6, 8}, e.g. --tp-sizes 2,4,8",
)
parser.add_argument(
    "-p",
    "--pp",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="pp size. e.g. -p 1",
)
parser.add_argument(
    "-g",
    "--graphon",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="use CUDA graph: 1=on (default), 0=off. e.g. -g 0",
)
parser.add_argument(
    "--jit-warmup",
    type=int,
    default=JIT_WARMUP_ITERS,
    help=f"JIT warmup iterations per path before timing (default {JIT_WARMUP_ITERS})",
)
parser.add_argument(
    "--force-fused",
    action="store_true",
    help="use force_fused=True in mhc_fused_post_pre / fused AR+MHC path",
)
l_test_types = ["split", "fused"]
parser.add_argument(
    "--test",
    type=str,
    choices=l_test_types,
    nargs="*",
    default=None,
    help="test type(s) to run. e.g. --test split fused",
)


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.shape is not None:
        l_shape = args.shape
    if args.tp_sizes is not None:
        l_tp = args.tp_sizes
    elif args.tp is not None:
        l_tp = [args.tp]
    if args.pp is not None:
        l_pp = [args.pp]
    if args.graphon is not None:
        l_graph = [bool(args.graphon)]
    if args.force_fused:
        l_force_fused = [True]
    run_tests = tuple(args.test if args.test else l_test_types)
    df = []
    for shape, tp, pp, graph_on, force_fused in itertools.product(
        l_shape, l_tp, l_pp, l_graph, l_force_fused
    ):
        row = {
            "tp_size": tp,
            "shape": shape,
            "force_fused": force_fused,
            "withGraph": graph_on,
            "jit_warmup": args.jit_warmup,
        }
        ret = test_ar_mhc_rmsnorm(
            tp,
            pp,
            shape,
            force_fused,
            withGraph=graph_on,
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
            run_tests=run_tests,
            jit_warmup_iters=args.jit_warmup,
        )
        row.update(ret)
        df.append(row)
    df = pd.DataFrame(df)
    show_cols = [
        "tp_size",
        "shape",
        "force_fused",
        "withGraph",
        "jit_warmup",
        "split_median_us",
        "split_min_us",
        "split_max_us",
        "fused_median_us",
        "fused_min_us",
        "fused_max_us",
        "fused_err",
        "speedup_pct",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    logger.info(
        "fused allreduce mhc rmsnorm summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
