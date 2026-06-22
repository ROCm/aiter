# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multigpu tests: split (AR + mhc_post) vs fused AR+mhc_post epilogue."""

from __future__ import annotations

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method
from typing import Optional

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
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
from aiter.ops.custom_all_reduce import (
    fused_allreduce_mhc_post_only,
    fused_allreduce_mhc_post_one_stage,
    fused_allreduce_mhc_post_split,
)
from aiter.test_common import benchmark, checkAllclose

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

WARMUP = 5
BENCH_WARMUP = 2
BENCH_ITERS = 101
DEFAULT_SHAPES = (
    (1, 4096),
    (2, 4096),
    (4, 4096),
    (16, 4096),
    (32, 4096),
    (128, 4096),
    (1024, 4096),
    (2048, 4096),
    (8192, 4096),
)


def _make_inputs(m: int, hidden_size: int, rank: int, device: torch.device):
    torch.manual_seed(20260617)
    hc_mult = 4
    base_layer_input = torch.randn(
        m, hidden_size, dtype=aiter.dtypes.bf16, device=device
    )
    return {
        "layer_input": base_layer_input * float(rank + 1),
        "residual_in": torch.randn(
            m, hc_mult, hidden_size, dtype=aiter.dtypes.bf16, device=device
        ),
        "post_layer_mix": torch.randn(
            m, hc_mult, 1, dtype=aiter.dtypes.fp32, device=device
        ),
        "comb_res_mix": torch.randn(
            m, hc_mult, hc_mult, dtype=aiter.dtypes.fp32, device=device
        ),
    }


def _event_mean_us(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end) * 1000.0)
    return sum(latencies) / len(latencies)


def _profile_worker(
    tp_size: int,
    rank_id: int,
    m: int,
    hidden_size: int,
    with_graph: bool,
    init_method: str,
    *,
    run_correctness: bool,
    breakdown: bool = False,
    compare_stages: bool = False,
):
    device = torch.device(f"cuda:{rank_id}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rank_id,
        distributed_init_method=init_method,
    )
    ensure_model_parallel_initialized(tp_size, 1)
    tensors = _make_inputs(m, hidden_size, rank_id, device)
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    ca_comm = get_tp_group().device_communicator.ca_comm
    next_residual_split = torch.empty_like(tensors["residual_in"])
    next_residual_fused = torch.empty_like(tensors["residual_in"])
    next_residual_1stage = torch.empty_like(tensors["residual_in"])
    next_residual_split_fused = torch.empty_like(tensors["residual_in"])
    reduced_buf = torch.empty_like(tensors["layer_input"])
    post_mix = tensors["post_layer_mix"]
    if post_mix.ndim == 3:
        post_mix = post_mix.squeeze(-1)

    def _reg():
        if ca_comm is None or ca_comm.disabled:
            return 0, 0
        return ca_comm._pool["input"].data_ptr, ca_comm._pool["input"].max_size

    def split_ar_post():
        reduced = tensor_model_parallel_all_reduce(tensors["layer_input"])
        aiter.mhc_post(
            next_residual_split,
            reduced,
            tensors["residual_in"],
            post_mix,
            tensors["comb_res_mix"],
        )

    def fused_ar_post(*, registered: bool):
        reg_ptr, reg_bytes = (0, 0) if registered else _reg()
        fused_allreduce_mhc_post_only(
            ca_comm._ptr,
            tensors["layer_input"],
            next_residual_fused,
            tensors["residual_in"],
            tensors["post_layer_mix"],
            tensors["comb_res_mix"],
            reg_ptr=reg_ptr,
            reg_bytes=reg_bytes,
        )

    def fused_ar_post_1stage(*, registered: bool):
        reg_ptr, reg_bytes = (0, 0) if registered else _reg()
        fused_allreduce_mhc_post_one_stage(
            ca_comm._ptr,
            tensors["layer_input"],
            next_residual_1stage,
            tensors["residual_in"],
            tensors["post_layer_mix"],
            tensors["comb_res_mix"],
            reg_ptr=reg_ptr,
            reg_bytes=reg_bytes,
        )

    def fused_ar_post_split(*, registered: bool):
        reg_ptr, reg_bytes = (0, 0) if registered else _reg()
        fused_allreduce_mhc_post_split(
            ca_comm._ptr,
            tensors["layer_input"],
            next_residual_split_fused,
            tensors["residual_in"],
            tensors["post_layer_mix"],
            tensors["comb_res_mix"],
            reg_ptr=reg_ptr,
            reg_bytes=reg_bytes,
        )

    def ar_only():
        nonlocal reduced_buf
        reduced_buf = tensor_model_parallel_all_reduce(tensors["layer_input"])

    def mhc_only():
        aiter.mhc_post(
            next_residual_split,
            reduced_buf,
            tensors["residual_in"],
            post_mix,
            tensors["comb_res_mix"],
        )

    err = 0.0
    if run_correctness:
        split_ar_post()
        ref = next_residual_split.clone()
        fused_ar_post(registered=False)
        err = max(
            err,
            checkAllclose(
                ref,
                next_residual_fused,
                msg=f"tp={tp_size} m={m} rank={rank_id} fused_auto",
            ),
        )
        fused_ar_post_1stage(registered=False)
        err = max(
            err,
            checkAllclose(
                ref,
                next_residual_1stage,
                msg=f"tp={tp_size} m={m} rank={rank_id} fused_1stage",
            ),
        )
        fused_ar_post_split(registered=False)
        err = max(
            err,
            checkAllclose(
                ref,
                next_residual_split_fused,
                msg=f"tp={tp_size} m={m} rank={rank_id} fused_2stage",
            ),
        )

    for _ in range(WARMUP):
        split_ar_post()
        fused_ar_post(registered=False)
        fused_ar_post_1stage(registered=False)
        fused_ar_post_split(registered=False)
    torch.cuda.synchronize()

    ar_us = mhc_us = fused_split_us = fused_1stage_us = fused_auto_us = 0.0

    if with_graph:
        graph_split = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph_split, stream=gc.stream):
                reduced = tensor_model_parallel_all_reduce(tensors["layer_input"])
                aiter.mhc_post(
                    next_residual_split,
                    reduced,
                    tensors["residual_in"],
                    post_mix,
                    tensors["comb_res_mix"],
                )
        next_residual_split.zero_()

        graph_fused = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph_fused, stream=gc.stream):
                fused_ar_post(registered=True)
        next_residual_fused.zero_()

        split_us = _event_mean_us(
            graph_split.replay, warmup=BENCH_WARMUP, iters=BENCH_ITERS
        )
        fused_us = _event_mean_us(
            graph_fused.replay, warmup=BENCH_WARMUP, iters=BENCH_ITERS
        )
        if breakdown:
            graph_fused_split = torch.cuda.CUDAGraph()
            with graph_capture() as gc:
                with torch.cuda.graph(graph_fused_split, stream=gc.stream):
                    fused_ar_post_split(registered=True)
            next_residual_split_fused.zero_()
            fused_split_us = _event_mean_us(
                graph_fused_split.replay, warmup=BENCH_WARMUP, iters=BENCH_ITERS
            )
    else:
        if breakdown or compare_stages:
            ar_only()
            ar_us = _event_mean_us(ar_only, warmup=BENCH_WARMUP, iters=BENCH_ITERS)
            mhc_us = _event_mean_us(mhc_only, warmup=BENCH_WARMUP, iters=BENCH_ITERS)
        if compare_stages:
            fused_1stage_us = _event_mean_us(
                lambda: fused_ar_post_1stage(registered=False),
                warmup=BENCH_WARMUP,
                iters=BENCH_ITERS,
            )
            fused_split_us = _event_mean_us(
                lambda: fused_ar_post_split(registered=False),
                warmup=BENCH_WARMUP,
                iters=BENCH_ITERS,
            )
            fused_auto_us = _event_mean_us(
                lambda: fused_ar_post(registered=False),
                warmup=BENCH_WARMUP,
                iters=BENCH_ITERS,
            )
        elif breakdown:
            fused_split_us = _event_mean_us(
                lambda: fused_ar_post_split(registered=False),
                warmup=BENCH_WARMUP,
                iters=BENCH_ITERS,
            )
        split_us = _event_mean_us(split_ar_post, warmup=BENCH_WARMUP, iters=BENCH_ITERS)
        fused_us = _event_mean_us(
            lambda: fused_ar_post(registered=False),
            warmup=BENCH_WARMUP,
            iters=BENCH_ITERS,
        )

    destroy_model_parallel()
    destroy_distributed_environment()

    return {
        "rank": rank_id,
        "split_us": split_us,
        "fused_us": fused_us,
        "ar_us": ar_us,
        "mhc_us": mhc_us,
        "fused_split_us": fused_split_us,
        "fused_1stage_us": fused_1stage_us,
        "fused_auto_us": fused_auto_us,
        "err": err,
    }


def _run_profile(
    tp_size: int,
    m: int,
    hidden_size: int,
    with_graph: bool,
    init_method: Optional[str] = None,
    *,
    run_correctness: bool = False,
    breakdown: bool = False,
    compare_stages: bool = False,
):
    if init_method is None:
        init_method = get_distributed_init_method(get_ip(), get_open_port())
    pool = Pool(processes=tp_size)
    rets = [
        pool.apply_async(
            _profile_worker,
            args=(tp_size, r, m, hidden_size, with_graph, init_method),
            kwds={
                "run_correctness": run_correctness,
                "breakdown": breakdown,
                "compare_stages": compare_stages,
            },
        )
        for r in range(tp_size)
    ]
    pool.close()
    pool.join()
    rows = [r.get() for r in rets]
    split = max(x["split_us"] for x in rows)
    fused = max(x["fused_us"] for x in rows)
    ar = max(x["ar_us"] for x in rows)
    mhc = max(x["mhc_us"] for x in rows)
    fused_split = max(x["fused_split_us"] for x in rows)
    fused_1stage = max(x["fused_1stage_us"] for x in rows)
    fused_auto = max(x["fused_auto_us"] for x in rows)
    saved = split - fused
    speedup = (saved / split * 100.0) if split > 0 else 0.0
    err = max(x["err"] for x in rows)
    input_bytes = m * hidden_size * 2
    use_split = tp_size >= 4 and input_bytes > 512 * 1024
    auto_path = "2stage" if use_split else "1stage"
    best_fused = min(
        (fused_1stage, "1stage"),
        (fused_split, "2stage"),
        (fused_auto, "auto"),
        key=lambda x: x[0],
    )[1]
    return {
        "split_mean_us": split,
        "fused_mean_us": fused,
        "ar_mean_us": ar,
        "mhc_mean_us": mhc,
        "fused_split_mean_us": fused_split,
        "fused_1stage_mean_us": fused_1stage,
        "fused_auto_mean_us": fused_auto if fused_auto > 0 else fused,
        "input_bytes": input_bytes,
        "auto_path": auto_path,
        "best_fused_path": best_fused,
        "saved_us": saved,
        "speedup_pct": speedup,
        "err": err,
    }


@benchmark()
def test_ar_mhc_post_only_profile(
    tp_size: int,
    m: int,
    hidden_size: int,
    with_graph: bool = False,
    distributed_init_method: Optional[str] = None,
    run_correctness: bool = False,
    breakdown: bool = False,
    compare_stages: bool = False,
):
    stats = _run_profile(
        tp_size,
        m,
        hidden_size,
        with_graph,
        distributed_init_method,
        run_correctness=run_correctness,
        breakdown=breakdown,
        compare_stages=compare_stages,
    )
    return {
        "tp_size": tp_size,
        "m": m,
        "hidden_size": hidden_size,
        "withGraph": with_graph,
        **stats,
    }


try:
    import pytest

    @pytest.mark.parametrize("m", [16, 4096])
    def test_fused_ar_mhc_post_only_tp2_smoke(m: int):
        if torch.cuda.device_count() < 2:
            pytest.skip(f"requires >=2 GPUs, got {torch.cuda.device_count()}")
        ret = test_ar_mhc_post_only_profile(
            tp_size=2,
            m=m,
            hidden_size=4096,
            with_graph=False,
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
            run_correctness=True,
        )
        assert ret["err"] == 0

    @pytest.mark.parametrize("m", [16, 128, 4096, 8192])
    def test_fused_auto_dispatch_tp2(m: int):
        if torch.cuda.device_count() < 2:
            pytest.skip(f"requires >=2 GPUs, got {torch.cuda.device_count()}")
        ret = test_ar_mhc_post_only_profile(
            tp_size=2,
            m=m,
            hidden_size=4096,
            with_graph=False,
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
            run_correctness=True,
        )
        assert ret["err"] == 0
        assert ret["auto_path"] == "1stage"

    @pytest.mark.parametrize("m", [16, 8192])
    def test_fused_auto_dispatch_tp4(m: int):
        if torch.cuda.device_count() < 4:
            pytest.skip(f"requires >=4 GPUs, got {torch.cuda.device_count()}")
        ret = test_ar_mhc_post_only_profile(
            tp_size=4,
            m=m,
            hidden_size=4096,
            with_graph=False,
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
            run_correctness=True,
        )
        assert ret["err"] == 0
        input_bytes = m * 4096 * 2
        if input_bytes > 512 * 1024:
            assert ret["auto_path"] == "2stage"
        else:
            assert ret["auto_path"] == "1stage"

except ImportError:
    pass


def _parse_shapes(raw: str) -> list[tuple[int, int]]:
    shapes = []
    for tok in raw.split():
        m_s, h_s = tok.split(",")
        shapes.append((int(m_s), int(h_s)))
    return shapes


def _print_table(
    tp_size: int,
    with_graph: bool,
    rows: list[dict],
    *,
    breakdown: bool,
    compare_stages: bool,
):
    mode = "graph-on" if with_graph else "graph-off"
    print(f"## TP={tp_size} {mode}")
    if compare_stages:
        print("M\tbytes\tsplit\t1stage\t2stage\tauto\tauto_path\tbest")
        for row in rows:
            print(
                f"{row['m']}\t{row['input_bytes']}\t{row['split_mean_us']:.1f}\t"
                f"{row['fused_1stage_mean_us']:.1f}\t{row['fused_split_mean_us']:.1f}\t"
                f"{row['fused_auto_mean_us']:.1f}\t{row['auto_path']}\t"
                f"{row['best_fused_path']}"
            )
    elif breakdown:
        print("M\tar\tmhc\tsplit\tfused_1stage\tfused_2stage")
        for row in rows:
            print(
                f"{row['m']}\t{row['ar_mean_us']:.1f}\t{row['mhc_mean_us']:.1f}\t"
                f"{row['split_mean_us']:.1f}\t{row['fused_mean_us']:.1f}\t"
                f"{row['fused_split_mean_us']:.1f}"
            )
    else:
        print("M\tsplit\tfused\tsaved\tspeedup")
        for row in rows:
            print(
                f"{row['m']}\t{row['split_mean_us']:.1f}\t"
                f"{row['fused_mean_us']:.1f}\t{row['saved_us']:.1f}\t"
                f"{row['speedup_pct']:+.1f}%"
            )
    print()


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(
        description="Profile split AR+mhc_post vs fused post-only epilogue"
    )
    parser.add_argument("-t", "--tp-size", type=int, nargs="+", default=[2])
    parser.add_argument(
        "-s",
        "--shapes",
        type=str,
        default=" ".join(f"{m},{h}" for m, h in DEFAULT_SHAPES),
    )
    parser.add_argument("-g", "--graph", type=int, default=-1, choices=[-1, 0, 1])
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="also report AR-only / mhc_post-only / fused 2-stage split path",
    )
    parser.add_argument(
        "--compare-stages",
        action="store_true",
        help="compare split vs fused 1-stage vs 2-stage vs auto-dispatch (graph-off)",
    )
    args = parser.parse_args()

    if args.compare_stages:
        graph_modes = [False]
    else:
        graph_modes = [False, True] if args.graph < 0 else [bool(args.graph)]

    shapes = _parse_shapes(args.shapes)

    print("# AR + mhc_post only (split vs fused epilogue)")
    print(f"# HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}")
    print(f"# warmup={WARMUP} bench_warmup={BENCH_WARMUP} bench_iters={BENCH_ITERS}")
    print("# graph capture: split via registered AR; fused reg_ptr=0 (registered=True)")
    print("# metric=rank-max mean (us)")
    print()

    df_rows = []
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    for tp_size in args.tp_size:
        for with_graph in graph_modes:
            mode_rows = []
            for m, hidden_size in shapes:
                ret = test_ar_mhc_post_only_profile(
                    tp_size,
                    m,
                    hidden_size,
                    with_graph=with_graph,
                    distributed_init_method=init_method,
                    breakdown=args.breakdown,
                    compare_stages=args.compare_stages,
                )
                mode_rows.append(ret)
                df_rows.append(ret)
            _print_table(
                tp_size,
                with_graph,
                mode_rows,
                breakdown=args.breakdown,
                compare_stages=args.compare_stages,
            )

    df = pd.DataFrame(df_rows)
    logger.info(
        "AR+mhc_post profile summary (markdown):\n%s",
        df.to_markdown(index=False),
    )
