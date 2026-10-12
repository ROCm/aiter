# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multigpu tests: split (AR + mhc_post) vs fused AR+mhc_post epilogue."""

from __future__ import annotations

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method

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
    fused_allreduce_mhc_post_one_stage,
    fused_allreduce_mhc_post_only,
    fused_allreduce_mhc_post_split,
)
from aiter.test_common import checkAllclose

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


def barrier_before_teardown():
    """Align all ranks before tearing down the distributed groups.

    Drain this rank's GPU work, then join a barrier so no rank starts freeing
    IPC buffers / destroying process groups while a peer is still inside a
    NCCL / custom-all-reduce collective -- that race intermittently hangs when
    these comm UTs run back-to-back in CI. No-op if dist is uninitialized.
    """
    if not dist.is_initialized():
        return
    torch.cuda.synchronize()
    get_tp_group().barrier()
    torch.cuda.synchronize()


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


def _run_mhc_post_case(
    tp_size: int,
    rank_id: int,
    m: int,
    hidden_size: int,
    with_graph: bool,
    device: torch.device,
    graphs: list,
    *,
    run_correctness: bool,
    breakdown: bool,
    compare_stages: bool,
):
    """Profile one (m, hidden_size, with_graph) case on an initialized rank.

    In graph mode the captured graphs and the buffers they reference are
    appended to ``graphs`` and must outlive the whole sweep: custom all-reduce
    caches the peer IPC address of every captured buffer by its local pointer,
    so freeing one and capturing a later case at the same address would
    replay with stale peer pointers.
    """
    tensors = _make_inputs(m, hidden_size, rank_id, device)
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
        with graph_capture() as gc, torch.cuda.graph(graph_split, stream=gc.stream):
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
        with graph_capture() as gc, torch.cuda.graph(graph_fused, stream=gc.stream):
            fused_ar_post(registered=True)
        next_residual_fused.zero_()
        captured = [graph_split, graph_fused]

        split_us = _event_mean_us(
            graph_split.replay, warmup=BENCH_WARMUP, iters=BENCH_ITERS
        )
        fused_us = _event_mean_us(
            graph_fused.replay, warmup=BENCH_WARMUP, iters=BENCH_ITERS
        )
        if breakdown:
            graph_fused_split = torch.cuda.CUDAGraph()
            with graph_capture() as gc, torch.cuda.graph(
                graph_fused_split, stream=gc.stream
            ):
                fused_ar_post_split(registered=True)
            next_residual_split_fused.zero_()
            captured.append(graph_fused_split)
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

    if with_graph:
        graphs.append(
            (
                captured,
                tensors,
                next_residual_split,
                next_residual_fused,
                next_residual_split_fused,
            )
        )

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


def _profile_sweep_worker(
    tp_size: int,
    rank_id: int,
    cases: list[tuple[int, int, bool]],
    init_method: str,
    *,
    run_correctness: bool,
    breakdown: bool = False,
    compare_stages: bool = False,
):
    """Run every ``(m, hidden_size, with_graph)`` case inside one init.

    Setting up the TP group dominates a single case (seconds vs. microseconds
    of kernel time), so graph-on and graph-off cases share one group that is
    created and torn down once.
    """
    device = torch.device(f"cuda:{rank_id}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rank_id,
        distributed_init_method=init_method,
    )
    ensure_model_parallel_initialized(tp_size, 1)
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    graphs = []
    results = [
        _run_mhc_post_case(
            tp_size,
            rank_id,
            m,
            hidden_size,
            with_graph,
            device,
            graphs,
            run_correctness=run_correctness,
            breakdown=breakdown,
            compare_stages=compare_stages,
        )
        for m, hidden_size, with_graph in cases
    ]

    barrier_before_teardown()
    destroy_model_parallel()
    destroy_distributed_environment()
    graphs.clear()
    return results


def _summarize_profile(tp_size: int, m: int, hidden_size: int, rows: list[dict]):
    """Reduce one case's per-rank timings to rank-max stats."""
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


def test_ar_mhc_post_only_profile(
    tp_size: int,
    cases: list[tuple[int, int, bool]],
    distributed_init_method: str | None = None,
    run_correctness: bool = False,
    breakdown: bool = False,
    compare_stages: bool = False,
):
    """Profile ``(m, hidden_size, with_graph)`` cases on one TP group and
    return one summary row per case."""
    if distributed_init_method is None:
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp_size) as pool:
        rets = [
            pool.apply_async(
                _profile_sweep_worker,
                args=(tp_size, r, cases, distributed_init_method),
                kwds={
                    "run_correctness": run_correctness,
                    "breakdown": breakdown,
                    "compare_stages": compare_stages,
                },
            )
            for r in range(tp_size)
        ]
        per_rank = [r.get() for r in rets]
    return [
        {
            "tp_size": tp_size,
            "m": m,
            "hidden_size": hidden_size,
            "withGraph": with_graph,
            "run_correctness": run_correctness,
            "breakdown": breakdown,
            "compare_stages": compare_stages,
            **_summarize_profile(
                tp_size, m, hidden_size, [results[i] for results in per_rank]
            ),
        }
        for i, (m, hidden_size, with_graph) in enumerate(cases)
    ]


try:
    import pytest

    def _correctness_rows(tp_size: int, ms: list[int]) -> list[dict]:
        if torch.cuda.device_count() < tp_size:
            pytest.skip(f"requires >={tp_size} GPUs, got {torch.cuda.device_count()}")
        return test_ar_mhc_post_only_profile(
            tp_size,
            [(m, 4096, False) for m in ms],
            run_correctness=True,
        )

    def test_fused_ar_mhc_post_only_tp2_smoke():
        for ret in _correctness_rows(2, [16, 4096]):
            assert ret["err"] == 0, ret["m"]

    def test_fused_auto_dispatch_tp2():
        for ret in _correctness_rows(2, [16, 128, 4096, 8192]):
            assert ret["err"] == 0, ret["m"]
            assert ret["auto_path"] == "1stage", ret["m"]

    def test_fused_auto_dispatch_tp4():
        for ret in _correctness_rows(4, [16, 8192]):
            m = ret["m"]
            assert ret["err"] == 0, m
            input_bytes = m * 4096 * 2
            if input_bytes > 512 * 1024:
                assert ret["auto_path"] == "2stage", m
            else:
                assert ret["auto_path"] == "1stage", m

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

    # One TP group per tp_size; graph-on/off modes and shapes share it.
    df_rows = []
    for tp_size in args.tp_size:
        rows = test_ar_mhc_post_only_profile(
            tp_size,
            [(m, h, with_graph) for with_graph in graph_modes for m, h in shapes],
            breakdown=args.breakdown,
            compare_stages=args.compare_stages,
        )
        df_rows.extend(rows)
        for with_graph in graph_modes:
            _print_table(
                tp_size,
                with_graph,
                [row for row in rows if row["withGraph"] == with_graph],
                breakdown=args.breakdown,
                compare_stages=args.compare_stages,
            )

    df = pd.DataFrame(df_rows)
    logger.info(
        "AR+mhc_post profile summary (markdown):\n%s",
        df.to_markdown(index=False),
    )
