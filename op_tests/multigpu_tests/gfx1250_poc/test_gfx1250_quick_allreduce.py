# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import multiprocessing
import os
import time

import pytest
import torch
import torch.distributed as dist

import aiter as ops
from aiter.dist.device_communicators.quick_all_reduce import (
    QuickAllReduce,
    QuickReduceRegime,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

DTYPES = {"bf16": torch.bfloat16}
REGIMES = ("FP", "FP8", "INT6", "INT4", "INT3")


def _tolerances(regime: str, world_size: int) -> tuple[float, float]:
    atol_per_rank = {
        "FP": 0.125,
        "FP8": 1.0,
        "INT6": 0.75,
        "INT4": 1.0,
        "INT3": 2.0,
    }
    return atol_per_rank[regime] * world_size, 2e-2


def _run_quickreduce_checks(
    rank: int,
    world_size: int,
    init_method: str,
    regime: str,
    dtype_name: str,
) -> None:
    device = torch.device(f"cuda:{rank}")
    dtype = DTYPES[dtype_name]
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    quick_ar = None
    try:
        quick_ar = QuickAllReduce(group=dist.group.WORLD, device=device)
        assert not quick_ar.disabled, "QuickReduce failed to initialize"
        if regime == "INT4":
            assert quick_ar.uses_pull_q4, "INT4 test requires PullQ4"
            assert quick_ar.uses_pull_q4_bulk, "INT4 test requires bulk PullQ4"
            route_min_bytes = {2: 16, 4: 8}[world_size] * 1024 * 1024
            route_probe = torch.empty(
                route_min_bytes // torch.tensor([], dtype=dtype).element_size(),
                dtype=dtype,
                device=device,
            )
            assert quick_ar.should_quick_allreduce(route_probe)
            assert not quick_ar.should_quick_allreduce(route_probe[:-1])
            assert not quick_ar.should_quick_allreduce(route_probe.to(torch.float16))
        else:
            assert not quick_ar.uses_pull_q4, f"{regime} must not select PullQ4"
            assert (
                not quick_ar.uses_pull_q4_bulk
            ), f"{regime} must not select bulk PullQ4"

        rank_sum = world_size * (world_size + 1) // 2
        atol, rtol = _tolerances(regime, world_size)
        mismatch_inp = torch.full((1024,), rank + 1, dtype=dtype, device=device)
        with pytest.raises(RuntimeError, match="does not match"):
            ops.qr_all_reduce(
                quick_ar._ptr,
                mismatch_inp,
                torch.empty_like(mismatch_inp),
                (quick_ar.qr_quant_level.value + 1)
                % (QuickReduceRegime.INT3.value + 1),
                quick_ar.use_fp16_kernels,
            )

        numels = (
            (1, 7, 1023, 16391, 1 << 20) if regime == "INT4" else (8, 1024, 1 << 20)
        )
        for numel in numels:
            inp = torch.full((numel,), rank + 1, dtype=dtype, device=device)
            out = quick_ar.quick_all_reduce(inp)
            torch.cuda.synchronize()
            expected = torch.full_like(out, float(rank_sum))
            torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)

        pattern = torch.linspace(-2.0, 2.0, 1024, dtype=dtype, device=device)
        inp = pattern + float(rank + 1)
        out = quick_ar.quick_all_reduce(inp)
        torch.cuda.synchronize()
        reference = pattern * world_size + rank_sum
        torch.testing.assert_close(out, reference, atol=atol, rtol=rtol)

        if regime == "INT3":
            residual = torch.ones_like(mismatch_inp)
            weight = torch.ones(1024, dtype=dtype, device=device)
            with pytest.raises(RuntimeError, match="does not support INT3"):
                ops.qr_all_reduce_rmsnorm(
                    quick_ar._ptr,
                    mismatch_inp,
                    residual,
                    torch.empty_like(residual),
                    torch.empty_like(mismatch_inp),
                    weight,
                    1e-5,
                    1024,
                    QuickReduceRegime.INT3.value,
                    quick_ar.use_fp16_kernels,
                )

        graphs = []
        inputs = []
        outputs = []
        # Alternate a fused (<1 MiB) graph with a bulk graph for INT4.
        # Other codecs exercise their two-shot graph route at the same sizes.
        for size_bytes in (512 * 1024, 64 * 1024 * 1024):
            numel = size_bytes // torch.empty((), dtype=dtype).element_size()
            inp = torch.full((numel,), rank + 1, dtype=dtype, device=device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = quick_ar.quick_all_reduce(inp)
            inputs.append(inp)
            outputs.append(out)
            graphs.append(graph)

        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            time.sleep(0.2)

        replay_count = int(os.environ.get("AITER_QUICK_REDUCE_GRAPH_REPLAYS", "1000"))
        positions = [[0, inp.numel() // 3, inp.numel() - 1] for inp in inputs]
        offsets = (-1.0, 0.5, 2.0)
        samples = [
            torch.empty((replay_count, len(offsets)), dtype=dtype, pin_memory=True)
            for _ in graphs
        ]
        expected = [[], []]
        replay_values = ((8.0, 40.0, 72.0, 104.0), (12.0, 52.0, 92.0, 132.0))
        for replay in range(1, replay_count + 1):
            replay_value = replay_values[0][(replay - 1) % len(replay_values[0])]
            scalar = rank + 1 + replay_value
            inputs[0].fill_(scalar)
            for position, offset in zip(positions[0], offsets):
                inputs[0][position] = scalar + offset
            graphs[0].replay()
            samples[0][replay - 1].copy_(
                outputs[0].flatten()[positions[0]], non_blocking=True
            )
            expected[0].append(
                [rank_sum + world_size * (replay_value + offset) for offset in offsets]
            )

            replay_value = replay_values[1][(replay - 1) % len(replay_values[1])]
            scalar = 2 * (rank + 1) + replay_value
            inputs[1].fill_(scalar)
            for position, offset in zip(positions[1], offsets):
                inputs[1][position] = scalar + offset
            graphs[1].replay()
            samples[1][replay - 1].copy_(
                outputs[1].flatten()[positions[1]], non_blocking=True
            )
            expected[1].append(
                [
                    2 * rank_sum + world_size * (replay_value + offset)
                    for offset in offsets
                ]
            )
        torch.cuda.synchronize()

        for graph_index, actual in enumerate(samples):
            expected_tensor = torch.tensor(expected[graph_index], dtype=torch.float32)
            assert torch.isfinite(actual).all()
            assert torch.all(actual != 0)
            torch.testing.assert_close(
                actual.float(), expected_tensor, atol=atol, rtol=rtol
            )
            current_error = (actual.float()[1:] - expected_tensor[1:]).abs()
            stale_error = (actual.float()[1:] - expected_tensor[:-1]).abs()
            assert torch.all(current_error < stale_error), (
                f"{regime} graph {graph_index} output is closer to the previous "
                "replay than the current replay"
            )
        # No rank may free its exported payload while a peer can still read it.
        dist.barrier()
    finally:
        if quick_ar is not None:
            quick_ar.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()


def _run_tp_test(
    tp_size: int,
    regime: str = "INT4",
    dtype_name: str = "bf16",
) -> None:
    if "gfx1250" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx1250-only test")
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"TP{tp_size} requires {tp_size} GPUs")
    if regime == "INT3" and tp_size != 2:
        pytest.skip("INT3 is supported only for TP2")

    env_updates = {
        "AITER_QUICK_REDUCE_QUANTIZATION": regime,
        "AITER_QUICK_REDUCE_FORCE_PULL_Q4": None,
        "AITER_QUICK_REDUCE_FORCE_PULL_Q4_BULK": None,
    }
    previous = {name: os.environ.get(name) for name in env_updates}
    try:
        for name, value in env_updates.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        init_method = get_distributed_init_method(get_ip(), get_open_port())

        ctx = multiprocessing.get_context("spawn")
        processes = [
            ctx.Process(
                target=_run_quickreduce_checks,
                args=(rank, tp_size, init_method, regime, dtype_name),
            )
            for rank in range(tp_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=180)
        hung = [process for process in processes if process.is_alive()]
        if hung:
            for process in hung:
                process.terminate()
                process.join()
            raise RuntimeError(
                f"TP{tp_size} {dtype_name} {regime} QuickReduce test timed out"
            )
        failed = [process.exitcode for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(
                f"TP{tp_size} {dtype_name} {regime} QuickReduce workers failed: {failed}"
            )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_gfx1250_quick_allreduce_quantization_validation() -> None:
    if "gfx1250" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx1250-only test")
    torch.cuda.set_device(0)
    for invalid in (
        -1,
        QuickReduceRegime.NONE.value,
        len(QuickReduceRegime),
        2**32 + QuickReduceRegime.INT4.value,
    ):
        with pytest.raises((ValueError, RuntimeError), match="quantization level"):
            ops.init_custom_qr(0, 2, 1024 * 1024, invalid)

    # The original three-argument source API remains valid and intentionally
    # creates an unconfigured legacy communicator.
    ptr = ops.init_custom_qr(0, 2, 1024 * 1024)
    try:
        assert not ops.qr_uses_pull_q4(ptr)
        assert not ops.qr_uses_pull_q4_bulk(ptr)
    finally:
        ops.qr_destroy(ptr)


@pytest.mark.parametrize("regime", ["FP", "FP8", "INT6", "INT4"])
@pytest.mark.parametrize("tp_size", [2, 4])
def test_gfx1250_quick_allreduce(tp_size: int, regime: str) -> None:
    _run_tp_test(tp_size, regime, "bf16")


def test_gfx1250_quick_allreduce_int3_tp2() -> None:
    _run_tp_test(2, "INT3", "bf16")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, choices=(2, 4), required=True)
    parser.add_argument("--regime", choices=REGIMES, default="INT4")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    args = parser.parse_args()
    _run_tp_test(args.tp_size, args.regime, args.dtype)
