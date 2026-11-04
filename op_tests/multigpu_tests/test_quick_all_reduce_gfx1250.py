# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import multiprocessing
import os
import sys
import time
from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist

import aiter as ops
from aiter.dist.device_communicators.quick_all_reduce import (
    QuickAllReduce,
    QuickReduceRegime,
    qr_exchange_handles,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

MB = 1024 * 1024
TP_SIZES = (2, 4, 8)
# INT4-LEGACY is the two-shot INT4 route of an unconfigured
# (quant_level=None) communicator.
LEGACY_REGIMES = ("FP", "FP8", "INT6", "INT4-LEGACY", "INT3")
REGIMES = ("INT4",) + LEGACY_REGIMES
DTYPE = torch.bfloat16
LEGACY_MAX_SIZE = 16 * MB
QR_ENV = (
    "AITER_QUICK_REDUCE_QUANTIZATION",
    "AITER_QUICK_REDUCE_CAST_BF16_TO_FP16",
    "AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB",
)


def _is_gfx1250() -> bool:
    return (
        torch.cuda.is_available()
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    )


def _tolerances(regime: str, world_size: int) -> tuple[float, float]:
    atol_per_rank = {
        "FP": 0.125,
        "FP8": 1.0,
        "INT6": 0.75,
        "INT4": 1.0,
        "INT4-LEGACY": 1.0,
        "INT3": 2.0,
    }
    return atol_per_rank[regime] * world_size, 2e-2


def _check_eager(
    reduce_fn: Callable[[torch.Tensor], torch.Tensor],
    numels: tuple[int, ...],
    rank: int,
    world_size: int,
    atol: float,
    rtol: float,
) -> None:
    device = torch.device(f"cuda:{rank}")
    rank_sum = world_size * (world_size + 1) // 2
    for numel in numels:
        inp = torch.full((numel,), rank + 1, dtype=DTYPE, device=device)
        out = reduce_fn(inp)
        torch.cuda.synchronize()
        expected = torch.full_like(out, float(rank_sum))
        torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)

    pattern = torch.linspace(-2.0, 2.0, 1024, dtype=DTYPE, device=device)
    out = reduce_fn(pattern + float(rank + 1))
    torch.cuda.synchronize()
    reference = pattern * world_size + rank_sum
    torch.testing.assert_close(out, reference, atol=atol, rtol=rtol)


def _check_graph_replays(
    reduce_fn: Callable[[torch.Tensor], torch.Tensor],
    graph_sizes: tuple[int, int],
    rank: int,
    world_size: int,
    atol: float,
    rtol: float,
    label: str,
) -> None:
    """Alternate two captured sizes with changing inputs, rank skew, and no
    per-replay synchronization, then reject zero and stale samples."""
    device = torch.device(f"cuda:{rank}")
    rank_sum = world_size * (world_size + 1) // 2
    graphs = []
    inputs = []
    outputs = []
    for size_bytes in graph_sizes:
        numel = size_bytes // torch.empty((), dtype=DTYPE).element_size()
        inp = torch.full((numel,), rank + 1, dtype=DTYPE, device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = reduce_fn(inp)
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
        torch.empty((replay_count, len(offsets)), dtype=DTYPE, pin_memory=True)
        for _ in graphs
    ]
    expected = [[], []]
    replay_values = ((8.0, 40.0, 72.0, 104.0), (12.0, 52.0, 92.0, 132.0))
    for replay in range(replay_count):
        for index, scale in ((0, 1), (1, 2)):
            values = replay_values[index]
            replay_value = values[replay % len(values)]
            scalar = scale * (rank + 1) + replay_value
            inputs[index].fill_(scalar)
            for position, offset in zip(positions[index], offsets):
                inputs[index][position] = scalar + offset
            graphs[index].replay()
            samples[index][replay].copy_(
                outputs[index].flatten()[positions[index]], non_blocking=True
            )
            expected[index].append(
                [
                    scale * rank_sum + world_size * (replay_value + offset)
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
            f"{label} graph {graph_index} output is closer to the previous "
            "replay than the current replay"
        )


def _new_quick_all_reduce(device: torch.device, **env: str) -> QuickAllReduce:
    for name in QR_ENV:
        os.environ.pop(name, None)
    os.environ.update(env)
    return QuickAllReduce(group=dist.group.WORLD, device=device)


def _run_pull_q4_checks(rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    for regime in ("FP", "FP8", "INT6", "INT3"):
        disabled = _new_quick_all_reduce(device, AITER_QUICK_REDUCE_QUANTIZATION=regime)
        assert disabled.disabled and not disabled._ptr, regime
    disabled = _new_quick_all_reduce(
        device,
        AITER_QUICK_REDUCE_QUANTIZATION="INT4",
        AITER_QUICK_REDUCE_CAST_BF16_TO_FP16="0",
    )
    assert disabled.disabled and not disabled._ptr

    quick_ar = _new_quick_all_reduce(device, AITER_QUICK_REDUCE_QUANTIZATION="INT4")
    try:
        assert not quick_ar.disabled, "QuickReduce failed to initialize"
        assert quick_ar.uses_pull_q4, "gfx1250 INT4 must select PullQ4"
        assert quick_ar.qr_max_size == 512 * MB

        route_min_bytes = QuickAllReduce._GFX1250_MIN_SIZE[world_size]
        route_probe = torch.empty(route_min_bytes // 2, dtype=DTYPE, device=device)
        assert quick_ar.should_quick_allreduce(route_probe)
        assert not quick_ar.should_quick_allreduce(route_probe[:-1])
        assert not quick_ar.should_quick_allreduce(route_probe.to(torch.float16))
        assert not quick_ar.should_quick_allreduce_rmsnorm(
            route_probe, route_probe, route_probe[:1024], 1024
        )

        small = torch.ones(1024, dtype=DTYPE, device=device)
        with pytest.raises(RuntimeError, match="cannot mix legacy"):
            ops.qr_all_reduce(
                quick_ar._ptr,
                small,
                torch.empty_like(small),
                QuickReduceRegime.FP.value,
                True,
            )
        with pytest.raises(RuntimeError, match="requires BF16 input"):
            ops.qr_all_reduce(
                quick_ar._ptr,
                small,
                torch.empty_like(small),
                QuickReduceRegime.INT4.value,
                False,
            )
        with pytest.raises(RuntimeError, match="does not support the legacy fused"):
            ops.qr_all_reduce_rmsnorm(
                quick_ar._ptr,
                small,
                small,
                torch.empty_like(small),
                torch.empty_like(small),
                small,
                1e-5,
                1024,
                QuickReduceRegime.INT4.value,
                True,
            )

        atol, rtol = _tolerances("INT4", world_size)
        _check_eager(
            quick_ar.quick_all_reduce,
            (1, 7, 1023, 16391, 1 << 20),
            rank,
            world_size,
            atol,
            rtol,
        )
        _check_graph_replays(
            quick_ar.quick_all_reduce,
            (512 * 1024, 64 * MB),
            rank,
            world_size,
            atol,
            rtol,
            "PullQ4",
        )
        # No rank may free its exported payload while a peer can still read it.
        dist.barrier()
    finally:
        quick_ar.close()


def _run_legacy_checks(rank: int, world_size: int, regime: str) -> None:
    """Exercise shared two-shot codecs on gfx1250 through direct op calls.
    These are not production routes on gfx1250."""
    device = torch.device(f"cuda:{rank}")
    quant_level = QuickReduceRegime["INT4" if regime == "INT4-LEGACY" else regime]
    configured_level = None if regime == "INT4-LEGACY" else quant_level.value
    ptr = ops.init_custom_qr(rank, world_size, LEGACY_MAX_SIZE, configured_level)
    try:
        qr_exchange_handles(ptr, world_size, dist.group.WORLD)
        assert not ops.qr_uses_pull_q4(ptr), f"{regime} must not select PullQ4"

        def reduce_fn(inp: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(inp)
            ops.qr_all_reduce(ptr, inp, out, quant_level.value, True)
            return out

        if regime == "INT3":
            small = torch.ones(1024, dtype=DTYPE, device=device)
            with pytest.raises(RuntimeError, match="does not support INT3"):
                ops.qr_all_reduce_rmsnorm(
                    ptr,
                    small,
                    small,
                    torch.empty_like(small),
                    torch.empty_like(small),
                    small,
                    1e-5,
                    1024,
                    quant_level.value,
                    True,
                )

        atol, rtol = _tolerances(regime, world_size)
        _check_eager(reduce_fn, (8, 1024, 1 << 20), rank, world_size, atol, rtol)
        _check_graph_replays(
            reduce_fn,
            (512 * 1024, LEGACY_MAX_SIZE // 2),
            rank,
            world_size,
            atol,
            rtol,
            regime,
        )
        dist.barrier()
    finally:
        ops.qr_destroy(ptr)


def _worker(rank: int, world_size: int, init_method: str, regime: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        if regime == "INT4":
            _run_pull_q4_checks(rank, world_size)
        else:
            _run_legacy_checks(rank, world_size, regime)
    finally:
        dist.destroy_process_group()
        torch.cuda.empty_cache()


def _skip_reason(tp_size: int, regime: str) -> str | None:
    if not _is_gfx1250():
        return "gfx1250-only test"
    if torch.cuda.device_count() < tp_size:
        return f"TP{tp_size} requires {tp_size} GPUs"
    if regime == "INT3" and tp_size != 2:
        return "INT3 is supported only for TP2"
    return None


def _run_tp_test(tp_size: int, regime: str) -> None:
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(target=_worker, args=(rank, tp_size, init_method, regime))
        for rank in range(tp_size)
    ]
    # Spawned workers decide QuickReduce availability when they import it.
    previous = os.environ.get("AITER_QUICK_REDUCE_QUANTIZATION")
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"
    try:
        for process in processes:
            process.start()
    finally:
        if previous is None:
            os.environ.pop("AITER_QUICK_REDUCE_QUANTIZATION")
        else:
            os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = previous
    for process in processes:
        process.join(timeout=300)
    hung = [process for process in processes if process.is_alive()]
    if hung:
        for process in hung:
            process.terminate()
            process.join()
        raise RuntimeError(f"TP{tp_size} {regime} QuickReduce test timed out")
    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"TP{tp_size} {regime} QuickReduce workers failed: {failed}")


def test_quick_all_reduce_gfx1250_quantization_validation() -> None:
    if not _is_gfx1250():
        pytest.skip("gfx1250-only test")
    torch.cuda.set_device(0)
    for invalid in (
        -1,
        QuickReduceRegime.NONE.value,
        len(QuickReduceRegime),
        2**32 + QuickReduceRegime.INT4.value,
    ):
        with pytest.raises((ValueError, RuntimeError), match="quantization level"):
            ops.init_custom_qr(0, 2, MB, invalid)

    # The original three-argument API remains valid and creates an
    # unconfigured legacy communicator.
    ptr = ops.init_custom_qr(0, 2, MB)
    try:
        assert not ops.qr_uses_pull_q4(ptr)
    finally:
        ops.qr_destroy(ptr)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("tp_size", TP_SIZES)
def test_quick_all_reduce_gfx1250(tp_size: int, regime: str) -> None:
    reason = _skip_reason(tp_size, regime)
    if reason is not None:
        pytest.skip(reason)
    _run_tp_test(tp_size, regime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, choices=TP_SIZES, nargs="*")
    parser.add_argument("--regime", choices=REGIMES, nargs="*")
    args = parser.parse_args()
    if not _is_gfx1250():
        print("Skipping gfx1250 QuickReduce test on a non-gfx1250 device.")
        sys.exit(0)
    test_quick_all_reduce_gfx1250_quantization_validation()
    for tp_size in args.tp_size or TP_SIZES:
        for regime in args.regime or REGIMES:
            reason = _skip_reason(tp_size, regime)
            if reason is not None:
                print(f"Skipping TP{tp_size} {regime}: {reason}")
                continue
            _run_tp_test(tp_size, regime)
            print(f"PASS TP{tp_size} {regime}")
