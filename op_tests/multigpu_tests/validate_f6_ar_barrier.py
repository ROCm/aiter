# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP4 validation for gfx1250 F6 fused-AR end-barrier folding.

Run only on four idle gfx1250 GPUs:

    python op_tests/multigpu_tests/validate_f6_ar_barrier.py

For M=2517 and 4536 at hidden=6144, this compares the two-stage prefill path
bitwise with the one-stage fixed-rank-order implementation, stresses eager and
HIP-graph rounds under rank arrival skew, checks all-rank bitwise digests, and
profiles the two-stage call. Expected targeted kernels per call are:

    start_sync_kernel -> ar_rmsnorm_reduce_scatter_gfx1250
                      -> ar_rmsnorm_gfx1250

There must be no standalone end_sync_kernel. A process-pool timeout turns a
barrier hang into a failing run instead of leaving workers behind indefinitely.
"""

import argparse
import hashlib
import json
import logging
import statistics
import time
from multiprocessing import TimeoutError, get_context

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from aiter import dtypes
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port


LOG = logging.getLogger("aiter.f6")
TP = 4
HIDDEN = 6144
DEFAULT_ROWS = (2517, 4536)
EPS = 1e-6
TARGET_KERNELS = (
    "start_sync_kernel",
    "ar_rmsnorm_reduce_scatter_gfx1250",
    "ar_rmsnorm_gfx1250",
    "end_sync_kernel",
)
TIMING_WARMUP = 5
TIMING_ITERS = 50


def _is_gfx1250(device: torch.device) -> bool:
    return "gfx1250" in torch.cuda.get_device_properties(device).gcnArchName


def _skew(rank: int, step: int) -> None:
    # Deliberately vary host arrival. This wait is not reported as removable F6
    # gain: start_sync must still wait for the slowest rank.
    time.sleep(((rank * 17 + step * 29) % 8) * 0.0002)


def _buffers(shape: tuple[int, int], quant: bool, device: torch.device):
    residual_out = torch.empty(shape, dtype=torch.bfloat16, device=device)
    if quant:
        return (
            torch.empty(shape, dtype=dtypes.fp8, device=device),
            residual_out,
            torch.empty((shape[0], 1), dtype=torch.float32, device=device),
        )
    return torch.empty(shape, dtype=torch.bfloat16, device=device), residual_out


def _call(ca, x, residual, weight, buffers, *, use_1stage: bool, quant: bool):
    kwargs = {
        "res_out": buffers[1],
        "out": buffers[0],
        "w": weight,
        "eps": EPS,
        "registered": False,
        "use_1stage": use_1stage,
        "post_per_token_quant": quant,
        "gemma_norm": True,
    }
    if quant:
        kwargs["scale_out"] = buffers[2]
    result = ca.fused_ar_rms(x, residual, **kwargs)
    assert all(actual is expected for actual, expected in zip(result, buffers))
    return result


def _assert_bitwise(label: str, expected, actual) -> None:
    assert len(expected) == len(actual)
    for index, (want, got) in enumerate(zip(expected, actual)):
        want_bits = want.contiguous().view(torch.uint8)
        got_bits = got.contiguous().view(torch.uint8)
        if not torch.equal(want_bits, got_bits):
            mismatches = torch.count_nonzero(want_bits != got_bits).item()
            raise AssertionError(
                f"{label} tensor {index} differs in {mismatches} bytes"
            )


def _digest(tensors) -> str:
    value = hashlib.sha256()
    for tensor in tensors:
        value.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return value.hexdigest()


def _assert_all_rank_bits(label: str, tensors, world_size: int) -> None:
    digest = _digest(tensors)
    digests = [None] * world_size
    dist.all_gather_object(digests, digest)
    if len(set(digests)) != 1:
        raise AssertionError(f"{label} differs across ranks: {digests}")


def _capture(ca, fn):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with ca.capture(), torch.cuda.graph(graph, stream=stream):
        outputs = fn()
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier()
    return graph, outputs


def _time_us(fn) -> float:
    for _ in range(TIMING_WARMUP):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(TIMING_ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(TIMING_ITERS)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(
        start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)
    )


def _profile_kernel_count(fn) -> tuple[dict[str, int], dict[str, float]]:
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = list(prof.events())
    names = [event.name for event in events]
    counts = {
        needle: sum(needle in name for name in names) for needle in TARGET_KERNELS
    }
    expected = {
        "start_sync_kernel": 1,
        "ar_rmsnorm_reduce_scatter_gfx1250": 1,
        "ar_rmsnorm_gfx1250": 1,
        "end_sync_kernel": 0,
    }
    if counts != expected:
        matching = sorted(
            {name for name in names if any(k in name for k in TARGET_KERNELS)}
        )
        raise AssertionError(
            f"target kernel count {counts}, expected {expected}; events={matching}"
        )
    kernel_us = {}
    for needle in TARGET_KERNELS:
        kernel_us[needle] = sum(
            float(
                getattr(
                    event,
                    "device_time_total",
                    getattr(event, "cuda_time_total", 0.0),
                )
            )
            for event in events
            if needle in event.name
        )
    return counts, kernel_us


def _run_case(
    rank: int,
    ca,
    rows: int,
    quant: bool,
    eager_iters: int,
    graph_iters: int,
    device: torch.device,
):
    shape = (rows, HIDDEN)
    common_gen = torch.Generator(device=device).manual_seed(393_000 + rows)
    rank_gen = torch.Generator(device=device).manual_seed(393_100 + rows + rank)
    residual = torch.empty(shape, dtype=torch.bfloat16, device=device)
    x = torch.empty(shape, dtype=torch.bfloat16, device=device)
    residual.uniform_(-0.25, 0.25, generator=common_gen)
    x.uniform_(-0.25, 0.25, generator=rank_gen)
    weight = torch.empty(HIDDEN, dtype=torch.bfloat16, device=device)
    weight.uniform_(-0.1, 0.1, generator=common_gen)

    eager_ref = _buffers(shape, quant, device)
    eager_fused = _buffers(shape, quant, device)
    for step in range(eager_iters):
        residual.uniform_(-0.25, 0.25, generator=common_gen)
        x.uniform_(-0.25, 0.25, generator=rank_gen)
        _skew(rank, step)
        reference = _call(
            ca,
            x,
            residual,
            weight,
            eager_ref,
            use_1stage=True,
            quant=quant,
        )
        fused = _call(
            ca,
            x,
            residual,
            weight,
            eager_fused,
            use_1stage=False,
            quant=quant,
        )
        torch.cuda.synchronize()
        _assert_bitwise(f"eager M={rows} step={step}", reference, fused)
    _assert_all_rank_bits(f"eager M={rows}", eager_fused, TP)

    graph_ref_buffers = _buffers(shape, quant, device)
    graph_fused_buffers = _buffers(shape, quant, device)
    reference_graph, reference_graph_out = _capture(
        ca,
        lambda: _call(
            ca,
            x,
            residual,
            weight,
            graph_ref_buffers,
            use_1stage=True,
            quant=quant,
        ),
    )
    fused_graph, fused_graph_out = _capture(
        ca,
        lambda: _call(
            ca,
            x,
            residual,
            weight,
            graph_fused_buffers,
            use_1stage=False,
            quant=quant,
        ),
    )
    for step in range(graph_iters):
        residual.uniform_(-0.25, 0.25, generator=common_gen)
        x.uniform_(-0.25, 0.25, generator=rank_gen)
        _skew(rank, eager_iters + step)
        reference_graph.replay()
        fused_graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(
            f"graph M={rows} replay={step}", reference_graph_out, fused_graph_out
        )
    _assert_all_rank_bits(f"graph M={rows}", fused_graph_out, TP)

    # All timing blocks run without the deliberate host skew. Report every
    # rank: the spread captures residual rank-arrival wait in start_sync and is
    # not counted as removable F6 work.
    dist.barrier()
    eager_reference_us = _time_us(
        lambda: _call(
            ca,
            x,
            residual,
            weight,
            eager_ref,
            use_1stage=True,
            quant=quant,
        )
    )
    dist.barrier()
    eager_fused_us = _time_us(
        lambda: _call(
            ca,
            x,
            residual,
            weight,
            eager_fused,
            use_1stage=False,
            quant=quant,
        )
    )
    dist.barrier()
    graph_reference_us = _time_us(reference_graph.replay)
    dist.barrier()
    graph_fused_us = _time_us(fused_graph.replay)

    # Profile an eager two-stage call. Extra copy/setup kernels are allowed;
    # the F6 sequence itself must contain exactly these three targeted kernels.
    dist.barrier()
    counts, kernel_us = _profile_kernel_count(
        lambda: _call(
            ca,
            x,
            residual,
            weight,
            eager_fused,
            use_1stage=False,
            quant=quant,
        )
    )
    dist.barrier()
    return {
        "rank": rank,
        "rows": rows,
        "quant": quant,
        "eager_rounds": eager_iters,
        "graph_replays": graph_iters,
        "bitwise": True,
        "kernel_counts": counts,
        "kernel_us": kernel_us,
        "counter_inclusive_stage_us": (
            kernel_us["ar_rmsnorm_reduce_scatter_gfx1250"]
            + kernel_us["ar_rmsnorm_gfx1250"]
        ),
        "eager_reference_us": eager_reference_us,
        "eager_fused_us": eager_fused_us,
        "graph_reference_us": graph_reference_us,
        "graph_fused_us": graph_fused_us,
    }


def _worker(
    rank: int,
    init_method: str,
    rows: tuple[int, ...],
    eager_iters: int,
    graph_iters: int,
):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if not _is_gfx1250(device):
        raise RuntimeError(f"rank {rank}: F6 validation requires gfx1250")
    dist.init_process_group(
        "gloo", world_size=TP, rank=rank, init_method=init_method
    )
    ca = CustomAllreduce(dist.group.WORLD, device)
    if ca is None or ca.disabled:
        raise RuntimeError(f"rank {rank}: gfx1250 custom all-reduce is disabled")

    retained = []
    try:
        dist.barrier()
        torch.cuda.synchronize()
        results = []
        for m in rows:
            for quant in (False, True):
                result = _run_case(
                    rank,
                    ca,
                    m,
                    quant,
                    eager_iters,
                    graph_iters,
                    device,
                )
                retained.append(result)
                results.append(result)
        torch.cuda.synchronize()
        dist.barrier()
        return results
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        retained.clear()
        ca.close()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=list(DEFAULT_ROWS))
    parser.add_argument("--eager-iters", type=int, default=20)
    parser.add_argument("--graph-iters", type=int, default=100)
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="whole TP4 run timeout in seconds; expiry is treated as a deadlock",
    )
    parser.add_argument("--json", default="", help="optional summary output path")
    args = parser.parse_args()
    if tuple(args.rows) != DEFAULT_ROWS:
        LOG.warning("non-acceptance rows requested: %s", args.rows)
    if any(m <= 0 for m in args.rows):
        raise ValueError("--rows must be positive")

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    ctx = get_context("spawn")
    pool = ctx.Pool(processes=TP)
    try:
        pending = pool.starmap_async(
            _worker,
            [
                (
                    rank,
                    init_method,
                    tuple(args.rows),
                    args.eager_iters,
                    args.graph_iters,
                )
                for rank in range(TP)
            ],
        )
        per_rank = pending.get(timeout=args.timeout)
        pool.close()
        pool.join()
    except TimeoutError as error:
        pool.terminate()
        pool.join()
        raise RuntimeError(
            f"TP4 F6 validation exceeded {args.timeout}s; possible deadlock"
        ) from error
    except BaseException:
        pool.terminate()
        pool.join()
        raise

    summary = {
        "tp": TP,
        "hidden": HIDDEN,
        "rows": args.rows,
        "eager_iters": args.eager_iters,
        "graph_iters": args.graph_iters,
        "ranks": per_rank,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json:
        with open(args.json, "w") as output:
            output.write(text + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
