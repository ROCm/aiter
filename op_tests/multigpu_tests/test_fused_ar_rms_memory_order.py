# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os
import statistics
from multiprocessing import Pool, set_start_method

import torch
import torch.distributed as dist
import torch.nn.functional as F

from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import checkAllclose

set_start_method("spawn", force=True)

DEFAULT_SHAPES = [
    (1, 4096),
    (16, 4096),
    (64, 2880),
    (128, 2880),
    (1024, 2880),
    (1319, 2880),
    (2048, 2880),
]


def _parse_shapes(raw: str) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in shape.split(",")) for shape in raw.split(";")]


def _time_us(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0)
    return timings


def _run_rank(
    tp_size: int,
    rank: int,
    shapes: list[tuple[int, int]],
    iters: int,
    distributed_init_method: str,
    benchmark: bool,
    bench_warmup: int,
    bench_iters: int,
):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, 1)
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
    )

    rows = []
    for shape in shapes:
        max_abs_err = 0.0
        first_bad = -1
        for idx in range(iters):
            torch.manual_seed(1009 + idx)
            x = torch.randn(shape, dtype=torch.bfloat16, device=device)
            residual = torch.randn(shape, dtype=torch.bfloat16, device=device)
            weight = torch.randn((shape[-1],), dtype=torch.bfloat16, device=device)
            out, residual_out = tensor_model_parallel_fused_allreduce_rmsnorm(
                x, residual, weight, 1e-6
            )
            ref_residual = x * tp_size + residual
            ref = F.rms_norm(ref_residual, (shape[-1],), weight=weight, eps=1e-6)
            checkAllclose(
                ref,
                out,
                msg=f"fused_ar_rms output {shape=} iter={idx} rank={rank}",
                atol=0.5,
                rtol=0.5,
            )
            checkAllclose(
                ref_residual,
                residual_out,
                msg=f"fused_ar_rms residual {shape=} iter={idx} rank={rank}",
                atol=0.5,
                rtol=0.5,
            )
            err = max(
                (ref.float() - out.float()).abs().max().item(),
                (ref_residual.float() - residual_out.float()).abs().max().item(),
            )
            max_abs_err = max(max_abs_err, err)
            if err > 0.5 and first_bad < 0:
                first_bad = idx
                break

        row = {
            "shape": shape,
            "rank": rank,
            "max_abs_err": max_abs_err,
            "first_bad": first_bad,
        }

        if benchmark:
            torch.manual_seed(2026)
            x = torch.randn(shape, dtype=torch.bfloat16, device=device)
            residual = torch.randn(shape, dtype=torch.bfloat16, device=device)
            weight = torch.randn((shape[-1],), dtype=torch.bfloat16, device=device)

            def fused_call():
                return tensor_model_parallel_fused_allreduce_rmsnorm(
                    x, residual, weight, 1e-6
                )

            timings = _time_us(fused_call, bench_warmup, bench_iters)
            row.update(
                {
                    "p50_us": statistics.median(timings),
                    "p90_us": sorted(timings)[int(0.9 * (len(timings) - 1))],
                    "p99_us": sorted(timings)[int(0.99 * (len(timings) - 1))],
                }
            )
        rows.append(row)

    destroy_model_parallel()
    destroy_distributed_environment()
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Regression test for fused_allreduce_rmsnorm memory ordering"
    )
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--shapes",
        default=";".join(f"{m},{n}" for m, n in DEFAULT_SHAPES),
        help="semicolon-separated M,N shapes",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--bench-warmup", type=int, default=100)
    parser.add_argument("--bench-iters", type=int, default=500)
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49393"
    distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
    shapes = _parse_shapes(args.shapes)
    with Pool(processes=args.tp) as pool:
        futures = [
            pool.apply_async(
                _run_rank,
                args=(
                    args.tp,
                    rank,
                    shapes,
                    args.iters,
                    distributed_init_method,
                    args.benchmark,
                    args.bench_warmup,
                    args.bench_iters,
                ),
            )
            for rank in range(args.tp)
        ]
        all_rows = [row for future in futures for row in future.get()]

    failed = [row for row in all_rows if row["first_bad"] >= 0]
    for row in all_rows:
        print(row)
    if failed:
        raise AssertionError(f"memory-order regression detected: {failed[:4]}")


if __name__ == "__main__":
    main()
