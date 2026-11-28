# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP4 FP4 row-gather latency and per-rank payload bandwidth on gfx1250."""

import csv
import functools
import os
from pathlib import Path

import torch
import torch.distributed as dist

from aiter import logger
from aiter.ops.triton.comms.all_gather_fp4 import Fp4AllGather, all_gather_fp4
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.comms.test_all_gather_fp4 import generate_fp4_gather_inputs

MODELS = [
    ("kimi_k3", 7168, 16),
    ("minimax_m31_flash", 6144, 4),
    ("dsv41_flash", 5120, 6),
]


def main():
    parser = get_parser("TP4 FP4 all-gather")
    parser.add_argument("--rows", nargs="+", type=int, default=[8192, 16384])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--providers", nargs="+", choices=["gluon"], default=["gluon"])
    args = parser.parse_args()
    if any(m <= 0 or m % 4 for m in args.rows):
        raise ValueError("global row counts must be positive multiples of four")
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("gloo")
    cpu = dist.group.WORLD
    results = []
    try:
        for model, hidden, topk in MODELS:
            for include_metadata in (False, True):
                metadata_bytes = hidden // 32 + 8 * topk if include_metadata else 0
                comm = Fp4AllGather(
                    cpu, max(args.rows) // 4, hidden // 2, metadata_bytes, device=device
                )
                try:
                    for m in args.rows:
                        local_rows = m // 4
                        for provider in args.providers:
                            bencher = RotatingBufferBench.from_dict(
                                {
                                    **get_do_bench_graph_kwargs(args),
                                    **get_rot_bufs_kwargs(args),
                                }
                            )
                            bencher.build_pool(
                                [
                                    functools.partial(
                                        generate_fp4_gather_inputs,
                                        local_rows,
                                        hidden // 2,
                                        metadata_bytes,
                                        device,
                                    )
                                ],
                                [["codes", "metadata", "out"]],
                            )
                            fn = functools.partial(all_gather_fp4, comm=comm)
                            ms = bencher.bench_synced(fn, dist_group=cpu)
                            times = [None] * 4
                            dist.all_gather_object(times, ms, group=cpu)
                            us = max(times) * 1000
                            full_bytes = m * (hidden // 2 + metadata_bytes)
                            row = {
                                "model": model,
                                "hidden": hidden,
                                "topk": topk,
                                "global_M": m,
                                "local_M": local_rows,
                                "payload": (
                                    "fp4+metadata" if include_metadata else "fp4"
                                ),
                                "provider": provider,
                                "us": us,
                                "output_bytes": full_bytes,
                                "rank_input_bytes": full_bytes // 4,
                                "algorithm_GBs": full_bytes / us / 1000,
                                "bus_GBs": 0.75 * full_bytes / us / 1000,
                                "bidirectional_GBs": 1.5 * full_bytes / us / 1000,
                            }
                            results.append(row)
                            if rank == 0:
                                logger.info(
                                    "%s M=%d %s %s: %.3f us, %.2f GB/s payload, %.2f GB/s bus, %.2f GB/s TX+RX",
                                    model,
                                    m,
                                    row["payload"],
                                    provider,
                                    us,
                                    row["algorithm_GBs"],
                                    row["bus_GBs"],
                                    row["bidirectional_GBs"],
                                )
                                path = Path(args.output)
                                path.parent.mkdir(parents=True, exist_ok=True)
                                with path.open("w") as f:
                                    writer = csv.DictWriter(f, fieldnames=list(row))
                                    writer.writeheader()
                                    writer.writerows(results)
                finally:
                    errors = comm.errors.item()
                    comm.close()
                    if errors:
                        raise RuntimeError(f"{errors} peer readiness waits timed out")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
