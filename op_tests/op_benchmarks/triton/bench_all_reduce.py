# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon all-reduce against RCCL and the aiter custom all-reduce on 4 gfx1250 GPUs.
The time is the slowest rank, in a HIP graph."""

import multiprocessing as mp

import torch
import torch.distributed as dist
import triton

import aiter
from aiter import logger
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms.all_reduce import (
    all_reduce,
    all_reduce_free,
    all_reduce_init,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton_synced,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
)

WORLD_SIZE = 4
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def benchmark(args, rank, device, tp, inboxes, calls):
    ca = tp.device_communicator.ca_comm
    providers = ["gluon", "rccl"]
    if ca is not None and not ca.disabled:
        providers.append("car")
    fns = {
        "gluon": lambda x, out: all_reduce(x, tp.cpu_group, inboxes, calls),
        "rccl": lambda x, out: dist.all_reduce(x, group=tp.device_group),
        "car": lambda x, out: ca.all_reduce(x, out=out, registered_input=False),
    }
    unit = "us" if args.metric == "time" else "GB/s"
    config = triton.testing.Benchmark(
        x_names=["tokens"],
        x_vals=args.tokens,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(tokens, provider):
        numel = tokens * args.hidden

        def gen():
            x = torch.randn(numel, dtype=torch.bfloat16, device=device)
            return x, torch.empty_like(x)

        bufs = generate_rotating_buffers_pool(
            [gen], [["x", "out"]], **get_rot_bufs_kwargs(args)
        )
        ms = do_bench_aiter_triton_synced(
            fns[provider],
            bufs,
            dist_group=tp.cpu_group,
            **get_do_bench_graph_kwargs(args),
        )
        per_rank = [None] * WORLD_SIZE
        dist.all_gather_object(per_rank, ms, group=tp.cpu_group)
        ms = max(per_rank)
        if args.metric == "time":
            return ms * 1e3
        return numel * 2 * 1e-9 / (ms * 1e-3)

    write_csv = args.o and rank == 0
    _run.run(save_path="." if write_csv else None, print_data=rank == 0)


def _worker(rank, init_method, args):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    max_bytes = max(args.tokens) * args.hidden * 2
    inboxes, calls = all_reduce_init(tp.cpu_group, max_bytes)
    try:
        benchmark(args, rank, device, tp, inboxes, calls)
        assert calls[-1].item() == 0, "a Gluon all-reduce poll timed out"
    finally:
        all_reduce_free(tp.cpu_group, inboxes)
        aiter.destroy_dist_env()


def parse_args():
    parser = get_parser(kernel_name="all-reduce")
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=TOKENS,
        help="token counts; a message is tokens x hidden bf16 elements",
    )
    parser.add_argument("--hidden", type=int, default=2048, help="hidden size")
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to CSV"
    )
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    # Small pools: a 256 MB pool of 4 KB messages is 32768 calls in one graph.
    override_parser_action(parser, action="--rot-bufs-target-mb", default=1)
    args = parser.parse_args()
    assert all(t * args.hidden % 8 == 0 for t in args.tokens), "16-byte messages"
    return args


def main():
    args = parse_args()
    if torch.cuda.device_count() < WORLD_SIZE or arch_info.get_arch() != "gfx1250":
        logger.warning("bench_all_reduce needs %d gfx1250 GPUs: skipped", WORLD_SIZE)
        return
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=WORLD_SIZE) as pool:
        results = [
            pool.apply_async(_worker, (rank, init_method, args))
            for rank in range(WORLD_SIZE)
        ]
        for result in results:
            result.get(timeout=1800)


if __name__ == "__main__":
    main()
