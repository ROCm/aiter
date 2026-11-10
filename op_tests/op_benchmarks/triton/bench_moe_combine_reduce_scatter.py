# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon MoE top-k combine + reduce-scatter on 4 gfx950 GPUs. The time is the
slowest rank, in a HIP graph."""

import multiprocessing as mp

import torch
import torch.distributed as dist
import triton

import aiter
from aiter import logger
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms.fused.moe_combine_reduce_scatter import (
    MoeCombineReduceScatter,
    fused_moe_combine_reduce_scatter,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method
from op_tests.multigpu_tests.triton_test.test_moe_combine_reduce_scatter import (
    _routing,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

WORLD_SIZE = 4
TOKENS = [2343, 4096, 5857, 8192, 14394, 16383]


def benchmark(args, rank, device, tp, comm):
    config = triton.testing.Benchmark(
        x_names=["tokens"],
        x_vals=args.tokens,
        line_arg="provider",
        line_vals=["gluon"],
        line_names=["gluon"],
        ylabel="us",
        plot_name=get_caller_name_no_ext(),
        args={},
    )
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([config])
    def _run(tokens, provider):
        sort = _routing(
            tokens, args.topk, args.experts, args.hidden, 0, device, tp.device_group
        )
        sw, rev = sort[1], sort[6]
        rows = triton.cdiv(tokens, WORLD_SIZE)

        def gen_rot():
            inter = torch.randn(
                (sw.numel(), args.hidden), dtype=torch.bfloat16, device=device
            )
            out = torch.empty((rows, args.hidden), dtype=torch.bfloat16, device=device)
            return inter, out

        bencher.build_pool(
            [gen_rot], [["inter", "out"]], [lambda: (rev, sw)], [["rev", "sw"]]
        )

        def fn(inter, out, rev, sw):
            return fused_moe_combine_reduce_scatter(
                inter, rev, sw, tokens, comm, out=out
            )

        ms = bencher.bench_synced(fn, dist_group=tp.cpu_group)
        per_rank = [None] * WORLD_SIZE
        dist.all_gather_object(per_rank, ms, group=tp.cpu_group)
        return max(per_rank) * 1e3

    write_csv = args.o and rank == 0
    _run.run(save_path="." if write_csv else None, print_data=rank == 0)


def _worker(rank, init_method, args):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    comm = MoeCombineReduceScatter(
        tp.cpu_group, max(args.tokens), args.hidden, args.topk
    )
    try:
        benchmark(args, rank, device, tp, comm)
        assert comm.error() == [0, 0, 0, 0], "a wait timed out"
    finally:
        comm.close()
        aiter.destroy_dist_env()


def parse_args():
    parser = get_parser(kernel_name="MoE combine + reduce-scatter")
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=TOKENS, help="token counts"
    )
    parser.add_argument("--hidden", type=int, default=6144, help="hidden size")
    parser.add_argument("--topk", type=int, default=4, help="experts of a token")
    parser.add_argument("--experts", type=int, default=128, help="number of experts")
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to CSV"
    )
    override_parser_action(parser, action="--metric", choices=["time"], default="time")
    # A launch moves up to 1 GB: 100 launches a graph are enough.
    override_parser_action(parser, action="--graph-kernels-per-graph", default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    if torch.cuda.device_count() < WORLD_SIZE or arch_info.get_arch() != "gfx950":
        logger.warning("bench needs %d gfx950 GPUs: skipped", WORLD_SIZE)
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
