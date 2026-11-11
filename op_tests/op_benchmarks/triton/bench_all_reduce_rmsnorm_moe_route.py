# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon all-reduce + residual add + RMSNorm (+ MXFP8) + router + top-k + moe_sorting on 2,
4 or 8 gfx950 GPUs. The defaults are the MiniMax-M3.1 decode shape (hidden 6144, 128 experts,
top-4 sigmoid + bias, Gemma norm, fp32 router). The time is the slowest rank, in a HIP
graph."""

import multiprocessing as mp

import torch
import torch.distributed as dist
import triton

import aiter
from aiter import logger
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms.fused.all_reduce_rmsnorm_moe_route import (
    AllReduceRmsnormMoeRoute,
    fused_all_reduce_rmsnorm_moe_route,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method
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

TOKENS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256]


def benchmark(args, device, tp, comm):
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
    wdt = torch.float32 if args.router_fp32 else torch.bfloat16
    norm_w = 0.1 * torch.randn((args.hidden,), dtype=torch.float32, device=device)
    norm_w = norm_w.to(torch.bfloat16)
    router_w = 0.02 * torch.randn(
        (args.experts, args.hidden), dtype=torch.float32, device=device
    )
    router_w = router_w.to(wdt)
    bias = 0.01 * torch.randn((args.experts,), dtype=torch.float32, device=device)

    @triton.testing.perf_report([config])
    def _run(tokens, provider):
        shape = (tokens, args.hidden)

        def gen_rot():
            x = torch.randn(shape, dtype=torch.bfloat16, device=device)
            residual = torch.randn(shape, dtype=torch.bfloat16, device=device)
            return x, residual

        bencher.build_pool(
            [gen_rot],
            [["x", "residual"]],
            [lambda: (norm_w, router_w, bias)],
            [["norm_w", "router_w", "bias"]],
        )

        def fn(x, residual, norm_w, router_w, bias):
            return fused_all_reduce_rmsnorm_moe_route(
                x,
                residual,
                norm_w,
                1e-6,
                router_w,
                comm,
                correction_bias=bias,
                scoring="sigmoid",
                renormalize=True,
                routed_scaling_factor=2.0,
                gemma=True,
                quant=True,
            )

        ms = bencher.bench_synced(fn, dist_group=tp.cpu_group)
        per_rank = [None] * args.world_size
        dist.all_gather_object(per_rank, ms, group=tp.cpu_group)
        return max(per_rank) * 1e3

    rank = dist.get_rank(tp.cpu_group)
    write_csv = args.o and rank == 0
    _run.run(save_path="." if write_csv else None, print_data=rank == 0)


def _worker(rank, init_method, args):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(args.world_size, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    comm = AllReduceRmsnormMoeRoute(
        tp.cpu_group,
        max(args.tokens),
        args.hidden,
        args.experts,
        args.topk,
        router_fp32=args.router_fp32,
    )
    try:
        benchmark(args, device, tp, comm)
        assert comm.error() == [0, 0, 0, 0], "a wait timed out"
    finally:
        comm.close()
        aiter.destroy_dist_env()


def parse_args():
    parser = get_parser(kernel_name="All-reduce + RMSNorm + MoE route")
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=TOKENS, help="token counts"
    )
    parser.add_argument(
        "--world-size", type=int, default=4, choices=[2, 4, 8], help="ranks"
    )
    parser.add_argument("--hidden", type=int, default=6144, help="hidden size")
    parser.add_argument("--experts", type=int, default=128, help="routed experts")
    parser.add_argument("--topk", type=int, default=4, help="experts a token")
    parser.add_argument(
        "--router-bf16",
        dest="router_fp32",
        action="store_false",
        default=True,
        help="bf16 router weights (default fp32)",
    )
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to CSV"
    )
    override_parser_action(parser, action="--metric", choices=["time"], default="time")
    return parser.parse_args()


def main():
    args = parse_args()
    if torch.cuda.device_count() < args.world_size or arch_info.get_arch() != "gfx950":
        logger.warning("bench needs %d gfx950 GPUs: skipped", args.world_size)
        return
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=args.world_size) as pool:
        results = [
            pool.apply_async(_worker, (rank, init_method, args))
            for rank in range(args.world_size)
        ]
        for result in results:
            result.get(timeout=1800)


if __name__ == "__main__":
    main()
