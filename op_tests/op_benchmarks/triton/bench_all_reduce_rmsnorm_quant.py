# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon all-reduce + add + RMSNorm against RCCL + add_rmsnorm, CAR + add_rmsnorm
and the aiter fused CAR on 4 gfx1250 GPUs. The time is the slowest rank, in a HIP
graph."""

import multiprocessing as mp

import torch
import torch.distributed as dist
import triton

import aiter
from aiter import logger, rmsnorm2d_fwd_with_add
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import get_tp_group, graph_capture
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms.all_reduce import all_reduce_free, all_reduce_init
from aiter.ops.triton.comms.fused.all_reduce_rmsnorm_quant import (
    all_reduce_rmsnorm_quant,
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
EPS = 1e-6
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def benchmark(args, rank, device, tp, inboxes, calls):
    ca = tp.device_communicator.ca_comm
    providers = ["gluon", "gluon_fp8", "rccl"]
    if ca is not None and not ca.disabled:
        providers += ["car", "car_fused"]
    config = triton.testing.Benchmark(
        x_names=["tokens"],
        x_vals=args.tokens,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        ylabel="us",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(tokens, provider):
        shape = (tokens, args.hidden)
        weight = torch.randn(args.hidden, dtype=torch.bfloat16, device=device)
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
        res_out = torch.empty_like(out)
        group = tp.cpu_group

        def rccl(x, res):
            dist.all_reduce(x, group=tp.device_group)
            rmsnorm2d_fwd_with_add(out, x, res, res_out, weight, EPS, True, 0)

        def car(x, res):
            ar = tensor_model_parallel_all_reduce(x)
            rmsnorm2d_fwd_with_add(out, ar, res, res_out, weight, EPS, True, 0)

        fns = {
            "gluon": lambda x, res: all_reduce_rmsnorm_quant(
                x, res, weight, EPS, True, False, group, inboxes, calls
            ),
            "gluon_fp8": lambda x, res: all_reduce_rmsnorm_quant(
                x, res, weight, EPS, True, True, group, inboxes, calls
            ),
            "rccl": rccl,
            "car": car,
            # The fused CAR kernel itself: the dispatch runs 2 kernels at small sizes.
            "car_fused": lambda x, res: ca.custom_fused_ar_rms(
                x, res, weight, EPS, True, gemma_norm=True
            ),
        }

        def gen():
            x = torch.randn(shape, dtype=torch.bfloat16, device=device)
            return x, torch.randn(shape, dtype=torch.bfloat16, device=device)

        bufs = generate_rotating_buffers_pool(
            [gen], [["x", "res"]], **get_rot_bufs_kwargs(args)
        )
        with graph_capture():  # the CAR paths capture as they do in a model
            ms = do_bench_aiter_triton_synced(
                fns[provider],
                bufs,
                dist_group=tp.cpu_group,
                **get_do_bench_graph_kwargs(args),
            )
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
    max_bytes = max(args.tokens) * args.hidden * 2
    inboxes, calls = all_reduce_init(tp.cpu_group, max_bytes)
    try:
        benchmark(args, rank, device, tp, inboxes, calls)
        assert calls[-1].item() == 0, "a Gluon poll timed out"
    finally:
        all_reduce_free(tp.cpu_group, inboxes)
        aiter.destroy_dist_env()


def parse_args():
    parser = get_parser(kernel_name="all-reduce + RMSNorm")
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=TOKENS, help="token counts"
    )
    parser.add_argument("--hidden", type=int, default=2048, help="hidden size")
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to CSV"
    )
    override_parser_action(parser, action="--metric", choices=["time"], default="time")
    override_parser_action(parser, action="--graph-kernels-per-graph", default=100)
    override_parser_action(parser, action="--rot-bufs-target-mb", default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if torch.cuda.device_count() < WORLD_SIZE or arch_info.get_arch() != "gfx1250":
        logger.warning("bench needs %d gfx1250 GPUs: skipped", WORLD_SIZE)
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
