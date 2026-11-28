# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the batch-invariant KDA decode step (Gluon kda_bi_decode, gfx950) against
AITER's KDA decode (fused_recurrent_kda) for one token per request."""

import functools

import torch
import triton

from aiter.ops.triton.attention.kda import fused_recurrent_kda
from aiter.ops.triton.kimi_delta_attn.batch_invariant.kda_bi_decode import (
    kda_bi_decode,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.kimi_delta_attn.batch_invariant.test_kda_bi_decode import (
    generate_kda_bi_decode_inputs,
)

_REQUESTS = [1, 8, 32, 128, 256]
_KEYS = ["q", "k", "v", "g", "beta", "A_log", "dt_bias", "lower_bound", "state",
         "cache", "slots"]  # fmt: skip
_REC_KEYS = ["q", "k", "v", "g", "beta", "A_log", "dt_bias", "initial_state",
             "cu_seqlens", "ssm_state_indices"]  # fmt: skip
D = 128


def _recurrent_inputs(R, H, seed=0):
    """fused_recurrent_kda arguments for R requests, one token each ([V, K] state)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v, g = (
        torch.randn((1, R, H, D), dtype=torch.bfloat16, device="cuda", generator=gen)
        for _ in range(4)
    )
    beta = torch.rand((1, R, H), dtype=torch.bfloat16, device="cuda", generator=gen)
    A_log = torch.rand(H, dtype=torch.float32, device="cuda", generator=gen)
    dt_bias = 0.1 * torch.randn(
        H * D, dtype=torch.float32, device="cuda", generator=gen
    )
    state = 0.05 * torch.randn(
        (R, H, D, D), dtype=torch.float32, device="cuda", generator=gen
    )
    cu = torch.arange(R + 1, device="cuda")
    idx = torch.randperm(R, device="cuda", generator=gen).to(torch.int32)
    return q, k, v, g, beta, A_log, dt_bias, state, cu, idx


def run_benchmark(args) -> None:
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    benchmark = triton.testing.Benchmark(
        x_names=["R"],
        x_vals=[args.requests] if args.requests else _REQUESTS,
        line_arg="provider",
        line_vals=["bi", "existing"],
        line_names=["kda_bi_decode", "fused_recurrent_kda"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="Time (ms)",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench(R: int, provider: str) -> float:
        if provider == "bi":
            gen = functools.partial(generate_kda_bi_decode_inputs, R, args.heads)
            bencher.build_pool([gen], [_KEYS])
            return bencher.bench(kda_bi_decode)
        gen = functools.partial(_recurrent_inputs, R, args.heads)
        bencher.build_pool([gen], [_REC_KEYS])
        fn = functools.partial(
            fused_recurrent_kda,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=-5.0,
            inplace_final_state=True,
        )
        return bencher.bench(fn)

    bench.run(print_data=True, show_plots=False)


def main(args=None) -> None:
    parser = get_parser("KDA decode batch invariant")
    parser.add_argument("--requests", type=int, default=None, help="one request count")
    parser.add_argument("--heads", type=int, default=12, help="heads per GPU")
    run_benchmark(parser.parse_args(args))


if __name__ == "__main__":
    main()
