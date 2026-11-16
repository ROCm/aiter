# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the batch-invariant KDA chunk forward (Gluon kda_bi_fwd, gfx950) against the
chunked KDA forward (chunk_kimi_delta_attn) on the same inputs."""

import argparse

import torch
import triton

from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn
from aiter.ops.triton.kimi_delta_attn.batch_invariant.kda_bi import kda_bi_fwd
from op_tests.triton_tests.kimi_delta_attn.batch_invariant.test_kda_bi import (
    LOWER,
    make_inputs,
)

_TOKENS = [1024, 4096, 16384, 65536]


def run_benchmark(args) -> None:
    benchmark = triton.testing.Benchmark(
        x_names=["T"],
        x_vals=[args.tokens] if args.tokens else _TOKENS,
        line_arg="provider",
        line_vals=["kda_bi", "chunk_kda"],
        line_names=["kda_bi (batch invariant)", "chunk_kimi_delta_attn"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="Time (ms)",
        plot_name="bench_kda_bi",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench(T: int, provider: str) -> float:
        x = make_inputs(T, 0, h=args.heads)
        lens = [T // args.seqs] * args.seqs
        lens[-1] += T - sum(lens)
        cu_cpu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist())
        cu = cu_cpu.cuda()
        q, k, v, g, beta, a_log, dt_bias = x
        if provider == "kda_bi":

            def fn():
                return kda_bi_fwd(*x, LOWER, cu, cu_seqlens_cpu=cu_cpu)

        else:

            def fn():
                return chunk_kimi_delta_attn(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    A_log=a_log,
                    dt_bias=dt_bias,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    safe_gate=True,
                    lower_bound=LOWER,
                    cu_seqlens=cu,
                )

        return triton.testing.do_bench(fn)

    bench.run(print_data=True, show_plots=False)


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="KDA batch-invariant chunk forward")
    parser.add_argument(
        "--tokens", type=int, default=None, help="One T (default: a sweep)."
    )
    parser.add_argument("--heads", type=int, default=12, help="KDA heads on this GPU.")
    parser.add_argument(
        "--seqs", type=int, default=1, help="Equal-length sequences in T."
    )
    run_benchmark(parser.parse_args(args))


if __name__ == "__main__":
    main()
