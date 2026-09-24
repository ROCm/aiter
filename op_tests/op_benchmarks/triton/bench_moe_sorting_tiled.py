# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare existing and opted-in tiled M3 sorting; no config search."""

import argparse

import torch
import triton

from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton.moe.moe_sorting_tiled import try_m3_tiled_sort
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext


def benchmark(save_path=None):
    torch.manual_seed(0)
    routed = torch.rand((32768, 128), device="cuda").topk(4, dim=1).indices.int()
    ids = torch.cat(
        (routed, torch.full((32768, 1), 128, dtype=torch.int32, device="cuda")), dim=1
    )
    weights = torch.rand(ids.shape, device=ids.device)
    weights[:, -1] = 1
    config = triton.testing.Benchmark(
        x_names=["aux"],
        x_vals=[False, True],
        line_arg="provider",
        line_vals=["existing", "tiled"],
        line_names=["existing dispatcher", "tiled"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report(config)
    def run(aux, provider):
        output_aux = "opus" if aux else False
        assert (
            try_m3_tiled_sort(
                ids,
                weights,
                129,
                6144,
                torch.bfloat16,
                64,
                accumulate=False,
                output_aux=output_aux,
            )
            is not None
        ), "M3 tiled sorting is unavailable for this configuration"
        fn = lambda: moe_sorting(
            ids,
            weights,
            129,
            6144,
            torch.bfloat16,
            64,
            accumulate=False,
            output_aux=output_aux,
            use_tiled_sort=provider == "tiled",
        )
        return 1000 * triton.testing.do_bench_cudagraph(fn, rep=100)

    run.run(save_path=save_path, print_data=True, show_plots=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-path", default=None, help="Optional perf_report output directory"
    )
    args = parser.parse_args()
    if (
        not torch.cuda.is_available()
        or torch.version.hip is None
        or get_gfx_runtime() != "gfx950"
    ):
        parser.error("requires a gfx950 ROCm device")
    benchmark(args.save_path)
