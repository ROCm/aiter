# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-GPU checks for the fused expert-sum + all-reduce op.

The cross-rank fused kernel itself needs >= 2 GPUs to exercise (custom AR is
disabled at world_size==1). What is verifiable on a single GPU:

  * the expert weighted-sum (moe_sum) that the fusion folds in, vs a torch ref
  * the torch.library fake/meta shape (torch.compile trace-safety)
  * the world_size==1 functional path (fused op degrades to a local moe_sum)
  * a baseline moe_sum microbench (the standalone pass the fusion removes)

Run:  HIP_VISIBLE_DEVICES=7 python op_tests/test_fused_moe_sum_all_reduce_single_gpu.py
      HIP_VISIBLE_DEVICES=7 python op_tests/test_fused_moe_sum_all_reduce_single_gpu.py --bench
"""

import argparse

import torch

import aiter
from aiter.test_common import checkAllclose, run_perftest

SHAPES = [(17, 2, 6144), (128, 8, 6144), (1, 8, 6144)]


def _torch_ref(x):
    # moe_sum reduces over the topk axis (dim=-2).
    return x.float().sum(dim=-2).to(x.dtype)


def test_moe_sum_matches_torch():
    for dtype in (torch.bfloat16, torch.float16):
        for (m, topk, n) in SHAPES:
            x = torch.randn((m, topk, n), dtype=dtype, device="cuda") * 0.1
            out = torch.empty((m, n), dtype=dtype, device="cuda")
            aiter.moe_sum(x, out)
            checkAllclose(
                _torch_ref(x),
                out,
                msg=f"moe_sum m={m} topk={topk} n={n} {dtype}",
                atol=5e-2,
                rtol=5e-2,
            )


def test_fake_meta_shape():
    from aiter.dist.parallel_state import fused_moe_sum_all_reduce_fake

    x = torch.empty((17, 8, 6144), dtype=torch.bfloat16, device="meta")
    out = fused_moe_sum_all_reduce_fake(x, group_name="tp")
    assert out.shape == (17, 6144), out.shape
    assert out.dtype == x.dtype


def test_world_size_1_path():
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        set_custom_all_reduce,
    )
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    torch.cuda.set_device(0)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
    )
    ensure_model_parallel_initialized(1, 1)
    try:
        x = torch.randn((128, 8, 6144), dtype=torch.bfloat16, device="cuda") * 0.1
        out = get_tp_group().fused_moe_sum_all_reduce(x)
        ref = torch.empty((128, 6144), dtype=x.dtype, device="cuda")
        aiter.moe_sum(x, ref)
        # world_size==1 routes through the same moe_sum -> must match exactly.
        checkAllclose(ref, out, msg="world_size=1 fused == moe_sum", atol=0, rtol=0)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def bench_moe_sum():
    print("baseline moe_sum microbench (the standalone pass the fusion removes):")
    for dtype in (torch.bfloat16,):
        for (m, topk, n) in [(1, 8, 6144), (128, 8, 6144), (2048, 8, 6144), (8192, 8, 6144)]:
            x = torch.randn((m, topk, n), dtype=dtype, device="cuda") * 0.1
            out = torch.empty((m, n), dtype=dtype, device="cuda")
            _, us = run_perftest(aiter.moe_sum, x, out, num_warmup=5, num_iters=50)
            moved_gb = (x.numel() + out.numel()) * x.element_size() / 1e9
            print(
                f"  m={m:6d} topk={topk} n={n} {dtype}: "
                f"{us:8.2f} us   {moved_gb / (us * 1e-6):7.1f} GB/s"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="also run the moe_sum microbench")
    args = ap.parse_args()

    test_moe_sum_matches_torch()
    print("moe_sum arithmetic vs torch: OK")
    test_fake_meta_shape()
    print("fake/meta shape (trace-safety): OK")
    if args.bench:
        bench_moe_sum()
    # Run the dist path last since it initializes/destroys a process group.
    test_world_size_1_path()
    print("world_size=1 functional path: OK")
