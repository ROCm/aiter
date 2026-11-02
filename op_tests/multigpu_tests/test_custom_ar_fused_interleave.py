# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mixed-size custom all-reduce traffic must stay correct across skewed ranks.

Speculative decoding alternates a small fused all-reduce + RMSNorm (the draft
model's final norm) and small all-reduces with larger ones (the target model),
all through one custom all-reduce instance. Ranks drift apart while the host
does per-step work, so each step here delays every rank by a different amount.
A collective that is only correct when ranks arrive together corrupts a later
collective on some ranks, or never completes.
"""

import argparse
import logging
from multiprocessing import Pool, set_start_method

import torch
import torch.distributed as dist

from aiter.dist.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
)
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

logger = logging.getLogger("aiter")
set_start_method("spawn", force=True)


def _inputs(seed, shape, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=gen).to(device=device, dtype=torch.bfloat16)


def _skew(rank, step):
    # Up to ~1 ms of GPU-side delay, different per rank and step.
    torch.cuda._sleep(((rank * 7919 + step * 104729) % 1000) * 2000)


def _reduced(seeds, shape, device):
    return sum(_inputs(s, shape, device).float() for s in seeds)


def _is_gfx1250() -> bool:
    return (
        torch.cuda.is_available()
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    )


def _rmsnorm(h, weight, eps):
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def _worker(
    tp_size, rank, init_method, steps, hidden, small_rows, big_rows, fused_big_rows
):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp_size, 1)
    eps = 1e-6
    weight = _inputs(-1, (hidden,), device)
    small, big = (small_rows, hidden), (big_rows, hidden)
    fused_big_shapes = tuple((rows, hidden) for rows in fused_big_rows)
    worst = {"fused": 0.0, "fused_big": 0.0, "small": 0.0, "big": 0.0, "gather": 0.0}
    for step in range(1, steps + 1):
        seeds = [8 * (step * tp_size + r) for r in range(tp_size)]
        residual = _inputs(-2 - step, small, device)
        fused_big = fused_big_shapes[step % len(fused_big_shapes)]
        residual_big = _inputs(-3 - step, fused_big, device)

        _skew(rank, step)
        out, _ = tensor_model_parallel_fused_allreduce_rmsnorm(
            _inputs(seeds[rank], small, device), residual, weight, eps
        )
        # Above 4 MiB, where gfx1250 also takes the fused kernel.
        out_big, _ = tensor_model_parallel_fused_allreduce_rmsnorm(
            _inputs(seeds[rank] + 3, fused_big, device), residual_big, weight, eps
        )
        _skew(rank, step + 1)
        y_small = tensor_model_parallel_all_reduce(
            _inputs(seeds[rank] + 1, small, device)
        )
        _skew(rank, step + 2)
        y_big = tensor_model_parallel_all_reduce(_inputs(seeds[rank] + 2, big, device))
        # The barrier-synchronized path (the LM head's logits gather).
        y_gather = tensor_model_parallel_all_gather(
            _inputs(seeds[rank] + 4, big, device), use_custom=True
        )

        h = _reduced(seeds, small, device) + residual.float()
        h_big = (
            _reduced([s + 3 for s in seeds], fused_big, device) + residual_big.float()
        )
        errs = {
            "fused": out.float() - _rmsnorm(h, weight, eps),
            "fused_big": out_big.float() - _rmsnorm(h_big, weight, eps),
            "small": y_small.float() - _reduced([s + 1 for s in seeds], small, device),
            "big": y_big.float() - _reduced([s + 2 for s in seeds], big, device),
            "gather": y_gather.float()
            - torch.cat([_inputs(s + 4, big, device).float() for s in seeds], dim=-1),
        }
        for k, e in errs.items():
            worst[k] = max(worst[k], e.abs().max().item())
    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    if dist.is_initialized():
        dist.destroy_process_group()
    return worst


def test_mixed_size_custom_ar_under_skew(
    tp_size=4,
    steps=300,
    hidden=6144,
    small_rows=4,
    big_rows=16,
    fused_big_rows=(384, 1024),
    timeout_s=600,
):
    # The kernels and dispatch this guards are gfx1250-only.
    if not _is_gfx1250():
        return
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    args = [
        (tp_size, r, init_method, steps, hidden, small_rows, big_rows, fused_big_rows)
        for r in range(tp_size)
    ]
    with Pool(processes=tp_size) as pool:
        results = pool.starmap_async(_worker, args).get(timeout=timeout_s)
    for rank, worst in enumerate(results):
        logger.info(f"rank {rank}: max abs error {worst}")
        # bf16 sums of tp_size N(0, 1) inputs stay well under 0.1 of the fp32 sum.
        for k, err in worst.items():
            assert err < 0.1, f"rank {rank}: {k} result corrupted (max err {err})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    test_mixed_size_custom_ar_under_skew(tp_size=args.tp, steps=args.steps)
    print("PASSED" if _is_gfx1250() else "SKIPPED (gfx1250 only)")
