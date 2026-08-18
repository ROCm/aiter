# SPDX-License-Identifier: MIT
"""EP8 local-expert A4W4 SiLU compute baseline (no fused transport yet)."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def _logits_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    x, y = lhs.double(), rhs.double()
    return float(1.0 - (2.0 * (x * y).sum() / (x.square().sum() + y.square().sum())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activation", choices=("silu", "swiglu", "situv2"), default="silu"
    )
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 8:
        raise RuntimeError("requires EP world_size=8")
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("gloo")

    from aiter.ops.flydsl.kernels.megamoe_tile import a4w4_dense_reference, prepare_local_a4w4_weights
    from aiter.ops.flydsl.kernels.megamoe_tile.compute_v2 import run_local_ep_a4w4

    # Resource-light semantic case: one accepted route per rank.
    torch.manual_seed(2026)
    m, h, inter, experts, topk = 4, 1024, 256, 32, 2
    epr = experts // world
    x = (torch.randn(m, h, device=dev) * 0.1).to(torch.bfloat16)
    w1_global = (
        torch.randn(experts, 2 * inter, h, device=dev) * 0.05
    ).to(torch.bfloat16)
    w2_global = (
        torch.randn(experts, h, inter, device=dev) * 0.05
    ).to(torch.bfloat16)
    ids = torch.tensor(
        [[0, 16], [4, 20], [8, 24], [12, 28]],
        dtype=torch.int32,
        device=dev,
    )
    weights = torch.tensor([[0.6, 0.4]] * m, dtype=torch.float32, device=dev)

    lo, hi = rank * epr, (rank + 1) * epr
    local_weights = prepare_local_a4w4_weights(
        w1_global[lo:hi].contiguous(), w2_global[lo:hi].contiguous()
    )
    mask = torch.zeros(experts, dtype=torch.int32, device=dev)
    mask[lo:hi] = 1
    partial = run_local_ep_a4w4(
        x,
        ids,
        weights,
        local_weights,
        global_experts=experts,
        local_expert_mask=mask,
        activation=args.activation,
    )

    # Sum rank-local expert contributions.  Gloo CPU reduction keeps this smoke
    # independent from RCCL and isolates the expert-remap/computation semantics.
    combined = partial.float().cpu()
    dist.all_reduce(combined, op=dist.ReduceOp.SUM)

    if rank == 0:
        global_prepared = prepare_local_a4w4_weights(w1_global, w2_global)
        ref = a4w4_dense_reference(
            x, ids, weights, global_prepared, activation=args.activation
        ).cpu()
        diff = _logits_diff(combined, ref)
        print(
            "MEGAMOE_TILE_EP8_COMPUTE",
            f"activation={args.activation}",
            "logits_diff=",
            diff,
            flush=True,
        )
        ok = diff < 2e-2 and bool(torch.isfinite(combined).all())
    else:
        ok = True
    ok_tensor = torch.tensor([int(ok)], dtype=torch.int32)
    dist.broadcast(ok_tensor, src=0)
    dist.barrier()
    if not ok_tensor.item():
        raise AssertionError("EP8 A4W4 local-expert partial sum failed reference")
    if rank == 0:
        print("MEGAMOE_TILE_EP8_COMPUTE_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
