# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Multi-GPU correctness test for vocab-parallel cross-entropy under TP.

Spawns ``world_size`` processes, each holding a ``[B, SQ, V_local]`` logit
shard.  The test verifies that:

1. The scalar loss returned by every rank matches the single-GPU reference
   (computed with the full ``[B, SQ, V]`` logits on rank 0 with
   ``dist_group=None``).
2. The per-token gradient written into the logit shard (in-place) matches the
   corresponding columns of the single-GPU gradient after ``all_gather``.

This exercises the stat-merge loop (``for i in range(1, world_size)``) and
the ``all_gather_into_tensor`` memory layout — the paths that the single-GPU
tests cannot reach.
"""

import multiprocessing as mp
import sys
import traceback

import torch

import aiter
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.cross_entropy import cross_entropy_forward
from aiter.test_common import ensure_spawn_method


def _worker(
    rank: int,
    world_size: int,
    logits_cpu: torch.Tensor,
    target_cpu: torch.Tensor,
    ignore_idx: int,
    reduce_loss: bool,
    distributed_init_method: str,
):
    """Per-rank worker.  Returns ``(loss_scalar_or_tensor, grad_shard_cpu)``."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    try:
        aiter.init_dist_env(
            world_size,
            rank,
            distributed_init_method=distributed_init_method,
        )
        group = get_tp_group().device_group

        _B, _SQ, V = logits_cpu.shape
        V_local = V // world_size
        logits_shard = (
            logits_cpu[..., rank * V_local : (rank + 1) * V_local]
            .contiguous()
            .to(device)
        )
        target = target_cpu.to(device)

        loss, grad = cross_entropy_forward(
            logits_shard,
            target,
            label_smoothing=0.0,
            reduce_loss=reduce_loss,
            dist_group=group,
            ignore_idx=ignore_idx,
        )

        if reduce_loss:
            loss_out = loss.item()
        else:
            loss_out = loss.cpu()

        return loss_out, grad.cpu()

    except Exception as e:
        print(
            f"[rank {rank}] ERROR: {e}\n"
            + "".join(traceback.format_exception(*sys.exc_info())),
            flush=True,
        )
        raise
    finally:
        aiter.destroy_dist_env()


def _run_tp(world_size, logits, target, ignore_idx, reduce_loss, init_method):
    """Spawn world_size workers and collect (loss, grad_shard) per rank."""
    pool = mp.Pool(processes=world_size)
    futures = [
        pool.apply_async(
            _worker,
            args=(
                rank,
                world_size,
                logits.cpu(),
                target.cpu(),
                ignore_idx,
                reduce_loss,
                init_method,
            ),
        )
        for rank in range(world_size)
    ]
    pool.close()
    pool.join()
    return [f.get() for f in futures]


def test_cross_entropy_tp_loss(world_size: int = 2):
    """Loss under TP matches the single-GPU reference."""
    torch.manual_seed(42)
    B, SQ, V = 2, 8, 512  # V must be divisible by world_size
    assert V % world_size == 0
    dtype = torch.bfloat16

    logits = torch.randn(B, SQ, V, dtype=dtype)
    target = torch.randint(0, V, (B, SQ))

    # Single-GPU reference on rank-0 GPU.
    device = torch.device("cuda:0")
    ref_loss, _ = cross_entropy_forward(
        logits.to(device),
        target.to(device),
        label_smoothing=0.0,
        reduce_loss=True,
        dist_group=None,
        ignore_idx=-100,
    )
    ref_loss = ref_loss.item()

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    results = _run_tp(world_size, logits, target, -100, True, init_method)

    for rank, (loss_val, _) in enumerate(results):
        assert abs(loss_val - ref_loss) < 1e-2, (
            f"rank {rank}: TP loss {loss_val:.6f} diverges from "
            f"single-GPU ref {ref_loss:.6f}"
        )

    print(
        f"PASSED test_cross_entropy_tp_loss  "
        f"world_size={world_size}  ref={ref_loss:.6f}  "
        f"tp={[f'{r[0]:.6f}' for r in results]}"
    )


def test_cross_entropy_tp_grad(world_size: int = 2):
    """Gradient shards under TP match the corresponding columns of the single-GPU gradient."""
    torch.manual_seed(7)
    B, SQ, V = 2, 8, 512
    assert V % world_size == 0
    dtype = torch.bfloat16
    V_local = V // world_size

    logits = torch.randn(B, SQ, V, dtype=dtype)
    target = torch.randint(0, V, (B, SQ))

    # Single-GPU reference gradient.
    device = torch.device("cuda:0")
    _, ref_grad = cross_entropy_forward(
        logits.to(device),
        target.to(device),
        label_smoothing=0.0,
        reduce_loss=True,
        dist_group=None,
        ignore_idx=-100,
    )
    ref_grad = ref_grad.cpu().float()

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    results = _run_tp(world_size, logits, target, -100, True, init_method)

    for rank, (_, grad_shard) in enumerate(results):
        ref_cols = ref_grad[..., rank * V_local : (rank + 1) * V_local]
        diff = (grad_shard.float() - ref_cols).abs().max().item()
        assert diff < 1e-2, f"rank {rank}: max grad diff {diff:.4e} exceeds tolerance"

    print(
        f"PASSED test_cross_entropy_tp_grad  "
        f"world_size={world_size}  "
        f"max_diffs={[f'{results[r][1].float().sub(ref_grad[..., r*V_local:(r+1)*V_local]).abs().max().item():.2e}' for r in range(world_size)]}"
    )


def test_cross_entropy_tp_ignore_index(world_size: int = 2):
    """Ignored rows contribute zero loss and zero gradient under TP."""
    torch.manual_seed(3)
    B, SQ, V = 2, 8, 512
    assert V % world_size == 0
    dtype = torch.bfloat16
    ignore_idx = -100

    logits = torch.randn(B, SQ, V, dtype=dtype)
    target = torch.randint(0, V, (B, SQ))
    # Mask out half the rows.
    target.view(-1)[::2] = ignore_idx

    device = torch.device("cuda:0")
    ref_loss, _ref_grad = cross_entropy_forward(
        logits.to(device),
        target.to(device),
        label_smoothing=0.0,
        reduce_loss=True,
        dist_group=None,
        ignore_idx=ignore_idx,
    )
    ref_loss = ref_loss.item()

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    results = _run_tp(world_size, logits, target, ignore_idx, True, init_method)

    for rank, (loss_val, _) in enumerate(results):
        assert (
            abs(loss_val - ref_loss) < 1e-2
        ), f"rank {rank}: TP loss {loss_val:.6f} != ref {ref_loss:.6f} with ignore_index"

    print(
        f"PASSED test_cross_entropy_tp_ignore_index  "
        f"world_size={world_size}  ref={ref_loss:.6f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-GPU vocab-parallel cross-entropy correctness test"
    )
    parser.add_argument(
        "-n", "--num_gpus", type=int, default=2, help="TP world size (default: 2)"
    )
    args = parser.parse_args()

    ensure_spawn_method()

    n = args.num_gpus
    print(f"Running TP cross-entropy tests with world_size={n}\n")

    try:
        test_cross_entropy_tp_loss(n)
        test_cross_entropy_tp_grad(n)
        test_cross_entropy_tp_ignore_index(n)
        print("\nAll TP cross-entropy tests PASSED.")
    except Exception as e:  # noqa: BLE001
        print(f"\nTEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
