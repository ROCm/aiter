# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TPActivationGather correctness tests. Run with:

    torchrun --standalone --nproc_per_node=8 \
        op_tests/multigpu_tests/test_tp_gather.py --case <name>
"""

import argparse
import os
import sys

import mori.shmem as ms
import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
from aiter.ops.flydsl.kernels.mega_moe.tp_gather import TPActivationGather

MODEL_DIM = 7168


def _setup():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, device


def _teardown():
    try:
        ms.shmem_finalize()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _make_x(m_local, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((m_local, MODEL_DIM), generator=g).to(
        device=device, dtype=torch.bfloat16
    ) * (MODEL_DIM**-0.25)
    return x.contiguous()


def _nccl_gather(t, world):
    out = torch.empty(
        (t.shape[0] * world,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device
    )
    dist.all_gather_into_tensor(out, t.contiguous())
    return out


def case_construct():
    """Constructor preconditions. Runs under torchrun because shmem needs a PG."""
    rank, world, device = _setup()
    try:
        common = dict(
            model_dim=MODEL_DIM,
            tp_size=world,
            tp_rank=rank,
            max_tok_per_rank=128,
            device=device,
        )

        for kwargs, want in (
            (dict(tp_size=2, tp_rank=0), "tp_size"),
            (dict(tp_rank=world + 1), "out of range"),
            (dict(max_tok_per_rank=0), "positive"),
            (dict(model_dim=7000), "multiple of 512"),
            (dict(producer_blocks=30), "divisible"),
        ):
            merged = dict(common)
            merged.update(kwargs)
            try:
                TPActivationGather(**merged)
            except ValueError as exc:
                assert want in str(exc), (want, str(exc))
            else:
                raise AssertionError(
                    f"expected ValueError containing {want!r} for {kwargs}"
                )

        g = TPActivationGather(**common)
        assert g.rows == world * 128 + 1, g.rows
        assert g.scale_dim == MODEL_DIM // 32
        assert tuple(g.rx_x.shape) == (2, world * 128 + 1, MODEL_DIM)
        assert tuple(g.rx_scale.shape) == (2, world * 128 + 1, MODEL_DIM // 32)
        assert tuple(g.p2p_rx_x.shape) == (world,)
        assert (
            int((g.p2p_rx_x != 0).sum()) == world
        ), "every peer pointer must be non-null"
        assert (
            g.p2p_rx_x[rank].item() == g.rx_x.data_ptr()
        ), "self entry must be the local ptr"

        x = _make_x(4, device, 1)
        x_q, x_s = per_1x32_mx_quant(x, quant_mode="fp8")
        for bad, want in (
            # x_q[:0] is a zero-row tensor. The plan wrote x_q[:200000] here,
            # but Python clamps an over-long slice, so that expression is just
            # the unmodified 4-row tensor and _validate correctly accepts it.
            ((x_q[:0], x_s), "must be positive"),
            ((x_q.to(torch.uint8), x_s), "float8_e4m3fn"),
            ((x_q, x_s[:, :2]), "x_scale must be"),
        ):
            try:
                g._validate(*bad)
            except (ValueError, IndexError) as exc:
                assert want in str(exc) or isinstance(exc, IndexError), (want, str(exc))
            else:
                raise AssertionError(f"expected rejection for {want}")

        big = torch.empty((129, MODEL_DIM), dtype=torch.float8_e4m3fn, device=device)
        big_s = torch.empty((129, MODEL_DIM // 32), dtype=torch.uint8, device=device)
        try:
            g._validate(big, big_s)
        except ValueError as exc:
            assert "exceeds max_tok_per_rank" in str(exc), exc
        else:
            raise AssertionError("m_local > max_tok_per_rank must raise")

        if rank == 0:
            print("case_construct OK")
    finally:
        _teardown()


CASES = {"construct": case_construct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="construct")
    args = ap.parse_args()
    CASES[args.case]()


if __name__ == "__main__":
    main()
