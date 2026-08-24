"""TPMoEStage1 correctness tests. Run with:

    torchrun --standalone --nproc_per_node=8 \
        op_tests/multigpu_tests/test_tp_moe_stage1.py --case <name>

Single-rank cases (construct/capacity) also run as plain `python3 <file>`.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.mega_moe.tp_moe_stage1 import (
    TPMoEStage1,
    TPMoEStage1Output,
)

NETWORK = dict(
    model_dim=7168,
    experts=384,
    topk=6,
    swiglu_limit=10.0,
)
STAGE1_KERNEL = "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_xcd4_kw4_fp8"


def _fake_w1(experts, inter_dim, model_dim, device):
    """Byte-shaped stand-in for a preshuffled MXFP4 W1 (values are irrelevant here)."""
    w1 = torch.zeros(
        (experts, 2 * inter_dim, model_dim // 2), dtype=torch.uint8, device=device
    )
    w1_scale = torch.full(
        (experts, 2 * inter_dim, model_dim // 32), 0x7F, dtype=torch.uint8, device=device
    )
    return w1, w1_scale


def case_construct_validates():
    device = torch.device("cuda", 0)
    inter_dim = 384
    w1, w1_scale = _fake_w1(NETWORK["experts"], inter_dim, NETWORK["model_dim"], device)

    # tp_size must be 4 or 8
    try:
        TPMoEStage1(
            model_dim=NETWORK["model_dim"],
            inter_dim=inter_dim,
            experts=NETWORK["experts"],
            topk=NETWORK["topk"],
            w1=w1,
            w1_scale=w1_scale,
            tp_size=2,
            tp_rank=0,
            device=device,
        )
    except ValueError as exc:
        assert "tp_size" in str(exc), exc
    else:
        raise AssertionError("tp_size=2 must be rejected")

    # sort_block_m must equal the stage1 tile_m
    try:
        TPMoEStage1(
            model_dim=NETWORK["model_dim"],
            inter_dim=inter_dim,
            experts=NETWORK["experts"],
            topk=NETWORK["topk"],
            w1=w1,
            w1_scale=w1_scale,
            tp_size=8,
            tp_rank=0,
            device=device,
            sort_block_m=48,
        )
    except ValueError as exc:
        assert "sort_block_m" in str(exc), exc
    else:
        raise AssertionError("sort_block_m=48 must be rejected")

    # supplying only one of tp_size / tp_rank must be rejected with "together"
    try:
        TPMoEStage1(
            model_dim=NETWORK["model_dim"],
            inter_dim=inter_dim,
            experts=NETWORK["experts"],
            topk=NETWORK["topk"],
            w1=w1,
            w1_scale=w1_scale,
            tp_size=8,
            tp_rank=None,
            device=device,
        )
    except ValueError as exc:
        assert "together" in str(exc), exc
    else:
        raise AssertionError("tp_size=8, tp_rank=None must be rejected")

    op = TPMoEStage1(
        model_dim=NETWORK["model_dim"],
        inter_dim=inter_dim,
        experts=NETWORK["experts"],
        topk=NETWORK["topk"],
        w1=w1,
        w1_scale=w1_scale,
        tp_size=8,
        tp_rank=0,
        device=device,
        swiglu_limit=NETWORK["swiglu_limit"],
        stage1_kernel_name=STAGE1_KERNEL,
    )
    assert op.tp_size == 8
    assert op.sort_block_m == 32
    assert op.stage1_params["tile_m"] == 32
    assert op.stage1_params["tile_n"] == 64
    assert op.stage1_params["tile_k"] == 256
    assert op.stage1_params["gate_mode"] == "interleave"
    print("case_construct_validates OK")


def case_capacity():
    device = torch.device("cuda", 0)
    inter_dim = 384
    w1, w1_scale = _fake_w1(NETWORK["experts"], inter_dim, NETWORK["model_dim"], device)
    op = TPMoEStage1(
        model_dim=NETWORK["model_dim"],
        inter_dim=inter_dim,
        experts=NETWORK["experts"],
        topk=NETWORK["topk"],
        w1=w1,
        w1_scale=w1_scale,
        tp_size=8,
        tp_rank=0,
        device=device,
        stage1_kernel_name=STAGE1_KERNEL,
    )
    # M_global = tp_size * m_local; max_sorted matches moe_sorting's own formula.
    assert op.m_logical_for(1) == 8
    assert op.m_logical_for(128) == 1024
    # 8*6 + 384*32 - 6
    assert op.max_sorted_for(1) == 8 * 6 + 384 * 32 - 6
    # 1024*6 + 384*32 - 6
    assert op.max_sorted_for(128) == 1024 * 6 + 384 * 32 - 6
    print("case_capacity OK")


def _setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    return rank, world, device


def case_all_gather():
    rank, world, device = _setup_dist()
    assert world in (4, 8), f"run this case with 4 or 8 ranks, got {world}"
    m_local = 5
    inter_dim = 384
    w1, w1_scale = _fake_w1(NETWORK["experts"], inter_dim, NETWORK["model_dim"], device)
    op = TPMoEStage1(
        model_dim=NETWORK["model_dim"],
        inter_dim=inter_dim,
        experts=NETWORK["experts"],
        topk=NETWORK["topk"],
        w1=w1,
        w1_scale=w1_scale,
        device=device,
        stage1_kernel_name=STAGE1_KERNEL,
    )
    assert op.tp_size == world and op.tp_rank == rank

    # Rank-identifiable payloads so we can assert the concatenation order.
    x = torch.full(
        (m_local, NETWORK["model_dim"]), float(rank), dtype=torch.bfloat16, device=device
    )
    ids = torch.full(
        (m_local, NETWORK["topk"]), rank, dtype=torch.int32, device=device
    )
    wts = torch.full(
        (m_local, NETWORK["topk"]), float(rank), dtype=torch.float32, device=device
    )

    gx, gwts, gids = op._all_gather_inputs(x, wts, ids)

    m_global = world * m_local
    assert gx.shape == (m_global, NETWORK["model_dim"]), gx.shape
    assert gids.shape == (m_global, NETWORK["topk"]), gids.shape
    assert gwts.shape == (m_global, NETWORK["topk"]), gwts.shape
    assert gx.dtype == torch.bfloat16 and gids.dtype == torch.int32
    assert gwts.dtype == torch.float32
    assert gx.is_contiguous() and gids.is_contiguous() and gwts.is_contiguous()

    for src in range(world):
        lo, hi = src * m_local, (src + 1) * m_local
        assert torch.all(gx[lo:hi] == float(src)), f"activation block {src} misordered"
        assert torch.all(gids[lo:hi] == src), f"topk_ids block {src} misordered"
        assert torch.all(gwts[lo:hi] == float(src)), f"weights block {src} misordered"

    if rank == 0:
        print("case_all_gather OK")
    dist.barrier()
    dist.destroy_process_group()


CASES = {
    "construct": case_construct_validates,
    "capacity": case_capacity,
    "all_gather": case_all_gather,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="construct")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    CASES[args.case]()


if __name__ == "__main__":
    main()
