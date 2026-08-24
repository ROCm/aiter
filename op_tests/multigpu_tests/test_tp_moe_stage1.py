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

    # sort_block_m must divide the stage1 tile_m
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
    assert op.m_logical(1) == 8
    assert op.m_logical(128) == 1024
    # 8*6 + 384*32 - 6
    assert op.max_sorted(1) == 8 * 6 + 384 * 32 - 6
    # 1024*6 + 384*32 - 6
    assert op.max_sorted(128) == 1024 * 6 + 384 * 32 - 6
    print("case_capacity OK")


CASES = {
    "construct": case_construct_validates,
    "capacity": case_capacity,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="construct")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    CASES[args.case]()


if __name__ == "__main__":
    main()
