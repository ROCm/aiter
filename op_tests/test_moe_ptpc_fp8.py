# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 HSACO ``fused_moe_ptpc_fp8`` op test.

**Numeric gate:** HSACO vs CK ``fused_moe`` (``AITER_MOE_SMALL_BATCH=0``) — tight cosine.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from aiter.fused_moe_ptpc_fp8 import fused_moe_ptpc_fp8
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose, run_perftest


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return float(1 - sim)


def _ref_fused_moe_ck(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    prev = os.environ.get("AITER_MOE_SMALL_BATCH")
    os.environ["AITER_MOE_SMALL_BATCH"] = "0"
    try:
        return fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            expert_mask=None,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_Token,
            doweight_stage1=False,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            num_local_tokens=None,
            moe_sorting_dispatch_policy=0,
        )
    finally:
        if prev is None:
            os.environ.pop("AITER_MOE_SMALL_BATCH", None)
        else:
            os.environ["AITER_MOE_SMALL_BATCH"] = prev


def test_fused_moe_ptpc_fp8_hsaco(
    batch_size: int,
    num_experts: int,
    topk: int,
    hidden_size: int,
    inter_dim: int,
    *,
    num_iters: int = 11,
    num_warmup: int = 2,
) -> Optional[dict[str, Any]]:
    if not torch.cuda.is_available():
        print("skip test_fused_moe_ptpc_fp8_hsaco: CUDA not available")
        return None

    if get_gfx() != "gfx942":
        print(f"skip test_fused_moe_ptpc_fp8_hsaco: unsupported platform {get_gfx()!r} (need gfx942)")
        return None

    if topk != 10 or inter_dim != 128 or hidden_size != 4096:
        print(
            "skip test_fused_moe_ptpc_fp8_hsaco: dimensions must match prebuilt "
            f"HSACO (got topk={topk}, inter_dim={inter_dim}, hidden_size={hidden_size})"
        )
        return None

    device = torch.device("cuda")
    weight_dtype = torch.float8_e4m3fnuz
    torch.manual_seed(0)

    hidden_states = torch.randn(
        batch_size, hidden_size, dtype=torch.bfloat16, device=device
    )
    w1_bf16 = torch.randn(
        num_experts, 2 * inter_dim, hidden_size, dtype=torch.bfloat16, device=device
    )
    w2_bf16 = torch.randn(
        num_experts, hidden_size, inter_dim, dtype=torch.bfloat16, device=device
    )

    torch_quant = aiter.get_torch_quant(QuantType.per_Token)
    w1, w1_scale = torch_quant(w1_bf16, quant_dtype=weight_dtype)
    w2, w2_scale = torch_quant(w2_bf16, quant_dtype=weight_dtype)

    topk_weight = torch.randn(batch_size, topk, dtype=torch.float32, device=device)
    topk_ids = torch.randperm(batch_size * topk, dtype=torch.int32, device=device).reshape(batch_size, topk)

    ref_ck = _ref_fused_moe_ck(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        w1_scale,
        w2_scale,
    )

    hsaco_ret, dt_us = run_perftest(
        fused_moe_ptpc_fp8,
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        ActivationType.Silu,
        QuantType.per_Token,
        w1_scale,
        w2_scale,
        None,
        None,
        0,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )

    if hsaco_ret is None:
        print(
            "test_fused_moe_ptpc_fp8_hsaco: fused_moe_ptpc_fp8 returned None "
            "(gates / missing .co / shape mismatch)"
        )
        return None

    assert ref_ck.shape == hsaco_ret.shape, f"{ref_ck.shape=} {hsaco_ret.shape=}"
    assert torch.isfinite(hsaco_ret).all(), "HSACO output contains NaN/Inf"

    err_ratio_ck = checkAllclose(
        ref_ck,
        hsaco_ret,
        rtol=1e-2,
        atol=1e-2,
        msg="HSACO vs fused_moe (CK, AITER_MOE_SMALL_BATCH=0)",
    )
    diff_ck = calc_diff(ref_ck, hsaco_ret)

    print(
        f"{batch_size=} {diff_ck=:.6f} "
        f"close_mismatch_ratio_ck={err_ratio_ck:.6f} "
        f"time_us={dt_us:.1f}"
    )

    assert diff_ck < 0.02, f"HSACO vs CK logits_diff too large: {diff_ck}"

    return {
        "batch_size": batch_size,
        "num_experts": num_experts,
        "topk": topk,
        "diff_hsaco_vs_ck": diff_ck,
        "close_mismatch_ratio_ck": err_ratio_ck,
        "time(us)": f"{dt_us:.0f}",
    }


if __name__ == "__main__":
    torch.set_default_device("cuda")
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=4)
    torch.manual_seed(0)

    summary = []
    for B in (1, 2, 4, 8, 10, 12, 16, 32):
        ret = test_fused_moe_ptpc_fp8_hsaco(
            batch_size=B,
            num_experts=512,
            topk=10,
            hidden_size=4096,
            inter_dim=128,
            num_iters=11,
            num_warmup=2,
        )
        if ret is not None:
            summary.append(ret)

    if summary:
        try:
            import pandas as pd

            print(pd.DataFrame(summary).to_markdown(index=False))
        except Exception as e:
            print(e)
            print(summary)
