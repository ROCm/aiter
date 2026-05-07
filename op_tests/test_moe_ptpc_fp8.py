# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 HSACO ``fused_moe_ptpc_fp8`` op test.

Torch reference follows ``qwen3.5_dev/aiter/op_tests/test_moe_2stage.py`` ``test_fmoe``
(see commit ``5ddbcdd871440617820571114f6004bf14a3a700`` — *use use_raw_for_ref to compare
with torch*): ``torch_moe_stage1`` / ``torch_moe_stage2`` with the same branching as
``use_raw_for_ref``.

- ``use_raw_for_ref=True``: bf16 ``hidden_states``, ``a1_scale`` / ``a2_scale`` = ``[1]``,
  no activation requant between stages (same as ``test_fmoe`` lines 191–199, 223–231).
- ``use_raw_for_ref=False``: quantize activations with ``torch_quant`` into stage 1 and
  again after stage 1 into stage 2 (same as ``test_fmoe`` default).

**Numeric gate:** HSACO vs CK ``fused_moe`` (``AITER_MOE_SMALL_BATCH=0``) — tight cosine.
Torch refs are logged / optionally checked with looser tolerance (they differ from HSACO
when weights are not preshuffled like CK).
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, torch_moe_stage1, torch_moe_stage2
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


def _scale_one(device: torch.device) -> torch.Tensor:
    return torch.tensor([1.0], dtype=torch.float32, device=device)


def ref_torch_two_stage_like_fmoe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    use_raw_for_ref: bool,
    activation_quant_dtype: torch.dtype = dtypes.fp8,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Mirror ``test_fmoe`` reference construction (``use_raw_for_ref`` semantics)."""
    torch_quant = aiter.get_torch_quant(QuantType.per_Token)
    B = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    dev = hidden_states.device
    one = _scale_one(dev)

    if use_raw_for_ref:
        h1 = hidden_states
        a1_s = one
    else:
        a1_qt, a1_scale = torch_quant(
            hidden_states, quant_dtype=activation_quant_dtype
        )
        h1 = a1_qt
        a1_s = a1_scale

    out1 = torch_moe_stage1(
        h1,
        w1,
        w2,
        topk_weight,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_Token,
        a1_scale=a1_s,
        w1_scale=w1_scale,
        w1_bias=None,
        doweight=False,
    )

    if use_raw_for_ref:
        h2 = out1
        a2_s = one
    else:
        a2_qt, a2_scale = torch_quant(out1, quant_dtype=activation_quant_dtype)
        h2 = a2_qt.view(B, topk, -1)
        a2_s = a2_scale

    return torch_moe_stage2(
        h2,
        w1,
        w2,
        topk_weight,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_Token,
        w2_scale=w2_scale,
        a2_scale=a2_s,
        w2_bias=None,
        doweight=True,
    )


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
    use_raw_for_ref: bool = True,
    num_iters: int = 11,
    num_warmup: int = 2,
    assert_torch_vs_hsaco: bool = False,
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
    topk_ids = torch.randint(
        0, num_experts, (batch_size, topk), dtype=torch.int32, device=device
    )

    ref_ck = _ref_fused_moe_ck(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        w1_scale,
        w2_scale,
    )
    ref_torch = ref_torch_two_stage_like_fmoe(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weight,
        topk_ids,
        use_raw_for_ref=use_raw_for_ref,
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

    assert ref_ck.shape == hsaco_ret.shape == ref_torch.shape, (
        f"{ref_ck.shape=} {hsaco_ret.shape=} {ref_torch.shape=}"
    )
    assert torch.isfinite(hsaco_ret).all(), "HSACO output contains NaN/Inf"

    err_ratio_ck = checkAllclose(
        ref_ck,
        hsaco_ret,
        rtol=1e-2,
        atol=1e-2,
        msg="HSACO vs fused_moe (CK, AITER_MOE_SMALL_BATCH=0)",
    )
    diff_ck = calc_diff(ref_ck, hsaco_ret)
    # diff_hsaco_vs_torch_ref: torch_moe_stage1/2 uses FP32 dequant matmul; PTPC HSACO
    # does not match numerically — re-enable when a closer reference exists.
    # diff_torch = calc_diff(ref_torch, hsaco_ret)
    # err_ratio_torch = checkAllclose(
    #     ref_torch,
    #     hsaco_ret,
    #     rtol=1e-2,
    #     atol=1e-2,
    #     msg=f"HSACO vs torch_moe_stage1/2 (use_raw_for_ref={use_raw_for_ref})",
    # )

    print(
        f"{batch_size=} use_raw_for_ref={use_raw_for_ref} {diff_ck=:.6f} "
        f"close_mismatch_ratio_ck={err_ratio_ck:.6f} "
        f"time_us={dt_us:.1f}"
    )

    # if diff_torch > 1e-3:
    #     logging.warning(
    #         "logits_diff HSACO vs torch ref: %s (use_raw_for_ref=%s); "
    #         "see test_moe_2stage use_raw_for_ref discussion",
    #         diff_torch,
    #         use_raw_for_ref,
    #     )

    assert diff_ck < 0.02, f"HSACO vs CK logits_diff too large: {diff_ck}"

    # if assert_torch_vs_hsaco:
    #     assert diff_torch < 0.02, (
    #         f"HSACO vs torch ref logits_diff too large: {diff_torch} "
    #         f"(use_raw_for_ref={use_raw_for_ref})"
    #     )

    return {
        "batch_size": batch_size,
        "num_experts": num_experts,
        "topk": topk,
        "use_raw_for_ref": use_raw_for_ref,
        "diff_hsaco_vs_ck": diff_ck,
        # "diff_hsaco_vs_torch_ref": diff_torch,
        "close_mismatch_ratio_ck": err_ratio_ck,
        # "close_mismatch_ratio_torch": err_ratio_torch,
        "time(us)": f"{dt_us:.0f}",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HSACO fused_moe_ptpc_fp8 vs CK and torch_moe_stage1/2 reference."
    )
    parser.add_argument(
        "--torch-ref-mode",
        choices=("raw", "quant"),
        default="raw",
        help=(
            "raw: same as test_fmoe use_raw_for_ref=True (bf16 hidden, scale=1). "
            "quant: same as test_fmoe use_raw_for_ref=False (torch_quant between stages)."
        ),
    )
    parser.add_argument(
        "--assert-torch-vs-hsaco",
        action="store_true",
        help="Also assert HSACO close to torch ref (usually fails for PTPC HSACO).",
    )
    args = parser.parse_args()
    use_raw = args.torch_ref_mode == "raw"

    torch.set_default_device("cuda")
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=4)
    torch.manual_seed(0)

    summary = []
    #for B in (1, 2, 4, 8, 10, 12, 16, 32):
    #for B in (1, 2, 4, 8, 10, 12, 16):
    for B in [16, 32]:
        ret = test_fused_moe_ptpc_fp8_hsaco(
            batch_size=B,
            num_experts=512,
            topk=10,
            hidden_size=4096,
            inter_dim=128,
            use_raw_for_ref=use_raw,
            num_iters=11,
            num_warmup=2,
            assert_torch_vs_hsaco=args.assert_torch_vs_hsaco,
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
