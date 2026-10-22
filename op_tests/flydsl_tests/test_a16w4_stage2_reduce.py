# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""a16w4 stage2 reduce must replace the caller buffer, not atomic-add into it.

Atomic mode adds into the supplied output, so a nonzero prefill is part of the
result. Reduce mode writes one row per expert slot and then stores the sum over
that output, so the same prefill is discarded.

fp4 is not native on gfx942, so this is asserted on gfx950 only.
"""

from __future__ import annotations

import pytest
import torch

from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

_PREFILL = 3.0

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx950",
    reason="fp4 stage2 reduce is asserted on gfx950; fp4 is not native on gfx942",
)


def _mismatch_ratio(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return (
        (~torch.isclose(reference, actual, rtol=1e-2, atol=1e-2)).float().mean().item()
    )


def _sorted_activation(
    slots, topk_ids, sorted_ids, sorted_expert_ids, num_valid, block_m
):
    token, _topk, inter_dim = slots.shape
    gathered = torch.zeros(
        (sorted_ids.shape[0], inter_dim), dtype=torch.bfloat16, device=slots.device
    )
    token_of = sorted_ids[:num_valid] & 0x00FFFFFF
    rows = (token_of < token).nonzero(as_tuple=True)[0]
    for row in rows.tolist():
        token_id = int(token_of[row].item())
        expert = int(sorted_expert_ids[row // block_m].item())
        slot = (topk_ids[token_id] == expert).nonzero(as_tuple=True)[0]
        if slot.numel() == 0:
            continue
        gathered[row] = slots[token_id, int(slot[0].item())]
    return gathered


def _case(tile_m, tile_k):
    token, model_dim, inter_dim, expert, topk = 8, 256, 256, 8, 2
    assert inter_dim % tile_k == 0
    torch.manual_seed(0)
    slots = torch.randn((token, topk, inter_dim), device="cuda", dtype=dtypes.bf16) / 4
    w2 = torch.randn((expert, model_dim, inter_dim), device="cuda", dtype=dtypes.bf16)
    w2 = w2 / 10
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_q = w2_q.view(expert, model_dim, inter_dim // 2)
    topk_ids = torch.stack(
        [(torch.arange(token, device="cuda") + slot) % expert for slot in range(topk)],
        dim=1,
    ).to(torch.int32)
    topk_weights = torch.rand((token, topk), device="cuda", dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _moe_buf = (
        moe_sorting(topk_ids, topk_weights, expert, model_dim, dtypes.bf16, tile_m)
    )
    needed = sorted_expert_ids.shape[0] * tile_m
    if sorted_ids.shape[0] < needed:
        pad = needed - sorted_ids.shape[0]
        sorted_ids = torch.cat(
            [
                sorted_ids,
                torch.full(
                    (pad,), token, dtype=sorted_ids.dtype, device=sorted_ids.device
                ),
            ]
        )
        sorted_weights = torch.cat(
            [
                sorted_weights,
                torch.zeros(
                    pad, dtype=sorted_weights.dtype, device=sorted_weights.device
                ),
            ]
        )
    inter_sorted = _sorted_activation(
        slots,
        topk_ids,
        sorted_ids,
        sorted_expert_ids,
        int(num_valid_ids.reshape(-1)[0].item()),
        tile_m,
    )
    return {
        "token": token,
        "model_dim": model_dim,
        "topk": topk,
        "inter_sorted": inter_sorted,
        "w2": shuffle_weight_a16w4(w2_q, 16, False),
        "w2_scale": shuffle_scale_a16w4(w2_scale, expert, False),
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
    }


def _run(data, tile_m, tile_k, mode, prefill):
    output = torch.full(
        (data["token"], data["model_dim"]),
        prefill,
        dtype=dtypes.bf16,
        device="cuda",
    )
    flydsl_moe_stage2(
        inter_states=data["inter_sorted"],
        w2=data["w2"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        out=output,
        topk=data["topk"],
        tile_m=tile_m,
        tile_n=128,
        tile_k=tile_k,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype="bf16",
        mode=mode,
        w2_scale=data["w2_scale"],
        sorted_weights=data["sorted_weights"],
    )
    torch.cuda.synchronize()
    assert torch.isfinite(output).all(), mode
    return output


@pytest.mark.parametrize(
    ("tile_m", "tile_k"),
    [
        pytest.param(16, 128, id="t16x128x128"),
        pytest.param(32, 256, id="t32x128x256"),
    ],
)
def test_prefilled_output_atomic_keeps_reduce_replaces(tile_m, tile_k):
    data = _case(tile_m, tile_k)
    atomic_zero = _run(data, tile_m, tile_k, "atomic", 0.0)
    atomic_prefill = _run(data, tile_m, tile_k, "atomic", _PREFILL)
    reduce_prefill = _run(data, tile_m, tile_k, "reduce", _PREFILL)

    atomic_ratio = _mismatch_ratio(atomic_zero, atomic_prefill)
    reduce_ratio = _mismatch_ratio(atomic_zero, reduce_prefill)
    assert atomic_ratio > 0.5, {"atomic": atomic_ratio, "reduce": reduce_ratio}
    assert reduce_ratio <= 0.01, {"atomic": atomic_ratio, "reduce": reduce_ratio}
