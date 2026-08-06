# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.moe_sorting_kernel import moe_sorting_flydsl
from aiter.ops.flydsl.mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2
from aiter.ops.flydsl.mxfp4_v2_tune_utils import (
    populate_v2_intermediate_from_ref,
)
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import shuffle_scale, shuffle_weight
from aiter.test_common import checkAllclose
from aiter.utility import fp4_utils

_SKIP_GFX950_FLYDSL = pytest.mark.skipif(
    get_gfx() != "gfx950" or not is_flydsl_available(),
    reason="gfx950 FlyDSL required",
)


def _check_close(ref, out, label):
    assert torch.isfinite(out).all(), f"{label}: output contains NaN or Inf"
    ref_f32 = ref.float().reshape(1, -1)
    out_f32 = out.float().reshape(1, -1)
    cosine = torch.nn.functional.cosine_similarity(
        ref_f32, out_f32, dim=1, eps=1e-12
    ).item()
    assert cosine >= 0.999, (
        f"{label}: cosine similarity {cosine:.6f} is below 0.999"
    )
    err = checkAllclose(ref, out, msg=label, atol=0.01, rtol=0.05)
    assert err == 0 or err <= 0.05, f"{label}: error ratio {err} exceeds 0.05"


def test_check_close_rejects_all_zero_output():
    ref = torch.tensor((0.6, -0.4, 0.2), dtype=torch.float32)
    out = torch.zeros_like(ref)
    with pytest.raises(AssertionError, match="cosine similarity"):
        _check_close(ref, out, "all_zero")


def _dequant_fp4(value, scale, cols):
    values = fp4_utils.mxfp4_to_f32(value).view(*value.shape[:-1], cols // 32, 32)
    scales = fp4_utils.e8m0_to_f32(scale).view(*value.shape[:-1], cols // 32, 1)
    return (values * scales).view(*value.shape[:-1], cols)


def prepare_direct_stage2_case(inter_dim, bk, *, bm=32, seed=123):
    token, model_dim, expert, topk = 33, 256, 2, 2
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    token_idx = torch.arange(token, dtype=torch.int32, device=device)
    topk_ids = torch.stack((token_idx % expert, (token_idx + 1) % expert), dim=1)
    topk_weights = torch.tensor(
        (0.625, 0.375), dtype=torch.float32, device=device
    ).repeat(token, 1)

    max_padded = token * topk + expert * bm - topk
    max_sorted = (max_padded + bm - 1) // bm * bm
    sorted_ids = torch.empty(max_sorted, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_sorted, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(max_sorted // bm, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    moe_buf = torch.empty((token, model_dim), dtype=torch.bfloat16, device=device)
    moe_sorting_flydsl(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        expert,
        bm,
    )
    torch.cuda.synchronize()
    n_sorted = int(num_valid_ids[0].item())

    inter = (
        torch.randn(
            (token, topk, inter_dim), dtype=torch.bfloat16, device=device
        )
        / 10
    )
    w2 = (
        torch.randn(
            (expert, model_dim, inter_dim), dtype=torch.bfloat16, device=device
        )
        / 10
    )

    inter_q, inter_scale = per_1x32_f4_quant(
        inter.view(token * topk, inter_dim),
        quant_dtype=dtypes.fp4x2,
        shuffle=False,
    )
    w2_q, w2_scale = per_1x32_f4_quant(
        w2, quant_dtype=dtypes.fp4x2, shuffle=False
    )

    layout = {
        "sti": sorted_ids,
        "swt": sorted_weights,
        "sei": sorted_expert_ids,
        "cumsum": num_valid_ids,
        "n": n_sorted,
        "max_sorted": max_sorted,
    }
    # Reconstruct the native GEMM1 sorted-row payload and E8M0 layout. The v2
    # helper is byte-compatible when sort block and compute block are both BM32.
    populate_v2_intermediate_from_ref(
        {"ref1": inter, "stage2_adtype": "fp4"},
        layout,
        token,
        topk,
        bm,
    )

    inter_dequant = _dequant_fp4(inter_q, inter_scale, inter_dim).view(
        token, topk, inter_dim
    )
    w2_dequant = _dequant_fp4(w2_q, w2_scale, inter_dim)
    ref = torch.zeros((token, model_dim), dtype=torch.float32, device=device)
    for slot in range(topk):
        expert_weight = w2_dequant[topk_ids[:, slot].long()]
        route = torch.bmm(
            expert_weight, inter_dequant[:, slot].unsqueeze(-1)
        ).squeeze(-1)
        ref += route * topk_weights[:, slot, None]

    out = torch.zeros((token, model_dim), dtype=torch.bfloat16, device=device)
    launch_kwargs = {
        "inter_sorted_quant": layout["isq"],
        "inter_sorted_shuffled_scale": layout["iss"],
        "w2_u8": shuffle_weight(
            w2_q, (16, 16), is_guinterleave=False, gate_up=False
        ),
        "w2_scale_u8": shuffle_scale(
            w2_scale.view(expert * model_dim, inter_dim // 32),
            expert,
            is_guinterleave=False,
            gate_up=False,
        ),
        "sorted_expert_ids": sorted_expert_ids,
        "cumsum_tensor": num_valid_ids,
        "sorted_token_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "flat_out": out,
        "M_logical": token,
        "max_sorted": max_sorted,
        "BM": bm,
        "use_nt": False,
        "atomic": True,
        "mxfp4out": False,
        "NE": expert,
        "D_HIDDEN": model_dim,
        "D_INTER": inter_dim,
        "topk": topk,
        "BN": 256,
        "BK": bk,
    }
    return {"ref": ref.to(torch.bfloat16), "out": out, "kwargs": launch_kwargs}


def run_direct_stage2(case):
    case["out"].zero_()
    flydsl_mxfp4_gemm2(**case["kwargs"])
    return case["out"]


@pytest.mark.parametrize(
    ("inter_dim", "bk"),
    [
        pytest.param(384, 128, id="k384-bk128"),
        pytest.param(512, 256, id="k512-bk256"),
    ],
)
@_SKIP_GFX950_FLYDSL
def test_native_atomic_bm32_direct_stage2(inter_dim, bk):
    case = prepare_direct_stage2_case(inter_dim, bk)
    out = run_direct_stage2(case)
    _check_close(
        case["ref"].float(), out.float(), f"native_bm32_k{inter_dim}_bk{bk}"
    )


@pytest.mark.parametrize(
    "inter_dim",
    [
        pytest.param(512, id="k512-4tiles"),
        pytest.param(1024, id="k1024-8tiles"),
    ],
)
@_SKIP_GFX950_FLYDSL
def test_native_atomic_bm16_bk128_deep_k(inter_dim):
    case = prepare_direct_stage2_case(inter_dim, 128, bm=16)
    out = run_direct_stage2(case)
    _check_close(
        case["ref"].float(), out.float(), f"native_bm16_k{inter_dim}_bk128"
    )
