# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL MXFP4 MoE with expert weights past 4 GiB (Kimi-K3.1 TP1: 9.9 GB w1, 4.9 GB w2).

Only the routed tail experts carry data; the weight and scale preshuffles are expert-major.
"""

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, moe_sorting, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import (
    moe_shuffle_scale,
    moe_shuffle_weight,
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.test_common import checkAllclose
from aiter.utility.fp4_utils import e8m0_shuffle

Q_TYPE = QuantType.per_1x32
BETA, LINEAR_BETA = 2.0, 1.5

_SKIP = pytest.mark.skipif(
    get_gfx_runtime() != "gfx950", reason="gfx950 FlyDSL required"
)


def _check_close(ref, out, label, atol=1.0, rtol=0.05, max_err_ratio=0.05):
    assert not out.isnan().any(), f"{label}: output has NaN"
    err = checkAllclose(ref, out, msg=label, atol=atol, rtol=rtol)
    assert err <= max_err_ratio, f"{label}: {err:.2%} of elements outside tolerance"


def _at_tail(tensor, tail, experts, *shape):
    rows = tensor.view(torch.uint8).reshape(tail, -1)
    full = torch.zeros((experts, rows.shape[1]), dtype=torch.uint8, device="cuda")
    full[-tail:] = rows
    return full.view(tensor.dtype).view(*shape)


@_SKIP
def test_a8w4_mixed_moe_experts_past_4gib():
    """Experts past 4 GiB of w1/w2 (Kimi-K3.1 TP1: 9.9 GB w1, 4.9 GB w2) read their own
    weights in both stages. Only the routed tail experts carry data; shuffles are expert-major.
    """
    torch.set_default_device("cuda")
    token, model_dim, inter_dim, E, topk, tail, block_m = 64, 3584, 3072, 896, 8, 16, 32
    assert E * model_dim * inter_dim // 2 > 2**32  # w2, so w1 too
    torch.manual_seed(31)
    inp = torch.randn((token, model_dim), dtype=torch.bfloat16) / 4
    w1 = torch.randn((tail, inter_dim * 2, model_dim), dtype=torch.bfloat16) / 4
    w2 = torch.randn((tail, model_dim, inter_dim), dtype=torch.bfloat16) / 4
    local_ids = torch.stack([torch.randperm(tail)[:topk] for _ in range(token)])
    topk_ids = (local_ids + (E - tail)).to(torch.int32)
    topk_weights = torch.rand((token, topk))
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, torch.bfloat16, block_m
    )
    a_q, a_scale = per_1x32_f8_scale_f8_quant(
        inp, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    w1_q, w1_scale = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(tail, inter_dim * 2, model_dim // 2)
    w2_q = w2_q.view(tail, model_dim, inter_dim // 2)
    ref1 = torch_moe_stage1(
        a_q,
        w1_q,
        w2_q,
        topk_weights,
        local_ids,
        dtype=torch.bfloat16,
        activation=ActivationType.Situv2,
        quant_type=Q_TYPE,
        a1_scale=a_scale,
        w1_scale=w1_scale,
        situ_beta=BETA,
        situ_linear_beta=LINEAR_BETA,
    )
    ref2 = torch_moe_stage2(
        ref1,
        w1_q,
        w2_q,
        topk_weights,
        local_ids,
        dtype=torch.bfloat16,
        quant_type=Q_TYPE,
        w2_scale=w2_scale,
        a2_scale=None,
        doweight=True,
    )

    kw = {
        "sorted_token_ids": sorted_ids,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "topk": topk,
        "tile_m": 32,
        "tile_n": 256,
        "tile_k": 256,
        "a_dtype": "fp8",
        "b_dtype": "fp4",
        "out_dtype": "bf16",
    }
    s1 = flydsl_moe_stage1(
        a=a_q,
        w1=_at_tail(
            shuffle_weight(w1_q, (16, 16)), tail, E, E, inter_dim * 2, model_dim // 2
        ),
        act="situv2",
        situ_beta=BETA,
        situ_linear_beta=LINEAR_BETA,
        w1_scale=_at_tail(
            e8m0_shuffle(w1_scale), tail, E, E * inter_dim * 2, model_dim // 32
        ),
        a1_scale=mxfp4_moe_sort_fwd(
            a_scale,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token,
            cols=model_dim,
        ),
        gate_mode="separated",
        **kw,
    )
    torch.cuda.synchronize()
    _check_close(ref1.float(), s1.float(), "situv2_past_4gib_stage1")
    a2_q, a2_scale = per_1x32_f8_scale_f8_quant(
        s1, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    out = flydsl_moe_stage2(
        inter_states=a2_q.view(token, topk, inter_dim),
        w2=_at_tail(
            shuffle_weight_a16w4(w2_q, 16, False), tail, E, E, model_dim, inter_dim // 2
        ),
        mode="atomic",
        w2_scale=_at_tail(
            shuffle_scale_a16w4(w2_scale, tail, False),
            tail,
            E,
            E * model_dim,
            inter_dim // 32,
        ),
        a2_scale=mxfp4_moe_sort_fwd(
            a2_scale,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token,
            cols=inter_dim,
        ),
        sorted_weights=sorted_weights,
        inter_dim_pad=0,
        model_dim_pad=0,
        **kw,
    )
    torch.cuda.synchronize()
    _check_close(ref2.float(), out.float(), "situv2_past_4gib_e2e")


@_SKIP
def test_a4w4_mxmoe_experts_past_4gib(monkeypatch):
    """fused_moe's A4W4 path (AITER_SITUV2_A4W4=1) through the MXMOE stage 1."""
    import aiter.ops.flydsl.mxfp4_gemm1_kernels as gemm1_kernels

    token, model_dim, inter_dim, E, topk, tail = 512, 3584, 3072, 896, 16, 32
    assert E * 2 * inter_dim * model_dim // 2 > 2**32  # w1
    monkeypatch.setenv("AITER_SITUV2_A4W4", "1")
    stage1_calls = []
    launch = gemm1_kernels.flydsl_mxfp4_gemm1

    def counted_launch(*args, **kwargs):
        stage1_calls.append(1)
        return launch(*args, **kwargs)

    monkeypatch.setattr(gemm1_kernels, "flydsl_mxfp4_gemm1", counted_launch)
    torch.manual_seed(41)
    dev = "cuda"
    x = torch.randn((token, model_dim), dtype=torch.bfloat16, device=dev) / 4
    w1 = torch.randn((tail, 2 * inter_dim, model_dim), dtype=torch.bfloat16, device=dev)
    w2 = torch.randn((tail, model_dim, inter_dim), dtype=torch.bfloat16, device=dev)
    local_ids = torch.stack(
        [torch.randperm(tail, device=dev)[:topk] for _ in range(token)]
    ).to(torch.int32)
    topk_weights = torch.rand((token, topk), device=dev)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    w1_q, w1_scale = per_1x32_f4_quant(w1 / 8, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = per_1x32_f4_quant(w2 / 8, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(tail, 2 * inter_dim, model_dim // 2)
    w2_q = w2_q.view(tail, model_dim, inter_dim // 2)
    # Reference with the kernel's quantization points: fp4 input, fp4 intermediate.
    a_q, a_scale = per_1x32_f4_quant(x, quant_dtype=dtypes.fp4x2)
    ref1 = torch_moe_stage1(
        a_q.view(token, model_dim // 2),
        w1_q,
        w2_q,
        topk_weights,
        local_ids,
        dtype=torch.bfloat16,
        activation=ActivationType.Situv2,
        quant_type=Q_TYPE,
        a1_scale=a_scale,
        w1_scale=w1_scale,
        situ_beta=BETA,
        situ_linear_beta=LINEAR_BETA,
    )
    i_q, i_scale = per_1x32_f4_quant(
        ref1.view(token * topk, inter_dim), quant_dtype=dtypes.fp4x2
    )
    ref = torch_moe_stage2(
        i_q.view(token, topk, inter_dim // 2),
        w1_q,
        w2_q,
        topk_weights,
        local_ids,
        dtype=torch.bfloat16,
        quant_type=Q_TYPE,
        w2_scale=w2_scale,
        a2_scale=i_scale,
        doweight=True,
    )
    w1_sh = moe_shuffle_weight(w1_q, tail, is_guinterleave=False, gate_up=True)
    w2_sh = moe_shuffle_weight(w2_q, tail, is_guinterleave=False, gate_up=False)
    s1_sh = moe_shuffle_scale(
        w1_scale.view(-1, model_dim // 32), tail, is_guinterleave=False, gate_up=True
    )
    s2_sh = moe_shuffle_scale(
        w2_scale.view(-1, inter_dim // 32), tail, is_guinterleave=False, gate_up=False
    )
    w1_full = _at_tail(w1_sh, tail, E, E, 2 * inter_dim, model_dim // 2)
    w2_full = _at_tail(w2_sh, tail, E, E, model_dim, inter_dim // 2)
    w1_full.is_shuffled = w2_full.is_shuffled = True
    out = fused_moe(
        x,
        w1_full,
        w2_full,
        topk_weights,
        local_ids + (E - tail),
        activation=ActivationType.Situv2,
        quant_type=Q_TYPE,
        w1_scale=_at_tail(s1_sh, tail, E, E * 2 * inter_dim, model_dim // 32),
        w2_scale=_at_tail(s2_sh, tail, E, E * model_dim, inter_dim // 32),
        beta=BETA,
        linear_beta=LINEAR_BETA,
    )
    torch.cuda.synchronize()
    assert stage1_calls, "fused_moe did not run the MXMOE stage 1"
    ref, out = ref.float(), out.float()
    rel = (out - ref).norm(dim=-1) / ref.norm(dim=-1)
    assert rel.max() < 0.1, f"a4w4 past 4 GiB: max row relative error {rel.max():.3f}"
    # Per element, in row-RMS units: MXFP4 rounding flips of the intermediate
    # reach 0.18 with this PR; experts read at the wrong offset put 82% past 0.2.
    rms = ref.square().mean(dim=-1, keepdim=True).sqrt()
    err = checkAllclose(
        ref / rms, out / rms, rtol=0.1, atol=0.25, msg="a4w4 past 4 GiB"
    )
    assert err == 0, f"a4w4 past 4 GiB: {err:.2%} of elements outside rtol/atol"
