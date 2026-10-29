# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CK-Tile MXFP4 MoE stage 1 must only split K into whole 256-wide K tiles.

bf16 activations with MXFP4 weights run stage 1 on the CK-Tile split-k kernel
(split_k >= 2 for SwiGLU). The kernel runs whole K tiles per split without
masking, so a split of model_dim that is not a multiple of 256 (model_dim=256,
768, 1280, ...) used to return silently wrong results or fault.
"""

import inspect

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import _cktile_gemm1_k_batch, fused_moe, fused_topk, moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

_SKIP = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="CK-Tile MXFP4 MoE instances are gfx950-only"
)

INTER_DIM = 256
MAX_REL_L2 = 0.02


@pytest.mark.parametrize(
    ("K", "split_k", "want"),
    [
        (256, 2, 1),
        (512, 2, 2),
        (512, 4, 2),
        (768, 2, 1),
        (768, 3, 3),
        (1280, 2, 1),
        (6144, 2, 2),
        (6144, 16, 12),
        (384, 2, 1),
        (3072, 64, 12),
        (512, 1, 1),
        (128, 2, 1),
    ],
)
def test_cktile_gemm1_k_batch(K, split_k, want):
    assert _cktile_gemm1_k_batch(K, split_k) == want


def _quant(w):
    E, N, K = w.shape
    q, s = per_1x32_f4_quant(w.reshape(E * N, K), quant_dtype=dtypes.fp4x2)
    q = q.view(E, N, K // 2)
    s = s.view(torch.uint8).view(E * N, K // 32)
    deq = fp4_utils.mxfp4_to_f32(q) * fp4_utils.e8m0_to_f32(s).repeat_interleave(
        32, -1
    ).view(E, N, K)
    scale = fp4_utils.e8m0_shuffle(s).view(dtypes.fp8_e8m0)
    return shuffle_weight(q, (16, 16)), scale, deq.double()


def _make(model_dim, E, topk, token, hidden_pad=0):
    torch.manual_seed(0)
    x = torch.randn(token, model_dim, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(E, 2 * INTER_DIM, model_dim, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(E, model_dim, INTER_DIM, device="cuda", dtype=torch.bfloat16)
    real_dim = model_dim - hidden_pad
    x[:, real_dim:] = 0
    w1[:, :, real_dim:] = 0
    w2[:, real_dim:] = 0
    score = torch.randn(token, E, device="cuda", dtype=torch.bfloat16)
    topk_weight, topk_ids = fused_topk(x, score, topk, True)
    w1q, w1_scale, w1_ref = _quant(w1 / model_dim**0.5)
    w2q, w2_scale, w2_ref = _quant(w2 / INTER_DIM**0.5)
    return x, topk_weight, topk_ids, (w1q, w1_scale, w1_ref), (w2q, w2_scale, w2_ref)


def _ref(x, topk_weight, topk_ids, w1_ref, w2_ref, activation):
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float64)
    for m in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = topk_ids[m, k]
            h = w1_ref[e] @ x[m].double()
            if activation == ActivationType.Swiglu:
                gate, up = h.split(INTER_DIM)
                gate, up = gate.clamp(max=7.0), up.clamp(-7.0, 7.0)
                a2 = gate * torch.sigmoid(1.702 * gate) * (up + 1)
            else:
                # fp4 A/W Silu post-activation reads gate/up interleaved in
                # 16-row blocks (post_activation_layout="auto").
                gate, up = h.view(INTER_DIM // 16, 2, 16).unbind(1)
                a2 = torch.nn.functional.silu(gate.flatten()) * up.flatten()
            out[m] += topk_weight[m, k].double() * (
                w2_ref[e] @ a2.to(torch.bfloat16).double()
            )
    return out


def _run_fused_moe(monkeypatch, model_dim, E, topk, token, activation, hidden_pad=0):
    x, topk_weight, topk_ids, w1, w2 = _make(model_dim, E, topk, token, hidden_pad)
    launched = []
    gemm1 = aiter.moe_cktile2stages_gemm1
    sig = inspect.signature(gemm1)

    def spy(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        launched.append(bound.arguments["split_k"])
        return gemm1(*args, **kwargs)

    monkeypatch.setattr(aiter, "moe_cktile2stages_gemm1", spy)
    fused_moe_module.get_ksplit.cache_clear()
    fused_moe_module.get_2stage_cfgs.cache_clear()
    try:
        out = fused_moe(
            x,
            w1[0],
            w2[0],
            topk_weight,
            topk_ids,
            activation=activation,
            quant_type=QuantType.per_1x32,
            w1_scale=w1[1],
            w2_scale=w2[1],
            hidden_pad=hidden_pad,
        )
    finally:
        fused_moe_module.get_ksplit.cache_clear()
        fused_moe_module.get_2stage_cfgs.cache_clear()
    ref = _ref(x, topk_weight, topk_ids, w1[2], w2[2], activation)
    rel_l2 = ((out.double() - ref).norm() / ref.norm()).item()
    return launched, rel_l2


@_SKIP
@pytest.mark.parametrize("model_dim", [256, 512, 768, 1024, 1280, 6144])
@pytest.mark.parametrize(("E", "topk", "token"), [(4, 2, 4), (128, 4, 16)])
def test_bf16_swiglu_cktile_stage1_split_alignment(
    monkeypatch, model_dim, E, topk, token
):
    launched, rel_l2 = _run_fused_moe(
        monkeypatch, model_dim, E, topk, token, ActivationType.Swiglu
    )
    # SwiGLU forces split_k >= 2 on top of the CU-count based ksplit.
    requested = max(
        fused_moe_module.get_ksplit(token, topk, E, INTER_DIM, model_dim), 2
    )
    assert launched == [_cktile_gemm1_k_batch(model_dim, requested)]
    assert rel_l2 < MAX_REL_L2


@_SKIP
@pytest.mark.parametrize("hidden_pad", [192, 256])
def test_bf16_swiglu_cktile_stage1_padded_k_does_not_split(monkeypatch, hidden_pad):
    # The CK-Tile pipeline skips the k_padded_zeros tail in every split.
    launched, rel_l2 = _run_fused_moe(
        monkeypatch, 1024, 4, 2, 4, ActivationType.Swiglu, hidden_pad
    )
    assert launched == [1]
    assert rel_l2 < MAX_REL_L2


@_SKIP
def test_bf16_swiglu_cktile_stage1_rejects_partial_k_tile(monkeypatch):
    with pytest.raises(RuntimeError, match="must be a multiple of k_batch"):
        _run_fused_moe(monkeypatch, 384, 4, 2, 4, ActivationType.Swiglu)


_GEMM1_NAME = "moe_cktile2stages_gemm1_256x16x128x256_1x4_16x16x32_1perCU_1x32_swiglu"


@_SKIP
@pytest.mark.parametrize(
    ("model_dim", "k_batch", "out_width", "extra", "match"),
    [
        (768, 2, 2 * INTER_DIM, {}, "must be a multiple of k_batch"),
        (512, 2, INTER_DIM, {}, "split-k workspace"),
        (512, 1, INTER_DIM, {"kernel_name": _GEMM1_NAME + "_SplitK"}, "does not match"),
        (512, 1, 2 * INTER_DIM, {"kernel_name": _GEMM1_NAME}, "does not match"),
        (1024, 2, 2 * INTER_DIM, {"k_padded_zeros": 256}, "k_padded_zeros"),
    ],
)
def test_cktile_gemm1_rejects_bad_split(model_dim, k_batch, out_width, extra, match):
    E, topk, token, block_m = 4, 2, 4, 16
    x, topk_weight, topk_ids, w1, _ = _make(model_dim, E, topk, token)
    sorted_ids, _, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weight, E, model_dim, torch.bfloat16, block_m
    )
    out = torch.zeros(token * topk, out_width, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=match):
        aiter.moe_cktile2stages_gemm1(
            x,
            w1[0],
            out,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            w_scale=w1[1],
            activation=ActivationType.Swiglu,
            block_m=block_m,
            split_k=k_batch,
            **extra,
        )


@_SKIP
@pytest.mark.parametrize(
    ("model_dim", "split_k", "k_batch"), [(512, 4, 2), (768, 2, 1), (1024, 4, 4)]
)
def test_silu_cktile_stage1_forced_split(monkeypatch, model_dim, split_k, k_batch):
    # Silu reaches the same kernel through the ksplit > 1 branch.
    monkeypatch.setenv("AITER_KSPLIT", str(split_k))
    monkeypatch.setenv("AITER_FLYDSL_FORCE", "0")
    launched, rel_l2 = _run_fused_moe(
        monkeypatch, model_dim, 128, 4, 1, ActivationType.Silu
    )
    assert launched == [k_batch]
    assert rel_l2 < MAX_REL_L2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
