# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CK 2-stage MoE with hidden sizes that are not multiples of the block tile.

The bf16/fp16 NKPadding instances take weights zero-padded only to the preshuffle
granularity (16 rows, 32 columns), while the activations and the output keep the
model's hidden size.
"""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    ck_moe_stage1,
    fused_moe,
    moe_sorting,
    torch_moe,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in ("gfx942", "gfx950"),
    reason="CK 2-stage MoE instances are built for gfx942 and gfx950",
)

EXPERTS = 8
TOPK = 2


def _make_moe(tokens, model_dim, inter_dim, dtype=dtypes.bf16):
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(tokens, model_dim, generator=gen, device="cuda") / 10
    w1 = torch.randn(EXPERTS, 2 * inter_dim, model_dim, generator=gen, device="cuda")
    w2 = torch.randn(EXPERTS, model_dim, inter_dim, generator=gen, device="cuda")
    score = torch.randn(tokens, EXPERTS, generator=gen, device="cuda")
    topk_weights, topk_ids = torch.topk(score.softmax(-1), TOPK, dim=-1)
    return (
        x.to(dtype),
        (w1 / 10).to(dtype),
        (w2 / 10).to(dtype),
        topk_weights.float(),
        topk_ids.to(dtypes.i32),
    )


def _shuffle_with_padding(w1, w2):
    return (
        shuffle_weight(w1, (16, 16), pad_k_to=32),
        shuffle_weight(w2, (16, 16), pad_n_to=16),
    )


def _run_ck_stages(x, w1, w2, topk_weights, topk_ids, inter_dim, block_m):
    model_dim = x.shape[-1]
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out = moe_sorting(
        topk_ids, topk_weights, EXPERTS, model_dim, x.dtype, block_m
    )[:5]
    a2 = torch.empty(x.shape[0], TOPK, inter_dim, dtype=x.dtype, device="cuda")
    ck_moe_stage1(
        x,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        a2,
        TOPK,
        block_m,
        None,
        None,
        activation=ActivationType.Silu,
        dtype=x.dtype,
    )
    aiter.ck_moe_stage2_fwd(
        a2,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        TOPK,
        "",
        None,
        None,
        block_m,
        sorted_weights,
        QuantType.No,
        ActivationType.Silu,
    )
    return out


def _assert_matches_reference(out, ref):
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


def test_shuffle_weight_pad_n_to_pads_rows():
    w = torch.zeros(2, 200, 256, dtype=dtypes.bf16, device="cuda")

    shuffled = shuffle_weight(w, (16, 16), pad_n_to=16)

    assert shuffled.shape == (2, 208, 256)
    assert shuffled.aiter_original_n == 200
    assert shuffled.aiter_padded_n == 208


# 2880 and 1024 need no weight padding; 2880 is still not a multiple of the
# 128-wide stage-2 tiles. 8 and 200 pad both w1 K and w2 rows, 2896 only w1 K.
@pytest.mark.parametrize("model_dim", [8, 200, 1024, 2880, 2896])
@pytest.mark.parametrize("inter_dim", [192, 256])
@pytest.mark.parametrize("block_m", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [dtypes.bf16, dtypes.fp16])
def test_ck_moe_stages_unaligned_model_dim(model_dim, inter_dim, block_m, dtype):
    x, w1, w2, topk_weights, topk_ids = _make_moe(128, model_dim, inter_dim, dtype)
    ref = torch_moe(x, w1, w2, topk_weights, topk_ids)
    w1, w2 = _shuffle_with_padding(w1, w2)

    out = _run_ck_stages(x, w1, w2, topk_weights, topk_ids, inter_dim, block_m)

    _assert_matches_reference(out, ref)


@pytest.mark.parametrize("model_dim", [200, 2880, 2896])
@pytest.mark.parametrize("tokens", [1, 64, 2048])
def test_fused_moe_unaligned_model_dim(model_dim, tokens):
    inter_dim = 256
    x, w1, w2, topk_weights, topk_ids = _make_moe(tokens, model_dim, inter_dim)
    ref = torch_moe(x, w1, w2, topk_weights, topk_ids)
    w1, w2 = _shuffle_with_padding(w1, w2)

    out = fused_moe(x, w1, w2, topk_weights, topk_ids)

    assert out.shape == (tokens, model_dim)
    _assert_matches_reference(out, ref)


def test_ck_moe_stage2_rejects_weights_padded_past_preshuffle_tile():
    x, w1, w2, topk_weights, topk_ids = _make_moe(16, 200, 256)
    w1 = shuffle_weight(w1, (16, 16), pad_k_to=32)
    w2 = shuffle_weight(w2, (16, 16), pad_n_to=128)

    with pytest.raises(RuntimeError, match="expected 208"):
        _run_ck_stages(x, w1, w2, topk_weights, topk_ids, 256, 32)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
