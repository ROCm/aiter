# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import pytest
import torch

from op_tests.triton_tests.utils import minimax_m3 as m3

pytestmark = pytest.mark.skipif(
    os.getenv("AITER_TEST_MINIMAX_M3_ATOM") != "1",
    reason="Install ATOM and set AITER_TEST_MINIMAX_M3_ATOM=1",
)


@pytest.mark.parametrize("tokens", [16, 128, 3072])
def test_norm_rope_cache(tokens):
    data = m3.norm_rope_inputs(tokens)
    references = []
    for name, heads in (("q", 16), ("k", 1)):
        x = data[name].float().reshape(tokens, heads, 128)
        norm = data[name + "_norm"]
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.variance_epsilon)
        x = x * (1 + norm.weight.float())
        cos = data["rotary_emb"].cos_cache[:tokens].repeat(1, 2)[:, None]
        sin = data["rotary_emb"].sin_cache[:tokens].repeat(1, 2)[:, None]
        rotated = torch.cat((-x[..., 64:], x[..., :64]), dim=-1)
        references.append((x * cos + rotated * sin).reshape(tokens, -1))
    q, k = m3.norm_rope(**data)
    for actual, expected in zip((q, k), references):
        torch.testing.assert_close(actual, expected.bfloat16(), atol=0.04, rtol=0.02)
    cached_k = data["k_cache"].float().permute(0, 3, 1, 2, 4).reshape(-1, 128)[:tokens]
    cached_v = data["v_cache"].float().permute(0, 2, 4, 1, 3).reshape(-1, 128)[:tokens]
    for actual, scale_name, expected in (
        (cached_k, "k_scale", references[1]),
        (cached_v, "v_scale", data["v"].float()),
    ):
        scale = data[scale_name].flatten()[:tokens, None]
        expected_scale = expected.abs().amax(-1, keepdim=True).clamp(min=1e-6) / 448
        torch.testing.assert_close(scale, expected_scale, atol=1e-5, rtol=0.02)
        torch.testing.assert_close(actual * scale, expected, atol=0.15, rtol=0.08)
