# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness coverage for the context-parallel GDN K5 path."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.linear_attention_prefill_kernels import (
    chunk_gated_delta_rule_fwd_h_flydsl_opt,
    gdn_prepare_fwd_flydsl,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.gdn_segment_scan import (
    gdn_segment_scan_fwd,
)

pytestmark = pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 kernel")


def test_gdn_segment_scan_matches_flydsl():
    T, H, HG, K, V = 1024, 16, 8, 128, 128
    torch.manual_seed(0)
    k = torch.randn(1, T, HG, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, T, H, V, device="cuda", dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16)
    v = torch.nn.functional.normalize(v.float(), dim=-1).to(torch.bfloat16)
    g_raw = torch.full((1, T, H), -0.05, device="cuda")
    beta = torch.full((1, T, H), 0.7, device="cuda")
    h0 = torch.randn(1, H, V, K, device="cuda")
    w, u, g = gdn_prepare_fwd_flydsl(k=k, v=v, g=g_raw, beta=beta, use_exp2=True)

    expected = chunk_gated_delta_rule_fwd_h_flydsl_opt(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=h0,
        output_final_state=True,
        use_exp2=True,
        g_head_major=True,
    )
    actual = gdn_segment_scan_fwd(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=h0,
        output_final_state=True,
        seq_lens=(T,),
        chunks_per_segment=4,
    )

    torch.testing.assert_close(actual[0], expected[0], atol=0.032, rtol=0.05)
    torch.testing.assert_close(actual[1], expected[1], atol=0.016, rtol=0.05)
    torch.testing.assert_close(actual[2], expected[2], atol=6e-4, rtol=0.05)
