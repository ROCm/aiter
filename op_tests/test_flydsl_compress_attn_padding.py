# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CSA padding must ignore non-finite state, without hiding valid-state NaNs."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.fused_compress_attn import flydsl_fused_compress_attn


@pytest.mark.parametrize("quant_mode", ["none", "group_fp8"])
@pytest.mark.parametrize("num_waves", [1, 2, 4])
@pytest.mark.parametrize("plan_capacity", [1, 128, 2048])
def test_csa_padding_state(quant_mode, num_waves, plan_capacity):
    if get_gfx() != "gfx950":
        pytest.skip("wave64 CSA padding regression validated on gfx950")

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(1234)
    kv = torch.randn(
        (4, 1024), device=device, dtype=torch.bfloat16, generator=generator
    )
    scores = torch.zeros_like(kv)
    state = torch.zeros((1, 15, 1024), device=device, dtype=torch.float32)
    state_scores = torch.zeros_like(state)
    plan = torch.full((plan_capacity, 4), -1, device=device, dtype=torch.int32)
    slot_mapping = torch.zeros(1, device=device, dtype=torch.int32)
    block_tables = torch.zeros((1, 1), device=device, dtype=torch.int32)
    ape = torch.zeros((4, 1024), device=device, dtype=torch.float32)
    norm = torch.ones(512, device=device, dtype=torch.float32)
    cos = torch.ones((16, 32), device=device, dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)

    def run(position, fill):
        state.fill_(fill)
        # Positions -4..-1 are padding at the first compression boundary.
        # Position 7 instead consumes real history at positions 0..3.
        state_scores.fill_(float("nan") if position == 3 else 0.0)
        plan[0] = torch.tensor([3, 0, position, 4], device=device, dtype=torch.int32)
        cache = torch.zeros(
            (1, 64, 512),
            device=device,
            dtype=torch.bfloat16 if quant_mode == "none" else torch.float8_e4m3fn,
        )
        rope = torch.zeros((1, 64, 64), device=device, dtype=torch.bfloat16)
        flydsl_fused_compress_attn(
            kv_in=kv,
            score_in=scores,
            kv_state=state,
            score_state=state_scores,
            plan_gpu=plan,
            state_slot_mapping=slot_mapping,
            ape=ape,
            rms_weight=norm,
            rms_eps=1e-6,
            cos_cache=cos,
            sin_cache=sin,
            kv_cache=cache,
            block_tables=block_tables,
            k_per_block=64,
            overlap=True,
            ratio=4,
            head_dim=512,
            rope_head_dim=64,
            quant=quant_mode != "none",
            quant_mode=quant_mode,
            k_rope_cache=rope if quant_mode == "group_fp8" else None,
            k_split_num_waves=num_waves,
        )
        index = position // 4
        return cache[0, index].cpu(), rope[0, index].cpu()

    expected_bytes = None
    for fill in (0.0, float("nan"), float("inf"), -float("inf")):
        output, rope_output = run(3, fill)
        bits = output.view(torch.uint8)
        if quant_mode == "none":
            assert torch.isfinite(output).all()
            # Scores and APE are zero: the four valid values have equal weight.
            pooled = kv[:, 512:].float().cpu().mean(dim=0)
            expected = pooled * torch.rsqrt(pooled.square().mean() + 1e-6)
            torch.testing.assert_close(output.float(), expected, rtol=0.01, atol=0.02)
        else:
            assert torch.isfinite(output[:448].float()).all()
            assert (bits[448:462] != 255).all()
            assert torch.isfinite(rope_output).all()
        actual_bytes = (bits, rope_output.view(torch.uint8))
        if expected_bytes is None:
            expected_bytes = actual_bytes
        else:
            assert all(torch.equal(a, b) for a, b in zip(actual_bytes, expected_bytes))

    # The same physical state becomes semantically valid at the next boundary.
    # It must still affect the result, including propagating a genuine NaN.
    finite, finite_rope = run(7, 0.5)
    bad, bad_rope = run(7, float("nan"))
    if quant_mode == "none":
        assert torch.isfinite(finite).all()
        assert torch.isnan(bad).all()
    else:
        assert torch.isfinite(finite[:448].float()).all()
        assert torch.isfinite(finite_rope).all()
        assert torch.isnan(bad[:448].float()).all()
        assert torch.isnan(bad_rope).all()
