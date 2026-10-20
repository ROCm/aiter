# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.torch_ref.fused_compress_attn import fused_compress_attn
from aiter.ops.triton.fusions.fused_compress_attn import fused_compress_attn_fp4
from aiter.ops.triton.utils._triton import arch_info

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx1250", reason="gfx1250 Gluon kernel"
)

HEAD_DIM = 128
ROPE_DIM = 64
RATIO = 4
STATE_SIZE = 8
K_PER_BLOCK = 64


def make_inputs(num_rows):
    """Decode plan. Sequence i has 4 * i + 4 tokens. Its window has 7 tokens from
    the state ring and 1 token from the input. Every 5th plan row is empty."""
    torch.manual_seed(0)
    seq = torch.arange(num_rows, device="cuda")
    last_pos = RATIO * seq + RATIO - 1
    num_from_state = torch.full_like(seq, 2 * RATIO - 1)
    # Each plan row is [last_token_row, seq, last_pos, num_from_state].
    plan = torch.stack([seq, seq, last_pos, num_from_state], dim=1).int()
    plan[2::5] = -1
    max_pos = RATIO * num_rows
    # Sequence i writes cache entry i. Give each sequence blocks for num_rows entries.
    blocks_per_seq = num_rows // K_PER_BLOCK + 1
    num_blocks = num_rows * blocks_per_seq
    return {
        "kv_in": torch.randn(num_rows, 2 * HEAD_DIM, device="cuda").bfloat16(),
        "score_in": torch.randn(num_rows, 2 * HEAD_DIM, device="cuda").bfloat16(),
        "kv_state": torch.randn(num_rows, STATE_SIZE, 2 * HEAD_DIM, device="cuda"),
        "score_state": torch.randn(num_rows, STATE_SIZE, 2 * HEAD_DIM, device="cuda"),
        "state_slots": torch.randperm(num_rows, device="cuda").int(),
        "plan": plan,
        "ape": torch.randn(RATIO, 2 * HEAD_DIM, device="cuda"),
        "rms_weight": torch.rand(HEAD_DIM, device="cuda").bfloat16() + 0.5,
        "rms_eps": 1e-6,
        "cos": torch.rand(max_pos, ROPE_DIM // 2, device="cuda").bfloat16(),
        "sin": torch.rand(max_pos, ROPE_DIM // 2, device="cuda").bfloat16(),
        "kv_cache": torch.zeros(
            num_blocks, K_PER_BLOCK, HEAD_DIM // 2, dtype=torch.uint8, device="cuda"
        ),
        "cache_scale": torch.zeros(
            num_blocks, K_PER_BLOCK, HEAD_DIM // 32, dtype=torch.uint8, device="cuda"
        ),
        "block_tables": torch.randperm(num_blocks, device="cuda")
        .view(num_rows, -1)
        .int(),
    }


@pytest.mark.parametrize("num_rows", [512])
def test_fused_compress_attn_fp4(num_rows):
    inputs = make_inputs(num_rows)
    ref_kv_cache = inputs["kv_cache"].clone()
    ref_cache_scale = inputs["cache_scale"].clone()
    fused_compress_attn(
        kv_in=inputs["kv_in"],
        score_in=inputs["score_in"],
        kv_state=inputs["kv_state"],
        score_state=inputs["score_state"],
        plan_gpu=inputs["plan"],
        state_slot_mapping=inputs["state_slots"],
        ape=inputs["ape"],
        rms_weight=inputs["rms_weight"],
        rms_eps=inputs["rms_eps"],
        cos_cache=inputs["cos"],
        sin_cache=inputs["sin"],
        kv_cache=ref_kv_cache,
        cache_scale=ref_cache_scale,
        block_tables=inputs["block_tables"],
        k_per_block=K_PER_BLOCK,
        overlap=True,
        ratio=RATIO,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_DIM,
        quant_mode="fp4",
        preshuffle=False,
    )
    fused_compress_attn_fp4(**inputs)
    torch.testing.assert_close(inputs["kv_cache"], ref_kv_cache)
    torch.testing.assert_close(inputs["cache_scale"], ref_cache_scale)
