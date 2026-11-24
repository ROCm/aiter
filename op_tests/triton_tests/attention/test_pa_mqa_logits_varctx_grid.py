# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes
from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits,
    deepgemm_fp8_paged_mqa_logits_schedule,
    enable_jit_gluon_pa_mqa_logits_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_weight


@pytest.mark.parametrize("batch", [79, 80, 81, 128])
@torch.inference_mode()
def test_varctx_grid_covers_every_query(batch):
    if (
        arch_info.get_arch() not in ("gfx942", "gfx950")
        or not enable_jit_gluon_pa_mqa_logits_kernel
    ):
        pytest.skip("Requires the CDNA Gluon JIT paged MQA kernel")

    # Emulate the 80-CU budget on larger CDNA GPUs without changing the device.
    cu_count = 80
    next_n, heads, head_dim, page_size, width = 2, 64, 128, 64, 512
    lengths = width - page_size * (torch.arange(batch, device="cuda") % 4)
    lengths = lengths.to(torch.int32).contiguous()
    pages = batch * (width // page_size)
    page_table = torch.arange(pages, device="cuda", dtype=torch.int32).view(batch, -1)
    q = torch.ones((batch, next_n, heads, head_dim), device="cuda", dtype=torch.float32)
    q = q.to(dtypes.fp8)
    k = torch.ones((pages, page_size, head_dim), device="cuda", dtype=torch.float32)
    k = shuffle_weight(k.to(dtypes.fp8), layout=(16, 16))
    scales = torch.ones((pages, page_size), device="cuda", dtype=torch.float32)
    cache = torch.cat(
        (k.view(pages, -1).view(torch.uint8), scales.view(torch.uint8)), dim=1
    )
    cache = cache.view(pages, page_size, 1, head_dim + 4)
    weights = torch.ones((batch * next_n, heads), device="cuda", dtype=torch.float32)
    logits = torch.full(
        (batch * next_n, width), float("-inf"), device="cuda", dtype=torch.float32
    )
    schedule = deepgemm_fp8_paged_mqa_logits_schedule(
        batch, next_n, lengths, width, TotalCuCount=cu_count
    )
    deepgemm_fp8_paged_mqa_logits(
        q,
        cache,
        weights,
        logits,
        lengths,
        page_table,
        width,
        Preshuffle=True,
        KVBlockSize=page_size,
        TotalCuCount=cu_count,
        VarCtxSchedule=schedule,
    )

    # Unit Q, K, scale and weight give heads * head_dim at each valid position.
    q_len = lengths[:, None] - next_n + 1 + torch.arange(next_n, device="cuda")
    valid = torch.arange(width, device="cuda") < q_len.reshape(-1, 1)
    expected = torch.full_like(logits, float("-inf"))
    expected.masked_fill_(valid, heads * head_dim)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
