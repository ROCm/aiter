# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits,
    enable_jit_gluon_pa_mqa_logits_kernel,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

pytestmark = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950") or not enable_jit_gluon_pa_mqa_logits_kernel,
    reason="Requires the CDNA Gluon JIT paged MQA kernel",
)


@pytest.mark.parametrize("block_size", [1, 16, 64, 128])
@pytest.mark.parametrize("chunk_k", [64, 256])
@pytest.mark.parametrize("padded_table", [False, True], ids=["compact", "padded"])
@pytest.mark.parametrize(
    "context_lengths,next_n,heads,hidden_dim",
    [
        pytest.param((997, 63, 1), 1, 32, 128, id="decode"),
        pytest.param((3000, 257, 3, 0), 3, 64, 128, id="mtp"),
        pytest.param((511, 7), 1, 64, 64, id="dim64"),
    ],
)
@torch.inference_mode()
def test_paged_mqa_logits_non_preshuffle(
    block_size: int,
    chunk_k: int,
    padded_table: bool,
    context_lengths: tuple[int, ...],
    next_n: int,
    heads: int,
    hidden_dim: int,
) -> None:
    """Check block paging, per-token scales, short tails and the prefetch loop.

    Compact page tables expose token-indexed OOB reads; padded tables keep
    those reads in allocated memory and expose incorrect scores instead.
    Block size 1 also checks the original per-token paging layout.
    """
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(5591)
    fp8_dtype = get_fp8_e4m3_dtype()
    batch = len(context_lengths)
    max_context = max(context_lengths)
    max_pages = (max_context + block_size - 1) // block_size
    num_blocks = 2 * max_pages + 3

    q = torch.randn(
        batch,
        next_n,
        heads,
        hidden_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).to(fp8_dtype)
    kv = torch.randn(
        num_blocks,
        block_size,
        hidden_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).to(fp8_dtype)
    scales = 0.25 + torch.rand(
        num_blocks, block_size, generator=generator, device=device
    )
    weights = torch.randn(batch * next_n, heads, generator=generator, device=device)

    # Each physical block stores all FP8 values followed by all FP32 scales.
    packed = torch.empty(
        num_blocks, block_size * (hidden_dim + 4), dtype=torch.uint8, device=device
    )
    value_bytes = block_size * hidden_dim
    packed[:, :value_bytes] = kv.view(num_blocks, -1).view(torch.uint8)
    packed[:, value_bytes:] = scales.view(torch.uint8)
    cache = packed.view(num_blocks, block_size, 1, hidden_dim + 4)

    table_width = max(4096, max_pages) if padded_table else max_pages
    block_tables = torch.zeros(batch, table_width, dtype=torch.int32, device=device)
    for b, context_length in enumerate(context_lengths):
        pages = (context_length + block_size - 1) // block_size
        block_tables[b, :pages] = torch.randperm(
            num_blocks, generator=generator, device=device
        )[:pages]
    context_lens = torch.tensor(context_lengths, dtype=torch.int32, device=device)

    # Guard each output row, including the negative offsets of a short chunk.
    guard = 16
    sentinel = 12345.0
    storage = torch.full(
        (batch * next_n, max_context + 2 * guard), sentinel, device=device
    )
    out = storage[:, guard:-guard]
    out.fill_(float("-inf"))
    deepgemm_fp8_paged_mqa_logits(
        q,
        cache,
        weights,
        out,
        context_lens,
        block_tables,
        max_context,
        Preshuffle=False,
        KVBlockSize=block_size,
        ChunkK=chunk_k,
        # Force multiple chunks per CTA to exercise the steady-state prefetch,
        # even on a GPU with enough CUs to assign one CTA to every chunk.
        TotalCuCount=1,
        WavePerEU=2,
    )

    reference = torch.full_like(out, float("-inf"))
    dequantized_kv = kv.float() * scales[..., None]
    for b, context_length in enumerate(context_lengths):
        if context_length == 0:
            continue
        pages = (context_length + block_size - 1) // block_size
        page_ids = block_tables[b, :pages].long()
        keys = dequantized_kv[page_ids].reshape(-1, hidden_dim)[:context_length]
        scores = (q[b].float() @ keys.T).relu()
        row_weights = weights[b * next_n : (b + 1) * next_n]
        logits = (scores * row_weights[..., None]).sum(dim=1)
        positions = torch.arange(context_length, device=device)
        query_positions = context_length - next_n + torch.arange(next_n, device=device)
        logits.masked_fill_(
            positions[None, :] > query_positions[:, None], float("-inf")
        )
        reference[b * next_n : (b + 1) * next_n, :context_length] = logits

    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)
    assert torch.all(storage[:, :guard] == sentinel)
    assert torch.all(storage[:, -guard:] == sentinel)
