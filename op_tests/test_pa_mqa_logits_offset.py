#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for paged MQA logits block paging and output offsets.

The non-preshuffle cases check block tables and per-token scales against FP32.

Background
----------
The producer stores logits with AMD ``buffer_store``, whose voffset is a 32-bit
byte offset. For a wide dense output ``[rows, max_model_len]`` the store address
of row ``r`` is ``r * stride_out_batch * 4`` (fp32). With ``max_model_len=1<<20``
that reaches ``2**31`` at row 512, so the store silently overflows and rows
``512..`` are never written. Top-k then consumes the zero/garbage tail -> wrong
sparse-KV indices -> GLM sparse-MLA MTP acceptance collapse at concurrency>=256.

This is a *silent* bug (no crash), so it needs a test that actually crosses the
boundary. The existing pa-mqa tests use small ``max_model_len`` and never trigger
it. Here we use the real failing layout and assert:

1. every output row is written (no sentinel left), and
2. the wide output matches a compact-width reference bit-for-bit, especially the
   rows ``>= 512`` that live past the 2**31 boundary.

The compact reference uses ``out_width = context_len`` so its own largest row
offset (``rows * context_len * 4``) stays well below 2**31 and is therefore
always computed correctly.
"""

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits,
    enable_jit_gluon_pa_mqa_logits_kernel,
)

dev = "cuda"
SEED = 1234
HEADS = 128
HEAD_DIM = 128
BLOCK_SIZE = 256
NEXT_N = 4
CONTEXT_LEN = 112592
# 1<<20: row 512 hits exactly 512 * (1<<20) * 4 == 2**31 bytes (the failing shape)
WIDE_MAX_MODEL_LEN = 1 << 20
SENTINEL = 12345.0


def _make_inputs(batch_size, next_n, context_len):
    torch.manual_seed(SEED)
    q_bits = torch.randint(
        1, 64, (batch_size, next_n, HEADS, HEAD_DIM), dtype=torch.uint8, device=dev
    )
    q_fp8 = q_bits.view(dtypes.fp8)

    max_block_len = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    kv_bits = torch.randint(
        1,
        64,
        (max_block_len, BLOCK_SIZE, 1, HEAD_DIM + 4),
        dtype=torch.uint8,
        device=dev,
    )
    # last 4 fp8 bytes per token are read as one fp32 scale -> force float32(1.0)
    kv_bits[..., HEAD_DIM:] = torch.tensor(
        [0, 0, 128, 63], dtype=torch.uint8, device=dev
    )
    kv_cache = kv_bits.view(dtypes.fp8)

    weights = torch.ones((batch_size * next_n, HEADS), dtype=torch.float32, device=dev)
    context_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=dev)
    block_tables = torch.arange(max_block_len, dtype=torch.int32, device=dev).repeat(
        batch_size, 1
    )
    return q_fp8, kv_cache, weights, context_lens, block_tables


def _run(q, kv, w, ctx_lens, block_tables, out_width):
    rows = q.shape[0] * q.shape[1]
    out = torch.full((rows, out_width), SENTINEL, dtype=torch.float32, device=dev)
    deepgemm_fp8_paged_mqa_logits(
        q,
        kv,
        w,
        out,
        ctx_lens,
        block_tables,
        out_width,
        Preshuffle=True,
        KVBlockSize=BLOCK_SIZE,
        ChunkK=256,
        WavePerEU=2,
    )
    torch.cuda.synchronize()
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
# rows = batch_size * next_n; both cross the 2**31 boundary at row 512 with
# max_model_len=1<<20 (516 rows: just past 512; 1024 rows: the GLM con256 shape).
@pytest.mark.parametrize("batch_size", [129, 256])
def test_paged_mqa_logits_wide_output_no_tail_drop(batch_size):
    rows = batch_size * NEXT_N
    assert rows > 512, "shape must cross the 2**31 boundary (needs batch*next_n > 512)"

    q, kv, w, ctx_lens, block_tables = _make_inputs(batch_size, NEXT_N, CONTEXT_LEN)

    # golden reference: compact output width never crosses 2**31
    # (rows * CONTEXT_LEN * 4 << 2**31), so every row is computed correctly.
    ref = _run(q, kv, w, ctx_lens, block_tables, CONTEXT_LEN)

    # under test: the real failing layout (wide dense logits, stride 1<<20).
    wide = _run(q, kv, w, ctx_lens, block_tables, WIDE_MAX_MODEL_LEN)

    # (1) every row must be written -- directly catches the silent tail drop.
    unwritten = (wide == SENTINEL).all(dim=1)
    first_untouched = int(unwritten.nonzero()[0]) if bool(unwritten.any()) else None
    assert not bool(unwritten.any()), (
        f"buffer_store overflow: {int(unwritten.sum())}/{rows} rows left unwritten "
        f"(first_untouched_row={first_untouched}); expected all rows written."
    )

    # (2) values must be bit-identical to the compact reference -- the tail rows
    # (>=512) are the ones the overflow used to corrupt.
    valid = wide[:, :CONTEXT_LEN]
    assert torch.equal(valid, ref), "wide-output logits differ from compact reference"
    assert torch.equal(
        valid[512:], ref[512:]
    ), "rows >= 512 (past the 2**31 offset boundary) differ from the reference"


@pytest.mark.skipif(
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
    fp8_dtype = dtypes.fp8
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


if __name__ == "__main__":

    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
