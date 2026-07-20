#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression test for the ``deepgemm_fp8_paged_mqa_logits`` 2**31 output-offset
overflow (silent tail-row drop).

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
from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

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

    # (1) every row must be written — directly catches the silent tail drop.
    unwritten = (wide == SENTINEL).all(dim=1)
    first_untouched = int(unwritten.nonzero()[0]) if bool(unwritten.any()) else None
    assert not bool(unwritten.any()), (
        f"buffer_store overflow: {int(unwritten.sum())}/{rows} rows left unwritten "
        f"(first_untouched_row={first_untouched}); expected all rows written."
    )

    # (2) values must be bit-identical to the compact reference — the tail rows
    # (>=512) are the ones the overflow used to corrupt.
    valid = wide[:, :CONTEXT_LEN]
    assert torch.equal(valid, ref), "wide-output logits differ from compact reference"
    assert torch.equal(
        valid[512:], ref[512:]
    ), "rows >= 512 (past the 2**31 offset boundary) differ from the reference"


if __name__ == "__main__":

    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
