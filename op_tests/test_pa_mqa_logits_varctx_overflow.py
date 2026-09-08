#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression test for the ``deepgemm_fp8_paged_mqa_logits`` VarCtx path 2**31
KV-read-offset overflow (silent zeroed logits).

Background
----------
The VarCtx gluon kernel
``_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx`` loaded K/scale with
``gl.amd.cdna3.buffer_load(offsets=context_kv_idx * stride_k_seq + ...)``. The
buffer-load voffset is a **32-bit** element offset, while ``context_kv_idx`` is an
int32 block index. For a wide paged cache the product ``block_index * stride_k_seq``
crosses ``2**31`` and wraps, so the load reads out of bounds and returns 0 -> the
logits for those positions silently collapse to 0.

``stride_k_seq = KVBlockSize * (head_dim + 4)`` (the +4 fp32-scale bytes make the
row stride non-power-of-two, e.g. 64*(128+4)=8448). Overflow starts at
``block_index >= 2**31 / 8448 = 254201``. The non-VarCtx (SplitKV / plain
preshuffle) path never triggers this because it loads with 64-bit pointer
arithmetic ``gl.load(KV_buffer + context_kv_idx.to(int64) * stride_k_seq)``.

The bug only shows up when (a) the VarCtx schedule is used, and (b) the paged
cache is wide enough that block indices reach the 2**31 boundary -- i.e. large
batch with distinct per-sequence blocks. All existing pa-mqa tests either don't
exercise the VarCtx path, or share a small block table across sequences, so they
never cross the boundary.

This test uses distinct per-sequence blocks and a batch big enough to cross
2**31, then asserts the VarCtx output matches the known-good SplitKV output and
that no valid position was zeroed.
"""

import pytest
import torch

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits,
    deepgemm_fp8_paged_mqa_logits_schedule,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

dev = "cuda"
SEED = 256
HEADS = 64
HEAD_DIM = 128
KV_BLOCK = 64
CHUNK_K = 256
INDEX_DIM = HEAD_DIM + 4
STRIDE_K_SEQ = (
    KV_BLOCK * INDEX_DIM
)  # 8448 -> block_index*this overflows int32 at 254201
FP8 = get_fp8_e4m3_dtype()


def _variable_ctx(batch, lo=2048, hi=65536):
    g = torch.Generator().manual_seed(SEED)
    return [
        max(KV_BLOCK, (c // KV_BLOCK) * KV_BLOCK)
        for c in torch.randint(lo, hi, (batch,), generator=g).tolist()
    ]


def _make_inputs(batch, ctx_list):
    """Distinct per-sequence paged fp8 KV cache (block_tables = arange) so global
    block indices grow with batch and cross the 2**31 read-offset boundary."""
    max_ctx = max(ctx_list)
    max_blocks = max(
        (max_ctx + CHUNK_K - 1) // CHUNK_K * (CHUNK_K // KV_BLOCK), CHUNK_K // KV_BLOCK
    )
    t_max = max_blocks * KV_BLOCK
    num_blocks = max_blocks * batch
    context_lens = torch.tensor(ctx_list, dtype=torch.int32, device=dev)

    torch.manual_seed(0)
    q_bf16 = torch.randn(batch, 1, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    kv_bf16 = torch.randn(batch, t_max, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    weights = torch.randn(batch, HEADS, dtype=torch.float32, device=dev) * 0.1
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(
        batch, max_blocks
    )

    kv_blocks = kv_bf16.reshape(num_blocks, KV_BLOCK, 1, HEAD_DIM)
    sf = kv_blocks.abs().float().amax(dim=3, keepdim=True).clamp(1e-4) / 240.0
    x_scaled = (kv_blocks * (1.0 / sf)).to(FP8)
    kvc = torch.empty((num_blocks, KV_BLOCK * INDEX_DIM), dtype=torch.uint8, device=dev)
    kvc[:, : KV_BLOCK * HEAD_DIM] = x_scaled.reshape(
        num_blocks, KV_BLOCK * HEAD_DIM
    ).view(torch.uint8)
    kvc[:, KV_BLOCK * HEAD_DIM :] = sf.reshape(num_blocks, KV_BLOCK).view(torch.uint8)
    kvc = kvc.view(num_blocks, KV_BLOCK, 1, INDEX_DIM)
    flat = kvc.view(num_blocks, KV_BLOCK * INDEX_DIM)
    data = shuffle_weight(
        flat[:, : KV_BLOCK * HEAD_DIM].contiguous().view(num_blocks, KV_BLOCK, HEAD_DIM)
    )
    flat[:, : KV_BLOCK * HEAD_DIM] = data.reshape(num_blocks, KV_BLOCK * HEAD_DIM)

    q_fp8 = q_bf16.to(FP8).contiguous()
    return (
        q_fp8,
        kvc,
        weights.contiguous(),
        context_lens,
        block_tables,
        t_max,
        num_blocks,
    )


def _run(q, kvc, w, ctx_lens, block_tables, t_max, vcs):
    out = torch.full(
        (q.shape[0], t_max), float("-inf"), dtype=torch.float32, device=dev
    )
    deepgemm_fp8_paged_mqa_logits(
        q,
        kvc,
        w,
        out,
        ctx_lens,
        block_tables,
        t_max,
        ChunkK=CHUNK_K,
        Preshuffle=True,
        KVBlockSize=KV_BLOCK,
        WavePerEU=2,
        VarCtxSchedule=vcs,
    )
    torch.cuda.synchronize()
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
@pytest.mark.parametrize("batch", [256])
def test_varctx_kv_read_offset_no_overflow(batch):
    ctx_list = _variable_ctx(batch)
    q, kvc, w, ctx_lens, block_tables, t_max, num_blocks = _make_inputs(batch, ctx_list)

    # the config must actually cross the 2**31 KV-read boundary, otherwise the
    # test would pass even with the buggy (int32) kernel.
    max_read_off = (num_blocks - 1) * STRIDE_K_SEQ
    assert max_read_off >= (1 << 31), (
        f"shape must cross 2**31 (max block-read offset={max_read_off} < 2**31); "
        f"need a wider cache (num_blocks={num_blocks}, stride_k_seq={STRIDE_K_SEQ})"
    )

    # golden: SplitKV path (VarCtxSchedule=None) uses 64-bit pointer arithmetic
    # and is always correct.
    golden = _run(q, kvc, w, ctx_lens, block_tables, t_max, None)

    sched = deepgemm_fp8_paged_mqa_logits_schedule(
        batch, 1, ctx_lens, t_max, ChunkK=CHUNK_K, WavePerEU=2
    )
    var = _run(q, kvc, w, ctx_lens, block_tables, t_max, sched)

    # (1) no valid position may be silently zeroed where the golden is non-zero
    #     (the buffer_load OOB-returns-0 signature).
    worst_row, worst_zeroed = -1, 0
    for b in range(batch):
        c = ctx_list[b]
        g = golden[b, :c]
        v = var[b, :c]
        zeroed = int(((v == 0) & (g.abs() > 1e-3)).sum())
        if zeroed > worst_zeroed:
            worst_zeroed, worst_row = zeroed, b
    assert worst_zeroed == 0, (
        f"VarCtx zeroed {worst_zeroed} valid logits in row {worst_row} "
        f"(ctx={ctx_list[worst_row]}): int32 overflow in the K/scale load offset."
    )

    # (2) per-row cosine vs the golden SplitKV output must be ~1.0 everywhere.
    min_cos, min_row = 1.0, -1
    for b in range(batch):
        c = ctx_list[b]
        g = golden[b, :c].double()
        v = var[b, :c].double()
        cos = float((g * v).sum() / (g.norm() * v.norm() + 1e-12))
        if cos < min_cos:
            min_cos, min_row = cos, b
    assert min_cos > 0.999, (
        f"VarCtx diverges from SplitKV: min per-row cos={min_cos:.6f} at row "
        f"{min_row} (ctx={ctx_list[min_row]})."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
