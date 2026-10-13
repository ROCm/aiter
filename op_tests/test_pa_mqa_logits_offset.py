#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for paged MQA logits block paging and input/output offsets.

The non-preshuffle cases check block tables, per-token scales and large KV
offsets against FP32.

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
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits,
    deepgemm_fp8_paged_mqa_logits_schedule,
    enable_jit_gluon_pa_mqa_logits_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_weight as _varctx_shuffle_weight
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

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
@pytest.mark.parametrize("block_size", [1, 8, 16, 64, 128])
@pytest.mark.parametrize("chunk_k", [64, 256])
@pytest.mark.parametrize("padded_table", [False, True], ids=["compact", "padded"])
@pytest.mark.parametrize(
    "context_lengths,next_n,heads,hidden_dim",
    [
        pytest.param((997, 63, 1), 1, 32, 128, id="decode"),
        pytest.param((3000, 257, 3, 0), 3, 64, 128, id="mtp"),
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


@pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950") or not enable_jit_gluon_pa_mqa_logits_kernel,
    reason="Requires the CDNA Gluon JIT paged MQA kernel",
)
# Page 8 is half the 16-token MFMA tile, 16 is exactly one, and 32 through 128
# are several row groups per page. 256 exceeds ChunkK // 2 and so takes the
# one-page-per-stage branch instead.
@pytest.mark.parametrize("block_size", [8, 16, 32, 64, 128, 256])
# The two ChunkK the callers ask for: the bench runs preshuffle at 128, the
# indexer at 256.
@pytest.mark.parametrize("chunk_k", [128, 256])
@pytest.mark.parametrize("padded_table", [False, True], ids=["compact", "padded"])
@pytest.mark.parametrize(
    "context_lengths,next_n,heads,hidden_dim",
    [
        pytest.param((997, 63, 1), 1, 32, 128, id="decode"),
        pytest.param((3000, 257, 3, 0), 3, 64, 128, id="mtp"),
    ],
)
@torch.inference_mode()
def test_paged_mqa_logits_preshuffle(
    block_size: int,
    chunk_k: int,
    padded_table: bool,
    context_lengths: tuple[int, ...],
    next_n: int,
    heads: int,
    hidden_dim: int,
) -> None:
    """Sweep the preshuffle path the way the plain one is swept.

    The shuffled KV feeds the MFMA B operand directly, so the page size decides
    how the tile is assembled: below 16 it spans several pages, at 16 it is one
    group, and above 16 the kernel walks row groups inside a single page.
    """
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(5614)
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

    packed = torch.empty(
        num_blocks, block_size * (hidden_dim + 4), dtype=torch.uint8, device=device
    )
    value_bytes = block_size * hidden_dim
    packed[:, :value_bytes] = (
        shuffle_weight(kv, layout=(min(block_size, 16), 16))
        .reshape(num_blocks, -1)
        .view(torch.uint8)
    )
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
        Preshuffle=True,
        KVBlockSize=block_size,
        ChunkK=chunk_k,
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


@pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950") or not enable_jit_gluon_pa_mqa_logits_kernel,
    reason="Requires the CDNA Gluon JIT paged MQA kernel",
)
@pytest.mark.parametrize(
    "layout,block_size",
    [
        pytest.param("plain", 1, id="plain-B1"),
        pytest.param("plain", 64, id="plain-B64"),
        # Preshuffle splits on ChunkKPerStage % KVBlockSize: B64 loads a page
        # index per lane, B256 keeps one page per stage. B8 is shorter than the
        # 16-token MFMA tile, so the tile spans two pages.
        pytest.param("preshuffle", 8, id="preshuffle-B8"),
        pytest.param("preshuffle", 64, id="preshuffle-B64"),
        pytest.param("preshuffle", 256, id="preshuffle-B256"),
    ],
)
@pytest.mark.parametrize("boundary_bits", [31, 32, 33], ids=["2GiB", "4GiB", "8GiB"])
@torch.inference_mode()
def test_paged_mqa_logits_large_kv_offsets(
    layout: str,
    block_size: int,
    boundary_bits: int,
) -> None:
    """Cross buffer bounds and K/FP32-scale offset overflow with a small batch."""
    preshuffle = layout != "plain"
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(5614)
    heads, hidden_dim = 32, 128
    # Unaligned lengths with an output width equal to the longest context: the
    # tail of row 0 must not reach into row 1.
    context_lengths = (3001, 513)
    batch, max_context = len(context_lengths), max(context_lengths)
    block_bytes = block_size * (hidden_dim + 4)
    boundary_page = (1 << boundary_bits) // block_bytes
    physical_pages = torch.tensor(
        [0, boundary_page - 1, boundary_page, boundary_page + 1],
        device=device,
    )

    q = torch.randn(
        batch,
        1,
        heads,
        hidden_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).to(dtypes.fp8)
    kv = torch.randn(
        len(physical_pages),
        block_size,
        hidden_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).to(dtypes.fp8)
    scales = 0.25 + torch.rand(
        len(physical_pages), block_size, device=device, generator=generator
    )
    weights = torch.randn(batch, heads, device=device, generator=generator)
    compact = torch.empty(
        len(physical_pages), block_bytes, dtype=torch.uint8, device=device
    )
    value_bytes = block_size * hidden_dim
    # A page shorter than the 16-token MFMA tile can only be shuffled in groups
    # of its own length; the kernel then reads the tile from several pages.
    values = shuffle_weight(kv, layout=(min(block_size, 16), 16)) if preshuffle else kv
    compact[:, :value_bytes] = values.reshape(len(physical_pages), -1).view(torch.uint8)
    compact[:, value_bytes:] = scales.view(torch.uint8)
    # Only four pages are referenced. Place them around the address boundary
    # without constructing a multi-GiB FP32 reference or initializing other pages.
    packed = torch.empty(
        boundary_page + 2, block_bytes, dtype=torch.uint8, device=device
    )
    packed[physical_pages] = compact
    cache = packed.view(-1, block_size, 1, hidden_dim + 4)
    table_width = (max_context + block_size - 1) // block_size
    compact_tables = torch.randint(
        len(physical_pages), (batch, table_width), device=device, generator=generator
    )
    block_tables = physical_pages[compact_tables].to(torch.int32)
    context_lens = torch.tensor(context_lengths, dtype=torch.int32, device=device)
    out = torch.full((batch, max_context), float("-inf"), device=device)
    deepgemm_fp8_paged_mqa_logits(
        q,
        cache,
        weights,
        out,
        context_lens,
        block_tables,
        max_context,
        Preshuffle=preshuffle,
        KVBlockSize=block_size,
        # More than ten chunks force both the initial loads and loop prefetch.
        TotalCuCount=1,
    )

    reference = torch.full_like(out, float("-inf"))
    dequantized_kv = kv.float() * scales[..., None]
    for b, context_length in enumerate(context_lengths):
        pages = (context_length + block_size - 1) // block_size
        keys = dequantized_kv[compact_tables[b, :pages]]
        keys = keys.reshape(-1, hidden_dim)[:context_length]
        scores = (q[b, 0].float() @ keys.T).relu()
        reference[b, :context_length] = (scores * weights[b, :, None]).sum(dim=0)
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)


# VarCtx uses a 32-bit buffer_load offset for small caches and a 64-bit
# pointer load for large caches. These tests cover both paths and both
# page-loading branches around the 2 GiB KV-read boundary.
# The VarCtx Gluon branch (_gluon_..._preshuffle_varctx) runs only on CDNA
# gfx942/gfx950. On gfx1250 the wrapper discards VarCtxSchedule and falls back to
# the non-VarCtx path, so the regression would pass vacuously there.
_VARCTX_DEVICE_ARCH = arch_info.get_arch()
_VARCTX_ARCHS = ("gfx942", "gfx950")

_VARCTX_DEV = "cuda"
_VARCTX_SEED = 256
_VARCTX_HEADS = 64
_VARCTX_HEAD_DIM = 128
_VARCTX_KV_BLOCK = 64
_VARCTX_CHUNK_K = 256
_VARCTX_INDEX_DIM = _VARCTX_HEAD_DIM + 4
_VARCTX_STRIDE_K_SEQ = (
    _VARCTX_KV_BLOCK * _VARCTX_INDEX_DIM
)  # 8448 -> block_index*this overflows int32 at 254201
_VARCTX_FP8 = get_fp8_e4m3_dtype()


def _varctx_variable_ctx(batch, lo=2048, hi=65536):
    g = torch.Generator(device="cpu").manual_seed(_VARCTX_SEED)
    return [
        max(_VARCTX_KV_BLOCK, (c // _VARCTX_KV_BLOCK) * _VARCTX_KV_BLOCK)
        for c in torch.randint(lo, hi, (batch,), generator=g, device="cpu").tolist()
    ]


def _make_varctx_inputs(batch, ctx_list):
    """Distinct per-sequence paged fp8 KV cache (block_tables = arange) so global
    block indices grow with batch and cross the 2**31 read-offset boundary."""
    max_ctx = max(ctx_list)
    max_blocks = max(
        (max_ctx + _VARCTX_CHUNK_K - 1)
        // _VARCTX_CHUNK_K
        * (_VARCTX_CHUNK_K // _VARCTX_KV_BLOCK),
        _VARCTX_CHUNK_K // _VARCTX_KV_BLOCK,
    )
    t_max = max_blocks * _VARCTX_KV_BLOCK
    num_blocks = max_blocks * batch
    context_lens = torch.tensor(ctx_list, dtype=torch.int32, device=_VARCTX_DEV)

    torch.manual_seed(0)
    q_bf16 = torch.randn(
        batch,
        1,
        _VARCTX_HEADS,
        _VARCTX_HEAD_DIM,
        dtype=torch.bfloat16,
        device=_VARCTX_DEV,
    )
    kv_bf16 = torch.randn(
        batch, t_max, _VARCTX_HEAD_DIM, dtype=torch.bfloat16, device=_VARCTX_DEV
    )
    weights = (
        torch.randn(batch, _VARCTX_HEADS, dtype=torch.float32, device=_VARCTX_DEV) * 0.1
    )
    block_tables = torch.arange(
        num_blocks, dtype=torch.int32, device=_VARCTX_DEV
    ).reshape(batch, max_blocks)

    kv_blocks = kv_bf16.reshape(num_blocks, _VARCTX_KV_BLOCK, 1, _VARCTX_HEAD_DIM)
    sf = kv_blocks.abs().float().amax(dim=3, keepdim=True).clamp(1e-4) / 240.0
    x_scaled = (kv_blocks * (1.0 / sf)).to(_VARCTX_FP8)
    kvc = torch.empty(
        (num_blocks, _VARCTX_KV_BLOCK * _VARCTX_INDEX_DIM),
        dtype=torch.uint8,
        device=_VARCTX_DEV,
    )
    kvc[:, : _VARCTX_KV_BLOCK * _VARCTX_HEAD_DIM] = x_scaled.reshape(
        num_blocks, _VARCTX_KV_BLOCK * _VARCTX_HEAD_DIM
    ).view(torch.uint8)
    kvc[:, _VARCTX_KV_BLOCK * _VARCTX_HEAD_DIM :] = sf.reshape(
        num_blocks, _VARCTX_KV_BLOCK
    ).view(torch.uint8)
    kvc = kvc.view(num_blocks, _VARCTX_KV_BLOCK, 1, _VARCTX_INDEX_DIM)
    flat = kvc.view(num_blocks, _VARCTX_KV_BLOCK * _VARCTX_INDEX_DIM)
    data = _varctx_shuffle_weight(
        flat[:, : _VARCTX_KV_BLOCK * _VARCTX_HEAD_DIM]
        .contiguous()
        .view(num_blocks, _VARCTX_KV_BLOCK, _VARCTX_HEAD_DIM)
    )
    flat[:, : _VARCTX_KV_BLOCK * _VARCTX_HEAD_DIM] = data.reshape(
        num_blocks, _VARCTX_KV_BLOCK * _VARCTX_HEAD_DIM
    )

    q_fp8 = q_bf16.to(_VARCTX_FP8).contiguous()
    return (
        q_fp8,
        kvc,
        weights.contiguous(),
        context_lens,
        block_tables,
        t_max,
        num_blocks,
    )


def _run_varctx(q, kvc, w, ctx_lens, block_tables, t_max, vcs):
    out = torch.full(
        (q.shape[0], t_max), float("-inf"), dtype=torch.float32, device=_VARCTX_DEV
    )
    deepgemm_fp8_paged_mqa_logits(
        q,
        kvc,
        w,
        out,
        ctx_lens,
        block_tables,
        t_max,
        ChunkK=_VARCTX_CHUNK_K,
        Preshuffle=True,
        KVBlockSize=_VARCTX_KV_BLOCK,
        WavePerEU=2,
        VarCtxSchedule=vcs,
    )
    torch.cuda.synchronize()
    return out


@pytest.mark.skipif(
    not torch.cuda.is_available() or _VARCTX_DEVICE_ARCH not in _VARCTX_ARCHS,
    reason=f"VarCtx Gluon path requires gfx942/gfx950; got {_VARCTX_DEVICE_ARCH}",
)
@pytest.mark.parametrize("batch", [256])
def test_varctx_kv_read_offset_no_overflow(batch):
    ctx_list = _varctx_variable_ctx(batch)
    q, kvc, w, ctx_lens, block_tables, t_max, num_blocks = _make_varctx_inputs(
        batch, ctx_list
    )

    # the config must actually cross the 2**31 KV-read boundary, otherwise the
    # test would pass even with the buggy (int32) kernel.
    max_read_off = (num_blocks - 1) * _VARCTX_STRIDE_K_SEQ
    assert max_read_off >= (1 << 31), (
        f"shape must cross 2**31 (max block-read offset={max_read_off} < 2**31); "
        f"need a wider cache (num_blocks={num_blocks}, stride_k_seq={_VARCTX_STRIDE_K_SEQ})"
    )

    # golden: SplitKV path (VarCtxSchedule=None) uses 64-bit pointer arithmetic
    # and is always correct.
    golden = _run_varctx(q, kvc, w, ctx_lens, block_tables, t_max, None)

    sched = deepgemm_fp8_paged_mqa_logits_schedule(
        batch, 1, ctx_lens, t_max, ChunkK=_VARCTX_CHUNK_K, WavePerEU=2
    )
    var = _run_varctx(q, kvc, w, ctx_lens, block_tables, t_max, sched)

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


@pytest.mark.skipif(
    not torch.cuda.is_available() or _VARCTX_DEVICE_ARCH not in _VARCTX_ARCHS,
    reason=f"VarCtx Gluon path requires gfx942/gfx950; got {_VARCTX_DEVICE_ARCH}",
)
def test_varctx_small_cache_buffer_load():
    ctx_list = [2048, 4096]
    q, kvc, w, ctx_lens, block_tables, t_max, _ = _make_varctx_inputs(2, ctx_list)
    assert kvc.numel() < 2**31 - 1

    golden = _run_varctx(q, kvc, w, ctx_lens, block_tables, t_max, None)
    sched = deepgemm_fp8_paged_mqa_logits_schedule(
        2, 1, ctx_lens, t_max, ChunkK=_VARCTX_CHUNK_K, WavePerEU=2
    )
    var = _run_varctx(q, kvc, w, ctx_lens, block_tables, t_max, sched)
    for b, ctx_len in enumerate(ctx_list):
        actual = var[b, :ctx_len].double()
        expected = golden[b, :ctx_len].double()
        cos = (actual * expected).sum() / (actual.norm() * expected.norm())
        assert cos > 0.999


@pytest.mark.skipif(
    not torch.cuda.is_available() or _VARCTX_DEVICE_ARCH not in _VARCTX_ARCHS,
    reason=f"VarCtx Gluon path requires gfx942/gfx950; got {_VARCTX_DEVICE_ARCH}",
)
def test_varctx_large_page_offset_no_overflow():
    """KVBlockSize=256 exercises the VarCtx one-page-per-stage load branch."""
    block_size = 256
    page_bytes = block_size * _VARCTX_INDEX_DIM
    high_page = (2**31 + page_bytes - 1) // page_bytes
    assert high_page * page_bytes >= 2**31

    cache = torch.zeros(
        (high_page + 2, block_size, 1, _VARCTX_INDEX_DIM),
        dtype=torch.uint8,
        device=_VARCTX_DEV,
    )
    packed = cache.view(high_page + 2, page_bytes)
    fp8_one = torch.ones((), dtype=torch.float32, device=_VARCTX_DEV).to(_VARCTX_FP8)
    for pages in (slice(0, 2), slice(high_page, high_page + 2)):
        packed[pages, : block_size * _VARCTX_HEAD_DIM] = fp8_one.view(torch.uint8)
        packed[pages, block_size * _VARCTX_HEAD_DIM :].view(torch.float32).fill_(1.0)

    q = torch.ones(
        (1, 1, _VARCTX_HEADS, _VARCTX_HEAD_DIM), dtype=torch.float32, device=_VARCTX_DEV
    ).to(_VARCTX_FP8)
    weights = torch.ones((1, _VARCTX_HEADS), dtype=torch.float32, device=_VARCTX_DEV)
    ctx_lens = torch.tensor([2 * block_size], dtype=torch.int32, device=_VARCTX_DEV)
    sched = deepgemm_fp8_paged_mqa_logits_schedule(
        1, 1, ctx_lens, 2 * block_size, ChunkK=_VARCTX_CHUNK_K, WavePerEU=2
    )

    for pages in ((0, 1), (high_page, high_page + 1)):
        block_tables = torch.tensor([pages], dtype=torch.int32, device=_VARCTX_DEV)
        out = torch.full((1, 2 * block_size), float("-inf"), device=_VARCTX_DEV)
        deepgemm_fp8_paged_mqa_logits(
            q,
            cache,
            weights,
            out,
            ctx_lens,
            block_tables,
            2 * block_size,
            ChunkK=_VARCTX_CHUNK_K,
            Preshuffle=True,
            KVBlockSize=block_size,
            WavePerEU=2,
            VarCtxSchedule=sched,
        )
        torch.testing.assert_close(
            out, torch.full_like(out, _VARCTX_HEADS * _VARCTX_HEAD_DIM)
        )


if __name__ == "__main__":

    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
