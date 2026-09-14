# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Byte-exact native paged-loader regression checks from FlyDSL #1066."""

import pytest
import torch

from aiter.ops.flydsl.kernels.fmha_gfx950.paged_memory import (
    load as _load,
)
from aiter.ops.flydsl.kernels.fmha_gfx950.paged_memory import (
    store as _store,
)
from op_tests.flydsl_tests.test_flydsl_paged_fmha import gfx950


@gfx950
@pytest.mark.parametrize("num_words", [4, 8])
@pytest.mark.parametrize("token_base", [0, 2**31 - 128])
def test_paged_fp8_value_mask_is_byte_exact(num_words, token_base):
    """Preserve every active byte and clear all tails, including signed-int32 boundaries."""
    from types import SimpleNamespace

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import gpu

    from aiter.ops.flydsl.kernels.fmha_gfx950.paged_lds import (
        DualwaveFp8KvGmemToLdsLoader,
    )

    bit_patterns = [4269833985, 305419896, 4294967295, 16909060] * (num_words // 4)
    signed_words = [word if word < 2**31 else word - 2**32 for word in bit_patterns]

    @flyc.kernel
    def mask(output: fx.Tensor):
        lane = fx.Int64(gpu.thread_idx.x)
        group = fx.Int64(gpu.block_idx.x)
        ctx = SimpleNamespace(seqlen_kv_v=fx.Int64(token_base) + lane)
        source = fx.Vector.from_elements(
            [fx.Int32(word) for word in signed_words], fx.Int32
        )
        result = DualwaveFp8KvGmemToLdsLoader._mask_v_fp8_group(
            ctx, source, fx.Int64(token_base) + group * 16
        )
        offset = (group * 128 + lane) * num_words
        _store(fx.add_offset(fx.get_iter(output), offset), fx.Vector(result))

    @flyc.jit
    def launch(output: fx.Tensor):
        mask(output).launch(grid=(9, 1, 1), block=(128, 1, 1))

    output = torch.empty((9, 128, num_words), device="cuda", dtype=torch.int32)
    launch(output.view(-1))
    torch.cuda.synchronize()
    expected = [
        [
            [
                word & (1 << min(max(length - group * 16 - i % 4 * 4, 0), 4) * 8) - 1
                for (i, word) in enumerate(bit_patterns)
            ]
            for length in range(128)
        ]
        for group in range(9)
    ]
    torch.testing.assert_close(
        output,
        torch.tensor(expected, device="cuda", dtype=torch.int64).to(torch.int32),
        rtol=0,
        atol=0,
    )


@gfx950
@pytest.mark.parametrize("active_rows", [1, 15, 16, 17, 63, 64])
def test_paged_fp8_page1_shared_page_ids_are_byte_exact(active_rows):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.fmha_gfx950.paged_lds import _page1_k_page_ids

    @flyc.kernel
    def permute_ids(source: fx.Tensor, output: fx.Tensor):
        lane = fx.Int32(fx.thread_idx.x)
        value = _load(fx.add_offset(fx.get_iter(source), lane), dtype=fx.Int32, count=1)
        permuted = _page1_k_page_ids(value)
        _store(fx.add_offset(fx.get_iter(output), lane), fx.Int32(permuted))

    @flyc.jit
    def launch(source: fx.Tensor, output: fx.Tensor):
        permute_ids(source, output).launch(grid=(1, 1, 1), block=(64, 1, 1))

    torch.manual_seed(314)
    source = torch.randint(0, 2**30, (64,), device="cuda", dtype=torch.int32)
    source[active_rows:].zero_()
    output = torch.empty_like(source)
    launch(source, output)
    torch.cuda.synchronize()
    lane = torch.arange(64, device="cuda")
    sigma = lane & 3 | (lane & 8) >> 1 | (lane & 4) << 1 | lane & ~15
    torch.testing.assert_close(output, source.index_select(0, sigma), rtol=0, atol=0)


@gfx950
@pytest.mark.parametrize("active_rows", [1, 15, 16, 17, 63, 64])
@pytest.mark.parametrize("stages", [2, 4])
def test_paged_fp8_page1_transpose_is_byte_exact(active_rows, stages):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.fmha_gfx950.paged_lds import _transpose_v_fp8_16x16

    @flyc.kernel
    def transpose_bytes(source: fx.Tensor, output: fx.Tensor):
        lane = fx.Int32(fx.thread_idx.x)
        values = _load(
            fx.add_offset(fx.get_iter(source), lane * 4), dtype=fx.Int32, count=4
        )
        transposed = _transpose_v_fp8_16x16(values, lane, stages=stages)
        _store(fx.add_offset(fx.get_iter(output), lane * 4), transposed)

    @flyc.jit
    def launch(source: fx.Tensor, output: fx.Tensor):
        transpose_bytes(source, output).launch(grid=(1, 1, 1), block=(64, 1, 1))

    torch.manual_seed(47)
    source = torch.randint(0, 256, (64, 16), device="cuda", dtype=torch.uint8)
    source[active_rows:].zero_()
    output = torch.empty_like(source)
    launch(source.view(torch.int32), output.view(torch.int32))
    torch.cuda.synchronize()
    if stages == 2:
        expected = source.reshape(16, 4, 4, 4).permute(0, 3, 2, 1)
    else:
        expected = source.reshape(4, 16, 16).transpose(1, 2)
    torch.testing.assert_close(output, expected.reshape(64, 16), rtol=0, atol=0)


@gfx950
@pytest.mark.parametrize("value_dim", [128, 192])
@pytest.mark.parametrize("active_rows", [1, 15, 16, 17, 63, 64])
def test_paged_fp8_page1_word_scatter_matches_lds_layout(value_dim, active_rows):
    from types import SimpleNamespace

    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.fmha_gfx950.paged_lds import (
        DualwaveFp8KvGmemToLdsLoader,
        _transpose_v_fp8_16x16,
    )

    stride = 80 if value_dim == 128 else 64

    @flyc.kernel
    def scatter(source: fx.Tensor, output: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % 64
        wave = tid // 64
        offset = lane * (value_dim // 4) + wave * 4
        prefix = _load(
            fx.add_offset(fx.get_iter(source), offset), dtype=fx.Int32, count=4
        )
        words = _transpose_v_fp8_16x16(prefix, lane, stages=2)
        if value_dim == 192:
            tail_offset = (wave < 4).select(offset + 32, 0)
            tail = _load(
                fx.add_offset(fx.get_iter(source), tail_offset), dtype=fx.Int32, count=4
            )
            tail_words = _transpose_v_fp8_16x16(tail, lane, stages=2)
            words = words.shuffle(tail_words, [0, 1, 2, 3, 4, 5, 6, 7])
        ctx = SimpleNamespace(
            traits=SimpleNamespace(
                HEAD_DIM_V=value_dim,
                FP8_V_ROW_STRIDE=stride,
                FP8_PV_SEGMENTED=value_dim == 192,
                FP8_V_H1=128,
                FP8_V_H2=value_dim - 128,
                KV_VEC_SIZE=16,
            ),
            lane_in_warp=fx.Int64(lane),
            wave_id=fx.Int64(wave),
            wave_id_uni=fx.Int64(wave),
            lds_vt_base_idx=fx.Int64(0),
            v_lds_i32_tiles=output,
        )
        DualwaveFp8KvGmemToLdsLoader._store_v_fp8_page1(
            ctx, words.ir_value(), fx.Int64(0)
        )

    @flyc.jit
    def launch(source: fx.Tensor, output: fx.Tensor):
        scatter(source, output).launch(grid=(1, 1, 1), block=(512, 1, 1))

    torch.manual_seed(777)
    source = torch.randint(0, 256, (64, value_dim), device="cuda", dtype=torch.uint8)
    source[active_rows:].zero_()
    storage = torch.full(
        (value_dim * stride + 128,), 171, dtype=torch.uint8, device="cuda"
    )
    output = storage[64:-64]
    launch(source.view(torch.int32).reshape(-1), output.view(torch.int32))
    torch.cuda.synchronize()
    expected = torch.full_like(storage, 171)
    expected_tile = expected[64:-64].reshape(value_dim, stride)
    token = torch.arange(64, device="cuda")
    page = token // 16
    word = token % 16 // 4
    offset = page % 2 * 4 + page // 2 * 16 + word % 2 * 32 + word // 2 * 8 + token % 4
    rows = torch.arange(value_dim, device="cuda")[:, None]
    offsets = offset[None, :].expand(value_dim, -1)
    if value_dim == 192:
        offsets = offsets ^ rows // 4 % 4 * 16
    expected_tile[rows, offsets] = source.T
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)


@gfx950
@pytest.mark.parametrize("page_size", [1, 16, 64, 1024])
@pytest.mark.parametrize("csr", [False, True])
@pytest.mark.parametrize(
    "num_pages,uniform,start_page",
    [(0, True, 0), (1, True, 0), (3, True, 4), (3, False, 0), (3, False, 3)],
)
def test_scalar_page_ids_ignore_padding(page_size, csr, num_pages, uniform, start_page):
    from types import SimpleNamespace

    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.fmha_gfx950.paged_pipeline import (
        DualwaveFp8KernelContext,
    )

    page_base = 5
    length = (num_pages - 1) * page_size + 1 if num_pages else 0

    @flyc.kernel
    def lookup(table: fx.Tensor, output: fx.Tensor):
        lane = fx.Int64(fx.thread_idx.x)
        local_page = fx.Int64(start_page)
        if fx.const_expr(not uniform):
            local_page = local_page + lane % 5
        ctx = SimpleNamespace(
            traits=SimpleNamespace(
                PAGE_SIZE=page_size,
                BLOCK_N=64,
                CSR_PAGE_TABLE=csr,
                PAIRED_PAGE_IDS=False,
            ),
            BlockTable=table,
            batch_idx=fx.Int64(1),
            block_table_stride=fx.Int64(page_base),
            page_base=fx.Int64(page_base),
            request_page_count=fx.Int64(num_pages),
            num_kv_tiles=fx.Int64((length + 63) // 64),
            seqlen_kv_v=fx.Int64(length),
        )
        DualwaveFp8KernelContext.init_page_table(ctx)
        page = DualwaveFp8KernelContext.load_page_id(
            ctx, local_page * page_size, uniform=uniform
        )
        _store(
            fx.add_offset(fx.get_iter(output), lane),
            fx.Vector.from_elements([fx.Int32(page)], fx.Int32),
        )

    @flyc.jit
    def launch(table: fx.Tensor, output: fx.Tensor):
        lookup(table, output).launch(grid=(1, 1, 1), block=(64, 1, 1))

    table = torch.full((page_base + 8,), -1, dtype=torch.int32, device="cuda")
    table[page_base : page_base + num_pages] = torch.arange(
        6, 6 + num_pages, dtype=torch.int32, device="cuda"
    )
    output = torch.empty(64, dtype=torch.int32, device="cuda")
    launch(table, output)
    pages = torch.full_like(output, start_page)
    if not uniform:
        pages += torch.arange(64, dtype=torch.int32, device="cuda") % 5
    torch.testing.assert_close(
        output, torch.where(pages < num_pages, pages + 6, 0), rtol=0, atol=0
    )
