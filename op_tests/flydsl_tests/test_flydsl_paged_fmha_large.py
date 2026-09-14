# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Real large-cache addresses and descriptor-boundary regressions."""

import math

import pytest
import torch

from aiter.ops.flydsl.kernels import flash_attn_paged_fp8_func_gfx950 as paged
from op_tests.flydsl_tests.test_flydsl_paged_fmha import (
    DIMS,
    LAYOUTS,
    gfx950,
    make_case,
    run_case,
)


def _large_case(page, layout, d, dv, high_page):
    hkv = 2
    npages = high_page + 1
    needed = npages * page * hkv * (d + dv)
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < needed + 2 * 2**30:
        pytest.skip(f"requires {needed / 2**30:.1f} GiB cache plus 2 GiB headroom")
    case = make_case(page, layout, d, dv, qlens=(1,), klens=(1,))
    if layout == "vectorized":
        kshape = (npages, hkv, d // 16, page, 16)
        vshape = (npages, hkv, page // 16, dv, 16)
    elif layout == "linear3d":
        kshape, vshape = (npages, hkv, d), (npages, hkv, dv)
    else:
        kshape, vshape = (npages, page, hkv, d), (npages, page, hkv, dv)
    case.k = torch.empty(kshape, dtype=torch.float8_e4m3fn, device="cuda")
    case.v = torch.empty(vshape, dtype=torch.float8_e4m3fn, device="cuda")
    case.k[high_page].zero_()
    case.v[high_page].fill_(float("nan"))
    first = (1.0 + torch.arange(hkv * dv, device="cuda").reshape(hkv, dv) % 7 / 8).to(
        torch.float8_e4m3fn
    )
    if layout == "vectorized":
        case.v[high_page, :, 0, :, 0].copy_(first)
    else:
        case.v[high_page].reshape(page, hkv, dv)[0].copy_(first)
    case.vs.fill_(1.0)
    case.table.fill_(-1)
    case.table[0, 0] = high_page
    expected = first.float().repeat_interleave(case.hq // hkv, dim=0)[None]
    return case, expected


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
@pytest.mark.parametrize("csr", [False, True])
def test_cache_offsets_above_4gib(page, layout, d, dv, csr):
    high_page = math.ceil(2**32 / (page * 2 * min(d, dv)))
    case, expected = _large_case(page, layout, d, dv, high_page)
    options = {}
    if csr:
        options = {
            "block_table": None,
            "kv_indptr": torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            "kv_page_indices": torch.tensor(
                [high_page], dtype=torch.int32, device="cuda"
            ),
            "kv_last_page_lens": torch.tensor([1], dtype=torch.int32, device="cuda"),
        }
    actual = run_case(case, **options)
    torch.cuda.synchronize()
    assert high_page * page * case.hkv * d >= 2**32
    assert high_page * page * case.hkv * dv >= 2**32
    assert bool(actual.isfinite().all())
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)


@gfx950
@pytest.mark.parametrize("csr", [False, True])
def test_page16_scalar_offset_near_i32_limit(monkeypatch, csr):
    page, d, dv = 16, 192, 192
    high_page = paged.PAGED_FP8_BUFFER_LIMIT_BYTES // (page * 2 * d) - 1
    case, expected = _large_case(page, "vectorized", d, dv, high_page)
    assert case.k.numel() <= paged.PAGED_FP8_BUFFER_LIMIT_BYTES
    assert high_page * page * case.hkv * d > 2**31 - 16384
    original = paged._build
    selected = []

    def build(**kwargs):
        selected.append(kwargs["buffered"])
        return original(**kwargs)

    monkeypatch.setattr(paged, "_build", build)
    options = {}
    if csr:
        options = {
            "block_table": None,
            "kv_indptr": torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            "kv_page_indices": torch.tensor(
                [high_page], dtype=torch.int32, device="cuda"
            ),
            "kv_last_page_lens": torch.tensor([1], dtype=torch.int32, device="cuda"),
        }
    actual = run_case(case, **options)
    torch.cuda.synchronize()
    assert selected == [True]
    assert bool(actual.isfinite().all())
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
