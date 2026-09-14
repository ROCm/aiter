# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Source-derived masking, rescaling and launch-contract regressions."""

import math

import pytest
import torch

from aiter.ops.flydsl.kernels import flash_attn_paged_fp8_func_gfx950 as paged
from op_tests.flydsl_tests.test_flydsl_paged_fmha import (
    DIMS,
    LAYOUTS,
    check_case,
    csr_metadata,
    gfx950,
    make_case,
    reference,
    run_case,
)


def _native_cache(case, key, value):
    pages = key.shape[0]
    if case.layout == "vectorized":
        key = key.view(pages, case.page, case.hkv, case.d // 16, 16).permute(
            0, 2, 3, 1, 4
        )
        value = value.view(pages, case.page // 16, 16, case.hkv, case.dv).permute(
            0, 3, 1, 4, 2
        )
    elif case.layout == "linear3d":
        key, value = key[:, 0], value[:, 0]
    case.k, case.v = key.contiguous(), value.contiguous()


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
@pytest.mark.parametrize("csr", [False, True])
def test_inactive_nan_bytes_and_empty_request(page, layout, d, dv, csr):
    # Unit byte-mask tests are exhaustive; these lengths cross token groups,
    # 64/128-token tiles, the K/V pipeline and the 1024-token physical page.
    for length in (1, 16, 17, 63, 64, 65, 127, 128, 129, 513, 1025):
        case = make_case(page, layout, d, dv, qlens=(17, 17), klens=(length, 0))
        case.q.zero_()
        key = torch.full(
            (case.k.shape[0], page, case.hkv, d),
            float("nan"),
            dtype=torch.float32,
            device="cuda",
        )
        value = torch.full(
            (case.k.shape[0], page, case.hkv, dv),
            float("nan"),
            dtype=torch.float32,
            device="cuda",
        )
        count = (length + page - 1) // page
        logical_key, logical_value = torch.full_like(
            key, float("nan")
        ), torch.full_like(value, float("nan"))
        logical_key.view(-1, case.hkv, d)[:length].zero_()
        logical_value.view(-1, case.hkv, dv)[:length].fill_(1)
        physical = case.table[0, :count].long()
        key[physical], value[physical] = logical_key[:count], logical_value[:count]
        _native_cache(case, key.to(case.q.dtype), value.to(case.q.dtype))
        case.qs.fill_(1)
        case.ks.fill_(1)
        case.vs.fill_(1)
        actual = run_case(case, **(csr_metadata(case, prefix=3) if csr else {}))
        expected = torch.zeros_like(actual)
        expected[max(0, 17 - length) : 17].fill_(1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
def test_shared_pages_loose_maxima_and_output_canaries(page, layout, d, dv):
    case = make_case(page, layout, d, dv, qlens=(17, 65), klens=(129, 65))
    count = (65 + page - 1) // page
    case.table[1, :count] = case.table[0, :count]
    # Oversized backing buffers make a broken bound observable as a canary
    # failure instead of an out-of-allocation memory access.
    case.maxq = 512
    query = torch.zeros((1024, case.hq, d), dtype=case.q.dtype, device="cuda")
    query[:82].copy_(case.q)
    case.q = query[:82]
    storage = torch.full((1026, case.hq, dv), 123, dtype=torch.bfloat16, device="cuda")
    case.out = storage[1:83]
    for csr in (False, True):
        case.out.fill_(float("nan"))
        check_case(case, **(csr_metadata(case, prefix=1) if csr else {}))
        assert bool((storage[:1] == 123).all())
        assert bool((storage[83:] == 123).all())


@gfx950
@pytest.mark.parametrize("mode", ["bounded", "escape", "mixed-waves", "negative"])
@pytest.mark.parametrize("batch,hkv", [(2, 1), (3, 2), (5, 1)])
@pytest.mark.parametrize("csr", [False, True])
def test_d128_query_bound_preserves_rescaling(mode, batch, hkv, csr):
    case = make_case(
        64,
        "vectorized",
        128,
        128,
        qlens=(300, 65, 257, 33, 127)[:batch],
        klens=(1024, 512, 768, 256, 384)[:batch],
        heads=(16, hkv),
        seed=29,
    )
    query = torch.zeros_like(case.q, dtype=torch.float32)
    key = torch.zeros_like(case.k, dtype=torch.float32)
    signs = torch.where(torch.arange(hkv, device="cuda") % 2 == 0, 1.0, -1.0)
    offset = 0
    for b, (qlen, klen) in enumerate(zip(case.qlens, case.klens)):
        rows = torch.arange(qlen, device="cuda")
        coefficients = torch.ones(qlen, device="cuda")
        if mode == "mixed-waves":
            coefficients = torch.where((rows // 32) % 2 == 0, 1.0, 4.0)
        query[offset : offset + qlen, :, 0] = coefficients[:, None]
        offset += qlen
        count = klen // 64
        levels = torch.where(
            (torch.arange(count, device="cuda") // 2) % 2 == 1, 1.0, -1.0
        )
        levels[:2] = 0
        if mode == "negative":
            levels.fill_(-1)
        key[case.table[b, :count].long(), :, 0, :, 0] = (
            levels[:, None, None] * signs[None, :, None] * 448
        )
    case.q, case.k = query.to(case.q.dtype), key.to(case.k.dtype)
    case.qs.fill_(1)
    # Use a non-default runtime scale: the query-bound proof must use the
    # same logit scale as the softmax, not an implicit rsqrt(D).
    scale = 0.137
    peak = {"bounded": 3.0, "escape": 16.0, "mixed-waves": 3.0, "negative": 2.5}[mode]
    case.ks.fill_(peak / (448 * scale * math.log2(math.e)))
    expected = reference(case, scale=scale)
    options = csr_metadata(case, prefix=5) if csr else {}
    for lazy in (True, False):
        actual = run_case(
            case, **options, softmax_scale=scale, dualwave_swp_lazy_rescale=lazy
        )
        assert bool(actual.isfinite().all())
        torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.005)


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
@pytest.mark.parametrize("csr", [False, True])
def test_explicit_compile_then_launch(monkeypatch, page, layout, d, dv, csr):
    case = make_case(page, layout, d, dv, qlens=(65, 17), klens=(128, 97))
    original = paged._build
    compiled = []

    def build(**kwargs):
        launcher = original(**kwargs)

        def run(*args, **options):
            compiled.append(launcher.compile(*args, **options))
            return launcher(*args, **options)

        return run

    monkeypatch.setattr(paged, "_build", build)
    check_case(case, **(csr_metadata(case, prefix=1) if csr else {}))
    assert len(compiled) == 1 and compiled[0] is not None


@gfx950
@pytest.mark.parametrize("hq,hkv", [(0, 1), (16, 0), (6, 4)])
def test_invalid_head_counts_rejected_before_build(monkeypatch, hq, hkv):
    case = make_case(1, "linear3d", 128, 128, qlens=(17,), klens=(65,))
    case.q = torch.empty((17, hq, 128), dtype=case.q.dtype, device="cuda")
    case.k = torch.empty((65, hkv, 128), dtype=case.k.dtype, device="cuda")
    case.v = torch.empty_like(case.k)
    monkeypatch.setattr(paged, "_build", lambda **kw: pytest.fail("reached builder"))
    with pytest.raises(ValueError, match="positive heads"):
        run_case(case)


@gfx950
@pytest.mark.parametrize("d,dv", DIMS)
def test_flat_int32_guard_precedes_compilation(monkeypatch, d, dv):
    case = make_case(1, "linear3d", d, dv, qlens=(17,), klens=(65,))
    monkeypatch.setattr(paged, "_MAX_FLAT_ELEMS", case.q.numel())
    monkeypatch.setattr(paged, "_build", lambda **kw: pytest.fail("reached builder"))
    with pytest.raises(NotImplementedError, match="flattened Q/O"):
        run_case(case)


@gfx950
@pytest.mark.parametrize("bound", [-1, 1 << 31, 1.5])
@pytest.mark.parametrize("field", ["maxq", "maxkv"])
def test_invalid_launch_bounds_rejected_before_build(monkeypatch, field, bound):
    case = make_case(1, "linear3d", 128, 128, qlens=(17,), klens=(65,))
    setattr(case, field, bound)
    monkeypatch.setattr(paged, "_build", lambda **kw: pytest.fail("reached builder"))
    with pytest.raises((ValueError, TypeError, NotImplementedError), match="maxima"):
        run_case(case)


@gfx950
@pytest.mark.parametrize("csr", [False, True])
def test_oversized_metadata_rejected_before_copy(monkeypatch, csr):
    case = make_case(64, "vectorized", 128, 128, qlens=(17,), klens=(65,))
    count = paged.PAGED_FP8_BUFFER_LIMIT_BYTES // 4 + 1
    options = {}
    if csr:
        options = csr_metadata(case)
        options["kv_page_indices"] = options["kv_page_indices"][:1].expand(count)
    else:
        case.table = case.table[:, :1].expand(1, count)
    monkeypatch.setattr(paged, "_build", lambda **kw: pytest.fail("reached builder"))
    with pytest.raises(NotImplementedError, match="metadata.*byte limit"):
        run_case(case, **options)


@gfx950
@pytest.mark.parametrize("compile_only", [False, True])
@pytest.mark.parametrize(
    "missing",
    [
        "cu_seqlens_q",
        "kv_metadata",
        "kv_last_page_lens",
        "block_table",
        "block_table_stride",
        "q_descale",
        "k_descale",
        "v_descale",
    ],
)
def test_direct_launcher_requires_metadata(monkeypatch, compile_only, missing):
    from aiter.ops.flydsl.kernels.fmha_gfx950 import (
        flash_attn_paged_fp8_gfx950 as kernel,
    )

    case = make_case(16, "vectorized", 128, 128, qlens=(17,), klens=(65,))
    meta = csr_metadata(case)
    launch = paged._build(
        case.hq, case.hkv, 128, 128, 16, "vectorized", False, 1, True, True, "csr", True
    )
    options = {
        "seq_len_kv": case.maxkv,
        "cu_seqlens_q": case.cuq,
        "kv_metadata": meta["kv_indptr"],
        "kv_last_page_lens": meta["kv_last_page_lens"],
        "block_table": meta["kv_page_indices"],
        "block_table_stride": 0,
        "q_descale": case.qs,
        "k_descale": case.ks,
        "v_descale": case.vs,
    }
    options[missing] = None
    monkeypatch.setattr(
        kernel, "_run_compiled", lambda *a: pytest.fail("reached launch")
    )
    monkeypatch.setattr(
        kernel.flyc, "compile", lambda *a: pytest.fail("reached compile")
    )
    call = launch.compile if compile_only else launch
    with pytest.raises(ValueError, match="requires"):
        call(
            case.q.flatten(),
            case.k,
            case.v,
            case.out.flatten(),
            1,
            case.maxq,
            **options
        )
