# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The fake device addresses of the FlyDSL AOT dry run (CPU only).

Host code on a replayed path compares pointers and the byte ranges they span,
so FakeTensor storages must look like disjoint allocations while views keep
their storage's address."""

import itertools

import torch

from aiter.aot.flydsl.dry_run import dry_run


def _span(tensor):
    start = tensor.data_ptr()
    return start, start + tensor.untyped_storage().nbytes()


def test_fake_storages_are_disjoint_and_aligned():
    with dry_run("gfx1250", 256):
        tensors = [
            torch.empty(shape, dtype=dtype, device="cuda")
            for shape, dtype in (
                ((1, 4096), torch.bfloat16),
                ((128, 4096), torch.bfloat16),
                ((1, 128), torch.bfloat16),
                ((3,), torch.uint8),
                ((8, 4096), torch.float32),
                ((96, 3072, 3584), torch.uint8),
            )
        ]
        spans = [_span(t) for t in tensors]
    assert all(lo % 4096 == 0 for lo, _ in spans)
    for (lo_a, hi_a), (lo_b, hi_b) in itertools.combinations(spans, 2):
        assert hi_a <= lo_b or hi_b <= lo_a


def test_views_share_their_storage_address():
    with dry_run("gfx1250", 256):
        base = torch.empty((8, 128), dtype=torch.bfloat16, device="cuda")
        view = base[2:]
        column = base[:, 64:]
        assert base.data_ptr() == base.data_ptr()
        assert view.data_ptr() == base.data_ptr() + 2 * 128 * 2
        assert column.data_ptr() == base.data_ptr() + 64 * 2
        assert torch.empty(0, device="cuda").data_ptr() == 0


def test_tdm_gemm_output_does_not_overlap_its_inputs():
    """The gfx1250 TDM GEMM refuses an output overlapping its inputs; a fresh
    output must pass that check in a replay, as it does on a GPU."""
    from aiter.ops.flydsl.gemm_a16w16_tdm import _overlaps, _storage_span

    with dry_run("gfx1250", 256):
        for m in (1, 2, 4, 8, 9):
            a = torch.empty((m, 4096), dtype=torch.bfloat16, device="cuda")
            b = torch.empty((128, 4096), dtype=torch.bfloat16, device="cuda")
            out = torch.empty((m, 128), dtype=torch.bfloat16, device="cuda")
            out_lo, out_hi = _storage_span(out)
            assert not _overlaps(out_lo, out_hi, a)
            assert not _overlaps(out_lo, out_hi, b)
            assert _overlaps(*_storage_span(a[:1]), a)
