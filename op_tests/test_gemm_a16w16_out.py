# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Caller-owned output for tuned a16w16 GEMM."""

import torch
import torch.nn.functional as F

from aiter.tuned_gemm import gemm_a16w16, skinny_gemm


def _check(out, ref, produced):
    torch.testing.assert_close(produced, ref)
    assert produced.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, ref)


def test_out_matches_linear_and_overwrites():
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
    ref = F.linear(x, weight)
    out = torch.full_like(ref, 1000)
    produced = gemm_a16w16(x, weight, out=out)
    _check(out, ref, produced)


def test_bad_out_shape_raises():
    x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(2, 4, dtype=torch.bfloat16, device="cuda")
    try:
        gemm_a16w16(x, weight, out=out)
    except ValueError:
        return
    raise AssertionError("a mismatched out shape must be rejected")


def test_unknown_skinny_solution_raises():
    x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
    try:
        skinny_gemm(x, weight, 9)
    except ValueError:
        return
    raise AssertionError("an unknown skinny solution must be rejected")


def test_nonviewable_batch_still_writes_out():
    base = torch.randn(3, 2, 16, dtype=torch.bfloat16, device="cuda")
    x = base.transpose(0, 1)
    try:
        x.view(-1, x.size(-1))
    except RuntimeError:
        pass
    else:
        raise AssertionError("this case needs a batch tensor view() rejects")
    weight = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
    ref = F.linear(x, weight)
    out = torch.full_like(ref, 1000)
    produced = gemm_a16w16(x, weight, out=out)
    _check(out, ref, produced)


if __name__ == "__main__":
    test_out_matches_linear_and_overwrites()
    test_bad_out_shape_raises()
    test_unknown_skinny_solution_raises()
    test_nonviewable_batch_still_writes_out()
    print("ok")
