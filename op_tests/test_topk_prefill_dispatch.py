#!/usr/bin/env python3
"""Routing tests for top_k_per_row_prefill -> AVO dispatch at stride0 >= 32768."""

import os
from unittest import mock

import torch

import aiter


def _random_logits(num_rows, width):
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.full((num_rows,), width, dtype=torch.int32, device="cuda")
    torch.manual_seed(0)
    logits = torch.randn(num_rows, width, dtype=torch.float32, device="cuda")
    for i, end in enumerate(row_ends.tolist()):
        logits[i, end:] = float("-inf")
    return logits, row_starts, row_ends


def test_prefill_avo_dispatch_routing():
    """stride0 >= 32768 routes top_k_per_row_prefill to the AVO kernels."""
    num_rows, top_k = 4, 2048
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

    def run_prefill(width, stable=False):
        logits, row_starts, row_ends = _random_logits(num_rows, width)
        aiter.top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            indices,
            None,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            k=top_k,
            stable=stable,
        )

    with mock.patch(
        "aiter.ops.topk._top_k_per_row_prefill_avo", autospec=True
    ) as avo_fn, mock.patch(
        "aiter.ops.topk._top_k_per_row_prefill", autospec=True
    ) as aiter_fn:
        avo_fn.side_effect = lambda *a, **k: None
        aiter_fn.side_effect = lambda *a, **k: None

        run_prefill(16384)
        assert aiter_fn.called and not avo_fn.called, "N=16384 should use aiter mb/ob"

        avo_fn.reset_mock()
        aiter_fn.reset_mock()
        run_prefill(65536)
        assert avo_fn.called and not aiter_fn.called, "N=65536 should use AVO"

        avo_fn.reset_mock()
        aiter_fn.reset_mock()
        run_prefill(65536, stable=True)
        assert aiter_fn.called and not avo_fn.called, "stable=True must use aiter"

        avo_fn.reset_mock()
        aiter_fn.reset_mock()
        with mock.patch.dict(os.environ, {"AITER_DISABLE_TOPK_AVO": "1"}):
            run_prefill(65536)
        assert aiter_fn.called and not avo_fn.called, "AITER_DISABLE_TOPK_AVO forces aiter"

    print("[prefill_avo_dispatch] PASS: routing by stride0 / stable / env")


if __name__ == "__main__":
    test_prefill_avo_dispatch_routing()
