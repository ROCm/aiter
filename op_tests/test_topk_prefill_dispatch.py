"""Routing tests for top_k_per_row_prefill -> `sampled` at stride0 >= the floor."""

import os
from unittest import mock

import torch

import aiter
from aiter.ops.topk import SAMPLED_MIN_STRIDE0


def _random_logits(num_rows, width):
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.full((num_rows,), width, dtype=torch.int32, device="cuda")
    torch.manual_seed(0)
    logits = torch.randn(num_rows, width, dtype=torch.float32, device="cuda")
    for i, end in enumerate(row_ends.tolist()):
        logits[i, end:] = float("-inf")
    return logits, row_starts, row_ends


def test_prefill_sampled_dispatch_routing():
    """stride0 >= SAMPLED_MIN_STRIDE0 routes top_k_per_row_prefill to `sampled`.

    Asserts on whether `sampled` ran and nothing else. Three paths can serve a
    prefill now -- `sampled`, FlyDSL one-block, and mb/ob -- and which of the
    other two takes a shape this routing declined is not this test's business.
    Asserting it made this test fail when upstream added FlyDSL, for a dispatch
    that was behaving correctly.
    """
    num_rows, top_k = 4, 2048
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    below = SAMPLED_MIN_STRIDE0 // 2
    at_or_above = SAMPLED_MIN_STRIDE0

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
        "aiter.ops.topk._top_k_per_row_prefill_sampled", autospec=True
    ) as sampled_fn:
        sampled_fn.side_effect = lambda *a, **k: None

        run_prefill(below)
        assert (
            not sampled_fn.called
        ), f"stride0={below} is below the floor and must not use `sampled`"

        sampled_fn.reset_mock()
        run_prefill(at_or_above)
        assert (
            sampled_fn.called
        ), f"stride0={at_or_above} is at the floor and must use `sampled`"

        sampled_fn.reset_mock()
        run_prefill(at_or_above, stable=True)
        assert not sampled_fn.called, "stable=True must not use `sampled`"

        sampled_fn.reset_mock()
        with mock.patch.dict(os.environ, {"AITER_DISABLE_TOPK_SAMPLED": "1"}):
            run_prefill(at_or_above)
        assert not sampled_fn.called, "AITER_DISABLE_TOPK_SAMPLED must withdraw it"

    print("[prefill_sampled_dispatch] PASS: routing by stride0 / stable / env")


if __name__ == "__main__":
    test_prefill_sampled_dispatch_routing()
