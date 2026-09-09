# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the opus serving miss path: what a tuned row naming a kernel
id this build does not contain does to a live request.

Every kernel is faked, so these run without a GPU.
"""

import contextlib
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UNBAKED_ROW = {"solidx": 999999, "splitK": 0, "kernelName": "unbaked"}
OOM_ROW = {"solidx": 7, "splitK": 0, "kernelName": "oom"}


class _FakeUnbaked(RuntimeError):
    """Stands in for the module's UnbakedKernelError, which needs a built .so."""


def _raise_unbaked(*_a, **_k):
    raise _FakeUnbaked(
        "[AITER] Kernel id 999999 not found in a16w16 bf16 tune lookup table"
    )


def _raise_oom(*_a, **_k):
    raise RuntimeError("HIP out of memory")


def _operands():
    import torch

    return (
        torch.zeros(16, 64, dtype=torch.bfloat16),
        torch.zeros(64, 64, dtype=torch.bfloat16),
    )


@contextlib.contextmanager
def _faked_opus(tuned_row, tuned_kernel):
    """opus with every kernel faked.

    `tuned_row` is what the CSV lookup returns, `tuned_kernel` is what the tuned
    kid does when called. Yields the module and a count of what ran.
    """
    import aiter.ops.opus.gemm_op_a16w16 as g

    calls = {"tune": 0, "heuristic": 0}

    def tune(*a, **k):
        calls["tune"] += 1
        return tuned_kernel(*a, **k)

    def heuristic(*_a, **_k):
        calls["heuristic"] += 1

    with mock.patch.multiple(
        g,
        opus_gemm_a16w16_tune=tune,
        _opus_gemm_bf16_dispatch=heuristic,
        _unbaked_kernel_error=lambda: _FakeUnbaked,
        _UNBAKED_KIDS=set(),
    ), mock.patch.object(g._opus_common, "lookup_tuned", lambda **_k: tuned_row):
        yield g, calls


def test_unbaked_kid_falls_back_and_is_not_retried():
    with _faked_opus(UNBAKED_ROW, _raise_unbaked) as (g, calls):
        g.gemm_a16w16_opus(*_operands())
        g.gemm_a16w16_opus(*_operands())
        assert calls == {"tune": 1, "heuristic": 2}, calls


def test_no_tuned_row_falls_back_to_the_heuristic():
    with _faked_opus(None, _raise_unbaked) as (g, calls):
        g.gemm_a16w16_opus(*_operands())
        assert calls == {"tune": 0, "heuristic": 1}, calls


def test_workspace_oom_is_not_swallowed_as_unbaked():
    with _faked_opus(OOM_ROW, _raise_oom) as (g, calls):
        with pytest.raises(RuntimeError, match="out of memory"):
            g.gemm_a16w16_opus(*_operands())
        assert calls == {"tune": 1, "heuristic": 0}, calls
        assert (7, 0) not in g._UNBAKED_KIDS, g._UNBAKED_KIDS


def test_tuned_gemm_routes_opus_through_the_guard():
    import aiter.ops.opus.gemm_op_a16w16 as g
    import aiter.tuned_gemm as tg

    assert tg._opus_tune is g.try_opus_gemm_a16w16_tune, tg._opus_tune


def test_tuned_gemm_falls_back_to_torch_on_an_unbaked_kid():
    import torch

    import aiter.ops.opus.gemm_op_a16w16 as g
    import aiter.tuned_gemm as tg

    calls = {"tune": 0, "torch": 0}

    def tune(*_a, **_k):
        calls["tune"] += 1
        _raise_unbaked()

    def torch_gemm(inp, weights, *_a, **_k):
        calls["torch"] += 1
        return torch.zeros(
            inp.shape[0], weights.shape[0], dtype=inp.dtype, device=inp.device
        )

    with mock.patch.multiple(
        g,
        opus_gemm_a16w16_tune=tune,
        _unbaked_kernel_error=lambda: _FakeUnbaked,
        _UNBAKED_KIDS=set(),
    ), mock.patch.object(tg, "torch_gemm", torch_gemm):
        out = tg.opus_gemm(*_operands(), 999999)
        assert calls == {"tune": 1, "torch": 1}, calls
        assert out.shape == (16, 64), out.shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
