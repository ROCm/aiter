# SPDX-License-Identifier: MIT
"""GPU regressions for scale-ring lifetime and caller-owned split-K workspace."""

import pytest
import torch

from aiter import dtypes
from aiter.jit.core import get_gfx
from aiter.ops.opus import gemm_op_a8w8
from csrc.opus_gemm.opus_bmm_mxscale_tune import (
    gen_bmm_mxscale_data,
    run_bmm_mxscale_bench,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950", reason="requires gfx950"
)


@pytest.mark.parametrize("m", [48, 64, 1024, 1025])
@pytest.mark.parametrize("dtype", [dtypes.bf16, dtypes.fp32])
def test_mx32_preload_scale_ring_is_repeatable(m, dtype):
    # A mean-error gate missed this race: a few rows changed between identical
    # calls. Large M increases the number of workgroups and scheduling variation.
    torch.backends.cuda.matmul.allow_tf32 = False
    data = gen_bmm_mxscale_data(8, m, 1024, 4096, 20260924 + m, dtype, 9324, 1)
    first = None
    for _ in range(20):
        data[2].fill_(float("nan"))
        run_bmm_mxscale_bench(*data[:6], 9324, 1)
        torch.cuda.synchronize()
        assert torch.isfinite(data[2]).all()
        if first is None:
            first = data[2].clone()
            mismatch = (
                (~torch.isclose(first, data[6], rtol=0.01, atol=0.01)).float().mean()
            )
            assert mismatch.item() <= 0.001
        else:
            assert torch.equal(data[2], first)


@pytest.mark.parametrize("kid", [8326, 9326])
def test_owned_workspace_uses_checked_launch_without_python_planning(kid, monkeypatch):
    torch.backends.cuda.matmul.allow_tf32 = False
    data = gen_bmm_mxscale_data(2, 96, 128, 2048, 991, dtypes.bf16, kid, 2)
    run_bmm_mxscale_bench(*data[:5], None, kid, 2)
    torch.cuda.synchronize()
    expected = data[2].clone()

    def unexpected(*args, **kwargs):
        raise AssertionError("caller-owned workspace entered Python planning")

    monkeypatch.setattr(gemm_op_a8w8, "_get_cached_a8w8_mxscale_bmm_plan", unexpected)
    monkeypatch.setattr(gemm_op_a8w8, "_validate_a8w8_mxscale_bmm_tensors", unexpected)
    run_bmm_mxscale_bench(*data[:6], kid, 2)
    torch.cuda.synchronize()
    assert torch.equal(data[2], expected)


@pytest.mark.parametrize(
    "kid,split_k", [(9325, 1), (9326, 1), (9326, 2), (9327, 1), (9327, 2)]
)
def test_other_mx32_preload_instances_share_the_ring_fix(kid, split_k):
    torch.backends.cuda.matmul.allow_tf32 = False
    data = gen_bmm_mxscale_data(8, 1024, 1024, 4096, 991, dtypes.bf16, kid, split_k)
    first = None
    for _ in range(20):
        data[2].fill_(float("nan"))
        run_bmm_mxscale_bench(*data[:6], kid, split_k)
        torch.cuda.synchronize()
        assert torch.isfinite(data[2]).all()
        if first is None:
            first = data[2].clone()
            assert (
                ~torch.isclose(first, data[6], rtol=0.01, atol=0.01)
            ).float().mean().item() <= 0.001
        else:
            assert torch.equal(data[2], first)


@pytest.mark.parametrize("kid", [8326, 9326])
@pytest.mark.parametrize(
    "case", ["capacity", "dtype", "layout", "alignment", "device", "object"]
)
def test_owned_workspace_is_still_validated(kid, case):
    data = gen_bmm_mxscale_data(2, 96, 128, 2048, 991, dtypes.bf16, kid, 2)
    size = data[5].numel()
    if case == "capacity":
        workspace = data[5][:-1]
    elif case == "dtype":
        workspace = torch.empty(size, device="cuda", dtype=torch.bfloat16)
    elif case == "layout":
        workspace = torch.empty(size * 2, device="cuda")[::2]
    elif case == "alignment":
        workspace = torch.empty(size + 1, device="cuda")[1:]
    elif case == "device":
        workspace = torch.empty(size, device="cpu")
    else:
        workspace = object()
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        run_bmm_mxscale_bench(*data[:5], workspace, kid, 2)


@pytest.mark.parametrize("m", [16384, 24576, 16385])
def test_large_b16_batch_pair_schedule(m):
    # 24576 has an odd panel count; 16385 retains the partial-tile path.
    torch.backends.cuda.matmul.allow_tf32 = False
    data = gen_bmm_mxscale_data(16, m, 1024, 4096, 77, dtypes.bf16, 9158, 1)
    first = None
    for _ in range(3):
        data[2].fill_(float("nan"))
        run_bmm_mxscale_bench(*data[:6], 9158, 1)
        torch.cuda.synchronize()
        assert torch.isfinite(data[2]).all()
        if first is None:
            first = data[2].clone()
            mismatch = (
                (~torch.isclose(first, data[6], rtol=0.01, atol=0.01)).float().mean()
            )
            assert mismatch.item() <= 0.001
        else:
            assert torch.equal(first, data[2])


@pytest.mark.parametrize("m", [16384, 20480, 16385])
def test_large_b16_gs128_batch_pair_schedule(m):
    # 20480 has an odd panel count; 16385 retains the partial-tile path.
    torch.backends.cuda.matmul.allow_tf32 = False
    data = gen_bmm_mxscale_data(16, m, 1024, 4096, 77, dtypes.bf16, 8158, 1)
    first = None
    for _ in range(3):
        data[2].fill_(float("nan"))
        run_bmm_mxscale_bench(*data[:6], 8158, 1)
        torch.cuda.synchronize()
        assert torch.isfinite(data[2]).all()
        if first is None:
            first = data[2].clone()
            mismatch = (
                (~torch.isclose(first, data[6], rtol=0.01, atol=0.01)).float().mean()
            )
            assert mismatch.item() <= 0.001
        else:
            assert torch.equal(first, data[2])
