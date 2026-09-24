# SPDX-License-Identifier: MIT
"""GPU regressions for scale-ring lifetime and caller-owned split-K workspace."""

import json
import os
import subprocess
import sys

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


def test_raw_unknown_kid_throws_on_fresh_thread():
    # AITER_CHECK uses thread-local exception state. A subprocess contains an
    # accidental abort; the worker must not make an earlier successful call.
    code = """
import resource
import threading
import torch
from aiter import dtypes
from aiter.ops.opus.gemm_op_a8w8 import _opus_gemm_a8w8_mxscale_bmm_launch_raw
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
x = torch.empty((32, 2, 256), device='cuda', dtype=dtypes.fp8)
w = torch.empty((2, 128, 256), device='cuda', dtype=dtypes.fp8)
y = torch.empty((32, 2, 128), device='cuda', dtype=dtypes.bf16)
xs = torch.empty((32, 2, 2), device='cuda', dtype=torch.uint8)
ws = torch.empty((2, 1, 2), device='cuda', dtype=torch.uint8)
errors = []
def run():
    try:
        _opus_gemm_a8w8_mxscale_bmm_launch_raw(x, w, y, xs, ws, None, 999999, 1)
    except RuntimeError as error:
        errors.append(str(error))
thread = threading.Thread(target=run)
thread.start()
thread.join()
assert len(errors) == 1 and 'unknown exact OPUS a8w8_mxscale_bmm kid' in errors[0], errors
print('fresh-thread RuntimeError caught')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fresh-thread RuntimeError caught" in result.stdout


def _check_large_direct_grid(kid, dtype_name):
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(dtypes, dtype_name)
    for seed in (77, 991):
        data = gen_bmm_mxscale_data(4, 16384, 1024, 4096, seed, dtype, kid, 1)
        first = None
        mismatch = None
        for _ in range(10):
            data[2].fill_(float("nan"))
            run_bmm_mxscale_bench(*data[:6], kid, 1)
            torch.cuda.synchronize()
            assert torch.isfinite(data[2]).all()
            if first is None:
                first = data[2].clone()
                mismatch = (
                    (~torch.isclose(first, data[6], rtol=0.01, atol=0.01))
                    .float()
                    .mean()
                    .item()
                )
                assert mismatch <= 0.001, mismatch
            else:
                assert torch.equal(first, data[2])
        print(
            json.dumps(
                {
                    "kid": kid,
                    "dtype": dtype_name,
                    "seed": seed,
                    "b": 4,
                    "m": 16384,
                    "n": 1024,
                    "k": 4096,
                    "split_k": 1,
                    "repeats": 10,
                    "mismatch_fraction": mismatch,
                    "finite": True,
                    "bitwise_repeatable": True,
                }
            ),
            flush=True,
        )
        del data, first


@pytest.mark.parametrize("kid", [8646, 9646])
@pytest.mark.parametrize("dtype_name", ["bf16", "fp32"])
def test_direct_only_barrier_at_large_grid(kid, dtype_name):
    code = (
        "import resource; resource.setrlimit(resource.RLIMIT_CORE, (0,0)); "
        "from op_tests.test_opus_mxscale_regressions import _check_large_direct_grid; "
        f"_check_large_direct_grid({kid}, {dtype_name!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env=dict(os.environ, AITER_REBUILD="0"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)


@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("split_k", [1, 2])
@pytest.mark.parametrize("compiled", [False, True])
def test_public_mxscale_group_matches_reference(
    group_size, split_k, compiled, monkeypatch
):
    from aiter.ops import batched_gemm_op_a8w8 as bmm
    from aiter.ops.opus import policy

    # Use the real CSV policy and cache with a controlled winner to exercise
    # both direct and allocated-workspace public paths, including torch.compile.
    kid = 9326 if group_size == 32 else 8326
    torch.backends.cuda.matmul.allow_tf32 = False
    data = gen_bmm_mxscale_data(2, 96, 128, 2048, 991, dtypes.bf16, kid, split_k)

    def lookup(*args, **kwargs):
        assert kwargs["group_size"] == group_size
        return {"kernelId": kid, "splitK": split_k, "libtype": "opus"}

    monkeypatch.setattr(policy, "lookup_mxscale_bmm_config", lookup)
    bmm._get_mxscale_bmm_launch_plan.cache_clear()
    try:
        entry = bmm.batched_gemm_a8w8_mxscale
        if compiled:
            entry = torch.compile(entry, backend="aot_eager", fullgraph=True)
        kwargs = {} if group_size == 128 else {"group_size": 32}
        result = entry(
            data[0].transpose(0, 1), data[1], data[3].transpose(0, 1), data[4], **kwargs
        )
        torch.cuda.synchronize()
        assert torch.isfinite(result).all()
        assert (
            ~torch.isclose(result, data[6], rtol=0.01, atol=0.01)
        ).float().mean().item() <= 0.001
    finally:
        bmm._get_mxscale_bmm_launch_plan.cache_clear()


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
