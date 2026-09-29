# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Named Gluon row dispatch, CPU fallbacks and optional GPU graph replay."""

from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import aiter.tuned_gemm as tuned

GFX, CU_NUM = "gfx950", 256
N = K = 2048
BF16 = str(torch.bfloat16)


def _key(
    m, n=N, k=K, bias=False, dtype=BF16, otype=BF16, scaleAB=False, bpreshuffle=False
):
    return (GFX, CU_NUM, m, n, k, bias, dtype, otype, scaleAB, bpreshuffle)


def _row(libtype, kernel_name):
    return {"libtype": libtype, "solidx": 0, "splitK": 0, "kernelName": kernel_name}


def _padded_m(m, _n, _k, gl):
    # Stand-in for the native helper: gl 0 rounds up to a multiple of 16, gl 1 to
    # the next power of two.
    return -(-m // 16) * 16 if gl == 0 else 1 << (m - 1).bit_length()


class FakeKernel:
    """Computes only M = 128, reads only 16-byte aligned weights, and records the
    calls it gets; raises if ``fail``."""

    def __init__(self, m=128):
        self.m = m
        self.calls = []
        self.support_calls = []
        self.fail = False

    def supported(self, M, N, K, bias, dtype, otype):
        self.support_calls.append((M, N, K, bias, dtype, otype))
        return (M, N, K, bias, dtype, otype) == (
            self.m,
            2048,
            2048,
            False,
            torch.bfloat16,
            torch.bfloat16,
        )

    def accepts(self, x, w):
        return (
            w.is_contiguous()
            and w.data_ptr() % 16 == 0
            and not getattr(w, "is_shuffled_16x32", False)
        )

    def gemm(self, x, w):
        self.calls.append(x.shape[0])
        if self.fail:
            raise RuntimeError("stand-in for a kernel that does not compile")
        return F.linear(x, w)


@pytest.fixture
def table(monkeypatch):
    """Point the lookup at a tuned table the test fills, on a 256-CU gfx950."""
    rows = {}
    kernel = FakeKernel()
    warnings = []
    monkeypatch.setattr(tuned, "get_GEMM_A16W16_config_", lambda: rows)
    monkeypatch.setattr(tuned, "get_gfx", lambda: GFX)
    monkeypatch.setattr(tuned, "get_cu_num", lambda: CU_NUM)
    monkeypatch.setattr(tuned, "get_padded_m", _padded_m)
    monkeypatch.setattr(tuned, "_failed_gluon_gemm_kernels", set())
    monkeypatch.setattr(
        tuned,
        "_get_gluon_gemm_kernels",
        lambda: {"fake_m128": (kernel.supported, kernel.accepts, kernel.gemm)},
    )
    monkeypatch.setattr(
        tuned.logger, "warning", lambda message, *args: warnings.append(message % args)
    )
    tuned.get_GEMM_A16W16_config.cache_clear()
    yield rows, kernel, warnings
    tuned.get_GEMM_A16W16_config.cache_clear()


def _lookup(m, n=N, k=K, **kwargs):
    tuned.get_GEMM_A16W16_config.cache_clear()
    return tuned.get_GEMM_A16W16_config(*_key(m, n, k, **kwargs)[2:])


@pytest.fixture
def inputs():
    return torch.randn(128, K).to(torch.bfloat16), torch.randn(N, K).to(torch.bfloat16)


@pytest.mark.parametrize("m", [16, 32, 64, 128])
def test_mm_selects_registered_shape_but_rejects_a_padded_call(
    table, inputs, monkeypatch, m
):
    rows, _, _ = table
    without = _lookup(m - 1)
    kernel = FakeKernel(m)
    monkeypatch.setattr(
        tuned,
        "_get_gluon_gemm_kernels",
        lambda: {"named": (kernel.supported, kernel.accepts, kernel.gemm)},
    )
    rows[_key(m)] = _row("gluon", "named")
    assert _lookup(m)["libtype"] == "gluon"
    assert kernel.support_calls == [(m, N, K, False, torch.bfloat16, torch.bfloat16)]
    x, w = inputs
    torch.testing.assert_close(tuned.tgemm.mm(x[:m], w), F.linear(x[:m], w))
    kernel.support_calls.clear()
    assert _lookup(m - 1) == without
    assert kernel.support_calls and all(
        call[0] == m - 1 for call in kernel.support_calls
    )
    torch.testing.assert_close(tuned.tgemm.mm(x[: m - 1], w), F.linear(x[: m - 1], w))
    assert kernel.calls == [m]


@pytest.mark.parametrize(
    "kwargs", [{"bias": True}, {"scaleAB": True}, {"bpreshuffle": True}]
)
def test_incompatible_row_resolves_as_absent_without_importing_kernels(
    table, monkeypatch, kwargs
):
    rows, _, _ = table
    without = _lookup(128, **kwargs)
    rows[_key(128, **kwargs)] = _row("gluon", "fake_m128")

    def forbidden_registry():
        pytest.fail("an incompatible row must not import its registered kernels")

    monkeypatch.setattr(tuned, "_get_gluon_gemm_kernels", forbidden_registry)
    assert _lookup(128, **kwargs) == without


@pytest.mark.parametrize(
    "n,k,dtype,otype",
    [
        (N, K, torch.float16, torch.bfloat16),
        (N, K, torch.bfloat16, torch.float32),
        (1024, K, torch.bfloat16, torch.bfloat16),
        (N, 1024, torch.bfloat16, torch.bfloat16),
    ],
)
def test_supported_receives_actual_shape_and_dtypes_for_rejection(
    table, n, k, dtype, otype
):
    rows, kernel, _ = table
    kwargs = {"n": n, "k": k, "dtype": str(dtype), "otype": str(otype)}
    without = _lookup(128, **kwargs)
    rows[_key(128, **kwargs)] = _row("gluon", "fake_m128")
    assert _lookup(128, **kwargs) == without
    assert set(kernel.support_calls) == {(128, n, k, False, dtype, otype)}


@pytest.mark.parametrize("reason", ["unsupported", "disabled"])
def test_unusable_row_leaves_the_next_padded_candidate_selected(table, reason):
    rows, kernel, _ = table
    # M100 first pads to 112, then 128. Disable an otherwise eligible kernel so
    # the second case tests failed-name rejection, not merely its shape guard.
    if reason == "disabled":
        kernel.m = 100
        tuned._failed_gluon_gemm_kernels.add("fake_m128")
    next_row = _row("torch", "native")
    rows[_key(112)] = _row("gluon", "fake_m128")
    rows[_key(128)] = next_row
    assert _lookup(100) == next_row


def test_unregistered_name_falls_back_with_a_warning(table):
    rows, _, warnings = table
    rows[_key(128)] = _row("gluon", "no_such_kernel")
    assert _lookup(128)["libtype"] != "gluon"
    assert warnings and all("no_such_kernel" in w for w in warnings)


def test_failure_disables_only_that_name_and_invalidates_cached_selection(
    table, inputs, monkeypatch
):
    rows, kernel, warnings = table
    rows[_key(128)] = _row("gluon", "fake_m128")
    kernel.fail = True
    other = FakeKernel(64)
    rows[_key(64)] = _row("gluon", "other")
    monkeypatch.setattr(
        tuned,
        "_get_gluon_gemm_kernels",
        lambda: {
            "fake_m128": (kernel.supported, kernel.accepts, kernel.gemm),
            "other": (other.supported, other.accepts, other.gemm),
        },
    )
    x, w = inputs
    torch.testing.assert_close(tuned.tgemm.mm(x[:64], w), F.linear(x[:64], w))
    for _ in range(2):
        torch.testing.assert_close(tuned.tgemm.mm(x, w), F.linear(x, w))
    assert kernel.calls == [128]
    assert len(warnings) == 1 and "fake_m128" in warnings[0]
    config = tuned.get_GEMM_A16W16_config(128, N, K, False, BF16, BF16)
    assert config["libtype"] == "torch"
    assert _lookup(64)["kernelName"] == "other"
    torch.testing.assert_close(tuned.tgemm.mm(x[:64], w), F.linear(x[:64], w))
    assert other.calls == [64, 64]
    assert tuned._failed_gluon_gemm_kernels == {"fake_m128"}


@pytest.mark.parametrize("layout", ["misaligned", "strided", "shuffled_tag"])
def test_operands_the_kernel_rejects_run_torch_for_that_call_only(
    table, inputs, layout
):
    rows, kernel, warnings = table
    rows[_key(128)] = _row("gluon", "fake_m128")
    tuned.get_GEMM_A16W16_config.cache_clear()
    x, w = inputs
    if layout == "misaligned":
        rejected = torch.empty(N * K + 1, dtype=torch.bfloat16)[1:].view(N, K)
        rejected.copy_(w)
    elif layout == "strided":
        rejected = w.t()
    else:
        rejected = w.clone()
        rejected.is_shuffled_16x32 = True
    torch.testing.assert_close(tuned.tgemm.mm(x, rejected), F.linear(x, rejected))
    assert kernel.calls == [] and not warnings
    torch.testing.assert_close(tuned.tgemm.mm(x, w), F.linear(x, w))
    assert kernel.calls == [128]


def test_batched_mm_restores_leading_dimensions_and_output_dtype(table, inputs):
    rows, kernel, _ = table
    rows[_key(128)] = _row("gluon", "fake_m128")
    x, w = inputs
    x = x.view(2, 64, K)
    out = tuned.tgemm.mm(x, w)
    assert kernel.calls == [128]
    assert out.shape == (2, 64, N) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out, F.linear(x, w))


def test_unsupported_output_dtype_keeps_incumbent_result(table, inputs):
    rows, kernel, _ = table
    rows[_key(128, otype=str(torch.float16))] = _row("gluon", "fake_m128")
    x, w = inputs
    out = tuned.tgemm.mm(x, w, otype=torch.float16)
    assert not kernel.calls and out.dtype == torch.float16
    torch.testing.assert_close(out, F.linear(x, w).to(torch.float16))


def test_scale_c_keeps_incumbent_arguments_without_disabling_row(
    table, inputs, monkeypatch
):
    rows, kernel, _ = table
    rows[_key(128)] = _row("gluon", "fake_m128")
    seen = []
    original = tuned.torch_gemm

    def incumbent(*args, **kwargs):
        seen.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(tuned, "torch_gemm", incumbent)
    x, w = inputs
    scale_c = torch.tensor(2.0)
    out = tuned.tgemm.mm(x, w, scale_c=scale_c)
    assert not kernel.calls and len(seen) == 1
    assert seen[0][0][7] is scale_c
    # Existing BF16 torch_gemm ignores scale_c; this checks dispatch parity, not
    # newly implemented scale semantics.
    torch.testing.assert_close(out, F.linear(x, w))
    assert not tuned._failed_gluon_gemm_kernels
    assert _lookup(128)["kernelName"] == "fake_m128"


def test_warmed_host_path_uses_cached_lookup_without_device_queries(
    table, inputs, monkeypatch
):
    rows, kernel, _ = table
    rows[_key(128)] = _row("gluon", "fake_m128")
    x, w = inputs
    tuned.tgemm.mm(x, w)

    def forbidden_query(*args, **kwargs):
        pytest.fail("a warmed dispatch must not query device or padded-M metadata")

    for name in ("get_gfx", "get_cu_num", "get_padded_m"):
        monkeypatch.setattr(tuned, name, forbidden_query)
    for _ in range(3):
        torch.testing.assert_close(tuned.tgemm.mm(x, w), F.linear(x, w))
    assert kernel.calls == [128] * 4


def test_shipped_gluon_rows_name_registered_kernels():
    configs = Path(__file__).resolve().parents[1] / "aiter" / "configs"
    tables = [configs / "bf16_tuned_gemm.csv"]
    tables += sorted((configs / "model_configs").glob("*bf16_tuned_gemm*.csv"))
    rows = pd.concat([pd.read_csv(path) for path in tables])
    names = set(rows.loc[rows["libtype"] == "gluon", "kernelName"])
    assert names <= set(tuned._get_gluon_gemm_kernels())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU graph replay needs a GPU"
)
@pytest.mark.parametrize("m", [16, 32, 64, 128])
def test_registered_gpu_operation_replays_changed_inputs(table, monkeypatch, m):
    """Exercise dispatch/capture with existing F.linear, not a new Gluon kernel."""
    rows, _, _ = table
    kernel = FakeKernel(m)
    monkeypatch.setattr(
        tuned,
        "_get_gluon_gemm_kernels",
        lambda: {"gpu_linear": (kernel.supported, kernel.accepts, kernel.gemm)},
    )
    rows[_key(m)] = _row("gluon", "gpu_linear")
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    original_w = w.clone()
    torch.testing.assert_close(tuned.tgemm.mm(x, w), F.linear(x, w))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = tuned.tgemm.mm(x, w)
    for _ in range(5):
        x.copy_(torch.randn_like(x))
        before_x = x.clone()
        expected = F.linear(x, w)
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected)
        assert torch.equal(x, before_x) and torch.equal(w, original_w)
    # Python dispatch runs during eager warm-up and capture, never on replay.
    assert kernel.calls == [m, m]
    assert kernel.support_calls == [(m, N, K, False, torch.bfloat16, torch.bfloat16)]
    assert not tuned._failed_gluon_gemm_kernels


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
