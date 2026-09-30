from types import SimpleNamespace

import pytest
import torch

from aiter import tuned_gemm


class DSLCompileError(Exception):
    pass


DSLCompileError.__module__ = "flydsl.compiler.diagnostics"


def test_flydsl_compile_error_falls_back_once(monkeypatch):
    calls = {"flydsl": 0, "fallback": 0}
    expected = torch.empty(2, 4)

    def fail_compile(*args, **kwargs):
        calls["flydsl"] += 1
        raise DSLCompileError("lld invocation failed")

    kernels = SimpleNamespace(
        get_flydsl_hgemm_kernel_params=lambda _name: {
            "block_m": 16,
            "block_n": 16,
            "block_k": 16,
            "split_k": 1,
            "m_waves": 1,
            "n_waves": 1,
            "k_waves": 1,
            "stages": 1,
            "group_m": 1,
            "use_half_tile_interleaved": False,
        },
        flydsl_hgemm=fail_compile,
    )

    def fallback(*args, **kwargs):
        calls["fallback"] += 1
        return expected

    monkeypatch.setattr(tuned_gemm, "_get_flydsl_gemm_kernels", lambda: kernels)
    monkeypatch.setattr(tuned_gemm, "torch_gemm", fallback)
    tuned_gemm._flydsl_compile_failures.clear()

    config = {"kernelName": "broken-kernel"}
    inp = torch.empty(2, 3)
    weights = torch.empty(4, 3)
    assert tuned_gemm.flydsl_gemm(inp, weights, 0, config=config) is expected
    assert tuned_gemm.flydsl_gemm(inp, weights, 0, config=config) is expected
    assert calls == {"flydsl": 1, "fallback": 2}


def test_flydsl_runtime_error_is_not_hidden(monkeypatch):
    kernels = SimpleNamespace(
        get_flydsl_hgemm_kernel_params=lambda _name: {
            "block_m": 16,
            "block_n": 16,
            "block_k": 16,
            "split_k": 1,
            "m_waves": 1,
            "n_waves": 1,
            "k_waves": 1,
            "stages": 1,
            "group_m": 1,
            "use_half_tile_interleaved": False,
        },
        flydsl_hgemm=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("launch failed")
        ),
    )
    monkeypatch.setattr(tuned_gemm, "_get_flydsl_gemm_kernels", lambda: kernels)
    tuned_gemm._flydsl_compile_failures.clear()

    with pytest.raises(RuntimeError, match="launch failed"):
        tuned_gemm.flydsl_gemm(
            torch.empty(2, 3),
            torch.empty(4, 3),
            0,
            config={"kernelName": "runtime-failure"},
        )
