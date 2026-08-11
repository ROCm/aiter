# SPDX-License-Identifier: MIT

"""Unified exact-M BF16 decode policy correctness and architecture tests."""

from __future__ import annotations

import csv
import os
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

pytest.importorskip("flydsl")

import flydsl.expr as fx

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels import gemm_decode as gemm_decode_module
from aiter.ops.flydsl.kernels.gemm_decode import (
    compile_gemm_decode_bf16,
    gemm_decode_bf16,
    gemm_decode_bf16_configured,
    launch_gemm_decode_kernel_name,
)
from aiter.ops.flydsl.kernels.gemm_decode_config import (
    ActivationSource,
    BlockMfmaDecodeConfig,
    ContractionMode,
    OutputRounding,
    ReductionMode,
    SIGNED_INT32_MAX,
    WaveDecodeConfig,
    block_mfma_estimated_live_vgprs,
    block_mfma_lds_bytes,
    gemm_decode_kernel_name,
    iter_gemm_decode_configs,
    parse_gemm_decode_kernel_name,
    validate_block_mfma_grid_i32,
    validate_wave_i32_addressing,
)

ARCH = get_gfx_runtime()
pytestmark = pytest.mark.skipif(
    ARCH not in ("gfx942", "gfx950"),
    reason="unified BF16 decode requires gfx942 or gfx950",
)
ATOL = 0.125
RTOL = 0.01


def test_tuned_decode_bias_contract_rejected():
    import aiter.tuned_gemm as tuned_gemm

    m, n, k = 3, 64, 128
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="does not support fused bias"):
        tuned_gemm.flydsl_decode_gemm(
            a,
            b,
            0,
            bias=bias,
            otype=torch.bfloat16,
            config={
                "kernelName": gemm_decode_kernel_name(
                    "gfx942", m, n, k, _block()
                )
            },
        )


def test_installed_decode_rows_do_not_capture_unmatched_contracts(monkeypatch):
    import aiter.tuned_gemm as tuned_gemm

    tuned_gemm.get_GEMM_A16W16_config_.cache_clear()
    tuned_gemm.get_GEMM_A16W16_config.cache_clear()
    cases = (
        (5, 896, 7168, False),
        (3, 64, 128, False),
        (3, 2304, 1536, True),
    )
    for m, n, k, bias in cases:
        selected = tuned_gemm.get_GEMM_A16W16_config(
            m,
            n,
            k,
            bias,
            "torch.bfloat16",
            "torch.bfloat16",
        )
        assert selected["libtype"] != "flydsl_decode"

    monkeypatch.setattr(tuned_gemm, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(tuned_gemm, "get_cu_num", lambda: 256)
    tuned_gemm.get_GEMM_A16W16_config.cache_clear()
    selected = tuned_gemm.get_GEMM_A16W16_config(
        3,
        2304,
        1536,
        False,
        "torch.bfloat16",
        "torch.bfloat16",
    )
    assert selected["libtype"] != "flydsl_decode"


@contextmanager
def _fresh_compile():
    previous = os.environ.get("FLYDSL_RUNTIME_ENABLE_CACHE")
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("FLYDSL_RUNTIME_ENABLE_CACHE", None)
        else:
            os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = previous


def _wave(m: int, k: int):
    return WaveDecodeConfig(
        m_per_wave=m,
        n_per_wave=1,
        kvec=2,
        reduction=(
            ReductionMode.DPP
            if k % 2 == 0
            else ReductionMode.BPERMUTE
        ),
        contraction=(
            ContractionMode.DOT2_BF16
            if ARCH == "gfx950"
            else ContractionMode.SCALAR_F32
        ),
    )


def _block(
    source=ActivationSource.GLOBAL,
    *,
    persistent_n=False,
    workgroups_per_cu=1,
    b_load_width=4,
):
    return BlockMfmaDecodeConfig(
        waves_per_workgroup=4,
        columns_per_wave=2,
        activation_source=source,
        b_load_width=b_load_width,
        persistent_n=persistent_n,
        workgroups_per_cu=workgroups_per_cu,
    )


def _run(m, n, k, config, seed=0):
    torch.manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    output = torch.full(
        (m, n),
        torch.nan,
        dtype=torch.bfloat16,
        device="cuda",
    )
    gemm_decode_bf16_configured(
        a,
        b,
        output,
        m,
        n,
        k,
        config,
        fx.Stream(None),
        arch=ARCH,
    )
    torch.cuda.synchronize()
    reference = (a.float() @ b.float().T).bfloat16()
    assert torch.isfinite(output).all(), "NaN-poisoned output was not overwritten"
    torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)
    return a, b, output, reference


@pytest.mark.parametrize("m", range(1, 6))
@pytest.mark.parametrize("with_bias", (False, True))
def test_public_api_uses_unified_default_and_preserves_output(m, with_bias):
    n, k = 65, 257
    torch.manual_seed(1701 + m)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    bias = (
        torch.randn((n,), dtype=torch.bfloat16, device="cuda")
        if with_bias
        else None
    )
    output = torch.full(
        (m, n),
        torch.nan,
        dtype=torch.bfloat16,
        device="cuda",
    )
    returned = gemm_decode_bf16(
        a,
        b,
        output,
        m,
        n,
        k,
        bias=bias,
    )
    torch.cuda.synchronize()
    reference = (a.float() @ b.float().T).bfloat16()
    if bias is not None:
        reference.add_(bias)
    assert returned is output
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)


def test_public_api_m_above_5_fails_explicitly():
    m, n, k = 6, 8, 8
    a = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match=r"exact M in \[1, 5\]"):
        gemm_decode_bf16(a, b, output, m, n, k)


@pytest.mark.parametrize("with_bias", (False, True))
def test_stream_none_uses_non_default_current_stream_in_order(with_bias):
    m, n, k = 2, 17, 33
    a = torch.zeros((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.zeros((n, k), dtype=torch.bfloat16, device="cuda")
    bias = (
        torch.full((n,), 2, dtype=torch.bfloat16, device="cuda")
        if with_bias
        else None
    )
    output = torch.full(
        (m, n), torch.nan, dtype=torch.bfloat16, device="cuda"
    )
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    done = torch.cuda.Event()
    with torch.cuda.stream(side):
        torch.cuda._sleep(2_000_000)
        a.fill_(1)
        b.fill_(1)
        returned = gemm_decode_bf16(
            a,
            b,
            output,
            m,
            n,
            k,
            stream=None,
            bias=bias,
        )
        done.record()
    done.synchronize()
    expected = torch.full_like(output, k + (2 if with_bias else 0))
    assert returned is output
    assert torch.equal(output, expected)


@pytest.mark.parametrize("wrapped", (False, True))
def test_bias_supports_positive_raw_stream_pointer(wrapped):
    m, n, k = 2, 17, 32
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda")
    output = torch.full(
        (m, n), torch.nan, dtype=torch.bfloat16, device="cuda"
    )
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    stream = fx.Stream(side.cuda_stream) if wrapped else side.cuda_stream
    gemm_decode_bf16(
        a,
        b,
        output,
        m,
        n,
        k,
        stream=stream,
        bias=bias,
    )
    side.synchronize()
    reference = (a.float() @ b.float().T).bfloat16()
    reference.add_(bias)
    torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)


def test_invalid_bias_stream_rejected_before_compile_and_output_mutation(
    monkeypatch,
):
    m, n, k = 2, 17, 32
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda")
    output = torch.full(
        (m, n), torch.nan, dtype=torch.bfloat16, device="cuda"
    )
    compile_calls = {"count": 0}

    def unexpected_compile(*args, **kwargs):
        compile_calls["count"] += 1
        raise AssertionError("invalid stream reached compilation")

    monkeypatch.setattr(
        gemm_decode_module,
        "compile_gemm_decode_bf16",
        unexpected_compile,
    )
    with pytest.raises(ValueError, match="raw stream 0"):
        gemm_decode_bf16(
            a,
            b,
            output,
            m,
            n,
            k,
            stream=fx.Stream(0),
            bias=bias,
        )
    assert compile_calls["count"] == 0
    assert torch.isnan(output).all()


def test_bias_free_raw_default_stream_remains_supported():
    m, n, k = 2, 17, 32
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    output = torch.full(
        (m, n), torch.nan, dtype=torch.bfloat16, device="cuda"
    )
    gemm_decode_bf16(
        a,
        b,
        output,
        m,
        n,
        k,
        stream=fx.Stream(0),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output,
        (a.float() @ b.float().T).bfloat16(),
        atol=ATOL,
        rtol=RTOL,
    )


def test_explicit_pytorch_stream_is_asynchronous_and_ordered():
    m, n, k = 3, 65, 257
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda")
    output = torch.full(
        (m, n), torch.nan, dtype=torch.bfloat16, device="cuda"
    )
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    gemm_decode_bf16(
        a,
        b,
        output,
        m,
        n,
        k,
        stream=side,
        bias=bias,
    )
    done = torch.cuda.Event()
    done.record(side)
    done.synchronize()
    reference = (a.float() @ b.float().T).bfloat16()
    reference.add_(bias)
    torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)


def test_public_api_stream_and_graph_with_bias():
    m, n, k = 3, 65, 257
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    reference = (a.float() @ b.float().T).bfloat16()
    reference.add_(bias)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        returned = gemm_decode_bf16(
            a,
            b,
            output,
            m,
            n,
            k,
            fx.Stream(side),
            bias=bias,
        )
    side.synchronize()
    assert returned is output

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        gemm_decode_bf16(
            a,
            b,
            output,
            m,
            n,
            k,
            fx.Stream(side),
            bias=bias,
        )
    side.synchronize()
    output.fill_(torch.nan)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        graph.replay()
    side.synchronize()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("m", range(1, 6))
@pytest.mark.parametrize("policy", ("wave", "block"))
def test_exact_m_1_through_5_with_odd_n_k(m, policy):
    n, k = 65, 257
    config = _wave(m, k) if policy == "wave" else _block()
    _run(m, n, k, config)
    if isinstance(config, WaveDecodeConfig):
        assert config.m_per_wave == m


@pytest.mark.parametrize("m", (3, 5))
def test_wave_exact_divisor_mp1_has_no_m_tail(m):
    n, k = 65, 257
    config = _wave(m, k)
    config = WaveDecodeConfig(
        m_per_wave=1,
        n_per_wave=config.n_per_wave,
        kvec=config.kvec,
        reduction=config.reduction,
        contraction=config.contraction,
    )
    _run(m, n, k, config)


@pytest.mark.parametrize("m", (3, 5))
def test_block_mfma_emits_exact_row_loops(m):
    config = _block()
    launcher = compile_gemm_decode_bf16(m, 65, 257, config, arch=ARCH)
    a = torch.randn((m, 257), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((65, 257), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((m, 65), dtype=torch.bfloat16, device="cuda")
    with _fresh_compile():
        launcher(a, b, c)
    torch.cuda.synchronize()
    source_ir = launcher._last_compiled[1].source_ir
    assert "mfma" in source_ir
    assert source_ir.count("rocdl.mfma") == 4 * m
    assert "persistent" not in source_ir
    assert "scf.while" not in source_ir
    assert launcher.persistent_turns == 1


def test_deterministic_output():
    m, n, k = 5, 65, 257
    a, b, first, reference = _run(m, n, k, _block(), seed=11)
    first = first.clone()
    for _ in range(2):
        output = torch.full_like(first, torch.nan)
        gemm_decode_bf16_configured(
            a,
            b,
            output,
            m,
            n,
            k,
            _block(),
            arch=ARCH,
        )
        torch.cuda.synchronize()
        assert torch.equal(output, first)
        torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)


def test_configured_launch_rejects_explicit_arch_mismatch():
    m, n, k = 3, 65, 257
    other_arch = "gfx950" if ARCH == "gfx942" else "gfx942"
    a = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="does not match runtime architecture"):
        gemm_decode_bf16_configured(
            a,
            b,
            c,
            m,
            n,
            k,
            _wave(m, k),
            arch=other_arch,
        )


@pytest.mark.parametrize("policy", ("wave", "block"))
def test_graph_replay_on_non_default_stream(policy):
    m, n, k = 3, 65, 257
    config = _wave(m, k) if policy == "wave" else _block()
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    c = torch.full((m, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    reference = (a.float() @ b.float().T).bfloat16()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    gemm_decode_bf16_configured(
        a, b, c, m, n, k, config, fx.Stream(side), arch=ARCH
    )
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        gemm_decode_bf16_configured(
            a, b, c, m, n, k, config, fx.Stream(side), arch=ARCH
        )
    side.synchronize()
    c.fill_(torch.nan)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        graph.replay()
    side.synchronize()
    assert torch.isfinite(c).all()
    torch.testing.assert_close(c, reference, atol=ATOL, rtol=RTOL)


def test_registry_is_legal_deduplicated_and_roundtrips():
    for arch in ("gfx942", "gfx950"):
        for m in range(1, 6):
            configs = list(iter_gemm_decode_configs(m, 65, 7168, arch))
            assert configs
            assert len(configs) == len(set(configs))
            assert {
                config.reduction
                for config in configs
                if isinstance(config, WaveDecodeConfig)
            } == set(ReductionMode)
            assert {
                config.m_per_wave
                for config in configs
                if isinstance(config, WaveDecodeConfig)
            } == {mp for mp in range(1, m + 1) if m % mp == 0}
            for config in configs:
                config.validate(m=m, n=65, k=7168, arch=arch)
                name = gemm_decode_kernel_name(arch, m, 65, 7168, config)
                assert parse_gemm_decode_kernel_name(name) == (
                    arch,
                    m,
                    65,
                    7168,
                    config,
                )
                if isinstance(config, WaveDecodeConfig):
                    assert m % config.m_per_wave == 0
                assert config.output_rounding == OutputRounding.RNE


def test_registry_targets_persistent_tuning_without_shape_explosion():
    target_configs = list(iter_gemm_decode_configs(3, 2304, 1536, "gfx942"))
    persistent = [
        config
        for config in target_configs
        if isinstance(config, BlockMfmaDecodeConfig) and config.persistent_n
    ]
    assert persistent
    assert len(persistent) == len(set(persistent))
    assert any(
        config.waves_per_workgroup == 8
        and config.columns_per_wave == 1
        and config.b_load_width == 8
        and config.k_unroll == 2
        and config.workgroups_per_cu == 1
        and config.waves_per_eu == 0
        for config in persistent
    )
    unrelated = iter_gemm_decode_configs(3, 2305, 1536, "gfx942")
    assert not any(
        isinstance(config, BlockMfmaDecodeConfig) and config.persistent_n
        for config in unrelated
    )


def test_block_mfma_persistence_legality_and_stale_name_rejection():
    with pytest.raises(ValueError, match="full-A LDS"):
        _block(persistent_n=True).validate(
            m=3, n=2304, k=1536, arch="gfx942"
        )
    with pytest.raises(ValueError, match="only applies"):
        _block(workgroups_per_cu=2).validate(
            m=3, n=2304, k=1536, arch="gfx942"
        )
    with pytest.raises(ValueError, match="full A LDS"):
        _block(
            ActivationSource.FULL_LDS,
            persistent_n=True,
            workgroups_per_cu=4,
        ).validate(m=5, n=20480, k=7168, arch="gfx942")
    legal = _block(
        ActivationSource.FULL_LDS,
        persistent_n=True,
        workgroups_per_cu=4,
        b_load_width=8,
    )
    legal.validate(m=3, n=2304, k=1536, arch="gfx942")
    name = gemm_decode_kernel_name("gfx942", 3, 2304, 1536, legal)
    assert "_pblock2_" in name
    assert "_pn1_g4_" in name
    assert parse_gemm_decode_kernel_name(name)[-1] == legal
    stale = name.replace("_pblock2_", "_pblock_").replace("_pn1_g4_", "_")
    with pytest.raises(ValueError, match="invalid BlockMFMA"):
        parse_gemm_decode_kernel_name(stale)


def test_block_mfma_signed_i32_address_boundaries():
    config = _block()
    # 32767 * 32769 == floor(INT32_MAX / sizeof(bf16)).
    config.validate(m=1, n=32767, k=32769, arch="gfx942")
    with pytest.raises(ValueError, match="B byte extent"):
        config.validate(m=1, n=32768, k=32769, arch="gfx942")

    max_a_k = SIGNED_INT32_MAX // (5 * 2)
    config.validate(m=5, n=1, k=max_a_k, arch="gfx942")
    with pytest.raises(ValueError, match="A byte extent"):
        config.validate(m=5, n=1, k=max_a_k + 1, arch="gfx942")

    max_c_n = SIGNED_INT32_MAX // (5 * 2)
    config.validate(m=5, n=max_c_n, k=1, arch="gfx942")
    with pytest.raises(ValueError, match="C byte extent"):
        config.validate(m=5, n=max_c_n + 1, k=1, arch="gfx942")

    max_stride_k = SIGNED_INT32_MAX // 2
    config.validate(m=1, n=1, k=max_stride_k, arch="gfx942")
    with pytest.raises(ValueError, match="row stride bytes"):
        config.validate(m=1, n=1, k=max_stride_k + 1, arch="gfx942")


def _boundary_wave(k: int) -> WaveDecodeConfig:
    return WaveDecodeConfig(
        m_per_wave=1,
        n_per_wave=1,
        kvec=2,
        reduction=(
            ReductionMode.DPP if k % 2 == 0 else ReductionMode.BPERMUTE
        ),
        contraction=ContractionMode.SCALAR_F32,
    )


def test_wave_signed_i32_address_boundaries():
    _boundary_wave(32769).validate(
        m=1, n=32767, k=32769, arch="gfx942"
    )
    with pytest.raises(ValueError, match="B byte extent"):
        _boundary_wave(32769).validate(
            m=1, n=32768, k=32769, arch="gfx942"
        )

    max_stride_k = SIGNED_INT32_MAX // 2
    _boundary_wave(max_stride_k).validate(
        m=1, n=1, k=max_stride_k, arch="gfx942"
    )
    with pytest.raises(ValueError, match="row stride bytes"):
        _boundary_wave(max_stride_k + 1).validate(
            m=1, n=1, k=max_stride_k + 1, arch="gfx942"
        )

    max_a_k = SIGNED_INT32_MAX // (5 * 2)
    _boundary_wave(max_a_k).validate(
        m=5, n=1, k=max_a_k, arch="gfx942"
    )
    with pytest.raises(ValueError, match="A byte extent"):
        _boundary_wave(max_a_k + 1).validate(
            m=5, n=1, k=max_a_k + 1, arch="gfx942"
        )

    max_c_n = SIGNED_INT32_MAX // (5 * 2)
    _boundary_wave(1).validate(m=5, n=max_c_n, k=1, arch="gfx942")
    with pytest.raises(ValueError, match="C byte extent"):
        _boundary_wave(1).validate(
            m=5, n=max_c_n + 1, k=1, arch="gfx942"
        )

    with pytest.raises(ValueError, match="A element extent"):
        _boundary_wave(SIGNED_INT32_MAX // 5 + 1).validate(
            m=5,
            n=1,
            k=SIGNED_INT32_MAX // 5 + 1,
            arch="gfx942",
        )
    with pytest.raises(ValueError, match="B element extent"):
        _boundary_wave(3).validate(
            m=1,
            n=SIGNED_INT32_MAX // 2,
            k=3,
            arch="gfx942",
        )
    with pytest.raises(ValueError, match="C element extent"):
        _boundary_wave(1).validate(
            m=5,
            n=SIGNED_INT32_MAX // 5 + 1,
            k=1,
            arch="gfx942",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("m_per_wave", 0, "m_per_wave"),
        ("n_per_wave", 3, "n_per_wave"),
        ("kvec", 3, "kvec"),
        ("prefetch_depth", 3, "prefetch_depth"),
        ("waves_per_eu", 3, "waves_per_eu"),
    ),
)
def test_wave_malformed_axes_rejected_before_addressing(field, value, message):
    values = {
        "m_per_wave": 1,
        "n_per_wave": 1,
        "kvec": 2,
        "prefetch_depth": 0,
        "waves_per_eu": 4,
        "reduction": ReductionMode.DPP,
        "contraction": ContractionMode.SCALAR_F32,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        WaveDecodeConfig(**values).validate(
            m=1,
            n=1,
            k=SIGNED_INT32_MAX,
            arch="gfx942",
        )


def test_wave_runtime_and_aot_reject_before_lowering_or_output_mutation(
    monkeypatch,
):
    unsafe_k = SIGNED_INT32_MAX // 2 + 1
    config = _boundary_wave(unsafe_k)
    lowered = {"count": 0}

    def unexpected_lowering(*args, **kwargs):
        lowered["count"] += 1
        raise AssertionError("unsafe Wave shape reached kernel construction")

    monkeypatch.setattr(
        gemm_decode_module,
        "compile_gemm_decode_wave_bf16",
        unexpected_lowering,
    )
    with pytest.raises(ValueError, match="row stride bytes"):
        compile_gemm_decode_bf16(
            1,
            1,
            unsafe_k,
            config,
            arch="gfx942",
        )

    a = torch.ones((1, 2), dtype=torch.bfloat16, device="cuda")
    b = torch.ones((1, 2), dtype=torch.bfloat16, device="cuda")
    output = torch.full(
        (1, 1), torch.nan, dtype=torch.bfloat16, device="cuda"
    )
    monkeypatch.setattr(
        gemm_decode_module,
        "validate_gemm_decode_tensors",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="row stride bytes"):
        gemm_decode_bf16_configured(
            a,
            b,
            output,
            1,
            1,
            unsafe_k,
            config,
            arch=ARCH,
        )
    assert lowered["count"] == 0
    assert torch.isnan(output).all()


def test_all_installed_decode_rows_pass_address_validation():
    csv_path = (
        Path(__file__).resolve().parents[1]
        / "aiter/configs/bf16_tuned_gemm.csv"
    )
    with csv_path.open(newline="") as source:
        rows = [
            row
            for row in csv.DictReader(source)
            if row["libtype"] == "flydsl_decode"
        ]
    assert len(rows) == 52
    for row in rows:
        arch, m, n, k, config = parse_gemm_decode_kernel_name(
            row["kernelName"]
        )
        config.validate(m=m, n=n, k=k, arch=arch)
        if isinstance(config, WaveDecodeConfig):
            validate_wave_i32_addressing(m, n, k, config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("waves_per_workgroup", 0, "waves_per_workgroup"),
        ("waves_per_workgroup", 3, "waves_per_workgroup"),
        ("columns_per_wave", 0, "columns_per_wave"),
        ("columns_per_wave", 3, "columns_per_wave"),
        ("b_load_width", 0, "b_load_width"),
        ("k_unroll", 0, "k_unroll"),
        ("prefetch_stages", 0, "prefetch_stages"),
        ("workgroups_per_cu", 0, "workgroups_per_cu"),
        ("waves_per_eu", 3, "waves_per_eu"),
    ),
)
def test_block_mfma_invalid_axes_raise_intentional_value_error(
    field, value, message
):
    values = {
        "waves_per_workgroup": 8,
        "columns_per_wave": 1,
        "b_load_width": 4,
        "k_unroll": 1,
        "prefetch_stages": 1,
        "workgroups_per_cu": 1,
        "waves_per_eu": 0,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        BlockMfmaDecodeConfig(**values).validate(
            m=3,
            n=2304,
            k=1536,
            arch="gfx942",
        )


def test_persistent_grid_cap_i32_boundaries_and_semantics():
    config = _block(
        ActivationSource.FULL_LDS,
        persistent_n=True,
        workgroups_per_cu=4,
    )
    # The multiplier caps grid size; it does not assert four resident groups.
    config.validate(m=3, n=2304, k=1536, arch="gfx942")
    max_cus = SIGNED_INT32_MAX // config.workgroups_per_cu
    validate_block_mfma_grid_i32(2304, config, num_cus=max_cus)
    with pytest.raises(ValueError, match="positive integer"):
        validate_block_mfma_grid_i32(2304, config, num_cus=0)
    with pytest.raises(ValueError, match="grid-cap workgroups"):
        validate_block_mfma_grid_i32(2304, config, num_cus=max_cus + 1)


@pytest.mark.skipif(ARCH != "gfx942", reason="gfx942 persistence coverage")
@pytest.mark.parametrize(("m", "n"), ((3, 2304), (4, 2305)))
def test_persistent_block_exact_m_turns_one_and_n_tail(m, n):
    config = BlockMfmaDecodeConfig(
        waves_per_workgroup=16,
        columns_per_wave=1,
        activation_source=ActivationSource.FULL_LDS,
        b_load_width=8,
        k_unroll=2,
        persistent_n=True,
        workgroups_per_cu=1,
    )
    _run(m, n, 1536, config)
    launcher = compile_gemm_decode_bf16(
        m,
        n,
        1536,
        config,
        arch=ARCH,
        num_cus=torch.cuda.get_device_properties(0).multi_processor_count,
    )
    assert launcher.persistent_turns == 1


@pytest.mark.skipif(ARCH != "gfx942", reason="gfx942 persistence coverage")
def test_persistent_block_multiple_turns_stages_full_a_once():
    m, n, k = 3, 5001, 257
    config = BlockMfmaDecodeConfig(
        waves_per_workgroup=4,
        columns_per_wave=1,
        activation_source=ActivationSource.FULL_LDS,
        b_load_width=8,
        persistent_n=True,
        workgroups_per_cu=1,
    )
    a, b, output, reference = _run(m, n, k, config)
    launcher = compile_gemm_decode_bf16(
        m,
        n,
        k,
        config,
        arch=ARCH,
        num_cus=torch.cuda.get_device_properties(0).multi_processor_count,
    )
    with _fresh_compile():
        launcher(a, b, output)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)
    source_ir = launcher._last_compiled[1].source_ir
    assert launcher.persistent_turns > 1
    assert source_ir.count("gpu.barrier") == 1
    assert "scf.while" in source_ir
    assert source_ir.index("gpu.barrier") < source_ir.index("scf.while")


def test_architecture_feature_rejections_and_lds_limits():
    with pytest.raises(ValueError, match="dot2"):
        WaveDecodeConfig(
            m_per_wave=1,
            contraction=ContractionMode.DOT2_BF16,
        ).validate(m=1, n=64, k=128, arch="gfx942")
    with pytest.raises(ValueError, match="stochastic"):
        WaveDecodeConfig(
            m_per_wave=1,
            output_rounding=OutputRounding.STOCHASTIC,
        ).validate(m=1, n=64, k=128, arch="gfx942")
    BlockMfmaDecodeConfig(b_load_width=8).validate(
        m=3, n=64, k=128, arch="gfx942"
    )
    with pytest.raises(ValueError, match="two-stage"):
        BlockMfmaDecodeConfig(prefetch_stages=2).validate(
            m=3, n=64, k=128, arch="gfx942"
        )
    full_lds = BlockMfmaDecodeConfig(
        activation_source=ActivationSource.FULL_LDS
    )
    assert block_mfma_lds_bytes(5, 7168) > 64 * 1024
    with pytest.raises(ValueError, match="65536-byte"):
        full_lds.validate(m=5, n=20480, k=7168, arch="gfx942")
    full_lds.validate(m=5, n=20480, k=7168, arch="gfx950")
    BlockMfmaDecodeConfig(
        activation_source=ActivationSource.GLOBAL
    ).validate(m=5, n=20480, k=7168, arch="gfx942")
    unsafe_prefetch = BlockMfmaDecodeConfig(
        columns_per_wave=4,
        b_load_width=8,
        k_unroll=2,
        prefetch_stages=2,
    )
    assert block_mfma_estimated_live_vgprs(3, unsafe_prefetch) > 192
    with pytest.raises(ValueError, match="live prefetch state"):
        unsafe_prefetch.validate(m=3, n=64, k=7168, arch="gfx950")


def test_gfx950_only_configs_are_static_legal():
    WaveDecodeConfig(
        m_per_wave=3,
        kvec=8,
        prefetch_depth=2,
        contraction=ContractionMode.DOT2_BF16,
        output_rounding=OutputRounding.STOCHASTIC,
    ).validate(m=3, n=64, k=7168, arch="gfx950")
    BlockMfmaDecodeConfig(
        waves_per_workgroup=8,
        columns_per_wave=2,
        b_load_width=8,
        k_unroll=2,
        prefetch_stages=2,
    ).validate(m=3, n=65, k=7168, arch="gfx950")


@pytest.mark.parametrize(
    "policy",
    ("wave", "block", "block_boundary", "persistent_block"),
)
def test_gfx950_cross_compile_and_final_isa(policy, tmp_path, monkeypatch):
    from aiter.aot.flydsl.gemm import compile_one_config

    if policy == "wave":
        m, n, k = 3, 2304, 1536
        config = WaveDecodeConfig(
            m_per_wave=m,
            n_per_wave=4,
            kvec=8,
            prefetch_depth=2,
            waves_per_eu=2,
            contraction=ContractionMode.DOT2_BF16,
        )
    elif policy == "block":
        m, n, k = 3, 6288, 7168
        config = BlockMfmaDecodeConfig(
            waves_per_workgroup=8,
            columns_per_wave=2,
            b_load_width=8,
            k_unroll=2,
            prefetch_stages=2,
            waves_per_eu=2,
        )
    elif policy == "block_boundary":
        m, n, k = 4, 6288, 7168
        config = BlockMfmaDecodeConfig(
            waves_per_workgroup=8,
            columns_per_wave=4,
            b_load_width=8,
            k_unroll=2,
            prefetch_stages=1,
            waves_per_eu=2,
        )
        assert block_mfma_estimated_live_vgprs(m, config) == 176
    else:
        m, n, k = 3, 2304, 1536
        config = BlockMfmaDecodeConfig(
            waves_per_workgroup=8,
            columns_per_wave=1,
            activation_source=ActivationSource.FULL_LDS,
            b_load_width=8,
            k_unroll=2,
            persistent_n=True,
            workgroups_per_cu=1,
        )
    name = gemm_decode_kernel_name("gfx950", m, n, k, config)
    dump_dir = tmp_path / policy
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    result = compile_one_config(
        name,
        "decode",
        m,
        n,
        k,
        cu_num=256,
        arch="gfx950",
        config=config,
    )
    assert result["compile_time"] is not None
    isa_paths = list(dump_dir.rglob("*final_isa.s"))
    assert len(isa_paths) == 1
    isa = isa_paths[0].read_text()
    assert "scratch_load" not in isa
    assert "scratch_store" not in isa
    if policy == "wave":
        assert "v_dot2_f32_bf16" in isa
    else:
        assert "buffer_load_dwordx4" in isa
        assert "v_mfma_f32_4x4x4_16b_bf16" in isa


@pytest.mark.skipif(ARCH != "gfx942", reason="gfx942 ISA check")
def test_gfx942_scalar_wave_final_isa_is_legal(tmp_path, monkeypatch):
    m, n, k = 3, 67, 259
    config = WaveDecodeConfig(
        m_per_wave=m,
        n_per_wave=1,
        kvec=2,
        reduction=ReductionMode.BPERMUTE,
        contraction=ContractionMode.SCALAR_F32,
    )
    dump_dir = tmp_path / "gfx942_scalar"
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    _run(m, n, k, config)
    isa_paths = list(dump_dir.rglob("*final_isa.s"))
    assert len(isa_paths) == 1
    isa = isa_paths[0].read_text()
    assert "v_fma_f32" in isa
    assert "v_dot2_f32_bf16" not in isa
    assert "scratch_load" not in isa
    assert "scratch_store" not in isa


def test_kernel_name_runtime_launch_and_tuned_gemm_bridge():
    from aiter.tuned_gemm import flydsl_decode_gemm

    m, n, k = 5, 33, 127
    config = _wave(m, k)
    name = gemm_decode_kernel_name(ARCH, m, n, k, config)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    c = torch.full((m, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    launch_gemm_decode_kernel_name(a, b, c, name)
    bridged = flydsl_decode_gemm(
        a,
        b,
        0,
        otype=torch.bfloat16,
        config={"kernelName": name},
    )
    torch.cuda.synchronize()
    reference = (a.float() @ b.float().T).bfloat16()
    torch.testing.assert_close(c, reference, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(bridged, reference, atol=ATOL, rtol=RTOL)


def test_synthetic_tuned_csv_dispatches_exact_decode_kernel(tmp_path, monkeypatch):
    import aiter.tuned_gemm as tuned_gemm
    from aiter.jit.core import AITER_CONFIGS

    m, n, k = 3, 65, 257
    cu_num = torch.cuda.get_device_properties(0).multi_processor_count
    config = _wave(m, k)
    name = gemm_decode_kernel_name(ARCH, m, n, k, config)
    csv_path = tmp_path / "bf16_tuned.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,"
        "libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw\n"
        f"{ARCH},{cu_num},{m},{n},{k},False,torch.bfloat16,torch.bfloat16,"
        f"False,False,flydsl_decode,0,0,1.0,{name},0.0,0.0,0.0\n"
    )
    try:
        with monkeypatch.context() as isolated:
            isolated.setenv("AITER_CONFIG_GEMM_BF16", str(csv_path))
            AITER_CONFIGS.get_config_file.cache_clear()
            tuned_gemm.get_GEMM_A16W16_config_.cache_clear()
            tuned_gemm.get_GEMM_A16W16_config.cache_clear()
            selected = tuned_gemm.get_GEMM_A16W16_config(
                m,
                n,
                k,
                False,
                str(torch.bfloat16),
                str(torch.bfloat16),
            )
            assert selected["libtype"] == "flydsl_decode"
            assert selected["kernelName"] == name
            a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
            output = tuned_gemm.gemm_a16w16(a, b)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output,
                (a.float() @ b.float().T).bfloat16(),
                atol=ATOL,
                rtol=RTOL,
            )
    finally:
        AITER_CONFIGS.get_config_file.cache_clear()
        tuned_gemm.get_GEMM_A16W16_config_.cache_clear()
        tuned_gemm.get_GEMM_A16W16_config.cache_clear()


def test_name_launch_reloads_from_deployed_cache(tmp_path, monkeypatch):
    from aiter.ops.flydsl.kernels.gemm_decode_wave import (
        compile_gemm_decode_wave_bf16,
    )

    m, n, k = 3, 65, 257
    name = gemm_decode_kernel_name(ARCH, m, n, k, _wave(m, k))
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    cache_dir = tmp_path / "cache"
    try:
        with monkeypatch.context() as isolated:
            isolated.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(cache_dir))
            isolated.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
            isolated.delenv("FLYDSL_RUNTIME_RUN_ONLY", raising=False)
            isolated.delenv("FLYDSL_DUMP_IR", raising=False)
            compile_gemm_decode_wave_bf16.cache_clear()
            first = torch.full(
                (m, n), torch.nan, dtype=torch.bfloat16, device="cuda"
            )
            launch_gemm_decode_kernel_name(a, b, first, name)
            torch.cuda.synchronize()
            cache_files = [path for path in cache_dir.rglob("*") if path.is_file()]
            assert cache_files, "first launch did not populate the disk cache"

            isolated.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
            isolated.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
            compile_gemm_decode_wave_bf16.cache_clear()
            second = torch.full(
                (m, n), torch.nan, dtype=torch.bfloat16, device="cuda"
            )
            launch_gemm_decode_kernel_name(a, b, second, name)
            torch.cuda.synchronize()
            assert torch.equal(first, second)
            torch.testing.assert_close(
                second,
                (a.float() @ b.float().T).bfloat16(),
                atol=ATOL,
                rtol=RTOL,
            )
    finally:
        compile_gemm_decode_wave_bf16.cache_clear()


def test_run_only_missing_unified_artifact_raises(tmp_path, monkeypatch):
    from aiter.ops.flydsl.kernels.gemm_decode_wave import (
        compile_gemm_decode_wave_bf16,
    )

    m, n, k = 3, 65, 257
    name = gemm_decode_kernel_name(ARCH, m, n, k, _wave(m, k))
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    with monkeypatch.context() as isolated:
        isolated.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "empty-cache"))
        isolated.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        isolated.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
        compile_gemm_decode_wave_bf16.cache_clear()
        with pytest.raises(RuntimeError, match=r"(?i)(run.only.*cache|cache.*run.only)"):
            launch_gemm_decode_kernel_name(a, b, output, name)
    compile_gemm_decode_wave_bf16.cache_clear()


def test_aot_csv_parser_recognizes_decode_rows(tmp_path):
    from aiter.aot.flydsl.gemm import parse_csv

    m, n, k = 3, 65, 257
    cu_num = 304 if ARCH == "gfx942" else 256
    config = _block()
    name = gemm_decode_kernel_name(ARCH, m, n, k, config)
    csv_path = tmp_path / "decode.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,bias,libtype,kernelName\n"
        f"{ARCH},{cu_num},{m},{n},{k},False,flydsl_decode,{name}\n"
    )
    jobs = parse_csv(str(csv_path))
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "decode"
    assert jobs[0]["arch"] == ARCH
    assert jobs[0]["config"] == config


@pytest.mark.parametrize(
    ("csv_arch", "cu_num"),
    (("gfx950", 304), ("gfx942", 256), ("gfx942", 999)),
)
def test_aot_csv_rejects_decode_arch_metadata_mismatch(
    tmp_path, csv_arch, cu_num
):
    from aiter.aot.flydsl.gemm import parse_csv

    m, n, k = 3, 65, 257
    name = gemm_decode_kernel_name("gfx942", m, n, k, _block())
    csv_path = tmp_path / "decode_mismatch.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,bias,libtype,kernelName\n"
        f"{csv_arch},{cu_num},{m},{n},{k},False,flydsl_decode,{name}\n"
    )
    with pytest.raises(ValueError, match="architecture metadata mismatch|recognized"):
        parse_csv(str(csv_path))


def test_aot_rejects_unsafe_block_address_name_before_compile(tmp_path):
    from aiter.aot.flydsl.gemm import parse_csv

    safe = gemm_decode_kernel_name("gfx942", 1, 32767, 32769, _block())
    unsafe = safe.replace("_n32767_", "_n32768_")
    csv_path = tmp_path / "unsafe_decode.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,bias,libtype,kernelName\n"
        f"gfx942,304,1,32768,32769,False,flydsl_decode,{unsafe}\n"
    )
    with pytest.raises(ValueError, match="B byte extent"):
        parse_csv(str(csv_path))


@pytest.mark.skipif(ARCH != "gfx942", reason="gfx942 persistent cache coverage")
def test_persistent_block_aot_cache_hit_and_isolation(tmp_path, monkeypatch):
    from aiter.aot.flydsl.gemm import compile_one_config, parse_csv
    from aiter.ops.flydsl.kernels.gemm_decode_block_mfma import (
        compile_gemm_decode_block_mfma_bf16,
    )

    m, n, k = 3, 257, 128
    configs = [
        BlockMfmaDecodeConfig(
            waves_per_workgroup=4,
            columns_per_wave=1,
            activation_source=ActivationSource.FULL_LDS,
            b_load_width=8,
            k_unroll=2,
            persistent_n=True,
            workgroups_per_cu=grid_cap,
        )
        for grid_cap in (1, 2)
    ]
    names = [
        gemm_decode_kernel_name("gfx942", m, n, k, config)
        for config in configs
    ]
    csv_path = tmp_path / "persistent.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,bias,libtype,kernelName\n"
        + "".join(
            f"gfx942,304,{m},{n},{k},False,flydsl_decode,{name}\n"
            for name in names
        )
    )
    jobs = parse_csv(str(csv_path))
    assert [job["config"] for job in jobs] == configs

    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.delenv("FLYDSL_RUNTIME_RUN_ONLY", raising=False)
    compile_gemm_decode_block_mfma_bf16.cache_clear()
    seen_files = set()
    for job in jobs:
        result = compile_one_config(**job)
        assert result["compile_time"] is not None
        current_files = {
            path.relative_to(cache_dir)
            for path in cache_dir.rglob("*")
            if path.is_file()
        }
        assert current_files - seen_files
        seen_files = current_files

    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    reference = (a.float() @ b.float().T).bfloat16()
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
    for name in names:
        compile_gemm_decode_block_mfma_bf16.cache_clear()
        output = torch.full(
            (m, n),
            torch.nan,
            dtype=torch.bfloat16,
            device="cuda",
        )
        launch_gemm_decode_kernel_name(a, b, output, name)
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(ARCH != "gfx942", reason="gfx942 persistent cache coverage")
def test_persistent_same_name_different_cu_grid_cache_isolation(
    tmp_path, monkeypatch
):
    from aiter.ops.flydsl.kernels.gemm_decode_block_mfma import (
        compile_gemm_decode_block_mfma_bf16,
    )

    m, n, k = 3, 5001, 128
    config = BlockMfmaDecodeConfig(
        waves_per_workgroup=4,
        columns_per_wave=1,
        activation_source=ActivationSource.FULL_LDS,
        b_load_width=8,
        k_unroll=2,
        persistent_n=True,
        workgroups_per_cu=1,
    )
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.delenv("FLYDSL_RUNTIME_RUN_ONLY", raising=False)
    compile_gemm_decode_block_mfma_bf16.cache_clear()
    launchers = [
        compile_gemm_decode_block_mfma_bf16(
            m,
            n,
            k,
            config,
            "gfx942",
            num_cus=num_cus,
        )
        for num_cus in (1, 2)
    ]
    assert launchers[0].kernel_name == launchers[1].kernel_name
    assert launchers[0].grid_workgroups == 1
    assert launchers[1].grid_workgroups == 2
    assert launchers[0].persistent_turns != launchers[1].persistent_turns

    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    reference = (a.float() @ b.float().T).bfloat16()
    seen_files = set()
    for launcher in launchers:
        output = torch.full(
            (m, n), torch.nan, dtype=torch.bfloat16, device="cuda"
        )
        launcher(a, b, output)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)
        current_files = {
            path.relative_to(cache_dir)
            for path in cache_dir.rglob("*")
            if path.is_file()
        }
        assert current_files - seen_files
        seen_files = current_files

    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
    compile_gemm_decode_block_mfma_bf16.cache_clear()
    for num_cus, expected_grid in ((1, 1), (2, 2)):
        launcher = compile_gemm_decode_block_mfma_bf16(
            m,
            n,
            k,
            config,
            "gfx942",
            num_cus=num_cus,
        )
        assert launcher.grid_workgroups == expected_grid
        output = torch.full(
            (m, n), torch.nan, dtype=torch.bfloat16, device="cuda"
        )
        launcher(a, b, output)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(ARCH != "gfx942", reason="runtime architecture check")
def test_persistent_wrong_architecture_name_rejected_before_launch():
    config = BlockMfmaDecodeConfig(
        activation_source=ActivationSource.FULL_LDS,
        persistent_n=True,
    )
    name = gemm_decode_kernel_name("gfx950", 3, 257, 128, config)
    a = torch.empty((3, 128), dtype=torch.bfloat16, device="cuda")
    b = torch.empty((257, 128), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((3, 257), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(
        ValueError,
        match="targets gfx950, but the runtime device is gfx942",
    ):
        launch_gemm_decode_kernel_name(a, b, c, name)


@pytest.mark.skipif(ARCH != "gfx942", reason="runtime architecture check")
def test_wave_wrong_architecture_name_reports_config_legality():
    config = WaveDecodeConfig(
        m_per_wave=1,
        contraction=ContractionMode.SCALAR_F32,
    )
    name = gemm_decode_kernel_name("gfx942", 1, 256, 128, config).replace(
        "_agfx942_", "_agfx950_", 1
    )
    a = torch.empty((1, 128), dtype=torch.bfloat16, device="cuda")
    b = torch.empty((256, 128), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((1, 256), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(
        ValueError,
        match="gfx950 wave policy uses native dot2 BF16 contraction",
    ):
        launch_gemm_decode_kernel_name(a, b, c, name)
