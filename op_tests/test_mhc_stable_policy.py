# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Policy and numerical regressions for mHC pre across runtime batch sizes."""

from unittest.mock import Mock

import pytest
import torch

from aiter.ops import mhc

STABLE_ARCH = "gfx950"
STABLE_CU = 256
HIDDEN_SIZE = 4096
HC_HIDDEN_SIZE = 4 * HIDDEN_SIZE
SEGMENT_ROWS = 511
BOUNDARY_M = (
    1,
    2,
    63,
    64,
    65,
    511,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
    4095,
    4096,
    4097,
    8191,
    8192,
    8193,
    16384,
)


@pytest.fixture(autouse=True)
def reset_selector_state():
    mhc._get_mhc_pre_splitk.cache_clear()
    mhc._MHC_STABLE_PRE_LOGGED.clear()
    mhc._MHC_STABLE_RMSNORM_LOGGED.clear()
    yield
    mhc._get_mhc_pre_splitk.cache_clear()
    mhc._MHC_STABLE_PRE_LOGGED.clear()
    mhc._MHC_STABLE_RMSNORM_LOGGED.clear()


@pytest.fixture
def stable_family(monkeypatch):
    monkeypatch.setattr(mhc, "get_gfx_runtime", lambda: STABLE_ARCH)
    monkeypatch.setattr(mhc, "get_cu_num", lambda: STABLE_CU)


@pytest.mark.parametrize("packed", [False, True])
def test_pre_selectors_are_cross_m_stable(stable_family, packed):
    for m in BOUNDARY_M:
        for selector in (mhc.get_mhc_pre_splitk, mhc.get_mhc_pre_splitk_large_m):
            assert selector(m, HC_HIDDEN_SIZE, w_preshuffle_bf16=packed) == (16, 128)


def test_selector_cache_is_keyed_by_hardware(monkeypatch):
    monkeypatch.setattr(mhc, "get_cu_num", lambda: 256)
    monkeypatch.setattr(mhc, "get_gfx_runtime", lambda: "gfx1250")
    assert mhc.get_mhc_pre_splitk(1, HC_HIDDEN_SIZE) != (16, 128)
    monkeypatch.setattr(mhc, "get_gfx_runtime", lambda: STABLE_ARCH)
    assert mhc.get_mhc_pre_splitk(1, HC_HIDDEN_SIZE) == (16, 128)
    monkeypatch.setattr(mhc, "get_cu_num", lambda: 128)
    assert mhc.get_mhc_pre_splitk(1, HC_HIDDEN_SIZE) != (16, 128)


@pytest.mark.parametrize(
    "arch,cu,hidden",
    [
        ("gfx950", 256, 4 * 7168),
        ("gfx950", 128, HC_HIDDEN_SIZE),
        ("gfx942", 256, HC_HIDDEN_SIZE),
        ("gfx1250", 256, HC_HIDDEN_SIZE),
    ],
)
@pytest.mark.parametrize("packed", [False, True])
def test_unregistered_families_keep_dynamic_policy(
    monkeypatch, arch, cu, hidden, packed
):
    monkeypatch.setattr(mhc, "get_gfx_runtime", lambda: arch)
    monkeypatch.setattr(mhc, "get_cu_num", lambda: cu)
    small = mhc.get_mhc_pre_splitk(1, hidden, w_preshuffle_bf16=packed)
    large = mhc.get_mhc_pre_splitk(16384, hidden, w_preshuffle_bf16=packed)
    assert small != large
    for m in (8191, 8192, 16384):
        expected = (
            (8, 64)
            if arch == "gfx950" and m >= 8192
            else mhc.get_mhc_pre_splitk(m, hidden, w_preshuffle_bf16=packed)
        )
        assert (
            mhc.get_mhc_pre_splitk_large_m(m, hidden, w_preshuffle_bf16=packed)
            == expected
        )


def test_stable_families_log_once(stable_family, monkeypatch):
    logger = Mock()
    monkeypatch.setattr(mhc, "logger", logger)
    for m in (1, 8192, 16384):
        mhc.get_mhc_pre_splitk(m, HC_HIDDEN_SIZE)
        mhc.get_mhc_pre_splitk_large_m(m, HC_HIDDEN_SIZE)
        mhc._log_stable_mhc_rmsnorm_family(HIDDEN_SIZE)
    assert logger.info.call_count == 2
    assert logger.info.call_args_list[0].args[1:] == (
        STABLE_ARCH,
        STABLE_CU,
        HC_HIDDEN_SIZE,
        16,
        128,
    )
    assert logger.info.call_args_list[1].args[1:] == (
        STABLE_ARCH,
        STABLE_CU,
        HIDDEN_SIZE,
        2,
        512,
        512,
    )
    mhc._log_stable_mhc_rmsnorm_family(7168)
    assert logger.info.call_count == 2


@pytest.fixture(scope="module")
def gpu_device():
    # Defer hardware detection until a GPU test runs; policy tests use mocks.
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm")
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if (
        props.gcnArchName.split(":")[0] != STABLE_ARCH
        or props.multi_processor_count != STABLE_CU
    ):
        pytest.skip("requires gfx950 with 256 CUs")
    if mhc.get_gfx_runtime() != STABLE_ARCH or mhc.get_cu_num() != STABLE_CU:
        pytest.skip("runtime selector does not report the registered family")
    return device


def make_inputs(m, device):
    generator = torch.Generator(device=device).manual_seed(0)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=generator, dtype=dtype, device=device)

    residual = randn(m, 4, HIDDEN_SIZE, dtype=torch.bfloat16)
    fn = randn(24, HC_HIDDEN_SIZE) * 0.02
    scale = randn(3)
    base = randn(24) * 0.1
    norm_weight = randn(HIDDEN_SIZE, dtype=torch.bfloat16)
    return residual, fn, scale, base, norm_weight


def assert_exact(actual, expected):
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        # Also distinguish signed zero; torch.equal alone does not do that.
        bits = torch.int16 if a.element_size() == 2 else torch.int32
        assert torch.equal(a.contiguous().view(bits), b.contiguous().view(bits)), (
            f"output {index}: {(a != b).sum().item()}/{a.numel()} elements differ; "
            f"max abs={(a.float() - b.float()).abs().max().item():.6e}"
        )


@pytest.mark.parametrize("whole_m", [11 * SEGMENT_ROWS, 8193])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("with_norm", [False, True])
@torch.inference_mode()
def test_mhc_pre_outputs_are_exact_across_runtime_m(
    gpu_device, whole_m, packed, with_norm
):
    residual, fn, scale, base, norm_weight = make_inputs(whole_m, gpu_device)
    if packed:
        fn = mhc.mhc_shuffle_fn(fn)
    if not with_norm:
        norm_weight = None

    def run(rows, large_m=False):
        return mhc.mhc_pre(
            rows,
            fn,
            scale,
            base,
            norm_weight=norm_weight,
            large_m_splitk=large_m,
            w_preshuffle_bf16=int(packed),
        )

    whole = run(residual)
    # Exercise the m >= 8192 override and the odd final row block.
    assert_exact(run(residual, large_m=True), whole)
    for start in range(0, whole_m, SEGMENT_ROWS):
        stop = min(start + SEGMENT_ROWS, whole_m)
        assert_exact(run(residual[start:stop]), tuple(out[start:stop] for out in whole))
    if whole_m == 8193:
        for m in BOUNDARY_M:
            if m < whole_m:
                assert_exact(run(residual[:m]), tuple(out[:m] for out in whole))


@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_mhc_head_is_cross_m_stable(gpu_device, packed):
    residual, fn, scale, base, _ = make_inputs(1025, gpu_device)
    fn, base = fn[:4].contiguous(), base[:4].contiguous()
    if packed:
        fn = mhc.mhc_shuffle_fn(fn)

    def run(rows):
        # Head mode only writes layer_input; mix outputs are intentionally unused.
        return (
            mhc.mhc_pre(
                rows, fn, scale, base, sinkhorn_repeat=0, w_preshuffle_bf16=int(packed)
            )[2],
        )

    whole = run(residual)
    for start in range(0, 1025, SEGMENT_ROWS):
        stop = min(start + SEGMENT_ROWS, 1025)
        assert_exact(run(residual[start:stop]), (whole[0][start:stop],))


@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_large_m_post_pre_outputs_are_exact_across_runtime_m(gpu_device, packed):
    residual, fn, scale, base, norm_weight = make_inputs(8193, gpu_device)
    if packed:
        fn = mhc.mhc_shuffle_fn(fn)
    # Nontrivial post mixing, with the same inputs in each chunk.
    layer_input = residual[:, 0, :].contiguous()
    post = torch.full((8193, 4), 0.3, device=gpu_device)
    comb = torch.full((8193, 4, 4), 0.2, device=gpu_device)

    def run(start, stop):
        return mhc.mhc_fused_post_pre_large_m(
            layer_input[start:stop],
            residual[start:stop],
            post[start:stop],
            comb[start:stop],
            fn,
            scale,
            base,
            norm_weight=norm_weight,
            w_preshuffle_bf16=packed,
        )

    whole = run(0, 8193)
    for start in range(0, 8193, 1025):
        stop = min(start + 1025, 8193)
        assert_exact(run(start, stop), tuple(out[start:stop] for out in whole))


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("with_norm", [False, True])
@torch.inference_mode()
def test_mhc_pre_matches_reference(gpu_device, packed, with_norm):
    residual, fn, scale, base, norm_weight = make_inputs(8, gpu_device)
    # Independent FP64 reference; cross-M equality alone could pass for a
    # consistently wrong kernel. RMSNorm uses the unrounded sum for its RMS
    # and a BF16 intermediate for the numerator, matching the native contract.
    x, weights, s, bias = (t.cpu().double() for t in (residual, fn, scale, base))
    flat = x.flatten(1)
    mixes = (flat @ weights.T) * (flat.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    scales = torch.cat((s[0].expand(4), s[1].expand(4), s[2].expand(16)))
    mixes = mixes * scales + bias
    pre = mixes[:, :4].sigmoid() + 1e-6
    post = mixes[:, 4:8].sigmoid().unsqueeze(-1)
    comb = mixes[:, 8:].reshape(-1, 4, 4).softmax(-1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    layer = (x * pre.unsqueeze(-1)).sum(1)
    if with_norm:
        rms = (layer.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        layer = layer.bfloat16().double() * rms * norm_weight.cpu().double()
    actual = mhc.mhc_pre(
        residual,
        mhc.mhc_shuffle_fn(fn) if packed else fn,
        scale,
        base,
        norm_weight=norm_weight if with_norm else None,
        w_preshuffle_bf16=int(packed),
    )
    for a, b in zip(actual[:2], (post, comb), strict=True):
        torch.testing.assert_close(a.cpu().double(), b, rtol=1e-4, atol=5e-6)
    # Allow BF16 rounding in the intermediate and output (two roundings with
    # RMSNorm), plus small absolute error for cancellation near zero.
    torch.testing.assert_close(actual[2].cpu().double(), layer, rtol=0.016, atol=2e-3)
