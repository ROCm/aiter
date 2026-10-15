# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import pytest
import torch

import aiter
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cdna4_num_xcds, get_gfx_runtime
from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_8wave import (
    flydsl_8wave_gemm_a8,
    parse_8wave_kernel_name,
    run_gemm_a8w8_bpreshuffle_8wave,
)
from aiter.ops.gemm_op_a8w8 import get_GEMM_config_with_quant_type
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as gluon_gemm_a8w8


def test_int8_8wave_kernel_name_contract():
    name = "flydsl_bpreshuffle_8w_256x256x128_I8_I8_B16_1x8"
    assert parse_8wave_kernel_name(name) == {
        "block_m": 256,
        "block_n": 256,
        "block_k": 128,
        "q_dtype_a": "I8",
        "q_dtype_w": "I8",
        "dtype": "B16",
        "waves_per_eu": 1,
        "xcd_swizzle": 8,
    }


@pytest.mark.parametrize("cu_num,num_xcds", [(64, 2), (128, 4), (256, 8)])
def test_cdna4_xcd_count_from_cus(cu_num, num_xcds):
    assert get_cdna4_num_xcds(cu_num) == num_xcds


@pytest.mark.parametrize("cu_num", [0, 96, 129, 288])
def test_cdna4_xcd_count_rejects_invalid_cus(cu_num):
    with pytest.raises(ValueError, match="CDNA4"):
        get_cdna4_num_xcds(cu_num)


def test_8wave_compile_cache_keys_xcd_count(monkeypatch):
    import aiter.ops.flydsl.gemm_a8w8_bpreshuffle_8wave as mod

    monkeypatch.setattr(mod, "_validate", lambda **_kwargs: None)
    monkeypatch.setattr(mod, "_compile_fp8_gemm_8w", lambda **kwargs: kwargs)
    mod.compile_8wave_gemm.cache_clear()
    common = {
        "K": 256,
        "block_m": 256,
        "block_n": 256,
        "waves_per_eu": 1,
        "xcd_swizzle": 8,
        "q_dtype": "int8",
    }
    xcd4 = mod.compile_8wave_gemm(**common, num_xcds=4)
    xcd8 = mod.compile_8wave_gemm(**common, num_xcds=8)
    assert xcd4["num_xcds"] == 4
    assert xcd8["num_xcds"] == 8
    assert xcd4 is not xcd8
    mod.compile_8wave_gemm.cache_clear()


def test_8wave_kernel_name_rejects_dtype_mismatch():
    x = torch.empty((1, 256), dtype=torch.int8)
    weight = torch.empty((256, 256), dtype=torch.int8)
    scale_a = torch.ones((1, 1), dtype=torch.float32)
    scale_b = torch.ones((1, 256), dtype=torch.float32)
    out = torch.empty((1, 256), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requests fp8"):
        run_gemm_a8w8_bpreshuffle_8wave(
            x,
            weight,
            scale_a,
            scale_b,
            out,
            "flydsl_bpreshuffle_8w_128x256x128_F8_F8_B16_1x0",
        )


@pytest.mark.skipif(get_gfx_runtime() != "gfx950", reason="gfx950 FlyDSL kernel")
@torch.no_grad()
def test_flydsl_int8_8wave_bit_exact_runtime_tails():
    # K=384 exercises the steady-state body as well as both peeled pipeline
    # steps. M/N deliberately tail the 128x256 workgroup tile.
    m, n, k = 131, 272, 384
    generator = torch.Generator(device="cuda").manual_seed(20261013)
    x = torch.randint(
        -31, 32, (m, k), dtype=torch.int8, device="cuda", generator=generator
    )
    weight = torch.randint(
        -31, 32, (n, k), dtype=torch.int8, device="cuda", generator=generator
    )
    x_scale = torch.rand(
        (m, 1), dtype=torch.float32, device="cuda", generator=generator
    )
    w_scale = torch.rand(
        (1, n), dtype=torch.float32, device="cuda", generator=generator
    )
    weight_shuffled = shuffle_weight(weight, layout=(16, 16))
    actual = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    flydsl_8wave_gemm_a8(
        x,
        weight_shuffled,
        x_scale,
        w_scale,
        actual,
        block_m=256,
        block_n=256,
        waves_per_eu=1,
        xcd_swizzle=0,
    )

    # Values are intentionally small enough that the FP32 linear reference
    # represents every integer product and sum exactly. Match the kernel's
    # single post-reduction scale product and require BF16 bit identity.
    accum = torch.nn.functional.linear(x.float(), weight.float())
    expected = (accum * (x_scale * w_scale)).to(torch.bfloat16)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(get_gfx_runtime() != "gfx950", reason="gfx950 FlyDSL kernel")
@torch.no_grad()
def test_flydsl_8wave_fp8_regression():
    m, n, k = 17, 256, 256
    generator = torch.Generator(device="cuda").manual_seed(20261013)
    x_float = torch.randn(
        (m, k), dtype=torch.float32, device="cuda", generator=generator
    )
    w_float = torch.randn(
        (n, k), dtype=torch.float32, device="cuda", generator=generator
    )
    x_scale = x_float.abs().amax(dim=1, keepdim=True).clamp_min_(1e-12) / 448.0
    w_scale = (
        w_float.abs().amax(dim=1, keepdim=True).T.contiguous().clamp_min_(1e-12) / 448.0
    )
    x = (x_float / x_scale).to(torch.float8_e4m3fn)
    weight = (w_float / w_scale.T).to(torch.float8_e4m3fn)
    weight_shuffled = shuffle_weight(weight, layout=(16, 16))
    actual = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    flydsl_8wave_gemm_a8(
        x,
        weight_shuffled,
        x_scale,
        w_scale,
        actual,
        block_m=128,
        block_n=256,
        waves_per_eu=1,
        xcd_swizzle=0,
    )

    expected = torch.nn.functional.linear(x.float(), weight.float())
    expected = (expected * (x_scale * w_scale)).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=1e-2)


# ---------------------------------------------------------------------------
# Public ``aiter.gemm_a8w8_bpreshuffle`` coverage (frozen INT8 ABI)
# ---------------------------------------------------------------------------

TUNED_KERNEL = "flydsl_bpreshuffle_8w_256x256x128_I8_I8_B16_1x8"

# Required (tuned) Wan 14B rows: 75,600 tokens at SP degree 8 -> M=9450.
WAN_TUNED_ROWS = (
    (9450, 5120, 5120),
    (9450, 13824, 5120),
    (9450, 5120, 13824),
)
# Other SP / A2A-padding / cross-attention M values for the 5120x5120 Wan
# projection. They have no tuned row: they must stay correct through the
# default-kernel fallback (not NaN/garbage from the FP8-only CK fallback).
WAN_UNTUNED_ROWS = (
    (512, 5120, 5120),  # cross-attention K/V (~512 text tokens)
    (9472, 5120, 5120),  # packed A2A padding (75,776 / 8)
    (18900, 5120, 5120),  # SP4
    (37800, 5120, 5120),  # SP2
    (75600, 5120, 5120),  # SP1
)

requires_gfx950 = pytest.mark.skipif(
    get_gfx_runtime() != "gfx950", reason="gfx950 FlyDSL kernel"
)


@functools.lru_cache(maxsize=4)
def _make_case(m, n, k, seed=20261013):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x_float = torch.randn((m, k), device="cuda", generator=generator)
    w_float = torch.randn((n, k), device="cuda", generator=generator)
    x_scale = x_float.abs().amax(dim=1, keepdim=True).clamp_min_(1e-12) / 127.0
    w_scale = (
        w_float.abs().amax(dim=1, keepdim=True).T.contiguous().clamp_min_(1e-12) / 127.0
    )
    x = (x_float / x_scale).to(torch.int8)
    weight = (w_float / w_scale.T).to(torch.int8)
    return {
        "x": x,
        "weight": weight,
        "weight_shuffled": shuffle_weight(weight, layout=(16, 16)),
        "x_scale": x_scale,
        "w_scale": w_scale,
    }


def _public(case, **overrides):
    args = {
        "XQ": case["x"],
        "WQ": case["weight_shuffled"],
        "x_scale": case["x_scale"],
        "w_scale": case["w_scale"],
        "bias": None,
        "dtype": torch.bfloat16,
    }
    args.update(overrides)
    return aiter.gemm_a8w8_bpreshuffle(
        args["XQ"],
        args["WQ"],
        args["x_scale"],
        args["w_scale"],
        args["bias"],
        args["dtype"],
    )


def _gluon(case):
    m, n = case["x"].shape[0], case["weight"].shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    return gluon_gemm_a8w8(
        case["x"],
        case["weight"],
        case["x_scale"],
        case["w_scale"],
        None,
        torch.bfloat16,
        out,
        backend="gluon",
    )


@pytest.fixture
def spy_8wave(monkeypatch):
    import aiter.ops.flydsl.gemm_a8w8_bpreshuffle_8wave as mod

    calls = []
    original = mod.run_gemm_a8w8_bpreshuffle_8wave

    def spy(XQ, WQ, x_scale, w_scale, Out, kernel_name):
        calls.append(kernel_name)
        return original(XQ, WQ, x_scale, w_scale, Out, kernel_name)

    monkeypatch.setattr(mod, "run_gemm_a8w8_bpreshuffle_8wave", spy)
    return calls


@requires_gfx950
@pytest.mark.parametrize("m,n,k", WAN_TUNED_ROWS)
@torch.no_grad()
def test_public_wan_rows_select_tuned_kernel_and_match_gluon(spy_8wave, m, n, k):
    config = get_GEMM_config_with_quant_type(
        m, n, k, torch.int8, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE
    )
    assert config is not None, f"missing tuned INT8 row for {(m, n, k)}"
    assert config["libtype"] == "flydsl"
    assert config["kernelName"] == TUNED_KERNEL

    case = _make_case(m, n, k)
    first = _public(case)
    assert spy_8wave == [TUNED_KERNEL]
    assert first.dtype == torch.bfloat16 and tuple(first.shape) == (m, n)

    expected = _gluon(case)
    mismatch = torch.count_nonzero(first != expected).item()
    assert mismatch == 0, f"{mismatch}/{m * n} BF16 elements differ from Gluon"

    # Repeated launches (fresh output each time) must be bit-stable.
    for _ in range(3):
        assert torch.equal(_public(case), first)


@requires_gfx950
@pytest.mark.parametrize("m,n,k", WAN_UNTUNED_ROWS)
@torch.no_grad()
def test_public_untuned_wan_m_uses_correct_fallback(spy_8wave, m, n, k):
    case = _make_case(m, n, k)
    actual = _public(case)
    assert spy_8wave == [TUNED_KERNEL]
    assert torch.isfinite(actual.float()).all()
    mismatch = torch.count_nonzero(actual != _gluon(case)).item()
    assert mismatch == 0, f"{mismatch}/{m * n} BF16 elements differ from Gluon"


@requires_gfx950
@torch.no_grad()
def test_public_scales_accept_flat_and_2d_layouts():
    case = _make_case(*WAN_TUNED_ROWS[0])
    expected = _public(case)
    flat = _public(
        case,
        x_scale=case["x_scale"].reshape(-1),
        w_scale=case["w_scale"].reshape(-1),
    )
    assert torch.equal(flat, expected)


def _small_case(m=256, n=256, k=256):
    return _make_case(m, n, k, seed=7)


@requires_gfx950
@pytest.mark.parametrize(
    "override,exc,match",
    [
        ({"dtype": torch.float16}, ValueError, "BF16 output"),
        ({"dtype": torch.float32}, ValueError, "BF16 output"),
        ({"bias": "bias"}, ValueError, "bias"),
        ({"x_scale": "x_scale_fp16"}, ValueError, "FP32"),
        ({"w_scale": "w_scale_fp16"}, ValueError, "FP32"),
        ({"x_scale": "x_scale_short"}, ValueError, "per-row x_scale"),
        ({"w_scale": "w_scale_short"}, ValueError, "per-column w_scale"),
        ({"XQ": "x_noncontig"}, ValueError, "contiguous"),
        ({"WQ": "w_noncontig"}, ValueError, "contiguous"),
        ({"XQ": "x_1d"}, ValueError, "2-D"),
        ({"XQ": "x_fp8"}, ValueError, "must match"),
        ({"WQ": "w_fp8"}, ValueError, "must match"),
    ],
)
@torch.no_grad()
def test_public_rejects_contract_violations(override, exc, match):
    case = _small_case()
    m, n, k = 256, 256, 256
    variants = {
        "bias": torch.zeros((1, n), dtype=torch.float32, device="cuda"),
        "x_scale_fp16": case["x_scale"].half(),
        "w_scale_fp16": case["w_scale"].half(),
        "x_scale_short": case["x_scale"][: m // 2],
        "w_scale_short": case["w_scale"][:, : n // 2],
        "x_noncontig": torch.zeros((k, m), dtype=torch.int8, device="cuda").T,
        "w_noncontig": torch.zeros((k, n), dtype=torch.int8, device="cuda").T,
        "x_1d": case["x"].reshape(-1),
        "x_fp8": case["x"].view(torch.uint8).view(torch.float8_e4m3fn),
        "w_fp8": case["weight_shuffled"].view(torch.uint8).view(torch.float8_e4m3fn),
    }
    resolved = {
        key: variants[value] if isinstance(value, str) else value
        for key, value in override.items()
    }
    with pytest.raises(exc, match=match):
        _public(case, **resolved)


@requires_gfx950
@pytest.mark.parametrize("k", [128, 200, 320])
@torch.no_grad()
def test_public_rejects_unsupported_k(k):
    # K must be a multiple of 128 and >= 256 for the 8-wave pipeline.
    m, n = 256, 256
    x = torch.zeros((m, k), dtype=torch.int8, device="cuda")
    w = torch.zeros((n, k), dtype=torch.int8, device="cuda")
    scale_x = torch.ones((m, 1), dtype=torch.float32, device="cuda")
    scale_w = torch.ones((1, n), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="K must be"):
        aiter.gemm_a8w8_bpreshuffle(x, w, scale_x, scale_w, None, torch.bfloat16)


@pytest.mark.skipif(
    get_gfx_runtime() != "gfx950" or torch.cuda.device_count() < 2,
    reason="needs two gfx950 devices",
)
@torch.no_grad()
def test_public_rejects_mixed_devices():
    case = _small_case()
    with pytest.raises(ValueError, match="share a device"):
        _public(case, x_scale=case["x_scale"].to("cuda:1"))


@requires_gfx950
@torch.no_grad()
def test_public_untuned_wrong_arch_not_silently_ck(monkeypatch):
    """Non-gfx950 INT8 must raise rather than reach the FP8-only CK fallback."""
    import aiter.ops.gemm_op_a8w8 as op

    monkeypatch.setattr(op, "get_gfx", lambda: "gfx942")
    case = _small_case(512, 256, 256)
    with pytest.raises(NotImplementedError, match="gfx950"):
        _public(case)


@requires_gfx950
@torch.no_grad()
def test_public_torch_compile_fullgraph_matches_eager():
    torch._dynamo.reset()
    case = _make_case(*WAN_TUNED_ROWS[0])
    eager = _public(case)  # warm the kernel cache before tracing

    def run(x, w, xs, ws):
        return aiter.gemm_a8w8_bpreshuffle(x, w, xs, ws, None, torch.bfloat16)

    compiled = torch.compile(run, fullgraph=True)
    args = (case["x"], case["weight_shuffled"], case["x_scale"], case["w_scale"])
    for _ in range(2):
        out = compiled(*args)
        torch.cuda.synchronize()
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, eager)
    torch._dynamo.reset()


@requires_gfx950
@torch.no_grad()
def test_public_cuda_graph_capture_and_replay():
    m, n, k = WAN_TUNED_ROWS[0]
    case = _make_case(m, n, k)
    static_x = case["x"].clone()
    weight = case["weight_shuffled"]
    x_scale = case["x_scale"].clone()
    w_scale = case["w_scale"]

    def run():
        return aiter.gemm_a8w8_bpreshuffle(
            static_x, weight, x_scale, w_scale, None, torch.bfloat16
        )

    # JIT/compile must finish before capture; compiling during capture would
    # issue forbidden host-side operations on the capturing stream.
    warm = run()
    torch.cuda.synchronize()
    assert torch.equal(warm, _public(case))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = run()
    out_ptr = static_out.data_ptr()

    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, warm)

    # New activations/scales copied into the captured buffers change the result
    # in place at a stable address.
    other = _make_case(m, n, k, seed=11)
    static_x.copy_(other["x"])
    x_scale.copy_(other["x_scale"])
    graph.replay()
    torch.cuda.synchronize()
    assert static_out.data_ptr() == out_ptr
    reference = aiter.gemm_a8w8_bpreshuffle(
        other["x"], weight, other["x_scale"], w_scale, None, torch.bfloat16
    )
    assert torch.equal(static_out, reference)

    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    assert static_out.data_ptr() == out_ptr
    assert torch.equal(static_out, reference)


@requires_gfx950
@torch.no_grad()
def test_public_runs_on_current_nondefault_stream():
    case = _make_case(*WAN_TUNED_ROWS[0])
    expected = _public(case)
    torch.cuda.synchronize()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out = _public(case)
        # Stream-ordered consumer: only correct if the GEMM was enqueued on
        # ``side`` rather than the legacy/default stream.
        doubled = out.float() * 2
    side.synchronize()
    assert torch.equal(out, expected)
    assert torch.equal(doubled, expected.float() * 2)
