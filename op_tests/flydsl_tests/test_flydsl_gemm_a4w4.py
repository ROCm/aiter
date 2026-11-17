# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util

import pytest
import torch

_SKIP = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or torch.cuda.get_device_properties().gcnArchName.split(":", 1)[0] != "gfx950"
    or importlib.util.find_spec("flydsl") is None,
    reason="gfx950 and FlyDSL are required",
)


def _quantize_roundup(inputs):
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.utility.mx_types import MxScaleRoundModeInt

    packed, scales = per_1x32_f4_quant(inputs, round_mode=MxScaleRoundModeInt.RoundUp)
    return packed.view(torch.uint8), scales.view(torch.uint8)


@pytest.mark.parametrize(
    "m,n,k,expected",
    [
        (127, 8448, 7168, 16),
        (128, 8448, 7168, 64),
        (511, 7168, 4224, 16),
        (512, 7168, 4224, 64),
        (2047, 1536, 7168, 16),
        (2048, 1536, 7168, 64),
        (511, 7168, 768, 16),
        (512, 7168, 768, 64),
        (4096, 256, 7168, 16),
    ],
)
def test_flydsl_dense_a4w4_bm_selection(m, n, k, expected):
    from aiter.ops.flydsl.gemm_a4w4 import _select_bm

    assert _select_bm(m, n, k) == expected


@_SKIP
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1536, 7168),
        (17, 1536, 7168),
        (1, 1536, 4224),
        (17, 1536, 4224),
    ],
)
def test_flydsl_dense_a4w4_matches_triton(m, n, k):
    from aiter.ops.flydsl.gemm_a4w4 import (
        flydsl_gemm_a4w4,
        prepare_gemm_a4w4_weight,
    )
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    torch.manual_seed(1)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    w, w_scale = _quantize_roundup(w_bf16)
    a_quant, a_scale = _quantize_roundup(a)

    expected = gemm_afp4wfp4(a_quant, w, a_scale, w_scale, dtype=torch.bfloat16)
    prepared = prepare_gemm_a4w4_weight(w, w_scale)
    actual = flydsl_gemm_a4w4(a, prepared, _bm=16)

    actual_f32, expected_f32 = actual.float(), expected.float()
    assert torch.isfinite(actual_f32).all()
    assert torch.isfinite(expected_f32).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    relative_rmse = (actual_f32 - expected_f32).square().mean().sqrt()
    relative_rmse /= expected_f32.square().mean().sqrt().clamp_min(1e-6)
    assert cosine > 0.998
    assert relative_rmse < 0.01


@_SKIP
@pytest.mark.parametrize("k", [7168, 4224])
@pytest.mark.parametrize("m", [1, 15, 16, 17, 63, 64, 65, 127, 128, 129])
def test_flydsl_dense_a4w4_bm64_boundaries(m, k):
    from aiter.ops.flydsl.gemm_a4w4 import (
        flydsl_gemm_a4w4,
        prepare_gemm_a4w4_weight,
    )
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    n = 256
    torch.manual_seed(2)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w, w_scale = _quantize_roundup(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    )
    a_quant, a_scale = _quantize_roundup(a)
    expected = gemm_afp4wfp4(a_quant, w, a_scale, w_scale, dtype=torch.bfloat16)

    actual = flydsl_gemm_a4w4(a, prepare_gemm_a4w4_weight(w, w_scale), _bm=64)
    actual_f32, expected_f32 = actual.float(), expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    relative_rmse = (actual_f32 - expected_f32).square().mean().sqrt()
    relative_rmse /= expected_f32.square().mean().sqrt().clamp_min(1e-6)
    assert torch.isfinite(actual_f32).all()
    assert cosine > 0.998
    assert relative_rmse < 0.01


@_SKIP
def test_flydsl_dense_a4w4_bm64_scale_boundaries_and_row_groups():
    from aiter.ops.flydsl.gemm_a4w4 import (
        flydsl_gemm_a4w4,
        prepare_gemm_a4w4_weight,
    )
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    m, n, k = 64, 256, 768
    base = torch.linspace(-1.0625, 1.0625, 32, device="cuda")
    rows = []
    for row in range(m):
        exponent = (row % 12) - 6
        block = base * (2.0**exponent)
        rows.append(block.repeat(k // 32))
    a = torch.stack(rows).to(torch.bfloat16)
    torch.manual_seed(3)
    w, w_scale = _quantize_roundup(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    )
    a_quant, a_scale = _quantize_roundup(a)
    expected = gemm_afp4wfp4(a_quant, w, a_scale, w_scale, dtype=torch.bfloat16)
    actual = flydsl_gemm_a4w4(a, prepare_gemm_a4w4_weight(w, w_scale), _bm=64)

    actual_f32, expected_f32 = actual.float(), expected.float()
    assert torch.isfinite(actual_f32).all()
    assert torch.isfinite(expected_f32).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    relative_rmse = (actual_f32 - expected_f32).square().mean().sqrt()
    relative_rmse /= expected_f32.square().mean().sqrt().clamp_min(1e-6)
    assert cosine > 0.998
    assert relative_rmse < 0.01


@_SKIP
@pytest.mark.parametrize(
    "n,k",
    [(8448, 7168), (7168, 4224), (1536, 7168), (7168, 768), (1536, 4224)],
)
def test_flydsl_dense_a4w4_required_shapes_compile_and_run(n, k):
    from aiter.ops.flydsl.gemm_a4w4 import flydsl_gemm_a4w4

    a = torch.zeros((1, k), device="cuda", dtype=torch.bfloat16)
    w, w_scale = _quantize_roundup(
        torch.zeros((n, k), device="cuda", dtype=torch.bfloat16)
    )
    out = flydsl_gemm_a4w4(a, w, w_scale)
    assert out.shape == (1, n)
    assert not out.isnan().any()


@_SKIP
def test_flydsl_dense_a4w4_k4224_scale_padding():
    from aiter.ops.flydsl.gemm_a4w4 import prepare_gemm_a4w4_weight

    weight = torch.zeros((256, 2112), device="cuda", dtype=torch.uint8)
    weight_scale = torch.zeros((256, 132), device="cuda", dtype=torch.uint8)
    prepared = prepare_gemm_a4w4_weight(weight, weight_scale)
    assert prepared.weight.shape == (256, 2112)
    assert prepared.scale.shape == (256, 136)


@_SKIP
def test_flydsl_dense_a4w4_rejects_non_128_aligned_k():
    from aiter.ops.flydsl.gemm_a4w4 import prepare_gemm_a4w4_weight

    weight = torch.zeros((256, 96), device="cuda", dtype=torch.uint8)
    weight_scale = torch.zeros((256, 6), device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="positive multiple of 128"):
        prepare_gemm_a4w4_weight(weight, weight_scale)


@pytest.mark.parametrize(
    "m,n,k,message",
    [
        (1, 256, 256, "three BK tiles"),
        (1, 256, 8320, "at most 8192"),
        (2**21, 256, 1024, "M\\*K\\*2"),
        (2**20, 2048, 768, "M\\*N"),
        (1, 2**23, 768, "Packed weight"),
    ],
)
def test_flydsl_dense_a4w4_rejects_unaddressable_extents(m, n, k, message):
    from aiter.ops.flydsl.gemm_a4w4 import PreshuffledA4W4Weight, flydsl_gemm_a4w4

    inputs = torch.empty((m, k), dtype=torch.bfloat16, device="meta")
    prepared = PreshuffledA4W4Weight(
        torch.empty((n, k // 2), dtype=torch.uint8, device="meta"),
        torch.empty((n, ((k + 255) // 256) * 8), dtype=torch.uint8, device="meta"),
        n,
        k,
    )
    with pytest.raises(ValueError, match=message):
        flydsl_gemm_a4w4(inputs, prepared)


def test_flydsl_dense_a4w4_validates_prepared_scale_layout():
    from aiter.ops.flydsl.gemm_a4w4 import (
        PreshuffledA4W4Weight,
        _validate_prepared,
    )

    weight = torch.empty(256, 2112, dtype=torch.uint8, device="meta")
    for scale in (
        torch.empty(256, 132, dtype=torch.uint8, device="meta"),
        torch.empty(256, 136, dtype=torch.float32, device="meta"),
        torch.empty(136, 256, dtype=torch.uint8, device="meta").T,
    ):
        with pytest.raises(ValueError, match="Prepared"):
            _validate_prepared(PreshuffledA4W4Weight(weight, scale, 256, 4224))
    _validate_prepared(
        PreshuffledA4W4Weight(
            weight, torch.empty(256, 136, dtype=torch.uint8, device="meta"), 256, 4224
        )
    )


def test_flydsl_dense_a4w4_checks_tensor_device_architecture(monkeypatch):
    from aiter.ops.flydsl import gemm_a4w4 as mod

    devices = []

    def architecture(device):
        devices.append(device)
        return "gfx942"

    monkeypatch.setattr(mod, "get_gfx_from_device", architecture)
    with pytest.raises(RuntimeError, match="gfx942"):
        mod._require_gfx950(torch.device("cuda", 1))
    assert devices == [torch.device("cuda", 1)]


@pytest.mark.parametrize("invalid", ["rank", "dtype", "device", "bm"])
def test_flydsl_dense_a4w4_rejects_invalid_call_metadata(invalid):
    from aiter.ops.flydsl.gemm_a4w4 import PreshuffledA4W4Weight, flydsl_gemm_a4w4

    prepared = PreshuffledA4W4Weight(
        torch.empty(256, 384, dtype=torch.uint8),
        torch.empty(256, 24, dtype=torch.uint8),
        256,
        768,
    )
    inputs = torch.empty(1, 768, dtype=torch.bfloat16)
    if invalid == "rank":
        inputs = inputs.reshape(-1)
    elif invalid == "dtype":
        inputs = inputs.float()
    expected = {
        "rank": (ValueError, "2D"),
        "dtype": (TypeError, "BF16"),
        "device": (ValueError, "CUDA/ROCm"),
        "bm": (ValueError, "_bm"),
    }
    error, message = expected[invalid]
    with pytest.raises(error, match=message):
        flydsl_gemm_a4w4(inputs, prepared, _bm=32 if invalid == "bm" else None)


@_SKIP
@pytest.mark.parametrize("invalid", ["shape", "dtype", "stride", "device"])
def test_flydsl_dense_a4w4_rejects_invalid_output(invalid):
    from aiter.ops.flydsl.gemm_a4w4 import flydsl_gemm_a4w4, prepare_gemm_a4w4_weight

    inputs = torch.zeros(2, 768, dtype=torch.bfloat16, device="cuda")
    prepared = prepare_gemm_a4w4_weight(
        torch.zeros(256, 384, dtype=torch.uint8, device="cuda"),
        torch.full((256, 24), 127, dtype=torch.uint8, device="cuda"),
    )
    output = torch.empty(
        (256, 2) if invalid == "stride" else (2, 255 if invalid == "shape" else 256),
        dtype=torch.float32 if invalid == "dtype" else torch.bfloat16,
        device="cpu" if invalid == "device" else "cuda",
    )
    if invalid == "stride":
        output = output.T
    with pytest.raises(ValueError, match="out must be contiguous BF16"):
        flydsl_gemm_a4w4(inputs, prepared, out=output)


@_SKIP
@pytest.mark.parametrize("preallocated", [False, True])
def test_flydsl_dense_a4w4_empty_batch_does_not_compile(monkeypatch, preallocated):
    from aiter.ops.flydsl.gemm_a4w4 import flydsl_gemm_a4w4, prepare_gemm_a4w4_weight
    from aiter.ops.flydsl.kernels import gemm_a4w4 as kernel

    prepared = prepare_gemm_a4w4_weight(
        torch.zeros(256, 384, dtype=torch.uint8, device="cuda"),
        torch.full((256, 24), 127, dtype=torch.uint8, device="cuda"),
    )
    monkeypatch.setattr(
        kernel,
        "compile_gemm_a4w4",
        lambda **kwargs: pytest.fail("Empty batch compiled"),
    )
    inputs = torch.empty(0, 768, dtype=torch.bfloat16, device="cuda")
    output = (
        torch.empty(0, 256, dtype=torch.bfloat16, device="cuda")
        if preallocated
        else None
    )
    actual = flydsl_gemm_a4w4(inputs, prepared, out=output)
    assert actual.shape == (0, 256) and actual.dtype == torch.bfloat16
    if output is not None:
        assert actual is output


@_SKIP
def test_flydsl_dense_a4w4_rejects_output_alias():
    from aiter.ops.flydsl.gemm_a4w4 import flydsl_gemm_a4w4, prepare_gemm_a4w4_weight

    inputs = torch.zeros(1, 768, dtype=torch.bfloat16, device="cuda")
    prepared = prepare_gemm_a4w4_weight(
        torch.zeros(768, 384, dtype=torch.uint8, device="cuda"),
        torch.full((768, 24), 127, dtype=torch.uint8, device="cuda"),
    )
    with pytest.raises(ValueError, match="must not overlap"):
        flydsl_gemm_a4w4(inputs, prepared, out=inputs)


@_SKIP
@pytest.mark.parametrize(
    "tile_m,rows,k", [(16, 17, 4224), (64, 65, 4224), (16, 17, 768), (64, 65, 768)]
)
def test_flydsl_dense_a4w4_random_scales_match_quantized_oracle(tile_m, rows, k):
    from aiter.ops.flydsl.gemm_a4w4 import flydsl_gemm_a4w4, prepare_gemm_a4w4_weight
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.utility import fp4_utils

    torch.manual_seed(rows + k)
    inputs = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    packed = torch.randint(0, 256, (256, k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.randint(0x78, 0x87, (256, k // 32), device="cuda", dtype=torch.uint8)
    quantized, input_scales = _quantize_roundup(inputs)
    decoded_inputs = fp4_utils.mxfp4_to_f32(quantized) * fp4_utils.e8m0_to_f32(
        input_scales
    ).repeat_interleave(32, -1)
    decoded_weights = fp4_utils.mxfp4_to_f32(packed) * fp4_utils.e8m0_to_f32(
        scales
    ).repeat_interleave(32, -1)
    expected = decoded_inputs.float() @ decoded_weights.float().T
    baseline = gemm_afp4wfp4(
        quantized, packed, input_scales, scales, dtype=torch.bfloat16
    )
    prepared = prepare_gemm_a4w4_weight(packed, scales)
    output = torch.empty(rows, 256, dtype=torch.bfloat16, device="cuda")
    actual = flydsl_gemm_a4w4(inputs, prepared, out=output, _bm=tile_m)
    assert actual is output
    for value in (actual, baseline):
        assert torch.isfinite(value).all()
        relative_l2 = (value.float() - expected).norm() / expected.norm().clamp_min(
            1e-30
        )
        assert relative_l2 < 0.01
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flydsl_gemm_a4w4(inputs, prepared, out=output, _bm=tile_m)
    graph.replay()
    torch.cuda.synchronize()
    assert (output.float() - expected).norm() / expected.norm().clamp_min(1e-30) < 0.01


@_SKIP
def test_flydsl_dense_a4w4_e8m0_classification():
    from aiter.ops.flydsl.gemm_a4w4 import flydsl_gemm_a4w4, prepare_gemm_a4w4_weight
    from aiter.utility import fp4_utils

    inputs = torch.zeros(1, 768, device="cuda", dtype=torch.bfloat16)
    inputs[0, 0] = 1
    packed = torch.full((256, 384), 0x22, device="cuda", dtype=torch.uint8)
    scales = torch.full((256, 24), 127, device="cuda", dtype=torch.uint8)
    codes = torch.tensor(
        [0, 1, 0x78, 0x7F, 0x86, 0xFE, 0xFF], device="cuda", dtype=torch.uint8
    )
    scales[:, 0] = codes[torch.arange(256, device="cuda") % len(codes)]
    expected = fp4_utils.e8m0_to_f32(scales[:, 0]).to(torch.bfloat16)[None, :]
    actual = flydsl_gemm_a4w4(inputs, prepare_gemm_a4w4_weight(packed, scales))
    torch.testing.assert_close(
        torch.isnan(actual), torch.isnan(expected), rtol=0, atol=0
    )
    torch.testing.assert_close(
        torch.isinf(actual), torch.isinf(expected), rtol=0, atol=0
    )
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], rtol=0, atol=0)


@_SKIP
@pytest.mark.parametrize("tile_m,inline_quant", [(16, True), (32, False)])
def test_dense_changes_preserve_shared_moe_gemm1(tile_m, inline_quant):
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight
    from aiter.utility import fp4_utils

    torch.manual_seed(4782)
    hidden, intermediate = 768, 128
    inputs = torch.randn(tile_m, hidden, device="cuda", dtype=torch.bfloat16)
    packed = torch.randint(
        0, 256, (1, 2 * intermediate, hidden // 2), device="cuda", dtype=torch.uint8
    )
    scales = torch.randint(
        123, 130, (2 * intermediate, hidden // 32), device="cuda", dtype=torch.uint8
    )
    quantized, input_scales = _quantize_roundup(inputs)
    output = torch.empty(tile_m, intermediate // 2, device="cuda", dtype=torch.uint8)
    output_scales = torch.empty(32, 8, device="cuda", dtype=torch.uint8)
    flydsl_mxfp4_gemm1(
        a_quant=quantized,
        a_scale_sorted_shuffled=fp4_utils.e8m0_shuffle(input_scales),
        w1_u8=shuffle_weight(packed, layout=(16, 16)),
        w1_scale_u8=shuffle_scale(scales),
        sorted_expert_ids=torch.zeros(1, device="cuda", dtype=torch.int32),
        cumsum_tensor=torch.tensor([tile_m], device="cuda", dtype=torch.int32),
        m_indices=torch.arange(tile_m, device="cuda", dtype=torch.int32),
        inter_sorted_quant=output,
        inter_sorted_shuffled_scale=output_scales,
        hidden_states=inputs,
        n_tokens=tile_m,
        BM=tile_m,
        use_nt=True,
        inline_quant=inline_quant,
        NE=1,
        D_HIDDEN=hidden,
        D_INTER=intermediate,
        topk=1,
        swiglu_limit=float("inf"),
    )
    decoded_inputs = fp4_utils.mxfp4_to_f32(quantized) * fp4_utils.e8m0_to_f32(
        input_scales
    ).repeat_interleave(32, -1)
    decoded_weights = fp4_utils.mxfp4_to_f32(packed[0]) * fp4_utils.e8m0_to_f32(
        scales
    ).repeat_interleave(32, -1)
    gate, up = (decoded_inputs @ decoded_weights.T).chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    reference_packed, reference_scales = _quantize_roundup(activated)
    reference_scales = torch.nn.functional.pad(reference_scales, (0, 4, 0, 32 - tile_m))
    shuffled = fp4_utils.e8m0_shuffle(reference_scales)[:32]
    active = fp4_utils.e8m0_shuffle(
        torch.nn.functional.pad(
            torch.ones(tile_m, 4, device="cuda", dtype=torch.uint8),
            (0, 4, 0, 32 - tile_m),
        )
    )[:32].bool()
    torch.testing.assert_close(output_scales[active], shuffled[active], rtol=0, atol=0)
    actual = fp4_utils.mxfp4_to_f32(output)
    expected = fp4_utils.mxfp4_to_f32(reference_packed)
    assert torch.isfinite(actual).all()
    assert (actual - expected).norm() / expected.norm().clamp_min(1e-30) < 0.01
