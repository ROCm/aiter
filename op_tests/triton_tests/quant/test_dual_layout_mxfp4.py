# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.quant import dual_layout_quant_mxfp4
from aiter.ops.triton.quant.dual_layout_mxfp4 import _philox_streams
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.mxfp4 import (
    mxfp4_data_shuffle_supported,
    mxfp4_scale_swizzle_supported,
)
from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm, shuffle_weight
from aiter.utility.fp4_utils import (
    e8m0_to_f32,
    f32_to_mx_e8m0_scale,
    f32_to_mxfp4,
)
from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

_BLOCK_SIZE = 32
_HADAMARD_SIZE = 16
_MAX_PHILOX_COUNTER = (1 << 63) - 1


requires_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_info.get_arch() != "gfx950",
    reason="dual-layout MXFP4 quantization requires gfx950",
)


def _hadamard16(device: torch.device) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    while matrix.shape[0] < _HADAMARD_SIZE:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / (_HADAMARD_SIZE**0.5)


def _h16_transposed(x: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    blocks = x.T.float().reshape(N, M // _HADAMARD_SIZE, _HADAMARD_SIZE)
    rotated = (blocks * sign.float().reshape(1, 1, -1)) @ _hadamard16(x.device)
    return rotated.reshape(N, M)


def _quantize_rtn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    blocks = x.float().reshape(M, N // _BLOCK_SIZE, _BLOCK_SIZE)
    scales = f32_to_mx_e8m0_scale(
        blocks.abs().amax(dim=-1),
        mode=MxScaleRoundModeInt.Even,
        dtype=MxDtypeInt.FP4_E2M1,
    ).view(torch.uint8)
    scales = scales.clamp_max(254)
    scaled = blocks / e8m0_to_f32(scales).float().unsqueeze(-1)
    packed = f32_to_mxfp4(scaled.reshape(M, N)).view(torch.uint8)
    return packed, scales


def _signs(device: torch.device) -> torch.Tensor:
    return torch.where(
        torch.arange(_HADAMARD_SIZE, device=device) % 3 == 0,
        -torch.ones(_HADAMARD_SIZE, device=device),
        torch.ones(_HADAMARD_SIZE, device=device),
    ).to(torch.bfloat16)


@requires_gfx950
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(32, 32), (64, 96), (96, 64), (256, 256)])
def test_dual_layout_quant_mxfp4_rtn_matches_reference(shape, dtype):
    torch.manual_seed(17)
    x = torch.randn(shape, dtype=dtype, device="cuda") * 4.0
    sign = _signs(x.device)

    actual = dual_layout_quant_mxfp4(x, sign)
    row_expected = _quantize_rtn(x)
    transposed_expected = _quantize_rtn(_h16_transposed(x, sign))

    for result, expected in zip(actual, (*row_expected, *transposed_expected)):
        torch.testing.assert_close(result, expected, atol=0, rtol=0)

    M, N = shape
    assert actual[0].shape == (M, N // 2)
    assert actual[1].shape == (M, N // _BLOCK_SIZE)
    assert actual[2].shape == (N, M // 2)
    assert actual[3].shape == (N, M // _BLOCK_SIZE)
    assert all(t.dtype == torch.uint8 and t.is_contiguous() for t in actual)


@requires_gfx950
def test_dual_layout_quant_mxfp4_h16_is_not_plain_transpose():
    torch.manual_seed(23)
    x = torch.randn((64, 96), dtype=torch.bfloat16, device="cuda")
    sign = _signs(x.device)

    _, _, transposed, _ = dual_layout_quant_mxfp4(x, sign)
    plain_transposed, _ = _quantize_rtn(x.T.contiguous())

    assert not torch.equal(transposed, plain_transposed)


@requires_gfx950
def test_dual_layout_quant_mxfp4_fused_gemm_layouts_match_shared_shuffles():
    torch.manual_seed(25)
    x = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
    sign = _signs(x.device)

    row, row_scale, col, col_scale = dual_layout_quant_mxfp4(x, sign)
    fused = dual_layout_quant_mxfp4(
        x,
        sign,
        swizzle_scale=True,
        shuffle_col=True,
    )

    torch.testing.assert_close(fused[0], row, atol=0, rtol=0)
    torch.testing.assert_close(
        fused[1],
        shuffle_scale_gemm(
            row_scale,
            arch="gfx950",
            preshuffle_factor=32,
            scale_kwidth=8,
        ),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        fused[2],
        shuffle_weight(col, arch="gfx950", layout=(16, 16)),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        fused[3],
        shuffle_scale_gemm(
            col_scale,
            arch="gfx950",
            preshuffle_factor=32,
            scale_kwidth=8,
        ),
        atol=0,
        rtol=0,
    )


@requires_gfx950
def test_dual_layout_quant_mxfp4_preserves_e8m0_endpoints():
    sign = torch.ones(_HADAMARD_SIZE, dtype=torch.bfloat16, device="cuda")

    zeros = torch.zeros((32, 32), dtype=torch.float32, device="cuda")
    zero_row, zero_row_scale, zero_transposed, zero_transposed_scale = (
        dual_layout_quant_mxfp4(zeros, sign)
    )
    assert torch.count_nonzero(zero_row).item() == 0
    assert torch.count_nonzero(zero_transposed).item() == 0
    assert torch.all(zero_row_scale == 0)
    assert torch.all(zero_transposed_scale == 0)

    smallest_value = 2.0**-125
    smallest = torch.full((32, 32), smallest_value, dtype=torch.float32, device="cuda")
    smallest_row, smallest_row_scale, _, _ = dual_layout_quant_mxfp4(smallest, sign)
    assert torch.all(smallest_row_scale == 0)
    assert torch.all(smallest_row == 0x66)

    # [4*x, 0, ..., 0] @ H16/4 is the constant vector x, so this gives
    # the rotated-transposed layout its own raw-zero scale oracle.
    smallest_transposed_input = torch.zeros_like(smallest)
    smallest_transposed_input[::_HADAMARD_SIZE, :] = 4.0 * smallest_value
    _, _, smallest_transposed, smallest_transposed_scale = dual_layout_quant_mxfp4(
        smallest_transposed_input, sign
    )
    assert torch.all(smallest_transposed_scale == 0)
    assert torch.all(smallest_transposed == 0x66)

    largest = torch.zeros((32, 32), dtype=torch.float32, device="cuda")
    largest[:, 0] = torch.finfo(torch.float32).max
    largest_row, largest_row_scale, _, _ = dual_layout_quant_mxfp4(largest, sign)
    largest_row_expected = torch.zeros_like(largest_row)
    largest_row_expected[:, 0] = 0x04
    assert torch.all(largest_row_scale == 254)
    torch.testing.assert_close(largest_row, largest_row_expected, atol=0, rtol=0)

    # A constant block c maps to [4*c, 0, ..., 0]. Choosing max/4 verifies
    # that normalization happens before the butterfly's intermediate sums.
    largest_transposed_input = torch.full(
        (32, 32),
        torch.finfo(torch.float32).max / 4.0,
        dtype=torch.float32,
        device="cuda",
    )
    _, _, largest_transposed, largest_transposed_scale = dual_layout_quant_mxfp4(
        largest_transposed_input, sign
    )
    largest_transposed_expected = torch.zeros_like(largest_transposed)
    largest_transposed_expected[:, 0] = 0x04
    largest_transposed_expected[:, _HADAMARD_SIZE // 2] = 0x04
    assert torch.all(largest_transposed_scale == 254)
    torch.testing.assert_close(
        largest_transposed, largest_transposed_expected, atol=0, rtol=0
    )


@requires_gfx950
@pytest.mark.parametrize(
    "use_sr_row,use_sr_transposed",
    [(True, False), (False, True), (True, True)],
)
@requires_gfx950
def test_dual_layout_quant_mxfp4_sr_is_reproducible_and_reuses_rtn_scales(
    use_sr_row,
    use_sr_transposed,
):
    torch.manual_seed(29)
    x = torch.randn((64, 96), dtype=torch.bfloat16, device="cuda")
    sign = _signs(x.device)
    kwargs = {
        "use_sr_row": use_sr_row,
        "use_sr_transposed": use_sr_transposed,
        "philox_seed": 1234,
        "philox_offset": (1 << 32) + 17,
    }

    actual = dual_layout_quant_mxfp4(x, sign, **kwargs)
    repeated = dual_layout_quant_mxfp4(x, sign, **kwargs)
    rtn = dual_layout_quant_mxfp4(x, sign)

    for result, expected in zip(actual, repeated):
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[1], rtn[1], atol=0, rtol=0)
    torch.testing.assert_close(actual[3], rtn[3], atol=0, rtol=0)
    if use_sr_row:
        assert not torch.equal(actual[0], rtn[0])
    else:
        torch.testing.assert_close(actual[0], rtn[0], atol=0, rtol=0)
    if use_sr_transposed:
        assert not torch.equal(actual[2], rtn[2])
    else:
        torch.testing.assert_close(actual[2], rtn[2], atol=0, rtol=0)


@requires_gfx950
def test_dual_layout_quant_mxfp4_assigns_adjacent_nonoverlapping_streams():
    sign = torch.ones(_HADAMARD_SIZE, dtype=torch.float32, device="cuda")
    h16 = _hadamard16(sign.device)
    block_h16 = torch.block_diag(h16, h16, h16, h16)
    x = torch.eye(64, dtype=torch.float32, device="cuda") + block_h16
    counters_per_layout = x.numel() // 8

    row_only = dual_layout_quant_mxfp4(
        x,
        sign,
        use_sr_row=True,
        philox_seed=1234,
        philox_offset=17,
    )
    transposed_only = dual_layout_quant_mxfp4(
        x,
        sign,
        use_sr_transposed=True,
        philox_seed=1234,
        philox_offset=17,
    )
    both = dual_layout_quant_mxfp4(
        x,
        sign,
        use_sr_row=True,
        use_sr_transposed=True,
        philox_seed=1234,
        philox_offset=17,
    )
    shifted_row = dual_layout_quant_mxfp4(
        x,
        sign,
        use_sr_row=True,
        philox_seed=1234,
        philox_offset=17 + counters_per_layout,
    )

    torch.testing.assert_close(row_only[1], transposed_only[3], atol=0, rtol=0)
    torch.testing.assert_close(row_only[0], transposed_only[2], atol=0, rtol=0)
    torch.testing.assert_close(both[0], row_only[0], atol=0, rtol=0)
    torch.testing.assert_close(both[2], shifted_row[0], atol=0, rtol=0)
    assert not torch.equal(both[0], both[2])


@requires_gfx950
def test_dual_layout_quant_mxfp4_does_not_reuse_counters_across_tiles():
    torch.manual_seed(31)
    tile = torch.randn((32, 32), dtype=torch.bfloat16, device="cuda")
    x = tile.repeat(2, 3)
    sign = torch.ones(_HADAMARD_SIZE, dtype=torch.bfloat16, device="cuda")

    row, scales, _, _ = dual_layout_quant_mxfp4(
        x,
        sign,
        use_sr_row=True,
        philox_seed=1234,
        philox_offset=(1 << 32) + 17,
    )

    torch.testing.assert_close(scales[:32, :1], scales[32:, :1], atol=0, rtol=0)
    torch.testing.assert_close(scales[:, :1], scales[:, 1:2], atol=0, rtol=0)
    assert not torch.equal(row[:32, :16], row[32:, :16])
    assert not torch.equal(row[:, :16], row[:, 16:32])


def test_dual_layout_quant_mxfp4_validates_input_contract():
    sign = torch.ones(_HADAMARD_SIZE, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="must be 2-D"):
        dual_layout_quant_mxfp4(torch.empty(32, dtype=torch.bfloat16), sign)
    with pytest.raises(TypeError, match="torch.bfloat16 or torch.float32"):
        dual_layout_quant_mxfp4(torch.empty((32, 32), dtype=torch.float16), sign)
    with pytest.raises(ValueError, match="block_size must be 32"):
        dual_layout_quant_mxfp4(
            torch.empty((32, 32), dtype=torch.bfloat16),
            sign,
            block_size=16,
        )
    with pytest.raises(ValueError, match="g must be 16"):
        dual_layout_quant_mxfp4(
            torch.empty((32, 32), dtype=torch.bfloat16),
            sign,
            g=32,
        )


def test_mxfp4_fused_layout_capability_checks():
    assert mxfp4_scale_swizzle_supported(32, 8)
    assert not mxfp4_scale_swizzle_supported(16, 8)
    assert mxfp4_data_shuffle_supported(16, 32)
    assert not mxfp4_data_shuffle_supported(16, 16)


@pytest.mark.parametrize(
    "use_sr_row,use_sr_transposed,expected_row_delta,expected_transposed_delta",
    [
        (False, False, None, None),
        (True, False, 0, None),
        (False, True, None, 0),
        (True, True, 0, 1),
    ],
)
def test_philox_streams_pack_only_enabled_layouts(
    use_sr_row,
    use_sr_transposed,
    expected_row_delta,
    expected_transposed_delta,
):
    M, N = 64, 96
    base = 17
    counters_per_layout = M * N // 8
    seed = 1234 if use_sr_row or use_sr_transposed else None

    actual_seed, row_offset, transposed_offset = _philox_streams(
        M,
        N,
        use_sr_row,
        use_sr_transposed,
        seed,
        base if seed is not None else 0,
    )

    assert actual_seed == (seed or 0)
    assert row_offset == (
        0
        if expected_row_delta is None
        else base + expected_row_delta * counters_per_layout
    )
    assert transposed_offset == (
        0
        if expected_transposed_delta is None
        else base + expected_transposed_delta * counters_per_layout
    )


@pytest.mark.parametrize(
    "use_sr_row,use_sr_transposed", [(True, False), (False, True), (True, True)]
)
def test_philox_streams_accept_inclusive_last_counter_boundary(
    use_sr_row,
    use_sr_transposed,
):
    M, N = 32, 32
    enabled_layouts = int(use_sr_row) + int(use_sr_transposed)
    reserved = enabled_layouts * M * N // 8
    last_valid_offset = _MAX_PHILOX_COUNTER - reserved + 1

    _philox_streams(
        M,
        N,
        use_sr_row,
        use_sr_transposed,
        1,
        last_valid_offset,
    )
    with pytest.raises(ValueError, match="leave room for enabled layouts"):
        _philox_streams(
            M,
            N,
            use_sr_row,
            use_sr_transposed,
            1,
            last_valid_offset + 1,
        )


@requires_gfx950
def test_dual_layout_quant_mxfp4_validates_shape_and_sign():
    good_sign = torch.ones(_HADAMARD_SIZE, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError, match="must be non-zero"):
        dual_layout_quant_mxfp4(
            torch.empty((0, 32), dtype=torch.bfloat16, device="cuda"), good_sign
        )
    with pytest.raises(ValueError, match="both be divisible by 32"):
        dual_layout_quant_mxfp4(
            torch.empty((32, 48), dtype=torch.bfloat16, device="cuda"), good_sign
        )
    with pytest.raises(ValueError, match="cannot be swizzled"):
        dual_layout_quant_mxfp4(
            torch.empty((32, 32), dtype=torch.bfloat16, device="cuda"),
            good_sign,
            swizzle_scale=True,
        )
    with pytest.raises(ValueError, match="cannot be shuffled"):
        dual_layout_quant_mxfp4(
            torch.empty((32, 32), dtype=torch.bfloat16, device="cuda"),
            good_sign,
            shuffle_col=True,
        )
    noncontiguous = torch.empty((32, 64), dtype=torch.bfloat16, device="cuda")[:, ::2]
    with pytest.raises(ValueError, match="must be contiguous"):
        dual_layout_quant_mxfp4(noncontiguous, good_sign)
    with pytest.raises(TypeError, match="sign_vector must have dtype"):
        dual_layout_quant_mxfp4(
            torch.empty((32, 32), dtype=torch.bfloat16, device="cuda"),
            good_sign.to(torch.float16),
        )
    with pytest.raises(ValueError, match="must contain 16 elements"):
        dual_layout_quant_mxfp4(
            torch.empty((32, 32), dtype=torch.bfloat16, device="cuda"),
            good_sign[:8],
        )
    invalid_sign = good_sign.clone()
    invalid_sign[0] = 0
    with pytest.raises(ValueError, match=r"finite and equal to \+1 or -1"):
        dual_layout_quant_mxfp4(
            torch.empty((32, 32), dtype=torch.bfloat16, device="cuda"),
            invalid_sign,
        )

    with torch.inference_mode():
        inference_sign = torch.ones(_HADAMARD_SIZE, dtype=torch.bfloat16, device="cuda")
    dual_layout_quant_mxfp4(
        torch.empty((32, 32), dtype=torch.bfloat16, device="cuda"),
        inference_sign,
    )

    mutable_sign = good_sign.clone()
    dual_layout_quant_mxfp4(
        torch.empty((32, 32), dtype=torch.bfloat16, device="cuda"), mutable_sign
    )
    mutable_sign[0] = 0
    with pytest.raises(ValueError, match=r"finite and equal to \+1 or -1"):
        dual_layout_quant_mxfp4(
            torch.empty((32, 32), dtype=torch.bfloat16, device="cuda"), mutable_sign
        )


@requires_gfx950
def test_dual_layout_quant_mxfp4_validates_philox_contract():
    x = torch.empty((32, 32), dtype=torch.bfloat16, device="cuda")
    sign = torch.ones(_HADAMARD_SIZE, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="philox_seed is required"):
        dual_layout_quant_mxfp4(x, sign, use_sr_row=True)
    with pytest.raises(TypeError, match="must be integers"):
        dual_layout_quant_mxfp4(x, sign, use_sr_row=True, philox_seed=1.5)
    with pytest.raises(ValueError, match="philox_seed must be"):
        dual_layout_quant_mxfp4(x, sign, use_sr_row=True, philox_seed=-1)
    with pytest.raises(ValueError, match="leave room for enabled layouts"):
        dual_layout_quant_mxfp4(
            x,
            sign,
            use_sr_transposed=True,
            philox_seed=1,
            philox_offset=_MAX_PHILOX_COUNTER - x.numel() // 8 + 2,
        )
    with pytest.raises(ValueError, match="only valid"):
        dual_layout_quant_mxfp4(x, sign, philox_seed=1)


def test_dual_layout_quant_mxfp4_rejects_unsupported_arch(monkeypatch):
    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx942")
    x = torch.empty((32, 32), dtype=torch.bfloat16)
    sign = torch.ones(_HADAMARD_SIZE, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires gfx950"):
        dual_layout_quant_mxfp4(x, sign)


@requires_gfx950
def test_dual_layout_quant_mxfp4_requires_published_config(monkeypatch):
    def missing_config(path, required=True):
        assert required
        assert path.endswith("/gfx950/triton/fusions/dual_layout_mxfp4/DEFAULT.json")
        raise FileNotFoundError(path)

    monkeypatch.setattr(
        "aiter.ops.triton.quant.dual_layout_mxfp4.load_config_json", missing_config
    )
    x = torch.zeros((32, 32), dtype=torch.bfloat16, device="cuda")
    sign = torch.ones(_HADAMARD_SIZE, dtype=torch.bfloat16, device=x.device)
    with pytest.raises(FileNotFoundError, match="dual_layout_mxfp4/DEFAULT.json"):
        dual_layout_quant_mxfp4(x, sign)


@requires_gfx950
def test_dual_layout_quant_mxfp4_graph_capture():
    x = torch.randn((32, 32), dtype=torch.bfloat16, device="cuda")
    sign = _signs(x.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = dual_layout_quant_mxfp4(x, sign)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = dual_layout_quant_mxfp4(x, sign)
    graph.replay()
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
