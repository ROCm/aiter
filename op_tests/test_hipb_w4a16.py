# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import dtypes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

N, K = 512, 1024


@pytest.fixture(scope="module", autouse=True)
def hipb_extension():
    aiter.hipb_create_extension()
    yield
    aiter.hipb_destroy_extension()


@pytest.fixture(scope="module")
def requires_w4a16(hipb_extension):
    if not aiter.hipb_w4a16_is_supported(dtypes.bf16, N, K, 128, has_zp=False):
        pytest.skip(
            "no hipBLASLt w4a16 support (headers without the w4a16 API "
            "or no kernels for this device)"
        )


def pack_int4_k_order(q: torch.Tensor, row_pad: int = 0) -> torch.Tensor:
    """[N, K] values in [0, 15] -> [N, K // 2] uint8, rows `row_pad` bytes apart extra."""
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)
    if not row_pad:
        return packed
    buf = torch.zeros(
        (packed.shape[0], packed.shape[1] + row_pad), dtype=torch.uint8, device=q.device
    )
    buf[:, : packed.shape[1]] = packed
    return buf[:, : packed.shape[1]]


def pack_zeros_along_n(zp: torch.Tensor) -> torch.Tensor:
    """[N, G] values in [0, 15] -> [ceil(N / 8), G] int32, row n in nibble n % 8."""
    n, g = zp.shape
    padded = torch.zeros(((n + 7) // 8 * 8, g), dtype=torch.int64, device=zp.device)
    padded[:n] = zp
    shifts = (4 * torch.arange(8, device=zp.device))[None, :, None]
    words = (padded.view(-1, 8, g) << shifts).sum(dim=1)
    return words.to(torch.uint32).view(torch.int32)


@pytest.mark.usefixtures("requires_w4a16")
@pytest.mark.parametrize("dtype", [dtypes.bf16, dtypes.fp16])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("has_zp", [False, True])
@pytest.mark.parametrize("m", [1, 7, 300])
@pytest.mark.parametrize("row_pad", [0, 128])
def test_hipb_w4a16_mm_matches_dequant_reference(dtype, group_size, has_zp, m, row_pad):
    """`row_pad` pads the rows of the weights, the activations and the output."""
    torch.manual_seed(0)
    q = torch.randint(0, 16, (N, K), device="cuda")
    scales = (torch.rand(N, K // group_size, device="cuda") * 0.02 + 0.005).to(dtype)
    zp = (
        torch.randint(0, 16, (N, K // group_size), device="cuda")
        if has_zp
        else torch.full((N, K // group_size), 8, device="cuda")
    )
    pad = row_pad // 2  # elements of a 16-bit row
    x = torch.randn(m, K + pad, device="cuda", dtype=dtype)[:, :K]
    out = torch.empty(m, N + pad, device="cuda", dtype=dtype)[:, :N]
    bias = torch.randn(N, device="cuda", dtype=dtype)

    w_q = pack_int4_k_order(q, row_pad)
    assert aiter.hipb_w4a16_is_supported(
        dtype,
        N,
        K,
        group_size,
        has_zp,
        lda=w_q.stride(0) * 2,
        ldb=x.stride(0),
        ldd=out.stride(0),
    )
    scale = aiter.hipb_w4a16_scale_buffer(
        scales, pack_zeros_along_n(zp) if has_zp else None
    )
    assert (
        aiter.hipb_w4a16_mm(x, w_q, scale, group_size, has_zp, bias=bias, out=out)
        is out
    )

    w = (
        q - zp.repeat_interleave(group_size, dim=1)
    ).float() * scales.float().repeat_interleave(group_size, dim=1)
    ref = x.float() @ w.t() + bias.float()
    assert out.dtype == dtype and out.shape == (m, N)
    torch.testing.assert_close(
        out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item()
    )


@pytest.mark.usefixtures("requires_w4a16")
def test_hipb_w4a16_mm_writes_into_out():
    x = torch.randn(4, K, device="cuda", dtype=dtypes.bf16)
    w_q = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
    scale = torch.full((N, K // 128), 0.01, device="cuda", dtype=dtypes.bf16)
    out = torch.empty(4, N, device="cuda", dtype=dtypes.bf16)
    assert aiter.hipb_w4a16_mm(x, w_q, scale, 128, out=out) is out


@pytest.mark.usefixtures("requires_w4a16")
@pytest.mark.parametrize("dtype", [dtypes.bf16, dtypes.fp16])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("has_zp", [False, True])
@pytest.mark.parametrize("m", [7, 300])
def test_hipb_w4a16_mm_signed_encoding(dtype, group_size, has_zp, m):
    """SIGNED weights (and zero-points) in [-8, 7], stored as two's-complement nibbles."""
    signed = aiter.HipblasLtInt4Encoding.SIGNED
    torch.manual_seed(0)
    q = torch.randint(-8, 8, (N, K), device="cuda")
    zp = (
        torch.randint(-8, 8, (N, K // group_size), device="cuda")
        if has_zp
        else torch.zeros((N, K // group_size), device="cuda", dtype=torch.int64)
    )
    scales = (torch.rand(N, K // group_size, device="cuda") * 0.02 + 0.005).to(dtype)
    x = torch.randn(m, K, device="cuda", dtype=dtype)

    w_q = pack_int4_k_order(q & 0xF)
    assert aiter.hipb_w4a16_is_supported(
        dtype, N, K, group_size, has_zp, int4_encoding=signed
    )
    scale = aiter.hipb_w4a16_scale_buffer(
        scales, pack_zeros_along_n(zp & 0xF) if has_zp else None
    )
    out = aiter.hipb_w4a16_mm(x, w_q, scale, group_size, has_zp, int4_encoding=signed)

    # Independent of the packing: dequantize the signed integers directly.
    w = (q - zp.repeat_interleave(group_size, dim=1)).float()
    w *= scales.float().repeat_interleave(group_size, dim=1)
    ref = x.float() @ w.t()
    torch.testing.assert_close(
        out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item()
    )


@pytest.mark.usefixtures("requires_w4a16")
@pytest.mark.parametrize("has_zp", [False, True])
@pytest.mark.parametrize("int4_encoding", list(aiter.HipblasLtInt4Encoding))
def test_hipb_w4a16_mm_out_op_contract(has_zp, int4_encoding):
    """The registered op writes a row-padded `out` in place and aliases nothing."""
    q = torch.randint(0, 16, (N, K), device="cuda")
    zp = torch.randint(0, 16, (N, K // 128), device="cuda")
    scales = torch.full((N, K // 128), 0.01, device="cuda", dtype=dtypes.bf16)
    scale = aiter.hipb_w4a16_scale_buffer(
        scales, pack_zeros_along_n(zp) if has_zp else None
    )
    x = torch.randn(7, K + 64, device="cuda", dtype=dtypes.bf16)[:, :K]
    out = torch.empty(7, N + 64, device="cuda", dtype=dtypes.bf16)[:, :N]
    torch.library.opcheck(
        torch.ops.aiter.hipb_w4a16_mm_out.default,
        (x, pack_int4_k_order(q, 128), scale, out, 128, has_zp, int(int4_encoding)),
    )


@pytest.mark.parametrize(
    "dtype, group_size, lda",
    [
        (dtypes.fp32, 128, K),  # activations hipBLASLt has no w4a16 kernel for
        (dtypes.bf16, 16, K),  # no scale mode for this group size
        (dtypes.bf16, 128, K + 4),  # lda must be a multiple of 8
    ],
)
def test_hipb_w4a16_is_supported_rejects_unexpressible_layers(dtype, group_size, lda):
    assert not aiter.hipb_w4a16_is_supported(dtype, N, K, group_size, False, lda=lda)


@pytest.mark.usefixtures("requires_w4a16")
def test_hipb_w4a16_mm_raises_for_unexpressible_layer():
    """A layer hipBLASLt rejects raises instead of aborting the process."""
    x = torch.randn(4, K, device="cuda", dtype=dtypes.bf16)
    # Row stride K / 2 + 2 bytes: lda = K + 4 int4 elements, not a multiple of 8.
    w_q = torch.zeros(N, K // 2 + 2, device="cuda", dtype=torch.uint8)[:, : K // 2]
    scale = torch.full((N, K // 128), 0.01, device="cuda", dtype=dtypes.bf16)
    with pytest.raises(RuntimeError, match="no w4a16 kernel"):
        aiter.hipb_w4a16_mm(x, w_q, scale, 128)


def test_hipb_w4a16_scale_buffer_rejects_non_16bit_scales():
    """Packed as bytes, fp32 scales would be read as pairs of 16-bit scales."""
    scales = torch.full((N, K // 128), 0.01, device="cuda", dtype=torch.float32)
    zeros = torch.zeros((N // 8, K // 128), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        aiter.hipb_w4a16_scale_buffer(scales, zeros)


def test_hipb_w4a16_scale_buffer_single_group_transposed():
    """One group per row: transposed [1, N] inputs are contiguous but not unit-stride."""
    scales = torch.rand((1, N), device="cuda", dtype=dtypes.bf16).t()
    zeros = torch.randint(
        0, 2**31 - 1, (1, N // 8), device="cuda", dtype=torch.int32
    ).t()
    dense = aiter.hipb_w4a16_scale_buffer(
        scales.clone(memory_format=torch.contiguous_format)
    )
    assert torch.equal(aiter.hipb_w4a16_scale_buffer(scales), dense)
    expected = aiter.hipb_w4a16_scale_buffer(
        scales.reshape(-1).clone().reshape(N, 1),
        zeros.reshape(-1).clone().reshape(N // 8, 1),
    )
    assert torch.equal(aiter.hipb_w4a16_scale_buffer(scales, zeros), expected)


@pytest.mark.usefixtures("requires_w4a16")
def test_hipb_w4a16_mm_rejects_unsupported_group_size():
    x = torch.randn(4, K, device="cuda", dtype=dtypes.bf16)
    w_q = torch.zeros(N, K // 2, device="cuda", dtype=torch.uint8)
    scale = torch.full((N, K // 16), 0.01, device="cuda", dtype=dtypes.bf16)
    with pytest.raises(RuntimeError, match="group_size must be 32, 64 or 128"):
        aiter.hipb_w4a16_mm(x, w_q, scale, 16)


@pytest.mark.usefixtures("requires_w4a16")
def test_hipb_w4a16_mm_concurrent_streams_with_workspace():
    """Launches on two streams, each needing a workspace, do not share scratch."""
    from aiter.ops.gradlib import _hipb_w4a16_workspace_size

    m, n, k, g = 128, 5120, 17408, 32  # a Stream-K pick that keeps partial sums
    dtype = dtypes.fp16
    torch.manual_seed(0)
    problems = []
    for _ in range(2):
        q = torch.randint(0, 16, (n, k), device="cuda")
        zp = torch.randint(0, 16, (n, k // g), device="cuda")
        scales = (torch.rand(n, k // g, device="cuda") * 0.02 + 0.005).to(dtype)
        x = torch.randn(m, k, device="cuda", dtype=dtype)
        w = (
            q - zp.repeat_interleave(g, dim=1)
        ).float() * scales.float().repeat_interleave(g, dim=1)
        problems.append(
            (
                x,
                pack_int4_k_order(q),
                aiter.hipb_w4a16_scale_buffer(scales, pack_zeros_along_n(zp)),
                x.float() @ w.t(),
            )
        )
    out0 = torch.empty(m, n, device="cuda", dtype=dtype)
    if (
        _hipb_w4a16_workspace_size(problems[0][0], problems[0][1], out0, g, True, 1)
        == 0
    ):
        pytest.skip("this hipBLASLt picks no workspace-using kernel for the shape")

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outs = [torch.empty(m, n, device="cuda", dtype=dtype) for _ in streams]
    torch.cuda.synchronize()
    # A shared workspace corrupts about 2% of outputs here, so iterate enough to see it.
    for _ in range(100):
        for stream, out, (x, w_q, scale, _) in zip(streams, outs, problems):
            with torch.cuda.stream(stream):
                aiter.hipb_w4a16_mm(x, w_q, scale, g, True, out=out)
        torch.cuda.synchronize()
        for out, (*_, ref) in zip(outs, problems):
            torch.testing.assert_close(
                out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item()
            )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.usefixtures("requires_w4a16")
def test_hipb_w4a16_rejects_other_current_device():
    """The launch stream comes from the current device, so it must be the tensors'."""
    x = torch.randn(4, K, device="cuda:0", dtype=dtypes.bf16)
    w_q = torch.zeros(N, K // 2, device="cuda:0", dtype=torch.uint8)
    scale = torch.full((N, K // 128), 0.01, device="cuda:0", dtype=dtypes.bf16)
    with torch.cuda.device(1):
        with pytest.raises(RuntimeError, match="current device"):
            aiter.hipb_w4a16_mm(x, w_q, scale, 128)
        with pytest.raises(RuntimeError, match="current device"):
            aiter.hipb_w4a16_is_supported(dtypes.bf16, N, K, 128, False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
