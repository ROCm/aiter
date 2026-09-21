# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton as grouped_gemm_module
from aiter.ops.gradlib import hipb_grouped_mm, hipb_multistream_mm
from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton import (
    _registered_host_cu_seqlens,
    clear_registered_host_cu_seqlens,
    grouped_gemm,
    register_host_cu_seqlens,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _offsets(counts, device="cuda"):
    return torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()],
        dtype=torch.int64,
        device=device,
    )


@pytest.mark.parametrize("counts", ([7, 0, 13], [1, 19, 4]))
def test_hipb_grouped_forward_matches_torch(counts):
    torch.manual_seed(17)
    dtype = torch.float16
    total, experts, k, n = sum(counts), len(counts), 64, 96
    a = torch.randn(total, k, device="cuda", dtype=dtype)
    b = torch.randn(experts, k, n, device="cuda", dtype=dtype)
    out = torch.empty(total, n, device="cuda", dtype=dtype)

    hipb_grouped_mm(a, b.transpose(1, 2).contiguous(), _offsets(counts), out)

    expected = torch.cat(
        [chunk @ b[index] for index, chunk in enumerate(torch.split(a, counts))]
    )
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("offsets_device", ("cpu", "cuda"))
def test_hipb_multistream_wgrad_matches_torch(offsets_device):
    torch.manual_seed(23)
    dtype = torch.bfloat16
    counts = [5, 0, 11]
    total, experts, k, n = sum(counts), len(counts), 64, 80
    a = torch.randn(total, k, device="cuda", dtype=dtype)
    b = torch.randn(total, n, device="cuda", dtype=dtype)
    out = torch.full((experts, k, n), 7.0, device="cuda", dtype=dtype)

    hipb_multistream_mm(a, b, _offsets(counts, offsets_device), out, True)

    expected = torch.stack(
        [
            (
                a_chunk.T @ b_chunk
                if count
                else torch.zeros(k, n, device="cuda", dtype=dtype)
            )
            for count, a_chunk, b_chunk in zip(
                counts, torch.split(a, counts), torch.split(b, counts)
            )
        ]
    )
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("input_dtype", "output_dtype"),
    (
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ),
)
@pytest.mark.parametrize("offsets_device", ("cpu", "cuda"))
def test_hipb_multistream_forward_matches_torch(
    input_dtype, output_dtype, offsets_device
):
    torch.manual_seed(29)
    counts = [7, 0, 13]
    total, experts, k, n = sum(counts), len(counts), 64, 96
    a = torch.randn(total, k, device="cuda", dtype=input_dtype)
    b = torch.randn(experts, k, n, device="cuda", dtype=input_dtype)
    out = torch.empty(total, n, device="cuda", dtype=output_dtype)

    hipb_multistream_mm(a, b, _offsets(counts, offsets_device), out)

    expected = torch.cat(
        [
            chunk.to(output_dtype) @ b[index].to(output_dtype)
            for index, chunk in enumerate(torch.split(a, counts))
        ]
    )
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("offsets_device", ("cpu", "cuda"))
def test_hipb_multistream_transposed_weight_matches_torch(offsets_device):
    torch.manual_seed(31)
    dtype = torch.bfloat16
    counts = [9, 0, 15]
    total, experts, k, n = sum(counts), len(counts), 64, 80
    a = torch.randn(total, k, device="cuda", dtype=dtype)
    b_transposed = torch.randn(experts, n, k, device="cuda", dtype=dtype)
    out = torch.empty(total, n, device="cuda", dtype=dtype)

    hipb_multistream_mm(
        a,
        b_transposed,
        _offsets(counts, offsets_device),
        out,
        False,
        None,
        True,
    )

    expected = torch.cat(
        [
            chunk @ b_transposed[index].T
            for index, chunk in enumerate(torch.split(a, counts))
        ]
    )
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


def test_host_offset_cache_rejects_reused_storage_alias():
    offsets = _offsets([7, 13])
    host_offsets = offsets.cpu()
    register_host_cu_seqlens(offsets, host_offsets)

    assert torch.equal(_registered_host_cu_seqlens(offsets), host_offsets)
    alias = offsets.detach()
    assert alias is not offsets
    assert alias.data_ptr() == offsets.data_ptr()
    assert _registered_host_cu_seqlens(alias) is None

    clear_registered_host_cu_seqlens(offsets)


@pytest.mark.parametrize("invalid_option", ("bias", "scatter_idx", "transposed_b"))
def test_grouped_wgrad_rejects_incompatible_options(invalid_option):
    a = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4, 6, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, 8, 6, device="cuda", dtype=torch.bfloat16)
    kwargs = {}
    if invalid_option == "bias":
        kwargs["bias"] = torch.zeros(1, 6, device="cuda", dtype=torch.bfloat16)
    elif invalid_option == "scatter_idx":
        kwargs["scatter_idx"] = torch.arange(4, device="cuda", dtype=torch.int32)
    else:
        kwargs["B_is_transposed"] = True

    with pytest.raises(ValueError):
        grouped_gemm(
            a,
            b,
            _offsets([4]),
            out=out,
            A_is_transposed=True,
            **kwargs,
        )


def test_hipb_grouped_wgrad_zeroes_all_empty_experts():
    torch.manual_seed(37)
    dtype = torch.bfloat16
    counts = [0, 0, 0]
    total, experts, k, n = sum(counts), len(counts), 64, 80
    a = torch.randn(total, k, device="cuda", dtype=dtype)
    b = torch.randn(total, n, device="cuda", dtype=dtype)
    out = torch.full((experts, k, n), 7.0, device="cuda", dtype=dtype)

    hipb_grouped_mm(a, b, _offsets(counts), out, True)

    expected = torch.stack(
        [
            (
                a_chunk.T @ b_chunk
                if count
                else torch.zeros(k, n, device="cuda", dtype=dtype)
            )
            for count, a_chunk, b_chunk in zip(
                counts, torch.split(a, counts), torch.split(b, counts)
            )
        ]
    )
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


def test_hipb_grouped_scratch_reuse_across_streams():
    torch.manual_seed(41)
    dtype = torch.float16
    counts = [7, 13]
    total, experts, k, n = sum(counts), len(counts), 64, 96
    a = torch.randn(total, k, device="cuda", dtype=dtype)
    b = torch.randn(experts, k, n, device="cuda", dtype=dtype)
    expected = torch.cat(
        [chunk @ b[index] for index, chunk in enumerate(torch.split(a, counts))]
    )
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    outputs = []

    for iteration in range(8):
        out = torch.empty_like(expected)
        with torch.cuda.stream(streams[iteration % 2]):
            hipb_grouped_mm(
                a,
                b.transpose(1, 2).contiguous(),
                _offsets(counts),
                out,
            )
        outputs.append(out)

    torch.cuda.synchronize()
    for out in outputs:
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


def test_hipb_grouped_compile_schema_and_execution():
    schema = str(torch.ops.aiter.hipb_grouped_mm.default._schema)
    assert "Tensor(a3!) out" in schema
    assert schema.endswith("-> ()")

    counts = [4, 4]
    a = torch.randn(8, 32, device="cuda", dtype=torch.float16)
    b = torch.randn(2, 32, 48, device="cuda", dtype=torch.float16)
    offsets = _offsets(counts)

    @torch.compile(fullgraph=True)
    def run_grouped(a, b, offsets, out):
        hipb_grouped_mm(a, b.transpose(1, 2).contiguous(), offsets, out)
        return out

    out = torch.empty(8, 48, device="cuda", dtype=torch.float16)
    actual = run_grouped(a, b, offsets, out)
    expected = torch.cat(
        [chunk @ b[index] for index, chunk in enumerate(torch.split(a, counts))]
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_hipb_validation_rejects_invalid_metadata_and_bias():
    a = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 48, 32, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(8, 48, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="nondecreasing"):
        hipb_multistream_mm(
            a,
            b.transpose(1, 2).contiguous(),
            torch.tensor([0, 6, 5, 8], dtype=torch.int64),
            out,
        )

    with pytest.raises(RuntimeError, match="device and dtype"):
        hipb_grouped_mm(
            a,
            b,
            _offsets([4, 4]),
            out,
            bias=torch.zeros(2, 48, device="cuda", dtype=torch.float16),
        )

    with pytest.raises(RuntimeError, match="does not support bias"):
        hipb_multistream_mm(
            a,
            torch.randn(8, 48, device="cuda", dtype=torch.bfloat16),
            _offsets([4, 4], "cpu"),
            torch.empty(2, 32, 48, device="cuda", dtype=torch.bfloat16),
            True,
            torch.zeros(2, 48, device="cuda", dtype=torch.bfloat16),
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_hipb_grouped_context_is_device_scoped():
    for device_index in (0, 1):
        device = f"cuda:{device_index}"
        counts = [3, 5]
        a = torch.randn(8, 32, device=device, dtype=torch.float16)
        b = torch.randn(2, 32, 48, device=device, dtype=torch.float16)
        out = torch.empty(8, 48, device=device, dtype=torch.float16)
        offsets = _offsets(counts, device)

        hipb_grouped_mm(a, b.transpose(1, 2).contiguous(), offsets, out)

        expected = torch.cat(
            [chunk @ b[index] for index, chunk in enumerate(torch.split(a, counts))]
        )
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


def test_configured_triton_launch_has_single_group_size(monkeypatch):
    class FakeKernel:
        kwargs = None

        def __getitem__(self, _grid):
            def launch(*_args, **kwargs):
                self.kwargs = kwargs

            return launch

    fake_kernel = FakeKernel()
    config = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 2,
    }
    monkeypatch.setattr(grouped_gemm_module, "_grouped_gemm_kernel", fake_kernel)
    monkeypatch.setattr(
        grouped_gemm_module, "get_grouped_gemm_fwd_config", lambda *_args: config
    )
    monkeypatch.setattr(grouped_gemm_module, "_use_qwen3_tuned_configs", lambda: False)

    a = torch.randn(8, 32, device="cuda", dtype=torch.float16)
    b = torch.randn(2, 32, 48, device="cuda", dtype=torch.float16)
    grouped_gemm_module._grouped_gemm_triton(a, b, _offsets([4, 4]))

    assert fake_kernel.kwargs["GROUP_SIZE_M"] == 4


def test_triton_dispatch_unwraps_local_tensors(monkeypatch):
    class LocalTensorWrapper:
        def __init__(self, local):
            self.local = local

        def to_local(self):
            return self.local

    local_a = torch.randn(8, 32, device="cuda", dtype=torch.float16)
    local_b = torch.randn(2, 32, 48, device="cuda", dtype=torch.float16)
    local_offsets = _offsets([4, 4])
    local_out = torch.empty(8, 48, device="cuda", dtype=torch.float16)
    wrapped_out = LocalTensorWrapper(local_out)

    def fake_triton(a, b, offsets, out, *_args):
        assert a is local_a
        assert b is local_b
        assert offsets is local_offsets
        assert out is local_out
        return out

    monkeypatch.setenv("SONIC_MOE_GROUPED_GEMM_BACKEND", "triton")
    monkeypatch.setattr(grouped_gemm_module, "_grouped_gemm_triton", fake_triton)
    result = grouped_gemm_module.grouped_gemm(
        LocalTensorWrapper(local_a),
        LocalTensorWrapper(local_b),
        LocalTensorWrapper(local_offsets),
        out=wrapped_out,
    )

    assert result is wrapped_out
