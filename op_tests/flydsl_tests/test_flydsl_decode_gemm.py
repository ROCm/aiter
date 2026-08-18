# SPDX-License-Identifier: MIT

"""Runtime correctness tests for the public BF16 decode GEMM operation."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("flydsl")

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gemm_kernels import (
    ActivationSource,
    BlockMfmaDecodeConfig,
    ContractionMode,
    ReductionMode,
    WaveDecodeConfig,
    gemm_decode_bf16,
)

ARCH = get_gfx_runtime()
SUPPORTED_ARCHS = ("gfx942", "gfx950")
ATOL = 0.125
RTOL = 0.01

pytestmark = pytest.mark.skipif(
    ARCH not in SUPPORTED_ARCHS,
    reason="BF16 decode GEMM requires gfx942 or gfx950",
)


def _inputs(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.arange(m, device="cuda", dtype=torch.int32)[:, None]
    col = torch.arange(n, device="cuda", dtype=torch.int32)[:, None]
    red = torch.arange(k, device="cuda", dtype=torch.int32)[None, :]
    a = (((row * 7 + red * 3) % 23) - 11).to(torch.float32).div_(16).bfloat16()
    b = (((col * 5 + red * 7) % 29) - 14).to(torch.float32).div_(16).bfloat16()
    return a, b


def _bias(n: int) -> torch.Tensor:
    return (
        (((torch.arange(n, device="cuda", dtype=torch.int32) * 3) % 17) - 8)
        .to(torch.float32)
        .div_(16)
        .bfloat16()
    )


def _reference(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    product = a.float() @ b.float().T
    if bias is not None:
        product = product + bias.float()
    return product.bfloat16()


def _output(m: int, n: int) -> torch.Tensor:
    return torch.full((m, n), torch.nan, device="cuda", dtype=torch.bfloat16)


def _assert_output(
    output: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)


def _block_config(
    source: ActivationSource,
    *,
    columns_per_wave: int,
    persistent_n: bool = False,
) -> BlockMfmaDecodeConfig:
    return BlockMfmaDecodeConfig(
        waves_per_workgroup=4,
        columns_per_wave=columns_per_wave,
        activation_source=source,
        b_load_width=8,
        k_unroll=2,
        prefetch_stages=2 if ARCH == "gfx950" else 1,
        persistent_n=persistent_n,
        workgroups_per_cu=1,
        waves_per_eu=2 if ARCH == "gfx950" else 0,
    )


def _wave_config(m: int, k: int) -> WaveDecodeConfig:
    return WaveDecodeConfig(
        m_per_wave=m,
        n_per_wave=1,
        kvec=2,
        prefetch_depth=0,
        waves_per_eu=4,
        reduction=ReductionMode.DPP if k % 2 == 0 else ReductionMode.BPERMUTE,
        contraction=(
            ContractionMode.DOT2_BF16
            if ARCH == "gfx950"
            else ContractionMode.SCALAR_F32
        ),
    )


def _run_config(
    m: int,
    n: int,
    k: int,
    config: BlockMfmaDecodeConfig,
    *,
    with_bias: bool = False,
) -> None:
    a, b = _inputs(m, n, k)
    bias = _bias(n) if with_bias else None
    output = _output(m, n)
    returned = gemm_decode_bf16(
        a,
        b,
        output,
        config,
        bias=bias,
    )
    torch.cuda.synchronize()
    assert returned is output
    _assert_output(output, _reference(a, b, bias))


def test_wave_public_path_no_bias() -> None:
    m, n, k = 1, 64, 128
    a, b = _inputs(m, n, k)
    output = _output(m, n)
    returned = gemm_decode_bf16(a, b, output, _wave_config(m, k))
    torch.cuda.synchronize()
    assert returned is output
    _assert_output(output, _reference(a, b))


def test_wave_public_path_bias_and_odd_n_and_k_tails() -> None:
    m, n, k = 5, 65, 257
    a, b = _inputs(m, n, k)
    bias = _bias(n)
    output = _output(m, n)
    returned = gemm_decode_bf16(
        a,
        b,
        output,
        _wave_config(m, k),
        bias=bias,
    )
    torch.cuda.synchronize()
    assert returned is output
    _assert_output(output, _reference(a, b, bias))


def test_block_mfma_global() -> None:
    _run_config(
        3,
        65,
        257,
        _block_config(ActivationSource.GLOBAL, columns_per_wave=2),
        with_bias=True,
    )


def test_block_mfma_full_lds_k_padding_and_n_boundary() -> None:
    _run_config(
        5,
        17,
        129,
        _block_config(ActivationSource.FULL_LDS, columns_per_wave=1),
    )


def test_block_mfma_persistent_n_multiple_turns_and_partial_group() -> None:
    _run_config(
        3,
        5001,
        257,
        _block_config(
            ActivationSource.FULL_LDS,
            columns_per_wave=1,
            persistent_n=True,
        ),
        with_bias=True,
    )


def test_block_mfma_graph_replay_on_non_default_stream() -> None:
    m, n, k = 3, 65, 257
    config = _block_config(ActivationSource.GLOBAL, columns_per_wave=2)
    a, b = _inputs(m, n, k)
    output = _output(m, n)
    reference = _reference(a, b)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())

    # Compile and warm the exact configured launch before graph capture.
    gemm_decode_bf16(a, b, output, config, stream=side)
    side.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        gemm_decode_bf16(a, b, output, config, stream=side)
    side.synchronize()

    output.fill_(torch.nan)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        graph.replay()
    side.synchronize()
    _assert_output(output, reference)


def test_output_must_not_overlap_bias() -> None:
    m, n, k = 3, 64, 128
    a, b = _inputs(m, n, k)
    output = _output(m, n)
    with pytest.raises(ValueError, match="overlap bias"):
        gemm_decode_bf16(a, b, output, _wave_config(m, k), bias=output[0])
