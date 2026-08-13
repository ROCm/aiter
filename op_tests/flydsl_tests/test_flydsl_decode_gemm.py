# SPDX-License-Identifier: MIT

"""Runtime correctness tests for the public BF16 decode GEMM operation."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("flydsl")

import flydsl.expr as fx

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gemm_kernels import (
    ActivationSource,
    BlockMfmaDecodeConfig,
    gemm_decode_bf16,
    gemm_decode_bf16_configured,
)


ARCH = get_gfx_runtime()
SUPPORTED_ARCHS = ("gfx942", "gfx950")
ATOL = 0.125
RTOL = 0.01

pytestmark = pytest.mark.skipif(
    ARCH not in SUPPORTED_ARCHS,
    reason="BF16 decode GEMM requires gfx942 or gfx950",
)


def _inputs(
    m: int, n: int, k: int, *, with_bias: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    row = torch.arange(m, device="cuda", dtype=torch.int32)[:, None]
    col = torch.arange(n, device="cuda", dtype=torch.int32)[:, None]
    red = torch.arange(k, device="cuda", dtype=torch.int32)[None, :]
    a = (((row * 7 + red * 3) % 23) - 11).to(torch.float32).div_(16).bfloat16()
    b = (((col * 5 + red * 7) % 29) - 14).to(torch.float32).div_(16).bfloat16()
    bias = None
    if with_bias:
        bias = (
            ((torch.arange(n, device="cuda", dtype=torch.int32) * 3) % 17) - 8
        ).to(torch.float32).div_(16).bfloat16()
    return a, b, bias


def _reference(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    product = a.float() @ b.float().T
    output = product.bfloat16()
    if bias is not None:
        # The public decode epilogue adds BF16 bias to the rounded BF16 GEMM output.
        output.add_(bias)
    return output


def _output(m: int, n: int) -> torch.Tensor:
    return torch.full(
        (m, n), torch.nan, device="cuda", dtype=torch.bfloat16
    )


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


def _run_configured(
    m: int,
    n: int,
    k: int,
    config: BlockMfmaDecodeConfig,
) -> None:
    a, b, _ = _inputs(m, n, k)
    output = _output(m, n)
    returned = gemm_decode_bf16_configured(
        a, b, output, m, n, k, config, arch=ARCH
    )
    torch.cuda.synchronize()
    assert returned is output
    _assert_output(output, _reference(a, b))


def test_wave_public_path_no_bias() -> None:
    m, n, k = 1, 64, 128
    a, b, _ = _inputs(m, n, k)
    output = _output(m, n)
    returned = gemm_decode_bf16(a, b, output, m, n, k)
    torch.cuda.synchronize()
    assert returned is output
    _assert_output(output, _reference(a, b))


def test_wave_public_path_bias_and_odd_boundaries() -> None:
    m, n, k = 5, 65, 257
    a, b, bias = _inputs(m, n, k, with_bias=True)
    output = _output(m, n)
    returned = gemm_decode_bf16(a, b, output, m, n, k, bias=bias)
    torch.cuda.synchronize()
    assert returned is output
    _assert_output(output, _reference(a, b, bias))


def test_block_mfma_global() -> None:
    _run_configured(
        3,
        65,
        257,
        _block_config(ActivationSource.GLOBAL, columns_per_wave=2),
    )


def test_block_mfma_full_lds_k_padding_and_n_boundary() -> None:
    _run_configured(
        5,
        17,
        129,
        _block_config(ActivationSource.FULL_LDS, columns_per_wave=1),
    )


def test_block_mfma_persistent_n_multiple_turns_and_partial_group() -> None:
    _run_configured(
        3,
        5001,
        257,
        _block_config(
            ActivationSource.FULL_LDS,
            columns_per_wave=1,
            persistent_n=True,
        ),
    )


def test_block_mfma_graph_replay_on_non_default_stream() -> None:
    m, n, k = 3, 65, 257
    config = _block_config(ActivationSource.GLOBAL, columns_per_wave=2)
    a, b, _ = _inputs(m, n, k)
    output = _output(m, n)
    reference = _reference(a, b)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())

    # Compile and warm the exact configured launch before graph capture.
    gemm_decode_bf16_configured(
        a, b, output, m, n, k, config, fx.Stream(side), arch=ARCH
    )
    side.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        gemm_decode_bf16_configured(
            a, b, output, m, n, k, config, fx.Stream(side), arch=ARCH
        )
    side.synchronize()

    output.fill_(torch.nan)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        graph.replay()
    side.synchronize()
    _assert_output(output, reference)
