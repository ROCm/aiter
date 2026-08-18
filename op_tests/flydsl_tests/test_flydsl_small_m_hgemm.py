# SPDX-License-Identifier: MIT

"""Runtime correctness tests for the public BF16 small-M HGEMM operation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

pytest.importorskip("flydsl")

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gemm_kernels import flydsl_small_m_hgemm

ARCH = get_gfx_runtime()
SUPPORTED_ARCHS = ("gfx942", "gfx950")
ATOL = 0.125
RTOL = 0.01
SPLIT_K_ATOL = 0.5
SPLIT_K_RTOL = 0.02

pytestmark = pytest.mark.skipif(
    ARCH not in SUPPORTED_ARCHS,
    reason="BF16 small-M HGEMM requires gfx942 or gfx950",
)


@dataclass(frozen=True)
class Case:
    name: str
    m: int
    n: int
    k: int
    tile_n: int
    tile_k: int
    split_k: int = 1
    block_n_warps: int = 2
    n_tile_repeat: int = 1
    persistent_n_tiles: int = 1
    b_to_lds: bool = False
    with_bias: bool = False


CASES = [
    Case(
        "lower-boundary",
        m=1,
        n=128,
        k=32,
        tile_n=128,
        tile_k=32,
        b_to_lds=True,
    ),
    Case(
        "n-repeat",
        m=7,
        n=192,
        k=128,
        tile_n=64,
        tile_k=64,
        block_n_warps=1,
        n_tile_repeat=2,
    ),
    Case(
        "split-k",
        m=7,
        n=128,
        k=256,
        tile_n=128,
        tile_k=64,
        split_k=2,
    ),
    Case(
        "persistent-bias",
        m=16,
        n=384,
        k=256,
        tile_n=128,
        tile_k=64,
        split_k=2,
        persistent_n_tiles=2,
        b_to_lds=True,
        with_bias=True,
    ),
]
if ARCH == "gfx942":
    CASES.append(
        Case(
            "vgpr-b",
            m=8,
            n=128,
            k=64,
            tile_n=16,
            tile_k=64,
            block_n_warps=1,
            b_to_lds=False,
        )
    )


def _inputs(
    case: Case,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    row = torch.arange(case.m, device="cuda", dtype=torch.int32)[:, None]
    col = torch.arange(case.n, device="cuda", dtype=torch.int32)[:, None]
    red = torch.arange(case.k, device="cuda", dtype=torch.int32)[None, :]
    a = (((row * 11 + red * 5) % 31) - 15).to(torch.float32).div_(16).bfloat16()
    b = (((col * 7 + red * 3) % 37) - 18).to(torch.float32).div_(16).bfloat16()
    bias = None
    if case.with_bias:
        bias = (
            (((torch.arange(case.n, device="cuda", dtype=torch.int32) * 5) % 19) - 9)
            .to(torch.float32)
            .div_(16)
            .bfloat16()
        )
    return a, b, bias


def _reference(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    output = a.float() @ b.float().T
    if bias is not None:
        output.add_(bias.float())
    return output.bfloat16()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_small_m_hgemm_matches_fp32_oracle(case: Case) -> None:
    a, b, bias = _inputs(case)
    output = torch.full(
        (case.m, case.n),
        torch.nan,
        device="cuda",
        dtype=torch.bfloat16,
    )
    returned = flydsl_small_m_hgemm(
        a,
        b,
        out=output,
        bias=bias,
        tile_n=case.tile_n,
        tile_k=case.tile_k,
        split_k=case.split_k,
        block_n_warps=case.block_n_warps,
        n_tile_repeat=case.n_tile_repeat,
        persistent_n_tiles=case.persistent_n_tiles,
        b_to_lds=case.b_to_lds,
    )
    torch.cuda.synchronize()

    assert returned is output
    assert torch.isfinite(output).all()
    atol, rtol = (SPLIT_K_ATOL, SPLIT_K_RTOL) if case.split_k > 1 else (ATOL, RTOL)
    torch.testing.assert_close(
        output,
        _reference(a, b, bias),
        atol=atol,
        rtol=rtol,
    )
