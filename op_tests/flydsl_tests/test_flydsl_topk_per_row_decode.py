# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL coverage for per-row decode TopK.

Run with:
    pytest op_tests/flydsl_tests/test_flydsl_topk_per_row_decode.py -q
"""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops import topk as public_topk_impl
from aiter.ops.flydsl.utils import is_flydsl_available

torch.set_default_device("cuda")

pytestmark = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950") or not is_flydsl_available(),
    reason="CDNA (gfx942/gfx950) + FlyDSL required for decode TopK",
)
_GFX950_ONLY = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="public FlyDSL dispatch is gfx950-only"
)


def run_torch(
    logits: torch.Tensor,
    context_lens: list[int],
    next_n: int,
    top_k: int,
    stable: bool,
) -> torch.Tensor:
    rows = []
    for row in range(logits.shape[0]):
        row_len = min(
            logits.shape[1],
            max(
                0,
                context_lens[row // next_n] - next_n + row % next_n + 1,
            ),
        )
        valid_k = min(row_len, top_k)
        padded = torch.full((top_k,), -1, dtype=torch.int64, device=logits.device)
        if stable:
            selected = torch.argsort(
                logits[row, :row_len], descending=True, stable=True
            )[:valid_k]
            padded[:valid_k] = torch.sort(selected).values
        elif valid_k > 0:
            padded[:valid_k] = torch.topk(
                logits[row, :row_len], valid_k, sorted=False
            ).indices
        rows.append(padded)
    return torch.stack(rows).to(torch.int32)


def _run_public(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    top_k: int,
    stable: bool,
) -> None:
    public_topk_impl.top_k_per_row_decode(
        logits,
        1,
        seq_lens,
        output,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        top_k,
        stable,
    )


@pytest.mark.parametrize(
    "arch,stable,top_k,rows,width,expected",
    [
        ("gfx942", True, 2048, 1, 65_536, False),
        ("gfx950", True, 128, 1, 65_536, False),
        ("gfx950", True, 4096, 1, 65_536, True),
        ("gfx950", False, 2048, 1, 131_071, False),
        ("gfx950", False, 2048, 32, 131_072, True),
        ("gfx950", True, 2048, 128, 20_000, True),
        ("gfx950", True, 2048, 129, 20_000, False),
        ("gfx950", True, 2048, 1, 32_767, False),
        ("gfx950", True, 2048, 16, 32_768, True),
        ("gfx950", True, 2048, 17, 32_768, False),
        ("gfx950", True, 2048, 32, 65_536, True),
        ("gfx950", True, 2048, 33, 65_536, False),
    ],
)
def test_public_topk_decode_gate(arch, stable, top_k, rows, width, expected):
    """Check architecture, K, width, and row-count dispatch boundaries."""
    supported = arch in public_topk_impl._FLYDSL_TOPK_DECODE_GATES
    actual = supported and public_topk_impl._flydsl_topk_decode_shape_supported(
        arch,
        stable,
        width,
        rows,
        top_k,
    )
    assert actual is expected


@pytest.mark.parametrize(
    "stable,top_k,rows,width,seq_len,use_flydsl",
    [
        (True, 2048, 128, 20_000, 20_000, True),
        (True, 2048, 4, 20_000, 1024, True),
        (True, 512, 16, 32_768, 32_768, True),
        (True, 2048, 4, 65_537, 65_533, True),
        (False, 4096, 32, 131_072, 131_072, True),
        (True, 2048, 17, 32_768, 32_768, False),
        (True, 256, 1, 65_536, 65_536, False),
        (True, 2048, 1, 32_767, 32_767, False),
    ],
)
@_GFX950_ONLY
def test_public_topk_decode_e2e(stable, top_k, rows, width, seq_len, use_flydsl):
    """Check both FlyDSL kernels and HIP fallbacks against the Torch reference."""
    torch.manual_seed(0)
    logits = torch.randn((rows, width), dtype=torch.float32, device="cuda")
    seq_lens = torch.full((rows,), seq_len, dtype=torch.int32, device="cuda")
    output = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    reference = run_torch(logits, [seq_len] * rows, 1, top_k, stable)

    assert (
        public_topk_impl._should_use_flydsl_topk_decode(
            logits,
            rows,
            top_k,
            stable,
        )
        is use_flydsl
    )
    _run_public(logits, seq_lens, output, top_k, stable)
    torch.cuda.synchronize()
    if stable:
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    else:
        output_values = (
            torch.gather(logits, 1, output.to(torch.int64)).sort(dim=1).values
        )
        reference_values = (
            torch.gather(logits, 1, reference.to(torch.int64)).sort(dim=1).values
        )
        torch.testing.assert_close(
            output_values,
            reference_values,
            rtol=0,
            atol=0,
        )
    if seq_len < top_k:
        assert torch.all(output[:, seq_len:] == -1)


@_GFX950_ONLY
def test_public_topk_decode_cuda_graph():
    """Check capture and replay of the long-row multi-kernel path."""
    rows, width, top_k = 1, 65_536, 2048
    torch.manual_seed(0)
    logits = torch.randn((rows, width), dtype=torch.float32, device="cuda")
    seq_lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    output = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    reference = run_torch(logits, [width], 1, top_k, True)

    _run_public(logits, seq_lens, output, top_k, True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_public(logits, seq_lens, output, top_k, True)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
