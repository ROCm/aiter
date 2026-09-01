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

_GFX = get_gfx()
pytestmark = pytest.mark.skipif(
    _GFX not in ("gfx942", "gfx950") or not is_flydsl_available(),
    reason="CDNA (gfx942/gfx950) + FlyDSL required for decode TopK",
)

_E2E_CASES = {
    "gfx942": [
        (True, 2048, 64, 20_000, 20_000, 1, False),
        (True, 2048, 4, 20_000, 1024, 1, True),
        (True, 512, 16, 131_072, 131_072, 1, False),
        (True, 2048, 4, 131_073, 131_069, 1, True),
        (False, 4096, 32, 524_288, 524_288, 1, True),
        pytest.param(
            True,
            2048,
            8,
            20_000,
            (4_099, 12_347),
            4,
            False,
            id="one-workgroup-mtp4",
        ),
        pytest.param(
            True,
            2048,
            8,
            131_072,
            (65_539, 131_065),
            4,
            True,
            id="multi-cta-mtp4",
        ),
    ],
    "gfx950": [
        (True, 2048, 128, 20_000, 20_000, 1, False),
        (True, 2048, 4, 20_000, 1024, 1, True),
        (True, 512, 16, 32_768, 32_768, 1, False),
        (True, 2048, 4, 65_537, 65_533, 1, True),
        (False, 4096, 32, 131_072, 131_072, 1, False),
        pytest.param(
            True,
            2048,
            8,
            20_000,
            (4_099, 12_347),
            4,
            False,
            id="one-workgroup-mtp4",
        ),
        pytest.param(
            True,
            2048,
            8,
            65_536,
            (32_771, 65_529),
            4,
            True,
            id="multi-cta-mtp4",
        ),
    ],
}.get(_GFX, [])

_GRAPH_WIDTH = 131_072 if _GFX == "gfx942" else 65_536
_TIE_CASES = [
    pytest.param(4, 4_096, 2_048, id="one-workgroup"),
    pytest.param(4, _GRAPH_WIDTH, 2_048, id="multi-cta"),
]
_NAN_CASES = [
    pytest.param(4_096, id="one-workgroup"),
    pytest.param(_GRAPH_WIDTH, id="multi-cta"),
]


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
    values: torch.Tensor | None = None,
    next_n: int = 1,
) -> None:
    public_topk_impl.top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        output,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        top_k,
        stable,
        values,
    )


@pytest.mark.parametrize(
    "arch,stable,top_k,rows,width,expected",
    [
        ("gfx942", True, 2048, 1, 65_536, False),
        ("gfx942", True, 2048, 64, 20_000, True),
        ("gfx942", True, 2048, 65, 20_000, False),
        ("gfx942", True, 2048, 1, 131_071, False),
        ("gfx942", True, 2048, 16, 131_072, True),
        ("gfx942", True, 2048, 17, 131_072, False),
        ("gfx942", False, 2048, 1, 524_287, False),
        ("gfx942", False, 2048, 32, 524_288, True),
        ("gfx942", False, 2048, 33, 524_288, False),
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
    "stable,top_k,rows,width,context_lens,next_n,write_values",
    _E2E_CASES,
)
def test_public_topk_decode_e2e(
    stable,
    top_k,
    rows,
    width,
    context_lens,
    next_n,
    write_values,
):
    """Check FlyDSL kernels against the Torch reference."""
    torch.manual_seed(0)
    if isinstance(context_lens, int):
        context_lens = [context_lens] * ((rows + next_n - 1) // next_n)
        logits = torch.randn((rows, width), dtype=torch.float32, device="cuda")
    else:
        context_lens = list(context_lens)
        logits = torch.arange(width, dtype=torch.float32, device="cuda").repeat(
            rows, 1
        )
    seq_lens = torch.tensor(context_lens, dtype=torch.int32, device="cuda")
    output = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    values = (
        torch.empty((rows, top_k), dtype=torch.float32, device="cuda")
        if write_values
        else None
    )
    reference = run_torch(logits, context_lens, next_n, top_k, stable)

    _run_public(
        logits,
        seq_lens,
        output,
        top_k,
        stable,
        values,
        next_n,
    )
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
    padding = reference == -1
    assert torch.all(output[padding] == -1)
    if values is not None:
        valid = output >= 0
        gathered = torch.gather(logits, 1, output.clamp_min(0).to(torch.int64))
        torch.testing.assert_close(values[valid], gathered[valid], rtol=0, atol=0)
        assert torch.all(torch.isneginf(values[~valid]))


@pytest.mark.parametrize("rows,width,top_k", _TIE_CASES)
def test_public_topk_decode_stable_tie_break(rows, width, top_k):
    """Prefer the smallest indices when many values equal the TopK threshold."""
    logits = torch.zeros((rows, width), dtype=torch.float32, device="cuda")
    expected_rows = []
    for row in range(rows):
        above_count = top_k // 4 + row * 17
        above_indices = (
            torch.arange(above_count, dtype=torch.int64, device="cuda") * 17
            + row * 13
        ) % width
        logits[row, above_indices] = 1.0

        above = torch.nonzero(logits[row] > 0, as_tuple=False).flatten()
        equal = torch.nonzero(logits[row] == 0, as_tuple=False).flatten()
        selected = torch.cat((above, equal[: top_k - above_count]))
        expected_rows.append(torch.sort(selected).values)

    expected = torch.stack(expected_rows).to(torch.int32)
    seq_lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")

    def run():
        output = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
        _run_public(logits, seq_lens, output, top_k, True)
        return output

    first, second = run(), run()
    torch.cuda.synchronize()
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", _NAN_CASES)
def test_public_topk_decode_nan_ordering(width):
    """Canonicalize NaNs above +inf and tie-break them by smallest index."""
    top_k = 512
    nan_count = 700
    logits = torch.full((1, width), float("inf"), dtype=torch.float32, device="cuda")
    nan_indices = (
        torch.arange(nan_count, dtype=torch.int64, device="cuda") * 17 + 3
    ) % width
    nan_bits = torch.tensor(
        [0x7FC00001, 0x7FC12345, 0xFFC00002 - (1 << 32)],
        dtype=torch.int32,
        device="cuda",
    )
    nan_values = nan_bits.view(torch.float32)
    logits[0, nan_indices] = nan_values[
        torch.arange(nan_count, device="cuda") % nan_values.numel()
    ]

    seq_lens = torch.tensor([width], dtype=torch.int32, device="cuda")
    output = torch.empty((1, top_k), dtype=torch.int32, device="cuda")
    expected = torch.sort(nan_indices).values[:top_k].view(1, top_k).to(torch.int32)

    _run_public(logits, seq_lens, output, top_k, True)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_public_topk_decode_cuda_graph():
    """Check capture and replay of the long-row multi-kernel path."""
    rows, width, top_k = 1, _GRAPH_WIDTH, 2048
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
