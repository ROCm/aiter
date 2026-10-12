# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_indexed_gated_residual import (
    fused_indexed_gated_residual,
)


def reference(residual, x, gate, indices, round_intermediate=False):
    """What the fused op replaces; with rounding it is the eager expression."""
    gated = gate.index_select(0, indices).float() * x.float()
    if round_intermediate:
        gated = gated.to(x.dtype).float()
    return (residual.float() + gated).to(residual.dtype)


def make(M, N, G, dtype, device, *, runs=True, seed=0):
    """``runs`` mirrors a packed diffusion sequence: consecutive tokens share a
    gate index. The scattered case is the adversarial one."""
    gen = torch.Generator(device=device).manual_seed(seed)
    residual = torch.randn(M, N, generator=gen, device=device, dtype=dtype) * 4
    x = torch.randn(M, N, generator=gen, device=device, dtype=dtype)
    gate = torch.randn(G, N, generator=gen, device=device, dtype=dtype)
    if runs:
        edges = torch.linspace(0, M, G + 1).round().long()
        indices = torch.zeros(M, dtype=torch.int64, device=device)
        for g in range(G):
            indices[edges[g] : edges[g + 1]] = g
    else:
        indices = torch.randint(
            0, G, (M,), generator=gen, device=device, dtype=torch.int64
        )
    return residual, x, gate, indices


@pytest.mark.parametrize("round_intermediate", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "M,N,G",
    [
        (1, 5376, 3),  # single token
        (7, 5376, 3),  # fewer rows than BLOCK_M
        (128, 5376, 9),  # MiniMax-H3: hidden 5376, 3 modalities x 3 timesteps
        (1024, 1536, 2),
        (333, 320, 4),  # N not a multiple of BLOCK_N, M not a multiple of BLOCK_M
        (64, 8192, 6),  # power-of-two row
        (5, 3, 2),  # row narrower than one tile
    ],
)
@pytest.mark.parametrize("runs", [True, False])
def test_matches_gathered_gate_then_add(dtype, M, N, G, runs, round_intermediate):
    dev = "cuda"
    residual, x, gate, indices = make(M, N, G, dtype, dev, runs=runs)
    got = fused_indexed_gated_residual(
        residual, x, gate, indices, round_intermediate=round_intermediate
    )
    want = reference(residual, x, gate, indices, round_intermediate)

    assert got.shape == residual.shape and got.dtype == residual.dtype
    if round_intermediate:
        # One rounding of the product and one of the sum, like the eager
        # expression, so the result is reproduced bit for bit.
        torch.testing.assert_close(got, want, atol=0, rtol=0)
        if dtype != torch.float32:
            torch.testing.assert_close(
                got, residual + gate.index_select(0, indices) * x, atol=0, rtol=0
            )
    else:
        # The product stays in fp32; only the sum is rounded to the output dtype.
        tol = {torch.float32: 2e-6, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}[dtype]
        torch.testing.assert_close(got.float(), want.float(), atol=tol, rtol=tol)


def test_round_intermediate_changes_the_result():
    dev = "cuda"
    residual, x, gate, indices = make(512, 5376, 3, torch.bfloat16, dev)
    rounded = fused_indexed_gated_residual(
        residual, x, gate, indices, round_intermediate=True
    )
    full = fused_indexed_gated_residual(residual, x, gate, indices)
    assert not torch.equal(rounded, full), "test input does not expose rounding"


def test_uniform_and_scattered_indices_agree():
    """The uniform-index fast path must not change the result.

    The kernel broadcasts a single gate row when every row in a block shares an
    index, and gathers [BLOCK_M, BLOCK_N] otherwise. Two identical table entries
    make the same gate reachable both ways, so the outputs must be equal bit
    for bit.
    """
    dev = "cuda"
    M, N, G = 256, 5376, 6
    residual, x, gate, _ = make(M, N, G, torch.bfloat16, dev)
    gate[5] = gate[2]

    same = torch.full((M,), 2, dtype=torch.int64, device=dev)
    fast = fused_indexed_gated_residual(residual, x, gate, same)
    alternating = torch.where(
        torch.arange(M, device=dev) % 2 == 0,
        torch.full((M,), 2, dtype=torch.int64, device=dev),
        torch.full((M,), 5, dtype=torch.int64, device=dev),
    )
    slow = fused_indexed_gated_residual(residual, x, gate, alternating)
    torch.testing.assert_close(fast, slow, atol=0, rtol=0)


def test_every_index_is_honoured():
    """One row per table entry: a kernel that broadcast row 0 would pass a
    uniform test and fail this one."""
    dev = "cuda"
    N, G = 512, 6
    residual, x, gate, _ = make(G, N, G, torch.float32, dev)
    indices = torch.arange(G, device=dev, dtype=torch.int64)
    got = fused_indexed_gated_residual(residual, x, gate, indices)
    torch.testing.assert_close(
        got, reference(residual, x, gate, indices), atol=2e-6, rtol=2e-6
    )


def test_out_parameter_and_in_place_update():
    dev = "cuda"
    residual, x, gate, indices = make(64, 1024, 3, torch.bfloat16, dev)
    want = reference(residual, x, gate, indices, round_intermediate=True)
    dst = torch.empty_like(residual)
    got = fused_indexed_gated_residual(
        residual, x, gate, indices, out=dst, round_intermediate=True
    )
    assert got.data_ptr() == dst.data_ptr()
    torch.testing.assert_close(dst, want, atol=0, rtol=0)

    got = fused_indexed_gated_residual(
        residual, x, gate, indices, out=residual, round_intermediate=True
    )
    assert got.data_ptr() == residual.data_ptr()
    torch.testing.assert_close(residual, want, atol=0, rtol=0)


def test_strided_rows():
    """Rows may be views into wider buffers, e.g. one chunk of a fused projection."""
    dev = "cuda"
    M, N, G = 96, 640, 4
    residual, x, gate, indices = make(M, N, G, torch.bfloat16, dev, runs=False)
    wide = torch.zeros(M, 3 * N, device=dev, dtype=torch.bfloat16)
    wide[:, N : 2 * N] = x
    gate_wide = torch.zeros(G, 2 * N, device=dev, dtype=torch.bfloat16)
    gate_wide[:, :N] = gate
    out_wide = torch.zeros(M, 2 * N, device=dev, dtype=torch.bfloat16)
    got = fused_indexed_gated_residual(
        residual,
        wide[:, N : 2 * N],
        gate_wide[:, :N],
        indices,
        out=out_wide[:, N:],
        round_intermediate=True,
    )
    torch.testing.assert_close(
        got, reference(residual, x, gate, indices, True), atol=0, rtol=0
    )
    assert torch.count_nonzero(out_wide[:, :N]) == 0


def test_int32_indices_are_accepted():
    dev = "cuda"
    residual, x, gate, indices = make(64, 1024, 3, torch.bfloat16, dev, runs=False)
    got = fused_indexed_gated_residual(residual, x, gate, indices.to(torch.int32))
    want = fused_indexed_gated_residual(residual, x, gate, indices)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_rejects_mismatched_shapes():
    dev = "cuda"
    residual, x, gate, indices = make(32, 256, 3, torch.bfloat16, dev)
    with pytest.raises(AssertionError):
        fused_indexed_gated_residual(residual, x[:, :-1], gate, indices)
    with pytest.raises(AssertionError):
        fused_indexed_gated_residual(residual, x, gate[:, :-1], indices)
    with pytest.raises(AssertionError):
        fused_indexed_gated_residual(residual, x, gate, indices[:-1])
    with pytest.raises(AssertionError):
        fused_indexed_gated_residual(residual, x, gate, indices.float())
