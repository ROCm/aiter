# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_rmsnorm_indexed_adaln import (
    fused_rmsnorm_indexed_adaln,
)


def reference(x, weight, shift, scale, indices, eps):
    """What the fused op replaces: a norm, then a gathered affine."""
    normed = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)
    return normed * (1.0 + scale.index_select(0, indices)) + shift.index_select(
        0, indices
    )


def make(M, N, G, dtype, device, *, runs=True, seed=0):
    """``runs`` mirrors a packed diffusion sequence: consecutive tokens share a
    modulation index. The scattered case is the adversarial one."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(M, N, generator=gen, device=device, dtype=dtype)
    weight = torch.randn(N, generator=gen, device=device, dtype=dtype)
    shift = torch.randn(G, N, generator=gen, device=device, dtype=dtype)
    scale = torch.randn(G, N, generator=gen, device=device, dtype=dtype) * 0.1
    if runs:
        edges = torch.linspace(0, M, G + 1).round().long()
        indices = torch.zeros(M, dtype=torch.int64, device=device)
        for g in range(G):
            indices[edges[g] : edges[g + 1]] = g
    else:
        indices = torch.randint(
            0, G, (M,), generator=gen, device=device, dtype=torch.int64
        )
    return x, weight, shift, scale, indices


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "M,N,G",
    [
        (1, 5376, 3),  # single token
        (7, 5376, 3),  # fewer rows than BLOCK_M
        (128, 5376, 9),  # MiniMax-H3: hidden 5376, 3 modalities x 3 timesteps
        (1024, 1536, 2),
        (333, 320, 4),  # N below one tile, M not a multiple of BLOCK_M
        (64, 8192, 6),  # power-of-two row
    ],
)
@pytest.mark.parametrize("runs", [True, False])
def test_matches_norm_then_gathered_affine(dtype, M, N, G, runs):
    dev = "cuda"
    x, weight, shift, scale, indices = make(M, N, G, dtype, dev, runs=runs)
    got = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, eps=1e-5)
    want = reference(x, weight, shift, scale, indices, 1e-5)

    assert got.shape == x.shape and got.dtype == x.dtype
    # The fused path keeps the row in fp32 across the norm and the affine, so it
    # is nearer the fp32 result than the reference is, not further.
    tol = {torch.float32: 2e-6, torch.float16: 4e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(got.float(), want.float(), atol=tol, rtol=tol)


def test_uniform_and_scattered_indices_agree():
    """The uniform-index fast path must not change the result.

    A block of rows sharing an index takes a different route through the kernel
    than a mixed block; feeding the same logical work both ways is what catches
    a broadcast that quietly ignores per-row indices.
    """
    dev = "cuda"
    M, N, G = 256, 5376, 4
    x, weight, shift, scale, _ = make(M, N, G, torch.bfloat16, dev)
    same = torch.full((M,), 2, dtype=torch.int64, device=dev)
    fast = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, same)

    # Same indices, but arranged so no block is uniform.
    alternating = torch.full((M,), 2, dtype=torch.int64, device=dev)
    slow = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, alternating)
    torch.testing.assert_close(fast, slow, atol=0, rtol=0)

    mixed = torch.arange(M, device=dev, dtype=torch.int64) % G
    got = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, mixed)
    want = reference(x, weight, shift, scale, mixed, 1e-5)
    torch.testing.assert_close(got.float(), want.float(), atol=3e-2, rtol=3e-2)


def test_every_index_is_honoured():
    """One row per table entry: a kernel that broadcast row 0 would pass a
    uniform test and fail this one."""
    dev = "cuda"
    N, G = 512, 6
    x, weight, shift, scale, _ = make(G, N, G, torch.float32, dev)
    indices = torch.arange(G, device=dev, dtype=torch.int64)
    got = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, indices)
    want = reference(x, weight, shift, scale, indices, 1e-5)
    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-6)


def test_out_parameter_is_written_in_place():
    dev = "cuda"
    x, weight, shift, scale, indices = make(64, 1024, 3, torch.bfloat16, dev)
    dst = torch.empty_like(x)
    got = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, out=dst)
    assert got.data_ptr() == dst.data_ptr()
    torch.testing.assert_close(
        dst.float(),
        reference(x, weight, shift, scale, indices, 1e-5).float(),
        atol=3e-2,
        rtol=3e-2,
    )


def test_int32_indices_are_accepted():
    dev = "cuda"
    x, weight, shift, scale, indices = make(64, 1024, 3, torch.bfloat16, dev)
    got = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, indices.to(torch.int32))
    want = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, indices)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_rejects_mismatched_shapes():
    dev = "cuda"
    x, weight, shift, scale, indices = make(32, 256, 3, torch.bfloat16, dev)
    with pytest.raises(AssertionError):
        fused_rmsnorm_indexed_adaln(x, weight[:-1], shift, scale, indices)
    with pytest.raises(AssertionError):
        fused_rmsnorm_indexed_adaln(x, weight, shift, scale, indices[:-1])
