# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_rmsnorm_indexed_adaln import (
    fused_layernorm_indexed_adaln,
    fused_rmsnorm_indexed_adaln,
)


def reference(x, weight, shift, scale, indices, eps, round_intermediate=False):
    """What the fused op replaces, with optional eager-style store rounding."""
    normed = torch.nn.functional.rms_norm(
        x.float(), (x.shape[-1],), weight.float(), eps
    )
    if round_intermediate:
        normed = normed.to(x.dtype).float()
    modulated = normed * (1.0 + scale.index_select(0, indices).float())
    if round_intermediate:
        modulated = modulated.to(x.dtype).float()
    return (modulated + shift.index_select(0, indices).float()).to(x.dtype)


def make(M, N, G, dtype, device, *, runs=True, seed=0):
    """``runs`` mirrors a packed diffusion sequence: consecutive tokens share a
    modulation index. The scattered case is the adversarial one."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(M, N, generator=gen, device=device, dtype=dtype)
    weight = torch.randn(N, generator=gen, device=device, dtype=dtype)
    shift = torch.randn(G, N, generator=gen, device=device, dtype=dtype)
    scale = torch.randn(G, N, generator=gen, device=device, dtype=dtype) * 0.1
    if runs:
        edges = (
            torch.linspace(0, M, G + 1, dtype=torch.float32, device=device)
            .round()
            .long()
        )
        indices = torch.zeros(M, dtype=torch.int64, device=device)
        for g in range(G):
            indices[edges[g] : edges[g + 1]] = g
    else:
        indices = torch.randint(
            0, G, (M,), generator=gen, device=device, dtype=torch.int64
        )
    # Drawn last, so the tensors above keep their existing values.
    bias = torch.randn(N, generator=gen, device=device, dtype=dtype)
    return x, weight, shift, scale, indices, bias


@pytest.mark.parametrize("round_intermediate", [False, True])
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
def test_matches_norm_then_gathered_affine(dtype, M, N, G, runs, round_intermediate):
    dev = "cuda"
    x, weight, shift, scale, indices, _ = make(M, N, G, dtype, dev, runs=runs)
    got = fused_rmsnorm_indexed_adaln(
        x,
        weight,
        shift,
        scale,
        indices,
        eps=1e-5,
        round_intermediate=round_intermediate,
    )
    want = reference(
        x, weight, shift, scale, indices, 1e-5, round_intermediate=round_intermediate
    )

    assert got.shape == x.shape and got.dtype == x.dtype
    if round_intermediate and dtype != torch.float32:
        full_precision = reference(
            x, weight, shift, scale, indices, 1e-5, round_intermediate=False
        )
        assert not torch.equal(
            want, full_precision
        ), "test input does not expose intermediate rounding"
        unrounded = fused_rmsnorm_indexed_adaln(
            x, weight, shift, scale, indices, eps=1e-5, round_intermediate=False
        )
        assert not torch.equal(
            got, unrounded
        ), "round_intermediate=True did not change the kernel output"
        tol = 2 * torch.finfo(dtype).eps
    else:
        # Without intermediate rounding the kernel keeps the row in fp32 across
        # the norm and affine, so compare against the fp32-intermediate reference.
        tol = {torch.float32: 2e-6, torch.float16: 4e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(got.float(), want.float(), atol=tol, rtol=tol)


def test_uniform_and_scattered_indices_agree():
    """The uniform-index fast path must not change the result.

    The kernel asks whether every row in a block shares an index
    (``tl.min(idx) == tl.max(idx)``) and, when they do, broadcasts a single
    modulation row instead of gathering [BLOCK_M, BLOCK_N]. To compare the two
    branches the same modulation has to be reachable both ways, so two table
    entries are made identical: indexing them uniformly takes the broadcast, and
    alternating between them takes the gather, for a result that must be equal
    bit for bit.
    """
    dev = "cuda"
    M, N, G = 256, 5376, 6
    x, weight, shift, scale, _, _ = make(M, N, G, torch.bfloat16, dev)
    # Entries 2 and 5 now modulate identically, so which one a row picks is
    # invisible in the output but decides which branch its block takes.
    shift[5] = shift[2]
    scale[5] = scale[2]

    same = torch.full((M,), 2, dtype=torch.int64, device=dev)
    fast = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, same, eps=1e-5)

    # Same modulation, but no block is uniform.
    alternating = torch.where(
        torch.arange(M, device=dev) % 2 == 0,
        torch.full((M,), 2, dtype=torch.int64, device=dev),
        torch.full((M,), 5, dtype=torch.int64, device=dev),
    )
    slow = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, alternating, eps=1e-5)
    torch.testing.assert_close(fast, slow, atol=0, rtol=0)

    mixed = torch.arange(M, device=dev, dtype=torch.int64) % G
    got = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, mixed, eps=1e-5)
    want = reference(x, weight, shift, scale, mixed, 1e-5)
    torch.testing.assert_close(got.float(), want.float(), atol=3e-2, rtol=3e-2)


def test_every_index_is_honoured():
    """One row per table entry: a kernel that broadcast row 0 would pass a
    uniform test and fail this one."""
    dev = "cuda"
    N, G = 512, 6
    x, weight, shift, scale, _, _ = make(G, N, G, torch.float32, dev)
    indices = torch.arange(G, device=dev, dtype=torch.int64)
    got = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, eps=1e-5)
    want = reference(x, weight, shift, scale, indices, 1e-5)
    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-6)


def test_out_parameter_is_written_in_place():
    dev = "cuda"
    x, weight, shift, scale, indices, _ = make(64, 1024, 3, torch.bfloat16, dev)
    dst = torch.empty_like(x)
    got = fused_rmsnorm_indexed_adaln(
        x, weight, shift, scale, indices, out=dst, eps=1e-5
    )
    assert got.data_ptr() == dst.data_ptr()
    torch.testing.assert_close(
        dst.float(),
        reference(x, weight, shift, scale, indices, 1e-5).float(),
        atol=3e-2,
        rtol=3e-2,
    )


def test_int32_indices_are_accepted():
    dev = "cuda"
    x, weight, shift, scale, indices, _ = make(64, 1024, 3, torch.bfloat16, dev)
    got = fused_rmsnorm_indexed_adaln(
        x, weight, shift, scale, indices.to(torch.int32), eps=1e-5
    )
    want = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, eps=1e-5)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_rejects_mismatched_shapes():
    dev = "cuda"
    x, weight, shift, scale, indices, _ = make(32, 256, 3, torch.bfloat16, dev)
    with pytest.raises(AssertionError):
        fused_rmsnorm_indexed_adaln(x, weight[:-1], shift, scale, indices)
    with pytest.raises(AssertionError):
        fused_rmsnorm_indexed_adaln(x, weight, shift, scale, indices[:-1])


def layernorm_reference(
    x, weight, bias, shift, scale, indices, eps, round_intermediate=False
):
    """LayerNorm, then the gathered affine, with optional eager-style rounding."""
    normed = torch.nn.functional.layer_norm(
        x.float(),
        (x.shape[-1],),
        None if weight is None else weight.float(),
        None if bias is None else bias.float(),
        eps,
    )
    if round_intermediate:
        normed = normed.to(x.dtype).float()
    modulated = normed * (1.0 + scale.index_select(0, indices).float())
    if round_intermediate:
        modulated = modulated.to(x.dtype).float()
    return (modulated + shift.index_select(0, indices).float()).to(x.dtype)


@pytest.mark.parametrize("round_intermediate", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "M,N,G",
    [
        (1, 3072, 1),  # single token
        (7, 1152, 2),  # fewer rows than BLOCK_M
        (333, 320, 4),  # N below one tile, M not a multiple of BLOCK_M
        (1024, 1536, 2),  # Wan2.1-1.3B hidden size, one row per CFG sample
        (64, 2432, 2),  # SD3.5-Large: N not a multiple of BLOCK_N
        (64, 8192, 6),  # power-of-two row, several tiles
    ],
)
@pytest.mark.parametrize("affine", ["none", "weight", "weight_bias"])
@pytest.mark.parametrize("runs", [True, False])
def test_layernorm_matches_norm_then_gathered_affine(
    dtype, M, N, G, affine, runs, round_intermediate
):
    dev = "cuda"
    x, weight, shift, scale, indices, bias = make(M, N, G, dtype, dev, runs=runs)
    if affine == "none":
        weight = None
    if affine != "weight_bias":
        bias = None
    got = fused_layernorm_indexed_adaln(
        x,
        shift,
        scale,
        indices,
        eps=1e-6,
        weight=weight,
        bias=bias,
        round_intermediate=round_intermediate,
    )
    want = layernorm_reference(
        x, weight, bias, shift, scale, indices, 1e-6, round_intermediate
    )

    assert got.shape == x.shape and got.dtype == x.dtype
    if round_intermediate and dtype != torch.float32:
        full_precision = layernorm_reference(
            x, weight, bias, shift, scale, indices, 1e-6, round_intermediate=False
        )
        assert not torch.equal(
            want, full_precision
        ), "test input does not expose intermediate rounding"
        unrounded = fused_layernorm_indexed_adaln(
            x, shift, scale, indices, eps=1e-6, weight=weight, bias=bias
        )
        assert not torch.equal(
            got, unrounded
        ), "round_intermediate=True did not change the kernel output"
    # One table for both paths: a rounded intermediate can land on the other
    # side of a midpoint from the reference, and the add does not shrink the gap.
    tol = {torch.float32: 2e-6, torch.float16: 4e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(got.float(), want.float(), atol=tol, rtol=tol)


@pytest.mark.parametrize("N", [320, 2432, 8192])
def test_layernorm_large_mean(N):
    """A row with a large mean and a small spread. A plain fp32 sum loses the
    variance here. The reference is fp64, because fp32 torch.nn.LayerNorm also
    has errors near 1e-2 on this input."""
    dev = "cuda"
    M, G = 64, 2
    x, _, shift, scale, indices, _ = make(M, N, G, torch.float32, dev)
    x = x * 0.01 + 1000.0
    got = fused_layernorm_indexed_adaln(x, shift, scale, indices, eps=1e-6)
    xn = torch.nn.functional.layer_norm(x.double(), (N,), eps=1e-6)
    want = xn * (1 + scale.double()[indices]) + shift.double()[indices]
    torch.testing.assert_close(got.double(), want, atol=1e-5, rtol=1e-5)


def test_layernorm_leading_outlier():
    """Column 0 holds a value large enough that fp32 cannot resolve the rest
    against it. An anchor taken from that one column would round the other
    values together and lose the variance."""
    dev = "cuda"
    N = 8192
    x = torch.empty(1, N, device=dev, dtype=torch.float32)
    x[0, 0] = 2.0**24
    x[0, 1::2] = 0.0
    x[0, 2::2] = -1.0
    shift = torch.zeros(1, N, device=dev, dtype=torch.float32)
    scale = torch.zeros(1, N, device=dev, dtype=torch.float32)
    indices = torch.zeros(1, device=dev, dtype=torch.int64)
    got = fused_layernorm_indexed_adaln(x, shift, scale, indices, eps=1e-6)
    want = torch.nn.functional.layer_norm(x.double(), (N,), eps=1e-6)
    torch.testing.assert_close(got.double(), want, atol=2e-6, rtol=2e-6)


def test_layernorm_near_fp32_limit():
    """Every value is finite, but a tile of them sums past the fp32 limit. The
    anchor has to scale before it reduces, or the row comes back as NaN."""
    dev = "cuda"
    N = 8192
    x = torch.full((1, N), 1e36, device=dev, dtype=torch.float32)
    shift = torch.zeros(1, N, device=dev, dtype=torch.float32)
    scale = torch.zeros(1, N, device=dev, dtype=torch.float32)
    indices = torch.zeros(1, device=dev, dtype=torch.int64)
    got = fused_layernorm_indexed_adaln(x, shift, scale, indices, eps=1e-6)
    want = torch.nn.functional.layer_norm(x.double(), (N,), eps=1e-6)
    torch.testing.assert_close(got.double(), want, atol=2e-6, rtol=2e-6)


def test_layernorm_per_sample_modulation():
    """The common DiT case: a [B, N] modulation and indices = sample of each
    token."""
    dev = "cuda"
    B, S, N = 2, 333, 1152
    x, _, shift, scale, _, _ = make(B * S, N, B, torch.bfloat16, dev)
    indices = torch.arange(B, device=dev).repeat_interleave(S)
    got = fused_layernorm_indexed_adaln(x, shift, scale, indices, eps=1e-6)
    xn = torch.nn.functional.layer_norm(x.float().view(B, S, N), (N,), eps=1e-6)
    want = xn * (1 + scale.float()[:, None]) + shift.float()[:, None]
    torch.testing.assert_close(got.float().view(B, S, N), want, atol=3e-2, rtol=3e-2)


def test_layernorm_rejects_mismatched_shapes():
    dev = "cuda"
    x, weight, shift, scale, indices, _ = make(32, 256, 3, torch.bfloat16, dev)
    with pytest.raises(AssertionError):
        fused_layernorm_indexed_adaln(x, shift, scale, indices, weight=weight[:-1])
    with pytest.raises(AssertionError):
        fused_layernorm_indexed_adaln(x, shift, scale, indices, bias=weight[:-1])


def test_rejects_column_strided_table():
    """Only the row stride reaches the kernel, so a column-strided table would
    take the wrong elements. A row-strided view still has to work: that is one
    chunk of a packed table, which is how a DiT holds its modulation."""
    dev = "cuda"
    N = 256
    x, weight, shift, scale, indices, _ = make(32, N, 3, torch.bfloat16, dev)
    packed = torch.cat([shift, scale], dim=1)

    strided_shift, strided_scale = packed[:, ::2], packed[:, 1::2]
    assert strided_shift.shape == shift.shape and strided_shift.stride(1) == 2
    with pytest.raises(AssertionError):
        fused_rmsnorm_indexed_adaln(x, weight, strided_shift, strided_scale, indices)
    with pytest.raises(AssertionError):
        fused_layernorm_indexed_adaln(x, strided_shift, strided_scale, indices)

    # The two chunks of that table: row stride 2N, columns still adjacent.
    chunk_shift, chunk_scale = packed[:, :N], packed[:, N:]
    assert chunk_shift.stride(0) == 2 * N and chunk_shift.stride(1) == 1
    got = fused_layernorm_indexed_adaln(x, chunk_shift, chunk_scale, indices, eps=1e-6)
    want = layernorm_reference(x, None, None, shift, scale, indices, 1e-6)
    torch.testing.assert_close(got.float(), want.float(), atol=3e-2, rtol=3e-2)


def test_rejects_strided_affine():
    """A strided view has the right shape, but the kernel reads it with the
    column offset alone, so it would take the wrong elements."""
    dev = "cuda"
    x, weight, shift, scale, indices, _ = make(32, 256, 3, torch.bfloat16, dev)
    strided = weight.repeat(2)[::2]
    assert strided.shape == weight.shape and not strided.is_contiguous()
    with pytest.raises(AssertionError):
        fused_rmsnorm_indexed_adaln(x, strided, shift, scale, indices)
    with pytest.raises(AssertionError):
        fused_layernorm_indexed_adaln(x, shift, scale, indices, weight=strided)
    with pytest.raises(AssertionError):
        fused_layernorm_indexed_adaln(x, shift, scale, indices, bias=strided)
