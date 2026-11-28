# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.fusions.fused_indexed_gated_residual_layernorm_adaln import (
    fused_indexed_gated_residual_layernorm_adaln,
)


def reference(
    residual,
    x,
    gate,
    shift,
    scale,
    indices,
    eps=1e-6,
    weight=None,
    bias=None,
    round_intermediate=False,
):
    """The unfused expression, with h rounded to the output dtype as it is when
    stored between the two steps."""
    gated = x.float()
    if gate is not None:
        gated = gate.index_select(0, indices).float() * gated
        if round_intermediate:
            gated = gated.to(x.dtype).float()
    h = (residual.float() + gated).to(residual.dtype)
    normed = F.layer_norm(
        h.float(),
        (h.shape[1],),
        None if weight is None else weight.float(),
        None if bias is None else bias.float(),
        eps,
    )
    out = (
        normed * (1 + scale.index_select(0, indices).float())
        + shift.index_select(0, indices).float()
    )
    return out.to(residual.dtype), h


def make(M, N, G, dtype, device, *, runs=True, seed=0):
    """``runs`` mirrors a packed diffusion sequence: consecutive tokens share a
    table index. The scattered case is the adversarial one."""
    gen = torch.Generator(device=device).manual_seed(seed)
    # An offset mean checks that the statistics do not lose precision to it.
    residual = torch.randn(M, N, generator=gen, device=device, dtype=dtype) * 4 + 3
    x = torch.randn(M, N, generator=gen, device=device, dtype=dtype)
    gate, shift, scale = (
        torch.randn(G, N, generator=gen, device=device, dtype=dtype) for _ in range(3)
    )
    if runs:
        edges = (
            torch.linspace(0, M, G + 1, device=device, dtype=torch.float32)
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
    return residual, x, gate, shift, scale, indices


def affine(N, dtype, device, seed=1):
    gen = torch.Generator(device=device).manual_seed(seed)
    weight = 1 + 0.1 * torch.randn(N, generator=gen, device=device, dtype=dtype)
    bias = 0.1 * torch.randn(N, generator=gen, device=device, dtype=dtype)
    return weight, bias


def check(got, want):
    out, h = got
    want_out, want_h = want
    assert out.shape == want_out.shape and out.dtype == want_out.dtype
    # h is rounded once to the output dtype in both.
    torch.testing.assert_close(h, want_h, atol=0, rtol=0)
    # out differs by the summation order of the statistics and one rounding.
    torch.testing.assert_close(out.float(), want_out.float(), atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "M,N,G",
    [
        (1, 5120, 3),  # single token
        (7, 3072, 3),  # fewer rows than BLOCK_M
        (256, 5120, 2),  # Wan 14B hidden size
        (256, 1536, 2),  # Wan 1.3B, SD3.5
        (128, 3072, 9),  # Flux, Qwen-Image
        (333, 320, 4),  # N not a multiple of BLOCK_N, M not a multiple of BLOCK_M
        (64, 8192, 6),  # power-of-two row wider than a tile
        (32, 12000, 3),  # several tiles and a partial one
        (5, 3, 2),  # row narrower than one tile
    ],
)
@pytest.mark.parametrize("runs", [True, False])
def test_matches_unfused(M, N, G, runs, dtype):
    dev = "cuda"
    residual, x, gate, shift, scale, indices = make(M, N, G, dtype, dev, runs=runs)
    # round_intermediate makes h exactly reproducible; without it, h may differ
    # from the eager expression by one rounding of the product.
    got = fused_indexed_gated_residual_layernorm_adaln(
        residual, x, gate, shift, scale, indices, round_intermediate=True
    )
    check(
        got,
        reference(residual, x, gate, shift, scale, indices, round_intermediate=True),
    )


@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("N", [1536, 5120, 12000])
def test_layernorm_affine(N, has_weight, has_bias):
    dev = "cuda"
    residual, x, gate, shift, scale, indices = make(96, N, 3, torch.bfloat16, dev)
    weight, bias = affine(N, torch.bfloat16, dev)
    weight = weight if has_weight else None
    bias = bias if has_bias else None
    got = fused_indexed_gated_residual_layernorm_adaln(
        residual,
        x,
        gate,
        shift,
        scale,
        indices,
        eps=1e-5,
        weight=weight,
        bias=bias,
        round_intermediate=True,
    )
    check(
        got,
        reference(
            residual,
            x,
            gate,
            shift,
            scale,
            indices,
            1e-5,
            weight,
            bias,
            round_intermediate=True,
        ),
    )


@pytest.mark.parametrize("N", [1536, 12000])
def test_without_gate(N):
    """gate=None is a plain residual add, as in a block without a gate."""
    dev = "cuda"
    residual, x, _, shift, scale, indices = make(64, N, 3, torch.bfloat16, dev)
    got = fused_indexed_gated_residual_layernorm_adaln(
        residual, x, None, shift, scale, indices
    )
    check(got, reference(residual, x, None, shift, scale, indices))
    torch.testing.assert_close(got[1], residual + x, atol=0, rtol=0)


def test_fp32_product_by_default():
    """Without round_intermediate the product stays in fp32, so h is within one
    rounding of the eager value."""
    dev = "cuda"
    residual, x, gate, shift, scale, indices = make(512, 5120, 3, torch.bfloat16, dev)
    _, h = fused_indexed_gated_residual_layernorm_adaln(
        residual, x, gate, shift, scale, indices
    )
    want_h = (residual.float() + gate.index_select(0, indices).float() * x.float()).to(
        residual.dtype
    )
    # The kernel may contract the product and the add into an fma.
    torch.testing.assert_close(h.float(), want_h.float(), atol=1.6e-2, rtol=1.6e-2)
    rounded_h = fused_indexed_gated_residual_layernorm_adaln(
        residual, x, gate, shift, scale, indices, round_intermediate=True
    )[1]
    assert not torch.equal(h, rounded_h), "test input does not expose rounding"


def test_uniform_and_scattered_indices_agree():
    """The uniform-index fast path must not change the result.

    The kernel broadcasts a single table row when every row in a block shares
    an index, and gathers [BLOCK_M, BLOCK_N] otherwise. Two identical table
    entries make the same row reachable both ways, so the outputs must be
    equal bit for bit.
    """
    dev = "cuda"
    M, N, G = 256, 3072, 6
    residual, x, gate, shift, scale, _ = make(M, N, G, torch.bfloat16, dev)
    for t in (gate, shift, scale):
        t[5] = t[2]
    same = torch.full((M,), 2, dtype=torch.int64, device=dev)
    alternating = torch.where(torch.arange(M, device=dev) % 2 == 0, 2, 5)
    fast = fused_indexed_gated_residual_layernorm_adaln(
        residual, x, gate, shift, scale, same
    )
    slow = fused_indexed_gated_residual_layernorm_adaln(
        residual, x, gate, shift, scale, alternating
    )
    for a, b in zip(fast, slow):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_every_index_is_honoured():
    """One row per table entry: a kernel that broadcast row 0 would pass a
    uniform test and fail this one."""
    dev = "cuda"
    N, G = 1536, 6
    residual, x, gate, shift, scale, _ = make(G, N, G, torch.bfloat16, dev)
    indices = torch.arange(G, device=dev, dtype=torch.int64)
    got = fused_indexed_gated_residual_layernorm_adaln(
        residual, x, gate, shift, scale, indices, round_intermediate=True
    )
    check(
        got,
        reference(residual, x, gate, shift, scale, indices, round_intermediate=True),
    )


def test_per_frame_table():
    """A per-frame modulation [B, F, 1, N] flattened to [B * F, N], each token
    indexing the row of its frame."""
    dev = "cuda"
    B, F_, T, N = 2, 5, 48, 1536
    M = B * F_ * T
    residual, x, _, _, _, _ = make(M, N, 1, torch.bfloat16, dev)
    gen = torch.Generator(device=dev).manual_seed(3)
    gate, shift, scale = (
        torch.randn(B, F_, 1, N, generator=gen, device=dev, dtype=torch.bfloat16)
        for _ in range(3)
    )
    frame = torch.arange(B * F_, device=dev).repeat_interleave(T)
    got = fused_indexed_gated_residual_layernorm_adaln(
        residual,
        x,
        gate.reshape(B * F_, N),
        shift.reshape(B * F_, N),
        scale.reshape(B * F_, N),
        frame,
        round_intermediate=True,
    )
    # The eager broadcast over [B, F, T, N].
    r4, x4 = residual.view(B, F_, T, N), x.view(B, F_, T, N)
    h = r4 + gate * x4
    out = F.layer_norm(h.float(), (N,), eps=1e-6) * (1 + scale.float()) + shift.float()
    check(got, (out.to(torch.bfloat16).view(M, N), h.view(M, N)))


def test_out_parameters_and_in_place_update():
    dev = "cuda"
    residual, x, gate, shift, scale, indices = make(64, 3072, 3, torch.bfloat16, dev)
    want = reference(residual, x, gate, shift, scale, indices, round_intermediate=True)
    out, h = torch.empty_like(residual), torch.empty_like(residual)
    got = fused_indexed_gated_residual_layernorm_adaln(
        residual,
        x,
        gate,
        shift,
        scale,
        indices,
        out=out,
        h_out=h,
        round_intermediate=True,
    )
    assert got[0].data_ptr() == out.data_ptr() and got[1].data_ptr() == h.data_ptr()
    check(got, want)

    got = fused_indexed_gated_residual_layernorm_adaln(
        residual,
        x,
        gate,
        shift,
        scale,
        indices,
        h_out=residual,
        round_intermediate=True,
    )
    assert got[1].data_ptr() == residual.data_ptr()
    check(got, want)


def test_strided_rows():
    """Rows may be views into wider buffers, e.g. one chunk of a fused projection."""
    dev = "cuda"
    M, N, G = 96, 640, 4
    residual, x, gate, shift, scale, indices = make(
        M, N, G, torch.bfloat16, dev, runs=False
    )
    want = reference(residual, x, gate, shift, scale, indices, round_intermediate=True)
    x_wide = torch.zeros(M, 3 * N, device=dev, dtype=torch.bfloat16)
    x_wide[:, N : 2 * N] = x
    # Modulation tables are often chunks of one [G, 6 * N] projection.
    table = torch.zeros(G, 6 * N, device=dev, dtype=torch.bfloat16)
    table[:, :N], table[:, 2 * N : 3 * N], table[:, 4 * N : 5 * N] = gate, shift, scale
    out_wide = torch.zeros(M, 2 * N, device=dev, dtype=torch.bfloat16)
    h_wide = torch.zeros(M, 2 * N, device=dev, dtype=torch.bfloat16)
    got = fused_indexed_gated_residual_layernorm_adaln(
        residual,
        x_wide[:, N : 2 * N],
        table[:, :N],
        table[:, 2 * N : 3 * N],
        table[:, 4 * N : 5 * N],
        indices,
        out=out_wide[:, N:],
        h_out=h_wide[:, :N],
        round_intermediate=True,
    )
    check(got, want)
    assert torch.count_nonzero(out_wide[:, :N]) == 0
    assert torch.count_nonzero(h_wide[:, N:]) == 0


def test_int32_indices_are_accepted():
    dev = "cuda"
    residual, x, gate, shift, scale, indices = make(
        64, 1536, 3, torch.bfloat16, dev, runs=False
    )
    got = fused_indexed_gated_residual_layernorm_adaln(
        residual, x, gate, shift, scale, indices.to(torch.int32)
    )
    want = fused_indexed_gated_residual_layernorm_adaln(
        residual, x, gate, shift, scale, indices
    )
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_rejects_mismatched_shapes():
    dev = "cuda"
    residual, x, gate, shift, scale, indices = make(32, 256, 3, torch.bfloat16, dev)
    f = fused_indexed_gated_residual_layernorm_adaln
    with pytest.raises(AssertionError):
        f(residual, x[:, :-1], gate, shift, scale, indices)
    with pytest.raises(AssertionError):
        f(residual, x, gate[:, :-1], shift, scale, indices)
    with pytest.raises(AssertionError):
        f(residual, x, gate, shift, scale[:-1], indices)
    with pytest.raises(AssertionError):
        f(residual, x, gate, shift, scale, indices[:-1])
    with pytest.raises(AssertionError):
        f(residual, x, gate, shift, scale, indices.float())
    with pytest.raises(AssertionError):
        f(residual, x, gate, shift, scale, indices, weight=gate[0, :-1])


def test_rejects_unsupported_dtypes():
    dev = "cuda"
    residual, x, gate, shift, scale, indices = make(32, 256, 3, torch.bfloat16, dev)
    f = fused_indexed_gated_residual_layernorm_adaln
    with pytest.raises(AssertionError):
        f(*(t.float() for t in (residual, x, gate, shift, scale)), indices)
    with pytest.raises(AssertionError):
        f(residual, x, gate.float(), shift, scale, indices)
