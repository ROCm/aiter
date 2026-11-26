# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_indexed_gated_residual import (
    fused_indexed_gated_residual,
)
from aiter.ops.triton.fusions.fused_rmsnorm_indexed_adaln import (
    fused_gated_residual_rmsnorm_indexed_adaln_fp8_per_token_quant,
    fused_rmsnorm_indexed_adaln,
    fused_rmsnorm_indexed_adaln_fp8_per_token_quant,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from op_tests.triton_tests.utils.per_token_fp8_ref import per_token_fp8_ref


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
    return x, weight, shift, scale, indices


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
    x, weight, shift, scale, indices = make(M, N, G, dtype, dev, runs=runs)
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
    x, weight, shift, scale, _ = make(M, N, G, torch.bfloat16, dev)
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
    x, weight, shift, scale, _ = make(G, N, G, torch.float32, dev)
    indices = torch.arange(G, device=dev, dtype=torch.int64)
    got = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, eps=1e-5)
    want = reference(x, weight, shift, scale, indices, 1e-5)
    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-6)


def test_out_parameter_is_written_in_place():
    dev = "cuda"
    x, weight, shift, scale, indices = make(64, 1024, 3, torch.bfloat16, dev)
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
    x, weight, shift, scale, indices = make(64, 1024, 3, torch.bfloat16, dev)
    got = fused_rmsnorm_indexed_adaln(
        x, weight, shift, scale, indices.to(torch.int32), eps=1e-5
    )
    want = fused_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, eps=1e-5)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


# --- dynamic per-token FP8 outputs ---------------------------------------------

FP8 = get_fp8_e4m3_dtype()


def assert_quantizes(q, s, want):
    """q * s is a per-token FP8 quantization of want.

    The kernel quantizes its own y, which may sit one input ulp away from the
    reference's, so scales are compared to 1%, values to half an fp8 step plus
    that ulp, and fp8 codes must agree almost everywhere.
    """
    M, N = want.shape
    assert q.shape == (M, N) and q.dtype == FP8
    assert s.shape == (M, 1) and s.dtype == torch.float32
    q_ref, s_ref = per_token_fp8_ref(want, FP8)
    torch.testing.assert_close(s, s_ref, rtol=1e-2, atol=0)
    wf = want.float()
    err = (q.float() * s - wf).abs()
    assert (err <= wf.abs() * (2.0**-4 + 2.0**-6) + s * 2.0**-9).all()
    differ = (q.view(torch.uint8) != q_ref.view(torch.uint8)).float().mean().item()
    assert differ < 1e-2, f"{differ:.2%} of fp8 codes differ from the reference"


FP8_SHAPES = [
    (1, 5376, 3),
    (128, 5376, 9),  # MiniMax-H3 hidden size
    (333, 320, 4),
    (64, 8192, 6),
    (5, 3072, 2),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("M,N,G", FP8_SHAPES)
@pytest.mark.parametrize("runs", [True, False])
def test_fp8_per_token_quant_matches_adaln_then_quant(dtype, M, N, G, runs):
    x, weight, shift, scale, indices = make(M, N, G, dtype, "cuda", runs=runs)
    q, s = fused_rmsnorm_indexed_adaln_fp8_per_token_quant(
        x, weight, shift, scale, indices, eps=1e-5
    )
    assert_quantizes(q, s, reference(x, weight, shift, scale, indices, 1e-5))


def make_gated(M, N, G, dtype, runs, seed=1):
    residual, weight, shift, scale, indices = make(M, N, G, dtype, "cuda", runs=runs)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    branch = torch.randn(M, N, generator=gen, device="cuda", dtype=dtype)
    gate = torch.randn(G, N, generator=gen, device="cuda", dtype=dtype) * 0.5
    return residual, gate, branch, weight, shift, scale, indices


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("M,N,G", FP8_SHAPES)
@pytest.mark.parametrize("runs", [True, False])
def test_gated_fp8_per_token_quant_matches_unfused(dtype, M, N, G, runs):
    residual, gate, branch, weight, shift, scale, indices = make_gated(
        M, N, G, dtype, runs
    )
    residual_out, q, s = fused_gated_residual_rmsnorm_indexed_adaln_fp8_per_token_quant(
        residual, gate, branch, weight, shift, scale, indices, eps=1e-5
    )
    want_residual = (
        residual.float() + gate.index_select(0, indices).float() * branch.float()
    ).to(dtype)
    assert residual_out.dtype == dtype
    tol = {torch.float32: 1e-6, torch.float16: 1e-3, torch.bfloat16: 8e-3}[dtype]
    torch.testing.assert_close(residual_out, want_residual, atol=tol, rtol=tol)
    # Quantize the norm of the residual the kernel actually stored.
    want = reference(residual_out, weight, shift, scale, indices, 1e-5)
    assert_quantizes(q, s, want)


def test_gated_residual_out_matches_fused_indexed_gated_residual():
    residual, gate, branch, weight, shift, scale, indices = make_gated(
        256, 5376, 9, torch.bfloat16, runs=False
    )
    residual_out, _, _ = fused_gated_residual_rmsnorm_indexed_adaln_fp8_per_token_quant(
        residual, gate, branch, weight, shift, scale, indices
    )
    want = fused_indexed_gated_residual(residual, branch, gate, indices)
    torch.testing.assert_close(residual_out, want, atol=0, rtol=0)


def test_gated_fp32_product_is_rounded_before_the_add():
    # An fma would keep gate * branch = 2**46 - 1 exact and store -1, not 0.
    M, N = 2, 256
    residual, gate, branch, weight, shift, scale, indices = make_gated(
        M, N, 3, torch.float32, runs=True
    )
    residual[:, 0] = -(2.0**46)
    gate[:, 0] = 2.0**23 + 1
    branch[:, 0] = 2.0**23 - 1
    residual_out, _, _ = fused_gated_residual_rmsnorm_indexed_adaln_fp8_per_token_quant(
        residual, gate, branch, weight, shift, scale, indices
    )
    want = residual + gate.index_select(0, indices) * branch
    assert torch.equal(residual_out, want)
    assert not residual_out[:, 0].any()


def test_gated_rejects_mismatched_gate_table():
    residual, gate, branch, weight, shift, scale, indices = make_gated(
        8, 256, 3, torch.bfloat16, runs=True
    )
    with pytest.raises(AssertionError):
        fused_gated_residual_rmsnorm_indexed_adaln_fp8_per_token_quant(
            residual, gate[:1], branch, weight, shift, scale, indices
        )


def test_fp8_per_token_quant_zero_row_and_int32_indices():
    x, weight, shift, scale, indices = make(16, 1024, 3, torch.bfloat16, "cuda")
    shift[1] = 0
    scale[1] = -1  # rows indexed 1 modulate to exactly zero
    indices[3] = 1
    q, s = fused_rmsnorm_indexed_adaln_fp8_per_token_quant(
        x, weight, shift, scale, indices.to(torch.int32)
    )
    q64, s64 = fused_rmsnorm_indexed_adaln_fp8_per_token_quant(
        x, weight, shift, scale, indices
    )
    assert torch.equal(q.view(torch.uint8), q64.view(torch.uint8))
    assert torch.equal(s, s64)
    fp8_max = torch.finfo(FP8).max
    assert (
        s[3].item() == pytest.approx(1.0 / (fp8_max * 512)) and not q[3].float().any()
    )


def test_fp8_per_token_quant_empty_and_rejects():
    x, weight, shift, scale, indices = make(0, 256, 3, torch.bfloat16, "cuda")
    q, s = fused_rmsnorm_indexed_adaln_fp8_per_token_quant(
        x, weight, shift, scale, indices
    )
    assert q.shape == (0, 256) and s.shape == (0, 1)
    x, weight, shift, scale, indices = make(8, 256, 3, torch.bfloat16, "cuda")
    with pytest.raises(AssertionError):
        fused_rmsnorm_indexed_adaln_fp8_per_token_quant(
            x, weight[:-1], shift, scale, indices
        )
    with pytest.raises(AssertionError):
        fused_rmsnorm_indexed_adaln_fp8_per_token_quant(
            x[:, ::2], weight[:128], shift[:, :128], scale[:, :128], indices
        )
    with pytest.raises(AssertionError):
        fused_rmsnorm_indexed_adaln_fp8_per_token_quant(
            x, weight, shift, scale, indices.float()
        )
