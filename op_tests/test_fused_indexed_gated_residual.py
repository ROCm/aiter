# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import sys

import pytest
import torch

from aiter import fused_indexed_gated_residual


def reference(residual, x, gate, indices, round_intermediate=False):
    """What the fused op replaces; with rounding it is the eager expression."""
    gated = gate.index_select(0, indices).float() * x.float()
    if round_intermediate:
        gated = gated.to(x.dtype).float()
    return (residual.float() + gated).to(residual.dtype)


def make(M, N, G, dtype, *, runs=True, seed=0):
    """``runs`` mirrors a packed diffusion sequence: consecutive tokens share a
    gate index. The scattered case is the adversarial one."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(M, N, generator=gen, device="cuda", dtype=dtype) * 4
    x = torch.randn(M, N, generator=gen, device="cuda", dtype=dtype)
    gate = torch.randn(G, N, generator=gen, device="cuda", dtype=dtype)
    if runs:
        indices = (torch.arange(M, device="cuda") * G // max(M, 1)).to(torch.int64)
    else:
        indices = torch.randint(
            0, G, (M,), generator=gen, device="cuda", dtype=torch.int64
        )
    return residual, x, gate, indices


@pytest.mark.parametrize("round_intermediate", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "M,N,G",
    [
        (1, 5376, 3),  # single token
        (128, 5376, 9),  # MiniMax-H3: hidden 5376, 3 modalities x 3 timesteps
        (1024, 1536, 2),
        (333, 320, 4),
        (64, 8192, 6),
        (17, 5, 2),  # N not a multiple of the vector width: scalar path
    ],
)
@pytest.mark.parametrize("runs", [True, False])
def test_matches_gathered_gate_then_add(dtype, M, N, G, runs, round_intermediate):
    residual, x, gate, indices = make(M, N, G, dtype, runs=runs)
    got = fused_indexed_gated_residual(
        residual, x, gate, indices, round_intermediate=round_intermediate
    )
    want = reference(residual, x, gate, indices, round_intermediate)
    assert got.shape == residual.shape and got.dtype == residual.dtype
    if round_intermediate:
        # One rounding of the product and one of the sum, like the eager
        # expression, so the result is reproduced bit for bit.
        torch.testing.assert_close(got, want, atol=0, rtol=0)
        torch.testing.assert_close(
            got, residual + gate.index_select(0, indices) * x, atol=0, rtol=0
        )
    else:
        # The product stays in fp32; only the sum is rounded, like the reference.
        torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_round_intermediate_changes_the_result():
    residual, x, gate, indices = make(512, 5376, 3, torch.bfloat16)
    rounded = fused_indexed_gated_residual(
        residual, x, gate, indices, round_intermediate=True
    )
    full = fused_indexed_gated_residual(residual, x, gate, indices)
    assert not torch.equal(rounded, full), "test input does not expose rounding"


def test_every_index_is_honoured():
    N, G = 512, 6
    residual, x, gate, _ = make(G, N, G, torch.float32)
    indices = torch.arange(G, device="cuda", dtype=torch.int64)
    got = fused_indexed_gated_residual(residual, x, gate, indices)
    torch.testing.assert_close(
        got, reference(residual, x, gate, indices), atol=0, rtol=0
    )


def test_out_parameter_and_in_place_update():
    residual, x, gate, indices = make(64, 1024, 3, torch.bfloat16)
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


@pytest.mark.parametrize("offset", [0, 1])
def test_strided_and_unaligned_rows(offset):
    """Rows may be views into wider buffers; an odd column offset breaks 16-byte
    alignment and must fall back to the scalar path, not misread."""
    M, N, G = 96, 640, 4
    residual, x, gate, indices = make(M, N, G, torch.bfloat16, runs=False)
    wide = torch.zeros(M, 3 * N + 1, device="cuda", dtype=torch.bfloat16)
    wide[:, N + offset : 2 * N + offset] = x
    gate_wide = torch.zeros(G, 2 * N, device="cuda", dtype=torch.bfloat16)
    gate_wide[:, :N] = gate
    out_wide = torch.zeros(M, 2 * N + 1, device="cuda", dtype=torch.bfloat16)
    got = fused_indexed_gated_residual(
        residual,
        wide[:, N + offset : 2 * N + offset],
        gate_wide[:, :N],
        indices,
        out=out_wide[:, N + offset : 2 * N + offset],
        round_intermediate=True,
    )
    torch.testing.assert_close(
        got, reference(residual, x, gate, indices, True), atol=0, rtol=0
    )
    assert torch.count_nonzero(out_wide[:, : N + offset]) == 0


def test_int32_indices_are_accepted():
    residual, x, gate, indices = make(64, 1024, 3, torch.bfloat16, runs=False)
    got = fused_indexed_gated_residual(residual, x, gate, indices.to(torch.int32))
    want = fused_indexed_gated_residual(residual, x, gate, indices)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_special_values_match_eager():
    residual, x, gate, indices = make(8, 64, 2, torch.bfloat16)
    residual[0, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0])
    x[1, :3] = torch.tensor([float("inf"), float("nan"), 1e38])
    gate[indices[1], :3] = torch.tensor([0.0, 1.0, 1e3])
    got = fused_indexed_gated_residual(
        residual, x, gate, indices, round_intermediate=True
    )
    want = residual + gate.index_select(0, indices) * x
    torch.testing.assert_close(got, want, atol=0, rtol=0, equal_nan=True)


def test_empty_rows():
    residual, x, gate, indices = make(0, 256, 2, torch.bfloat16)
    assert fused_indexed_gated_residual(residual, x, gate, indices).shape == (0, 256)


def test_rejects_mismatched_inputs():
    residual, x, gate, indices = make(32, 256, 3, torch.bfloat16)
    with pytest.raises(ValueError):
        fused_indexed_gated_residual(residual, x[:, :-1], gate, indices)
    with pytest.raises(ValueError):
        fused_indexed_gated_residual(residual, x, gate[:, :-1], indices)
    with pytest.raises(ValueError):
        fused_indexed_gated_residual(residual, x, gate, indices[:-1])
    with pytest.raises(ValueError):
        fused_indexed_gated_residual(residual, x, gate.half(), indices)
    with pytest.raises(ValueError):
        fused_indexed_gated_residual(residual, x, gate, indices.float())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
