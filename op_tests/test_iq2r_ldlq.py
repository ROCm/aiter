# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LDLQ for IQ2R-v2 (aiter/ops/iq2r_v2.py): CPU tests on small matrices."""

import pytest
import torch

from aiter.ops.iq2r_v2 import (
    IQ2R_V2_LEVELS_C4,
    iq2r_v2_codebook_copies,
    iq2r_v2_encode,
    iq2r_v2_ldlq,
    iq2r_v2_ldlq_factor,
    iq2r_v2_materialize,
    iq2r_v2_seed_codebook,
    iq2r_v2_write_auxiliary,
    iq2r_v2_write_data,
)

LEVELS = IQ2R_V2_LEVELS_C4


def _correlated_hessian(k: int, seed: int) -> torch.Tensor:
    """A Hessian with strong low-rank structure, like a layer's MoE input."""
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn((k, 16), generator=generator, dtype=torch.float64)
    x = torch.randn((4096, 16), generator=generator, dtype=torch.float64) @ basis.T
    x += 0.3 * torch.randn((4096, k), generator=generator, dtype=torch.float64)
    return (x.T @ x / x.shape[0]).float()


def _problem(n: int, k: int, entries: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn((n, k), generator=generator) * 0.02
    book = iq2r_v2_seed_codebook("cpu", entries)
    enc = iq2r_v2_encode(weight, torch.ones(k), book, LEVELS)
    return weight, book, enc


def _output_error(weight, reconstruction, hessian):
    delta = (reconstruction - weight).double()
    h = hessian.double()
    return float(((delta @ h) * delta).sum() / ((weight.double() @ h) * weight).sum())


def _nearest_codewords(weight, enc, book):
    """Reference without feedback: each vector's nearest codeword of its copy."""
    n, k = weight.shape
    copies = iq2r_v2_codebook_copies(book, LEVELS)
    level = enc.levels.long()
    scale = torch.exp2(
        (enc.d.float() - 127).repeat_interleave(8, 1) + (level // 4).float()
    ).repeat_interleave(4, 1)
    copy = (level % 4).repeat_interleave(4, 1)
    vectors = weight.abs().reshape(n, k // 8, 8) / scale[..., None]
    candidates = copies[copy]  # [N, K/8, E, 8]
    index = (candidates - vectors[..., None, :]).square().sum(-1).argmin(-1)
    return index


def test_ldlq_without_correlation_is_nearest_codeword():
    """H = I, no damping: no feedback, so every vector takes its nearest codeword."""
    weight, book, enc = _problem(32, 512, 512, 0)
    factor = iq2r_v2_ldlq_factor(torch.eye(512), damp=0.0)
    out = iq2r_v2_ldlq(weight, enc, book, factor, LEVELS)
    assert torch.equal(out.indices, _nearest_codewords(weight, enc, book))
    assert torch.equal(out.negative, weight < 0)
    assert torch.equal(out.levels, enc.levels) and torch.equal(out.d, enc.d)


@pytest.mark.parametrize("entries", [512, 1024])
def test_ldlq_lowers_the_output_error(entries):
    weight, book, enc = _problem(64, 1024, entries, 1)
    hessian = _correlated_hessian(1024, 2)
    out = iq2r_v2_ldlq(weight, enc, book, iq2r_v2_ldlq_factor(hessian), LEVELS)
    before = _output_error(weight, enc.reconstruction, hessian)
    after = _output_error(weight, out.reconstruction, hessian)
    assert after < 0.8 * before


def test_ldlq_batch_does_not_change_the_choice():
    weight, book, enc = _problem(16, 512, 512, 3)
    factor = iq2r_v2_ldlq_factor(_correlated_hessian(512, 4))
    lazy = iq2r_v2_ldlq(weight, enc, book, factor, LEVELS, batch=128)
    eager = iq2r_v2_ldlq(weight, enc, book, factor, LEVELS, batch=8)
    # Same decisions up to FP32 rounding of the feedback order.
    assert (lazy.indices == eager.indices).float().mean() > 0.999
    assert (lazy.negative == eager.negative).float().mean() > 0.999


@pytest.mark.parametrize("entries", [512, 1024])
def test_ldlq_encoding_round_trips_through_the_records(entries):
    """The records written from an LDLQ encoding decode to its reconstruction."""
    weight, book, enc = _problem(32, 512, entries, 5)
    factor = iq2r_v2_ldlq_factor(_correlated_hessian(512, 6))
    out = iq2r_v2_ldlq(weight, enc, book, factor, LEVELS)
    data = iq2r_v2_write_data(out.indices, out.negative, out.levels, LEVELS, entries)
    aux = iq2r_v2_write_auxiliary(book[None], out.d, (1.0, 1.0))
    dense = iq2r_v2_materialize(data, aux, 32, 512, LEVELS, codebook_entries=entries)
    assert torch.equal(dense, out.reconstruction)


def test_ldlq_factor_inverts_the_damped_hessian():
    hessian = _correlated_hessian(64, 7)
    factor = iq2r_v2_ldlq_factor(hessian, damp=1.0)
    damped = hessian.double() + hessian.diagonal().double().mean() * torch.eye(64)
    u = factor.u.double()
    torch.testing.assert_close(u.T @ u, torch.linalg.inv(damped), rtol=1e-4, atol=1e-6)
    for j in range(8):
        block = u[8 * j : 8 * j + 8, 8 * j : 8 * j + 8]
        torch.testing.assert_close(
            factor.block_inverses[j].double(),
            torch.linalg.inv(block),
            rtol=1e-4,
            atol=1e-4,
        )


def test_ldlq_rejects_mismatched_shapes():
    weight, book, enc = _problem(8, 256, 512, 8)
    with pytest.raises(ValueError, match="does not match"):
        iq2r_v2_ldlq(weight, enc, book, iq2r_v2_ldlq_factor(torch.eye(512)), LEVELS)
    with pytest.raises(ValueError, match="batch"):
        iq2r_v2_ldlq(
            weight, enc, book, iq2r_v2_ldlq_factor(torch.eye(256)), LEVELS, batch=12
        )
