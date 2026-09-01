# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for the vocab-parallel cross-entropy Triton kernels.

Tests the single-GPU (dist_group=None) path of cross_entropy_forward,
cross_entropy_forward_chunked, and cross_entropy_backward against
torch.nn.functional.cross_entropy as the reference.

Multi-GPU / TP-parallel paths require a real process group and are not
covered here; those are exercised by the distributed integration tests.
"""

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.cross_entropy import (
    cross_entropy_backward,
    cross_entropy_forward,
    cross_entropy_forward_chunked,
)

# ── helpers ──────────────────────────────────────────────────────────


def _ref_forward(logits_2d, target_1d, label_smoothing, reduce_loss, ignore_idx):
    """PyTorch reference: F.cross_entropy on a [n_rows, V] view."""
    reduction = "mean" if reduce_loss else "none"
    return F.cross_entropy(
        logits_2d.float(),
        target_1d,
        label_smoothing=label_smoothing,
        reduction=reduction,
        ignore_index=ignore_idx,
    )


# ── forward correctness ──────────────────────────────────────────────


@pytest.mark.parametrize("V", [128, 512, 1000, 4096, 32001])
@pytest.mark.parametrize("B_SQ", [1, 4, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cross_entropy_forward_basic(V, B_SQ, dtype):
    """Loss values match F.cross_entropy for standard CE (no smoothing)."""
    torch.manual_seed(42)
    logits = torch.randn(B_SQ, 1, V, dtype=dtype, device="cuda")
    target = torch.randint(0, V, (B_SQ, 1), device="cuda")

    loss_triton, _ = cross_entropy_forward(
        logits.clone(),
        target,
        label_smoothing=0.0,
        reduce_loss=False,
        dist_group=None,
        ignore_idx=-100,
    )

    ref = _ref_forward(
        logits.reshape(B_SQ, V), target.reshape(B_SQ), 0.0, False, -100
    ).reshape(B_SQ, 1)

    torch.testing.assert_close(loss_triton.float(), ref.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("reduce_loss", [True, False])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.2])
def test_cross_entropy_forward_label_smoothing(reduce_loss, label_smoothing):
    torch.manual_seed(0)
    B, SQ, V = 2, 8, 256
    logits = torch.randn(B, SQ, V, dtype=torch.float32, device="cuda")
    target = torch.randint(0, V, (B, SQ), device="cuda")

    loss_triton, _ = cross_entropy_forward(
        logits.clone(),
        target,
        label_smoothing=label_smoothing,
        reduce_loss=reduce_loss,
        dist_group=None,
        ignore_idx=-100,
    )

    ref = _ref_forward(
        logits.reshape(B * SQ, V),
        target.reshape(B * SQ),
        label_smoothing,
        reduce_loss,
        -100,
    )
    if not reduce_loss:
        ref = ref.reshape(B, SQ)

    torch.testing.assert_close(loss_triton.float(), ref.float(), atol=1e-3, rtol=1e-3)


def test_cross_entropy_forward_ignore_index():
    """Rows with target == ignore_idx must produce loss 0 and zero gradient."""
    torch.manual_seed(7)
    B, SQ, V = 2, 4, 128
    ignore_idx = -100
    logits = torch.randn(B, SQ, V, device="cuda")
    target = torch.randint(0, V, (B, SQ), device="cuda")
    # mask the first row of each batch
    target[0, 0] = ignore_idx
    target[1, 2] = ignore_idx

    logits_clone = logits.clone()
    loss_triton, grad = cross_entropy_forward(
        logits_clone,
        target,
        label_smoothing=0.0,
        reduce_loss=False,
        dist_group=None,
        ignore_idx=ignore_idx,
    )

    ref = _ref_forward(
        logits.reshape(B * SQ, V), target.reshape(B * SQ), 0.0, False, ignore_idx
    ).reshape(B, SQ)

    torch.testing.assert_close(loss_triton.float(), ref.float(), atol=1e-3, rtol=1e-3)

    # grad rows for ignored tokens must be all-zero
    grad_2d = grad.reshape(B * SQ, V)
    target_1d = target.reshape(B * SQ)
    for row in range(B * SQ):
        if target_1d[row] == ignore_idx:
            assert grad_2d[row].abs().max().item() == 0.0, f"row {row} grad not zero"


@pytest.mark.parametrize("V", [127, 511, 32001])  # non-power-of-2 vocab sizes
def test_cross_entropy_forward_nonaligned_vocab(V):
    """Unaligned vocab sizes must not cause silent OOB reads."""
    torch.manual_seed(3)
    B, SQ = 4, 8
    logits = torch.randn(B, SQ, V, device="cuda")
    target = torch.randint(0, V, (B, SQ), device="cuda")

    loss_triton, _ = cross_entropy_forward(
        logits.clone(),
        target,
        label_smoothing=0.0,
        reduce_loss=True,
        dist_group=None,
        ignore_idx=-100,
    )

    ref = _ref_forward(
        logits.reshape(B * SQ, V), target.reshape(B * SQ), 0.0, True, -100
    )
    torch.testing.assert_close(loss_triton.float(), ref.float(), atol=1e-3, rtol=1e-3)


# ── gradient correctness ─────────────────────────────────────────────


def test_cross_entropy_backward_grad_scale():
    """cross_entropy_backward scales the stored gradient by grad_output."""
    torch.manual_seed(5)
    B, SQ, V = 2, 4, 64

    logits = torch.randn(B, SQ, V, device="cuda")
    target = torch.randint(0, V, (B, SQ), device="cuda")

    _, grad_stored = cross_entropy_forward(
        logits.clone(),
        target,
        label_smoothing=0.0,
        reduce_loss=False,
        dist_group=None,
        ignore_idx=-100,
    )
    # Save a copy before in-place backward
    grad_stored_copy = grad_stored.clone()

    grad_output = torch.tensor(2.5, device="cuda")
    grad_out = cross_entropy_backward(grad_stored.clone(), grad_output)

    torch.testing.assert_close(grad_out, grad_stored_copy * 2.5, atol=1e-5, rtol=1e-5)


def test_cross_entropy_backward_identity_skip():
    """grad_output == 1.0 must return the gradient tensor unchanged (fast path)."""
    B, SQ, V = 2, 4, 64
    logits = torch.randn(B, SQ, V, device="cuda")
    target = torch.randint(0, V, (B, SQ), device="cuda")

    _, grad_stored = cross_entropy_forward(
        logits.clone(), target, 0.0, False, None, -100
    )
    grad_id = torch.tensor(1.0, device="cuda")
    result = cross_entropy_backward(
        grad_stored.clone(), grad_id, is_cg_capturable=False
    )
    # fast path: returns the tensor without launching the scale kernel
    torch.testing.assert_close(result, grad_stored, atol=0, rtol=0)


def test_cross_entropy_autograd_e2e():
    """End-to-end autograd: triton loss gradient matches autograd through F.cross_entropy."""
    torch.manual_seed(9)
    B, SQ, V = 2, 6, 128

    logits_ref = torch.randn(B, SQ, V, device="cuda", requires_grad=True)
    logits_tri = logits_ref.detach().clone().requires_grad_(True)
    target = torch.randint(0, V, (B, SQ), device="cuda")

    # Reference: F.cross_entropy mean
    ref_loss = F.cross_entropy(
        logits_ref.reshape(B * SQ, V), target.reshape(B * SQ), reduction="mean"
    )
    ref_loss.backward()

    # Triton path: reduce_loss=True returns a scalar, grad already in grad_input
    _loss_tri, grad_input = cross_entropy_forward(
        logits_tri.reshape(B, SQ, V),
        target,
        label_smoothing=0.0,
        reduce_loss=True,
        dist_group=None,
        ignore_idx=-100,
    )
    # backward scales the already-normalised grad by grad_output=1.0 (no-op fast path)
    grad_out = torch.tensor(1.0, device="cuda")
    cross_entropy_backward(grad_input, grad_out)

    torch.testing.assert_close(
        grad_input.reshape(B * SQ, V).float(),
        logits_ref.grad.reshape(B * SQ, V).float(),
        atol=1e-3,
        rtol=1e-3,
    )


# ── chunked forward ──────────────────────────────────────────────────


@pytest.mark.parametrize("chunk_rows", [1, 4, 16])
@pytest.mark.parametrize("B_SQ", [7, 32])
def test_cross_entropy_forward_chunked_matches_full(chunk_rows, B_SQ):
    """Chunked forward must produce identical loss to the non-chunked path."""
    torch.manual_seed(11)
    V = 256
    logits = torch.randn(B_SQ, 1, V, device="cuda")
    target = torch.randint(0, V, (B_SQ, 1), device="cuda")

    loss_full, _ = cross_entropy_forward(logits.clone(), target, 0.0, False, None, -100)
    loss_chunked, _ = cross_entropy_forward_chunked(
        logits.clone(), target, 0.0, False, None, -100, chunk_rows=chunk_rows
    )

    torch.testing.assert_close(
        loss_full.float(), loss_chunked.float(), atol=1e-4, rtol=1e-4
    )
