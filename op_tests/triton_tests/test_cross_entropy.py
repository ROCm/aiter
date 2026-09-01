# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for the vocab-parallel cross-entropy Triton kernels.

Tests the single-GPU (dist_group=None) path of cross_entropy_forward,
cross_entropy_forward_chunked, and cross_entropy_backward against
torch.nn.functional.cross_entropy as the reference.

The multi-GPU / TP-parallel path (world_size > 1: gathered-stat layout and
TP label smoothing) requires a real process group and is currently NOT
covered by any test — it needs a multi-process TP harness (follow-up).
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


@pytest.mark.parametrize("V", [128, 32001])  # aligned + large non-aligned
@pytest.mark.parametrize("B_SQ", [1, 64])
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


# ls=0.0 (standard CE) is covered by test_cross_entropy_forward_basic; the
# smoothing term is linear in ls, so one non-zero value is enough.
@pytest.mark.parametrize("reduce_loss", [True, False])
@pytest.mark.parametrize("label_smoothing", [0.1])
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


def test_cross_entropy_forward_reduce_with_ignore_index():
    """reduce_loss=True must divide by the non-ignored row count, not n_rows."""
    torch.manual_seed(13)
    B, SQ, V = 4, 8, 256
    ignore_idx = -100
    logits = torch.randn(B, SQ, V, device="cuda")
    target = torch.randint(0, V, (B, SQ), device="cuda")
    # ignore a third of the rows
    flat = target.reshape(-1)
    flat[::3] = ignore_idx

    loss_triton, grad = cross_entropy_forward(
        logits.clone(),
        target,
        label_smoothing=0.0,
        reduce_loss=True,
        dist_group=None,
        ignore_idx=ignore_idx,
    )

    # F.cross_entropy(reduction="mean") already averages over non-ignored rows.
    ref_loss = _ref_forward(
        logits.reshape(B * SQ, V), target.reshape(B * SQ), 0.0, True, ignore_idx
    )
    torch.testing.assert_close(
        loss_triton.float(), ref_loss.float(), atol=1e-3, rtol=1e-3
    )

    # gradient must also be normalized by the non-ignored count
    logits_ref = logits.clone().requires_grad_(True)
    F.cross_entropy(
        logits_ref.reshape(B * SQ, V),
        target.reshape(B * SQ),
        reduction="mean",
        ignore_index=ignore_idx,
    ).backward()
    torch.testing.assert_close(
        grad.reshape(B * SQ, V).float(),
        logits_ref.grad.reshape(B * SQ, V).float(),
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize("V", [127, 511])  # non-power-of-2 vocab sizes
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


# ── input-layout robustness ──────────────────────────────────────────


@pytest.mark.parametrize("chunked", [False, True])
def test_cross_entropy_forward_noncontiguous_input(chunked):
    """Non-contiguous [B, SQ, V] input must still produce correct loss+grad."""
    torch.manual_seed(21)
    B, SQ, V = 3, 5, 128
    # Build a non-contiguous [B, SQ, V] view (inner stride 1, wrong batch stride).
    base = torch.randn(B, V, SQ, device="cuda")
    logits = base.transpose(1, 2)  # [B, SQ, V], not contiguous
    assert not logits.is_contiguous()
    target = torch.randint(0, V, (B, SQ), device="cuda")

    if chunked:
        loss, grad = cross_entropy_forward_chunked(
            logits, target, 0.0, False, None, -100, chunk_rows=2
        )
    else:
        loss, grad = cross_entropy_forward(logits, target, 0.0, False, None, -100)

    ref = _ref_forward(
        logits.reshape(B * SQ, V), target.reshape(B * SQ), 0.0, False, -100
    ).reshape(B, SQ)
    torch.testing.assert_close(loss.float(), ref.float(), atol=1e-3, rtol=1e-3)

    logits_ref = logits.contiguous().requires_grad_(True)
    F.cross_entropy(
        logits_ref.reshape(B * SQ, V), target.reshape(B * SQ), reduction="sum"
    ).backward()
    torch.testing.assert_close(
        grad.reshape(B * SQ, V).float(),
        logits_ref.grad.reshape(B * SQ, V).float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_cross_entropy_backward_rejects_bad_grad_shape():
    """grad_output whose numel is neither 1 nor n_rows must raise."""
    B, SQ, V = 2, 4, 64
    logits = torch.randn(B, SQ, V, device="cuda")
    target = torch.randint(0, V, (B, SQ), device="cuda")
    _, grad_stored = cross_entropy_forward(
        logits.clone(), target, 0.0, False, None, -100
    )

    bad = torch.ones(B * SQ + 1, device="cuda")  # wrong length
    with pytest.raises(ValueError):
        cross_entropy_backward(grad_stored, bad)


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


# chunk=1 fully splits; chunk=16 > B_SQ=7 exercises the last-partial-chunk path.
@pytest.mark.parametrize("chunk_rows", [1, 16])
@pytest.mark.parametrize("B_SQ", [7])
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


@pytest.mark.parametrize("reduce_loss", [True, False])
def test_cross_entropy_forward_chunked_ignore_index(reduce_loss):
    """Chunked path must zero ignored rows and match the non-chunked path."""
    torch.manual_seed(17)
    B_SQ, V, ignore_idx = 12, 256, -100
    logits = torch.randn(B_SQ, 1, V, device="cuda")
    target = torch.randint(0, V, (B_SQ, 1), device="cuda")
    target.reshape(-1)[::4] = ignore_idx  # ignore every 4th row

    loss_full, grad_full = cross_entropy_forward(
        logits.clone(), target, 0.0, reduce_loss, None, ignore_idx
    )
    loss_chunked, grad_chunked = cross_entropy_forward_chunked(
        logits.clone(), target, 0.0, reduce_loss, None, ignore_idx, chunk_rows=5
    )

    torch.testing.assert_close(
        loss_chunked.float(), loss_full.float(), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        grad_chunked.float(), grad_full.float(), atol=1e-4, rtol=1e-4
    )


def test_cross_entropy_forward_chunked_rejects_zero_chunk():
    """chunk_rows < 1 must raise instead of looping forever."""
    logits = torch.randn(4, 1, 64, device="cuda")
    target = torch.randint(0, 64, (4, 1), device="cuda")
    with pytest.raises(ValueError):
        cross_entropy_forward_chunked(
            logits, target, 0.0, False, None, -100, chunk_rows=0
        )
