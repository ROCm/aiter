"""Tests for the FlyDSL HSTU attention backward kernel.

Driven test-first per docs/2026-07-07_HSTU_backward_plan.md. The numerical
oracle is torch.autograd.grad on the PyTorch reference torch_hstu_attention;
dO is synthetic. The FlyDSL forward is only needed for the end-to-end autograd
phase, so these tests do not depend on it.
"""

import pytest
import torch

import aiter.ops.flydsl.hstu_attention_kernels as hstu_kernels
from aiter.ops.flydsl.hstu_attention_kernels import (
    flydsl_hstu_attention_bwd,
    _validate_bwd_inputs,
)

# Reuse the forward test's self-contained input generator.
from op_tests.flydsl_tests.test_flydsl_hstu_attention import (
    generate_hstu_attn_inputs,
)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA device"
)


# --------------------------------------------------------------------------- #
# Reference gradient oracle
# --------------------------------------------------------------------------- #


def hstu_bwd_reference(
    N,
    alpha,
    q,
    k,
    v,
    seq_offsets,
    causal,
    num_targets,
    max_attn_len,
    contextual_seq_len,
    dout,
):
    """Ground-truth (dq, dk, dv) via autograd on the torch reference.

    Computed in fp32 for a clean oracle; the FlyDSL kernel runs in {f16,bf16}
    and is compared with relaxed tolerances established in Phase 3.
    """
    from op_tests.triton_tests.utils.hstu_attention_ref import torch_hstu_attention

    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)

    out = torch_hstu_attention(
        N,
        alpha,
        qf,
        kf,
        vf,
        seq_offsets,
        causal,
        dropout_pr=0.0,
        training=False,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        min_full_attn_seq_len=0,
    )
    dq, dk, dv = torch.autograd.grad(out, (qf, kf, vf), grad_outputs=dout.float())
    return dq, dk, dv


@requires_cuda
def test_reference_oracle_runs():
    """The oracle itself runs and returns correctly-shaped grads (no FlyDSL)."""
    batch, heads, attn_dim, hidden_dim = 8, 1, 64, 64
    max_seq_len = 256

    q, k, v, seq_offsets, num_targets = generate_hstu_attn_inputs(
        batch_size=batch,
        max_seq_len=max_seq_len,
        sparsity=0.5,
        heads=heads,
        attn_dim=attn_dim,
        hidden_dim=hidden_dim,
        target_size=0,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    dout = torch.randn_like(v)
    alpha = 1.0 / attn_dim * 10000

    dq, dk, dv = hstu_bwd_reference(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        True,
        num_targets,
        0,
        0,
        dout,
    )

    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape
    assert torch.isfinite(dq).all()
    assert torch.isfinite(dk).all()
    assert torch.isfinite(dv).all()


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def _qkv_do(batch=2, tokens=8, heads=4, attn_dim=128, hidden_dim=128, device="cuda"):
    q = torch.zeros((tokens, heads, attn_dim), dtype=torch.bfloat16, device=device)
    k = torch.zeros_like(q)
    v = torch.zeros((tokens, heads, hidden_dim), dtype=torch.bfloat16, device=device)
    dout = torch.zeros_like(v)
    seq_offsets = torch.zeros(batch + 1, dtype=torch.int64, device=device)
    return q, k, v, dout, seq_offsets


@requires_cuda
def test_validate_bwd_inputs_ok():
    q, k, v, dout, seq_offsets = _qkv_do(batch=2, heads=4, attn_dim=128, hidden_dim=96)
    actual = _validate_bwd_inputs(q, k, v, dout, seq_offsets, None)
    assert actual == (2, 4, 128, 96, "bf16")


@requires_cuda
def test_validate_bwd_inputs_rejects_dout_shape_mismatch():
    q, k, v, dout, seq_offsets = _qkv_do()
    dout = torch.zeros(
        (v.shape[0], v.shape[1], v.shape[2] + 16), dtype=v.dtype, device=v.device
    )
    with pytest.raises(ValueError):
        _validate_bwd_inputs(q, k, v, dout, seq_offsets, None)


@requires_cuda
def test_validate_bwd_inputs_rejects_dout_dtype_mismatch():
    q, k, v, dout, seq_offsets = _qkv_do()
    dout = dout.to(torch.float16)
    with pytest.raises(ValueError):
        _validate_bwd_inputs(q, k, v, dout, seq_offsets, None)


def test_validate_bwd_inputs_rejects_cpu_tensors():
    q, k, v, dout, seq_offsets = _qkv_do(device="cpu")
    with pytest.raises(ValueError):
        _validate_bwd_inputs(q, k, v, dout, seq_offsets, None)


# --------------------------------------------------------------------------- #
# Kernel entry point (drives Phase 1+): currently a stub
# --------------------------------------------------------------------------- #


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,heads,attn_dim,hidden_dim,max_seq_len",
    [
        (8, 1, 64, 64, 256),  # single/few tiles
        (8, 4, 64, 64, 512),  # multi-tile
        (16, 2, 128, 128, 1024),  # larger, multi-tile
    ],
)
def test_flydsl_bwd_dv_causal(batch, heads, attn_dim, hidden_dim, max_seq_len, dtype):
    alpha = 1.0 / attn_dim * 10000

    q, k, v, seq_offsets, num_targets = generate_hstu_attn_inputs(
        batch_size=batch,
        max_seq_len=max_seq_len,
        sparsity=0.5,
        heads=heads,
        attn_dim=attn_dim,
        hidden_dim=hidden_dim,
        target_size=0,
        dtype=dtype,
        device=torch.device("cuda"),
    )
    dout = torch.randn_like(v)

    _, _, dv_ref = hstu_bwd_reference(
        max_seq_len, alpha, q, k, v, seq_offsets, True, num_targets, 0, 0, dout
    )
    _, _, dv = flydsl_hstu_attention_bwd(
        max_seq_len, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0
    )
    # Relaxed tolerance: bf16/f16 inputs + fast-math SiLU (non-IEEE) recompute.
    torch.testing.assert_close(dv.float(), dv_ref, atol=2e-2, rtol=2e-2)
