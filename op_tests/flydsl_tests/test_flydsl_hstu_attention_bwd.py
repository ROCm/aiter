"""Tests for the FlyDSL HSTU attention backward kernel.

Driven test-first per docs/2026-07-07_HSTU_backward_plan.md. The numerical
oracle is torch.autograd.grad on the PyTorch reference torch_hstu_attention;
dO is synthetic. The FlyDSL forward is only needed for the end-to-end autograd
phase, so these tests do not depend on it.
"""

import csv

import pytest
import torch

import aiter.ops.flydsl.hstu_attention_kernels as hstu_kernels
from aiter.ops.flydsl.hstu_attention_kernels import (
    flydsl_hstu_attention_bwd,
    flydsl_hstu_attention,
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


def hstu_bwd_reference_causal_dense(N, alpha, q, k, v, seq_offsets, dout):
    """Causal-only dense oracle that supports attn_dim != hidden_dim.

    The vendored torch_hstu_attention reshapes v with q's head_dim, so it only works
    when attn_dim == hidden_dim. This hand-rolled per-sequence reference avoids that
    and is used for the asymmetric-dim tests.
    """
    import torch.nn.functional as F

    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)

    offs = seq_offsets.tolist()
    outs = []
    for b in range(len(offs) - 1):
        s, e = offs[b], offs[b + 1]
        n = e - s
        if n == 0:
            continue
        Q, K, V = qf[s:e], kf[s:e], vf[s:e]  # (n, H, d)
        scores = torch.einsum("xha,yha->hxy", Q, K) * alpha
        attn = F.silu(scores) / N
        mask = torch.tril(torch.ones(n, n, device=q.device))  # i >= j (causal + diagonal)
        attn = attn * mask.unsqueeze(0)
        outs.append(torch.einsum("hxy,yhv->xhv", attn, V))  # (n, H, dv)

    out = torch.cat(outs, dim=0)
    return torch.autograd.grad(out, (qf, kf, vf), grad_outputs=dout.float())


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "attn_dim,hidden_dim",
    [(128, 64), (64, 128), (128, 256)],
)
def test_flydsl_bwd_asymmetric_dims(attn_dim, hidden_dim, dtype):
    batch, heads, max_seq_len = 16, 2, 512
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

    dq_ref, dk_ref, dv_ref = hstu_bwd_reference_causal_dense(
        max_seq_len, alpha, q, k, v, seq_offsets, dout
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(
        max_seq_len, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0
    )
    torch.testing.assert_close(dv.float(), dv_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dk.float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dq.float(), dq_ref, atol=3e-2, rtol=3e-2)


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


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "max_attn_len,contextual_seq_len,target_size",
    [
        (0, 0, 20),  # num_targets
        (64, 0, 0),  # sliding window
        (0, 64, 0),  # contextual prefix
        (64, 0, 20),  # window + targets
    ],
)
def test_flydsl_bwd_variants(max_attn_len, contextual_seq_len, target_size, dtype):
    batch, heads, attn_dim, hidden_dim, max_seq_len = 32, 4, 128, 128, 512
    alpha = 1.0 / attn_dim * 10000

    q, k, v, seq_offsets, num_targets = generate_hstu_attn_inputs(
        batch_size=batch,
        max_seq_len=max_seq_len,
        sparsity=0.5,
        heads=heads,
        attn_dim=attn_dim,
        hidden_dim=hidden_dim,
        target_size=target_size,
        dtype=dtype,
        device=torch.device("cuda"),
    )
    dout = torch.randn_like(v)

    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        max_seq_len, alpha, q, k, v, seq_offsets, True, num_targets,
        max_attn_len, contextual_seq_len, dout,
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(
        max_seq_len, alpha, q, k, v, dout, seq_offsets, True, num_targets,
        max_attn_len, contextual_seq_len,
    )
    torch.testing.assert_close(dv.float(), dv_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dk.float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dq.float(), dq_ref, atol=3e-2, rtol=3e-2)


def test_silu_derivative_formula():
    """Lock the SiLU-derivative gate silu'(s) = sigma(s)*(1 + s*(1-sigma(s)))
    against torch autograd, independent of the kernel."""
    import torch.nn.functional as F

    s = torch.linspace(-8.0, 8.0, 257, dtype=torch.float64, requires_grad=True)
    (grad_ref,) = torch.autograd.grad(F.silu(s).sum(), s)

    sig = torch.sigmoid(s.detach())
    grad_formula = sig * (1.0 + s.detach() * (1.0 - sig))

    torch.testing.assert_close(grad_formula, grad_ref, atol=1e-10, rtol=1e-10)


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
def test_flydsl_bwd_all_causal(batch, heads, attn_dim, hidden_dim, max_seq_len, dtype):
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

    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        max_seq_len, alpha, q, k, v, seq_offsets, True, num_targets, 0, 0, dout
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(
        max_seq_len, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0
    )
    # Relaxed tolerance: bf16/f16 inputs + fast-math SiLU (non-IEEE) recompute.
    # dQ/dK carry extra matmuls (dA and dS reductions) so they accumulate more error than dV.
    torch.testing.assert_close(dv.float(), dv_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dk.float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dq.float(), dq_ref, atol=3e-2, rtol=3e-2)


# --------------------------------------------------------------------------- #
# Tiling / block-size overrides
# --------------------------------------------------------------------------- #


@requires_cuda
@pytest.mark.parametrize(
    "block_m,block_n,num_waves,waves_per_eu",
    [
        (64, 32, 4, 0),  # default
        (128, 32, 4, 0),
        (128, 64, 4, 0),
        (192, 32, 4, 0),  # tuned dV/dK pick at N=2048
        (128, 32, 4, 2),  # tuned dQ pick (forced-occupancy, small scratch spill)
        (64, 32, 2, 0),
    ],
)
def test_flydsl_bwd_block_size_overrides(block_m, block_n, num_waves, waves_per_eu):
    """Grads stay correct across explicit tile configs (tiling independence)."""
    batch, heads, attn_dim, hidden_dim, max_seq_len = 16, 2, 128, 128, 1024
    alpha = 1.0 / attn_dim * 10000

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

    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        max_seq_len, alpha, q, k, v, seq_offsets, True, num_targets, 0, 0, dout
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(
        max_seq_len, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0,
        block_m=block_m, block_n=block_n, num_waves=num_waves,
        waves_per_eu=waves_per_eu,
    )
    torch.testing.assert_close(dv.float(), dv_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dk.float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dq.float(), dq_ref, atol=3e-2, rtol=3e-2)


# --------------------------------------------------------------------------- #
# Backward tuned-CSV loading
# --------------------------------------------------------------------------- #


def _bwd_row(**overrides) -> dict:
    row = dict(
        arch=hstu_kernels._GPU_ARCH,
        dtype="bf16",
        num_heads=4,
        head_dim=128,
        hidden_dim=128,
        batch=256,
        max_seq_len=1024,
        has_window="False",
        has_contextual="False",
        has_targets="False",
        kernel="dv",
        block_m=128,
        block_n=64,
        num_waves=4,
        waves_per_eu=2,
        duration=1.0,
    )
    row.update(overrides)
    return row


def _write_bwd_csv(path, rows) -> str:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=hstu_kernels._BWD_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def test_bwd_tuned_csv_is_picked_up(tmp_path):
    path = _write_bwd_csv(tmp_path / "tuned_bwd.csv", [_bwd_row()])

    config_map = hstu_kernels._bwd_tuned_config_map(path)

    assert len(config_map) == 1
    (config,) = config_map.values()
    assert config == dict(block_m=128, block_n=64, num_waves=4, waves_per_eu=2)


def test_bwd_tuned_csv_missing_file_returns_empty(tmp_path):
    assert hstu_kernels._bwd_tuned_config_map(str(tmp_path / "nope.csv")) == {}


def test_bwd_tuned_csv_best_duration_wins(tmp_path):
    path = _write_bwd_csv(
        tmp_path / "tuned_bwd.csv",
        [
            _bwd_row(duration=5.0, block_m=64),
            _bwd_row(duration=1.0, block_m=256),
        ],
    )

    config_map = hstu_kernels._bwd_tuned_config_map(path)

    (config,) = config_map.values()
    assert config["block_m"] == 256


def test_bwd_tuned_csv_per_kernel_configs(tmp_path):
    """dV, dK and dQ each resolve independent tuned configs for the same problem."""
    path = _write_bwd_csv(
        tmp_path / "tuned_bwd.csv",
        [
            _bwd_row(kernel="dv", block_m=192, block_n=32, num_waves=4, waves_per_eu=0),
            _bwd_row(kernel="dk", block_m=192, block_n=32, num_waves=4, waves_per_eu=2),
            _bwd_row(kernel="dq", block_m=128, block_n=32, num_waves=4, waves_per_eu=2),
        ],
    )

    config_map = hstu_kernels._bwd_tuned_config_map(path)

    assert len(config_map) == 3
    # Keys are (problem_key, kernel); pull each kernel's config.
    by_kernel = {kern: cfg for (_, kern), cfg in config_map.items()}
    assert by_kernel["dv"] == dict(block_m=192, block_n=32, num_waves=4, waves_per_eu=0)
    assert by_kernel["dk"] == dict(block_m=192, block_n=32, num_waves=4, waves_per_eu=2)
    assert by_kernel["dq"] == dict(block_m=128, block_n=32, num_waves=4, waves_per_eu=2)


# --------------------------------------------------------------------------- #
# End-to-end autograd integration
# --------------------------------------------------------------------------- #


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "max_attn_len,contextual_seq_len,target_size",
    [
        (0, 0, 0),    # dense causal
        (0, 0, 20),   # targets
        (64, 0, 0),   # window
    ],
)
def test_flydsl_autograd_end_to_end(max_attn_len, contextual_seq_len, target_size, dtype):
    """FlydslHstuAttention.apply is drop-in differentiable: .grad after .backward()
    matches torch.autograd.grad on the torch reference."""
    batch, heads, attn_dim, hidden_dim, max_seq_len = 32, 4, 128, 128, 512
    alpha = 1.0 / attn_dim * 10000

    q, k, v, seq_offsets, num_targets = generate_hstu_attn_inputs(
        batch_size=batch,
        max_seq_len=max_seq_len,
        sparsity=0.5,
        heads=heads,
        attn_dim=attn_dim,
        hidden_dim=hidden_dim,
        target_size=target_size,
        dtype=dtype,
        device=torch.device("cuda"),
    )
    dout = torch.randn_like(v)

    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        max_seq_len, alpha, q, k, v, seq_offsets, True, num_targets,
        max_attn_len, contextual_seq_len, dout,
    )

    qd = q.detach().clone().requires_grad_(True)
    kd = k.detach().clone().requires_grad_(True)
    vd = v.detach().clone().requires_grad_(True)

    out = flydsl_hstu_attention(
        max_seq_len, alpha, qd, kd, vd, seq_offsets, True, num_targets,
        max_attn_len, contextual_seq_len,
    )
    assert out.requires_grad
    out.backward(dout)

    torch.testing.assert_close(vd.grad.float(), dv_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(kd.grad.float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(qd.grad.float(), dq_ref, atol=3e-2, rtol=3e-2)
