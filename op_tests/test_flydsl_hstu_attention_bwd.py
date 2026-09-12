# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL HSTU attention backward: pytest correctness suite + perf sweep.

The numerical oracle is ``torch.autograd.grad`` on the PyTorch reference
``torch_hstu_attention`` (fp32); dO is synthetic. The FlyDSL forward is only
needed by the end-to-end autograd test, so the gradient correctness tests do not
depend on it.

Imported / collected by pytest, this file is the correctness gate: gradient
values across mask variants, asymmetric dims, tiling overrides, input
validation, tuned-CSV loading, and the end-to-end autograd wiring.

Run as a script instead, it is the op-test-standard perf + correctness sweep:
each swept (b, h, n, d, dtype, mask) case records the kernel ``us`` plus
TFLOPS / TB/s rooflines and the max grad error against the fp32 autograd oracle,
emitting one markdown table per dtype. The oracle pads to dense [B, H, N, N]
fp32 scores, so it is skipped (``err`` empty) for cases where that would not fit;
``--no-ref`` skips it everywhere.

      python op_tests/test_flydsl_hstu_attention_bwd.py -b 120 -s 512 1024 --mask causal hstu
"""

import argparse
import csv
import itertools

import pandas as pd
import pytest
import torch

import aiter
import aiter.ops.flydsl.hstu_attention_kernels as hstu_kernels
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.hstu_attention_kernels import (
    _validate_bwd_inputs,
    flydsl_hstu_attention,
    flydsl_hstu_attention_bwd,
)
from aiter.ops.flydsl.kernels.hstu_attention_bwd import validate_hstu_attention_bwd
from aiter.test_common import benchmark, checkAllclose, run_perftest

# Reuse the forward test's self-contained input generator.
from op_tests.test_flydsl_hstu_attention import (
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

    Computed in fp32 for a clean oracle; the FlyDSL kernel runs in {f16,bf16}.
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
        mask = torch.tril(
            torch.ones(n, n, device=q.device)
        )  # i >= j (causal + diagonal)
        attn = attn * mask.unsqueeze(0)
        outs.append(torch.einsum("hxy,yhv->xhv", attn, V))  # (n, H, dv)

    out = torch.cat(outs, dim=0)
    return torch.autograd.grad(out, (qf, kf, vf), grad_outputs=dout.float())


# --------------------------------------------------------------------------- #
# Gradient comparison
# --------------------------------------------------------------------------- #
TOL_DV = 2e-2
TOL_DQK = 3e-2


def assert_grad_close(name, got, ref, tol):
    """Compare a kernel gradient to the fp32 oracle, atol scaled to the oracle."""
    torch.testing.assert_close(
        got.float(),
        ref,
        atol=tol * ref.abs().max().item(),
        rtol=tol,
        msg=lambda generated: f"{name}: {generated}",
    )


def assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref):
    assert_grad_close("dv", dv, dv_ref, TOL_DV)
    assert_grad_close("dk", dk, dk_ref, TOL_DQK)
    assert_grad_close("dq", dq, dq_ref, TOL_DQK)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "attn_dim,hidden_dim",
    # (96, 128) is the non-64-aligned attn_dim case: the kernel rounds the Q/K LDS
    # stride up to 128, so the padded columns exercise both DMAs' source clamps.
    # (96, 96) and (128, 192) additionally have a hidden_dim the default tile cannot
    # serve, so they only build via the fallback's candidate search.
    [(128, 64), (64, 128), (128, 256), (96, 128), (96, 96), (128, 192)],
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
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "max_attn_len,contextual_seq_len,target_size",
    [
        (0, 0, 20),  # num_targets
        (64, 0, 0),  # sliding window
        (0, 64, 0),  # contextual prefix
        (64, 0, 20),  # window + targets
        # semi_local_fig: window well below seq_len + targets. Exercises the
        # dV/dK targets-aware n_q_tiles cap -- KV tiles far from the target tail get
        # their query sweep capped, while tiles inside the window of max_id must
        # reopen to seq_len so the (raw-position-distant, id-clamped) target queries
        # still contribute. A raw-position-only cap would drop them.
        (256, 0, 20),  # semi_local_fig (window << seq + targets)
        (128, 0, 20),
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
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        True,
        num_targets,
        max_attn_len,
        contextual_seq_len,
        dout,
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        dout,
        seq_offsets,
        True,
        num_targets,
        max_attn_len,
        contextual_seq_len,
    )
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


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
    actual = _validate_bwd_inputs(q, k, v, dout, seq_offsets, None, max_seq_len=8)
    assert actual == (2, 4, 128, 96, "bf16")


@requires_cuda
def test_validate_bwd_inputs_rejects_dout_shape_mismatch():
    q, k, v, dout, seq_offsets = _qkv_do()
    dout = torch.zeros(
        (v.shape[0], v.shape[1], v.shape[2] + 16), dtype=v.dtype, device=v.device
    )
    with pytest.raises(ValueError):
        _validate_bwd_inputs(q, k, v, dout, seq_offsets, None, max_seq_len=8)


@requires_cuda
def test_validate_bwd_inputs_rejects_dout_dtype_mismatch():
    q, k, v, dout, seq_offsets = _qkv_do()
    dout = dout.to(torch.float16)
    with pytest.raises(ValueError):
        _validate_bwd_inputs(q, k, v, dout, seq_offsets, None, max_seq_len=8)


def test_validate_bwd_inputs_rejects_cpu_tensors():
    q, k, v, dout, seq_offsets = _qkv_do(device="cpu")
    with pytest.raises(ValueError):
        _validate_bwd_inputs(q, k, v, dout, seq_offsets, None, max_seq_len=8)


# --------------------------------------------------------------------------- #
# Dense causal correctness across problem shapes
# --------------------------------------------------------------------------- #


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,heads,attn_dim,hidden_dim,max_seq_len",
    [
        (8, 1, 64, 64, 256),  # single/few tiles
        (8, 4, 64, 64, 512),  # multi-tile
        (16, 2, 128, 128, 1024),  # larger, multi-tile
        (3, 4, 64, 64, 256),  # batch*heads=12: grid pads the last group
        (5, 1, 64, 64, 256),  # batch*heads=5 < NUM_GRID_GROUPS: mostly padding
    ],
)
def test_flydsl_bwd_dense_causal_shapes(
    batch, heads, attn_dim, hidden_dim, max_seq_len, dtype
):
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
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


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
        max_seq_len,
        alpha,
        q,
        k,
        v,
        dout,
        seq_offsets,
        True,
        num_targets,
        0,
        0,
        block_m=block_m,
        block_n=block_n,
        num_waves=num_waves,
        waves_per_eu=waves_per_eu,
    )
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


# --------------------------------------------------------------------------- #
# Untuned fallback config
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("kernel", ["dvdk", "dq"])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
@pytest.mark.parametrize("hidden_dim", [64, 96, 128, 192, 256])
def test_bwd_default_config_validates(arch, kernel, head_dim, hidden_dim):
    """The untuned fallback must never hand back a config the kernel rejects.

    hidden_dim 96/192 do not divide the dO/V DMA pass at the default wave count, and
    on gfx950 the wider dwordx4 pass rules out more of the space still -- both are
    only reachable through the fallback's candidate search. Arch-parametrized so the
    gfx950 geometry is covered from a gfx942 host.
    """
    config = hstu_kernels._get_bwd_default_config(
        kernel, head_dim=head_dim, hidden_dim=hidden_dim, arch=arch
    )
    validate_hstu_attention_bwd(
        1,
        head_dim,
        hidden_dim,
        1,
        True,
        0,
        0,
        False,
        1.0,
        "bf16",
        512,
        arch=arch,
        **config,
    )


def test_bwd_validation_honors_arch_argument():
    """Tile checks must follow the requested arch, not the local device.

    gfx950's dwordx4 DMA gives a 4x wider pass, so hidden_dim=16 at the default tile
    is fine on gfx942 and invalid on gfx950. If the arch argument were dropped, this
    would agree with whichever device happens to be running the suite.
    """
    config = {"block_m": 128, "block_n": 32, "num_waves": 4, "waves_per_eu": 0}
    args = (1, 128, 16, 1, True, 0, 0, False, 1.0, "bf16", 512)
    validate_hstu_attention_bwd(*args, arch="gfx942", **config)
    with pytest.raises(ValueError, match="dO DMA tile"):
        validate_hstu_attention_bwd(*args, arch="gfx950", **config)


# --------------------------------------------------------------------------- #
# Backward tuned-CSV loading
# --------------------------------------------------------------------------- #


def _bwd_row(**overrides) -> dict:
    row = {
        "arch": hstu_kernels._GPU_ARCH,
        "dtype": "bf16",
        "num_heads": 4,
        "head_dim": 128,
        "hidden_dim": 128,
        "batch": 256,
        "max_seq_len": 1024,
        "has_window": "False",
        "has_contextual": "False",
        "has_targets": "False",
        "kernel": "dvdk",
        "block_m": 128,
        "block_n": 64,
        "num_waves": 4,
        "waves_per_eu": 2,
        "duration_us": 1.0,
    }
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
    assert config == {"block_m": 128, "block_n": 64, "num_waves": 4, "waves_per_eu": 2}


def test_bwd_tuned_csv_missing_file_returns_empty(tmp_path):
    assert hstu_kernels._bwd_tuned_config_map(str(tmp_path / "nope.csv")) == {}


def test_bwd_tuned_csv_best_duration_wins(tmp_path):
    path = _write_bwd_csv(
        tmp_path / "tuned_bwd.csv",
        [
            _bwd_row(duration_us=5.0, block_m=64),
            _bwd_row(duration_us=1.0, block_m=256),
        ],
    )

    config_map = hstu_kernels._bwd_tuned_config_map(path)

    (config,) = config_map.values()
    assert config["block_m"] == 256


def test_bwd_tuned_csv_per_kernel_configs(tmp_path):
    """The fused dV+dK kernel and dQ each resolve independent tuned configs for the
    same problem."""
    path = _write_bwd_csv(
        tmp_path / "tuned_bwd.csv",
        [
            _bwd_row(
                kernel="dvdk", block_m=96, block_n=16, num_waves=2, waves_per_eu=0
            ),
            _bwd_row(kernel="dq", block_m=128, block_n=32, num_waves=4, waves_per_eu=2),
        ],
    )

    config_map = hstu_kernels._bwd_tuned_config_map(path)

    assert len(config_map) == 2
    # Keys are (problem_key, kernel); pull each kernel's config.
    by_kernel = {kern: cfg for (_, kern), cfg in config_map.items()}
    assert by_kernel["dvdk"] == {
        "block_m": 96,
        "block_n": 16,
        "num_waves": 2,
        "waves_per_eu": 0,
    }
    assert by_kernel["dq"] == {
        "block_m": 128,
        "block_n": 32,
        "num_waves": 4,
        "waves_per_eu": 2,
    }


# --------------------------------------------------------------------------- #
# End-to-end autograd integration
#
# Scope is the autograd wiring, not a second numerics sweep: the gradient values
# themselves are covered by the direct flydsl_hstu_attention_bwd tests above,
# which pin failures to the kernel without involving the FlyDSL forward. What
# only this path can catch is FlydslHstuAttention dropping or mangling state on
# the ctx round-trip, so each case below turns on one masking argument that
# forward has to hand back to backward.
# --------------------------------------------------------------------------- #


@requires_cuda
@pytest.mark.parametrize(
    "max_attn_len,contextual_seq_len,target_size",
    [
        (0, 0, 20),  # num_targets forwarded
        (64, 0, 0),  # max_attn_len forwarded
        (0, 64, 0),  # contextual_seq_len forwarded
    ],
)
def test_flydsl_autograd_end_to_end(max_attn_len, contextual_seq_len, target_size):
    """FlydslHstuAttention.apply is drop-in differentiable: .grad after .backward()
    matches torch.autograd.grad on the torch reference."""
    dtype = torch.bfloat16
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
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        True,
        num_targets,
        max_attn_len,
        contextual_seq_len,
        dout,
    )

    qd = q.detach().clone().requires_grad_(True)
    kd = k.detach().clone().requires_grad_(True)
    vd = v.detach().clone().requires_grad_(True)

    out = flydsl_hstu_attention(
        max_seq_len,
        alpha,
        qd,
        kd,
        vd,
        seq_offsets,
        True,
        num_targets,
        max_attn_len,
        contextual_seq_len,
    )
    assert out.requires_grad
    out.backward(dout)

    assert_grads_close(qd.grad, kd.grad, vd.grad, dq_ref, dk_ref, dv_ref)


@requires_cuda
def test_flydsl_autograd_rejects_non_causal():
    """causal=False must fail in forward, not later inside .backward()."""
    q, k, v, seq_offsets, _ = generate_hstu_attn_inputs(
        batch_size=8,
        max_seq_len=256,
        sparsity=0.5,
        heads=4,
        attn_dim=64,
        hidden_dim=64,
        target_size=0,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )

    with pytest.raises(ValueError, match="causal"):
        flydsl_hstu_attention(
            256, 1.0 / 64 * 10000, q, k, v, seq_offsets, False, None, 0, 0
        )


# --------------------------------------------------------------------------- #
# Perf + correctness sweep (script entry point)
#
# Run via ``python op_tests/test_flydsl_hstu_attention_bwd.py ...`` -- times the
# flydsl bwd wrapper (which dispatches internally, covering every supported gfx9
# arch) and checks (dq, dk, dv) against the fp32 autograd oracle above. Excluded
# from pytest collection (``__test__ = False``) because the sweep entry is driven
# by ``main()``/argparse rather than fixtures.
# --------------------------------------------------------------------------- #

# The flydsl bwd wrapper dispatches internally; validated on the gfx9 family.
SUPPORTED_GFX = ["gfx942", "gfx950"]

# Realistic jagged length distribution for perf (uniform[1,N]*SPARSITY). The pytest
# suite's generator applies aggressive length sampling that collapses sequences to
# ~2 tokens (great for mask edge-cases, useless for a perf table), so we build our
# own realistic lengths here — matching the perf benches.
SPARSITY = 0.5

# The oracle pads to dense [B, H, N, N] fp32 scores, so its footprint grows with N^2
# and blows past device memory long before the kernel does (n=16384 at b=120/h=4 asks
# for 480 GiB). Autograd retains roughly this many of those tensors across the einsum
# / silu / mask chain, plus one [B, N, N] mask.
_ORACLE_DENSE_COPIES = 4
# Leave room for the kernel's own inputs and workspace alongside the oracle.
_ORACLE_MEM_HEADROOM = 0.7
# Set from --no-ref: report timings only, leaving the err column empty.
_SKIP_REFERENCE = False


def _oracle_bytes(b, h, n):
    """Estimated peak device bytes for the dense fp32 autograd oracle."""
    fp32 = 4
    return _ORACLE_DENSE_COPIES * b * h * n * n * fp32 + b * n * n * fp32


def _oracle_fits(b, h, n):
    if _SKIP_REFERENCE:
        return False
    free, _total = torch.cuda.mem_get_info()
    return _oracle_bytes(b, h, n) <= free * _ORACLE_MEM_HEADROOM


# label -> (max_attn_len, contextual_seq_len, target_size)
MASKS = {
    "causal": (0, 0, 0),
    "hstu": (0, 0, 20),
}


def _build_inputs(b, h, d, n, target_size, dtype, seed=1001):
    """(q, k, v, seq_offsets, num_targets) with realistic jagged lengths; attn_dim ==
    hidden_dim == d (the torch oracle requires them equal)."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    lengths = torch.randint(1, n + 1, (b,), device=dev, generator=g)
    lengths = (lengths.float() * SPARSITY).clamp(min=1.0).to(torch.int64)
    seq_offsets = torch.zeros(b + 1, dtype=torch.int64, device=dev)
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    total = int(seq_offsets[-1].item())
    x = torch.empty((total, h, 2 * d + d), dtype=dtype, device=dev).uniform_(
        -0.01, 0.01
    )
    q, k, v = torch.split(x, [d, d, d], dim=-1)
    num_targets = None
    if target_size > 0:
        num_targets = torch.randint(1, target_size + 1, (b,), device=dev, generator=g)
        num_targets = torch.minimum(num_targets, lengths).to(torch.int32)
    return q.contiguous(), k.contiguous(), v.contiguous(), seq_offsets, num_targets


@benchmark()
def test_flydsl_hstu_bwd(b, h, n, d, dtype, mask):
    """One (batch, heads, seq_len, head_dim, dtype, mask) case: time the flydsl bwd
    and check (dq, dk, dv) against the torch autograd oracle in fp32."""
    max_attn_len, contextual_seq_len, target_size = MASKS[mask]
    alpha = 1.0 / d * 10000  # matches the pytest suite's alpha/tolerance regime

    q, k, v, seq_offsets, num_targets = _build_inputs(b, h, d, n, target_size, dtype)
    dout = torch.randn_like(v)

    (dq, dk, dv), us = run_perftest(
        flydsl_hstu_attention_bwd,
        n,
        alpha,
        q,
        k,
        v,
        dout,
        seq_offsets,
        True,
        num_targets,
        max_attn_len,
        contextual_seq_len,
    )

    msg = f"{mask} B{b}H{h}N{n}d{d}"

    # Tolerances scale with the oracle's peak, matching the pytest suite: these
    # gradients run ~1e-3, so a fixed atol would exceed the data and pass on
    # all-zero output. dQ/dK carry the extra dA/dS reductions, so they accumulate
    # more bf16/fast-math error than dV.
    def _check(got, ref, tol, name):
        ref_f32 = ref.to(dtypes.fp32)
        scale = ref_f32.abs().max().item()
        return checkAllclose(
            got.to(dtypes.fp32),
            ref_f32,
            rtol=tol,
            atol=tol * scale,
            msg=f"{msg}: {name}",
        )

    # Reference only (fp32 autograd oracle): compared, never timed / tabled. Skipped
    # when it would not fit, the pytest suite owns correctness at sizes where the oracle
    # is affordable.
    if _oracle_fits(b, h, n):
        dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
            n,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            True,
            num_targets,
            max_attn_len,
            contextual_seq_len,
            dout,
        )
        err = max(
            _check(dv, dv_ref, TOL_DV, "dv"),
            _check(dk, dk_ref, TOL_DQK, "dk"),
            _check(dq, dq_ref, TOL_DQK, "dq"),
        )
    else:
        err = float("nan")
        aiter.logger.warning(
            "%s: skipping fp32 oracle (needs ~%.0f GiB); perf only",
            msg,
            _oracle_bytes(b, h, n) / 1024**3,
        )

    # Roofline. Causal pairs per sequence = L*(L+1)/2; bwd = 3*f1 + 2*f2 with
    # f1 = 2*attn_dim (S recompute / dK / dQ share the attn-dim contraction),
    # f2 = 2*hidden_dim (dA=dO*V^T and dV share the hidden-dim contraction); ad=hd=d.
    lengths = (seq_offsets[1:] - seq_offsets[:-1]).to(torch.float64)
    pairs = float((lengths * (lengths + 1) / 2).sum().item()) * h
    total = int(seq_offsets[-1].item())
    flops = (3 * 2 * d + 2 * 2 * d) * pairs
    # DRAM traffic: read q,k (attn_dim) + v,dO (hidden_dim); write dq,dk (attn_dim) +
    # dv (hidden_dim) -> (4*ad + 3*hd) elems/token = 7*d when ad=hd=d.
    nbytes = total * h * (4 * d + 3 * d) * q.element_size()

    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": err,
    }


# Script-only perf entry point: driven by main()/argparse, not pytest fixtures.
test_flydsl_hstu_bwd.__test__ = False


def main():
    torch.set_default_device("cuda")

    # Whole-op arch gate lives here (an in-fn return from @benchmark would still
    # emit an args-only NaN row). Positive allow-list so an unknown card skips.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl hstu attention bwd unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="""Data types to sweep (one table each).
        e.g.: -d bf16 fp16""",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[120],
        help="""Batch sizes.
        e.g.: -b 120 1024""",
    )
    parser.add_argument(
        "-s",
        "--seqlen",
        type=int,
        nargs="*",
        default=[512, 1024, 2048],
        help="""Max sequence lengths.
        e.g.: -s 512 1024 2048""",
    )
    parser.add_argument(
        "-H",
        "--heads",
        type=int,
        nargs="*",
        default=[4, 8],
        help="""Head counts.
        e.g.: -H 4 8""",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        nargs="*",
        default=[64, 128],
        help="""Head dim (attn_dim == hidden_dim; the torch oracle requires them equal).
        e.g.: --head-dim 64 128""",
    )
    parser.add_argument(
        "--mask",
        type=str,
        nargs="*",
        default=list(MASKS),
        choices=list(MASKS),
        help="""Mask presets to sweep.
        e.g.: --mask causal hstu""",
    )
    parser.add_argument(
        "--no-ref",
        action="store_true",
        help="""Skip the fp32 autograd oracle everywhere (perf only, err column empty).
        It is skipped automatically for cases whose dense [B, H, N, N] scores would
        not fit in device memory.""",
    )
    args = parser.parse_args()

    global _SKIP_REFERENCE
    _SKIP_REFERENCE = args.no_ref

    for dtype in args.dtype:  # one table per dtype
        df = []
        for mask, b, h, n, d in itertools.product(
            args.mask, args.batch, args.heads, args.seqlen, args.head_dim
        ):
            df.append(test_flydsl_hstu_bwd(b, h, n, d, dtype, mask))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "flydsl hstu bwd summary [%s] (markdown):\n%s",
            dtype,
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
