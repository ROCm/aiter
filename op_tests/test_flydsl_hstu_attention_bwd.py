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
import time
import weakref

import pandas as pd
import pytest
import torch

import aiter
import aiter.ops.flydsl.hstu_attention as hstu_kernels
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.hstu_attention import (
    _validate_bwd_inputs,
    flydsl_hstu_attention,
    flydsl_hstu_attention_bwd,
)
from aiter.ops.flydsl.kernels.hstu.hstu_attention_bwd import validate_hstu_attention_bwd
from aiter.ops.flydsl.kernels.hstu.hstu_attention_bwd_fallback_dvdk import (
    validate_hstu_attention_bwd_fallback,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

# Reuse the forward test's self-contained input generator and torch-reference
# loader. Mirror the forward's sys.path fallback (#5394) so this file also works
# when run directly as a script (`python op_tests/test_*.py`), where `op_tests`
# is not importable unless the repo root is on sys.path.
try:
    from op_tests.test_flydsl_hstu_attention import (
        _load_torch_hstu_reference,
        generate_hstu_attn_inputs,
        record_launch_allocation_streams,
    )
except ModuleNotFoundError as exc:
    missing = exc.name or ""
    if not (missing == "op_tests" or missing.startswith(("op_tests.", "triton_tests"))):
        raise

    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from op_tests.test_flydsl_hstu_attention import (
        _load_torch_hstu_reference,
        generate_hstu_attn_inputs,
        record_launch_allocation_streams,
    )

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA device"
)


@pytest.fixture(autouse=True)
def _validate_metadata_values(monkeypatch):
    """Every case also checks the metadata values, which production skips."""
    monkeypatch.setenv("AITER_HSTU_VALIDATE", "1")


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
    torch_hstu_attention = _load_torch_hstu_reference()

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


# --------------------------------------------------------------------------- #
# Shared input construction
#
# Used by the fused correctness cases, the perf sweep below, and
# op_tests/op_benchmarks/flydsl/bench_hstu_bwd.py. Unlike the forward suite's
# generate_hstu_attn_inputs, it draws lengths AND data from one seeded
# generator, so inputs are reproducible across processes, and it can give
# every sequence the same target count.
# --------------------------------------------------------------------------- #

# Realistic jagged length distribution (uniform[1,N]*SPARSITY).
SPARSITY = 0.5

# label -> (max_attn_len, contextual_seq_len, target_size)
MASKS = {
    "causal": (0, 0, 0),
    "hstu": (0, 0, 20),
}


def build_bwd_inputs(
    b, h, d, n, target_size, dtype, seed=1001, sparsity=SPARSITY, fixed_targets=False
):
    """(q, k, v, seq_offsets, num_targets) with realistic jagged lengths; attn_dim ==
    hidden_dim == d (the torch oracle requires them equal).

    `sparsity` scales the sampled lengths (`length = uniform[1, n] * sparsity`), so
    HIGHER means longer sequences and quadratically more work, not less.
    `fixed_targets` gives every sequence exactly `target_size` candidates instead of
    sampling; both exist for the ticket protocol the bench reproduces.
    """
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    lengths = torch.randint(1, n + 1, (b,), device=dev, generator=g)
    lengths = (lengths.float() * sparsity).clamp(min=1.0).to(torch.int64)
    seq_offsets = torch.zeros(b + 1, dtype=torch.int64, device=dev)
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    total = int(seq_offsets[-1].item())
    # `generator=g` matters: without it `uniform_` draws from the global RNG, so
    # `seed` pins the sequence LENGTHS but not the data, and two processes get
    # different q/k/v. That silently voids any cross-process bitwise A/B.
    x = torch.empty((total, h, 2 * d + d), dtype=dtype, device=dev).uniform_(
        -0.01, 0.01, generator=g
    )
    q, k, v = torch.split(x, [d, d, d], dim=-1)
    num_targets = None
    if target_size > 0:
        if fixed_targets:
            num_targets = torch.full((b,), target_size, device=dev)
        else:
            num_targets = torch.randint(
                1, target_size + 1, (b,), device=dev, generator=g
            )
        num_targets = torch.minimum(num_targets, lengths).to(torch.int32)
    return q.contiguous(), k.contiguous(), v.contiguous(), seq_offsets, num_targets


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
        # contextual + window: the window is measured in effective IDs, where the
        # whole prefix is ID 0. Query tiles starting inside (w, w + c - 1] still
        # reach prefix keys that a raw-position bound would skip.
        (20, 50, 0),
        (20, 50, 20),
        (256, 100, 20),
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
    seq_offsets = (torch.arange(batch + 1, device=device) * tokens) // batch
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


@requires_cuda
@pytest.mark.parametrize(
    "make_offsets,make_targets,match",
    [
        (lambda so: so[None, :], None, "seq_offsets must be rank 1"),
        (lambda so: so.float(), None, "seq_offsets must be int32 or int64"),
        (lambda so: so, lambda nt: nt[:, None], "num_targets must be rank 1"),
        (lambda so: so, lambda nt: nt.to(torch.int16), "num_targets must be int32"),
    ],
)
def test_validate_bwd_inputs_rejects_bad_metadata_layout(
    monkeypatch, make_offsets, make_targets, match
):
    """Rank and dtype are checked on every call, AITER_HSTU_VALIDATE or not."""
    monkeypatch.delenv("AITER_HSTU_VALIDATE")
    q, k, v, dout, seq_offsets = _qkv_do()
    num_targets = torch.zeros(2, dtype=torch.int32, device=q.device)
    if make_targets is not None:
        num_targets = make_targets(num_targets)
    with pytest.raises(ValueError, match=match):
        _validate_bwd_inputs(
            q, k, v, dout, make_offsets(seq_offsets), num_targets, max_seq_len=8
        )


@requires_cuda
@pytest.mark.parametrize(
    "offsets,targets,match",
    [
        ([1, 4, 8], None, r"seq_offsets\[0\] must be 0"),
        ([0, 4, 6], None, r"seq_offsets\[-1\] must equal the token count \(8\)"),
        ([0, 6, 4, 8], None, "non-decreasing"),
        ([0, 0, 8], None, r"must not exceed max_seq_len \(6\)"),
        ([0, 4, 8], [1, -1], "num_targets must be non-negative"),
        ([0, 4, 8], [5, 0], "must not exceed each sequence's length"),
    ],
)
def test_validate_bwd_inputs_checks_metadata_values(offsets, targets, match):
    """With AITER_HSTU_VALIDATE=1, invalid offsets and targets are refused."""
    q, k, v, dout, _ = _qkv_do(batch=len(offsets) - 1)
    seq_offsets = torch.tensor(offsets, dtype=torch.int64, device=q.device)
    num_targets = None
    if targets is not None:
        num_targets = torch.tensor(targets, dtype=torch.int32, device=q.device)
    with pytest.raises(ValueError, match=match):
        _validate_bwd_inputs(q, k, v, dout, seq_offsets, num_targets, max_seq_len=6)


@requires_cuda
def test_validate_bwd_inputs_compares_mixed_widths_in_int64():
    """int64 targets next to int32 offsets are not narrowed: 2**32 would wrap
    to 0 and pass."""
    q, k, v, dout, _ = _qkv_do()
    seq_offsets = torch.tensor([0, 4, 8], dtype=torch.int32, device=q.device)
    num_targets = torch.tensor([2**32, 0], dtype=torch.int64, device=q.device)
    with pytest.raises(ValueError, match="must not exceed each sequence's length"):
        _validate_bwd_inputs(q, k, v, dout, seq_offsets, num_targets, max_seq_len=8)


@requires_cuda
def test_validate_bwd_inputs_skips_values_by_default(monkeypatch):
    """Value checks synchronise with the device, so they are opt-in."""
    monkeypatch.delenv("AITER_HSTU_VALIDATE")
    q, k, v, dout, _ = _qkv_do()
    seq_offsets = torch.tensor([0, 6, 4], dtype=torch.int64, device=q.device)
    _validate_bwd_inputs(q, k, v, dout, seq_offsets, None, max_seq_len=8)


# --------------------------------------------------------------------------- #
# Dense causal correctness across problem shapes
# --------------------------------------------------------------------------- #


@requires_cuda
# False pins the fallback pair; None is the default dispatch, which runs the
# fused kernel for the bf16 / 64-dim cases on gfx942.
@pytest.mark.parametrize("fused", [False, None])
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
    batch, heads, attn_dim, hidden_dim, max_seq_len, dtype, fused
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
        max_seq_len, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0,
        fused=fused,
    )  # fmt: skip
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


# --------------------------------------------------------------------------- #
# Fused dV/dK/dQ kernel (the default; split-K dQ + reduction)
#
# The fused kernel hard-gates gfx942 / bf16 / head_dim == hidden_dim == 64 and
# the (256, 16, 8) tile, so everything here is skipped elsewhere. The cases
# above mostly sit outside that gate and so exercise the fallback pair.
#
# `fused=True` rather than the default: the default degrades to the fallback
# pair SILENTLY when the gate rejects a shape, which would make these cases
# pass while testing something else entirely.
#
# Inputs come from `build_bwd_inputs` at sparsity 0.95, so most sequences span
# several of the fused kernel's 256-row kv tiles, which is what exercises the
# split-K accumulation and the kv-chunking.
# --------------------------------------------------------------------------- #

requires_fused = pytest.mark.skipif(
    not torch.cuda.is_available()
    # The tests allocate on "cuda", i.e. the current device.
    or hstu_kernels._device_arch_and_cus(torch.cuda.current_device())
    != ("gfx942", 304),
    reason="hstu bwd fused is validated on the 304-CU gfx942 class only",
)

# B=32/H=4/N=1024/d=64 spans ~2-4 kv tiles per sequence at BLOCK_M=256, which is
# what makes split-K and kv-chunking non-trivial, while keeping the dense fp32
# oracle affordable.
_FUSED_CASE = {"b": 32, "h": 4, "n": 1024, "d": 64}


def _fused_inputs(b=32, h=4, n=1024, d=64, target_size=300):
    q, k, v, seq_offsets, num_targets = build_bwd_inputs(
        b, h, d, n, target_size, torch.bfloat16, sparsity=0.95, fixed_targets=True
    )
    g = torch.Generator(device=q.device).manual_seed(1234)
    dout = torch.randn(v.shape, generator=g, device=v.device, dtype=v.dtype)
    return q, k, v, dout, seq_offsets, num_targets


@requires_fused
@pytest.mark.parametrize("groups", [1, 2, 3])
@pytest.mark.parametrize("target_size", [0, 300])
def test_flydsl_bwd_fused_matches_oracle(target_size, groups):
    """All three fused gradients match the fp32 autograd oracle.

    `groups` is the kv-chunking depth: at G > 1 the kernel runs once per tile
    span into one reused scratch, with an fp32 carry between passes.
    """
    b, h, n, d = (_FUSED_CASE[key] for key in ("b", "h", "n", "d"))
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, num_targets = _fused_inputs(b, h, n, d, target_size)

    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        n, alpha, q, k, v, seq_offsets, True, num_targets, 0, 0, dout
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(
        n, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0,
        fused=True, fused_groups=groups,
    )  # fmt: skip
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


@requires_fused
def test_flydsl_bwd_fused_identity_perm():
    """b % NUM_GRID_GROUPS != 0 takes the identity permutation, not the balanced deal."""
    b, h, n, d = 33, 4, 1024, 64
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, num_targets = _fused_inputs(b, h, n, d)

    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        n, alpha, q, k, v, seq_offsets, True, num_targets, 0, 0, dout
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(
        n, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0, fused=True
    )
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


@requires_fused
@pytest.mark.parametrize("target_size", [0, 300])
def test_flydsl_bwd_fused_unsorted(target_size):
    """sort_by_length=False compiles the kernel without the permutation.

    The permutation only reassigns sequences to workgroups, so both orders must
    agree EXACTLY; b=32 is a multiple of NUM_GRID_GROUPS, so the sorted run
    really takes the balanced deal rather than the identity.
    """
    b, h, n, d = (_FUSED_CASE[key] for key in ("b", "h", "n", "d"))
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, num_targets = _fused_inputs(b, h, n, d, target_size)

    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        n, alpha, q, k, v, seq_offsets, True, num_targets, 0, 0, dout
    )
    args = (n, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0)
    unsorted = flydsl_hstu_attention_bwd(*args, fused=True, sort_by_length=False)
    assert_grads_close(*unsorted, dq_ref, dk_ref, dv_ref)
    for name, got, ref in zip(
        ("dq", "dk", "dv"),
        unsorted,
        flydsl_hstu_attention_bwd(*args, fused=True, sort_by_length=True),
    ):
        assert torch.equal(got, ref), f"{name} depends on sort_by_length"


@requires_fused
def test_flydsl_bwd_fused_group_invariance():
    """G=1, G=2 and G=4 must agree EXACTLY, not merely to tolerance.

    kv-chunking changes only WHEN partials are summed, never the order within a
    q row, so any difference at all means the tile-group split or the per-pass
    scratch offsets are wrong -- a class of bug that a tolerance check would
    wave through as rounding.
    """
    b, h, n, d = (_FUSED_CASE[key] for key in ("b", "h", "n", "d"))
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, num_targets = _fused_inputs(b, h, n, d)

    args = (n, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0)
    dq1, dk1, dv1 = flydsl_hstu_attention_bwd(*args, fused=True, fused_groups=1)
    for groups in (2, 4):
        dq2, dk2, dv2 = flydsl_hstu_attention_bwd(
            *args, fused=True, fused_groups=groups
        )
        assert torch.equal(dq1, dq2), (
            f"G=1 vs G={groups} differ in "
            f"{int((dq1 != dq2).sum()):,}/{dq1.numel():,} dq elems"
        )
        # dV/dK are not split across passes at all, so they are the control: a
        # difference there means a pass perturbed state it does not own.
        assert torch.equal(dk1, dk2)
        assert torch.equal(dv1, dv2)


@requires_fused
def test_flydsl_bwd_fused_rejects_unsupported_shape():
    """`fused=True` raises outside the gate; the default quietly uses dvdk + dq.

    The fallback must still be CORRECT, which is the half of this that a plain
    `pytest.raises` would miss.
    """
    b, h, n, d = 8, 2, 512, 128  # head_dim 128 is outside the fused gate
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, num_targets = _fused_inputs(b, h, n, d)

    args = (n, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0)
    with pytest.raises(ValueError, match="head_dim == hidden_dim == 64"):
        flydsl_hstu_attention_bwd(*args, fused=True)

    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        n, alpha, q, k, v, seq_offsets, True, num_targets, 0, 0, dout
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(*args, fused=None)
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


@requires_fused
@pytest.mark.parametrize("max_attn_len", [64, 256])
@pytest.mark.parametrize("target_size", [0, 300])
def test_flydsl_bwd_fused_sliding_window(max_attn_len, target_size):
    """A windowed mask sizes each scratch slot to its window; the reduction
    reads a slot row exactly when the tile serves it.

    The split-K scratch is allocated with `empty`, so a read outside a slot is
    garbage (or another tile's partial). Both halves of the predicate are
    exercised here: without targets the served tiles form a contiguous span
    below the row, and with targets the clamped rows are served by tiles
    arbitrarily far below, which is the case that a span-only bound drops.
    """
    b, h, n, d = 32, 4, 1024, 64
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, num_targets = _fused_inputs(b, h, n, d, target_size)
    args = (n, alpha, q, k, v, dout, seq_offsets, True, num_targets, max_attn_len, 0)

    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        n, alpha, q, k, v, seq_offsets, True, num_targets, max_attn_len, 0, dout
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(*args, fused=True)
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


@requires_fused
def test_flydsl_bwd_fused_rejects_contextual():
    """A contextual prefix must RAISE, not return wrong gradients.

    The prefix needs dQ from kv tiles above the causal diagonal, which the
    compact scratch does not store, so the fused path cannot be correct here.
    """
    b, h, n, d = 32, 4, 1024, 64
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, num_targets = _fused_inputs(b, h, n, d, 0)
    args = (n, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 64)

    with pytest.raises(ValueError, match="contextual prefix"):
        flydsl_hstu_attention_bwd(*args, fused=True)

    # The fallback must still be CORRECT, which a bare `raises` would miss.
    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        n, alpha, q, k, v, seq_offsets, True, num_targets, 0, 64, dout
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(*args, fused=None)
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


@requires_fused
def test_flydsl_bwd_fused_custom_stream():
    """`stream=` waits for the caller's stream and keeps its inputs alive.

    The inputs are produced on the caller's stream behind a long sleep, so a
    launch stream that did not wait for it would read them before they exist.
    The caller then drops them while the launch stream is still busy: without
    `record_stream` the allocator would hand their blocks straight to the
    caller's next allocations.
    """
    b, h, n, d = (_FUSED_CASE[key] for key in ("b", "h", "n", "d"))
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, num_targets = _fused_inputs(b, h, n, d)
    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        n, alpha, q, k, v, seq_offsets, True, num_targets, 0, 0, dout
    )
    # Compile outside the stream juggling below.
    flydsl_hstu_attention_bwd(
        n, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0, fused=True
    )
    torch.cuda.synchronize()

    caller, side = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(caller):
        torch.cuda._sleep(100_000_000)
        inputs = [t.clone() for t in (q, k, v, dout)]
        input_ptrs = {t.data_ptr() for t in inputs}
        dq, dk, dv = flydsl_hstu_attention_bwd(
            n, alpha, *inputs, seq_offsets, True, num_targets, 0, 0,
            fused=True, stream=side,
        )  # fmt: skip
        with torch.cuda.stream(side):
            torch.cuda._sleep(100_000_000)
        del inputs
        reused = [torch.full_like(t, float("nan")) for t in (q, k, v, dout)]
        assert not input_ptrs & {t.data_ptr() for t in reused}
    torch.cuda.synchronize()
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


@requires_cuda
@pytest.mark.parametrize("fused", [False, pytest.param(True, marks=requires_fused)])
@pytest.mark.parametrize("sort_by_length", [True, False])
def test_flydsl_bwd_custom_stream_temporaries(monkeypatch, fused, sort_by_length):
    """Outputs and temporaries are allocated on `stream`, not the caller's.

    Non-contiguous inputs force contiguous copies, and the call also builds
    its own `num_targets` and permutation. Allocated on the caller's stream,
    they would return to its pool when the call exits and could be handed to
    the caller's next allocation while the side stream still reads them.
    """
    b, h, n, d = (_FUSED_CASE[key] for key in ("b", "h", "n", "d"))
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, _ = _fused_inputs(b, h, n, d, target_size=0)
    kwargs = {"fused": fused, "sort_by_length": sort_by_length}
    expected = flydsl_hstu_attention_bwd(
        n, alpha, q, k, v, dout, seq_offsets, True, None, 0, 0, **kwargs
    )
    strided = [torch.cat([t, t], dim=-1)[..., :d] for t in (q, k, v, dout)]
    assert not strided[0].is_contiguous()

    launches = record_launch_allocation_streams(monkeypatch, [*strided, seq_offsets])
    side = torch.cuda.Stream()
    got = flydsl_hstu_attention_bwd(
        n, alpha, *strided, seq_offsets, True, None, 0, 0, stream=side, **kwargs
    )
    torch.cuda.synchronize()
    assert launches and all(s == {side.cuda_stream} for s in launches), launches
    for name, g, e in zip(("dq", "dk", "dv"), got, expected):
        assert torch.equal(g, e), name


@requires_cuda
@pytest.mark.parametrize("aligned", [True, False])
def test_balance_perm_cache_is_safe_across_streams(aligned):
    """A cache miss on a stalled stream is complete before another stream reads.

    The cached gather index (identity perm for unaligned batches) is built on
    whichever stream misses first. A second stream that read it while the
    first was still stalled would see whatever the block held before. The
    stalled stream's free blocks are zero-filled first, so such a read gives a
    perm full of repeats instead of, by luck, the right one; the reader copies
    the perm at once, so a premature read is captured.

    The ordering must not come from the host waiting: the cache miss returns
    while the stalled stream is still asleep.
    """
    groups = hstu_kernels.NUM_GRID_GROUPS
    batch = groups * 3 + (0 if aligned else 1)
    g = torch.Generator(device="cuda").manual_seed(3)
    lengths = torch.randint(0, 100, (batch,), device="cuda", generator=g)
    seq_offsets = torch.zeros(batch + 1, dtype=torch.int64, device="cuda")
    seq_offsets[1:] = torch.cumsum(lengths, 0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda._sleep(200_000_000)
    torch.cuda.synchronize()
    sleep_s = time.perf_counter() - start

    for _ in range(5):
        hstu_kernels._balance_gather_index.cache_clear()
        hstu_kernels._identity_perm.cache_clear()
        stalled, reader = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(stalled):
            poison = [
                torch.zeros(batch, dtype=torch.int64, device="cuda") for _ in range(32)
            ]
        torch.cuda.synchronize()
        del poison
        with torch.cuda.stream(stalled):
            torch.cuda._sleep(200_000_000)
            start = time.perf_counter()
            hstu_kernels._build_balance_perm(seq_offsets)
            assert time.perf_counter() - start < sleep_s / 2, "cache miss blocked"
        with torch.cuda.stream(reader):
            perm = hstu_kernels._build_balance_perm(seq_offsets).clone()
        torch.cuda.synchronize()
        assert sorted(perm.tolist()) == list(range(batch))


def _inputs_with_lengths(lengths, h=4, d=64, seed=7):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    lengths = torch.tensor(lengths, dtype=torch.int64, device=dev)
    seq_offsets = torch.zeros(len(lengths) + 1, dtype=torch.int64, device=dev)
    seq_offsets[1:] = torch.cumsum(lengths, 0)
    total = int(seq_offsets[-1])
    x = torch.empty((total, h, 4 * d), dtype=torch.bfloat16, device=dev)
    x.uniform_(-0.01, 0.01, generator=g)
    q, k, v, dout = (t.contiguous() for t in torch.split(x, [d] * 4, dim=-1))
    return q, k, v, dout * 100, seq_offsets


_FUSED_OR_FALLBACK = [
    False,
    pytest.param(True, marks=requires_fused),
]


@requires_cuda
@pytest.mark.parametrize("fused", _FUSED_OR_FALLBACK)
def test_flydsl_bwd_empty_sequences(fused):
    """Empty sequences -- leading, interior and trailing -- contribute nothing.

    A trailing empty sequence starts one past the last token, so any load
    anchored at its start is out of bounds.
    """
    n, d = 1024, 64
    alpha = 1.0 / d * 10000
    lengths = [0, 700, 0, 1, 300, 1024, 0, 0]
    q, k, v, dout, seq_offsets = _inputs_with_lengths(lengths, d=d)
    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        n, alpha, q, k, v, seq_offsets, True, None, 0, 0, dout
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(
        n, alpha, q, k, v, dout, seq_offsets, True, None, 0, 0, fused=fused
    )
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


@requires_cuda
@pytest.mark.parametrize("fused", _FUSED_OR_FALLBACK)
def test_flydsl_bwd_all_empty(fused):
    """A batch without a single token returns empty gradients, launching nothing."""
    q, k, v, dout, seq_offsets = _inputs_with_lengths([0, 0, 0, 0])
    dq, dk, dv = flydsl_hstu_attention_bwd(
        512, 1.0, q, k, v, dout, seq_offsets, True, None, 0, 0, fused=fused
    )
    assert dq.shape == q.shape and dk.shape == k.shape and dv.shape == v.shape


@requires_fused
@pytest.mark.parametrize("h", [17, 32])
def test_flydsl_bwd_fused_many_heads(h):
    """Past 16 heads a reduction row no longer fits one 1024-thread workgroup
    pass, so the reduction narrows its row group; 17 also makes it partial-wave.
    """
    b, n, d = 4, 1024, 64
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, num_targets = _fused_inputs(b, h, n, d)
    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        n, alpha, q, k, v, seq_offsets, True, num_targets, 0, 0, dout
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(
        n, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0, fused=True
    )
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


def test_bwd_fused_gate_rejects_too_many_heads():
    """A head count whose reduction row exceeds a workgroup is gated out."""
    with pytest.raises(ValueError, match="dq_reduce needs"):
        validate_hstu_attention_bwd(
            80, 64, 64, True, 0, 0, False, 1.0, "bf16", 1024,
            block_m=256, block_n=16, num_waves=8, waves_per_eu=2, arch="gfx942",
            num_cus=304,
        )  # fmt: skip


@pytest.mark.parametrize("waves_per_eu", [0, 1, 3, 4, 99, -1])
def test_bwd_fused_gate_requires_measured_occupancy(waves_per_eu):
    """Only the measured occupancy, waves_per_eu == 2, passes the fused gate."""
    with pytest.raises(ValueError, match="waves_per_eu == 2"):
        validate_hstu_attention_bwd(
            4, 64, 64, True, 0, 0, False, 1.0, "bf16", 1024,
            block_m=256, block_n=16, num_waves=8, waves_per_eu=waves_per_eu,
            arch="gfx942", num_cus=304,
        )  # fmt: skip


@pytest.mark.parametrize("num_cus", [80, 228, 256])
def test_bwd_fused_gate_requires_validated_device(num_cus):
    """gfx942 products other than the validated MI300X (e.g. MI308X with 80
    CUs, MI300A with 228) are gated out; MI300X's 304 passes."""
    args = (4, 64, 64, True, 0, 0, False, 1.0, "bf16", 1024)
    tiles = {"block_m": 256, "block_n": 16, "num_waves": 8, "waves_per_eu": 2}
    with pytest.raises(ValueError, match="304-CU gfx942 class only"):
        validate_hstu_attention_bwd(*args, **tiles, arch="gfx942", num_cus=num_cus)
    validate_hstu_attention_bwd(*args, **tiles, arch="gfx942", num_cus=304)


@requires_fused
def test_flydsl_bwd_fused_gate_reads_the_tensors_device(monkeypatch):
    """The gate checks the device holding q, not the process's current one.

    A node can mix gfx942 parts. Here q's device reports an 80-CU MI308X
    while the current device stays the validated one, so the default must
    fall back and `fused=True` must refuse.
    """
    real = hstu_kernels._device_arch_and_cus
    queried = []

    def fake(index):
        queried.append(index)
        return ("gfx942", 80) if index == q.device.index else real(index)

    b, h, n, d = (_FUSED_CASE[key] for key in ("b", "h", "n", "d"))
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, num_targets = _fused_inputs(b, h, n, d)
    monkeypatch.setattr(hstu_kernels, "_device_arch_and_cus", fake)
    args = (n, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0)
    with pytest.raises(ValueError, match="304-CU gfx942 class only"):
        flydsl_hstu_attention_bwd(*args, fused=True)
    assert queried and set(queried) == {q.device.index}

    calls = []
    real_runners = hstu_kernels._make_bwd_fused_runners
    monkeypatch.setattr(
        hstu_kernels,
        "_make_bwd_fused_runners",
        lambda *a, **kw: calls.append(1) or real_runners(*a, **kw),
    )
    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        n, alpha, q, k, v, seq_offsets, True, num_targets, 0, 0, dout
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(*args)
    assert not calls
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


@requires_fused
def test_bwd_fused_oom_retry_releases_failed_attempt(monkeypatch):
    """An OOM on the carry steps G up WITHOUT keeping the failed scratch alive."""
    real_empty, real_zeros = torch.empty, torch.zeros
    scratch_refs = []
    zeros_calls = []

    def fake_empty(*args, **kwargs):
        for ref in scratch_refs:
            assert ref() is None, "the failed attempt's scratch is still alive"
        out = real_empty(*args, **kwargs)
        scratch_refs.append(weakref.ref(out))
        return out

    def fake_zeros(*args, **kwargs):
        zeros_calls.append(args)
        if len(zeros_calls) == 1:
            raise torch.cuda.OutOfMemoryError("synthetic carry OOM")
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1 << 40,) * 2)
    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(torch, "zeros", fake_zeros)
    groups, spans, _dq_part, dq_carry = hstu_kernels._plan_fused_groups(
        [300, 200, 100],
        block_m=256,
        num_heads=1,
        head_dim=64,
        max_seq_len=768,
        total_tokens=600,
        groups="auto",
        device=torch.device("cuda"),
    )
    assert groups == 2 and len(spans) == 2
    assert dq_carry.shape == (600, 64)


@requires_fused
@pytest.mark.parametrize(
    "env,d,expect_fused",
    [
        (None, 64, True),  # the default inside the gate
        (None, 128, False),  # outside the gate: the fallback pair
        ("0", 64, False),  # the opt-out
    ],
)
def test_flydsl_bwd_default_dispatch(monkeypatch, env, d, expect_fused):
    """Autograd's backward runs the fused kernel wherever its gate accepts the
    shape, and the (correct) fallback pair elsewhere or when opted out."""
    calls = []
    real = hstu_kernels._make_bwd_fused_runners

    def spy(*args, **kwargs):
        calls.append(args[2].shape)
        return real(*args, **kwargs)

    monkeypatch.setattr(hstu_kernels, "_make_bwd_fused_runners", spy)
    if env is None:
        monkeypatch.delenv("AITER_HSTU_BWD_FUSED", raising=False)
    else:
        monkeypatch.setenv("AITER_HSTU_BWD_FUSED", env)

    b, h, n = 8, 2, 512
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, num_targets = _fused_inputs(b, h, n, d)
    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        n, alpha, q, k, v, seq_offsets, True, num_targets, 0, 0, dout
    )
    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    out = flydsl_hstu_attention(n, alpha, *leaves, seq_offsets, True, num_targets, 0, 0)
    out.backward(dout)
    assert bool(calls) == expect_fused, f"head_dim={d}: fused calls {calls}"
    assert_grads_close(*(t.grad for t in leaves), dq_ref, dk_ref, dv_ref)


@requires_fused
def test_flydsl_bwd_explicit_tiles_use_fallback(monkeypatch):
    """Explicit tiles are the fallback pair's knobs, so the default honours
    them there instead of failing the fused gate."""
    calls = []
    real = hstu_kernels._make_bwd_fused_runners

    def spy(*args, **kwargs):
        calls.append(args[2].shape)
        return real(*args, **kwargs)

    monkeypatch.setattr(hstu_kernels, "_make_bwd_fused_runners", spy)
    b, h, n, d = (_FUSED_CASE[key] for key in ("b", "h", "n", "d"))
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, num_targets = _fused_inputs(b, h, n, d)
    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        n, alpha, q, k, v, seq_offsets, True, num_targets, 0, 0, dout
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(
        n, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0,
        block_m=64, block_n=32, num_waves=4,
    )  # fmt: skip
    assert not calls
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


@requires_fused
def test_flydsl_bwd_unmeasured_occupancy_uses_fallback():
    """A waves_per_eu the fused kernel was not measured at runs the fallback
    by default, and is refused when the fused kernel is required."""
    b, h, n, d = (_FUSED_CASE[key] for key in ("b", "h", "n", "d"))
    alpha = 1.0 / d * 10000
    q, k, v, dout, seq_offsets, num_targets = _fused_inputs(b, h, n, d)
    args = (n, alpha, q, k, v, dout, seq_offsets, True, num_targets, 0, 0)
    with pytest.raises(ValueError, match="waves_per_eu == 2"):
        flydsl_hstu_attention_bwd(*args, waves_per_eu=99, fused=True)

    dq_ref, dk_ref, dv_ref = hstu_bwd_reference(
        n, alpha, q, k, v, seq_offsets, True, num_targets, 0, 0, dout
    )
    dq, dk, dv = flydsl_hstu_attention_bwd(*args, waves_per_eu=4)
    assert_grads_close(dq, dk, dv, dq_ref, dk_ref, dv_ref)


def test_fused_scratch_budget_ignores_split_free_blocks(monkeypatch):
    """Free pieces of partly-used segments do not count toward the scratch.

    Counting them made the planner try a G whose scratch could not be placed;
    the allocator then released the previous call's reusable scratch before
    failing, and every call paid a full reallocation.
    """
    gib = 1 << 30
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device=None: (10 * gib, 192 * gib)
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda device=None: {
            "reserved_bytes.all.current": 100 * gib,
            "allocated_bytes.all.current": 30 * gib,
            "inactive_split_bytes.all.current": 50 * gib,
        },
    )
    budget = hstu_kernels._fused_scratch_budget(torch.device("cuda"))
    # 10 free + (100 - 30 - 50) whole cached segments.
    assert budget == pytest.approx(30 * gib * hstu_kernels._BWD_FUSED_MEM_BUDGET)


def test_bwd_tuned_csv_accepts_fused_discriminator(tmp_path):
    """A `fused` row resolves like any other kernel's, independently of dvdk/dq."""
    path = _write_bwd_csv(
        tmp_path / "tuned_bwd.csv",
        [
            _bwd_row(
                kernel="fused",
                head_dim=64,
                hidden_dim=64,
                block_m=256,
                block_n=16,
                num_waves=8,
                waves_per_eu=2,
            )
        ],
    )

    config_map = hstu_kernels._bwd_tuned_config_map(path)

    ((_problem_key, kernel),) = config_map.keys()
    assert kernel == "fused"
    assert config_map[(_problem_key, kernel)] == hstu_kernels._BWD_FUSED_TILES


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
    validate_hstu_attention_bwd_fallback(
        1,
        head_dim,
        hidden_dim,
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
    args = (1, 128, 16, True, 0, 0, False, 1.0, "bf16", 512)
    validate_hstu_attention_bwd_fallback(*args, arch="gfx942", **config)
    with pytest.raises(ValueError, match="dO DMA tile"):
        validate_hstu_attention_bwd_fallback(*args, arch="gfx950", **config)


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
        (20, 50, 0),  # contextual prefix and window together
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


@benchmark()
def test_flydsl_hstu_bwd(b, h, n, d, dtype, mask):
    """One (batch, heads, seq_len, head_dim, dtype, mask) case: time the flydsl bwd
    and check (dq, dk, dv) against the torch autograd oracle in fp32."""
    max_attn_len, contextual_seq_len, target_size = MASKS[mask]
    alpha = 1.0 / d * 10000  # matches the pytest suite's alpha/tolerance regime

    q, k, v, seq_offsets, num_targets = build_bwd_inputs(b, h, d, n, target_size, dtype)
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
