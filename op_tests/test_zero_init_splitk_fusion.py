# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Test the SplitK fused zero-init optimization.

A SplitK (KBatch > 1) blockscale GEMM accumulates its partial products into the
output Y with atomic_add, so Y must be zero before launch. Normally the GEMM
zeroes Y itself (an extra `Y.zero_()` launch). The fusion lets the upstream
producer that already streams over memory (a quant / norm+quant op feeding the
GEMM) pre-zero Y as a side effect, so the GEMM can be invoked with
`y_is_zeroed=True` and skip its own zero-init.

This file tests *only* the zeroing feature; each producer's numerical
correctness is covered by its own op_test. Two concerns:

  test_producer_zero_init_matches_explicit
      The producer-side half, parametrized over every producer wired with
      `gemm_out_zero_init` (HIP per_group_quant_hip / fused_qk_rmsnorm_group_quant
      / gated_rmsnorm_fp8_group_quant and Triton fused_rms_fp8_group_quant /
      fused_rms_gated_fp8_group_quant / act_mul_and_fp8_group_quant). Running a
      producer with a dirty buffer must leave it byte-for-byte identical to an
      explicit `torch.zeros`, and the producer's own output must be unchanged by
      the side effect (the zero-init is pure).

  test_splitk_gemm_fusion
      The GEMM-side half. This is producer-independent (one GEMM kernel honoring
      `y_is_zeroed`), so it runs once over a shape sweep using per_group_quant_hip
      as the representative pre-zeroing producer, comparing three dirty buffers:
        Y_ref   : y_is_zeroed=False                      -> GEMM zeroes Y itself.
        Y_fused : producer pre-zeros + y_is_zeroed=True   -> GEMM skips zero-init.
        Y_dirty : y_is_zeroed=True, nobody zeros Y        -> dirty buffer leaks.
      Y_fused must match Y_ref (only bf16 atomic-add reordering noise), while
      Y_dirty must diverge sharply -- the negative control proving that passing
      y_is_zeroed=True without a real zero-init genuinely fails.
"""
import pytest
import torch

import aiter  # noqa: F401  (registers the blockscale GEMM / quant ops)
from aiter import dtypes
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant
from aiter.ops.gated_rmsnorm_fp8_group_quant import gated_rmsnorm_fp8_group_quant
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle_cktile
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.activation import act_mul_and_fp8_group_quant
from aiter.ops.triton.quant.fused_fp8_quant import (
    fused_rms_fp8_group_quant,
    fused_rms_gated_fp8_group_quant,
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm/CUDA GPU"
)

GROUP_SIZE = 128
SPLITK = 2  # KBatch > 1 so the atomic-add zero-init path is actually exercised.
DIRTY_FILL = 100.0  # >> output magnitude, so a skipped zero-init is unmistakable.

# Producer-side check: each producer consumes a (ROWS, FEAT_K) activation and is
# asked to pre-zero a separate (ROWS, BUF_N) buffer. BUF_N is deliberately unequal
# to the producer's own output width so the test confirms the grid-strided
# zero-init covers the whole buffer regardless of the producer's launch grid.
# ROWS*BUF_N*2 bytes must be a multiple of 16 (HIP writes uint4, Triton int32).
ROWS = 8
FEAT_K = 4096
BUF_N = 2048

# GEMM-side check: (M, N, K) decode-shaped cases for the end-to-end fusion.
SHAPES = [
    (8, 5120, 2048),
    (8, 2048, 4096),
    (8, 1024, 2048),
]


def _dirty(rows, cols):
    return torch.full((rows, cols), DIRTY_FILL, dtype=torch.bfloat16, device="cuda")


# --------------------------------------------------------------------------- #
# Producer adapters: run(zero_init) -> tuple of the producer's primary outputs.
# Each builds deterministic inputs (so two calls are bit-identical) and, when
# `zero_init` is given, asks the producer to additionally zero that buffer.
# --------------------------------------------------------------------------- #
def _run_per_group_quant_hip(zero_init):
    torch.manual_seed(0)
    x = torch.randn(ROWS, FEAT_K, device="cuda", dtype=torch.bfloat16) * 0.1
    y, scale = per_group_quant_hip(
        x,
        quant_dtype=dtypes.fp8,
        group_size=GROUP_SIZE,
        transpose_scale=True,
        gemm_out_zero_init=zero_init,
    )
    return (y, scale)


def _run_fused_qk_rmsnorm_group_quant(zero_init):
    torch.manual_seed(0)
    q = torch.randn(ROWS, FEAT_K, device="cuda", dtype=torch.bfloat16) / 10
    q_weight = torch.randn(FEAT_K, device="cuda", dtype=torch.bfloat16)
    q_out = torch.empty(ROWS, FEAT_K, dtype=dtypes.fp8, device="cuda")
    n_scale = FEAT_K // GROUP_SIZE
    # transpose_scale=True keeps the public [ROWS, n_scale] shape with column-major storage.
    q_scale = torch.empty((n_scale, ROWS), dtype=torch.float32, device="cuda").view(
        ROWS, n_scale
    )
    fused_qk_rmsnorm_group_quant(
        q_out_quantized=q_out,
        q_out_scale=q_scale,
        q=q,
        q_weight=q_weight,
        q_epsilon=1e-6,
        group_size=GROUP_SIZE,
        transpose_scale=True,
        gemm_out_zero_init=zero_init,
    )
    return (q_out, q_scale)


def _run_gated_rmsnorm_fp8_group_quant(zero_init):
    torch.manual_seed(0)
    num_heads = FEAT_K // 128  # head_dim is fixed at 128 for this kernel.
    x = torch.randn(ROWS, num_heads, 128, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(ROWS, num_heads, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(ROWS, num_heads * 128, dtype=dtypes.fp8, device="cuda")
    scales = torch.empty((ROWS, num_heads), dtype=torch.float32, device="cuda")
    gated_rmsnorm_fp8_group_quant(
        out,
        scales,
        x,
        z,
        weight,
        1e-6,
        GROUP_SIZE,
        True,
        gemm_out_zero_init=zero_init,
    )
    return (out, scales)


def _run_fused_rms_fp8_group_quant(zero_init):
    torch.manual_seed(0)
    inp1 = torch.randn(ROWS, FEAT_K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(FEAT_K, device="cuda", dtype=torch.bfloat16)
    # Returns ((out1_fp8, out1_bs), out1, out2, out_res1); keep the quant + scale.
    (out1_fp8, out1_bs), *_ = fused_rms_fp8_group_quant(
        inp1,
        weight,
        1e-6,
        group_size=GROUP_SIZE,
        dtype_quant=dtypes.fp8,
        transpose_scale=True,
        gemm_out_zero_init=zero_init,
    )
    return (out1_fp8, out1_bs)


def _run_fused_rms_gated_fp8_group_quant(zero_init):
    torch.manual_seed(0)
    x = torch.randn(ROWS, FEAT_K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(FEAT_K, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(ROWS, FEAT_K, device="cuda", dtype=torch.bfloat16)
    y_q, scales = fused_rms_gated_fp8_group_quant(
        x,
        weight,
        None,
        z,
        1e-5,
        norm_before_gate=True,
        use_ue8m0=False,
        activation="silu",
        group_size=GROUP_SIZE,
        gemm_out_zero_init=zero_init,
    )
    return (y_q, scales)


def _run_act_mul_and_fp8_group_quant(zero_init):
    torch.manual_seed(0)
    # The gated activation halves the feature dim, so the input is 2*FEAT_K wide.
    x = torch.randn(ROWS, 2 * FEAT_K, device="cuda", dtype=torch.bfloat16)
    x_q, scales = act_mul_and_fp8_group_quant(
        x,
        activation="silu",
        group_size=GROUP_SIZE,
        dtype_quant=dtypes.fp8,
        gemm_out_zero_init=zero_init,
    )
    return (x_q, scales)


PRODUCERS = [
    ("per_group_quant_hip", _run_per_group_quant_hip),
    ("fused_qk_rmsnorm_group_quant", _run_fused_qk_rmsnorm_group_quant),
    ("gated_rmsnorm_fp8_group_quant", _run_gated_rmsnorm_fp8_group_quant),
    ("fused_rms_fp8_group_quant", _run_fused_rms_fp8_group_quant),
    ("fused_rms_gated_fp8_group_quant", _run_fused_rms_gated_fp8_group_quant),
    ("act_mul_and_fp8_group_quant", _run_act_mul_and_fp8_group_quant),
]


@requires_gpu
@pytest.mark.parametrize("name, run", PRODUCERS, ids=[p[0] for p in PRODUCERS])
def test_producer_zero_init_matches_explicit(name, run):
    """Each producer's fused zero-init must equal an explicit Y.zero_()."""
    explicit = torch.zeros(ROWS, BUF_N, dtype=torch.bfloat16, device="cuda")

    Y = _dirty(ROWS, BUF_N)
    out_fused = run(Y)
    torch.cuda.synchronize()

    # The fused producer must have zeroed the dirty buffer byte-for-byte, exactly
    # as an explicit zero-init would.
    assert torch.equal(
        Y, explicit
    ), f"{name}: gemm_out_zero_init buffer not fully zeroed"

    # Running without the buffer must produce an identical primary output: the
    # zero-init is a pure side effect that never perturbs the producer's result.
    out_plain = run(None)
    torch.cuda.synchronize()
    for i, (a, b) in enumerate(zip(out_fused, out_plain)):
        assert torch.equal(
            a, b
        ), f"{name}: output[{i}] changed when gemm_out_zero_init was supplied"


@requires_gpu
@pytest.mark.parametrize("M, N, K", SHAPES)
def test_splitk_gemm_fusion(M, N, K):
    """End-to-end: a producer-zeroed buffer + y_is_zeroed=True must match the
    GEMM's own zero-init, and leaving the buffer dirty must corrupt the output."""
    torch.manual_seed(0)

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w_q = shuffle_weight(w_bf16.to(dtypes.fp8).clone(), layout=(16, 16))
    w_scale = torch.ones(
        (N + 127) // 128, (K + 127) // 128, device="cuda", dtype=torch.float32
    )

    def run(y_is_zeroed, producer_zero_init):
        # Start from a known-dirty buffer so zero-init (or its absence) is observable.
        Y = _dirty(M, N)
        # The producer pre-zeros Y as a side effect only on the fused path.
        x_q, x_scale = per_group_quant_hip(
            x,
            quant_dtype=dtypes.fp8,
            group_size=GROUP_SIZE,
            transpose_scale=True,
            gemm_out_zero_init=(Y if producer_zero_init else None),
        )
        # Op schema: (XQ, WQ, x_scale, w_scale, Out, isBpreshuffled=True, splitK=0,
        #             kernelName="", y_is_zeroed=False). kernelName keeps its
        # default (untuned heuristic kernel).
        gemm_a8w8_blockscale_bpreshuffle_cktile(
            x_q,
            w_q,
            x_scale,
            w_scale,
            Y,
            isBpreshuffled=True,
            splitK=SPLITK,
            y_is_zeroed=y_is_zeroed,
        )
        torch.cuda.synchronize()
        return Y

    # Baseline: the GEMM performs its own Y.zero_() before the splitK atomic-add.
    Y_ref = run(y_is_zeroed=False, producer_zero_init=False)
    # Fusion under test: the producer pre-zeros Y, the GEMM skips its zero-init.
    Y_fused = run(y_is_zeroed=True, producer_zero_init=True)
    # Negative control: y_is_zeroed=True but nobody zeros Y -> dirty buffer leaks.
    Y_dirty = run(y_is_zeroed=True, producer_zero_init=False)

    ymax = Y_ref.abs().max().item()
    assert ymax > 0, "reference output is all zeros; inputs are degenerate"
    d_fused = (Y_fused - Y_ref).abs().max().item() / ymax
    d_dirty = (Y_dirty - Y_ref).abs().max().item() / ymax

    # Producer-fused zero-init must match the kernel's own zero-init; the only
    # expected difference is bf16 splitK atomic-add reordering noise.
    assert (
        d_fused < 2e-2
    ), f"producer-fused y_is_zeroed diverged from kernel zero-init: rel={d_fused:.2e}"
    # y_is_zeroed=True must genuinely skip the zero-init, so leaving Y dirty has
    # to visibly corrupt the output (guards against the flag being a no-op).
    assert (
        d_dirty > 1.0
    ), f"y_is_zeroed=True did not skip zero-init (dirty buffer had no effect): rel={d_dirty:.2e}"
