# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Op-test-standard perf + correctness sweep for the FlyDSL HSTU attention backward.

Single kernel vs a torch-autograd reference (not timed): ``flydsl_hstu_attention_bwd``
recomputes S/sigma from q,k and returns (dq, dk, dv); the reference is
``torch.autograd.grad`` on the vendored ``torch_hstu_attention`` (fp32 oracle). Each
swept (b, h, n, d, dtype, mask) case records the kernel ``us`` plus TFLOPS / TB/s
rooflines and the max grad error, and emits one markdown table per dtype.

The oracle pads to dense [B, H, N, N] fp32 scores, so it is skipped (``err`` empty)
for cases where that would not fit. ``--no-ref`` skips it everywhere.

The public ``flydsl_hstu_attention_bwd`` wrapper reads the tuned CSV / heuristic and
launches the KV-owned dV/dK kernel + the Q-owned dQ kernel, so this one test covers
every supported gfx9 arch. Correctness edge cases (mask variants, tiling overrides,
input validation, autograd end-to-end) stay in the pytest suite
``op_tests/flydsl_tests/test_flydsl_hstu_attention_bwd.py``; this file is the
perf-oriented gate.

      python op_tests/test_flydsl_hstu_attention_bwd.py -b 120 -s 512 1024 --mask causal hstu
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.hstu_attention_kernels import flydsl_hstu_attention_bwd
from aiter.test_common import benchmark, checkAllclose, run_perftest

# torch-autograd oracle and tolerances from the pytest correctness suite (single
# source of truth; they live at module top level for exactly this reuse).
from op_tests.flydsl_tests.test_flydsl_hstu_attention_bwd import (
    TOL_DQK,
    TOL_DV,
    hstu_bwd_reference,
)

torch.set_default_device("cuda")

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


def main():
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
