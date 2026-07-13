# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""op_test for jagged_dense_bmm_broadcast_add (FlyDSL grouped BF16 GEMM + bias).

For each group ``b`` over its packed row slice ``[seq_offsets[b], seq_offsets[b+1])``::

    Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]

Shapes map to the model's ``(B, D, Kout, Mi)``:
  B    = n_groups                          (per-group dense weight panels)
  D    = reduction K  (jagged/dense inner) -> kernel ``K``
  Kout = output N     (dense/out columns)  -> kernel ``N``
  Mi   = max_seq_len  (per-group rows in uniform; envelope in skew)
  L    = seq_offsets[-1] = total packed rows (swept implicitly by B*Mi)

``regime`` is a real deployment axis (uniform per-group length vs skewed
variable length), not a behavior toggle. Candidates: the FlyDSL dispatch
wrapper (the production path) and the generative-recommenders Triton baseline
(only when importable).
"""

from __future__ import annotations

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.jit.utils.chip_info import get_gfx

import flydsl.compiler as flyc
from aiter.ops.flydsl.jagged_dense_bmm_dispatch import jagged_dense_bmm_dispatched
from aiter.ops.flydsl.kernels.jagged_dense_bmm_gen import BLOCK_M as _BLOCK_M

torch.set_default_device("cuda")

# gfx942 (MI300X) is the only arch the kernel is tuned/validated for
# (jagged_dense_bmm_dispatch.json ships a single gfx942 section).
SUPPORTED_GFX = ["gfx942"]

SEED = 1234  # deterministic skew seq_offsets

# Optional Triton baseline from facebookresearch/generative-recommenders.
try:
    from generative_recommenders.ops.triton.triton_jagged import (
        triton_jagged_dense_bmm_add_fwd,
    )

    _HAS_TRITON = True
except Exception as _exc:  # pragma: no cover - depends on external dep on PYTHONPATH
    _HAS_TRITON = False
    _TRITON_ERR = _exc


def _make_seq_offsets(B, Mi, regime, seed=SEED):
    """int32 (B+1,) prefix sum. uniform: every group == Mi; skew: ~20% empty,
    one full-envelope + one near-full group, the rest heavily right-skewed."""
    if regime == "uniform":
        return torch.arange(0, (B + 1) * Mi, Mi, dtype=torch.int32, device="cuda")
    # Build on CPU (the seeded generator is a CPU generator) then move to cuda.
    g = torch.Generator().manual_seed(seed)
    t = (Mi * (torch.rand(B, generator=g, device="cpu") ** 4)).floor().to(torch.int64)
    t[: max(1, B // 5)] = 0  # ~20% empty groups
    t[-1] = Mi  # full-envelope group
    if B > 1:
        t[-2] = int(0.9 * Mi)  # near-full group
    off = torch.zeros(B + 1, dtype=torch.int32, device="cpu")
    off[1:] = torch.cumsum(t, 0).to(torch.int32)
    return off.cuda()


def run_torch(jagged, dense, bias, seq_offsets, N, dtype=dtypes.bf16):
    """Reference only: per-group fp32 matmul + bias broadcast, cast back.
    Not timed, not in the table."""
    L = jagged.shape[0]
    out = torch.zeros((L, N), dtype=dtype, device="cuda")
    so = seq_offsets.cpu().tolist()
    for b in range(dense.shape[0]):
        s, e = so[b], so[b + 1]
        if e > s:
            out[s:e] = (
                jagged[s:e].to(dtypes.fp32) @ dense[b].to(dtypes.fp32)
                + bias[b].to(dtypes.fp32)[None, :]
            ).to(dtype)
    return out


@benchmark()  # (B, D, Kout, Mi, regime, dtype) become the table's left columns
def test_jdbba(B, D, Kout, Mi, regime, dtype):
    N, K = Kout, D
    seq_offsets = _make_seq_offsets(B, Mi, regime)
    L = int(seq_offsets[-1].item())
    uniform = regime == "uniform"

    # Inputs/outputs exactly as the dispatch wrapper is invoked in deployment:
    #   jagged  (L, K)     bf16  packed rows
    #   dense   (B, K, N)  bf16 -> host pre-transposed tall (B*N, K)
    #   bias    (B, N)     bf16 -> flat (B*N,)
    #   out     (L, N)     bf16  preallocated with a BLOCK_M tail pad (kernel writes
    #                            full M-tiles; the pad absorbs the partial tail tile)
    torch.manual_seed(0)
    jagged = torch.randn(max(L, 1), K, dtype=dtype, device="cuda")
    dense = torch.randn(B, K, N, dtype=dtype, device="cuda")
    bias = torch.randn(B, N, dtype=dtype, device="cuda")

    dense_tall = dense.transpose(1, 2).reshape(B * N, K).contiguous()
    bias_flat = bias.reshape(B * N).contiguous()
    out = torch.zeros(L + _BLOCK_M, N, dtype=dtype, device="cuda")
    tA = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
    st = torch.cuda.current_stream()

    ref = run_torch(jagged, dense, bias, seq_offsets, N, dtype)

    def _flydsl():
        jagged_dense_bmm_dispatched(
            tC, tA, dense_tall, bias_flat, seq_offsets, B, Mi,
            stream=st, uniform_seqlen=uniform,
        )
        return out[:L]

    candidates = {"flydsl": _flydsl}
    # Triton baseline only when the external package is on PYTHONPATH; leave its
    # cells nan otherwise (never a wrong-but-fast row).
    if _HAS_TRITON:
        so64 = seq_offsets.to(torch.int64)
        candidates["triton"] = lambda: triton_jagged_dense_bmm_add_fwd(
            Mi, so64, jagged, dense, bias
        )[0][:L]

    # grouped GEMM over L packed rows: FLOPs = 2*L*K*N (mul-add);
    # bytes = jagged + dense weights + bias + out (bf16 element_size handles dtype).
    flops = 2 * L * K * N
    nbytes = (L * K + B * K * N + B * N + L * N) * jagged.element_size()

    ret = {"gfx": get_gfx(), "L": L}
    for name, fn in candidates.items():
        got, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            got.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: jagged_dense_bmm ({regime})",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "jagged_dense_bmm unsupported on %s; skipping", get_gfx()
        )
        return
    if not _HAS_TRITON:
        aiter.logger.warning(
            "generative-recommenders Triton baseline unavailable (%s); "
            "running flydsl-only", _TRITON_ERR,
        )

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
        help="""Data type (op is bf16-only).
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            # headline deployment shapes (Mi=7680): (B, D, Kout, Mi)
            (120, 256, 256, 7680),
            (120, 512, 512, 7680),
            (1024, 256, 256, 7680),
            (1024, 512, 512, 7680),
            # production shape (JSON winner B64D512K1024N8192)
            (64, 512, 1024, 8192),
        ],
        help="""(B, D, Kout, Mi) tuples: B groups, D reduction-K, Kout output-N,
        Mi max_seq_len.
        e.g.: -s 64,512,1024,8192""",
    )
    parser.add_argument(
        "-r",
        "--regime",
        type=str,
        choices=["uniform", "skew"],
        nargs="*",
        default=["uniform", "skew"],
        help="""Sequence-length distribution:
        uniform = every group has Mi rows; skew = variable per-group lengths.
        e.g.: -r uniform""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for regime, (B, D, Kout, Mi) in itertools.product(args.regime, args.shapes):
            df.append(test_jdbba(B, D, Kout, Mi, regime, dtype))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "jagged_dense_bmm summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
