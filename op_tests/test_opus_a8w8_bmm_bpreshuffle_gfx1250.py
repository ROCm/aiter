# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Probe / correctness gate for gfx1250 a8w8 mxscale BMM with preshuffled B.

Runs a small identity-weight case (shuffle_weight(16,16) wo_a, scale=0x7F) so
frag_a / frag_b / store_c mistakes show up as wrong numbers, not plausible noise.

Usage (on gfx1250 hardware, inside the dev container):
    rm -f aiter/jit/module_deepgemm_opus.so
    rm -rf aiter/jit/build/module_deepgemm_opus
    GPU_ARCHS=gfx1250 PYTHONPATH=. python3 op_tests/test_opus_a8w8_bmm_bpreshuffle_gfx1250.py
"""

from __future__ import annotations

import sys

import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.opus.bmm_op import _opus_bmm_a8w8_mxscale_bpreshuffle_raw
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")

GROUP_K = (
    128  # DS V4 / gfx950 flatmm: 1x128 blockscale (pack_e8m0x4 broadcast in kernel)
)
E8M0_ONE = 127  # 0x7F -> 2^0 scale factor


def _ref_bmm(
    O_fp8: torch.Tensor,
    W_fp8: torch.Tensor,
    x_scale_u8: torch.Tensor,
    w_scale_u8: torch.Tensor,
) -> torch.Tensor:
    """[G,M,K] x [G,N,K] -> [G,M,N] with 1x128 e8m0 blockscale dequant."""
    G, M, K = O_fp8.shape
    N = W_fp8.shape[1]
    act = O_fp8.to(dtypes.fp32).view(G, M, K // GROUP_K, GROUP_K)
    act = act * torch.exp2(x_scale_u8.to(dtypes.fp32) - 127.0).unsqueeze(-1)
    act = act.reshape(G, M, K)
    w = W_fp8.to(dtypes.fp32).view(G, N, K // GROUP_K, GROUP_K)
    w = w * torch.exp2(w_scale_u8.to(dtypes.fp32) - 127.0).unsqueeze(-1)
    w = w.reshape(G, N, K)
    return torch.einsum("gmk,gnk->gmn", act, w)


def run_probe(m: int = 128, n: int = 128, k: int = 256, g: int = 1) -> None:
    assert m % 16 == 0 and n % 16 == 0 and k % 32 == 0, "shape alignment"

    # A[m,k] = (m % 16) + 1 in fp8 -- small magnitudes, unique within 16x16 tile.
    O = torch.zeros(g, m, k, dtype=dtypes.fp8)
    for mi in range(m):
        O[:, mi, :] = float((mi % 16) + 1)

    # Identity weight before shuffle: W[g,n,k] = 1 when n==k else 0.
    W = torch.zeros(g, n, k, dtype=dtypes.fp8)
    diag = min(n, k)
    for i in range(diag):
        W[:, i, i] = torch.tensor(1.0, dtype=dtypes.fp8)

    W_shuf = shuffle_weight(W, layout=(16, 16)).contiguous()

    xs = torch.full((g, m, k // GROUP_K), E8M0_ONE, dtype=torch.uint8)
    ws = torch.full((g, n, k // GROUP_K), E8M0_ONE, dtype=torch.uint8)

    O_in = O.transpose(0, 1).contiguous()  # [m,g,k]
    xs_in = xs.transpose(0, 1).contiguous()
    ref = _ref_bmm(O, W, xs, ws).transpose(0, 1)  # [m,g,n]

    Y = torch.empty(m, g, n, dtype=dtypes.bf16)
    _opus_bmm_a8w8_mxscale_bpreshuffle_raw(O_in, W_shuf, Y, xs_in, ws, 1, 0)

    checkAllclose(
        Y.to(dtypes.fp32), ref, msg="bpreshuffle probe: ", rtol=0.05, atol=0.5
    )
    print(
        f"PASS  m={m} n={n} k={k} g={g}  max_err={ (Y.to(dtypes.fp32)-ref).abs().max().item():.4g}"
    )


def main() -> int:
    gfx = get_gfx_runtime()
    print(f"runtime gfx = {gfx}")
    if gfx != "gfx1250":
        print(
            "SKIP: opus_bmm_a8w8_mxscale_bpreshuffle is gfx1250-only; "
            f"this node reports {gfx}.",
            file=sys.stderr,
        )
        return 2

    # Tile is 128x128x256; use full-tile K and a 128x128 output region first.
    run_probe(m=128, n=128, k=256, g=1)
    # Minimal probe from pipeline header comment.
    run_probe(m=16, n=16, k=128, g=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
