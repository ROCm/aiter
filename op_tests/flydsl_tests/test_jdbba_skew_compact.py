# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Skew compact TILE_MAP dispatch correctness.

Verifies ``jagged_dense_bmm_dispatched(..., uniform_seqlen=False)`` on variable
``seq_offsets`` (empty groups, partial tiles, full envelope) for headline shapes.
Skew routing policy is documented in ``jagged_dense_bmm_dispatch.json``.
"""

from __future__ import annotations

import sys

import torch
import flydsl.compiler as flyc
from aiter.ops.flydsl.jagged_dense_bmm_dispatch import (
    clear_skew_tile_map_cache,
    jagged_dense_bmm_dispatched,
)

MI = 7680
SEED = 1234
BLOCK_M = 128


def _make_seq_offsets(B):
    g = torch.Generator().manual_seed(SEED)
    u = torch.rand(B, generator=g)
    t = (MI * (u**4)).floor().to(torch.int64)
    t[: max(1, B // 5)] = 0
    t[-1] = MI
    t = torch.clamp(t, max=MI)
    off = torch.zeros(B + 1, dtype=torch.int32)
    off[1:] = torch.cumsum(t, 0).to(torch.int32)
    return off.cuda()


def _torch_ref(jagged, dense, bias, seq_offsets, N):
    L = jagged.shape[0]
    out = torch.zeros((L, N), dtype=torch.bfloat16, device="cuda")
    so = seq_offsets.cpu().tolist()
    for b in range(dense.shape[0]):
        s, e = so[b], so[b + 1]
        if e > s:
            out[s:e] = (
                jagged[s:e].float() @ dense[b].float() + bias[b].float()
            ).bfloat16()
    return out


def _run_shape(B, D, Kout):
    N, K = Kout, D
    torch.manual_seed(0)
    seq_offsets = _make_seq_offsets(B)
    L = int(seq_offsets[-1].item())
    jagged = torch.randn(max(L, 1), K, dtype=torch.bfloat16, device="cuda")
    dense = torch.randn(B, K, N, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(B, N, dtype=torch.bfloat16, device="cuda")
    dense_tall = dense.transpose(1, 2).reshape(B * N, K).contiguous()
    bias_flat = bias.reshape(B * N).contiguous()
    out = torch.zeros(L + BLOCK_M, N, dtype=torch.bfloat16, device="cuda")
    tA = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)

    clear_skew_tile_map_cache()
    jagged_dense_bmm_dispatched(
        tC,
        tA,
        dense_tall,
        bias_flat,
        seq_offsets,
        B,
        MI,
        stream=torch.cuda.current_stream(),
        uniform_seqlen=False,
    )
    torch.cuda.synchronize()
    ref = _torch_ref(jagged, dense, bias, seq_offsets, N)
    cos = torch.nn.functional.cosine_similarity(
        out[:L].float().flatten(), ref.float().flatten(), dim=0
    ).item()
    tag = f"B{B}_D{D}"
    ok = cos > 0.999
    print(f"  {'PASS' if ok else 'FAIL'} {tag} skew compact dispatch  cos={cos:.4f}")
    return ok


def main() -> int:
    ok = True
    for B, D, Kout in [
        (120, 256, 256),
        (120, 512, 512),
        (1024, 256, 256),
        (1024, 512, 512),
    ]:
        ok &= _run_shape(B, D, Kout)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
