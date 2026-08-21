# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compile smoke test for the persistent / mega-kernel grouped a8w4 GEMM.

Launches the smallest legal grouped shape twice -- plain persistent, then with
the fused gather prologue -- so a trace or codegen regression surfaces without
needing the multi-GPU EP stack. It checks that the kernels build and run, not
that the numerics are right.
"""

import torch

from aiter.ops.flydsl.batched_gemm_mxfp4 import flydsl_grouped_gemm_a8w4_masked
from aiter.ops.flydsl.kernels.grid_barrier import grid_barrier_counter
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.types import FusedGatherContext
from aiter.ops.flydsl.moe_kernels import flydsl_moe_wire_gather_preshuffle

E = 2
K = 256
N = 256
TILE_M, TILE_N, TILE_K = 64, 256, 128
M_WARP, N_WARP = 1, 4
WMMA_REP = (TILE_M // M_WARP) // 16
CONTIG_M = 128
TOPK = 2


def _operands(device):
    # e4m3 / e2m1 codes are kept small and the e8m0 exponents at 2^0 so the
    # result stays finite: a bitwise compare is meaningless once NaN is in it.
    a = torch.randint(0, 0x30, (1, CONTIG_M, K), dtype=torch.uint8, device=device)
    a_scales = torch.full(
        (CONTIG_M // WMMA_REP, (K // 32) * WMMA_REP),
        127,
        dtype=torch.uint8,
        device=device,
    )
    w = torch.randint(0, 0x30, (E, N, K // 2), dtype=torch.uint8, device=device)
    w_scales = torch.full(
        (E, N // 32, (K // 32) * 32), 127, dtype=torch.uint8, device=device
    )
    # Tile-aligned per-expert exclusive end rows.
    psum = torch.tensor([TILE_M, CONTIG_M], dtype=torch.int32, device=device)
    out = torch.empty((1, CONTIG_M, N), dtype=torch.bfloat16, device=device)
    return a, a_scales, w, w_scales, psum, out


def _launch(out, a, a_scales, w, w_scales, psum, **kw):
    flydsl_grouped_gemm_a8w4_masked(
        out,
        a,
        w,
        a_scales.view(torch.int32),
        w_scales.view(torch.int32),
        psum,
        n_experts=E,
        contiguous_m=CONTIG_M,
        N=N,
        K=K,
        tile_m=TILE_M,
        tile_n=TILE_N,
        tile_k=TILE_K,
        m_warp=M_WARP,
        n_warp=N_WARP,
        num_buffers=2,
        **kw,
    )
    torch.cuda.synchronize()


def main():
    device = torch.device("cuda:0")
    a, a_scales, w, w_scales, psum, out = _operands(device)

    _launch(out, a, a_scales, w, w_scales, psum)
    ref = out.clone()
    print("[ok] non-persistent baseline")

    out.zero_()
    _launch(out, a, a_scales, w, w_scales, psum, persistent=True)
    assert torch.equal(out, ref), "persistent grid diverged from the baseline"
    print("[ok] persistent grid, bit-identical to baseline")

    # One block for every tile is the easy case; squeeze the grid so each block
    # has to walk several tiles and reuse its LDS arena between them.
    out.zero_()
    _launch(out, a, a_scales, w, w_scales, psum, persistent=True, grid_blocks=1)
    assert torch.equal(out, ref), "single-block persistent loop diverged"
    print("[ok] persistent grid with 1 block walking every tile")

    # Fused gather. Each route owns its own contiguous row here, so the
    # prologue has to fill the whole A operand from the wire buffer.
    wire_nbytes = K + K // 32
    numel = CONTIG_M
    wire = torch.randint(
        0, 0x30, (numel // TOPK, wire_nbytes), dtype=torch.uint8, device=device
    )
    wire[:, K:] = 127
    topids_to_rows = torch.arange(numel, dtype=torch.int32, device=device)

    # Reference: the standalone gather kernel this prologue replaces, feeding
    # the same persistent GEMM.
    gathered_a, gathered_s = flydsl_moe_wire_gather_preshuffle(
        wire,
        1,
        CONTIG_M,
        wmma_rep=WMMA_REP,
        topids_to_rows=topids_to_rows,
        source_topk=TOPK,
        feat_dim=K,
    )
    out.zero_()
    _launch(out, gathered_a, gathered_s, w, w_scales, psum, persistent=True)
    gather_ref = out.clone()

    fused_a = torch.zeros_like(gathered_a)
    fused_s = torch.zeros_like(gathered_s)
    ctx = FusedGatherContext(
        wire=wire.reshape(-1),
        topids_to_rows=topids_to_rows,
        num_valid_routes=torch.empty(0, dtype=torch.int32, device=device),
        grid_bar=grid_barrier_counter(device),
        numel=numel,
        feat_dim=K,
        wmma_rep=WMMA_REP,
        source_topk=TOPK,
    )
    out.zero_()
    _launch(
        out, fused_a, fused_s, w, w_scales, psum, persistent=True, fused_gather=ctx
    )
    assert torch.equal(fused_a, gathered_a), "prologue built a different A payload"
    assert torch.equal(fused_s, gathered_s), "prologue built a different A scale"
    assert torch.equal(
        out.view(torch.int16), gather_ref.view(torch.int16)
    ), "mega-kernel diverged from gather + persistent GEMM"
    print("[ok] fused gather prologue matches the standalone gather kernel")

    # One barrier generation, so the counter lands on exactly the block count.
    print("grid_bar =", int(ctx.grid_bar.item()))


if __name__ == "__main__":
    main()
