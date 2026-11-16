# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tile work and unique buffer bytes for the gfx1250 grouped MoE GEMMs.

This is a mathematical byte model, not an HBM counter or a sum of WGP
requests. Each touched byte is counted once per input buffer and once per
output buffer. Reads and writes of the intermediate buffer remain separate.
"""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class GemmGeometry:
    n: int
    k: int
    tile_m: int
    tile_n: int
    tile_k: int
    m_warp: int
    n_warp: int
    a_is_fp4: bool
    a_preshuffled: bool
    stage1_act: bool = False
    stage1_quant_out: bool = False
    output_element_bytes: int = 2

    def validate(self) -> None:
        wmma_n = 32 if self.a_is_fp4 else 16
        if min(self.n, self.k, self.tile_m, self.tile_n, self.tile_k,
               self.m_warp, self.n_warp) <= 0:
            raise ValueError("GEMM dimensions and wave counts must be positive")
        if self.tile_m % (16 * self.m_warp) or self.tile_n % (wmma_n * self.n_warp):
            raise ValueError("tile dimensions must contain complete wave WMMA tiles")
        if self.k % 128 or self.tile_k % 128:
            raise ValueError("K and tile_k must be WMMA-K128 aligned")
        # The current grouped TDM path has no live-N extent on B or ScaleB.
        # Do not invent safe tail traffic for a shape that reads beyond them.
        if self.n % self.tile_n:
            raise ValueError("unique byte accounting requires N divisible by tile_n")
        if self.a_preshuffled and not self.a_is_fp4:
            raise ValueError("A-preshuffled payload requires FP4")
        if self.stage1_act and self.n % 2:
            raise ValueError("gated activation requires complete gate/up pairs")
        if self.stage1_act and self.stage1_quant_out and self.n % 256:
            raise ValueError("fused MX output requires complete K128 scale words")
        if self.output_element_bytes <= 0:
            raise ValueError("output element width must be positive")


def _stage_work(route_counts: list[int], expert_ends: list[int],
                geometry: GemmGeometry) -> dict:
    tm, tn, tk = geometry.tile_m, geometry.tile_n, geometry.tile_k
    input_rows = output_rows = output_scale_rows = 0
    used_experts: set[int] = set()
    m_tiles = _align_up(expert_ends[-1], tm) // tm
    quant_output = geometry.stage1_act and geometry.stage1_quant_out

    # Reproduce ownership from the actual contiguous-M prefix bounds. This
    # also handles empty experts and gaps when the stages have different M
    # tiles but share the larger alignment. M intervals do not overlap, so
    # their row lengths give the union of touched bytes without an N-tile
    # or cluster multiplier.
    for base in range(0, expert_ends[-1], tm):
        expert = bisect_right(expert_ends, base)
        live = min(tm, expert_ends[expert] - base)
        m16 = _align_up(live, 16)
        used_experts.add(expert)
        input_rows += m16 if geometry.a_preshuffled else live
        output_rows += m16 if quant_output and geometry.a_preshuffled else live
        output_scale_rows += m16 if quant_output else 0

    compute_rows = m_tiles * tm
    output_n = geometry.n // 2 if geometry.stage1_act else geometry.n
    input_bits = 4 if geometry.a_is_fp4 else 8
    output_bits = input_bits if quant_output else geometry.output_element_bytes * 8
    weights = len(used_experts)
    transfers = {
        "A": input_rows * geometry.k * input_bits // 8,
        "B": weights * geometry.n * geometry.k // 2,
        "ScaleA": compute_rows * geometry.k // 32,
        "ScaleB": weights * geometry.n * geometry.k // 32,
        "output": output_rows * output_n * output_bits // 8,
        "output_scale": output_scale_rows * output_n // 32,
    }
    wmma_n = 32 if geometry.a_is_fp4 else 16
    # The K tail executes only its remaining complete K128 WMMA steps. The
    # requested full-body policy ignores dynamic M-fragment skipping, not
    # the statically shorter last K tile.
    wmma_instructions = (
        m_tiles * (geometry.n // tn) * (tm // 16) * (tn // wmma_n)
        * (geometry.k // 128)
    )
    return {
        "tile": [tm, tn, tk],
        "wmma_shape": [16, wmma_n, 128],
        "wmma_instructions": wmma_instructions,
        "flops": wmma_instructions * 2 * 16 * wmma_n * 128,
        "bytes": sum(transfers.values()),
        "read_bytes": sum(transfers[key] for key in ("A", "B", "ScaleA", "ScaleB")),
        "write_bytes": transfers["output"] + transfers["output_scale"],
        "transfers": transfers,
        "route_rows": sum(route_counts),
        "compute_rows": compute_rows,
        "input_rows": input_rows,
        "input_scale_rows": compute_rows,
        "output_rows": output_rows,
        "output_scale_rows": output_scale_rows,
        "weight_experts": sorted(used_experts),
        "m_tiles": m_tiles,
        "n_tiles": geometry.n // tn,
    }


def grouped_moe_work_metrics(
    route_counts: Sequence[int],
    gemm1: GemmGeometry,
    gemm2: GemmGeometry,
    *,
    expert_ends: Sequence[int] | None = None,
    capacity_rows: int | None = None,
) -> dict:
    """Count complete WMMA tiles and unique OOB-filtered buffer bytes.

    ``route_counts`` must come from the same topk_ids used by the timed MoE.
    Negative/dropped routes must be removed by the caller. ``expert_ends``
    optionally checks these counts against the captured device-side psum.
    No WGP, cluster, or repeated-request multiplier enters the byte model.
    """
    counts = [int(count) for count in route_counts]
    if not counts or any(count < 0 for count in counts):
        raise ValueError("route counts must be a nonempty sequence of nonnegative integers")
    gemm1.validate()
    gemm2.validate()
    row_alignment = max(gemm1.tile_m, gemm2.tile_m)
    if row_alignment % gemm1.tile_m or row_alignment % gemm2.tile_m:
        raise ValueError("both stage M tiles must divide the shared row alignment")

    starts, ends = [], []
    start = 0
    for count in counts:
        starts.append(start)
        ends.append(start + count)
        start += _align_up(count, row_alignment)
    if expert_ends is not None and [int(end) for end in expert_ends] != ends:
        raise ValueError("route histogram does not match the captured grouped-M psum")
    if capacity_rows is not None and (
        capacity_rows % row_alignment or capacity_rows < start
    ):
        raise ValueError("grouped buffer capacity does not cover the aligned expert rows")

    work1 = _stage_work(counts, ends, gemm1)
    work2 = _stage_work(counts, ends, gemm2)
    return {
        "expert_route_counts": counts,
        "expert_starts": starts,
        "expert_ends": ends,
        "row_alignment": row_alignment,
        "byte_policy": "unique buffer bytes after descriptor OOB; repeated reads counted once",
        "wmma_policy": "full WMMA body for each scheduled M tile; dynamic M skipping ignored",
        "gemm1": work1,
        "gemm2": work2,
        "total": {
            "flops": work1["flops"] + work2["flops"],
            "bytes": work1["bytes"] + work2["bytes"],
            "wmma_instructions": work1["wmma_instructions"] + work2["wmma_instructions"],
        },
    }
