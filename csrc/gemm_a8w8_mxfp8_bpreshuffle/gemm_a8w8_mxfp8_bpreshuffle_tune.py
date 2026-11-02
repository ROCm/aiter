# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune gemm_a8w8_mxfp8_bpreshuffle (FP8 GEMM, e8m0 microscales) on gfx950.

gfx950 runs two FlyDSL pipelines on unshuffled row-major scales (x_scale
[M, K/32], w_scale [N, K/32]): the mxpsh kernel and the 8-wave 256x256 one
(ids from 1_000_000). Every candidate of both that fits the shape is timed and
the fastest lands in the same table the gfx1250 rows live in, keyed by gfx and
w_scale_block (1x32: a shape list without the column is one):

    python3 csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \\
        -i aiter/configs/a8w8_mxfp8_bpreshuffle_untuned_gemm.csv \\
        -o aiter/configs/model_configs/dsv4_a8w8_mxfp8_bpreshuffle_tuned_gemm.csv

--mxscale_table tunes gemm_a8w8_blockscale_bpreshuffle's e8m0 table instead,
the only one the 32x32 (w_scale [N/32, K/32], as bmm's) and 128x128 blocks
(w_scale [N/128, K/128], x_scale 1x128 as its column-major bytes) have, and
the one its 1x32 calls read: the batched GEMM at B = 1 (bmm) joins the mxfp8
pipelines on every block (its candidates: the table's tuned bmm kernels for
M/2..2M, its heuristic pick, and their split-K neighbours), and a row's
kernelId names the backend that runs it, "mxfp8" or "bmm":

    python3 csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \\
        --mxscale_table \\
        -i <shapes with a w_scale_block column> \\
        -o aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv

On gfx1250 --mxscale_table times the mxfp8_128 kernels the 2-D dispatch runs
(flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common, row-major A) on 128x128
shapes, and a row's kernelId is the kernel's id in that list.

OPUS uses one normal pass: shape/traits filtering -> benchmark candidates ->
select the fastest valid result. No historical OPUS seeds or time-based retries.
Every available legal CO and JIT geometry baseline is retained. Joint compute
settings stay in a shape/resource-derived range; secondary layouts get mode
coverage. The range accounts for CU fill, the A ring, and scale-panel LDS.
--opus_max_candidates 0 runs the full legal search directly.
"""

import ast
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import torch
import torch.nn.functional as F

from aiter import dtypes, logger
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
    MXFP8_ROW_PIPELINES,
    MXFP8_ROW_W32X32_PIPELINES,
    MXFP8_ROW_W128X128_PIPELINES,
)
from aiter.ops.gemm_op_common import with_mxscale_w_scale_block
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

TUNED_FILE = "aiter/configs/model_configs/dsv4_a8w8_mxfp8_bpreshuffle_tuned_gemm.csv"
# The block of a shape list without a w_scale_block column, and of a tuned
# table from before the column.
W_SCALE_BLOCK = "1x32"
PIPELINES_BY_W_SCALE_BLOCK = {
    "1x32": MXFP8_ROW_PIPELINES,
    "32x32": MXFP8_ROW_W32X32_PIPELINES,
    "128x128": MXFP8_ROW_W128X128_PIPELINES,
}


def generate_data(m, n, k, seed, w_scale_block=W_SCALE_BLOCK, device="cuda"):
    """fp8 operands with e8m0 scales around 2^-3..2^3, unshuffled, per
    w_scale_block: 1x32 x_scale [m, k/32] with a 1x32 [n, k/32] or 32x32
    [n/32, k/32] w_scale; or the blockscale pair, a 1x128 x_scale (a contiguous
    (m, k/128) holding the column-major [k/128, m] bytes) and a 128x128
    [n/128, k/128] w_scale."""
    blocks = {"1x32": (1, 32), "32x32": (32, 32), "128x128": (128, 128)}
    assert w_scale_block in blocks, f"no {w_scale_block} w_scale data"
    w_rows, kb = blocks[w_scale_block]
    torch.manual_seed(seed)
    x = (torch.randn((m, k), device=device) * 2).to(dtypes.fp8)
    weight = (torch.randn((n, k), device=device) * 2).to(dtypes.fp8)
    x_scale = torch.randint(124, 131, (m, k // kb), dtype=torch.uint8, device=device)
    w_scale = torch.randint(
        124, 131, (n // w_rows, k // kb), dtype=torch.uint8, device=device
    )

    def dequant(q, s, rows=1):
        s = torch.exp2(s.float() - 127).repeat_interleave(rows, 0)
        return q.to(dtypes.fp32) * s.repeat_interleave(kb, 1)

    # A 1x128 x_scale is passed as its column-major bytes.
    x_scale_arg = x_scale if kb == 32 else x_scale.T.contiguous().view(m, k // kb)
    return {
        "x": x,
        "weight_shuffle": shuffle_weight(weight, layout=(16, 16)),
        "x_scale": x_scale_arg.view(dtypes.fp8_e8m0),
        "w_scale": w_scale.view(dtypes.fp8_e8m0),
        "out": torch.empty(m, n, dtype=dtypes.bf16, device=device),
        "x_deq": dequant(x, x_scale),
        "w_deq": dequant(weight, w_scale, w_rows),
    }


def run_torch(x_deq, w_deq, dtype=dtypes.bf16):
    return F.linear(x_deq, w_deq).to(dtype)


def opus_kernel_name(kid, *, cu_num):
    from csrc.opus_gemm.opus_gemm_common import get_kernel_instance

    instance = get_kernel_instance(
        "gfx950", "a8w8_blockscale_bpreshuffle", int(kid), cu_num=cu_num
    )
    if instance is None:
        raise ValueError(f"Unknown or unavailable OPUS MXFP8 bpreshuffle kid {kid}")
    return instance.name


def _opus_geometry(instance):
    entry = instance.bpreshuffle_co or instance.bpreshuffle_entry
    return (
        entry["pipeline"],
        instance.B_M,
        instance.B_N,
        instance.B_K,
        instance.T_M,
        instance.T_N,
    )


def _normalize_opus_geometry(value):
    """Accept [pipeline, BM, BN, BK, TM, TN] or named geometry fields."""
    if isinstance(value, dict):
        fields = {str(key).lower().replace("_", ""): val for key, val in value.items()}
        tile = fields.get("tile")
        waves = fields.get("wavelayout")
        if tile is not None and waves is not None:
            value = (fields["pipeline"], *tile, *waves)
        else:
            value = tuple(
                fields[key] for key in ("pipeline", "bm", "bn", "bk", "tm", "tn")
            )
    if not isinstance(value, (tuple, list)) or len(value) != 6:
        raise ValueError("geometry must contain pipeline, BM, BN, BK, TM, TN")
    pipeline = str(value[0]).strip()
    if not pipeline:
        raise ValueError("geometry pipeline must not be empty")
    dimensions = tuple(int(str(dimension).strip()) for dimension in value[1:])
    if min(dimensions) <= 0:
        raise ValueError("geometry dimensions must be positive integers")
    return (pipeline, *dimensions)


def load_opus_geometry_file(path):
    """Read shape-specific geometries without importing any GPU code.

    CSV rows have M,N,K and pipeline,BM,BN,BK,TM,TN (B_M etc. also work),
    or a JSON old_opus_geometry/opus_geometry field containing the same tuple
    or named fields. Historical tuple literals are accepted too. Repeated rows
    select multiple geometries; blank geometry fields in analysis tables skip
    shapes for which no historical geometry was recorded.
    """
    by_shape = defaultdict(set)
    with open(path, newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.DictReader(stream), start=2):
            try:
                fields = {
                    key.lower().replace("_", ""): value for key, value in row.items()
                }
                shape = tuple(int(fields[key]) for key in ("m", "n", "k"))
                if min(shape) <= 0:
                    raise ValueError("shape dimensions must be positive integers")
                encoded = fields.get("oldopusgeometry") or fields.get("opusgeometry")
                if not encoded and (
                    "oldopusgeometry" in fields or "opusgeometry" in fields
                ):
                    continue
                if encoded:
                    try:
                        value = json.loads(encoded)
                    except json.JSONDecodeError:
                        value = ast.literal_eval(encoded)
                    singular = isinstance(value, dict) or (
                        isinstance(value, (list, tuple))
                        and len(value) == 6
                        and isinstance(value[0], str)
                    )
                    values = [value] if singular else value
                else:
                    values = [fields]
                if not values:
                    raise ValueError("shape has no requested OPUS geometry")
                by_shape[shape].update(
                    _normalize_opus_geometry(value) for value in values
                )
            except (ValueError, TypeError, KeyError, SyntaxError) as exc:
                raise ValueError(
                    f"{path}:{line}: invalid OPUS geometry: {exc}"
                ) from exc
    if not by_shape:
        raise ValueError(f"{path}: no OPUS geometry rows")
    return dict(by_shape)


_OPUS_TRAIT_DEFAULTS = {
    "prefetch_b_vgpr": (None,) * 4
    + (
        "0",
        "false",
        "7168",
        "4",
        "1",
        "0",
        "2",
        "0",
        "256",
        "false",
        "false",
        "false",
        "0",
    ),
    "prefetch_b_vgpr_overlap": (None,) * 4
    + (
        "0",
        "false",
        "7168",
        "4",
        "1",
        "0",
        "2",
        "0",
        "256",
        "false",
        "false",
        "false",
        "0",
        "0",
    ),
    "pingpong_lds": (
        "192",
        "256",
        "128",
        "0",
        "4",
        "0",
        "0",
        "0",
        "false",
        "false",
        "false",
        "0",
        "8",
    ),
    "wave_interleave": ("256", "0", "4", "true", "0", "0", "4"),
}


@lru_cache(maxsize=65536)
def _opus_schedule_profile(pipeline, traits):
    """Normalize omitted defaults using the current pipelines' traits interface."""
    match = re.fullmatch(r"\s*\w+\s*<([^<>]*)>\s*", traits)
    if not match:
        raise ValueError(f"Unsupported OPUS traits: {traits}")
    body = match[1].strip()
    args = tuple(arg.strip().lower() for arg in body.split(",")) if body else ()
    defaults = _OPUS_TRAIT_DEFAULTS[pipeline]
    args += defaults[len(args) :]
    if len(args) != len(defaults) or any(
        arg is None or not re.fullmatch(r"-?\d+|true|false", arg) for arg in args
    ):
        raise ValueError(f"Incomplete OPUS traits: {traits}")
    return args


@lru_cache(maxsize=65536)
def _opus_effective_schedule(pipeline, traits, bm, bn, bk, wn, wg):
    """Compare B prefetch distances by the queue depth the traits actually use."""
    profile = list(_opus_schedule_profile(pipeline, traits))
    if pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
        stages, distance = int(profile[2]), int(profile[3])
        mr, nr = bm // 16, bn // (wn * 16)
        room = (256 if wg == 2 else 512) - mr * nr * 4 - (bk // 128) * mr * 8 - 48
        max_ahead = max(room // 8 - 1, 1)
        rotate = wg == 2 or (bk // 128) * mr >= 16
        ahead = (1 if stages == 2 else 4) if rotate else max_ahead
        if (
            not rotate
            and bk == 128
            and nr >= 4
            and profile[14] == "true"
            and int(profile[6]) >= 1536
        ):
            ahead = 3
        profile[3] = f"ahead{min(distance * (bk // 128) * nr, ahead)}"
    return tuple(profile)


def _opus_variant_profile(instance, m, n):
    entry = instance.bpreshuffle_entry
    profile = _opus_effective_schedule(
        entry["pipeline"],
        entry["traits"],
        instance.B_M,
        instance.B_N,
        instance.B_K,
        instance.T_N,
        instance.WG_PER_CU,
    )
    if entry["pipeline"] == "pingpong_lds" and (
        int(profile[5]) > 0 or n // instance.B_N < 256 or m <= instance.B_M
    ):
        # GroupM overwrites the N swizzle. Otherwise GroupN is used only in
        # the wide-N branch of the m-repeat pipeline's launch-order mapping.
        profile = (*profile[:12], "inactive")
    return profile


def _opus_jit_lds(instance, k):
    """LDS allocation from the four registered JIT pipelines' traits formulas."""
    entry = instance.bpreshuffle_entry
    profile = _opus_schedule_profile(entry["pipeline"], entry["traits"])
    bm, bn, threads = instance.B_M, instance.B_N, instance.BLOCK_SIZE
    if entry["pipeline"] in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
        scale = math.ceil(bm * (k // 128) / (threads * 16)) * threads * 16
        return int(profile[2]) * bm * instance.B_K + scale + threads * 4
    if entry["pipeline"] == "pingpong_lds":
        packs = math.ceil((bm // (instance.T_M * 16)) / 4)
        return 264 * (bm + bn) + int(profile[2]) * (packs * instance.T_M * 64 + 4)
    rows = threads // 8
    a_slot = math.ceil((bm // 2) / rows) * rows * 128
    return 2 * (2 * a_slot + 2 * 128 * 128) + 127 * 256 + 128 * 2


def _opus_schedule_modes(pipeline, profile):
    """Cover all trait values and the pipelines' coupled scheduling choices."""
    # A new trait value is covered automatically. Coupled groups additionally
    # retain combinations affecting queue/resource limits or execution routes.
    indices = [(index,) for index in range(len(profile))]
    if pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
        # stage/ahead/WG, A/B cache, store/output, contiguous/pins, XCD/groupM
        indices += [(2, 3, 8), (4, 9), (10, 11), (13, 14, 15), (5, 16)]
        if pipeline == "prefetch_b_vgpr_overlap":
            # Publication timing depends on the A ring and output route.
            indices += [(2, 17), (11, 17)]
    elif pipeline == "pingpong_lds":
        # panel/fixedK, fixedK/fragments, cache/fragments/guard, launch groups
        indices += [(2, 3), (3, 8), (6, 7, 8, 10), (5, 12), (5, 9)]
    elif pipeline == "wave_interleave":
        # groupM/XCD and A/B cache
        indices += [(2, 3), (4, 5)]
    return {
        (positions, tuple(profile[index] for index in positions))
        for positions in indices
    }


def _opus_representative_variants(kids, registry, m, n, k):
    """Low-LDS baseline followed by greedy coverage of legal scheduling modes.

    The count follows the modes present in the registry, with no fixed number
    of variants per tile and no use of historical timing or winning kernels.
    Covering these modes does not cover their entire Cartesian product.
    """
    profiles = {}
    for kid in sorted(kids):
        profile = _opus_variant_profile(registry[kid], m, n)
        profiles.setdefault(profile, kid)
    profiles = {kid: profile for profile, kid in profiles.items()}
    footprints = {kid: _opus_jit_lds(registry[kid], k) for kid in profiles}
    pipeline = registry[kids[0]].bpreshuffle_entry["pipeline"]
    modes = {
        kid: _opus_schedule_modes(pipeline, profile)
        for kid, profile in profiles.items()
    }
    baseline = min(profiles, key=lambda kid: (footprints[kid], kid))
    selected = [baseline]
    uncovered = set().union(*modes.values()) - modes.pop(baseline)
    while uncovered:
        kid = min(
            modes,
            key=lambda kid: (-len(modes[kid] & uncovered), footprints[kid], kid),
        )
        selected.append(kid)
        uncovered.difference_update(modes.pop(kid))
    return selected


def _opus_rank_geometries(groups, m, n, cu_num, top_geometries):
    """Keep the existing shape ranking independently for each pipeline/BK/layout."""
    if not top_geometries:
        return set(groups)
    by_layout = defaultdict(list)
    for geometry in groups:
        pipeline, bm, bn, bk, wm, wn = geometry
        tiles = math.ceil(m / bm) * math.ceil(n / bn)
        padding = math.ceil(m / bm) * bm / m * math.ceil(n / bn) * bn / n
        occupancy = tiles / max(cu_num, 1)
        cost = (
            0.25 * (padding - 1)
            + 0.25 * max(0, 1 - occupancy)
            + 0.1 * (math.ceil(occupancy) / occupancy - 1)
            + 8 / bm
            + 8 / bn
        )
        by_layout[(pipeline, bk, wm, wn)].append((cost, geometry))
    return {
        geometry
        for geometries in by_layout.values()
        for _, geometry in sorted(geometries)[:top_geometries]
    }


def _opus_geometry_cost(geometry, m, n, cu_num):
    """Relative padding, CU fill, tail, and matrix-request cost of a tile."""
    bm, bn = geometry[1:3]
    tiles = math.ceil(m / bm) * math.ceil(n / bn)
    padding = math.ceil(m / bm) * bm / m * math.ceil(n / bn) * bn / n
    occupancy = tiles / max(cu_num, 1)
    return (
        0.25 * (padding - 1)
        + 0.25 * max(0, 1 - occupancy)
        + 0.1 * (math.ceil(occupancy) / occupancy - 1)
        + 8 / bm
        + 8 / bn
    )


def _opus_m_repeat_cost(geometry, m, n, k, cu_num):
    """Estimate core-tile work, including the extra rolling A-fragment wait.

    The coefficients are relative work weights, not measured GPU latencies.
    A fixed startup cost competes with the K-dependent refill cost; M/N tile
    padding and dispatch tails can keep either E_M=3 or E_M=4 in the range.
    """
    _, bm, bn, _, wm, wn = geometry
    em, en = bm // (16 * wm), bn // (16 * wn)
    mfma = em * en * 32 * wm * wn / 4
    transfer = (bm + bn) * 128 / 512
    refill = max(0, em - 3) * 64
    body = (k / 128) * (mfma + transfer + refill) + 3072 + bm * bn * 2 / 512
    tiles = math.ceil(m / bm) * math.ceil(n / bn)
    occupancy = tiles / max(cu_num, 1)
    tail = 1 + 0.1 * (math.ceil(occupancy) / occupancy - 1)
    return tiles * body * tail / (m * n * k)


def _opus_refine_geometry_range(primary, secondary, m, n, k, cu_num):
    """Expand joint parameters near the computed costs and adequate CU supply."""
    dropped = set()
    core = [
        g
        for g in primary
        if g[0] == "pingpong_lds" and g[4] * g[5] == 8 and g[1] // (g[4] * 16) >= 3
    ]
    if core:
        costs = {g: _opus_m_repeat_cost(g, m, n, k, cu_num) for g in core}
        best = min(costs.values())
        dropped.update(g for g in core if costs[g] > 1.025 * best)
    by_layout = defaultdict(list)
    for geometry in primary:
        if geometry[0] in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
            by_layout[(geometry[0], *geometry[3:])].append(geometry)
    for geometries in by_layout.values():
        costs = {g: _opus_geometry_cost(g, m, n, cu_num) for g in geometries}
        best = min(costs.values())
        # Two-wave blocks have more residency-dependent CU-fill choices.
        margin = 4 / 3 if geometries[0][4] * geometries[0][5] < 4 else 1.1
        dropped.update(g for g in geometries if costs[g] > margin * best)
    # Below half a CU grid, retain resource-mode representatives rather than
    # multiplying every parameter on a poorly supplied large tile.
    dropped.update(
        g
        for g in primary
        if math.ceil(m / g[1]) * math.ceil(n / g[2]) < max(cu_num, 1) / 2
    )
    return primary - dropped, (secondary | dropped) - (primary - dropped)


def _opus_calculate_geometry_range(groups, registry, m, n, k, cu_num):
    """Calculate primary tile ranges and secondary resource states, without times.

    B tiles keep the largest full-A geometry. When it underfills the CU grid,
    allow up to 1.5x more M tiles and 2x more N tiles to recover parallelism.
    The 4-wave layout gets full settings; other wave counts get them while
    their grid has too few blocks to fill four SIMDs per CU.

    M-repeat core tiles fill the three-fragment A ring without requiring a
    second packed scale word (E_M=3..4). If those tiles underfill the CUs,
    also search the largest layouts whose matrix/scale LDS permits two blocks.
    """
    by_pipeline = defaultdict(list)
    for geometry in groups:
        by_pipeline[geometry[0]].append(geometry)
    primary, secondary = set(), set()
    cu_num = max(cu_num, 1)
    for pipeline, tiles in by_pipeline.items():
        if pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
            layouts = defaultdict(list)
            native = any(
                wm * wn == 4 and (bk // 128) * (bm // 16) < 16
                for _, bm, _, bk, wm, wn in tiles
            )
            for geometry in tiles:
                layouts[(geometry[3], geometry[4], geometry[5])].append(geometry)
            for (bk, wm, wn), layout in layouts.items():
                full_a = [g for g in layout if (bk // 128) * (g[1] // 16) < 16]
                if full_a:
                    largest = max(full_a, key=lambda g: (g[1] * g[2], g))
                    max_bm, max_bn = largest[1:3]
                    grid = math.ceil(m / max_bm) * math.ceil(n / max_bn)
                    if grid < cu_num:
                        chosen = {
                            g
                            for g in full_a
                            if 3 * g[1] >= 2 * max_bm and 2 * g[2] >= max_bn
                        }
                    else:
                        chosen = {largest}
                    waves = wm * wn
                    if native and waves != 4 and grid >= cu_num * max(1, 4 / waves):
                        secondary.update(chosen)
                    else:
                        primary.update(chosen)
                rotated = [g for g in layout if (bk // 128) * (g[1] // 16) >= 16]
                if rotated:
                    # This changes the A live set, LDS waits, and B queue clamp.
                    # Cover its resource modes even when full-A tiles rank higher.
                    secondary.update(
                        _opus_rank_geometries(
                            {g: groups[g] for g in rotated}, m, n, cu_num, 1
                        )
                    )
        elif pipeline == "pingpong_lds":
            core = [
                g for g in tiles if g[4] * g[5] == 8 and 3 <= g[1] // (g[4] * 16) <= 4
            ]
            underfilled = True
            if core:
                max_bn = max(g[2] for g in core)
                core = [g for g in core if g[2] == max_bn]
                primary.update(core)
                largest = max(core, key=lambda g: (g[1] * g[2], g))
                grid = math.ceil(m / largest[1]) * math.ceil(n / largest[2])
                underfilled = grid < cu_num
            for waves in (4, 8):
                resident = [
                    g
                    for g in tiles
                    if g[4] * g[5] == waves
                    and min(_opus_jit_lds(registry[kid], k) for kid in groups[g])
                    <= 160 * 1024 // 2
                ]
                if resident:
                    max_area = max(g[1] * g[2] for g in resident)
                    chosen = {g for g in resident if g[1] * g[2] == max_area}
                    (primary if underfilled else secondary).update(chosen)
        elif pipeline == "wave_interleave":
            ranked = _opus_rank_geometries(
                {g: groups[g] for g in tiles}, m, n, cu_num, 1
            )
            for geometry in ranked:
                (primary if geometry[4] * geometry[5] == 8 else secondary).add(geometry)
        else:
            secondary.update(tiles)
        if not any(g in primary for g in tiles) and not any(
            g in secondary for g in tiles
        ):
            # A registry/shape may have no core or two-resident-block layout.
            secondary.update(
                _opus_rank_geometries({g: groups[g] for g in tiles}, m, n, cu_num, 1)
            )
    return _opus_refine_geometry_range(primary, secondary, m, n, k, cu_num)


def _opus_secondary_variants(kids, registry, m, n, k):
    """Cover resource modes and early/middle/late publication in secondary tiles.

    PublishM moves the A request within the first B fragment's M repeats.
    Complete publication/queue/cache products remain in the primary range.
    """
    pipeline = registry[kids[0]].bpreshuffle_entry["pipeline"]
    if pipeline == "prefetch_b_vgpr_overlap":
        profiles = {
            kid: _opus_schedule_profile(
                pipeline, registry[kid].bpreshuffle_entry["traits"]
            )
            for kid in kids
        }
        publications = sorted({int(profile[17]) for profile in profiles.values()})
        if len(publications) > 3:
            midpoint = (publications[0] + publications[-1]) / 2
            middle = min(publications, key=lambda value: abs(value - midpoint))
            retained = {publications[0], middle, publications[-1]}
            kids = [kid for kid in kids if int(profiles[kid][17]) in retained]
    return _opus_representative_variants(kids, registry, m, n, k)


def _opus_joint_routes(kids, registry, m, n):
    """Keep complete joint settings on baseline compute and tile-order routes.

    B-VGPR and overlap keep all queues/caches/publication points on the default
    LDS-output route. M-repeat keeps all K/panel/cache/fragment/guard settings
    with ungrouped or the smallest registered M grouping. Interleave is small
    enough to retain all its settings. Alternate output/M-group routes also get
    trait-mode coverage below. No historical kernel or time enters this filter.
    """
    pipeline = registry[kids[0]].bpreshuffle_entry["pipeline"]
    profiles = {
        kid: _opus_schedule_profile(pipeline, registry[kid].bpreshuffle_entry["traits"])
        for kid in kids
    }
    if pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
        default_output = _OPUS_TRAIT_DEFAULTS[pipeline][11]
        primary = [kid for kid in kids if profiles[kid][11] == default_output]
    elif pipeline == "pingpong_lds":
        nonzero = min(
            (int(profile[5]) for profile in profiles.values() if int(profile[5]) > 0),
            default=0,
        )
        primary = [kid for kid in kids if int(profiles[kid][5]) in (0, nonzero)]
    else:
        primary = kids
    # B distances can clamp to the same effective queue; GroupN can be
    # overwritten or inactive for this shape. Keep one canonical representative
    # of each such identical route, without merging different joint settings.
    unique = {}
    for kid in sorted(primary):
        unique.setdefault(_opus_variant_profile(registry[kid], m, n), kid)
    return list(unique.values())


def _opus_calculated_joint_routes(kids, registry, m, n, k):
    """Keep complete combinations within actual panel/queue/order resources."""
    instance = registry[kids[0]]
    pipeline = instance.bpreshuffle_entry["pipeline"]
    if pipeline == "pingpong_lds":
        em = instance.B_M // (instance.T_M * 16)
        core = instance.T_M * instance.T_N == 8 and 3 <= em <= 4
        if not core:
            # This geometry was admitted for two resident blocks. A larger
            # panel can change it to one, so check each variant's actual LDS.
            resident = [
                kid
                for kid in kids
                if _opus_jit_lds(registry[kid], k) <= 160 * 1024 // 2
            ]
            if resident:
                kids = resident
    routes = _opus_joint_routes(kids, registry, m, n)
    if not routes:
        return routes
    if pipeline == "wave_interleave":
        profiles = {
            kid: _opus_schedule_profile(
                pipeline, registry[kid].bpreshuffle_entry["traits"]
            )
            for kid in routes
        }
        smallest = min(
            (int(p[2]) for p in profiles.values() if int(p[2]) > 0), default=0
        )
        routes = [kid for kid in routes if int(profiles[kid][2]) in (0, smallest)]
    elif pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
        aheads = {
            kid: int(_opus_variant_profile(registry[kid], m, n)[3][5:])
            for kid in routes
        }
        # Bound B-ring words by the full-A fragment live set, keeping the next
        # available queue level when the registered distances straddle it.
        target = max((instance.B_K // 128) * (instance.B_M // 16) - 1, 1)
        bound = min(
            (ahead for ahead in aheads.values() if ahead >= target),
            default=max(aheads.values()),
        )
        routes = [kid for kid in routes if aheads[kid] <= bound]
    return routes


def _opus_filtered_candidates(
    candidates, registry, m, n, k, cu_num, top_geometries, cap
):
    """Calculate the search range, retaining every legal CO/geometry baseline."""
    selected = {
        kid for kid, instance in candidates if instance.bpreshuffle_co is not None
    }
    co_count = len(selected)
    groups = defaultdict(list)
    for kid, instance in candidates:
        if instance.bpreshuffle_co is None:
            groups[_opus_geometry(instance)].append(kid)
    if top_geometries is None:
        primary, secondary = _opus_calculate_geometry_range(
            groups, registry, m, n, k, cu_num
        )
    else:
        primary = _opus_rank_geometries(groups, m, n, cu_num, top_geometries)
        secondary = set()
    for kids in groups.values():
        selected.add(min(kids, key=lambda kid: (_opus_jit_lds(registry[kid], k), kid)))
    for geometry in primary:
        selected.update(
            _opus_calculated_joint_routes(groups[geometry], registry, m, n, k)
            if top_geometries is None
            else _opus_joint_routes(groups[geometry], registry, m, n)
        )
    required = len(selected)
    if cap is not None and cap < required:
        raise ValueError(
            f"OPUS cap {cap} would discard the protected search range for {(m, n, k)}; "
            f"need at least {required}, or 0 for the full legal search"
        )
    policies = {
        geometry: [
            kid
            for kid in (
                _opus_secondary_variants(groups[geometry], registry, m, n, k)
                if geometry in secondary
                else _opus_representative_variants(groups[geometry], registry, m, n, k)
            )
            + (
                _opus_secondary_variants(groups[geometry], registry, m, n, k)
                if top_geometries == 0
                else []
            )
            if kid not in selected
        ]
        for geometry in sorted(primary | secondary)
    }
    for rank in range(max(map(len, policies.values()), default=0)):
        for geometry, variants in policies.items():
            if cap is not None and len(selected) >= cap:
                return [(kid, registry[kid]) for kid in sorted(selected)]
            if rank < len(variants):
                selected.add(variants[rank])
    print(
        f"[OPUS range] shape={(m, n, k)} primary={len(primary)} "
        f"secondary={len(secondary)} geometries={len(groups)} "
        f"co={co_count} protected={required} candidates={len(selected)}",
        flush=True,
    )
    return [(kid, registry[kid]) for kid in sorted(selected)]


def opus_candidates(
    m, n, k, *, cu_num, top_geometries=None, geometries=None, max_candidates=None
):
    """Filter legal kernels by shape/traits, then tune the returned set once.

    The default computes a tile/resource range using shape and kernel traits.
    Full joint settings remain in that range; secondary states get mode coverage.
    A nonnegative top_geometries explicitly selects the prior geometry ranking.
    max_candidates=0 runs the full legal space. Explicit geometry files retain
    the full requested space unless a positive cap requests parameter filtering.
    """
    from csrc.opus_gemm.opus_gemm_common import candidate_kids_for_shape, get_registry

    if top_geometries is not None and (
        type(top_geometries) is not int or top_geometries < 0
    ):
        raise ValueError("opus_top_geometries must be None or an integer >= 0")
    if max_candidates is not None and (
        type(max_candidates) is not int or max_candidates < 0
    ):
        raise ValueError("opus_max_candidates must be an integer >= 0")
    registry = get_registry("a8w8_blockscale_bpreshuffle", "gfx950", cu_num)
    candidates = [
        (kid, registry[kid])
        for kid in candidate_kids_for_shape("gfx950", m, n, k, cu_num=cu_num)
    ]
    legal_count = len(candidates)
    if geometries is not None:
        requested = {_normalize_opus_geometry(geometry) for geometry in geometries}
        if not requested:
            raise ValueError(f"No requested OPUS geometry for shape {(m, n, k)}")
        missing = requested - {_opus_geometry(instance) for _, instance in candidates}
        if missing:
            raise ValueError(
                f"OPUS geometries unavailable for {(m, n, k)}: {sorted(missing)}"
            )
        candidates = [
            (kid, instance)
            for kid, instance in candidates
            if _opus_geometry(instance) in requested
        ]
    if not candidates:
        return []
    exhaustive = max_candidates == 0 or (
        geometries is not None and max_candidates is None
    )
    if exhaustive:
        result = candidates
    else:
        result = _opus_filtered_candidates(
            candidates,
            registry,
            m,
            n,
            k,
            cu_num,
            0 if geometries is not None else top_geometries,
            max_candidates,
        )
    print(
        f"[OPUS search] shape={(m, n, k)} legal={legal_count} "
        f"candidates={len(result)} mode={'full' if exhaustive else 'filtered'}",
        flush=True,
    )
    return result


def ensure_opus_kids_compiled(kids, *, cu_num):
    if not kids:
        return False
    from csrc.opus_gemm.opus_gemm_common import get_registry

    registry = get_registry("a8w8_blockscale_bpreshuffle", "gfx950", cu_num)
    missing = set(kids) - registry.keys()
    if missing:
        raise FileNotFoundError(
            f"OPUS COs are unavailable for kernel IDs {sorted(missing)}; "
            "run csrc/opus_gemm/gen_co/build_co.py --arch gfx950 first."
        )
    opus_dir = str(Path(__file__).resolve().parents[1] / "opus_gemm")
    added_path = opus_dir not in sys.path
    if added_path:
        sys.path.insert(0, opus_dir)
    try:
        from opus_gemm_tune import _ensure_kids_compiled

        return _ensure_kids_compiled(
            kids, family="a8w8_blockscale_bpreshuffle", gfx="gfx950", cu_num=cu_num
        )
    finally:
        if added_path:
            sys.path.remove(opus_dir)


def run_gemm_opus(x, weight_shuffle, x_scale, w_scale, out, kid):
    from aiter.ops.opus import opus_gemm

    return opus_gemm(
        x,
        weight_shuffle,
        out,
        kid=kid,
        family="a8w8_blockscale_bpreshuffle",
        layout="bpreshuffle",
        x_scale=x_scale,
        w_scale=w_scale,
    )


# The batched GEMM at B = 1 in --mxscale_table mode: its ids follow mxfp8's
# (kernelId records the backend in that table).
KERNEL_ID_BASE_BMM = 3_000_000
MXSCALE_KERNEL_ID = "mxfp8"
MXSCALE_BMM_KERNEL_ID = "bmm"


def _bmm_scale_args(w_scale_block):
    from aiter.ops.gemm_op_common import mxscale_bmm_groups_of_block

    rows, cols = mxscale_bmm_groups_of_block(w_scale_block)
    # generate_data's x_scale is as wide as the w_scale block; a 1x128 one
    # (blockscale) is passed as its column-major bytes.
    return {
        "x_scale_k": cols,
        "w_scale_n": rows,
        "w_scale_k": cols,
        "x_scale_transposed": cols == 128,
    }


def bmm_candidates(m, n, k, w_scale_block, seeds=()):
    """bmm kernelNames to time for a shape: ``seeds`` (tuned bmm rows of nearby
    M), the heuristic pick, and their split-K neighbours that run it. A
    1x128 x_scale is read row-major at M = 1, which a row may serve through
    padded M, so 128x128 candidates must run both layouts."""
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (
        bmm_candidate_names,
        bmm_config_runs,
        parse_bmm_kernel_name,
    )

    scale_args = _bmm_scale_args(w_scale_block)
    names = bmm_candidate_names(1, m, n, k, seeds, **scale_args)
    row_major = {**scale_args, "x_scale_transposed": False}
    out = []
    for name in names:
        cfg = parse_bmm_kernel_name(name)
        tiles = math.ceil(m / cfg["tile_m"]) * (n // cfg["tile_n"])
        if cfg["splits"] > 1 and tiles > (1 << 16):
            continue
        if scale_args["x_scale_transposed"] and not bmm_config_runs(
            1, n, k, row_major, cfg
        ):
            continue
        out.append(name)
    return out


def run_bmm_flydsl(x, weight_shuffle, x_scale, w_scale, out, kernel_name):
    from aiter.ops.flydsl.batched_gemm_a8w8 import run_bmm_a8w8_mxfp8

    m, k = x.shape
    n = weight_shuffle.shape[0]
    run_bmm_a8w8_mxfp8(
        x.view(m, 1, k),
        weight_shuffle.view(1, n, k),
        x_scale.view(m, 1, x_scale.shape[-1]),
        w_scale.view(1, *w_scale.shape),
        out.view(m, 1, n),
        kernel_name=kernel_name,
        x_scale_transposed=k // x_scale.shape[-1] == 128,
    )
    return out


def run_gemm_flydsl(x, weight_shuffle, x_scale, w_scale, out, kernel_name):
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import run_gemm_a8w8_mxfp8_gfx950

    return run_gemm_a8w8_mxfp8_gfx950(
        x, weight_shuffle, x_scale, w_scale, out, kernel_name
    )


def gfx1250_candidates(m, n, k):
    """(kernelId, kernel) of the gfx1250 mxfp8_128 kernels the 2-D dispatch can
    run for the shape (row-major A)."""
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common import (
        kernel_fits_shape,
        kernels_list,
    )

    return [
        (kernel_id, ki)
        for kernel_id, ki in kernels_list.items()
        if not ki.a_preshuffle and kernel_fits_shape(ki, m, n, k)
    ]


def run_gemm_flydsl_gfx1250(x, weight_shuffle, x_scale, w_scale, out, kernel_name):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_mxfp8_128_bpreshuffle_flydsl

    return gemm_a8w8_mxfp8_128_bpreshuffle_flydsl(
        x, weight_shuffle, x_scale, w_scale, out, {"kernelName": kernel_name}
    )


class GemmA8W8MXFP8BpreshuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": TUNED_FILE,
        "untune_file": "aiter/configs/a8w8_mxfp8_bpreshuffle_untuned_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",
    }

    def _setup_common_arguments(self):
        super()._setup_common_arguments()
        for option in (
            "--run_config",
            "--e2e_tune",
            "--compare",
            "--update_improved",
            "--min_improvement_pct",
        ):
            action = self.parser._option_string_actions.pop(option)
            self.parser._remove_action(action)
            self.parser._optionals._group_actions.remove(action)
        # The shared run() reads these switches before entering candidate tuning.
        self.parser.set_defaults(run_config=False, compare=False, update_improved=False)
        self.parser._option_string_actions["--errRatio"].help = (
            "Tolerable error ratio (default 0.05); candidates above this are rejected."
        )

    def getKernelName(self, kernelId):
        # Pipeline id ranges are disjoint (8-wave ids start at 1_000_000).
        for pipe in MXFP8_ROW_PIPELINES:
            if kernelId in pipe.kernels_list:
                return pipe.kernels_list[kernelId].name
        return None

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--mxscale_table",
            action="store_true",
            help="Tune gemm_a8w8_blockscale_bpreshuffle's e8m0 table: bmm (the "
            "batched GEMM at B = 1) joins on 1x32 / 32x32 / 128x128; OPUS JIT/CO "
            "joins on 128x128. FlyDSL kernelId names mxfp8 or bmm; "
            "OPUS IDs are numeric.",
        )
        self.parser.add_argument(
            "--libtype",
            choices=("both", "opus", "flydsl"),
            default="both",
            help="Backends to search; OPUS calculates a JIT tile/resource range "
            "from the shape and includes every legal prebuilt CO, "
            "using shape and kernel metadata before compiling candidates "
            "and requires --mxscale_table and 128x128. "
            "Prebuild COs with csrc/opus_gemm/gen_co/build_co.py --arch gfx950.",
        )

        self.parser.add_argument(
            "--opus_top_geometries",
            type=int,
            default=None,
            help="Default: calculate tile/resource ranges from shape and traits. "
            "Set an integer to use the previous top-N geometry ranking per "
            "pipeline/BK/wave layout; 0 applies parameter filters to all geometries.",
        )
        self.parser.add_argument(
            "--opus_max_candidates",
            type=int,
            default=None,
            help="Default: calculated shape/resource ranges with no fixed count. "
            "0 runs the full legal search. A positive cap cannot remove the "
            "protected range; "
            "too-small caps raise an error. There is no automatic expansion.",
        )
        self.parser.add_argument(
            "--opus_geometry_file",
            help="Optional CSV selecting pipeline,BM,BN,BK,TM,TN per M,N,K. "
            "Search the complete requested space unless a positive cap is set.",
        )

    def pre_process(self, args):
        self._mxscale_table = args.mxscale_table
        if args.opus_top_geometries is not None and args.opus_top_geometries < 0:
            self.parser.error("--opus_top_geometries must be >= 0")
        if args.opus_max_candidates is not None and args.opus_max_candidates < 0:
            self.parser.error("--opus_max_candidates must be >= 0")
        if args.opus_geometry_file and (
            not self._mxscale_table or args.libtype == "flydsl"
        ):
            self.parser.error(
                "--opus_geometry_file requires --mxscale_table and OPUS tuning"
            )
        if args.libtype == "opus" and not self._mxscale_table:
            self.parser.error("OPUS tuning requires --mxscale_table")
        super().pre_process(args)

    def _bmm_seeds(self):
        """{(N, K, w_scale_block): [(M, bmm kernelName)]} of the tuned
        gemm_a8w8_blockscale_bpreshuffle table, this arch and CU count."""
        from aiter.jit.core import AITER_CONFIGS

        path = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE_FILE
        try:
            df = with_mxscale_w_scale_block(pd.read_csv(path), path)
        except (OSError, pd.errors.EmptyDataError):
            return {}
        df = df[
            (df["gfx"] == self.get_gfx())
            & (df["cu_num"] == self.get_cu_num())
            & (df["kernelId"].astype(str) == MXSCALE_BMM_KERNEL_ID)
        ]
        seeds = {}
        for r in df.itertuples():
            key = (int(r.N), int(r.K), str(r.w_scale_block))
            seeds.setdefault(key, []).append((int(r.M), str(r.kernelName)))
        return seeds

    def get_untuned_gemm_list(self, untuned_gemm_file):
        df = super().get_untuned_gemm_list(untuned_gemm_file)
        if "w_scale_block" not in df.columns:
            df["w_scale_block"] = W_SCALE_BLOCK
        df["w_scale_block"] = df["w_scale_block"].fillna(W_SCALE_BLOCK)
        unknown = set(df["w_scale_block"]) - set(PIPELINES_BY_W_SCALE_BLOCK)
        if unknown:
            raise ValueError(
                f"{untuned_gemm_file}: no gfx950 mxfp8 kernels tuned for w_scale "
                f"block {sorted(unknown)}; tunable: {sorted(PIPELINES_BY_W_SCALE_BLOCK)}"
            )
        return df

    def get_tuned_gemm_list(self, tuned_gemm_file, columns=None):
        df = super().get_tuned_gemm_list(tuned_gemm_file, columns)
        df = with_mxscale_w_scale_block(df, tuned_gemm_file, default=W_SCALE_BLOCK)
        rest = [c for c in df.columns if c not in self.keys]
        return df[[c for c in self.keys if c in df.columns] + rest]

    def calculate(self, results, bpes=(1, 1, 2)):
        return super().calculate(results, bpes=(1, 1, 2))

    def tune(self, untunedf, tunedf, args):
        gfx = self.get_gfx()
        if gfx not in ("gfx950", "gfx1250"):
            logger.warning(
                f"gemm_a8w8_mxfp8_bpreshuffle tuner covers gfx950 and gfx1250, got {gfx}"
            )
            return []
        cu_num = self.get_cu_num()
        gemm_keys = ["x", "weight_shuffle", "x_scale", "w_scale", "out"]
        ref_args = (["x_deq", "w_deq"], dtypes.bf16)
        run_kwargs = {"num_warmup": args.warmup, "num_iters": args.iters}
        seed = 0

        task = []
        self._mxscale_table = args.mxscale_table
        coarse = sorted(set(untunedf["w_scale_block"]) - {W_SCALE_BLOCK})
        if coarse and not self._mxscale_table:
            raise ValueError(
                f"w_scale block {coarse} is tuned for gemm_a8w8_blockscale_"
                f"bpreshuffle only: pass --mxscale_table"
            )
        if gfx == "gfx1250":
            if not self._mxscale_table or set(untunedf["w_scale_block"]) != {"128x128"}:
                raise ValueError(
                    "gfx1250 tunes the mxfp8_128 kernels of gemm_a8w8_blockscale_"
                    "bpreshuffle's table only: pass --mxscale_table and 128x128 shapes"
                )
            if args.libtype == "opus":
                raise ValueError("OPUS bpreshuffle tuning supports gfx950 only")
        seeds = (
            self._bmm_seeds()
            if self._mxscale_table and gfx == "gfx950" and args.libtype != "opus"
            else {}
        )
        opus_by_shape = {}
        geometry_path = getattr(args, "opus_geometry_file", None)
        geometry_by_shape = (
            load_opus_geometry_file(geometry_path) if geometry_path else None
        )
        if gfx == "gfx950" and args.libtype != "flydsl" and self._mxscale_table:
            if args.libtype == "opus" and set(untunedf.w_scale_block) != {"128x128"}:
                raise ValueError("OPUS supports A 1x128 / B 128x128 scales only")
            for row in untunedf.itertuples():
                if row.w_scale_block == "128x128":
                    shape = (int(row.M), int(row.N), int(row.K))
                    if geometry_by_shape is not None and shape not in geometry_by_shape:
                        raise ValueError(
                            f"{geometry_path}: no OPUS geometry for shape {shape}"
                        )
                    opus_by_shape[shape] = opus_candidates(
                        *shape,
                        cu_num=cu_num,
                        top_geometries=getattr(args, "opus_top_geometries", None),
                        geometries=(
                            geometry_by_shape[shape]
                            if geometry_by_shape is not None
                            else None
                        ),
                        max_candidates=getattr(args, "opus_max_candidates", None),
                    )
                    if args.libtype == "opus" and not opus_by_shape[shape]:
                        raise ValueError(f"No OPUS candidate supports {shape}")
        for i in range(len(untunedf)):
            M = int(untunedf.loc[i, "M"])
            N = int(untunedf.loc[i, "N"])
            K = int(untunedf.loc[i, "K"])
            block = str(untunedf.loc[i, "w_scale_block"])
            info_keys = (gfx, cu_num, M, N, K, block)
            opus = opus_by_shape.get((M, N, K), ()) if block == "128x128" else ()
            cands = (
                gfx1250_candidates(M, N, K)
                if gfx == "gfx1250"
                else [
                    (kernel_id, ki)
                    for pipe in PIPELINES_BY_W_SCALE_BLOCK[block]
                    for kernel_id, ki in pipe.kernels_list.items()
                    if args.libtype != "opus" and pipe.fits(ki, M, N, K)
                ]
            )
            run_fn = run_gemm_flydsl_gfx1250 if gfx == "gfx1250" else run_gemm_flydsl
            runs = [(kernel_id, ki.split_k, ki.name, run_fn) for kernel_id, ki in cands]
            if self._mxscale_table and gfx == "gfx950" and args.libtype != "opus":
                near = [
                    name
                    for mm, name in seeds.get((N, K, block), ())
                    if M / 2 <= mm <= M * 2
                ]
                for j, name in enumerate(bmm_candidates(M, N, K, block, near)):
                    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (
                        parse_bmm_kernel_name,
                    )

                    splits = parse_bmm_kernel_name(name)["splits"]
                    runs.append((KERNEL_ID_BASE_BMM + j, splits, name, run_bmm_flydsl))
            runs.extend(
                (kid, 1, opus_kernel_name(kid, cu_num=cu_num), run_gemm_opus)
                for kid, _ in opus
            )
            for kernel_id, split_k, name, run in runs:
                is_opus = run is run_gemm_opus
                info = (
                    info_keys,
                    kernel_id,
                    split_k,
                    name,
                    "opus" if is_opus else "flydsl",
                )
                task.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, block),
                        run,
                        (gemm_keys, kernel_id) if is_opus else (gemm_keys, name),
                        dict(run_kwargs),
                        run_torch,
                        ref_args,
                        {},
                        None,
                        1e-2,
                        0.01,
                        None,
                        None,
                        ("out",),
                    )
                )
            if not runs:
                logger.warning(f"no mxfp8 {block} kernel fits M={M} N={N} K={K}")

        if not task:
            return []
        return self._run_tasks(task, args)

    def _run_tasks(self, tasks, args):
        if not tasks:
            return []
        ensure_opus_kids_compiled(
            {task[0][1] for task in tasks if task[0][-1] == "opus"},
            cu_num=self.get_cu_num(),
        )
        counts = Counter(task[0][0] for task in tasks)
        return mp_tuner(
            tasks,
            [(count, ()) for count in counts.values()],
            args.mp,
            False,
            args.shape_grouped,
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    def result_to_df(self, results):
        rows = []
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype = info
            if time in (self.INVALID_TIME, self.INF_TIME):
                kernelName = "None"
            elif libtype == "opus":
                kernelName = opus_kernel_name(kernelId, cu_num=int(keys[1]))
            elif libtype == "flydsl" and self._mxscale_table and keys[0] == "gfx950":
                # gfx1250 rows keep their kernels_list id, as the table's do.
                kernelId = (
                    MXSCALE_BMM_KERNEL_ID
                    if kernelId >= KERNEL_ID_BASE_BMM
                    else MXSCALE_KERNEL_ID
                )
            tflops, bw = self.calculate((info, time, err_ratio))
            key_dict = dict(zip(self.keys, keys))
            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is "
                    f"{kernelName} splitK={splitK}, {time}us, err_ratio={err_ratio}, "
                    f"tflops={tflops} TFLOPS, bw={bw} GB/s"
                )
            key_dict.update(
                {
                    "libtype": [libtype],
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                    "tflops": [tflops],
                    "bw": [bw],
                    "errRatio": [err_ratio],
                }
            )
            rows.append(key_dict)
        if not rows:
            return pd.DataFrame(columns=self.columns)
        return pd.concat([pd.DataFrame(r) for r in rows], ignore_index=True)


if __name__ == "__main__":
    key = ["gfx", "cu_num", "M", "N", "K", "w_scale_block"]
    resultList = [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]
    tuner = GemmA8W8MXFP8BpreshuffleTuner(
        "a8w8_mxfp8_bpreshuffle_tuned_gemm",
        key=key,
        resultList=resultList,
        description="Tune gfx950 MXFP8 bpreshuffle GEMM (FlyDSL and OPUS JIT/CO)",
    )
    args = tuner.parse_args()
    tuner.run(args, False)
