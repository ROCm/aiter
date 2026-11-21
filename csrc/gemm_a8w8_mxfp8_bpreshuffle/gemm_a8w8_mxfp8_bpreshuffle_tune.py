# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune gemm_a8w8_mxfp8_bpreshuffle (FP8 GEMM, e8m0 microscales).

On gfx1250, 1x32 tuning searches every legal direct, dedicated-small-M, and
B=1 batched FlyDSL candidate with CUDA Graph timing, then keeps a replacement
only after paired stable confirmation clears the configured gain threshold.

gfx950 runs two FlyDSL pipelines on unshuffled row-major scales (x_scale
[M, K/32], w_scale [N, K/32]): the mxpsh kernel and the 8-wave 256x256 one
(ids from 1_000_000). Every candidate of both that fits the shape is timed,
and so is the batched GEMM at B = 1 (bmm, on the same scales). Its candidates
are the tuned bmm kernels of the shape's N and K at M/2..2M in this table and
in gemm_a8w8_blockscale_bpreshuffle's e8m0 table (any block), else that
table's bmm kernels of any N and K at those M, plus the heuristic pick and
their split-K neighbours. The fastest lands in the same
table the gfx1250 rows live in, keyed by gfx and w_scale_block (1x32: a shape
list without the column is one); a bmm row's kernelId is "bmm", which
gemm_a8w8_mxfp8_bpreshuffle runs as the batched GEMM:

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

"""

import gc
import math
import re
import statistics
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
    KERNEL_ID_BASE_PF,
    MXFP8_ROW_PIPELINES,
    MXFP8_ROW_W32X32_PIPELINES,
    MXFP8_ROW_W128X128_PIPELINES,
)
from aiter.ops.gemm_op_common import with_mxscale_w_scale_block
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

TUNED_FILE = "aiter/configs/model_configs/dsv4_a8w8_mxfp8_bpreshuffle_tuned_gemm.csv"
MXSCALE_TUNED_FILE = "aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"
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


def generate_data_gfx1250_1x32(m, n, k, seed, device="cuda"):
    """Generate the n32k4 scale layout consumed by gfx1250's 1x32 op."""
    from aiter.ops.shuffle import shuffle_mxfp8fp4_scale

    data = generate_data(m, n, k, seed, W_SCALE_BLOCK, device)
    data["w_scale"] = shuffle_mxfp8fp4_scale(data["w_scale"])
    return data


def run_torch(x_deq, w_deq, dtype=dtypes.bf16):
    return F.linear(x_deq, w_deq).to(dtype)


_OPUS_B_DEFAULTS = (None,) * 4 + tuple(
    map(str, (0, "false", 64, 4, 1, 0, 0, 256, "false", "false", "false"))
)
_OPUS_TRAIT_DEFAULTS = {
    "prefetch_b_vgpr": _OPUS_B_DEFAULTS,
    "prefetch_b_vgpr_overlap": (*_OPUS_B_DEFAULTS, "0"),
    "pingpong_lds": tuple(map(str, (192, 256, 128, 4, 0, 0, 0, "false", "false", 8))),
}


@lru_cache(maxsize=65536)
def _opus_kernel_profile(geometry, traits, wg, threads):
    """Return effective traits and LDS bytes, reused across shapes."""
    pipeline, bm, bn, bk, wm, wn = geometry
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
    profile = list(args)
    if pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
        alignment = threads * 16
        scale_bytes = math.ceil(bm * int(profile[6]) / alignment) * alignment
        lds = int(profile[2]) * bm * bk + scale_bytes + threads * 4
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
            and (
                profile[13] == "true" or (profile[14] == "false" and mr * nr * 4 >= 128)
            )
        ):
            ahead = 3
        profile[3] = f"ahead{min(distance * (bk // 128) * nr, ahead)}"
    elif pipeline == "pingpong_lds":
        packs = math.ceil((bm // (wm * 16)) / 4)
        lds = 264 * (bm + bn) + int(profile[2]) * (packs * wm * 64 + 4)
    return tuple(profile), lds


def _opus_select_variants(geometry, kids, kernel_profiles, footprints, m, n, mode):
    """Keep a baseline for every tile; expand primary or secondary modes.

    Primary tiles add joint parameter routes and scheduling representatives.
    Secondary tiles add representatives, sampling overlap publication timing.
    """
    selected = {min(kids, key=lambda kid: (footprints[kid], kid))}
    if mode == "baseline":
        return selected
    pipeline, bm, bn, bk, wm, wn = geometry
    unique = {}
    for kid in sorted(kids):
        profile = kernel_profiles[kid]
        if pipeline == "pingpong_lds" and (
            int(profile[4]) > 0 or n // bn < 256 or m <= bm
        ):
            # GroupM overrides GroupN; narrow-N and single-M-tile routes ignore it.
            profile = (*profile[:9], "inactive")
        unique.setdefault(profile, kid)
    profiles = {kid: profile for profile, kid in unique.items()}
    if mode == "primary":
        routes = list(profiles)
        if pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
            routes = [kid for kid in routes if profiles[kid][10] == "0"]
            if routes:
                aheads = {kid: int(profiles[kid][3][5:]) for kid in routes}
                # Retain the queue level nearest to or above the full-A live set.
                target = max((bk // 128) * (bm // 16) - 1, 1)
                bound = min(
                    (ahead for ahead in aheads.values() if ahead >= target),
                    default=max(aheads.values()),
                )
                routes = [kid for kid in routes if aheads[kid] <= bound]
        elif pipeline == "pingpong_lds":
            em = bm // (wm * 16)
            if not (wm * wn == 8 and 3 <= em <= 4):
                # Include each variant's scale panel in two-block residency.
                resident = {kid for kid in kids if footprints[kid] <= 80 * 1024}
                if resident:
                    routes = [kid for kid in routes if kid in resident]
            group_index = 4
            smallest = min(
                (
                    int(profiles[kid][group_index])
                    for kid in routes
                    if int(profiles[kid][group_index]) > 0
                ),
                default=0,
            )
            routes = [
                kid
                for kid in routes
                if int(profiles[kid][group_index]) in (0, smallest)
            ]
        selected.update(routes)
    elif pipeline == "prefetch_b_vgpr_overlap":
        publications = sorted({int(profile[15]) for profile in profiles.values()})
        if len(publications) > 3:
            midpoint = (publications[0] + publications[-1]) / 2
            middle = min(publications, key=lambda value: abs(value - midpoint))
            retained = {publications[0], middle, publications[-1]}
            profiles = {
                kid: profile
                for kid, profile in profiles.items()
                if int(profile[15]) in retained
            }

    # Cover individual trait values and resource/execution-route combinations.
    indices = [(index,) for index in range(len(next(iter(profiles.values()))))]
    if pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
        # stage/ahead/WG, A/B cache, contiguous/pins
        indices += [(2, 3, 8), (4, 9), (12, 13, 14)]
        if pipeline == "prefetch_b_vgpr_overlap":
            indices += [(2, 15), (10, 15)]
    elif pipeline == "pingpong_lds":
        # fragments/cache/guard, launch groups
        indices += [(5, 6, 7, 8), (4, 9)]
    modes = {
        kid: {
            (positions, tuple(profile[index] for index in positions))
            for positions in indices
        }
        for kid, profile in profiles.items()
    }
    baseline = min(profiles, key=lambda kid: (footprints[kid], kid))
    selected.add(baseline)
    uncovered = set().union(*modes.values()) - modes.pop(baseline)
    while uncovered:
        kid = min(
            modes,
            key=lambda kid: (-len(modes[kid] & uncovered), footprints[kid], kid),
        )
        selected.add(kid)
        uncovered.difference_update(modes.pop(kid))
    return selected


def opus_candidates(m, n, k, *, cu_num):
    """Calculate tile ranges and variants from shape and cached kernel resources."""
    from csrc.opus_gemm.opus_gemm_common import candidate_kids_for_shape, get_registry

    registry = get_registry("a8w8_blockscale_bpreshuffle", "gfx950", cu_num)
    legal = candidate_kids_for_shape("gfx950", m, n, k, cu_num=cu_num)
    if not legal:
        return []
    selected = {kid for kid in legal if registry[kid].bpreshuffle_co is not None}
    co_count = len(selected)
    groups, profiles, footprints = defaultdict(list), {}, {}
    for kid in legal:
        instance = registry[kid]
        if instance.bpreshuffle_co is not None:
            continue
        geometry = (
            instance.bpreshuffle_entry["pipeline"],
            instance.B_M,
            instance.B_N,
            instance.B_K,
            instance.T_M,
            instance.T_N,
        )
        groups[geometry].append(kid)
        profile, lds = _opus_kernel_profile(
            geometry,
            instance.bpreshuffle_entry["traits"],
            instance.WG_PER_CU,
            instance.BLOCK_SIZE,
        )
        profiles[kid] = profile
        footprints[kid] = lds

    # Compute each tile's grid, relative cost, and minimum LDS once.
    cu_grid = max(cu_num, 1)
    grids, costs, min_lds, by_pipeline = {}, {}, {}, defaultdict(list)
    for geometry, kids in groups.items():
        pipeline, bm, bn, _, _, _ = geometry
        tiles = math.ceil(m / bm) * math.ceil(n / bn)
        padding = math.ceil(m / bm) * bm / m * math.ceil(n / bn) * bn / n
        occupancy = tiles / cu_grid
        grids[geometry] = tiles
        costs[geometry] = (
            0.25 * (padding - 1)
            + 0.25 * max(0, 1 - occupancy)
            + 0.1 * (math.ceil(occupancy) / occupancy - 1)
            + 8 / bm
            + 8 / bn
        )
        min_lds[geometry] = min(footprints[kid] for kid in kids)
        by_pipeline[pipeline].append(geometry)

    primary, secondary = set(), set()
    for pipeline, tiles in by_pipeline.items():
        layouts = defaultdict(list)
        for geometry in tiles:
            layouts[geometry[3:]].append(geometry)
        ranked = {
            min(layout, key=lambda g: (costs[g], g)) for layout in layouts.values()
        }
        if pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
            native = any(
                wm * wn == 4 and (bk // 128) * (bm // 16) < 16
                for _, bm, _, bk, wm, wn in tiles
            )
            for (bk, wm, wn), layout in layouts.items():
                full_a = [g for g in layout if (bk // 128) * (g[1] // 16) < 16]
                if full_a:
                    largest = max(full_a, key=lambda g: (g[1] * g[2], g))
                    max_bm, max_bn = largest[1:3]
                    grid = grids[largest]
                    chosen = (
                        {
                            g
                            for g in full_a
                            if 3 * g[1] >= 2 * max_bm and 2 * g[2] >= max_bn
                        }
                        if grid < cu_grid
                        else {largest, min(full_a, key=lambda g: (costs[g], g))}
                    )
                    if grid < cu_grid:
                        supplied = [g for g in full_a if grids[g] >= cu_grid]
                        if supplied:
                            area = max(g[1] * g[2] for g in supplied)
                            chosen.update(g for g in supplied if g[1] * g[2] == area)
                    waves = wm * wn
                    if native and waves != 4 and grid >= cu_grid * max(1, 4 / waves):
                        secondary.update(chosen)
                    else:
                        primary.update(chosen)
                rotated = [g for g in layout if (bk // 128) * (g[1] // 16) >= 16]
                if rotated:
                    # Cover the changed A live set, LDS waits, and B queue clamp.
                    secondary.add(min(rotated, key=lambda g: (costs[g], g)))
        elif pipeline == "pingpong_lds":
            # Fill the A ring with E_M=3..4; add two-resident-block tiles if needed.
            core = [
                g for g in tiles if g[4] * g[5] == 8 and 3 <= g[1] // (g[4] * 16) <= 4
            ]
            underfilled = True
            if core:
                max_bn = max(g[2] for g in core)
                core = [g for g in core if g[2] == max_bn]
                primary.update(core)
                largest = max(core, key=lambda g: (g[1] * g[2], g))
                underfilled = grids[largest] < cu_grid
            for waves in (4, 8):
                resident = [
                    g for g in tiles if g[4] * g[5] == waves and min_lds[g] <= 80 * 1024
                ]
                if resident:
                    max_area = max(g[1] * g[2] for g in resident)
                    chosen = {g for g in resident if g[1] * g[2] == max_area}
                    (primary if underfilled else secondary).update(chosen)
        else:
            secondary.update(tiles)
        if not (primary | secondary).intersection(tiles):
            secondary.update(ranked)

    # Tighten primary ranges by relative work and grid supply, retaining modes.
    dropped, core_costs, by_layout = set(), {}, defaultdict(list)
    for geometry in primary:
        _, bm, bn, _, wm, wn = geometry
        if geometry[0] == "pingpong_lds" and wm * wn == 8 and bm // (wm * 16) >= 3:
            em, en = bm // (16 * wm), bn // (16 * wn)
            mfma = em * en * 32 * wm * wn / 4
            transfer = (bm + bn) * 128 / 512
            refill = max(0, em - 3) * 64
            body = (k / 128) * (mfma + transfer + refill) + 3072 + bm * bn * 2 / 512
            occupancy = grids[geometry] / cu_grid
            tail = 1 + 0.1 * (math.ceil(occupancy) / occupancy - 1)
            core_costs[geometry] = grids[geometry] * body * tail / (m * n * k)
        elif geometry[0] in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
            by_layout[(geometry[0], *geometry[3:])].append(geometry)
        if grids[geometry] < cu_grid / 2:
            dropped.add(geometry)
    if core_costs:
        best = min(core_costs.values())
        dropped.update(g for g in core_costs if core_costs[g] > 1.025 * best)
    for geometries in by_layout.values():
        best = min(costs[g] for g in geometries)
        margin = 4 / 3 if geometries[0][4] * geometries[0][5] < 4 else 1.1
        dropped.update(g for g in geometries if costs[g] > margin * best)
    primary -= dropped
    secondary = (secondary | dropped) - primary

    for geometry, kids in groups.items():
        mode = "baseline"
        if geometry in primary:
            mode = "primary"
        elif geometry in secondary:
            mode = "secondary"
        variants = _opus_select_variants(
            geometry, kids, profiles, footprints, m, n, mode
        )
        if geometry[0] in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
            signatures = {profiles[kid] for kid in variants}
            variants.update(kid for kid in kids if profiles[kid] in signatures)
        selected.update(variants)
    print(
        f"[OPUS search] shape={(m, n, k)} legal={len(legal)} "
        f"primary={len(primary)} secondary={len(secondary)} "
        f"co={co_count} candidates={len(selected)}",
        flush=True,
    )
    return [(kid, registry[kid]) for kid in sorted(selected)]


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
# Decode weight-streaming mxfp8_32 kernels (flydsl_mxfp8_32_bpreshuffle_stream_*),
# run through the same B = 1 bmm dispatch.
KERNEL_ID_BASE_STREAM = 3_900_000  # inside the bmm id range: rows get kernelId "bmm"
MXSCALE_KERNEL_ID = "mxfp8"
MXSCALE_BMM_KERNEL_ID = "bmm"
_GFX1250_1X32_LAST_KERNEL = None
_GFX1250_1X32_LAST_USED_WORKSPACE = False


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


def gfx1250_1x32_runs(m, n, k):
    """Every legal direct and B=1 batched 1x32 candidate."""
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import (
        bmm_candidates as gfx1250_bmm_candidates,
    )
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import (
        bmm_mx32_kernel_name,
    )
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common import (
        kernels_list,
        mxfp8_32_kernel_fits_shape,
    )

    runs = []
    for kernel_id, ki in kernels_list.items():
        if ki.a_preshuffle or not mxfp8_32_kernel_fits_shape(ki, m, n, k):
            continue
        runs.append(
            (
                kernel_id,
                ki.split_k,
                ki.name.replace("mxfp8_128", "mxfp8_32"),
                run_gemm_flydsl_gfx1250_1x32,
            )
        )

    for index, cfg in enumerate(gfx1250_bmm_candidates(1, m, n, k)):
        runs.append(
            (
                KERNEL_ID_BASE_BMM + index,
                1,
                bmm_mx32_kernel_name(**cfg),
                run_bmm_flydsl,
            )
        )
    from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
        mxfp8_32_stream_candidates,
    )

    for index, name in enumerate(mxfp8_32_stream_candidates(m, n, k)):
        runs.append((KERNEL_ID_BASE_STREAM + index, 1, name, run_bmm_flydsl))
    return list({entry[2]: entry for entry in runs}.values())


def run_gemm_flydsl_gfx1250(x, weight_shuffle, x_scale, w_scale, out, kernel_name):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_mxfp8_128_bpreshuffle_flydsl

    return gemm_a8w8_mxfp8_128_bpreshuffle_flydsl(
        x, weight_shuffle, x_scale, w_scale, out, {"kernelName": kernel_name}
    )


def run_gemm_flydsl_gfx1250_1x32(x, weight_shuffle, x_scale, w_scale, out, kernel_name):
    global _GFX1250_1X32_LAST_KERNEL, _GFX1250_1X32_LAST_USED_WORKSPACE
    from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
        clear_mxfp8_32_workspace_cache,
        parse_mxfp8_32_wmma_kernel_name,
        run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250,
    )

    if _GFX1250_1X32_LAST_KERNEL != kernel_name:
        if _GFX1250_1X32_LAST_USED_WORKSPACE:
            clear_mxfp8_32_workspace_cache()
            gc.collect()
            torch.cuda.empty_cache()
        config = parse_mxfp8_32_wmma_kernel_name(kernel_name)
        _GFX1250_1X32_LAST_USED_WORKSPACE = bool(
            config is not None and config.get("split_k", 1) > 1
        )
        _GFX1250_1X32_LAST_KERNEL = kernel_name
    return run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250(
        x, weight_shuffle, x_scale, w_scale, out, kernel_name
    )


class GemmA8W8MXFP8BpreshuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": TUNED_FILE,
        "untune_file": "aiter/configs/a8w8_mxfp8_bpreshuffle_untuned_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",
        "config_env_name": "AITER_CONFIG_GEMM_A8W8_MXFP8_BPRESHUFFLE",
    }

    def get_arg_defaults(self):
        defaults = super().get_arg_defaults()
        if getattr(self, "_mxscale_table", False):
            defaults.update(
                tune_file=MXSCALE_TUNED_FILE,
                config_env_name="AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
            )
        return defaults

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
            "--confirm-rounds",
            type=int,
            default=7,
            help="Interleaved baseline/candidate Graph timing rounds for gfx1250 1x32.",
        )
        self.parser.add_argument(
            "--min-gain-pct",
            type=float,
            default=2.0,
            help="Minimum median paired gain required to replace a gfx1250 1x32 row.",
        )

    def pre_process(self, args):
        self._mxscale_table = args.mxscale_table
        if self._mxscale_table and args.tune_file == TUNED_FILE:
            args.tune_file = MXSCALE_TUNED_FILE
        if args.libtype == "opus" and not self._mxscale_table:
            self.parser.error("OPUS tuning requires --mxscale_table")
        self._baseline_tunedf = self.get_tuned_gemm_list(args.tune_file)
        super().pre_process(args)

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a8w8 as op

        if getattr(self, "_mxscale_table", False):
            op.get_mxscale_bpreshuffle_config.cache_clear()
            op._load_mxscale_bpreshuffle_tuned.cache_clear()
            op._mxscale_bpreshuffle_tuned_nk.cache_clear()
            op._MXSCALE_UNTUNED_REPORTED.clear()
        else:
            op._get_mxfp8_bpreshuffle_config.cache_clear()
            op._load_mxfp8_bpreshuffle_tuned.cache_clear()
            op._mxfp8_bpreshuffle_tuned_nk.cache_clear()

    def _restore_config_env(self, env_name, old_val, old_rebuild=0):
        super()._restore_config_env(env_name, old_val, old_rebuild)
        from aiter.jit.core import AITER_CONFIGS

        AITER_CONFIGS.get_config_file.cache_clear()
        self._clear_op_caches()

    def _bmm_seeds_mxfp8(self):
        """bmm seeds of a gfx950 1x32 shape in this op's table: ({(N, K):
        [(M, kernelName)]}, [(M, kernelName)]). The first holds the tuned bmm
        rows of this table and of gemm_a8w8_blockscale_bpreshuffle's e8m0 one
        (any w_scale block: a bmm kernelName runs on every block its config
        admits), the second every such row of the e8m0 table, the pool a shape
        without rows of its own N and K draws on."""
        from aiter.jit.core import AITER_CONFIGS

        gfx, cu_num = self.get_gfx(), self.get_cu_num()
        by_nk, pool = {}, []
        for path, is_mxfp8 in (
            (AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_MXFP8_BPRESHUFFLE_FILE, True),
            (
                AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE_FILE,
                False,
            ),
        ):
            try:
                df = pd.read_csv(path)
            except (OSError, pd.errors.EmptyDataError):
                continue
            df = df[
                (df["gfx"] == gfx)
                & (df["cu_num"] == cu_num)
                & (df["kernelId"].astype(str) == MXSCALE_BMM_KERNEL_ID)
            ]
            for r in df.itertuples():
                seed = (int(r.M), str(r.kernelName))
                by_nk.setdefault((int(r.N), int(r.K)), []).append(seed)
                if not is_mxfp8:
                    pool.append(seed)
        return by_nk, pool

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

    @staticmethod
    def _shape_key(info):
        gfx, cu_num, m, n, k, block = info[0]
        return str(gfx), int(cu_num), int(m), int(n), int(k), str(block)

    def _gfx1250_1x32_baselines(self, tunedf, tasks):
        from aiter.ops.gemm_op_a8w8 import _mxfp8_32_fallback_kernel_name

        existing = {}
        required = set(self.keys) | {"kernelName", "us"}
        if required.issubset(tunedf.columns):
            for _, row in tunedf.iterrows():
                key = (
                    str(row["gfx"]),
                    int(row["cu_num"]),
                    int(row["M"]),
                    int(row["N"]),
                    int(row["K"]),
                    str(row["w_scale_block"]),
                )
                existing[key] = (
                    str(row["kernelName"]),
                    float(row["us"]),
                    float(row.get("errRatio", 0.0)),
                )

        tasks_by_shape = defaultdict(dict)
        for task in tasks:
            tasks_by_shape[self._shape_key(task[0])][task[0][3]] = task

        baselines = {}
        for shape, by_name in tasks_by_shape.items():
            baseline = existing.get(shape)
            if baseline is None or baseline[0] not in by_name:
                _, _, m, n, k, _ = shape
                name = _mxfp8_32_fallback_kernel_name(m, n, k)
                baseline = (name, None, None)
            if baseline[0] not in by_name:
                raise RuntimeError(
                    f"gfx1250 1x32 baseline {baseline[0]!r} is not a legal "
                    f"candidate for {shape}"
                )
            baselines[shape] = baseline
        return baselines, tasks_by_shape

    def _confirm_gfx1250_1x32(self, coarse, tasks, tunedf, args):
        baselines, tasks_by_shape = self._gfx1250_1x32_baselines(tunedf, tasks)
        coarse_by_shape = defaultdict(list)
        for result in coarse:
            info, us, err = result
            if math.isfinite(us) and us > 0 and err <= args.errRatio:
                coarse_by_shape[self._shape_key(info)].append(result)

        choices = {}
        confirm_tasks = []
        rounds = max(1, int(args.confirm_rounds))
        for shape, by_name in tasks_by_shape.items():
            baseline_name, stored_us, stored_err = baselines[shape]
            valid = sorted(coarse_by_shape.get(shape, ()), key=lambda result: result[1])
            candidate_name = valid[0][0][3] if valid else baseline_name
            choices[shape] = (baseline_name, candidate_name, stored_us, stored_err)
            for round_index in range(rounds):
                order = (
                    (baseline_name, candidate_name)
                    if round_index % 2 == 0
                    else (candidate_name, baseline_name)
                )
                for name in dict.fromkeys(order):
                    confirm_tasks.append(by_name[name])

        confirmed = self._run_tasks(confirm_tasks, args)
        measured = defaultdict(list)
        for result in confirmed:
            info, us, err = result
            measured[(self._shape_key(info), info[3])].append((float(us), float(err)))

        selected = []
        min_speedup = 1.0 + float(args.min_gain_pct) / 100.0
        for shape, (
            baseline_name,
            candidate_name,
            stored_us,
            stored_err,
        ) in choices.items():
            baseline = measured[(shape, baseline_name)]
            candidate = measured[(shape, candidate_name)]
            if len(baseline) != rounds or len(candidate) != rounds:
                raise RuntimeError(
                    f"incomplete gfx1250 1x32 confirmation for {shape}: "
                    f"baseline={len(baseline)}, candidate={len(candidate)}, "
                    f"expected={rounds}"
                )
            baseline_us = [value[0] for value in baseline]
            candidate_us = [value[0] for value in candidate]
            paired = [
                base / cand
                for base, cand in zip(baseline_us, candidate_us)
                if base > 0 and cand > 0
            ]
            candidate_median = statistics.median(candidate_us)
            keep = (
                candidate_name != baseline_name
                and len(paired) == rounds
                and min(paired) > 1.0
                and statistics.median(paired) >= min_speedup
                and (stored_us is None or stored_us / candidate_median >= min_speedup)
            )
            name = candidate_name if keep else baseline_name
            values = candidate if keep else baseline
            info = tasks_by_shape[shape][name][0]
            selected_us = (
                float(stored_us)
                if not keep and stored_us is not None
                else statistics.median(value[0] for value in values)
            )
            selected_err = (
                float(stored_err)
                if not keep and stored_err is not None
                else max(value[1] for value in values)
            )
            selected.append(
                (
                    info,
                    round(selected_us, 4),
                    selected_err,
                )
            )
        return selected

    def run_config(self, args):
        from aiter.ops.gemm_op_a8w8 import (
            gemm_a8w8_blockscale_bpreshuffle,
            gemm_a8w8_mxfp8_bpreshuffle,
        )
        from aiter.test_common import checkAllclose, run_perftest

        production_op = (
            gemm_a8w8_blockscale_bpreshuffle
            if self._mxscale_table
            else gemm_a8w8_mxfp8_bpreshuffle
        )
        results = []
        for _, row in self.untunedf.iterrows():
            m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
            block = row.get("w_scale_block", W_SCALE_BLOCK)
            shape = f"({m}, {n}, {k}, {block})"
            allowed, allowed_desc = self._get_run_config_err_ratio_limit(row, args)
            try:
                if not self._mxscale_table and block != W_SCALE_BLOCK:
                    raise ValueError(
                        f"{block} production validation requires --mxscale_table"
                    )
                data = generate_data(m, n, k, 0, block)
                if self.get_gfx() == "gfx1250" and block == W_SCALE_BLOCK:
                    from aiter.ops.shuffle import shuffle_mxfp8fp4_scale

                    data["w_scale"] = shuffle_mxfp8fp4_scale(data["w_scale"])
                output, us = run_perftest(
                    production_op,
                    data["x"],
                    data["weight_shuffle"],
                    data["x_scale"],
                    data["w_scale"],
                    out=data["out"],
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                    testGraph=(self.get_gfx() == "gfx1250" and block == W_SCALE_BLOCK),
                    use_cuda_event=(
                        self.get_gfx() == "gfx1250" and block == W_SCALE_BLOCK
                    ),
                    num_rotate_args=1,
                )
                if not math.isfinite(us) or us <= 0:
                    raise RuntimeError(f"invalid production timing {us}")
                err = checkAllclose(
                    run_torch(data["x_deq"], data["w_deq"]),
                    output,
                    msg=f"run_config {shape}",
                    catastrophic_check=True,
                )
                status = (
                    "ok"
                    if err <= allowed
                    else f"mismatch:err_ratio={err:.6g}(>{allowed_desc})"
                )
                results.append({"shape": shape, "e2e_us": us, "status": status})
            except Exception as error:  # noqa: BLE001
                results.append(
                    {"shape": shape, "e2e_us": -1, "status": f"error:{error}"}
                )
            finally:
                torch.cuda.empty_cache()
        return results

    def tune(self, untunedf, tunedf, args):
        gfx = self.get_gfx()
        if gfx not in ("gfx950", "gfx1250"):
            logger.warning(
                f"gemm_a8w8_mxfp8_bpreshuffle tuner covers gfx950 and gfx1250, got {gfx}"
            )
            return []
        if args.libtype == "opus" and gfx != "gfx950":
            raise ValueError("OPUS bpreshuffle tuning supports gfx950 only")
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
        blocks = set(untunedf["w_scale_block"])
        self._gfx1250_1x32 = (
            gfx == "gfx1250" and not self._mxscale_table and blocks == {W_SCALE_BLOCK}
        )
        gfx1250_128x128 = (
            gfx == "gfx1250" and self._mxscale_table and blocks == {"128x128"}
        )
        if gfx == "gfx1250" and not (self._gfx1250_1x32 or gfx1250_128x128):
            raise ValueError(
                "gfx1250 supports this op's 1x32 table, or the mxfp8_128 kernels "
                "of gemm_a8w8_blockscale_bpreshuffle with --mxscale_table and "
                "128x128 shapes"
            )
        seeds = (
            self._bmm_seeds() if self._mxscale_table and args.libtype != "opus" else {}
        )
        # This op's gfx950 1x32 table: bmm joins the mxpsh / 8-wave kernels too.
        self._bmm_1x32 = gfx == "gfx950" and not self._mxscale_table
        seeds_nk, seed_pool = self._bmm_seeds_mxfp8() if self._bmm_1x32 else ({}, [])
        opus_by_shape = {}
        if gfx == "gfx950" and args.libtype != "flydsl" and self._mxscale_table:
            if args.libtype == "opus" and set(untunedf.w_scale_block) != {"128x128"}:
                raise ValueError("OPUS supports A 1x128 / B 128x128 scales only")
            for row in untunedf.itertuples():
                if row.w_scale_block == "128x128":
                    shape = (int(row.M), int(row.N), int(row.K))
                    opus_by_shape[shape] = opus_candidates(*shape, cu_num=cu_num)
                    if args.libtype == "opus" and not opus_by_shape[shape]:
                        raise ValueError(f"No OPUS candidate supports {shape}")
        for i in range(len(untunedf)):
            M = int(untunedf.loc[i, "M"])
            N = int(untunedf.loc[i, "N"])
            K = int(untunedf.loc[i, "K"])
            block = str(untunedf.loc[i, "w_scale_block"])
            info_keys = (gfx, cu_num, M, N, K, block)
            opus = opus_by_shape.get((M, N, K), ()) if block == "128x128" else ()
            data_generator = generate_data
            data_args = (M, N, K, seed, block)
            candidate_run_kwargs = dict(run_kwargs)
            if self._gfx1250_1x32:
                runs = gfx1250_1x32_runs(M, N, K)
                data_generator = generate_data_gfx1250_1x32
                data_args = (M, N, K, seed)
                candidate_run_kwargs.update(
                    testGraph=True, use_cuda_event=True, num_rotate_args=1
                )
            elif gfx == "gfx1250":
                cands = gfx1250_candidates(M, N, K)
                runs = [
                    (kernel_id, ki.split_k, ki.name, run_gemm_flydsl_gfx1250)
                    for kernel_id, ki in cands
                ]
            else:
                cands = [
                    (kernel_id, ki)
                    for pipe in PIPELINES_BY_W_SCALE_BLOCK[block]
                    for kernel_id, ki in pipe.kernels_list.items()
                    if args.libtype != "opus" and pipe.fits(ki, M, N, K)
                ]
                runs = [
                    (kernel_id, ki.split_k, ki.name, run_gemm_flydsl)
                    for kernel_id, ki in cands
                ]
            near = None
            if self._mxscale_table and gfx == "gfx950" and args.libtype != "opus":
                near = [
                    name
                    for mm, name in seeds.get((N, K, block), ())
                    if M / 2 <= mm <= M * 2
                ]
            elif self._bmm_1x32:
                near = [
                    name
                    for mm, name in seeds_nk.get((N, K), ())
                    if M / 2 <= mm <= M * 2
                ] or sorted({name for mm, name in seed_pool if M / 2 <= mm <= M * 2})
            if near is not None:
                for j, name in enumerate(bmm_candidates(M, N, K, block, near)):
                    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (
                        parse_bmm_kernel_name,
                    )

                    splits = parse_bmm_kernel_name(name)["splits"]
                    runs.append((KERNEL_ID_BASE_BMM + j, splits, name, run_bmm_flydsl))
            runs.extend(
                (kid, 1, instance.name, run_gemm_opus) for kid, instance in opus
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
                        data_generator,
                        data_args,
                        run,
                        (gemm_keys, kernel_id) if is_opus else (gemm_keys, name),
                        candidate_run_kwargs,
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
        results = self._run_tasks(task, args)
        if self._gfx1250_1x32:
            return self._confirm_gfx1250_1x32(
                results, task, getattr(self, "_baseline_tunedf", tunedf), args
            )
        return results

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
            args.shape_grouped or self._gfx1250_1x32,
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
            elif libtype == "flydsl" and (
                keys[0] == "gfx1250"
                or (
                    keys[0] == "gfx950"
                    and (
                        getattr(self, "_mxscale_table", False)
                        or getattr(self, "_bmm_1x32", False)
                    )
                )
            ):
                # bmm's ids sit between fsk's and the _pf ones. Direct gfx1250
                # rows keep their stable kernels_list / dedicated-small-M ID.
                is_bmm = KERNEL_ID_BASE_BMM <= kernelId < KERNEL_ID_BASE_PF
                if is_bmm:
                    kernelId = MXSCALE_BMM_KERNEL_ID
                elif keys[0] == "gfx950" and self._mxscale_table:
                    kernelId = MXSCALE_KERNEL_ID
                # This op's table keeps the mxpsh / 8-wave kernelId.
            tflops, bw = self.calculate(el)
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
