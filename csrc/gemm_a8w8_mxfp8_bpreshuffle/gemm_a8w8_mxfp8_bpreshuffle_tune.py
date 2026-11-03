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

OPUS calculates a shape/resource range, benchmarks it once, and selects the
fastest valid result. Legal COs and low-LDS baselines in the selected JIT range
are retained; small tiles cover resource and execution axes without expanding
their full geometry/Split-K cross-product.
"""

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


_OPUS_B_DEFAULTS = (None,) * 4 + tuple(
    map(str, (0, "false", 7168, 4, 1, 0, 2, 0, 256, "false", "false", "false", 0))
)
_OPUS_M_REPEAT_STAGE_DEFAULTS = (None,) * 7 + ("1", "false", "0", "false")
_OPUS_TRAIT_DEFAULTS = {
    "b_vgpr": _OPUS_B_DEFAULTS,
    "b_vgpr_overlap": (*_OPUS_B_DEFAULTS, "0"),
    "m_repeat": tuple(
        map(str, (192, 256, 128, 0, 4, 0, 0, 0, "false", "false", "false", 0, 8))
    ),
    "wave_interleave": tuple(map(str, (256, 0, 4, "true", 0, 0, 4))),
    "b_vgpr_panel": (*_OPUS_B_DEFAULTS[:16], "false", "1", "false"),
    "m_repeat_scale": tuple(
        map(
            str,
            (192, 256, 128, 0, 4, 0, 0, 0, "false", "false", "false", "false", "false"),
        )
    ),
    "operand": (
        (None,) * 8 + ("0",),
        tuple(
            map(
                str,
                (
                    "true",
                    "true",
                    "false",
                    1,
                    0,
                    "true",
                    "true",
                    "true",
                    "false",
                    3,
                    0,
                    "false",
                ),
            )
        ),
    ),
    "m_repeat_stage": _OPUS_M_REPEAT_STAGE_DEFAULTS,
}
_OPUS_TRAIT_WRAPPERS = {
    "b_vgpr_panel": (
        "opus_gemm_bpreshuffle_bounded_thread_traits",
        "opus_gemm_bpreshuffle_simple_tail_traits",
        "opus_gemm_bpreshuffle_native_c_traits",
    ),
    "operand": ("opus_gemm_bpreshuffle_mfirst_traits",),
}
_OPUS_SMALL_PIPELINES = frozenset(("b_vgpr_panel", "m_repeat_scale", "operand", "m_repeat_stage"))


# Small-M resource limits used by the shared candidate workflow.
_OPUS_SMALL_PADDING_RATIO = 2
_OPUS_SMALL_TILES_PER_LAYOUT = 4
_OPUS_SMALL_RESIDENT_LDS = 80 * 1024
_OPUS_SMALL_TILE_COST_RATIO = 4 / 3
_OPUS_SMALL_PADDED_COST_RATIO = 1.75
_OPUS_SMALL_PRIMARY_COST_RATIO = 1.1
_OPUS_SMALL_LIVE_REGISTERS = {"b_vgpr_panel": 384, "operand": 288}


@lru_cache(maxsize=65536)
def _opus_kernel_profile(geometry, traits, wg, threads):
    """Return effective traits, fixed LDS bytes, and A-scale LDS alignment.

    Shape-dependent A-scale storage is added by opus_candidates. Cache keys
    contain immutable kernel metadata, so this work is reused across shapes.
    """
    pipeline, bm, bn, bk, wm, wn, *_, split = geometry
    original_traits, wrappers = traits, []
    while wrapper := re.fullmatch(
        r"\s*(opus_gemm_bpreshuffle_(?:bounded_thread|simple_tail|native_c)_traits"
        r"|opus_gemm_bpreshuffle_mfirst_traits)\s*<(.+)>\s*",
        traits,
        re.DOTALL,
    ):
        wrappers.append(wrapper[1])
        traits = wrapper[2]
    if pipeline not in _OPUS_TRAIT_DEFAULTS or any(
        wrapper not in _OPUS_TRAIT_WRAPPERS.get(pipeline, ()) for wrapper in wrappers
    ):
        raise ValueError(f"Unsupported OPUS traits: {original_traits}")
    if pipeline == "operand":
        match = re.fullmatch(
            r"\s*opus_gemm_bpreshuffle_(?:m_repeat|b_vgpr)_operand_traits\s*<\s*"
            r"opus_gemm_bpreshuffle_stage_geometry\s*<([^<>]*)>\s*,\s*"
            r"opus_gemm_bpreshuffle_stage_policy\s*<([^<>]*)>\s*>\s*",
            traits,
        )
        defaults = _OPUS_TRAIT_DEFAULTS[pipeline]
    else:
        match = re.fullmatch(r"\s*\w+\s*<([^<>]*)>\s*", traits)
        defaults = (_OPUS_TRAIT_DEFAULTS[pipeline],)
    if not match:
        raise ValueError(f"Unsupported OPUS traits: {original_traits}")
    profile, scale_alignment = [], 0
    for body, default in zip(match.groups(), defaults):
        body = body.strip()
        args = tuple(arg.strip().lower() for arg in body.split(",")) if body else ()
        args += default[len(args) :]
        if len(args) != len(default) or any(
            arg is None or not re.fullmatch(r"-?\d+|true|false", arg) for arg in args
        ):
            raise ValueError(f"Incomplete OPUS traits: {original_traits}")
        profile.extend(args)
    if pipeline in ("b_vgpr", "b_vgpr_overlap"):
        lds = int(profile[2]) * bm * bk + threads * 4
        scale_alignment = threads * 16
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
    elif pipeline == "m_repeat":
        packs = math.ceil((bm // (wm * 16)) / 4)
        lds = 264 * (bm + bn) + int(profile[2]) * (packs * wm * 64 + 4)
    elif pipeline == "wave_interleave":
        rows = threads // 8
        a_slot = math.ceil((bm // 2) / rows) * rows * 128
        lds = 2 * (2 * a_slot + 2 * 128 * 128) + 127 * 256 + 128 * 2
    elif pipeline == "b_vgpr_panel":
        # Whole-panel B lookahead is BD itself, unlike the fragment queue above.
        scale_cols = int(profile[6]) // (128 * split)
        scale_rows = math.ceil(bm / 16) * 16
        byte_scale = profile[18] == "true"
        copy_elems = threads * (1 if byte_scale else 16)
        scale_bytes = (
            math.ceil(scale_rows * scale_cols / copy_elems)
            * threads
            * (4 if byte_scale else 16)
        )
        lds = int(profile[2]) * bm * bk + scale_bytes + threads * 4
    elif pipeline == "m_repeat_scale":
        # Raw panels retain one byte per M row, rather than packed A-scale rows.
        lds = 264 * (bm + bn) + int(profile[2]) * (bm + 4)
    elif pipeline == "operand":
        stages, total_k = int(profile[5]), int(profile[6])
        compact, per_stage, direct_b = (
            profile[index] == "true" for index in (9, 10, 11)
        )
        dma = threads * 16
        a_bytes = math.ceil(bm * bk / dma) * dma
        b_bytes = 0 if direct_b else math.ceil(bn * bk / dma) * dma
        scale_cols = bk // 128 if per_stage else total_k // (128 * split)
        scale_copy = threads * (4 if compact else 1)
        sa_bytes = math.ceil(bm * scale_cols / scale_copy) * threads * 4
        sb_bytes = math.ceil(math.ceil(bn / 128) * scale_cols / threads) * threads * 4
        scale_bytes = sa_bytes + sb_bytes
        stage_bytes = a_bytes + b_bytes + (scale_bytes if per_stage else 0)
        main_bytes = stages * stage_bytes + (0 if per_stage else scale_bytes)
        lds = max(main_bytes, bm * (bn + 8) * 2)
        # These parameters do not change the selected execution path. Keep
        # their original template arguments in the registry, but deduplicate
        # the effective profile before covering variants.
        if not direct_b:
            profile[12] = "0"
        if int(profile[18]) < 2:
            profile[18] = "0"
        if split > 1 or "opus_gemm_bpreshuffle_mfirst_traits" in wrappers:
            profile[20] = "false"
    else:
        stages, total_k = int(profile[4]), int(profile[5])
        scale_cols = total_k // (128 * split)
        scale_copy = threads * (4 if profile[8] == "true" else 1)
        sa_bytes = math.ceil(bm * scale_cols / scale_copy) * threads * 4
        lds = stages * (bm + bn) * bk + sa_bytes + threads * 4
    if pipeline in _OPUS_TRAIT_WRAPPERS:
        # Wrappers change thread bounds, C storage, or traversal independently.
        profile.append(tuple(wrappers))
    return tuple(profile), lds, scale_alignment


def _opus_select_variants(
    geometry, kids, kernel_profiles, footprints, m, n, mode, *, small_m=False
):
    """Cover variants for both shape ranges, preserving their policy limits.

    Small-M profiles include launch metadata so distinct launch routes remain
    visible. Its primary native routes retain complete policies; all other
    routes share the baseline and secondary trait-coverage implementation.
    """
    pipeline = geometry[0]
    if small_m and mode == "primary" and pipeline in _OPUS_SMALL_PIPELINES:
        unique = {}
        for kid in sorted(kids):
            unique.setdefault(kernel_profiles[kid], kid)
        if pipeline not in ("b_vgpr_panel", "operand"):
            return set(unique.values())
        selected, representatives = set(), {}
        for kid in unique.values():
            profile, launch = kernel_profiles[kid]
            if pipeline == "b_vgpr_panel":
                main = (profile[13], profile[5], profile[16]) == (
                    "true",
                    "false",
                    "true",
                )
                route = (
                    launch,
                    profile[5],
                    profile[13],
                    profile[16],
                    profile[11],
                    profile[19],
                )
                indices = ((2, 3, 8), (4, 9, 18), (10, 14, 15))
            else:
                main = (profile[14], profile[15], profile[18]) == ("true", "true", "3")
                route = launch, profile[14], profile[15], profile[18], profile[21]
                indices = ((5, 9, 10, 11, 12), (9, 10, 11, 16, 13, 19), (8, 17, 20))
            if main:
                selected.add(kid)
                continue
            resource = footprints[kid]
            for positions in indices:
                key = route, positions, tuple(profile[index] for index in positions)
                previous = representatives.get(key)
                if previous is None or (resource, kid) < previous:
                    representatives[key] = resource, kid
        selected.update(kid for _, kid in representatives.values())
        return selected

    launch_profiles = kernel_profiles if small_m else None
    if small_m:
        kernel_profiles = {kid: kernel_profiles[kid][0] for kid in kids}
    small_primary = small_m and mode == "primary"
    if small_primary:
        mode = "secondary"
    selected = {min(kids, key=lambda kid: (footprints[kid], kid))}
    if mode != "baseline":
        pipeline, bm, bn, bk, wm, wn = geometry[:6]
        unique = {}
        for kid in sorted(kids):
            profile = kernel_profiles[kid]
            if pipeline == "m_repeat" and (
                int(profile[5]) > 0 or n // bn < 256 or m <= bm
            ):
                # GroupM overrides GroupN; narrow-N and single-M-tile routes ignore it.
                profile = (*profile[:12], "inactive")
            unique.setdefault(profile, kid)
        profiles = {kid: profile for profile, kid in unique.items()}
        if mode == "primary" and pipeline in (
            "b_vgpr",
            "b_vgpr_overlap",
            "m_repeat",
            "wave_interleave",
        ):
            routes = list(profiles)
            if pipeline in ("b_vgpr", "b_vgpr_overlap"):
                routes = [kid for kid in routes if profiles[kid][11] == "0"]
                if routes:
                    aheads = {kid: int(profiles[kid][3][5:]) for kid in routes}
                    # Retain the queue level nearest to or above the full-A live set.
                    target = max((bk // 128) * (bm // 16) - 1, 1)
                    bound = min(
                        (ahead for ahead in aheads.values() if ahead >= target),
                        default=max(aheads.values()),
                    )
                    routes = [kid for kid in routes if aheads[kid] <= bound]
            elif pipeline in ("m_repeat", "wave_interleave"):
                if pipeline == "m_repeat":
                    em = bm // (wm * 16)
                    if not (wm * wn == 8 and 3 <= em <= 4):
                        # Include each variant's scale panel in two-block residency.
                        resident = {kid for kid in kids if footprints[kid] <= 80 * 1024}
                        if resident:
                            routes = [kid for kid in routes if kid in resident]
                    group_index = 5
                else:
                    group_index = 2
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
        elif pipeline == "b_vgpr_overlap":
            publications = sorted({int(profile[17]) for profile in profiles.values()})
            if len(publications) > 3:
                midpoint = (publications[0] + publications[-1]) / 2
                middle = min(publications, key=lambda value: abs(value - midpoint))
                retained = {publications[0], middle, publications[-1]}
                profiles = {
                    kid: profile
                    for kid, profile in profiles.items()
                    if int(profile[17]) in retained
                }

        # Cover individual trait values and resource/execution-route combinations.
        indices = [(index,) for index in range(len(next(iter(profiles.values()))))]
        if pipeline in ("b_vgpr", "b_vgpr_overlap"):
            # stage/ahead/WG, A/B cache, store/output, contiguous/pins, XCD/groupM
            indices += [(2, 3, 8), (4, 9), (10, 11), (13, 14, 15), (5, 16)]
            if pipeline == "b_vgpr_overlap":
                indices += [(2, 17), (11, 17)]
        elif pipeline == "m_repeat":
            # panel/fixedK, fragments/cache/guard, launch groups
            indices += [(2, 3), (3, 8), (6, 7, 8, 10), (5, 12), (5, 9)]
        elif pipeline == "wave_interleave":
            indices += [(2, 3), (4, 5)]
        elif pipeline == "b_vgpr_panel":
            # Panel/stages/WG, joint WG/cache/scale policies, traversal, output/wrapper.
            indices += [(13, 16, 18), (11, 18), (11, 19)]
            if mode == "primary":
                indices += [(2, 3, 8), (8, 9, 4, 18), (5, 16)]
        elif pipeline == "m_repeat_scale":
            indices += [(5, 12)]
            if mode == "primary":
                indices += [(2, 3), (3, 8), (6, 7, 8)]
        elif pipeline == "operand":
            # LDS/ring loading, scale/MFMA layout, cache/output, address/traversal.
            indices += [(10, 11), (11, 16), (17, 19, 21)]
            if mode == "primary":
                indices += [
                    (5, 10, 11, 12),
                    (9, 10, 11, 16),
                    (13, 18),
                    (14, 15, 19),
                    (20, 21),
                ]
        elif pipeline == "m_repeat_stage":
            indices += [(8, 9)]
            if mode == "primary":
                indices += [(4, 8)]
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
    if small_m:
        if small_primary:
            if pipeline == "b_vgpr":
                # Keep the full-A direct-store route beyond the large-M clamp.
                selected.update(kid for kid in kids if kernel_profiles[kid][11] == "0")
            elif pipeline == "m_repeat":
                # Preserve complete ungrouped panel policies, including GroupM.
                selected.update(kid for kid in kids if kernel_profiles[kid][5] == "0")
        covered = {kernel_profiles[kid] for kid in selected}
        unique = {}
        for kid in sorted(kids):
            if kernel_profiles[kid] in covered:
                unique.setdefault(launch_profiles[kid], kid)
        selected.update(unique.values())
    return selected


def opus_candidates(m, n, k, *, cu_num, small_m=None):
    """Select candidates through shared metadata, coverage and output stages.

    M <= 1024 uses the small-M resource range by default. An explicit small_m
    selects a range independently of M, including CPU checks of either range.
    Both tile ranges are applied here, retaining their rules and legal COs.
    """
    small_m = m <= 1024 if small_m is None else small_m
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
        entry = instance.bpreshuffle_entry
        pipeline = entry["pipeline"]
        if entry.get(f"{pipeline}_schedule") == "operand":
            # Both families share the same operand geometry/resource coverage.
            pipeline = "operand"
        elif pipeline == "b_vgpr":
            schedule = entry.get("b_vgpr_schedule", "fragment")
            pipeline = "b_vgpr" if schedule == "fragment" else "b_vgpr_" + schedule
        elif pipeline == "m_repeat":
            pipeline = ("m_repeat_stage" if entry.get("m_repeat_schedule") == "staged"
                        else "m_repeat_scale" if entry.get("scale_on_read") else "m_repeat")
        geometry = (
            pipeline,
            instance.B_M,
            instance.B_N,
            instance.B_K,
            instance.T_M,
            instance.T_N,
            # Loop and publication modes keep their existing resource coverage.
            *(
                (instance.bpreshuffle_entry["loop"], instance.bpreshuffle_split_k > 1)
                if pipeline == "m_repeat_stage"
                else ()
            ),
            instance.bpreshuffle_split_k,
        )
        profile, lds, alignment = _opus_kernel_profile(
            geometry,
            instance.bpreshuffle_entry["traits"],
            instance.WG_PER_CU,
            instance.BLOCK_SIZE,
        )
        if alignment:
            lds += math.ceil(instance.B_M * (k // 128) / alignment) * alignment
        launch = (
            instance.WG_PER_CU,
            instance.BLOCK_SIZE,
            entry.get("kernel"),
            entry.get("host_kind"),
            entry.get("grid_kind"),
            tuple(entry.get("workspace_rule", ())),
            entry.get("workspace_zero_init"),
            entry.get("reduce_kernel"),
            tuple(entry.get("headers", ())),
            tuple(entry.get("host_traits_headers", ())),
            getattr(instance, "m_align", None),
            getattr(instance, "max_k", None),
            getattr(instance, "bpreshuffle_fixed_k", None),
            getattr(instance, "bpreshuffle_pad_n", None),
            getattr(instance, "max_tensor_bytes", None),
        )
        profiles[kid] = profile, launch
        footprints[kid] = lds
        groups[geometry].append(kid)

    # Compute the same grid, padding and relative tile cost for both ranges.
    cu_grid = max(cu_num, 1)
    grids, paddings, costs = {}, {}, {}
    for geometry in groups:
        bm, bn, split = geometry[1], geometry[2], geometry[-1]
        grid = math.ceil(m / bm) * math.ceil(n / bn) * split
        padding = math.ceil(m / bm) * bm / m * math.ceil(n / bn) * bn / n
        occupancy = grid / cu_grid
        grids[geometry], paddings[geometry] = grid, padding
        costs[geometry] = (
            0.25 * (padding - 1)
            + 0.25 * max(0, 1 - occupancy)
            + 0.1 * (math.ceil(occupancy) / occupancy - 1)
            + 8 / bm
            + 8 / bn
        )
    retained = set()
    if small_m:
        # Small-M ranges bound fragments, grid supply and execution resources.
        tile_resources, layouts = {}, defaultdict(list)
        for geometry in groups:
            pipeline, bm, bn, bk, wm, wn, *_, split = geometry
            em, en = bm // (16 * wm), bn // (16 * wn)
            tile_resources[geometry] = (
                em,
                en,
                k // (split * bk),
                em * en * (bk // 128),
                4 * em * en,
                grids[geometry] * min(wm * wn, 4),
            )
            layouts[(pipeline, *geometry[3:])].append(geometry)
        tile_modes = {}
        for geometries in layouts.values():
            minimum = min(paddings[g] for g in geometries)
            ranked = sorted(
                (
                    g
                    for g in geometries
                    if paddings[g] <= _OPUS_SMALL_PADDING_RATIO * minimum
                ),
                key=lambda g: (costs[g], g),
            )
            for geometry in ranked[:_OPUS_SMALL_TILES_PER_LAYOUT]:
                pipeline, bm, _, bk, wm, wn = geometry[:6]
                native = (
                    pipeline == "b_vgpr"
                    and wm * wn <= 4
                    and (bk // 128) * (bm // 16) < 16
                ) or (
                    pipeline == "m_repeat"
                    and wm * wn == 8
                    and 3 <= bm // (wm * 16) <= 4
                )
                tile_modes[geometry] = (
                    "primary"
                    if pipeline in _OPUS_SMALL_PIPELINES or native
                    else "secondary"
                )

        def narrow(current, allowed, prefer=None):
            # Fallbacks stay in the preceding pool. A stricter range can prefer
            # its preceding resource range before ranking LDS within the family.
            result, fallbacks = {}, {}
            for geometry, kids in current.items():
                eligible = {kid for kid in kids if allowed(geometry, kid)}
                if eligible:
                    result[geometry] = eligible
                else:
                    fallback = min(
                        (
                            0 if prefer is None else int(not prefer(geometry, kid)),
                            footprints[kid],
                            kid,
                            geometry,
                        )
                        for kid in kids
                    )
                    family = (geometry[0], *geometry[6:-1])
                    previous = fallbacks.get(family)
                    if previous is None or fallback < previous:
                        fallbacks[family] = fallback
            available = {(geometry[0], *geometry[6:-1]) for geometry in result}
            for family, (_, _, kid, geometry) in fallbacks.items():
                if family not in available:
                    result.setdefault(geometry, set()).add(kid)
            return result

        pools = {
            geometry: _opus_select_variants(
                geometry,
                groups[geometry],
                profiles,
                footprints,
                m,
                n,
                tile_modes[geometry],
                small_m=True,
            )
            for geometry in sorted(tile_modes)
        }

        def family_eligible(geometry, kid):
            pipeline, _, _, _, wm, wn, *_, split = geometry
            if pipeline not in _OPUS_SMALL_PIPELINES:
                return True
            em, en, *_ = tile_resources[geometry]
            profile = profiles[kid][0]
            if split > 1 and grids[geometry] > 2 * cu_grid * max(1, 4 / (wm * wn)):
                return False
            if pipeline == "b_vgpr_panel":
                return (
                    wn >= 2
                    and em >= 2
                    and (profile[13], profile[5], profile[16])
                    == ("true", "false", "true")
                )
            if pipeline == "operand":
                return (
                    en <= 2 and (wm == 1 or (split > 1 and em <= 4))
                    if profile[11] == "true"
                    else em <= 4 and en <= 4
                )
            if pipeline == "m_repeat_stage":
                return em <= 4 and en <= 2 and (wm, wn) in ((1, 1), (1, 2), (2, 2))
            return profile[5] == "0"  # Ungrouped LDS panels.

        def resident_eligible(geometry, kid):
            return family_eligible(geometry, kid) and (
                geometry[0] not in ("b_vgpr_panel", "operand")
                or grids[geometry] <= 2 * cu_grid
                or footprints[kid] <= _OPUS_SMALL_RESIDENT_LDS
            )

        pools = narrow(pools, resident_eligible, prefer=family_eligible)
        stage_counts = {
            kid: int(
                profiles[kid][0][{"b_vgpr_panel": 2, "operand": 5}.get(geometry[0], 4)]
            )
            for geometry, kids in pools.items()
            if geometry[0] in _OPUS_SMALL_PIPELINES and geometry[0] != "m_repeat_scale"
            for kid in kids
        }
        min_panels, live_registers = {}, {}
        for geometry, kids in pools.items():
            pipeline, _, _, bk, *_ = geometry
            if pipeline == "m_repeat_scale":
                min_panels[geometry] = min(int(profiles[kid][0][2]) for kid in kids)
            elif pipeline in ("b_vgpr_panel", "operand"):
                em, en, _, _, c_registers, _ = tile_resources[geometry]
                for kid in kids:
                    ahead = int(profiles[kid][0][3 if pipeline == "b_vgpr_panel" else 12])
                    live_registers[kid] = (
                        c_registers + 8 * (bk // 128) * (em + (ahead + 1) * en) + 48
                    )

        def resource_eligible(geometry, kid):
            pipeline, bm, _, _, wm, wn, *_, split = geometry
            grid = grids[geometry]
            waves = wm * wn
            em, en, panels, work, c_registers, supply = tile_resources[geometry]
            profile, _ = profiles[kid]
            if pipeline == "m_repeat_stage":
                # Resident/streaming rings cover grids that cannot sustain the
                # bulk operand paths; avoid repeating them across many CU grids.
                return grid <= 2 * cu_grid
            if pipeline == "m_repeat_scale":
                return int(profile[2]) == min_panels[geometry]
            if pipeline not in ("b_vgpr_panel", "operand"):
                return True

            stages = stage_counts[kid]
            ahead = int(profile[3 if pipeline == "b_vgpr_panel" else 12])
            if live_registers[kid] > _OPUS_SMALL_LIVE_REGISTERS[pipeline]:
                return False
            if pipeline == "b_vgpr_panel":
                if grid > cu_grid and footprints[kid] > _OPUS_SMALL_RESIDENT_LDS:
                    return False
                # A steady whole-panel B ring needs an additional A stage and
                # enough matrix work and wave supply to amortize its refill.
                if panels > 4 and (stages <= ahead + 1 or work < 4):
                    return False
                if supply < 2 * cu_grid:
                    return False
                if grid > (4 if panels <= 4 else 2) * cu_grid:
                    return False
                # Packed BF16 stores keep a substantial C tile and packed scale
                # copies, bounding simultaneous scale expansion and C packing.
                return profile[11] != "2" or (
                    split == 1 and c_registers >= 32 and profile[18] == "false"
                )

            direct_b, per_stage, normal = (
                profile[index] == "true" for index in (11, 10, 16)
            )
            if wn < 2 or grid > 2 * cu_grid:
                return False
            if supply < (2 * cu_grid if waves >= 4 else cu_grid / 2):
                return False
            # Normal MFMA uses scalar C stores. Keep its smaller accumulator
            # banks on underfilled grids, including a padded single-M tile.
            if normal and (c_registers > 32 or (grid > cu_grid and m >= bm)):
                return False
            if direct_b:
                if panels > 4 and work < 8 and stages <= ahead + 1:
                    return False
                if wn <= 2 and split == 1 and work < 8:
                    return False
            else:
                # Skinny shared-A/B tiles and deep rings favor underfilled grids
                # or small accumulator banks instead of duplicating bulk routes.
                if min(em, en) == 1 and grid > cu_grid and m >= bm:
                    return False
                if stages > 4 and c_registers > 24:
                    return False
            # FP32 Split-K partials favor vector C stores and resident scales.
            if split > 1 and (normal or per_stage):
                return False
            # Long loops need enough queued MFMA work to cover operand latency;
            # short loops may spend most of their time filling/draining the ring.
            return panels <= 2 * stages or (stages - 1) * work >= 8

        pools = narrow(pools, resource_eligible)

        # Select one useful Split-K level per tile and prefetch depth. A partition
        # must contain a full ring before extra splits can improve steady supply.
        split_levels = defaultdict(set)
        for geometry, kids in pools.items():
            pipeline, _, _, bk, _, _, *_, split = geometry
            if pipeline not in _OPUS_SMALL_PIPELINES or pipeline == "m_repeat_scale":
                continue
            for kid in kids:
                stages = stage_counts[kid]
                if split == 1 or k // (split * bk) >= stages:
                    split_levels[(geometry[:-1], stages)].add(split)
        split_targets = {}
        for (base, stages), levels in split_levels.items():
            supply = (
                math.ceil(m / base[1])
                * math.ceil(n / base[2])
                * min(base[4] * base[5], 4)
            )
            split_targets[(base, stages)] = min(
                (split for split in levels if split * supply >= 2 * cu_grid),
                default=max(levels),
            )

        pools = narrow(
            pools,
            lambda geometry, kid: geometry[-1] == 1
            or geometry[0] not in _OPUS_SMALL_PIPELINES
            or geometry[0] == "m_repeat_scale"
            or geometry[-1] == split_targets.get((geometry[:-1], stage_counts[kid]), 1),
        )

        # Rank the surviving layouts once for both range and policy coverage.
        by_layout = defaultdict(list)
        for geometry in pools:
            by_layout[(geometry[0], *geometry[3:])].append(geometry)
        best_costs = {
            layout: min(costs[geometry] for geometry in geometries)
            for layout, geometries in by_layout.items()
        }
        pools = {
            geometry: kids
            for geometry, kids in pools.items()
            if costs[geometry]
            <= (
                _OPUS_SMALL_PADDED_COST_RATIO
                if m < geometry[1]
                else _OPUS_SMALL_TILE_COST_RATIO
            )
            * best_costs[(geometry[0], *geometry[3:])]
        }
        best_reuse = {
            layout: min(1 / g[1] + 1 / g[2] for g in geometries if g in pools)
            for layout, geometries in by_layout.items()
        }
        variant_modes, covered = {}, {}
        for geometry, kids in pools.items():
            pipeline, bm, bn, bk, wm, wn, *_, split = geometry
            supply = tile_resources[geometry][5]
            mode = "primary"
            if pipeline in ("b_vgpr_panel", "operand"):
                layout = pipeline, *geometry[3:]
                if not (
                    k // (split * bk) <= 4
                    or costs[geometry]
                    <= _OPUS_SMALL_PRIMARY_COST_RATIO * best_costs[layout]
                    or 1 / bm + 1 / bn <= best_reuse[layout]
                ) or (pipeline == "b_vgpr_panel" and m < bm):
                    mode = "secondary"
            elif pipeline == "b_vgpr":
                if not (
                    wm * wn <= 4
                    and (bk // 128) * (bm // 16) < 16
                    and m >= bm
                    and supply >= 2 * cu_grid
                ):
                    mode = "baseline"
            elif pipeline in ("m_repeat_scale", "m_repeat"):
                if not (
                    wm * wn == 8 and 3 <= bm // (wm * 16) <= 4 and supply >= 2 * cu_grid
                ):
                    mode = "baseline"
            elif pipeline in ("b_vgpr_overlap", "wave_interleave"):
                mode = "baseline"
            variant_modes[geometry] = mode
            covered[geometry] = (
                kids
                if mode == "primary"
                else _opus_select_variants(
                    geometry, kids, profiles, footprints, m, n, mode, small_m=True
                )
            )
        pools = covered

        def execution_eligible(geometry, kid, *, refine=False):
            """Bound steady loops and wave supply with one execution check."""
            pipeline, bm, bn, bk, wm, wn, *_, split = geometry
            profile, launch = profiles[kid]
            _, _, panels, work, c_registers, supply = tile_resources[geometry]
            stages = stage_counts.get(kid, 0)
            if split > 1:
                publication = 8 * (split - 1) / k
                if (
                    grids[geometry] > cu_grid
                    or publication * m * n > (m + n) / 2
                    or publication * bm * bn > (bm + bn) / 12
                    or (pipeline == "b_vgpr_panel" and work < 8)
                ):
                    return False
            if pipeline == "m_repeat_stage":
                return (panels <= 4 or stages >= 4) and (
                    stages <= 4 or panels <= stages or work <= 4
                )
            if pipeline not in ("b_vgpr_panel", "operand"):
                return True
            waves = wm * wn
            if waves > 4 and grids[geometry] * waves > 8 * cu_grid:
                return False
            if pipeline == "b_vgpr_panel":
                if work < 4 or (profile[11] == "2" and work < 12):
                    return False
                if stages >= 4 and launch[0] == 1 and work < 8:
                    return False
                if panels > 4:
                    bound = (3.5 if refine else 3) * cu_grid
                    if launch[0] == 2 and stages < 4 and supply < bound:
                        return False
                    if refine and work < 8 and supply < 3.5 * cu_grid:
                        return False
                return True
            compact, per_stage, direct_b, normal = (
                profile[index] == "true" for index in (9, 10, 11, 16)
            )
            if bk == 128 and (
                not (compact or per_stage) or (normal and c_registers < 32)
            ):
                return False
            if direct_b and (
                (stages > 4 and panels > 2 * stages)
                or (split > 1 and work < 8 and stages < 6)
            ):
                return False
            if split == 1 and waves <= 2 and grids[geometry] < cu_grid / 2:
                return False
            if panels > 2 * stages and (stages - 1) * work < 10:
                return False
            if refine:
                if (
                    split == 1
                    and panels > 2 * stages
                    and grids[geometry] < 0.75 * cu_grid
                ):
                    return False
                if waves > 4 and live_registers[kid] > 192:
                    return False
                if (
                    stages > 4
                    and panels > stages
                    and (
                        (direct_b and panels > stages + 1)
                        or (not direct_b and work > 4)
                    )
                ):
                    return False
            return True

        pools = narrow(
            pools,
            lambda geometry, kid: execution_eligible(geometry, kid, refine=True),
            prefer=execution_eligible,
        )
        final_primary, final_secondary, baselines = 0, 0, 0
        for geometry, kids in pools.items():
            pipeline, bm, _, _, _, wn = geometry[:6]
            mode = variant_modes[geometry]
            if pipeline == "b_vgpr" and math.ceil(m / bm) < max(8, 2 * wn):
                # Complete full-A queues amortize their preload over repeated M
                # tiles. Other shapes keep a legal representative of that tile.
                retained.update(
                    _opus_select_variants(
                        geometry,
                        kids,
                        profiles,
                        footprints,
                        m,
                        n,
                        "baseline",
                        small_m=True,
                    )
                )
                mode = "baseline"
            else:
                retained.update(kids)
            final_primary += mode == "primary"
            final_secondary += mode == "secondary"
            baselines += mode == "baseline"
        summary = (
            f"primary={final_primary} secondary={final_secondary} baseline={baselines} "
            f"resource_tiles={len(pools)}"
        )
    else:
        # Large-M ranges keep the established tile and execution-route rules.
        profiles = {kid: profile[0] for kid, profile in profiles.items()}
        min_lds, by_pipeline = {}, defaultdict(list)
        for geometry, kids in groups.items():
            min_lds[geometry] = min(footprints[kid] for kid in kids)
            by_pipeline[(geometry[0], *geometry[6:-1])].append(geometry)
        primary, secondary, small_baselines = set(), set(), set()
        for (pipeline, *_), tiles in by_pipeline.items():
            layouts = defaultdict(list)
            for geometry in tiles:
                layouts[geometry[3:]].append(geometry)
            ranked = {
                min(layout, key=lambda g: (costs[g], g)) for layout in layouts.values()
            }
            if pipeline in ("b_vgpr", "b_vgpr_overlap"):
                native = any(
                    wm * wn == 4 and (bk // 128) * (bm // 16) < 16
                    for _, bm, _, bk, wm, wn, _ in tiles
                )
                for (bk, wm, wn, _), layout in layouts.items():
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
                            else {largest}
                        )
                        waves = wm * wn
                        if (
                            native
                            and waves != 4
                            and grid >= cu_grid * max(1, 4 / waves)
                        ):
                            secondary.update(chosen)
                        else:
                            primary.update(chosen)
                    rotated = [g for g in layout if (bk // 128) * (g[1] // 16) >= 16]
                    if rotated:
                        # Cover the changed A live set, LDS waits, and B queue clamp.
                        secondary.add(min(rotated, key=lambda g: (costs[g], g)))
            elif pipeline == "m_repeat":
                # Fill the A ring with E_M=3..4; add two-resident-block tiles if needed.
                core = [
                    g
                    for g in tiles
                    if g[4] * g[5] == 8 and 3 <= g[1] // (g[4] * 16) <= 4
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
                        g
                        for g in tiles
                        if g[4] * g[5] == waves and min_lds[g] <= 80 * 1024
                    ]
                    if resident:
                        max_area = max(g[1] * g[2] for g in resident)
                        chosen = {g for g in resident if g[1] * g[2] == max_area}
                        (primary if underfilled else secondary).update(chosen)
            elif pipeline == "wave_interleave":
                for geometry in ranked:
                    (primary if geometry[4] * geometry[5] == 8 else secondary).add(
                        geometry
                    )
            elif pipeline in _OPUS_SMALL_PIPELINES:
                # Model padded work, matrix traffic, MFMA fragment storage, wave
                # supply, and Split-K publication traffic. Multiplicative work
                # avoids letting a large M-padding term hide differences in N/BK.
                scores, wave_bands, axes, routes = (
                    {},
                    defaultdict(list),
                    defaultdict(list),
                    defaultdict(list),
                )
                for geometry in tiles:
                    _, bm, bn, bk, wm, wn, *_, split = geometry
                    waves = wm * wn
                    em, en = bm // (16 * wm), bn // (16 * wn)
                    padding = math.ceil(m / bm) * bm / m * math.ceil(n / bn) * bn / n
                    fragment_regs = 4 * em * en + 8 * (bk // 128) * (em + en)
                    wave_supply = grids[geometry] * min(waves, 4) / cu_grid
                    publication = 8 * m * n * (split - 1) / (k * (m + n))
                    scores[geometry] = (
                        padding
                        * (1 + 16 / bm + 16 / bn)
                        * (1 + 0.25 * max(0, 4 / wave_supply - 1))
                        * (1 + fragment_regs / 512)
                        * (1 + 0.1 * min_lds[geometry] / (160 * 1024))
                        * (1 + publication)
                    )
                    wave_bands[waves <= 2].append(geometry)
                    # Cover each independent launch axis with its best tile, not
                    # every BK/wave/Split-K combination.
                    axes[("bk", bk)].append(geometry)
                    axes[("waves", wm, wn)].append(geometry)
                    axes[("split", split)].append(geometry)
                    for kid in groups[geometry]:
                        profile = profiles[kid]
                        if pipeline == "b_vgpr_panel":
                            routes[("load", profile[18], profile[11])].append(geometry)
                            if profile[19]:
                                routes[("wrapper", profile[19])].append(geometry)
                        elif pipeline == "operand":
                            routes[("load", profile[10], profile[11])].append(geometry)
                            if profile[21]:
                                routes[("wrapper", profile[21])].append(geometry)

                for band in wave_bands.values():
                    core = min(band, key=lambda g: (scores[g], g))
                    primary.add(core)
                    resident = [
                        g
                        for g in band
                        if min_lds[g] <= 80 * 1024 and scores[g] <= 1.25 * scores[core]
                    ]
                    if resident:
                        small_baselines.add(
                            min(resident, key=lambda g: (min_lds[g], scores[g], g))
                        )
                secondary.update(
                    min(axis, key=lambda g: (scores[g], g)) for axis in axes.values()
                )
                secondary.update(
                    min(route, key=lambda g: (scores[g], g))
                    for route in routes.values()
                )
            else:
                secondary.update(tiles)
            if not (primary | secondary).intersection(tiles):
                secondary.update(ranked)

        # Tighten primary ranges by relative work and grid supply, retaining modes.
        dropped, core_costs, by_layout = set(), {}, defaultdict(list)
        for geometry in primary:
            _, bm, bn, _, wm, wn = geometry[:6]
            if geometry[0] == "m_repeat" and wm * wn == 8 and bm // (wm * 16) >= 3:
                em, en = bm // (16 * wm), bn // (16 * wn)
                mfma = em * en * 32 * wm * wn / 4
                transfer = (bm + bn) * 128 / 512
                refill = max(0, em - 3) * 64
                body = (k / 128) * (mfma + transfer + refill) + 3072 + bm * bn * 2 / 512
                occupancy = grids[geometry] / cu_grid
                tail = 1 + 0.1 * (math.ceil(occupancy) / occupancy - 1)
                core_costs[geometry] = grids[geometry] * body * tail / (m * n * k)
            elif geometry[0] in ("b_vgpr", "b_vgpr_overlap"):
                by_layout[(geometry[0], *geometry[3:])].append(geometry)
            if (
                geometry[0] not in _OPUS_SMALL_PIPELINES
                and grids[geometry] < cu_grid / 2
            ):
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
        retained_small = primary | secondary | small_baselines

        for geometry, kids in groups.items():
            if geometry[0] in _OPUS_SMALL_PIPELINES and geometry not in retained_small:
                continue
            mode = "baseline"
            if geometry in primary:
                mode = "primary"
            elif geometry in secondary:
                mode = "secondary"
            retained.update(
                _opus_select_variants(geometry, kids, profiles, footprints, m, n, mode)
            )
        summary = (
            f"primary={len(primary)} secondary={len(secondary)} "
            f"small_geometries={sum(g[0] in _OPUS_SMALL_PIPELINES for g in retained_small)}"
        )
    selected.update(retained)
    tag = "OPUS small search" if small_m else "OPUS search"
    print(
        f"[{tag}] shape={(m, n, k)} legal={len(legal)} "
        f"{summary} co={co_count} candidates={len(selected)}",
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

    def pre_process(self, args):
        self._mxscale_table = args.mxscale_table
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
        if gfx == "gfx1250" and (
            not self._mxscale_table or set(untunedf["w_scale_block"]) != {"128x128"}
        ):
            raise ValueError(
                "gfx1250 tunes the mxfp8_128 kernels of gemm_a8w8_blockscale_"
                "bpreshuffle's table only: pass --mxscale_table and 128x128 shapes"
            )
        seeds = (
            self._bmm_seeds() if self._mxscale_table and args.libtype != "opus" else {}
        )
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
                (
                    kid,
                    instance.bpreshuffle_split_k,
                    opus_kernel_name(kid, cu_num=cu_num),
                    run_gemm_opus,
                )
                for kid, instance in opus
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
                    # bmm's ids sit between fsk's and the _pf ones
                    if KERNEL_ID_BASE_BMM <= kernelId < KERNEL_ID_BASE_PF
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
