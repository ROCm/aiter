# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host policy of the FlyDSL MXFP6 MoE: which kernel each stage of a call runs.

`select_config` is what the op runs: the tuned row of the call's shape where the tuned
config (aiter/configs/tuned_mxfp6_moe.csv and its model_configs siblings, or the file
AITER_CONFIG_MXFP6_MOE names) has one, else the heuristic.

`default_config` holds the heuristic: the routing tile, and per stage the kernel tile,
LDS ring, tile-group size, non-temporal weight loads, padded rows, L2 warm and half
tiles. The per-stage choices change the schedule of the two grouped GEMMs, no output
bit. The rules were measured on gfx950 on 88 (layer, token count) points, 8 alternating
rounds each.

Host code only: no kernel is built here (aiter.ops.flydsl.mxfp6_moe_kernels).
"""

from __future__ import annotations

import csv
import functools
import math
import os
import re
from dataclasses import dataclass

from .mxfp6_moe_quant import FORMATS

H_LAYOUTS = ("tile_step", "rows")

# gfx950 dispatches blocks round-robin over 8 XCDs, each with its own L2 (the XCD remap
# of the GEMM kernel's tile_order)
XCDS = 8
# GM 4 grids warm only the first quarter of each next m-tile's K steps (measured on
# 32k-token grids)
WARM_DEN_GM4 = 4
# n-tiles re-reading a padded row (N / BN) from which the padded copy pays on 256-row
# tiles: 16 lose 2-3%, 20 and more win 1-5% (measured)
PAD_MIN_REUSE = 20
# A8W6 256x256 gate: keep 256x128 while its ceil(rounds) x RQC < the 256x256 grid's
# (fitted on one shape)
RQC = 0.6
# shortcut: gfx950 MALL size hard-coded (256 MiB), read it from the device before
# targeting other parts
MALL_BYTES = 256 << 20


@dataclass(frozen=True)
class Mxfp6StageConfig:
    """The kernel one grouped GEMM of the op runs.

    tile_m x tile_n: rows x columns of a block's tile (128 or 256 each); tile_n covers
        whole weight slabs of the prepared layer.
    ring: LDS ring depth in K steps (2 or 3).
    group_m: m-tiles per tile group (the order the blocks take the tiles in).
    nt_weights: non-temporal weight loads.
    pad_rows: stage 1, 6-bit activations: the gathered rows are first copied to 128 B
        per K step (one more launch, tokens x K transient bytes).
    warm: stage 2 reading a tile-step-major h: 0 = off, n = the store epilogue warms
        the first 1 / n of the next tile group's h rows in L2.
    half_tiles: stage 1: tile_m-row tiles over routing tiles of 2 x tile_m rows.
    """

    tile_m: int
    tile_n: int
    ring: int = 3
    group_m: int = 1
    nt_weights: bool = True
    pad_rows: bool = False
    warm: int = 0
    half_tiles: bool = False

    @property
    def route_m(self) -> int:
        """Rows of the routing tile this kernel runs on."""
        return self.tile_m * (2 if self.half_tiles else 1)


@dataclass(frozen=True)
class Mxfp6MoeConfig:
    """The kernels of one MXFP6 MoE call.

    route_m: rows of a routing tile (128 or 256): the router pads each expert's rows to
        whole tiles, and both stages run on them.
    h_layout: layout of a 6-bit h between the stages: "tile_step" ([m-tile][K step]
        [rows][96 B], stage 2 reads whole cache lines) or "rows" (row-major). 8-bit h
        has one layout; the field has no effect there.
    """

    route_m: int
    stage1: Mxfp6StageConfig
    stage2: Mxfp6StageConfig
    h_layout: str = "tile_step"


def pick_bm(rows_per_expert: float) -> int:
    """Rows of the routing tile: 256-row tiles waste most of the work when experts get
    ~100 rows (many experts / few tokens)."""
    return 128 if rows_per_expert <= 160 else 256


def stage_bn(N: int, a_fmt: str) -> int:
    """Weight slab width of a stage, and its default N tile. Fixed per stage at prepare
    time: the weights are preshuffled N-block-major for it. 128 when N % 256 != 0
    (hidden a multiple of 128 only) or for FP8 activations (256 x 256 with FP8 A rows
    does not fit three LDS stages), else 256."""
    return 128 if (N % 256 != 0 or a_fmt == "e4m3") else 256


def route_tiles(entries: int, experts: int, route_m: int) -> int:
    """m-tiles of the routed grid: the bound the router allocates, the tiles of the
    routed rows plus one partly filled tile per expert."""
    return (entries + route_m - 1) // route_m + experts


def _round_bytes_gm(A, B, tpe, r, gms=(1, 2, 4, 8, 16)):
    """The GM minimising the bytes one round of r co-resident tiles brings into its
    XCD's L2: GM m-tiles of rows (A each) and r / GM n-tiles of weight slab (B each) per
    expert the group spans, ceil(GM / tpe) of them; ties to the larger GM."""
    # a round smaller than every group (under 16 CUs): the smallest
    gms = [g for g in gms if g <= r] or [min(gms)]
    cost = {g: g * A + max(1, math.ceil(g / max(tpe, 1))) * r / g * B for g in gms}
    return max(g for g in cost if cost[g] <= min(cost.values()) * (1 + 1e-9))


def stage1_gm(A, B, tpe, tiles, bm, ncu):
    """GM (m-tiles per tile group) of stage 1 (swiglu epilogues). A / B: bytes per unit
    of K of one m-tile's rows / one n-tile's weight slab; tpe: routed m-tiles per
    expert; tiles: routed tiles of the grid."""
    # one round, every tile co-resident: an XCD holds tiles / 8 of them
    if tiles <= ncu:
        return _round_bytes_gm(A, B, tpe, max(1, -(-tiles // XCDS)))
    # rounds of ncu / 8 tiles per XCD; a group's m-tiles share one expert's slab
    if tpe >= 2:
        return _round_bytes_gm(A, B, float("inf"), ncu // XCDS, (2, 4, 8, 16))
    # <= 1 m-tile per expert, several rounds: no slab is shared and the curve is flat
    # within ~3%. Measured constants: best on round-robin routing (0.5-1.5% over 8 on
    # 4k-token grids); random routing measured 4-8 better
    return 16 if bm == 256 else 4


def stage2_gm_warm(tpe, tile_step):
    """GM of stage 2 (store) = the m-tiles one expert owns (its group shares the weight
    slab), at most 4 (from 4 on the byte model ties 4 and 8; 4 measured >= 8), and the
    L2 warm of the next group's h rows, on wherever the kernel reads h
    tile-step-major."""
    return (1 if tpe < 2 else 2 if tpe < 4 else 4), int(tile_step)


def a8w6_tiles(tile_m, tile_n, N, m_tiles, E, stage1, ncu):
    """(tile_m, tile_n, ring, half tiles) for 8-bit activations, from the op's tile
    (256x128 or 128x128, 3 stages). The grid estimates count the tiles the routed rows
    fill (m_tiles less the router's E spare tiles) against the CUs."""
    routed = m_tiles - E
    t128 = N // 128 * routed  # 256x128 tiles of the routed rows
    if (
        (tile_m, tile_n) == (256, 128)
        and N % 256 == 0
        and t128 > ncu
        and not -(-t128 // ncu) * RQC < -(-(t128 // 2) // ncu)
    ):
        # 256x256 on a 2-stage ring: half the tiles and 35% fewer operand bytes per MFMA
        # (32k tokens: 1.16-1.17x); not while the 256x128 grid fits one round on the
        # CUs, nor where it needs fewer whole rounds (the RQC gate)
        return 256, 256, 2, False
    if (
        (tile_m, tile_n) == (128, 128)
        and N % 256 == 0
        and N // 128 * E * -(-routed // E) > ncu
    ):
        # 128x256 on two 128-wide slabs: 0.75-0.94x the stage time of 128x128 unless the
        # 128x128 grid fits one round. Balanced estimate E * ceil(routed / E): <= 160
        # rows per expert, so each routed expert has its own tiles
        return 128, 256, 3, False
    if stage1 and (tile_m, tile_n) == (256, 128) and 2 * t128 <= ncu:
        # half tiles: a 256x128 grid filling at most half the CUs runs 128x128 tiles
        # over the 256-row routing tiles split in halves, one round (1.25x on 10496 x
        # 1024 at 1024 tokens); 8-bit h is row-major, so a half tile writes its own rows
        return 128, 128, 3, True
    return tile_m, tile_n, 3, False


def nt_weights(bm, weight_bytes, stage1):
    """Non-temporal weight DMAs: the expert weights stream once per call and should not
    evict A from L2 (gating them to ~1 m-tile per expert lost 2-3% on 32k-token grids).
    Except on 128-row tiles of a layer whose weights fit the MALL: plain loads find them
    there from the previous call, nt re-reads HBM (plain wins up to 240 MiB, loses 3-7%
    from 288 MiB; it pays only when the layer is called back to back).

    weight_bytes: the stage's prepared weights and scales."""
    # the layer's W1 + W2, estimated from this stage's: W1 = 2 x W2
    layer_bytes = weight_bytes * (1.5 if stage1 else 3)
    return not (bm == 128 and layer_bytes <= MALL_BYTES)


def default_stage_config(
    *,
    stage1: bool,
    N: int,
    a_fmt: str,
    route_m: int,
    slab_n: int,
    m_tiles: int,
    experts: int,
    weight_bytes: int,
    a_bytes: int,
    cu_num: int,
    tile_step: bool,
    pad_rows: bool = True,
) -> Mxfp6StageConfig:
    """The heuristic's kernel of one stage: stage 1 (gathered rows, SwiGLU, quantized
    h; N = 2 x inter) or stage 2 (h @ W2^T; N = hidden).

    m_tiles: tiles of the routed grid (route_tiles); weight_bytes: the stage's prepared
    weights and scales; a_bytes: the stage's activation rows; tile_step: a 6-bit h is
    tile-step-major; pad_rows = False turns the padded copy of stage 1 off (it costs
    1.7-5.4% where the copy is on, and saves its tokens x K transient bytes)."""
    six = FORMATS[a_fmt][3] == 6
    tile_m, tile_n, ring, half = route_m, slab_n, 3, False
    if a_fmt == "e4m3":
        tile_m, tile_n, ring, half = a8w6_tiles(
            tile_m, tile_n, N, m_tiles, experts, stage1, cu_num
        )
    nt = nt_weights(tile_m, weight_bytes, stage1)
    # the padded copy where >= PAD_MIN_REUSE n-tiles of 256 rows re-read each padded
    # row. It is tokens x K bytes (4/3 of the rows): not where that would pass the
    # 32-bit offset limit
    pad = bool(
        pad_rows
        and stage1
        and six
        and tile_m == 256
        and N // tile_n >= PAD_MIN_REUSE
        and a_bytes // 3 * 4 + 4096 <= 2**32
    )
    routed = m_tiles - experts  # m-tiles of routed rows (the router's bound)
    # m-tiles per expert (half tiles: 128-row halves)
    tpe = routed / experts * (2 if half else 1)
    warm = 0
    if stage1:
        # balanced estimate, as in a8w6_tiles
        tiles = experts * -(-routed // experts) * (2 if half else 1) * (N // tile_n)
        gm = stage1_gm(
            tile_m * (0.75 if six and not pad else 1.0),
            tile_n * 0.75,
            tpe,
            tiles,
            tile_m,
            cu_num,
        )
    else:
        gm, warm = stage2_gm_warm(tpe, tile_step)
        warm = warm and (1 if gm <= 2 else WARM_DEN_GM4)
    return Mxfp6StageConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        ring=ring,
        group_m=gm,
        nt_weights=nt,
        pad_rows=pad,
        warm=warm,
        half_tiles=half,
    )


# cached: tiny calls (~75 us) are host-bound
@functools.cache
def default_config(
    *,
    tokens: int,
    topk: int,
    experts: int,
    model_dim: int,
    inter_dim: int,
    a_fmt: str,
    slab_n1: int,
    slab_n2: int,
    w1_bytes: int,
    w2_bytes: int,
    cu_num: int,
    pad_rows: bool = True,
    h_layout: str = "tile_step",
    route_m: int | None = None,
) -> Mxfp6MoeConfig:
    """The heuristic's kernels of one call: `tokens` rows of `model_dim`, `topk` of
    `experts` experts each.

    slab_n1 / slab_n2: weight slab widths of the prepared layer (stage_bn);
    w1_bytes / w2_bytes: bytes of each stage's prepared weights and scales; cu_num: the
    device's compute units; pad_rows / h_layout: see default_stage_config and
    Mxfp6MoeConfig; route_m: the routing tile the stages are picked for (128 or 256),
    None = the heuristic's (pick_bm)."""
    if h_layout not in H_LAYOUTS:
        raise ValueError(f"h_layout {h_layout!r}: use 'tile_step' or 'rows'")
    bits = FORMATS[a_fmt][3]
    if route_m is None:
        route_m = pick_bm(tokens * topk / experts)
    m_tiles = route_tiles(tokens * topk, experts, route_m)
    tile_step = bits == 6 and h_layout == "tile_step"
    shared = {
        "a_fmt": a_fmt,
        "route_m": route_m,
        "m_tiles": m_tiles,
        "experts": experts,
        "cu_num": cu_num,
        "tile_step": tile_step,
        "pad_rows": pad_rows,
    }
    stage1 = default_stage_config(
        stage1=True,
        N=2 * inter_dim,
        slab_n=slab_n1,
        weight_bytes=w1_bytes,
        a_bytes=tokens * model_dim * bits // 8,
        **shared,
    )
    stage2 = default_stage_config(
        stage1=False,
        N=model_dim,
        slab_n=slab_n2,
        weight_bytes=w2_bytes,
        a_bytes=m_tiles * route_m * inter_dim * bits // 8,
        **shared,
    )
    return Mxfp6MoeConfig(
        route_m=route_m, stage1=stage1, stage2=stage2, h_layout=h_layout
    )


# ---------------------------------------------------------------------------------
# tuned configs
# ---------------------------------------------------------------------------------
# A tuned row is keyed as tuned_fmoe.csv keys a fused_moe call. torch has no FP6 dtype,
# so the q_dtype_a / q_dtype_w cells of the 6-bit formats hold these names.
Q_DTYPE_NAMES = {
    "e2m3": "mxfp6_e2m3",
    "e3m2": "mxfp6_e3m2",
    "e4m3": "torch.float8_e4m3fn",
}
# the call every row describes: bf16 in and out, silu(gate) * up, MX scales per 32
# along K, routing weights applied by the combine
TUNED_CALL = {
    "act_type": "ActivationType.Silu",
    "dtype": "torch.bfloat16",
    "q_type": "QuantType.per_1x32",
    "use_g1u1": 1,
    "doweight_stage1": 0,
}
TUNED_KEYS = ("gfx", "cu_num", "token", "model_dim", "inter_dim", "expert", "topk")
TUNED_KEYS += ("act_type", "dtype", "q_dtype_a", "q_dtype_w", "q_type")
TUNED_KEYS += ("use_g1u1", "doweight_stage1")
# block_m: rows of the routing tile; kernelName1 / kernelName2: stage_kernel_name
TUNED_RESULTS = ("block_m", "h_layout", "kernelName1", "kernelName2")

_STAGE_NAME = re.compile(
    r"flydsl_mxfp6_moe(?P<stage>[12])_t(?P<tile_m>\d+)x(?P<tile_n>\d+)"
    r"_r(?P<ring>\d+)_gm(?P<group_m>\d+)(?P<nt>_nt)?(?P<pad>_pad)?"
    r"(?:_warm(?P<warm>\d+))?(?P<half>_half)?"
)


def stage_kernel_name(stage1: bool, config: Mxfp6StageConfig) -> str:
    """The name a tuned row holds for a stage's kernel:
    flydsl_mxfp6_moe{1|2}_t{rows}x{columns}_r{ring}_gm{group}[_nt][_pad][_warm{n}]
    [_half]."""
    return (
        f"flydsl_mxfp6_moe{1 if stage1 else 2}_t{config.tile_m}x{config.tile_n}"
        f"_r{config.ring}_gm{config.group_m}"
        + "_nt" * config.nt_weights
        + "_pad" * config.pad_rows
        + (f"_warm{config.warm}" if config.warm else "")
        + "_half" * config.half_tiles
    )


def stage_config_from_name(name: str) -> tuple[bool, Mxfp6StageConfig]:
    """(is stage 1, the kernel) of a stage_kernel_name. Raises ValueError for any
    other text and for a flag of the other stage."""
    m = _STAGE_NAME.fullmatch(name)
    stage1 = bool(m) and m["stage"] == "1"
    if not m or ((m["pad"] or m["half"]) and not stage1) or (m["warm"] and stage1):
        raise ValueError(f"not a MXFP6 MoE stage kernel name: {name!r}")
    return stage1, Mxfp6StageConfig(
        tile_m=int(m["tile_m"]),
        tile_n=int(m["tile_n"]),
        ring=int(m["ring"]),
        group_m=int(m["group_m"]),
        nt_weights=bool(m["nt"]),
        pad_rows=bool(m["pad"]),
        warm=int(m["warm"] or 0),
        half_tiles=bool(m["half"]),
    )


def _int_cell(value) -> int:
    """An integer cell; a table rewritten through pandas holds 256 as 256.0."""
    number = float(value)
    if number != int(number):
        raise ValueError(f"{value!r} is not an integer")
    return int(number)


def row_call(row) -> tuple[int, int, int, int, int, str, str]:
    """(token, model_dim, inter_dim, expert, topk, a_fmt, w_fmt) of a row of an
    untuned or tuned config file (a mapping column -> cell).

    Raises ValueError for a row that is not a MXFP6 MoE call (TUNED_CALL) or names
    another format, KeyError for a missing column."""
    for column, value in TUNED_CALL.items():
        cell = str(row[column]).strip()
        if (_int_cell(cell) if isinstance(value, int) else cell) != value:
            raise ValueError(f"{column} {cell!r} is not {value!r}")
    formats = {name: fmt for fmt, name in Q_DTYPE_NAMES.items()}
    a_name, w_name = (str(row[c]).strip() for c in ("q_dtype_a", "q_dtype_w"))
    if a_name not in formats or formats.get(w_name) not in ("e2m3", "e3m2"):
        raise ValueError(
            f"q_dtype_a {a_name!r} / q_dtype_w {w_name!r}: activations are one of "
            f"{sorted(formats)}, weights a mxfp6 one"
        )
    return (
        *(_int_cell(row[column]) for column in TUNED_KEYS[2:7]),
        formats[a_name],
        formats[w_name],
    )


def row_results(config: Mxfp6MoeConfig) -> dict[str, int | str]:
    """The TUNED_RESULTS cells of a tuned row that names config."""
    return {
        "block_m": config.route_m,
        "h_layout": config.h_layout,
        "kernelName1": stage_kernel_name(True, config.stage1),
        "kernelName2": stage_kernel_name(False, config.stage2),
    }


def row_config(row) -> Mxfp6MoeConfig:
    """The kernels a tuned row names (its TUNED_RESULTS cells). Raises ValueError for
    cells that name no MXFP6 MoE config, KeyError for a missing column."""
    first, stage1 = stage_config_from_name(str(row["kernelName1"]).strip())
    second, stage2 = stage_config_from_name(str(row["kernelName2"]).strip())
    config = Mxfp6MoeConfig(
        route_m=_int_cell(row["block_m"]),
        stage1=stage1,
        stage2=stage2,
        h_layout=str(row["h_layout"]).strip(),
    )
    if not first or second:
        raise ValueError("a stage kernel in the other stage's column")
    if config.h_layout not in H_LAYOUTS:
        raise ValueError(f"h_layout {config.h_layout!r}: use 'tile_step' or 'rows'")
    if stage1.route_m != config.route_m or stage2.route_m != config.route_m:
        raise ValueError(f"a kernel off the {config.route_m}-row routing tiles")
    return config


def load_tuned_configs(path: str) -> dict[tuple, Mxfp6MoeConfig]:
    """The rows of a tuned config file: (gfx, cu_num, token, model_dim, inter_dim,
    expert, topk, a_fmt, w_fmt) -> the kernels of that call. No file: no rows.

    Raises ValueError, naming the file and line, for a row that is not a MXFP6 MoE
    call (row_call), names no config (row_config) or repeats a key."""
    configs: dict[tuple, Mxfp6MoeConfig] = {}
    if not path or not os.path.exists(path):
        return configs
    with open(path, newline="", encoding="utf-8") as source:
        for line, row in enumerate(csv.DictReader(source), 2):
            try:
                key = (row["gfx"].strip(), _int_cell(row["cu_num"]), *row_call(row))
                if key in configs:
                    raise ValueError("a repeated key")
                configs[key] = row_config(row)
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(
                    f"{path}, line {line}: not a tuned MXFP6 MoE row ({e})"
                ) from e
    return configs


@functools.cache
def _tuned_configs() -> dict[tuple, Mxfp6MoeConfig]:
    from aiter.jit.core import AITER_CONFIGS

    return load_tuned_configs(AITER_CONFIGS.AITER_CONFIG_MXFP6_MOE_FILE)


def clear_tuned_configs() -> None:
    """Forget the tuned config and every selection made from it: the next call reads
    the file AITER_CONFIG_MXFP6_MOE names then. For tuners and tests; a serving
    process reads it once."""
    from aiter.jit.core import AITER_CONFIGS

    type(AITER_CONFIGS).get_config_file.cache_clear()
    _tuned_configs.cache_clear()
    select_config.cache_clear()


# cached: tiny calls (~75 us) are host-bound
@functools.cache
def select_config(
    *,
    tokens: int,
    topk: int,
    experts: int,
    model_dim: int,
    inter_dim: int,
    a_fmt: str,
    w_fmt: str,
    slab_n1: int,
    slab_n2: int,
    w1_bytes: int,
    w2_bytes: int,
    cu_num: int,
    gfx: str,
) -> Mxfp6MoeConfig:
    """The kernels the op runs for a call: the tuned row of exactly this device and
    shape, else the heuristic's (default_config, whose arguments these are).

    Raises ValueError for a tuned row whose kernels do not fit the layer: a tile that
    does not cover whole weight slabs or does not divide the stage's columns."""
    key = (gfx, cu_num, tokens, model_dim, inter_dim, experts, topk, a_fmt, w_fmt)
    config = _tuned_configs().get(key)
    if config is None:
        return default_config(
            tokens=tokens,
            topk=topk,
            experts=experts,
            model_dim=model_dim,
            inter_dim=inter_dim,
            a_fmt=a_fmt,
            slab_n1=slab_n1,
            slab_n2=slab_n2,
            w1_bytes=w1_bytes,
            w2_bytes=w2_bytes,
            cu_num=cu_num,
        )
    for stage, n, slab in (
        (config.stage1, 2 * inter_dim, slab_n1),
        (config.stage2, model_dim, slab_n2),
    ):
        if stage.tile_n % slab or n % stage.tile_n:
            raise ValueError(
                f"tuned MXFP6 MoE row {key}: the {stage.tile_n}-column tile of "
                f"{stage} does not fit {n} columns in {slab}-column weight slabs"
            )
    return config
