# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The megakernel's host-side configuration: the switch and the step ranges,
the GEMM tiles of each layout, the admission checks, the arena regions and
the row capacities."""

from __future__ import annotations

import os

from ..config import _WAVE_SIZE as WAVE
from .protocol import (
    COMPACT_SCRATCH_MAX,
    SEG_BITS,
    TALLY_MAX,
    TALLY_MAXR,
    compact_scratch_bytes,
    ctl_arena_bytes,
    segs_pad,
    src_off,
)

#: AITER_MEGAKERNEL=1 runs steps of up to SLOTS_TOKENS tokens per rank on
#: slots and larger steps below COMPACT_MAX_TOKENS on compact; larger steps
#: run the staged path. A step's size is its receive bound per rank
#: (``recv_token_bound``, else ``world * max_tokens_per_rank``), the same on
#: every rank, so every rank picks the same path.
SLOTS_TOKENS = 512
#: With IOD-placed weights (AITER_MOE_LOC) slots runs steps of up to this many
#: tokens per rank, by (hidden, inter): placed, slots claims by IOD domain, and
#: compact's 128-row tile is faster past it. Where it crosses depends on the
#: model: compact's fixed cost (every expert's weights read once, the dispatch
#: ahead of the first m-tile) weighs more with more experts per rank and fewer
#: rows each. EP4, MoE alone, steady state at the power limit, us/layer, slots /
#: compact:
#: - DeepSeek-R1 (recorded routing): 128 tokens 129.6 / 145.9, 192 150.9 /
#:   154.9, 256 173.5 / 159.3, 384 195.0 / 176.8.
#: - DSV4.1 Flash (uniform routing): 512 tokens 168.5 / 184.4, 768 173.8 /
#:   189.5, 896 195.6 / 193.5, 1024 203.4 / 195.2.
SLOTS_TOKENS_PLACED = {
    (7168, 2048): 192,  # DeepSeek-R1
    (5120, 2304): 768,  # DSV4.1 Flash
    None: 256,
}
COMPACT_MAX_TOKENS = 8192

#: slots' GEMM tiles: the a4w4 CSV's decode row for this shape family. The
#: staged optimum (t16x512x128 b2 GEMM1) is slower here: its m-tiles finish
#: later and GEMM2 waits on them. A tile never changes the bits (every output
#: element sums K in the same order).
SLOTS_TILE = dict(  # noqa: C408 - the builder's keyword names
    tile_m=16,
    tile_n=256,
    tile_k=256,
    num_buffers=3,
    tile_n2=512,
    tile_k2=128,
    num_buffers2=3,
)
#: Deeper pipelines for the 16-row tile: with few rows an item is bound by the
#: bytes a CTA keeps in flight, not by HBM. The 64-row tile's 4 stages do not
#: fit 2 CTAs per CU.
SLOTS_TILE_M16 = dict(  # noqa: C408 - the builder's keyword names
    num_buffers=4,
    tile_k2=256,
    num_buffers2=2,
)
#: Tokens per rank past which the 32-row, then the 64-row tile is faster.
SLOTS_TILE_M32_TOKENS = 64
SLOTS_TILE_M64_TOKENS = 256
SLOTS_CTAS_PER_CU = 2


def slots_tile(max_tokens_per_rank: int) -> dict:
    """The decode tiles for an instance serving up to this many tokens."""
    tile = dict(SLOTS_TILE)
    if int(max_tokens_per_rank) > SLOTS_TILE_M64_TOKENS:
        tile["tile_m"] = 64
    elif int(max_tokens_per_rank) > SLOTS_TILE_M32_TOKENS:
        tile["tile_m"] = 32
    else:
        tile.update(SLOTS_TILE_M16)
        from ..s1f_ctl import iod_placement_reason

        if iod_placement_reason() is not None:
            # A placed W2 (512 x 128 layout) is read by 512 x 128 tiles: the
            # 16-row tile keeps the other tiles' GEMM2.
            tile.update(
                tile_k2=SLOTS_TILE["tile_k2"], num_buffers2=SLOTS_TILE["num_buffers2"]
            )
    return tile


def read_megakernel() -> bool:
    return os.environ.get("AITER_MEGAKERNEL", "0").strip() in ("1", "true", "True")


def slots_reason(config) -> str | None:
    """Why this MegaMoE cannot run the slots megakernel, or None if it can."""
    if not read_megakernel() or SLOTS_TOKENS <= 0:
        return "AITER_MEGAKERNEL is off"
    from ..s1f_ctl import iod_placement_reason

    tile = slots_tile(slots_tokens(config))
    why = iod_placement_reason()
    if why is not None:
        # Placed weights stop slots only when its tiles cannot read them
        # (GEMM1 256 x 256 and GEMM2 512 x 128 read the layouts as laid).
        from .kernel import BUILDER_LOC

        if not BUILDER_LOC or not _reads_placed(tile):
            return why
    return _mega_reason(config, tile)


def compact_reason(config) -> str | None:
    """Why this MegaMoE cannot run the compact megakernel, or None if it can."""
    if not read_megakernel():
        return "AITER_MEGAKERNEL is off"
    geo = config.geometry
    if geo.max_tokens_per_rank > 0xFFFF:
        return "X1 holds a source's rows of an expert in 16 bits"
    if geo.max_tokens_per_rank * config.topk > TALLY_MAX * TALLY_MAXR:
        return "more routes per rank than the tally CTAs count"
    npes, epr = config.world_size, config.experts_per_rank
    segs = segs_pad(npes=npes, experts_per_rank=epr)
    if segs > (1 << SEG_BITS):
        return "too many (rank, expert) segments for a route slot"
    if npes * segs * 4 > src_off(epr):
        return "the X1 rows do not fit the control region"
    if compact_scratch_bytes(npes=npes, experts_per_rank=epr) > COMPACT_SCRATCH_MAX:
        return "compact's dispatch LDS does not fit the GEMM arena"
    variants = compact_variants(config)
    if not variants:
        return "slots serves every step of this instance"
    for _, _, tile, _ in variants:
        why = _mega_reason(config, tile)
        if why is not None:
            return why
    return _placed_reason(variants)


def _reads_placed(tile) -> bool:
    """Whether a compact tile's GEMMs read the IOD-placed W1 and W2."""
    from aiter.ops.flydsl import iod_locality

    return bool(
        iod_locality.placed_ratio(iod_locality.W13_TILE, tile["tile_n"], tile["tile_k"])
        and iod_locality.placed_ratio(
            iod_locality.W2_TILE, tile["tile_n2"], tile["tile_k2"]
        )
    )


def _compact_tiles():
    """COMPACT_TILES, less the rows whose tiles cannot read IOD-placed
    weights when they are placed: the next row serves their steps."""
    from ..s1f_ctl import iod_placement_reason

    if iod_placement_reason() is None:
        return COMPACT_TILES
    return tuple(r for r in COMPACT_TILES if r[0] is None or _reads_placed(r[1]))


def _placed_reason(variants) -> str | None:
    """Why the compact megakernel cannot read IOD-placed weights, or None
    (also when nothing is placed). Its GEMMs read them through the builder's
    ``b_placed`` when every variant's tiles serve the placed layouts."""
    from ..s1f_ctl import iod_placement_reason

    why = iod_placement_reason()
    if why is None:
        return None
    from .kernel import BUILDER_LOC

    if not BUILDER_LOC or not all(_reads_placed(t) for _, _, t, _ in variants):
        return why
    return None


def _mega_reason(config, t) -> str | None:
    from aiter import ActivationType, QuantType

    from ..recipe import CombineKind, DispatchKind

    recipe, geo = config.recipe, config.geometry
    if recipe.dispatch is not DispatchKind.COMPACT or config.dispatch_wire != "fp4":
        return "needs the compact dispatch on the fp4 wire"
    if recipe.combine is not CombineKind.FUSED_SCATTER:
        return "needs the fused-scatter combine"
    comp = recipe.compute
    if comp.activation not in (ActivationType.Silu, ActivationType.Swiglu):
        return f"activation {comp.activation} not supported"
    if (
        comp.quant_type != QuantType.per_1x32
        or comp.hidden_pad
        or comp.intermediate_pad
    ):
        return "needs per_1x32 quant and no padding"
    H, I = config.hidden_dim, geo.inter_dim
    if H % (32 * WAVE) or (2 * I) % t["tile_n"] or H % t["tile_n2"]:
        return f"hidden {H} / inter {I} do not tile"
    if config.world_size > WAVE or geo.experts % config.world_size:
        return "experts must split evenly over at most 32 ranks"
    return None


def slots_tokens(config) -> int:
    """Tokens per rank slots serves: SLOTS_TOKENS, at most the instance's
    ``max_tokens_per_rank``."""
    from ..s1f_ctl import iod_placement_reason

    if iod_placement_reason() is None:
        cap = SLOTS_TOKENS
    else:
        cap = SLOTS_TOKENS_PLACED.get(
            (config.hidden_dim, config.geometry.inter_dim), SLOTS_TOKENS_PLACED[None]
        )
    return min(cap, int(config.geometry.max_tokens_per_rank))


def slots_recv_cap(config) -> int:
    """The largest receive bound (tokens) a step may have to run slots."""
    return config.world_size * slots_tokens(config)


def slots_cap(config) -> int:
    """Rows each local expert reserves. A token's top-k experts are distinct, so
    an expert gets at most one row per token its rank receives, and a served
    step receives at most ``slots_recv_cap`` tokens."""
    tm = slots_tile(slots_tokens(config))["tile_m"]
    rows = slots_recv_cap(config)
    return (rows + tm - 1) // tm * tm


#: compact's GEMM tile by step size: (tokens per rank up to which a step runs
#: it, None for every larger step; the tile of both GEMMs; CTAs per CU). Each
#: row is its own kernel with its own control block, covering the steps whose
#: receive bound falls in its range (see compact_variants). The large tile
#: reads each expert's weights fewer times once steps grow past 512 tokens per
#: rank; its arena fits 2 CTAs per CU, the small tile's (~70 KB) 4.
COMPACT_TILES = (
    (
        512,
        dict(  # noqa: C408 - the builder's keyword names
            tile_m=64,
            tile_n=256,
            tile_k=128,
            num_buffers=3,
            tile_n2=256,
            tile_k2=128,
            num_buffers2=3,
        ),
        4,
    ),
    (
        None,
        dict(  # noqa: C408 - the builder's keyword names
            tile_m=128,
            tile_n=256,
            tile_k=256,
            num_buffers=2,
            tile_n2=256,
            tile_k2=256,
            num_buffers2=2,
        ),
        2,
    ),
)
#: Rows the tile table may have: each kernel has its own arena control region
#: (compact_ctl, compact_ctl1, ... in arena._ARENA_REGION_ORDER).
COMPACT_MAX_KERNELS = 4
assert len(COMPACT_TILES) <= COMPACT_MAX_KERNELS


def compact_variants(config) -> list:
    """(recv_lo, recv_hi, tile, CTAs per CU) of each compact kernel: the rows
    of COMPACT_TILES cut to the receive bounds compact serves, from past
    slots' (or 0) to below COMPACT_MAX_TOKENS per rank, at most the
    instance's. A step runs the kernel whose (recv_lo, recv_hi] holds its
    bound."""
    lo = slots_recv_cap(config) if slots_reason(config) is None else 0
    top = min(config.max_recv, config.world_size * COMPACT_MAX_TOKENS - 1)
    out = []
    for upto, tile, ctas in _compact_tiles():
        if lo >= top:
            break
        hi = top if upto is None else min(top, config.world_size * int(upto))
        if hi > lo:
            out.append((lo, hi, dict(tile), int(ctas)))
            lo = hi
    # STREAM packs the plan, which leaves the large tile room for a third
    # stage (EP4, 2048 tokens per rank).
    if (
        out
        and out[-1][2]["tile_m"] == 128
        and stream_reason(config, *out[-1][2:]) is None
    ):
        out[-1][2].update(
            STREAM_TILES.get(
                (config.hidden_dim, config.geometry.inter_dim), STREAM_TILES[None]
            )
        )
    return _split_g2_early(config, out)


#: Early GEMM2 CTAs (the builder's ``g2_early``, after ROCm/aiter#7788's
#: early_k) by (hidden, inter): ((tokens per rank up to which a step runs the
#: setting, None for every larger step; one CTA in KE of each XCC group, 0 =
#: off), ...). A kernel row of compact_variants whose range spans a boundary
#: splits there into its own kernels. GEMM2 is XGMI-bound and the links idle
#: while GEMM1 runs, so the early CTAs start the scatter sooner; the gain grows
#: with the step. Measured EP4, MoE alone, KE 2 against off, three interleaved
#: repeats of 100 iterations: DSR1 -1.3 us (-0.4%) at 1536 tokens per rank,
#: -14 us (-2.1%) at 4096; DSV4.1 Flash none, so it stays off.
G2_EARLY_TILES = {
    (7168, 2048): ((None, 2),),  # DSR1
    None: ((None, 0),),
}
#: GEMM1 items an early CTA runs before it switches.
G2_EARLY_AFTER = 1


def _split_g2_early(config, variants) -> list:
    """compact_variants' rows cut at the G2_EARLY_TILES boundaries of this
    instance's shape, each with its ``g2_early`` / ``g2_early_after``.
    Splitting stops at COMPACT_MAX_KERNELS kernels; a row past it keeps the
    setting of its upper bound."""
    rows = G2_EARLY_TILES.get(
        (config.hidden_dim, config.geometry.inter_dim), G2_EARLY_TILES[None]
    )
    bounds = [
        (None if upto is None else config.world_size * int(upto), int(ke))
        for upto, ke in rows
    ]

    def ke_at(recv):  # the setting of a step whose receive bound is recv
        for hi, ke in bounds:
            if hi is None or recv <= hi:
                return ke
        return 0

    out = []
    for lo, hi, tile, ctas in variants:
        cuts = [b for b, _ in bounds if b is not None and lo < b < hi]
        if len(out) + len(cuts) + 1 > COMPACT_MAX_KERNELS:
            cuts = []
        for edge in [*cuts, hi]:
            t = dict(tile, g2_early=ke_at(edge), g2_early_after=G2_EARLY_AFTER)
            out.append((lo, edge, t, ctas))
            lo = edge
    return out


#: compact STREAM: the dispatch is the
#: S1F producer pipeline inside the kernel. Every CTA quantizes its tokens
#: into local wire rows, ``STREAM_PRODUCERS`` CTAs ship them a run (<= 8 rows
#: of one destination flag tile) at a time and publish a flag per (flag tile,
#: source), and each GEMM item waits only on its own m-tile's flags.
STREAM = True
STREAM_PRODUCERS = 128
STREAM_RUN_ROWS = 8
#: STREAM's large-tile stages by (hidden, inter): DSV4.1 Flash takes a third
#: GEMM2 stage, DSR1 (and the rest) a third GEMM1 stage only.
STREAM_TILES = {
    (5120, 2304): dict(num_buffers=3, num_buffers2=3),  # noqa: C408
    None: dict(num_buffers=3),  # noqa: C408
}


def _pow2_at_least(n: int, lo: int) -> int:
    p = lo
    while p < n:
        p *= 2
    return p


def stream_flag_m(tile_m: int) -> int:
    from ..s1f_stage1 import s1f_flag_m

    return s1f_flag_m(tile_m)


def stream_reason(config, tile, ctas_per_cu) -> str | None:
    """Why this compact kernel cannot run STREAM, or None if it can."""
    from ..s1f_stage1 import s1f_producer_lds

    if not STREAM:
        return "STREAM is off"
    npes, tm = config.world_size, int(tile["tile_m"])
    fm = stream_flag_m(tm)
    if tm % fm or fm % STREAM_RUN_ROWS or (tm // fm) * npes > WAVE or npes > 16:
        return f"tile_m={tm} / {npes} ranks: flag tiles do not fit one poll"
    row = wire_row(config)
    if row > 0xFFFF or config.geometry.max_tokens_per_rank > 0xFFFF:
        return "the producer gathers 16-bit rows of a 16-bit width"
    lds = s1f_producer_lds(run_rows=STREAM_RUN_ROWS, row_stride=row)["end"]
    lds += stream_table_lds(config)
    if int(ctas_per_cu) * ((lds + 4096 + 2047) // 2048 * 2048) > 327680:
        return f"the producer ring ({lds} B) does not fit {ctas_per_cu} CTAs per CU"
    return None


def stream_npos(config) -> int:
    """(destination, local expert) positions of the run order, padded to a
    power of two of at least one per thread."""
    return _pow2_at_least(config.world_size * config.experts_per_rank, 128)


def stream_table_lds(config) -> int:
    """Bytes of a producer's LDS copy of the segment table."""
    return (4 * stream_npos(config) + 32) * 4


def stream_layout(*, npes, experts_per_rank, max_tok_per_rank, row, tile_cap):
    """Byte offsets of STREAM's local scratch: the send list (route ids,
    segment s at s * max_tok_per_rank), the segment table (run starts, row
    starts, rows and segment per run-order position, then the run count),
    the per-flag-tile source masks and the wire rows."""
    nseg = int(npes) * int(experts_per_rank)
    npos = _pow2_at_least(nseg, 128)

    def al(x):
        return (x + 255) // 256 * 256

    send = 0
    segt = al(send + nseg * int(max_tok_per_rank) * 4)
    tsrc = al(segt + (4 * npos + 32) * 4)
    wire = al(tsrc + int(tile_cap) * 4)
    return {
        "send": send,
        "send_n": nseg * int(max_tok_per_rank),
        "segt": segt,
        "npos": npos,
        "tsrc": tsrc,
        "wire": wire,
        "bytes": al(wire + int(max_tok_per_rank) * int(row)),
    }


def stream_flag_rel(*, npes, experts_per_rank) -> int:
    """STREAM's flag tiles start this far into compact's control region."""
    b = ctl_arena_bytes(npes=npes, experts_per_rank=experts_per_rank)
    return (b + 4095) // 4096 * 4096


def stream_tile_cap(rows: int, tile_m: int) -> int:
    fm = stream_flag_m(tile_m)
    return (int(rows) + fm - 1) // fm


def compact_ctl_region(i: int) -> str:
    """The arena control region of compact's i-th kernel."""
    return "compact_ctl" if i == 0 else f"compact_ctl{i}"


def compact_caps(config, recv_bound: int, tile_m: int) -> int:
    """Compact rows a step with this receive bound may fill: the staged
    plan's capacity at this compact kernel's alignment."""
    from ..compact_plan import compact_row_capacity

    return compact_row_capacity(
        max_recv=int(recv_bound),
        topk=config.topk,
        experts_per_rank=config.experts_per_rank,
        tile_m=int(tile_m),
    )


def compact_regions(config) -> dict:
    """The arena regions compact needs ({} when it is off): a control region
    per kernel, and the staged compact rows / rowmap at least as large as the
    widest kernel's alignment needs (merged with the staged sizes)."""
    if compact_reason(config) is not None:
        return {}
    variants = compact_variants(config)
    caps = max(compact_caps(config, hi, t["tile_m"]) for _, hi, t, _ in variants)
    rows, rowmap = compact_row_regions()
    regions = {rows: caps * wire_row(config), rowmap: (caps + 1) * 8}
    for i, (_, hi, t, ctas) in enumerate(variants):
        nb = ctl_arena_bytes(
            npes=config.world_size, experts_per_rank=config.experts_per_rank
        )
        if stream_reason(config, t, ctas) is None:
            # + STREAM's flag tiles (one FLAG_STRIDE slot each).
            nb = stream_flag_rel(
                npes=config.world_size, experts_per_rank=config.experts_per_rank
            ) + 256 * stream_tile_cap(
                compact_caps(config, hi, t["tile_m"]), t["tile_m"]
            )
        regions[compact_ctl_region(i)] = nb
    return regions


def compact_row_regions() -> tuple:
    """compact's wire rows and rowmap: the staged path's."""
    return "disp_out", "ep_rowmap"


def wire_row(config) -> int:
    h = config.hidden_dim
    return (h // 2 + h // 32 + 127) // 128 * 128


def slots_regions(config) -> dict:
    """The arena regions the slots megakernel adds ({} when it is off)."""
    if slots_reason(config) is not None:
        return {}
    epr, cap = config.experts_per_rank, slots_cap(config)
    return {
        "slots_rows": epr * cap * wire_row(config),
        "slots_rowmap": epr * cap * 8,
        "slots_ctl": ctl_arena_bytes(npes=config.world_size, experts_per_rank=epr),
    }
