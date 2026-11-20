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
    return tile


def read_megakernel() -> bool:
    return os.environ.get("AITER_MEGAKERNEL", "0").strip() in ("1", "true", "True")


def slots_reason(config) -> str | None:
    """Why this MegaMoE cannot run the slots megakernel, or None if it can."""
    if not read_megakernel() or SLOTS_TOKENS <= 0:
        return "AITER_MEGAKERNEL is off"
    return _mega_reason(config, slots_tile(slots_tokens(config)))


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
    return min(SLOTS_TOKENS, int(config.geometry.max_tokens_per_rank))


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
    for upto, tile, ctas in COMPACT_TILES:
        if lo >= top:
            break
        hi = top if upto is None else min(top, config.world_size * int(upto))
        if hi > lo:
            out.append((lo, hi, dict(tile), int(ctas)))
            lo = hi
    return out


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
    for i in range(len(variants)):
        regions[compact_ctl_region(i)] = ctl_arena_bytes(
            npes=config.world_size, experts_per_rank=config.experts_per_rank
        )
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
