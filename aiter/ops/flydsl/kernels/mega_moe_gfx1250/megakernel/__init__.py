# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MegaMoE megakernel: quant, dispatch, GEMM1, GEMM2 and combine as ONE kernel
(gfx1250).

The single-kernel counterpart of the staged compact path (plan, quant,
dispatch, GEMM1, GEMM2 + EP scatter, combine sync, reduce: seven launches),
after the gfx950 small-M MegaMoE (ROCm/aiter#6881). Two layouts share one
protocol: slots (fixed per-expert slots, for small steps) and compact (the
staged compact layout, up to 16k tokens per rank). Every CTA of an
all-resident grid of 4-wave CTAs runs the same phases:

1. DISPATCH, slots. One wave per local route (token, k) reserves a slot of the
   destination expert with one SCOPE_SYS atomic on that rank's
   ``running[parity][expert]``, quantizes the token's row to the fp4 wire in
   registers (the staged quant's bytes) and stores the row and its
   ``(meta, w)`` rowmap pair into the slot. The waves completing this rank's
   slot atomics, then its row stores, publish ``reserved[rank]`` and
   ``src_done[rank]`` to every destination.
2. PLAN. CTA 0 alone polls the peers' flags, reads the expert counts and
   publishes them in ``PLAN_REPS`` replicas; each CTA lays the m-tiles
   (expert-major, ``tile_m`` rows) out in LDS from its replica. CTA 0 then
   opens the rows-landed gates.
3. GEMM. One dynamic queue: every GEMM1 item ``(m, n)``, then every GEMM2
   item, the bodies of ``build_gemm_a8w4_tdm_task``. GEMM1 (act + fp4 quant
   into a2) waits for the rows-landed gate and publishes ``g1_ready[m]`` once
   an m-tile's items are done; GEMM2 waits for ``g1_ready`` and scatters each
   weighted row into its token owner's ``comb_inp`` slot.
4. CLOSE. The count completing this rank's GEMM2 units stores the generation
   into every peer's ``comb_flag[rank]``.
5. REDUCE. Once every peer closed, each token's top-k slots are summed in k
   order in fp32, the staged reduce's arithmetic: the output is bitwise the
   staged one.

compact replaces 1. with a count-first dispatch
on the staged compact layout (expert-major, each expert padded to ``tile_m``,
sources in rank order), so its rows, m-tiles and GEMMs are the staged path's:

1a. TALLY. CTAs 1..T count chunks of this rank's routes per segment (a
    destination rank's expert) and take each route's rank in its segment; the
    last tally CTA sends the segment totals, tagged with the launch's 16-bit
    tag, into every peer's X1 row for this source.
1b. LAYOUT. CTA 0 waits until every X1 row carries this launch's tag, lays
    every destination's segments out (all ranks compute the same layout),
    publishes this rank's send bases and opens the send gates.
1c. SEND. One wave per token quantizes its row and stores it, with its rowmap
    pair, at each route's row.

Forward progress: dispatch waits only on the peers' tallies (compact); the
plan waits only on dispatch; GEMM1 waits on nothing else; GEMM2 waits only on
GEMM1 items already claimed; the reduce waits on the peers' closes. No other
kernel may hold CUs while it runs; a wait traps after ``TIMEOUT_S``.

Cross-launch reuse: rank s writes rank d's rows, rowmap and counters in launch
L+1 only after its launch L ended, which needed d's close of L, which comes
after d's last read of them in L. ``comb_inp`` rows on s are written in L+1
only for tokens s produced in L+1. compact sends to d only once d's X1 row of
the same launch arrived; the staged path, which shares ``disp_out``,
``ep_rowmap`` and ``comb_inp``, waits for d's counts likewise.

Modules: ``config`` (tiles, switches, admission, arena regions), ``protocol``
(control block, entry ticket, waits and gates, plan, GEMM queue, close,
reduce), ``dispatch_slots`` / ``dispatch_compact`` (phase 1. of each layout),
``kernel`` (make_megakernel) and ``host`` (the MegaKernel classes).
"""

from .config import (
    compact_reason,
    compact_regions,
    compact_variants,
    read_megakernel,
    slots_reason,
    slots_regions,
)
from .host import (
    CompactMega,
    CompactMegaSet,
    MegaKernel,
    SlotsMega,
)
from .kernel import (
    make_megakernel,
)
from .protocol import (
    TR_N,
    ctl_arena_bytes,
    ctl_words,
    epr_pad,
)

__all__ = [
    "TR_N",
    "CompactMega",
    "CompactMegaSet",
    "MegaKernel",
    "SlotsMega",
    "compact_reason",
    "compact_regions",
    "compact_variants",
    "ctl_arena_bytes",
    "ctl_words",
    "epr_pad",
    "make_megakernel",
    "read_megakernel",
    "slots_reason",
    "slots_regions",
]
