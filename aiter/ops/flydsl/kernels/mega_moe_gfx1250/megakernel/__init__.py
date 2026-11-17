# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MegaMoE megakernel: quant, dispatch, GEMM1, GEMM2 and combine as ONE kernel
(gfx1250).

The single-kernel counterpart of the staged compact path (plan, quant,
dispatch, GEMM1, GEMM2 + EP scatter, combine sync, reduce: seven launches),
after the gfx950 small-M MegaMoE (ROCm/aiter#6881), in two layouts over one
core: slots (fixed per-expert slots, for small steps) and compact (the staged
compact layout, up to 16k tokens per rank). Every CTA of an all-resident grid
of 4-wave CTAs runs the same phases (1. is the slots front end):

1. PRODUCE. One wave per local route (token, k). Lane 0 reserves a slot of the
   destination expert with one SCOPE_SYS buffer atomic on that rank's
   ``running[parity][expert]`` (expert ``e`` owns rows ``[e * cap, (e + 1) *
   cap)`` of ``slots_rows``; ``cap`` covers every token of every rank); the
   token's bf16 row, loaded first, is quantized to the fp4 wire in registers
   meanwhile (per-1x32 MX with the HIP quant's RoundUp scale and amax floor,
   so the bytes match the staged wire); the row and its ``(meta, w)`` rowmap
   pair then go straight into the slot with SCOPE_SYS stores. Once every slot
   atomic of this rank has returned, the wave completing that count stores the
   generation into every destination's ``reserved[rank]`` (its counters from
   this rank are final); once every row has retired, the wave completing that
   count stores it into ``src_done[rank]``. A step of at least
   ``PREF_MIN_TOK`` tokens then has every CTA but 0 read the W1 slice its
   first item most likely needs into cache (HBM is idle until the plan).
2. PLAN. CTA 0 alone polls the peers' ``reserved`` (hundreds of CTAs polling
   one uncached line starve every access queued behind it), reads the expert
   counts and publishes them in ``PLAN_REPS`` replicas, each dword tagged with
   the generation in one 64-bit store (16-row tiles) or followed by every
   CTA's own gate line. Each CTA lays the m-tiles (expert-major, ``tile_m``
   rows) out in LDS from its replica. CTA 0 then polls the peers'
   ``src_done`` and opens the rows-landed gates.
3. GEMM. One dynamic queue: every GEMM1 item ``(m, n)``, then every GEMM2
   item. A CTA's first item is static, later ones come from one agent claim
   each. GEMM1 is the stage-1 body of ``build_gemm_a8w4_tdm_task`` (act + fp4
   quant into a2, stored SCOPE_DEV); its A owners wait for the rows-landed
   gate while its B owners stream W1, and the item completing an m-tile's
   ``g1_cnt`` publishes ``g1_ready[m] = gen``. A GEMM2 item's A owners wait for
   ``g1_ready`` while its B owners stream W2; its EP-scatter epilogue writes
   each weighted row into its token owner's ``comb_inp`` slot with SCOPE_SYS
   gather stores. All GEMM1 items are claimed before any GEMM2 item.
4. CLOSE. Each CTA counts its GEMM2 units (CTA 0 one more, for resetting the
   next launch's counters); the count completing the total stores the
   generation into every peer's ``comb_flag[rank]``.
5. REDUCE. CTA 0 polls the peers' ``comb_flag`` and opens the gates; each CTA
   then sums its tokens' top-k slots in k order in fp32, the staged reduce's
   arithmetic, so the output is bitwise the staged one.

compact (``layout="compact"``, AITER_MEGAKERNEL_COMPACT) replaces 1. with a count-first
front end on the staged compact layout (expert-major, each expert padded to
``tile_m``, sources in rank order inside it), so its rows, m-tiles and GEMMs
are the staged path's, without per-expert slots sized for every token:

1a. TALLY. CTAs 1..T each count a chunk of this rank's routes in an LDS
    histogram, then take one device atomic per (CTA, segment) for the chunk's
    base in the segment (a segment is a destination rank's expert); each
    route's (segment, rank in segment) goes to a route table. The last tally
    CTA sends the segment totals, every dword tagged with the launch's 16-bit
    tag, into each peer's X1 row for this source.
1b. LAYOUT. CTA 0 polls the X1 rows until every dword carries this launch's
    tag, lays every destination's segments out from the matrix (all ranks
    compute the same layout), keeps this rank's receive counts, publishes its
    send bases in ``N_SREP`` replicas and opens every CTA's send gate.
1c. SEND. One wave per token quantizes the row in registers, waits for its
    CTA's send gate and stores the row and its rowmap pair at each route's row.
    Then 2. to 5., the plan from the receive counts CTA 0 already holds.

Forward progress: producers wait on nothing (compact's on CTA 0's layout, which
waits only on the peers' tallies); the plan waits only on producers (all ranks
run theirs first); a GEMM1 item waits on nothing; a GEMM2 item waits only on
GEMM1 items already claimed; the reduce waits on the peers' closes, which
depend only on the above. All CTAs must be resident at once (the grid is the CU
count times ``SLOTS_CTAS_PER_CU`` for slots, the ``COMPACT_TILES`` row's CTAs per CU
for a compact kernel, and no other kernel may hold CUs). A wait
traps after ``TIMEOUT_S``: a peer's launch may start seconds late.

Cross-launch reuse: rank s writes d's ``slots_rows`` / ``slots_rowmap`` /
``running`` in launch L+1 only after s's launch L ended, which needed d's close
of L, which comes after every read d made of them in L (and after CTA 0 reset
the counters launch L+1 uses). Rows of ``comb_inp`` on s are written in L+1
only for tokens s produced in L+1, i.e. after s's reduce of L. compact sends to
d only once d's X1 row of the same launch arrived, i.e. once d's previous
launch ended; that holds whichever path (compact or staged, which share
``disp_out``, ``ep_rowmap`` and ``comb_inp``) ran it, as the staged dispatch
likewise waits for d's counts.

Start handshake (off by default, ``AITER_DMK_NO_HS=0`` turns it on): slots
then sends only once every peer's ``started`` flag carries this launch's
generation, which CTA 0 polls and passes on through each CTA's send gate. It
was added when writes that reached a rank still between two launches came out
corrupted now and then (about 1 in 400 calls had one token wrong). That no
longer reproduces: without it, ~72k calls with up to 1 ms of random launch skew
between ranks and staged calls interleaved matched the staged path bit for
bit (4x gfx1250), and it costs ~5 us per launch. compact needs no such step:
its rows go out after every peer's X1.

Generation and parity come from u64 entry tickets (``G`` per launch over
``N_TICKET`` counters): every rank launches once per layer, so they agree.

Modules: ``config`` (tiles, switches, admission, arena regions), ``core``
(control block, entry ticket, waits and gates, plan, GEMM queue, close,
reduce), ``frontend_slots`` / ``frontend_compact`` (phase 1. of each layout),
``kernel`` (make_megakernel) and ``host`` (the MegaKernel classes).
"""

from .config import (
    compact_reason,
    compact_regions,
    compact_tile_table,
    compact_variants,
    read_compact,
    read_slots_tokens,
    slots_reason,
    slots_regions,
)
from .core import (
    TR_N,
    ctl_arena_bytes,
    ctl_words,
    epr_pad,
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

__all__ = [
    "TR_N",
    "CompactMega",
    "CompactMegaSet",
    "MegaKernel",
    "SlotsMega",
    "compact_reason",
    "compact_regions",
    "compact_tile_table",
    "compact_variants",
    "ctl_arena_bytes",
    "ctl_words",
    "epr_pad",
    "make_megakernel",
    "read_compact",
    "read_slots_tokens",
    "slots_reason",
    "slots_regions",
]
