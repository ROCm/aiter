# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GEMM1 + GEMM2 + ReduceScatter in one kernel launch.

:mod:`.stage2_rs` folds the ReduceScatter into GEMM2. This goes one step
further and hosts *both* GEMMs, so the tail of the TP MoE layer collapses from

    gemm1 -> gemm2 -> rs_publish -> rs_pull        (4 launches)

to a single one. Neither GEMM is reimplemented: both come from the
``_composition`` hook on their own compiler
(:func:`~..mxfp4_gemm1.compile_gemm1_a4w4_port` and
:func:`~..mxmoe_dispatcher.compile_gemm2_a4w4_port`), so the tuned tiles are
emitted verbatim.

    one CTA per sort block:
        its ``g1_n_blocks`` GEMM1 tiles   -> that block's FP4 intermediate
        -- s_waitcnt + workgroup barrier --
        its ``G2_N_BLOCKS`` GEMM2 tiles   -> arena partial
    tail      the shared ReduceScatter tail from :mod:`.rs_tail`

No grid-wide barrier: GEMM2's row block reads only the GEMM1 output of the same
row block, because each GEMM1 n-tile pairs its gate slice with the matching up
slice and writes ``BN//2`` intermediate columns, so one row block's tiles cover
its intermediate exactly. Dropping the barrier also drops the requirement that
every CTA be resident, which had capped the launch at one CTA per CU; GEMM1 uses
32.5 KB of LDS at BM32 against 160 KB per CU on gfx950, so that cost a factor of
four in occupancy against the standalone kernels.

Why this pair and not any pair
------------------------------
The two GEMMs must agree on a block size, because one kernel has one. GEMM2 is
always 256 threads; GEMM1 is ``num_waves * k_wave * 64``, which is 256 exactly
when ``num_waves=4, k_wave=1`` -- the BM16 inline-quant rows small ``M`` tunes
onto. :func:`stage12_supported` checks that rather than assuming it. LDS is the
union of the two (16640 B and 8192 B for kimi3 BM16), allocated once and handed
to both, since the phases are disjoint.

The handoff is a bare ``s_waitcnt`` plus a workgroup barrier. The agent-scope
release/acquire the two-phase version needed was there because MI355X L2 is
per-XCD and a GEMM2 tile could land on a different XCD than the GEMM1 tile that
fed it; with producer and consumer now in the same CTA that cannot happen.
"""

# NOTE: no ``from __future__ import annotations`` here. It would turn the
# ``@fx.struct`` field annotation below into a string, and FlyDSL resolves those
# eagerly -- the failure is "type fx.Array[...] does not implement the Storable
# protocol". The other kernel modules omit it for the same reason.

import functools
import hashlib
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int8, T

from ..mxfp4_gemm1 import compile_gemm1_a4w4_port
from ..mxfp4_gemm_common import _udiv, _umod, global_typed_ptr
from .. import communication_ops_utils as comm
from .p2p import desc_slot
from ..mxmoe_dispatcher import compile_gemm2_a4w4_port
from ..tensor_shim import _run_compiled as run_compiled, buf_copy_atom, ptr_buf_tensor
from .reduce_scatter import MAX_SERVICE_BLOCKS, RS_UNIT_ELEMS
from .rs_tail import (
    emit_phase_barrier,
    emit_rs_tail,
    read_epoch,
    rs_service_low as _rs_service_low,
    rs_tail_slots,
)

__all__ = [
    "compile_stage12_rs",
    "hosts_reduce_as_atomic",
    "run_stage12_rs",
    "stage12_supported",
]

_BLOCK = 256
_SERVICE_BLOCKS = min(
    MAX_SERVICE_BLOCKS, int(os.environ.get("AITER_TP_STAGE12_RS_SERVICE", "128"))
)
#: Override the CTA count. The default is one CTA per sort block; this exists to
#: A/B that against the old persistent grid, which was pinned to the CU count
#: because the inter-phase barrier required every CTA resident.
_GRID_CU = int(os.environ.get("AITER_TP_STAGE12_GRID", "0"))
#: Cap the grid when the AllGather is hosted, in CTAs. 0 = no cap.
#:
#: With the AG inside, every CTA that is not pushing spins at the gate and
#: cannot retire until the cross-rank AllGather completes. If the grid exceeds
#: what the device holds at once, the surplus CTAs are queued *behind* those
#: spinners -- they cannot start, and nothing they would do can overlap the
#: collective. Capping the grid at residency removes the queue; the loop is
#: grid-strided so correctness does not depend on the count.
_AG_GRID_CAP = int(os.environ.get("AITER_TP_MEGA_AG_GRID_CAP", "0"))
#: Minimum waves per EU to compile for, i.e. a VGPR cap. 0 leaves it to the
#: register allocator.
#:
#: This kernel's register budget is the *peak* over quantize, push, A-scale
#: shuffle, both GEMMs and the ReduceScatter tail, and its LDS is
#: ``max(g1, g2)`` -- so GEMM2 runs at GEMM1's footprint rather than its own.
#: Left to itself the allocator sometimes lands well below what the GEMMs want.
#:
#: 2 is a net win over the whole sweep (mean 1.128 -> 1.136, median 1.076 ->
#: 1.183, 14 losses -> 11) but it is really a *per-shape* choice -- dsv4 gains
#: 15-17% at M=64..2048 and glm5 2-10%, while kimi3 and dsv3 give back 0.4-4.5%
#: at small M -- so it is also a tuned axis, carried on the plan and passed as
#: ``waves_per_eu``. This env value is only the fallback for an untuned shape.
#: 3 and 4 are worse everywhere (spills), so the axis is two-valued.
_WAVES_PER_EU = int(os.environ.get("AITER_TP_MEGA_WAVES_PER_EU", "2"))
#: Hard VGPR cap handed to the backend as ``--amdgpu-num-vgpr``. 0 leaves it
#: to the allocator. Not in the kernel name -- it rides ``compile_hints``,
#: which the disk cache key does not cover, so A/B it with the cache wiped.
_MAXNREG = int(os.environ.get("AITER_TP_MEGA_MAXNREG", "0"))
#: Skip a phase of the merged kernel. WRONG RESULTS; a timing probe only.
#: "g1" / "g2" / "both" drop those tiles, so the difference against a full run
#: is that phase's cost *inside this kernel* -- which is what has to be
#: compared against the standalone kernel's, not the total. Use "both" to price
#: GEMM1: "g1" on its own leaves GEMM2 reading uninitialised E8M0 scales and
#: the run dies, so take (skip g2) - (skip both) instead.
#:
#: This is how the large-M deficit was pinned down. At M=8192, GEMM2 costs
#: **1.6-2.8x inside this kernel** what it costs as its own launch:
#:
#:     kimi3  947.5 us in-kernel vs 410.7 standalone  (2.31x)
#:     dsv4   890.8            vs 552.2              (1.61x)
#:     glm5   994.2            vs 355.8              (2.79x)
#:
#: and that excess (537 / 339 / 638 us) is most of the whole gap against the
#: split path (817 / 500 / 855 us).
#:
#: **Both** GEMMs are inflated, not just GEMM2 -- G1 1.38-1.79x, G2 1.62-2.78x
#: -- so whatever it is, it is a whole-kernel effect. Four candidates were
#: measured; all four are too small, and each is recorded because each looks
#: like the answer until it is measured:
#:
#:   * *LDS occupancy.* Forcing GEMM1 to a 19 KB tile (8 CTA/CU) instead of
#:     65 KB (2 CTA/CU) -- a 4x swing -- moved GEMM2 in-kernel by **9%**
#:     (891.6 -> 812.9 us), ratio still ~2x, and the total got worse because
#:     GEMM1 degrades at BM32.
#:   * *L2 swizzle.* The merged kernel cannot use GEMM2's `spart` partitioning
#:     (`stage12_supported` rejects it). Disabling `MXFP4_G2_SPART` on the
#:     standalone kernel costs it **0-4%**.
#:   * *Register spill.* The merged kernel does spill (256 VGPR, 172 B scratch,
#:     against 230/0 for GEMM1 and 128/0 for GEMM2 standalone). But the
#:     BM32/BN128 variant compiles to 229 VGPR and **0 scratch** and still
#:     shows 2.03x.
#:   * *n-tile serialisation.* GEMM2's tiles run back to back in one CTA with a
#:     `gpu.barrier()` between them, so nothing overlaps. Halving the tile count
#:     (BN 256 -> 512, 14 -> 7 blocks) is worth **13%** (893.0 -> 775.6, ratio
#:     2.22 -> 1.93). Real, but not the bulk.
#:
#: What is left, and what fits all four negatives: the producer-consumer
#: round trip. A CTA writes its m-block's FP4 intermediate to global, drains it
#: with `s_waitcnt vmcnt(0)`, then reads it straight back for GEMM2. In the
#: split path that handoff is a kernel boundary and GEMM2 streams an
#: already-settled buffer. Here it is a full memory round trip per m-block with
#: nothing to overlap it, because a CTA holds one m-block at a time.
#: Untested fix for that: give each CTA two m-blocks and software-pipeline
#: them -- GEMM1(A), GEMM1(B), one drain, GEMM2(A), GEMM2(B) -- so B's compute
#: covers A's drain. Local to this file. Until then `mega=0` at M>=2048.
_SKIP = os.environ.get("AITER_TP_MEGA_SKIP", "")
#: Probe: drop the AllGather front (push + barrier + gate). WRONG RESULTS;
#: prices it in-kernel against the standalone push kernel.
_SKIP_AG = os.environ.get("AITER_TP_MEGA_SKIP_AG", "0") == "1"
#: Probe: keep the AllGather's push but drop its cross-rank barrier and gate.
#: WRONG RESULTS. Bisects the AG for the >1024 fault (v33 showed SKIP_AG=1
#: makes glm5/2048 pass, so the fault is somewhere in this phase): if the push
#: alone still faults it is in the data path, if it stops the fault is in the
#: rendezvous. Compile-time, so the default build is byte-identical.
_SKIP_AG_SYNC = os.environ.get("AITER_TP_MEGA_SKIP_AG_SYNC", "0") == "1"
#: Probe: divide the row count handed to the AllGather push. WRONG RESULTS.
#: Narrows the >1024 fault (v39 put it in the push, not the rendezvous) by
#: shrinking the range the push walks: 1 is the real value, 2 halves it. At
#: glm5/2048 the real count is 256 rows per rank and the working 1024 case is
#: 128, so a divisor of 2 asks "is it the row count".
_AG_ROWS_DIV = int(os.environ.get("AITER_TP_MEGA_AG_ROWS_DIV", "1"))
#: Probe: start the AllGather push at the *upper* half of its unit range,
#: leaving the row count (and therefore every buffer bound) at its real value.
#: WRONG RESULTS. This is the one probe that separates "the high addresses are
#: bad" from "the sheer number of units is": ``_AG_ROWS_DIV=2`` shrinks the
#: loop *and* the buffers, so it cannot tell them apart.
_AG_PUSH_UPPER = os.environ.get("AITER_TP_MEGA_AG_PUSH_UPPER", "0") == "1"
#: Probe: drop the ReduceScatter tail. WRONG RESULTS; prices the tail against
#: the standalone RS kernel's device time.
_SKIP_TAIL = os.environ.get("AITER_TP_MEGA_SKIP_TAIL", "0") == "1"
#: Probe: drop the ``s_waitcnt`` that drains GEMM1's intermediate stores before
#: GEMM2 reads them back. WRONG RESULTS; it prices the producer-consumer round
#: trip, which is the last structural difference against the split path that
#: has never been measured on its own. Software-pipelining two m-blocks was the
#: other way to attack it and is already known to be worse (see the comment at
#: the handoff), so this probe only answers "is the drain worth attacking".
_SKIP_DRAIN = os.environ.get("AITER_TP_MEGA_SKIP_DRAIN", "0") == "1"
#: Run the expert sort inside the merged kernel instead of on the host.
#:
#: The point is not the saved launches. With the sort on the host it sits
#: *between* the route AllGather and this kernel, so the route has to be pushed
#: on its own and the activation pushed again here -- two cross-rank
#: rendezvous where the unfused path needs one, measured at +33 us on kimi3
#: M=8 (see ``opt_0921_v13.txt``). With the sort in here, both can ride one
#: push and one gate.
#:
#: Only the ``oneshot`` sorter is inlinable: it does the whole sort in block 0
#: out of LDS. That caps it at ``ONESHOT_MAX_T`` global tokens -- 16 for every
#: model here -- which is exactly where the fused path is furthest behind.
_FUSE_SORT = os.environ.get("AITER_TP_MEGA_FUSE_SORT", "0") == "1"
#: Token bound the inlined sort is sized for. The oneshot sorter is only chosen
#: up to ``ONESHOT_MAX_T`` global tokens (16 for every model here), and its LDS
#: mesh is ``sub_tokens x (E+1)``, so sizing it for more would cost LDS that can
#: never be used.
_SORT_MAX_T = 16
#: Bump when the inlined-sort source changes.
#:
#: The kernel name is the module cache key, and the name's hash covers the env
#: knobs -- not this file's contents. Editing the sort wiring without changing
#: any knob therefore re-runs the *previous* binary, which reads as "the fix
#: changed nothing" with a byte-identical number.
_SORT_REV = int(os.environ.get("AITER_TP_MEGA_SORT_REV", "8"))
#: Copy the ``sorted_ids`` this kernel produced into the debug sink, so the host
#: can diff it against the reference sort. Three rounds of inferring what the
#: kernel wrote from rel_l2 each got part of it wrong; this reads it directly.
#: Let the host sort take its fast path, and derive ``m_indices`` in here.
#:
#: ``moe_sorting`` only dispatches to the FlyDSL sorter when no aux outputs are
#: asked for. The merged path asks for two -- ``m_indices`` and
#: ``reverse_sorted`` -- and so lands on the slower opus/CK implementation. But
#: it needs neither: ``reverse_sorted`` is only read by the non-atomic scatter
#: path, which this kernel does not take, and ``m_indices`` is exactly
#: ``sorted_ids & 0xFFFFFF``. Measured cost of that choice: 104.5 us of the
#: 170.7 us layer at kimi3 M=8 -- more than the whole split path.
_FAST_SORT = os.environ.get("AITER_TP_MEGA_FAST_SORT", "0") == "1"
#: Count what the merged kernel actually executed, into the debug sink.
#:
#: Slot 0: m-block iterations. Slot 1: GEMM1 tiles. Slot 2: GEMM2 tiles.
#:
#: Needed because neither accuracy nor the reported ``stage12`` mode is
#: evidence of execution: this path falls back silently, and `stage12_mode`
#: reports the *decision*. `AITER_TP_MEGA_SKIP=both` emitted no tiles at all and
#: still passed the 0.06 gate -- which invalidated four rounds of "the GEMMs are
#: free" before anyone noticed.
#: Double-buffer GEMM1's LDS across n-tiles.
#:
#: The tile loop currently puts a full `gpu.barrier()` between every tile,
#: because consecutive tiles share one LDS region. That barrier is what stops
#: tile n+1's weight fetch from overlapping tile n's MFMA, and the merged
#: kernel reaches only ~19% of peak HBM bandwidth against ~40% for the
#: standalone pair (kimi3 M=8: 264 MB of expert weights in 170 us vs 83 us).
#: With two regions the tiles no longer alias, so only the first tile of each
#: m-block needs to synchronise.
_DBUF = os.environ.get("AITER_TP_MEGA_DBUF", "0") == "1"
#: Bump when the double-buffering wiring changes. The kernel name's hash covers
#: this file's env knobs, not the GEMM emitters' source -- so editing
#: ``mxmoe_dispatcher.py`` without bumping this re-runs the previous binary.
_DBUF_REV = int(os.environ.get("AITER_TP_MEGA_DBUF_REV", "8"))
#: Give GEMM1 and GEMM2 their own LDS regions instead of aliasing one.
#:
#: `stage12_supported`'s BM16 exclusion names this aliasing as the suspect for
#: an unexplained numerical failure ("this kernel reuses one LDS region across
#: phases ... neither of which the standalone pair does"). Separating them is
#: worth testing on its own -- it removes the cross-phase dependency that forces
#: the drain between GEMM1 and GEMM2 -- and if it also makes BM16 correct, that
#: halves the decode row padding.
#:
#: Costs another 2x LDS on top of `_DBUF`: ~32 KB -> ~128 KB, which fits
#: gfx950's 160 KB but drops occupancy to one CTA per CU.
_SPLIT_LDS = os.environ.get("AITER_TP_MEGA_SPLIT_LDS", "0") == "1"
#: Replace "one CTA owns one m-block through both GEMMs" with a task queue.
#:
#: The serial chain per CTA today is every GEMM1 tile of its block followed by
#: every GEMM2 tile, and the grid is only as wide as the m blocks -- 128 CTAs on
#: 256 CUs at kimi3 M=8. Flattening the tile space alone is *worse* (measured:
#: 128 -> 256 -> 512 CTAs is 172.6 -> 215.9 -> 261.4 us) because it needs two
#: grid-wide phase barriers. A queue keeps the flattening but replaces those
#: barriers with a per-m-block dependency: block B's GEMM2 waits only on block
#: B's GEMM1, not on the whole grid.
#:
#: No live-block pool is needed here, unlike DeepGEMM's: the intermediate buffer
#: is already ``max_sorted`` rows, so every m block owns a slot and GEMM1 can
#: run arbitrarily far ahead.
_TASKQ = os.environ.get("AITER_TP_MEGA_TASKQ", "0") == "1"
#: Bump when the task-queue body changes; the kernel name's hash does not cover
#: this file's contents.
_TASKQ_REV = int(os.environ.get("AITER_TP_MEGA_TASKQ_REV", "8"))
#: Tasks claimed per atomic. One-at-a-time made every one of the ~2300 tasks
#: contend on a single counter and pay an LDS broadcast plus two workgroup
#: barriers, which cost more than the finer dependency saved (-19.8% at kimi3
#: M=8). Claiming a run of tasks amortises all three.
_TASKQ_BATCH = int(os.environ.get("AITER_TP_MEGA_TASKQ_BATCH", "1"))
#: CTA count for the task-queue form.
_TASKQ_GRID = int(os.environ.get("AITER_TP_MEGA_TASKQ_GRID", "512"))
#: Push the payload twice inside the kernel. The push is idempotent, so the
#: result is unchanged -- unlike a skip probe, this one can be run under the
#: 0.06 accuracy gate, and the delta is the in-kernel AllGather's cost.
_AGPUSH_TWICE = os.environ.get("AITER_TP_MEGA_AGPUSH_TWICE", "0") == "1"
_COUNT = os.environ.get("AITER_TP_MEGA_COUNT", "0") == "1"
_SORT_DEBUG = os.environ.get("AITER_TP_MEGA_SORT_DEBUG", "0") == "1"


#: Largest expert count the inlined sort is worth it for.
#:
#: Inlining forces the sort to this kernel's 256-thread block, and for
#: ``E > 256`` the prefix sum falls back to a serial extension in thread 0 that
#: is ``E - 256`` experts long. Measured at M=8/16, graph mode, against the same
#: kernel with the sort on the host:
#:
#:     dsv3   E=256  extend   0   +10.2% / +10.4%
#:     glm5   E=257  extend   1   +11.3% /  +8.3%
#:     dsv4   E=384  extend 128   +13.8% / +11.8%
#:     kimi3  E=896  extend 640    -2.1% / -12.3%
#:
#: so the crossover is between 128 and 640 experts of extension.
#:
#: Replacing that serial extension with a chunked parallel scan (round 28) got
#: kimi3 back to 167.5 / 232.6 -- +1.1% at M=8 but still -8.8% at M=16 -- so
#: the extension was not its only problem. Its sort LDS is 63 KB against
#: 18-27 KB for the others, which halves occupancy. Still gated out.
_SORT_MAX_E = int(os.environ.get("AITER_TP_MEGA_SORT_MAX_E", "512"))


#: Run the sort inside the merged kernel via the **multiphase** path.
#:
#: The oneshot sorter caps at 16 tokens because its LDS mesh is
#: ``sub_tokens x (E+1)``; the multiphase path keeps that mesh in HBM and needs
#: only counters in LDS, so it reaches decode. For ``T <= 2048`` it is exactly
#: two phases -- p0v2 (clear+scatter+count) and p23 (prefix-sum+scatter) --
#: which is why the bound below is 2048 and not the sorter's own limit.
#:
#: What this buys is not launches. With the sort on the host it sits *between*
#: the route AllGather and this kernel, so route and payload cannot share a
#: push: measured at 14-24% across the decode matrix (``opt_data_0921_v9``).
_FUSE_SORT_MP = os.environ.get("AITER_TP_MEGA_FUSE_SORT_MP", "0") == "1"
_SORT_MP_MAX_T = int(os.environ.get("AITER_TP_MEGA_SORT_MP_MAX_T", "2048"))


def kernel_sorts_mp(global_tokens: int, num_experts: int) -> bool:
    """Whether the merged kernel runs the multiphase sort for this shape."""
    return _FUSE_SORT_MP and int(global_tokens) <= _SORT_MP_MAX_T


def kernel_sorts(global_tokens: int, num_experts: int) -> bool:
    """Whether the merged kernel does the expert sort for this token count.

    Bounded by the oneshot sorter's LDS mesh, which is sized for
    ``_SORT_MAX_T`` tokens. Past that the standalone sort goes multiphase and
    there is nothing single-block to inline, so the host keeps sorting --
    silently running the oneshot body on more tokens than its mesh holds gives
    wrong results (dsv3 at 32 tokens: rel_l2 0.70).
    """
    return (
        _FUSE_SORT
        and int(global_tokens) <= _SORT_MAX_T
        and int(num_experts) <= _SORT_MAX_E
    )
#: Emit the n-block loops as runtime loops instead of ``range_constexpr``
#: unrolls.
#:
#: The unrolled form emits one full copy of the tile body per n block --
#: ``g1_n_blocks`` of GEMM1 and ``G2_N_BLOCKS`` of GEMM2. At kimi3 M=8192 that
#: is 3 and 14 copies, 2016 MFMA instructions against 448 + 48 for the two
#: standalone kernels, and 18k lines of ISA (~144 KB) against a 32 KB L1
#: instruction cache. Every tile is *different* code, so there is no
#: instruction reuse across them.
#:
#: The tile emitters already take a runtime block index -- the flattened
#: ``_GRID_CU`` branch passes one -- so rolling the loops costs nothing but the
#: loop overhead.
#:
#: The bound has to come from a *kernel argument*, not from the Python
#: constant. A ``range(fx.Int32(0), fx.Int32(14), fx.Int32(1))`` is folded and
#: fully unrolled again: with that form the emitted ISA was byte-identical to
#: the constexpr one (18100 lines, 2016 MFMA), which reads as "rolling does not
#: help" when in fact nothing rolled.
#: ``auto`` rolls once GEMM2 has at least ``_ROLL_MIN_NB`` n blocks. Measured
#: on decode with a wiped compile cache -- the earlier "rolling does not help"
#: reading was taken through a stale binary, see ``data_0921/opt_data_0921_v2``.
#: The gain tracks the copy count, which is what the I-cache argument predicts:
#:
#:     glm5  G2_N_BLOCKS=24   M=256 +3.3%  M=512 +3.8%  M=1024 +13.9%  M=2048 0%
#:     kimi3 G2_N_BLOCKS=14   M=512 +1.0%
#:     dsv3  G2_N_BLOCKS=14   M=1024 +0.1%
#:
#: It also lowers the register peak (glm5 M=512: 254 -> 229 VGPR) and the spill
#: (M=1024: 356 -> 228 bytes of scratch). 14 blocks is inside the noise either
#: way, so the threshold sits between the two families rather than at 1.
#: GEMM1's n-block loop is deliberately **not** rolled, unlike GEMM2's.
#:
#: Tried and refuted (2026-09-21). With only 2-3 n blocks a runtime index stops
#: the tile specialising -- the block number no longer folds into the addressing
#: -- and the loop-carried state costs registers rather than saving them:
#:
#:     glm5/1024   396.9 -> 534.0 us   scratch 228 -> 260   (0.646 -> 0.482)
#:     kimi3/512   473.5 -> 688.3 us   vgpr 233 -> 240      (0.771 -> 0.527)
#:
#: Worse, merely *having* the rolled branch in the source cost the same 35% even
#: with the knob off -- the nested tile helper and the early ``return`` inside
#: the traced ``_do_gemm1`` changed what was emitted for the unrolled path too.
#: So it is removed rather than left behind a default-off flag.
#:
#: Same shape as ``_ROLL``'s own threshold: rolling only pays once there are
#: enough copies for the I-cache to notice, hence ``G2_N_BLOCKS >= 16`` there.

_ROLL_ENV = os.environ.get("AITER_TP_MEGA_ROLL", "auto")
_ROLL_MIN_NB = int(os.environ.get("AITER_TP_MEGA_ROLL_MIN_NB", "16"))


def _roll_enabled(g2_n_blocks: int) -> bool:
    """Whether to roll GEMM2's n-block loop for this shape."""
    if _ROLL_ENV == "auto":
        return int(g2_n_blocks) >= _ROLL_MIN_NB
    return _ROLL_ENV == "1"
#: Run the in-kernel A-scale shuffle. Whether GEMM1 wants the gathered scale in
#: token order or in sorted+swizzled order decides this, and the two differ by a
#: whole kernel, so it is a knob until measured rather than an assumption.
_ASCALE_SHUFFLE = os.environ.get("AITER_TP_MEGA_ASCALE_SHUFFLE", "1") == "1"


def _mx_scale_shuffle_idx(scaleN_pad: int, x, y):
    """Byte offset of scale column ``y`` of sorted row ``x`` in GEMM1's layout.

    Port of ``aiter::mx_scale_shuffle_idx`` (csrc/include/mx_quant_utils.h).

    ``y`` may be a Python int or a runtime value; when it is an int the whole
    column term folds to a constant. Getting that wrong is silent -- the kernel
    still runs, it just scatters to the wrong bytes -- so the two forms are
    written out rather than left to duck typing.
    """
    row_term = (
        _udiv(x, fx.Int32(32)) * fx.Int32(scaleN_pad * 32)
        + _umod(x, fx.Int32(16)) * fx.Int32(4)
        + _udiv(_umod(x, fx.Int32(32)), fx.Int32(16))
    )
    if isinstance(y, int):
        return row_term + fx.Int32((y // 8) * 256 + (y % 4) * 64 + (y % 8) // 4 * 2)
    return (
        row_term
        + _udiv(y, fx.Int32(8)) * fx.Int32(256)
        + _umod(y, fx.Int32(4)) * fx.Int32(64)
        + _udiv(_umod(y, fx.Int32(8)), fx.Int32(4)) * fx.Int32(2)
    )


@flyc.jit
def emit_ascale_shuffle(
    arg_scale_in,
    arg_scale_out,
    arg_stids,
    arg_num_valid,
    i32_ntok,
    row0,
    row_stride,
    tid,
    stride,
    total_sorted,
    *,
    scale_per_row: int,
    scaleN_pad: int,
):
    """Reorder the gathered E8M0 scales into the layout GEMM1 reads.

    ``fused_moe_2stages`` runs this as ``mxfp4_moe_sort_fwd`` between the
    AllGather and the GEMMs, because it needs both: the gathered scales and the
    sort. That makes it part of the fused region -- the chain is really
    ``quant -> AG -> scale shuffle -> GEMM1 -> GEMM2 -> RS`` -- so a single
    kernel has to host it too.

    ``topk`` is 1 on this path (``mxfp4_moe_sort_fwd`` does not forward one), so
    the source row is just the token id; the per-slot ``token*topk + slot``
    addressing of the C++ kernel does not apply here.
    """
    src = global_typed_ptr(arg_scale_in, T.i8, align=1)
    dst = global_typed_ptr(arg_scale_out, T.i8, align=1)
    num_valid = global_typed_ptr(arg_num_valid, T.i32)[0]
    # Row-major over threads, column-major within one: thread ``t`` walks the
    # *columns* of one row while the whole block sits on the same row, so a
    # wavefront reads ``scale_per_row`` consecutive bytes.
    #
    # The other way round -- one row per thread, the full column loop inside --
    # is what this used to do, and it is the worst pattern available: adjacent
    # lanes land ``scale_per_row`` bytes apart (192 for model_dim 6144), so every
    # single-byte load pulls a whole cache line and uses one byte of it. Measured
    # at ~107 GB/s against 8 TB/s of HBM -- 1.3% of peak for 3.2 MB of scales,
    # which is where ~30 us of the 73 us of per-m-block scaffolding went.
    #
    # Both loops stay runtime rather than ``range_constexpr``: fully unrolling
    # the inner one means 112-224 live byte loads depending on model_dim, and
    # this code shares a register budget with both GEMMs and the RS tail.
    # Rows across waves, columns across lanes. The all-threads-on-one-row form
    # this replaces had a serial row depth of BM, which is why it won on the
    # BM=32 cells and lost on the only BM=64 one; splitting rows over the four
    # waves cuts that depth to BM/4 while a wave still reads 64 contiguous
    # bytes. The per-row guard work (num_valid, the sorted id, the token check)
    # also drops from 256-way redundant to 64-way.
    _nw = stride // fx.Int32(64)
    _wv = tid // fx.Int32(64)
    _ln = _umod(tid, fx.Int32(64))
    for raw in range(row0 + _wv * row_stride, total_sorted, row_stride * _nw):
        row = fx.Int32(raw)
        # Rows past the sort's valid count are padding: the reference leaves
        # them untouched, so they must stay untouched here as well.
        if row < num_valid:
            info = global_typed_ptr(arg_stids, T.i32)[row]
            token = info & fx.Int32(0xFFFFFF)
            if token < i32_ntok:
                base = token * fx.Int32(scale_per_row)
                for col in range(_ln, fx.Int32(scale_per_row), fx.Int32(64)):
                    c = fx.Int32(col)
                    dst[_mx_scale_shuffle_idx(scaleN_pad, row, c)] = src[base + c]


# The staged routes are written by one CTA and read back, inside the same
# launch, by another that may sit on a different XCD -- and MI355X L2 is
# per-XCD. Rather than bracket that with a release/acquire fence pair (an L2
# writeback and a full invalidate, per sort block, which would throw away the
# GEMM weight locality this kernel depends on), give the two accesses a cache
# policy: `sc1` on the store puts the line past L2, `sc0` on the load fetches
# past it. Same pairing the a8w4 comm-fused megakernel uses for its own route
# buffer.
_ROUTE_STORE_SC1 = 0x10
_ROUTE_LOAD_SC0 = 0x1
#: What a ``reduce``-tuned row does inside the merged kernel.
#:
#: ``atomic`` re-emits GEMM2 with the atomic epilogue, keeping the tuned tile
#: but not the tuned epilogue. ``inline`` keeps the staging buffer and hosts the
#: reduction here (:func:`emit_route_reduce`).
#:
#: ``atomic`` is the default: ``inline`` cannot be made fast, and the reason
#: is structural rather than a missing optimisation. ``atomic`` needs the
#: partial zeroed, which the sort does not do for a reduce-tuned row -- see
#: ``zero_partial`` in :meth:`MegaMoeTP._fused_stage12`.
#:
#: ``inline`` cannot be made fast, and the reason is structural rather than a
#: missing optimisation. Staging moves
#: ``M * topk * H`` twice where the atomic epilogue moves ``M * H`` once -- at
#: kimi3 M=8192 that is 939 MB out and 939 MB back in, against 117 MB. The
#: three-kernel path absorbs that because its reduction is its own launch with
#: one CTA per token; here it runs inside CTAs whose LDS and register budget
#: are set by the GEMMs, which leaves about two waves per SIMD. Measured at
#: kimi3 M=8192: the gather alone costs 1528 us for 1.05 GB (~0.7 TB/s, a tenth
#: of what the part is worth), and it scales linearly with route count -- one
#: route instead of eight costs 182 us. Latency-bound at GEMM occupancy, not
#: fixable by a better loop; four loop shapes were tried.
_ROUTE_HOST = os.environ.get("AITER_TP_MEGA_REDUCE_HOST", "atomic")
#: How the staged routes are made visible to the consuming CTA.
#:   0 - plain stores/loads, release+acquire fences  (default)
#:   1 - `sc1` stores, `sc0` loads, no fences
#:   2 - `sc1` stores, plain loads, release+acquire fences
#:
#: 1 and 2 both give the wrong answer, and since 2 keeps the fences the fault
#: is the `sc1` *store*, not the visibility protocol: passing
#: ``_reduce_store_cache_modifier`` switches GEMM2 to a different 128-bit
#: epilogue branch (``mxmoe_gemm_v2``), which only the a8w4 comm-fused producer
#: exercises and which evidently does not reproduce the default branch for this
#: configuration. Left as a knob because it is the way to drop the release
#: fence; the fence form is correct and is what ships.
_ROUTE_VIS = int(os.environ.get("AITER_TP_MEGA_ROUTE_VIS", "0"))
#:   3 - plain stores, `sc0` loads, release fence only (no L2 invalidate)
#:   4 - nothing. WRONG RESULTS; exists only to price the fences.
#:   5 - counter bumps but no reduction. WRONG RESULTS; prices the gather.
#:   6 - neither. WRONG RESULTS; prices the `reduce` epilogue on its own.
#:   7 - gather one route instead of `topk`. WRONG RESULTS; prices the width.
_ROUTE_SC1_STORE = _ROUTE_VIS in (1, 2)
_ROUTE_RELEASE = _ROUTE_VIS in (0, 2, 3)
_ROUTE_ACQUIRE = _ROUTE_VIS in (0, 2)
_ROUTE_LOAD_MOD = 0 if _ROUTE_VIS in (0, 2) else _ROUTE_LOAD_SC0
_ROUTE_COUNT = _ROUTE_VIS != 6
_ROUTE_GATHER = _ROUTE_VIS not in (5, 6)
_ROUTE_WIDTH1 = _ROUTE_VIS == 7
#: bf16 elements per route access -- 16 bytes, the widest buffer op.
_ROUTE_VEC = 8


def _route_view(arg):
    """Flat bf16 V# over *arg*, indexed in ``_ROUTE_VEC``-element units.

    Unit-strided, not element-strided. ``unit_stride=1`` would let a caller
    slice at any element, but it also declares the pointer 2-byte aligned,
    which is all a 16-byte access could then rely on -- and a 128-bit copy atom
    under a 2-byte alignment gets scalarised. Every offset here is a multiple
    of ``_ROUTE_VEC`` anyway, so the strided form costs nothing and keeps the
    access a single ``buffer_load_dwordx4``.
    """
    return ptr_buf_tensor(arg, fx.BFloat16, unit_elems=_ROUTE_VEC)


@flyc.jit
def emit_route_reduce(
    arg_stids,
    arg_counter,
    arg_target,
    arg_partial,
    mb,
    i32_M,
    i32_num_valid,
    tid,
    lds_raw,
    *,
    BM: int,
    topk: int,
    model_dim: int,
):
    """Reduce a token's topk staged routes as soon as the last one is written.

    The ``reduce`` epilogue stages GEMM2 output at ``target[token][slot][:]``
    and a separate pass sums the ``topk`` slots of each token. Hosting that pass
    here looks like it needs a grid-wide barrier after GEMM2 -- a token's routes
    go to different experts, so they land in different sort blocks, and no CTA
    holds all of them. A barrier would put the grid back under the co-residency
    cap this kernel was built to escape.

    It does not. The dependency is per *token*, not grid-wide: bump a counter
    for each row this block produced, and whichever CTA takes a token's count to
    ``topk`` owns that token's reduction. CTAs never wait on each other, so the
    grid can stay as large as the sort block count.

    The counter needs no memset beyond its first: the winner subtracts ``topk``
    on its way out, and since a launch performs exactly ``topk`` increments per
    token, the counter is back at zero by the time the next launch starts.
    Generation counting was the obvious alternative and is wrong twice over --
    the ReduceScatter epoch advances on *atomic* launches too, so a shared
    counter would drift, and a host-side generation is frozen by CUDA-graph
    capture, which this path is meant to run under.
    """
    # LDS scratch: for each of this block's rows, the token it won, or -1.
    # Safe to reuse the GEMM region -- this runs between a block's last GEMM2
    # tile and the next block's first GEMM1 tile, with barriers on both sides.
    if const_expr(not _ROUTE_COUNT):
        return
    scratch = fx.recast_iter(fx.Int32, lds_raw)
    base = mb * fx.Int32(BM)
    if tid < fx.Int32(BM):
        row = base + tid
        tok = fx.Int32(-1)
        if row < i32_num_valid:
            info = global_typed_ptr(arg_stids, T.i32)[row]
            # Same bound GEMM2's reduce epilogue applies when it stages the
            # row: a row it skipped must not be counted as an arrival.
            t = info & fx.Int32(0xFFFFFF)
            if t < i32_M:
                addr = fx.Int64(arg_counter) + fx.Int64(t) * fx.Int64(4)
                prev = fx.Int32(comm.atomic_add_agent(addr, fx.Int32(1)))
                if prev == fx.Int32(topk - 1):
                    # Last route in for this token: reduce it, and hand the
                    # counter back at zero for the next launch.
                    comm.atomic_add_agent(addr, fx.Int32(-topk))
                    tok = t
        scratch[tid] = tok
    gpu.barrier()
    if const_expr(not _ROUTE_GATHER):
        return

    # Rows outside, columns inside. The flattened (row, column) form needs an
    # integer division per iteration to recover the row, and a non-winning row
    # still costs its whole share of iterations; this way a non-winning row is
    # one LDS read and a branch for the entire workgroup.
    #
    # No barrier in this loop. The first version had one per row and was 30x
    # slower than the three-kernel path.
    NCOL = model_dim // _ROUTE_VEC
    n_route = 1 if _ROUTE_WIDTH1 else topk
    src = _route_view(arg_target)
    dst = _route_view(arg_partial)
    load = buf_copy_atom(_ROUTE_VEC * 2, fx.BFloat16, cache_modifier=_ROUTE_LOAD_MOD)
    store = buf_copy_atom(_ROUTE_VEC * 2, fx.BFloat16)
    # One fragment per route, not one reused across them: sharing it makes each
    # load wait for the previous one to be consumed, so the routes serialise at
    # full memory latency with no other wave to hide them.
    frags = [fx.make_fragment_like(fx.slice(src, (0, None))) for _ in range(n_route)]
    for r in range_constexpr(BM):
        t = scratch[fx.Int32(r)]
        if t >= fx.Int32(0):
            tbase = t * fx.Int32(topk * NCOL)
            obase = t * fx.Int32(NCOL)
            for cg in range(tid, fx.Int32(NCOL), fx.Int32(_BLOCK)):
                c = fx.Int32(cg)
                for k in range_constexpr(n_route):
                    fx.copy(
                        load,
                        fx.slice(src, (tbase + c + fx.Int32(k * NCOL), None)),
                        frags[k],
                    )
                acc = [fx.Float32(0.0) for _ in range(_ROUTE_VEC)]
                for k in range_constexpr(n_route):
                    v = fx.Vector(fx.memref_load_vec(frags[k]))
                    for e in range_constexpr(_ROUTE_VEC):
                        acc[e] = acc[e] + fx.Float32(v[e])
                fx.memref_store_vec(
                    fx.Vector.from_elements(acc, fx.Float32).to(fx.BFloat16), frags[0]
                )
                fx.copy(store, frags[0], fx.slice(dst, (obase + c, None)))


def hosts_reduce_as_atomic() -> bool:
    """Whether a ``reduce``-tuned row runs its GEMM2 atomically in here.

    The sort has to know: an atomic epilogue accumulates and needs its output
    buffer zeroed, a reduce epilogue stages elsewhere and does not.
    """
    return _ROUTE_HOST != "inline"


def stage12_supported(g1_cfg, g2_cfg, model_dim: int) -> bool:
    """Whether this tuned (GEMM1, GEMM2) pair can share one kernel.

    The binding constraint is the block size: one kernel has one, GEMM2 is
    always 256 threads, and GEMM1 is ``num_waves * k_wave * 64``.
    """
    if g1_cfg is None or g2_cfg is None:
        return False
    if model_dim % RS_UNIT_ELEMS:
        return False
    if g1_cfg.get("a_dtype") != "fp4" or g1_cfg.get("out_dtype") != "fp4":
        return False
    if int(g1_cfg.get("num_waves", 4)) * int(g1_cfg.get("k_wave", 1)) * 64 != _BLOCK:
        return False
    # BM16 is excluded from the merged kernel, not from the layer. It is correct
    # in the three-kernel path at every shape tested, and correct *here* at
    # glm5 M=8/16/64 and kimi3 M=8/16 -- but glm5 M=128 returns NaN and kimi3
    # M=64 paired with a reduce GEMM2 returns rel_l2 0.43. The pattern is not
    # understood, and the BM16 output scale layout
    # (``native_scale_layout_for(16, "fp4")`` is True) is the obvious suspect:
    # this kernel reuses one LDS region across phases and drives GEMM2 from the
    # same CTA, neither of which the standalone pair does. Gating on "the cases
    # that happened to pass" is how the BM16 bug got shipped once already.
    if int(g1_cfg.get("BM", 0)) == 16:
        return False
    # ``reduce`` is hosted, not excluded. Its reduction sums a token's topk
    # routes, which live in different sort blocks, so it looks like it needs a
    # grid-wide barrier after GEMM2 -- the one thing this kernel gave up to let
    # its grid exceed what the device holds at once. It does not: the dependency
    # is per token, and ``emit_route_reduce`` expresses it with a counter.
    #
    # Block size was never the obstacle, contrary to an earlier reading here:
    # capping the standalone reduction at 256 threads measured 1.00-1.06x
    # against its natural 512/1024.
    if g2_cfg.get("epilog") not in ("atomic", "reduce") or g2_cfg.get("persist"):
        return False
    # One CTA owns one sort block through both GEMMs, so the two tile_m must
    # agree -- a GEMM2 tile narrower than the GEMM1 block would read rows this
    # CTA did not produce.
    if int(g2_cfg.get("tile_m", 0)) != int(g1_cfg.get("BM", -1)):
        return False
    # ``spart`` re-orders the flattened (m, n) output space; with m owned by the
    # CTA and n walked inside it there is no flattened space left to re-order.
    if g2_cfg.get("spart"):
        return False
    return True


@functools.cache
def compile_stage12_rs(
    tp_size: int,
    model_dim: int,
    *,
    # -- GEMM1 -----------------------------------------------------------
    g1_BM: int,
    g1_use_nt: bool,
    g1_inline_quant: bool,
    g1_act: str,
    g1_situ_beta: float,
    g1_situ_linear_beta: float,
    g1_swiglu_limit: float,
    g1_native_scale_layout: bool,
    g1_interleave: bool,
    g1_xcd_swizzle: int,
    g1_num_waves: int,
    g1_k_wave: int,
    D_HIDDEN: int,
    D_INTER: int,
    NE: int,
    # -- GEMM2 -----------------------------------------------------------
    g2_BM: int,
    g2_BN: int,
    g2_BK: int,
    g2_use_nt: bool,
    g2_SBM: int,
    g2_spart: int | None,
    g2_bf16_lds: bool | None,
    g2_kstatic: bool,
    HIDDEN_MAX: int,
    INTER_MAX: int,
    a_dtype: str,
    b_dtype: str,
    topk: int = 0,
    fuse_sort: bool = False,
    sort_mp: bool = False,
    sort_tokens: int = 0,
    fuse_ag: bool = False,
    g2_epilog: str = "atomic",
    waves_per_eu: int = 0,
    service_blocks: int = _SERVICE_BLOCKS,
):
    """Build the fused GEMM1 + GEMM2 + ReduceScatter launcher for one row pair."""
    # A ``reduce``-tuned row can run either epilogue here; see ``_ROUTE_HOST``.
    # ``g2_tag`` keeps the *tuned* epilogue in the kernel name even when the
    # hosted one is rewritten: the name is the module cache key, and a
    # reduce-tuned row that ends up atomic would otherwise collide with a
    # genuinely atomic row that happens to share tile sizes -- different GEMM1
    # configs, one compiled kernel.
    # Captured before any of it is normalised below, so the signature covers
    # exactly what the caller asked for.
    _config = dict(locals())
    g2_tag = "_red" if g2_epilog == "reduce" else ""
    if g2_epilog == "reduce" and _ROUTE_HOST != "inline":
        g2_epilog = "atomic"
        g2_tag = "_reda"
    if not 1 <= tp_size <= 8:
        raise ValueError(f"tp_size must be in [1, 8], got {tp_size}")

    # Mirror the GEMM2 knob resolution so the partitioner replayed here matches
    # the one the tile was compiled with.
    if g2_spart is None:
        g2_spart = int(os.environ.get("MXFP4_G2_SPART", "402"))
    g2_spart = int(g2_spart)
    tail_slots = rs_tail_slots(tp_size)

    # -- collect the GEMM1 tile emitter ----------------------------------
    g1: dict = {}

    def g1_compose(**hook):
        g1.update(hook)
        return _G1Handle()

    compile_gemm1_a4w4_port(
        BM=g1_BM,
        use_nt=g1_use_nt,
        inline_quant=g1_inline_quant,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        interleave=g1_interleave,
        xcd_swizzle=g1_xcd_swizzle,
        a_dtype="fp4",
        out_dtype="fp4",
        act=g1_act,
        situ_beta=g1_situ_beta,
        situ_linear_beta=g1_situ_linear_beta,
        swiglu_limit=g1_swiglu_limit,
        native_scale_layout=g1_native_scale_layout,
        num_waves=g1_num_waves,
        k_wave=g1_k_wave,
        _composition=g1_compose,
    )
    if int(g1["block_threads"]) != _BLOCK:
        raise ValueError(
            f"GEMM1 wants {g1['block_threads']} threads but GEMM2 is fixed at "
            f"{_BLOCK}; this pair cannot share a kernel"
        )
    emit_gemm1_tile = g1["emit_gemm1_tile"]
    g1_n_blocks = int(g1["n_blocks"])
    g1_lds_bytes = int(g1["lds_bytes"])

    # -- optionally pull the quantize-and-push AllGather in front ----------
    ag: dict = {}
    if fuse_ag:
        from .allgather_quant_push import (
            compile_allgather_quant_push,
            quant_push_supported,
        )

        if not quant_push_supported(model_dim, topk):
            raise ValueError(
                f"model_dim={model_dim} topk={topk} has no fused quant push, so "
                "the AllGather cannot move into this kernel"
            )
        compile_allgather_quant_push(
            tp_size,
            model_dim,
            topk,
            block=_BLOCK,
            _composition=lambda **hook: ag.update(hook),
        )

    # -- build the merged kernel inside the GEMM2 composition -------------
    emit_quant_payload_push = ag.get("emit_quant_payload_push")
    emit_route_push = ag.get("emit_route_push")
    emit_ag_barrier = ag.get("emit_ag_barrier")
    emit_ag_gate = ag.get("emit_ag_gate")
    # Every push goes to the lowest ``PUSH_CTAS`` block ids. Workgroups are
    # dispatched in increasing id order, so those are the ones guaranteed to be
    # resident -- which means the grid above them can be as large as we like and
    # a queued CTA never starves the push it is waiting on. Without this the
    # grid is capped at the CU count, and that cap is expensive: the same GEMM
    # work measured 391 us at one CTA per sort block against 533 us at 256.
    push_ctas = 0
    if fuse_ag:
        from aiter.jit.utils.chip_info import get_cu_num as _cu

        push_ctas = int(_cu())
    if fuse_ag:
        from .allgather_push import AG_DESC_DONE, AG_DESC_EPOCH
        from .p2p import desc_size as _p2p_desc_size

        ag_done_index = _p2p_desc_size(tp_size, 4) + AG_DESC_DONE
        assert AG_DESC_EPOCH == AG_DESC_DONE - 1  # the -1 read above
    else:
        ag_done_index = 0

    def g2_compose(*, module_name, emit_gemm2_tile, shared_storage, lds_bytes, **_):
        # ``2 * g2_BM`` int32 of that region is also the route-reduce scratch.
        # -- the expert sort, when this kernel runs it itself ----------------
        #
        # Only the oneshot sorter can be inlined, and it fixes its own block
        # size from the expert count (256 for E <= 256, else 512). A CTA cannot
        # change block size mid-kernel, so a shape whose sorter wants 512 is
        # rejected here rather than silently run at the wrong width.
        sort_emit = None
        sort_geom = {}
        sort_lds_bytes = 0
        _SMEM_COLS = 0
        _SUB_TOKENS = 0
        sort_emit_p0v2 = None
        sort_emit_p23 = None
        _MP_MESH_STRIDE = 0
        _MP_MESH_I32 = 0
        _MP_WS_I32 = 0
        _K4_COLS = 0
        _P0V2_WAVES = 0
        _K4_WAVES = 0
        if fuse_sort and fuse_ag and sort_mp:
            # Decode token counts. The oneshot sorter's LDS mesh is
            # ``sub_tokens x (E+1)`` and cannot reach past 16 tokens; the
            # multiphase path keeps its mesh in HBM and needs only counters in
            # LDS, so it scales. For T <= 2048 it is exactly two phases --
            # p0v2 (clear+scatter+count) and p23 (prefix-sum+scatter) -- with
            # one grid-wide barrier between them, which this kernel already
            # emits for the AllGather.
            from ..moe_sorting_kernel import _compile_moe_sorting_multiphase

            _compile_moe_sorting_multiphase(
                num_experts=NE,
                topk=topk,
                unit_size=g2_BM,
                # A CTA cannot change width mid-kernel, so both phases run at
                # this kernel's. p0v2 defaults to 512 (CK's choice) and k4 to
                # 256; both stride by the value rather than assuming it.
                p0v2_block=_BLOCK,
                k4_block=_BLOCK,
                _composition=lambda **hook: sort_geom.update(hook),
            )
            sort_emit_p0v2 = sort_geom["emit_p0v2"]
            sort_emit_p23 = sort_geom["emit_p23"]
            _K4_COLS = int(sort_geom["k4_smem_cols"])
            _P0V2_WAVES = int(sort_geom["p0v2_num_waves"])
            _K4_WAVES = int(sort_geom["k4_num_waves"])
            # Both phases' regions, laid out back to back in the GEMM union.
            # p0v2 runs to completion before p23 starts, so they could overlap,
            # but together they are under 4 KB against a 33 KB union -- not
            # worth the aliasing hazard.
            sort_lds_bytes = 4 * (_P0V2_WAVES + _K4_COLS + _K4_WAVES)
            # HBM mesh geometry, same formula the standalone host path uses:
            # one uint8 row per expert, each padded to the consumer's tile-M,
            # then E+1 i32 of expert_cumsum after it.
            _MP_MESH_STRIDE = (
                (int(sort_tokens) + g2_BM - 1) // g2_BM
            ) * g2_BM
            _MP_MESH_I32 = (NE * _MP_MESH_STRIDE + 3) // 4
            _MP_WS_I32 = _MP_MESH_I32 + NE + 1
        elif fuse_sort and fuse_ag:
            from ..moe_sorting_kernel import _compile_moe_sorting_oneshot

            _compile_moe_sorting_oneshot(
                num_experts=NE,
                topk=topk,
                max_tokens=_SORT_MAX_T,
                # The sort pads each expert's run to the GEMM tile-M the
                # consumer reads it with, which in here is GEMM2's sort block.
                unit_size=g2_BM,
                # A CTA cannot change width mid-kernel, so the sort runs at this
                # kernel's. Legal at any width -- the body strides by it and
                # already has an ``E > block`` path -- but for E > block that
                # path is a serial extension in thread 0, so it is not free.
                block_override=_BLOCK,
                _composition=lambda **hook: sort_geom.update(hook),
            )
            assert int(sort_geom["block"]) == _BLOCK
            sort_emit = sort_geom["emit_sort_oneshot"]
            _SMEM_COLS = int(sort_geom["smem_cols"])
            _SUB_TOKENS = int(sort_geom["sub_tokens"])
            sort_lds_bytes = 4 * (
                2 * int(sort_geom["smem_cols"])
                + int(sort_geom["sub_tokens"]) * int(sort_geom["smem_cols"])
                + int(sort_geom["num_waves"])
            )

        # The sort finishes before GEMM1 starts, so its LDS aliases the GEMM
        # union rather than adding to it.
        merged_lds = max(
            int(lds_bytes), g1_lds_bytes, 2 * g2_BM * 4, sort_lds_bytes
        )
        # One region per (phase, parity). `_DBUF` buys the second parity;
        # `_SPLIT_LDS` buys GEMM2 its own pair instead of aliasing GEMM1's.
        _unit = merged_lds
        _split = _DBUF and _SPLIT_LDS
        _dbuf_half = _unit if _DBUF else 0     # distance between parities
        _g2_off = 2 * _unit if _split else 0   # distance to GEMM2's own pair
        merged_lds = _unit * (1 + (1 if _DBUF else 0) + (2 if _split else 0))
        # model_dim and g2_BN are both compile-time here, so the GEMM2 n-block
        # count is too -- which lets the inner loop be unrolled instead of
        # re-deriving the bound from i32_hidden on every row block.
        G2_N_BLOCKS = int(model_dim) // int(g2_BN)
        # Rolling is a per-shape decision and the block count is only known
        # here, so the effective value shadows the module-level policy. It goes
        # into ``sig`` below like any other knob that changes the binary.
        _ROLL = _roll_enabled(G2_N_BLOCKS)

        # The name is the module cache key, so everything that changes the
        # binary has to be in it. The readable part is not enough on its own:
        # GEMM1's BN/BK, the ``nt`` flags, GEMM2's BK and ``waves_per_eu`` all
        # change the code and none of them appear there. Cells that differ only
        # in those shared a name, and whichever compiled first in the process
        # won -- five dsv4 cells moved together by ~19% between runs because of
        # it, which reads as irreproducible timing rather than as a bug.
        sig = hashlib.sha256(
            repr(
                (
                    _config, _ROUTE_HOST, _ROUTE_VIS, _ASCALE_SHUFFLE,
                    _SKIP, _ROLL, _SKIP_TAIL, _SKIP_AG, _SKIP_AG_SYNC,
                    _AG_ROWS_DIV, _AG_PUSH_UPPER,
                    _rs_service_low(),
                    _SKIP_DRAIN, fuse_sort, sort_mp, _SORT_REV, _SORT_DEBUG,
                    _FAST_SORT,
                    _COUNT, _DBUF, _DBUF_REV, _SPLIT_LDS, _TASKQ, _TASKQ_REV,
                    _TASKQ_BATCH, _TASKQ_GRID, _AGPUSH_TWICE,
                )
            ).encode()
        ).hexdigest()[:12]
        name = (
            f"mega_moe_tp_{'mega' if fuse_ag else 'stage12'}_rs_tp{tp_size}"
            f"_h{model_dim}_g1bm{g1_BM}_g2bm{g2_BM}x{g2_BN}"
            f"{g2_tag}_sv{service_blocks}_{sig}"
        )
        if os.environ.get("AITER_TP_MEGA_ECHO", "0") == "1":
            print(
                f"[ECHO] fuse_ag={fuse_ag} _SKIP={_SKIP!r} "
                f"g1_n_blocks={g1_n_blocks} G2_N_BLOCKS={G2_N_BLOCKS} "
                f"_GRID_CU={_GRID_CU} g1_lds={g1_lds_bytes} g2_lds={int(lds_bytes)} "
                f"merged_lds={merged_lds} g1_BM={g1_BM} g2_BM={g2_BM} name={name}",
                flush=True,
            )

        @fx.struct
        class MergedStorage:
            # One region for both phases: they never overlap in time, so the
            # union is enough and the two bodies each take the raw base.
            buf: fx.Array[Int8, merged_lds, 16]
            # Broadcast slot for the task id thread 0 claims.
            task: fx.Array[fx.Int32, 4, 16]

        @flyc.kernel(name=name, known_block_size=[_BLOCK, 1, 1])
        def stage12_kernel(
            arg_hidden: fx.Int64,
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_w1: fx.Int64,
            arg_w1_scale: fx.Int64,
            arg_w2: fx.Int64,
            arg_w2_scale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_stids: fx.Int64,
            arg_sweights: fx.Int64,
            arg_mind: fx.Int64,
            arg_bias1: fx.Int64,
            arg_bias2: fx.Int64,
            arg_aqout: fx.Int64,
            arg_ascaleout: fx.Int64,
            arg_out: fx.Int64,
            i32_ntok: fx.Int32,
            i32_M: fx.Int32,
            i32_max_m_blocks: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            arg_desc: fx.Int64,
            i32_rank: fx.Int32,
            i32_rows: fx.Int32,
            arg_ag_desc: fx.Int64,
            arg_ascale_raw: fx.Int64,
            i32_max_sorted: fx.Int32,
            arg_route_target: fx.Int64,
            arg_route_counter: fx.Int64,
            arg_tk_ids: fx.Int64,
            arg_tk_weights: fx.Int64,
            arg_num_valid: fx.Int64,
            arg_dbg: fx.Int64,
            arg_taskq: fx.Int64,
            arg_sort_ws: fx.Int64,
        ):
            tx_i32 = fx.Int32(gpu.thread_id("x"))
            bx_i32 = fx.Int32(gpu.block_id("x"))
            lane = tx_i32 % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
            grid_nb = fx.Int32(gpu.grid_dim.x)
            # Phase-barrier slots. They share one monotone counter, so the
            # values a launch publishes must be strictly above the previous
            # launch's. ``epoch`` bumps by 1, so a window of width W gives
            # slots 1..W. The inlined multiphase sort needs two more of them.
            _BSTEP = 4 if (fuse_sort and sort_mp) else 2
            lds = fx.SharedAllocator().allocate(MergedStorage).peek()
            lds_raw = lds.buf.ptr
            # Only one SharedAllocator is allowed per kernel, so the second
            # GEMM1 region is an offset into the same (widened) storage.
            lds_b_raw = lds_raw + fx.Int64(_dbuf_half) if _DBUF else lds_raw
            if const_expr(fuse_sort and sort_mp):
                # Multiphase: p0v2's cross-wave reduce, then p23's cumsum and
                # its cross-wave scratch. Element offsets into the same union
                # the GEMM phases take raw.
                _sort_lds = fx.recast_iter(fx.Int32, lds_raw)
                mp_reduce = _sort_lds
                mp_cumsum = _sort_lds + fx.Int64(_P0V2_WAVES)
                mp_scatter = _sort_lds + fx.Int64(_P0V2_WAVES + _K4_COLS)
            elif const_expr(fuse_sort):
                # The sorter addresses LDS as i32 elements, so its four regions
                # are offsets into the same union the GEMM phases take raw.
                _sort_lds = fx.recast_iter(fx.Int32, lds_raw)
                sort_cumsum = _sort_lds
                sort_cumdup = _sort_lds + fx.Int64(_SMEM_COLS)
                sort_mesh = _sort_lds + fx.Int64(2 * _SMEM_COLS)
                sort_scratch = _sort_lds + fx.Int64(
                    (2 + _SUB_TOKENS) * _SMEM_COLS
                )

            # Read at kernel entry, before anything can bump it -- and before
            # the AllGather, because its gate uses this value. The AllGather's
            # own epoch cannot serve: it is bumped mid-kernel, so a CTA the
            # hardware dispatched late reads the bumped value and waits for one
            # more than will ever be published.
            epoch_addr, epoch = read_epoch(arg_desc, tail_slots)

            # -- quantize this rank's rows and AllGather them ----------------
            #
            # This is where the kernel stops being a GEMM and becomes the whole
            # fused region: quant -> AG -> GEMM1 -> GEMM2 -> RS. The routing
            # AllGather and the expert sort ran before the launch, because the
            # sort reads the route and GEMM1 reads the sort.
            #
            # An in-kernel collective looks like it forces a persistent grid --
            # every CTA waits here and then continues, so a queued CTA would
            # starve the pushers it is waiting on. It does not, because the push
            # is pinned to the lowest ``push_ctas`` block ids: those are
            # dispatched first, so they always get to run, and every other CTA
            # only spins on a flag, which costs nothing while queued. See the
            # grid sizing in :func:`run_stage12_rs`.
            if fuse_ag and not _SKIP_AG:
                ag_epoch0 = epoch
                # Clamp to the actual grid: a shape whose sort blocks number
                # fewer than the CU count would otherwise have the barrier wait
                # for arrivals from CTAs that do not exist. glm5 at M=16 is 254
                # blocks against 256 CUs, so this is not a corner case.
                n_push = fx.min(fx.Int32(push_ctas), grid_nb)
                if bx_i32 < n_push:
                    if const_expr(_AGPUSH_TWICE):
                        # Push the same bytes an extra time. Idempotent, so the
                        # result is unchanged and this runs under the real
                        # accuracy gate; the delta against a normal run is one
                        # payload AllGather's cost inside this kernel.
                        emit_quant_payload_push(
                            arg_ag_desc,
                            i32_rank,
                            i32_rows,
                            bx_i32,
                            bx_i32 * fx.Int32(_BLOCK) + tx_i32,
                            n_push * fx.Int32(_BLOCK),
                        )
                    emit_quant_payload_push(
                        arg_ag_desc,
                        i32_rank,
                        i32_rows
                        if const_expr(_AG_ROWS_DIV == 1)
                        else _udiv(i32_rows, fx.Int32(_AG_ROWS_DIV)),
                        bx_i32,
                        bx_i32 * fx.Int32(_BLOCK) + tx_i32
                        if const_expr(not _AG_PUSH_UPPER)
                        else bx_i32 * fx.Int32(_BLOCK)
                        + tx_i32
                        + _udiv(i32_rows * fx.Int32(model_dim // 128), fx.Int32(2)),
                        n_push * fx.Int32(_BLOCK),
                    )
                    if const_expr(fuse_sort):
                        # The route rides the same push, and therefore the same
                        # barrier. On the host it needs its own AllGather -- and
                        # therefore its own cross-rank rendezvous -- because the
                        # sort sits between it and this kernel. With the sort in
                        # here, one rendezvous covers both.
                        emit_route_push(
                            arg_ag_desc,
                            i32_rank,
                            i32_rows,
                            bx_i32,
                            bx_i32 * fx.Int32(_BLOCK) + tx_i32,
                            n_push * fx.Int32(_BLOCK),
                        )
                    if const_expr(not _SKIP_AG_SYNC):
                        emit_ag_barrier(
                            arg_ag_desc,
                            i32_rank,
                            n_push,
                            tx_i32,
                            entry_epoch=ag_epoch0,
                        )
                # Pushers fall through already satisfied; the rest wait here.
                if const_expr(not _SKIP_AG_SYNC):
                    emit_ag_gate(arg_ag_desc, ag_epoch0, tx_i32)

            # -- the expert sort ---------------------------------------------
            #
            # It reads the gathered route, which the gate above just made
            # visible, and writes the arrays GEMM1 is about to read. Only block
            # 0 sorts, so a workgroup barrier cannot order this: the handoff is
            # grid-wide, and across XCDs, so it needs the same agent-scope
            # release/acquire as any other inter-phase handoff here.
            #
            # ``i32_moe_buf_elems = 0`` because the output buffer is zeroed
            # before the launch; the sorter's zeroing pass is for the standalone
            # kernel, where nothing else has done it.
            if const_expr(fuse_sort and sort_mp):
                # Two phases, one grid-wide barrier between them. Experts are
                # strided **by block**: each phase body carries CTA-scope
                # barriers, so every thread of a block has to walk the same
                # expert sequence (see the emitters' docstrings).
                _mp_ws = fx.get_iter(_sort_buf(arg_sort_ws))
                # The mesh scatter is byte-addressed, and always under an
                # in-bounds guard, so it wants a plain typed global pointer --
                # ``recast_iter`` is for LDS, which carries a memspace.
                # ``align=1``: the mesh byte offset is ``eid * stride + token``,
                # which is arbitrarily aligned. The standalone sorter's
                # ``_i8_global_ptr`` builds the pointer type with alignment 1
                # for the same reason; the default 4 here is a miscompile
                # waiting to happen.
                _mp_ws_i8 = global_typed_ptr(arg_sort_ws, T.i8, align=1)
                _mp_tk = fx.get_iter(_sort_buf(arg_tk_ids))
                _mp_msk = fx.get_iter(_sort_buf(arg_tk_ids))  # no mask build
                _mp_ltok = fx.get_iter(_sort_buf(arg_tk_ids))  # no local_tokens
                for _e0 in range(bx_i32, fx.Int32(NE), grid_nb):
                    sort_emit_p0v2(
                        fx.Int32(_e0),
                        tx_i32,
                        lane,
                        wave,
                        _mp_ws,
                        _mp_ws_i8,
                        _mp_msk,
                        _mp_tk,
                        _mp_ltok,
                        mp_reduce,
                        i32_ntok,
                        fx.Int32(_MP_MESH_STRIDE),
                        fx.Int32(_MP_MESH_I32),
                    )
                emit_phase_barrier(
                    arg_desc,
                    fx.Int32(_BSTEP) * epoch + fx.Int32(1),
                    bx_i32,
                    grid_nb,
                    tx_i32,
                    tp_size=tp_size,
                )
                for _e1 in range(bx_i32, fx.Int32(NE), grid_nb):
                    sort_emit_p23(
                        fx.Int32(_e1),
                        tx_i32,
                        lane,
                        wave,
                        _mp_ws,
                        fx.get_iter(_sort_buf(arg_tk_weights)),
                        fx.get_iter(_sort_buf(arg_stids)),
                        fx.get_iter(_sort_buf(arg_sweights)),
                        fx.get_iter(_sort_buf(arg_eids)),
                        fx.get_iter(_sort_buf(arg_cumsum)),
                        _mp_msk,
                        _mp_ltok,
                        mp_cumsum,
                        mp_scatter,
                        i32_ntok,
                        fx.Int32(_MP_MESH_STRIDE),
                        fx.Int32(_MP_MESH_I32),
                    )
                # Every block wrote its own expert's slice of ``sorted_ids``;
                # the m_indices pass below reads all of it from block 0.
                emit_phase_barrier(
                    arg_desc,
                    fx.Int32(_BSTEP) * epoch + fx.Int32(2),
                    bx_i32,
                    grid_nb,
                    tx_i32,
                    tp_size=tp_size,
                )
            elif const_expr(fuse_sort):
                sort_emit(
                    fx.get_iter(_sort_buf(arg_tk_ids)),
                    fx.get_iter(_sort_buf(arg_tk_weights)),
                    fx.get_iter(_sort_buf(arg_stids)),
                    fx.get_iter(_sort_buf(arg_sweights)),
                    fx.get_iter(_sort_buf(arg_eids)),
                    # The sort's ``num_valid_ids[0]`` *is* what this kernel reads
                    # back as ``cumsum0`` a few lines below to size the m loop,
                    # so point them at one tensor rather than hoping the host
                    # passed the same one twice.
                    fx.get_iter(_sort_buf(arg_cumsum)),
                    fx.get_iter(_sort_buf(arg_out)),
                    # expert_mask / local_tokens: this build has neither, so the
                    # emitter never reads them, but the slots want a mapped view.
                    fx.get_iter(_sort_buf(arg_tk_ids)),
                    fx.get_iter(_sort_buf(arg_tk_ids)),
                    # This is not just the loop bound: the sort also ORs it into
                    # the sentinel it writes into every padding slot, and the
                    # consumer recognises padding by comparing the unpacked
                    # token index against GEMM1's ``n_tokens``. So it has to be
                    # the value GEMM1 will compare against, not the gathered M.
                    i32_ntok,
                    fx.Int32(0),
                    bx_i32,
                    tx_i32,
                    grid_nb,
                    sort_cumsum,
                    sort_cumdup,
                    sort_mesh,
                    sort_scratch,
                )
                # GEMM1 reads ``m_indices``, which the sort does not write: it
                # packs the topk slot into the high byte of ``sorted_ids`` and
                # ``m_indices`` is the low 24 bits of that. The standalone sort
                # has a separate aux pass for it; here it is one grid-stride
                # loop over the rows the sort just produced.
                # Block 0, not grid-stride: ``sorted_ids`` is written by block
                # 0 alone and the barrier that publishes it is *below*, so any
                # other block reading it here would race. It read the host's
                # copy and looked correct for exactly as long as the host was
                # still sorting.
                _mind_n = global_typed_ptr(arg_cumsum, T.i32)[0]
                _stids_p = global_typed_ptr(arg_stids, T.i32)
                _mind_p = global_typed_ptr(arg_mind, T.i32)
                _mind_lo = (bx_i32 == fx.Int32(0)).select(tx_i32, _mind_n)
                for _mi in range(_mind_lo, _mind_n, fx.Int32(_BLOCK)):
                    _mind_p[_mi] = _stids_p[_mi] & fx.Int32(0xFFFFFF)
                    if const_expr(_SORT_DEBUG):
                        global_typed_ptr(arg_dbg, T.i32)[_mi] = _stids_p[_mi]
                emit_phase_barrier(
                    arg_desc,
                    fx.Int32(_BSTEP) * epoch + fx.Int32(_BSTEP - 1),
                    bx_i32,
                    grid_nb,
                    tx_i32,
                    tp_size=tp_size,
                )

            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]

            # -- GEMM1 then GEMM2 ------------------------------------------
            #
            # Two shapes, picked by whether the AllGather lives in this kernel.
            #
            # ``fuse_ag=0``: one CTA owns one sort block and runs that block's
            # GEMM1 tiles then its GEMM2 tiles. GEMM2's row block reads only the
            # GEMM1 output of the *same* row block -- each GEMM1 n-tile pairs
            # its gate slice with the matching up slice and writes ``BN//2``
            # intermediate columns, so one row block's tiles cover its
            # intermediate exactly -- so no grid-wide barrier is needed, and the
            # launch is free to be one CTA per sort block at full occupancy.
            #
            # ``fuse_ag=1``: the AllGather barrier already forces every CTA to be
            # resident, so the grid is the CU count no matter what. Owning a
            # whole sort block then balances badly: at ~430 blocks over 256 CTAs
            # some CTAs get two and some get one, a 2x tail. Walking the
            # flattened (m, n) tile space instead gives ~40 tiles per CTA, and
            # the phase barrier it needs is free here -- co-residency is already
            # a precondition, not a new cost.
            g1_total_m = _udiv(cumsum0, g1_BM)
            _NXCD = 8
            _xq = _udiv(g1_total_m, _NXCD)
            _xr = _umod(g1_total_m, _NXCD)

            def _g1_flat_tile(pid, bound):
                """XCD-round-robin over the flattened (m, n) GEMM1 tile space.

                The tuned swizzle GEMM1 ships; reproduced here because the
                flattened branch walks the same space the standalone kernel
                does.
                """
                if const_expr(g1_xcd_swizzle <= 0):
                    return pid
                xq = _udiv(bound, _NXCD)
                xr = _umod(bound, _NXCD)
                xc = _umod(pid, _NXCD)
                wgid = xc * xq + fx.min(xc, xr) + _udiv(pid, _NXCD)
                ng = fx.Int32(g1_xcd_swizzle * g1_n_blocks)
                group_id = wgid // ng
                first_pid_m = group_id * fx.Int32(g1_xcd_swizzle)
                remaining_m = g1_total_m - first_pid_m
                group_size_m = fx.min(remaining_m, fx.Int32(g1_xcd_swizzle))
                wig = wgid % ng
                m_block = first_pid_m + (wig % group_size_m)
                n_block = wig // group_size_m
                return m_block * fx.Int32(g1_n_blocks) + n_block

            def _m_block(pid):
                """Spread consecutive row blocks across XCDs, as GEMM1 does.

                The tuned swizzle permutes the flattened (m, n) tile space; with
                n handled inside the CTA there is only m left to permute, so this
                is the same round-robin restricted to that axis.
                """
                if const_expr(g1_xcd_swizzle <= 0):
                    return pid
                xc = _umod(pid, _NXCD)
                return xc * _xq + fx.min(xc, _xr) + _udiv(pid, _NXCD)

            if _GRID_CU > 0:
                # Flattened tile space, one grid-stride loop per phase. Legal
                # only when the grid is forced persistent: the phase barriers
                # below require every CTA resident.
                if _ASCALE_SHUFFLE:
                    emit_ascale_shuffle(
                        arg_ascale_raw,
                        arg_ascale,
                        arg_stids,
                        arg_cumsum,
                        i32_ntok,
                        bx_i32,
                        grid_nb,
                        tx_i32,
                        fx.Int32(_BLOCK),
                        i32_max_sorted,
                        scale_per_row=model_dim // 32,
                        scaleN_pad=((model_dim // 32 + 7) // 8) * 8,
                    )
                    # Grid-strided, so a CTA reads rows another CTA wrote: this
                    # needs the same grid-wide release/acquire as the GEMM1
                    # handoff below, not a workgroup barrier.
                    # Two barriers in one launch need two gate values: the gate
                    # is ``spin_until_ge(gate, epoch)``, so reusing one epoch
                    # makes the second call pass straight through. 2*e+1 and
                    # 2*e+2 stay monotone across launches and are never 0, which
                    # a zero-initialised gate would satisfy for free.
                    emit_phase_barrier(
                        arg_desc,
                        fx.Int32(_BSTEP) * epoch + fx.Int32(_BSTEP - 1),
                        bx_i32,
                        grid_nb,
                        tx_i32,
                        tp_size=tp_size,
                    )
                g1_bound = g1_total_m * fx.Int32(g1_n_blocks)
                for raw in range(bx_i32, g1_bound, grid_nb):
                    gpu.barrier()
                    TILE = _g1_flat_tile(fx.Int32(raw), g1_bound)
                    emit_gemm1_tile(
                        arg_aq,
                        arg_ascale,
                        arg_w1,
                        arg_w1_scale,
                        arg_eids,
                        arg_mind,
                        arg_aqout,
                        arg_ascaleout,
                        arg_hidden,
                        arg_bias1,
                        TILE,
                        lane,
                        wave,
                        i32_ntok,
                        g1_total_m,
                        lds_raw,
                    )
                emit_phase_barrier(
                    arg_desc,
                    fx.Int32(_BSTEP) * epoch + fx.Int32(_BSTEP),
                    bx_i32,
                    grid_nb,
                    tx_i32,
                    tp_size=tp_size,
                )
                g2_bound = g1_total_m * fx.Int32(G2_N_BLOCKS)
                for raw2 in range(bx_i32, g2_bound, grid_nb):
                    gpu.barrier()
                    unit = fx.Int32(raw2)
                    MBLK = _udiv(unit, fx.Int32(G2_N_BLOCKS))
                    NBLK = unit - MBLK * fx.Int32(G2_N_BLOCKS)
                    emit_gemm2_tile(
                        arg_aqout,
                        arg_ascaleout,
                        arg_w2,
                        arg_w2_scale,
                        arg_eids,
                        arg_stids,
                        arg_sweights,
                        arg_bias2,
                        arg_out,
                        MBLK,
                        NBLK,
                        lane,
                        wave,
                        i32_M,
                        i32_max_m_blocks,
                        i32_inter,
                        i32_hidden,
                        lds,
                    )
            else:
                # One CTA owns one sort block: its GEMM1 tiles, then its
                # GEMM2 tiles, with the handoff below.

                def _do_shuffle(mb):
                    if fuse_ag and _ASCALE_SHUFFLE:
                        # Only this block's rows. Doing it grid-stride instead
                        # would make CTA i shuffle rows CTA j consumes, and a
                        # workgroup barrier cannot order that -- the same reason
                        # the GEMM handoff is per sort block rather than
                        # grid-wide.
                        emit_ascale_shuffle(
                            arg_ascale_raw,
                            arg_ascale,
                            arg_stids,
                            arg_cumsum,
                            i32_ntok,
                            mb * fx.Int32(g1_BM),
                            fx.Int32(1),
                            tx_i32,
                            fx.Int32(_BLOCK),
                            (mb + fx.Int32(1)) * fx.Int32(g1_BM),
                            scale_per_row=model_dim // 32,
                            scaleN_pad=((model_dim // 32 + 7) // 8) * 8,
                        )
                        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        gpu.barrier()

                def _bump(slot):
                    if const_expr(_COUNT):
                        if tx_i32 == fx.Int32(0):
                            comm.atomic_add_agent(
                                fx.Int64(arg_dbg) + fx.Int64(slot) * fx.Int64(4),
                                fx.Int32(1),
                            )

                def _do_mind(mb):
                    """``m_indices`` for one m block, when the fast sort skipped it.

                    Only this CTA's own block: the same CTA runs that block's
                    GEMM1, so a workgroup barrier is enough. Deriving it
                    grid-stride would need the grid-wide release the
                    sort-inlining path pays for.
                    """
                    if const_expr(_FAST_SORT):
                        _sp = global_typed_ptr(arg_stids, T.i32)
                        _mp = global_typed_ptr(arg_mind, T.i32)
                        _mb0 = mb * fx.Int32(g1_BM)
                        for _mi in range(tx_i32, fx.Int32(g1_BM), fx.Int32(_BLOCK)):
                            _mp[_mb0 + _mi] = _sp[_mb0 + _mi] & fx.Int32(0xFFFFFF)
                        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        gpu.barrier()

                def _g1_flat_one(raw):
                    """One GEMM1 tile addressed by its flattened (m, n) index."""
                    gpu.barrier()
                    emit_gemm1_tile(
                        arg_aq, arg_ascale, arg_w1, arg_w1_scale, arg_eids,
                        arg_mind, arg_aqout, arg_ascaleout, arg_hidden,
                        arg_bias1,
                        _g1_flat_tile(raw, g1_total_m * fx.Int32(g1_n_blocks)),
                        lane, wave, i32_ntok, g1_total_m, lds_raw,
                    )

                def _do_gemm1(mb):
                    n1 = 0 if _SKIP in ("g1", "both") else g1_n_blocks
                    for nb1 in range_constexpr(n1):
                        _bump(1)
                        # With one LDS region every tile must wait for the
                        # previous one to be done reading it. With two, only the
                        # first tile of the block has to sync -- after that the
                        # fetch for tile n+1 overlaps tile n's math.
                        if const_expr(not _DBUF) or nb1 == 0:
                            gpu.barrier()
                        _g1_lds = lds_b_raw if (_DBUF and nb1 % 2) else lds_raw
                        emit_gemm1_tile(
                            arg_aq,
                            arg_ascale,
                            arg_w1,
                            arg_w1_scale,
                            arg_eids,
                            arg_mind,
                            arg_aqout,
                            arg_ascaleout,
                            arg_hidden,
                            arg_bias1,
                            mb * fx.Int32(g1_n_blocks)
                            + (nb1 if _ROLL else fx.Int32(nb1)),
                            lane,
                            wave,
                            i32_ntok,
                            g1_total_m,
                            _g1_lds,
                        )

                def _g2_lds_base(par):
                    """LDS base for GEMM2 tile parity ``par`` (a Python int).

                    Never called with ``None``: a plain ``if`` inside a traced
                    body is rewritten into an ``scf.if`` and *both* arms get
                    traced, so a ``None`` guard here would still evaluate
                    ``par % 2``. Callers with a runtime tile index use
                    :func:`_g2_tile_rt` instead.
                    """
                    if not _DBUF:
                        return None
                    if not _split:
                        return lds_b_raw if (par % 2) else None
                    return lds_raw + fx.Int64(_g2_off + (par % 2) * _dbuf_half)

                def _g2_tile_rt(mb, nb2):
                    """One GEMM2 tile at a runtime n index: single-buffer form."""
                    _bump(2)
                    gpu.barrier()
                    emit_gemm2_tile(
                        arg_aqout, arg_ascaleout, arg_w2, arg_w2_scale,
                        arg_eids, arg_stids, arg_sweights, arg_bias2,
                        arg_route_target if g2_epilog == "reduce" else arg_out,
                        mb, nb2, lane, wave, i32_M, i32_max_m_blocks,
                        i32_inter, i32_hidden, lds,
                    )

                def _g2_tile(mb, nb2, par):
                    # ``par`` is the trace-time tile index. The rolled loop only
                    # has a runtime one, so it falls back to the single-buffer
                    # form; the constexpr loop alternates regions and only has
                    # to synchronise on its first tile.
                    _bump(2)
                    if const_expr(not _DBUF) or par == 0:
                        gpu.barrier()
                    emit_gemm2_tile(
                        arg_aqout, arg_ascaleout, arg_w2, arg_w2_scale,
                        arg_eids, arg_stids, arg_sweights, arg_bias2,
                        arg_route_target if g2_epilog == "reduce" else arg_out,
                        mb, nb2, lane, wave, i32_M, i32_max_m_blocks,
                        i32_inter, i32_hidden, lds,
                        lds_base=_g2_lds_base(par),
                    )

                def _do_gemm2(mb):
                    n2 = 0 if _SKIP in ("g2", "both") else G2_N_BLOCKS
                    # Two syntactic loops, not one over a precomputed iterator:
                    # the tracer recognises ``for x in range(...)`` by *shape*
                    # and otherwise falls back to iterating it in Python, which
                    # fails with "dynamic 'ArithValue' has no Python integer
                    # representation". The bound also has to come from a kernel
                    # argument -- a constant one is folded and re-unrolled.
                    if _ROLL and n2:
                        for nb2 in range(
                            fx.Int32(0), _udiv(i32_hidden, fx.Int32(g2_BN)),
                            fx.Int32(1),
                        ):
                            _g2_tile_rt(mb, nb2)
                    else:
                        for nb2 in range_constexpr(n2):
                            _g2_tile(mb, fx.Int32(nb2), nb2)

                def _do_reduce(mb):
                    if g2_epilog == "reduce":
                        # GEMM2 staged this block's rows at
                        # target[token][slot][:]; every slot of those tokens is
                        # only complete once every expert they routed to has
                        # run. The counter expresses exactly that, per token,
                        # instead of a grid-wide barrier -- see
                        # :func:`emit_route_reduce`.
                        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        if const_expr(_ROUTE_RELEASE and _ROUTE_GATHER):
                            comm.fence_agent_release()
                        gpu.barrier()
                        emit_route_reduce(
                            arg_stids,
                            arg_route_counter,
                            arg_route_target,
                            arg_out,
                            mb,
                            i32_M,
                            cumsum0,
                            tx_i32,
                            lds_raw,
                            BM=g2_BM,
                            topk=topk,
                            model_dim=model_dim,
                        )

                if const_expr(_TASKQ):
                    # ---- task queue ------------------------------------
                    # Task space, in claim order:
                    #   [0, M)                prep  (A-scale shuffle, m_indices)
                    #   [M, M + M*G1N)        GEMM1 tiles
                    #   [M + M*G1N, total)    GEMM2 tiles
                    #
                    # Claim order matters for liveness: a GEMM2 id is only
                    # handed out after every GEMM1 id has been, and a CTA holding
                    # a GEMM1 task is never blocked, so a CTA spinning on a GEMM2
                    # dependency always has someone able to satisfy it.
                    _M = g1_total_m
                    _n_g1 = _M * fx.Int32(g1_n_blocks)
                    _g1_base = _M
                    _g2_base = _M + _n_g1
                    _total = _g2_base + _M * fx.Int32(G2_N_BLOCKS)
                    _ctr_a = fx.Int64(arg_taskq)
                    _prep_a = _ctr_a + fx.Int64(4)
                    _g1d_a = _prep_a + fx.Int64(4) * fx.Int64(i32_max_m_blocks)
                    _slot = lds.task.ptr
                    # Counters are never reset; every wait is against a value
                    # derived from this launch's epoch instead.
                    _need1 = epoch * fx.Int32(g1_n_blocks)
                    # Each launch must consume a *fixed* number of claims for
                    # the modulo below to identify the task within it, so the
                    # loop is bounded by the rounded-up claim space rather than
                    # by the task count.
                    _stride = grid_nb * fx.Int32(_TASKQ_BATCH)
                    _claims = _udiv(
                        _total + _stride - fx.Int32(1), _stride
                    ) * _stride
                    for _it in range(fx.Int32(0), _claims, _stride):
                        if tx_i32 == fx.Int32(0):
                            fx.ptr_store(
                                fx.Int32(
                                    comm.atomic_add_agent(
                                        _ctr_a, fx.Int32(_TASKQ_BATCH)
                                    )
                                ),
                                _slot,
                            )
                        gpu.barrier()
                        _tb = _umod(fx.ptr_load(_slot), _claims)
                        for _k in range_constexpr(_TASKQ_BATCH):
                            _t = _tb + fx.Int32(_k)
                            # Three independent, mutually exclusive tests rather
                            # than a nested if/else chain: a traced ``else`` arm is
                            # rewritten into scf.if and the nesting did not survive
                            # it -- the GEMM2 arm never ran.
                            _is_prep = _t < _g1_base
                            _is_g1 = (_t >= _g1_base) & (_t < _g2_base)
                            _is_g2 = (_t >= _g2_base) & (_t < _total)
                            if _is_prep:
                                _bump(0)
                                _mbp = _m_block(_t)
                                _do_shuffle(_mbp)
                                _do_mind(_mbp)
                                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                                gpu.barrier()
                                if tx_i32 == fx.Int32(0):
                                    comm.store_i32_global_agent_release(
                                        _prep_a + fx.Int64(_mbp) * fx.Int64(4), epoch
                                    )
                            if _is_g1:
                                _r = _t - _g1_base
                                _mb1 = _udiv(_r, fx.Int32(g1_n_blocks))
                                if tx_i32 == fx.Int32(0):
                                    comm.spin_until_ge_i32_agent(
                                        _prep_a + fx.Int64(_mb1) * fx.Int64(4), epoch
                                    )
                                gpu.barrier()
                                _bump(1)
                                _g1_flat_one(_r)
                                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                                gpu.barrier()
                                if tx_i32 == fx.Int32(0):
                                    comm.atomic_add_agent(
                                        _g1d_a + fx.Int64(_mb1) * fx.Int64(4),
                                        fx.Int32(1),
                                    )
                            if _is_g2:
                                _r2 = _t - _g2_base
                                _mb2 = _udiv(_r2, fx.Int32(G2_N_BLOCKS))
                                _nb2 = _r2 - _mb2 * fx.Int32(G2_N_BLOCKS)
                                if tx_i32 == fx.Int32(0):
                                    comm.spin_until_ge_i32_agent(
                                        _g1d_a + fx.Int64(_mb2) * fx.Int64(4), _need1
                                    )
                                gpu.barrier()
                                _g2_tile_rt(_mb2, _nb2)
                                _do_reduce(_mb2)
                            gpu.barrier()
                if const_expr(not _TASKQ):
                  for raw in range(bx_i32, g1_total_m, grid_nb):
                    _bump(0)
                    mb = _m_block(fx.Int32(raw))
                    _do_shuffle(mb)
                    _do_mind(mb)
                    _do_gemm1(mb)
                    # This block's intermediate is complete. GEMM1 wrote it to
                    # HBM and the GEMM2 tiles below read it back, so the stores
                    # have to retire first -- and every wave in the CTA has to
                    # see that, hence the barrier after the wait rather than
                    # instead of it. Same CTA, so same XCD and same L2: no
                    # agent-scope release is needed, which is what made the old
                    # inter-phase handoff costly.
                    #
                    # Running *two* blocks' GEMM1 before a single drain, so the
                    # second covers the first's store latency, was tried and is
                    # **worse** -- 3 to 40% across every cell. The drain is not
                    # idle time for the machine: while one CTA waits on its own
                    # stores the other CTAs resident on the CU keep issuing, so
                    # there is nothing for a within-CTA pipeline to recover.
                    # It also needs the out-of-range second block clamped onto
                    # the last valid one, which re-runs that block's GEMM1 while
                    # another CTA is reading its output -- same values, but a
                    # race, and pure waste. Removed rather than left behind a
                    # knob.
                    if const_expr(not _SKIP_DRAIN):
                        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    gpu.barrier()
                    _do_gemm2(mb)
                    _do_reduce(mb)

            # -- ReduceScatter -----------------------------------------------
            if const_expr(not _SKIP_TAIL):
                emit_rs_tail(
                    arg_desc,
                    i32_rank,
                    i32_rows,
                    epoch_addr,
                    epoch,
                    bx_i32,
                    grid_nb,
                    tx_i32,
                    tp_size=tp_size,
                    model_dim=model_dim,
                    block=_BLOCK,
                    service_blocks=service_blocks,
                )

        @flyc.jit
        def launch_stage12_rs(
            arg_hidden: fx.Int64,
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_w1: fx.Int64,
            arg_w1_scale: fx.Int64,
            arg_w2: fx.Int64,
            arg_w2_scale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_stids: fx.Int64,
            arg_sweights: fx.Int64,
            arg_mind: fx.Int64,
            arg_bias1: fx.Int64,
            arg_bias2: fx.Int64,
            arg_aqout: fx.Int64,
            arg_ascaleout: fx.Int64,
            arg_out: fx.Int64,
            i32_ntok: fx.Int32,
            i32_M: fx.Int32,
            i32_max_m_blocks: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            arg_desc: fx.Int64,
            i32_rank: fx.Int32,
            i32_rows: fx.Int32,
            arg_ag_desc: fx.Int64,
            arg_ascale_raw: fx.Int64,
            i32_max_sorted: fx.Int32,
            arg_route_target: fx.Int64,
            arg_route_counter: fx.Int64,
            arg_tk_ids: fx.Int64,
            arg_tk_weights: fx.Int64,
            arg_num_valid: fx.Int64,
            arg_dbg: fx.Int64,
            arg_taskq: fx.Int64,
            arg_sort_ws: fx.Int64,
            i32_grid: fx.Int32,
            stream: fx.Stream,
        ):
            stage12_kernel(
                arg_hidden,
                arg_aq,
                arg_ascale,
                arg_w1,
                arg_w1_scale,
                arg_w2,
                arg_w2_scale,
                arg_eids,
                arg_cumsum,
                arg_stids,
                arg_sweights,
                arg_mind,
                arg_bias1,
                arg_bias2,
                arg_aqout,
                arg_ascaleout,
                arg_out,
                i32_ntok,
                i32_M,
                i32_max_m_blocks,
                i32_inter,
                i32_hidden,
                arg_desc,
                i32_rank,
                i32_rows,
                arg_ag_desc,
                arg_ascale_raw,
                i32_max_sorted,
                arg_route_target,
                arg_route_counter,
                arg_tk_ids,
                arg_tk_weights,
                arg_num_valid,
                arg_dbg,
                arg_taskq,
                arg_sort_ws,
            ).launch(
                grid=(fx.Int64(i32_grid), 1, 1),
                block=(_BLOCK, 1, 1),
                stream=stream,
            )

        launch_stage12_rs.block = _BLOCK
        launch_stage12_rs.g2_n_blocks = G2_N_BLOCKS
        # 0 when the kernel does not sort; the host allocates exactly this many
        # i32 and hands the pointer in as ``arg_sort_ws``.
        launch_stage12_rs.sort_ws_i32 = _MP_WS_I32
        return launch_stage12_rs

    return compile_gemm2_a4w4_port(
        BM=g2_BM,
        BN=g2_BN,
        BK=g2_BK,
        use_nt=g2_use_nt,
        HIDDEN_MAX=HIDDEN_MAX,
        epilog=g2_epilog,
        INTER_MAX=INTER_MAX,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        topk=topk if g2_epilog == "reduce" else 1,
        SBM=g2_SBM,
        persist=False,
        g2_spart=g2_spart,
        g2_bf16_lds=g2_bf16_lds,
        g2_kstatic=g2_kstatic,
        out_dtype="bf16",
        enable_bias=False,
        _composition=g2_compose,
        # The staged routes are read back inside this same launch, by a CTA that
        # may sit on another XCD, so they have to land past the per-XCD L2.
        _reduce_store_cache_modifier=(
            _ROUTE_STORE_SC1 if (g2_epilog == "reduce" and _ROUTE_SC1_STORE) else None
        ),
    )


class _G1Handle:
    """Stand-in for the launcher ``compile_gemm1_a4w4_port`` would return.

    The GEMM1 compiler sets ``compile_hints`` on whatever a composition hands
    back; this kernel is launched through the GEMM2 side, so that object is
    discarded and only needs to accept the attribute.
    """

    compile_hints: dict = {}


_ROUTE_COUNTERS: dict = {}


def _route_counter(M, device):
    """The per-token arrival counter for the hosted ``reduce`` epilogue.

    Zeroed once and cached: :func:`emit_route_reduce`'s winner hands it back at
    zero, so it is self-maintaining from then on and costs no memset per launch.
    """
    key = (int(M), str(device))
    ctr = _ROUTE_COUNTERS.get(key)
    if ctr is None:
        ctr = torch.zeros(int(M), dtype=torch.int32, device=device)
        _ROUTE_COUNTERS[key] = ctr
    return ctr


_ROUTE_TARGETS = {}


def _route_target(M, topk, model_dim, dtype, device):
    """Staging buffer for the hosted ``reduce`` epilogue.

    Cached rather than allocated per call, for the same reason ``_taskq_ptr``
    and ``_alloc_sorting`` are: **a CUDA graph records the address**, and a
    buffer that is freed and re-allocated between capture and replay leaves the
    kernel writing to whatever took its place.

    That was a real fault, not a hypothetical: dsv3 and dsv4 at 2048 tokens
    SIGSEGV'd under graph capture while passing eagerly. 2048 is where the
    tuned row switches to the ``reduce`` epilogue, so it is the first bucket
    that allocates this at all -- and at (2048, 8, 7168) bf16 it is 235 MB, big
    enough that the allocator hands the block back out to something else.

    Sizing is exact rather than a high-water mark: the shape is part of the key,
    so a sweep over token counts keeps one buffer per bucket. That is the same
    trade the other graph-safe buffers here make.
    """
    key = (int(M), int(topk), int(model_dim), str(dtype), str(device))
    buf = _ROUTE_TARGETS.get(key)
    if buf is None:
        buf = torch.empty(
            (int(M), int(topk), int(model_dim)), dtype=dtype, device=device
        )
        _ROUTE_TARGETS[key] = buf
    return buf


def _ptr(t, fallback):
    """Device address of ``t``, or of ``fallback`` when the kernel ignores it."""
    return int((fallback if t is None else t).data_ptr())


#: Scratch ``m_indices`` buffers, keyed by (elements, device). The fast sort
#: does not produce one and the kernel writes it per m-block; a cached buffer
#: keeps the address stable across CUDA-graph replays.
_MIND_BUFS: dict = {}


def _mind_ptr(m_indices, sorted_token_ids):
    """Address GEMM1 should read ``m_indices`` from."""
    if m_indices is not None:
        return int(m_indices.data_ptr())
    if not _FAST_SORT:
        # Unused by this build; any mapped address will do.
        return int(sorted_token_ids.data_ptr())
    key = (int(sorted_token_ids.numel()), str(sorted_token_ids.device))
    buf = _MIND_BUFS.get(key)
    if buf is None:
        buf = torch.empty_like(sorted_token_ids)
        _MIND_BUFS[key] = buf
    return int(buf.data_ptr())


#: Execution counters for :data:`_COUNT`, one buffer per device.
_COUNT_BUFS: dict = {}


def _count_buffer(device):
    buf = _COUNT_BUFS.get(str(device))
    if buf is None:
        buf = torch.zeros(4, dtype=torch.int32, device=device)
        _COUNT_BUFS[str(device)] = buf
    return buf


def read_counts(device, reset: bool = True):
    """(m_blocks, gemm1_tiles, gemm2_tiles) the merged kernel actually ran."""
    buf = _COUNT_BUFS.get(str(device))
    if buf is None:
        return None
    out = tuple(int(v) for v in buf[:3].tolist())
    if reset:
        buf.zero_()
    return out


def _dbg_ptr(dbg, sorted_token_ids):
    if _COUNT:
        return int(_count_buffer(sorted_token_ids.device).data_ptr())
    return int((dbg if dbg is not None else sorted_token_ids).data_ptr())


#: Task-queue state, keyed by (m_blocks, device): one global task counter plus
#: two per-m-block dependency counters.
_TASKQ_BUFS: dict = {}


def _taskq_ptr(sorted_token_ids, max_sorted, g2_BM):
    """Device address of the task queue's counters.

    Layout: [0] = the claim counter, [1 .. 1+B) = "prep done" per m-block,
    [1+B .. 1+2B) = "GEMM1 tiles done" per m-block.

    Never zeroed after the first allocation: every wait compares against a
    value derived from the launch epoch, so the counters only have to be
    monotone. Zeroing per launch would need either host work (which CUDA-graph
    capture freezes) or a pass over the whole array.
    """
    blocks = int((int(max_sorted) + int(g2_BM) - 1) // int(g2_BM))
    key = (blocks, str(sorted_token_ids.device))
    buf = _TASKQ_BUFS.get(key)
    if buf is None:
        buf = torch.zeros(1 + 2 * blocks, dtype=torch.int32,
                          device=sorted_token_ids.device)
        _TASKQ_BUFS[key] = buf
    return int(buf.data_ptr())



_SORT_WS_BUFS = {}


def _sort_ws_ptr(ref, n_i32: int) -> int:
    """Device address of the inlined sort's HBM workspace.

    Layout matches the standalone multiphase path: a uint8 expert mesh of
    ``E x mesh_stride`` bytes, then ``E+1`` i32 of expert_cumsum.

    Cached per (size, device) rather than allocated per call, because a CUDA
    graph replay needs the same address every time. No zeroing here either:
    p0v2's first phase clears its own mesh row and its third writes the cumsum
    entry, so nothing carries over between launches.
    """
    if n_i32 <= 0:
        return int(ref.data_ptr())  # unused; any mapped address
    key = (int(n_i32), str(ref.device))
    buf = _SORT_WS_BUFS.get(key)
    if buf is None:
        buf = torch.empty(int(n_i32), dtype=torch.int32, device=ref.device)
        _SORT_WS_BUFS[key] = buf
    return int(buf.data_ptr())

def _sort_buf(arg):
    """An i32 buffer view with the OOB semantics the sort relies on.

    The sort masks every conditional store as ``.select(idx, 0x7FFFFFFF)`` and
    counts on the hardware dropping it. That only works if the descriptor's
    ``num_records`` -- which is in **bytes** -- is smaller than that index
    scaled by 4. ``ptr_buf_tensor``'s default is ``0xFFFFFFFF`` *elements*,
    which both overflows the 32-bit field and leaves the sentinel in range, so
    the masked stores are not dropped and the real ones do not land. Cap it in
    bytes instead, which is what the standalone sorter's ``max_size=True``
    does.
    """
    return ptr_buf_tensor(arg, fx.Int32, num_records_bytes=fx.Int64(0xFFFFFFFF))


def _as_u8(t):
    if t is not None and t.element_size() == 1 and t.dtype != torch.uint8:
        return t.view(torch.uint8)
    return t


def run_stage12_rs(
    *,
    g1,
    w2,
    w2_scale,
    sorted_token_ids,
    sorted_weights,
    partial,
    output,
    desc_ptr,
    rank,
    tp_size,
    local_rows,
    M_logical,
    model_dim,
    inter_dim,
    g2_cfg,
    block_m=None,
    ag_desc_ptr=0,
    ascale_raw=None,
    topk=0,
    fuse_ag=False,
    waves_per_eu=None,
    route_target=None,
    route_counter=None,
    #: The gathered route and the sort's valid count. Only read when the kernel
    #: runs the sort itself (``AITER_TP_MEGA_FUSE_SORT``); otherwise the host
    #: sort has already produced ``sorted_token_ids`` and these stay unused, so
    #: any mapped address will do.
    tk_ids=None,
    tk_weights=None,
    num_valid=None,
    #: Debug sink: when ``AITER_TP_MEGA_SORT_DEBUG`` is set the kernel copies
    #: the ``sorted_ids`` it produced here, so the host can diff it against the
    #: reference sort instead of inferring from accuracy.
    dbg=None,
    stream=None,
):
    """Launch GEMM1 + GEMM2 + ReduceScatter as one kernel.

    ``g1`` is the keyword dict ``_mxfp4_a4w4_stage1`` would have handed to
    ``flydsl_mxfp4_gemm1`` -- captured through its ``_gemm1_launch`` hook, so
    every operand and every compile knob is the one the tuned path derived
    rather than a copy of that derivation.
    """
    g2_BM = g2_cfg["tile_m"]
    g2_SBM = g2_cfg["sort_block_m"] or (int(block_m) if block_m else g2_BM)
    wpe = _WAVES_PER_EU if waves_per_eu is None else int(waves_per_eu)
    kstatic = os.environ.get("MXFP4_G2_KSTATIC", "1") == "1"
    launch = compile_stage12_rs(
        int(tp_size),
        int(model_dim),
        g1_BM=int(g1["BM"]),
        g1_use_nt=bool(g1["use_nt"]),
        g1_inline_quant=bool(g1["inline_quant"]),
        g1_act=g1["act"],
        g1_situ_beta=float(g1["situ_beta"]),
        g1_situ_linear_beta=float(g1["situ_linear_beta"]),
        g1_swiglu_limit=float(g1["swiglu_limit"]),
        g1_native_scale_layout=bool(g1["native_scale_layout"]),
        g1_interleave=bool(g1["interleave"]),
        g1_xcd_swizzle=int(g1["xcd_swizzle"]),
        g1_num_waves=int(g1["num_waves"]),
        g1_k_wave=int(g1["k_wave"]),
        D_HIDDEN=int(g1["D_HIDDEN"]),
        D_INTER=int(g1["D_INTER"]),
        NE=int(g1["NE"]),
        g2_BM=g2_BM,
        g2_BN=g2_cfg["tile_n"],
        g2_BK=g2_cfg["tile_k"],
        g2_use_nt=bool(g2_cfg["use_nt"]),
        g2_SBM=g2_SBM,
        g2_spart=g2_cfg["spart"],
        g2_bf16_lds=g2_cfg["bf16_lds"],
        g2_kstatic=kstatic,
        HIDDEN_MAX=8192,
        INTER_MAX=int(inter_dim) if kstatic else 8192,
        a_dtype=g2_cfg["a_dtype"],
        b_dtype=g2_cfg["b_dtype"],
        topk=int(topk),
        fuse_ag=bool(fuse_ag),
        fuse_sort=(
            kernel_sorts(M_logical, int(g1["NE"]))
            or kernel_sorts_mp(M_logical, int(g1["NE"]))
        ),
        sort_mp=kernel_sorts_mp(M_logical, int(g1["NE"])),
        sort_tokens=int(M_logical),
        g2_epilog=g2_cfg["epilog"],
        # Part of the compile key *and* the kernel name: it changes the binary
        # without changing any tile, so two cells that differ only here would
        # otherwise share one compiled module.
        waves_per_eu=wpe,
    )

    aqout = g1["inter_sorted_quant"]
    max_sorted = int(sorted_token_ids.shape[0])
    # One CTA per sort block, with or without the in-kernel AllGather.
    #
    # The push is pinned to the lowest block ids, which the dispatcher schedules
    # first, and everyone else only waits on a flag -- so the whole grid does
    # not have to be co-resident. An earlier version of this hung part-way
    # through a sweep because the gate value was read per CTA from the
    # AllGather's own epoch, which is bumped *mid-kernel*: a late-dispatched CTA
    # read the bumped value and waited for one more than would ever be
    # published. The gate now uses the ReduceScatter epoch, which is bumped only
    # after every CTA has reached the tail and is therefore the same number for
    # every CTA no matter when it started.
    if g2_cfg["epilog"] == "reduce":
        # GEMM2 stages one row per route and the reduction runs later in the
        # same kernel, so the staging buffer the 3-kernel path allocates is
        # still needed -- it just never outlives the launch now.
        if route_target is None:
            route_target = _route_target(
                M_logical, topk, model_dim, partial.dtype, partial.device
            )
        if route_counter is None:
            route_counter = _route_counter(M_logical, partial.device)

    # ``max_sorted`` is what ``moe_sorting`` *allocated*: the all-experts-busy
    # worst case. The blocks that actually carry rows are bounded far tighter,
    # and from the host: at most one block per non-empty expert -- and no more
    # experts can be non-empty than there are routes -- plus the blocks the
    # routes themselves need.
    #
    # This only matters **with the AllGather hosted**, and that is not obvious.
    # Measured with the AG outside it is worth exactly nothing (+0.7/-0.3/-0.1%
    # at M=8/16/64): idle CTAs that walk an empty grid-stride loop are cheap.
    # With the AG inside they are not: every CTA that is not pushing spins at
    # the gate and cannot retire, so a grid larger than the device holds leaves
    # the surplus queued *behind* spinners, unable to start and unable to
    # overlap anything. kimi3 M=8 goes 896 CTAs -> 132 and gains 5.5%.
    #
    # The bound saturates on its own once routes >= E, so large M is untouched
    # -- which it has to be: forcing 128 CTAs at kimi3 M=1024 costs 73%.
    grid_blocks = (max_sorted + g2_BM - 1) // g2_BM
    if _TASKQ:
        # CTAs are not bound to m blocks here, so the grid is sized by what the
        # device can hold rather than by the block count. That is the whole
        # point of the queue: the static form caps parallelism at the number of
        # m blocks (132 CTAs on 256 CUs at kimi3 M=8).
        grid_blocks = max(grid_blocks, _TASKQ_GRID)
    if fuse_ag:
        routes = int(M_logical) * max(1, int(topk))
        live = min(int(g1["NE"]), routes) + (routes + g2_BM - 1) // g2_BM
        grid_blocks = min(grid_blocks, max(1, live))
    if fuse_ag and _AG_GRID_CAP > 0:
        grid_blocks = min(grid_blocks, _AG_GRID_CAP)
    if _GRID_CU > 0:
        grid_blocks = _GRID_CU  # override kept for A/B measurement
    # The jit compiles on first call, not on build, so the hint has to still be
    # in scope at the launch.
    hints = {"waves_per_eu": wpe} if wpe else {}
    # A hard VGPR cap, distinct from waves_per_eu's target: the backend turns it
    # into ``--amdgpu-num-vgpr``. This kernel's peak is 254 across quantize,
    # push, both GEMMs and the RS tail -- 2 waves/SIMD against the standalone
    # GEMM2's 7 -- and ``waves_per_eu`` is inert here (0/2/3/4 compile to the
    # same binary), so the allocator will not go below on its own. Capping
    # forces spills; whether that trades well is an open question, because the
    # kernel is bandwidth-bound and more waves in flight may buy more than the
    # scratch traffic costs.
    if _MAXNREG:
        hints["maxnreg"] = _MAXNREG
    with CompilationContext.compile_hints(hints):
        if const_expr(_SKIP_TAIL):
            # Probe only. Without the tail nothing writes ``output``, and an
            # uninitialised buffer reads back as NaN -- which the harness
            # asserts on *before* the timing loop, so the probe could never be
            # timed. Zeroing makes it a finite wrong answer instead, which
            # ``--fused-rtol 99`` lets through. A device op, so capture-safe.
            output.zero_()
        run_compiled(
            launch,
            int(g1["hidden_states"].data_ptr()),
            _ptr(g1["a_quant"], g1["hidden_states"]),
            _ptr(g1["a_scale_sorted_shuffled"], g1["hidden_states"]),
            int(_as_u8(g1["w1_u8"]).data_ptr()),
            int(_as_u8(g1["w1_scale_u8"]).data_ptr()),
            int(_as_u8(w2).data_ptr()),
            int(_as_u8(w2_scale).data_ptr()),
            int(g1["sorted_expert_ids"].data_ptr()),
            int(g1["cumsum_tensor"].data_ptr()),
            int(sorted_token_ids.data_ptr()),
            int(sorted_weights.data_ptr()),
            _mind_ptr(g1["m_indices"], sorted_token_ids),
            _ptr(g1["bias"], partial),
            int(partial.data_ptr()),  # unused bias2; any mapped address
            int(aqout.data_ptr()),
            int(g1["inter_sorted_shuffled_scale"].data_ptr()),
            int(partial.data_ptr()),
            int(g1["n_tokens"]),
            int(M_logical),
            int((max_sorted + g2_BM - 1) // g2_BM),
            int(inter_dim),
            int(model_dim),
            int(desc_ptr),
            int(rank),
            int(local_rows),
            int(ag_desc_ptr),
            _ptr(ascale_raw, sorted_token_ids),
            int(max_sorted),
            _ptr(route_target, sorted_token_ids),
            _ptr(route_counter, sorted_token_ids),
            _ptr(tk_ids, sorted_token_ids),
            _ptr(tk_weights, sorted_token_ids),
            _ptr(num_valid, sorted_token_ids),
            _dbg_ptr(dbg, sorted_token_ids),
            _taskq_ptr(sorted_token_ids, max_sorted, g2_BM),
            _sort_ws_ptr(sorted_token_ids, getattr(launch, "sort_ws_i32", 0)),
            # One CTA per sort block. Not a persistent grid any more: without the
            # inter-phase barrier nothing requires co-residency, so this is the same
            # shape the standalone GEMM1 launches and the hardware is free to hold
            # as many CTAs per CU as LDS allows.
            int(grid_blocks),
            stream if stream is not None else torch.cuda.current_stream(),
        )
    return output[:local_rows]
