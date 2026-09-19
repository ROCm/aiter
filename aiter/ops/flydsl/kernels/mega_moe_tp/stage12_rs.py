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
from .rs_tail import emit_phase_barrier, emit_rs_tail, read_epoch, rs_tail_slots

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
#: The cause is **register pressure**, not LDS and not L2 locality. Compiled
#: ISA (`FLYDSL_DUMP_IR=1`), kimi3 M=8192:
#:
#:     GEMM1 standalone   65 KB LDS   230 VGPR   0 B scratch
#:     GEMM2 standalone   32 KB LDS   128 VGPR   0 B scratch
#:     merged             65 KB LDS   256 VGPR   172 B scratch
#:
#: The merged kernel sits at the 256 arch-VGPR ceiling and spills, with the
#: scratch traffic spread through the MFMA-heavy regions rather than confined
#: to a cold path. Dropping the AllGather front (`fuse_ag=False`) leaves it
#: byte-identical at 256/172, so it is the two GEMMs alone that do it.
#:
#: Two plausible explanations were measured and rejected first, both worth
#: recording because they look right:
#:   * LDS occupancy. Forcing GEMM1 to a 19 KB tile (8 CTA/CU) instead of 65 KB
#:     (2 CTA/CU) -- a 4x occupancy swing -- moved GEMM2 in-kernel by **9%**
#:     (891.6 -> 812.9 us) and left the ratio at ~2x. The total got *worse*
#:     because GEMM1 degrades at BM32.
#:   * L2 swizzle. The merged kernel cannot use GEMM2's `spart` partitioning
#:     (`stage12_supported` rejects it). Disabling `MXFP4_G2_SPART` on the
#:     standalone kernel costs it **0-4%**, so that is not the gap either.
#:
#: One function means one register allocation, and two register-hungry GEMMs
#: do not fit. Not removable without un-fusing, which is what `mega=0` at
#: M>=2048 does.
_SKIP = os.environ.get("AITER_TP_MEGA_SKIP", "")
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
    gid,
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
    for raw in range(gid, total_sorted, stride):
        row = fx.Int32(raw)
        # Rows past the sort's valid count are padding: the reference leaves
        # them untouched, so they must stay untouched here as well.
        if row < num_valid:
            info = global_typed_ptr(arg_stids, T.i32)[row]
            token = info & fx.Int32(0xFFFFFF)
            if token < i32_ntok:
                base = token * fx.Int32(scale_per_row)
                # Runtime loop, not ``range_constexpr``. Fully unrolling it
                # means 112-224 live byte loads depending on model_dim, and this
                # code shares a register budget with both GEMMs and the RS tail:
                # the kernel's occupancy is the worst point in it, so an
                # unrolled copy loop here costs CTAs per CU everywhere else.
                for col in range(fx.Int32(0), fx.Int32(scale_per_row), fx.Int32(1)):
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
        merged_lds = max(int(lds_bytes), g1_lds_bytes, 2 * g2_BM * 4)
        # model_dim and g2_BN are both compile-time here, so the GEMM2 n-block
        # count is too -- which lets the inner loop be unrolled instead of
        # re-deriving the bound from i32_hidden on every row block.
        G2_N_BLOCKS = int(model_dim) // int(g2_BN)
        # The name is the module cache key, so everything that changes the
        # binary has to be in it. The readable part is not enough on its own:
        # GEMM1's BN/BK, the ``nt`` flags, GEMM2's BK and ``waves_per_eu`` all
        # change the code and none of them appear there. Cells that differ only
        # in those shared a name, and whichever compiled first in the process
        # won -- five dsv4 cells moved together by ~19% between runs because of
        # it, which reads as irreproducible timing rather than as a bug.
        sig = hashlib.sha256(
            repr(
                (_config, _ROUTE_HOST, _ROUTE_VIS, _ASCALE_SHUFFLE, _SKIP)
            ).encode()
        ).hexdigest()[:12]
        name = (
            f"mega_moe_tp_{'mega' if fuse_ag else 'stage12'}_rs_tp{tp_size}"
            f"_h{model_dim}_g1bm{g1_BM}_g2bm{g2_BM}x{g2_BN}"
            f"{g2_tag}_sv{service_blocks}_{sig}"
        )

        @fx.struct
        class MergedStorage:
            # One region for both phases: they never overlap in time, so the
            # union is enough and the two bodies each take the raw base.
            buf: fx.Array[Int8, merged_lds, 16]

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
        ):
            tx_i32 = fx.Int32(gpu.thread_id("x"))
            bx_i32 = fx.Int32(gpu.block_id("x"))
            lane = tx_i32 % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
            grid_nb = fx.Int32(gpu.grid_dim.x)
            lds = fx.SharedAllocator().allocate(MergedStorage).peek()
            lds_raw = lds.buf.ptr

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
            if fuse_ag:
                ag_epoch0 = epoch
                # Clamp to the actual grid: a shape whose sort blocks number
                # fewer than the CU count would otherwise have the barrier wait
                # for arrivals from CTAs that do not exist. glm5 at M=16 is 254
                # blocks against 256 CUs, so this is not a corner case.
                n_push = fx.min(fx.Int32(push_ctas), grid_nb)
                if bx_i32 < n_push:
                    emit_quant_payload_push(
                        arg_ag_desc,
                        i32_rank,
                        i32_rows,
                        bx_i32,
                        bx_i32 * fx.Int32(_BLOCK) + tx_i32,
                        n_push * fx.Int32(_BLOCK),
                    )
                    emit_ag_barrier(
                        arg_ag_desc,
                        i32_rank,
                        n_push,
                        tx_i32,
                        entry_epoch=ag_epoch0,
                    )
                # Pushers fall through already satisfied; the rest wait here.
                emit_ag_gate(arg_ag_desc, ag_epoch0, tx_i32)

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
                        bx_i32 * fx.Int32(_BLOCK) + tx_i32,
                        grid_nb * fx.Int32(_BLOCK),
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
                        fx.Int32(2) * epoch + fx.Int32(1),
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
                    fx.Int32(2) * epoch + fx.Int32(2),
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
                for raw in range(bx_i32, g1_total_m, grid_nb):
                    mb = _m_block(fx.Int32(raw))
                    if fuse_ag and _ASCALE_SHUFFLE:
                        # Only this block's rows. Doing it grid-stride instead would
                        # make CTA i shuffle rows CTA j consumes, and a workgroup
                        # barrier cannot order that -- the same reason the GEMM
                        # handoff is per sort block rather than grid-wide.
                        emit_ascale_shuffle(
                            arg_ascale_raw,
                            arg_ascale,
                            arg_stids,
                            arg_cumsum,
                            i32_ntok,
                            mb * fx.Int32(g1_BM) + tx_i32,
                            fx.Int32(_BLOCK),
                            (mb + fx.Int32(1)) * fx.Int32(g1_BM),
                            scale_per_row=model_dim // 32,
                            scaleN_pad=((model_dim // 32 + 7) // 8) * 8,
                        )
                        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        gpu.barrier()
                    for nb1 in range_constexpr(0 if _SKIP in ("g1", "both") else g1_n_blocks):
                        # Unconditional: reusing LDS across tiles is not safe
                        # without it, and a Python-level "skip the first one" flag
                        # would be evaluated at trace time and emit nothing.
                        gpu.barrier()
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
                            mb * fx.Int32(g1_n_blocks) + fx.Int32(nb1),
                            lane,
                            wave,
                            i32_ntok,
                            g1_total_m,
                            lds_raw,
                        )

                    # This block's intermediate is complete. GEMM1 wrote it to HBM
                    # and the GEMM2 tiles below read it back, so the stores have to
                    # retire first -- and every wave in the CTA has to see that,
                    # hence the barrier after the wait rather than instead of it.
                    # Same CTA, so same XCD and same L2: no agent-scope release is
                    # needed, which is what made the old inter-phase handoff costly.
                    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    gpu.barrier()

                    for nb2 in range_constexpr(0 if _SKIP in ("g2", "both") else G2_N_BLOCKS):
                        gpu.barrier()
                        m_block_idx = mb
                        n_block_idx = fx.Int32(nb2)
                        emit_gemm2_tile(
                            arg_aqout,
                            arg_ascaleout,
                            arg_w2,
                            arg_w2_scale,
                            arg_eids,
                            arg_stids,
                            arg_sweights,
                            arg_bias2,
                            arg_route_target if g2_epilog == "reduce" else arg_out,
                            m_block_idx,
                            n_block_idx,
                            lane,
                            wave,
                            i32_M,
                            i32_max_m_blocks,
                            i32_inter,
                            i32_hidden,
                            lds,
                        )

                    if g2_epilog == "reduce":
                        # GEMM2 staged this block's rows at
                        # target[token][slot][:]; every slot of those tokens is
                        # only complete once every expert they routed to has run.
                        # The counter below expresses exactly that, per token,
                        # instead of a grid-wide barrier -- see
                        # :func:`emit_route_reduce`.
                        # `sc1` already put the staged rows past L2, so the
                        # release is just the wait; without it this needs the
                        # writeback fence.
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

            # -- ReduceScatter -----------------------------------------------
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
            ).launch(
                grid=(fx.Int64(i32_grid), 1, 1),
                block=(_BLOCK, 1, 1),
                stream=stream,
            )

        launch_stage12_rs.block = _BLOCK
        launch_stage12_rs.g2_n_blocks = G2_N_BLOCKS
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


def _ptr(t, fallback):
    """Device address of ``t``, or of ``fallback`` when the kernel ignores it."""
    return int((fallback if t is None else t).data_ptr())


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
            route_target = torch.empty(
                (int(M_logical), int(topk), int(model_dim)),
                dtype=partial.dtype,
                device=partial.device,
            )
        if route_counter is None:
            route_counter = _route_counter(M_logical, partial.device)

    grid_blocks = (max_sorted + g2_BM - 1) // g2_BM
    if _GRID_CU > 0:
        grid_blocks = _GRID_CU  # override kept for A/B measurement
    # The jit compiles on first call, not on build, so the hint has to still be
    # in scope at the launch.
    hints = {"waves_per_eu": wpe} if wpe else {}
    with CompilationContext.compile_hints(hints):
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
            _ptr(g1["m_indices"], sorted_token_ids),
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
            # One CTA per sort block. Not a persistent grid any more: without the
            # inter-phase barrier nothing requires co-residency, so this is the same
            # shape the standalone GEMM1 launches and the hardware is free to hold
            # as many CTAs per CU as LDS allows.
            int(grid_blocks),
            stream if stream is not None else torch.cuda.current_stream(),
        )
    return output[:local_rows]
