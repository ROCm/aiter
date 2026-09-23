# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""FlyDSL intranode EP dispatch driven by the gfx1250 Tensor Data Mover.

A TDM moves a whole payload row on one descriptor. Routes are one lane each
(``WAVE / topk`` tokens per wave); a wave dedups a token's same-peer routes
with a ballot + mbcnt_lo match-any. Recv slots are reserved one remote atomic
per (block, peer) off an LDS histogram. idx / weights / srcmap -- and, on a
quantized wire, the token's e8m0 scale row -- are gathered into a destTokId-
ordered SoA so the cross-GPU metadata write is a few bulk TDM runs.

Slots are handed out block-local; nothing indexes by slot order.

The payload is bf16/f32, fp8 or fp4; the last two carry a per-token e8m0 scale
row alongside it, padded to a 128-byte stride so a run of them is something the
engine can move. ``scale_bytes == 0`` compiles the whole scale path away.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.rocdl import ballot, ds_bpermute, mbcnt_lo, readfirstlane, readlane
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

from . import tdm_prims as TDM
from .config import (
    _LANE_MASK as LANE_MASK,
)
from .config import (
    _LOG2_WAVE_SIZE as LOG2_WAVE,
)
from .config import (
    _WAVE_SIZE as WAVE,
)

#: gfx1250 LDS per workgroup. The payload tiles are the whole budget.
_LDS_BUDGET = 327680


def _align(v, a):
    return (v + a - 1) // a * a


def tdm_tokens_per_wave(experts_per_token):
    """Tokens one wave covers per iteration: enough lanes for a token's routes.

    ``WAVE / topk`` when the routes tile the wave exactly, so COUNT reads
    ``topk`` consecutive indices per token with every lane busy. Otherwise one
    token per wave with the tail lanes idle -- correct, just wider.
    """
    topk = experts_per_token
    return WAVE // topk if (0 < topk <= WAVE and WAVE % topk == 0) else 1


def tdm_stage_capacity(*, npes, max_recv):
    """(per-peer destTokId capacity, total staging slots).

    Staging is indexed ``peer * cap + destTokId``: a block's reserved run
    ``[s_base, s_base+n)`` is already contiguous in destTokId space, so META can
    TDM-copy it without a block-local pack. ``max_recv`` is the live bound -- a
    destTokId at or past it is dropped.
    """
    return max_recv, npes * max_recv


def tdm_lds_bytes(*, hidden_dim, hidden_elem_size, warp_num_per_block, npes):
    """LDS the kernel asks for: one payload tile per warp, plus the counters."""
    tile = _align(hidden_dim * hidden_elem_size, 128)
    return warp_num_per_block * tile + _align(3 * npes * 4, 128)


def _tile_bytes(payload_bytes, slab_bytes):
    """A warp's LDS tile: the payload row, or the metadata batch if that is wider.

    The two share the tile, so it is sized in bytes rather than payload
    elements. Without the floor an fp4 payload would halve the tile exactly
    when the metadata grew by a scale row.
    """
    return _align(max(payload_bytes, slab_bytes), 128)


def tdm_max_warps(*, hidden_dim, hidden_elem_size, npes, slab_bytes=0):
    """Widest power-of-two warp count whose payload tiles fit the LDS budget.

    A 7168-wide bf16 tile is 14 KB and 32 of them want 448 KB against a 320 KB
    budget. A caller clamping to this keeps the tuned block count -- which is
    what paces the grid barrier -- and gives up only the warp width.
    """
    tile = _tile_bytes(hidden_dim * hidden_elem_size, slab_bytes)
    room = (_LDS_BUDGET - _align(3 * npes * 4, 128)) // tile
    if room < 1:
        raise ValueError(
            f"a single {tile}B payload tile (hidden_dim={hidden_dim}, "
            f"{hidden_elem_size}B elements) does not fit the {_LDS_BUDGET}B LDS "
            f"budget; the TDM dispatch cannot serve this hidden size"
        )
    warps = 1
    while warps * 2 <= room:
        warps *= 2
    return warps


# ── The compact payload wire, as a callable emitter ─────────────────────────
#
# Everything a compact dispatch does to a payload row lives here rather than
# inline in the kernel below, because a fused stage-1 kernel has to run this
# exact sequence off its own work partition. ``compact_plan`` has already
# chosen every route's destination row, so a producer only moves bytes and
# publishes the rowmap entry -- no histogram, no slot reservation, no metadata
# staging, none of the LDS or grid budget those need.
#
# These are ordinary Python functions executed while FlyDSL traces a
# ``@flyc.kernel`` body. They never read ``fx.block_idx``/``fx.thread_idx``:
# the work id, the lane and the LDS tile all come in as arguments, so the
# caller owns the partition and the LDS layout.


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactPayloadSpec:
    """Compile-time geometry of one compact dispatch payload row.

    ``compact_row_stride`` is the wire row -- the payload plane followed by the
    token's padded e8m0 scale plane -- and is what a destination row index is
    scaled by. ``tile_bytes`` is the LDS tile the caller lends the emitter: the
    whole wire row is assembled there, so it must hold one.

    ``tok_map`` on a compact wire is not the dispatch kernel's ``pe * max_recv
    + tok`` encoding; ``compact_plan`` packs ``(dest_tok << peer_bits) |
    dest_pe`` and leaves a dead route negative.
    """

    rank: int
    npes: int
    topk: int
    hidden_dim: int
    hidden_elem_size: int
    compact_row_stride: int
    tile_bytes: int
    max_tok_slot_stride: int
    off_out_tok: int
    off_ep_rowmap: int
    scale_bytes: int = 0

    def __post_init__(self):
        if WAVE != 32:
            raise ValueError(
                f"the compact payload wire is gfx1250-only (wave32); wave size "
                f"resolved to {WAVE}"
            )
        if self.npes < 1:
            raise ValueError(f"npes must be positive, got {self.npes}")
        if not 0 <= self.rank < self.npes:
            raise ValueError(
                f"rank must be in [0, {self.npes}), got {self.rank}"
            )
        if not 1 <= self.topk <= WAVE:
            raise ValueError(
                f"topk={self.topk} must fit one wave; a token's routes are "
                f"read by one lane each"
            )
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.hidden_elem_size not in (1, 2, 4, 8):
            raise ValueError(
                "TDM data_size encodes 1/2/4/8-byte elements, got "
                f"{self.hidden_elem_size}"
            )
        if self.payload_bytes % 128:
            raise ValueError(
                f"TDM payload rows must be 128-byte aligned, got "
                f"{self.payload_bytes}B"
            )
        if self.scale_bytes < 0 or self.scale_bytes % 4:
            raise ValueError(
                f"scale rows must be dword-sized, got {self.scale_bytes}B"
            )
        if (
            self.compact_row_stride < self.payload_bytes + self.scale_bytes
            or self.compact_row_stride % 128
        ):
            raise ValueError(
                "compact_row_stride must contain payload and scale and be "
                f"128-byte aligned, got {self.compact_row_stride}B"
            )
        if self.compact_row_stride > TDM.TDM_MAX_DIM:
            raise ValueError(
                f"a compact wire row of {self.compact_row_stride}B exceeds the "
                f"{TDM.TDM_MAX_DIM}-byte TDM extent"
            )
        if self.tile_bytes % 128 or self.tile_bytes < self.compact_row_stride:
            raise ValueError(
                "compact wire row does not fit a 128-byte aligned LDS tile: "
                f"row={self.compact_row_stride}B tile={self.tile_bytes}B"
            )
        if self.max_tok_slot_stride < 1:
            raise ValueError(
                "max_tok_slot_stride must be positive; it is what keeps one "
                "rank's packed rowmap slots off another's"
            )
        if self.off_out_tok < 0 or self.off_ep_rowmap < 0:
            raise ValueError("symmetric arena offsets must be non-negative")

    @property
    def payload_bytes(self):
        """Bytes of one source token row."""
        return self.hidden_dim * self.hidden_elem_size

    @property
    def scale_src_dw(self):
        """Dwords of the caller's packed e8m0 row; 0 on an unquantized wire."""
        return self.scale_bytes // 4

    @property
    def peer_bits(self):
        """Low bits of a packed ``tok_map`` entry holding ``dest_pe``."""
        return max(1, (self.npes - 1).bit_length())

    @property
    def peer_mask(self):
        return (1 << self.peer_bits) - 1


@dataclass(frozen=True, slots=True)
class CompactPayloadInvariants:
    """Per-wave values :func:`emit_compact_payload_token` would otherwise rebuild.

    The two GROUP1 descriptors fold to eight SGPR constants each and the lane
    probe is one select, so hoisting them out of a token loop keeps a
    multi-token caller from re-emitting both at every iteration.
    """

    group1_load: object
    group1_store: object
    probe_off: object


def compact_payload_invariants(spec, *, lane):
    """Build the loop-invariant half of the compact payload emitter."""
    return CompactPayloadInvariants(
        group1_load=TDM.tdm_group1(spec.hidden_dim, 1, spec.hidden_elem_size),
        group1_store=TDM.tdm_group1(spec.compact_row_stride, 1, 1),
        probe_off=arith.select(lane < spec.topk, lane, 0),
    )


@comm_ops.traced
def emit_compact_payload_token(
    spec,
    *,
    token,
    token_limit,
    lane,
    tile_addr,
    window,
    addr_inp_tok,
    rsrc_tok_map,
    rsrc_inp_wts,
    rsrc_inp_scale=None,
    invariants=None,
    on_route_complete=None,
):
    """Emit one token's compact payload sends: one TDM load, one store per route.

    Call this from a ``@flyc.kernel`` body with the whole wave converged.
    ``token`` is a wave-uniform source token id, guarded here against
    ``token_limit`` so a caller may hand over a padded partition.
    ``tile_addr`` is a wave-uniform i32 LDS byte address of a
    ``spec.tile_bytes`` tile the wave owns; it is overwritten. ``window`` is a
    ``cco.Window`` over the symmetric arena.

    ``on_route_complete`` is an optional per-route arrival callback, invoked
    once for each surviving route of this token, called wave-uniformly with
    every lane active, after the route's payload TDM store and its rowmap store
    have both been drained and a system release fence separates them from
    whatever the callback publishes. That ordering is the point of the hook: a
    consumer woken by it may read the row. It receives keyword arguments
    ``dest_pe`` and ``dest_tok`` (both wave-uniform i32, the destination row the
    route just landed on), plus ``token``, ``route`` (the python topk slot) and
    ``lane``; take ``**_`` so later arguments do not break the callback. A
    callback with dynamic control flow of its own has to be traceable the same
    way these emitters are -- nested in the kernel body, or decorated with
    ``@comm_ops.traced``.

    Without a callback the rowmap stores are left in flight -- the drain that
    covers them belongs with whatever the caller publishes next.
    """
    if spec.scale_bytes and rsrc_inp_scale is None:
        raise ValueError(
            "a quantized compact wire carries a per-token e8m0 row; "
            "rsrc_inp_scale is required when spec.scale_bytes > 0"
        )
    if invariants is None:
        invariants = compact_payload_invariants(spec, lane=lane)
    topk = spec.topk
    nbytes = spec.payload_bytes
    scale_src_dw = spec.scale_src_dw
    if token < token_limit:
        flat = buffer_load(
            rsrc_tok_map,
            token * topk + invariants.probe_off,
            vec_width=1,
            dtype=T.i32,
        )
        # `flat >= 0` rejects the host's -1 fill, so a slot the planner never
        # published can never name a route.
        live = (lane < topk) & (flat >= 0)
        live_mask = ballot(T.i32, live)
        wt_bits = arith.constant(0)
        packed_meta = arith.constant(0)
        if live:
            wt_bits = arith.bitcast(
                T.i32,
                buffer_load(
                    rsrc_inp_wts,
                    token * topk + lane,
                    vec_width=1,
                    dtype=T.f32,
                ),
            )
            packed_meta = (
                fx.Int32(spec.rank) * fx.Int32(spec.max_tok_slot_stride)
                + token * fx.Int32(topk)
                + lane
            )
        if live_mask != 0:
            TDM.tdm_load(
                TDM.tdm_group0(
                    tile_addr,
                    fx.Int64(addr_inp_tok) + fx.Int64(token) * fx.Int64(nbytes),
                ),
                invariants.group1_load,
            )
            # All compact rows of this token carry the same scale. Read it once
            # into the unused tail of this wave's LDS tile instead of
            # re-reading the global row per route.
            scale_lds = fx.Int64(tile_addr) + fx.Int64(nbytes)
            if const_expr(spec.scale_bytes > 0):
                scale_vec = 2 if scale_src_dw % 2 == 0 else 1
                for si in range(lane * scale_vec, scale_src_dw, WAVE * scale_vec):
                    val = buffer_load(
                        rsrc_inp_scale,
                        token * scale_src_dw + si,
                        vec_width=scale_vec,
                        dtype=T.i32,
                    )
                    if const_expr(scale_vec == 2):
                        comm_ops.store_i32_lds(
                            scale_lds + fx.Int64(si) * fx.Int64(4), val[0]
                        )
                        comm_ops.store_i32_lds(
                            scale_lds + fx.Int64(si + 1) * fx.Int64(4), val[1]
                        )
                    else:
                        comm_ops.store_i32_lds(
                            scale_lds + fx.Int64(si) * fx.Int64(4), val
                        )
                fx.rocdl.s_wait_dscnt(0)
            TDM.tdm_wait(0)
            # Unroll the live routes so each TDM descriptor is a distinct issue
            # site. The ctpop walk serializes too many SGPR descriptor rebuilds
            # through one loop.
            for k in range_constexpr(topk):
                bit = arith.constant(1 << k)
                if (live_mask & bit) != 0:
                    flat_l = readlane(T.i32, flat, k)
                    dest_pe = flat_l & spec.peer_mask
                    dest_tok = flat_l >> spec.peer_bits
                    TDM.tdm_store(
                        TDM.tdm_group0(
                            tile_addr,
                            fx.Int64(window.lsa_ptr(dest_pe, spec.off_out_tok))
                            + fx.Int64(dest_tok) * fx.Int64(spec.compact_row_stride),
                        ),
                        invariants.group1_store,
                    )
                    if lane == k:
                        buffer_store(
                            fx.Vector.from_elements(
                                [packed_meta, wt_bits], dtype=fx.Int32
                            ),
                            create_buffer_resource_from_addr(
                                fx.Int64(
                                    window.lsa_ptr(dest_pe, spec.off_ep_rowmap)
                                )
                            ),
                            dest_tok * 2,
                        )
            TDM.tdm_wait(0)
            if on_route_complete is not None:
                # The wait above retired the payload on the tensor counter,
                # which storecnt does not track, so the rowmap stores need
                # their own drain before anything says the row is readable.
                # The destinations are recomputed rather than carried out of
                # the store loop: they are two ALU ops off `flat`, and the
                # values there live inside a dynamic branch.
                comm_ops.waitcnt_stores()
                comm_ops.fence_system_release()
                for k in range_constexpr(topk):
                    bit = arith.constant(1 << k)
                    if (live_mask & bit) != 0:
                        flat_l = readlane(T.i32, flat, k)
                        on_route_complete(
                            dest_pe=flat_l & spec.peer_mask,
                            dest_tok=flat_l >> spec.peer_bits,
                            token=token,
                            route=k,
                            lane=lane,
                        )


@comm_ops.traced
def emit_compact_payload_rows(
    spec,
    *,
    work_id,
    work_stride,
    token_limit,
    lane,
    tile_addr,
    window,
    addr_inp_tok,
    rsrc_tok_map,
    rsrc_inp_wts,
    rsrc_inp_scale=None,
    on_route_complete=None,
):
    """Emit a wave's whole share of the compact payload, one token at a time.

    ``work_id``/``work_stride`` are the caller's partition -- a global wave id
    and the number of waves in the grid for the dispatch kernel, but any
    strided cover of ``[0, token_limit)`` does. ``work_stride`` must be
    positive at runtime: a zero step is a loop that never advances, a hang no
    correctness check can report because the check hangs with it.

    Unlike the dispatch kernel's token-major phase, every route of a token is
    read from ``tok_map`` here, which a compact wire may do from any wave --
    the planner filled ``tok_map`` in a prior kernel, so no entry depends on
    what this wave itself wrote. The remaining arguments are
    :func:`emit_compact_payload_token`'s.
    """
    invariants = compact_payload_invariants(spec, lane=lane)
    for token in range(work_id, token_limit, work_stride):
        emit_compact_payload_token(
            spec,
            token=token,
            token_limit=token_limit,
            lane=lane,
            tile_addr=tile_addr,
            window=window,
            addr_inp_tok=addr_inp_tok,
            rsrc_tok_map=rsrc_tok_map,
            rsrc_inp_wts=rsrc_inp_wts,
            rsrc_inp_scale=rsrc_inp_scale,
            invariants=invariants,
            on_route_complete=on_route_complete,
        )


def _make_dispatch_tdm(
    *,
    rank,
    npes,
    experts_per_rank,
    experts_per_token,
    hidden_dim,
    max_tok_per_rank,
    max_recv,
    block_num,
    warp_num_per_block,
    off_tok_off,
    off_recv_num,
    off_tis,
    off_out_idx,
    off_out_wts,
    off_out_tok,
    off_out_scales=0,
    scale_bytes=0,
    scale_stride=0,
    hidden_elem_size=2,
    slab_bytes=0,
    enable_signal=True,
    meta_tdm=True,
    clear_route_counter=False,
    compact_plan=False,
    compact_row_stride=0,
    off_ep_rowmap=0,
    max_tok_slot_stride=0,
):
    """Build the TDM dispatch kernel. Returns a ``@flyc.jit`` launcher.

    The launcher takes five extra pointers between ``addr_total_recv`` and
    ``my_lsa_rank`` -- four staging bases and the caller's scale buffer.
    ``meta_tdm=False`` routes the metadata through per-lane stores instead of
    the TDM engine.

    ``scale_bytes`` is the caller's packed e8m0 row and ``scale_stride`` the
    padded one the wire uses; ``slab_bytes`` floors the LDS tile so an fp4
    payload does not shrink the tile exactly when the metadata grew by a scale
    row. All three are 0 on a bf16 wire and the scale path disappears.
    """
    compact_plan = bool(compact_plan)
    compact_row_stride = int(compact_row_stride)
    off_ep_rowmap = int(off_ep_rowmap)
    max_tok_slot_stride = int(max_tok_slot_stride)
    if WAVE != 32:
        raise ValueError(
            f"TDM dispatch is gfx1250-only (wave32); wave size resolved to {WAVE}"
        )
    topk = experts_per_token
    nbytes = hidden_dim * hidden_elem_size
    if nbytes % 4:
        raise ValueError(
            f"token payload must be a whole number of dwords, got {nbytes}B"
        )
    if nbytes % 128:
        raise ValueError(f"TDM payload rows must be 128-byte aligned, got {nbytes}B")
    if compact_plan:
        if compact_row_stride < nbytes + scale_bytes or compact_row_stride % 128:
            raise ValueError(
                "compact_row_stride must contain payload and scale and be "
                f"128-byte aligned, got {compact_row_stride}B"
            )
    elif compact_row_stride:
        raise ValueError("compact_row_stride requires compact_plan")
    if scale_bytes:
        if scale_bytes % 4:
            raise ValueError(f"scale rows must be dword-sized, got {scale_bytes}B")
        if scale_stride < scale_bytes or scale_stride % 128:
            raise ValueError(
                "scale_stride must contain the source row and be 128-byte "
                f"aligned, got row={scale_bytes}B stride={scale_stride}B"
            )
        if not off_out_scales:
            raise ValueError("scale transport requires off_out_scales")
    elif scale_stride:
        raise ValueError("scale_stride requires a nonzero scale_bytes")
    scale_src_dw = scale_bytes // 4
    scale_dst_dw = scale_stride // 4
    if topk > WAVE:
        raise ValueError(
            f"topk={topk} exceeds the wave size; a token's routes must fit one wave"
        )

    tpi = tdm_tokens_per_wave(topk)
    warps_total = block_num * warp_num_per_block
    block_threads = warp_num_per_block * WAVE
    stg_cap, _stage_slots = tdm_stage_capacity(npes=npes, max_recv=max_recv)
    # sentinel: tok_map dropped-slot marker whose dest_pe (value // max_recv) == npes.
    sentinel_val = npes * max_recv

    tile_bytes = _tile_bytes(nbytes, slab_bytes)
    if compact_plan and compact_row_stride > tile_bytes:
        raise ValueError(
            "compact wire row does not fit the warp LDS tile: "
            f"row={compact_row_stride}B tile={tile_bytes}B"
        )
    ctl_bytes = _align(3 * npes * 4, 128)
    lds_bytes = warp_num_per_block * tile_bytes + ctl_bytes
    if lds_bytes > _LDS_BUDGET:
        raise ValueError(
            f"TDM dispatch needs {lds_bytes}B of LDS ({warp_num_per_block} warps x "
            f"{tile_bytes}B payload tile) over the {_LDS_BUDGET}B budget; lower "
            f"warp_num_per_block"
        )

    # The compact payload phase is shared with the fused stage-1 producer, so
    # its geometry is a value rather than a set of closed-over locals.
    if compact_plan:
        compact_spec = CompactPayloadSpec(
            rank=rank,
            npes=npes,
            topk=topk,
            hidden_dim=hidden_dim,
            hidden_elem_size=hidden_elem_size,
            compact_row_stride=compact_row_stride,
            tile_bytes=tile_bytes,
            max_tok_slot_stride=max_tok_slot_stride,
            off_out_tok=off_out_tok,
            off_ep_rowmap=off_ep_rowmap,
            scale_bytes=scale_bytes,
        )

    # A metadata batch reuses the warp's payload tile for four runs: idx,
    # weights, srcmap and (for mxfp4/a8w4) the padded e8m0 scale row. Only the
    # token count per batch is fixed here -- each run's split into a scalar
    # head, a TDM body (peeled rows or a short 2D whole-tile) and a scalar
    # tail is decided on device from the real source and destination addresses,
    # because a destination token id is a remote atomic's return value and
    # lands wherever it lands. 128B of tile slack per field pays for rounding
    # each body up to a 32-dword LDS region.
    meta_fields = 4 if scale_bytes else 3
    meta_per_tok = topk * 4 * 2 + 4 + scale_stride
    meta_cap = (tile_bytes - meta_fields * 128) // meta_per_tok
    meta_cap = min(meta_cap, TDM.TDM_MAX_DIM)
    use_meta_tdm = bool(meta_tdm) and meta_cap > 0

    # One warp per peer would leave every warp past the world size idle, so the
    # peers are split across the warps that exist. mori collapses that split to
    # 1 once ``numTokens <= warps * 2``: a whole-tile metadata TDM may rewrite
    # the rest of a straddled 128B block, and two warps sharing a peer's run
    # would clobber each other. Large batches keep the split -- each sub-run
    # is then long enough for a peeled row-split that owns whole rows.
    peer_split_max = max(1, warp_num_per_block // npes) if npes else 1

    @flyc.kernel(
        name=f"ep_dispatch_tdm_{block_num}x{warp_num_per_block}",
        known_block_size=[block_threads, 1, 1],
    )
    def ep_dispatch_tdm(
        arena: Int64,
        addr_inp_tok: Int64,
        addr_inp_idx: Int64,
        addr_inp_wts: Int64,
        addr_tok_map: Int64,
        addr_dest_pe_ctr: Int64,
        addr_disp_bar: Int64,
        addr_total_recv: Int64,
        addr_stg_idx: Int64,
        addr_stg_wt: Int64,
        addr_stg_src: Int64,
        addr_stg_scale: Int64,
        addr_inp_scale: Int64,
        addr_route_counter: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        global_warp_id = bid * warp_num_per_block + warp
        window = cco.Window(arena)

        rsrc_inp_idx = create_buffer_resource_from_addr(addr_inp_idx)
        rsrc_inp_wts = create_buffer_resource_from_addr(addr_inp_wts)
        rsrc_tok_map = create_buffer_resource_from_addr(addr_tok_map)
        rsrc_dest_ctr = create_buffer_resource_from_addr(addr_dest_pe_ctr)
        rsrc_disp_bar = create_buffer_resource_from_addr(addr_disp_bar)
        rsrc_stg_idx = create_buffer_resource_from_addr(addr_stg_idx)
        rsrc_stg_wt = create_buffer_resource_from_addr(addr_stg_wt)
        rsrc_stg_src = create_buffer_resource_from_addr(addr_stg_src)
        rsrc_route_counter = create_buffer_resource_from_addr(addr_route_counter)
        if const_expr(scale_bytes > 0):
            rsrc_inp_scale = create_buffer_resource_from_addr(addr_inp_scale)
            rsrc_stg_scale = create_buffer_resource_from_addr(addr_stg_scale)

        # LDS: `warp_num_per_block` payload tiles, then the three per-peer
        # counters (committed count / reserved remote base / handout cursor).
        smem = fx.SharedAllocator(static=False)
        tile_ptr = smem.allocate(warp_num_per_block * tile_bytes, 128)._ptr
        ctl_ptr = smem.allocate(ctl_bytes, 128)._ptr
        tile_base_i32 = arith.index_cast(
            T.i32, fx.index_cast(T.index, fx.ptrtoint(tile_ptr))
        )
        my_tile = arith.addi(
            tile_base_i32,
            arith.muli(
                readfirstlane(T.i32, warp), arith.constant(tile_bytes, type=T.i32)
            ),
        )
        ctl = fx.Int64(fx.ptrtoint(ctl_ptr))

        def s_n(p):
            return ctl + fx.Int64(p) * fx.Int64(4)

        def s_base(p):
            return ctl + (fx.Int64(p) + fx.Int64(npes)) * fx.Int64(4)

        def s_run(p):
            return ctl + (fx.Int64(p) + fx.Int64(2 * npes)) * fx.Int64(4)

        # Tokens a warp takes per iteration. A fixed quota of `tpi` only fills the
        # grid once there are `warps_total * tpi` tokens to go round: below that
        # every token lands on a low warp id, which is block-major, so most of the
        # grid sends no payload at all and the kernel costs the same at 8 tokens
        # as at 512. Capping the quota at what it takes to cover the grid spreads
        # them. COUNT gives up its full-warp index burst when the cap bites, which
        # is the cheaper half of that trade.
        #
        # The clamp to 1 is load-bearing: `ceil(n / warps_total)` is 0 for n == 0,
        # and a token loop stepping by `warps_total * 0` never advances -- a hang
        # no correctness check can report, because the check hangs with it.
        if const_expr(tpi > 1):
            q_tok = (inp_cur_tok + (warps_total - 1)) // warps_total
            etpi = arith.select(
                q_tok < 1,
                fx.Int32(1),
                arith.select(q_tok < tpi, q_tok, fx.Int32(tpi)),
            )
        else:
            etpi = fx.Int32(1)

        # Lane -> (which of the wave's tokens, which of that token's routes). The
        # grouping follows `etpi`, not `tpi`: left on `tpi` the surplus lanes
        # would keep routing tokens the loops below hand to another warp.
        if const_expr(tpi > 1):
            s_lane = lane // topk
            e_lane = lane - s_lane * topk
        else:
            s_lane = fx.Int32(0)
            e_lane = lane
        lane_act = (s_lane < etpi) & (e_lane < topk)

        if const_expr(not compact_plan):
            if tid < npes:
                comm_ops.store_i32_lds(s_n(tid), arith.constant(0))
                comm_ops.store_i32_lds(s_run(tid), arith.constant(0))
            fx.barrier()

        # Route resolution, shared by COUNT and FINALIZE. No dynamic `if`: it
        # runs under both phases' loops and selects rather than branches, so
        # inactive lanes stay in bounds instead of being masked off.
        def resolve(tok_base):
            tok = tok_base + s_lane
            act = lane_act & (tok < inp_cur_tok)
            tok_s = arith.select(tok < inp_cur_tok, tok, inp_cur_tok - 1)
            e_s = arith.select(lane_act, e_lane, 0)
            slot_off = tok_s * topk + e_s
            expert = buffer_load(rsrc_inp_idx, slot_off, vec_width=1, dtype=T.i32)
            dest_pe = expert // experts_per_rank
            valid = act & (expert >= 0) & (dest_pe < npes)
            # Dedup: a token routed to several experts on one peer is sent once.
            # A match-any on (s_lane, dest_pe): one ballot per peer (and per
            # token-slot when tpi>1), then mbcnt_lo picks the lowest lane in that
            # group -- cheaper than a topk x 2 ds_bpermute probe, which dominated
            # COUNT/FINALIZE at small batches.
            keep = valid & (e_lane < 0)  # False, same predicate type as valid
            zero_i32 = arith.constant(0, type=T.i32)
            if const_expr(tpi == 1):
                for p in range_constexpr(npes):
                    pred = valid & (dest_pe == p)
                    m = ballot(T.i32, pred)
                    below = mbcnt_lo(T.i32, m, zero_i32)
                    keep = keep | (pred & (below == 0))
            else:
                for s in range_constexpr(tpi):
                    for p in range_constexpr(npes):
                        pred = valid & (s_lane == s) & (dest_pe == p)
                        m = ballot(T.i32, pred)
                        below = mbcnt_lo(T.i32, m, zero_i32)
                        keep = keep | (pred & (below == 0))
            # slot_off rather than the weight itself: COUNT has no use for the
            # weight, so the load belongs in FINALIZE.
            return tok, act, expert, slot_off, dest_pe, keep

        # ── COUNT: block-local histogram of routes per destination peer ──
        if const_expr(not compact_plan):
            for tok_base in range(
                global_warp_id * etpi, inp_cur_tok, warps_total * etpi
            ):
                _tok, _act, _expert, _off, dest_pe, keep = resolve(tok_base)
                if keep:
                    comm_ops.atomic_add_lds(s_n(dest_pe), arith.constant(1))
            fx.barrier()

            # ── RESERVE: one remote atomic per (block, peer), not per route ──
            if tid < npes:
                n = comm_ops.load_i32_lds(s_n(tid))
                base = arith.constant(0)
                if n > 0:
                    base = comm_ops.atomic_add_system(
                        fx.Int64(window.lsa_ptr(tid, off_tok_off)), n
                    )
                    comm_ops.atomic_add_system(
                        fx.Int64(addr_dest_pe_ctr) + fx.Int64(tid) * fx.Int64(4), n
                    )
                comm_ops.store_i32_lds(s_base(tid), base)
            fx.barrier()

            # ── FINALIZE: hand out the reserved slots and gather the metadata ──
            # destTokId = s_base + block-local j; staging is peer-major destTokId
            # SoA, so a block's reserved run is already a contiguous TDM source.
            for tok_base in range(
                global_warp_id * etpi, inp_cur_tok, warps_total * etpi
            ):
                tok, act, expert, slot_off, dest_pe, keep = resolve(tok_base)
                wt = buffer_load(rsrc_inp_wts, slot_off, vec_width=1, dtype=T.f32)
                j = arith.constant(0)
                if keep:
                    j = comm_ops.atomic_add_lds(s_run(dest_pe), arith.constant(1))
                base = arith.constant(0)
                if keep:
                    base = comm_ops.load_i32_lds(s_base(dest_pe))
                dest_tok = base + j
                pub = keep & (dest_tok < stg_cap) & (dest_tok < max_recv)
                if act:
                    buffer_store(
                        arith.select(pub, dest_pe * max_recv + dest_tok, sentinel_val),
                        rsrc_tok_map,
                        tok * topk + e_lane,
                    )
                slot = dest_pe * stg_cap + dest_tok
                src_encoded = rank * max_tok_per_rank + tok
                pub_i = arith.select(pub, fx.Int32(1), fx.Int32(0))
                for e in range_constexpr(topk):
                    probe = (s_lane * topk + e) * 4
                    pub_e = ds_bpermute(T.i32, probe, pub_i)
                    slot_e = ds_bpermute(T.i32, probe, slot)
                    if lane_act & (pub_e != 0):
                        buffer_store(expert, rsrc_stg_idx, slot_e * topk + e_lane)
                        buffer_store(
                            arith.bitcast(T.i32, wt),
                            rsrc_stg_wt,
                            slot_e * topk + e_lane,
                        )
                        if e_lane == 0:
                            buffer_store(src_encoded, rsrc_stg_src, slot_e)
                        if const_expr(scale_bytes > 0):
                            # The caller's row is packed, the staged one is padded to
                            # `scale_stride` so a run of them starts on a TDM row. The
                            # token's route lanes split both halves between them.
                            for si in range(e_lane, scale_src_dw, topk):
                                buffer_store(
                                    buffer_load(
                                        rsrc_inp_scale,
                                        tok * scale_src_dw + si,
                                        vec_width=1,
                                        dtype=T.i32,
                                    ),
                                    rsrc_stg_scale,
                                    slot_e * scale_dst_dw + si,
                                )
                            # Zeroed rather than left over: the pad crosses into a
                            # peer's memory on the next TDM run.
                            for si in range(scale_src_dw + e_lane, scale_dst_dw, topk):
                                buffer_store(
                                    arith.constant(0),
                                    rsrc_stg_scale,
                                    slot_e * scale_dst_dw + si,
                                )

            # tok_map is written here and re-read by the payload phase, and the
            # staging arrays are written here and read by the metadata phase; both
            # go through global memory, so the stores have to land before either.
            comm_ops.waitcnt_stores()
            fx.barrier()

            # ── META: the staged runs leave as bulk cross-GPU writes ──
            def _copy_edge(src_rsrc, dst_rsrc, src_off, dst_off, head, body, total):
                """Scalar-copy the leading and trailing elements the body misses."""
                for i in range(lane, head, WAVE):
                    buffer_store(
                        buffer_load(src_rsrc, src_off + i, vec_width=1, dtype=T.i32),
                        dst_rsrc,
                        dst_off + i,
                    )
                for i in range(head + body + lane, total, WAVE):
                    buffer_store(
                        buffer_load(src_rsrc, src_off + i, vec_width=1, dtype=T.i32),
                        dst_rsrc,
                        dst_off + i,
                    )

            def _pad_body_elems(n):
                """Round a TDM body up to 32 dwords so the next LDS region stays 128B-aligned."""
                return arith.andi(
                    arith.addi(n, arith.constant(31, type=T.i32)),
                    arith.constant(-32, type=T.i32),
                )

            def _tdm_load(lds, glob, head, body, dim0, dim1):
                if body > 0:
                    TDM.tdm_load(
                        TDM.tdm_group0(lds, glob + fx.Int64(head) * fx.Int64(4)),
                        TDM.tdm_group1_2d(dim0, dim1),
                    )

            def _tdm_store(lds, glob, head, body, dim0, dim1):
                if body > 0:
                    TDM.tdm_store(
                        TDM.tdm_group0(lds, glob + fx.Int64(head) * fx.Int64(4)),
                        TDM.tdm_group1_2d(dim0, dim1),
                    )

            def _ship_meta(
                peer_id, n_tok, src_tok, dst_tok, p_idx, p_wts, p_tis, p_scales
            ):
                """Move one batch of staged metadata for ``n_tok`` tokens to a peer.

                Every field is planned on its own: they start at unrelated phases
                within a 128B row, so one can earn a TDM body where the next goes
                entirely scalar. A short leftover (or a phase mismatch) can still
                take a dense 2D whole-tile. Tile regions are packed at each body's
                size rounded up to 32 dwords, so every region stays as 128B-aligned
                as the tile base.
                """
                n_kv = n_tok * topk
                kv_bytes = fx.Int64(src_tok) * fx.Int64(topk * 4)
                kv_dbytes = fx.Int64(dst_tok) * fx.Int64(topk * 4)
                s_idx = fx.Int64(addr_stg_idx) + kv_bytes
                d_idx = fx.Int64(window.lsa_ptr(peer_id, off_out_idx)) + kv_dbytes
                s_wt = fx.Int64(addr_stg_wt) + kv_bytes
                d_wt = fx.Int64(window.lsa_ptr(peer_id, off_out_wts)) + kv_dbytes
                s_src = fx.Int64(addr_stg_src) + fx.Int64(src_tok) * fx.Int64(4)
                d_src = fx.Int64(window.lsa_ptr(peer_id, off_tis)) + fx.Int64(
                    dst_tok
                ) * fx.Int64(4)
                h_idx, b_idx, d0_idx, d1_idx = TDM.tdm_plan_xfer_4b(
                    s_idx, d_idx, n_kv
                )
                h_wt, b_wt, d0_wt, d1_wt = TDM.tdm_plan_xfer_4b(s_wt, d_wt, n_kv)
                h_src, b_src, d0_src, d1_src = TDM.tdm_plan_xfer_4b(
                    s_src, d_src, n_tok
                )
                l_idx = my_tile
                l_wt = l_idx + _pad_body_elems(b_idx) * 4
                l_src = l_wt + _pad_body_elems(b_wt) * 4
                _tdm_load(l_idx, s_idx, h_idx, b_idx, d0_idx, d1_idx)
                _tdm_load(l_wt, s_wt, h_wt, b_wt, d0_wt, d1_wt)
                _tdm_load(l_src, s_src, h_src, b_src, d0_src, d1_src)
                if const_expr(scale_bytes > 0):
                    n_sc = n_tok * scale_dst_dw
                    s_sc = fx.Int64(addr_stg_scale) + fx.Int64(src_tok) * fx.Int64(
                        scale_stride
                    )
                    d_sc = fx.Int64(window.lsa_ptr(peer_id, off_out_scales)) + fx.Int64(
                        dst_tok
                    ) * fx.Int64(scale_stride)
                    h_sc, b_sc, d0_sc, d1_sc = TDM.tdm_plan_xfer_4b(
                        s_sc, d_sc, n_sc
                    )
                    l_sc = l_src + _pad_body_elems(b_src) * 4
                    _tdm_load(l_sc, s_sc, h_sc, b_sc, d0_sc, d1_sc)
                # The edges are global-to-global and owe the tile nothing, so they
                # run while the loads above are still in flight.
                _copy_edge(
                    rsrc_stg_idx,
                    p_idx,
                    src_tok * topk,
                    dst_tok * topk,
                    h_idx,
                    b_idx,
                    n_kv,
                )
                _copy_edge(
                    rsrc_stg_wt, p_wts, src_tok * topk, dst_tok * topk, h_wt, b_wt, n_kv
                )
                _copy_edge(rsrc_stg_src, p_tis, src_tok, dst_tok, h_src, b_src, n_tok)
                if const_expr(scale_bytes > 0):
                    _copy_edge(
                        rsrc_stg_scale,
                        p_scales,
                        src_tok * scale_dst_dw,
                        dst_tok * scale_dst_dw,
                        h_sc,
                        b_sc,
                        n_sc,
                    )
                TDM.tdm_wait(0)
                _tdm_store(l_idx, d_idx, h_idx, b_idx, d0_idx, d1_idx)
                _tdm_store(l_wt, d_wt, h_wt, b_wt, d0_wt, d1_wt)
                _tdm_store(l_src, d_src, h_src, b_src, d0_src, d1_src)
                if const_expr(scale_bytes > 0):
                    _tdm_store(l_sc, d_sc, h_sc, b_sc, d0_sc, d1_sc)
                TDM.tdm_wait(0)

            split = arith.select(
                inp_cur_tok <= (warps_total * 2),
                fx.Int32(1),
                fx.Int32(peer_split_max),
            )
            n_runs = fx.Int32(npes) * split
            for run_id in range(warp, n_runs, warp_num_per_block):
                peer = run_id // split
                part = run_id - peer * split
                cnt_all = comm_ops.load_i32_lds(s_n(peer))
                base_all = comm_ops.load_i32_lds(s_base(peer))
                # Split the peer's run across `split` warps, remainder to the
                # low parts so the sub-runs differ by at most one token.
                q = cnt_all // split
                rem = cnt_all - q * split
                my_beg = part * q + arith.select(part < rem, part, rem)
                my_cnt = q + arith.select(part < rem, fx.Int32(1), fx.Int32(0))
                # RESERVE counted every route, FINALIZE published only those that
                # fit; without this the surplus names slots the peer never
                # allocated and the metadata run walks off its recv buffer.
                room = arith.constant(min(max_recv, stg_cap)) - (base_all + my_beg)
                my_cnt = arith.select(my_cnt < room, my_cnt, room)
                my_cnt = arith.select(my_cnt < 0, fx.Int32(0), my_cnt)
                peer_idx = create_buffer_resource_from_addr(
                    fx.Int64(window.lsa_ptr(peer, off_out_idx))
                )
                peer_wts = create_buffer_resource_from_addr(
                    fx.Int64(window.lsa_ptr(peer, off_out_wts))
                )
                peer_tis = create_buffer_resource_from_addr(
                    fx.Int64(window.lsa_ptr(peer, off_tis))
                )
                if const_expr(scale_bytes > 0):
                    peer_scales = create_buffer_resource_from_addr(
                        fx.Int64(window.lsa_ptr(peer, off_out_scales))
                    )
                stg_beg = peer * stg_cap + base_all + my_beg
                step = meta_cap if use_meta_tdm else 1
                for cs in range(0, my_cnt, step):
                    dst = base_all + my_beg + cs
                    src = stg_beg + cs
                    if const_expr(use_meta_tdm):
                        left = my_cnt - cs
                        n_tok = arith.select(left < step, left, fx.Int32(step))
                        _ship_meta(
                            peer,
                            n_tok,
                            src,
                            dst,
                            peer_idx,
                            peer_wts,
                            peer_tis,
                            peer_scales if scale_bytes else None,
                        )
                    else:
                        for i in range(lane, topk, WAVE):
                            buffer_store(
                                buffer_load(
                                    rsrc_stg_idx,
                                    src * topk + i,
                                    vec_width=1,
                                    dtype=T.i32,
                                ),
                                peer_idx,
                                dst * topk + i,
                            )
                            buffer_store(
                                buffer_load(
                                    rsrc_stg_wt,
                                    src * topk + i,
                                    vec_width=1,
                                    dtype=T.i32,
                                ),
                                peer_wts,
                                dst * topk + i,
                            )
                        if lane == 0:
                            buffer_store(
                                buffer_load(
                                    rsrc_stg_src, src, vec_width=1, dtype=T.i32
                                ),
                                peer_tis,
                                dst,
                            )
                        if const_expr(scale_bytes > 0):
                            for i in range(lane, scale_dst_dw, WAVE):
                                buffer_store(
                                    buffer_load(
                                        rsrc_stg_scale,
                                        src * scale_dst_dw + i,
                                        vec_width=1,
                                        dtype=T.i32,
                                    ),
                                    peer_scales,
                                    dst * scale_dst_dw + i,
                                )

        # ── PAYLOAD: one TDM load per token, one TDM store per surviving route ──
        # No barrier before this. The tile a warp is about to overwrite is the
        # one it just drained itself; the cross-warp state (staging, s_base) was
        # published by the barrier after FINALIZE.
        if const_expr(compact_plan):
            # The compact wire is the shared emitter's, warp-strided over the
            # grid: the planner filled tok_map in a prior low-LDS kernel, so
            # every warp may read every entry.
            emit_compact_payload_rows(
                compact_spec,
                work_id=global_warp_id,
                work_stride=warps_total,
                token_limit=inp_cur_tok,
                lane=lane,
                tile_addr=my_tile,
                window=window,
                addr_inp_tok=addr_inp_tok,
                rsrc_tok_map=rsrc_tok_map,
                rsrc_inp_wts=rsrc_inp_wts,
                rsrc_inp_scale=rsrc_inp_scale if scale_bytes else None,
            )
        else:
            # The token partition has to be FINALIZE's, walked one token at a
            # time. tok_map goes through global memory but the only barrier
            # between the two phases is a workgroup one, so a warp may read back
            # nothing but the entries it wrote itself. A grid-strided
            # `range(global_warp_id, ...)` reads slots other BLOCKS own: at 512
            # tokens on a 64x8 grid it is warps 0..127 (blocks 0..15) that route,
            # and every one of the remaining 48 blocks would send payload off
            # entries still holding the host's -1 fill. -1 passes a `< sentinel`
            # liveness test and decodes to dest_pe 0, dest_tok -1, i.e. a TDM
            # store one whole token BEFORE a peer's recv buffer -- an
            # out-of-bounds fabric write, which is what wedges the engine rather
            # than merely corrupting the result.
            #
            # `sub` is a runtime loop and not `range_constexpr` on purpose: the
            # route loop below already unrolls `topk` descriptor sites, and
            # unrolling this one too would multiply them by `tpi`.
            g_payload_load = TDM.tdm_group1(hidden_dim, 1, hidden_elem_size)
            g_payload_store = TDM.tdm_group1(hidden_dim, 1, hidden_elem_size)
            probe_off = arith.select(lane < topk, lane, 0)
            for tok_base in range(
                global_warp_id * etpi, inp_cur_tok, warps_total * etpi
            ):
                for sub in range(etpi):
                    tok = tok_base + sub
                    if tok < inp_cur_tok:
                        flat = buffer_load(
                            rsrc_tok_map,
                            tok * topk + probe_off,
                            vec_width=1,
                            dtype=T.i32,
                        )
                        # `flat >= 0` rejects the host's -1 fill as well as the
                        # sentinel, so a slot FINALIZE never published can never
                        # name a route.
                        live = (
                            (lane < topk) & (flat >= 0) & (flat < sentinel_val)
                        )
                        live_mask = ballot(T.i32, live)
                        if live_mask != 0:
                            TDM.tdm_load(
                                TDM.tdm_group0(
                                    my_tile,
                                    fx.Int64(addr_inp_tok)
                                    + fx.Int64(tok) * fx.Int64(nbytes),
                                ),
                                g_payload_load,
                            )
                            TDM.tdm_wait(0)
                            rest = live_mask
                            for _ in range(fx.ctpop(live_mask)):
                                src_lane = fx.cttz(rest)
                                rest = rest & (rest - 1)
                                flat_l = readlane(T.i32, flat, src_lane)
                                dest_pe = flat_l // max_recv
                                dest_tok = flat_l - dest_pe * max_recv
                                TDM.tdm_store(
                                    TDM.tdm_group0(
                                        my_tile,
                                        fx.Int64(window.lsa_ptr(dest_pe, off_out_tok))
                                        + fx.Int64(dest_tok) * fx.Int64(nbytes),
                                    ),
                                    g_payload_store,
                                )
                            TDM.tdm_wait(0)

        if const_expr(enable_signal and compact_plan):
            TDM.tdm_wait(0)
            comm_ops.waitcnt_stores()
            fx.barrier()

            arrive_ticket = arith.constant(-1)
            if tid == 0:
                arrive_ticket = comm_ops.atomic_add_system(
                    fx.Int64(addr_disp_bar), arith.constant(1)
                )
            if warp == 0:
                arrive_ticket = readlane(T.i32, arrive_ticket, 0)
            is_last = (warp == 0) & (arrive_ticket == block_num - 1)
            if is_last:
                if lane == 0:
                    buffer_store(arith.constant(0), rsrc_disp_bar, 0)
                comm_ops.fence_system_release()
                for dest_pe in range(lane, npes, WAVE):
                    recv_num_remote_addr = fx.Int64(
                        window.lsa_ptr(dest_pe, off_recv_num)
                    ) + fx.Int64(rank) * fx.Int64(4)
                    comm_ops.store_i32_system(
                        recv_num_remote_addr, arith.constant(0), arith.constant(1)
                    )
                comm_ops.waitcnt_stores()

                local_recv_num = fx.Int64(window.lsa_ptr(my_lsa_rank, off_recv_num))
                for src_pe in range(lane, npes, WAVE):
                    recv_num_src_addr = local_recv_num + fx.Int64(src_pe) * fx.Int64(4)
                    comm_ops.spin_until_gt_i32(recv_num_src_addr, 0)
                    comm_ops.store_i32_system(
                        recv_num_src_addr, arith.constant(0), arith.constant(0)
                    )

        if const_expr(enable_signal and not compact_plan):
            # TDM stores retire on the tensor counter, which storecnt does not
            # track, so the grid barrier needs both drains to cover the payload
            # and the plain stores.
            TDM.tdm_wait(0)
            comm_ops.waitcnt_stores()
            fx.barrier()

            # Tickets on the arrival counter instead of a spin on it: every
            # block draws one, the first arriver draws a second once its drain
            # read is done, and the highest draw implies both preconditions of
            # the signal store -- every block is in, so dest_pe_ctr is a
            # complete sum, and the peer's mailbox has been observed empty.
            #
            # The point of the second ticket is that the drain is uncached peer
            # memory, a full fabric round trip even when the slot has long been
            # zero. Parking it on the first block to finish rather than on
            # block 0 overlaps it with every other block's remaining work;
            # block 0 has no reason to be the one that finishes first.
            no_ticket = -1
            arrive_ticket = arith.constant(no_ticket)
            if tid == 0:
                arrive_ticket = comm_ops.atomic_add_system(
                    fx.Int64(addr_disp_bar), arith.constant(1)
                )
            if warp == 0:
                arrive_ticket = readlane(T.i32, arrive_ticket, 0)
            is_first = (warp == 0) & (arrive_ticket == 0)

            local_recv_num = fx.Int64(window.lsa_ptr(my_lsa_rank, off_recv_num))
            drain_ticket = arith.constant(no_ticket)
            if is_first:
                for dest_pe in range(lane, npes, WAVE):
                    comm_ops.spin_until_eq_i32(
                        fx.Int64(window.lsa_ptr(dest_pe, off_recv_num))
                        + fx.Int64(rank) * fx.Int64(4),
                        0,
                    )
                comm_ops.waitcnt_stores()
                if lane == 0:
                    drain_ticket = comm_ops.atomic_add_system(
                        fx.Int64(addr_disp_bar), arith.constant(1)
                    )
                drain_ticket = readlane(T.i32, drain_ticket, 0)

            # atomic_add returns the pre-increment value, so the block_num + 1
            # draws read 0..block_num: the highest ticket is the grid size, not
            # the counter's final value.
            holds_highest = (warp == 0) & (
                (arrive_ticket == block_num) | (drain_ticket == block_num)
            )
            if holds_highest:
                # Reset once, outside the peer loop: a wide EP (npes > wave)
                # runs that loop more than once per lane, and a second pass
                # waiting on a barrier it had already consumed would hang.
                if lane == 0:
                    buffer_store(arith.constant(0), rsrc_disp_bar, 0)
                # The following route kernel uses this local array for slot
                # atomics. This tail already has a unique owner, so folding the
                # reset here needs neither another launch nor a grid barrier.
                if const_expr(clear_route_counter):
                    for expert in range(lane, experts_per_rank, WAVE):
                        buffer_store(arith.constant(0), rsrc_route_counter, expert)
                for dest_pe in range(lane, npes, WAVE):
                    recv_num_remote_addr = fx.Int64(
                        window.lsa_ptr(dest_pe, off_recv_num)
                    ) + fx.Int64(rank) * fx.Int64(4)
                    signal_value = (
                        buffer_load(rsrc_dest_ctr, dest_pe, vec_width=1, dtype=T.i32)
                        + 1
                    )
                    comm_ops.store_i32_system(
                        recv_num_remote_addr, arith.constant(0), signal_value
                    )
                    # Cleared here and not in the inbound loop: that one runs on
                    # another block now, and a peer signalling this rank says
                    # nothing about whether this rank has read its own counts.
                    buffer_store(arith.constant(0), rsrc_dest_ctr, dest_pe)

            # Inbound rides the first arriver so it is already parked when the
            # peers' signals land. It cannot just move above the outbound store
            # on the same warp: this rank would wait on a peer that is waiting
            # on this rank, with neither having sent. The separate tickets are
            # what break that cycle.
            if is_first:
                if lane == 0:
                    comm_ops.store_i32_system(
                        fx.Int64(addr_total_recv),
                        arith.constant(0),
                        arith.constant(0),
                    )
                # Same wave, so the reset is ahead of every lane's add in
                # program order; the wait is what makes it ahead in memory too.
                comm_ops.waitcnt_stores()
                for src_pe in range(lane, npes, WAVE):
                    recv_num_src_addr = local_recv_num + fx.Int64(src_pe) * fx.Int64(4)
                    signal_value = comm_ops.spin_until_gt_i32(recv_num_src_addr, 0)
                    comm_ops.store_i32_system(
                        recv_num_src_addr, arith.constant(0), arith.constant(0)
                    )
                    comm_ops.atomic_add_system(
                        fx.Int64(addr_total_recv), signal_value - 1
                    )
                if lane == 0:
                    local_tok_off = fx.Int64(window.lsa_ptr(my_lsa_rank, off_tok_off))
                    comm_ops.store_i32_system(
                        local_tok_off, arith.constant(0), arith.constant(0)
                    )

    @flyc.jit
    def run(
        arena: Int64,
        addr_inp_tok: Int64,
        addr_inp_idx: Int64,
        addr_inp_wts: Int64,
        addr_tok_map: Int64,
        addr_dest_pe_ctr: Int64,
        addr_disp_bar: Int64,
        addr_total_recv: Int64,
        addr_stg_idx: Int64,
        addr_stg_wt: Int64,
        addr_stg_src: Int64,
        addr_stg_scale: Int64,
        addr_inp_scale: Int64,
        addr_route_counter: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        ep_dispatch_tdm(
            arena,
            addr_inp_tok,
            addr_inp_idx,
            addr_inp_wts,
            addr_tok_map,
            addr_dest_pe_ctr,
            addr_disp_bar,
            addr_total_recv,
            addr_stg_idx,
            addr_stg_wt,
            addr_stg_src,
            addr_stg_scale,
            addr_inp_scale,
            addr_route_counter,
            my_lsa_rank,
            inp_cur_tok,
        ).launch(
            grid=(block_num, 1, 1),
            block=[block_threads, 1, 1],
            stream=stream,
        )

    return run
