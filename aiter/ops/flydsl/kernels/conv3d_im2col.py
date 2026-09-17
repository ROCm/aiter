# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Advanced Micro Devices, Inc.

"""The im2col side of the implicit-GEMM conv3d.

An implicit GEMM differs from an ordinary one in one place only. B is a plain
(N, K) matrix, but A is not a matrix at all: each of its (row, k) entries is a
tap into NDHWC input, reached by decomposing the row into (n, ot, oh, ow) and
k into (c, kt, kh, kw). This module owns that decomposition and the addressing
it implies, which is what lets the main loop in ``conv3d_implicit_gfx950``
read like the one in ``gemm_a16w16_gfx950``: its ``async_load_a_to_lds`` walks
the taps of a K tile and issues one DMA each, and never mentions a
convolution.

Geometry against addressing is where the split falls. The output extents, npq
and CRS belong to the caller -- the grid and the epilogue are shaped by them
too, so they are derived once there and passed in. What lives here is what
only the gather cares about: the tap fixup each padding mode implies, the
div/mod folding of the K axis against C/groups and the filter extents, and the
buffer descriptors, including the rebasing a 32-bit voffset needs when the
input outgrows what one can reach.
"""

from typing import NamedTuple

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from .conv3d_gfx950_utils import (
    BF16_BYTES,
    LDG_VEC,
    OOB_SENTINEL_BYTES,
    OOB_SENTINEL_ELEM,
    ConvGeometry,
    TileConfig,
    dil,
    flat_buffer_view,
    gather_valid,
    in_range,
)

PADDING_MODES = ("zeros", "reflect", "replicate", "circular")

# num_records of a rebased BIG_IN resource: the most a 32-bit voffset reaches.
BIG_IN_NR = 0x80000000


class Im2colPlan(NamedTuple):
    """What one conv's gather knows before the kernel runs.

    Every field is a compile-time constant -- folding the div/mod of the K axis
    against the filter extents and C/groups is where much of this kernel's
    performance comes from -- so a plan belongs to one compiled artifact.
    Build it with ``make_im2col_plan``, which derives the addressing decisions
    and asserts what this gather cannot express.

    A NamedTuple rather than a dataclass, and that is load-bearing: FlyDSL
    keys a compiled kernel on its source plus the *scalar* values its closure
    captures, where scalar means int/float/bool/str/None/tuple/Enum. An
    ordinary object is not one, so a plan held as anything else would drop
    every constant in it out of the cache key, and two convolutions that
    differ only in padding mode or dilation would silently share one binary.
    """

    # Input and filter. Flat rather than the param struct they come from,
    # because only tuples and scalars reach the cache key (see above).
    c: int
    d: int
    h: int
    w: int
    kh: int
    kw: int
    st: int
    sh: int
    sw: int
    pt: int
    ph: int
    pw: int
    dt: int
    dh: int
    dw: int
    pad_mode: str
    groups: int

    # The output grid the rows decompose against. Nested, which a NamedTuple
    # may be: it is a tuple, so its own fields still reach the cache key.
    geom: ConvGeometry

    # The A tile this gather fills. Nested, which a NamedTuple may be: it is
    # a tuple, so its own fields still reach the cache key.
    cfg: TileConfig

    # Addressing decisions derived from the above.
    temporal_only_fast: bool
    scalar_k: bool
    big_in: bool
    big_in_n1: bool
    big_in_nm: bool
    t_aligned: bool
    x_bytes: int
    x_sample_elems: int


def make_im2col_plan(param, geom, cfg):
    """An Im2colPlan for one problem and launch config, or an assertion.

    Takes the problem as the ``Conv3dImplicitParam`` the caller already has,
    and the grid as the ``ConvGeometry`` it already derived, so the two cannot
    disagree with what the rest of the kernel was built against.

    The asserts here are the ones about reach: whether the input fits what a
    buffer descriptor addresses, on its own and as rebased per sample or per
    tile. They are the gather's, not the launch config's, so they cannot move
    into ``validate_launch_config``.
    """
    tile_m, tile_k = cfg.tile_m, cfg.tile_k
    n, c, d, h, w = param.n, param.c, param.d, param.h, param.w
    kt, kh, kw = param.kt, param.kh, param.kw
    st, sh, sw = param.st, param.sh, param.sw
    pt, ph, pw = param.pt, param.ph, param.pw
    dt, dh, dw = param.dt, param.dh, param.dw
    pad_mode, groups = param.pad_mode, param.groups
    do, ho, wo, hw_o = geom.do, geom.ho, geom.wo, geom.hw_o

    assert pad_mode in PADDING_MODES, (
        f"pad_mode must be one of {PADDING_MODES}, got {pad_mode!r}"
    )

    x_elems = n * c * d * h * w
    x_bytes = x_elems * BF16_BYTES
    big_in = x_elems > 0x7FFFFFFF
    assert x_bytes < OOB_SENTINEL_BYTES or big_in, f"input {x_bytes}B exceeds limit"
    assert pad_mode == "zeros" or not big_in, (
        "non-zero pad_mode requires the non-BIG_IN address path"
    )

    big_in_n1 = big_in and n == 1
    big_in_nm = big_in and n > 1
    x_sample_elems = c * d * h * w

    # n > 1 rebases the descriptor once per sample, so a tap can sit anywhere in
    # the sample and the whole sample has to fit the 2 GB num_records -- unlike
    # the per-tile rebasing below, whose reach is bounded by the tile. Without
    # this check, taps past 2 GB fall outside num_records and read as zero, which
    # is silently wrong rather than an error. Note how little room that leaves:
    # BIG_IN needs n * sample > 2 GiB of elements, so at n == 2 the only sample
    # size that both trips BIG_IN and fits is exactly 2 GB.
    assert not big_in_nm or x_sample_elems * BF16_BYTES <= BIG_IN_NR, (
        f"batched input sample too large for the 32-bit gather: one sample spans "
        f"{x_sample_elems * BF16_BYTES / 2**30:.2f} GiB, past the "
        f"{BIG_IN_NR / 2**30:.0f} GiB the per-sample buffer descriptor addresses. "
        f"Loop over N instead of batching."
    )

    t_aligned = big_in_n1 and hw_o % tile_m == 0
    if big_in_n1:
        ot_span = (tile_m - 1) // hw_o + (1 if t_aligned else 2)
        t_span = min(d - 1, (ot_span - 1) * st + dt * (kt - 1))
        h_span = (
            min(h - 1, ((tile_m - 1) // wo + 1) * sh + dh * (kh - 1))
            if t_aligned
            else h - 1
        )
        span = (((t_span * h + h_span) * w + (w - 1)) * c + c) * BF16_BYTES
        assert span <= BIG_IN_NR, (
            f"input sample too large for the 32-bit gather: a {tile_m}-row tile reaches "
            f"{span / 2**30:.2f} GiB from its rebased origin, past the "
            f"{BIG_IN_NR / 2**30:.0f} GiB the buffer descriptor addresses. Split the batch "
            f"over N, or pass a narrower tile=(TILE_M, ...)."
        )

    return Im2colPlan(
        c=c,
        d=d,
        h=h,
        w=w,
        kh=kh,
        kw=kw,
        st=st,
        sh=sh,
        sw=sw,
        pt=pt,
        ph=ph,
        pw=pw,
        dt=dt,
        dh=dh,
        dw=dw,
        pad_mode=pad_mode,
        groups=groups,
        geom=geom,
        cfg=cfg,
        # A 1x1 filter at unit stride and no spatial padding leaves the H and W
        # taps fixed, so the row's own offset already addresses them and only
        # the T tap moves: a whole filter's worth of division collapses.
        temporal_only_fast=(
            kh == 1
            and kw == 1
            and st == 1
            and sh == 1
            and sw == 1
            and ph == 0
            and pw == 0
            and do == d
            and ho == h
            and wo == w
        ),
        # A K tile that never straddles a channel boundary makes the channel and
        # the filter tap uniform across the tile, so they are hoisted per tile
        # rather than recomputed per tap.
        scalar_k=geom.cgp % tile_k == 0,
        big_in=big_in,
        big_in_n1=big_in_n1,
        big_in_nm=big_in_nm,
        t_aligned=t_aligned,
        x_bytes=x_bytes,
        x_sample_elems=x_sample_elems,
    )


class Im2colGather:
    """The gather of one kernel, then of one block.

    Two stages because the kernel's input descriptor and the block's rows are
    fixed at different points: build this where the other buffer descriptors
    are built, and ``bind_block`` once the block knows its own ``m_offset``,
    which is all that decides which output element each A vector belongs to.
    ``taps`` then walks a K tile, handing back the source and offset of every
    A vector for the caller to DMA.
    """

    def __init__(self, plan, x):
        self._plan = plan
        self._x = x
        self._tid = self._ch_base = None
        self._nbase = self._base_t = self._base_h = None
        self._rows = None
        # BIG_IN has no kernel-wide descriptor: it rebases per block (n == 1)
        # or per tap's own sample (n > 1), neither of which is known yet.
        self._x_src = (
            None
            if const_expr(plan.big_in)
            else flat_buffer_view(
                fx.get_iter(x), plan.x_bytes // BF16_BYTES, plan.x_bytes
            )
        )

    def bind_block(self, tid, m_offset, ch_base=None):
        """Resolve this block's rows, and its descriptor if it rebases per block."""
        self._tid = tid
        self._ch_base = ch_base
        if const_expr(self._plan.big_in_n1):
            # One sample, too large to address whole: rebase the descriptor on
            # the corner of the input this block's own tile reaches back to.
            self._rebase_on_tile(m_offset)
        self._rows = self._decode_rows(m_offset)

    def taps(self, k_base):
        """Yield ``(i, src, voff)`` per A vector of the K tile at ``k_base``."""
        assert self._rows is not None, "bind_block() before taps()"
        plan, geom, cfg = self._plan, self._plan.geom, self._plan.cfg
        kbase_i = fx.Int64(k_base)
        cc_base = ckk_base = None
        if const_expr(plan.scalar_k):
            cc_base = kbase_i % geom.cgp
            if const_expr(plan.groups > 1):
                cc_base = self._ch_base + cc_base
            ckk_base = kbase_i // geom.cgp
        for i in range_constexpr(cfg.ldg_a_count):
            g_off, valid, sample = self._tap_addr(i, kbase_i, cc_base, ckk_base)
            yield (
                i,
                self._tap_src(sample),
                valid.select(g_off, fx.Int32(OOB_SENTINEL_ELEM)),
            )

    def _rebased(self, off_elems):
        """A descriptor based ``off_elems`` into the input, capped at its reach."""
        ptr = fx.add_offset(fx.get_iter(self._x), fx.make_int_tuple(off_elems))
        return flat_buffer_view(ptr, BIG_IN_NR // BF16_BYTES, BIG_IN_NR)

    def _rebase_on_tile(self, m_offset):
        plan, geom = self._plan, self._plan.geom
        self._nbase = m_offset // geom.dhw
        rem0 = m_offset % geom.dhw
        ot_base0 = rem0 // geom.hw_o

        self._base_t = fx.max(
            ot_base0 * fx.Int64(plan.st) - fx.Int64(plan.pt), fx.Int64(0)
        )
        if const_expr(plan.t_aligned):
            oh_base0 = (rem0 % geom.hw_o) // geom.wo
            self._base_h = fx.max(
                oh_base0 * fx.Int64(plan.sh) - fx.Int64(plan.ph), fx.Int64(0)
            )
        else:
            self._base_h = fx.Int64(0)
        base_row = (self._nbase * fx.Int64(plan.d) + self._base_t) * fx.Int64(
            plan.h
        ) + self._base_h
        x_base_elem = base_row * fx.Int64(plan.w) * fx.Int64(plan.c)
        self._x_src = self._rebased(fx.Int64(x_base_elem))

    def _tap_src(self, sample):
        if const_expr(self._plan.big_in_nm):
            return self._rebased(fx.Int64(sample) * fx.Int64(self._plan.x_sample_elems))
        return self._x_src

    def _decode_rows(self, m_offset):
        """Per A vector, the output element its GEMM row is, as (n, ot, oh, ow).

        Held as the input coordinate each of those taps starts from, since the
        filter tap is all that is added per K tile.
        """
        plan, geom, cfg = self._plan, self._plan.geom, self._plan.cfg
        rows = []
        for i in range_constexpr(cfg.ldg_a_count):
            linear = (self._tid + i * cfg.block_threads) * LDG_VEC
            local_m = linear // cfg.tile_k
            local_k = linear % cfg.tile_k
            row = m_offset + local_m
            row_valid = row < fx.Int64(geom.npq)
            if const_expr(plan.temporal_only_fast):
                out_t = (row // geom.hw_o) % plan.d
                rows.append((local_k, row, row_valid, out_t))
            else:
                n_idx = row // geom.dhw
                rem = row % geom.dhw
                ot = rem // geom.hw_o
                rem2 = rem % geom.hw_o
                oh = rem2 // geom.wo
                ow = rem2 % geom.wo
                in_t0 = ot * plan.st - plan.pt
                in_h0 = oh * plan.sh - plan.ph
                in_w0 = ow * plan.sw - plan.pw
                # A tile-rebased descriptor is based on this block's own sample,
                # so the row's sample index becomes an offset from that one.
                n_or_di = (n_idx - self._nbase) if const_expr(plan.big_in_n1) else n_idx
                rows.append((local_k, row_valid, n_or_di, in_t0, in_h0, in_w0))
        return rows

    def _pad_coord(self, v, ext, pad):
        """Tap coordinate -> in-bounds input coordinate; returns (coord, mask).

        "zeros" leaves the coordinate alone and returns a range mask, which the
        caller folds into the OOB-sentinel routing so the load reads as zero. Every
        other mode resolves the coordinate into [0, ext) instead and returns no mask
        """
        pad_mode = self._plan.pad_mode
        if const_expr(pad_mode == "zeros"):
            return v, in_range(v, ext)
        u = v + fx.Int64(pad)
        low = u < fx.Int64(pad)  # v < 0
        high = u >= fx.Int64(pad + ext)  # v >= ext
        mid = u - fx.Int64(pad)  # v, where in range
        if const_expr(pad_mode == "replicate"):
            r = high.select(fx.Int64(ext - 1), mid)
            r = low.select(fx.Int64(0), r)
        elif const_expr(pad_mode == "reflect"):
            # [a b c d e] pad 2 -> [c b a b c d e d c]: -v near, 2*(ext-1) - v far.
            r = high.select(fx.Int64(2 * (ext - 1) + pad) - u, mid)
            r = low.select(fx.Int64(pad) - u, r)
        else:  # circular: v + ext near, v - ext far
            r = high.select(u - fx.Int64(pad + ext), mid)
            r = low.select(u + fx.Int64(ext - pad), r)
        return r, None

    def _tap_addr(self, i, kbase_i, cc_base, ckk_base):
        """A vector ``i`` of this K tile as (element offset, valid, sample).

        The K axis decomposes against CGP (per-group channels) while every
        offset below keeps ``c`` (padded total channels) as the NDHWC row
        stride. ``cc`` is the absolute input channel: the group base plus the
        offset within the group. ``sample`` is which sample to rebase on, and
        only the per-sample descriptor path has one.
        """
        plan, geom = self._plan, self._plan.geom
        dec = self._rows[i]
        local_k = dec[0]
        k_abs = kbase_i + fx.Int64(local_k)
        if const_expr(plan.scalar_k):
            cc = cc_base + fx.Int64(local_k)  # cc_base already carries ch_base
        else:
            cc = k_abs % geom.cgp
            if const_expr(plan.groups > 1):
                cc = self._ch_base + cc
        k_valid = k_abs < fx.Int64(geom.crs)

        if const_expr(plan.temporal_only_fast):
            _, row, row_valid, out_t = dec
            kt_i = ckk_base if const_expr(plan.scalar_k) else k_abs // geom.cgp
            temporal_delta = dil(kt_i, plan.dt) - plan.pt
            in_t, m_t = self._pad_coord(out_t + temporal_delta, plan.d, plan.pt)
            valid = gather_valid(row_valid & k_valid, m_t)

            delta = (
                temporal_delta
                if const_expr(plan.pad_mode == "zeros")
                else (in_t - out_t)
            )
            if const_expr(plan.big_in_n1):
                g_off = (
                    (row + delta * geom.hw_o)
                    - (fx.Int64(self._nbase) * geom.dhw + self._base_t * geom.hw_o)
                ) * plan.c + cc
            else:
                g_off = (row + delta * geom.hw_o) * plan.c + cc
            return fx.Int32(g_off), valid, None

        ckk = ckk_base if const_expr(plan.scalar_k) else k_abs // geom.cgp
        kw_i = ckk % plan.kw
        ckk2 = ckk // plan.kw
        kh_i = ckk2 % plan.kh
        kt_i = ckk2 // plan.kh
        _, row_valid, n_or_di, in_t0, in_h0, in_w0 = dec
        in_t, m_t = self._pad_coord(in_t0 + dil(kt_i, plan.dt), plan.d, plan.pt)
        in_h, m_h = self._pad_coord(in_h0 + dil(kh_i, plan.dh), plan.h, plan.ph)
        in_w, m_w = self._pad_coord(in_w0 + dil(kw_i, plan.dw), plan.w, plan.pw)
        valid = gather_valid(row_valid & k_valid, m_t, m_h, m_w)
        if const_expr(plan.big_in_n1):
            row_off = (
                (n_or_di * plan.d + (in_t - self._base_t)) * plan.h
                + (in_h - self._base_h)
            ) * plan.w + in_w
            return fx.Int32(row_off * plan.c + cc), valid, None
        if const_expr(plan.big_in_nm):
            g_off = ((in_t * plan.h + in_h) * plan.w + in_w) * plan.c + cc
            return fx.Int32(g_off), valid, n_or_di
        g_off = (
            ((n_or_di * plan.d + in_t) * plan.h + in_h) * plan.w + in_w
        ) * plan.c + cc
        return fx.Int32(g_off), valid, None
