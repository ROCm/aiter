# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Im2col gather for implicit-GEMM conv3d: tap addressing, padding, buffer rebasing."""

from typing import NamedTuple

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from .conv3d_gfx950_utils import (
    BF16_BYTES,
    LDG_VEC,
    OOB_SENTINEL_BYTES,
    OOB_SENTINEL_ELEM,
    TileConfig,
    dil,
    flat_buffer_view,
    gather_valid,
    in_range,
)

PADDING_MODES = ("zeros", "reflect", "replicate", "circular")

# num_records of a rebased BIG_IN resource: the most a 32-bit voffset reaches.
BIG_IN_NR = 0x80000000


def _bytes_of(elems):
    """Element count -> byte count, as a literal where the count is one.

    num_records is a plain integer to the descriptor, so a literal stays an
    immediate and a runtime extent becomes the scalar it is.
    """
    return (
        elems * BF16_BYTES if isinstance(elems, int) else fx.Int32(elems) * BF16_BYTES
    )


class Im2colPlan(NamedTuple):
    """Gather compile constants. NamedTuple so FlyDSL keys on the fields."""

    # Spatial extents live in ConvExtents, not here (dyn_hw cache key).
    c: int
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

    # The two parts of ``ConvGeometry`` the gather needs, flattened out. Both
    # follow from C/groups and the filter, never from the resolution, so they
    # stay compile-time while the output extents next to them in the geometry
    # do not; holding the whole struct would drag those along.
    cgp: int
    crs: int

    # The A tile this gather fills. Nested, which a NamedTuple may be: it is
    # a tuple, so its own fields still reach the cache key.
    cfg: TileConfig

    # Addressing decisions derived from the above. All booleans, and that is
    # what keeps them here: a boolean in the key costs a constant number of
    # artifacts, while the extent it was derived from would cost one per
    # resolution. The host derives them from the real shape either way.
    temporal_only_fast: bool
    scalar_k: bool
    big_in: bool
    big_in_n1: bool
    big_in_nm: bool
    t_aligned: bool


def make_im2col_plan(param, geom, cfg):
    """Im2colPlan, or an assertion about gather reach."""
    tile_m, tile_k = cfg.tile_m, cfg.tile_k
    n, c, d, h, w = param.n, param.c, param.d, param.h, param.w
    kt, kh, kw = param.kt, param.kh, param.kw
    st, sh, sw = param.st, param.sh, param.sw
    pt, ph, pw = param.pt, param.ph, param.pw
    dt, dh, dw = param.dt, param.dh, param.dw
    pad_mode, groups = param.pad_mode, param.groups
    do, ho, wo, hw_o = geom.do, geom.ho, geom.wo, geom.hw_o

    assert (
        pad_mode in PADDING_MODES
    ), f"pad_mode must be one of {PADDING_MODES}, got {pad_mode!r}"

    x_elems = n * c * d * h * w
    x_bytes = x_elems * BF16_BYTES
    big_in = x_elems > 0x7FFFFFFF
    assert x_bytes < OOB_SENTINEL_BYTES or big_in, f"input {x_bytes}B exceeds limit"
    assert (
        pad_mode == "zeros" or not big_in
    ), "non-zero pad_mode requires the non-BIG_IN address path"

    big_in_n1 = big_in and n == 1
    big_in_nm = big_in and n > 1
    x_sample_elems = c * d * h * w

    # n>1 rebases per sample; the whole sample must fit BIG_IN_NR.
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
        cgp=geom.cgp,
        crs=geom.crs,
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
    )


class Im2colGather:
    """Gather: bind kernel descriptor, then the block's rows."""

    def __init__(self, plan, x, ext):
        self._plan = plan
        self._ext = ext
        self._x = x
        self._tid = self._ch_base = None
        self._nbase = self._base_t = self._base_h = None
        self._rows = None
        # BIG_IN has no kernel-wide descriptor: it rebases per block (n == 1)
        # or per tap's own sample (n > 1), neither of which is known yet.
        self._x_src = (
            None
            if const_expr(plan.big_in)
            else flat_buffer_view(fx.get_iter(x), ext.x_elems, _bytes_of(ext.x_elems))
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
        plan, cfg = self._plan, self._plan.cfg
        kbase_i = fx.Int64(k_base)
        cc_base = ckk_base = None
        if const_expr(plan.scalar_k):
            cc_base = kbase_i % plan.cgp
            if const_expr(plan.groups > 1):
                cc_base = self._ch_base + cc_base
            ckk_base = kbase_i // plan.cgp
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
        plan, ext = self._plan, self._ext
        self._nbase, rem0 = ext.div_dhw.divmod(m_offset)
        ot_base0, rem1 = ext.div_hw_o.divmod(rem0)

        self._base_t = fx.max(
            ot_base0 * fx.Int64(plan.st) - fx.Int64(plan.pt), fx.Int64(0)
        )
        if const_expr(plan.t_aligned):
            oh_base0 = ext.div_wo.div(rem1)
            self._base_h = fx.max(
                oh_base0 * fx.Int64(plan.sh) - fx.Int64(plan.ph), fx.Int64(0)
            )
        else:
            self._base_h = fx.Int64(0)
        base_row = (self._nbase * fx.Int64(ext.d) + self._base_t) * fx.Int64(
            ext.h
        ) + self._base_h
        x_base_elem = base_row * fx.Int64(ext.w) * fx.Int64(plan.c)
        self._x_src = self._rebased(fx.Int64(x_base_elem))

    def _tap_src(self, sample):
        if const_expr(self._plan.big_in_nm):
            return self._rebased(fx.Int64(sample) * fx.Int64(self._ext.x_sample_elems))
        return self._x_src

    def _decode_rows(self, m_offset):
        """Per A vector, the output element its GEMM row is, as (n, ot, oh, ow).

        Held as the input coordinate each of those taps starts from, since the
        filter tap is all that is added per K tile.
        """
        plan, ext, cfg = self._plan, self._ext, self._plan.cfg
        rows = []
        for i in range_constexpr(cfg.ldg_a_count):
            linear = (self._tid + i * cfg.block_threads) * LDG_VEC
            local_m = linear // cfg.tile_k
            local_k = linear % cfg.tile_k
            row = m_offset + local_m
            row_valid = row < fx.Int64(ext.npq)
            if const_expr(plan.temporal_only_fast):
                out_t = ext.div_d.mod(ext.div_hw_o.div(row))
                rows.append((local_k, row, row_valid, out_t))
            else:
                n_idx, rem = ext.div_dhw.divmod(row)
                ot, rem2 = ext.div_hw_o.divmod(rem)
                oh, ow = ext.div_wo.divmod(rem2)
                in_t0 = ot * plan.st - plan.pt
                in_h0 = oh * plan.sh - plan.ph
                in_w0 = ow * plan.sw - plan.pw
                # A tile-rebased descriptor is based on this block's own sample,
                # so the row's sample index becomes an offset from that one.
                n_or_di = (n_idx - self._nbase) if const_expr(plan.big_in_n1) else n_idx
                rows.append((local_k, row_valid, n_or_di, in_t0, in_h0, in_w0))
        return rows

    def _pad_coord(self, v, extent, pad):
        """Tap -> in-bounds coord. zeros: range mask; other modes wrap into [0, extent)."""
        pad_mode = self._plan.pad_mode
        ext_i = fx.Int64(extent)
        if const_expr(pad_mode == "zeros"):
            return v, in_range(v, ext_i)
        u = v + fx.Int64(pad)
        low = u < fx.Int64(pad)  # v < 0
        high = u >= fx.Int64(pad) + ext_i  # v >= extent
        mid = u - fx.Int64(pad)  # v, where in range
        if const_expr(pad_mode == "replicate"):
            r = high.select(ext_i - fx.Int64(1), mid)
            r = low.select(fx.Int64(0), r)
        elif const_expr(pad_mode == "reflect"):
            # [a b c d e] pad 2 -> [c b a b c d e d c]: -v near, 2*(extent-1) - v far.
            r = high.select(
                (ext_i - fx.Int64(1)) * fx.Int64(2) + fx.Int64(pad) - u, mid
            )
            r = low.select(fx.Int64(pad) - u, r)
        else:  # circular: v + extent near, v - extent far
            r = high.select(u - fx.Int64(pad) - ext_i, mid)
            r = low.select(u + ext_i - fx.Int64(pad), r)
        return r, None

    def _tap_addr(self, i, kbase_i, cc_base, ckk_base):
        """Vector ``i`` of this K tile: (element offset, valid, sample)."""
        plan, ext = self._plan, self._ext
        dec = self._rows[i]
        local_k = dec[0]
        k_abs = kbase_i + fx.Int64(local_k)
        if const_expr(plan.scalar_k):
            cc = cc_base + fx.Int64(local_k)  # cc_base already carries ch_base
        else:
            cc = k_abs % plan.cgp
            if const_expr(plan.groups > 1):
                cc = self._ch_base + cc
        k_valid = k_abs < fx.Int64(plan.crs)

        if const_expr(plan.temporal_only_fast):
            _, row, row_valid, out_t = dec
            kt_i = ckk_base if const_expr(plan.scalar_k) else k_abs // plan.cgp
            temporal_delta = dil(kt_i, plan.dt) - plan.pt
            in_t, m_t = self._pad_coord(out_t + temporal_delta, ext.d, plan.pt)
            valid = gather_valid(row_valid & k_valid, m_t)

            delta = (
                temporal_delta
                if const_expr(plan.pad_mode == "zeros")
                else (in_t - out_t)
            )
            if const_expr(plan.big_in_n1):
                g_off = (
                    (row + delta * ext.hw_o)
                    - (fx.Int64(self._nbase) * ext.dhw + self._base_t * ext.hw_o)
                ) * plan.c + cc
            else:
                g_off = (row + delta * ext.hw_o) * plan.c + cc
            return fx.Int32(g_off), valid, None

        ckk = ckk_base if const_expr(plan.scalar_k) else k_abs // plan.cgp
        kw_i = ckk % plan.kw
        ckk2 = ckk // plan.kw
        kh_i = ckk2 % plan.kh
        kt_i = ckk2 // plan.kh
        _, row_valid, n_or_di, in_t0, in_h0, in_w0 = dec
        in_t, m_t = self._pad_coord(in_t0 + dil(kt_i, plan.dt), ext.d, plan.pt)
        in_h, m_h = self._pad_coord(in_h0 + dil(kh_i, plan.dh), ext.h, plan.ph)
        in_w, m_w = self._pad_coord(in_w0 + dil(kw_i, plan.dw), ext.w, plan.pw)
        valid = gather_valid(row_valid & k_valid, m_t, m_h, m_w)
        if const_expr(plan.big_in_n1):
            row_off = (
                (n_or_di * fx.Int64(ext.d) + (in_t - self._base_t)) * fx.Int64(ext.h)
                + (in_h - self._base_h)
            ) * fx.Int64(ext.w) + in_w
            return fx.Int32(row_off * plan.c + cc), valid, None
        if const_expr(plan.big_in_nm):
            g_off = (
                (in_t * fx.Int64(ext.h) + in_h) * fx.Int64(ext.w) + in_w
            ) * plan.c + cc
            return fx.Int32(g_off), valid, n_or_di
        g_off = (
            (n_or_di * fx.Int64(ext.d) + in_t) * fx.Int64(ext.h) + in_h
        ) * fx.Int64(ext.w) + in_w
        g_off = g_off * plan.c + cc
        return fx.Int32(g_off), valid, None
