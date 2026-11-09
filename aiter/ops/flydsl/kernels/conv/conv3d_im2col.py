# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Im2col gather for implicit-GEMM conv3d: tap addressing, padding, buffer rebasing."""

from typing import NamedTuple

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from .conv3d_gfx950_utils import (
    BF16_BYTES,
    KW_REUSE_STAGES,
    LDG_VEC,
    LDS_BYTES_PER_CU,
    OOB_SENTINEL_BYTES,
    OOB_SENTINEL_ELEM,
    WARP_SIZE,
    TileConfig,
    b_park_elems,
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

    # kW reuse: the KW taps of one (kt, kh, channel block) share a single LDS
    # strip of input pixels, read back at a row shift of kw*dw. The strip
    # holds the block's rows plus a halo per output-row segment; tile_m <= Wo
    # keeps that to two segments. Only the boolean may follow the resolution;
    # the strip sizes come from the filter and tile alone.
    kw_reuse: bool
    strip_halo: int
    strip_rows: int
    strip_ldg: int

    # kH x kW reuse: a block owns an out_block = (rows, cols) patch of output
    # pixels in one (n, t) plane, and all kH*kW taps of one (kt, channel
    # block) read a single LDS patch of patch_rows x patch_cols input pixels.
    # Takes precedence over kW reuse. Sizes come from the filter and tile;
    # out_block is None unless kh_reuse.
    kh_reuse: bool
    out_block: object
    patch_rows: int
    patch_cols: int
    patch_ldg: int
    patch_alloc_vecs: int


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

    scalar_k = geom.cgp % tile_k == 0
    # What both reuse paths need of the layer; AITER_CONV3D_KW_REUSE=0 turns
    # off every LDS-reuse path.
    reuse_ok = bool(
        param.kw_reuse
        and scalar_k
        and pad_mode == "zeros"
        and groups == 1
        and param.splitk <= 1
    )
    kw_layer_ok = reuse_ok and kw > 1 and sw == 1
    strip_halo = (kw - 1) * dw if kw_layer_ok else 0
    strip_rows = tile_m + 2 * strip_halo if kw_layer_ok else 0
    strip_vecs = strip_rows * (tile_k // LDG_VEC)
    strip_ldg = -(-strip_vecs // cfg.block_threads)
    # The KW B tiles per stage can push LDS past what keeps the default path's
    # blocks resident (128x192: 80 KB -> 96 KB, two blocks per CU -> one), and
    # measured on the VAE layers that loses more than the cut L2 traffic buys.
    reuse_lds = _reuse_lds_bytes(cfg, strip_ldg, kw)
    base_lds = _base_lds_bytes(cfg)
    keeps_occupancy = LDS_BYTES_PER_CU // reuse_lds >= LDS_BYTES_PER_CU // base_lds
    kw_reuse = kw_layer_ok and keeps_occupancy and not big_in and wo >= tile_m

    # kH x kW reuse. cols % 16 keeps every MFMA atom's 16 rows inside one
    # image row of the patch. The epilogue masks the H/W tails through the
    # output descriptor, which BIG_OUT does not have.
    rows = param.block_rows
    cols = tile_m // rows if rows > 0 and tile_m % rows == 0 else 0
    kh_layer_ok = (
        reuse_ok
        and rows > 1
        and cols > 0
        and cols % 16 == 0
        and (kh > 1 or kw > 1)
        and sh == 1
        and sw == 1
    )
    patch_rows = rows + (kh - 1) * dh if kh_layer_ok else 0
    patch_cols = cols + (kw - 1) * dw if kh_layer_ok else 0
    patch_vecs = patch_rows * patch_cols * (tile_k // LDG_VEC)
    patch_ldg = -(-patch_vecs // cfg.block_threads)
    # The last round is padded to whole waves; the waves past it park.
    patch_alloc_vecs = -(-patch_vecs // WARP_SIZE) * WARP_SIZE
    big_out = param.k * geom.npq * BF16_BYTES > 0x7FFFFFFF
    # Against the path this layer would otherwise take: kW reuse can already
    # hold two blocks per CU where the default path holds one (256x96).
    other_lds = reuse_lds if kw_reuse else base_lds
    kh_keeps_occupancy = kh_layer_ok and (
        LDS_BYTES_PER_CU // _patch_lds_bytes(cfg, patch_ldg, patch_alloc_vecs)
        >= LDS_BYTES_PER_CU // other_lds
    )
    kh_reuse = kh_layer_ok and kh_keeps_occupancy and not big_in and not big_out
    if kh_reuse:
        kw_reuse = False

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
        scalar_k=scalar_k,
        big_in=big_in,
        big_in_n1=big_in_n1,
        big_in_nm=big_in_nm,
        t_aligned=t_aligned,
        kw_reuse=kw_reuse,
        strip_halo=strip_halo,
        strip_rows=strip_rows,
        strip_ldg=strip_ldg,
        kh_reuse=kh_reuse,
        out_block=(rows, cols) if kh_reuse else None,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        patch_ldg=patch_ldg,
        patch_alloc_vecs=patch_alloc_vecs,
    )


def strip_elems(plan):
    """Elements of one LDS strip stage, padded to whole DMA rounds."""
    return plan.strip_ldg * plan.cfg.block_threads * LDG_VEC


def _base_lds_bytes(cfg):
    return BF16_BYTES * (cfg.lds_a_elems + cfg.lds_b_elems + b_park_elems(cfg))


def _reuse_lds_bytes(cfg, strip_ldg, kw):
    strip = strip_ldg * cfg.block_threads * LDG_VEC
    stage = strip + kw * cfg.tile_n * cfg.tile_k
    return BF16_BYTES * (KW_REUSE_STAGES * stage + b_park_elems(cfg))


def patch_needs_park(cfg, patch_ldg, patch_alloc_vecs):
    """Whether the patch's last DMA round has waves past the padded patch."""
    return patch_ldg * cfg.block_threads > patch_alloc_vecs


def _patch_lds_bytes(cfg, patch_ldg, patch_alloc_vecs):
    park = b_park_elems(cfg) or (
        WARP_SIZE * LDG_VEC if patch_needs_park(cfg, patch_ldg, patch_alloc_vecs) else 0
    )
    patch = KW_REUSE_STAGES * patch_alloc_vecs * LDG_VEC
    return BF16_BYTES * (patch + cfg.lds_b_elems + park)


def plan_lds_bytes(plan):
    """LDS one block allocates under this plan (what make_shared_storage lays out)."""
    if plan.kh_reuse:
        return _patch_lds_bytes(plan.cfg, plan.patch_ldg, plan.patch_alloc_vecs)
    if plan.kw_reuse:
        return _reuse_lds_bytes(plan.cfg, plan.strip_ldg, plan.kw)
    return _base_lds_bytes(plan.cfg)


class StripGather:
    """kW-reuse gather: one LDS strip per (kt, kh, channel block).

    Strip row r holds 32 channels of one input pixel. The block's GEMM rows
    m = 0..tile_m-1 are consecutive output positions; they wrap to the next
    output row at m_w = Wo - ow0 at most once (tile_m <= Wo). Segment 0 fills
    strip rows [0, m_w + halo), segment 1 the rows after, so tap kw of row m
    reads strip row s(m) + kw*dw with s(m) = m + halo * (m >= m_w).
    """

    def __init__(self, plan, x, ext):
        assert plan.kw_reuse
        self._plan = plan
        self._ext = ext
        self._x_src = flat_buffer_view(
            fx.get_iter(x), ext.x_elems, _bytes_of(ext.x_elems)
        )
        self._m_w = None
        self._vecs = None

    def bind_block(self, tid, m_offset):
        plan, ext, cfg = self._plan, self._ext, self._plan.cfg
        halo = plan.strip_halo
        m0 = fx.Int64(m_offset)
        n0, rem = ext.div_dhw.divmod(m0)
        ot0, rem = ext.div_hw_o.divmod(rem)
        oh0, ow0 = ext.div_wo.divmod(rem)
        m_w = fx.Int64(ext.wo) - ow0
        m1 = m0 + m_w
        n1, rem = ext.div_dhw.divmod(m1)
        ot1, rem = ext.div_hw_o.divmod(rem)
        oh1 = ext.div_wo.div(rem)
        npq = fx.Int64(ext.npq)
        exists0 = m0 < npq
        exists1 = m1 < npq
        r1 = m_w + fx.Int64(halo)
        d, h, w = fx.Int64(ext.d), fx.Int64(ext.h), fx.Int64(ext.w)

        vecs_per_row = cfg.tile_k // LDG_VEC
        self._vecs = []
        for i in range_constexpr(plan.strip_ldg):
            v = fx.Int64(tid) + fx.Int64(i * cfg.block_threads)
            r = v // vecs_per_row
            q = v % vecs_per_row
            seg1 = r >= r1
            n = seg1.select(n1, n0)
            ot = seg1.select(ot1, ot0)
            oh = seg1.select(oh1, oh0)
            iw = seg1.select(r - r1, ow0 + r) - fx.Int64(plan.pw)
            it0 = ot * plan.st - plan.pt
            ih0 = oh * plan.sh - plan.ph
            base = (((n * d + it0) * h + ih0) * w + iw) * plan.c + q * LDG_VEC
            ok = (
                seg1.select(exists1, exists0)
                & in_range(iw, w)
                & (r < fx.Int64(plan.strip_rows))
            )
            self._vecs.append((base, ok, it0, ih0))
        self._m_w = m_w

    def taps(self, kt, kh, cc):
        """Yield ``(i, src, voff)`` per strip vector of group (kt, kh, cc)."""
        assert self._vecs is not None, "bind_block() before taps()"
        plan, ext = self._plan, self._ext
        t_off = dil(kt, plan.dt)
        h_off = dil(kh, plan.dh)
        delta = (fx.Int64(t_off) * fx.Int64(ext.h) + fx.Int64(h_off)) * fx.Int64(
            ext.w
        ) * plan.c + cc * plan.cfg.tile_k
        for i in range_constexpr(plan.strip_ldg):
            base, ok, it0, ih0 = self._vecs[i]
            valid = gather_valid(
                ok,
                in_range(it0 + fx.Int64(t_off), ext.d),
                in_range(ih0 + fx.Int64(h_off), ext.h),
            )
            yield i, self._x_src, valid.select(
                fx.Int32(base + delta), fx.Int32(OOB_SENTINEL_ELEM)
            )

    def row_offset(self, m):
        """Strip element offset of GEMM row ``m`` at tap kw = 0."""
        halo = self._plan.strip_halo
        m = fx.Int64(m)
        s = m + (m >= self._m_w).select(fx.Int64(halo), fx.Int64(0))
        return s * self._plan.cfg.tile_k


class PatchGather:
    """kH x kW reuse gather: one LDS patch per (kt, channel block).

    The block owns a ``rows x cols`` patch of output pixels in one (n, t)
    plane (``plan.out_block``). The LDS patch holds ``patch_rows x
    patch_cols`` input pixels (32 channels each), and tap (kh, kw) of output
    (r, c) reads patch pixel (r + kh*dh, c + kw*dw). Unit H/W stride.
    """

    def __init__(self, plan, x, ext):
        assert plan.kh_reuse
        cfg = plan.cfg
        self._plan, self._ext = plan, ext
        self.rows, self.cols = plan.out_block
        self.prow, self.pcol = plan.patch_rows, plan.patch_cols
        self._vpp = cfg.tile_k // LDG_VEC
        self.vecs = self.prow * self.pcol * self._vpp
        self.ldg = plan.patch_ldg
        self.alloc_vecs = plan.patch_alloc_vecs
        self.stage_elems = self.alloc_vecs * LDG_VEC
        self._x_src = flat_buffer_view(
            fx.get_iter(x), ext.x_elems, _bytes_of(ext.x_elems)
        )
        self._vecs = None
        self._tid = None

    def bind_block(self, tid, m0):
        plan, ext, cfg = self._plan, self._ext, self._plan.cfg
        m0 = fx.Int64(m0)
        n0, rem = ext.div_dhw.divmod(m0)
        ot0, rem = ext.div_hw_o.divmod(rem)
        oh0, ow0 = ext.div_wo.divmod(rem)
        d, h, w = fx.Int64(ext.d), fx.Int64(ext.h), fx.Int64(ext.w)
        it0 = ot0 * plan.st - plan.pt
        self._tid = tid
        self._vecs = []
        for i in range_constexpr(self.ldg):
            v = fx.Int64(tid) + fx.Int64(i * cfg.block_threads)
            p = v // self._vpp
            q = v % self._vpp
            pr = p // self.pcol
            pc = p % self.pcol
            ih = oh0 + pr - fx.Int64(plan.ph)
            iw = ow0 + pc - fx.Int64(plan.pw)
            base = (((n0 * d + it0) * h + ih) * w + iw) * plan.c + q * LDG_VEC
            ok = (v < fx.Int64(self.vecs)) & in_range(ih, h) & in_range(iw, w)
            self._vecs.append((base, ok))
        self._it0 = it0

    def taps(self, kt, cc):
        """Yield ``(i, src, voff, live)`` per patch vector of group (kt, cc).

        ``live`` is None, or the wave-uniform predicate of a round that is
        only partly inside the padded patch (the rest of the waves park).
        """
        plan, ext, cfg = self._plan, self._ext, self._plan.cfg
        t_off = dil(kt, plan.dt)
        delta = (
            fx.Int64(t_off) * fx.Int64(ext.h) * fx.Int64(ext.w) * plan.c
            + cc * cfg.tile_k
        )
        t_ok = in_range(self._it0 + fx.Int64(t_off), ext.d)
        for i in range_constexpr(self.ldg):
            base, ok = self._vecs[i]
            voff = (ok & t_ok).select(
                fx.Int32(base + delta), fx.Int32(OOB_SENTINEL_ELEM)
            )
            live = None
            if const_expr((i + 1) * cfg.block_threads > self.alloc_vecs):
                wave_v = (
                    fx.Int32(self._tid) // WARP_SIZE
                ) * WARP_SIZE + i * cfg.block_threads
                live = wave_v < fx.Int32(self.alloc_vecs)
            yield i, self._x_src, voff, live

    def lane_offset(self, m):
        """Patch element offset of block-local output row ``m`` at tap (0, 0)."""
        m = fx.Int64(m)
        return ((m // self.cols) * self.pcol + m % self.cols) * self._plan.cfg.tile_k

    def tap_offset(self, kh, kw):
        plan = self._plan
        return (dil(kh, plan.dh) * self.pcol + dil(kw, plan.dw)) * plan.cfg.tile_k


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
            r = high.select(ext_i - 1, mid)
            r = low.select(fx.Int64(0), r)
        elif const_expr(pad_mode == "reflect"):
            # [a b c d e] pad 2 -> [c b a b c d e d c]: -v near, 2*(extent-1) - v far.
            r = high.select((ext_i - 1) * 2 + fx.Int64(pad) - u, mid)
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
