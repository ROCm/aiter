# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""16-bit (bf16/fp16) Q/K/V staging managers for the gfx1250 MHA kernels.

Each manager owns the ``global -> LDS (async) -> VGPR (WMMA fragment)`` path for one
16-bit-element operand (Q, K or V): the LDS layout, the global->LDS copy schedule and
the fragment read. They are self-contained — a caller passes only the configuration it
already maintains (hdim, gqa_ratio, kv block width, wave count) through the constructor
plus the runtime ``warp_idx``/``lane_idx``. Nothing here reads a tiling constant from
the kernel; the facts intrinsic to the managed layout live below as private constants.

The ``16b`` suffix names the element width: every swizzle here assumes a ``b128 == 8``
element chunk. An 8-bit (fp8) variant would need its own manager family.

Contents: ``{Q,K,V,O}Manager16bV{1,2}`` plus ``OManager16bV3`` — V1 stages through
``cluster_load_async_to_lds_b128``, V2 through TDM (hardware OOB), V3 writes O back with
``global_store_async_from_lds_b128``. The K/V pairs are one class each plus a transport
mixin: V2 *inherits* V1 and overrides only ``load_descriptor`` (see the K/V staging
section). Q and O keep independent V1/V2. Each class's docstring has the details.

Target: gfx1250 (MI400 / mi450), wave32, 8 waves per threadgroup (256 threads).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import rocdl
from flydsl.expr.rocdl import tdm_ops

from aiter.ops.flydsl.kernels import buffer_ops

from ..kernels_common import create_llvm_ptr
from ..tensor_shim import _to_raw as _ir

# ============================================================================
# Manager-intrinsic tiling constants: fixed by the WMMA instruction and the swizzles
# implemented here, so NOT parameters. What the caller genuinely chooses (hdim,
# gqa_ratio, kv block width, wave count) arrives through a constructor argument.
# ============================================================================

# v_wmma_f32_16x16x32_bf16/f16 shape.
_WMMA_M = 16
_WMMA_K = 32
_BF16_BYTES = 2
_CHUNK_ELEMS = 8  # b128 = 8 bf16
_CHUNK_BYTES = _CHUNK_ELEMS * _BF16_BYTES  # 16

# gfx1250 is wave32; every swizzle/reshape here assumes 32 lanes per wave. FlyDSL
# exposes the wave size only compiler-side (GPUTarget.warp_size), not as a trace-time
# Python int, hence the named constant.
_WAVE_LANES = 32

# Default 8-wave ("m16x8") threadgroup; override via the ``num_waves`` ctor arg.
_DEFAULT_NUM_WAVES = 8

# KV sequence block choices (columns of one QK GEMM tile).
_N_BLOCK_CHOICES = (32, 64, 128, 256)
_DEFAULT_N_BLOCK = 64

# O epilogue software pipeline (OManager16b, see its docstring). 64-col transpose
# units keep each coalesced global store a full 128B row; _O_INFLIGHT_UNITS units
# stay resident in an LDS ring and overlap.
_O_COLS_PER_TILE = 64
_O_INFLIGHT_UNITS = 2
_O_DSCNT_MAX = 63  # s_wait_dscnt SIMM16[5:0]

# LDS budget the O ring fits inside (the caller's non-current K|V slot):
# 8 waves * 2 units * 2KB = 32KB.
_O_LDS_BUDGET_BYTES = 32 * 1024

# Global->LDS async write tile (the V1 transport): 8(kv) x 32(hdim) per warp call,
# one b128 per lane -- lane ``l`` writes row ``l // 4``, 8-element chunk ``l % 4``. K and
# V share it; so does the padded LDS layout they land in (see the K/V staging section).
_WR_TILE_KV = 8
_WR_TILE_HD = _WMMA_K

# O staging (OManager16b): no padding -- an XOR swizzle on the 8-bf16 chunk index makes
# both the b128 store and b128 read bank-conflict-free. MI400 LDS = 64 banks x 4 B, so a
# 16 B chunk spans one 4-bank group and bank_group(q, slot) = (q*G + slot) % 16 with
# G = v_hdim/8 chunks per row. The store touches a fixed chunk-column across all 16
# q-rows, so the swizzle must spread the slot over all 16 q; the read touches one full
# row, so any within-row bijection is already conflict-free. ``slot = chunk ^ (q >>
# shift)`` with shift = max(0, 4 - log2(G)) satisfies both -- the shift folds the low q
# bits the ``q*G`` term already carries when G < 16. Verified for v_hdim in {64,128,256}.


def _assert_multiple(name, val, mult):
    if isinstance(val, int):
        assert val % mult == 0, f"{name} must be a multiple of {mult}; got {val}"


def _as_bases(ptr_lds):
    """Normalize ``ptr_lds`` to a list of LDS sub-buffer bases; a bare fx.Int32 means one
    unsplit buffer. The CALLER places the sub-buffers, so the split count is just
    ``len()``, known at trace time."""
    return list(ptr_lds) if isinstance(ptr_lds, (list, tuple)) else [ptr_lds]


# ---- gfx1250 Expert Scheduling Mode 2 --------------------------------------
# DEP_MODE=2 turns the HW VA_VDST/VM_VSRC issue interlocks OFF. The kernel enables it
# with the `amdgpu-expert-scheduling-mode` LLVM hint at jit time (see _ensure_*_kernel);
# LLVM then emits the setreg AND inserts every dependency cover itself (post-RA depctr
# waits) for the plain intrinsic memory ops below -- the SSA-visible RAW/WAR hazards and
# the LDS RAW between the async global->LDS load and the ds_load that reads it back. So
# this file emits nothing but ordinary flydsl intrinsics in BOTH modes, and mode 2 is
# codegen-identical to mode 0 plus the one setreg.
#
# That is also why every memory op here is a plain intrinsic (``create_llvm_ptr`` +
# ``llvm_dialect.load``/``store``, ``rocdl.ds_load_tr16_b128``,
# ``buffer_ops.buffer_store``) rather than an opaque inline-asm block with hand-written
# covers. An opaque ds_load hides the RAW against the async global->LDS store -- no SSA
# edge, LDS unmodeled -- and LLVM mis-orders it under DEP_MODE=2, producing silent NaN
# at scale. See memory fmha-flydsl-0-3-x-migration / fmha-m16x8-sched-mode2-unsafe.
#
# The kernel module imports this flag and flips the hint in lockstep; it no longer
# changes codegen in this file. False -> mode 0.
ENABLE_SCHED_MODE2 = True


def _async_load_to_lds(gptrs, lds_ptrs, *, cluster, imm_offs=None):
    """Issue a BATCH of async 16B (b128) global->LDS loads. Pure issue, no address math:
    the managers' ``global_load_ptrs`` already built the pointers.

    ``gptrs``/``lds_ptrs`` are equal-length lists of address-space-1 sources and
    address-space-3 destinations (a scalar is treated as a 1-load batch); ``cluster``
    selects the MCAST form (K/V) over plain global (Q).

    ``imm_offs`` is an optional list of per-load compile-time byte immediates, b128
    aligned. The async immediate hits BOTH the global source and the LDS dest by the same
    amount in lockstep (GPU-verified), so a caller wanting a global-only stride must
    pre-subtract it from ``lds_ptrs`` (see ``global_load_ptrs``)."""
    if not isinstance(gptrs, (list, tuple)):
        gptrs = [gptrs]
    if not isinstance(lds_ptrs, (list, tuple)):
        lds_ptrs = [lds_ptrs]
    n = len(gptrs)
    if len(lds_ptrs) != n:
        raise ValueError(f"gptrs/lds_ptrs length mismatch: {n} vs {len(lds_ptrs)}")
    if imm_offs is None:
        imm_offs = [0] * n
    elif len(imm_offs) != n:
        raise ValueError(f"gptrs/imm_offs length mismatch: {n} vs {len(imm_offs)}")

    for gptr, lds_ptr, imm in zip(gptrs, lds_ptrs, imm_offs):
        if cluster:
            # The generic wrapper in FlyDSL 0.3.2 uses an older argument order.
            # The public b128 overload preserves (gptr, lds_ptr, offset, mask).
            mask0 = _ir(fx.Int32(0))
            rocdl.cluster_load_async_to_lds_b128(gptr, lds_ptr, imm, mask0)
        else:
            rocdl.global_load_async_to_lds_b128(gptr, lds_ptr, imm)


# ============================================================================
# K/V staging: row-major PADDED LDS + the two global->LDS transports
#
# Both K/V manager families stage a ``[n_block, hdim]`` tile into plain ROW-MAJOR LDS
# with a per-row pad: element ``(row, col)`` lives at ``row*row_bytes + col*2``, where
# ``row_elems = hdim + pad_elems`` (K pads 8 elems / 16 B, V 16 elems / 32 B) is sized to
# keep the WMMA ``ds_load`` / ``ds_load_tr16`` fetch bank-conflict-free. Everything
# downstream of LDS -- ``ds_load_ptrs``, the fragment plan, the fragment reads -- is
# therefore layout-only and lives ONCE, in the V1 base classes; V2 inherits it untouched.
#
# What the two families actually differ in is the TRANSPORT that fills that LDS:
#
#   V1  ``cluster_load_async_to_lds_b128`` -- per-lane addresses, software OOB clamp,
#       one b128 per lane per 8x32 write tile; bumps ``asynccnt``.
#   V2  TDM (``fx.copy_atom_call`` on a ``make_tdm_atom`` view) -- one descriptor-driven
#       copy per band, no per-lane address VALU (so far fewer address VGPRs), and the
#       per-dim extent gives HARDWARE OOB zero-fill; bumps ``tensorcnt``.
#
# A row-major destination is what makes the two interchangeable at all. The async op's
# compile-time immediate hits source AND destination in lockstep, and here the LDS column
# step is byte-for-byte the global one (``col * 2``), so that immediate simply IS the
# column stride on both sides. (Under V1's former XOR-swizzled layout it had to be
# pre-SUBTRACTED from the LDS pointer to cancel there -- and the swizzle also forced a
# second, round-robin address path whenever a warp spanned more than one row block.
# Adopting the padded layout deleted both.)
#
# Either transport reaches the caller as a ``BufferOpDescriptor``: built PURE (no memory
# op) so the address VALU can be hoisted away from the issue point, then issued via
# ``async_load()`` and fenced against the counters it declares.
# ============================================================================

_K_PAD_ELEMS = 8  # 4 DW = 16 B per K row
_V_PAD_ELEMS = 16  # 8 DW = 32 B per V row
_Q_PAD_ELEMS = 8  # 4 DW = 16 B per Q row (matches K)
_O_PAD_ELEMS = 8  # 4 DW = 16 B per O row (conflict-free ds_store_b128)


class ProducerCtx:
    """Who is issuing a global<->LDS op, and how the producers partition the tile.

    A tile is copied by ``num_producer_warps`` waves; wave ``producer_warp`` owns the
    dense row band ``[w*rows, (w+1)*rows)`` and copies it alone. That partition is the
    CALLER's (it is the same warp specialization the kernel is built on), so it arrives
    here rather than being re-derived per manager. ``lane_idx`` is read only by a
    transport with per-lane addresses (V1); TDM ignores it."""

    def __init__(self, *, producer_warp, num_producer_warps, lane_idx=None):
        self.producer_warp = producer_warp
        self.num_producer_warps = num_producer_warps
        self.lane_idx = lane_idx


class BufferOpDescriptor:
    """Everything needed to issue ONE wave's share of a global<->LDS tile op, and to
    fence it afterwards.

    Building a descriptor is PURE -- it emits no memory op, only address arithmetic --
    which is the whole reason this is an object rather than a call: the caller builds it
    early, lets the address VALU sink into an unrelated load's shadow, and calls
    ``async_load()`` much later. The concrete subclass is the TRANSPORT (``kind``), so a
    caller that just wants "fill this tile" never branches on the loader family. The op is
    named rather than anonymous (``issue()``) to leave room for a store counterpart.

    ``asynccnt`` / ``tensorcnt`` are how many of THIS wave's copies the op will put on
    each hardware counter. Both are always present; a transport that does not touch a
    counter reports 0 for it. Turning those into fence depths is the caller's job -- 0
    here means "none of ours are on this counter", which is NOT the same instruction as
    waiting for that counter to reach 0."""

    kind = None
    asynccnt = 0
    tensorcnt = 0

    def async_load(self):
        raise NotImplementedError


class AsyncCopyDescriptor(BufferOpDescriptor):
    """A batch of ``cluster_load_async_to_lds_b128`` (V1 transport); on ``asynccnt``."""

    kind = "async"

    def __init__(self, gptrs, lds_ptrs, imm_offs, *, cluster=True):
        self.gptrs = gptrs
        self.lds_ptrs = lds_ptrs
        self.imm_offs = imm_offs
        self.cluster = cluster
        self.asynccnt = len(gptrs)
        self.tensorcnt = 0

    def async_load(self):
        _async_load_to_lds(
            self.gptrs, self.lds_ptrs, cluster=self.cluster, imm_offs=self.imm_offs
        )


class TdmCopyDescriptor(BufferOpDescriptor):
    """A list of TDM ``(atom, g_view, lds_view)`` copies (V2 transport); on ``tensorcnt``."""

    kind = "tdm"

    def __init__(self, views):
        self.views = views
        self.asynccnt = 0
        self.tensorcnt = len(views)

    def async_load(self):
        for view in self.views:
            fx.copy_atom_call(*view)


def _warp_band(*, bases, n_block, row_bytes, producer_warp, num_producer_warps):
    """Which rows of the tile this producer wave owns, and where they land in LDS.

    Wave ``producer_warp`` (0..num_producer_warps-1, RUNTIME) copies the dense band
    ``[w*rows, (w+1)*rows)`` by itself, so the waves of one LDS split stay contiguous and
    the destination is dense inside its sub-buffer. Returns ``(lds_base, row0,
    num_rows)``, ``row0`` relative to the tile. Transport-agnostic -- the async and the
    TDM path place their band identically."""
    num_splits = len(bases)
    _assert_multiple("n_block", n_block, num_producer_warps)
    if num_producer_warps % num_splits:
        raise ValueError(
            f"{num_producer_warps} producer waves do not divide over {num_splits} LDS splits"
        )
    num_rows = n_block // num_producer_warps
    warps_per_split = num_producer_warps // num_splits
    base = bases[-1]
    for s in range(num_splits - 2, -1, -1):
        base = (producer_warp < fx.Int32((s + 1) * warps_per_split)).select(
            bases[s], base
        )
    r0 = producer_warp * fx.Int32(num_rows)
    lds_base = base + (producer_warp % fx.Int32(warps_per_split)) * fx.Int32(
        num_rows * row_bytes
    )
    return lds_base, r0, num_rows


def _async_band_descriptor(
    *,
    bases,
    n_block,
    hdim,
    row_bytes,
    ptr_x,
    stride_seq,
    stride_head,
    head,
    row0,
    valid,
    ctx,
):
    """This wave's band of a ``[n_block, hdim]`` tile as ``cluster_load_async_to_lds_b128``
    source/destination pointers. Pure index arithmetic -- no memory op.

    The band is cut into 8(kv) x 32(hdim) write tiles, one b128 per lane (lane ``l`` ->
    row ``l//4``, chunk ``l%4``), but only ONE pointer pair is built per 8-row block, at
    hdim column 0: the ``hdim/32`` columns are walked by the compile-time immediate, which
    the async op applies to source and destination alike -- precisely the column stride
    both want now that LDS is row-major. So a band costs ``num_rows/8`` address
    computations, not one per copy.

    Rows past ``valid`` are clamped to the tile's row 0 on the GLOBAL side only (the LDS
    position stays unclamped): in-bounds garbage, which softmax masks off later. V2 gets
    the same protection from the TDM extent, which zero-fills instead.

    ``row0``/``valid`` describe the whole tile; the band's offset into it is applied here.
    Strides are in ELEMENTS."""
    lds_base, band_r0, num_rows = _warp_band(
        bases=bases,
        n_block=n_block,
        row_bytes=row_bytes,
        producer_warp=ctx.producer_warp,
        num_producer_warps=ctx.num_producer_warps,
    )
    _assert_multiple("producer band rows", num_rows, _WR_TILE_KV)
    _assert_multiple("hdim", hdim, _WR_TILE_HD)
    base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_x)))
    wr_row = ctx.lane_idx // 4  # kv row within the 8-row write tile [0,8)
    chunk = ctx.lane_idx % 4  # which 8-element b128 [0,4) of the 32-wide tile
    gptrs, lds_ptrs, imm_offs = [], [], []
    for r in fx.range_constexpr(num_rows // _WR_TILE_KV):
        band_row = fx.Int32(r * _WR_TILE_KV) + wr_row  # row within this wave's band
        tile_row = band_r0 + band_row  # row within the tile (what ``valid`` bounds)
        safe_row = (tile_row < valid).select(tile_row, fx.Int32(0))  # clamp OOB
        g_base = (
            (row0 + safe_row) * stride_seq + head * stride_head + chunk * _CHUNK_ELEMS
        ) * _BF16_BYTES
        gptr = create_llvm_ptr(base_i64 + fx.Int64(g_base), address_space=1)
        lds_ptr = create_llvm_ptr(
            lds_base + band_row * fx.Int32(row_bytes) + chunk * fx.Int32(_CHUNK_BYTES),
            address_space=3,
        )
        for c in fx.range_constexpr(hdim // _WR_TILE_HD):
            gptrs.append(gptr)  # one source per row block; columns ride the immediate
            lds_ptrs.append(lds_ptr)
            imm_offs.append(c * _WR_TILE_HD * _BF16_BYTES)
    return AsyncCopyDescriptor(gptrs, lds_ptrs, imm_offs, cluster=True)


def _pow2_segments(width):
    """Split ``width`` (elements) into power-of-two column segments, largest first -- the TDM
    pad_interval must be a power of two. 192 -> [(0,128),(128,64)]."""
    segs, c0, rem = [], 0, width
    while rem > 0:
        w = 1 << (rem.bit_length() - 1)  # largest power of two <= rem
        segs.append((c0, w))
        c0 += w
        rem -= w
    return segs


def _tdm_load_views(
    *,
    ptr_x,
    stride_seq,
    stride_head,
    head,
    row0,
    valid,
    num_rows,
    hdim,
    pad_elems,
    lds_base,
    elem_dtype,
    num_warps=_DEFAULT_NUM_WAVES,
):
    """Build a LIST of ``(atom, g_view, lds_view)`` TDM global->LDS copies for one
    ``[num_rows, hdim]`` tile into row-major padded LDS -- PURE (no memory op); issue each with
    ``fx.copy_atom_call(*view)``, then drain with ``tensor_wait(0)``.

    One copy per ``_pow2_segments`` column segment ``(c0, w)``, carrying ``pad_interval=w`` and
    ``pad_amount=(hdim + pad_elems - w)`` so the LDS row still advances by the padded stride.
    The per-row extent ``valid`` is what gives HW OOB zero-fill. ``num_warps`` waves split the
    tile by rows and the lowering takes the share from ``wave_id % num_warps``, so waves 4..7
    issuing a ``num_warps=4`` copy cover the same tile as waves 0..3. Strides in ELEMENTS.
    """
    row_elems = hdim + pad_elems
    off = fx.Int64(row0) * fx.Int64(stride_seq) + fx.Int64(head) * fx.Int64(stride_head)
    base_iter = fx.get_iter(ptr_x)
    lds_ptr_ty = fx.PointerType.get(
        elem_ty=elem_dtype.ir_type,
        address_space=fx.AddressSpace.Shared,
        alignment=16,
    )
    views = []
    for c0, w in _pow2_segments(hdim):
        gbase = fx.add_offset(base_iter, off + fx.Int64(c0))
        g_view = fx.Tensor(fx.make_view(gbase, fx.make_layout((num_rows, w), (w, 1))))
        atom = fx.rocdl.make_tdm_atom(
            g_view,
            [valid, None],
            strides=[stride_seq, None],
            num_warps=num_warps,
            pad_interval=w,
            pad_amount=row_elems - w,
        )
        lds_iter = fx.inttoptr(lds_ptr_ty, lds_base + fx.Int32(c0 * _BF16_BYTES))
        lds_view = fx.Tensor(
            fx.make_view(lds_iter, fx.make_layout((num_rows, w), (row_elems, 1)))
        )
        views.append((atom, g_view, lds_view))
    return views


def _tdm_band_descriptor(
    *,
    bases,
    n_block,
    hdim,
    pad_elems,
    row_bytes,
    ptr_x,
    stride_seq,
    stride_head,
    head,
    row0,
    valid,
    ctx,
    elem_dtype,
):
    """This wave's band of a ``[n_block, hdim]`` tile as TDM copies (one per pow2 hdim
    segment: 1 for 128/256, 2 for 192). The wave copies its band ALONE (``num_warps=1``),
    so it issues one ``tensor_load`` per segment rather than a share of every band's copy.
    Pure, like the async form. ``row0``/``valid`` describe the whole tile."""
    lds_base, band_r0, num_rows = _warp_band(
        bases=bases,
        n_block=n_block,
        row_bytes=row_bytes,
        producer_warp=ctx.producer_warp,
        num_producer_warps=ctx.num_producer_warps,
    )
    return TdmCopyDescriptor(
        _tdm_load_views(
            ptr_x=ptr_x,
            stride_seq=stride_seq,
            stride_head=stride_head,
            head=head,
            row0=row0 + band_r0,
            valid=fx.max(valid - band_r0, fx.Int32(0)),
            num_rows=num_rows,
            hdim=hdim,
            pad_elems=pad_elems,
            lds_base=lds_base,
            elem_dtype=elem_dtype,
            num_warps=1,
        )
    )


class _AsyncTransport:
    """global->LDS by ``cluster_load_async_to_lds_b128`` (the V1 transport).

    Mixed into the K and V BASE managers. It is the only thing those classes hold that is
    not about the LDS layout, which is why the V2 managers can inherit them whole and
    swap just this in."""

    def load_descriptor(
        self,
        *,
        ptr_lds,
        ptr_src,
        stride_seq,
        stride_head,
        head,
        row0,
        valid,
        ctx,
    ):
        """An ``AsyncCopyDescriptor`` for this wave's band of one tile. See
        ``load_descriptor`` on the base managers for the argument contract."""
        return _async_band_descriptor(
            bases=_as_bases(ptr_lds),
            n_block=self.n_block,
            hdim=self.hdim,
            row_bytes=self.row_bytes,
            ptr_x=ptr_src,
            stride_seq=stride_seq,
            stride_head=stride_head,
            head=head,
            row0=row0,
            valid=valid,
            ctx=ctx,
        )


class _TdmTransport:
    """global->LDS by TDM (the V2 transport). Mixed in FRONT of a base manager, so it
    overrides ``load_descriptor`` and inherits the entire LDS read side."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.num_waves != _DEFAULT_NUM_WAVES:
            raise NotImplementedError("V2 TDM loader assumes 8 waves")

    def load_descriptor(
        self,
        *,
        ptr_lds,
        ptr_src,
        stride_seq,
        stride_head,
        head,
        row0,
        valid,
        ctx,
    ):
        """A ``TdmCopyDescriptor`` for this wave's band of one tile. See
        ``load_descriptor`` on the base managers for the argument contract."""
        return _tdm_band_descriptor(
            bases=_as_bases(ptr_lds),
            n_block=self.n_block,
            hdim=self.hdim,
            pad_elems=self.pad_elems,
            row_bytes=self.row_bytes,
            ptr_x=ptr_src,
            stride_seq=stride_seq,
            stride_head=stride_head,
            head=head,
            row0=row0,
            valid=valid,
            ctx=ctx,
            elem_dtype=self.elem_dtype,
        )


# ============================================================================
# Q loader (global -> LDS async -> VGPR WMMA fragments)
# ============================================================================


class QManager16bV1:
    """Owns everything about Q: its LDS footprint and the global->LDS->VGPR load.

    The compute core only asks how much LDS to reserve (``get_lds_size_in_byte``) and
    hands back the raw allocation base; the per-warp sub-offset and the swizzled staging
    layout stay the manager's business.

    Staging layout (per warp, tile-major): each of ``qk_hdim // _WMMA_K`` K-tiles is 16
    rows x _WMMA_K cols bf16 (1024 B), cut into 4x4 subtiles of 4 rows x 8 cols whose
    b128 chunk index is XOR-swizzled with the subtile row to spread LDS banks. Waves
    stack their tiles contiguously.
    """

    def __init__(
        self,
        *,
        qk_hdim,
        gqa_ratio,
        num_waves=_DEFAULT_NUM_WAVES,
        lds_tiles=None,
        q_tiles_per_wave=1,
        elem_dtype=fx.BFloat16,
    ):
        self.elem_dtype = elem_dtype
        if qk_hdim % _WMMA_K != 0:
            raise ValueError(f"qk_hdim must be a multiple of {_WMMA_K}; got {qk_hdim}")
        self.qk_hdim = qk_hdim  # compile-time
        self.gqa_ratio = gqa_ratio  # compile-time
        self.num_waves = num_waves  # compile-time (threadgroup wave count)
        # Each wave owns q_tiles_per_wave adjacent 16-row Q WMMA tiles (contiguous).
        self.q_tiles_per_wave = q_tiles_per_wave  # compile-time (R)
        self.block_m = _WMMA_M * q_tiles_per_wave * num_waves  # Q rows per threadgroup
        self.k_tiles = qk_hdim // _WMMA_K

        # Ring depth: how many of a wave's Q K-tiles live in LDS at once. Sets both the
        # LDS footprint and the inflight-async budget so they cannot drift. The default
        # (== k_tiles) keeps every async copy in flight; smaller trades that overlap for
        # LDS the K/V managers can use.
        if lds_tiles is None:
            lds_tiles = self.k_tiles
        if not 1 <= lds_tiles <= self.k_tiles:
            raise ValueError(
                f"lds_tiles must be in [1, {self.k_tiles}]; got {lds_tiles}"
            )
        # R>1 shares the async/ds counters across the wave's tiles; the gradual
        # ring drain assumes a single tile-chain, so require fully-resident there.
        if q_tiles_per_wave > 1 and lds_tiles != self.k_tiles:
            raise ValueError(
                "q_tiles_per_wave>1 requires fully-resident Q (lds_tiles==k_tiles)"
            )
        self.lds_tiles = lds_tiles
        # Per-tile LDS span; a wave stacks its R tiles contiguously.
        self._tile_stride = _WMMA_M * self.lds_tiles * _WMMA_K * _BF16_BYTES
        self._warp_stride = q_tiles_per_wave * self._tile_stride

    def get_lds_size_in_byte(self):
        """LDS bytes the caller must reserve for Q staging (all waves)."""
        return self.num_waves * self._warp_stride

    def warp_lds_size_in_byte(self):
        """LDS bytes of ONE wave's private Q region; the caller strides ``ptr_lds_warp`` by it."""
        return self._warp_stride

    def _lds_byte(self, row, col_chunk, tile):
        """Swizzled LDS byte offset (within a warp region) for a 8-col b128 chunk."""
        sw = col_chunk ^ (row // 4)  # 4x4 subtile XOR swizzle
        return (
            tile * (_WMMA_M * _WMMA_K * _BF16_BYTES)
            + row * (_WMMA_K * _BF16_BYTES)
            + sw * _CHUNK_BYTES
        )

    def _async_load_vram_to_lds(self, gptrs, lds_ptrs):
        """gfx1250 async 16B global->LDS copy — accepts a batch (equal-length pointer
        lists, or scalars for one load). Q uses the plain (non-MCAST) global form."""
        _async_load_to_lds(gptrs, lds_ptrs, cluster=False)

    def global_load_ptrs(
        self,
        *,
        ptr_Q,
        lds_q_base,  # fx.Int32: byte base of THIS warp's LDS region
        warp_row0,  # fx.Int32: global Q-row of this warp's row 0
        kv_head,
        q_start,
        q_len,
        stride_q_seq,
        stride_q_head,
        lane_idx,
    ):
        """Pointers for EVERY ``global_load_async_to_lds_b128`` of this warp's Q tile:
        ``(gptrs, lds_ptrs)``, ``k_tiles`` x 2 half-loads long. Pure index arithmetic (no
        memory op) so the caller can hoist ALL address VALU ahead of the load burst. Rows
        with seq >= q_len are clamped in-bounds (masked later in softmax)."""
        q_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_Q)))
        gptrs, lds_ptrs = [], []
        for tile in fx.range_constexpr(self.k_tiles):
            for half in fx.range_constexpr(2):
                row = lane_idx // 4 + half * 8  # row within warp [0,16)
                col_chunk = lane_idx % 4  # 8-col chunk within the 32-col tile [0,4)
                pr = warp_row0 + row
                q_head = kv_head * self.gqa_ratio + pr % self.gqa_ratio
                seq = pr // self.gqa_ratio
                safe_seq = (seq < q_len).select(seq, fx.Int32(0))  # clamp OOB
                token = q_start + safe_seq
                # stride_q_seq/head arrive in ELEMENTS (host convention); convert the
                # whole offset to bytes here (V2 TDM uses the element strides directly).
                g_off = (
                    token * stride_q_seq
                    + q_head * stride_q_head
                    + fx.Int32(tile * _WMMA_K)
                    + col_chunk * _CHUNK_ELEMS
                ) * fx.Int32(_BF16_BYTES)
                gptrs.append(
                    create_llvm_ptr(q_base_i64 + fx.Int64(g_off), address_space=1)
                )
                # LDS slot wraps mod lds_tiles: tile t reuses slot t % lds_tiles.
                slot = tile % self.lds_tiles
                lds_off = lds_q_base + self._lds_byte(row, col_chunk, slot)
                lds_ptrs.append(create_llvm_ptr(lds_off, address_space=3))
        return gptrs, lds_ptrs

    def ds_load_ptrs(self, *, lds_q_base, lane_idx):
        """LDS pointers for EVERY ``ds_load_b128`` read of this warp's Q tile: a flat list
        of ``k_tiles`` x 2, where read ``2t`` is tile ``t``'s low 8-col half and ``2t+1``
        its high half. The pair shuffles into the 16x32 WMMA A fragment. Pure index
        math."""
        row = lane_idx % _WMMA_M
        klane = lane_idx // _WMMA_M  # 0 or 1
        ptrs = []
        for tile in fx.range_constexpr(self.k_tiles):
            slot = tile % self.lds_tiles  # match global_load_ptrs ring slot
            lo = lds_q_base + self._lds_byte(row, klane, slot)
            hi = lds_q_base + self._lds_byte(row, klane + 2, slot)
            ptrs.append(create_llvm_ptr(lo, address_space=3))
            ptrs.append(create_llvm_ptr(hi, address_space=3))
        return ptrs

    def load_q_to_vgpr_part1(
        self,
        *,
        ptr_Q,
        stride_q_seq,
        stride_q_head,
        q_start,
        q_len,
        kv_head,
        block_x,  # fx.Int32: this workgroup's grid-x tile index
        warp_idx,
        lane_idx,
        ptr_lds_warp,  # fx.Int32: byte base of THIS warp's Q region (caller-placed)
    ):
        """Part 1 of the Q load: compute all global/LDS offsets (stashed as members, one
        list per of the wave's R q-tiles) and issue the prime async loads, qt-major then
        tile-major so part2's asynccnt countdown matches completion order. The trailing
        ``sched_barrier`` pins the loads above the caller's SALU so that work fills their
        shadow; call ``load_q_to_vgpr_part2`` after to drain and read."""
        R = self.q_tiles_per_wave
        warp_row0_wave = block_x * self.block_m + warp_idx * (R * _WMMA_M)

        self._q_gptrs = []
        self._q_lds_wr_ptrs = []
        self._q_ds_ptrs = []
        for qt in fx.range_constexpr(R):
            lds_q_base = ptr_lds_warp + qt * self._tile_stride
            warp_row0 = warp_row0_wave + qt * _WMMA_M
            # All address VALU up front (2 async b128 + 2 ds_load per tile).
            gptrs, lds_wr_ptrs = self.global_load_ptrs(
                ptr_Q=ptr_Q,
                lds_q_base=lds_q_base,
                warp_row0=warp_row0,
                kv_head=kv_head,
                q_start=q_start,
                q_len=q_len,
                stride_q_seq=stride_q_seq,
                stride_q_head=stride_q_head,
                lane_idx=lane_idx,
            )
            ds_ptrs = self.ds_load_ptrs(lds_q_base=lds_q_base, lane_idx=lane_idx)
            self._q_gptrs.append(gptrs)
            self._q_lds_wr_ptrs.append(lds_wr_ptrs)
            self._q_ds_ptrs.append(ds_ptrs)

        # Prime: issue the first lds_tiles tiles (2 loads each) of every q-tile.
        num_prime = 2 * self.lds_tiles
        for qt in fx.range_constexpr(R):
            self._async_load_vram_to_lds(
                self._q_gptrs[qt][:num_prime], self._q_lds_wr_ptrs[qt][:num_prime]
            )
        rocdl.sched_barrier(0)  # pin the prime loads above the caller's SALU

    def load_q_to_vgpr_part2(self, *, scale):
        """Part 2 of the Q load: drain part 1's async loads and read the tiles into WMMA A
        fragments, ``scale`` folded in (None leaves Q raw). The leading ``sched_barrier``
        keeps the waits and reads below the caller's SALU, inside the load shadow.

        Returns one list of ``k_tiles`` v16-bf16 A fragments per q-tile."""
        rocdl.sched_barrier(0)  # keep waits/reads below the caller's SALU
        R = self.q_tiles_per_wave
        k_tiles = self.k_tiles
        lds_tiles = self.lds_tiles
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, self.elem_dtype)
        scale_bf16 = None if scale is None else scale.to(self.elem_dtype)

        def _scaled(v):  # scale=None: the caller scales S after the QK gemm instead
            return v if scale_bf16 is None else v * scale_bf16

        def _read_tile(ds_ptrs, tile):
            lo = fx.Vector(llvm_dialect.load(v8_ty, ds_ptrs[2 * tile]))
            hi = fx.Vector(llvm_dialect.load(v8_ty, ds_ptrs[2 * tile + 1]))
            return lo, hi

        if lds_tiles == k_tiles:
            # Fully-resident (the only R>1 mode). Async loads complete in issue order, so
            # one GLOBAL asynccnt countdown reads tile (qt,tile) as soon as its 2 loads
            # land. Reduces to the single-chain drain when R==1.
            total = 2 * R * k_tiles
            q_frags_list = [[] for _ in range(R)]
            for qt in fx.range_constexpr(R):
                ds_ptrs = self._q_ds_ptrs[qt]
                for tile in fx.range_constexpr(k_tiles):
                    landed = 2 * (qt * k_tiles + tile) + 2  # this tile's lo+hi
                    rocdl.s_wait_asynccnt(total - landed)
                    lo, hi = _read_tile(ds_ptrs, tile)
                    rocdl.s_wait_dscnt(1)  # lo landed (in-order LDS return)
                    lo = _scaled(lo)
                    rocdl.s_wait_dscnt(0)  # hi landed
                    hi = _scaled(hi)
                    q_frags_list[qt].append(lo.shuffle(hi, list(range(16))))
            return q_frags_list

        # Non-fully-resident ring drain — R==1 only (guarded in __init__).
        ds_ptrs = self._q_ds_ptrs[0]
        gptrs = self._q_gptrs[0]
        lds_wr_ptrs = self._q_lds_wr_ptrs[0]

        def _refill(tile):
            lo = 2 * tile
            self._async_load_vram_to_lds(gptrs[lo : lo + 2], lds_wr_ptrs[lo : lo + 2])

        q_frags = []
        # Steady loop: read+evict tile (i-lds_tiles), refill tile i into its slot.
        for i in fx.range_constexpr(lds_tiles, k_tiles):
            rocdl.s_wait_asynccnt((lds_tiles - 1) * 2)  # oldest tile's 2 loads landed
            lo, hi = _read_tile(ds_ptrs, i - lds_tiles)
            rocdl.s_wait_dscnt(0)  # slot free to overwrite
            _refill(i)
            q_frags.append(_scaled(lo.shuffle(hi, list(range(16)))))
        # Drain loop: read the last lds_tiles tiles, no refill; overlap lo scale w/ hi load.
        for i in fx.range_constexpr(0, lds_tiles):
            rocdl.s_wait_asynccnt((lds_tiles - 1 - i) * 2)
            lo, hi = _read_tile(ds_ptrs, k_tiles - lds_tiles + i)
            rocdl.s_wait_dscnt(1)  # lo landed (in-order LDS return)
            lo = _scaled(lo)
            rocdl.s_wait_dscnt(0)  # hi landed
            hi = _scaled(hi)
            q_frags.append(lo.shuffle(hi, list(range(16))))
        return [q_frags]


# ============================================================================
# K loader (global -> LDS -> VGPR WMMA B-fragments)
# ============================================================================


class KManager16bV1(_AsyncTransport):
    """Owns K's LDS staging and the global->LDS->VGPR B-fragment load.

    Unlike QManager16b there is no ring buffer here: the manager reports the size of ONE
    ``n_block x qk_hdim`` K block and is not bound to a buffer -- the caller reserves as
    many ping-pong buffers as it wants, splits each as it wants, and passes the chosen
    bases (``ptr_lds``) into every method. The block is shared by all waves.

    LDS is row-major with a padded row stride (``qk_hdim + _K_PAD_ELEMS`` elements). The
    pad keeps the ``ds_load_b128`` fetch bank-conflict-free without a swizzle, which also
    makes the LDS column step equal the global one -- see the K/V staging section for why
    that is what lets the two transports share this class.

    The read API pulls 16x16 tiles; a ``col_idx``/``col_idx+16`` pair combines into one
    16x32 WMMA B operand. ``KManager16bV2`` inherits all of it and swaps in TDM.
    """

    def __init__(
        self,
        *,
        qk_hdim,
        n_block=_DEFAULT_N_BLOCK,
        num_waves=_DEFAULT_NUM_WAVES,
        elem_dtype=fx.BFloat16,
    ):
        self.elem_dtype = elem_dtype
        if qk_hdim % _WMMA_K != 0:
            raise ValueError(f"qk_hdim must be a multiple of {_WMMA_K}; got {qk_hdim}")
        if n_block not in _N_BLOCK_CHOICES:
            raise ValueError(
                f"n_block must be one of {_N_BLOCK_CHOICES}; got {n_block}"
            )
        self.qk_hdim = qk_hdim  # compile-time
        self.n_block = n_block  # compile-time
        self.num_waves = num_waves  # compile-time
        self.pad_elems = _K_PAD_ELEMS
        self.row_elems = qk_hdim + _K_PAD_ELEMS  # padded LDS row stride (elems)
        self.row_bytes = self.row_elems * _BF16_BYTES

    @property
    def hdim(self):
        """Transport-facing name for the managed operand's row width (K: ``qk_hdim``)."""
        return self.qk_hdim

    def get_lds_size_in_byte(self):
        """LDS bytes for one n_block x qk_hdim K block (one ping-pong buffer)."""
        return self.n_block * self.row_bytes

    # ------------------------------------------------------------------
    # ``load_descriptor(*, ptr_lds, ptr_src, stride_seq, stride_head, head, row0, valid,
    # ctx)`` comes from the transport mixin. Contract, identical for both transports:
    #
    #   ptr_lds     the CALLER-PLACED LDS sub-buffer bases of the target block (a list;
    #               a bare base is accepted as a 1-way split)
    #   ptr_src     the K tensor; ``stride_seq``/``stride_head`` its ELEMENT strides
    #   head        kv head index
    #   row0/valid  global token of the tile's row 0, and how many of its rows are real
    #   ctx         ``ProducerCtx`` -- which producer wave this is, of how many
    #
    # Returns a PURE ``BufferOpDescriptor`` that declares which counter its copies land on
    # (``asynccnt``/``tensorcnt``), so one fence can serve a mixed pair.
    # ------------------------------------------------------------------
    def ds_load_ptrs(self, *, ptr_lds, lane_idx):
        """One per-lane ds_load base pointer per LDS split; lane ``l`` fetches at row
        ``l%16``, d-byte ``(l//16)*16``. Every fragment ``(kv, dt, half)`` is its split's
        base plus a lane-independent compile-time immediate."""
        return [
            create_llvm_ptr(
                base
                + (lane_idx % _WMMA_M) * fx.Int32(self.row_bytes)
                + (lane_idx // _WMMA_M) * fx.Int32(_CHUNK_ELEMS * _BF16_BYTES),
                address_space=3,
            )
            for base in _as_bases(ptr_lds)
        ]

    def _rows_per_split(self, num_splits):
        """kv rows per LDS sub-buffer; each must hold whole 16-row WMMA tiles."""
        _assert_multiple("n_block", self.n_block, num_splits * _WMMA_M)
        return self.n_block // num_splits

    def num_ds_loads(self):
        return (self.n_block // _WMMA_M) * (self.qk_hdim // _WMMA_K) * 2

    def _ds_load_plan(self, num_splits, lds_imm_offset=0):
        """``[(base_idx, imm)]`` in ``_pv_qk_gemm``'s K order ``[(kv, dt, half)...]``, each
        immediate relative to the ``ds_load_ptrs`` base holding kv-tile ``kv``. Pure Python,
        so a ring driver can emit load ``j`` on demand."""
        NKV = self.n_block // _WMMA_M
        NDT = self.qk_hdim // _WMMA_K
        kv_per_split = self._rows_per_split(num_splits) // _WMMA_M
        plan = []
        for kv in range(NKV):
            split, kv_local = divmod(kv, kv_per_split)
            for dt in range(NDT):
                for half in range(2):
                    imm = (
                        kv_local * _WMMA_M * self.row_bytes
                        + (dt * _WMMA_K + half * _WMMA_M) * _BF16_BYTES
                        + lds_imm_offset
                    )
                    plan.append((split, imm))
        return plan

    def load_one_to_reg(self, base_ptrs, j, lds_imm_offset=0):
        """Emit the ``j``-th K ``ds_load_b128`` of the resident block."""
        v8_ty = fx.Vector.make_type(8, self.elem_dtype)
        base_idx, imm = self._ds_load_plan(len(base_ptrs), lds_imm_offset)[j]
        p = base_ptrs[base_idx]
        if imm:
            p = buffer_ops.get_element_ptr(p, static_byte_offset=imm)
        return fx.Vector(llvm_dialect.load(v8_ty, p))

    def load_all_to_reg(self, base_ptrs, lds_imm_offset=0):
        """Burst all K ``ds_load_b128`` for the resident block, in ``_pv_qk_gemm``'s K
        order."""
        return [
            self.load_one_to_reg(base_ptrs, j, lds_imm_offset)
            for j in range(self.num_ds_loads())
        ]


# ============================================================================
# V loader (global -> LDS -> VGPR WMMA A-fragments via transpose load)
# ============================================================================


class VManager16bV1(_AsyncTransport):
    """Owns V's LDS staging and the global->LDS->VGPR **transpose** load.

    PV computes O^T = V^T @ P^T, so V is the WMMA A-operand ``V^T[d, kv]``. V is stored
    ``[kv, d]`` but PV contracts over kv, so the LDS->VGPR read is ``ds_load_tr16_b128``
    rather than K's natural ``ds_load_b128``. See [[ds-load-tr16-b128-behavior]]: a
    per-lane b128 fetch (where bank conflicts live) followed by a fixed lane-indexed 8x8
    transpose crossbar.

    The layout is K's, only with a WIDER pad (32 B) -- what the transpose fetch needs to
    stay conflict-free where K's 16 B suffices. ``VManager16bV2`` inherits all of it and
    swaps in TDM.
    """

    def __init__(
        self,
        *,
        v_hdim,
        n_block=_DEFAULT_N_BLOCK,
        num_waves=_DEFAULT_NUM_WAVES,
        elem_dtype=fx.BFloat16,
    ):
        self.elem_dtype = elem_dtype
        if v_hdim % _WMMA_M != 0:
            raise ValueError(f"v_hdim must be a multiple of {_WMMA_M}; got {v_hdim}")
        if n_block % _WMMA_K != 0:
            raise ValueError(f"n_block must be a multiple of {_WMMA_K}; got {n_block}")
        self.v_hdim = v_hdim  # compile-time
        self.n_block = n_block  # compile-time
        self.num_waves = num_waves  # compile-time
        self.pad_elems = _V_PAD_ELEMS
        self.row_elems = v_hdim + _V_PAD_ELEMS  # padded LDS row stride (elems)
        self.row_bytes = self.row_elems * _BF16_BYTES

    @property
    def hdim(self):
        """Transport-facing name for the managed operand's row width (V: ``v_hdim``)."""
        return self.v_hdim

    def get_lds_size_in_byte(self):
        """LDS bytes for one ``n_block`` x ``v_hdim`` V block (one ping-pong buffer)."""
        return self.n_block * self.row_bytes

    # ``load_descriptor`` comes from the transport mixin; see ``KManager16bV1`` for the
    # argument contract (``ptr_src`` is the V tensor here).
    def ds_load_ptrs(self, *, ptr_lds, lane_idx):
        """One per-lane transpose-load base pointer per LDS split; the crossbar fetch sits
        at ``kv = (l//16)*8 + l%8``, ``d = ((l//8)%2)*8``. Every fragment ``(dt, kt, half)``
        is its split's base plus a lane-independent compile-time immediate."""
        lane_kv = (lane_idx // _WMMA_M) * fx.Int32(8) + lane_idx % fx.Int32(8)
        lane_d = ((lane_idx // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)
        return [
            create_llvm_ptr(
                base
                + lane_kv * fx.Int32(self.row_bytes)
                + lane_d * fx.Int32(_BF16_BYTES),
                address_space=3,
            )
            for base in _as_bases(ptr_lds)
        ]

    def _rows_per_split(self, num_splits):
        """kv rows per LDS sub-buffer; each must hold whole 32-kv contraction tiles."""
        _assert_multiple("n_block", self.n_block, num_splits * _WMMA_K)
        return self.n_block // num_splits

    def num_ds_loads(self):
        return (self.v_hdim // _WMMA_M) * (self.n_block // _WMMA_K) * 2

    def _ds_load_plan(self, num_splits, lds_imm_offset=0):
        """``[(base_idx, imm)]`` in ``_pv_qk_gemm``'s V order ``[(dt, kt, half)...]``, each
        immediate relative to the ``ds_load_ptrs`` base holding contraction tile ``kt``. Pure
        Python, so a ring driver can emit load ``j`` on demand."""
        d_tiles = self.v_hdim // _WMMA_M
        nkt = self.n_block // _WMMA_K
        kt_per_split = self._rows_per_split(num_splits) // _WMMA_K
        plan = []
        for dt in range(d_tiles):
            for kt in range(nkt):
                split, kt_local = divmod(kt, kt_per_split)
                for half in range(2):
                    imm = (
                        (kt_local * _WMMA_K + half * _WMMA_M) * self.row_bytes
                        + dt * _WMMA_M * _BF16_BYTES
                        + lds_imm_offset
                    )
                    plan.append((split, imm))
        return plan

    def load_one_to_reg(self, base_ptrs, j, lds_imm_offset=0):
        """Emit the ``j``-th V ``ds_load_tr16_b128`` of the resident block."""
        v8_ty = fx.Vector.make_type(8, self.elem_dtype)
        base_idx, imm = self._ds_load_plan(len(base_ptrs), lds_imm_offset)[j]
        p = base_ptrs[base_idx]
        if imm:
            p = buffer_ops.get_element_ptr(p, static_byte_offset=imm)
        return fx.Vector(rocdl.ds_load_tr16_b128(v8_ty, p))

    def load_all_to_reg(self, base_ptrs, lds_imm_offset=0):
        """Burst all V ``ds_load_tr16_b128`` for the resident block, in ``_pv_qk_gemm``'s
        V order."""
        return [
            self.load_one_to_reg(base_ptrs, j, lds_imm_offset)
            for j in range(self.num_ds_loads())
        ]


class QManager16bV2:
    """Q loader (per-warp TDM + row-major padded LDS). No ring buffer: each wave TDM-copies
    ALL of its Q rows in one shot (``num_warps=1``) into its own private, disjoint LDS region,
    so there is no cross-wave sync, then reads them into fragments. Same fragment output as
    ``QManager16bV1`` (scale folded), so the kernel switches V1<->V2 by swapping the class.

    A 3-D ``[num_seq, gqa_ratio, hdim]`` descriptor carries the GQA row-packing (packed row
    ``pr`` -> seq ``pr//gqa``, head ``kv_head*gqa + pr%gqa``), degenerating to
    ``[rows, 1, hdim]`` at gqa==1. LDS row stride matches K's (``hdim + _Q_PAD_ELEMS``).
    """

    _PART1_COUNTERS = ("tensorcnt",)  # part1 issues TDM copies only

    def __init__(
        self,
        *,
        qk_hdim,
        gqa_ratio,
        num_waves=_DEFAULT_NUM_WAVES,
        lds_tiles=None,
        q_tiles_per_wave=1,
        elem_dtype=fx.BFloat16,
    ):
        self.elem_dtype = elem_dtype
        if qk_hdim % _WMMA_K != 0:
            raise ValueError(f"qk_hdim must be a multiple of {_WMMA_K}; got {qk_hdim}")
        if num_waves != _DEFAULT_NUM_WAVES:
            raise NotImplementedError("V2 TDM loader assumes 8 waves")
        self.qk_hdim = qk_hdim  # compile-time
        self.gqa_ratio = gqa_ratio  # compile-time
        self.num_waves = num_waves
        self.q_tiles_per_wave = q_tiles_per_wave
        self.k_tiles = qk_hdim // _WMMA_K
        self.rows_per_warp = _WMMA_M * q_tiles_per_wave  # this wave's Q rows (32)
        self.block_m = self.rows_per_warp * num_waves  # 256
        self.row_elems = qk_hdim + _Q_PAD_ELEMS  # padded LDS row stride (elems)
        self.row_bytes = self.row_elems * _BF16_BYTES
        # lds_tiles is accepted for signature-compat with V1; V2 has no ring.

    def get_lds_size_in_byte(self):
        return self.block_m * self.row_bytes

    def warp_lds_size_in_byte(self):
        """LDS bytes of ONE wave's private Q region; the caller strides ``ptr_lds_warp`` by it."""
        return self.rows_per_warp * self.row_bytes

    def load_q_to_vgpr_part1(
        self,
        *,
        ptr_Q,
        stride_q_seq,
        stride_q_head,
        q_start,
        q_len,
        kv_head,
        block_x,
        warp_idx,
        lane_idx,
        ptr_lds_warp,  # fx.Int32: byte base of THIS warp's Q region (caller-placed)
    ):
        """Issue this wave's per-warp TDM copy of its ``rows_per_warp x qk_hdim`` Q tile into
        the caller-placed private region ``ptr_lds_warp``; strides in ELEMENTS. Drain and read
        in ``load_q_to_vgpr_part2``."""
        gqa = self.gqa_ratio
        # A wave holds rows_per_warp contiguous packed rows (seq outer, head inner): either
        # whole head-groups from head 0 (gqa <= rows_per_warp), or one seq's head-slice at
        # offset packed_row0%gqa (gqa > rows_per_warp). Either way no seq-straddle, which is
        # what the divisibility assert below enforces.
        assert (
            gqa % self.rows_per_warp == 0 or self.rows_per_warp % gqa == 0
        ), f"gqa_ratio={gqa} must divide or be a multiple of rows_per_warp={self.rows_per_warp}"
        num_head = min(gqa, self.rows_per_warp)
        num_seq = self.rows_per_warp // num_head
        packed_row0 = block_x * fx.Int32(self.block_m) + warp_idx * fx.Int32(
            self.rows_per_warp
        )
        seq0 = packed_row0 // fx.Int32(gqa)
        if gqa > self.rows_per_warp:
            head0 = kv_head * fx.Int32(gqa) + packed_row0 % fx.Int32(gqa)
        else:
            head0 = kv_head * fx.Int32(gqa)
        rem = q_len - seq0
        num_seq_valid = fx.max(rem, fx.Int32(0))

        off = fx.Int64(q_start + seq0) * fx.Int64(stride_q_seq) + fx.Int64(
            head0
        ) * fx.Int64(stride_q_head)
        base_iter = fx.get_iter(ptr_Q)
        warp_region = ptr_lds_warp
        lds_ptr_ty = fx.PointerType.get(
            elem_ty=self.elem_dtype.ir_type,
            address_space=fx.AddressSpace.Shared,
            alignment=16,
        )
        # One copy per pow2 column segment, carrying pad_amount=row_elems-w so the padded
        # LDS row stride survives the split (see _tdm_load_views).
        for c0, w in _pow2_segments(self.qk_hdim):
            gbase = fx.add_offset(base_iter, off + fx.Int64(c0))
            g_view = fx.Tensor(
                fx.make_view(
                    gbase, fx.make_layout((num_seq, num_head, w), (num_head * w, w, 1))
                )
            )
            atom = fx.rocdl.make_tdm_atom(
                g_view,
                [num_seq_valid, None, None],
                strides=[stride_q_seq, stride_q_head, None],
                num_warps=1,
                pad_interval=w,
                pad_amount=self.row_elems - w,
            )
            lds_iter = fx.inttoptr(lds_ptr_ty, warp_region + fx.Int32(c0 * _BF16_BYTES))
            lds_view = fx.Tensor(
                fx.make_view(
                    lds_iter,
                    fx.make_layout(
                        (num_seq, num_head, w),
                        (num_head * self.row_elems, self.row_elems, 1),
                    ),
                )
            )
            fx.copy_atom_call(atom, g_view, lds_view)
        self._warp_region = warp_region
        self._lane_idx = lane_idx

    def load_q_to_vgpr_part2(self, *, scale, skip_tensorcnt=-1, skip_asynccnt=-1):
        """Drain this wave's Q TDM and read its tile into WMMA B-fragments, ``scale`` folded
        (None leaves Q raw). Returns one list of ``k_tiles`` fragments per q-tile, like
        ``QManager16bV1``.

        The read collapses to 1 per-lane base + compile-time immediates (like K): lane ``l``
        reads row ``l%16``, d-byte ``(l//16)*16``.

        Each ``skip_*`` is how many copies the caller issued AFTER part1 that may stay in
        flight on that counter: the counters retire in issue order, so waiting down to that
        count drains Q alone. The default -1 means the caller named nothing, and the wait is
        then emitted at 0 for the counters part1 itself uses (``_PART1_COUNTERS``) and
        omitted for the rest."""
        for _cnt, _skip, _wait in (
            ("tensorcnt", skip_tensorcnt, tdm_ops.tensor_wait),
            ("asynccnt", skip_asynccnt, rocdl.s_wait_asynccnt),
        ):
            if _skip >= 0:
                _wait(_skip)
            elif _cnt in self._PART1_COUNTERS:
                _wait(0)
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, self.elem_dtype)
        scale_bf16 = None if scale is None else scale.to(self.elem_dtype)
        lane = self._lane_idx
        lane_base = (
            self._warp_region
            + (lane % _WMMA_M) * fx.Int32(self.row_bytes)
            + (lane // _WMMA_M) * fx.Int32(_CHUNK_ELEMS * _BF16_BYTES)
        )
        base = create_llvm_ptr(lane_base, address_space=3)
        q_frags_list = [[] for _ in range(self.q_tiles_per_wave)]
        for qt in range(self.q_tiles_per_wave):
            for tile in range(self.k_tiles):
                imm_lo = qt * _WMMA_M * self.row_bytes + tile * _WMMA_K * _BF16_BYTES
                imm_hi = imm_lo + _WMMA_M * _BF16_BYTES
                p_lo = (
                    base
                    if imm_lo == 0
                    else buffer_ops.get_element_ptr(base, static_byte_offset=imm_lo)
                )
                p_hi = buffer_ops.get_element_ptr(base, static_byte_offset=imm_hi)
                lo = fx.Vector(llvm_dialect.load(v8_ty, p_lo))
                hi = fx.Vector(llvm_dialect.load(v8_ty, p_hi))
                frag = lo.shuffle(hi, list(range(16)))
                q_frags_list[qt].append(
                    frag if scale_bf16 is None else frag * scale_bf16
                )
        return q_frags_list


class KManager16bV2(_TdmTransport, KManager16bV1):
    """K loader, TDM transport. The LDS layout, the ds_load bases, the fragment plan and
    the B-fragment reads are ``KManager16bV1``'s, inherited unchanged -- the ONLY
    difference is that ``load_descriptor`` returns a ``TdmCopyDescriptor`` (tensorcnt,
    hardware OOB, no per-lane address VALU) instead of an ``AsyncCopyDescriptor``."""


class VManager16bV2(_TdmTransport, VManager16bV1):
    """V loader, TDM transport. As ``KManager16bV2``: everything but the global->LDS copy
    is ``VManager16bV1``'s, including the 32 B row pad and the transpose read."""


# ============================================================================
# O writer (VGPR WMMA accumulator -> LDS reshape -> coalesced global store)
# ============================================================================


class OManager16bV1:
    """Owns the O epilogue: fp32 WMMA accumulator -> bf16 -> global VRAM.

    PV leaves each wave's 16 x v_hdim tile in the accumulator layout: for d-tile ``k`` lane
    ``l`` holds ``O[q = l%16, d = (l//16)*8 + 16*k + {0..7}]``. Adjacent lanes hold
    different q rows, so O is transposed through per-warp staging LDS -- store
    accumulator-indexed, re-read giving each lane 8 contiguous d of one q, then
    ``buffer_store`` coalesced.

    The warp tile splits into ``num_units = v_hdim / cols_per_tile`` self-contained
    transpose units over an ``inflight_units``-deep ring of 16(q) x cols_per_tile slots,
    XOR-swizzled (see the O-swizzle note at the top of the file) so no padding is needed.
    The units software-pipeline: unit u+1's ds_stores are issued ahead of unit u's re-read
    and buffer_store, hiding the LDS round-trip. ``cols_per_tile=64`` keeps each global
    store a full 128B row. The caller points ``ptr_lds`` at the non-current K|V slot, which
    holds a dead prefetch no wave reads, so no threadgroup barrier is needed.

    Ordering uses ``s_wait_dscnt``. DSCNT is a single in-order 6-bit counter shared by
    ds_store and ds_load, so every DS op's global issue index is tracked at compile time
    and a dependency awaited via ``clamp(issued - 1 - gidx, 0, 63)``.
    """

    def __init__(
        self,
        *,
        v_hdim,
        gqa_ratio,
        num_waves=_DEFAULT_NUM_WAVES,
        q_tiles_per_wave=1,
        cols_per_tile=_O_COLS_PER_TILE,
        inflight_units=_O_INFLIGHT_UNITS,
        lds_budget_bytes=_O_LDS_BUDGET_BYTES,
        elem_dtype=fx.BFloat16,
    ):
        self.elem_dtype = elem_dtype
        if v_hdim % _WMMA_M != 0:
            raise ValueError(f"v_hdim must be a multiple of {_WMMA_M}; got {v_hdim}")
        self.v_hdim = v_hdim
        self.gqa_ratio = gqa_ratio
        self.num_waves = num_waves
        # The wave's R tiles serialize through the SAME per-warp ring (caller loops qtile).
        self.q_tiles_per_wave = q_tiles_per_wave
        self.block_m = (
            _WMMA_M * q_tiles_per_wave * num_waves
        )  # Q/O rows per threadgroup
        self.d_tiles = v_hdim // _WMMA_M  # WMMA output tiles == frags/lane

        cpt = min(v_hdim, cols_per_tile)
        cpt = (cpt // _WMMA_M) * _WMMA_M  # 16-col aligned
        if cpt < _WMMA_M:
            raise ValueError(f"cols_per_tile={cols_per_tile} too small")
        if v_hdim % cpt != 0:
            raise NotImplementedError(
                f"v_hdim={v_hdim} not a multiple of cols_per_tile={cpt}"
            )
        self.cols_per_tile = cpt
        self.num_units = v_hdim // cpt
        self.tiles_per_unit = cpt // _WMMA_M  # ds_store ops per unit == ds_load ops
        self.inflight_units = min(inflight_units, self.num_units)

        # LDS geometry, per unit slot; the ring holds inflight_units slots.
        self._row_bytes = cpt * _BF16_BYTES
        self._slot_stride = _WMMA_M * self._row_bytes
        self._warp_stride = self.inflight_units * self._slot_stride
        self._chunks_per_row = cpt // _CHUNK_ELEMS  # G, b128 chunks per slot row
        # XOR swizzle: slot = chunk ^ (q // _sw_div), shift = max(0, 4 - log2(G)).
        shift = max(0, 4 - int(self._chunks_per_row).bit_length() + 1)
        self._sw_div = 1 << shift

        total = num_waves * self._warp_stride
        if total > lds_budget_bytes:
            raise ValueError(
                f"O ring {total}B (waves={num_waves} x inflight={self.inflight_units} x "
                f"slot={self._slot_stride}B) exceeds budget {lds_budget_bytes}B"
            )

    def get_lds_size_in_byte(self):
        """LDS bytes the caller must reserve for O staging (all waves, whole ring)."""
        return self.num_waves * self._warp_stride

    def warp_lds_size_in_byte(self):
        """LDS bytes of ONE wave's private O region; the caller strides ``ptr_lds_warp`` by it."""
        return self._warp_stride

    def _lds_byte(self, slot_idx, q_row, d_col):
        """Swizzled LDS byte offset (within a warp region) of ring-slot ``slot_idx``'s
        O(q_row, d_col) (``d_col`` is slot-local, in [0, cols_per_tile))."""
        chunk = d_col // _CHUNK_ELEMS
        sw = chunk ^ (q_row // self._sw_div)
        return (
            slot_idx * self._slot_stride
            + q_row * fx.Int32(self._row_bytes)
            + sw * _CHUNK_BYTES
        )

    def store_o_to_vram(
        self,
        *,
        ptr_O,  # fx.Pointer to the O tensor (buffer resource built internally)
        o_base_elems,  # fx.Int32: element offset of this (batch, ...) origin (0 for thd)
        stride_o_seq,  # elements per token step
        stride_o_head,  # elements per q-head step
        q_start,  # fx.Int32: first token of this request (varlen) / 0 (batch)
        q_len,  # fx.Int32: valid query rows; rows with seq >= q_len are masked off
        kv_head,  # fx.Int32: this workgroup's kv head
        block_x,  # fx.Int32: this workgroup's grid-x tile index
        warp_idx,
        lane_idx,
        ptr_lds_warp,  # fx.Int32: byte base of THIS warp's O region (caller-placed)
        o_frags,  # list[d_tiles] of v8 f32 (pre-normalized) WMMA accumulators
        qtile=0,  # which of this wave's q_tiles_per_wave tiles this call stores
    ):
        """Reshape this warp's 16 x v_hdim fp32 accumulator to bf16 and store it.

        ``o_frags[k]`` is this lane's v8 fp32 for d-tile ``k``, already normalized (O /
        row-sum). Rows with seq >= q_len are dropped by the buffer_store mask. Strides in
        ELEMENTS.
        """
        if len(o_frags) != self.d_tiles:
            raise ValueError(
                f"expected {self.d_tiles} O frags (v_hdim//{_WMMA_M}); got {len(o_frags)}"
            )
        # The store mask redirects a dropped row to byte 0x7FFFFFFF, so bound the resource
        # to the last valid token and that lands OOB instead of faulting. i64: an i32
        # product can overflow negative, and the descriptor then sign-extends to a huge
        # bound that defeats the drop.
        o_num_records_bytes = (
            fx.Int64(q_start + q_len) * fx.Int64(stride_o_seq) * fx.Int64(_BF16_BYTES)
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            ptr_O, num_records_bytes=_ir(o_num_records_bytes)
        )
        lds_warp = ptr_lds_warp
        q_st = lane_idx % _WMMA_M
        d_half = (lane_idx // _WMMA_M) * _CHUNK_ELEMS  # 0 or 8
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, self.elem_dtype)
        base_row = (
            block_x * self.block_m
            + warp_idx * (self.q_tiles_per_wave * _WMMA_M)
            + qtile * _WMMA_M
        )
        G = self._chunks_per_row
        TPU = self.tiles_per_unit  # ds_store ops per unit == ds_load rounds per unit
        NSL = self.inflight_units  # ring depth (units resident at once)

        # DSCNT is one in-order 6-bit counter for both ds_store and ds_load: op X has
        # retired <=> DSCNT <= issued-1-X. Hence the global issue index bookkeeping below.
        issued = 0  # DS ops emitted so far (global issue index)
        unit_last_store = {}  # unit -> gidx of its final ds_store (all TPU stores done)
        unit_loads = {}  # unit -> list of gidx of its TPU ds_loads

        def emit_write(u):
            """Issue unit ``u``'s TPU ds_stores (accumulator -> ring slot u%NSL)."""
            nonlocal issued
            slot = u % NSL
            prev = u - NSL  # last occupant of this slot
            if prev >= 0:
                # WAR: prev unit's re-reads must retire before we overwrite the slot.
                dep = unit_loads[prev][-1]
                rocdl.s_wait_dscnt(max(0, min(_O_DSCNT_MAX, issued - 1 - dep)))
            last = None
            for kk in range(TPU):
                k = u * TPU + kk
                d_col = d_half + fx.Int32(kk * _WMMA_M)  # slot-local column
                bf = o_frags[k].to(self.elem_dtype)
                addr = lds_warp + self._lds_byte(slot, q_st, d_col)
                lds_ptr = create_llvm_ptr(addr, address_space=3)
                llvm_dialect.store(_ir(bf), lds_ptr, alignment=_CHUNK_BYTES)
                last = issued
                issued += 1
            unit_last_store[u] = last

        def emit_read(u):
            """Re-read unit ``u`` coalesced (ring slot u%NSL) and buffer_store to VRAM."""
            nonlocal issued
            slot = u % NSL
            # RAW: every re-read pulls d-columns spanning all TPU stores of the unit,
            # so wait for the unit's final store before the first load.
            dep = unit_last_store[u]
            rocdl.s_wait_dscnt(max(0, min(_O_DSCNT_MAX, issued - 1 - dep)))
            loaded = []
            gidxs = []
            for r in range(TPU):
                f = fx.Int32(r * _WAVE_LANES) + lane_idx  # 32 b128 chunks per round
                q_out = f // G  # q-row within this warp [0,16)
                d_local = (f % G) * _CHUNK_ELEMS
                addr = lds_warp + self._lds_byte(slot, q_out, d_local)
                lds_ptr = create_llvm_ptr(addr, address_space=3)
                data = fx.Vector(llvm_dialect.load(v8_ty, lds_ptr))
                loaded.append((data, q_out, d_local))
                gidxs.append(issued)
                issued += 1
            unit_loads[u] = gidxs
            d_base = fx.Int32(u * self.cols_per_tile)  # global d origin of this unit
            for r in range(TPU):
                data, q_out, d_local = loaded[r]
                # RAW: this load's data must be in VGPR before storing it to VRAM.
                rocdl.s_wait_dscnt(max(0, min(_O_DSCNT_MAX, issued - 1 - gidxs[r])))
                pr = base_row + q_out  # global packed row
                q_head = kv_head * self.gqa_ratio + pr % self.gqa_ratio
                seq = pr // self.gqa_ratio
                valid = seq < q_len
                token = q_start + seq
                off_elems = (
                    o_base_elems
                    + token * stride_o_seq
                    + q_head * stride_o_head
                    + d_base
                    + d_local
                )
                off_bytes = off_elems * fx.Int32(_BF16_BYTES)
                off_masked = valid.select(off_bytes, fx.Int32(0x7FFFFFFF))
                buffer_ops.buffer_store(
                    data, o_rsrc, off_masked, mask=None, offset_is_bytes=True
                )

        # Two-stage pipeline: prime unit 0, then overlap unit u+1's write with unit
        # u's read. num_units == 1 collapses to a single write+read with one RAW wait.
        emit_write(0)
        for u in range(self.num_units):
            if u + 1 < self.num_units:
                emit_write(u + 1)
            emit_read(u)


class OManager16bV2:
    """O epilogue via TDM store: accumulator (fp32) -> bf16 -> contiguous private LDS -> global
    VRAM (per-warp ``tensor_store_from_lds``). No transpose re-read -- the TDM descriptor does
    the LDS->global reshape, and its extent drops rows ``seq >= q_len`` in hardware.

    The accumulator layout (d-tile ``k``, lane ``l`` -> ``O[q = l%16, d = 16*k + (l//16)*8 +
    {0..7}]``) maps straight to a row-major ``[16, v_hdim]`` LDS tile. The global side is the
    GQA-packed 3-D ``[num_seq, gqa, v_hdim]`` descriptor, degenerating to ``[rows,1,v_hdim]``
    at gqa==1; ``num_warps=1`` keeps each wave in its own LDS region, so no cross-wave sync.

    NOTE: the LDS staging is CONTIGUOUS, no pad. A TDM store IGNORES LDS padding on the
    LDS->memory direction (Shader Programming Guide 4.10.2: "there is no de-padding operation;
    padding is ignored"), so a padded tile would be read misaligned and only row 0 would land.
    The unpadded ``ds_store_b128`` therefore takes a bank conflict, negligible for a one-time
    epilogue."""

    def __init__(
        self,
        *,
        v_hdim,
        gqa_ratio,
        num_waves=_DEFAULT_NUM_WAVES,
        q_tiles_per_wave=1,
        elem_dtype=fx.BFloat16,
    ):
        self.elem_dtype = elem_dtype
        if v_hdim % _WMMA_M != 0:
            raise ValueError(f"v_hdim must be a multiple of {_WMMA_M}; got {v_hdim}")
        if num_waves != _DEFAULT_NUM_WAVES:
            raise NotImplementedError("V2 TDM loader assumes 8 waves")
        self.v_hdim = v_hdim
        self.gqa_ratio = gqa_ratio
        self.num_waves = num_waves
        self.q_tiles_per_wave = q_tiles_per_wave
        self.d_tiles = v_hdim // _WMMA_M
        self.rows_per_warp = _WMMA_M * q_tiles_per_wave  # 32
        self.block_m = self.rows_per_warp * num_waves  # 256
        self.row_elems = v_hdim  # CONTIGUOUS (TDM store ignores pad)
        self.row_bytes = self.row_elems * _BF16_BYTES

    def get_lds_size_in_byte(self):
        return self.num_waves * self.rows_per_warp * self.row_bytes

    def warp_lds_size_in_byte(self):
        """LDS bytes of ONE wave's private O region; the caller strides ``ptr_lds_warp`` by it."""
        return self.rows_per_warp * self.row_bytes

    def store_o_to_vram(
        self,
        *,
        ptr_O,
        o_base_elems,
        stride_o_seq,
        stride_o_head,
        q_start,
        q_len,
        kv_head,
        block_x,
        warp_idx,
        lane_idx,
        ptr_lds_warp,  # fx.Int32: byte base of THIS warp's O region (caller-placed)
        o_frags,
        qtile=0,
    ):
        """Reshape this warp's 16 x v_hdim fp32 accumulator to bf16 and TDM-store it.
        ``o_frags[k]`` is this lane's v8 fp32 for d-tile ``k``, already normalized; rows with
        seq >= q_len are dropped by the TDM extent. Strides in ELEMENTS."""
        if len(o_frags) != self.d_tiles:
            raise ValueError(
                f"expected {self.d_tiles} O frags (v_hdim//{_WMMA_M}); got {len(o_frags)}"
            )
        gqa = self.gqa_ratio
        warp_region = ptr_lds_warp
        tile_lds = warp_region + fx.Int32(qtile * _WMMA_M * self.row_bytes)

        # (1) Accumulator -> row-major LDS, one b128 per d-tile.
        lane_base = (
            tile_lds
            + (lane_idx % _WMMA_M) * fx.Int32(self.row_bytes)
            + (lane_idx // _WMMA_M) * fx.Int32(_CHUNK_ELEMS * _BF16_BYTES)
        )
        base_ptr = create_llvm_ptr(lane_base, address_space=3)
        for k in range(self.d_tiles):
            bf = o_frags[k].to(self.elem_dtype)
            imm = k * _WMMA_M * _BF16_BYTES
            p = (
                base_ptr
                if imm == 0
                else buffer_ops.get_element_ptr(base_ptr, static_byte_offset=imm)
            )
            llvm_dialect.store(_ir(bf), p, alignment=_CHUNK_BYTES)
        rocdl.s_wait_dscnt(0)  # drain b128 stores so the TDM read sees coherent LDS

        # (2) TDM store: private padded LDS tile -> global O (HW OOB drop on the seq axis).
        base_row = (
            block_x * fx.Int32(self.block_m)
            + warp_idx * fx.Int32(self.rows_per_warp)
            + fx.Int32(qtile * _WMMA_M)
        )
        seq0 = base_row // fx.Int32(gqa)
        head0 = kv_head * fx.Int32(gqa)
        rem = q_len - seq0
        num_seq_valid = fx.max(rem, fx.Int32(0))
        lds_ptr_ty = fx.PointerType.get(
            elem_ty=self.elem_dtype.ir_type,
            address_space=fx.AddressSpace.Shared,
            alignment=16,
        )
        lds_iter = fx.inttoptr(lds_ptr_ty, tile_lds)
        if gqa == 1:
            # gqa==1: packed rows == contiguous seqs of one head -> a plain 2-D store.
            off = (
                fx.Int64(o_base_elems)
                + fx.Int64(q_start + base_row) * fx.Int64(stride_o_seq)
                + fx.Int64(kv_head) * fx.Int64(stride_o_head)
            )
            gbase = fx.add_offset(fx.get_iter(ptr_O), off)
            g_view = fx.Tensor(
                fx.make_view(
                    gbase, fx.make_layout((_WMMA_M, self.v_hdim), (self.v_hdim, 1))
                )
            )
            atom = fx.rocdl.make_tdm_atom(
                g_view, [num_seq_valid, None], strides=[stride_o_seq, None], num_warps=1
            )
            lds_view = fx.Tensor(
                fx.make_view(
                    lds_iter,
                    fx.make_layout((_WMMA_M, self.v_hdim), (self.row_elems, 1)),
                )
            )
        else:
            # GQA: packed row pr -> seq pr//gqa, head kv_head*gqa + pr%gqa -> 3-D descriptor.
            num_seq = _WMMA_M // gqa
            off = (
                fx.Int64(o_base_elems)
                + fx.Int64(q_start + seq0) * fx.Int64(stride_o_seq)
                + fx.Int64(head0) * fx.Int64(stride_o_head)
            )
            gbase = fx.add_offset(fx.get_iter(ptr_O), off)
            g_view = fx.Tensor(
                fx.make_view(
                    gbase,
                    fx.make_layout(
                        (num_seq, gqa, self.v_hdim), (gqa * self.v_hdim, self.v_hdim, 1)
                    ),
                )
            )
            atom = fx.rocdl.make_tdm_atom(
                g_view,
                [num_seq_valid, None, None],
                strides=[stride_o_seq, stride_o_head, None],
                num_warps=1,
            )
            lds_view = fx.Tensor(
                fx.make_view(
                    lds_iter,
                    fx.make_layout(
                        (num_seq, gqa, self.v_hdim),
                        (gqa * self.row_elems, self.row_elems, 1),
                    ),
                )
            )
        fx.copy_atom_call(atom, lds_view, g_view)  # src=shared, dst=global => store


class OManager16bV3:
    """O writer: padded LDS ds_store + per-b128 ``global_store_async_from_lds_b128``, no TDM.
    Keeps V2's bank-conflict-free padded ds_store but dodges the TDM store's "padding ignored"
    limitation by addressing each b128 itself, and skips V1's VGPR re-read (the async store goes
    LDS->global directly).

    The accumulator lands in a row-major padded ``[16, v_hdim]`` LDS tile; its 16 x (v_hdim/8)
    b128 chunks then go out over ``num_rounds`` rounds of 32 lanes, round r lane l taking chunk
    ``c = r*32 + l`` -- consecutive lanes, consecutive global, so the store coalesces. Rows with
    seq >= q_len are EXEC-masked off, since an async store has no bounds check.
    """

    def __init__(
        self,
        *,
        v_hdim,
        gqa_ratio,
        num_waves=_DEFAULT_NUM_WAVES,
        q_tiles_per_wave=1,
        elem_dtype=fx.BFloat16,
    ):
        self.elem_dtype = elem_dtype
        if v_hdim % _WMMA_M != 0:
            raise ValueError(f"v_hdim must be a multiple of {_WMMA_M}; got {v_hdim}")
        if num_waves != _DEFAULT_NUM_WAVES:
            raise NotImplementedError("V3 assumes 8 waves")
        self.v_hdim = v_hdim
        self.gqa_ratio = gqa_ratio
        self.num_waves = num_waves
        self.q_tiles_per_wave = q_tiles_per_wave
        self.d_tiles = v_hdim // _WMMA_M
        self.rows_per_warp = _WMMA_M * q_tiles_per_wave
        self.block_m = self.rows_per_warp * num_waves
        self.row_elems = v_hdim + _O_PAD_ELEMS  # PADDED (conflict-free ds_store)
        self.row_bytes = self.row_elems * _BF16_BYTES
        self.chunks_per_row = v_hdim // _CHUNK_ELEMS  # b128 chunks per row
        self.num_rounds = (_WMMA_M * self.chunks_per_row) // _WAVE_LANES
        self._pending = []  # per-qtile (tile_lds, base_row)

    def get_lds_size_in_byte(self):
        return self.num_waves * self.rows_per_warp * self.row_bytes

    def warp_lds_size_in_byte(self):
        """LDS bytes of ONE wave's private O region; the caller strides ``ptr_lds_warp`` by it."""
        return self.rows_per_warp * self.row_bytes

    def store_o_to_vram(
        self,
        *,
        ptr_O,
        o_base_elems,
        stride_o_seq,
        stride_o_head,
        q_start,
        q_len,
        kv_head,
        block_x,
        warp_idx,
        lane_idx,
        ptr_lds_warp,  # fx.Int32: byte base of THIS warp's O region (caller-placed)
        o_frags,
        qtile=0,
    ):
        """Called once per q-tile: convert the accumulator to bf16, compute the LDS write
        pointers, and STASH. The LAST call flushes the whole warp -- ds_stores back-to-back, then
        ALL rows' global addresses together, then one ``s_wait_dscnt(0)`` and a single async
        burst. Computing every address at once keeps them in distinct registers, so no later
        store's address VALU overwrites the source register of one still in flight (splitting
        this per q-tile costs a long WAR halt). The warp's tiles are contiguous in LDS, so they
        flush as one block."""
        if len(o_frags) != self.d_tiles:
            raise ValueError(f"expected {self.d_tiles} O frags; got {len(o_frags)}")
        warp_region = ptr_lds_warp
        tile_lds = warp_region + fx.Int32(qtile * _WMMA_M * self.row_bytes)

        # (1) Per call: cvt fp32->bf16 + compute LDS write pointers (no issue yet). Stash.
        lane_base = (
            tile_lds
            + (lane_idx % _WMMA_M) * fx.Int32(self.row_bytes)
            + (lane_idx // _WMMA_M) * fx.Int32(_CHUNK_ELEMS * _BF16_BYTES)
        )
        base_ptr = create_llvm_ptr(lane_base, address_space=3)
        ds_ops = []
        for k in range(self.d_tiles):
            bf = o_frags[k].to(self.elem_dtype)  # cvt
            imm = k * _WMMA_M * _BF16_BYTES
            p = (
                base_ptr
                if imm == 0
                else buffer_ops.get_element_ptr(base_ptr, static_byte_offset=imm)
            )
            ds_ops.append((bf, p))
        self._pending.append(ds_ops)
        if qtile == 0:
            self._warp_region = warp_region
            self._warp_base = block_x * fx.Int32(self.block_m) + warp_idx * fx.Int32(
                self.rows_per_warp
            )
            self._cfg = {
                "ptr_O": ptr_O,
                "o_base_elems": o_base_elems,
                "stride_o_seq": stride_o_seq,
                "stride_o_head": stride_o_head,
                "q_start": q_start,
                "q_len": q_len,
                "kv_head": kv_head,
                "lane_idx": lane_idx,
            }

        if qtile != self.q_tiles_per_wave - 1:
            return

        # (2) LAST call -- flush the whole warp.
        rocdl.sched_barrier(0)
        for ds_ops in self._pending:
            for bf, p in ds_ops:
                llvm_dialect.store(_ir(bf), p, alignment=_CHUNK_BYTES)
        addrs, valid_rows = self._warp_addrs(
            self._warp_region, self._warp_base, **self._cfg
        )
        rocdl.sched_barrier(0)  # address VALU above, store burst below -- no interleave
        rocdl.s_wait_dscnt(
            0
        )  # all ds_stores landed (every async row reads a full padded row)

        @flyc.jit
        def _burst():
            if valid_rows > fx.Int32(0):
                for gdst, lsrc in addrs:
                    rocdl.global_store_async_from_lds_b128(_ir(gdst), _ir(lsrc), 0)

        _burst()
        # No s_wait_asynccnt: HW drains the async stores' LDS reads at workgroup retire.
        self._pending = []

    def _warp_addrs(
        self,
        warp_region,
        warp_base,
        *,
        ptr_O,
        o_base_elems,
        stride_o_seq,
        stride_o_head,
        q_start,
        q_len,
        kv_head,
        lane_idx,
    ):
        """Pure ALU: (gdst, lsrc) for every round of the warp's block, row-coalesced. OOB rows
        clamp to the warp's last valid row, which makes them redundant idempotent writes (a real
        lane of THIS warp stores the same bytes, and warps own disjoint rows) -- so every store
        issues unconditionally and the burst stays back-to-back. Returns (addrs, valid_rows);
        valid_rows <= 0 means the whole warp is OOB."""
        gqa = self.gqa_ratio
        cpr = self.chunks_per_row
        rpw = self.rows_per_warp
        num_rounds = (rpw * cpr) // _WAVE_LANES
        ptr_O_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_O)))
        # packed rows [warp_base, warp_base+rpw) valid iff pr//gqa < q_len iff pr < q_len*gqa.
        valid_rows = q_len * fx.Int32(gqa) - warp_base
        last_valid = fx.min(
            fx.max(valid_rows - fx.Int32(1), fx.Int32(0)), fx.Int32(rpw - 1)
        )
        addrs = []
        for r in range(num_rounds):
            c = fx.Int32(r * _WAVE_LANES) + lane_idx
            row = c // fx.Int32(cpr)
            d_chunk = c % fx.Int32(cpr)
            srow = fx.min(
                row, last_valid
            )  # OOB rows -> warp's last valid row (idempotent redirect)
            d = d_chunk * fx.Int32(_CHUNK_ELEMS)
            lds_src = (
                warp_region
                + srow * fx.Int32(self.row_bytes)
                + d_chunk * fx.Int32(_CHUNK_BYTES)
            )
            pr = warp_base + srow
            seq_g = pr // fx.Int32(gqa) if gqa > 1 else pr
            head = kv_head * fx.Int32(gqa) + pr % fx.Int32(gqa) if gqa > 1 else kv_head
            token = q_start + seq_g
            off64 = (
                fx.Int64(o_base_elems)
                + fx.Int64(token) * fx.Int64(stride_o_seq)
                + fx.Int64(head) * fx.Int64(stride_o_head)
                + fx.Int64(d)
            )
            gdst = create_llvm_ptr(
                ptr_O_i64 + off64 * fx.Int64(_BF16_BYTES), address_space=1
            )
            lsrc = create_llvm_ptr(lds_src, address_space=3)
            addrs.append((gdst, lsrc))
        return addrs, valid_rows
