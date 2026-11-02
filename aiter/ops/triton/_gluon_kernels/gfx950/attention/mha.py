##############################################################################
# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

"""Gluon MHA forward for gfx950 (CDNA4).

One kernel, `_attn_fwd`, covers bf16/fp16 and fp8 (e4m3); causal, sliding-window
and varlen attention; an optional PE slice; and padded head dims.  Top to bottom:

1. Layout helpers (constexpr): the padded LDS layouts K^T / V are staged in, the
   global->LDS async-copy address layouts, `kv_async_copy_ok` (the host calls it
   too), and `_head_slices` (a padded head dim issued as two power-of-two slices).
2. Aggregates:
   - `MhaConfig`: compile-time shape of a launch -- tiles, head slices, dtype,
     layouts.
   - `KVStage`: double-buffered LDS staging of K (+PE) and V, with its copies and
     reads.
   - `KeyMask`: per-row inputs of the element masks (causal, ragged K, window).
   - `ScoresOut`: where `return_scores` writes P.
3. Per-tile math: `_attn_qk` (Q@K^T and masks), `_attn_softmax_pv` /
   `_attn_softmax_pv_fp8` (online softmax and P@V), and the pipeline's pieces
   (`_dot_qk`, `_dot_pv`, `_sc_vec1`, `_sc_vec2`).
4. Two loops over the key blocks of one M-block:
   - `_attn_fwd_inner`: one tile at a time; applies any mask.
   - `_attn_fwd_pipelined`: the rotated four-cluster pipeline; unmasked blocks
     only, at least `_MIN_PIPE_BLOCKS` of them.
5. `_attn_fwd`: per M-block, set up, split the key range (`_classify_k_blocks`)
   into window-edge, full and masked blocks, run the full ones through the
   pipeline when there are enough and the rest through `_attn_fwd_inner`, then
   normalize and store.
6. `_get_config` (host): the tile for a shape, from mha.json.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import remap_xcd
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.common_utils import strip_annotate
from aiter.ops.triton.utils.config_utils import (
    AITER_TRITON_CONFIGS_PATH,
    load_config_json,
)

# ---------------------------------------------------------------------------
# 1. Layout helpers
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def _bits(n):
    """log2 of a power of two."""
    return n.bit_length() - 1


@gluon.constexpr_function
def _contig_strided(rows, cols, contig_dim):
    """(extent along the contiguous axis, extent along the strided axis)."""
    return (rows, cols) if contig_dim == 0 else (cols, rows)


@gluon.constexpr_function
def _kv_lds_params(elem_bits):
    """(padding interval, K^T pad, V pad), in elements, of the K/V staging tiles.

    16-bit: a 1 KB interval, the AMD backend's own padding granularity.  e4m3
    (chosen with the LDS counters): a 1 KB interval too -- at 512 elements the
    async copy's lowering declines the destination -- and a 16-element K^T pad.
    K^T is read with ds_read_b128, so its pad must stay a multiple of 16 bytes.
    """
    if elem_bits == 8:
        return 1024, 16, 32
    return 512, 8, 32


@gluon.constexpr_function
def _padded_staggered_layout(rows, cols, contig_dim, pad, interval):
    """Row-staggered padded shared layout for a K^T or V staging tile.

    Both tiles are written along their contiguous axis and read across it, which
    is conflict free only when consecutive strided-axis entries land far apart.
    The bases put the contiguous axis in the low bits, then the *high* bits of the
    strided index (the stagger), then its low bits.
    """
    c, r = _contig_strided(rows, cols, contig_dim)

    def base(cv, rv):
        return [cv, rv] if contig_dim == 0 else [rv, cv]

    rows_per_interval = max(interval // c, 1)
    split = max(r // rows_per_interval, 1)
    bases = [base(1 << i, 0) for i in range(_bits(c))]
    bases += [base(0, split << i) for i in range(_bits(rows_per_interval))]
    bases += [base(0, 1 << i) for i in range(_bits(split))]
    return gl.PaddedSharedLayout([[interval, pad]], bases, [], [rows, cols])


@gluon.constexpr_function
def _staggered_layout_ok(head_dim_pow2, block_n, interval):
    """Can the staggered layout be built: power-of-two extents, and BLOCK_N (the
    strided axis of both tiles) splitting evenly across the padding intervals."""
    if head_dim_pow2 & (head_dim_pow2 - 1) or block_n & (block_n - 1):
        return False
    if head_dim_pow2 < 16 or block_n < 16:
        return False
    return block_n % max(interval // head_dim_pow2, 1) == 0


@gluon.constexpr_function
def _make_kv_shared_layouts(head_dim_pow2, block_n, elem_bits):
    """(K^T, V) LDS layouts: `[head_dim, BLOCK_N]` and its transpose."""
    interval, k_pad, v_pad = _kv_lds_params(elem_bits)
    return (
        _padded_staggered_layout(head_dim_pow2, block_n, 0, k_pad, interval),
        _padded_staggered_layout(block_n, head_dim_pow2, 1, v_pad, interval),
    )


# Bits per lane the global->LDS copy's lowering accepts, widest first: 128 is a
# `*_dwordx4` per lane, 32 a plain `dword` (4x the instructions, 1/4 the alignment).
_ASYNC_COPY_BITS = (128, 32)


@gluon.constexpr_function
def _async_copy_lane_rows(rows, cols, contig_dim, interval):
    """(first strided-axis bit the copy's lanes take, strided bits left to them).

    The lanes must write consecutive LDS addresses, and the staggered destination
    stores the strided axis from bit `log2(split)` upward.
    """
    c, r = _contig_strided(rows, cols, contig_dim)
    split = max(r // max(interval // c, 1), 1)
    return _bits(split), _bits(r) - _bits(split)


@gluon.constexpr_function
def _async_copy_vec(
    rows,
    cols,
    contig_dim,
    num_warps,
    elem_bits,
    stride_align,
    interval,
    warp_size=64,
):
    """Elements per lane an async copy of this tile can move, or 0 if none can.

    Each lane must carry 128 or 32 bits aligned to their width (`stride_align` is
    the alignment the host verified, in elements), and the LDS writes must
    coalesce against the padded destination.
    """
    c, r = _contig_strided(rows, cols, contig_dim)
    n_threads = num_warps * warp_size
    if (rows * cols) % n_threads != 0:
        return 0
    per_lane = rows * cols // n_threads
    _, lane_rows = _async_copy_lane_rows(rows, cols, contig_dim, interval)
    for bits in _ASYNC_COPY_BITS:
        vec = bits // elem_bits
        if vec < 1 or vec > per_lane or vec > c or vec > stride_align:
            continue
        if (c // vec) > warp_size:
            continue
        r_lane_bits = _bits(warp_size) - _bits(c // vec)
        if r_lane_bits > lane_rows:
            continue
        if _bits(r) < _bits(num_warps) + r_lane_bits:
            continue
        return vec
    return 0


@gluon.constexpr_function
def _async_copy_layout(rows, cols, contig_dim, num_warps, vec, interval, warp_size=64):
    """Address layout for an async copy of a `[rows, cols]` tile.

    `vec` contiguous elements per lane; the rest of the contiguous axis across
    lanes, then the strided bits the destination stores next; the remaining
    strided bits across warps, then registers.
    """
    c, r = _contig_strided(rows, cols, contig_dim)

    def base(cv, rv):
        return [cv, rv] if contig_dim == 0 else [rv, cv]

    c_lane_bits = _bits(c // vec)
    c_reg_bits = _bits(c) - c_lane_bits
    r_warp_bits = _bits(num_warps)
    r_lane_bits = _bits(warp_size) - c_lane_bits
    r_lane_lo, _ = _async_copy_lane_rows(rows, cols, contig_dim, interval)
    r_rest = [
        i for i in range(_bits(r)) if not r_lane_lo <= i < r_lane_lo + r_lane_bits
    ]

    reg = [base(1 << i, 0) for i in range(c_reg_bits)]
    reg += [base(0, 1 << i) for i in r_rest[r_warp_bits:]]
    lane = [base(1 << (c_reg_bits + i), 0) for i in range(c_lane_bits)]
    lane += [base(0, 1 << (r_lane_lo + i)) for i in range(r_lane_bits)]
    warp = [base(0, 1 << i) for i in r_rest[:r_warp_bits]]
    return gl.DistributedLinearLayout(
        reg_bases=reg,
        lane_bases=lane,
        warp_bases=warp,
        block_bases=[],
        shape=[rows, cols],
    )


@gluon.constexpr_function
def _async_copy_ok(head_dim_pow2, block_n, num_warps, elem_bits, stride_align):
    """Can both the K^T and the V tile be async-copied, at one vector width?

    `stride_align` is 0 when the host could not verify any alignment (a K/V whose
    last axis is not contiguous), which disqualifies.
    """
    if elem_bits not in (8, 16):
        return False
    interval = _kv_lds_params(elem_bits)[0]
    if not _staggered_layout_ok(head_dim_pow2, block_n, interval):
        return False
    vec = _async_copy_vec(
        head_dim_pow2, block_n, 0, num_warps, elem_bits, stride_align, interval
    )
    if vec == 0:
        return False
    return (
        _async_copy_vec(
            block_n, head_dim_pow2, 1, num_warps, elem_bits, stride_align, interval
        )
        == vec
    )


@gluon.constexpr_function
def kv_async_copy_ok(
    head_dim_pow2, pe_head_dim, block_n, num_warps, elem_bits, stride_align
):
    """Can every K/V tile (and the PE slice) of this launch be async-copied?  The
    kernel has no other way to stage K/V; the host makes K/V contiguous when not."""
    return _async_copy_ok(
        head_dim_pow2, block_n, num_warps, elem_bits, stride_align
    ) and (
        pe_head_dim == 0
        or _async_copy_ok(pe_head_dim, block_n, num_warps, elem_bits, stride_align)
    )


@gluon.constexpr_function
def _kv_async_copy_layouts(head_dim_pow2, block_n, num_warps, elem_bits, stride_align):
    """(K^T, V) async-copy address layouts."""
    interval = _kv_lds_params(elem_bits)[0]
    vec = _async_copy_vec(
        head_dim_pow2, block_n, 0, num_warps, elem_bits, stride_align, interval
    )
    return (
        _async_copy_layout(head_dim_pow2, block_n, 0, num_warps, vec, interval),
        _async_copy_layout(block_n, head_dim_pow2, 1, num_warps, vec, interval),
    )


@gluon.constexpr_function
def _make_load_layout(block_dmodel, load_vec, num_warps, transposed, lanes=64):
    """Blocked layout for one tile load."""
    warp_elems = lanes * load_vec
    if transposed:
        return gl.BlockedLayout(
            [load_vec, 1],
            [block_dmodel // load_vec, warp_elems // block_dmodel],
            [1, num_warps],
            [0, 1],
        )
    return gl.BlockedLayout(
        [1, load_vec],
        [warp_elems // block_dmodel, block_dmodel // load_vec],
        [num_warps, 1],
        [1, 0],
    )


@gluon.constexpr_function
def head_split(head_dim, head_dim_pow2, min_hi):
    """Width of the upper slice when a padded head dim is issued as
    `[0, pow2 / 2)` plus a power-of-two `[pow2 / 2, +hi)`, or 0 when that saves
    nothing.  `min_hi` is the MFMA's narrowest operand on that side: 16 for the
    Q@K^T contraction, 32 for the P@V output."""
    if head_dim == head_dim_pow2:
        return 0
    lo = head_dim_pow2 // 2
    hi = max(1 << (head_dim - lo - 1).bit_length(), min_hi)
    return hi if lo + hi < head_dim_pow2 else 0


@gluon.constexpr_function
def _head_slices(head_dim, head_dim_pow2, has_pe, is_fp8):
    """`(D_LO, D_HI_QK, D_HI_PV)`: how the head dim is issued.

    A padded head (96 -> 128) wastes part of every MFMA on padding.  When it saves
    work it is issued as a lower slice plus a narrow upper one: on the Q@K^T side
    a second MFMA chain (never with PE, which is that chain already), on the P@V
    side a second accumulator `acc2` (only when Q@K^T splits too).  K and V stay
    one padded LDS tile; slices are sub-views.  fp8 never splits.  With no split,
    `D_LO` is the whole padded width.
    """
    hi_qk = 0 if has_pe or is_fp8 else head_split(head_dim, head_dim_pow2, 16)
    if hi_qk == 0:
        return head_dim_pow2, 0, 0
    return head_dim_pow2 // 2, hi_qk, head_split(head_dim, head_dim_pow2, 32)


# ---------------------------------------------------------------------------
# 2. Aggregates
# ---------------------------------------------------------------------------


@aggregate
@strip_annotate
class MhaConfig:
    """Compile-time shape of one launch, and the layouts derived from it."""

    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_DMODEL: gl.constexpr  # real head dim of V (and of Q / K without PE)
    BLOCK_DMODEL_POW2: gl.constexpr
    BLOCK_DMODEL_PE: gl.constexpr  # 0, or the PE slice's width
    HAS_PE: gl.constexpr
    PADDED_HEAD: gl.constexpr
    D_LO: gl.constexpr  # `_head_slices`
    D_HI_QK: gl.constexpr
    D_HI_PV: gl.constexpr
    HAS_PE_DOT: gl.constexpr  # a second Q@K^T chain: PE, or a split head
    IS_FP8: gl.constexpr
    SCALE_ON_Q: gl.constexpr  # qk_scale folded into Q once (bf16)
    RETURN_SCORES: gl.constexpr
    P_LIFT: gl.constexpr  # fp8: see `_FP8_P_LIFT`; 0.0 otherwise
    DTYPE: gl.constexpr  # K/V element type; also P's for P@V
    NUM_WARPS: gl.constexpr
    BUF_DEPTH: gl.constexpr  # LDS slots per staged tile
    LOAD_VEC: gl.constexpr  # elements per lane of a Q load (16 B)
    mfma: gl.constexpr
    dotQ: gl.constexpr
    dotK: gl.constexpr
    dotP: gl.constexpr
    dotV: gl.constexpr
    kt_shared: gl.constexpr
    v_shared: gl.constexpr
    kpe_shared: gl.constexpr
    kt_async: gl.constexpr
    v_async: gl.constexpr
    kpe_async: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        BLOCK_M,
        BLOCK_N,
        BLOCK_DMODEL,
        BLOCK_DMODEL_POW2,
        BLOCK_DMODEL_PE,
        IS_FP8,
        SCALE_ON_Q,
        RETURN_SCORES,
        DTYPE,
        KV_STRIDE_ALIGN,
        NUM_WARPS,
    ):
        has_pe = BLOCK_DMODEL_PE > 0
        d_lo, d_hi_qk, d_hi_pv = _head_slices(
            BLOCK_DMODEL, BLOCK_DMODEL_POW2, has_pe, IS_FP8
        )
        elem_bits = DTYPE.primitive_bitwidth
        mfma = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[32, 32, 64] if IS_FP8 else [32, 32, 16],
            transposed=True,
            warps_per_cta=[NUM_WARPS, 1],
        )
        k_width = 16 if IS_FP8 else 8
        kt_shared, v_shared = _make_kv_shared_layouts(
            BLOCK_DMODEL_POW2, BLOCK_N, elem_bits
        )
        kt_async, v_async = _kv_async_copy_layouts(
            BLOCK_DMODEL_POW2, BLOCK_N, NUM_WARPS, elem_bits, KV_STRIDE_ALIGN
        )
        if has_pe:
            # The PE tile is staged like K^T.
            kpe_shared = _make_kv_shared_layouts(BLOCK_DMODEL_PE, BLOCK_N, elem_bits)[0]
            kpe_async = _kv_async_copy_layouts(
                BLOCK_DMODEL_PE, BLOCK_N, NUM_WARPS, elem_bits, KV_STRIDE_ALIGN
            )[0]
        else:
            kpe_shared = None
            kpe_async = None

        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_DMODEL = gl.constexpr(BLOCK_DMODEL)
        self.BLOCK_DMODEL_POW2 = gl.constexpr(BLOCK_DMODEL_POW2)
        self.BLOCK_DMODEL_PE = gl.constexpr(BLOCK_DMODEL_PE)
        self.HAS_PE = gl.constexpr(has_pe)
        self.PADDED_HEAD = gl.constexpr(BLOCK_DMODEL != BLOCK_DMODEL_POW2)
        self.D_LO = gl.constexpr(d_lo)
        self.D_HI_QK = gl.constexpr(d_hi_qk)
        self.D_HI_PV = gl.constexpr(d_hi_pv)
        self.HAS_PE_DOT = gl.constexpr(has_pe or d_hi_qk > 0)
        self.IS_FP8 = gl.constexpr(IS_FP8)
        self.SCALE_ON_Q = gl.constexpr(SCALE_ON_Q)
        self.RETURN_SCORES = gl.constexpr(RETURN_SCORES)
        self.P_LIFT = gl.constexpr(_FP8_P_LIFT.value if IS_FP8 else 0.0)
        self.DTYPE = gl.constexpr(DTYPE)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.BUF_DEPTH = gl.constexpr(2)
        self.LOAD_VEC = gl.constexpr(16 if IS_FP8 else 8)
        self.mfma = gl.constexpr(mfma)
        self.dotQ = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=k_width)
        )
        self.dotK = gl.constexpr(
            gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=k_width)
        )
        self.dotP = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=4)
        )
        self.dotV = gl.constexpr(
            gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=4)
        )
        self.kt_shared = gl.constexpr(kt_shared)
        self.v_shared = gl.constexpr(v_shared)
        self.kpe_shared = gl.constexpr(kpe_shared)
        self.kt_async = gl.constexpr(kt_async)
        self.v_async = gl.constexpr(v_async)
        self.kpe_async = gl.constexpr(kpe_async)


@gluon.jit
def _async_copy_tile(smem, base, offsets, mask, HAS_MASK: gl.constexpr):
    """Issue one global->LDS async copy, without closing a commit group.

    A masked lane is zero-filled by the hardware (its offset is made out of
    bounds).  Do NOT pass `other`: that takes the lowering's general path, an
    exec-mask branch plus a `ds_write` per copy, whose predicates spill on padded
    heads and drain every copy in flight.
    """
    if HAS_MASK:
        cdna4_async.buffer_load_to_shared(smem, base, offsets, mask=mask)
    else:
        cdna4_async.buffer_load_to_shared(smem, base, offsets)


@aggregate
@strip_annotate
class KVStage:
    """K (+PE) and V tiles staged global->LDS by async copy, BUF_DEPTH slots deep.

    The copy offsets are relative to a tile's base pointer, so a tile is selected
    by moving `k_base` / `v_base` and the offsets never change inside a loop.  The
    PE slice shares K's base pointer and K's commit group.
    """

    cfg: MhaConfig
    smemK: gl.shared_memory_descriptor
    smemKpe: gl.shared_memory_descriptor | gl.constexpr  # None without PE
    smemV: gl.shared_memory_descriptor
    kt_off: gl.tensor
    kpe_off: gl.tensor | gl.constexpr
    v_off: gl.tensor
    kt_off_n: gl.tensor  # the offsets' axes, for the token / head-dim masks
    kt_off_d: gl.tensor
    kpe_off_n: gl.tensor | gl.constexpr
    v_off_n: gl.tensor
    v_off_d: gl.tensor
    k_head_mask: gl.tensor | gl.constexpr  # None unless the head is padded
    v_head_mask: gl.tensor | gl.constexpr
    stride_kn: gl.tensor | gl.constexpr
    stride_vn: gl.tensor | gl.constexpr

    @gluon.jit
    def initialize(cfg, k_ptr, v_ptr, stride_kk, stride_kn, stride_vk, stride_vn):
        smemK = gl.allocate_shared_memory(
            k_ptr.dtype.element_ty,
            [cfg.BUF_DEPTH, cfg.BLOCK_DMODEL_POW2, cfg.BLOCK_N],
            cfg.kt_shared,
        )
        smemV = gl.allocate_shared_memory(
            v_ptr.dtype.element_ty,
            [cfg.BUF_DEPTH, cfg.BLOCK_N, cfg.BLOCK_DMODEL_POW2],
            cfg.v_shared,
        )
        if cfg.HAS_PE:
            smemKpe = gl.allocate_shared_memory(
                k_ptr.dtype.element_ty,
                [cfg.BUF_DEPTH, cfg.BLOCK_DMODEL_PE, cfg.BLOCK_N],
                cfg.kpe_shared,
            )
        else:
            smemKpe = None

        kt_off_d = gl.arange(
            0, cfg.BLOCK_DMODEL_POW2, layout=gl.SliceLayout(1, cfg.kt_async)
        )
        kt_off_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.kt_async))
        kt_off = (kt_off_d[:, None] * stride_kk + kt_off_n[None, :] * stride_kn).to(
            gl.int32
        )
        v_off_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.v_async))
        v_off_d = gl.arange(
            0, cfg.BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, cfg.v_async)
        )
        v_off = (v_off_n[:, None] * stride_vn + v_off_d[None, :] * stride_vk).to(
            gl.int32
        )
        if cfg.HAS_PE:
            # The PE slice follows the NOPE head dims of K.
            kpe_off_d = cfg.BLOCK_DMODEL + gl.arange(
                0, cfg.BLOCK_DMODEL_PE, layout=gl.SliceLayout(1, cfg.kpe_async)
            )
            kpe_off_n = gl.arange(
                0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.kpe_async)
            )
            kpe_off = (
                kpe_off_d[:, None] * stride_kk + kpe_off_n[None, :] * stride_kn
            ).to(gl.int32)
        else:
            kpe_off_n = None
            kpe_off = None
        if cfg.PADDED_HEAD:
            k_head_mask = kt_off_d[:, None] < cfg.BLOCK_DMODEL
            v_head_mask = v_off_d[None, :] < cfg.BLOCK_DMODEL
        else:
            k_head_mask = None
            v_head_mask = None
        return KVStage(
            cfg,
            smemK,
            smemKpe,
            smemV,
            kt_off,
            kpe_off,
            v_off,
            kt_off_n,
            kt_off_d,
            kpe_off_n,
            v_off_n,
            v_off_d,
            k_head_mask,
            v_head_mask,
            stride_kn,
            stride_vn,
        )

    @gluon.jit
    def copy(self, slot, k_base, v_base, start_n, seqlen_k, MASK_STEPS: gl.constexpr):
        """Stage one K (+PE) and one V tile into `slot`: two commit groups.

        The token mask is built on MASK_STEPS blocks only; the head-dim mask on
        every block of a padded head.  Each mask carries only the axis it
        constrains, and is broadcast against the offsets.
        """
        cfg = self.cfg
        HAS_MASK: gl.constexpr = MASK_STEPS or cfg.PADDED_HEAD
        if MASK_STEPS:
            k_mask = (start_n + self.kt_off_n)[None, :] < seqlen_k
            v_mask = (start_n + self.v_off_n)[:, None] < seqlen_k
            if cfg.PADDED_HEAD:
                k_mask = k_mask & (self.kt_off_d[:, None] < cfg.BLOCK_DMODEL)
                v_mask = v_mask & (self.v_off_d[None, :] < cfg.BLOCK_DMODEL)
        elif cfg.PADDED_HEAD:
            k_mask = self.kt_off_d[:, None] < cfg.BLOCK_DMODEL
            v_mask = self.v_off_d[None, :] < cfg.BLOCK_DMODEL
        else:
            k_mask = None
            v_mask = None

        # The PE slice is never head-padded; it only takes the token mask.
        if cfg.HAS_PE and MASK_STEPS:
            kpe_mask = (start_n + self.kpe_off_n)[None, :] < seqlen_k
        else:
            kpe_mask = None
        _async_copy_tile(self.smemK.index(slot), k_base, self.kt_off, k_mask, HAS_MASK)
        if cfg.HAS_PE:
            _async_copy_tile(
                self.smemKpe.index(slot), k_base, self.kpe_off, kpe_mask, MASK_STEPS
            )
        cdna4_async.commit_group()
        _async_copy_tile(self.smemV.index(slot), v_base, self.v_off, v_mask, HAS_MASK)
        cdna4_async.commit_group()

    @gluon.jit
    def copy_k(self, slot, k_base, tile, kt_step):
        """K (+PE) of unmasked tile `tile` into `slot`, as one commit group."""
        _async_copy_tile(
            self.smemK.index(slot),
            k_base + tile * kt_step,
            self.kt_off,
            self.k_head_mask,
            self.cfg.PADDED_HEAD,
        )
        if self.cfg.HAS_PE:
            _async_copy_tile(
                self.smemKpe.index(slot),
                k_base + tile * kt_step,
                self.kpe_off,
                None,
                False,
            )
        cdna4_async.commit_group()

    @gluon.jit
    def copy_v(self, slot, v_base, tile, v_step):
        """V of unmasked tile `tile` into `slot`, as one commit group."""
        _async_copy_tile(
            self.smemV.index(slot),
            v_base + tile * v_step,
            self.v_off,
            self.v_head_mask,
            self.cfg.PADDED_HEAD,
        )
        cdna4_async.commit_group()

    @gluon.jit
    def read_k(self, slot):
        """K^T of one staged tile, and the second Q@K^T operand: the PE tile or a
        split head's upper slice (an alias of K with neither -- a loop-carried
        value has to be a real tensor)."""
        cfg = self.cfg
        if cfg.D_HI_QK > 0:
            k = cdna4_async.load_shared_relaxed(
                self.smemK.index(slot).slice(0, cfg.D_LO, 0), cfg.dotK
            )
            k_pe = cdna4_async.load_shared_relaxed(
                self.smemK.index(slot).slice(cfg.D_LO, cfg.D_HI_QK, 0), cfg.dotK
            )
        else:
            k = cdna4_async.load_shared_relaxed(self.smemK.index(slot), cfg.dotK)
            if cfg.HAS_PE:
                k_pe = cdna4_async.load_shared_relaxed(
                    self.smemKpe.index(slot), cfg.dotK
                )
            else:
                k_pe = k
        return k, k_pe

    @gluon.jit
    def read_v(self, slot):
        """V of one staged tile: `(v, v_hi)` on a split P@V, else the whole tile
        and an alias of it."""
        cfg = self.cfg
        if cfg.D_HI_PV > 0:
            v = cdna4_async.load_shared_relaxed(
                self.smemV.index(slot).slice(0, cfg.D_LO, 1), cfg.dotV
            )
            v_hi = cdna4_async.load_shared_relaxed(
                self.smemV.index(slot).slice(cfg.D_LO, cfg.D_HI_PV, 1), cfg.dotV
            )
        else:
            v = cdna4_async.load_shared_relaxed(self.smemV.index(slot), cfg.dotV)
            v_hi = v
        return v, v_hi


@aggregate
@strip_annotate
class KeyMask:
    """Per-row inputs of the element masks: causal diagonal, ragged K end, window."""

    offs_m: gl.tensor  # query rows of this M-block
    seqlen_q: gl.tensor | gl.constexpr
    seqlen_k: gl.tensor | gl.constexpr
    n_extra_tokens: gl.tensor | gl.constexpr  # keys in the last, partial block
    window_min: gl.tensor | gl.constexpr  # lowest key per row; None without window


@aggregate
@strip_annotate
class ScoresOut:
    """Where `return_scores` writes P for one M-block (all None when off)."""

    base: gl.tensor | gl.constexpr
    offsets: gl.tensor | gl.constexpr
    row_mask: gl.tensor | gl.constexpr
    stride_n: gl.tensor | gl.constexpr

    @gluon.jit
    def store(self, p, start_n, offs_n, seqlen_k, P_LIFT: gl.constexpr):
        """Store one tile of P, without fp8's lift.  Not rescaled to the final row
        max, so it differs from a reference softmax."""
        mask = self.row_mask[:, None] & ((start_n + offs_n)[None, :] < seqlen_k)
        if P_LIFT != 0.0:
            p = p * gl.exp2(-P_LIFT)
        gl.amd.cdna4.buffer_store(
            p.to(self.base.dtype.element_ty),
            ptr=self.base + start_n * self.stride_n,
            offsets=self.offsets,
            mask=mask,
        )


# ---------------------------------------------------------------------------
# 3. Per-tile math
# ---------------------------------------------------------------------------

# fp8 scales P statically: P = exp2(s - m) <= 1, so exp2(s - m + P_LIFT) puts it
# on e4m3's range with no per-tile amax.  The row sum carries the same factor, so
# it cancels in acc / l; only the LSE takes it back out.  log2(448), less a hair
# so exp2 rounding never carries P past e4m3's largest value.
_FP8_P_LIFT = gl.constexpr(8.8)


@gluon.jit
def _mma(a, b, acc, IS_FP8: gl.constexpr):
    """One MFMA chain step: `mfma`, or e4m3 `mfma_scaled` without scales."""
    if IS_FP8:
        return gl.amd.cdna4.mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)
    else:
        return gl.amd.cdna4.mfma(a, b, acc)


@gluon.jit
def _dot_qk(cfg, q, q_pe, k, k_pe):
    """Q@K^T for one tile, the second chain (PE or split head) first."""
    qk = gl.zeros([cfg.BLOCK_M, cfg.BLOCK_N], dtype=gl.float32, layout=cfg.mfma)
    if cfg.HAS_PE_DOT:
        qk = _mma(q_pe, k_pe, qk, cfg.IS_FP8)
    return _mma(q, k, qk, cfg.IS_FP8)


@gluon.jit
def _dot_pv(cfg, p_dot, v, v_hi, acc, acc2):
    """P@V for one tile, into `acc` and, on a split P@V, `acc2`."""
    acc = _mma(p_dot, v, acc, cfg.IS_FP8)
    if cfg.D_HI_PV > 0:
        acc2 = _mma(p_dot, v_hi, acc2, cfg.IS_FP8)
    return acc, acc2


@gluon.jit
def _attn_qk(
    cfg,
    km,
    q,
    k,
    q_pe,
    k_pe,
    start_n,
    offs_n,
    qk_scale,
    block_max,
    WINDOW_MASK: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    MASK_STEPS: gl.constexpr,
):
    """Masked fp32 scores of one staged key block.  fp8 leaves them unscaled: its
    softmax folds qk_scale into the exp2 FMA."""
    qk = _dot_qk(cfg, q, q_pe, k, k_pe)

    if not (cfg.SCALE_ON_Q or cfg.IS_FP8):
        qk = qk * qk_scale

    if MASK_STEPS or IS_CAUSAL or WINDOW_MASK:
        key_pos = start_n + offs_n
        mask = gl.full([cfg.BLOCK_M, cfg.BLOCK_N], True, dtype=gl.int1, layout=cfg.mfma)
        if MASK_STEPS:
            # Only the last visible block can be partial.
            bound_cond = (start_n + cfg.BLOCK_N == block_max) and (
                km.n_extra_tokens != 0
            )
            mask_partial = key_pos[None, :] < km.seqlen_k
            mask = gl.where(bound_cond, mask_partial, mask)
        if IS_CAUSAL:
            causal_boundary = key_pos + (km.seqlen_q - km.seqlen_k)
            mask = mask & (km.offs_m[:, None] >= causal_boundary[None, :])
        if WINDOW_MASK:
            mask = mask & (km.window_min[:, None] <= key_pos[None, :])
        qk = gl.where(mask, qk, float("-inf"))

    return qk


@gluon.jit
def _attn_softmax_pv(
    cfg, sd, acc, acc2, l_i, m_i, qk, v, v_hi, offs_n, start_n, seqlen_k
):
    """Online-softmax update and P@V for one key block; `qk` already scaled."""
    m_ij = gl.maximum(m_i, gl.max(qk, 1))
    p = gl.exp2(qk - m_ij[:, None])
    alpha = gl.exp2(m_i - m_ij)
    l_ij = gl.sum(p, 1)

    if cfg.RETURN_SCORES:
        sd.store(p, start_n, offs_n, seqlen_k, 0.0)

    acc = acc * alpha[:, None]
    if cfg.D_HI_PV > 0:
        acc2 = acc2 * alpha[:, None]
    p = gl.convert_layout(p.to(v.dtype), layout=cfg.dotP, assert_trivial=True)
    acc, acc2 = _dot_pv(cfg, p, v, v_hi, acc, acc2)

    l_i = l_i * alpha + l_ij
    m_i = m_ij

    return acc, acc2, l_i, m_i


@gluon.jit
def _attn_softmax_pv_fp8(
    cfg, sd, acc, l_i, m_i, qk, v, qk_scale, offs_n, start_n, seqlen_k
):
    """`_attn_softmax_pv` for e4m3: `qk` unscaled, P lifted by `_FP8_P_LIFT`."""
    m_ij = gl.maximum(m_i, gl.max(qk, 1) * qk_scale)
    p = gl.exp2(gl.fma(qk, qk_scale, _FP8_P_LIFT - m_ij[:, None]))
    alpha = gl.exp2(m_i - m_ij)
    l_ij = gl.sum(p, 1)

    if cfg.RETURN_SCORES:
        sd.store(p, start_n, offs_n, seqlen_k, _FP8_P_LIFT)

    acc = acc * alpha[:, None]
    p = gl.convert_layout(p.to(v.dtype), layout=cfg.dotP, assert_trivial=True)
    acc = _mma(p, v, acc, True)
    l_i = l_i * alpha + l_ij
    return acc, l_i, m_ij


@gluon.jit
def _sc_vec1(cfg, qk, m_run, qk_scale):
    """Pipeline VEC1 (in `dot2`): new row max, the exp2 burst, and alpha."""
    if cfg.SCALE_ON_Q:
        m_new = gl.maximum(m_run, gl.max(qk, 1))
        p = gl.exp2(qk - m_new[:, None])
    else:
        m_new = gl.maximum(m_run, gl.max(qk, 1) * qk_scale)
        # One fused FMA at the source, not an fmul/fsub pair for the backend to
        # contract after scheduling has already counted it.
        if cfg.P_LIFT != 0.0:
            p = gl.exp2(gl.fma(qk, qk_scale, cfg.P_LIFT - m_new[:, None]))
        else:
            p = gl.exp2(gl.fma(qk, qk_scale, -m_new[:, None]))
    alpha = gl.exp2(m_run - m_new)
    return m_new, p, alpha


@gluon.jit
def _sc_vec2(cfg, acc, acc2, l_i, p, alpha):
    """Pipeline VEC2 (in `dot1`): rescale, row sum and P downcast, for the tile
    VEC1 produced one stage earlier.  The rescale goes first: the part this
    cluster cannot hide then lands ahead of its MFMAs, with no hazard."""
    acc = acc * alpha[:, None]
    if cfg.D_HI_PV > 0:
        acc2 = acc2 * alpha[:, None]
    l_ij = gl.sum(p, axis=1)
    l_i = l_i * alpha + l_ij
    p_dot = gl.convert_layout(p.to(cfg.DTYPE), cfg.dotP)
    return acc, acc2, l_i, p_dot


# ---------------------------------------------------------------------------
# 4. K loops
# ---------------------------------------------------------------------------


@gluon.jit
def _attn_fwd_inner(
    cfg,
    kv,
    sd,
    km,
    acc,
    acc2,
    l_i,
    m_i,
    q,
    q_pe,
    k_base,
    v_base,
    block_min,
    block_max,
    qk_scale,
    WINDOW_MASK: gl.constexpr = False,
    IS_CAUSAL: gl.constexpr = False,
    MASK_STEPS: gl.constexpr = False,
):
    """Key blocks `[block_min, block_max)` one tile at a time, any mask applied.

    The copy of tile i+1 is in flight while tile i computes.  `acc2` is the
    split-head P@V accumulator (an alias of `acc` without one); fp8 passes it
    through.
    """
    if MASK_STEPS or IS_CAUSAL or WINDOW_MASK or cfg.RETURN_SCORES:
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.mfma))
    else:
        offs_n = None

    n_iter = (block_max - block_min) // cfg.BLOCK_N

    kv.copy(0, k_base, v_base, block_min, km.seqlen_k, MASK_STEPS)

    for i in range(n_iter):
        start_n = block_min + i * cfg.BLOCK_N

        cur = i % cfg.BUF_DEPTH
        nxt = cfg.BUF_DEPTH - 1 - cur
        # Clamped: the last iteration restages a tile nobody reads, which is
        # cheaper than a branch and keeps `wait_group` at a fixed depth.
        nxt_i = min(i + 1, n_iter - 1)
        kv.copy(
            nxt,
            k_base + nxt_i * cfg.BLOCK_N * kv.stride_kn,
            v_base + nxt_i * cfg.BLOCK_N * kv.stride_vn,
            block_min + nxt_i * cfg.BLOCK_N,
            km.seqlen_k,
            MASK_STEPS,
        )
        cdna4_async.wait_group(2)  # this tile's K and V; the prefetch stays in flight
        k, k_pe = kv.read_k(cur)
        v, v_hi = kv.read_v(cur)
        qk = _attn_qk(
            cfg,
            km,
            q,
            k,
            q_pe,
            k_pe,
            start_n,
            offs_n,
            qk_scale,
            block_max,
            WINDOW_MASK,
            IS_CAUSAL,
            MASK_STEPS,
        )
        if cfg.IS_FP8:
            acc, l_i, m_i = _attn_softmax_pv_fp8(
                cfg, sd, acc, l_i, m_i, qk, v, qk_scale, offs_n, start_n, km.seqlen_k
            )
        else:
            acc, acc2, l_i, m_i = _attn_softmax_pv(
                cfg, sd, acc, acc2, l_i, m_i, qk, v, v_hi, offs_n, start_n, km.seqlen_k
            )

        # WAR: this tile's LDS reads against the copy that next overwrites the slot.
        gl.barrier()

    cdna4_async.wait_group(0)

    return acc, acc2, l_i, m_i


# The rotated four-cluster pipeline.  One tile's work is cut into four clusters
# that alternate matrix and memory work, and the two waves of a SIMD run one
# cluster apart, so the softmax issues in the shadow of the other wave's MFMAs:
#
#   dot1   Q@K^T for tile j+1     VEC2 for tile j    (rescale, row sum, P downcast)
#   mem1   read V[j] from LDS     async-copy K[j+3] -> LDS
#   dot2   P@V for tile j         VEC1 for tile j+1  (row max, exp2 burst)
#   mem2   read K[j+2] from LDS   async-copy V[j+2] -> LDS
#
# The pipeline has a three-tile prologue and drain; below this many tiles they
# overlap and the stage barriers no longer order an LDS slot's read before the
# copy that overwrites it.  A correctness bound, not a tuning knob.
_MIN_PIPE_BLOCKS = gl.constexpr(8)


@gluon.jit
def _attn_fwd_pipelined(
    cfg, kv, acc, acc2, l_i, m_i, q, q_pe, k_base, v_base, n_full_blocks, qk_scale
):
    """Unmasked key tiles `[0, n_full_blocks)` through the rotated pipeline.

    Every tile is addressed as an offset from `k_base` / `v_base`, so no address
    VALU runs inside the loop.  The PE slice rides in its K tile's commit group:
    every `wait_group` count below is in commit groups.
    """
    kt_step = cfg.BLOCK_N * kv.stride_kn
    v_step = cfg.BLOCK_N * kv.stride_vn

    # Prologue: tile 0's QK and VEC1, K[1] in registers; K[0..2] and V[0..1]
    # staged.  Commit order K0, V0, K1, K2, V1 leaves {K2, V1} pending, the loop's
    # entry state.  K[2] reuses slot 0 once tile 0 is read -- hence the barrier.
    kv.copy_k(0, k_base, 0, kt_step)
    kv.copy_v(0, v_base, 0, v_step)
    kv.copy_k(1, k_base, 1, kt_step)

    cdna4_async.wait_group(2)  # K[0] has landed
    kt0, kpe0 = kv.read_k(0)
    qk = _dot_qk(cfg, q, q_pe, kt0, kpe0)
    m_run, p_c, alpha_c = _sc_vec1(cfg, qk, m_i, qk_scale)

    gl.barrier()  # WAR: tile 0's read against K[2]'s write into slot 0
    kv.copy_k(0, k_base, 2, kt_step)
    cdna4_async.wait_group(1)  # K[1] has landed
    kt_dot, kpe_dot = kv.read_k(1)
    kv.copy_v(1, v_base, 1, v_step)

    # Main loop, unrolled 2x: over a pair of tiles the slots swap back, so every
    # slot index is a compile-time constant.
    pairs = (n_full_blocks - 3) // 2
    for pair in range(pairs):
        # Tile blk reads slot cur and refills K into slot 1 - cur.
        for cur in gl.static_range(cfg.BUF_DEPTH):
            blk = pair * 2 + cur
            with warp_pipeline_stage("dot1"):
                qk = _dot_qk(cfg, q, q_pe, kt_dot, kpe_dot)
                acc, acc2, l_i, p_dot = _sc_vec2(cfg, acc, acc2, l_i, p_c, alpha_c)
            cdna4_async.wait_group(1)
            with warp_pipeline_stage("mem1"):
                v_dot, v_hi = kv.read_v(cur)
                kv.copy_k(1 - cur, k_base, blk + 3, kt_step)
            with warp_pipeline_stage("dot2"):
                acc, acc2 = _dot_pv(cfg, p_dot, v_dot, v_hi, acc, acc2)
                m_run, p_c, alpha_c = _sc_vec1(cfg, qk, m_run, qk_scale)
            cdna4_async.wait_group(1)
            with warp_pipeline_stage("mem2"):
                kt_dot, kpe_dot = kv.read_k(cur)
                kv.copy_v(cur, v_base, blk + 2, v_step)

    # An odd count leaves one tile over, always in slots 0/1.
    if (n_full_blocks - 3) % 2 == 1:
        blk = pairs * 2
        with warp_pipeline_stage("dot1"):
            qk = _dot_qk(cfg, q, q_pe, kt_dot, kpe_dot)
            acc, acc2, l_i, p_dot = _sc_vec2(cfg, acc, acc2, l_i, p_c, alpha_c)
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem1"):
            v_dot, v_hi = kv.read_v(0)
            kv.copy_k(1, k_base, blk + 3, kt_step)
        with warp_pipeline_stage("dot2"):
            acc, acc2 = _dot_pv(cfg, p_dot, v_dot, v_hi, acc, acc2)
            m_run, p_c, alpha_c = _sc_vec1(cfg, qk, m_run, qk_scale)
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem2"):
            kt_dot, kpe_dot = kv.read_k(0)
            kv.copy_v(0, v_base, blk + 2, v_step)

    # Drain: the last three tiles, with nothing left to prefetch.
    nm3 = n_full_blocks - 3
    nm2 = n_full_blocks - 2
    nm1 = n_full_blocks - 1
    s_nm3 = (nm3 % cfg.BUF_DEPTH).to(gl.int32)
    s_nm2 = (nm2 % cfg.BUF_DEPTH).to(gl.int32)
    s_nm1 = (nm1 % cfg.BUF_DEPTH).to(gl.int32)

    qk = _dot_qk(cfg, q, q_pe, kt_dot, kpe_dot)
    cdna4_async.wait_group(2)
    v_dot, v_hi = kv.read_v(s_nm3)
    acc, acc2, l_i, p_dot = _sc_vec2(cfg, acc, acc2, l_i, p_c, alpha_c)
    acc, acc2 = _dot_pv(cfg, p_dot, v_dot, v_hi, acc, acc2)
    m_run, p_c, alpha_c = _sc_vec1(cfg, qk, m_run, qk_scale)
    gl.barrier()  # WAR: tile n-3's V read against V[n-1]'s write into that slot
    kv.copy_v(s_nm1, v_base, nm1, v_step)
    cdna4_async.wait_group(2)
    kt_dot, kpe_dot = kv.read_k(s_nm1)

    qk = _dot_qk(cfg, q, q_pe, kt_dot, kpe_dot)
    cdna4_async.wait_group(1)
    v_dot, v_hi = kv.read_v(s_nm2)
    acc, acc2, l_i, p_dot = _sc_vec2(cfg, acc, acc2, l_i, p_c, alpha_c)
    acc, acc2 = _dot_pv(cfg, p_dot, v_dot, v_hi, acc, acc2)
    m_run, p_c, alpha_c = _sc_vec1(cfg, qk, m_run, qk_scale)

    cdna4_async.wait_group(0)
    v_dot, v_hi = kv.read_v(s_nm1)
    acc, acc2, l_i, p_dot = _sc_vec2(cfg, acc, acc2, l_i, p_c, alpha_c)
    acc, acc2 = _dot_pv(cfg, p_dot, v_dot, v_hi, acc, acc2)

    return acc, acc2, l_i, m_run


# ---------------------------------------------------------------------------
# 5. Kernel
# ---------------------------------------------------------------------------


@gluon.jit
def _stride(s, INT64: gl.constexpr):
    """A stride as int64 when INT64, so base-pointer arithmetic cannot overflow on
    huge tensors; per-tile offsets stay int32."""
    if INT64:
        s = gl.cast(s, gl.int64)
    return s


@gluon.jit
def _remap_xcd(
    NUM_BLOCKS_M,
    BATCH,
    NUM_Q_HEADS: gl.constexpr,
    NUM_XCD: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
):
    """program id -> (batch, q head, M-block): q heads spread across XCDs, and
    causal M-blocks in reverse so the longest rows start first."""
    pid = gl.program_id(axis=0)
    off_q_head = pid % NUM_Q_HEADS
    off_q_head = remap_xcd(off_q_head, NUM_Q_HEADS, NUM_XCD)
    start_m = (pid // NUM_Q_HEADS) % NUM_BLOCKS_M
    off_z = pid // (NUM_Q_HEADS * NUM_BLOCKS_M) % BATCH
    if IS_CAUSAL:
        start_m = NUM_BLOCKS_M - 1 - start_m
    return off_z, off_q_head, start_m


@gluon.jit
def _load_q(
    cfg,
    q_ptr,
    off_z,
    qh_off,
    cu_seqlens_q_start,
    start_m,
    seqlen_q,
    stride_qz,
    stride_qm,
    stride_qk,
    qk_scale,
    NUM_Q_HEADS: gl.constexpr,
):
    """Q of one M-block in its MFMA operand layout, and the second Q@K^T operand
    (the PE slice or a split head's upper slice).  With SCALE_ON_Q both are
    pre-scaled by qk_scale (one extra rounding of Q)."""
    qLoadLayout: gl.constexpr = _make_load_layout(
        cfg.D_LO, cfg.LOAD_VEC, cfg.NUM_WARPS, transposed=False
    )
    offs_qm = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, qLoadLayout))
    offs_qd = gl.arange(0, cfg.D_LO, layout=gl.SliceLayout(0, qLoadLayout))
    q_base = (
        q_ptr
        + off_z * stride_qz
        + qh_off
        + cu_seqlens_q_start * stride_qm
        + start_m * cfg.BLOCK_M * stride_qm
    )
    q_offsets = (offs_qm[:, None] * stride_qm + offs_qd[None, :] * stride_qk).to(
        gl.int32
    )
    q_mask = (start_m * cfg.BLOCK_M + offs_qm)[:, None] < seqlen_q
    if cfg.PADDED_HEAD and cfg.D_HI_QK == 0:
        # (A split head's lower slice is all real head dim.)
        q_mask = q_mask & (offs_qd[None, :] < cfg.BLOCK_DMODEL)
    # Stream Q (.cg) when one Q block spans at least one full head.
    if cfg.BLOCK_M >= NUM_Q_HEADS:
        q_cache_mod: gl.constexpr = ".cg"
    else:
        q_cache_mod: gl.constexpr = ""
    q = gl.amd.cdna4.buffer_load(
        ptr=q_base, offsets=q_offsets, mask=q_mask, other=0.0, cache=q_cache_mod
    )
    q = gl.convert_layout(q, layout=cfg.dotQ)
    if cfg.SCALE_ON_Q and cfg.D_HI_QK == 0:
        q = (q.to(gl.float32) * qk_scale).to(q_ptr.dtype.element_ty)

    if cfg.HAS_PE:
        # The PE slice follows the NOPE head dims of Q and K.
        qPeLoadLayout: gl.constexpr = _make_load_layout(
            cfg.BLOCK_DMODEL_PE, cfg.LOAD_VEC, cfg.NUM_WARPS, transposed=False
        )
        offs_qpm = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, qPeLoadLayout))
        offs_qpd = cfg.BLOCK_DMODEL + gl.arange(
            0, cfg.BLOCK_DMODEL_PE, layout=gl.SliceLayout(0, qPeLoadLayout)
        )
        q_pe_offsets = (
            offs_qpm[:, None] * stride_qm + offs_qpd[None, :] * stride_qk
        ).to(gl.int32)
        q_pe_mask = (start_m * cfg.BLOCK_M + offs_qpm)[:, None] < seqlen_q
        q_pe = gl.amd.cdna4.buffer_load(
            ptr=q_base,
            offsets=q_pe_offsets,
            mask=q_pe_mask,
            other=0.0,
            cache=q_cache_mod,
        )
        q_pe = gl.convert_layout(q_pe, layout=cfg.dotQ)
    elif cfg.D_HI_QK > 0:
        # A split head's upper slice `[D_LO, D_LO + D_HI_QK)`, head-dim masked.
        qHiLoadLayout: gl.constexpr = _make_load_layout(
            cfg.D_HI_QK, cfg.LOAD_VEC, cfg.NUM_WARPS, transposed=False
        )
        offs_qhm = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, qHiLoadLayout))
        offs_qhd = cfg.D_LO + gl.arange(
            0, cfg.D_HI_QK, layout=gl.SliceLayout(0, qHiLoadLayout)
        )
        q_hi_offsets = (
            offs_qhm[:, None] * stride_qm + offs_qhd[None, :] * stride_qk
        ).to(gl.int32)
        q_hi_mask = ((start_m * cfg.BLOCK_M + offs_qhm)[:, None] < seqlen_q) & (
            offs_qhd[None, :] < cfg.BLOCK_DMODEL
        )
        q_hi = gl.amd.cdna4.buffer_load(
            ptr=q_base,
            offsets=q_hi_offsets,
            mask=q_hi_mask,
            other=0.0,
            cache=q_cache_mod,
        )
        q_pe = gl.convert_layout(q_hi, layout=cfg.dotQ)
    else:
        q_pe = q  # unused; a returned value has to be a tensor
    if cfg.SCALE_ON_Q and cfg.D_HI_QK > 0:
        # Both slices loaded first, then scaled: ~2% faster at short sequences.
        q = (q.to(gl.float32) * qk_scale).to(q_ptr.dtype.element_ty)
    if cfg.SCALE_ON_Q and cfg.HAS_PE_DOT:
        q_pe = (q_pe.to(gl.float32) * qk_scale).to(q_ptr.dtype.element_ty)
    return q, q_pe


@gluon.jit
def _classify_k_blocks(
    start_m,
    seqlen_q,
    seqlen_k,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
):
    """Split this M-block's key range into, in order: `skipped_blocks` left of the
    window, `window_blocks` cut by its edge, `n_full_blocks` with no mask, and
    `masked_blocks` on the causal diagonal / ragged K end.

    Returns `(n_blocks, n_extra_tokens, skipped_blocks, window_blocks,
    masked_blocks, n_full_blocks)`; `n_blocks <= 0` is a causal M-block that sees
    no key.
    """
    n_blocks = gl.cdiv(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_causal = gl.cdiv(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )
        n_blocks = min(n_blocks, n_blocks_causal)

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = (not padded_block_k) and (seqlen_q % BLOCK_M == 0)

    skipped_blocks = 0
    if SLIDING_WINDOW > 0:
        window_start_n = start_m * BLOCK_M + seqlen_k - seqlen_q - SLIDING_WINDOW
        skipped_blocks = min(max(window_start_n, 0) // BLOCK_N, n_blocks)

    if IS_CAUSAL:
        # BLOCK_M // BLOCK_N diagonal blocks, one more when the lengths differ
        # (capped below).
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        masked_blocks = padded_block_k

    visible_blocks = n_blocks - skipped_blocks
    masked_blocks = min(masked_blocks, visible_blocks)

    # The window's lower edge is a diagonal: it cuts only the blocks left of the
    # last row's bound; the rest of a windowed row is dense.
    window_blocks = 0
    if SLIDING_WINDOW > 0:
        window_full_n = window_start_n + BLOCK_M - 1
        window_blocks = gl.cdiv(max(window_full_n, 0), BLOCK_N) - skipped_blocks
        window_blocks = min(max(window_blocks, 0), visible_blocks - masked_blocks)
    n_full_blocks = visible_blocks - masked_blocks - window_blocks
    return (
        n_blocks,
        n_extra_tokens,
        skipped_blocks,
        window_blocks,
        masked_blocks,
        n_full_blocks,
    )


@gluon.jit
def _store_out(
    acc,
    o_base,
    row0,
    seqlen_q,
    stride_om,
    stride_on,
    COL0: gl.constexpr,
    WIDTH: gl.constexpr,
    BLOCK_DMODEL_OUT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    LOAD_VEC: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Store output columns `[COL0, COL0 + WIDTH)` of one M-block, dropping rows
    past `seqlen_q` and columns past the caller's head dim."""
    storeLayout: gl.constexpr = _make_load_layout(
        WIDTH, LOAD_VEC, num_warps, transposed=False
    )
    out = gl.convert_layout(acc.to(o_base.dtype.element_ty), layout=storeLayout)

    offs_rm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, storeLayout))
    offs_od = COL0 + gl.arange(0, WIDTH, layout=gl.SliceLayout(0, storeLayout))
    o_offsets = (offs_rm[:, None] * stride_om + offs_od[None, :] * stride_on).to(
        gl.int32
    )

    out_mask = gl.full([BLOCK_M, 1], True, dtype=gl.int1, layout=storeLayout)
    if row0 + BLOCK_M > seqlen_q:
        out_mask = out_mask & ((row0 + offs_rm)[:, None] < seqlen_q)
    if COL0 + WIDTH > BLOCK_DMODEL_OUT:
        out_mask = out_mask & (offs_od[None, :] < BLOCK_DMODEL_OUT)
    gl.amd.cdna4.buffer_store(out, ptr=o_base, offsets=o_offsets, mask=out_mask)


_attn_fwd_repr = make_kernel_repr(
    "_attn_fwd_gluon",
    [
        "IS_CAUSAL",
        "NUM_Q_HEADS",
        "NUM_K_HEADS",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_DMODEL",
        "RETURN_SCORES",
        "HEAD_STRIDE_ALIGN",
        "IS_FP8",
        "VARLEN",
        "NUM_XCD",
        "USE_INT64_STRIDES",
        "ENABLE_SINK",
        "SLIDING_WINDOW",
    ],
)


@gluon.jit(repr=_attn_fwd_repr)
def _attn_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    softmax_lse_ptr,
    s_dmask_ptr,
    descale_q_ptr,
    descale_k_ptr,
    descale_v_ptr,
    sink_ptr,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    SEQLEN_Q,
    SEQLEN_K,
    stride_qz_in,
    stride_qh_in,
    stride_qm_in,
    stride_qk_in,
    stride_kz_in,
    stride_kh_in,
    stride_kn_in,
    stride_kk_in,
    stride_vz_in,
    stride_vh_in,
    stride_vn_in,
    stride_vk_in,
    stride_oz_in,
    stride_oh_in,
    stride_om_in,
    stride_on_in,
    stride_lse_z_in,
    stride_lse_h_in,
    stride_lse_m_in,
    stride_sd_z_in,
    stride_sd_h_in,
    stride_sd_m_in,
    stride_sd_n_in,
    stride_descale_q_z_in,
    stride_descale_k_z_in,
    stride_descale_v_z_in,
    NUM_Q_HEADS: gl.constexpr,
    NUM_K_HEADS: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    VARLEN: gl.constexpr,
    BATCH,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,  # zero, or a power of 2 >= 16
    BLOCK_DMODEL_OUT: gl.constexpr,
    NUM_XCD: gl.constexpr,
    USE_INT64_STRIDES: gl.constexpr,
    IS_FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
    ENABLE_SINK: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    RETURN_SCORES: gl.constexpr,
    HEAD_STRIDE_ALIGN: gl.constexpr,
    KV_STRIDE_ALIGN: gl.constexpr = 1,
    SCALE_ON_Q: gl.constexpr = False,
    num_warps: gl.constexpr = 4,
):
    gl.static_assert(not (IS_FP8 and BLOCK_DMODEL_PE > 0), "fp8 takes no PE slice")
    cfg = MhaConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_DMODEL,
        BLOCK_DMODEL_POW2,
        BLOCK_DMODEL_PE,
        IS_FP8,
        SCALE_ON_Q,
        RETURN_SCORES,
        k_ptr.dtype.element_ty,
        KV_STRIDE_ALIGN,
        num_warps,
    )
    # The host checks the same predicate and makes K/V contiguous when it fails.
    gl.static_assert(
        kv_async_copy_ok(
            BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE,
            BLOCK_N,
            num_warps,
            k_ptr.dtype.element_ty.primitive_bitwidth,
            KV_STRIDE_ALIGN,
        ),
        "K/V cannot be async-copied for this tile; the host must make K/V contiguous",
    )
    RCP_LN2: gl.constexpr = 1.4426950408889634
    PADDED_HEAD_OUT: gl.constexpr = BLOCK_DMODEL_OUT != BLOCK_DMODEL_POW2
    STORE_VEC: gl.constexpr = 8  # 16 B of the 16-bit output
    D_LO: gl.constexpr = cfg.D_LO
    D_HI_PV: gl.constexpr = cfg.D_HI_PV
    ACC_WIDTH: gl.constexpr = D_LO if D_HI_PV > 0 else BLOCK_DMODEL_POW2

    INT64: gl.constexpr = USE_INT64_STRIDES
    stride_qz = _stride(stride_qz_in, INT64)
    stride_qh = _stride(stride_qh_in, INT64)
    stride_qm = _stride(stride_qm_in, INT64)
    stride_qk = _stride(stride_qk_in, INT64)
    stride_kz = _stride(stride_kz_in, INT64)
    stride_kh = _stride(stride_kh_in, INT64)
    stride_kn = _stride(stride_kn_in, INT64)
    stride_kk = _stride(stride_kk_in, INT64)
    stride_vz = _stride(stride_vz_in, INT64)
    stride_vh = _stride(stride_vh_in, INT64)
    stride_vn = _stride(stride_vn_in, INT64)
    stride_vk = _stride(stride_vk_in, INT64)
    stride_descale_q_z = _stride(stride_descale_q_z_in, INT64)
    stride_descale_k_z = _stride(stride_descale_k_z_in, INT64)
    stride_descale_v_z = _stride(stride_descale_v_z_in, INT64)
    stride_oz = _stride(stride_oz_in, INT64)
    stride_oh = _stride(stride_oh_in, INT64)
    stride_om = _stride(stride_om_in, INT64)
    stride_on = _stride(stride_on_in, INT64)
    stride_lse_z = _stride(stride_lse_z_in, INT64)
    stride_lse_h = _stride(stride_lse_h_in, INT64)
    stride_lse_m = _stride(stride_lse_m_in, INT64)
    stride_sd_z = _stride(stride_sd_z_in, INT64)
    stride_sd_h = _stride(stride_sd_h_in, INT64)
    stride_sd_m = _stride(stride_sd_m_in, INT64)
    stride_sd_n = _stride(stride_sd_n_in, INT64)

    # The async copy only lowers when it can prove each lane's chunk is aligned to
    # its own size, and every K/V stride reaches that chunk's address.  Triton
    # infers alignment only for multiples of 16 elements (a padded head of 40 is
    # 16-byte aligned but misses it), so state what the host measured.
    if KV_STRIDE_ALIGN > 1:
        stride_kz = gl.multiple_of(stride_kz, KV_STRIDE_ALIGN)
        stride_kh = gl.multiple_of(stride_kh, KV_STRIDE_ALIGN)
        stride_kn = gl.multiple_of(stride_kn, KV_STRIDE_ALIGN)
        stride_vz = gl.multiple_of(stride_vz, KV_STRIDE_ALIGN)
        stride_vh = gl.multiple_of(stride_vh, KV_STRIDE_ALIGN)
        stride_vn = gl.multiple_of(stride_vn, KV_STRIDE_ALIGN)

    # SEQLEN_Q is the max query length, so the grid matches in varlen mode too.
    NUM_BLOCKS_M = gl.cdiv(SEQLEN_Q, BLOCK_M)
    off_z, off_q_head, start_m = _remap_xcd(
        NUM_BLOCKS_M, BATCH, NUM_Q_HEADS, NUM_XCD, IS_CAUSAL
    )

    # varlen: lengths from cu_seqlens, and the batch strides are 0.
    if VARLEN:
        cu_seqlens_q_start = gl.load(cu_seqlens_q + off_z)
        seqlen_q = gl.load(cu_seqlens_q + off_z + 1) - cu_seqlens_q_start
        if start_m * BLOCK_M >= seqlen_q:
            return
        cu_seqlens_k_start = gl.load(cu_seqlens_k + off_z)
        seqlen_k = gl.load(cu_seqlens_k + off_z + 1) - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = SEQLEN_Q
        seqlen_k = SEQLEN_K

    grp_sz: gl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    off_k_head = off_q_head // grp_sz

    if IS_FP8:
        descale_q = gl.load(descale_q_ptr + off_z * stride_descale_q_z + off_q_head)
        descale_k = gl.load(descale_k_ptr + off_z * stride_descale_k_z + off_k_head)
        descale_v = gl.load(descale_v_ptr + off_z * stride_descale_v_z + off_k_head)

    # HEAD_STRIDE_ALIGN: largest power of two (up to 16 B) dividing every head stride.
    qh_off = off_q_head * stride_qh
    kh_off = off_k_head * stride_kh
    vh_off = off_k_head * stride_vh
    if HEAD_STRIDE_ALIGN > 1:
        qh_off = gl.multiple_of(qh_off, HEAD_STRIDE_ALIGN)
        kh_off = gl.multiple_of(kh_off, HEAD_STRIDE_ALIGN)
        vh_off = gl.multiple_of(vh_off, HEAD_STRIDE_ALIGN)

    if IS_FP8:
        # Q stays e4m3 and the scale (with both descales) rides the softmax.  Kept
        # positive so a masked -inf * scale stays -inf.
        qk_scale = gl.maximum(sm_scale * RCP_LN2 * descale_q * descale_k, 1.0e-30)
    else:
        qk_scale = sm_scale * RCP_LN2

    q, q_pe = _load_q(
        cfg,
        q_ptr,
        off_z,
        qh_off,
        cu_seqlens_q_start,
        start_m,
        seqlen_q,
        stride_qz,
        stride_qm,
        stride_qk,
        qk_scale,
        NUM_Q_HEADS,
    )

    k_base = k_ptr + off_z * stride_kz + kh_off + cu_seqlens_k_start * stride_kn

    v_base = v_ptr + off_z * stride_vz + vh_off + cu_seqlens_k_start * stride_vn

    kv = KVStage.initialize(
        cfg, k_ptr, v_ptr, stride_kk, stride_kn, stride_vk, stride_vn
    )

    # Online-softmax state.  A sliding window can mask whole rows of a block; a
    # finite initial max keeps exp2(m_old - m_new) at 1 there instead of NaN.
    if ENABLE_SINK:
        m_i_init = gl.load(sink_ptr + off_q_head).to(gl.float32) * RCP_LN2
    elif SLIDING_WINDOW > 0:
        m_i_init = -1.0e30
    else:
        m_i_init = float("-inf")

    m_i = gl.full(
        [BLOCK_M], m_i_init, dtype=gl.float32, layout=gl.SliceLayout(1, cfg.mfma)
    )
    if IS_FP8:
        # l carries P's lift, and so does its initial 1.0 (the sink's term).
        l_init = gl.exp2(_FP8_P_LIFT)
    else:
        l_init = 1.0
    l_i = gl.full(
        [BLOCK_M], l_init, dtype=gl.float32, layout=gl.SliceLayout(1, cfg.mfma)
    )
    acc = gl.zeros([BLOCK_M, ACC_WIDTH], dtype=gl.float32, layout=cfg.mfma)
    if D_HI_PV > 0:
        acc2 = gl.zeros([BLOCK_M, D_HI_PV], dtype=gl.float32, layout=cfg.mfma)
    else:
        acc2 = acc  # never touched without a split P@V

    offs_m = start_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, cfg.mfma)
    )

    # Lowest key each row may attend; like the causal mask, bottom-right aligned.
    if SLIDING_WINDOW > 0:
        window_min = offs_m + (seqlen_k - seqlen_q - SLIDING_WINDOW)
    else:
        window_min = None

    if softmax_lse_ptr is not None:
        lse_base = (
            softmax_lse_ptr
            + off_z * stride_lse_z
            + off_q_head * stride_lse_h
            + cu_seqlens_q_start * stride_lse_m
            + start_m * BLOCK_M * stride_lse_m
        )
        offs_lse = (
            gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.mfma)) * stride_lse_m
        ).to(gl.int32)
        lse_mask = offs_m < seqlen_q
    else:
        lse_base = None
        offs_lse = None
        lse_mask = None

    if s_dmask_ptr is not None:
        sd_base = (
            s_dmask_ptr
            + off_z * stride_sd_z
            + off_q_head * stride_sd_h
            + start_m * BLOCK_M * stride_sd_m
        )
        offs_sd_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.mfma))
        offs_sd_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, cfg.mfma))
        sd_offsets = (
            offs_sd_m[:, None] * stride_sd_m + offs_sd_n[None, :] * stride_sd_n
        ).to(gl.int32)
        sd = ScoresOut(sd_base, sd_offsets, offs_m < seqlen_q, stride_sd_n)
    else:
        sd = ScoresOut(None, None, None, stride_sd_n)

    # The K loop's register allocation shifts with where the epilogue's scalars are
    # computed -- measured at 0.3-1.2% of kernel time -- so this pointer, the empty
    # causal block's store and the causal indices below each keep the form their
    # dtype's loop was tuned with, although the two forms are equivalent.
    if IS_FP8:
        o_base = (
            o_ptr
            + off_z * stride_oz
            + off_q_head * stride_oh
            + cu_seqlens_q_start * stride_om
            + start_m * BLOCK_M * stride_om
        )

    (
        n_blocks,
        n_extra_tokens,
        skipped_blocks,
        window_blocks,
        masked_blocks,
        n_full_blocks,
    ) = _classify_k_blocks(
        start_m, seqlen_q, seqlen_k, BLOCK_M, BLOCK_N, IS_CAUSAL, SLIDING_WINDOW
    )
    if IS_CAUSAL:  # noqa: SIM102
        if n_blocks <= 0:
            # A causal M-block that sees no key: zero output, zero LSE.
            if IS_FP8:
                # Built in _store_out's layout: converting from the MFMA layout
                # would stage an fp32 tile through LDS beside the K/V buffers.
                zeros = gl.zeros(
                    [BLOCK_M, BLOCK_DMODEL_POW2],
                    dtype=gl.float32,
                    layout=_make_load_layout(
                        BLOCK_DMODEL_POW2, STORE_VEC, num_warps, False
                    ),
                )
                _store_out(
                    zeros,
                    o_base,
                    start_m * BLOCK_M,
                    seqlen_q,
                    stride_om,
                    stride_on,
                    0,
                    BLOCK_DMODEL_POW2,
                    BLOCK_DMODEL_OUT,
                    BLOCK_M,
                    STORE_VEC,
                    num_warps,
                )
            else:  # see the note at o_base
                storeLayout: gl.constexpr = _make_load_layout(
                    BLOCK_DMODEL_POW2, STORE_VEC, num_warps, transposed=False
                )
                offs_od = gl.arange(
                    0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, storeLayout)
                )
                offs_rm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, storeLayout))
                offs_om = start_m * BLOCK_M + offs_rm
                o_base = (
                    o_ptr
                    + off_z * stride_oz
                    + off_q_head * stride_oh
                    + cu_seqlens_q_start * stride_om
                    + start_m * BLOCK_M * stride_om
                )
                o_offsets = (
                    offs_rm[:, None] * stride_om + offs_od[None, :] * stride_on
                ).to(gl.int32)
                zeros = gl.zeros(
                    [BLOCK_M, BLOCK_DMODEL_POW2],
                    dtype=o_ptr.dtype.element_ty,
                    layout=storeLayout,
                )
                o_mask = offs_om[:, None] < seqlen_q
                if PADDED_HEAD_OUT:
                    o_mask = o_mask & (offs_od[None, :] < BLOCK_DMODEL_OUT)
                gl.amd.cdna4.buffer_store(
                    zeros, ptr=o_base, offsets=o_offsets, mask=o_mask
                )

            if softmax_lse_ptr is not None:
                lse = gl.zeros(
                    [BLOCK_M], dtype=gl.float32, layout=gl.SliceLayout(1, cfg.mfma)
                )
                gl.amd.cdna4.buffer_store(
                    lse, ptr=lse_base, offsets=offs_lse, mask=lse_mask
                )
            return

    km = KeyMask(offs_m, seqlen_q, seqlen_k, n_extra_tokens, window_min)
    block_min = skipped_blocks * BLOCK_N
    block_max = n_blocks * BLOCK_N

    # Window-edge blocks.
    if SLIDING_WINDOW > 0:
        k_base += skipped_blocks * BLOCK_N * stride_kn
        v_base += skipped_blocks * BLOCK_N * stride_vn
        if window_blocks > 0:
            acc, acc2, l_i, m_i = _attn_fwd_inner(
                cfg,
                kv,
                sd,
                km,
                acc,
                acc2,
                l_i,
                m_i,
                q,
                q_pe,
                k_base,
                v_base,
                block_min,
                block_min + window_blocks * BLOCK_N,
                qk_scale,
                WINDOW_MASK=True,
            )
            k_base += window_blocks * BLOCK_N * stride_kn
            v_base += window_blocks * BLOCK_N * stride_vn
            block_min = block_min + window_blocks * BLOCK_N

    # Full blocks take the pipeline when there are enough of them -- checked per
    # workgroup, since causal M-blocks see different counts.  Not with returned
    # scores, and the PE slice only with 8 waves: its extra MFMA chain in dot1 needs
    # a second wave of this workgroup to overlap with (at 4 waves, PE 192/128, the
    # pipeline measured slower than the one-tile loop).
    PE_IN_PIPELINE: gl.constexpr = num_warps >= 8
    FAST_PATH: gl.constexpr = (not RETURN_SCORES) and (PE_IN_PIPELINE or not cfg.HAS_PE)

    pipelined = False
    if FAST_PATH:  # noqa: SIM102
        if n_full_blocks >= _MIN_PIPE_BLOCKS:
            pipelined = True
            acc, acc2, l_i, m_i = _attn_fwd_pipelined(
                cfg,
                kv,
                acc,
                acc2,
                l_i,
                m_i,
                q,
                q_pe,
                k_base,
                v_base,
                n_full_blocks,
                qk_scale,
            )
            block_min = block_min + n_full_blocks * BLOCK_N
            block_max = n_blocks * BLOCK_N
            gl.barrier()  # WAR: the pipeline's last LDS reads vs. the next copies

    if (not pipelined) and n_full_blocks > 0:
        block_max = block_min + n_full_blocks * BLOCK_N
        acc, acc2, l_i, m_i = _attn_fwd_inner(
            cfg,
            kv,
            sd,
            km,
            acc,
            acc2,
            l_i,
            m_i,
            q,
            q_pe,
            k_base,
            v_base,
            block_min,
            block_max,
            qk_scale,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    # Masked blocks: the causal diagonal and the ragged end of K.
    if masked_blocks > 0:
        k_base += n_full_blocks * BLOCK_N * stride_kn
        v_base += n_full_blocks * BLOCK_N * stride_vn
        acc, acc2, l_i, m_i = _attn_fwd_inner(
            cfg,
            kv,
            sd,
            km,
            acc,
            acc2,
            l_i,
            m_i,
            q,
            q_pe,
            k_base,
            v_base,
            block_min,
            block_max,
            qk_scale,
            WINDOW_MASK=SLIDING_WINDOW > 0,
            IS_CAUSAL=IS_CAUSAL,
            MASK_STEPS=True,
        )

    # Normalize: one reciprocal per row, then a multiply (a full IEEE divide is
    # several VALU ops per element).  fp8: the lift cancels in acc / l, and
    # descale_v is applied here, once.
    if IS_FP8:
        l_recip = descale_v / l_i
    else:
        l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]
    if D_HI_PV > 0:
        acc2 = acc2 * l_recip[:, None]

    # Rows before the causal start (seqlen_q > seqlen_k) saw only -inf scores and
    # hold NaN; zero them.  (Two equivalent orders: see the note at o_base.)
    if IS_FP8:
        causal_start_idx = seqlen_q - seqlen_k
        start_m_idx = start_m * BLOCK_M
        end_m_idx = start_m_idx + BLOCK_M
    else:
        end_m_idx = (start_m + 1) * BLOCK_M
        start_m_idx = start_m * BLOCK_M
        causal_start_idx = seqlen_q - seqlen_k
    if IS_CAUSAL:  # noqa: SIM102
        if (causal_start_idx > start_m_idx) and (causal_start_idx < end_m_idx):
            row_ok = (offs_m >= causal_start_idx)[:, None]
            acc = gl.where(row_ok, acc, 0.0)
            if D_HI_PV > 0:
                acc2 = gl.where(row_ok, acc2, 0.0)

    if softmax_lse_ptr is not None:
        LN2: gl.constexpr = 0.6931471824645996
        softmax_lse = m_i + gl.log2(l_i)  # base 2
        if IS_FP8:
            softmax_lse = softmax_lse - _FP8_P_LIFT
        softmax_lse = softmax_lse * LN2
        if IS_CAUSAL:
            softmax_lse = gl.where(offs_m < causal_start_idx, 0.0, softmax_lse)
        gl.amd.cdna4.buffer_store(
            softmax_lse, ptr=lse_base, offsets=offs_lse, mask=lse_mask
        )

    if IS_FP8:
        # Stored straight from the MFMA layout (4 consecutive columns per lane, a
        # 16 B store): converting would stage the fp32 tile through LDS and cap a CU
        # at one workgroup.
        offs_od = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, cfg.mfma))
        o_offsets = (
            (offs_m - start_m_idx)[:, None] * stride_om + offs_od[None, :] * stride_on
        ).to(gl.int32)
        o_mask = (offs_m < seqlen_q)[:, None]
        if PADDED_HEAD_OUT:
            o_mask = o_mask & (offs_od[None, :] < BLOCK_DMODEL_OUT)
        gl.amd.cdna4.buffer_store(
            acc.to(o_ptr.dtype.element_ty), ptr=o_base, offsets=o_offsets, mask=o_mask
        )
    elif D_HI_PV > 0:
        # A split head stores its two slices side by side.
        o_base = (
            o_ptr
            + off_z * stride_oz
            + off_q_head * stride_oh
            + cu_seqlens_q_start * stride_om
            + start_m * BLOCK_M * stride_om
        )
        _store_out(
            acc,
            o_base,
            start_m_idx,
            seqlen_q,
            stride_om,
            stride_on,
            0,
            D_LO,
            BLOCK_DMODEL_OUT,
            BLOCK_M,
            STORE_VEC,
            num_warps,
        )
        _store_out(
            acc2,
            o_base,
            start_m_idx,
            seqlen_q,
            stride_om,
            stride_on,
            D_LO,
            D_HI_PV,
            BLOCK_DMODEL_OUT,
            BLOCK_M,
            STORE_VEC,
            num_warps,
        )
    else:
        out = acc.to(o_ptr.dtype.element_ty)

        storeLayout: gl.constexpr = _make_load_layout(
            BLOCK_DMODEL_POW2, STORE_VEC, num_warps, transposed=False
        )
        out = gl.convert_layout(out, layout=storeLayout)

        offs_rm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, storeLayout))
        offs_om = start_m * BLOCK_M + offs_rm
        offs_od = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, storeLayout))

        o_base = (
            o_ptr
            + off_z * stride_oz
            + off_q_head * stride_oh
            + cu_seqlens_q_start * stride_om
            + start_m * BLOCK_M * stride_om
        )
        o_offsets = (offs_rm[:, None] * stride_om + offs_od[None, :] * stride_on).to(
            gl.int32
        )

        overflow_size = end_m_idx - seqlen_q
        out_mask = gl.full([BLOCK_M, 1], True, dtype=gl.int1, layout=storeLayout)
        if overflow_size > 0:
            out_mask = out_mask & (offs_om[:, None] < seqlen_q)
        if PADDED_HEAD_OUT:
            out_mask = out_mask & (offs_od[None, :] < BLOCK_DMODEL_OUT)
        gl.amd.cdna4.buffer_store(out, ptr=o_base, offsets=o_offsets, mask=out_mask)


# ---------------------------------------------------------------------------
# 6. Host-side config
# ---------------------------------------------------------------------------


def _get_config(
    is_fp8: bool,
    has_pe: bool = False,
    causal: bool = False,
    v_head_dim: int = 0,
    sliding_window: int = 0,
    seqlen_k: int = 0,
):
    """Tile / wave configuration for one masking + dtype mode.

    `BLOCK_M = 32 * num_warps`.  The wide tile (256 / 8 waves) is the default: two
    waves per SIMD overlap softmax with MFMA.  Causal attention below
    `causal_wide_min_seqlen_k` keys, and a causal sliding window, take the narrow
    tile (128 / 4 waves), which wastes less on the masked diagonal.
    """
    arch = arch_info.get_arch()
    fpath = f"{AITER_TRITON_CONFIGS_PATH}/{arch}/gluon/attention/mha/mha.json"
    fwd_cfg = load_config_json(fpath)["fwd"]
    if is_fp8:
        # A 64-wide head doubles BLOCK_N to amortise the softmax.  Only an exact
        # 64: padded heads (56, 32) spill at that tile.
        if v_head_dim == 64 and "fp8_small_head" in fwd_cfg:
            return fwd_cfg["fp8_small_head"]
        return fwd_cfg["fp8"]
    elif has_pe:
        # The wide tile fits only a V head <= 64; a wider one spills.
        if v_head_dim and v_head_dim <= 64 and "pe_narrow_v" in fwd_cfg:
            return fwd_cfg["pe_narrow_v"]
        return fwd_cfg["pe"]
    elif v_head_dim > 128 and "wide_head" in fwd_cfg:
        # Past 128 the default tile spills.
        return fwd_cfg["wide_head"]
    elif causal and sliding_window > 0 and "causal_window" in fwd_cfg:
        return fwd_cfg["causal_window"]
    elif (
        causal
        and "causal_short" in fwd_cfg
        and seqlen_k < fwd_cfg["causal_wide_min_seqlen_k"]
    ):
        return fwd_cfg["causal_short"]
    elif causal and "causal" in fwd_cfg:
        return fwd_cfg["causal"]
    else:
        return fwd_cfg["default"]
