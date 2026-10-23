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


from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import remap_xcd
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import (
    AITER_TRITON_CONFIGS_PATH,
    load_config_json,
)

# Padding interval, in elements, of the K/V shared layouts. 512 elements = 1 KB of
# 16-bit data, the granularity the AMD Triton backend itself pads at.
_PAD_INTERVAL = 512


@gluon.constexpr_function
def _bits(n):
    """log2 of a power of two."""
    return n.bit_length() - 1


@gluon.constexpr_function
def _contig_strided(rows, cols, contig_dim):
    """(extent along the contiguous axis, extent along the strided axis)."""
    return (rows, cols) if contig_dim == 0 else (cols, rows)


@gluon.constexpr_function
def _padded_staggered_layout(rows, cols, contig_dim, pad, interval=_PAD_INTERVAL):
    """Row-staggered padded shared layout for a K^T or V staging tile.

    Both tiles are written along their contiguous axis but read back by the MFMA
    across the other one, which is bank-conflict free only when consecutive
    strided-axis entries land far apart in LDS. The offset bases are therefore built
    by hand: the contiguous axis in the low bits, then the *high* bits of the strided
    index, then its low bits -- that middle group is the stagger. `pad` differs per
    tile because the reads do: K^T is read along its contiguous axis
    (`ds_read_b128`) while V is read across it by the hardware-transposing
    `ds_read_b64_tr_b16`.
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
def _staggered_layout_ok(head_dim_pow2, block_n, elem_bytes, interval=_PAD_INTERVAL):
    """Can the staggered layout be built for this tile pair?

    The bases above enumerate every bit of both axes exactly once, which needs both
    extents to be powers of two, and the strided axis has to split evenly across the
    padding intervals.
    """
    if elem_bytes != 2:
        return False
    if head_dim_pow2 & (head_dim_pow2 - 1) or block_n & (block_n - 1):
        return False
    if head_dim_pow2 < 16 or block_n < 16:
        return False
    # K^T is [head_dim, block_n] with dim 0 contiguous and V is its transpose, so
    # both tiles have the head dim on the contiguous axis and BLOCK_N on the strided
    # one.  The strided axis has to split evenly across the padding intervals.
    return block_n % max(interval // head_dim_pow2, 1) == 0


@gluon.constexpr_function
def _make_kv_shared_layouts(head_dim_pow2, block_n):
    """LDS layouts for the K^T / V staging tiles: the conflict-free row-staggered
    padded layout, which is also what lets the async copy write them coalesced.
    `kv_async_copy_ok` (via `_staggered_layout_ok`) guarantees it can be built."""
    return (
        _padded_staggered_layout(head_dim_pow2, block_n, 0, pad=8),
        _padded_staggered_layout(block_n, head_dim_pow2, 1, pad=32),
    )


# Bits per lane the global->LDS copy's lowering accepts, widest (and fastest) first.
# 128 is a `*_dwordx4` per lane; 32 is the plain-`dword` form, four times the
# instructions per tile but with a quarter of the per-lane footprint and alignment
# requirement.
_ASYNC_COPY_BITS = (128, 32)


@gluon.constexpr_function
def _async_copy_lane_rows(rows, cols, contig_dim, interval):
    """(first strided-axis bit the lanes take, strided-axis bits available to them).

    The copy's lanes have to write consecutive LDS addresses.  The staggered
    destination (`_padded_staggered_layout`) stores the contiguous axis in its low
    bits and, right above it, the strided axis starting at bit `log2(split)` -- so
    that is where the lanes' strided bits must start, and they can run up to the top
    of the axis.
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
    interval=_PAD_INTERVAL,
    warp_size=64,
):
    """Elements per lane a global->LDS async copy can move for this tile, or 0 if none can.

    `buffer_load_to_shared` wants every lane to carry exactly 128 or 32 bits, those
    bits aligned to their own width in global memory (`stride_align`, in elements,
    is what the host verified), and the resulting LDS writes have to coalesce against
    the padded destination.  The widest width that satisfies all three wins; the
    32-bit one is what keeps small tiles (head dim 16/32) and strides that are even
    but not 8-element aligned on the copy.  0 means the tile cannot be copied at
    all, and `kv_async_copy_ok` turns that into a host-side `.contiguous()`.
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
def _async_copy_layout(
    rows, cols, contig_dim, num_warps, vec, interval=_PAD_INTERVAL, warp_size=64
):
    """Address layout for a global->LDS async copy of a `[rows, cols]` tile.

    `buffer_load_to_shared` needs each lane to name `vec` *contiguous* elements
    and needs the resulting LDS writes to coalesce. The low `log2(vec)` bits of the
    contiguous axis go to registers and the rest of that axis to lanes; the lanes
    that axis leaves over take the strided bits the staggered destination stores
    next (`_async_copy_lane_rows`).  The remaining strided bits split low-to-high
    across warps, then registers.
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
    """Can BOTH the K^T and the V tile be moved by `buffer_load_to_shared`?

    `stride_align` is the largest power of two, in elements, that the host has
    verified divides every K/V stride -- and 0 when it could not verify anything,
    which is what the host reports for a tensor whose last axis is not contiguous.
    It has to reach the copy's vector width, because that is what makes every lane's
    chunk aligned to its own size; see the note at the `gl.multiple_of` call.
    Below it the copy would be wrong, so 0 disqualifies.
    """
    if not _staggered_layout_ok(head_dim_pow2, block_n, elem_bits // 8):
        return False
    vec = _async_copy_vec(head_dim_pow2, block_n, 0, num_warps, elem_bits, stride_align)
    if vec == 0:
        return False
    # K^T is [head_dim, block_n] with dim 0 contiguous; V is its transpose.  Both
    # tiles share one vector width (`KV_VEC` at the call sites).
    return (
        _async_copy_vec(block_n, head_dim_pow2, 1, num_warps, elem_bits, stride_align)
        == vec
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


@gluon.jit
def _async_copy_tile(smem, base, offsets, mask, HAS_MASK: gl.constexpr):
    """Issue one global->LDS async copy tile, WITHOUT closing a commit group.

    The copy never passes through VGPRs, so it needs no `s_waitcnt vmcnt(0)` and
    no `ds_write`; the wave issues it and walks away.

    A masked lane is zero-filled by the hardware: the lowering turns its offset
    out of bounds, and an out-of-bounds buffer load returns 0, which the copy
    writes to LDS like any other value.  Do NOT pass `other`: that sends the
    lowering down its general path, an exec-mask branch around every copy
    instruction plus a `ds_write` of the fill value, and on a padded head the
    branch predicates stay live across the loop, spill, and their reloads land in
    front of the copies as `s_waitcnt vmcnt(0)` -- draining every copy in flight.
    """
    if HAS_MASK:
        cdna4_async.buffer_load_to_shared(smem, base, offsets, mask=mask)
    else:
        cdna4_async.buffer_load_to_shared(smem, base, offsets)


@gluon.jit
def _async_copy_kv_tile(
    smemK,
    smemKpe,
    smemV,
    slot,
    k_base,
    v_base,
    kt_off,
    kpe_off,
    v_off,
    kt_off_n,
    kt_off_d,
    kpe_off_n,
    v_off_n,
    v_off_d,
    start_n,
    seqlen_k,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,
    MASK_STEPS: gl.constexpr,
    PADDED_HEAD: gl.constexpr,
    HAS_PE: gl.constexpr,
):
    """Stage one K (+PE) and V tile into LDS slot `slot` with buffer_load_to_shared.

    The KV-token mask is only built on the masked blocks; the head-dim mask is a
    property of the tensor, so it rides along on every block when the head dim is
    padded. `buffer_load_to_shared` broadcasts a mask against the offsets, so each
    half is handed over carrying only the axis it constrains.

    The PE slice is a third tile with its own LDS buffer but it shares K's commit
    group, so `ASYNC_GROUPS` in the caller stays 2 with or without it.
    """
    HAS_MASK: gl.constexpr = MASK_STEPS or PADDED_HEAD
    if MASK_STEPS:
        k_mask = (start_n + kt_off_n)[None, :] < seqlen_k
        v_mask = (start_n + v_off_n)[:, None] < seqlen_k
        if PADDED_HEAD:
            k_mask = k_mask & (kt_off_d[:, None] < BLOCK_DMODEL)
            v_mask = v_mask & (v_off_d[None, :] < BLOCK_DMODEL)
    elif PADDED_HEAD:
        k_mask = kt_off_d[:, None] < BLOCK_DMODEL
        v_mask = v_off_d[None, :] < BLOCK_DMODEL
    else:
        k_mask = None
        v_mask = None

    # The PE slice has no head-dim padding of its own (the host requires unpadded
    # powers of two there), so it only ever carries the KV-token mask.
    if HAS_PE and MASK_STEPS:
        kpe_mask = (start_n + kpe_off_n)[None, :] < seqlen_k
    else:
        kpe_mask = None
    _async_copy_tile(smemK.index(slot), k_base, kt_off, k_mask, HAS_MASK)
    if HAS_PE:
        _async_copy_tile(smemKpe.index(slot), k_base, kpe_off, kpe_mask, MASK_STEPS)
    cdna4_async.commit_group()
    _async_copy_tile(smemV.index(slot), v_base, v_off, v_mask, HAS_MASK)
    cdna4_async.commit_group()  # ACV


@gluon.jit
def _attn_qk(
    q,
    k,
    q_pe,
    k_pe,
    start_n,
    offs_n,
    window_min,
    qk_scale,
    mfmaLayout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HAS_PE: gl.constexpr,
    IS_FP8: gl.constexpr,
    WINDOW_MASK: gl.constexpr,
    SCALE_ON_Q: gl.constexpr = False,
    seqlen_q=None,
    seqlen_k=None,
    block_max=None,
    n_extra_tokens=None,
    offs_m=None,
    IS_CAUSAL: gl.constexpr = False,
    MASK_STEPS: gl.constexpr = False,
):
    """QK^T + scale + mask for one already-staged key block. `k` is already in
    its MFMA dot-operand layout; returns float32 scores in `mfmaLayout`. For
    FP8 the QK^T uses the CDNA4 scaled MFMA (32x32x64).
    """
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfmaLayout)
    if IS_FP8:
        if HAS_PE:
            qk = gl.amd.cdna4.mfma_scaled(q_pe, None, "e4m3", k_pe, None, "e4m3", qk)
        qk = gl.amd.cdna4.mfma_scaled(q, None, "e4m3", k, None, "e4m3", qk)
    else:
        if HAS_PE:
            qk = gl.amd.cdna4.mfma(q_pe, k_pe, qk)
        qk = gl.amd.cdna4.mfma(q, k, qk)

    if not SCALE_ON_Q:
        qk = qk * qk_scale

    if MASK_STEPS or IS_CAUSAL or WINDOW_MASK:
        key_pos = start_n + offs_n
        mask = gl.full([BLOCK_M, BLOCK_N], True, dtype=gl.int1, layout=mfmaLayout)
        if MASK_STEPS:
            # Only the last visible block can be partial (seqlen_k not a multiple
            # of BLOCK_N).
            bound_cond = (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0)
            mask_partial = key_pos[None, :] < seqlen_k
            mask = gl.where(bound_cond, mask_partial, mask)
        if IS_CAUSAL:
            causal_boundary = key_pos + (seqlen_q - seqlen_k)
            mask = mask & (offs_m[:, None] >= causal_boundary[None, :])
        if WINDOW_MASK:
            mask = mask & (window_min[:, None] <= key_pos[None, :])
        qk = gl.where(mask, qk, float("-inf"))

    return qk


@gluon.jit
def _attn_softmax_pv(
    acc,
    acc2,
    l_i,
    m_i,
    qk,
    v,
    v_hi,
    sd_base,
    sd_offsets,
    sd_q_mask,
    offs_n,
    start_n,
    seqlen_k,
    stride_sd_n,
    dotP: gl.constexpr,
    RETURN_SCORES: gl.constexpr,
    D_HI_PV: gl.constexpr,
):
    """Online-softmax rescale + P@V accumulation for one key block, over both
    accumulator slices.  Returns updated (acc, acc2, l_i, m_i)."""

    m_ij = gl.maximum(m_i, gl.max(qk, 1))
    p = gl.exp2(qk - m_ij[:, None])
    alpha = gl.exp2(m_i - m_ij)
    l_ij = gl.sum(p, 1)

    if RETURN_SCORES:
        # NOTE: the returned score is not the same as the reference because we
        # need to adjust as we find new maxes per block. We are not doing that
        p_mask = sd_q_mask[:, None] & ((start_n + offs_n)[None, :] < seqlen_k)
        gl.amd.cdna4.buffer_store(
            p.to(sd_base.dtype.element_ty),
            ptr=sd_base + start_n * stride_sd_n,
            offsets=sd_offsets,
            mask=p_mask,
        )

    acc = acc * alpha[:, None]
    if D_HI_PV > 0:
        acc2 = acc2 * alpha[:, None]
    p = gl.convert_layout(p.to(v.dtype), layout=dotP, assert_trivial=True)
    acc, acc2 = _dot_pv(p, v, v_hi, acc, acc2, D_HI_PV, False)

    l_i = l_i * alpha + l_ij
    m_i = m_ij

    return acc, acc2, l_i, m_i


# ---------------------------------------------------------------------------
# The rotated four-cluster pipeline (dense bf16/fp16 path)
#
# The generic loop above runs one tile at a time: stage it, read it, do both
# MFMA chains and the whole softmax, repeat.  Every wave on a SIMD is then in the
# same phase, so nothing covers anything.
#
# This path cuts one tile of work into four clusters that alternate matrix and
# memory, and runs the two waves of a SIMD one cluster apart.  A wave doing MFMAs
# always faces a wave doing loads, and -- because VALU and memory are separate issue
# ports -- the softmax rides in the MFMA's shadow instead of after it.
#
#   dot1   Q@K^T for tile j+1     VEC2 for tile j    (rescale, row sum, P downcast)
#   mem1   read V[j] from LDS     async-copy K[j+3] -> LDS
#   dot2   P@V for tile j         VEC1 for tile j+1  (row max, exp2 burst)
#   mem2   read K[j+2] from LDS   async-copy V[j+2] -> LDS
#
# Four pipeline stages are in flight at once, which is why the async-copy indices run up
# to three tiles ahead.  See kernels/attention/README.md of the gfx950 Gluon
# tutorial for the derivation.
# ---------------------------------------------------------------------------

# The pipeline has a three-tile prologue and a three-tile drain.  Below this many
# full tiles the two overlap, and the warp-pipeline stage barriers no longer
# separate an LDS slot's read from the async copy that overwrites it.  This is a
# correctness bound on the schedule, not a performance threshold.
_MIN_PIPE_BLOCKS = gl.constexpr(8)


@gluon.constexpr_function
def head_split(head_dim, head_dim_pow2, min_hi):
    """Width of the upper slice when a padded head dim is issued as
    `[0, head_dim_pow2 / 2)` plus `[head_dim_pow2 / 2, +hi)`; 0 when that saves
    nothing over the single padded tile.  Also called by the host to pick this
    kernel.

    `min_hi` is the narrowest slice the MFMA can take on that side of the product:
    16 on the Q@K^T contraction, 32 on the P@V output (the 32x32 instruction's N).
    """
    if head_dim == head_dim_pow2:
        return 0
    lo = head_dim_pow2 // 2
    hi = max(1 << (head_dim - lo - 1).bit_length(), min_hi)
    return hi if lo + hi < head_dim_pow2 else 0


@gluon.constexpr_function
def _head_slices(head_dim, head_dim_pow2, has_pe):
    """`(D_LO, D_HI_QK, D_HI_PV)` -- how `_attn_fwd` issues the head dim.

    A padded head dim run as one power-of-two tile spends part of every MFMA on zero
    padding (a quarter of it at 96 -> 128).  When it saves work, the head dim is
    issued as `[0, pow2 / 2)` plus a narrow power-of-two upper slice instead,
    independently on each side of the product:

    * Q@K^T contraction -- the upper slice is a second MFMA chain into the same
      score accumulator.  Never with PE: the PE slice is that side's second chain
      already, and it requires an unpadded head anyway.
    * P@V output -- the upper slice is a second accumulator, `acc2`.  Only when
      the contraction side also splits; 40 and 48 split there but not here, their
      upper slice being narrower than the 32x32 MFMA's N.

    K and V stay one padded LDS tile each; a slice is a `.slice()` sub-view of it.
    With no split, `D_LO` is the whole padded width and both upper slices are 0.
    """
    hi_qk = 0 if has_pe else head_split(head_dim, head_dim_pow2, 16)
    if hi_qk == 0:
        return head_dim_pow2, 0, 0
    return head_dim_pow2 // 2, hi_qk, head_split(head_dim, head_dim_pow2, 32)


@gluon.jit
def _read_k(
    smemK,
    smemKpe,
    slot,
    dotK: gl.constexpr,
    D_LO: gl.constexpr,
    D_HI_QK: gl.constexpr,
    HAS_PE: gl.constexpr,
):
    """K^T of one staged tile, and its PE operand `k_pe`.

    `k_pe` is the PE tile -- or, when the head is split (`_head_slices`), the upper
    head slice, which takes the PE operand's place in the Q@K^T chain.  With
    neither, it aliases K: a loop-carried value has to be a real tensor in every
    instantiation."""
    if D_HI_QK > 0:
        k = cdna4_async.load_shared_relaxed(smemK.index(slot).slice(0, D_LO, 0), dotK)
        k_pe = cdna4_async.load_shared_relaxed(
            smemK.index(slot).slice(D_LO, D_HI_QK, 0), dotK
        )
    else:
        k = cdna4_async.load_shared_relaxed(smemK.index(slot), dotK)
        if HAS_PE:
            k_pe = cdna4_async.load_shared_relaxed(smemKpe.index(slot), dotK)
        else:
            k_pe = k
    return k, k_pe


@gluon.jit
def _read_v(smemV, slot, dotV: gl.constexpr, D_LO: gl.constexpr, D_HI_PV: gl.constexpr):
    """V of one staged tile: `(v, v_hi)` when the P@V side is split, else the
    whole padded tile and an alias of it."""
    if D_HI_PV > 0:
        v = cdna4_async.load_shared_relaxed(smemV.index(slot).slice(0, D_LO, 1), dotV)
        v_hi = cdna4_async.load_shared_relaxed(
            smemV.index(slot).slice(D_LO, D_HI_PV, 1), dotV
        )
    else:
        v = cdna4_async.load_shared_relaxed(smemV.index(slot), dotV)
        v_hi = v
    return v, v_hi


@gluon.jit
def _mma(a, b, acc, IS_FP8: gl.constexpr):
    """One MFMA chain step: bf16/fp16 `mfma`, or e4m3 `mfma_scaled` (unscaled)."""
    if IS_FP8:
        return gl.amd.cdna4.mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)
    else:
        return gl.amd.cdna4.mfma(a, b, acc)


@gluon.jit
def _sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q: gl.constexpr, P_LIFT: gl.constexpr = 0.0):
    """VEC1 -- softmax numerator: new row max, the exp2 burst, and alpha.

    Placed in the `dot2` cluster.  `exp2` is a TRANS op and issues at half the
    rate of a plain VALU, so it is the most expensive item in the softmax and wants
    the roomier of the two shadows.  Its results are consumed one stage later.
    """
    if SCALE_ON_Q:
        # qk already carries the scale (folded into Q once, before the loop), so the
        # row max needs no multiply and the exponent argument is a plain subtract.
        m_new = gl.maximum(m_run, gl.max(qk, 1))
        p = gl.exp2(qk - m_new[:, None])
    else:
        m_new = gl.maximum(m_run, gl.max(qk, 1) * qk_scale)
        # Fused at the source (one llvm.fmuladd) rather than left as an fmul/fsub
        # pair for the backend to contract after scheduling has already counted it.
        if P_LIFT != 0.0:
            # fp8: P scaled onto e4m3's range (see _FP8_P_LIFT); rides the addend.
            p = gl.exp2(gl.fma(qk, qk_scale, P_LIFT - m_new[:, None]))
        else:
            p = gl.exp2(gl.fma(qk, qk_scale, -m_new[:, None]))
    alpha = gl.exp2(m_run - m_new)
    return m_new, p, alpha


@gluon.jit
def _sc_vec2(
    acc,
    acc2,
    l_i,
    p,
    alpha,
    dotP: gl.constexpr,
    DTYPE: gl.constexpr,
    D_HI_PV: gl.constexpr,
):
    """VEC2 -- denominator, accumulator rescale, and the P operand downcast.

    Placed in the `dot1` cluster, beside the Q@K^T MFMA; `p` and `alpha` were
    produced by VEC1 in the *previous* iteration.  The rescale goes first: this
    cluster's shadow cannot absorb all of it, and leading with it keeps the uncovered
    remainder ahead of the MFMAs, where an exposed packed op pays no back-to-back
    hazard.
    """
    acc = acc * alpha[:, None]
    if D_HI_PV > 0:
        acc2 = acc2 * alpha[:, None]
    l_ij = gl.sum(p, axis=1)
    l_i = l_i * alpha + l_ij
    p_dot = gl.convert_layout(p.to(DTYPE), dotP)
    return acc, acc2, l_i, p_dot


@gluon.jit
def _dot_qk(
    q,
    q_pe,
    k,
    k_pe,
    HAS_PE_DOT: gl.constexpr,
    IS_FP8: gl.constexpr,
    mfmaLayout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """Q@K^T for one tile, the PE chain first.  `q_pe` / `k_pe` may be the upper
    head slice of a split head (`_read_k`); `HAS_PE_DOT` says either is present."""
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfmaLayout)
    if HAS_PE_DOT:
        qk = _mma(q_pe, k_pe, qk, IS_FP8)
    return _mma(q, k, qk, IS_FP8)


@gluon.jit
def _dot_pv(p_dot, v, v_hi, acc, acc2, D_HI_PV: gl.constexpr, IS_FP8: gl.constexpr):
    """P@V for one tile, into `acc` and, when the P@V side is split, `acc2`."""
    acc = _mma(p_dot, v, acc, IS_FP8)
    if D_HI_PV > 0:
        acc2 = _mma(p_dot, v_hi, acc2, IS_FP8)
    return acc, acc2


@gluon.jit
def _attn_fwd_pipelined(
    acc,
    acc2,
    l_i,
    m_i,
    q_dot,
    q_pe,
    smemK,
    smemKpe,
    smemV,
    k_base,
    v_base,
    kt_off,
    kpe_off,
    v_off,
    k_mask,
    v_mask,
    kt_step,
    v_step,
    n_full_blocks,
    qk_scale,
    mfmaLayout: gl.constexpr,
    dotK: gl.constexpr,
    dotV: gl.constexpr,
    dotP: gl.constexpr,
    SCALE_ON_Q: gl.constexpr,
    DTYPE: gl.constexpr,
    HAS_MASK: gl.constexpr,
    HAS_PE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BUF_DEPTH: gl.constexpr,
    IS_FP8: gl.constexpr = False,
    P_LIFT: gl.constexpr = 0.0,
    D_LO: gl.constexpr = 0,
    D_HI_QK: gl.constexpr = 0,
    D_HI_PV: gl.constexpr = 0,
):
    """Run tiles `[0, n_full_blocks)` through the rotated pipeline.

    Also runs e4m3 (`IS_FP8`, from `_attn_fwd_fp8`): the MFMAs become
    `mfma_scaled` and P is lifted by `P_LIFT` onto e4m3's range.

    `k_base` / `v_base` point at tile 0 and every tile index is an offset from
    them, so the address VALU stays out of the loop entirely.

    `q_pe` is the Q operand of the PE chain.  On a split head (`D_HI_QK > 0`,
    `_head_slices`) it carries the upper head slice instead, as do `k_pe` /
    `kpe_dot` on the K side; `HAS_PE_DOT` says the chain is issued at all.  `acc2`
    is the upper P@V accumulator when `D_HI_PV > 0`, an alias of `acc` otherwise.
    """
    HAS_PE_DOT: gl.constexpr = HAS_PE or D_HI_QK > 0

    # -- Prologue: prime the rotation for tile 0 ---------------------------------
    # Compute all of tile 0's ahead-work (qk[0], m[0], p[0], alpha[0]) and the K
    # registers for tile 1, and stage K[0..2] / V[0..1].  K runs three tiles ahead,
    # so slot 0 is reused for K[2] once tile 0's read is done -- hence the barrier.
    # Commit order K0, V0, K1, K2, V1 leaves {K2, V1} pending, which is the loop's
    # steady-state entry condition.  The PE slice always rides in its K tile's commit
    # group: every `wait_group` count here is expressed in commit groups, so a group
    # of its own would shift all of them, and K and PE feed the same MFMA chain, so
    # there is never a reason to wait for one without the other.
    _async_copy_tile(smemK.index(0), k_base, kt_off, k_mask, HAS_MASK)
    if HAS_PE:
        _async_copy_tile(smemKpe.index(0), k_base, kpe_off, None, False)
    cdna4_async.commit_group()
    _async_copy_tile(smemV.index(0), v_base, v_off, v_mask, HAS_MASK)
    cdna4_async.commit_group()  # ACV
    _async_copy_tile(smemK.index(1), k_base + kt_step, kt_off, k_mask, HAS_MASK)
    if HAS_PE:
        _async_copy_tile(smemKpe.index(1), k_base + kt_step, kpe_off, None, False)
    cdna4_async.commit_group()

    cdna4_async.wait_group(2)  # K[0] has landed
    kt0, kpe0 = _read_k(smemK, smemKpe, 0, dotK, D_LO, D_HI_QK, HAS_PE)
    qk = _dot_qk(
        q_dot, q_pe, kt0, kpe0, HAS_PE_DOT, IS_FP8, mfmaLayout, BLOCK_M, BLOCK_N
    )
    m_run, p_c, alpha_c = _sc_vec1(qk, m_i, qk_scale, SCALE_ON_Q, P_LIFT)

    gl.barrier()  # WAR: tile 0's ds_read against K[2]'s write into the same slot
    _async_copy_tile(smemK.index(0), k_base + 2 * kt_step, kt_off, k_mask, HAS_MASK)
    if HAS_PE:
        _async_copy_tile(smemKpe.index(0), k_base + 2 * kt_step, kpe_off, None, False)
    cdna4_async.commit_group()
    cdna4_async.wait_group(1)  # K[1] has landed
    kt_dot, kpe_dot = _read_k(smemK, smemKpe, 1, dotK, D_LO, D_HI_QK, HAS_PE)
    _async_copy_tile(smemV.index(1), v_base + v_step, v_off, v_mask, HAS_MASK)
    cdna4_async.commit_group()  # ACV

    # -- Main loop, unrolled 2x ---------------------------------------------------
    # Over a pair of tiles the K and V slots exchange places; unrolling by BUF_DEPTH
    # returns each to where it started, so both slot indices are compile-time
    # constants and the loop body carries no `% BUF_DEPTH` slot arithmetic.
    pairs = (n_full_blocks - 3) // 2
    for pair in range(pairs):
        blk = pair * 2

        # even tile (blk): LDS slots cur=0, next=1
        with warp_pipeline_stage("dot1"):
            qk = _dot_qk(
                q_dot,
                q_pe,
                kt_dot,
                kpe_dot,
                HAS_PE_DOT,
                IS_FP8,
                mfmaLayout,
                BLOCK_M,
                BLOCK_N,
            )  # dot_qk
            acc, acc2, l_i, p_dot = _sc_vec2(
                acc, acc2, l_i, p_c, alpha_c, dotP, DTYPE, D_HI_PV
            )  # VEC2
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem1"):
            v_dot, v_hi = _read_v(smemV, 0, dotV, D_LO, D_HI_PV)  # LRV
            _async_copy_tile(  # ACK
                smemK.index(1), k_base + (blk + 3) * kt_step, kt_off, k_mask, HAS_MASK
            )
            if HAS_PE:
                _async_copy_tile(
                    smemKpe.index(1), k_base + (blk + 3) * kt_step, kpe_off, None, False
                )
            cdna4_async.commit_group()
        with warp_pipeline_stage("dot2"):
            acc, acc2 = _dot_pv(
                p_dot, v_dot, v_hi, acc, acc2, D_HI_PV, IS_FP8
            )  # dot_pv
            m_run, p_c, alpha_c = _sc_vec1(
                qk, m_run, qk_scale, SCALE_ON_Q, P_LIFT
            )  # VEC1
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem2"):
            kt_dot, kpe_dot = _read_k(
                smemK, smemKpe, 0, dotK, D_LO, D_HI_QK, HAS_PE
            )  # LRK
            _async_copy_tile(  # ACV
                smemV.index(0), v_base + (blk + 2) * v_step, v_off, v_mask, HAS_MASK
            )
            cdna4_async.commit_group()

        # odd tile (blk + 1): LDS slots cur=1, next=0
        with warp_pipeline_stage("dot1"):
            qk = _dot_qk(
                q_dot,
                q_pe,
                kt_dot,
                kpe_dot,
                HAS_PE_DOT,
                IS_FP8,
                mfmaLayout,
                BLOCK_M,
                BLOCK_N,
            )  # dot_qk
            acc, acc2, l_i, p_dot = _sc_vec2(
                acc, acc2, l_i, p_c, alpha_c, dotP, DTYPE, D_HI_PV
            )  # VEC2
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem1"):
            v_dot, v_hi = _read_v(smemV, 1, dotV, D_LO, D_HI_PV)  # LRV
            _async_copy_tile(  # ACK
                smemK.index(0),
                k_base + (blk + 1 + 3) * kt_step,
                kt_off,
                k_mask,
                HAS_MASK,
            )
            if HAS_PE:
                _async_copy_tile(
                    smemKpe.index(0),
                    k_base + (blk + 1 + 3) * kt_step,
                    kpe_off,
                    None,
                    False,
                )
            cdna4_async.commit_group()
        with warp_pipeline_stage("dot2"):
            acc, acc2 = _dot_pv(
                p_dot, v_dot, v_hi, acc, acc2, D_HI_PV, IS_FP8
            )  # dot_pv
            m_run, p_c, alpha_c = _sc_vec1(
                qk, m_run, qk_scale, SCALE_ON_Q, P_LIFT
            )  # VEC1
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem2"):
            kt_dot, kpe_dot = _read_k(
                smemK, smemKpe, 1, dotK, D_LO, D_HI_QK, HAS_PE
            )  # LRK
            _async_copy_tile(  # ACV
                smemV.index(1), v_base + (blk + 1 + 2) * v_step, v_off, v_mask, HAS_MASK
            )
            cdna4_async.commit_group()

    # An odd count leaves one tile over.  It is always an "even" tile (slots 0/1),
    # because each pair returns the buffers to where they started.
    if (n_full_blocks - 3) % 2 == 1:
        blk = pairs * 2

        # tail tile: LDS slots cur=0, next=1
        with warp_pipeline_stage("dot1"):
            qk = _dot_qk(
                q_dot,
                q_pe,
                kt_dot,
                kpe_dot,
                HAS_PE_DOT,
                IS_FP8,
                mfmaLayout,
                BLOCK_M,
                BLOCK_N,
            )  # dot_qk
            acc, acc2, l_i, p_dot = _sc_vec2(
                acc, acc2, l_i, p_c, alpha_c, dotP, DTYPE, D_HI_PV
            )  # VEC2
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem1"):
            v_dot, v_hi = _read_v(smemV, 0, dotV, D_LO, D_HI_PV)  # LRV
            _async_copy_tile(  # ACK
                smemK.index(1), k_base + (blk + 3) * kt_step, kt_off, k_mask, HAS_MASK
            )
            if HAS_PE:
                _async_copy_tile(
                    smemKpe.index(1), k_base + (blk + 3) * kt_step, kpe_off, None, False
                )
            cdna4_async.commit_group()
        with warp_pipeline_stage("dot2"):
            acc, acc2 = _dot_pv(
                p_dot, v_dot, v_hi, acc, acc2, D_HI_PV, IS_FP8
            )  # dot_pv
            m_run, p_c, alpha_c = _sc_vec1(
                qk, m_run, qk_scale, SCALE_ON_Q, P_LIFT
            )  # VEC1
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem2"):
            kt_dot, kpe_dot = _read_k(
                smemK, smemKpe, 0, dotK, D_LO, D_HI_QK, HAS_PE
            )  # LRK
            _async_copy_tile(  # ACV
                smemV.index(0), v_base + (blk + 2) * v_step, v_off, v_mask, HAS_MASK
            )
            cdna4_async.commit_group()

    # -- Drain: the last three tiles, with no prefetch left to issue ---------------
    nm3 = n_full_blocks - 3
    nm2 = n_full_blocks - 2
    nm1 = n_full_blocks - 1
    s_nm3 = (nm3 % BUF_DEPTH).to(gl.int32)
    s_nm2 = (nm2 % BUF_DEPTH).to(gl.int32)
    s_nm1 = (nm1 % BUF_DEPTH).to(gl.int32)

    qk = _dot_qk(
        q_dot, q_pe, kt_dot, kpe_dot, HAS_PE_DOT, IS_FP8, mfmaLayout, BLOCK_M, BLOCK_N
    )
    cdna4_async.wait_group(2)
    v_dot, v_hi = _read_v(smemV, s_nm3, dotV, D_LO, D_HI_PV)
    acc, acc2, l_i, p_dot = _sc_vec2(acc, acc2, l_i, p_c, alpha_c, dotP, DTYPE, D_HI_PV)
    acc, acc2 = _dot_pv(p_dot, v_dot, v_hi, acc, acc2, D_HI_PV, IS_FP8)
    m_run, p_c, alpha_c = _sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q, P_LIFT)
    gl.barrier()  # WAR: tile n-3's V read against V[n-1]'s write into that slot
    _async_copy_tile(smemV.index(s_nm1), v_base + nm1 * v_step, v_off, v_mask, HAS_MASK)
    cdna4_async.commit_group()  # ACV
    cdna4_async.wait_group(2)
    kt_dot, kpe_dot = _read_k(smemK, smemKpe, s_nm1, dotK, D_LO, D_HI_QK, HAS_PE)

    qk = _dot_qk(
        q_dot, q_pe, kt_dot, kpe_dot, HAS_PE_DOT, IS_FP8, mfmaLayout, BLOCK_M, BLOCK_N
    )
    cdna4_async.wait_group(1)
    v_dot, v_hi = _read_v(smemV, s_nm2, dotV, D_LO, D_HI_PV)
    acc, acc2, l_i, p_dot = _sc_vec2(acc, acc2, l_i, p_c, alpha_c, dotP, DTYPE, D_HI_PV)
    acc, acc2 = _dot_pv(p_dot, v_dot, v_hi, acc, acc2, D_HI_PV, IS_FP8)
    m_run, p_c, alpha_c = _sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q, P_LIFT)

    cdna4_async.wait_group(0)
    v_dot, v_hi = _read_v(smemV, s_nm1, dotV, D_LO, D_HI_PV)
    acc, acc2, l_i, p_dot = _sc_vec2(acc, acc2, l_i, p_c, alpha_c, dotP, DTYPE, D_HI_PV)
    acc, acc2 = _dot_pv(p_dot, v_dot, v_hi, acc, acc2, D_HI_PV, IS_FP8)

    return acc, acc2, l_i, m_run


@gluon.jit
def _attn_fwd_inner(
    acc,
    acc2,
    l_i,
    m_i,
    q,
    q_pe,
    k_base,
    v_base,
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
    stride_kn,
    stride_vn,
    seqlen_k,
    block_min,
    block_max,
    window_min,
    qk_scale,
    sd_base,
    sd_offsets,
    sd_q_mask,
    stride_sd_n,
    mfmaLayout: gl.constexpr,
    dotK: gl.constexpr,
    dotP: gl.constexpr,
    dotV: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,
    HAS_PE: gl.constexpr,
    WINDOW_MASK: gl.constexpr,
    RETURN_SCORES: gl.constexpr,
    SCALE_ON_Q: gl.constexpr,
    BUF_DEPTH: gl.constexpr,
    D_LO: gl.constexpr,
    D_HI_QK: gl.constexpr,
    D_HI_PV: gl.constexpr,
    seqlen_q=None,
    n_extra_tokens=None,
    offs_m=None,
    IS_CAUSAL: gl.constexpr = False,
    MASK_STEPS: gl.constexpr = False,
    IS_FP8: gl.constexpr = False,
):
    """Generic one-tile-at-a-time loop over key blocks `[block_min, block_max)`.

    `q_pe` / `acc2` as in `_attn_fwd_pipelined`.  `IS_FP8` takes the e4m3 QK^T and
    `_attn_softmax_pv_fp8` (no PE, no split head; `acc2` passes through).
    """
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2
    HAS_PE_DOT: gl.constexpr = HAS_PE or D_HI_QK > 0

    if MASK_STEPS or IS_CAUSAL or WINDOW_MASK or RETURN_SCORES:
        offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfmaLayout))
    else:
        offs_n = None

    n_iter = (block_max - block_min) // BLOCK_N

    # Commit groups issued per tile: one for K (carrying the PE slice with it when
    # present), one for V.
    ASYNC_GROUPS: gl.constexpr = 2

    # Prime slot 0, then run one tile ahead: the copy for tile i+1 is in flight
    # while tile i computes.  Because the copy lands straight in LDS there is no
    # `s_waitcnt vmcnt(0)` and no `ds_write` between the global read and the MFMA.
    _async_copy_kv_tile(
        smemK,
        smemKpe,
        smemV,
        0,
        k_base,
        v_base,
        kt_off,
        kpe_off,
        v_off,
        kt_off_n,
        kt_off_d,
        kpe_off_n,
        v_off_n,
        v_off_d,
        block_min,
        seqlen_k,
        BLOCK_DMODEL,
        BLOCK_DMODEL_PE,
        MASK_STEPS,
        PADDED_HEAD,
        HAS_PE,
    )

    for i in range(n_iter):
        start_n = block_min + i * BLOCK_N

        cur = i % BUF_DEPTH
        nxt = BUF_DEPTH - 1 - cur
        # Clamped so the speculative prefetch can never run off the end of K/V;
        # on the last iteration it restages a tile nobody reads, which is cheaper
        # than branching and keeps `wait_group` seeing a uniform queue depth.
        nxt_i = min(i + 1, n_iter - 1)
        _async_copy_kv_tile(
            smemK,
            smemKpe,
            smemV,
            nxt,
            k_base + nxt_i * BLOCK_N * stride_kn,
            v_base + nxt_i * BLOCK_N * stride_vn,
            kt_off,
            kpe_off,
            v_off,
            kt_off_n,
            kt_off_d,
            kpe_off_n,
            v_off_n,
            v_off_d,
            block_min + nxt_i * BLOCK_N,
            seqlen_k,
            BLOCK_DMODEL,
            BLOCK_DMODEL_PE,
            MASK_STEPS,
            PADDED_HEAD,
            HAS_PE,
        )
        # Drain exactly this tile's groups and leave the prefetch in flight.
        cdna4_async.wait_group(ASYNC_GROUPS)
        k, k_pe = _read_k(smemK, smemKpe, cur, dotK, D_LO, D_HI_QK, HAS_PE)
        v, v_hi = _read_v(smemV, cur, dotV, D_LO, D_HI_PV)
        qk = _attn_qk(
            q,
            k,
            q_pe,
            k_pe,
            start_n,
            offs_n,
            window_min,
            qk_scale,
            mfmaLayout=mfmaLayout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HAS_PE=HAS_PE_DOT,  # PE, or a split head's upper slice
            IS_FP8=IS_FP8,
            WINDOW_MASK=WINDOW_MASK,
            # fp8 leaves qk unscaled: its softmax folds qk_scale into the exp2 FMA.
            SCALE_ON_Q=SCALE_ON_Q or IS_FP8,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_max=block_max,
            n_extra_tokens=n_extra_tokens,
            offs_m=offs_m,
            IS_CAUSAL=IS_CAUSAL,
            MASK_STEPS=MASK_STEPS,
        )
        if IS_FP8:
            acc, l_i, m_i = _attn_softmax_pv_fp8(
                acc,
                l_i,
                m_i,
                qk,
                v,
                qk_scale,
                sd_base,
                sd_offsets,
                sd_q_mask,
                offs_n,
                start_n,
                seqlen_k,
                stride_sd_n,
                dotP,
                RETURN_SCORES,
            )
        else:
            acc, acc2, l_i, m_i = _attn_softmax_pv(
                acc,
                acc2,
                l_i,
                m_i,
                qk,
                v,
                v_hi,
                sd_base,
                sd_offsets,
                sd_q_mask,
                offs_n,
                start_n,
                seqlen_k,
                stride_sd_n,
                dotP,
                RETURN_SCORES,
                D_HI_PV,
            )

        # WAR: this tile's LDS reads against the copy that will overwrite the
        # same slot two iterations from now.
        gl.barrier()

    cdna4_async.wait_group(0)

    return acc, acc2, l_i, m_i


@gluon.jit
def _remap_xcd(
    NUM_BLOCKS_M,
    BATCH,
    NUM_Q_HEADS: gl.constexpr,
    NUM_XCD: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
):
    """
    Reorder off_z and off_q_head by xcd remapping.
    Reverse start_m if it is causal kernel.
    """
    pid = gl.program_id(axis=0)
    off_q_head = pid % NUM_Q_HEADS
    # Remap the q-head index across XCDs for better cache locality.
    off_q_head = remap_xcd(off_q_head, NUM_Q_HEADS, NUM_XCD)
    start_m = (pid // NUM_Q_HEADS) % NUM_BLOCKS_M
    off_z = pid // (NUM_Q_HEADS * NUM_BLOCKS_M) % BATCH
    if IS_CAUSAL:
        start_m = NUM_BLOCKS_M - 1 - start_m
    return off_z, off_q_head, start_m


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
    """Split this M-block's K range into, in order: `skipped_blocks` left of the
    window, `window_blocks` cut by the window's edge, `n_full_blocks` with no
    mask at all, and `masked_blocks` on the causal diagonal / ragged K end.

    Returns `(n_blocks, n_extra_tokens, skipped_blocks, window_blocks,
    masked_blocks, n_full_blocks)`; `n_blocks <= 0` is a causal M-block that sees
    no key at all.
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

    # Skip K blocks that are fully left of the earliest key position
    # reachable by this Q block.
    skipped_blocks = 0
    if SLIDING_WINDOW > 0:
        window_start_n = start_m * BLOCK_M + seqlen_k - seqlen_q - SLIDING_WINDOW
        skipped_blocks = min(max(window_start_n, 0) // BLOCK_N, n_blocks)

    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        masked_blocks = padded_block_k

    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.
    visible_blocks = n_blocks - skipped_blocks
    masked_blocks = min(masked_blocks, visible_blocks)

    # The window's lower edge is a diagonal, so it only cuts through the blocks
    # left of the LAST row's lower bound; every block from there on is inside every
    # row's window.  Only those leading blocks need the per-element window mask --
    # the rest of a windowed row is plain dense work, and takes the pipeline.
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
    gl.static_assert(not IS_FP8, "fp8 runs _attn_fwd_fp8")
    RCP_LN2: gl.constexpr = 1.4426950408889634
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2
    PADDED_HEAD_OUT: gl.constexpr = BLOCK_DMODEL_OUT != BLOCK_DMODEL_POW2
    HAS_PE: gl.constexpr = BLOCK_DMODEL_PE > 0
    # A padded head is issued as two power-of-two slices where that saves MFMA work
    # (96 -> 64 + 32, ...): D_LO is the lower slice (the whole padded width when
    # nothing splits), D_HI_QK / D_HI_PV the upper one on each side of the product.
    _SLICES: gl.constexpr = _head_slices(BLOCK_DMODEL, BLOCK_DMODEL_POW2, HAS_PE)
    D_LO: gl.constexpr = _SLICES[0]
    D_HI_QK: gl.constexpr = _SLICES[1]
    D_HI_PV: gl.constexpr = _SLICES[2]
    ACC_WIDTH: gl.constexpr = D_LO if D_HI_PV > 0 else BLOCK_DMODEL_POW2

    # NOTE:
    # Base-pointer and seqlen-loop offset arithmetic is performed using the
    # stride's integer width. With 32-bit strides, these products can overflow
    # and cause segfaults on very large tensors. Upcasting the strides to int64
    # ensures that this arithmetic uses 64-bit precision. The per-tile offset
    # tensors are still downcast to int32 for buffer_load, which is safe, as a
    # single tile's offsets are small.
    if USE_INT64_STRIDES:
        stride_qz = gl.cast(stride_qz_in, gl.int64)
        stride_qh = gl.cast(stride_qh_in, gl.int64)
        stride_qm = gl.cast(stride_qm_in, gl.int64)
        stride_qk = gl.cast(stride_qk_in, gl.int64)
        stride_kz = gl.cast(stride_kz_in, gl.int64)
        stride_kh = gl.cast(stride_kh_in, gl.int64)
        stride_kn = gl.cast(stride_kn_in, gl.int64)
        stride_kk = gl.cast(stride_kk_in, gl.int64)
        stride_vz = gl.cast(stride_vz_in, gl.int64)
        stride_vh = gl.cast(stride_vh_in, gl.int64)
        stride_vn = gl.cast(stride_vn_in, gl.int64)
        stride_vk = gl.cast(stride_vk_in, gl.int64)
        stride_oz = gl.cast(stride_oz_in, gl.int64)
        stride_oh = gl.cast(stride_oh_in, gl.int64)
        stride_om = gl.cast(stride_om_in, gl.int64)
        stride_on = gl.cast(stride_on_in, gl.int64)
        stride_lse_z = gl.cast(stride_lse_z_in, gl.int64)
        stride_lse_h = gl.cast(stride_lse_h_in, gl.int64)
        stride_lse_m = gl.cast(stride_lse_m_in, gl.int64)
        stride_sd_z = gl.cast(stride_sd_z_in, gl.int64)
        stride_sd_h = gl.cast(stride_sd_h_in, gl.int64)
        stride_sd_m = gl.cast(stride_sd_m_in, gl.int64)
        stride_sd_n = gl.cast(stride_sd_n_in, gl.int64)
    else:
        stride_qz = stride_qz_in
        stride_qh = stride_qh_in
        stride_qm = stride_qm_in
        stride_qk = stride_qk_in
        stride_kz = stride_kz_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kk = stride_kk_in
        stride_vz = stride_vz_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vk = stride_vk_in
        stride_oz = stride_oz_in
        stride_oh = stride_oh_in
        stride_om = stride_om_in
        stride_on = stride_on_in
        stride_lse_z = stride_lse_z_in
        stride_lse_h = stride_lse_h_in
        stride_lse_m = stride_lse_m_in
        stride_sd_z = stride_sd_z_in
        stride_sd_h = stride_sd_h_in
        stride_sd_m = stride_sd_m_in
        stride_sd_n = stride_sd_n_in

    # The global->LDS copy hands every lane 16 (or, on the 32-bit form, 4) contiguous
    # bytes, and its lowering only fires when it can PROVE that chunk is aligned to
    # its own size.  A lane's chunk
    # sits at `K + z*stride_kz + h*stride_kh + n*stride_kn + d0`, with `d0` a
    # multiple of the vector -- so EVERY stride that reaches the base pointer has to
    # carry the alignment, not just the sequence one.
    #
    # Triton infers `tt.divisibility = 16` for a stride argument only when the value
    # is a multiple of 16 *elements*.  A head dim padded to 40 gives 40, which is
    # 80 bytes and perfectly 16-byte aligned, but misses that test -- so the
    # conversion pattern silently declines to match and the op survives all the way
    # to LLVM translation as an unlowered `builtin.unrealized_conversion_cast`.
    # State the alignment the host actually measured instead.
    if KV_STRIDE_ALIGN > 1:
        stride_kz = gl.multiple_of(stride_kz, KV_STRIDE_ALIGN)
        stride_kh = gl.multiple_of(stride_kh, KV_STRIDE_ALIGN)
        stride_kn = gl.multiple_of(stride_kn, KV_STRIDE_ALIGN)
        stride_vz = gl.multiple_of(stride_vz, KV_STRIDE_ALIGN)
        stride_vh = gl.multiple_of(stride_vh, KV_STRIDE_ALIGN)
        stride_vn = gl.multiple_of(stride_vn, KV_STRIDE_ALIGN)

    # program -> (batch, q_head, query block). SEQLEN_Q is the max query length,
    # so NUM_BLOCKS_M matches the launch grid in both fixed and varlen mode.
    NUM_BLOCKS_M = gl.cdiv(SEQLEN_Q, BLOCK_M)
    off_z, off_q_head, start_m = _remap_xcd(
        NUM_BLOCKS_M, BATCH, NUM_Q_HEADS, NUM_XCD, IS_CAUSAL
    )

    # In varlen mode the lengths come from cu_seqlens and the batch axis is
    # collapsed (stride_*z == 0); in fixed mode use the SEQLEN_Q/SEQLEN_K args.
    if VARLEN:
        cu_seqlens_q_start = gl.load(cu_seqlens_q + off_z)
        seqlen_q = gl.load(cu_seqlens_q + off_z + 1) - cu_seqlens_q_start
        # This query block is entirely past the end of this batch's sequence.
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

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )

    K_WIDTH: gl.constexpr = 8
    PV_K_WIDTH: gl.constexpr = 4
    dotQ: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=K_WIDTH
    )
    dotK: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=K_WIDTH
    )
    dotP: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=PV_K_WIDTH
    )
    dotV: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=PV_K_WIDTH
    )

    LOAD_VEC: gl.constexpr = 8
    qLoadLayout: gl.constexpr = _make_load_layout(
        D_LO, LOAD_VEC, num_warps, transposed=False
    )
    oStoreLayout: gl.constexpr = _make_load_layout(
        BLOCK_DMODEL_POW2, LOAD_VEC, num_warps, transposed=False
    )

    # Shared layouts for the LDS staging of K and V.
    ELEM_BITS: gl.constexpr = k_ptr.dtype.element_ty.primitive_bitwidth
    _KV_SHARED: gl.constexpr = _make_kv_shared_layouts(BLOCK_DMODEL_POW2, BLOCK_N)

    # K, V and the PE slice reach LDS only by global->LDS async copy, double
    # buffered.  `kv_async_copy_ok` is the same predicate the host checks before
    # launch -- it makes K/V contiguous when a view fails it -- so this only fires
    # if the two ever disagree.
    gl.static_assert(
        kv_async_copy_ok(
            BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE,
            BLOCK_N,
            num_warps,
            ELEM_BITS,
            KV_STRIDE_ALIGN,
        ),
        "K/V cannot be async-copied for this tile; the host must make K/V contiguous",
    )
    BUF_DEPTH: gl.constexpr = 2
    kSharedLayout: gl.constexpr = _KV_SHARED[0]
    vSharedLayout: gl.constexpr = _KV_SHARED[1]

    if HAS_PE:
        qPeLoadLayout: gl.constexpr = _make_load_layout(
            BLOCK_DMODEL_PE, LOAD_VEC, num_warps, transposed=False
        )
        # The PE tile gets the same staggered padded layout as K: the async copy
        # writes it the same way.
        _KPE_SHARED: gl.constexpr = _make_kv_shared_layouts(BLOCK_DMODEL_PE, BLOCK_N)
        kPeSharedLayout: gl.constexpr = _KPE_SHARED[0]
    else:
        qPeLoadLayout: gl.constexpr = None
        kPeSharedLayout: gl.constexpr = None

    # HEAD_STRIDE_ALIGN is the largest power of two (capped at the 128-bit load
    # width) dividing every Q/K/V head-axis stride, in elements.
    qh_off = off_q_head * stride_qh
    kh_off = off_k_head * stride_kh
    vh_off = off_k_head * stride_vh
    if HEAD_STRIDE_ALIGN > 1:
        qh_off = gl.multiple_of(qh_off, HEAD_STRIDE_ALIGN)
        kh_off = gl.multiple_of(kh_off, HEAD_STRIDE_ALIGN)
        vh_off = gl.multiple_of(vh_off, HEAD_STRIDE_ALIGN)

    qk_scale = sm_scale * RCP_LN2

    # Load Q (stays resident for the whole key loop).
    offs_qm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, qLoadLayout))
    offs_qd = gl.arange(0, D_LO, layout=gl.SliceLayout(0, qLoadLayout))
    q_base = (
        q_ptr
        + off_z * stride_qz
        + qh_off
        + cu_seqlens_q_start * stride_qm
        + start_m * BLOCK_M * stride_qm
    )
    q_offsets = (offs_qm[:, None] * stride_qm + offs_qd[None, :] * stride_qk).to(
        gl.int32
    )
    q_mask = (start_m * BLOCK_M + offs_qm)[:, None] < seqlen_q
    if PADDED_HEAD and D_HI_QK == 0:
        # A split head's lower slice is all real head dim; its upper one is masked
        # below.
        q_mask = q_mask & (offs_qd[None, :] < BLOCK_DMODEL)
    # Cache Q at .cg when a single Q block spans at least one full head.
    if BLOCK_M >= NUM_Q_HEADS:
        q_cache_mod: gl.constexpr = ".cg"
    else:
        q_cache_mod: gl.constexpr = ""
    q = gl.amd.cdna4.buffer_load(
        ptr=q_base, offsets=q_offsets, mask=q_mask, other=0.0, cache=q_cache_mod
    )
    q = gl.convert_layout(q, layout=dotQ)
    if SCALE_ON_Q and D_HI_QK == 0:
        # Fold qk_scale into the Q operand ONCE, here, instead of scaling every
        # tile's [BLOCK_M, BLOCK_N] score matrix inside the loop.  q lives in
        # registers for the whole key loop, so this is a single pass over it, and it
        # also shortens the row-max dependency chain (the max needs no multiply and
        # the exponent argument becomes a plain subtract).  Costs one extra rounding
        # of Q to the input dtype.
        q = (q.to(gl.float32) * qk_scale).to(q_ptr.dtype.element_ty)

    # The PE slice sits immediately after the NOPE slice along the head dim of
    # Q and K, so it shares their base pointer and only shifts the head-dim
    # offsets by BLOCK_DMODEL. V and the output only span the NOPE slice.
    if HAS_PE:
        offs_qpm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, qPeLoadLayout))
        offs_qpd = BLOCK_DMODEL + gl.arange(
            0, BLOCK_DMODEL_PE, layout=gl.SliceLayout(0, qPeLoadLayout)
        )
        q_pe_offsets = (
            offs_qpm[:, None] * stride_qm + offs_qpd[None, :] * stride_qk
        ).to(gl.int32)
        q_pe_mask = (start_m * BLOCK_M + offs_qpm)[:, None] < seqlen_q
        q_pe = gl.amd.cdna4.buffer_load(
            ptr=q_base,
            offsets=q_pe_offsets,
            mask=q_pe_mask,
            other=0.0,
            cache=q_cache_mod,
        )
        q_pe = gl.convert_layout(q_pe, layout=dotQ)
    elif D_HI_QK > 0:
        # A split head: q_pe carries the upper head slice [D_LO, D_LO + D_HI_QK)
        # -- it takes the PE operand's place in the Q@K^T chain -- and that slice
        # carries the head-dim mask.
        qHiLoadLayout: gl.constexpr = _make_load_layout(
            D_HI_QK, LOAD_VEC, num_warps, transposed=False
        )
        offs_qhm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, qHiLoadLayout))
        offs_qhd = D_LO + gl.arange(0, D_HI_QK, layout=gl.SliceLayout(0, qHiLoadLayout))
        q_hi_offsets = (
            offs_qhm[:, None] * stride_qm + offs_qhd[None, :] * stride_qk
        ).to(gl.int32)
        q_hi_mask = ((start_m * BLOCK_M + offs_qhm)[:, None] < seqlen_q) & (
            offs_qhd[None, :] < BLOCK_DMODEL
        )
        q_hi = gl.amd.cdna4.buffer_load(
            ptr=q_base,
            offsets=q_hi_offsets,
            mask=q_hi_mask,
            other=0.0,
            cache=q_cache_mod,
        )
        q_pe = gl.convert_layout(q_hi, layout=dotQ)
    else:
        q_pe = None
    if SCALE_ON_Q and D_HI_QK > 0:
        # Both slices loaded first, then both scaled: scaling the lower one before
        # the upper one's load measured ~2% slower at short sequences (D=192).
        q = (q.to(gl.float32) * qk_scale).to(q_ptr.dtype.element_ty)
    if SCALE_ON_Q and (HAS_PE or D_HI_QK > 0):
        q_pe = (q_pe.to(gl.float32) * qk_scale).to(q_ptr.dtype.element_ty)

    k_base = k_ptr + off_z * stride_kz + kh_off + cu_seqlens_k_start * stride_kn

    v_base = v_ptr + off_z * stride_vz + vh_off + cu_seqlens_k_start * stride_vn

    # Shared-memory tiles for the K/V staging, double-buffered so the copy for tile
    # i+1 can be in flight while tile i is being consumed.
    smemK = gl.allocate_shared_memory(
        k_ptr.dtype.element_ty,
        [BUF_DEPTH, BLOCK_DMODEL_POW2, BLOCK_N],
        kSharedLayout,
    )
    smemV = gl.allocate_shared_memory(
        v_ptr.dtype.element_ty,
        [BUF_DEPTH, BLOCK_N, BLOCK_DMODEL_POW2],
        vSharedLayout,
    )
    if HAS_PE:
        smemKpe = gl.allocate_shared_memory(
            k_ptr.dtype.element_ty,
            [BUF_DEPTH, BLOCK_DMODEL_PE, BLOCK_N],
            kPeSharedLayout,
        )
    else:
        smemKpe = None

    # async copy address layouts and the intra-tile offset pattern.  The pattern never
    # changes from tile to tile -- successive tiles move the scalar base pointer --
    # so all of this stays out of the loop.
    KV_VEC: gl.constexpr = _async_copy_vec(
        BLOCK_DMODEL_POW2, BLOCK_N, 0, num_warps, ELEM_BITS, KV_STRIDE_ALIGN
    )
    kt_async_layout: gl.constexpr = _async_copy_layout(
        BLOCK_DMODEL_POW2, BLOCK_N, 0, num_warps, KV_VEC
    )
    v_async_layout: gl.constexpr = _async_copy_layout(
        BLOCK_N, BLOCK_DMODEL_POW2, 1, num_warps, KV_VEC
    )
    kt_off_d = gl.arange(
        0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(1, kt_async_layout)
    )
    kt_off_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kt_async_layout))
    kt_off = (kt_off_d[:, None] * stride_kk + kt_off_n[None, :] * stride_kn).to(
        gl.int32
    )
    v_off_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, v_async_layout))
    v_off_d = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, v_async_layout))
    v_off = (v_off_n[:, None] * stride_vn + v_off_d[None, :] * stride_vk).to(gl.int32)
    if HAS_PE:
        PE_VEC: gl.constexpr = _async_copy_vec(
            BLOCK_DMODEL_PE, BLOCK_N, 0, num_warps, ELEM_BITS, KV_STRIDE_ALIGN
        )
        kpe_async_layout: gl.constexpr = _async_copy_layout(
            BLOCK_DMODEL_PE, BLOCK_N, 0, num_warps, PE_VEC
        )
        kpe_off_d = BLOCK_DMODEL + gl.arange(
            0, BLOCK_DMODEL_PE, layout=gl.SliceLayout(1, kpe_async_layout)
        )
        kpe_off_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kpe_async_layout))
        kpe_off = (kpe_off_d[:, None] * stride_kk + kpe_off_n[None, :] * stride_kn).to(
            gl.int32
        )
    else:
        kpe_off_n = None
        kpe_off = None
    if PADDED_HEAD:
        # Broadcast against the offsets, so each half carries only its own axis.
        k_head_mask = kt_off_d[:, None] < BLOCK_DMODEL
        v_head_mask = v_off_d[None, :] < BLOCK_DMODEL
    else:
        k_head_mask = None
        v_head_mask = None
    # online-softmax state.
    if ENABLE_SINK:
        m_i_init = gl.load(sink_ptr + off_q_head).to(gl.float32) * RCP_LN2
    elif SLIDING_WINDOW > 0:
        # A sliding-window block can be fully masked for some rows, and -inf as the
        # running max would then make exp2(-inf - m_i) NaN. A finite floor keeps the
        # probabilities at 0 and the rescale factor at exactly 1.0.
        m_i_init = -1.0e30
    else:
        m_i_init = float("-inf")

    m_i = gl.full(
        [BLOCK_M], m_i_init, dtype=gl.float32, layout=gl.SliceLayout(1, mfmaLayout)
    )
    l_i = gl.full(
        [BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, mfmaLayout)
    )
    acc = gl.zeros([BLOCK_M, ACC_WIDTH], dtype=gl.float32, layout=mfmaLayout)
    if D_HI_PV > 0:
        acc2 = gl.zeros([BLOCK_M, D_HI_PV], dtype=gl.float32, layout=mfmaLayout)
    else:
        # Never touched without a P@V split; it only has to be a real tensor.
        acc2 = acc

    # Query positions used for the causal mask, in the MFMA result layout.
    offs_m = start_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout)
    )

    # Lowest key index each query row may attend. Like the causal mask, the
    # window is aligned to the bottom right corner.
    if SLIDING_WINDOW > 0:
        window_min = offs_m + (seqlen_k - seqlen_q - SLIDING_WINDOW)
    else:
        window_min = None

    # softmax_lse
    if softmax_lse_ptr is not None:
        lse_base = (
            softmax_lse_ptr
            + off_z * stride_lse_z
            + off_q_head * stride_lse_h
            + cu_seqlens_q_start * stride_lse_m
            + start_m * BLOCK_M * stride_lse_m
        )
        offs_lse = (
            gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout)) * stride_lse_m
        ).to(gl.int32)
        # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
        lse_mask = offs_m < seqlen_q
    else:
        lse_base = None
        offs_lse = None
        lse_mask = None

    # s_dmask (return_scores)
    if s_dmask_ptr is not None:
        sd_base = (
            s_dmask_ptr
            + off_z * stride_sd_z
            + off_q_head * stride_sd_h
            + start_m * BLOCK_M * stride_sd_m
        )
        offs_sd_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout))
        offs_sd_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfmaLayout))
        sd_offsets = (
            offs_sd_m[:, None] * stride_sd_m + offs_sd_n[None, :] * stride_sd_n
        ).to(gl.int32)
        sd_q_mask = offs_m < seqlen_q
    else:
        sd_base = None
        sd_offsets = None
        sd_q_mask = None

    # Classify key blocks: full (no boundary/causal mask) vs masked.
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
            storeLayout: gl.constexpr = oStoreLayout
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
            gl.amd.cdna4.buffer_store(zeros, ptr=o_base, offsets=o_offsets, mask=o_mask)

            if softmax_lse_ptr is not None:
                lse = gl.zeros(
                    [BLOCK_M], dtype=gl.float32, layout=gl.SliceLayout(1, mfmaLayout)
                )
                gl.amd.cdna4.buffer_store(
                    lse, ptr=lse_base, offsets=offs_lse, mask=lse_mask
                )
            return

    block_min = skipped_blocks * BLOCK_N
    block_max = n_blocks * BLOCK_N

    if SLIDING_WINDOW > 0:
        # k_base also anchors the PE slice, which shares K's base pointer.
        k_base += skipped_blocks * BLOCK_N * stride_kn
        v_base += skipped_blocks * BLOCK_N * stride_vn
        if window_blocks > 0:
            acc, acc2, l_i, m_i = _attn_fwd_inner(
                acc,
                acc2,
                l_i,
                m_i,
                q,
                q_pe,
                k_base,
                v_base,
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
                stride_kn,
                stride_vn,
                seqlen_k,
                block_min,
                block_min + window_blocks * BLOCK_N,
                window_min,
                qk_scale,
                sd_base,
                sd_offsets,
                sd_q_mask,
                stride_sd_n,
                mfmaLayout=mfmaLayout,
                dotK=dotK,
                dotP=dotP,
                dotV=dotV,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
                BLOCK_DMODEL_PE=BLOCK_DMODEL_PE,
                HAS_PE=HAS_PE,
                WINDOW_MASK=True,
                RETURN_SCORES=RETURN_SCORES,
                SCALE_ON_Q=SCALE_ON_Q,
                BUF_DEPTH=BUF_DEPTH,
                D_LO=D_LO,
                D_HI_QK=D_HI_QK,
                D_HI_PV=D_HI_PV,
            )
            k_base += window_blocks * BLOCK_N * stride_kn
            v_base += window_blocks * BLOCK_N * stride_vn
            block_min = block_min + window_blocks * BLOCK_N

    # The rotated pipeline is a dense-path specialisation: it carries no per-element
    # masking, so it only takes the configurations whose full
    # blocks are plain Q@K^T / P@V.  Everything it declines -- and every masked block
    # in every configuration -- goes to the generic loop below.
    #
    # The PE slice rides with K through the pipeline: same commit group in mem1,
    # read beside it in mem2, consumed beside it in dot1.  That puts a second MFMA
    # chain in dot1 with no extra memory work in mem1 to face it, which the
    # pipeline only absorbs when it has a second wave of this workgroup on the SIMD
    # to ping-pong with: with 8 waves (PE(128/64)) it measures well ahead of the
    # generic loop, with the 4 waves the wider PE heads need (PE(192/128)) it is
    # behind -- so the PE slice takes the pipeline only at 8.
    #
    # Moving the PE chain's MFMA into dot2 instead measured no better.  Moving its
    # LDS *read* into mem1 is not legal: a commit group is complete only once every
    # wave has waited on its own share of the copy, and the two wave groups run one
    # stage apart, so a tile is safe to read one stage after the wait that covers
    # it.  K[j+2] is waited on right before mem1(j), so it is read in mem2(j).
    PE_IN_PIPELINE: gl.constexpr = num_warps >= 8
    FAST_PATH: gl.constexpr = (not RETURN_SCORES) and (PE_IN_PIPELINE or not HAS_PE)

    pipelined = False
    if FAST_PATH:  # noqa: SIM102
        # Checked per workgroup, not per launch: under a causal mask each M-block
        # sees a different number of full tiles, so a long sequence still gives its
        # first M-blocks only a handful.
        if n_full_blocks >= _MIN_PIPE_BLOCKS:
            pipelined = True
            acc, acc2, l_i, m_i = _attn_fwd_pipelined(
                acc,
                acc2,
                l_i,
                m_i,
                q,
                q_pe,
                smemK,
                smemKpe,
                smemV,
                k_base,
                v_base,
                kt_off,
                kpe_off,
                v_off,
                k_head_mask,
                v_head_mask,
                BLOCK_N * stride_kn,
                BLOCK_N * stride_vn,
                n_full_blocks,
                qk_scale,
                mfmaLayout=mfmaLayout,
                dotK=dotK,
                dotV=dotV,
                dotP=dotP,
                SCALE_ON_Q=SCALE_ON_Q,
                DTYPE=v_ptr.dtype.element_ty,
                HAS_MASK=PADDED_HEAD,
                HAS_PE=HAS_PE,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BUF_DEPTH=BUF_DEPTH,
                D_LO=D_LO,
                D_HI_QK=D_HI_QK,
                D_HI_PV=D_HI_PV,
            )
            block_min = block_min + n_full_blocks * BLOCK_N
            block_max = n_blocks * BLOCK_N
            # WAR: the pipeline's last LDS reads against whatever stages next.
            gl.barrier()

    # Full blocks: no boundary mask, no causal mask.
    if (not pipelined) and n_full_blocks > 0:
        block_max = block_min + n_full_blocks * BLOCK_N
        acc, acc2, l_i, m_i = _attn_fwd_inner(
            acc,
            acc2,
            l_i,
            m_i,
            q,
            q_pe,
            k_base,
            v_base,
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
            stride_kn,
            stride_vn,
            seqlen_k,
            block_min,
            block_max,
            window_min,
            qk_scale,
            sd_base,
            sd_offsets,
            sd_q_mask,
            stride_sd_n,
            mfmaLayout=mfmaLayout,
            dotK=dotK,
            dotP=dotP,
            dotV=dotV,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE=BLOCK_DMODEL_PE,
            HAS_PE=HAS_PE,
            WINDOW_MASK=False,
            RETURN_SCORES=RETURN_SCORES,
            SCALE_ON_Q=SCALE_ON_Q,
            BUF_DEPTH=BUF_DEPTH,
            D_LO=D_LO,
            D_HI_QK=D_HI_QK,
            D_HI_PV=D_HI_PV,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    # Remaining blocks carry the boundary / causal masking.
    if masked_blocks > 0:
        k_base += n_full_blocks * BLOCK_N * stride_kn
        v_base += n_full_blocks * BLOCK_N * stride_vn
        acc, acc2, l_i, m_i = _attn_fwd_inner(
            acc,
            acc2,
            l_i,
            m_i,
            q,
            q_pe,
            k_base,
            v_base,
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
            stride_kn,
            stride_vn,
            seqlen_k,
            block_min,
            block_max,
            window_min,
            qk_scale,
            sd_base,
            sd_offsets,
            sd_q_mask,
            stride_sd_n,
            mfmaLayout=mfmaLayout,
            dotK=dotK,
            dotP=dotP,
            dotV=dotV,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE=BLOCK_DMODEL_PE,
            HAS_PE=HAS_PE,
            WINDOW_MASK=SLIDING_WINDOW > 0,
            RETURN_SCORES=RETURN_SCORES,
            SCALE_ON_Q=SCALE_ON_Q,
            BUF_DEPTH=BUF_DEPTH,
            D_LO=D_LO,
            D_HI_QK=D_HI_QK,
            D_HI_PV=D_HI_PV,
            seqlen_q=seqlen_q,
            n_extra_tokens=n_extra_tokens,
            offs_m=offs_m,
            IS_CAUSAL=IS_CAUSAL,
            MASK_STEPS=True,
        )

    # epilogue: normalize and write.  Take the reciprocal on the [BLOCK_M] vector and
    # multiply, rather than dividing the [BLOCK_M, BLOCK_DMODEL] accumulator: a full
    # IEEE divide expands to several VALU ops *per accumulator element*, and there are
    # BLOCK_DMODEL of them per row.
    l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]
    if D_HI_PV > 0:
        acc2 = acc2 * l_recip[:, None]

    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    if IS_CAUSAL:  # noqa: SIM102
        if (causal_start_idx > start_m_idx) and (causal_start_idx < end_m_idx):
            out_mask_boundary = gl.full(
                [ACC_WIDTH],
                causal_start_idx,
                dtype=gl.int32,
                layout=gl.SliceLayout(0, mfmaLayout),
            )
            mask_m_offsets = start_m_idx + gl.arange(
                0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout)
            )
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            acc = gl.where(out_ptrs_mask, acc, 0.0)
            if D_HI_PV > 0:
                acc2 = gl.where(
                    (mask_m_offsets >= causal_start_idx)[:, None], acc2, 0.0
                )

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    if softmax_lse_ptr is not None:
        LN2: gl.constexpr = 0.6931471824645996
        # compute log-sum-exp in base 2 units
        softmax_lse = m_i + gl.log2(l_i)
        # convert back to natural units
        softmax_lse = softmax_lse * LN2

        if IS_CAUSAL:
            # zero out nans caused by -infs when doing causal
            softmax_lse = gl.where(offs_m < causal_start_idx, 0.0, softmax_lse)

        gl.amd.cdna4.buffer_store(
            softmax_lse, ptr=lse_base, offsets=offs_lse, mask=lse_mask
        )

    if D_HI_PV > 0:
        # Two output slices; both go through _store_out against one base pointer
        # and one row bound.
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
            LOAD_VEC,
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
            LOAD_VEC,
            num_warps,
        )
    else:
        out = acc.to(o_ptr.dtype.element_ty)

        storeLayout: gl.constexpr = _make_load_layout(
            BLOCK_DMODEL_POW2, LOAD_VEC, num_warps, transposed=False
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
# fp8 (e4m3) -- a second kernel.
#
# fp8 halves the MFMA time of a tile but not its softmax, so the tile that is
# MFMA-bound in bf16 is VALU-bound here, and the bf16 kernel's fp8 branch pays for
# P's quantization on top: a workgroup-wide amax of P every tile (a cross-warp
# reduction through LDS and a barrier), then P@V into a fresh accumulator that is
# rescaled by `descale_p * descale_v` and added to `acc`.
#
# This kernel scales P statically instead.  P = exp2(s - m) is at most 1, so
# computing it as exp2(s - m + P_LIFT) with P_LIFT = log2(448) puts it on e4m3's
# full range with no reduction at all; the row sum `l` carries the same factor,
# so it cancels in `acc / l` and only the LSE has to take it back out.  P@V then
# accumulates straight into `acc` like bf16, and `descale_v` is applied once,
# in the epilogue.  (FlyDSL's gfx950 fp8 kernel does the same.)
#
# Everything else -- block classification, the sliding-window split, the masked
# Q@K^T, the copy helpers -- is shared with `_attn_fwd`.  No PE slice: the host
# rejects fp8 with PE.
# ---------------------------------------------------------------------------

# log2 of e4m3's largest finite value (448), less a hair so that exp2 rounding
# can never carry P past it.
_FP8_P_LIFT = gl.constexpr(8.8)


# The staggered layouts of _padded_staggered_layout, at 8 bits, chosen with the LDS
# counters: a 1 KB padding interval (1024 e4m3 elements; at 512 the async copy's
# lowering declines the destination), a 16-element pad on K^T and a 32-element pad
# on V.  0.7% of LDS-active cycles bank-conflicted, against 24% for the swizzle.
# K^T's pad must stay a multiple of 16 *bytes*: it is read with ds_read_b128, and
# the bf16 kernel's 8-element pad -- conflict-free here -- misaligns every other
# row and stalls 88% of LDS cycles on unaligned access, a cost the bank-conflict
# counter does not see.
_FP8_PAD_INTERVAL = 1024


@gluon.constexpr_function
def _fp8_staggered_ok(head_dim_pow2, block_n):
    """`_staggered_layout_ok` at 8 bits."""
    if head_dim_pow2 & (head_dim_pow2 - 1) or block_n & (block_n - 1):
        return False
    if head_dim_pow2 < 16 or block_n < 16:
        return False
    return block_n % max(_FP8_PAD_INTERVAL // head_dim_pow2, 1) == 0


@gluon.constexpr_function
def _fp8_kv_shared_layouts(head_dim_pow2, block_n):
    """K^T / V LDS layouts for e4m3: staggered and padded at the fp8 interval."""
    return (
        _padded_staggered_layout(
            head_dim_pow2, block_n, 0, pad=16, interval=_FP8_PAD_INTERVAL
        ),
        _padded_staggered_layout(
            block_n, head_dim_pow2, 1, pad=32, interval=_FP8_PAD_INTERVAL
        ),
    )


@gluon.constexpr_function
def _fp8_async_copy_ok(head_dim_pow2, block_n, num_warps, stride_align):
    """`_async_copy_ok` at 8 bits: a 16-element (128-bit) or 4-element (32-bit)
    vector per lane, the staggered destination, and a KV stride alignment reaching
    the vector."""
    if not _fp8_staggered_ok(head_dim_pow2, block_n):
        return False
    vec = _async_copy_vec(
        head_dim_pow2, block_n, 0, num_warps, 8, stride_align, _FP8_PAD_INTERVAL
    )
    if vec == 0:
        return False
    return (
        _async_copy_vec(
            block_n, head_dim_pow2, 1, num_warps, 8, stride_align, _FP8_PAD_INTERVAL
        )
        == vec
    )


@gluon.constexpr_function
def _fp8_async_copy_layouts(head_dim_pow2, block_n, num_warps, stride_align):
    """(K^T, V) async-copy address layouts at 8 bits, against the fp8 padding
    interval.  A constexpr function so the kernel never names `_FP8_PAD_INTERVAL`,
    a plain module global that `@gluon.jit` code cannot read."""
    vec = _async_copy_vec(
        head_dim_pow2, block_n, 0, num_warps, 8, stride_align, _FP8_PAD_INTERVAL
    )
    return (
        _async_copy_layout(
            head_dim_pow2, block_n, 0, num_warps, vec, _FP8_PAD_INTERVAL
        ),
        _async_copy_layout(
            block_n, head_dim_pow2, 1, num_warps, vec, _FP8_PAD_INTERVAL
        ),
    )


@gluon.constexpr_function
def kv_async_copy_ok(
    head_dim_pow2, pe_head_dim, block_n, num_warps, elem_bits, stride_align
):
    """Can every K/V tile (and the PE slice) of this launch go global->LDS by async
    copy?  The kernels have no other way to stage K/V, so they assert it; the host
    calls the same function before launch and makes K/V contiguous when it fails."""
    if elem_bits == 8:
        return _fp8_async_copy_ok(head_dim_pow2, block_n, num_warps, stride_align)
    return _async_copy_ok(
        head_dim_pow2, block_n, num_warps, elem_bits, stride_align
    ) and (
        pe_head_dim == 0
        or _async_copy_ok(pe_head_dim, block_n, num_warps, elem_bits, stride_align)
    )


@gluon.jit
def _attn_softmax_pv_fp8(
    acc,
    l_i,
    m_i,
    qk,
    v,
    qk_scale,
    sd_base,
    sd_offsets,
    sd_q_mask,
    offs_n,
    start_n,
    seqlen_k,
    stride_sd_n,
    dotP: gl.constexpr,
    RETURN_SCORES: gl.constexpr,
):
    """Online softmax + P@V for one key block, P statically scaled to e4m3.

    `qk` is unscaled (`qk_scale` folds the softmax scale and the Q/K descales);
    the scale rides the exp2 argument's FMA.
    """
    m_ij = gl.maximum(m_i, gl.max(qk, 1) * qk_scale)
    p = gl.exp2(gl.fma(qk, qk_scale, _FP8_P_LIFT - m_ij[:, None]))
    alpha = gl.exp2(m_i - m_ij)
    l_ij = gl.sum(p, 1)

    if RETURN_SCORES:
        # Store P on the same scale as the bf16 kernel does, without the lift.
        p_mask = sd_q_mask[:, None] & ((start_n + offs_n)[None, :] < seqlen_k)
        gl.amd.cdna4.buffer_store(
            (p * gl.exp2(-_FP8_P_LIFT)).to(sd_base.dtype.element_ty),
            ptr=sd_base + start_n * stride_sd_n,
            offsets=sd_offsets,
            mask=p_mask,
        )

    acc = acc * alpha[:, None]
    p = gl.convert_layout(p.to(v.dtype), layout=dotP, assert_trivial=True)
    acc = gl.amd.cdna4.mfma_scaled(p, None, "e4m3", v, None, "e4m3", acc)
    l_i = l_i * alpha + l_ij
    return acc, l_i, m_ij


_attn_fwd_fp8_repr = make_kernel_repr(
    "_attn_fwd_gluon_fp8",
    [
        "IS_CAUSAL",
        "NUM_Q_HEADS",
        "NUM_K_HEADS",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_DMODEL",
        "RETURN_SCORES",
        "HEAD_STRIDE_ALIGN",
        "VARLEN",
        "NUM_XCD",
        "USE_INT64_STRIDES",
        "ENABLE_SINK",
        "SLIDING_WINDOW",
    ],
)


@gluon.jit(repr=_attn_fwd_fp8_repr)
def _attn_fwd_fp8(
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
    BLOCK_DMODEL_PE: gl.constexpr,
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
    """Same arguments as `_attn_fwd`, so the host launches either one."""
    gl.static_assert(IS_FP8, "fp8 kernel")
    gl.static_assert(BLOCK_DMODEL_PE == 0, "fp8 kernel takes no PE slice")
    RCP_LN2: gl.constexpr = 1.4426950408889634
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2

    if USE_INT64_STRIDES:
        stride_qz = gl.cast(stride_qz_in, gl.int64)
        stride_qh = gl.cast(stride_qh_in, gl.int64)
        stride_qm = gl.cast(stride_qm_in, gl.int64)
        stride_qk = gl.cast(stride_qk_in, gl.int64)
        stride_kz = gl.cast(stride_kz_in, gl.int64)
        stride_kh = gl.cast(stride_kh_in, gl.int64)
        stride_kn = gl.cast(stride_kn_in, gl.int64)
        stride_kk = gl.cast(stride_kk_in, gl.int64)
        stride_vz = gl.cast(stride_vz_in, gl.int64)
        stride_vh = gl.cast(stride_vh_in, gl.int64)
        stride_vn = gl.cast(stride_vn_in, gl.int64)
        stride_vk = gl.cast(stride_vk_in, gl.int64)
        stride_descale_q_z = gl.cast(stride_descale_q_z_in, gl.int64)
        stride_descale_k_z = gl.cast(stride_descale_k_z_in, gl.int64)
        stride_descale_v_z = gl.cast(stride_descale_v_z_in, gl.int64)
        stride_oz = gl.cast(stride_oz_in, gl.int64)
        stride_oh = gl.cast(stride_oh_in, gl.int64)
        stride_om = gl.cast(stride_om_in, gl.int64)
        stride_on = gl.cast(stride_on_in, gl.int64)
        stride_lse_z = gl.cast(stride_lse_z_in, gl.int64)
        stride_lse_h = gl.cast(stride_lse_h_in, gl.int64)
        stride_lse_m = gl.cast(stride_lse_m_in, gl.int64)
        stride_sd_z = gl.cast(stride_sd_z_in, gl.int64)
        stride_sd_h = gl.cast(stride_sd_h_in, gl.int64)
        stride_sd_m = gl.cast(stride_sd_m_in, gl.int64)
        stride_sd_n = gl.cast(stride_sd_n_in, gl.int64)
    else:
        stride_qz = stride_qz_in
        stride_qh = stride_qh_in
        stride_qm = stride_qm_in
        stride_qk = stride_qk_in
        stride_kz = stride_kz_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kk = stride_kk_in
        stride_vz = stride_vz_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vk = stride_vk_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_oz = stride_oz_in
        stride_oh = stride_oh_in
        stride_om = stride_om_in
        stride_on = stride_on_in
        stride_lse_z = stride_lse_z_in
        stride_lse_h = stride_lse_h_in
        stride_lse_m = stride_lse_m_in
        stride_sd_z = stride_sd_z_in
        stride_sd_h = stride_sd_h_in
        stride_sd_m = stride_sd_m_in
        stride_sd_n = stride_sd_n_in

    # The async copy's alignment proof -- see the note at the same place in _attn_fwd.
    if KV_STRIDE_ALIGN > 1:
        stride_kz = gl.multiple_of(stride_kz, KV_STRIDE_ALIGN)
        stride_kh = gl.multiple_of(stride_kh, KV_STRIDE_ALIGN)
        stride_kn = gl.multiple_of(stride_kn, KV_STRIDE_ALIGN)
        stride_vz = gl.multiple_of(stride_vz, KV_STRIDE_ALIGN)
        stride_vh = gl.multiple_of(stride_vh, KV_STRIDE_ALIGN)
        stride_vn = gl.multiple_of(stride_vn, KV_STRIDE_ALIGN)

    NUM_BLOCKS_M = gl.cdiv(SEQLEN_Q, BLOCK_M)
    off_z, off_q_head, start_m = _remap_xcd(
        NUM_BLOCKS_M, BATCH, NUM_Q_HEADS, NUM_XCD, IS_CAUSAL
    )

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

    descale_q = gl.load(descale_q_ptr + off_z * stride_descale_q_z + off_q_head)
    descale_k = gl.load(descale_k_ptr + off_z * stride_descale_k_z + off_k_head)
    descale_v = gl.load(descale_v_ptr + off_z * stride_descale_v_z + off_k_head)

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    K_WIDTH: gl.constexpr = 16
    PV_K_WIDTH: gl.constexpr = 4
    dotQ: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=K_WIDTH
    )
    dotK: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=K_WIDTH
    )
    dotP: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=PV_K_WIDTH
    )
    dotV: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=PV_K_WIDTH
    )

    LOAD_VEC: gl.constexpr = 16
    qLoadLayout: gl.constexpr = _make_load_layout(
        BLOCK_DMODEL_POW2, LOAD_VEC, num_warps, transposed=False
    )

    _KV_SHARED: gl.constexpr = _fp8_kv_shared_layouts(BLOCK_DMODEL_POW2, BLOCK_N)
    gl.static_assert(
        kv_async_copy_ok(BLOCK_DMODEL_POW2, 0, BLOCK_N, num_warps, 8, KV_STRIDE_ALIGN),
        "K/V cannot be async-copied for this tile; the host must make K/V contiguous",
    )
    BUF_DEPTH: gl.constexpr = 2
    kSharedLayout: gl.constexpr = _KV_SHARED[0]
    vSharedLayout: gl.constexpr = _KV_SHARED[1]

    qh_off = off_q_head * stride_qh
    kh_off = off_k_head * stride_kh
    vh_off = off_k_head * stride_vh
    if HEAD_STRIDE_ALIGN > 1:
        qh_off = gl.multiple_of(qh_off, HEAD_STRIDE_ALIGN)
        kh_off = gl.multiple_of(kh_off, HEAD_STRIDE_ALIGN)
        vh_off = gl.multiple_of(vh_off, HEAD_STRIDE_ALIGN)

    # q stays e4m3: re-rounding q * scale into fp8 would lose more than the
    # per-tile multiply costs, so the scale (with both descales) rides the softmax.
    qk_scale = sm_scale * RCP_LN2 * descale_q * descale_k
    # Keep the scale positive so masked -inf * scale stays -inf, not NaN.
    qk_scale = gl.maximum(qk_scale, 1.0e-30)

    q_base = (
        q_ptr
        + off_z * stride_qz
        + qh_off
        + cu_seqlens_q_start * stride_qm
        + start_m * BLOCK_M * stride_qm
    )
    if BLOCK_M >= NUM_Q_HEADS:
        q_cache_mod: gl.constexpr = ".cg"
    else:
        q_cache_mod: gl.constexpr = ""
    offs_qm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, qLoadLayout))
    offs_qd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, qLoadLayout))
    q_offsets = (offs_qm[:, None] * stride_qm + offs_qd[None, :] * stride_qk).to(
        gl.int32
    )
    q_mask = (start_m * BLOCK_M + offs_qm)[:, None] < seqlen_q
    if PADDED_HEAD:
        q_mask = q_mask & (offs_qd[None, :] < BLOCK_DMODEL)
    q = gl.amd.cdna4.buffer_load(
        ptr=q_base, offsets=q_offsets, mask=q_mask, other=0.0, cache=q_cache_mod
    )
    q = gl.convert_layout(q, layout=dotQ)

    k_base = k_ptr + off_z * stride_kz + kh_off + cu_seqlens_k_start * stride_kn

    v_base = v_ptr + off_z * stride_vz + vh_off + cu_seqlens_k_start * stride_vn

    smemK = gl.allocate_shared_memory(
        k_ptr.dtype.element_ty, [BUF_DEPTH, BLOCK_DMODEL_POW2, BLOCK_N], kSharedLayout
    )
    smemV = gl.allocate_shared_memory(
        v_ptr.dtype.element_ty, [BUF_DEPTH, BLOCK_N, BLOCK_DMODEL_POW2], vSharedLayout
    )

    _KV_ASYNC: gl.constexpr = _fp8_async_copy_layouts(
        BLOCK_DMODEL_POW2, BLOCK_N, num_warps, KV_STRIDE_ALIGN
    )
    kt_async_layout: gl.constexpr = _KV_ASYNC[0]
    v_async_layout: gl.constexpr = _KV_ASYNC[1]
    kt_off_d = gl.arange(
        0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(1, kt_async_layout)
    )
    kt_off_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kt_async_layout))
    kt_off = (kt_off_d[:, None] * stride_kk + kt_off_n[None, :] * stride_kn).to(
        gl.int32
    )
    v_off_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, v_async_layout))
    v_off_d = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, v_async_layout))
    v_off = (v_off_n[:, None] * stride_vn + v_off_d[None, :] * stride_vk).to(gl.int32)
    if PADDED_HEAD:
        k_head_mask = kt_off_d[:, None] < BLOCK_DMODEL
        v_head_mask = v_off_d[None, :] < BLOCK_DMODEL
    else:
        k_head_mask = None
        v_head_mask = None
    if ENABLE_SINK:
        m_i_init = gl.load(sink_ptr + off_q_head).to(gl.float32) * RCP_LN2
    elif SLIDING_WINDOW > 0:
        m_i_init = -1.0e30  # see _attn_fwd
    else:
        m_i_init = float("-inf")

    m_i = gl.full(
        [BLOCK_M], m_i_init, dtype=gl.float32, layout=gl.SliceLayout(1, mfmaLayout)
    )
    # l carries P's static lift; its initial 1.0 (the sink's term) does too.
    l_i = gl.full(
        [BLOCK_M],
        gl.exp2(_FP8_P_LIFT),
        dtype=gl.float32,
        layout=gl.SliceLayout(1, mfmaLayout),
    )
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=gl.float32, layout=mfmaLayout)

    offs_m = start_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout)
    )
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
            gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout)) * stride_lse_m
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
        offs_sd_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout))
        offs_sd_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfmaLayout))
        sd_offsets = (
            offs_sd_m[:, None] * stride_sd_m + offs_sd_n[None, :] * stride_sd_n
        ).to(gl.int32)
        sd_q_mask = offs_m < seqlen_q
    else:
        sd_base = None
        sd_offsets = None
        sd_q_mask = None

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
            # Built directly in _store_out's layout: converting it from the MFMA
            # layout would stage a [BLOCK_M, BLOCK_DMODEL] fp32 tile through LDS
            # while the K/V buffers are live, and the sum no longer fits two
            # workgroups per CU.
            zeros = gl.zeros(
                [BLOCK_M, BLOCK_DMODEL_POW2],
                dtype=gl.float32,
                layout=_make_load_layout(BLOCK_DMODEL_POW2, 8, num_warps, False),
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
                8,
                num_warps,
            )
            if softmax_lse_ptr is not None:
                lse = gl.zeros(
                    [BLOCK_M], dtype=gl.float32, layout=gl.SliceLayout(1, mfmaLayout)
                )
                gl.amd.cdna4.buffer_store(
                    lse, ptr=lse_base, offsets=offs_lse, mask=lse_mask
                )
            return
    block_min = skipped_blocks * BLOCK_N
    block_max = n_blocks * BLOCK_N

    if SLIDING_WINDOW > 0:
        k_base += skipped_blocks * BLOCK_N * stride_kn
        v_base += skipped_blocks * BLOCK_N * stride_vn
        if window_blocks > 0:
            acc, _, l_i, m_i = _attn_fwd_inner(
                acc,
                acc,
                l_i,
                m_i,
                q,
                None,
                k_base,
                v_base,
                smemK,
                None,
                smemV,
                kt_off,
                None,
                v_off,
                kt_off_n,
                kt_off_d,
                None,
                v_off_n,
                v_off_d,
                stride_kn,
                stride_vn,
                seqlen_k,
                block_min,
                block_min + window_blocks * BLOCK_N,
                window_min,
                qk_scale,
                sd_base,
                sd_offsets,
                sd_q_mask,
                stride_sd_n,
                mfmaLayout=mfmaLayout,
                dotK=dotK,
                dotP=dotP,
                dotV=dotV,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
                BLOCK_DMODEL_PE=0,
                HAS_PE=False,
                WINDOW_MASK=True,
                RETURN_SCORES=RETURN_SCORES,
                SCALE_ON_Q=False,
                BUF_DEPTH=BUF_DEPTH,
                D_LO=BLOCK_DMODEL_POW2,
                D_HI_QK=0,
                D_HI_PV=0,
                IS_FP8=True,
            )
            k_base += window_blocks * BLOCK_N * stride_kn
            v_base += window_blocks * BLOCK_N * stride_vn
            block_min = block_min + window_blocks * BLOCK_N

    # Full blocks take the rotated pipeline -- _attn_fwd's, in its fp8 form -- when
    # the copy is asynchronous and the workgroup has enough of them (_attn_fwd).
    FAST_PATH: gl.constexpr = not RETURN_SCORES
    pipelined = False
    if FAST_PATH:  # noqa: SIM102
        if n_full_blocks >= _MIN_PIPE_BLOCKS:
            pipelined = True
            acc, _, l_i, m_i = _attn_fwd_pipelined(
                acc,
                acc,
                l_i,
                m_i,
                q,
                None,
                smemK,
                None,
                smemV,
                k_base,
                v_base,
                kt_off,
                None,
                v_off,
                k_head_mask,
                v_head_mask,
                BLOCK_N * stride_kn,
                BLOCK_N * stride_vn,
                n_full_blocks,
                qk_scale,
                mfmaLayout=mfmaLayout,
                dotK=dotK,
                dotV=dotV,
                dotP=dotP,
                SCALE_ON_Q=False,
                DTYPE=v_ptr.dtype.element_ty,
                HAS_MASK=PADDED_HEAD,
                HAS_PE=False,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BUF_DEPTH=BUF_DEPTH,
                IS_FP8=True,
                P_LIFT=_FP8_P_LIFT,
            )
            block_min = block_min + n_full_blocks * BLOCK_N
            block_max = n_blocks * BLOCK_N
            gl.barrier()

    if (not pipelined) and n_full_blocks > 0:
        block_max = block_min + n_full_blocks * BLOCK_N
        acc, _, l_i, m_i = _attn_fwd_inner(
            acc,
            acc,
            l_i,
            m_i,
            q,
            None,
            k_base,
            v_base,
            smemK,
            None,
            smemV,
            kt_off,
            None,
            v_off,
            kt_off_n,
            kt_off_d,
            None,
            v_off_n,
            v_off_d,
            stride_kn,
            stride_vn,
            seqlen_k,
            block_min,
            block_max,
            window_min,
            qk_scale,
            sd_base,
            sd_offsets,
            sd_q_mask,
            stride_sd_n,
            mfmaLayout=mfmaLayout,
            dotK=dotK,
            dotP=dotP,
            dotV=dotV,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE=0,
            HAS_PE=False,
            WINDOW_MASK=False,
            RETURN_SCORES=RETURN_SCORES,
            SCALE_ON_Q=False,
            BUF_DEPTH=BUF_DEPTH,
            D_LO=BLOCK_DMODEL_POW2,
            D_HI_QK=0,
            D_HI_PV=0,
            IS_FP8=True,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    if masked_blocks > 0:
        k_base += n_full_blocks * BLOCK_N * stride_kn
        v_base += n_full_blocks * BLOCK_N * stride_vn
        acc, _, l_i, m_i = _attn_fwd_inner(
            acc,
            acc,
            l_i,
            m_i,
            q,
            None,
            k_base,
            v_base,
            smemK,
            None,
            smemV,
            kt_off,
            None,
            v_off,
            kt_off_n,
            kt_off_d,
            None,
            v_off_n,
            v_off_d,
            stride_kn,
            stride_vn,
            seqlen_k,
            block_min,
            block_max,
            window_min,
            qk_scale,
            sd_base,
            sd_offsets,
            sd_q_mask,
            stride_sd_n,
            mfmaLayout=mfmaLayout,
            dotK=dotK,
            dotP=dotP,
            dotV=dotV,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE=0,
            HAS_PE=False,
            WINDOW_MASK=SLIDING_WINDOW > 0,
            RETURN_SCORES=RETURN_SCORES,
            SCALE_ON_Q=False,
            BUF_DEPTH=BUF_DEPTH,
            D_LO=BLOCK_DMODEL_POW2,
            D_HI_QK=0,
            D_HI_PV=0,
            seqlen_q=seqlen_q,
            n_extra_tokens=n_extra_tokens,
            offs_m=offs_m,
            IS_CAUSAL=IS_CAUSAL,
            MASK_STEPS=True,
            IS_FP8=True,
        )

    # Epilogue: the lift cancels in acc / l; descale_v is applied here, once.
    acc = acc * (descale_v / l_i)[:, None]

    causal_start_idx = seqlen_q - seqlen_k
    start_m_idx = start_m * BLOCK_M
    if IS_CAUSAL:  # noqa: SIM102
        if (causal_start_idx > start_m_idx) and (
            causal_start_idx < start_m_idx + BLOCK_M
        ):
            acc = gl.where((offs_m >= causal_start_idx)[:, None], acc, 0.0)

    if softmax_lse_ptr is not None:
        LN2: gl.constexpr = 0.6931471824645996
        softmax_lse = (m_i + gl.log2(l_i) - _FP8_P_LIFT) * LN2
        if IS_CAUSAL:
            softmax_lse = gl.where(offs_m < causal_start_idx, 0.0, softmax_lse)
        gl.amd.cdna4.buffer_store(
            softmax_lse, ptr=lse_base, offsets=offs_lse, mask=lse_mask
        )

    # Stored straight from the MFMA layout, where each lane already holds 4
    # consecutive output columns -- a 16 B store.  _store_out's conversion to a
    # blocked layout would stage the whole fp32 tile through LDS: 64 KB at a
    # 128-wide head, 128 KB at 256, which caps a CU at one workgroup.
    offs_od = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, mfmaLayout))
    o_offsets = (
        (offs_m - start_m_idx)[:, None] * stride_om + offs_od[None, :] * stride_on
    ).to(gl.int32)
    o_mask = (offs_m < seqlen_q)[:, None]
    if BLOCK_DMODEL_POW2 > BLOCK_DMODEL_OUT:
        o_mask = o_mask & (offs_od[None, :] < BLOCK_DMODEL_OUT)
    gl.amd.cdna4.buffer_store(
        acc.to(o_ptr.dtype.element_ty), ptr=o_base, offsets=o_offsets, mask=o_mask
    )


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
