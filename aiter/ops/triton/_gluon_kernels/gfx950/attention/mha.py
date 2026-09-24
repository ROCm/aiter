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

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.mha_kernel_utils import (
    _compute_fp8_scaling_factors,
)
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd
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
    index, then its low bits -- that middle group is the stagger. ``pad`` differs per
    tile because the reads do: K^T is read along its contiguous axis
    (``ds_read_b128``) while V is read across it by the hardware-transposing
    ``ds_read_b64_tr_b16``.
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
    padding intervals. Anything else keeps the old swizzled layout.
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
def _make_kv_shared_layouts(
    head_dim_pow2, elem_bytes, k_width=8, non_k_dim=16, banks=64, block_n=0
):
    """LDS layouts for the K/V staging tiles.

    Prefers the conflict-free row-staggered padded layout; falls back to the
    analytic swizzle for tiles it cannot describe (fp8, non-power-of-two extents).
    """
    if block_n and _staggered_layout_ok(head_dim_pow2, block_n, elem_bytes):
        k_shared = _padded_staggered_layout(head_dim_pow2, block_n, 0, pad=8)
        v_shared = _padded_staggered_layout(block_n, head_dim_pow2, 1, pad=32)
        return k_shared, v_shared

    bank_line_bytes = banks * 4
    bank_line_elems = bank_line_bytes // elem_bytes
    read_vec_bytes = min(k_width * elem_bytes, 16)
    num_threads_same_cycle = bank_line_bytes // read_vec_bytes
    per_phase = (bank_line_elems + head_dim_pow2 - 1) // head_dim_pow2
    swizzle_vec = min(k_width * max(1, per_phase // 2), read_vec_bytes // elem_bytes)
    max_phase = min(
        min(non_k_dim, num_threads_same_cycle) // per_phase,
        bank_line_elems // swizzle_vec,
    )
    k_shared = gl.SwizzledSharedLayout(swizzle_vec, per_phase, max_phase, order=[0, 1])
    v_shared = gl.SwizzledSharedLayout(swizzle_vec, per_phase, max_phase, order=[1, 0])
    return k_shared, v_shared


@gluon.constexpr_function
def _async_copy_vec(rows, cols, contig_dim, num_warps, elem_bits, warp_size=64):
    """Elements per lane a global->LDS async copy can move for this tile, or 0 if none can.

    ``buffer_load_to_shared`` wants every lane to carry a full 128 bits, and the
    resulting LDS writes have to coalesce against the padded destination. Tiles that
    cannot satisfy that keep the buffer_load + ds_write path.
    """
    c, _ = _contig_strided(rows, cols, contig_dim)
    n_threads = num_warps * warp_size
    if (rows * cols) % n_threads != 0:
        return 0
    per_lane = rows * cols // n_threads
    vec = 128 // elem_bits
    if vec < 1 or vec > per_lane or vec > c:
        return 0
    if (c // vec) > warp_size:
        return 0
    r_lane_bits = _bits(warp_size) - _bits(c // vec)
    if _bits(rows * cols // c) < _bits(num_warps) + r_lane_bits:
        return 0
    return vec


@gluon.constexpr_function
def _async_copy_layout(rows, cols, contig_dim, num_warps, vec, warp_size=64):
    """Address layout for a global->LDS async copy of a ``[rows, cols]`` tile.

    ``buffer_load_to_shared`` needs each lane to name ``vec`` *contiguous* elements
    and needs the resulting LDS writes to coalesce. Only one bit order satisfies
    both: the low ``log2(vec)`` bits of the contiguous axis go to registers, the rest
    of that axis to lanes, and the strided axis splits low-to-high across warps, then
    registers, then whatever lane bits the contiguous axis did not need.
    """
    c, r = _contig_strided(rows, cols, contig_dim)

    def base(cv, rv):
        return [cv, rv] if contig_dim == 0 else [rv, cv]

    c_lane_bits = _bits(c // vec)
    c_reg_bits = _bits(c) - c_lane_bits
    r_warp_bits = _bits(num_warps)
    r_lane_bits = _bits(warp_size) - c_lane_bits
    r_reg_bits = _bits(r) - r_warp_bits - r_lane_bits

    reg = [base(1 << i, 0) for i in range(c_reg_bits)]
    reg += [base(0, 1 << (r_warp_bits + i)) for i in range(r_reg_bits)]
    lane = [base(1 << (c_reg_bits + i), 0) for i in range(c_lane_bits)]
    lane += [base(0, 1 << (r_warp_bits + r_reg_bits + i)) for i in range(r_lane_bits)]
    warp = [base(0, 1 << i) for i in range(r_warp_bits)]
    return gl.DistributedLinearLayout(
        reg_bases=reg,
        lane_bases=lane,
        warp_bases=warp,
        block_bases=[],
        shape=[rows, cols],
    )


@gluon.constexpr_function
def _async_copy_ok(head_dim_pow2, block_n, num_warps, elem_bits, stride_align):
    """Can BOTH the K^T and the V tile be moved by ``buffer_load_to_shared``?

    All-or-nothing on purpose: a loop that async-copies one tile and register-stages the
    other still pays the blocking ``s_waitcnt vmcnt(0)`` for the second.

    ``stride_align`` is the largest power of two, in elements, that the host has
    verified divides both KV sequence strides -- and 0 when it could not verify
    anything, which is what the host reports for a tensor whose last axis is not
    contiguous.  It has to reach the copy's vector width, because that is what makes
    every lane's 16-byte chunk 16-byte aligned; see the note at the
    ``gl.multiple_of`` call.  Below it the copy is not merely unprofitable, it is
    wrong, so 0 disqualifies.
    """
    if not _staggered_layout_ok(head_dim_pow2, block_n, elem_bits // 8):
        return False
    vec = _async_copy_vec(head_dim_pow2, block_n, 0, num_warps, elem_bits)
    if vec == 0:
        return False
    if stride_align < vec:
        return False
    # K^T is [head_dim, block_n] with dim 0 contiguous; V is its transpose.
    return _async_copy_vec(block_n, head_dim_pow2, 1, num_warps, elem_bits) != 0


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
def _buffer_load_2d(
    base, offsets, offset_first, offset_second, boundary_first, boundary_second
):
    """buffer_load of one tile into registers; masked lanes read 0."""
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (
            offset_second[None, :] < boundary_second
        )
        tile = gl.amd.cdna4.buffer_load(ptr=base, offsets=offsets, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tile = gl.amd.cdna4.buffer_load(ptr=base, offsets=offsets, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tile = gl.amd.cdna4.buffer_load(ptr=base, offsets=offsets, mask=mask, other=0.0)
    else:
        tile = gl.amd.cdna4.buffer_load(ptr=base, offsets=offsets)
    return tile


@gluon.jit
def _load_k(
    k_base,
    k_offsets,
    k_pe_offsets,
    load_start_n,
    seqlen_k,
    kLoadLayout: gl.constexpr,
    kPeLoadLayout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,
    MASK_STEPS: gl.constexpr,
    PADDED_HEAD: gl.constexpr,
    HAS_PE: gl.constexpr,
):
    """buffer_load one K block ([BLOCK_DMODEL_POW2, BLOCK_N]), and its PE ([BLOCK_DMODEL_PE, BLOCK_N]) when PE is on (else None)."""
    k_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kLoadLayout))
    k_offs_d = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(1, kLoadLayout))
    if MASK_STEPS:
        k_n = load_start_n + k_offs_n
    else:
        k_n = None
    if PADDED_HEAD:
        k_d = k_offs_d
    else:
        k_d = None
    k_tile = _buffer_load_2d(k_base, k_offsets, k_d, k_n, BLOCK_DMODEL, seqlen_k)

    if HAS_PE:
        if MASK_STEPS:
            k_pe_n = load_start_n + gl.arange(
                0, BLOCK_N, layout=gl.SliceLayout(0, kPeLoadLayout)
            )
        else:
            k_pe_n = None
        return k_tile, _buffer_load_2d(
            k_base, k_pe_offsets, None, k_pe_n, BLOCK_DMODEL_PE, seqlen_k
        )
    else:
        return k_tile, None


@gluon.jit
def _load_v(
    v_base,
    v_offsets,
    load_start_n,
    seqlen_k,
    vLoadLayout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    MASK_STEPS: gl.constexpr,
    PADDED_HEAD: gl.constexpr,
):
    """buffer_load one V block ([BLOCK_N, BLOCK_DMODEL_POW2]) into registers."""
    v_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, vLoadLayout))
    v_offs_d = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, vLoadLayout))
    if MASK_STEPS:
        v_n = load_start_n + v_offs_n
    else:
        v_n = None
    if PADDED_HEAD:
        v_d = v_offs_d
    else:
        v_d = None
    return _buffer_load_2d(v_base, v_offsets, v_n, v_d, seqlen_k, BLOCK_DMODEL)


@gluon.jit
def _async_copy_tile(smem, base, offsets, mask, HAS_MASK: gl.constexpr):
    """Issue one global->LDS async copy tile, WITHOUT closing a commit group.

    The copy never passes through VGPRs, so it needs no ``s_waitcnt vmcnt(0)`` and
    no ``ds_write``; the wave issues it and walks away.
    """
    if HAS_MASK:
        cdna4_async.buffer_load_to_shared(smem, base, offsets, mask=mask, other=0.0)
    else:
        cdna4_async.buffer_load_to_shared(smem, base, offsets)


@gluon.jit
def _async_copy_k_group(
    smemK,
    smemKpe,
    k_base,
    kt_off,
    kpe_off,
    k_mask,
    kpe_mask,
    HAS_MASK: gl.constexpr,
    HAS_PE: gl.constexpr,
):
    """Stage K and, when present, its PE slice as a SINGLE commit group.

    Grouping them matters: every ``wait_group`` count in the pipeline is expressed
    in commit groups, so letting the PE slice open its own group would shift all of
    them.  K and its PE slice are consumed by the same MFMA chain in the same
    cluster, so there is never a reason to wait for one without the other.
    """
    _async_copy_tile(smemK, k_base, kt_off, k_mask, HAS_MASK)
    if HAS_PE:
        _async_copy_tile(smemKpe, k_base, kpe_off, kpe_mask, kpe_mask is not None)
    cdna4_async.commit_group()


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
    """Stage one K (+PE) and V tile into LDS slot ``slot`` with buffer_load_to_shared.

    The KV-token mask is only built on the masked blocks; the head-dim mask is a
    property of the tensor, so it rides along on every block when the head dim is
    padded. ``buffer_load_to_shared`` broadcasts a mask against the offsets, so each
    half is handed over carrying only the axis it constrains.

    The PE slice is a third tile with its own LDS buffer and its own commit group;
    ``ASYNC_GROUPS`` in the caller counts it.
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
    _async_copy_k_group(
        smemK.index(slot),
        smemKpe.index(slot) if HAS_PE else smemK.index(slot),
        k_base,
        kt_off,
        kpe_off,
        k_mask,
        kpe_mask,
        HAS_MASK,
        HAS_PE,
    )
    _async_copy_tile(smemV.index(slot), v_base, v_off, v_mask, HAS_MASK)
    cdna4_async.commit_group()  # ACV


@gluon.jit
def _store_k_smem(smemK, smemKpe, k_tile, k_pe_tile, HAS_PE: gl.constexpr):
    smemK.store(k_tile)
    if HAS_PE:
        smemKpe.store(k_pe_tile)


@gluon.jit
def _load_k_smem(smemK, smemKpe, dotK: gl.constexpr, HAS_PE: gl.constexpr):
    k = smemK.load(dotK)
    if HAS_PE:
        return k, smemKpe.load(dotK)
    else:
        return k, None


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
    SLIDING_WINDOW: gl.constexpr,
    SCALE_ON_Q: gl.constexpr = False,
    seqlen_q=None,
    seqlen_k=None,
    block_max=None,
    n_extra_tokens=None,
    offs_m=None,
    IS_CAUSAL: gl.constexpr = False,
    MASK_STEPS: gl.constexpr = False,
):
    """QK^T + scale + mask for one already-staged key block. ``k`` is already in
    its MFMA dot-operand layout; returns float32 scores in ``mfmaLayout``. For
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

    if MASK_STEPS or IS_CAUSAL or SLIDING_WINDOW > 0:
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
        if SLIDING_WINDOW > 0:
            mask = mask & (window_min[:, None] <= key_pos[None, :])
        qk = gl.where(mask, qk, float("-inf"))

    return qk


@gluon.jit
def _attn_softmax_pv(
    acc,
    l_i,
    m_i,
    qk,
    v,
    descale_v,
    sd_base,
    sd_offsets,
    sd_q_mask,
    offs_n,
    start_n,
    seqlen_k,
    stride_sd_n,
    dotP: gl.constexpr,
    mfmaLayout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    IS_FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
    RETURN_SCORES: gl.constexpr,
):
    """Online-softmax rescale + P@V accumulation for one key block.
    Returns updated (acc, l_i, m_i)."""

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

    if IS_FP8:
        scale_p, descale_p = _compute_fp8_scaling_factors(p, FP8_MAX)
        p = gl.convert_layout(
            (p * scale_p).to(v.dtype), layout=dotP, assert_trivial=True
        )
        pv = gl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=gl.float32, layout=mfmaLayout)
        pv = gl.amd.cdna4.mfma_scaled(p, None, "e4m3", v, None, "e4m3", pv)
        acc = acc + pv * (descale_p * descale_v)
    else:
        p = gl.convert_layout(p.to(v.dtype), layout=dotP, assert_trivial=True)
        acc = gl.amd.cdna4.mfma(p, v, acc)

    l_i = l_i * alpha + l_ij
    m_i = m_ij

    return acc, l_i, m_i


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


@gluon.jit
def _sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q: gl.constexpr):
    """VEC1 -- softmax numerator: new row max, the exp2 burst, and alpha.

    Placed in the ``dot2`` cluster.  ``exp2`` is a TRANS op and issues at half the
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
        p = gl.exp2(gl.fma(qk, qk_scale, -m_new[:, None]))
    alpha = gl.exp2(m_run - m_new)
    return m_new, p, alpha


@gluon.jit
def _sc_vec2(acc, l_i, p, alpha, dotP: gl.constexpr, DTYPE: gl.constexpr):
    """VEC2 -- denominator, accumulator rescale, and the P operand downcast.

    Placed in the ``dot1`` cluster, beside the Q@K^T MFMA; ``p`` and ``alpha`` were
    produced by VEC1 in the *previous* iteration.  The rescale goes first: this
    cluster's shadow cannot absorb all of it, and leading with it keeps the uncovered
    remainder ahead of the MFMAs, where an exposed packed op pays no back-to-back
    hazard.
    """
    acc = acc * alpha[:, None]
    l_ij = gl.sum(p, axis=1)
    l_i = l_i * alpha + l_ij
    p_dot = gl.convert_layout(p.to(DTYPE), dotP)
    return acc, l_i, p_dot


@gluon.jit
def _attn_fwd_pipelined(
    acc,
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
):
    """Run tiles ``[0, n_full_blocks)`` through the rotated pipeline.

    ``k_base`` / ``v_base`` point at tile 0 and every tile index is an offset from
    them, so the address VALU stays out of the loop entirely.
    """

    # -- Prologue: prime the rotation for tile 0 ---------------------------------
    # Compute all of tile 0's ahead-work (qk[0], m[0], p[0], alpha[0]) and the K
    # registers for tile 1, and stage K[0..2] / V[0..1].  K runs three tiles ahead,
    # so slot 0 is reused for K[2] once tile 0's read is done -- hence the barrier.
    # Commit order K0, V0, K1, K2, V1 leaves {K2, V1} pending, which is the loop's
    # steady-state entry condition.
    _async_copy_k_group(
        smemK.index(0),
        smemKpe.index(0) if HAS_PE else smemK.index(0),
        k_base,
        kt_off,
        kpe_off,
        k_mask,
        None,
        HAS_MASK,
        HAS_PE,
    )
    _async_copy_tile(smemV.index(0), v_base, v_off, v_mask, HAS_MASK)
    cdna4_async.commit_group()  # ACV
    _async_copy_k_group(
        smemK.index(1),
        smemKpe.index(1) if HAS_PE else smemK.index(1),
        k_base + kt_step,
        kt_off,
        kpe_off,
        k_mask,
        None,
        HAS_MASK,
        HAS_PE,
    )

    cdna4_async.wait_group(2)  # K[0] has landed
    kt0 = cdna4_async.load_shared_relaxed(smemK.index(0), dotK)
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfmaLayout)
    if HAS_PE:
        kpe0 = cdna4_async.load_shared_relaxed(smemKpe.index(0), dotK)
        qk = gl.amd.cdna4.mfma(q_pe, kpe0, qk)
    qk = gl.amd.cdna4.mfma(q_dot, kt0, qk)
    m_run, p_c, alpha_c = _sc_vec1(qk, m_i, qk_scale, SCALE_ON_Q)

    gl.barrier()  # WAR: tile 0's ds_read against K[2]'s write into the same slot
    _async_copy_k_group(
        smemK.index(0),
        smemKpe.index(0) if HAS_PE else smemK.index(0),
        k_base + 2 * kt_step,
        kt_off,
        kpe_off,
        k_mask,
        None,
        HAS_MASK,
        HAS_PE,
    )
    cdna4_async.wait_group(1)  # K[1] has landed
    kt_dot = cdna4_async.load_shared_relaxed(smemK.index(1), dotK)
    if HAS_PE:
        kpe_dot = cdna4_async.load_shared_relaxed(smemKpe.index(1), dotK)
    else:
        kpe_dot = kt_dot
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
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfmaLayout)
            if HAS_PE:
                qk = gl.amd.cdna4.mfma(q_pe, kpe_dot, qk)
            qk = gl.amd.cdna4.mfma(q_dot, kt_dot, qk)  # dot_qk
            acc, l_i, p_dot = _sc_vec2(acc, l_i, p_c, alpha_c, dotP, DTYPE)  # VEC2
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem1"):
            v_dot = cdna4_async.load_shared_relaxed(smemV.index(0), dotV)  # LRV
            _async_copy_k_group(  # ACK
                smemK.index(1),
                smemKpe.index(1) if HAS_PE else smemK.index(1),
                k_base + (blk + 3) * kt_step,
                kt_off,
                kpe_off,
                k_mask,
                None,
                HAS_MASK,
                HAS_PE,
            )
        with warp_pipeline_stage("dot2"):
            acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)  # dot_pv
            m_run, p_c, alpha_c = _sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q)  # VEC1
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem2"):
            kt_dot = cdna4_async.load_shared_relaxed(smemK.index(0), dotK)  # LRK
            if HAS_PE:
                kpe_dot = cdna4_async.load_shared_relaxed(smemKpe.index(0), dotK)
            else:
                # A loop-carried value has to be a real tensor in every instantiation,
                # so without a PE slice this aliases K rather than carrying None.
                kpe_dot = kt_dot
            _async_copy_tile(  # ACV
                smemV.index(0), v_base + (blk + 2) * v_step, v_off, v_mask, HAS_MASK
            )
            cdna4_async.commit_group()

        # odd tile (blk + 1): LDS slots cur=1, next=0
        with warp_pipeline_stage("dot1"):
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfmaLayout)
            if HAS_PE:
                qk = gl.amd.cdna4.mfma(q_pe, kpe_dot, qk)
            qk = gl.amd.cdna4.mfma(q_dot, kt_dot, qk)  # dot_qk
            acc, l_i, p_dot = _sc_vec2(acc, l_i, p_c, alpha_c, dotP, DTYPE)  # VEC2
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem1"):
            v_dot = cdna4_async.load_shared_relaxed(smemV.index(1), dotV)  # LRV
            _async_copy_k_group(  # ACK
                smemK.index(0),
                smemKpe.index(0) if HAS_PE else smemK.index(0),
                k_base + (blk + 1 + 3) * kt_step,
                kt_off,
                kpe_off,
                k_mask,
                None,
                HAS_MASK,
                HAS_PE,
            )
        with warp_pipeline_stage("dot2"):
            acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)  # dot_pv
            m_run, p_c, alpha_c = _sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q)  # VEC1
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem2"):
            kt_dot = cdna4_async.load_shared_relaxed(smemK.index(1), dotK)  # LRK
            if HAS_PE:
                kpe_dot = cdna4_async.load_shared_relaxed(smemKpe.index(1), dotK)
            else:
                # A loop-carried value has to be a real tensor in every instantiation,
                # so without a PE slice this aliases K rather than carrying None.
                kpe_dot = kt_dot
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
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfmaLayout)
            if HAS_PE:
                qk = gl.amd.cdna4.mfma(q_pe, kpe_dot, qk)
            qk = gl.amd.cdna4.mfma(q_dot, kt_dot, qk)  # dot_qk
            acc, l_i, p_dot = _sc_vec2(acc, l_i, p_c, alpha_c, dotP, DTYPE)  # VEC2
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem1"):
            v_dot = cdna4_async.load_shared_relaxed(smemV.index(0), dotV)  # LRV
            _async_copy_k_group(  # ACK
                smemK.index(1),
                smemKpe.index(1) if HAS_PE else smemK.index(1),
                k_base + (blk + 3) * kt_step,
                kt_off,
                kpe_off,
                k_mask,
                None,
                HAS_MASK,
                HAS_PE,
            )
        with warp_pipeline_stage("dot2"):
            acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)  # dot_pv
            m_run, p_c, alpha_c = _sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q)  # VEC1
        cdna4_async.wait_group(1)
        with warp_pipeline_stage("mem2"):
            kt_dot = cdna4_async.load_shared_relaxed(smemK.index(0), dotK)  # LRK
            if HAS_PE:
                kpe_dot = cdna4_async.load_shared_relaxed(smemKpe.index(0), dotK)
            else:
                # A loop-carried value has to be a real tensor in every instantiation,
                # so without a PE slice this aliases K rather than carrying None.
                kpe_dot = kt_dot
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

    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfmaLayout)
    if HAS_PE:
        qk = gl.amd.cdna4.mfma(q_pe, kpe_dot, qk)
    qk = gl.amd.cdna4.mfma(q_dot, kt_dot, qk)
    cdna4_async.wait_group(2)
    v_dot = cdna4_async.load_shared_relaxed(smemV.index(s_nm3), dotV)
    acc, l_i, p_dot = _sc_vec2(acc, l_i, p_c, alpha_c, dotP, DTYPE)
    acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)
    m_run, p_c, alpha_c = _sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q)
    gl.barrier()  # WAR: tile n-3's V read against V[n-1]'s write into that slot
    _async_copy_tile(smemV.index(s_nm1), v_base + nm1 * v_step, v_off, v_mask, HAS_MASK)
    cdna4_async.commit_group()  # ACV
    cdna4_async.wait_group(2)
    kt_dot = cdna4_async.load_shared_relaxed(smemK.index(s_nm1), dotK)
    if HAS_PE:
        kpe_dot = cdna4_async.load_shared_relaxed(smemKpe.index(s_nm1), dotK)

    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfmaLayout)
    if HAS_PE:
        qk = gl.amd.cdna4.mfma(q_pe, kpe_dot, qk)
    qk = gl.amd.cdna4.mfma(q_dot, kt_dot, qk)
    cdna4_async.wait_group(1)
    v_dot = cdna4_async.load_shared_relaxed(smemV.index(s_nm2), dotV)
    acc, l_i, p_dot = _sc_vec2(acc, l_i, p_c, alpha_c, dotP, DTYPE)
    acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)
    m_run, p_c, alpha_c = _sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q)

    cdna4_async.wait_group(0)
    v_dot = cdna4_async.load_shared_relaxed(smemV.index(s_nm1), dotV)
    acc, l_i, p_dot = _sc_vec2(acc, l_i, p_c, alpha_c, dotP, DTYPE)
    acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)

    return acc, l_i, m_run


@gluon.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_pe,
    k_base,
    k_offsets,
    k_pe_offsets,
    v_base,
    v_offsets,
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
    descale_v,
    sd_base,
    sd_offsets,
    sd_q_mask,
    stride_sd_n,
    mfmaLayout: gl.constexpr,
    dotK: gl.constexpr,
    dotP: gl.constexpr,
    dotV: gl.constexpr,
    kLoadLayout: gl.constexpr,
    kPeLoadLayout: gl.constexpr,
    vLoadLayout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,
    HAS_PE: gl.constexpr,
    IS_FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    RETURN_SCORES: gl.constexpr,
    SCALE_ON_Q: gl.constexpr,
    USE_ASYNC_COPY: gl.constexpr,
    BUF_DEPTH: gl.constexpr,
    seqlen_q=None,
    n_extra_tokens=None,
    offs_m=None,
    IS_CAUSAL: gl.constexpr = False,
    MASK_STEPS: gl.constexpr = False,
):
    """
    Inner loop for attention forward pass computation.
    """
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2

    if MASK_STEPS or IS_CAUSAL or SLIDING_WINDOW > 0 or RETURN_SCORES:
        offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfmaLayout))
    else:
        offs_n = None

    n_iter = (block_max - block_min) // BLOCK_N

    # Commit groups the async-copy path issues per tile: one for K (carrying the PE slice
    # with it when present), one for V.
    ASYNC_GROUPS: gl.constexpr = 2

    if USE_ASYNC_COPY:
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

        if USE_ASYNC_COPY:
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
            k = cdna4_async.load_shared_relaxed(smemK.index(cur), dotK)
            if HAS_PE:
                k_pe = cdna4_async.load_shared_relaxed(smemKpe.index(cur), dotK)
            else:
                k_pe = None
            v = cdna4_async.load_shared_relaxed(smemV.index(cur), dotV)
        else:
            cur = 0
            k_tile, k_pe_tile = _load_k(
                k_base + i * BLOCK_N * stride_kn,
                k_offsets,
                k_pe_offsets,
                start_n,
                seqlen_k,
                kLoadLayout,
                kPeLoadLayout,
                BLOCK_N,
                BLOCK_DMODEL,
                BLOCK_DMODEL_POW2,
                BLOCK_DMODEL_PE,
                MASK_STEPS,
                PADDED_HEAD,
                HAS_PE,
            )
            v_tile = _load_v(
                v_base + i * BLOCK_N * stride_vn,
                v_offsets,
                start_n,
                seqlen_k,
                vLoadLayout,
                BLOCK_N,
                BLOCK_DMODEL,
                BLOCK_DMODEL_POW2,
                MASK_STEPS,
                PADDED_HEAD,
            )
            _store_k_smem(
                smemK.index(0),
                smemKpe.index(0) if HAS_PE else None,
                k_tile,
                k_pe_tile,
                HAS_PE,
            )
            smemV.index(0).store(v_tile)
            k, k_pe = _load_k_smem(
                smemK.index(0), smemKpe.index(0) if HAS_PE else None, dotK, HAS_PE
            )
            v = smemV.index(0).load(dotV)

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
            HAS_PE=HAS_PE,
            IS_FP8=IS_FP8,
            SLIDING_WINDOW=SLIDING_WINDOW,
            SCALE_ON_Q=SCALE_ON_Q,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_max=block_max,
            n_extra_tokens=n_extra_tokens,
            offs_m=offs_m,
            IS_CAUSAL=IS_CAUSAL,
            MASK_STEPS=MASK_STEPS,
        )
        acc, l_i, m_i = _attn_softmax_pv(
            acc,
            l_i,
            m_i,
            qk,
            v,
            descale_v,
            sd_base,
            sd_offsets,
            sd_q_mask,
            offs_n,
            start_n,
            seqlen_k,
            stride_sd_n,
            dotP,
            mfmaLayout,
            BLOCK_M,
            BLOCK_DMODEL_POW2,
            IS_FP8,
            FP8_MAX,
            RETURN_SCORES,
        )

        if USE_ASYNC_COPY:
            # WAR: this tile's LDS reads against the copy that will overwrite the
            # same slot two iterations from now.
            gl.barrier()

    if USE_ASYNC_COPY:
        cdna4_async.wait_group(0)

    return acc, l_i, m_i


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
    RCP_LN2: gl.constexpr = 1.4426950408889634
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2
    PADDED_HEAD_OUT: gl.constexpr = BLOCK_DMODEL_OUT != BLOCK_DMODEL_POW2
    HAS_PE: gl.constexpr = BLOCK_DMODEL_PE > 0

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
        if IS_FP8:
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

    # The global->LDS copy hands every lane 16 contiguous bytes, and its lowering
    # only fires when it can PROVE that chunk is 16-byte aligned.  A lane's chunk
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
    pid = gl.program_id(axis=0)
    off_q_head = pid % NUM_Q_HEADS
    # Remap the q-head index across XCDs for better cache locality.
    off_q_head = remap_xcd(off_q_head, NUM_Q_HEADS, NUM_XCD)
    start_m = (pid // NUM_Q_HEADS) % NUM_BLOCKS_M
    off_z = pid // (NUM_Q_HEADS * NUM_BLOCKS_M) % BATCH

    if IS_CAUSAL:
        # Causal work is wildly uneven across M: with a bottom-right aligned mask,
        # M-block m walks ~(m+1) * BLOCK_M / BLOCK_N key blocks, so the last M-block
        # does tens of times the work of the first.  In index order the biggest
        # workgroups are also the last to be scheduled, and the dispatch ends with a
        # few huge workgroups running while most CUs sit idle.
        #
        # Reversing the M index is longest-processing-time-first scheduling: the big
        # workgroups start immediately and the short ones fill the gaps as CUs free
        # up.  Worth +6% to +16% here.
        #
        # It is deliberately a *monotone* remap.  An interleaved one (M-1, 0, M-2,
        # 1, ...) balances each scheduling round exactly and measures 5% SLOWER,
        # because M-block m and m+1 read nested K/V ranges and share L2 lines --
        # pairing a long block with a short one breaks that locality, and the lost
        # bandwidth costs more than the balance buys.
        start_m = NUM_BLOCKS_M - 1 - start_m

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

    if IS_FP8:
        descale_q = gl.load(descale_q_ptr + off_z * stride_descale_q_z + off_q_head)
        descale_k = gl.load(descale_k_ptr + off_z * stride_descale_k_z + off_k_head)
        descale_v = gl.load(descale_v_ptr + off_z * stride_descale_v_z + off_k_head)
    else:
        descale_q = 1.0
        descale_k = 1.0
        descale_v = 1.0

    MFMA_INSTR: gl.constexpr = [32, 32, 64] if IS_FP8 else [32, 32, 16]
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=MFMA_INSTR,
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )

    K_WIDTH: gl.constexpr = 16 if IS_FP8 else 8
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

    LOAD_VEC: gl.constexpr = 16 if IS_FP8 else 8
    qLoadLayout: gl.constexpr = _make_load_layout(
        BLOCK_DMODEL_POW2, LOAD_VEC, num_warps, transposed=False
    )
    kLoadLayout: gl.constexpr = _make_load_layout(
        BLOCK_DMODEL_POW2, LOAD_VEC, num_warps, transposed=True
    )
    vLoadLayout: gl.constexpr = _make_load_layout(
        BLOCK_DMODEL_POW2, LOAD_VEC, num_warps, transposed=False
    )

    # Shared layouts for the LDS staging of K and V.
    ELEM_BYTES: gl.constexpr = k_ptr.dtype.element_ty.primitive_bitwidth // 8
    ELEM_BITS: gl.constexpr = k_ptr.dtype.element_ty.primitive_bitwidth
    _KV_SHARED: gl.constexpr = _make_kv_shared_layouts(
        BLOCK_DMODEL_POW2, ELEM_BYTES, k_width=K_WIDTH, block_n=BLOCK_N
    )

    # Can the K/V tiles go global->LDS with buffer_load_to_shared, skipping VGPRs entirely?
    # All-or-nothing, including the PE slice: one register-staged tile reinstates the
    # blocking `s_waitcnt vmcnt(0)` the async copy exists to remove.
    # Can the K/V tiles go global->LDS with buffer_load_to_shared, skipping VGPRs entirely?
    # All-or-nothing, including the PE slice: one register-staged tile reinstates the
    # blocking `s_waitcnt vmcnt(0)` the async copy exists to remove.  The gate covers the
    # three things the copy's lowering actually requires -- a tile shape it can split
    # 128 bits per lane, a destination it can write coalesced (the staggered padded
    # layout, which also needs 16-bit elements), and a KV sequence stride whose
    # alignment reaches the vector width.  Falling any of them keeps the
    # buffer_load + ds_write staging, which is correct, just slower.
    USE_ASYNC_COPY: gl.constexpr = _async_copy_ok(
        BLOCK_DMODEL_POW2, BLOCK_N, num_warps, ELEM_BITS, KV_STRIDE_ALIGN
    ) and (
        (not HAS_PE)
        or _async_copy_ok(
            BLOCK_DMODEL_PE, BLOCK_N, num_warps, ELEM_BITS, KV_STRIDE_ALIGN
        )
    )
    # Double buffering only earns its LDS when the copy is asynchronous.
    BUF_DEPTH: gl.constexpr = 2 if USE_ASYNC_COPY else 1
    kSharedLayout: gl.constexpr = _KV_SHARED[0]
    vSharedLayout: gl.constexpr = _KV_SHARED[1]

    if HAS_PE:
        qPeLoadLayout: gl.constexpr = _make_load_layout(
            BLOCK_DMODEL_PE, LOAD_VEC, num_warps, transposed=False
        )
        kPeLoadLayout: gl.constexpr = _make_load_layout(
            BLOCK_DMODEL_PE, LOAD_VEC, num_warps, transposed=True
        )
        # block_n matters: without it this falls back to the analytic swizzle, and a
        # swizzled destination is not a legal target for buffer_load_to_shared unless
        # the swizzle stays inside a warp boundary.  The PE tile needs the same
        # staggered padded layout the K/V tiles get.
        _KPE_SHARED: gl.constexpr = _make_kv_shared_layouts(
            BLOCK_DMODEL_PE, ELEM_BYTES, k_width=K_WIDTH, block_n=BLOCK_N
        )
        kPeSharedLayout: gl.constexpr = _KPE_SHARED[0]
    else:
        qPeLoadLayout: gl.constexpr = None
        kPeLoadLayout: gl.constexpr = None
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
    if IS_FP8:
        qk_scale = qk_scale * descale_q * descale_k

    # Load Q (stays resident for the whole key loop).
    offs_qm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, qLoadLayout))
    offs_qd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, qLoadLayout))
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
    if PADDED_HEAD:
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
    if SCALE_ON_Q:
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
        if SCALE_ON_Q:
            q_pe = (q_pe.to(gl.float32) * qk_scale).to(q_ptr.dtype.element_ty)
    else:
        q_pe = None

    offs_kd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(1, kLoadLayout))
    offs_kn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kLoadLayout))
    k_base = k_ptr + off_z * stride_kz + kh_off + cu_seqlens_k_start * stride_kn
    k_offsets = (offs_kd[:, None] * stride_kk + offs_kn[None, :] * stride_kn).to(
        gl.int32
    )

    if HAS_PE:
        offs_kpd = BLOCK_DMODEL + gl.arange(
            0, BLOCK_DMODEL_PE, layout=gl.SliceLayout(1, kPeLoadLayout)
        )
        offs_kpn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kPeLoadLayout))
        k_pe_offsets = (
            offs_kpd[:, None] * stride_kk + offs_kpn[None, :] * stride_kn
        ).to(gl.int32)
    else:
        k_pe_offsets = None

    offs_vn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, vLoadLayout))
    offs_vd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, vLoadLayout))
    v_base = v_ptr + off_z * stride_vz + vh_off + cu_seqlens_k_start * stride_vn
    v_offsets = (offs_vn[:, None] * stride_vn + offs_vd[None, :] * stride_vk).to(
        gl.int32
    )

    # Shared-memory tiles for the K/V staging, double-buffered on the async-copy path so a
    # copy for tile i+1 can be in flight while tile i is being consumed.
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
    if USE_ASYNC_COPY:
        KV_VEC: gl.constexpr = _async_copy_vec(
            BLOCK_DMODEL_POW2, BLOCK_N, 0, num_warps, ELEM_BITS
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
        v_off_d = gl.arange(
            0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, v_async_layout)
        )
        v_off = (v_off_n[:, None] * stride_vn + v_off_d[None, :] * stride_vk).to(
            gl.int32
        )
        if HAS_PE:
            PE_VEC: gl.constexpr = _async_copy_vec(
                BLOCK_DMODEL_PE, BLOCK_N, 0, num_warps, ELEM_BITS
            )
            kpe_async_layout: gl.constexpr = _async_copy_layout(
                BLOCK_DMODEL_PE, BLOCK_N, 0, num_warps, PE_VEC
            )
            kpe_off_d = BLOCK_DMODEL + gl.arange(
                0, BLOCK_DMODEL_PE, layout=gl.SliceLayout(1, kpe_async_layout)
            )
            kpe_off_n = gl.arange(
                0, BLOCK_N, layout=gl.SliceLayout(0, kpe_async_layout)
            )
            kpe_off = (
                kpe_off_d[:, None] * stride_kk + kpe_off_n[None, :] * stride_kn
            ).to(gl.int32)
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
        if PADDED_HEAD:
            # A masked async copy simply does not write the masked lanes -- `other` is never
            # materialised in LDS.  The masked lanes are always the same ones (the
            # head-dim padding), so zeroing each slot once up front is enough; without
            # it the first tile multiplies q == 0 against uninitialised LDS, and
            # 0 * NaN is NaN.
            for _buf in gl.static_range(BUF_DEPTH):
                smemK.index(_buf).store(
                    gl.zeros(
                        [BLOCK_DMODEL_POW2, BLOCK_N],
                        dtype=k_ptr.dtype.element_ty,
                        layout=kLoadLayout,
                    )
                )
                smemV.index(_buf).store(
                    gl.zeros(
                        [BLOCK_N, BLOCK_DMODEL_POW2],
                        dtype=v_ptr.dtype.element_ty,
                        layout=vLoadLayout,
                    )
                )
            gl.barrier()
    else:
        kt_off = None
        kpe_off = None
        v_off = None
        kt_off_n = None
        kt_off_d = None
        kpe_off_n = None
        v_off_n = None
        v_off_d = None
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
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=gl.float32, layout=mfmaLayout)

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
    n_blocks = gl.cdiv(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_causal = gl.cdiv(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )
        n_blocks = min(n_blocks, n_blocks_causal)

        if n_blocks <= 0:
            storeLayout: gl.constexpr = qLoadLayout
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

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = (not padded_block_k) and (seqlen_q % BLOCK_M == 0)

    # Skip K blocks that are fully left of the earliest key position
    # reachable by this Q block. The first retained block can still be
    # partially outside the window, so we keep the per-element mask below.
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
    n_full_blocks = visible_blocks - masked_blocks
    block_min = skipped_blocks * BLOCK_N
    block_max = n_blocks * BLOCK_N

    if SLIDING_WINDOW > 0:
        # k_base also anchors the PE slice, which shares K's base pointer.
        k_base += skipped_blocks * BLOCK_N * stride_kn
        v_base += skipped_blocks * BLOCK_N * stride_vn

    # The rotated pipeline is a dense-path specialisation: it carries no per-element
    # masking and no fp8 rescale, so it only takes the configurations whose full
    # blocks are plain Q@K^T / P@V.  Everything it declines -- and every masked block
    # in every configuration -- goes to the generic loop below.
    #
    # The PE slice rides with K through the pipeline: same commit group in mem1,
    # read beside it in mem2, consumed beside it in dot1.  It is implemented but
    # disabled, because the four clusters are balanced on the assumption that each
    # matrix cluster faces one memory cluster.  The PE slice puts a SECOND MFMA
    # chain in dot1 without adding work to mem1 to overlap it with, so dot1 grows
    # and no memory work covers the growth.  Enabling it requires re-cutting the
    # clusters around three chains rather than flipping this flag.
    PE_IN_PIPELINE: gl.constexpr = False
    FAST_PATH: gl.constexpr = (
        USE_ASYNC_COPY
        and (not IS_FP8)
        and (not RETURN_SCORES)
        and SLIDING_WINDOW == 0
        and (PE_IN_PIPELINE or not HAS_PE)
    )

    pipelined = False
    if FAST_PATH:  # noqa: SIM102
        # Checked per workgroup, not per launch: under a causal mask each M-block
        # sees a different number of full tiles, so a long sequence still gives its
        # first M-blocks only a handful.
        if n_full_blocks >= _MIN_PIPE_BLOCKS:
            pipelined = True
            acc, l_i, m_i = _attn_fwd_pipelined(
                acc,
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
            )
            block_min = block_min + n_full_blocks * BLOCK_N
            block_max = n_blocks * BLOCK_N
            # WAR: the pipeline's last LDS reads against whatever stages next.
            gl.barrier()

    # Full blocks: no boundary mask, no causal mask.
    if (not pipelined) and n_full_blocks > 0:
        block_max = block_min + n_full_blocks * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_pe,
            k_base,
            k_offsets,
            k_pe_offsets,
            v_base,
            v_offsets,
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
            descale_v,
            sd_base,
            sd_offsets,
            sd_q_mask,
            stride_sd_n,
            mfmaLayout=mfmaLayout,
            dotK=dotK,
            dotP=dotP,
            dotV=dotV,
            kLoadLayout=kLoadLayout,
            kPeLoadLayout=kPeLoadLayout,
            vLoadLayout=vLoadLayout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE=BLOCK_DMODEL_PE,
            HAS_PE=HAS_PE,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            SLIDING_WINDOW=SLIDING_WINDOW,
            RETURN_SCORES=RETURN_SCORES,
            SCALE_ON_Q=SCALE_ON_Q,
            USE_ASYNC_COPY=USE_ASYNC_COPY,
            BUF_DEPTH=BUF_DEPTH,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    # Remaining blocks carry the boundary / causal masking.
    if masked_blocks > 0:
        k_base += n_full_blocks * BLOCK_N * stride_kn
        v_base += n_full_blocks * BLOCK_N * stride_vn
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_pe,
            k_base,
            k_offsets,
            k_pe_offsets,
            v_base,
            v_offsets,
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
            descale_v,
            sd_base,
            sd_offsets,
            sd_q_mask,
            stride_sd_n,
            mfmaLayout=mfmaLayout,
            dotK=dotK,
            dotP=dotP,
            dotV=dotV,
            kLoadLayout=kLoadLayout,
            kPeLoadLayout=kPeLoadLayout,
            vLoadLayout=vLoadLayout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE=BLOCK_DMODEL_PE,
            HAS_PE=HAS_PE,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            SLIDING_WINDOW=SLIDING_WINDOW,
            RETURN_SCORES=RETURN_SCORES,
            SCALE_ON_Q=SCALE_ON_Q,
            USE_ASYNC_COPY=USE_ASYNC_COPY,
            BUF_DEPTH=BUF_DEPTH,
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
    acc = acc * (1.0 / l_i)[:, None]

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
                [BLOCK_DMODEL_POW2],
                causal_start_idx,
                dtype=gl.int32,
                layout=gl.SliceLayout(0, mfmaLayout),
            )
            mask_m_offsets = start_m_idx + gl.arange(
                0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout)
            )
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            acc = gl.where(out_ptrs_mask, acc, 0.0)

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

    out = acc.to(o_ptr.dtype.element_ty)

    storeLayout: gl.constexpr = qLoadLayout
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


def _get_config(
    is_fp8: bool, has_pe: bool = False, causal: bool = False, v_head_dim: int = 0
):
    """Tile / wave configuration for one masking + dtype mode.

    The MFMA tiling fixes ``BLOCK_M = 32 * num_warps``, so the two knobs move
    together.  Both masking modes take the wide tile (256 / 8 waves): it amortises
    the per-tile softmax over twice the rows and puts two waves on every SIMD, which
    is what lets a wave's vector work issue in the other wave's MFMA shadow.  The
    narrow tile (128 / 4 waves) wastes less of a causal diagonal block, but it gives
    one wave per SIMD, which leaves the pipeline nothing to overlap against.
    """
    arch = arch_info.get_arch()
    fpath = f"{AITER_TRITON_CONFIGS_PATH}/{arch}/gluon/attention/mha/mha.json"
    fwd_cfg = load_config_json(fpath)["fwd"]
    if is_fp8:
        return fwd_cfg["fp8"]
    elif has_pe:
        # The PE tile is bounded by the accumulator, which is [BLOCK_M, v_head_dim]
        # fp32 -- v_head_dim/2 VGPRs per lane once BLOCK_M cancels against the wave
        # count.  A 64-wide V head leaves room for the wide tile and the two waves
        # per SIMD that go with it; a 128-wide one does not, and forcing it there
        # spills.
        if v_head_dim and v_head_dim <= 64 and "pe_narrow_v" in fwd_cfg:
            return fwd_cfg["pe_narrow_v"]
        return fwd_cfg["pe"]
    elif causal and "causal" in fwd_cfg:
        return fwd_cfg["causal"]
    else:
        return fwd_cfg["default"]
