# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 MQA logits (DeepSeek lightning indexer) -- FlyDSL gfx942/gfx950 kernel.

Compute for each query row ``m`` and KV position ``n``
inside that row's window ``[cu_starts[m], cu_ends[m])``::

    logits[m, n] = sum_h ReLU(<Q[m, h, :], K[n, :]> * kv_scale[n]) * weights[m, h]

The public ``flydsl_fp8_mqa_logits`` mirrors the Triton launcher
``aiter.ops.triton.attention.fp8_mqa_logits.fp8_mqa_logits`` exactly (same
arguments, same return tensor, same ``clean_logits`` semantics) so the two are
drop-in interchangeable in tests and benchmarks.
"""

# NOTE: do NOT add `from __future__ import annotations` to this file -- PEP 563
# stringizes annotations, which FlyDSL's kernel-argument typing relies on being
# real objects. (Matches the note in qk_norm_rope_quant.py.)

import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import _split_mfma_operands, _unwrap_mfma_operand
from flydsl.expr.numeric import ArithValue
from flydsl.expr.typing import T
from flydsl._mlir.dialects import scf, vector as mlir_vector
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl._mlir.dialects.rocdl import mfma_f32_32x32x16_fp8_fp8 as _ods_mfma32x32x16
from flydsl._mlir.dialects.rocdl import (
    mfma_scale_f32_32x32x64_f8f6f4 as _ods_mfma_scale32x32x64,
)
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .tensor_shim import GTensor, STensor, _run_compiled, _to_raw

Vec = fx.Vector

arch = get_rocm_arch()

# --------------------------------------------------------------------------- #
# MfmaAtom — bundles all MFMA-shape-derived constants and the rocdl functor.
#
# Adding a new MFMA shape only requires a new MfmaAtom instance and a new
# entry in _VARIANT_BUILDERS; the kernel builder is fully generic.
# --------------------------------------------------------------------------- #

def _make_operands_dense(a, b, acc):
    """Default ``MfmaAtom.make_operands``: dense-MFMA 6-operand convention
    ``[a, b, c, cbsz, abid/blgp, blgp]`` (all zero besides the fragments)."""
    return [a, b, acc, 0, 0, 0]


def _make_operands_scaled_identity(a, b, acc):
    """``MfmaAtom.make_operands`` for the CDNA4 scaled MFMA atoms (K=128/64).

    These instructions always carry ``scaleA``/``scaleB`` UE8M0 operands as
    part of their encoding; passing a compile-time identity scale (UE8M0
    bias-127, i.e. every byte == 127 -> multiplier 1.0) makes the hardware
    microscale a no-op. The kernel's own ``kv_scale`` is applied in f32
    after the MFMA (scale-hoisted), so this is the only scale these atoms
    need. NOTE: the identity encoding is standard OCP MX (bias 127) but not
    yet empirically re-verified on this instruction -- see the smoke test.
    """
    scale = arith.constant(0x7F7F7F7F, type=T.i32)
    return [a, b, acc, 0, 0, 0, scale, 0, scale]


@dataclass(frozen=True)
class MfmaAtom:
    """MFMA-shape descriptor for the fp8 MQA logits kernel.

    Fields
    ------
    name : str
        Human-readable shape tag embedded in kernel names (e.g. ``"16x16x32"``).
    MFMA_M, MFMA_N, MFMA_K : int
        MFMA output/input tile dimensions (M×N output, K fp8 elements/step).
    ACC_ELEMS : int
        Number of f32 accumulator elements per lane (``vec<ACC_ELEMS x f32>``).
    fn : Callable
        FlyDSL ``rocdl.mfma_*`` functor accepting ``(result_type, operands)``.
    shuffle_offsets : tuple[int, ...]
        ``shuffle_xor`` offsets for the in-wave head-reduce butterfly.
        Must cover all lane groups so the full H-wide sum is produced.
    acc_head_static_offsets : tuple[int, ...]
        Per-element compile-time head-offset within one MFMA_M tile.
        Length == ACC_ELEMS. For element ``ii`` in lane group ``g``:
        ``head_within_tile = acc_head_static_offsets[ii] + g * acc_head_group_stride``
        The full weight index is ``mi * MFMA_M + head_within_tile``.

        Derivation: the MFMA hardware stores acc element ``ii`` for the head
        whose *row* in the A-matrix is ``acc_head_static_offsets[ii] + g * stride``.
        For 16x16x32 (4 groups, ACC_ELEMS=4): the layout is sequential,
        giving ``static_offsets = (0, 1, 2, 3)`` and ``group_stride = 4``.
        For 32x32x16 (2 groups, ACC_ELEMS=16): the layout interleaves the two
        groups in blocks of 4, giving
        ``static_offsets = (0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27)``
        and ``group_stride = 4`` (empirically verified on gfx942/CDNA3).
        The CDNA4 scaled atoms (16x16x128, 32x32x64) reuse these fields
        verbatim from their K=32/K=16 siblings: ``tv_layout_c`` (read via
        standalone ``fly.MmaAtomType`` introspection) is identical across
        the K variants for a given (M, N) -- it depends only on the output
        tile shape, not the reduction depth.
    acc_head_group_stride : int
        Multiplier for the lane-group index (always 4 on gfx942/gfx950 fp8 MFMA).
    frag_bytes : int
        Bytes of A/B fragment data owned by one lane, one K-step. 8 for the
        dense K=32/K=16 atoms (i64, one buffer_load). 32 for the CDNA4 scaled
        K=128/K=64 atoms (vector<8xi32>, two dwordx4 buffer_loads).
    make_operands : Callable
        Builds the ``fn`` operand list from ``(a_frag, b_frag, acc)``.
        Dense atoms use ``_make_operands_dense`` (6-elem); the CDNA4 scaled
        atoms use ``_make_operands_scaled_identity`` (9-elem, appends an
        identity UE8M0 scale pair).
    """
    name: str
    MFMA_M: int
    MFMA_N: int
    MFMA_K: int
    ACC_ELEMS: int
    fn: Callable
    shuffle_offsets: tuple
    acc_head_static_offsets: tuple  # length == ACC_ELEMS
    acc_head_group_stride: int
    frag_bytes: int = 8
    make_operands: Callable = _make_operands_dense


def _mfma32x32x16_fp8_fp8_wrapper(result_type, operands, *, loc=None, ip=None):
    """Wrap the raw ODS ``mfma_f32_32x32x16_fp8_fp8`` to match the
    ``(result_type, operands_list)`` convention used by ``flydsl.expr.rocdl``."""
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands)
    return _ods_mfma32x32x16(result_type, a, b, c, cbsz, abid, blgp,
                              loc=loc, ip=ip).result


def _mfma32x32x64_fp8_fp8_scale_wrapper(result_type, operands, *, loc=None, ip=None):
    """Wrap the raw ODS ``mfma_scale_f32_32x32x64_f8f6f4`` to match the
    ``(result_type, operands_list)`` convention. ``operands`` follows the
    9-elem scaled-MFMA convention: ``[a, b, c, cbsz, blgp, opselA, scaleA,
    opselB, scaleB]`` (no ``abid``, unlike the dense atoms). Mirrors the
    ready-made ``rocdl.mfma_scale_f32_16x16x128_f8f6f4`` friendly wrapper,
    for which no 32x32x64 equivalent ships with FlyDSL yet."""
    a = _unwrap_mfma_operand(operands[0])
    b = _unwrap_mfma_operand(operands[1])
    c = _unwrap_mfma_operand(operands[2])
    cbsz = int(operands[3]) if len(operands) > 3 else 0
    blgp = int(operands[4]) if len(operands) > 4 else 0
    opselA = int(operands[5]) if len(operands) > 5 else 0
    scaleA = _unwrap_mfma_operand(operands[6]) if len(operands) > 6 else a
    opselB = int(operands[7]) if len(operands) > 7 else 0
    scaleB = _unwrap_mfma_operand(operands[8]) if len(operands) > 8 else b
    return _ods_mfma_scale32x32x64(
        result_type, a, b, c, cbsz, blgp, opselA, scaleA, opselB, scaleB,
        loc=loc, ip=ip,
    ).result


#: 16×16 output tile, K=32 fp8 elements/step.  Acc: vec<4 x f32>.
#: Fragment layout: lane l → A[row=l%16, k=(l//16)*8+0..7], col=l%16.
#: Writer: lane//16 == 0 (16 distinct output columns per tile).
#: Acc layout: acc[ii] at group g -> head g*4+ii.
_MFMA16 = MfmaAtom(
    name="16x16x32",
    MFMA_M=16, MFMA_N=16, MFMA_K=32,
    ACC_ELEMS=4,
    fn=rocdl.mfma_f32_16x16x32_fp8_fp8,
    shuffle_offsets=(16, 32),
    acc_head_static_offsets=(0, 1, 2, 3),
    acc_head_group_stride=4,
)

#: 32×32 output tile, K=16 fp8 elements/step.  Acc: vec<16 x f32>.
#: Fragment layout: lane l → A[row=l%32, k=(l//32)*8+0..7], col=l%32.
#: Writer: lane//32 == 0 (32 distinct output columns per tile).
#: Acc layout: acc[ii] at group g ->
#:   head (ii//4)*8 + g*4 + ii%4.  static_offsets = ii%4 + (ii//4)*8.
_MFMA32 = MfmaAtom(
    name="32x32x16",
    MFMA_M=32, MFMA_N=32, MFMA_K=16,
    ACC_ELEMS=16,
    fn=_mfma32x32x16_fp8_fp8_wrapper,
    shuffle_offsets=(32,),
    acc_head_static_offsets=(
        0, 1, 2, 3,    # ii=0..3:  head = g*4 + 0..3
        8, 9, 10, 11,  # ii=4..7:  head = g*4 + 8..11
        16, 17, 18, 19,# ii=8..11: head = g*4 + 16..19
        24, 25, 26, 27,# ii=12..15:head = g*4 + 24..27
    ),
    acc_head_group_stride=4,
)

#: gfx950/CDNA4 scaled MFMA: 16x16 output tile, K=128 fp8f6f4 elements/step.
#: Acc: vec<4 x f32> (same layout as _MFMA16 -- tv_layout_c depends only on
#: M,N). Fragment: vector<8xi32> (32 bytes/lane), 4x _MFMA16's 8-byte frag,
#: tracking the 4x K increase. Requires native FN operands (this instruction
#: rejects FNUZ) and head_size % 128 == 0 (enforced by the generic D%K assert
#: in _build_kernel_mfma_r_w).
_MFMA16_K128 = MfmaAtom(
    name="16x16x128",
    MFMA_M=16, MFMA_N=16, MFMA_K=128,
    ACC_ELEMS=4,
    fn=rocdl.mfma_scale_f32_16x16x128_f8f6f4,
    shuffle_offsets=(16, 32),
    acc_head_static_offsets=(0, 1, 2, 3),
    acc_head_group_stride=4,
    frag_bytes=32,
    make_operands=_make_operands_scaled_identity,
)

#: gfx950/CDNA4 scaled MFMA: 32x32 output tile, K=64 fp8f6f4 elements/step.
#: Acc: vec<16 x f32> (same layout as _MFMA32). Fragment: vector<8xi32>
#: (32 bytes/lane), 4x _MFMA32's 8-byte frag. Requires native FN operands
#: (rejects FNUZ); works for both head_size 64 and 128.
_MFMA32_K64 = MfmaAtom(
    name="32x32x64",
    MFMA_M=32, MFMA_N=32, MFMA_K=64,
    ACC_ELEMS=16,
    fn=_mfma32x32x64_fp8_fp8_scale_wrapper,
    shuffle_offsets=(32,),
    acc_head_static_offsets=(
        0, 1, 2, 3,    # ii=0..3:  head = g*4 + 0..3
        8, 9, 10, 11,  # ii=4..7:  head = g*4 + 8..11
        16, 17, 18, 19,# ii=8..11: head = g*4 + 16..19
        24, 25, 26, 27,# ii=12..15:head = g*4 + 24..27
    ),
    acc_head_group_stride=4,
    frag_bytes=32,
    make_operands=_make_operands_scaled_identity,
)


def _i32_add(a, b):
    """Add two fx.Int32 scalars -> fx.Int32 (i32 arithmetic)."""
    return fx.Int32(arith.addi(_to_raw(a), _to_raw(b)))

_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 2,
    "fast_fp_math": True,
}

# Don't split a row's KV window into chunks smaller than this many BKV tiles --
# below it the per-block Q/weight preload stops being amortized.
_MIN_TILES_PER_SPLIT = 8


@lru_cache(maxsize=8)
def _device_cu_count(device_index: int) -> int:
    """Compute-unit count for a CUDA/HIP device (cached); 304 if unavailable."""
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:
        return 304


def _auto_num_splits(
    seq_len_padded: int, seq_len_kv: int, rpb: int, block_kv: int, device_index: int
) -> int:
    """KV-column splits (grid.y) to fill the device when the row grid is small.

    For small-M / large-N shapes the ``ceil(seq_len/RPB)`` row grid leaves the
    device block-starved; splitting each row's window across ``grid.y`` recovers
    occupancy at no correctness cost (logits[m,n] are independent across n).
    Returns 1 once the row grid alone oversubscribes the device. Constants tuned
    on MI300X (304 CU): ~4x oversubscription, chunks >= _MIN_TILES_PER_SPLIT.
    """
    grid_x = seq_len_padded // rpb
    if grid_x == 0 or seq_len_kv < 4096:
        return 1
    target_blocks = 4 * _device_cu_count(device_index)
    if grid_x >= target_blocks:
        return 1
    max_splits = max(1, (seq_len_kv // block_kv) // _MIN_TILES_PER_SPLIT)
    return max(1, min(math.ceil(target_blocks / grid_x), max_splits))


def _build_kernel_mfma_r_w(*, num_heads: int, head_size: int, block_kv: int,
                           rows_per_block: int, waves_per_block: int,
                           mfma: MfmaAtom,
                           convert_q_fn: bool = False, convert_kv_fn: bool = False):
    """Multi-row, multi-wave MFMA kernel (generic over MFMA shape via MfmaAtom).

    Parameters
    ----------
    rows_per_block : int
        Query rows sharing one KV tile load; cuts KV global traffic by RPB.
    waves_per_block : int
        Waves per block; each wave owns a disjoint ``N_TILES // WPB`` column-tile
        slice, so all WPB waves run in parallel with no cross-wave LDS or barrier.
    mfma : MfmaAtom
        MFMA shape descriptor (dimensions, accumulator size, rocdl functor,
        head-reduce butterfly offsets).

    Thread decomposition
    --------------------
    * ``tid = wave * 64 + lane``  (tid: 0..MR_BLOCK_THREADS-1)
    * ``lane_div_N = lane // MFMA_N`` — k-group index; selects the K chunk each
      lane reads in the A/B frags.
    * ``lane_mod_N = lane % MFMA_N`` — column index within the MFMA_N-wide tile.
    * Wave ``w`` owns n-tiles ``[w*N_TILES_PER_WAVE, (w+1)*N_TILES_PER_WAVE)``
      within each BKV tile. A-frag layout and head-reduce use ``lane``, not ``tid``.

    Grid: ``(ceil(seq_len / RPB), 1, 1)``.  The host wrapper pads ``seq_len`` to
    a multiple of ``RPB`` so every block owns exactly ``RPB`` valid rows.
    """
    H   = num_heads
    D   = head_size
    BKV = block_kv
    RPB = rows_per_block
    WPB = waves_per_block
    MR_BLOCK_THREADS = 64 * WPB

    assert H % mfma.MFMA_M == 0, (
        f"num_heads={H} must be a multiple of MFMA_M={mfma.MFMA_M}"
    )
    assert BKV % mfma.MFMA_N == 0, (
        f"block_kv={BKV} must be a multiple of MFMA_N={mfma.MFMA_N}"
    )
    assert D % mfma.MFMA_K == 0, (
        f"head_size={D} must be a multiple of MFMA_K={mfma.MFMA_K}"
    )
    assert RPB >= 1, f"rows_per_block must be >= 1"
    assert WPB >= 1, f"waves_per_block must be >= 1"

    N_TILES = BKV // mfma.MFMA_N        # total column-tiles per BKV block
    assert N_TILES % WPB == 0, (
        f"BKV/MFMA_N={N_TILES} must be divisible by waves_per_block={WPB}"
    )
    N_TILES_PER_WAVE = N_TILES // WPB             # column-tiles per wave
    M_TILES          = H   // mfma.MFMA_M         # head row-tiles
    K_STEPS          = D   // mfma.MFMA_K         # MFMA K-steps over the head dim

    fm_fast = arith.FastMathFlags.fast
    mfma_fn = mfma.fn

    _cvt_tag = ""
    if convert_q_fn:
        _cvt_tag += "_cq"
    if convert_kv_fn:
        _cvt_tag += "_ck"
    _kname = (
        f"fp8_mqa_logits_H{H}_D{D}_mfma{mfma.name}"
        f"_bkv{BKV}_r{RPB}_w{WPB}{_cvt_tag}_flydsl"
    )

    @flyc.kernel(name=_kname, known_block_size=[MR_BLOCK_THREADS, 1, 1])
    def kernel(
        Q: fx.Tensor,            # [seq_len, H, D]       fp8 (bytes passed raw)
        KV: fx.Tensor,           # [seq_len_kv, D]       fp8 (bytes passed raw)
        kv_scales: fx.Tensor,    # [seq_len_kv]          f32
        weights: fx.Tensor,      # [seq_len, H]          f32
        cu_starts: fx.Tensor,    # [seq_len]             i32
        cu_ends: fx.Tensor,      # [seq_len]             i32
        logits: fx.Tensor,       # [seq_len, seq_len_kv] f32
        seq_len: fx.Int32,       # padded to a multiple of RPB
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
        num_splits: fx.Int32,    # grid.y KV-column splits (1 == no split)
    ):
        f32_0 = arith.constant(0.0, type=T.f32)
        _mfma_res_ty = Vec.make_type(mfma.ACC_ELEMS, fx.Float32)

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        # Block bid (reversed) owns rows [r0, r0+RPB).
        n_blocks = fx.Int32(arith.ceildivui(_to_raw(seq_len), _to_raw(fx.Int32(RPB))))
        r0 = fx.Int32(arith.muli(
            _to_raw(n_blocks - bid - fx.Int32(1)),
            _to_raw(fx.Int32(RPB)),
        ))

        # Decompose tid into wave index and in-wave lane.
        wave     = fx.Int32(arith.divui(_to_raw(tid), _to_raw(fx.Int32(64))))
        lane     = fx.Int32(arith.remui(_to_raw(tid), _to_raw(fx.Int32(64))))
        # lane_div_N: k-group index in the A/B frag layout.
        # lane_mod_N: output column index within the MFMA_N-wide tile.
        # lane8:      byte offset of this lane's K chunk (always 8 fp8 bytes per i64).
        #             = (lane // MFMA_N) * 8
        # Number of lane groups = 64 // MFMA_N:
        #   MFMA_N=16 → 4 groups, each K-step spans 32 bytes (need 64-bit per-group load)
        #   MFMA_N=32 → 2 groups, each K-step spans 16 bytes (fits in 128-bit load)
        lane_div_N = fx.Int32(arith.divui(_to_raw(lane), _to_raw(fx.Int32(mfma.MFMA_N))))
        lane_mod_N = fx.Int32(arith.remui(_to_raw(lane), _to_raw(fx.Int32(mfma.MFMA_N))))
        lane8      = fx.Int32(arith.muli(_to_raw(lane_div_N), _to_raw(fx.Int32(8))))
        # lane_frag_off: generalized per-lane-group byte offset for the wide
        # (frag_bytes=32) CDNA4 scaled atoms -- same role as lane8, just
        # scaled to the wider fragment (each lane group owns frag_bytes
        # contiguous bytes per K-step instead of 8).
        lane_frag_off = fx.Int32(
            arith.muli(_to_raw(lane_div_N), _to_raw(fx.Int32(mfma.frag_bytes)))
        )

        # fp8 operands are read as i32 dwords (v8i8 buffer_load fails to lower
        # on gfx942), then bitcast to i64 for the MFMA.
        q_i32 = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(KV, dtype=T.i32, shape=(-1,))
        sc_t = GTensor(kv_scales, dtype=T.f32, shape=(-1,))
        w_t = GTensor(weights, dtype=T.f32, shape=(-1, H))
        cs_t = GTensor(cu_starts, dtype=T.i32, shape=(-1,))
        ce_t = GTensor(cu_ends, dtype=T.i32, shape=(-1,))
        # out_t is built per-row inside the epilogue to avoid i32 byte-offset
        # overflow (row * stride * 4 > 2^31-1 for S >= 32768).
        # Each out_row_t is a 1-D view shifted by the row's i64 byte offset.
        _stride_i64_mfma = arith.extui(T.i64, _to_raw(stride_logits_s))

        def _make_out_row_t(row_i32):
            """1-D output GTensor for row_i32; byte base computed in i64."""
            _ri64 = arith.extui(T.i64, _to_raw(row_i32))
            _byte = arith.muli(arith.muli(_ri64, _stride_i64_mfma),
                               arith.constant(4, type=T.i64))
            _idx  = arith.index_cast(T.index, _byte)
            return GTensor(logits, dtype=T.f32, shape=(-1,),
                           static_bytes_offset_i64=_idx)

        # Number of lane groups per K-step tile.
        _N_LANE_GROUPS = 64 // mfma.MFMA_N  # compile-time constant

        # _is_group0: True for lanes in group 0 (lane_div_N == 0).
        # Used by the 128-bit load path (_N_LANE_GROUPS == 2) to select lo/hi i64.
        # Computed unconditionally; only referenced in the _N_LANE_GROUPS == 2 branch.
        _is_group0 = arith.cmpi(arith.CmpIPredicate.eq, _to_raw(lane_div_N),
                                 arith.constant(0, type=T.i32))

        def _load_pack_i64(i32_view, byte_off_i32):
            """64-bit load: returns i64 covering this lane's 8 fp8 bytes.

            byte_off_i32 must already include the lane's byte offset (lane8)
            so the load hits the correct 8-byte chunk for this lane group.
            """
            dword_off = fx.Int32(
                arith.divui(_to_raw(byte_off_i32), _to_raw(fx.Int32(4)))
            )
            v2 = i32_view.vec_load((dword_off,), vec_size=2)
            return Vec(v2).bitcast(fx.Int64)[0].ir_value()

        def _load_pack_i64x2(i32_view, byte_off_i32):
            """128-bit load: returns (lo_i64, hi_i64) covering 16 contiguous fp8 bytes.

            Valid only when there are 2 lane groups (MFMA_N=32): each K-step
            tile is 2 groups x 8 bytes = 16 bytes, fitting in one dwordx4 load.
              lane_div_N == 0  →  lo_i64  (bytes 0-7  relative to load base)
              lane_div_N == 1  →  hi_i64  (bytes 8-15 relative to load base)
            """
            dword_off = fx.Int32(
                arith.divui(_to_raw(byte_off_i32), _to_raw(fx.Int32(4)))
            )
            v4     = i32_view.vec_load((dword_off,), vec_size=4)
            v2xi64 = Vec(v4).bitcast(fx.Int64)   # vec<2 x i64>
            lo_i64 = v2xi64[0].ir_value()
            hi_i64 = v2xi64[1].ir_value()
            return lo_i64, hi_i64

        def _load_pack_i32x8(i32_view, byte_off_i32):
            """256-bit-equivalent load: returns a ``vector<8xi32>`` ir.Value
            covering this lane's 32 fp8 bytes (frag_bytes=32 CDNA4 scaled atoms).

            Hardware buffer_load tops out at dwordx4 (128 bits), so the
            fragment is split into two consecutive dwordx4 loads and
            concatenated via vector.shuffle. byte_off_i32 must already
            include the lane's byte offset (lane_frag_off) so the load hits
            the correct 32-byte chunk for this lane group.
            """
            dword_off = fx.Int32(
                arith.divui(_to_raw(byte_off_i32), _to_raw(fx.Int32(4)))
            )
            v4_lo = i32_view.vec_load((dword_off,), vec_size=4)
            dword_off_hi = _i32_add(dword_off, fx.Int32(4))
            v4_hi = i32_view.vec_load((dword_off_hi,), vec_size=4)
            return Vec(v4_lo).shuffle(v4_hi, list(range(8))).ir_value()

        def _load_pack_i64_swizzle(i32_view, byte_off_i32):
            """128-bit load + permlane32.swap: each lane gets its correct 8 bytes,
            zero wasted bandwidth.

            Valid only when _N_LANE_GROUPS == 2 (MFMA_N=32).

            All 64 lanes load the same 16-byte k-step tile at `byte_off_i32`
            (no lane8 offset):
              lo_i64 = bytes  0..7  → group 0's data
              hi_i64 = bytes 8..15  → group 1's data

            permlane32.swap exchanges hi_i64 between lane c (group 0) and
            lane c+32 (group 1). After the swap:
              group 0 lanes: use lo_i64              (their own bytes)
              group 1 lanes: use swapped hi_i64      (their own bytes, now received)

            The swap operates on i32; hi_i64 is split into two i32 halves,
            each swapped independently, then reassembled.
            """
            lo_i64, hi_i64 = _load_pack_i64x2(i32_view, byte_off_i32)
            # Split hi_i64 into two i32 halves for shuffle_xor.
            hi_lo32 = arith.TruncIOp(T.i32, hi_i64).result
            hi_hi32 = arith.TruncIOp(T.i32,
                arith.ShRUIOp(hi_i64, arith.constant(32, type=T.i64)).result
            ).result
            # gpu.shuffle xor offset=32: lane c ↔ lane c^32 (group 0 ↔ group 1).
            # width=64 (full wave). Each lane sends hi and receives its peer's hi.
            swapped_lo32 = ArithValue(hi_lo32).shuffle_xor(32, 64)
            swapped_hi32 = ArithValue(hi_hi32).shuffle_xor(32, 64)
            # Reassemble received i64.
            swapped_i64 = arith.OrIOp(
                arith.ExtUIOp(T.i64, swapped_lo32).result,
                arith.ShLIOp(
                    arith.ExtUIOp(T.i64, swapped_hi32).result,
                    arith.constant(32, type=T.i64),
                ).result,
            ).result
            # Group 0 uses lo_i64 (own data); group 1 uses the received hi_i64.
            return arith.select(_is_group0, lo_i64, swapped_i64)

        # ---- FN -> FNUZ in-kernel byte conversion ----
        def _fn_to_fnuz_i64(raw_i64):
            """Map FN byte 0x80 (neg-zero) -> 0x00 in 8 packed fp8 bytes."""
            lo_i32 = arith.TruncIOp(T.i32, raw_i64).result
            hi_i64 = arith.ShRUIOp(raw_i64, arith.constant(32, type=T.i64)).result
            hi_i32 = arith.TruncIOp(T.i32, hi_i64).result

            def _fix_i32(src):
                result = arith.constant(0, type=T.i32)
                for byte_idx in range_constexpr(4):
                    shift = arith.constant(byte_idx * 8, type=T.i32)
                    byte_val = arith.andi(
                        arith.shrui(src, shift),
                        arith.constant(0xFF, type=T.i32),
                    )
                    is_0x80 = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        byte_val,
                        arith.constant(0x80, type=T.i32),
                    )
                    cleaned = arith.select(
                        is_0x80,
                        arith.constant(0, type=T.i32),
                        byte_val,
                    )
                    result = arith.ori(result, arith.shli(cleaned, shift))
                return result

            lo_fix = _fix_i32(lo_i32)
            hi_fix = _fix_i32(hi_i32)
            lo_64 = arith.ExtUIOp(T.i64, lo_fix).result
            hi_64 = arith.ShLIOp(arith.ExtUIOp(T.i64, hi_fix).result, arith.constant(32, type=T.i64),).result
            return arith.OrIOp(lo_64, hi_64).result

        # ---- Preload per-row metadata, Q frags, and weights for all RPB rows ----
        # All WPB waves preload all RPB rows' Q frags. A-operand layout is per
        # in-wave lane (lane_div_N, lane_mod_N), so `lane` (not `tid`) is used here.
        starts  = [None] * RPB
        ends    = [None] * RPB
        a_packs = [None] * RPB
        w_frag  = [None] * RPB

        for j in range_constexpr(RPB):
            row = _i32_add(r0, fx.Int32(j))
            s = fx.Int32(cs_t[row])
            e = fx.Int32(ce_t[row])
            starts[j] = fx.Int32(arith.maxsi(_to_raw(s), _to_raw(fx.Int32(0))))
            ends[j]   = fx.Int32(arith.minsi(_to_raw(e), _to_raw(fx.Int32(seq_len_kv))))

            # A-frag layout: lane l -> Q[row, h = mi*MFMA_M + l%MFMA_N,
            #                              d = kk*MFMA_K + (l//MFMA_N)*8 + 0..7]
            row_a = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                h_a    = _i32_add(fx.Int32(mi * mfma.MFMA_M), lane_mod_N)
                row_h  = _i32_add(
                    fx.Int32(arith.muli(_to_raw(row), _to_raw(fx.Int32(H)))), h_a
                )
                base_a = fx.Int32(arith.muli(_to_raw(row_h), _to_raw(fx.Int32(D))))
                for kk in range_constexpr(K_STEPS):
                    if const_expr(mfma.frag_bytes == 32):
                        # CDNA4 scaled atoms (16x16x128/32x32x64): 256-bit
                        # fragment via two dwordx4 loads, offset by this
                        # lane group's frag_bytes-wide chunk. Native FN
                        # operands -- never FN->FNUZ converted.
                        raw = _load_pack_i32x8(
                            q_i32,
                            _i32_add(base_a, _i32_add(fx.Int32(kk * mfma.MFMA_K), lane_frag_off)),
                        )
                    elif const_expr(_N_LANE_GROUPS == 2):
                        # 32x32x16: 128-bit load + permlane32.swap.
                        # All lanes load 16 bytes at base (no lane8 offset);
                        # swap distributes the hi half to group 1, zero waste.
                        raw = _load_pack_i64_swizzle(
                            q_i32, _i32_add(base_a, fx.Int32(kk * mfma.MFMA_K))
                        )
                    else:
                        # 16x16x32: 64-bit load with lane8 group offset.
                        raw = _load_pack_i64(
                            q_i32, _i32_add(base_a, _i32_add(fx.Int32(kk * mfma.MFMA_K), lane8))
                        )
                    row_a[mi][kk] = _fn_to_fnuz_i64(raw) if convert_q_fn else raw
            a_packs[j] = row_a

            # weights[row, h] preloaded per (mi, ii):
            #   head = mi*MFMA_M + acc_head_static_offsets[ii] + lane_div_N*group_stride
            # acc_head_static_offsets and acc_head_group_stride encode the MFMA
            # hardware's accumulator-to-head mapping (shape-specific, empirically
            # verified).  For 16x16x32: static[ii]=ii, stride=4 (sequential).
            # For 32x32x16: static[ii]=(ii//4)*8+ii%4, stride=4 (interleaved).
            row_w = [[None] * mfma.ACC_ELEMS for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                for ii in range_constexpr(mfma.ACC_ELEMS):
                    static_off = mfma.acc_head_static_offsets[ii]
                    h_w = _i32_add(
                        fx.Int32(mi * mfma.MFMA_M + static_off),
                        fx.Int32(arith.muli(
                            _to_raw(lane_div_N),
                            _to_raw(fx.Int32(mfma.acc_head_group_stride)),
                        )),
                    )
                    row_w[mi][ii] = _to_raw(fx.Float32(w_t[row, h_w]))
            w_frag[j] = row_w

        # ---- Union window across all RPB rows ----
        tile_start = _to_raw(starts[0])
        tile_end   = _to_raw(ends[0])
        for j in range_constexpr(1, RPB):
            tile_start = arith.minsi(tile_start, _to_raw(starts[j]))
            tile_end   = arith.maxsi(tile_end,   _to_raw(ends[j]))
        # Align tile_start down to BKV boundary.
        tile_start = arith.muli(
            arith.divui(tile_start, _to_raw(fx.Int32(BKV))),
            _to_raw(fx.Int32(BKV)),
        )

        # KV-column split across grid.y. Block (.,by) takes a BKV-aligned
        # slice of the union window; logits[m,n] are independent across n.
        # The slices tile [start,end) exactly (disjoint, gap-free), 
        # so each column has one writer.
        # num_splits==1 collapses to the full window (by==0).
        by = fx.block_idx.y
        win_tiles = arith.ceildivui(
            arith.subi(tile_end, tile_start), _to_raw(fx.Int32(BKV))
        )
        split_cols = arith.muli(
            arith.ceildivui(win_tiles, _to_raw(num_splits)),
            _to_raw(fx.Int32(BKV)),
        )
        tile_start = arith.addi(tile_start, arith.muli(_to_raw(by), split_cols))
        tile_end = arith.minsi(arith.addi(tile_start, split_cols), tile_end)

        tile_lo   = _to_raw(fx.Index(fx.Int32(tile_start)))
        tile_hi   = _to_raw(fx.Index(fx.Int32(tile_end)))
        tile_step = _to_raw(fx.Index(fx.Int32(BKV)))
        tile_loop = scf.ForOp(tile_lo, tile_hi, tile_step, [])
        with ir.InsertionPoint(tile_loop.body):
            col0 = fx.Int32(arith.index_cast(T.i32, tile_loop.induction_variable))

            # ---- Load B-frags: each wave loads its own N_TILES_PER_WAVE columns ----
            # wave w owns absolute n-tiles [w*N_TILES_PER_WAVE, (w+1)*N_TILES_PER_WAVE).
            # No cross-wave sharing; each wave's column addresses are disjoint.
            wave_ni_base = fx.Int32(arith.muli(
                _to_raw(wave), _to_raw(fx.Int32(N_TILES_PER_WAVE))
            ))
            b_packs        = [[None] * K_STEPS for _ in range_constexpr(N_TILES_PER_WAVE)]
            kv_scales_tile = [None] * N_TILES_PER_WAVE
            cols           = [None] * N_TILES_PER_WAVE
            for ni in range_constexpr(N_TILES_PER_WAVE):
                abs_ni = _i32_add(wave_ni_base, fx.Int32(ni))
                # col = col0 + abs_ni*MFMA_N + lane_mod_N
                col = _i32_add(
                    _i32_add(
                        col0,
                        fx.Int32(arith.muli(
                            _to_raw(abs_ni), _to_raw(fx.Int32(mfma.MFMA_N))
                        )),
                    ),
                    lane_mod_N,
                )
                cols[ni] = col
                col_clamped = fx.Int32(
                    arith.minsi(_to_raw(col), _to_raw(fx.Int32(seq_len_kv) - fx.Int32(1)))
                )
                kv_scales_tile[ni] = _to_raw(fx.Float32(sc_t[col_clamped]))
                base_b = fx.Int32(arith.muli(_to_raw(col_clamped), _to_raw(fx.Int32(D))))
                for kk in range_constexpr(K_STEPS):
                    if const_expr(mfma.frag_bytes == 32):
                        # CDNA4 scaled atoms (16x16x128/32x32x64): 256-bit
                        # fragment via two dwordx4 loads, offset by this
                        # lane group's frag_bytes-wide chunk. Native FN
                        # operands -- never FN->FNUZ converted.
                        raw = _load_pack_i32x8(
                            kv_i32,
                            _i32_add(base_b, _i32_add(fx.Int32(kk * mfma.MFMA_K), lane_frag_off)),
                        )
                    elif const_expr(_N_LANE_GROUPS == 2):
                        # 32x32x16: 128-bit load + permlane32.swap.
                        # All lanes load 16 bytes at base (no lane8 offset);
                        # swap distributes the hi half to group 1, zero waste.
                        raw = _load_pack_i64_swizzle(
                            kv_i32, _i32_add(base_b, fx.Int32(kk * mfma.MFMA_K))
                        )
                    else:
                        # 16x16x32: 64-bit load with lane8 group offset.
                        raw = _load_pack_i64(
                            kv_i32, _i32_add(base_b, _i32_add(fx.Int32(kk * mfma.MFMA_K), lane8))
                        )
                    b_packs[ni][kk] = _fn_to_fnuz_i64(raw) if convert_kv_fn else raw

            # ---- Per-row MFMA + epilogue (inner loop over RPB rows) ----
            for j in range_constexpr(RPB):
                row = _i32_add(r0, fx.Int32(j))
                # 1-D output view for this row; base pointer shifted by row's
                # i64 byte offset so column offset stays safely in i32.
                out_row_t = _make_out_row_t(row)
                for ni in range_constexpr(N_TILES_PER_WAVE):
                    col      = cols[ni]
                    kv_scale = kv_scales_tile[ni]
                    col_sum  = _to_raw(f32_0)

                    # --- Head reduction step 1: in-register partial sum ---
                    # Each lane accumulates M_TILES * ACC_ELEMS MFMA output elements
                    # (covering different head subsets) into col_sum via ReLU/weight.
                    for mi in range_constexpr(M_TILES):
                        acc = Vec.filled(mfma.ACC_ELEMS, 0.0, fx.Float32)
                        for kk in range_constexpr(K_STEPS):
                            acc = mfma_fn(
                                _mfma_res_ty,
                                mfma.make_operands(a_packs[j][mi][kk], b_packs[ni][kk], acc),
                            )
                        for ii in range_constexpr(mfma.ACC_ELEMS):
                            score   = Vec(acc)[ii].ir_value()
                            relu    = arith.maximumf(score, _to_raw(f32_0))
                            wsc     = arith.MulFOp(relu, w_frag[j][mi][ii], fastmath=fm_fast).result
                            col_sum = arith.AddFOp(col_sum, wsc, fastmath=fm_fast).result
                    col_sum = arith.MulFOp(col_sum, kv_scale, fastmath=fm_fast).result

                    # --- Head reduction step 2: shuffle_xor butterfly (within wave) ---
                    # mfma.shuffle_offsets covers all lane groups so every lane ends up
                    # with the full H-wide head sum. Width is always 64 (per-wave).
                    for sh in mfma.shuffle_offsets:
                        peer    = _to_raw(ArithValue(col_sum).shuffle_xor(sh, 64))
                        col_sum = arith.AddFOp(col_sum, peer, fastmath=fm_fast).result

                    # --- Writer predicate: lane_div_N == 0 owns MFMA_N distinct cols ---
                    # `col >= start` is required: the tile loop is BKV-aligned
                    # below `start`, so it guards the pre-filled -inf in
                    # [aligned_start, start). (Both the single-grid alignment and
                    # the grid.y split can place the first tile below start.)
                    in_window = arith.andi(
                        _to_raw(arith.cmpi(
                            arith.CmpIPredicate.sge,
                            _to_raw(col),
                            _to_raw(starts[j]),
                        )),
                        _to_raw(arith.cmpi(
                            arith.CmpIPredicate.slt,
                            _to_raw(col),
                            _to_raw(ends[j]),
                        )),
                    )
                    is_writer = arith.andi(
                        _to_raw(arith.cmpi(
                            arith.CmpIPredicate.eq,
                            _to_raw(lane_div_N),
                            _to_raw(fx.Int32(0)),
                        )),
                        in_window,
                    )
                    with ir.InsertionPoint(scf.IfOp(is_writer).then_block):
                        out_row_t[col] = fx.Float32(col_sum)
                        scf.YieldOp([])

            scf.YieldOp([])

    @flyc.jit
    def launch_fp8_mqa_logits_mfma_r_w(
        Q: fx.Tensor,
        KV: fx.Tensor,
        kv_scales: fx.Tensor,
        weights: fx.Tensor,
        cu_starts: fx.Tensor,
        cu_ends: fx.Tensor,
        logits: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
        num_splits: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        n_blocks = arith.ceildivui(
            _to_raw(seq_len), _to_raw(fx.Int32(RPB))
        )
        gx = arith.index_cast(T.index, n_blocks)
        gy = arith.index_cast(T.index, _to_raw(num_splits))
        kernel._func.__name__ = _kname
        kernel(
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            logits,
            seq_len,
            seq_len_kv,
            stride_logits_s,
            num_splits,
        ).launch(grid=(gx, gy, 1), block=(MR_BLOCK_THREADS, 1, 1), stream=stream)

    return launch_fp8_mqa_logits_mfma_r_w


def _build_kernel_mfma_lds_pipe(*, num_heads: int, head_size: int, block_kv: int,
                                rows_per_block: int, waves_per_block: int,
                                mfma: MfmaAtom,
                                convert_q_fn: bool = False,
                                convert_kv_fn: bool = False,
                                swizzle: bool = False):
    """LDS double-buffered variant (gfx950 scaled atoms only).

    Parallel to ``_build_kernel_mfma_r_w`` but stages KV through a 2-slot LDS
    double buffer filled by async global->LDS DMA (``raw_ptr_buffer_load_lds``),
    with an explicit software pipeline (prefetch tile 0/1, then per-tile
    ``s_waitcnt`` + prefetch(i+2) + compute).

    Work partition (the key difference vs the direct-load builder):
      * All ``WPB`` waves cooperatively load ONE ``BKV``-wide K-tile into LDS and
        all read it -- so waves no longer own disjoint columns. Instead each wave
        owns a disjoint group of ``RPB`` query ROWS and iterates over ALL
        ``N_TILES`` columns of the shared LDS tile.
      * A block owns ``ROWS_PER_BLOCK = RPB * WPB`` query rows (wave ``w`` owns
        rows ``[w*RPB, (w+1)*RPB)``). KV reuse factor becomes ``RPB * WPB``.

    The epilogue (per-lane scalar ReLU/weight + ``shuffle_xor`` butterfly +
    predicated store) is kept identical to the direct-load builder.
    """
    H   = num_heads
    D   = head_size
    BKV = block_kv
    RPB = rows_per_block
    WPB = waves_per_block
    MR_BLOCK_THREADS = 64 * WPB
    ROWS_PER_BLOCK = RPB * WPB

    assert mfma.frag_bytes == 32, (
        "_build_kernel_mfma_lds_pipe currently supports only the CDNA4 scaled "
        "atoms (frag_bytes=32)."
    )
    assert H % mfma.MFMA_M == 0
    assert BKV % mfma.MFMA_N == 0
    assert D % mfma.MFMA_K == 0
    assert RPB >= 1 and WPB >= 1

    N_TILES = BKV // mfma.MFMA_N
    M_TILES = H   // mfma.MFMA_M
    K_STEPS = D   // mfma.MFMA_K

    # LDS double buffer: NUM_BUFFERS slots of [BKV, D] fp8 (row-major, row == KV
    # column index). Addressed as i32 dwords for the vector reads.
    NUM_BUFFERS = 2
    SLOT_BYTES  = BKV * D                 # fp8, 1 byte/elem
    SLOT_I32    = SLOT_BYTES // 4
    DMA_BYTES   = 16                       # dwordx4 async load per lane
    assert SLOT_BYTES % (MR_BLOCK_THREADS * DMA_BYTES) == 0, (
        f"SLOT_BYTES={SLOT_BYTES} must be divisible by "
        f"MR_BLOCK_THREADS*DMA_BYTES={MR_BLOCK_THREADS * DMA_BYTES}"
    )
    NUM_ASYNC_LOADS = SLOT_BYTES // (MR_BLOCK_THREADS * DMA_BYTES)

    # XOR swizzle (bank-conflict avoidance), aligned with the Gluon reference's
    # padded+swizzled shared K layout.  The slot stores [BKV, D] fp8 HEAD_SIZE-
    # contiguous, so per column the D bytes are DW_PER_COL i32 dwords.  A B-frag
    # read gathers a fixed CHUNK_DW-dword slice of the head dim across the 32/16
    # lanes of a KV-column group; since the per-column stride D/4 is a multiple
    # of the 32 LDS banks, every lane hits the same banks (up to 32-way
    # conflict).  XOR-ing the within-column chunk index with a function of the
    # column index n scatters the NC chunks across the banks, cutting the
    # conflict by NC (=D/frag_bytes) while keeping each frag read (and each
    # 16B DMA write) contiguous, because the XOR mask is a multiple of CHUNK_DW.
    #   phys_dword(n, c) = n*DW_PER_COL + (c XOR ((n & (NC-1)) * CHUNK_DW))
    DW_PER_COL = D // 4                    # i32 dwords per KV column (head dim)
    CHUNK_DW   = mfma.frag_bytes // 4      # dwords per B-frag read (=8)
    NC         = DW_PER_COL // CHUNK_DW    # chunks per column (D/frag_bytes)
    if swizzle:
        assert NC >= 2, (
            f"swizzle needs D/frag_bytes>=2 (D={D}, frag_bytes={mfma.frag_bytes})"
        )

    fm_fast = arith.FastMathFlags.fast
    mfma_fn = mfma.fn

    allocator = SmemAllocator(None, arch=arch)
    lds_off = allocator._align(allocator.ptr, 128)
    allocator.ptr = lds_off + NUM_BUFFERS * SLOT_BYTES

    _kname = (
        f"fp8_mqa_logits_H{H}_D{D}_mfma{mfma.name}"
        f"_bkv{BKV}_r{RPB}_w{WPB}_lds2{'_swizzled' if swizzle else ''}_flydsl"
    )

    @flyc.kernel(name=_kname, known_block_size=[MR_BLOCK_THREADS, 1, 1])
    def kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        kv_scales: fx.Tensor,
        weights: fx.Tensor,
        cu_starts: fx.Tensor,
        cu_ends: fx.Tensor,
        logits: fx.Tensor,
        seq_len: fx.Int32,          # padded to a multiple of ROWS_PER_BLOCK
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
        num_splits: fx.Int32,
    ):
        f32_0 = arith.constant(0.0, type=T.f32)
        _mfma_res_ty = Vec.make_type(mfma.ACC_ELEMS, fx.Float32)

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        # Block bid (reversed) owns rows [block_row0, block_row0+ROWS_PER_BLOCK).
        n_blocks = fx.Int32(arith.ceildivui(
            _to_raw(seq_len), _to_raw(fx.Int32(ROWS_PER_BLOCK))
        ))
        block_row0 = fx.Int32(arith.muli(
            _to_raw(n_blocks - bid - fx.Int32(1)),
            _to_raw(fx.Int32(ROWS_PER_BLOCK)),
        ))

        wave = fx.Int32(arith.divui(_to_raw(tid), _to_raw(fx.Int32(64))))
        lane = fx.Int32(arith.remui(_to_raw(tid), _to_raw(fx.Int32(64))))
        lane_div_N = fx.Int32(arith.divui(_to_raw(lane), _to_raw(fx.Int32(mfma.MFMA_N))))
        lane_mod_N = fx.Int32(arith.remui(_to_raw(lane), _to_raw(fx.Int32(mfma.MFMA_N))))
        lane_frag_off = fx.Int32(
            arith.muli(_to_raw(lane_div_N), _to_raw(fx.Int32(mfma.frag_bytes)))
        )
        # First row owned by THIS wave.
        wave_row0 = _i32_add(
            block_row0,
            fx.Int32(arith.muli(_to_raw(wave), _to_raw(fx.Int32(RPB)))),
        )

        q_i32 = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(KV, dtype=T.i32, shape=(-1,))
        kv_rsrc = kv_i32.rsrc
        sc_t = GTensor(kv_scales, dtype=T.f32, shape=(-1,))
        w_t = GTensor(weights, dtype=T.f32, shape=(-1, H))
        cs_t = GTensor(cu_starts, dtype=T.i32, shape=(-1,))
        ce_t = GTensor(cu_ends, dtype=T.i32, shape=(-1,))
        _stride_i64 = arith.extui(T.i64, _to_raw(stride_logits_s))

        def _make_out_row_t(row_i32):
            _ri64 = arith.extui(T.i64, _to_raw(row_i32))
            _byte = arith.muli(arith.muli(_ri64, _stride_i64),
                               arith.constant(4, type=T.i64))
            _idx  = arith.index_cast(T.index, _byte)
            return GTensor(logits, dtype=T.f32, shape=(-1,),
                           static_bytes_offset_i64=_idx)

        def _load_pack_i32x8(i32_view, byte_off_i32):
            dword_off = fx.Int32(
                arith.divui(_to_raw(byte_off_i32), _to_raw(fx.Int32(4)))
            )
            v4_lo = i32_view.vec_load((dword_off,), vec_size=4)
            dword_off_hi = _i32_add(dword_off, fx.Int32(4))
            v4_hi = i32_view.vec_load((dword_off_hi,), vec_size=4)
            return Vec(v4_lo).shuffle(v4_hi, list(range(8))).ir_value()

        # ---- LDS region + async-DMA base pointer ----
        base_ptr = allocator.get_base()
        region_ptr = SmemPtr(base_ptr, lds_off, T.i32,
                             shape=(NUM_BUFFERS * SLOT_I32,))
        lds_st = STensor(region_ptr, T.i32, shape=(NUM_BUFFERS * SLOT_I32,))
        lds_base_idx = memref_dialect.extract_aligned_pointer_as_index(lds_st.memptr)
        lds_ptr0 = buffer_ops.create_llvm_ptr(
            fx.Int64(lds_base_idx), address_space=3
        )

        def _dma_kv_tile_to_lds(slot_byte_i32, col0_i32):
            """Cooperatively async-copy KV[col0:col0+BKV, :] into LDS slot.

            All MR_BLOCK_THREADS threads participate; thread ``tid`` at load ``i``
            writes LDS byte ``(i*MR_BLOCK_THREADS + tid)*DMA_BYTES`` (relative to
            the slot), reading the matching linear byte of the row-major tile.
            OOB columns are clamped to ``seq_len_kv-1`` (harmless -- masked out in
            the epilogue by the per-row window predicate).
            """
            wave_slot_i32 = arith.addi(
                _to_raw(slot_byte_i32),
                arith.muli(_to_raw(wave), _to_raw(fx.Int32(64 * DMA_BYTES))),
            )
            wave_slot_scalar = rocdl.readfirstlane(
                fx.Int64.ir_type, arith.extui(T.i64, wave_slot_i32)
            )
            lds_ptr = buffer_ops.get_element_ptr(lds_ptr0, wave_slot_scalar)
            for i in range_constexpr(NUM_ASYNC_LOADS):
                lin_bytes = arith.muli(
                    arith.addi(_to_raw(fx.Int32(i * MR_BLOCK_THREADS)), _to_raw(tid)),
                    _to_raw(fx.Int32(DMA_BYTES)),
                )
                row_local = arith.divui(lin_bytes, _to_raw(fx.Int32(D)))
                d_off = arith.subi(
                    lin_bytes, arith.muli(row_local, _to_raw(fx.Int32(D)))
                )
                if const_expr(swizzle):
                    # The DMA writes lane-contiguously to physical byte lin_bytes,
                    # so to store the swizzled tile we fetch the *logical* element
                    # that maps to this physical slot: invert the within-column
                    # XOR (mask in bytes = (n & (NC-1)) * frag_bytes).
                    _mask_b = arith.muli(
                        arith.andi(row_local, _to_raw(fx.Int32(NC - 1))),
                        _to_raw(fx.Int32(mfma.frag_bytes)),
                    )
                    d_off = arith.xori(d_off, _mask_b)
                col = arith.addi(_to_raw(col0_i32), row_local)
                col_cl = arith.minsi(
                    col, _to_raw(fx.Int32(seq_len_kv) - fx.Int32(1))
                )
                voffset = arith.addi(
                    arith.muli(col_cl, _to_raw(fx.Int32(D))), d_off
                )
                if const_expr(i > 0):
                    lds_ptr = buffer_ops.get_element_ptr(
                        lds_ptr,
                        static_byte_offset=MR_BLOCK_THREADS * DMA_BYTES,
                    )
                rocdl.raw_ptr_buffer_load_lds(
                    kv_rsrc, lds_ptr, fx.Int32(DMA_BYTES), fx.Int32(voffset),
                    fx.Int32(0), fx.Int32(0), fx.Int32(1),
                )

        # ---- Preload this wave's RPB rows: window, Q A-frags, weights ----
        starts  = [None] * RPB
        ends    = [None] * RPB
        a_packs = [None] * RPB
        w_frag  = [None] * RPB
        for j in range_constexpr(RPB):
            row = _i32_add(wave_row0, fx.Int32(j))
            s = fx.Int32(cs_t[row])
            e = fx.Int32(ce_t[row])
            starts[j] = fx.Int32(arith.maxsi(_to_raw(s), _to_raw(fx.Int32(0))))
            ends[j]   = fx.Int32(arith.minsi(_to_raw(e), _to_raw(fx.Int32(seq_len_kv))))

            row_a = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                h_a    = _i32_add(fx.Int32(mi * mfma.MFMA_M), lane_mod_N)
                row_h  = _i32_add(
                    fx.Int32(arith.muli(_to_raw(row), _to_raw(fx.Int32(H)))), h_a
                )
                base_a = fx.Int32(arith.muli(_to_raw(row_h), _to_raw(fx.Int32(D))))
                for kk in range_constexpr(K_STEPS):
                    row_a[mi][kk] = _load_pack_i32x8(
                        q_i32,
                        _i32_add(base_a, _i32_add(fx.Int32(kk * mfma.MFMA_K), lane_frag_off)),
                    )
            a_packs[j] = row_a

            row_w = [[None] * mfma.ACC_ELEMS for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                for ii in range_constexpr(mfma.ACC_ELEMS):
                    static_off = mfma.acc_head_static_offsets[ii]
                    h_w = _i32_add(
                        fx.Int32(mi * mfma.MFMA_M + static_off),
                        fx.Int32(arith.muli(
                            _to_raw(lane_div_N),
                            _to_raw(fx.Int32(mfma.acc_head_group_stride)),
                        )),
                    )
                    row_w[mi][ii] = _to_raw(fx.Float32(w_t[row, h_w]))
            w_frag[j] = row_w

        # ---- Union KV window across ALL block rows (all waves cooperate) ----
        u_start = None
        u_end   = None
        for jj in range_constexpr(ROWS_PER_BLOCK):
            rr = _i32_add(block_row0, fx.Int32(jj))
            ss = arith.maxsi(_to_raw(fx.Int32(cs_t[rr])), _to_raw(fx.Int32(0)))
            ee = arith.minsi(_to_raw(fx.Int32(ce_t[rr])), _to_raw(fx.Int32(seq_len_kv)))
            if jj == 0:
                u_start = ss
                u_end   = ee
            else:
                u_start = arith.minsi(u_start, ss)
                u_end   = arith.maxsi(u_end, ee)
        tile_start = arith.muli(
            arith.divui(u_start, _to_raw(fx.Int32(BKV))),
            _to_raw(fx.Int32(BKV)),
        )
        tile_end = u_end

        # KV-column split across grid.y (identical to the direct-load builder).
        by = fx.block_idx.y
        win_tiles = arith.ceildivui(
            arith.subi(tile_end, tile_start), _to_raw(fx.Int32(BKV))
        )
        split_cols = arith.muli(
            arith.ceildivui(win_tiles, _to_raw(num_splits)),
            _to_raw(fx.Int32(BKV)),
        )
        tile_start = arith.addi(tile_start, arith.muli(_to_raw(by), split_cols))
        tile_end = arith.minsi(arith.addi(tile_start, split_cols), tile_end)

        n_tiles = arith.ceildivui(
            arith.maxsi(arith.subi(tile_end, tile_start), _to_raw(fx.Int32(0))),
            _to_raw(fx.Int32(BKV)),
        )

        # ---- Prologue: prefetch tile 0 -> buf 0, tile 1 -> buf 1 ----
        _dma_kv_tile_to_lds(fx.Int32(0), fx.Int32(tile_start))
        _dma_kv_tile_to_lds(
            fx.Int32(SLOT_BYTES),
            fx.Int32(arith.addi(tile_start, _to_raw(fx.Int32(BKV)))),
        )

        # ---- Steady-state software pipeline over BKV tiles ----
        lo   = _to_raw(fx.Index(fx.Int32(0)))
        hi   = _to_raw(fx.Index(fx.Int32(n_tiles)))
        step = _to_raw(fx.Index(fx.Int32(1)))
        tile_loop = scf.ForOp(lo, hi, step, [])
        with ir.InsertionPoint(tile_loop.body):
            t = fx.Int32(arith.index_cast(T.i32, tile_loop.induction_variable))
            col0 = arith.addi(
                tile_start, arith.muli(_to_raw(t), _to_raw(fx.Int32(BKV)))
            )
            parity = arith.remui(_to_raw(t), _to_raw(fx.Int32(2)))
            slot_byte  = arith.muli(parity, _to_raw(fx.Int32(SLOT_BYTES)))
            slot_dword = arith.muli(parity, _to_raw(fx.Int32(SLOT_I32)))

            # Wait for the current tile (keep the already-issued next tile in
            # flight), then sync so every wave sees the full LDS tile.
            rocdl.s_waitcnt(NUM_ASYNC_LOADS)
            gpu.barrier()

            # Read all B-frags for this tile from LDS into registers.
            b_packs        = [[None] * K_STEPS for _ in range_constexpr(N_TILES)]
            cols           = [None] * N_TILES
            kv_scales_tile = [None] * N_TILES
            for ni in range_constexpr(N_TILES):
                col = arith.addi(
                    arith.addi(col0, _to_raw(fx.Int32(ni * mfma.MFMA_N))),
                    _to_raw(lane_mod_N),
                )
                cols[ni] = fx.Int32(col)
                col_cl = fx.Int32(
                    arith.minsi(col, _to_raw(fx.Int32(seq_len_kv) - fx.Int32(1)))
                )
                kv_scales_tile[ni] = _to_raw(fx.Float32(sc_t[col_cl]))
                col_local = arith.addi(
                    _to_raw(fx.Int32(ni * mfma.MFMA_N)), _to_raw(lane_mod_N)
                )
                for kk in range_constexpr(K_STEPS):
                    if const_expr(swizzle):
                        # phys_dword = n*DW_PER_COL
                        #            + ((c_bytes/4) XOR ((n & (NC-1)) * CHUNK_DW))
                        c_dword = arith.divui(
                            arith.addi(
                                _to_raw(fx.Int32(kk * mfma.MFMA_K)),
                                _to_raw(lane_frag_off),
                            ),
                            _to_raw(fx.Int32(4)),
                        )
                        _mask_dw = arith.muli(
                            arith.andi(col_local, _to_raw(fx.Int32(NC - 1))),
                            _to_raw(fx.Int32(CHUNK_DW)),
                        )
                        swz_c = arith.xori(c_dword, _mask_dw)
                        frag_dword = arith.addi(
                            arith.muli(col_local, _to_raw(fx.Int32(DW_PER_COL))),
                            swz_c,
                        )
                    else:
                        frag_byte = arith.addi(
                            arith.addi(
                                arith.muli(col_local, _to_raw(fx.Int32(D))),
                                _to_raw(fx.Int32(kk * mfma.MFMA_K)),
                            ),
                            _to_raw(lane_frag_off),
                        )
                        frag_dword = arith.divui(frag_byte, _to_raw(fx.Int32(4)))

                    read_dword = arith.index_cast(
                        T.index, arith.addi(slot_dword, frag_dword)
                    )
                    b_packs[ni][kk] = lds_st.vec_load((read_dword,), vec_size=8)

            # Readers done -> safe to overwrite this slot with tile i+2.
            gpu.barrier()
            col0_next = arith.addi(
                tile_start,
                arith.muli(
                    arith.addi(_to_raw(t), _to_raw(fx.Int32(2))),
                    _to_raw(fx.Int32(BKV)),
                ),
            )
            _dma_kv_tile_to_lds(fx.Int32(slot_byte), fx.Int32(col0_next))

            # ---- Per-row MFMA + epilogue (this wave's RPB rows, all columns) ----
            for j in range_constexpr(RPB):
                row = _i32_add(wave_row0, fx.Int32(j))
                out_row_t = _make_out_row_t(row)
                for ni in range_constexpr(N_TILES):
                    col      = cols[ni]
                    kv_scale = kv_scales_tile[ni]
                    col_sum  = _to_raw(f32_0)
                    for mi in range_constexpr(M_TILES):
                        acc = Vec.filled(mfma.ACC_ELEMS, 0.0, fx.Float32)
                        for kk in range_constexpr(K_STEPS):
                            acc = mfma_fn(
                                _mfma_res_ty,
                                mfma.make_operands(a_packs[j][mi][kk], b_packs[ni][kk], acc),
                            )
                        for ii in range_constexpr(mfma.ACC_ELEMS):
                            score   = Vec(acc)[ii].ir_value()
                            relu    = arith.maximumf(score, _to_raw(f32_0))
                            wsc     = arith.MulFOp(relu, w_frag[j][mi][ii], fastmath=fm_fast).result
                            col_sum = arith.AddFOp(col_sum, wsc, fastmath=fm_fast).result
                    col_sum = arith.MulFOp(col_sum, kv_scale, fastmath=fm_fast).result

                    for sh in mfma.shuffle_offsets:
                        peer    = _to_raw(ArithValue(col_sum).shuffle_xor(sh, 64))
                        col_sum = arith.AddFOp(col_sum, peer, fastmath=fm_fast).result

                    in_window = arith.andi(
                        _to_raw(arith.cmpi(
                            arith.CmpIPredicate.sge,
                            _to_raw(col),
                            _to_raw(starts[j]),
                        )),
                        _to_raw(arith.cmpi(
                            arith.CmpIPredicate.slt,
                            _to_raw(col),
                            _to_raw(ends[j]),
                        )),
                    )
                    is_writer = arith.andi(
                        _to_raw(arith.cmpi(
                            arith.CmpIPredicate.eq,
                            _to_raw(lane_div_N),
                            _to_raw(fx.Int32(0)),
                        )),
                        in_window,
                    )
                    with ir.InsertionPoint(scf.IfOp(is_writer).then_block):
                        out_row_t[col] = fx.Float32(col_sum)
                        scf.YieldOp([])

            scf.YieldOp([])

    @flyc.jit
    def launch_fp8_mqa_logits_mfma_lds_pipe(
        Q: fx.Tensor,
        KV: fx.Tensor,
        kv_scales: fx.Tensor,
        weights: fx.Tensor,
        cu_starts: fx.Tensor,
        cu_ends: fx.Tensor,
        logits: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
        num_splits: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_blocks = arith.ceildivui(
            _to_raw(seq_len), _to_raw(fx.Int32(ROWS_PER_BLOCK))
        )
        gx = arith.index_cast(T.index, n_blocks)
        gy = arith.index_cast(T.index, _to_raw(num_splits))
        kernel._func.__name__ = _kname
        kernel(
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            logits,
            seq_len,
            seq_len_kv,
            stride_logits_s,
            num_splits,
        ).launch(grid=(gx, gy, 1), block=(MR_BLOCK_THREADS, 1, 1), stream=stream)

    return launch_fp8_mqa_logits_mfma_lds_pipe


# --------------------------------------------------------------------------- #
# Kernel-variant registry.
#
# Variant tags follow the scheme ``"mfma<MxNxK>_bkv<B>_r<RPB>_w<WPB>"`` where:
#   MxNxK  = MFMA tile dimensions (e.g. 16x16x32 or 32x32x16)
#   B      = block_kv (KV columns per tile-loop iteration)
#   RPB    = rows per block (Q-row KV-reuse factor)
#   WPB    = waves per block (column-split parallelism factor)
#
# To add a new variant, register a lambda below; no other call sites need
# changing (tests and benchmarks use the env var / ``variant=`` arg).
#
# Note: each MFMA-variant lambda hardcodes ``block_kv`` (bkv is part of the
# tag), overriding whatever ``block_kv`` the caller passed to
# ``compile_fp8_mqa_logits``.
# --------------------------------------------------------------------------- #

def _mfma16_bkv(bkv, r, w):
    """Helper: lambda for a 16x16x32-fp8 variant with given bkv/r/w."""
    return lambda **kw: _build_kernel_mfma_r_w(
        **{**kw, "block_kv": bkv},
        mfma=_MFMA16, rows_per_block=r, waves_per_block=w,
    )

def _mfma32_bkv(bkv, r, w):
    """Helper: lambda for a 32x32x16-fp8 variant with given bkv/r/w."""
    return lambda **kw: _build_kernel_mfma_r_w(
        **{**kw, "block_kv": bkv},
        mfma=_MFMA32, rows_per_block=r, waves_per_block=w,
    )

def _mfma16k128_bkv(bkv, r, w):
    """Helper: lambda for a gfx950 16x16x128-scaled-fp8 variant with given bkv/r/w."""
    return lambda **kw: _build_kernel_mfma_r_w(
        **{**kw, "block_kv": bkv},
        mfma=_MFMA16_K128, rows_per_block=r, waves_per_block=w,
    )

def _mfma32k64_bkv(bkv, r, w):
    """Helper: lambda for a gfx950 32x32x64-scaled-fp8 variant with given bkv/r/w."""
    return lambda **kw: _build_kernel_mfma_r_w(
        **{**kw, "block_kv": bkv},
        mfma=_MFMA32_K64, rows_per_block=r, waves_per_block=w,
    )

def _mfma32k64_lds_bkv(bkv, r, w, sw=False):
    """Helper: lambda for a gfx950 32x32x64-scaled-fp8 LDS double-buffered
    variant with given bkv/r/w (all WPB waves share one LDS K-tile).
    ``sw=True`` enables the XOR bank-conflict swizzle."""
    return lambda **kw: _build_kernel_mfma_lds_pipe(
        **{**kw, "block_kv": bkv},
        mfma=_MFMA32_K64, rows_per_block=r, waves_per_block=w, swizzle=sw,
    )

def _mfma16k128_lds_bkv(bkv, r, w, sw=False):
    """Helper: lambda for a gfx950 16x16x128-scaled-fp8 LDS double-buffered
    variant with given bkv/r/w (all WPB waves share one LDS K-tile).
    ``sw=True`` enables the XOR bank-conflict swizzle."""
    return lambda **kw: _build_kernel_mfma_lds_pipe(
        **{**kw, "block_kv": bkv},
        mfma=_MFMA16_K128, rows_per_block=r, waves_per_block=w, swizzle=sw,
    )

_VARIANT_BUILDERS = {
    # --- 16x16x32 fp8 variants (bkv=64) ---
    "mfma16x16x32_bkv64_r1_w1":   _mfma16_bkv(64,  1, 1),
    # --- 16x16x32 fp8 variants (bkv=128) ---
    "mfma16x16x32_bkv128_r1_w1":  _mfma16_bkv(128, 1, 1),
    "mfma16x16x32_bkv128_r2_w1":  _mfma16_bkv(128, 2, 1),
    "mfma16x16x32_bkv128_r4_w1":  _mfma16_bkv(128, 4, 1),
    "mfma16x16x32_bkv128_r1_w4":  _mfma16_bkv(128, 1, 4),
    "mfma16x16x32_bkv128_r2_w2":  _mfma16_bkv(128, 2, 2),
    "mfma16x16x32_bkv128_r2_w4":  _mfma16_bkv(128, 2, 4),
    "mfma16x16x32_bkv128_r4_w2":  _mfma16_bkv(128, 4, 2),
    # --- 16x16x32 fp8 variants (bkv=256) ---
    "mfma16x16x32_bkv256_r1_w1":  _mfma16_bkv(256, 1, 1),
    "mfma16x16x32_bkv256_r2_w2":  _mfma16_bkv(256, 2, 2),
    # --- 32x32x16 fp8 variants (bkv=128) ---
    "mfma32x32x16_bkv128_r1_w1":  _mfma32_bkv(128, 1, 1),
    "mfma32x32x16_bkv128_r2_w1":  _mfma32_bkv(128, 2, 1),
    "mfma32x32x16_bkv128_r2_w2":  _mfma32_bkv(128, 2, 2),
}

if arch == "gfx950":
    # CDNA4 scaled MFMA atoms (K=128/64) -- gfx950-only: these instructions
    # require native FN operands (reject FNUZ) and don't exist on gfx942.
    _VARIANT_BUILDERS.update({
        # Direct load variants (all waves load their own KV tile, no LDS sharing). WPB>=1 only.
        "mfma16x16x128_bkv128_r1_w1": _mfma16k128_bkv(128, 1, 1),
        "mfma16x16x128_bkv128_r2_w1": _mfma16k128_bkv(128, 2, 1),
        "mfma16x16x128_bkv128_r1_w2": _mfma16k128_bkv(128, 1, 2),
        "mfma16x16x128_bkv128_r2_w2": _mfma16k128_bkv(128, 2, 2),
        "mfma32x32x64_bkv128_r1_w1":  _mfma32k64_bkv(128, 1, 1),
        "mfma32x32x64_bkv128_r2_w1":  _mfma32k64_bkv(128, 2, 1),
        "mfma32x32x64_bkv128_r1_w2":  _mfma32k64_bkv(128, 1, 2),
        "mfma32x32x64_bkv128_r2_w2":  _mfma32k64_bkv(128, 2, 2),
        # LDS double-buffered variants (all WPB waves share one LDS K-tile,
        # waves partition rows -> block owns RPB*WPB rows). WPB>=2 only (the
        # whole point is cross-wave KV reuse via the shared LDS tile).
        # --- 32x32x64 (MFMA_N=32) ---
        "mfma32x32x64_bkv64_r1_w2_lds2":  _mfma32k64_lds_bkv(64,  1, 2, sw=True),
        "mfma32x32x64_bkv64_r2_w2_lds2":  _mfma32k64_lds_bkv(64,  2, 2, sw=True),
        "mfma32x32x64_bkv64_r2_w4_lds2":  _mfma32k64_lds_bkv(64,  2, 4, sw=True),
        "mfma32x32x64_bkv128_r1_w2_lds2": _mfma32k64_lds_bkv(128, 1, 2, sw=True),
        "mfma32x32x64_bkv128_r2_w2_lds2": _mfma32k64_lds_bkv(128, 2, 2, sw=True),
        "mfma32x32x64_bkv128_r2_w4_lds2": _mfma32k64_lds_bkv(128, 2, 4, sw=True),
        "mfma32x32x64_bkv256_r1_w2_lds2": _mfma32k64_lds_bkv(256, 1, 2, sw=True),
        "mfma32x32x64_bkv256_r2_w2_lds2": _mfma32k64_lds_bkv(256, 2, 2, sw=True),
        # --- 16x16x128 (MFMA_N=16, needs D>=128) ---
        "mfma16x16x128_bkv64_r2_w2_lds2":  _mfma16k128_lds_bkv(64,  2, 2, sw=True),
        "mfma16x16x128_bkv128_r1_w2_lds2": _mfma16k128_lds_bkv(128, 1, 2, sw=True),
        "mfma16x16x128_bkv128_r2_w2_lds2": _mfma16k128_lds_bkv(128, 2, 2, sw=True),
        "mfma16x16x128_bkv128_r2_w4_lds2": _mfma16k128_lds_bkv(128, 2, 4, sw=True),
        "mfma16x16x128_bkv256_r2_w2_lds2": _mfma16k128_lds_bkv(256, 2, 2, sw=True),
    })

KERNEL_VARIANTS = tuple(_VARIANT_BUILDERS.keys())
DEFAULT_VARIANT = "mfma16x16x32_bkv128_r2_w2" if arch == "gfx942" else "mfma32x32x64_bkv128_r2_w2"

def _auto_variant(seq_len, seq_len_kv):
    """Pick (RPB, WPB) from the problem shape: RPB=2 always; WPB=2 packs more
    column tiles per wave when M and N are both large, else WPB=4 for more
    wavefronts on small-M / short-window shapes."""
    wpb = 2 if (seq_len >= 2048 and seq_len_kv >= 8192) else 4
    return f"mfma16x16x32_bkv128_r2_w{wpb}"

def _resolve_variant(variant, seq_len, seq_len_kv):
    """Effective variant: explicit ``variant=`` > env var > shape-adaptive."""
    tag = (
        variant
        or os.environ.get("FLYDSL_FP8_MQA_LOGITS_VARIANT")
        or _auto_variant(seq_len, seq_len_kv)
    )
    if tag not in _VARIANT_BUILDERS:
        raise ValueError(
            f"unknown fp8_mqa_logits variant {tag!r}; "
            f"available: {list(KERNEL_VARIANTS)}"
        )
    return tag

@lru_cache(maxsize=32)
def compile_fp8_mqa_logits(
    *,
    num_heads: int,
    head_size: int,
    paged: bool = False,
    variant: str = DEFAULT_VARIANT,
    convert_q_fn: bool = False,
    convert_kv_fn: bool = False,
):
    """Return a cached, compiled FlyDSL launcher for the given shape config.

    Parameters
    ----------
    num_heads : int
        Number of indexer query heads (compile-time constant, power of two).
    head_size : int
        Head dimension D (compile-time constant, power of two; D in {64, 128}).
    paged : bool
        Reserved for the Phase-2 paged variant. Must be False for now.
    variant : str
        Which kernel version to build (see ``KERNEL_VARIANTS``). Tags follow
        ``"mfma<MxNxK>_bkv<B>_r<RPB>_w<WPB>"``.
    convert_q_fn : bool
        If True, Q bytes are FP8 FN and the kernel converts them to FNUZ
        in-register before the MFMA (applies to all ``mfma*`` variants).
    convert_kv_fn : bool
        If True, KV bytes are FP8 FN and the kernel converts them to FNUZ
        in-register before the MFMA (applies to all ``mfma*`` variants).
    """
    if paged:
        raise NotImplementedError(
            "Paged FlyDSL fp8_mqa_logits is not implemented."
        )
    if variant not in _VARIANT_BUILDERS:
        raise ValueError(
            f"unknown fp8_mqa_logits variant {variant!r}; "
            f"available: {list(KERNEL_VARIANTS)}"
        )
    launcher = _VARIANT_BUILDERS[variant](
        num_heads=num_heads, head_size=head_size,
        convert_q_fn=convert_q_fn, convert_kv_fn=convert_kv_fn,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


def flydsl_fp8_mqa_logits(
    Q,
    KV,
    kv_scales,
    weights,
    cu_starts,
    cu_ends,
    clean_logits=True,
    stream=None,
    variant=None,
):
    """FlyDSL gfx942 FP8 MQA logits -- drop-in for the Triton ``fp8_mqa_logits``.

    Q:            [seq_len, NUM_HEADS, HEAD_SIZE], dtype float8
    KV:           [seq_len_kv, HEAD_SIZE], dtype float8
    kv_scales:    [seq_len_kv], dtype float32
    weights:      [seq_len, NUM_HEADS], dtype float32
    cu_starts:    [seq_len], dtype int32, per-row window start (inclusive)
    cu_ends:      [seq_len], dtype int32, per-row window end (exclusive)
    clean_logits: bool. If True, positions outside [cu_starts[i], cu_ends[i])
                  in row i are written as -inf. If False, the kernel skips
                  those positions and the caller owns whatever is left there.
    stream:       optional HIP stream; defaults to the current stream.
    variant:      optional kernel-version tag (see ``KERNEL_VARIANTS``). If None,
                  resolved from the ``FLYDSL_FP8_MQA_LOGITS_VARIANT`` env var,
                  defaulting to ``DEFAULT_VARIANT`` (``"mfma16x16x32_bkv128_r2_w2"``).

    Returns
    -------
    logits: [seq_len, seq_len_kv], dtype float32.
    """
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."

    # FlyDSL's DLPack tensor adaptor rejects 0-dim tensors, but the per-token
    # ``kv_scales`` collapses to a scalar when seq_len_kv == 1 (and ``weights``
    # could too). Reshape the 1-D / 2-D inputs back to their logical rank so the
    # kernel always sees indexable tensors (matches the Triton pointer path).
    kv_scales = kv_scales.reshape(seq_len_kv)
    weights = weights.reshape(seq_len, num_heads)
    cu_starts = cu_starts.reshape(seq_len)
    cu_ends = cu_ends.reshape(seq_len)

    # The gfx942 fp8 MFMA (v_mfma_f32_16x16x32_fp8_fp8) always interprets
    # operands as e4m3 FNUZ (bias 8). When Q or KV arrive in e4m3 FN (OCP,
    # bias 7, max 448), two corrections are needed:
    #   1. In-kernel byte conversion: dequant-as-FNUZ × 2 -> requant-to-FNUZ
    #      recovers the true FN value in FNUZ encoding for 255/256 byte patterns
    #      (only 0x80 = FN -0 -> FNUZ NaN differs; values > 240 saturate).
    #   2. kv_scales compensation: multiply by 2 per FN operand. Since
    #      logits = sum_h ReLU(QK x scale) x w and ReLU is pos-homogeneous,
    #      this compensates the systematic 2x factor from the FN/FNUZ bias diff.
    # Combined, in-kernel conversion handles the per-byte encoding while
    # kv_scales compensation handles the overall numeric factor.
    # The 'mfma' lernel variant does in-kernel halve-and-requant:
    # dequant-as-FNUZ (= FN_value/2) * 0.5 -> requant-to-FNUZ, keeping all
    # values <= 120 < 240 (safely within FNUZ range). The 2x factor per FN
    # operand is compensated via kv_scales.
    if arch == "gfx942":
        _fnuz = torch.float8_e4m3fnuz
        convert_q_fn = Q.dtype != _fnuz
        convert_kv_fn = KV.dtype != _fnuz
        scale_mul = 1.0
        if convert_q_fn:
            scale_mul *= 2.0
        if convert_kv_fn:
            scale_mul *= 2.0
        if scale_mul != 1.0:
            kv_scales = kv_scales.to(torch.float32) * scale_mul
    else:
        convert_q_fn = False
        convert_kv_fn = False

    variant = _resolve_variant(variant, seq_len, seq_len_kv)

    launcher = compile_fp8_mqa_logits(
        num_heads=num_heads,
        head_size=head_size,
        paged=False,
        variant=variant,
        convert_q_fn=convert_q_fn,
        convert_kv_fn=convert_kv_fn,
    )

    # mfma*_r* kernels require seq_len padded to a multiple of rows_per_block so
    # every block owns exactly RPB rows.  Padded rows get empty windows (start ==
    # end == 0) so the kernel writes nothing for them; the output is sliced back
    # to the original seq_len after the launch.
    # Parse BKV, RPB and WPB from variant tag "mfma<shape>_bkv<B>_r<N>_w<M>".
    # For _lds2 variants all WPB waves share one LDS K-tile and partition rows,
    # so a block owns RPB*WPB rows -> seq_len must be padded to that multiple.
    _tag_match = re.match(r"mfma\d+x\d+x\d+_bkv(\d+)_r(\d+)_w(\d+)", variant)
    _BKV = int(_tag_match.group(1)) if _tag_match else 128
    _RPB = int(_tag_match.group(2)) if _tag_match else 1
    _WPB = int(_tag_match.group(3)) if _tag_match else 1
    _is_lds2 = "_lds2" in variant
    _rows_per_block_eff = _RPB * _WPB if _is_lds2 else _RPB
    seq_len_padded = (
        (seq_len + _rows_per_block_eff - 1) // _rows_per_block_eff
    ) * _rows_per_block_eff
    if seq_len_padded != seq_len:
        pad = seq_len_padded - seq_len
        Q = torch.cat([Q, Q.new_zeros((pad, num_heads, head_size))], dim=0)
        weights = torch.cat([weights, weights.new_zeros((pad, num_heads))], dim=0)
        cu_starts = torch.cat([cu_starts, cu_starts.new_zeros(pad)], dim=0)
        cu_ends = torch.cat([cu_ends, cu_ends.new_zeros(pad)], dim=0)

    # Match the Triton launcher's -inf-prefill / padding behavior so the two
    # produce identically-shaped, identically-masked outputs.
    aligned_size = 256
    seq_len_kv_aligned = (
        (seq_len_kv + aligned_size - 1) // aligned_size * aligned_size
    )
    if clean_logits:
        logits = torch.full(
            (seq_len_padded, seq_len_kv_aligned),
            fill_value=-float("inf"),
            dtype=torch.float32,
            device=Q.device,
        )[:, :seq_len_kv]
    else:
        logits = torch.empty(
            (seq_len_padded, seq_len_kv_aligned),
            dtype=torch.float32,
            device=Q.device,
        )[:, :seq_len_kv]

    num_splits = _auto_num_splits(
        seq_len_padded, seq_len_kv, _rows_per_block_eff, _BKV, Q.device.index
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    with torch.cuda.device(Q.device.index):
        _run_compiled(
            launcher,
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            logits,
            int(seq_len_padded),
            int(seq_len_kv),
            int(logits.stride(0)),
            int(num_splits),
            stream,
        )

    return logits[:seq_len, :]
