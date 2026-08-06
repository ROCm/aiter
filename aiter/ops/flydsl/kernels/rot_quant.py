# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused online Hadamard rotation + MXFP4 quantization (FlyDSL).

Entry point: ``flydsl_rot_quant(x, RS, ...)`` -- one kernel that block-diagonally
rotates a bf16 activation, quantizes it to MXFP4 and writes the e8m0 scales,
optionally already in the CK-gemm swizzle that ``aiter.gemm_a4w4`` consumes.

This is the activation-side counterpart of a rotated (QuaRot/SpinQuant-style) MXFP4
checkpoint: the offline rotation is folded into the weights, and the matching online
rotation of the activation has to happen at every layer, every token. Doing it as
rotate -> pack -> swizzle in three passes costs three round trips through HBM on a
purely memory-bound op; this fuses all three into one.

Relation to the existing aiter ops:
  - ``rotate_activation_fp4quant_inplace`` (csrc/kernels/dsv4_rotate_quant.cu) rotates,
    fp4-quantizes and **dequantizes back to bf16 in place** at a fixed ``dim`` of 128
    or 256. That is the simulation path; it does not produce a gemm input.
  - ``dynamic_mxfp4_quant`` produces packed fp4 + e8m0, but does not rotate.
  - ``shuffle_scale`` re-lays-out the scales in a separate pass.
This op is the fusion of all three, and is the only one that emits the CK scale
swizzle from inside the quantizing kernel.

Mapping: work = M * (K//COLS) independent thread-blocks; one thread per COLS-wide
chunk, which holds COLS//RS independent RS-wide Hadamard blocks (no cross-lane comms).
The FWHT is scalar-unrolled in registers (Sylvester, ``(a+b, a-b)`` pairing).
RS=64, COLS=64 => 64 f32 regs/thread.

Requires gfx950: the quantizer is a single ``v_cvt_scalef32_pk_fp4_f32`` per byte.

The e8m0 block scale is the shared ``quant_utils.emit_mx_e8m0_scale`` builder in
aiter's default ``MxScaleRoundMode.RoundUp`` -- no bespoke scale convention here.

Correctness is covered by ``op_tests/test_flydsl_rot_quant.py``, which checks the
packed fp4 and the e8m0 scales byte-for-byte against a torch reference, the fused
swizzle against ``aiter.ops.shuffle.shuffle_scale``, and the whole thing end-to-end
through ``gemm_a4w4``.
"""

# NOTE: do NOT add `from __future__ import annotations` to this file.
# PEP 563 turns all annotations into strings, which defeats flydsl's
# JitFunction._make_cache_key runtime detection:
#   is_runtime = hasattr(ann, "__get_c_pointers__")
# A string like 'fx.Int32' fails that check, so flydsl would treat the
# `total_blocks` Int32 parameter as a compile-time constant and embed its
# VALUE in the cache key -- every distinct M would then trigger a fresh
# JIT compile instead of hitting the in-memory CallState cache.

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as A
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import BFloat16, Float32, Int32, T, Vector
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import buffer_ops, vector

# Only the bare-int mirrors are imported here: the pybind11 MxScaleRoundMode /
# MxDtype in the same module lazy-load module_aiter_core on attribute access,
# which is incompatible with the FlyDSL AOT pass in setup.py (see quant_utils).
from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale
from aiter.utility.mx_types import MX_DEFAULT_ROUND_MODE as _DEFAULT_MODE
from aiter.utility.mx_types import MxDtypeInt as _D

from .tensor_shim import _run_compiled, _to_raw

QG = 32  # mxfp4 group size
LVEC = 8  # bf16 buffer_load width (128-bit loads)
SUPPORTED_RS = (32, 64, 128)


def _ic(v):
    return A.ConstantOp(T.i32, ir.IntegerAttr.get(T.i32, v)).result


def _fabs_raw(x):
    xi = A.BitcastOp(T.i32, _to_raw(x)).result
    m = A.AndIOp(xi, _ic(0x7FFFFFFF)).result
    return A.BitcastOp(T.f32, m).result


def _amax_raw(nums, tree=False):
    """abs-max over a group. ``tree`` uses a balanced reduction (shorter dep
    chain -> lower latency); bit-exact vs the chain form because max over the
    non-negative |v| is associative/commutative (no NaN, +0 only)."""
    absv = [_fabs_raw(t) for t in nums]
    if not tree:
        acc = absv[0]
        for t in absv[1:]:
            acc = A.MaximumFOp(acc, t).result
        return acc
    level = absv
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(A.MaximumFOp(level[i], level[i + 1]).result)
        if len(level) & 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def _e8m0_and_scale(amax_raw, fold_k=0):
    """Return (e8m0_i8_raw, hw_scale_f32_raw) for one mxfp4 group.

    The scale itself is the shared ``emit_mx_e8m0_scale`` builder in aiter's
    default ``RoundUp`` mode (``ceil_pow2(amax / 6)``) -- this op has no scale
    convention of its own.

    ``fold_k`` folds a global 2**(-fold_k) activation scale (the 1/sqrt(RS) FWHT
    normalization, when it is an exact power of two) into the exponent instead of a
    per-element float multiply. RoundUp is exponent-linear, so scaling the input by
    a power of two is exactly a shift of the result:
      stored e8m0 = max(e8m0(amax) - fold_k, 0)
      pack scale  = 2**(stored_e8m0 + fold_k - 127)     (so cvt(v, scale) ==
                                                         cvt(v * 2**-fold_k, 2**e8m0))
    fold_k=0 is the unfolded form (values pre-scaled by a float multiply instead).
    """
    e8 = _to_raw(
        emit_mx_e8m0_scale(ArithValue(amax_raw), mode=_DEFAULT_MODE, dtype=_D.FP4_E2M1)
    )
    if fold_k:
        e8 = A.MaxSIOp(A.SubIOp(e8, _ic(fold_k)).result, _ic(0)).result
        hs = A.ShLIOp(A.AddIOp(e8, _ic(fold_k)).result, _ic(23)).result
    else:
        hs = A.ShLIOp(e8, _ic(23)).result
    e8_i8 = A.TruncIOp(T.i8, e8).result
    hw = A.BitcastOp(T.f32, hs).result
    return e8_i8, hw


def _bits(v, shift, mask=None):
    """(v >> shift) & mask on raw i32 values; ``mask=None`` skips the AND (the
    high bits are already known zero), so no extra op is emitted."""
    if shift:
        v = A.ShRUIOp(v, _ic(shift)).result
    return A.AndIOp(v, _ic(mask)).result if mask is not None else v


def _horner(acc, terms):
    """acc = ((acc*m0 + a0)*m1 + a1)... over ``terms`` = [(mul, add), ...]."""
    for mul, add in terms:
        acc = A.AddIOp(A.MulIOp(acc, _ic(mul)).result, add).result
    return acc


def _pack4_i32(bytes4):
    """Pack 4 i8 raw values into one i32 (little-endian: b0 | b1<<8 | ...)."""
    acc = A.AndIOp(A.ExtUIOp(T.i32, bytes4[0]).result, _ic(0xFF)).result
    for k in range(1, 4):
        bk = A.AndIOp(A.ExtUIOp(T.i32, bytes4[k]).result, _ic(0xFF)).result
        bk = A.ShLIOp(bk, _ic(8 * k)).result
        acc = A.OrIOp(acc, bk).result
    return acc


def _pack_fp4(even_raw, odd_raw, hw_scale_raw):
    """v_cvt_scalef32_pk_fp4_f32: pack (even,odd) f32 -> byte (2x fp4), given scale."""
    packed = _llvm.inline_asm(
        T.i32,
        [even_raw, odd_raw, hw_scale_raw],
        "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        has_side_effects=False,
    )
    return A.TruncIOp(T.i8, packed).result


def _fwht_stage(vals, G, H):
    """One Sylvester FWHT stage: pair (lo, lo+H) -> (a+b, a-b)."""
    W = G * 2 * H
    out = [None] * W
    for g in range(G):
        base = g * 2 * H
        for h in range(H):
            lo = base + h
            hi = lo + H
            a, b = vals[lo], vals[hi]
            out[lo] = a + b
            out[hi] = a - b
    return out


def _fwht(vals, RS):
    stages = {
        32: [(16, 1), (8, 2), (4, 4), (2, 8), (1, 16)],
        64: [(32, 1), (16, 2), (8, 4), (4, 8), (2, 16), (1, 32)],
        128: [(64, 1), (32, 2), (16, 4), (8, 8), (4, 16), (2, 32), (1, 64)],
    }[RS]
    for G, H in stages:
        vals = _fwht_stage(vals, G, H)
    return vals


def _vec_i32(words):
    """Build a vector<len x i32> MLIR value from a list of i32 raw values."""
    vty = T.vec(len(words), T.i32)
    return vector.from_elements(vty, [_to_raw(w) for w in words])


def _build_rot_quant(
    K: int,
    RS: int,
    BLOCK: int,
    COLS: int | None = None,
    SVEC: int = 1,
    AMAX: str = "tree",
    SHUF: bool = False,
):
    # Each thread owns COLS contiguous cols = NG independent RS-blocks (COLS batching
    # -> longer HBM bursts). COLS must be a multiple of RS, QG and LVEC.
    if COLS is None:
        COLS = RS
    assert COLS % RS == 0 and COLS % QG == 0 and COLS % LVEC == 0 and K % COLS == 0
    NG = COLS // RS  # RS-blocks per thread
    NUM_QG = COLS // QG  # mxfp4 groups per thread
    HALF = COLS // 2  # fp4 bytes per thread
    NW = HALF // 4  # i32 store words
    NLOAD = COLS // LVEC  # vec8 loads per thread
    assert NW % SVEC == 0, f"NW={NW} not divisible by SVEC={SVEC}"
    # SHUF: write e8m0 scales directly at the canonical CK-gemm shuffle offset
    # (see aiter.ops.shuffle.shuffle_scale) instead of the natural [M, K/32] layout,
    # fusing the scale swizzle into this kernel -- no separate shuffle pass and no
    # padding-tile zeroing, since the pad cells are don't-care (the CK gemm slices
    # its output back to [:M]).
    NBCOL = K // COLS  # COLS-wide col-chunks per row (== scale-col/NUM_QG)
    SN = ((K // QG) + 7) // 8 * 8  # padded scale-col count (ceil8), constexpr
    SN8 = SN // 8
    rs_scale = 1.0 / math.sqrt(RS)
    _hl = 0.5 * math.log2(RS)
    fold = float(_hl).is_integer()
    fold_k = int(_hl) if fold else 0

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def rq_kernel(
        x_ptr: fx.Tensor,
        fp4_ptr: fx.Tensor,
        sc_ptr: fx.Tensor,
        total_blocks: Int32,
    ):
        i32 = T.i32
        bid = ArithValue(fx.block_idx.x)
        tid = ArithValue(fx.thread_idx.x)
        gid = bid * arith.constant(BLOCK, type=i32) + tid

        tb = _to_raw(total_blocks)
        x_nb = A.MulIOp(tb, _ic(COLS * 2)).result  # bf16 bytes
        fp4_nb = A.MulIOp(tb, _ic(HALF)).result  # fp4 bytes
        if const_expr(SHUF):
            # padded [sm, sn] scale buffer: bound to sm*sn bytes (sm=ceil256(M), sn
            # constexpr) so out-of-range lanes' scattered stores are dropped by the
            # buffer descriptor. M = total_blocks // NBCOL.
            m_tot = A.DivUIOp(tb, _ic(NBCOL)).result
            m_pad = A.ShLIOp(
                A.ShRUIOp(A.AddIOp(m_tot, _ic(255)).result, _ic(8)).result, _ic(8)
            ).result
            sc_nb = A.MulIOp(m_pad, _ic(SN)).result
        else:
            sc_nb = A.MulIOp(tb, _ic(NUM_QG)).result  # natural [M, K/32] e8m0 bytes
        x_rsrc = buffer_ops.create_buffer_resource(
            x_ptr, max_size=False, num_records_bytes=x_nb
        )
        fp4_rsrc = buffer_ops.create_buffer_resource(
            fp4_ptr, max_size=False, num_records_bytes=fp4_nb
        )
        sc_rsrc = buffer_ops.create_buffer_resource(
            sc_ptr, max_size=False, num_records_bytes=sc_nb
        )

        in_range = arith.cmpi(CmpIPredicate.slt, gid, total_blocks)
        blk_off = gid * arith.constant(COLS, type=i32)

        # wide vec8 loads of COLS contiguous bf16 -> f32 scalars
        vals = []
        for v in range_constexpr(NLOAD):
            off = blk_off + arith.constant(v * LVEC, type=i32)
            off = arith.select(in_range, off, arith.constant(0, type=i32))
            vec = buffer_ops.buffer_load(x_rsrc, off, vec_width=LVEC, dtype=T.bf16)
            vecw = Vector(vec, (LVEC,), BFloat16)
            for e in range_constexpr(LVEC):
                vals.append(vecw[e].to(Float32))

        # FWHT per RS sub-block (block-diagonal Hadamard)
        for n in range_constexpr(NG):
            vals[n * RS : (n + 1) * RS] = _fwht(vals[n * RS : (n + 1) * RS], RS)

        if const_expr(fold):
            vals = [x.to(BFloat16).to(Float32) for x in vals]
        else:
            sc = Float32(rs_scale)
            vals = [(x * sc).to(BFloat16).to(Float32) for x in vals]

        # per-group quant -> fp4 bytes + e8m0 scales
        fp4_bytes = []
        sc_bytes = []
        for g in range_constexpr(NUM_QG):
            grp = vals[g * QG : (g + 1) * QG]
            amax = _amax_raw(grp, tree=(AMAX == "tree"))
            e8_i8, hw = _e8m0_and_scale(amax, fold_k)
            sc_bytes.append(e8_i8)
            for p in range_constexpr(QG // 2):
                even = _to_raw(grp[2 * p])
                odd = _to_raw(grp[2 * p + 1])
                fp4_bytes.append(_pack_fp4(even, odd, hw))

        rng = _to_raw(in_range)
        words = [
            _pack4_i32(fp4_bytes[c * 4 : (c + 1) * 4]) for c in range_constexpr(NW)
        ]
        fp4_base = gid * arith.constant(NW, type=i32)
        for s in range_constexpr(NW // SVEC):
            off = fp4_base + arith.constant(s * SVEC, type=i32)
            if const_expr(SVEC == 1):
                buffer_ops.buffer_store(words[s], fp4_rsrc, off, mask=rng)
            else:
                vec = _vec_i32(words[s * SVEC : (s + 1) * SVEC])
                buffer_ops.buffer_store(vec, fp4_rsrc, off, mask=rng)

        if const_expr(SHUF):
            # Scatter each e8m0 byte to its canonical shuffle_scale offset:
            #   m = gid // NBCOL ; cb = gid % NBCOL ; n = cb*NUM_QG + q
            #   m0=m>>5, ma=(m>>4)&1, m2=m&15 ; k0=n>>3, ka=(n>>2)&1, k2=n&3
            #   dest = ((((m0*SN8 + k0)*4 + k2)*16 + m2)*2 + ka)*2 + ma
            gidr = _to_raw(gid)
            m_idx = A.DivUIOp(gidr, _ic(NBCOL)).result
            cb = A.RemUIOp(gidr, _ic(NBCOL)).result
            m0, ma, m2 = _bits(m_idx, 5), _bits(m_idx, 4, 1), _bits(m_idx, 0, 15)
            for q in range_constexpr(NUM_QG):
                n = A.AddIOp(A.MulIOp(cb, _ic(NUM_QG)).result, _ic(q)).result
                k0, ka, k2 = _bits(n, 3), _bits(n, 2, 1), _bits(n, 0, 3)
                off = _horner(m0, [(SN8, k0), (4, k2), (16, m2), (2, ka), (2, ma)])
                buffer_ops.buffer_store(sc_bytes[q], sc_rsrc, ArithValue(off), mask=rng)
        else:
            sc_base = gid * arith.constant(NUM_QG, type=i32)
            for q in range_constexpr(NUM_QG):
                off = sc_base + arith.constant(q, type=i32)
                buffer_ops.buffer_store(sc_bytes[q], sc_rsrc, off, mask=rng)

    @flyc.jit
    def launch(
        x_ptr: fx.Tensor,
        fp4_ptr: fx.Tensor,
        sc_ptr: fx.Tensor,
        total_blocks: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream = fx.Stream(  # noqa: B008  framework idiom: default is evaluated once at import on purpose
            None
        ),
    ):
        gx = arith.index_cast(T.index, grid_x)
        rq_kernel(x_ptr, fp4_ptr, sc_ptr, total_blocks).launch(
            grid=(gx, 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    return launch


@lru_cache(maxsize=32)
def compile_flydsl_rot_quant(
    *,
    K: int,
    RS: int,
    BLOCK: int,
    COLS: int,
    SVEC: int,
    AMAX: str,
    shuffle_scales: bool,
    WAVES: int | None = None,
    MAXNREG: int | None = None,
):
    """Compile (and cache) the launcher for one kernel config.

    Returns the ``@flyc.jit`` launcher; call it directly if you have already
    allocated the outputs and want to skip the torch-side work in
    :func:`flydsl_rot_quant`.
    """
    launcher = _build_rot_quant(K, RS, BLOCK, COLS, SVEC, AMAX, SHUF=shuffle_scales)
    hints = {}
    if WAVES:
        hints["waves_per_eu"] = WAVES
    if MAXNREG:
        hints["maxnreg"] = MAXNREG
    if hints:
        launcher.compile_hints = hints
    return launcher


def _pick_svec(COLS):
    """Widest i32 store vector (4 -> 128-bit) that divides the per-thread word count."""
    nw = (COLS // 2) // 4
    return next(s for s in (4, 2, 1) if nw % s == 0)


def _pick_block(M, K):
    """CTA size (threads/block) as a function of M, swept on MI350X gfx950.

    At prefill/mid batch (M>=1024) a 4-warp BLOCK=256 is a consistent 10-14% win
    over the single-warp BLOCK=64: larger CTAs keep more wide loads in flight and
    amortize launch/setup on this load-bound kernel. Decode / small prefill
    (M<=512) still prefers BLOCK=64 (more, smaller CTAs -> better memory-latency
    hiding). Large prefill (M>=4096) squeezes a further ~1-3% with an 8-warp
    BLOCK=512. These are gfx950 numbers; pass ``BLOCK=`` explicitly to override.
    """
    if M >= 4096:
        return 512
    if M >= 1024:
        return 256
    return 64


def flydsl_rot_quant(
    x,
    RS=64,
    BLOCK=None,
    COLS=None,
    SVEC=None,
    AMAX="tree",
    WAVES=None,
    MAXNREG=None,
    shuffle_scales=False,
):
    """Fused online Hadamard rotation + MXFP4 quantization.

    Args:
      x: bf16, **contiguous**, [M, K] (2D only -- reshape 3D activations first).
         Both are load-bearing: the kernel hardcodes bf16 buffer loads and computes
         element offsets linearly, so a fp16 or strided input would silently yield
         garbage. Both are checked below.
      RS: rotation size, one of {32, 64, 128}; K must be a multiple of it. The
         rotation is block-diagonal: K//RS independent RS-wide Hadamard blocks,
         each normalized by 1/sqrt(RS).
      shuffle_scales: emit the e8m0 scales already in the CK-gemm swizzle
         (``aiter.ops.shuffle.shuffle_scale`` layout) instead of the natural one.
         See the return contract below -- this changes the scale tensor's shape.
      BLOCK/COLS/SVEC/AMAX/WAVES/MAXNREG: tuning knobs; ``None`` picks the gfx950
         defaults (BLOCK adaptive per M -- see _pick_block; COLS=RS; SVEC=4 for
         128-bit stores; AMAX="tree", a balanced abs-max reduction with a shorter
         dep chain, ~2-3% over the chain form and bit-identical to it).
         WAVES/MAXNREG are compile hints (--amdgpu-waves-per-eu / --amdgpu-num-vgpr);
         measured to give no gain on gfx950 (the kernel already saturates HBM with
         many small CTAs) but kept for other shapes/hardware.

    Returns:
      (fp4, scales).
      fp4: uint8 [M, K//2], two fp4 (e2m1) values per byte, low nibble first.
      scales: uint8 e8m0.
        shuffle_scales=False -> [M, K//32], the natural per-32-element layout.
        shuffle_scales=True  -> **[ceil256(M), ceil8(K//32)]**, CK swizzle. The
          padding cells are *uninitialized* (torch.empty, same as
          ``shuffle_scale``): the CK gemm reads them but slices its output back to
          [:M], so they are don't-care. Do not compare, print or reuse them.

    Example:
        >>> xq, xs = flydsl_rot_quant(x, RS=64, shuffle_scales=True)
        >>> y = gemm_a4w4(xq, shuffle_weight(wq, layout=(16, 16)), xs,
        ...               shuffle_scale(ws), dtype=torch.bfloat16)[:M]
    """
    if RS not in SUPPORTED_RS:
        raise ValueError(f"RS must be one of {SUPPORTED_RS}, got {RS}")
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [M, K], got shape {tuple(x.shape)}")
    M, K = x.shape
    if K % RS:
        raise ValueError(f"K={K} must be a multiple of RS={RS}")
    # The kernel hardcodes bf16 loads and linear (contiguous) element offsets.
    # Without these two checks a wrong dtype/layout produces garbage silently --
    # no error, no warning, and the fp4 output still "looks" plausible.
    if x.dtype is not torch.bfloat16:
        raise ValueError(f"x must be bf16, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if AMAX not in ("tree", "chain"):
        raise ValueError(f"AMAX must be 'tree' or 'chain', got {AMAX!r}")
    arch = get_rocm_arch()
    if not str(arch).startswith("gfx950"):
        raise RuntimeError(
            "flydsl_rot_quant requires gfx950 (v_cvt_scalef32_pk_fp4_f32 MXFP4 "
            f"microscaling); got {arch}"
        )
    if BLOCK is None:
        BLOCK = _pick_block(M, K)
    if COLS is None:
        # COLS batching (>1 RS-block per thread) was measured to give no benefit on
        # gfx950: the kernel is load-bound and the vec8 loads already saturate the
        # burst, so COLS=RS with many small CTAs wins. _build_rot_quant still takes
        # COLS for other hardware.
        COLS = RS
    if SVEC is None:
        SVEC = _pick_svec(COLS)
    total_blocks = M * (K // COLS)
    grid_x = (total_blocks + BLOCK - 1) // BLOCK
    fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    if shuffle_scales:
        # Fused in-kernel CK-gemm scale swizzle: allocate the padded [sm, sn]
        # buffer the kernel scatters into. Padding cells are don't-care
        # (torch.empty, like shuffle_scale) -- the CK gemm slices output to [:M].
        n = K // QG
        sm = (M + 255) // 256 * 256
        sn = (n + 7) // 8 * 8
        sc = torch.empty((sm, sn), dtype=torch.uint8, device=x.device)
    else:
        sc = torch.empty((M, K // QG), dtype=torch.uint8, device=x.device)
    launcher = compile_flydsl_rot_quant(
        K=K,
        RS=RS,
        BLOCK=BLOCK,
        COLS=COLS,
        SVEC=SVEC,
        AMAX=AMAX,
        shuffle_scales=shuffle_scales,
        WAVES=WAVES,
        MAXNREG=MAXNREG,
    )
    _run_compiled(
        launcher, x, fp4, sc, total_blocks, grid_x, torch.cuda.current_stream()
    )
    return fp4, sc
