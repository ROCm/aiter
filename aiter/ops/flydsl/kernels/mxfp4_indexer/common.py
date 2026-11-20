"""Constants and FlyDSL helpers of the MXFP4 indexer kernels: buffer views, the scaled MFMA, a token's logit from its
accumulator, DPP / permlane lane exchange, the wave scan and the order-preserving fp32 keys.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..dpp_utils import permlane_swap_i32, update_dpp_i32
from ..tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor

H = 32  # index heads
D = 128  # index head dim
ENTRY_BYTES = D // 2 + D // 32  # e2m1 codes, then the e8m0 scales
BLOCK = 8  # tokens per candidate block
TOPK = 512
CANDIDATES = 2048  # candidate blocks of the two-level indexer
NBINS = 4096  # buckets of the select's histogram

F, I, I64, U = fx.Float32, fx.Int32, fx.Int64, fx.Uint32
NEG_INF = float("-inf")
INF = float("inf")
# A masked-off access goes to this element index: its byte offset (0x7FFFFFFC for 4-byte elements, 0x3FFFFFFE for
# 2-byte ones) is past every view's ``nbytes``, so the hardware range check drops the store / returns 0 for the load.
OFF_RANGE = 0x1FFFFFFF


def view(addr, nbytes, elem=None, unit=1):
    """Buffer tensor over the ``nbytes`` bytes at device address ``addr`` (descriptor in SGPRs, 32-bit offsets, range
    checked). ``unit`` > 1: ``unit``-element accesses at any element offset."""
    return ptr_buf_tensor(
        addr,
        elem=elem or I,
        unit_elems=unit,
        unit_stride=1 if unit > 1 else None,
        num_records_bytes=nbytes,
    )


def load(t, index, elem=None, unit=1, mask=None):
    """Element ``index`` of view ``t`` (``unit`` elements from it as a vector); 0 where ``mask`` is false."""
    if mask is not None:
        index = mask.select(index, I(OFF_RANGE))
    v = buf_copy_load(t, index, elem or I, unit)
    return Vec(v) if unit > 1 else (elem or I)(v)


def store(t, index, value, elem=None, mask=None):
    """Element ``index`` of view ``t`` = ``value``, where ``mask`` (if given) is true."""
    if mask is not None:
        index = mask.select(index, I(OFF_RANGE))
    buf_copy_store(t, index, value, elem or I)


def sload(t, index):
    """Element ``index`` of the i32 view ``t`` by a scalar load (``index`` wave-uniform): the value lands in an SGPR."""
    return fx.rocdl.s_buffer_load(fx.get_iter(t), index)


def wave_of(tx):
    """The wave of thread ``tx`` in its workgroup, wave-uniform. Raw ``rocdl.readfirstlane``: FlyDSL 0.5.0 has no typed
    broadcast, and without ``v_readfirstlane_b32`` the compiler keeps ``tx >> 6`` in a VGPR, so every LDS and buffer
    offset derived from it is per-lane VALU work."""
    return I(fx.rocdl.readfirstlane(T.i32, tx >> 6))


def bi(x):
    return x.bitcast(I)


def bf(x):
    return x.bitcast(F)


def ushr(a, n):
    return (a.bitcast(U) >> U(n)).bitcast(I)


def dpp(src, ctrl, old=None, bank_mask=0xF, row_mask=0xF, bound=True):
    """``v_mov_b32_dpp`` of an i32 (``dpp_utils.update_dpp_i32``; FlyDSL 0.5.0 has no typed DPP operation, and
    ``shuffle_xor`` lowers to ``ds_bpermute`` through LDS). ``old`` defaults to an undefined value, so the result does
    not need a copy of ``src`` first (raw ``llvm.mlir_undef``: there is no fx undef)."""
    o = _llvm.mlir_undef(T.i32) if old is None else old
    return I(update_dpp_i32(o, src, ctrl, row_mask, bank_mask, bound))


def xor_lane(v, m, lane):
    """The i32 ``v`` of lane ``lane ^ m`` (m a power of two below 64)."""
    if m == 1:
        return dpp(v, 0xB1)
    if m == 2:
        return dpp(v, 0x4E)
    if m == 4:
        return dpp(v, 0x104, old=dpp(v, 0x114, bank_mask=0xA), bank_mask=0x5)
    if m == 8:
        return dpp(v, 0x128)
    op = fx.rocdl.permlane16_swap if m == 16 else fx.rocdl.permlane32_swap
    a, b = permlane_swap_i32(op, v, v)
    return ((lane & m) != 0).select(a, b)


def wave_scan_incl(v):
    """Inclusive prefix sum of an i32 across the 64 lanes."""
    for ctrl in (0x111, 0x112, 0x114, 0x118):
        v = v + dpp(v, ctrl)
    v = v + dpp(v, 0x142, old=I(0), row_mask=0xA)
    return v + dpp(v, 0x143, old=I(0), row_mask=0xC)


def maxf(a, b):
    """``v_max_f32`` (``nnan``: no NaN canonicalisation around it; the operands are never NaN)."""
    return F(fx.maxnumf(a, b, fastmath=fx.arith.FastMathFlags.nnan))


def minf(a, b):
    return F(fx.minnumf(a, b, fastmath=fx.arith.FastMathFlags.nnan))


def not_nan(x):
    """False for a NaN fp32 (a traced IR compare, not a Python one)."""
    return x == x  # noqa: PLR0124


def key_of(v):
    """Order-preserving u32 key of an fp32 (as i32 bits); 0 for NaN and -inf, which no real logit maps to. A key of 0 is
    never selected, so a NaN logit (a key whose e8m0 scale is 0xFF) is never among the top-512, where a top-k over
    ``paged_mxfp4_mqa_logits``' output would rank it first."""
    b = v.bitcast(I)
    k = (b < I(0)).select(~b, b ^ I(-2147483648))
    return (not_nan(v) & (v > F(NEG_INF))).select(k, I(0))


def gptr(addr, align=16, ty=None):
    """Global pointer at a per-lane device address, for ``fx.generic_load``. The keys a select or scorer rescores are
    gathered: each lane's page differs and the page pool can span more than 4 GiB, so neither a buffer descriptor (a
    wave-uniform base in SGPRs, 32-bit offsets) nor a layout view of one tensor can address them; a 64-bit global load
    is the only form."""
    ptr_ty = fx.PointerType.get(
        ty or T.i32, address_space=fx.AddressSpace.Global, alignment=align
    )
    return fx.inttoptr(ptr_ty, I64(addr))


def mfma_fp4(a4, b4, acc, scale_a, scale_b, opsel_b=0):
    """``v_mfma_scale_f32_32x32x64_f8f6f4`` on e2m1 x e2m1: lane (r, h) of ``a4`` / ``b4`` holds 16 bytes (32 codes) of
    row / column r, K elements 32 h .. 32 h + 31 of the 64 of the step; ``scale_a`` is that lane's e8m0 scale,
    ``scale_b`` a dword whose byte ``opsel_b`` is the column's. Lane (c, h) of the result holds rows
    8 (i // 4) + 4 h + i % 4, column c."""
    fp4 = 4  # matrix format fp4_e2m1, for A (cbsz) and B (blgp)

    def pad(v):
        v, z = Vec(v), I(0)
        return Vec.from_elements([v[0], v[1], v[2], v[3], z, z, z, z], I)

    # The rocdl op wrapper: FlyDSL 0.5.0's MMA atoms (fx.rocdl.MFMA) take no per-operand e8m0 scales.
    return Vec(
        fx.rocdl.mfma_scale_f32_32x32x64_f8f6f4(
            Vec.make_type(16, F),
            [pad(a4), pad(b4), acc, fp4, fp4, 0, I(scale_a), opsel_b, I(scale_b)],
        )
    )


def token_logit(acc, w):
    """sum_h w_h relu(acc_h) of a lane's 16 head rows (four fma chains, summed in a fixed order) plus the other half
    wave's (lane ^ 32): every lane ends with its token's logit. The same arithmetic in every kernel, so a token scores
    to the same bits wherever it is computed."""
    s = [None] * 4
    for i in range_constexpr(16):
        r = maxf(acc[i], F(0.0))
        s[i % 4] = w[i] * r if i < 4 else fx.math.fma(w[i], r, s[i % 4])
    p = (s[0] + s[1]) + (s[2] + s[3])
    return _add_halves(p)


def _add_halves(p):
    a, b = permlane_swap_i32(fx.rocdl.permlane32_swap, bi(p), bi(p))
    return bf(a) + bf(b)


def pair_logit(acc_a, acc_b, w):
    """The logits of two tiles' tokens: lanes 0-31 of the result the tile A's (as ``token_logit``, same bits), lanes
    32-63 the tile B's, with one lane swap for both."""

    def partial(acc):
        s = [None] * 4
        for i in range_constexpr(16):
            r = maxf(acc[i], F(0.0))
            s[i % 4] = w[i] * r if i < 4 else fx.math.fma(w[i], r, s[i % 4])
        return (s[0] + s[1]) + (s[2] + s[3])

    pa, pb = partial(acc_a), partial(acc_b)
    a, b = permlane_swap_i32(fx.rocdl.permlane32_swap, bi(pa), bi(pb))
    return bf(a) + bf(b)
