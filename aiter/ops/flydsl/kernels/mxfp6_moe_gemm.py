# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL grouped GEMM for the MXFP6 MoE on gfx950: MXFP6 weights (W6, E2M3 / E3M2) x
MXFP6 or MXFP8 activations (A6 / A8), E8M0 scale per 32 along K,
V_MFMA_SCALE_F32_16X16X128_F8F6F4.

    C[m, n] = sum_k A[row(m), k] * B[e(tile(m)), n, k]

Storage formats are produced by aiter.ops.flydsl.mxfp6_moe_quant (packed 6-bit weights
preshuffled into MFMA fragments; activation rows in MFMA K order; scales packed 4
K-steps per dword) and by the row quantizer of mxfp6_moe_aux. The op runs two of these
GEMMs: stage 1, epilogue "swiglu_q" (gathered x rows @ W1^T, SwiGLU, h MX-quantized
into stage 2's A layout; B N-fragments gate/up interleaved), and stage 2, epilogue
"store" (h @ W2^T to bf16 rows). Epilogue "swiglu" stores the bf16 SwiGLU result
unquantized.

One block of compile_mxfp6_moe_gemm's kernel computes one BM x BN tile on 4 waves
(2 x 2; wave tile 128x128 / 128x64 / 64x128 / 64x64, FM x FN 16x16 fragments), by these
functions:
  tile_order   block -> (m-tile, n-tile) in GM-grouped, XCD-contiguous order; the
               tile's expert
  prologue     K steps 0 .. NSTAGE-2 and quad 0's scales: dma_b, dma_scale_b, then
               dma_a, dma_scale_a
  step         per K step (the K loop): k_barrier / early_barrier; later steps' DMAs
               (operand_dmas, scale_dmas) issued between the MFMAs; LDS reads a_frag,
               b_frag, load_scales; mma in mfma_group and last_group
  epilogue     stage_c or stage_h (SwiGLU) into LDS, warm_next_group (stage 2),
               store_c or quant_store_h (mx6_quant_store_hw, mx6_quant_store_exact)
LDS holds an NSTAGE-deep ring of K-step stages (A [BM rows][16*abits B], XOR-swizzled
pieces so the operand reads are bank-conflict-free, then B [BN/16 fragments][1536 B])
filled by async buffer->LDS DMA in 16-B pieces, two scale slots (quads alternate), and
at the end the staged output tile over them. Step s+NSTAGE-1's DMAs are issued while
step s computes; a step waits only for its own stage. Per-expert and per-tile bases are
64-bit (rebased buffer descriptors): the weights and outputs may exceed 4 GiB, the A
rows and one expert's weights may not (the launch wrapper in
aiter.ops.flydsl.mxfp6_moe_kernels refuses). The tile, ring and order of each call are
picked on the host by aiter.ops.flydsl.mxfp6_moe_policy.

Measured operand contract of the scaled MFMA (lane l: row = l % 16, g = l // 16):
  FP6 / FP4: lane holds K[32g, 32g+32) contiguous (element j at bits [6j, 6j+6) /
             [4j, 4j+4), little-endian dwords).
  FP8: two K=64 halves: bytes 0-15 = K[16g, 16g+16), bytes 16-31 =
       K[64+16g, 64+16g+16).
  E8M0 scale of (row r, K-block b of 32) in lane 16b + r; acc[i] of lane l =
  C[4*(l//16) + i, l % 16].
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from .act import sigmoid_batch
from .mxfp4_gemm_common import _lds_ptr3
from .mxfp6_moe_aux import QFMT, _i64
from .tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor

FMT_ELEM = {"e4m3": fx.Float8E4M3FN, "e2m3": fx.Float6E2M3FN, "e3m2": fx.Float6E3M2FN}
BITS = {"e4m3": 8, "e2m3": 6, "e3m2": 6}


def _v(x):  # i32 MLIR value, the raw DMA's operand form
    return fx.Int32(x).ir_value()


DMA_SLOTS = 4  # DMA issue slots per MFMA group, each right after the slot's first MFMA
# 2-stage ring: step s+1's first operand DMAs that go in step s's last MFMA group, after
# its barrier
N_EARLY_DMA = 8


def _abs_max_bits(words):
    """Bit pattern (i32) of the largest |bf16| of a block held as 32 bf16 in 16 dwords,
    by a packed u16 max (v_pk_max_u16, two elements per op)."""
    acc = None
    for w in words:
        a = fx.Vector.from_elements([w & fx.Int32(0x7FFF7FFF)], fx.Int32).bitcast(
            fx.Uint16
        )
        acc = a if acc is None else fx.max(acc, a)
    m = fx.Int32(acc.bitcast(fx.Int32)[0])
    return fx.max(m & fx.Int32(0xFFFF), fx.Int32(fx.Uint32(m) >> fx.Uint32(16)))


def _store_scale_byte(s_base, s_row_off, kb, sbyte):
    """E8M0 scale byte of K-block kb into the [R][KQ][4 b][4 j] h-scale layout (K step s
    = kb // 4 = 4 * quad + j)."""
    s = kb // fx.Int32(4)
    b = kb % fx.Int32(4)
    boff = (
        s_row_off
        + ((s // fx.Int32(4)) * fx.Int32(4) + b) * fx.Int32(4)
        + s % fx.Int32(4)
    )
    buf_copy_store(ptr_buf_tensor(s_base, fx.Int8), boff, fx.Int8(sbyte), fx.Int8, 1)


def mx6_nonfinite(words):
    """The block holds inf or NaN (|bf16| bit pattern >= 0x7F80): mx6_quant_store_hw's
    own maximum, shared by CSE."""
    return _abs_max_bits(words) >= fx.Int32(0x7F80)


def _cvt_fp8(words, scale):
    """32 bf16 (16 dwords) / scale -> 32 fp8 E4M3 codes (8 dwords):
    v_cvt_scalef32_pk_fp8_bf16, 2 bf16 into the lo / hi half of a dword per op; word wi
    = codes 4wi..4wi+3."""
    # Raw rocdl.cvt_scalef32_pk_fp8_bf16 (the upstream op, outside flydsl.expr.rocdl's
    # stable list): fx has no scaled convert. The stable form is mx_quant_store's
    # per-element arithmetic: 609 instructions per 32-element block against 138 with the
    # convert (ISA of the 256x256 swiglu_q kernel)
    out = []
    for wi in range_constexpr(8):
        t = fx.as_ir_value(
            fx.Vector.from_elements([fx.Int32(0)], fx.Int32).bitcast(fx.Int16)
        )
        for h in range_constexpr(2):
            src2 = fx.Vector.from_elements([words[2 * wi + h]], fx.Int32).bitcast(
                fx.BFloat16
            )
            t = rocdl.cvt_scalef32_pk_fp8_bf16(
                T.vec(2, T.i16), t, fx.as_ir_value(src2), fx.as_ir_value(scale), h == 1
            )
        out.append(fx.Int32(fx.Vector(t, (2,), fx.Int16).bitcast(fx.Int32)[0]))
    return out


def _cvt_fp6(words, scale, fmt):
    """32 bf16 (16 dwords) / scale -> 32 packed 6-bit codes (6 dwords): one
    v_cvt_scalef32_pk32_{fp6,bf6}_bf16."""
    # Raw rocdl.cvt_scalef32_pk32_{fp6,bf6}_bf16, as in _cvt_fp8: fx has no scaled
    # convert; mx_quant_store's arithmetic is 591 instructions per block against 90-92
    # (ISA)
    src = fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)
    cvt = (
        rocdl.cvt_scalef32_pk32_fp6_bf16
        if fmt == "e2m3"
        else rocdl.cvt_scalef32_pk32_bf6_bf16
    )
    return fx.Vector(
        cvt(T.vec(6, T.i32), fx.as_ir_value(src), fx.as_ir_value(scale)), (6,), fx.Int32
    )


def mx6_quant_store_hw(
    words, fmt, y_base, y_row_off, kb, s_base, s_row_off, y_dw, y_dw_t
):
    """mx_quant_store (mxfp6_moe_aux) of one 32-element block of h held as 32 bf16 in 16
    dwords (element 2k in the low half of word k): the same scale byte (2^ceil(log2(amax
    / vmax)), clamped), codes by the gfx950 hardware convert (divides by 2^e, RNE,
    saturating) instead of per-element bit arithmetic. 8-bit codes go to the row at byte
    y_row_off, 6-bit ones to the dword offsets y_dw (the block's first 16 B) and y_dw_t
    (its last 8 B)."""
    vmax = QFMT[fmt][0]
    amax = (_abs_max_bits(words) << fx.Int32(16)).bitcast(fx.Float32)
    # multiply by the rounded reciprocal instead of an IEEE divide (~11 VALU per block):
    # bit-identical scale byte here, since amax is bf16 (8 significant bits) and vmax
    # has 3-4: amax / vmax is a power of two only at amax = vmax * 2^j, where vmax *
    # fl(1/vmax) = 1 + (< 2^-24) rounds to 1, and otherwise lies >= 2^-9 away from one
    tb = (amax * fx.Float32(1.0 / vmax)).bitcast(fx.Int32)
    ex = (tb >> fx.Int32(23)) - fx.Int32(127)
    ex = ((tb & fx.Int32(0x7FFFFF)) != fx.Int32(0)).select(ex + fx.Int32(1), ex)
    ex = fx.max(ex, fx.Int32(-126))
    ex = fx.min(ex, fx.Int32(127))
    sbyte = (amax > fx.Float32(0.0)).select(ex + fx.Int32(127), fx.Int32(0))
    # 2^e: the cvt divides by it
    scale = ((ex + fx.Int32(127)) << fx.Int32(23)).bitcast(fx.Float32)
    if const_expr(fmt == "e4m3"):
        out = _cvt_fp8(words, scale)
        s = kb // fx.Int32(4)
        b = kb % fx.Int32(4)
        p0 = (
            y_row_off
            + s * fx.Int32(128)
            + (b % fx.Int32(2)) * fx.Int32(64)
            + (b // fx.Int32(2)) * fx.Int32(16)
        )
        y4 = ptr_buf_tensor(y_base, fx.Int32, unit_elems=4, unit_stride=1)
        buf_copy_store(
            y4,
            p0 // fx.Int32(4),
            fx.Vector.from_elements(out[:4], fx.Int32),
            fx.Int32,
            4,
        )
        buf_copy_store(
            y4,
            (p0 + fx.Int32(32)) // fx.Int32(4),
            fx.Vector.from_elements(out[4:], fx.Int32),
            fx.Int32,
            4,
        )
        _store_scale_byte(s_base, s_row_off, kb, sbyte)
        return
    packed = _cvt_fp6(words, scale, fmt)
    y4 = ptr_buf_tensor(y_base, fx.Int32, unit_elems=4, unit_stride=1)
    y2 = ptr_buf_tensor(y_base, fx.Int32, unit_elems=2, unit_stride=1)
    buf_copy_store(y4, y_dw, packed.shuffle(packed, [0, 1, 2, 3]), fx.Int32, 4)
    buf_copy_store(y2, y_dw_t, packed.shuffle(packed, [4, 5]), fx.Int32, 2)
    _store_scale_byte(s_base, s_row_off, kb, sbyte)


def mx6_quant_store_exact(
    words, fmt, y_base, y_row_off, kb, s_base, s_row_off, y_dw, y_dw_t
):
    """mx6_quant_store_hw's block with mxfp6_moe_aux.mx_quant_store's own arithmetic,
    for blocks holding inf or NaN: there its 2^-e is 0 (e clamps at 127), so finite
    elements get code 0 and inf / NaN the max code, at scale byte 254 (inf) / 0 (NaN);
    the hardware convert keeps inf / NaN (fp8 NaN codes) instead. mx_quant_store itself
    cannot be called: it has no store address for the tile-step-major 6-bit h."""
    vmax, mbits, minexp, cmax, bits = QFMT[fmt]
    vals = []
    for w in words:
        vals += [
            (w << fx.Int32(16)).bitcast(fx.Float32),
            (w & fx.Int32(-65536)).bitcast(fx.Float32),
        ]
    amax = fx.Float32(0.0)
    for v in vals:
        amax = fx.max(amax, fx.absf(v))
    tb = (amax / fx.Float32(vmax)).bitcast(fx.Int32)
    ex = (tb >> fx.Int32(23)) - fx.Int32(127)
    ex = ((tb & fx.Int32(0x7FFFFF)) != fx.Int32(0)).select(ex + fx.Int32(1), ex)
    ex = fx.max(ex, fx.Int32(-126))
    ex = fx.min(ex, fx.Int32(127))
    sbyte = (amax > fx.Float32(0.0)).select(ex + fx.Int32(127), fx.Int32(0))
    inv = ((fx.Int32(127) - ex) << fx.Int32(23)).bitcast(fx.Float32)  # 2^-e
    codes = []
    for v in vals:
        u = fx.absf(v) * inv
        ue = (u.bitcast(fx.Int32) >> fx.Int32(23)) - fx.Int32(127)
        ee = fx.max(ue, fx.Int32(minexp))
        inv_step = ((fx.Int32(127 + mbits) - ee) << fx.Int32(23)).bitcast(fx.Float32)
        qv = fx.Int32(fx.roundeven(u * inv_step))
        code = fx.min(
            (ee - fx.Int32(minexp)) * fx.Int32(1 << mbits) + qv, fx.Int32(cmax)
        )
        codes.append(
            code | (v < fx.Float32(0.0)).select(fx.Int32(1 << (bits - 1)), fx.Int32(0))
        )
    s = kb // fx.Int32(4)
    b = kb % fx.Int32(4)
    y4 = ptr_buf_tensor(y_base, fx.Int32, unit_elems=4, unit_stride=1)
    if const_expr(bits == 8):
        out = []
        for wi in range_constexpr(8):
            acc = fx.Int32(0)
            for bi in range_constexpr(4):
                acc = acc | ((codes[4 * wi + bi] & fx.Int32(0xFF)) << fx.Int32(8 * bi))
            out.append(acc)
        p0 = (
            y_row_off
            + s * fx.Int32(128)
            + (b % fx.Int32(2)) * fx.Int32(64)
            + (b // fx.Int32(2)) * fx.Int32(16)
        )
        buf_copy_store(
            y4,
            p0 // fx.Int32(4),
            fx.Vector.from_elements(out[:4], fx.Int32),
            fx.Int32,
            4,
        )
        buf_copy_store(
            y4,
            (p0 + fx.Int32(32)) // fx.Int32(4),
            fx.Vector.from_elements(out[4:], fx.Int32),
            fx.Int32,
            4,
        )
    else:
        out = []
        for wi in range_constexpr(6):
            acc = fx.Int32(0)
            for j in range_constexpr(32):
                lo = 6 * j
                if lo // 32 == wi:
                    acc = acc | (codes[j] << fx.Int32(lo % 32))
                elif (lo + 5) // 32 == wi:
                    acc = acc | fx.Int32(fx.Uint32(codes[j]) >> fx.Uint32(32 - lo % 32))
            out.append(acc)
        y2 = ptr_buf_tensor(y_base, fx.Int32, unit_elems=2, unit_stride=1)
        buf_copy_store(
            y4, y_dw, fx.Vector.from_elements(out[:4], fx.Int32), fx.Int32, 4
        )
        buf_copy_store(
            y2, y_dw_t, fx.Vector.from_elements(out[4:], fx.Int32), fx.Int32, 2
        )
    _store_scale_byte(s_base, s_row_off, kb, sbyte)


def _rsrc(addr_i64):
    """Buffer descriptor (V#) of a raw 64-bit address, for the raw async buffer->LDS DMA
    below."""
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(ptr_buf_tensor(addr_i64)))


def _at(layout, *crd):
    """Offset of coordinate crd in a layout (fx.crd2idx): an i32, or a Python int when
    crd is all static."""
    return fx.get_scalar(fx.crd2idx(crd, layout))


def _c_order(*shape):
    """Row-major layout of [shape[0]][shape[1]]...: the storage notation the comments
    use."""
    return fx.make_ordered_layout(shape, tuple(range(len(shape) - 1, -1, -1)))


def _split(idx, *shape):
    """Coordinates of index idx in the row-major [shape[0]][shape[1]]... (fx.idx2crd),
    outermost first. idx2crd takes each coordinate modulo its extent: idx must lie
    inside the layout."""
    return fx.idx2crd(
        idx, fx.make_ordered_layout(shape[::-1], tuple(range(len(shape))))
    ).unpack()[::-1]


# sched_barrier mask that lets every instruction cross: it only stops LLVM pairing two
# LDS reads
_SEP = 0xFFF


def _pin_agpr(v, k):
    """Accumulator k pinned to a[4k:4k+3]. Raw rocdl.experimental.sched_agpr (unstable:
    FlyDSL's experimental path, on a FlyDSL-local LLVM patch): MFMA_Scale / fx.gemm have
    no accumulator register class. With the launcher's AGPR-form MFMA selection
    (_agpr_form_mfma) it keeps the 256 loop-carried accumulators of an f8x8 tile in
    place: 0 v_accvgpr_* per K step against 109-130 and 12-100 B of scratch without the
    pin (ISA); the stable API alone ran the op at 0.92x (A6W6) / 0.72x (A8W6). Those
    kernels use 252-253 arch VGPRs: no headroom left.
    A FlyDSL without the pin or the option raises (_agpr_form_mfma, and the launch
    wrapper in mxfp6_moe_kernels). A FlyDSL that has both but gives them no effect
    compiles the slow form silently: FlyDSL 0.5.0 lets a kernel read neither its
    compiled register counts nor its scratch size, so only test_agpr_form in
    op_tests/test_flydsl_mxfp6_moe.py (compile only, no GPU) catches that."""
    return rocdl.experimental.sched_agpr(fx.as_ir_value(v), 4 * k)


# The error of a FlyDSL that cannot build the AGPR form; {} takes what is missing.
# Raised by _agpr_form_mfma, and by the launch wrapper where the LLVM rejects the option
AGPR_FORM_ERROR = (
    "the MXFP6 MoE GEMM's f8x8 tiles keep their 256 accumulators in AGPRs through "
    "rocdl.experimental.sched_agpr and the LLVM option amdgpu-mfma-vgpr-form=false, "
    "which FlyDSL 0.5.0 (with its local LLVM patch) provides; {}. Without them those "
    "tiles compile to a slow form (stable API only: 0.92x on A6W6, 0.72x on A8W6), so "
    "the kernel refuses; test_agpr_form in op_tests/test_flydsl_mxfp6_moe.py checks "
    "the compiled form without a GPU"
)


def _agpr_form_mfma(launch):
    """LLVM option on the launcher: the AGPR form of the MFMAs (dst and srcC in AGPRs).
    Needed with _pin_agpr: without it the pinned results are copied VGPR <-> AGPR,
    169-184 copies per K step and 656-788 B of scratch (ISA). FlyDSL 0.5.0 has no
    per-atom accumulator register class. Raises where the pin is missing; the launch
    wrapper does where the LLVM rejects the option."""
    if not hasattr(getattr(rocdl, "experimental", None), "sched_agpr"):
        raise RuntimeError(
            AGPR_FORM_ERROR.format("this FlyDSL has no rocdl.experimental.sched_agpr")
        )
    launch.compile_hints["llvm_options"] = {"amdgpu-mfma-vgpr-form": False}


def _nt_aux():
    """aux = 2 (non-temporal) for the raw async buffer->LDS DMA. Raw flydsl._mlir.ir
    attribute: the op takes its aux only so (an int raises TypeError), and the stable
    copy atoms hard-code aux 0. Plain weight loads lose 3-7% where the layer does not
    fit the MALL (mxfp6_moe_policy.nt_weights)."""
    return {"aux": _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), 2)}


# Raw rocdl.raw_ptr_buffer_load_async_lds: the stable copy atoms
# (cdna4.BufferLoadAsyncLDS128b / 32b through fx.copy) take no cache bits, and with them
# the K loop grew from ~2.8k to ~5.1k instructions with 12 spilled VGPRs: 8-12% slower
# (ISA of the K loop and op timing)
_dma_to_lds = rocdl.raw_ptr_buffer_load_async_lds


def _lds_load_b128(lds_i32, q):
    """One ds_read_b128 at i64 offset q. Raw llvm.LoadOp with alignment 16: fx.ptr_load
    states no alignment (a 16-B aligned view does not carry through fx.add_offset), LLVM
    then emits a ds_read2_b64, and the op loses up to 1.2% (A/B timing on 10 shapes: 4
    slower)."""
    return fx.Vector(
        _llvm.LoadOp(
            T.vec(2, T.i64), _lds_ptr3(lds_i32, q * fx.Int32(8)), alignment=16
        ).result
    )


def compile_mxfp6_moe_gemm(
    N: int,
    K: int,
    a_fmt: str,
    b_fmt: str,
    epi: str,
    gather: bool,
    FM: int = 8,
    FN: int = 8,
    WGM: int = 2,
    WGN: int = 2,
    NSTAGE: int = 3,
    GM: int = 1,
    NT_WEIGHTS: int = 0,
    H_TILE_STEP: int = 0,
    PAD_A: int = 0,
    WARM: int = 0,
    BN_SLAB: int = 0,
    HALF_TILES: int = 0,
):
    """One grouped-GEMM kernel and its launcher. Block tile BM x BN = 16*FM*WGM x
    16*FN*WGN on WGM x WGN waves, an NSTAGE-deep LDS ring, GM m-tiles per tile group.
    The host policy (mxfp6_moe_policy) picks the rest: NT_WEIGHTS non-temporal weight
    loads; H_TILE_STEP the 6-bit h tile-step-major (written by "swiglu_q", read by a
    gather=False "store"); PAD_A gathered 6-bit A rows padded to 128 B per K step (by
    compile_mxfp6_moe_pad_rows' kernel); WARM > 0: the store epilogue warms the first
    1 / WARM of the next tile group's h rows in L2; BN_SLAB the weight slab width
    (default BN); HALF_TILES 128-row tiles over 256-row routing tiles.

    Not cached here: mxfp6_moe_kernels keeps one launcher per argument set and
    device."""
    BM_, BN_ = 16 * FM * WGM, 16 * FN * WGN
    # B is preshuffled N-block-major [E][N/BN_SLAB][KS][BN_SLAB/16][1536] (BN_SLAB: the
    # prepared slab width): a BN-wide tile reads BN/BN_SLAB whole slabs, each one
    # contiguous run of SLB bytes per K step
    BN_SLAB = BN_SLAB or BN_
    abits = BITS[a_fmt]
    # SwiGLU epilogues: N/2 output columns (B N-fragments gate / up interleaved)
    SWI = epi in ("swiglu", "swiglu_q")
    # this kernel reads its A tile-step-major
    A_TILE_STEP = bool(H_TILE_STEP) and epi == "store"
    if (
        epi not in ("store", "swiglu", "swiglu_q")
        or GM < 1
        or NSTAGE not in (2, 3)
        or N % BN_
        or K % 128
        or BN_ % BN_SLAB
        or (SWI and FN % 2)
    ):
        raise ValueError(
            f"mxfp6_moe_gemm: epi {epi!r} must be 'store', 'swiglu' or 'swiglu_q', "
            f"GM {GM} >= 1, NSTAGE {NSTAGE} 2 or 3, N {N} a multiple of the "
            f"{BN_}-column tile, K {K} of 128, the tile of the {BN_SLAB}-column weight "
            "slabs"
        )
    if (
        (H_TILE_STEP and (abits != 6 or epi == "swiglu" or (epi == "store" and gather)))
        or (PAD_A and (abits != 6 or not gather))
        or (WARM and not A_TILE_STEP)
    ):
        raise ValueError(
            "mxfp6_moe_gemm: H_TILE_STEP is for 6-bit h (written by 'swiglu_q', read "
            "by a gather=False 'store'), PAD_A for gathered 6-bit rows, WARM for a "
            "store kernel that reads a tile-step-major A"
        )
    NW = WGM * WGN
    NT = 64 * NW
    KS = K // 128
    KQ = (KS + 3) // 4  # scale quads (4 K steps share one scale dword)
    KQF, KT = KS // 4, KS % 4

    # ---- A: row layout, padding, step stride ----
    RB = 16 * abits  # A bytes per row per K step
    PPR = RB // 16  # 16-B pieces per row-step
    a_row_bytes = K * abits // 8
    # Tile-step-major 6-bit h, [tile][K step][BM rows][96 B]: the swiglu_q epilogue
    # writes each n-tile's codes as one contiguous block, and stage 2's A DMA of a step
    # reads BM*96 contiguous bytes (8 whole lines per 1-KB DMA instead of ~16 straddled
    # ones for 96-B row pieces at the row pitch).
    # Padded rows: gathered 6-bit A rows re-laid by the launch wrapper to 128 B per K
    # step ([T][KS][128], 96 B used), so each row-step is one aligned line (the 96-B
    # pitch averages 1.5 lines per row-step; A DMAs stalled ~2x B's). Elsewhere 6-bit A
    # row-steps straddle lines between steps: operand_dmas alternates the row halves by
    # step parity
    ALT_HALVES = abits == 6 and not PAD_A and not A_TILE_STEP
    # global bytes between K steps of one A row
    ASTEP = BM_ * RB if A_TILE_STEP else (128 if PAD_A else RB)

    # ---- B: slabs ----
    b_frag_bytes = 1536
    b_expert_bytes = (N // 16) * KS * b_frag_bytes
    bs_expert_bytes = (N // 16) * KQ * 64 * 4
    BF = BN_ // 16
    BFS = BN_SLAB // 16
    # B bytes of one slab per K step (= the global B step stride)
    SLB = BFS * b_frag_bytes

    # ---- DMA plan: 1-KB pieces per step; a wave's pieces in groups of GS sharing one
    # M0 ----
    A_BYTES, B_BYTES = BM_ * RB, BF * b_frag_bytes
    IA, IB = A_BYTES // 1024, B_BYTES // 1024  # DMA instructions per step (block)
    IAW, IBW = IA // NW, IB // NW

    def _gs(n):
        ng = (n + 3) // 4
        return (n + ng - 1) // ng

    # a wave's operand DMAs t cover consecutive 1-KB LDS pieces; groups of GS share one
    # M0 and put step k = t % GS in the 12-bit immediate offset (added to the LDS and
    # the global address): the global voffset is pre-reduced by k KB and raised by
    # (GS-1) KB with the descriptor based (GS-1) KB lower (no negative voffset)
    GSA, GSB = _gs(IAW), _gs(IBW)
    CA, CB = (GSA - 1) * 1024, (GSB - 1) * 1024
    # 16-B A-scale DMA instructions per quad (one per 64 rows; waves >= NSA repeat wave
    # % NSA's)
    NSA = BM_ // 64
    # one quad of scales: A [rows][4 b] dwords, B [frags][64 lanes]
    SC_A, SC_B = BM_ * 16, BF * 256
    SBP = 16 if (SC_B // 1024) % NW == 0 else 4  # B-scale DMA piece bytes
    ISB_ = SC_B // (64 * SBP)  # B-scale DMA instrs (SBP-B pieces)
    ISBW = ISB_ // NW

    # ---- LDS: NSTAGE operand stages, two scale slots (quads alternate), the staged
    # output tile over them ----
    STAGE = A_BYTES + B_BYTES
    SLOT = SC_A + SC_B
    SC_BASE = NSTAGE * STAGE
    lds_bytes = SC_BASE + 2 * SLOT
    n_out = N // 2 if SWI else N
    BNO = BN_ // 2 if SWI else BN_  # output columns per block
    OSTR = BNO * 2 + 16  # padded LDS row stride (bytes) of the staged output tile
    if NSTAGE == 2:
        # a 2-stage ring is smaller than the staged C tile
        lds_bytes = max(lds_bytes, BM_ * OSTR)
    if (
        A_BYTES % 1024
        or B_BYTES % 1024
        or IA % NW
        or IB % NW
        or ISB_ % NW
        or BM_ * OSTR > lds_bytes
        or lds_bytes > 160 * 1024
    ):
        raise ValueError(
            f"mxfp6_moe_gemm: no {BM_}x{BN_} tile on {WGM}x{WGN} waves with a "
            f"{NSTAGE}-stage ring for {a_fmt} activations ({lds_bytes} B of LDS, limit "
            "160 KiB; whole 1-KB DMA pieces per wave)"
        )
    # swiglu_q: h is quantized in the epilogue to MX a_fmt (stage-2 "A" layout, see
    # aiter.ops.flydsl.mxfp6_moe_quant)
    q_row_bytes = n_out * abits // 8
    KQ2 = (n_out // 128 + 3) // 4

    # ---- stage-2 L2 warm of the next tile group's h rows (strip of NWI_ x NT dwords at
    # the top of LDS) ----
    if WARM:
        # blocks of a tile group; its last NWARM_ warm the next group
        GW_ = GM * (N // BN_)
        NWARM_ = min(32, GW_)
        # lines warmed per m-tile: its first 1 / WARM of the K steps
        LPM_ = (BM_ * a_row_bytes // 128) // WARM
        LINES_ = GM * LPM_
        LPB_ = (LINES_ + NWARM_ - 1) // NWARM_  # lines per warming block
        NWI_ = (LPB_ + NT - 1) // NT  # warm DMAs per lane
        if BM_ * OSTR > lds_bytes - NWI_ * NT * 4:
            WARM = 0  # no warm where its strip would overlap the staged tile

    # ---- stage-1 epilogue staging without the pre-staging barrier ----
    def staging_bases():
        """swiglu_q stages h in the ring stages the last K step does not read, with no
        barrier first. Invariant: every wave's DMAs into those stages must be complete;
        that needs KS >= NSTAGE - 1, so a K-loop barrier (after lgkmcnt(0)) follows each
        of them. At KS 1 the prologue's clamped DMA into stage 1 can land after another
        wave staged there. Returns the byte bases of staged rows [0, BM/2) and [BM/2,
        BM), or None: stage at 0 behind the epilogue's barrier."""
        half = BM_ // 2
        if not (
            epi == "swiglu_q"
            and NSTAGE == 3
            and half * OSTR <= STAGE
            and KS >= NSTAGE - 1
        ):
            return None
        free = [st for st in range(NSTAGE) if st != (KS - 1) % NSTAGE]
        if BM_ * OSTR <= STAGE or free[1] == free[0] + 1:
            return free[0] * STAGE, free[0] * STAGE
        return free[0] * STAGE, free[1] * STAGE - half * OSTR

    STAGING = staging_bases()

    # ---- LDS regions as byte ranges [lo, hi), and the overlaps that would be races.
    # Invariant checks: no caller input reaches them with the code above; they are here
    # to stop a later change of the regions from reintroducing one ----
    # operand stages: A part, then B part
    ring = [(k * STAGE, (k + 1) * STAGE) for k in range(NSTAGE)]
    # scale slots: A half, then B half
    slots = [(SC_BASE + q * SLOT, SC_BASE + (q + 1) * SLOT) for q in range(2)]
    # the staged output tile: at 0, or its two row halves at the barrier-less staging
    # bases
    half = BM_ // 2 * OSTR
    staged = (
        [(0, 2 * half)]
        if STAGING is None
        else [
            (STAGING[0], STAGING[0] + half),
            (STAGING[1] + half, STAGING[1] + 2 * half),
        ]
    )
    # the warm DMAs' destination, at the top
    strip = [(lds_bytes - NWI_ * NT * 4, lds_bytes)] if WARM else []

    def overlap(xs, ys):
        return any(x[0] < y[1] and y[0] < x[1] for x in xs for y in ys)

    if max(hi for _, hi in ring + slots + staged) > lds_bytes:
        raise ValueError(
            f"mxfp6_moe_gemm: an LDS region passes the {lds_bytes} B allocated"
        )
    # barrier-less staging: other waves still read the last K step's stage and the scale
    # slots, and their DMAs into the staged stages must have landed (staging_bases' KS
    # >= NSTAGE - 1)
    if STAGING is not None and overlap(staged, [ring[(KS - 1) % NSTAGE]] + slots):
        raise ValueError(
            f"mxfp6_moe_gemm: barrier-less staging at {STAGING} overlaps the last K "
            "step's stage or a scale slot"
        )
    # the warm strip is the top of LDS (scale slot 1; both slots and the end of the ring
    # for a long K over few n-tiles), read by lagging waves until the epilogue barrier:
    # the warm DMAs come after it, and the staged tile must stay clear
    if strip and (STAGING is not None or overlap(strip, staged)):
        raise ValueError(
            "mxfp6_moe_gemm: the warm strip needs the pre-staging barrier and must not "
            "overlap the staged tile"
        )

    # a step's last MFMA group also reads the next quad's scales, only with >= 3 full
    # quads: at K = 1024 (two quads) the 256x256 store kernel spilled
    SCALE_AHEAD = KQF >= 3
    # f8x8 tiles: the 256 accumulators live in AGPRs (_pin_agpr)
    ACC_AGPR = FM * FN >= 64
    name = (
        f"mxfp6_moe_gemm_{a_fmt}_{b_fmt}_{epi}_g{int(gather)}_n{N}_k{K}"
        f"_f{FM}x{FN}_w{WGM}x{WGN}_s{NSTAGE}_gm{GM}"
        + ("_ntb" if NT_WEIGHTS else "")
        + (f"_warm{WARM}" if WARM else "")
        + ("_hsm" if H_TILE_STEP else "")
        + ("_xp" if PAD_A else "")
        + ("_aal" if ALT_HALVES else "")
        + (f"_bs{BN_SLAB}" if BN_SLAB != BN_ else "")
        + ("_ht" if HALF_TILES else "")
    )
    GX = N // BN_
    # the expert weight stream is read once per call: non-temporal B and B-scale DMAs
    # (mxfp6_moe_policy.nt_weights)
    nt_aux = _nt_aux if NT_WEIGHTS else dict

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def gemm(
        A: fx.Pointer,
        AS: fx.Pointer,
        RIDX: fx.Pointer,
        TEXP: fx.Pointer,
        B: fx.Pointer,
        BS: fx.Pointer,
        C: fx.Pointer,
        CS: fx.Pointer,
        GY: fx.Int32,
    ):
        lds_raw = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        lds_i32 = fx.Int32(fx.ptrtoint(lds_raw))
        lds64 = fx.recast_iter(fx.Int64, lds_raw)
        tid = fx.thread_idx.x
        i32pt = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )

        def tile_order():
            """This block's (m-tile, n-tile) and expert (-1: past the used tiles, exit).
            XCD-contiguous remap (dispatch round-robins blocks over 8 XCDs), then
            grouped order: GM m-tiles x all n-tiles, m fastest, so co-resident blocks
            share A and B in L2. TEXP[GY] = number of used m-tiles; the remap runs over
            used tiles only, and the first-dispatched blocks are the used ones, so the 8
            XCDs stay balanced."""
            nb = fx.block_idx.x
            mb = fx.block_idx.y
            # grid rows: two half tiles per routing tile
            GYT = GY * fx.Int32(2) if HALF_TILES else GY
            nvalid = fx.inttoptr(i32pt, _i64(TEXP))[GY]
            if const_expr(HALF_TILES):
                nvalid = nvalid * fx.Int32(2)  # two half tiles per routing tile
            total = nvalid * fx.Int32(GX)
            L = mb * fx.Int32(GX) + nb
            L = (L < total).select(L, fx.Int32(-1))
            per = total // fx.Int32(8)
            lr = (total % fx.Int32(8) == fx.Int32(0)).select(
                (L % fx.Int32(8)) * per + L // fx.Int32(8), L
            )
            grp = lr // fx.Int32(GM * GX)
            w = lr % fx.Int32(GM * GX)
            rem = nvalid - grp * fx.Int32(GM)
            gm_eff = (rem < fx.Int32(GM)).select(rem, fx.Int32(GM))
            gm_eff = (gm_eff < fx.Int32(1)).select(fx.Int32(1), gm_eff)
            mb = (L >= fx.Int32(0)).select(grp * fx.Int32(GM) + w % gm_eff, GYT)
            nb = w // gm_eff
            # half tiles: expert of routing tile mb >> 1
            e = fx.inttoptr(i32pt, _i64(TEXP))[
                (mb >> fx.Int32(1)) if HALF_TILES else mb
            ]
            e = (mb < GYT).select(e, fx.Int32(-1))
            return mb, nb, e, nvalid, grp, w

        mb, nb, e, nvalid, grp, w = tile_order()
        if e >= fx.Int32(0):
            # thread -> wave (wm, wn) of the WGM x WGN wave grid, lane -> (g, r): row r
            # of a 16-row fragment, K group g
            wave, lane = _split(tid, NW, 64)
            wm, wn = fx.idx2crd(wave, _c_order(WGM, WGN)).unpack()
            g, r = _split(lane, 4, 16)

            a_rsrc = _rsrc(_i64(A) - fx.Int64(CA))
            as_rsrc = _rsrc(_i64(AS))
            ridx = fx.inttoptr(i32pt, _i64(RIDX))
            b_base = _i64(B) + fx.Int64(e) * fx.Int64(b_expert_bytes)
            bs_base = _i64(BS) + fx.Int64(e) * fx.Int64(bs_expert_bytes)
            b_rsrc = _rsrc(b_base - fx.Int64(CB))
            bs_rsrc = _rsrc(bs_base)
            nfb = nb * fx.Int32(BF)

            # ---- DMA sources and LDS destinations ----
            def a_pswz(rl, j):
                # XOR swizzle of a row's 16-B pieces, keyed on the row bits inside a
                # 16-row fragment. fp6 (96-B rows): rows r and r+8 share banks; swap
                # piece pairs on row bit 3 -> the
                #   three ds_read_b64 of a fragment (32-lane phases) cover all 64 banks.
                # fp8 (128-B rows): ds_read_b128 (16-lane phases): XOR by bits 1,2,3 of
                # r
                if const_expr(PPR == 6):
                    return j ^ ((rl >> fx.Int32(3)) & fx.Int32(1))
                f = (
                    (((rl >> fx.Int32(1)) & fx.Int32(1)) * fx.Int32(4))
                    ^ (((rl >> fx.Int32(2)) & fx.Int32(1)) * fx.Int32(3))
                    ^ ((rl >> fx.Int32(3)) & fx.Int32(1))
                )
                return j ^ f

            # a stage part is filled by NW * IxW DMAs, [wave][the wave's DMA t], of 64
            # 16-B pieces each, one per lane (1 KB)
            A_DMAS, B_DMAS = _c_order(NW, IAW), _c_order(NW, IBW)
            A_PIECES, B_PIECES = _c_order(IA, 64), _c_order(IB, 64)

            def pa(t):  # this lane's 16-B piece of A DMA t
                return _at(A_PIECES, _at(A_DMAS, wave, t), lane)

            def pb(t):
                return _at(B_PIECES, _at(B_DMAS, wave, t), lane)

            # LDS byte base (M0) of A DMA t's group of GSA, and t's immediate offset in
            # it
            def a_dst(buf_off, t):
                return (
                    buf_off
                    + _at(A_PIECES, _at(A_DMAS, wave, (t // GSA) * GSA), 0)
                    * fx.Int32(16),
                    (t % GSA) * 1024,
                )

            def b_dst(buf_off, t):
                dma0 = _at(B_DMAS, wave, (t // GSB) * GSB)
                return (
                    buf_off + fx.Int32(A_BYTES) + _at(B_PIECES, dma0, 0) * fx.Int32(16),
                    (t % GSB) * 1024,
                )

            def gathered_row(m):
                """Activation row of tile row m: ridx[m] in stage 1 (gathered x rows), m
                itself in stage 2."""
                return ridx[m] if const_expr(gather) else m

            # the outer extent of a row-major layout enters no stride: GY (tiles) stands
            # for the row counts too
            M_ROWS = _c_order(GY, BM_)  # rows of the routed grid: [m-tile][row]
            # global bytes of the A rows: tile-step-major h [tile][K
            # step][row][piece][16], else [row][K step][piece][16] with 8 pieces (128 B)
            # per K step when padded
            A_GLOBAL = (
                _c_order(GY, KS, BM_, PPR, 16)
                if A_TILE_STEP
                else _c_order(GY, KS, 8 if PAD_A else PPR, 16)
            )
            a_rows = []
            for t in range_constexpr(IAW):
                row, _ = _split(pa(t), BM_, PPR)
                a_rows.append(gathered_row(_at(M_ROWS, mb, row)))

            def build_a_src():
                # LDS slot (row, j) holds global piece a_pswz(row, j) (an involution):
                # bank-conflict-free reads
                out = []
                for t in range_constexpr(IAW):
                    row, j = _split(pa(t), BM_, PPR)
                    adj = fx.Int32(CA - (t % GSA) * 1024)
                    if const_expr(A_TILE_STEP):
                        out.append(_at(A_GLOBAL, mb, 0, row, a_pswz(row, j), 0) + adj)
                    else:
                        out.append(_at(A_GLOBAL, a_rows[t], 0, a_pswz(row, j), 0) + adj)
                return out

            # one expert's weights [N/BN_SLAB][KS][SLB bytes]; a tile's K-step stage is
            # BF/BFS whole slabs: [slab][SLB]
            B_SLABS, B_GLOBAL = _c_order(GX, BF // BFS), _c_order(N // BN_SLAB, KS, SLB)
            b_src = []
            for t in range_constexpr(IBW):
                slab, off = _split(pb(t) * fx.Int32(16), BF // BFS, SLB)
                b_src.append(
                    _at(B_GLOBAL, _at(B_SLABS, nb, slab), 0, off)
                    + fx.Int32(CB - (t % GSB) * 1024)
                )
            # A scales [R][KQ][4 b] dwords: one 16-B piece per (row, quad); lane -> row
            # (wave % NSA) * 64 + lane
            AS_GLOBAL, BS_GLOBAL = _c_order(GY, KQ, 16), _c_order(
                N // 16, KQ, 256 // SBP, SBP
            )
            sa_row = gathered_row(
                _at(_c_order(GY, NSA, 64), mb, wave % fx.Int32(NSA), lane)
            )
            a_src, sa_src = [], []  # filled by the prologue, after the B DMAs
            sb_src = []
            for t in range_constexpr(ISBW):
                tt = wave + fx.Int32(t * NW)
                LPF = 256 // SBP  # lanes per fragment
                f = tt * fx.Int32(64 // LPF) + lane // fx.Int32(LPF)
                sb_src.append(_at(BS_GLOBAL, nfb + f, 0, lane % fx.Int32(LPF), 0))

            # ---- LDS read offsets (i64 units for fragments, i32 for scales): this
            # lane's coordinate in each region's layout, plus per-fragment constants
            # (static coordinates), which fold into the ds_read immediates ----
            # LDS bytes: the operand stages, the scale slots
            RING, SLOTS = _c_order(NSTAGE, STAGE), _c_order(2, SLOT)
            # tile row of (wave row, fragment, lane row)
            TILE_ROWS = _c_order(WGM, FM, 16)
            # a stage's A part: [row][16-B piece][8-B half]
            A_PART = _c_order(BM_, PPR, 2)
            # its B part: [wave column][fragment][lane][8 B]
            B_PART = _c_order(WGN, FN, 64, 3)
            # a scale slot's parts, dwords
            A_SCALES, B_SCALES = _c_order(BM_, 4), _c_order(WGN, FN, 64)
            row0 = _at(TILE_ROWS, wm, 0, r)  # this lane's row in fragment 0
            if const_expr(PPR == 6 and (PAD_A or A_TILE_STEP)):
                # padded and tile-step-major rows are laid head / tail: the lane's 16-B
                # head is piece g, its 8-B tail half g % 2 of piece 4 + g // 2: one
                # ds_read_b128 + one ds_read_b64
                a_parts0 = [
                    (_at(A_PART, row0, a_pswz(r, g), 0), "b128"),
                    (
                        _at(
                            A_PART,
                            row0,
                            a_pswz(r, fx.Int32(4) + (g >> fx.Int32(1))),
                            g & fx.Int32(1),
                        ),
                        1,
                    ),
                ]
            elif const_expr(PPR == 6):
                # fp6: lane's 24 B are the row's 8-B units 3g+p: half u % 2 of piece u
                # // 2
                a_parts0 = []
                for p in range_constexpr(3):
                    u = g * fx.Int32(3) + fx.Int32(p)
                    a_parts0.append(
                        (
                            _at(A_PART, row0, a_pswz(r, u >> fx.Int32(1)), 0)
                            + (u & fx.Int32(1)),
                            1,
                        )
                    )
            else:
                # fp8: lane's 32 B are pieces 2g, 2g+1, each read as one 16-B
                # ds_read_b128
                a_parts0 = [
                    (_at(A_PART, row0, a_pswz(r, g * fx.Int32(2) + fx.Int32(jj)), 0), 2)
                    for jj in range_constexpr(2)
                ]
            b_lds0 = _at(B_PART, wn, 0, lane, 0)
            a_lds = [
                [(q + fx.Int32(_at(A_PART, 16 * mi, 0, 0)), n) for q, n in a_parts0]
                for mi in range_constexpr(FM)
            ]
            b_lds = [
                [
                    b_lds0 + fx.Int32(A_BYTES // 8 + _at(B_PART, 0, ni, 0, p))
                    for p in range_constexpr(3)
                ]
                for ni in range_constexpr(FN)
            ]
            l32 = fx.recast_iter(fx.Int32, lds_raw)
            sa_lds0, sb_lds0 = _at(A_SCALES, row0, g), _at(B_SCALES, wn, 0, lane)
            sa_lds = [
                sa_lds0 + fx.Int32(_at(A_SCALES, 16 * mi, 0))
                for mi in range_constexpr(FM)
            ]
            sb_lds = [
                sb_lds0 + fx.Int32(SC_A // 4 + _at(B_SCALES, 0, ni, 0))
                for ni in range_constexpr(FN)
            ]
            # the staged output tile, padded rows of OSTR bytes: [row][output
            # fragment][4 columns = 8 B] output fragment of (wave column, the wave's
            # fragment)
            OUT_FRAGS = _c_order(WGN, BNO // (16 * WGN))
            STAGED = fx.make_layout((BM_, BNO // 16, 4), (OSTR // 8, 4, 1))

            # ---- DMA (_dma_to_lds): every lane a 16-B piece from its own (gathered)
            # voffset into a wave-uniform LDS base (M0). The async form is tracked by
            # asyncmark / wait_asyncmark; the synchronous buffer->LDS copy made LLVM
            # wait vmcnt(0) before every ds_read (1463 -> 1849 TF/s on MI355X when
            # switched) ----
            def dma_a(t, buf_off, s, loop):
                """A DMA t of K step s into the stage at buf_off. loop: the step offset
                in the wave-uniform soffset (K loop); else in the voffset (prologue)."""
                dst, io = a_dst(buf_off, t)
                lds = _lds_ptr3(lds_i32, dst)
                voff, soff = (
                    (a_src[t], s * fx.Int32(ASTEP))
                    if loop
                    else (a_src[t] + s * fx.Int32(ASTEP), 0)
                )
                _dma_to_lds(a_rsrc, lds, _v(16), _v(voff), _v(soff), _v(io))

            def dma_b(t, buf_off, s, loop):
                dst, io = b_dst(buf_off, t)
                lds = _lds_ptr3(lds_i32, dst)
                voff, soff = (
                    (b_src[t], s * fx.Int32(SLB))
                    if loop
                    else (b_src[t] + s * fx.Int32(SLB), 0)
                )
                _dma_to_lds(b_rsrc, lds, _v(16), _v(voff), _v(soff), _v(io), **nt_aux())

            def dma_scale_a(slot_off, q):
                """Quad q's A scales into the slot at slot_off. Waves >= NSA repeat wave
                (w % NSA)'s DMA (same bytes, same place): no branch."""
                _dma_to_lds(
                    as_rsrc,
                    _lds_ptr3(
                        lds_i32,
                        slot_off
                        + _at(_c_order(NSA, 64, 16), wave % fx.Int32(NSA), 0, 0),
                    ),
                    _v(16),
                    _v(sa_src[0] + _at(AS_GLOBAL, 0, q, 0)),
                    _v(0),
                    _v(0),
                )

            def dma_scale_b(t, slot_off, q):
                dst = (
                    slot_off
                    + fx.Int32(SC_A)
                    + _at(_c_order(ISB_, 64, SBP), wave + fx.Int32(t * NW), 0, 0)
                )
                _dma_to_lds(
                    bs_rsrc,
                    _lds_ptr3(lds_i32, dst),
                    _v(SBP),
                    _v(sb_src[t] + _at(BS_GLOBAL, 0, q, 0, 0)),
                    _v(0),
                    _v(0),
                    **nt_aux(),
                )

            def operand_dmas(nxt, nbuf, j):
                """K step nxt's A and B DMAs into the stage at nbuf, as thunks the MFMA
                groups issue."""
                ath = [lambda t=t: dma_a(t, nbuf, nxt, True) for t in range(IAW)]
                bth = [lambda t=t: dma_b(t, nbuf, nxt, True) for t in range(IBW)]
                if const_expr(ALT_HALVES):
                    # 6-bit A rows share a 128-B line between consecutive steps (96-B
                    # chunks): one row half is issued last in step s and first in step
                    # s+1 (halves swap by step parity), so that half's shared line is
                    # re-touched while still in the vL1D
                    lo, hi = ath[: IAW // 2], ath[IAW // 2 :]
                    return (hi + bth + lo) if j % 2 == 0 else (lo + bth + hi)
                return ath + bth

            def scale_dmas(s, j):
                """Quad s//4+1's scale DMAs that step j of quad s//4 issues, as thunks:
                the A piece at j 0, the B pieces spread over steps j < SJ. Their slot
                ((s//4+1) % 2) was last read in quad s//4-1, at least 4 barriers back,
                each after lgkmcnt(0); they are waited by the barrier before the quad's
                first read."""
                # that barrier (quad step j = 3's) leaves the newest NSTAGE - 2 DMA
                # groups in flight: no scale piece in them
                SJ = 4 - (NSTAGE - 2)
                pieces = [("a", 0)] + [("b", t) for t in range(ISBW)]
                mine = [pc for i, pc in enumerate(pieces) if i * SJ // len(pieces) == j]
                if const_expr(len(mine) == 0):
                    return []
                qn = s // fx.Int32(4) + fx.Int32(1)
                slot_off = fx.Int32(SC_BASE) + _at(SLOTS, qn % fx.Int32(2), 0)
                return [
                    (
                        (lambda: dma_scale_a(slot_off, qn))
                        if kind == "a"
                        else (lambda t=t: dma_scale_b(t, slot_off, qn))
                    )
                    for kind, t in mine
                ]

            def prologue():
                """K steps 0 .. NSTAGE-2 and quad 0's scales. B of every prologue step
                and the B scales go first: they do not depend on the gathered rows, so
                they stream while the ridx loads behind the A offsets are in flight.
                Then A, one async group per step."""
                for p in range_constexpr(NSTAGE - 1):
                    for t in range_constexpr(IBW):
                        dma_b(t, fx.Int32(p * STAGE), fx.Int32(min(p, KS - 1)), False)
                for t in range_constexpr(ISBW):
                    dma_scale_b(t, fx.Int32(SC_BASE), fx.Int32(0))
                a_src.extend(build_a_src())
                sa_src.append(_at(AS_GLOBAL, sa_row, 0, 0))
                for p in range_constexpr(NSTAGE - 1):
                    for t in range_constexpr(IAW):
                        dma_a(t, fx.Int32(p * STAGE), fx.Int32(min(p, KS - 1)), False)
                    if const_expr(p == 0):
                        dma_scale_a(fx.Int32(SC_BASE), fx.Int32(0))
                    rocdl.asyncmark()

            # ---- barriers ----
            def k_barrier():
                """Stage s landed (all but the newest NSTAGE-2 DMA groups) and every
                wave finished reading stage s-1, which the DMAs issued after it
                overwrite. fx.barrier, not the raw s_barrier: it has memory semantics,
                so no ds_read, ds_write or DMA is moved across it."""
                rocdl.wait_asyncmark(NSTAGE - 2)
                fx.barrier()

            def early_barrier():
                """Step s+1's barrier, ahead of step s's last MFMA group, which then
                reads stage s+1. lgkmcnt(0): this wave's reads of the stage the next
                DMAs overwrite are complete before any wave passes it."""
                rocdl.asyncmark()
                rocdl.wait_asyncmark(NSTAGE - 2)
                rocdl.s_waitcnt(lgkmcnt=0)
                fx.barrier()

            # ---- LDS reads: one plain ds_read per fragment piece. Paired ds_read2_b64
            # conflict 2-way at these 24-B lane strides and cost ~3.7x a plain
            # ds_read_b64 in SQ_LDS_IDX_ACTIVE on gfx950; plain b64 (32-lane phases) and
            # b128 (16-lane phases) reads of this layout are conflict-free.
            # Raw rocdl.sched_barrier(_SEP) between them: FlyDSL 0.5.0 has no stable
            # scheduling fence (sched_dsrd / sched_dswr are group barriers, another
            # contract); without it LLVM pairs the reads and the op loses up to 2.4%
            # (A/B timing on 10 shapes: 5 slower) ----
            def a_frag(cur, mi):
                """A fragment mi of the stage at cur (i64 units)."""
                vals = []
                for q, n in a_lds[mi]:
                    rocdl.sched_barrier(_SEP)
                    if const_expr(n == "b128"):
                        hv = _lds_load_b128(lds_i32, cur + q)
                        vals += [fx.Int64(hv[0]), fx.Int64(hv[1])]
                    else:
                        vals += [
                            fx.ptr_load(fx.add_offset(lds64, cur + q + fx.Int32(k)))
                            for k in range_constexpr(n)
                        ]
                rocdl.sched_barrier(_SEP)
                return fx.Vector.from_elements(vals, fx.Int64).bitcast(fx.Int32)

            def b_frag(cur, ni):
                """B fragment ni of the stage at cur: 24 B per lane, three
                ds_read_b64."""
                vals = []
                for q in b_lds[ni]:
                    rocdl.sched_barrier(_SEP)
                    vals.append(fx.ptr_load(fx.add_offset(lds64, cur + q)))
                rocdl.sched_barrier(_SEP)
                return fx.Vector.from_elements(vals, fx.Int64).bitcast(fx.Int32)

            def read_scale(so, q):
                return fx.ptr_load(fx.add_offset(l32, so + q))

            def load_scales(qd):
                """Quad qd's A and B scales (one dword per fragment row / column), one
                ds_read_b32 each."""
                so = fx.Int32(SC_BASE // 4) + (qd % fx.Int32(2)) * fx.Int32(SLOT // 4)
                a_sc, b_sc = [], []
                for mi in range_constexpr(FM):
                    a_sc.append(read_scale(so, sa_lds[mi]))
                for ni in range_constexpr(FN):
                    b_sc.append(read_scale(so, sb_lds[ni]))
                return a_sc, b_sc

            def stage_off(s):
                return _at(RING, s % fx.Int32(NSTAGE), 0)

            # ---- MMA ----
            # scaled 16x16x128 MFMA atoms, one per E8M0 scale-byte select; B is the
            # MFMA's srcA (formats swapped)
            mma_atoms = [
                fx.make_mma_atom(
                    fx.rocdl.cdna4.MFMA_Scale(
                        16,
                        16,
                        128,
                        FMT_ELEM[b_fmt],
                        FMT_ELEM[a_fmt],
                        opsel_a=jj,
                        opsel_b=jj,
                    )
                )
                for jj in range(4)
            ]

            def _frag(vec_i32, nw):
                f = fx.make_rmem_tensor(fx.make_layout(nw, 1), fx.Int32)
                f.store(vec_i32)
                return f

            def mma(out, accs, mi, ni, av, bv, j, sc):
                """out[mi, ni] = accs[mi, ni] + MFMA over K step j of its quad (scale
                byte j). B is srcA, so the accumulator holds C^T per 16x16 tile: acc[i]
                of lane l = C[m = l % 16, n = 4*(l//16) + i], 4 consecutive output
                columns per lane."""
                k = mi * FN + ni
                cf = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                cf.store(accs[k])
                fx.gemm(
                    mma_atoms[j],
                    cf,
                    _frag(bv, 6),
                    _frag(av, abits),
                    cf,
                    scale_a=sc[1][ni],
                    scale_b=sc[0][mi],
                )
                if const_expr(ACC_AGPR):
                    out[k] = _pin_agpr(cf.load(), k)
                else:
                    out[k] = cf.load()

            # ---- one K step ----
            def mfma_group(out, accs, mi, av, bv, j, sc, cur, thunks, slot_of):
                """MFMA group mi: FN MFMAs in DMA_SLOTS slots. Slot h issues its DMAs
                right after its first MFMA (the M0 write fills the M0 -> LDS-DMA wait
                state instead of an s_nop); slot 0 then reads A fragment mi+1
                (returned)."""
                # Raw rocdl.sched_barrier(0), here and in last_group: FlyDSL 0.5.0 has
                # no stable scheduling fence. They keep each slot's DMAs and the lagged
                # fragment reads between the MFMAs as written; without them the
                # scheduler regroups them and the op loses up to 3.4% (A/B timing on 10
                # shapes: 6 slower)
                av_next = None
                for h in range_constexpr(DMA_SLOTS):
                    n0, n1 = h * FN // DMA_SLOTS, (h + 1) * FN // DMA_SLOTS
                    rocdl.sched_barrier(0)
                    mma(out, accs, mi, n0, av, bv[n0], j, sc)
                    for i in range_constexpr(len(thunks)):
                        if const_expr(slot_of[i] == mi * DMA_SLOTS + h):
                            thunks[i]()
                    if const_expr(h == 0 and mi + 1 < FM):
                        rocdl.sched_barrier(0)
                        av_next = a_frag(cur, mi + 1)
                    for ni in range_constexpr(n0 + 1, n1):
                        mma(out, accs, mi, ni, av, bv[ni], j, sc)
                return av_next

            def last_group(out, accs, av, bv, s, j, sc, soff, xs_issue):
                """Step s's last MFMA group, behind step s+1's barrier, reading stage
                s+1 under it: A fragment 0 right after MFMA 0, so step s+1's first MFMA
                waits for A0 + B0 only; B fragment ni-1 after MFMA ni (with early DMAs:
                ni after ni and its DMA); at a quad's last step with SCALE_AHEAD the
                next quad's scales, A scale 0 before the MFMAs. Every read targets a
                stage or slot this barrier waited for. Returns them (step s+1's pre)."""
                mi = FM - 1
                early_barrier()
                ncur = soff(1) // fx.Int32(8)
                nbv = []
                psc = SCALE_AHEAD and j == 3
                nsa, nsb = [], []
                if const_expr(psc):
                    so = fx.Int32(SC_BASE // 4) + (
                        (s + fx.Int32(1)) // fx.Int32(4) % fx.Int32(2)
                    ) * fx.Int32(SLOT // 4)
                    nsa = [read_scale(so, sa_lds[0])]
                xth = []
                if const_expr(xs_issue):
                    # 2-stage ring: step s+1's first N_EARLY_DMA DMAs; they target stage
                    # s+2 = stage s, whose reads all finished before this barrier
                    # (lgkmcnt(0)); this group's operands are in registers
                    s1 = s + fx.Int32(1)
                    nxt1 = s1 + fx.Int32(NSTAGE - 1)
                    xth = operand_dmas(
                        nxt1, stage_off(s1 + fx.Int32(NSTAGE - 1)), j + 1
                    )[:N_EARLY_DMA]
                lag = 0 if xs_issue else 1
                nav = None
                for ni in range_constexpr(FN):
                    mma(out, accs, mi, ni, av, bv[ni], j, sc)
                    rocdl.sched_barrier(0)
                    if const_expr(ni < len(xth)):
                        xth[ni]()
                    if const_expr(ni == 0):
                        nav = a_frag(ncur, 0)
                        rocdl.sched_barrier(0)
                    if const_expr(ni >= lag):
                        nbv.append(b_frag(ncur, ni - lag))
                        if const_expr(psc):
                            nsb.append(read_scale(so, sb_lds[ni - lag]))
                if const_expr(lag == 1):
                    nbv.append(b_frag(ncur, FN - 1))
                    if const_expr(psc):
                        nsb.append(read_scale(so, sb_lds[FN - 1]))
                if const_expr(psc):
                    nsa += [read_scale(so, sa_lds[m]) for m in range(1, FM)]
                return nbv, nav, ((nsa, nsb) if psc else None)

            def step(
                accs,
                s,
                j,
                sc,
                pre,
                has_next=False,
                s_static=None,
                ring=None,
                xs_skip=False,
                xs_issue=False,
            ):
                """K step s (j = s % 4) on stage s: issues step s + NSTAGE - 1's DMAs
                and the next quad's scale DMAs over the MFMA groups. pre: stage s's B
                fragments, A fragment 0 and (SCALE_AHEAD) the quad's scales, read by
                step s-1 under its last group, which also ran this step's barrier.
                has_next: run step s+1's barrier and reads in this step's last group.
                s_static: the step index when known at trace time (peeled quads): DMAs
                past the last K step are skipped. A loop step never issues past it: the
                last full quad is peeled, or KT >= NSTAGE - 1 steps follow it. ring
                (f8x8 main loop): k -> stage offset of step s + k from loop-carried
                rotating offsets."""
                soff = (lambda k: stage_off(s + fx.Int32(k))) if ring is None else ring
                nxt = s + fx.Int32(NSTAGE - 1)
                nbuf = soff(NSTAGE - 1)
                thunks = []
                if const_expr(s_static is None or s_static + NSTAGE - 1 < KS):
                    thunks = operand_dmas(nxt, nbuf, j)
                    if const_expr(xs_skip):
                        # issued by step s-1 after its early barrier
                        thunks = thunks[N_EARLY_DMA:]
                nab = len(thunks)
                if const_expr(s_static is None or s_static // 4 + 1 < KQ):
                    thunks = scale_dmas(s, j) + thunks
                    npre = len(thunks) - nab
                    if const_expr(ALT_HALVES and npre > 0 and nab > 0):
                        # scale DMAs after the first A half: nothing between the two
                        # touches of the shared line
                        thunks = (
                            thunks[npre : npre + IAW // 2]
                            + thunks[:npre]
                            + thunks[npre + IAW // 2 :]
                        )
                if const_expr(j == 0):
                    if const_expr(SCALE_AHEAD):
                        sc = pre[2]
                    else:
                        sc = load_scales(s // fx.Int32(4))
                cur = soff(0) // fx.Int32(8)
                bv, av_next = pre[0], pre[1]
                out = list(accs)
                # DMA issue slots: the early-barrier group has none
                NSL = (FM - (1 if has_next else 0)) * DMA_SLOTS
                # 2 stages: step s+1's DMAs must land by step s's last group, so they go
                # in its first half
                NSLD = NSL // 2 if NSTAGE == 2 else NSL
                slot_of = [i * NSLD // len(thunks) for i in range(len(thunks))]
                nxt_pre = None
                for mi in range_constexpr(FM):
                    av = av_next
                    if const_expr(has_next and mi == FM - 1):
                        nxt_pre = last_group(
                            out, accs, av, bv, s, j, sc, soff, xs_issue
                        )
                    else:
                        av_next = mfma_group(
                            out, accs, mi, av, bv, j, sc, cur, thunks, slot_of
                        )
                if const_expr(not has_next):
                    rocdl.asyncmark()
                return out, sc, nxt_pre

            # ---- epilogue ----
            def ep_base(row):
                if const_expr(STAGING is None or STAGING[0] == STAGING[1]):
                    return fx.Int32(0 if STAGING is None else STAGING[0])
                return (row >= fx.Int32(BM_ // 2)).select(
                    fx.Int32(STAGING[1]), fx.Int32(STAGING[0])
                )

            def stage_h(mi, accs):
                """SwiGLU of fragment group mi into LDS as bf16. Lane (r, g) holds row
                r, columns 4g..4g+3 of each fragment (B is srcA): 4 bf16 -> one 8-B LDS
                write."""
                gs, us = [], []
                for p in range_constexpr(FN // 2):
                    for i in range_constexpr(4):
                        gs.append(fx.Vector(accs[mi * FN + 2 * p])[i])
                        us.append(fx.Vector(accs[mi * FN + 2 * p + 1])[i])
                # h = silu(g) * u (act.silu_mul_batch). For the quantized h the last
                # product is an fma with +0.0: the same bits, except that an exactly
                # -0.0 product becomes +0.0. mx_quant_store gives -0.0 the +0 code (its
                # sign is v < 0), the hardware convert keeps the sign bit. Not covered:
                # a negative product that underflows to -0.0 in bf16 (|h| < 2^-134)
                # keeps the -0 code; the same value, the same final output bits
                sig = sigmoid_batch(gs)
                hs = [
                    (
                        fx.fma(gs[i] * sig[i], us[i], fx.Float32(0.0))
                        if epi == "swiglu_q"
                        else gs[i] * sig[i] * us[i]
                    )
                    for i in range(len(gs))
                ]
                rl = _at(TILE_ROWS, wm, mi, r)
                for p in range_constexpr(FN // 2):
                    pk = fx.Vector.from_elements(
                        [hs[p * 4 + i] for i in range(4)], fx.Float32
                    ).to(fx.BFloat16)
                    q8 = _at(STAGED, rl, _at(OUT_FRAGS, wn, p), g) + ep_base(
                        rl
                    ) // fx.Int32(8)
                    fx.ptr_store(
                        fx.Int64(pk.bitcast(fx.Int64)[0]), fx.add_offset(lds64, q8)
                    )

            def stage_c(accs):
                """The bf16 C tile into LDS (padded rows), one ds_write_b64 per fragment
                and lane."""
                for mi in range_constexpr(FM):
                    rl = _at(TILE_ROWS, wm, mi, r)
                    for ni in range_constexpr(FN):
                        av4 = fx.Vector(accs[mi * FN + ni])
                        pk = fx.Vector.from_elements(
                            [av4[i] for i in range(4)], fx.Float32
                        ).to(fx.BFloat16)
                        q8 = _at(STAGED, rl, _at(OUT_FRAGS, wn, ni), g)
                        fx.ptr_store(
                            fx.Int64(pk.bitcast(fx.Int64)[0]), fx.add_offset(lds64, q8)
                        )

            def warm_next_group():
                """Stage 2, tile-step-major h: the next GM group's A rows are one
                contiguous region. The last NWARM blocks of a group each touch 1/NWARM
                of its lines, one dword per 128-B line, by async DMA into the strip at
                the top of LDS, so the next group's first round reads A from L2 instead
                of fabric."""
                slot = w - fx.Int32(GW_ - NWARM_)
                nxt = (grp + fx.Int32(1)) * fx.Int32(GM)
                dow = (slot >= fx.Int32(0)) & (nxt < nvalid)
                slot_c = (slot >= fx.Int32(0)).select(slot, fx.Int32(0))
                base_m = dow.select(nxt, mb)
                for k in range_constexpr(NWI_):
                    idx = fx.Int32(k * NT) + wave * fx.Int32(64) + lane
                    idx = (idx < fx.Int32(LPB_)).select(idx, fx.Int32(LPB_ - 1))
                    li = slot_c * fx.Int32(LPB_) + idx
                    li = (li < fx.Int32(LINES_)).select(li, fx.Int32(LINES_ - 1))
                    # line li of the group's warmed lines: [m-tile][line]
                    mt, line = _split(li, GM, LPM_)
                    mt = dow.select(mt, fx.Int32(0))
                    m_abs = base_m + mt
                    # a partial last group: stay in used tiles
                    m_abs = (m_abs < nvalid).select(m_abs, nvalid - fx.Int32(1))
                    voff = _at(
                        _c_order(GY, BM_ * a_row_bytes // 128, 128), m_abs, line, 0
                    ) + fx.Int32(CA)
                    wdst = fx.Int32(lds_bytes - NWI_ * NT * 4) + _at(
                        _c_order(NWI_, NW, 256), k, wave, 0
                    )
                    _dma_to_lds(
                        a_rsrc, _lds_ptr3(lds_i32, wdst), _v(4), _v(voff), _v(0), _v(0)
                    )

            def quant_store_h():
                """Each lane a 32-element block of a staged h row: MX-quantized into
                stage 2's A layout (h codes and the [R][KQ][4] scale bytes). A block
                holding inf / NaN takes mx_quant_store's own arithmetic (rare)."""
                y_base = _i64(C) + fx.Int64(mb * fx.Int32(BM_)) * fx.Int64(q_row_bytes)
                s_base = _i64(CS) + fx.Int64(mb * fx.Int32(BM_)) * fx.Int64(16 * KQ2)
                BPR = BNO // 32  # 32-element blocks per row of the tile
                # the staged tile, dwords: [row][block][16]
                STAGED32 = fx.make_layout((BM_, BPR, 16), (OSTR // 4, 16, 1))
                # one tile's 6-bit h codes in dwords: tile-step-major [K step][row][24]
                # or row-major [row][K block][6]
                H_TILE = (
                    _c_order(n_out // 128, BM_, 24)
                    if H_TILE_STEP
                    else _c_order(BM_, n_out // 32, 6)
                )
                for it in range_constexpr(BM_ * BPR // NT):
                    u = tid + fx.Int32(it * NT)
                    row = u // fx.Int32(BPR)
                    cb = u % fx.Int32(BPR)
                    words = []
                    for c in range_constexpr(4):
                        v = fx.Vector(
                            fx.ptr_load(
                                fx.add_offset(
                                    l32,
                                    _at(STAGED32, row, cb, 4 * c)
                                    + ep_base(row) // fx.Int32(4),
                                ),
                                result_type=T.vec(4, T.i32),
                            )
                        )
                        for k in range_constexpr(4):
                            words.append(fx.Int32(v[k]))
                    kbv = nb * fx.Int32(BPR) + cb
                    y_dw, y_dw_t = None, None
                    if const_expr(H_TILE_STEP):
                        # tile-step-major: the 24-B block of K-block g = kbv % 4 as a
                        # 16-B head at 16g, an 8-B tail at 64 + 8g
                        rs = _at(H_TILE, kbv // fx.Int32(4), row, 0)
                        y_dw = rs + (kbv % fx.Int32(4)) * fx.Int32(4)
                        y_dw_t = rs + fx.Int32(16) + (kbv % fx.Int32(4)) * fx.Int32(2)
                    elif const_expr(abits == 6):
                        # row-major h (h_layout "rows"): the block's 24 B are contiguous
                        # in its row
                        y_dw = _at(H_TILE, row, kbv, 0)
                        y_dw_t = y_dw + fx.Int32(4)
                    if mx6_nonfinite(words):
                        mx6_quant_store_exact(
                            words,
                            a_fmt,
                            y_base,
                            row * fx.Int32(q_row_bytes),
                            kbv,
                            s_base,
                            row * fx.Int32(16 * KQ2),
                            y_dw,
                            y_dw_t,
                        )
                    else:
                        mx6_quant_store_hw(
                            words,
                            a_fmt,
                            y_base,
                            row * fx.Int32(q_row_bytes),
                            kbv,
                            s_base,
                            row * fx.Int32(16 * KQ2),
                            y_dw,
                            y_dw_t,
                        )

            def store_c():
                """The staged bf16 tile to C in 16-B coalesced non-temporal stores."""
                c_tile_base = _i64(C) + fx.Int64(mb * fx.Int32(BM_)) * fx.Int64(
                    n_out * 2
                )
                ct = ptr_buf_tensor(c_tile_base, fx.Int32, unit_elems=4, unit_stride=1)
                CPR = BNO // 8  # 16-B chunks per row
                for it in range_constexpr(BM_ * CPR // NT):
                    c = tid + fx.Int32(it * NT)
                    row = c // fx.Int32(CPR)
                    cc = c % fx.Int32(CPR)
                    v = fx.ptr_load(
                        fx.add_offset(
                            l32,
                            _at(
                                fx.make_layout((BM_, CPR, 4), (OSTR // 4, 4, 1)),
                                row,
                                cc,
                                0,
                            ),
                        ),
                        result_type=T.vec(4, T.i32),
                    )
                    buf_copy_store(
                        ct,
                        (row * fx.Int32(n_out) + nb * fx.Int32(BNO) + cc * fx.Int32(8))
                        // fx.Int32(2),
                        v,
                        fx.Int32,
                        4,
                        cache_modifier=2,
                    )

            # ---- the tile: prologue, K loop, epilogue ----
            prologue()
            zero = fx.Vector.filled(4, 0.0, fx.Float32)
            accs = [zero for _ in range(FM * FN)]
            # the first barrier and stage 0's reads ahead of the loop; every later
            # step's barrier runs inside the step before it
            k_barrier()
            c0 = stage_off(fx.Int32(0)) // fx.Int32(8)
            pre = (
                [b_frag(c0, ni) for ni in range_constexpr(FN)],
                a_frag(c0, 0),
                load_scales(fx.Int32(0)) if SCALE_AHEAD else None,
            )

            def carry(p):  # pre as loop-carried values
                return (
                    list(p[0])
                    + [p[1]]
                    + (list(p[2][0]) + list(p[2][1]) if SCALE_AHEAD else [])
                )

            def uncarry(st):
                o = FM * FN
                scs = (
                    (
                        st[o + FN + 1 : o + FN + 1 + FM],
                        st[o + FN + 1 + FM : o + FN + 1 + FM + FN],
                    )
                    if SCALE_AHEAD
                    else None
                )
                return st[o : o + FN], st[o + FN], scs

            # 2-stage ring: step s+1's first N_EARLY_DMA DMAs are issued in step s's
            # last group; needs a predecessor step for every loop step (the peeled first
            # quad)
            EARLY_DMA = NSTAGE == 2 and ACC_AGPR and KQF > 0

            def xdead(x):
                return x + NSTAGE - 1 >= KS

            def static_steps(accs, pre, s0, n, more):
                """K steps s0 .. s0 + n - 1 of one quad with trace-time step indices
                (the peeled quads, the K tail). more: a step follows the last one."""
                sc = None
                for i in range_constexpr(n):
                    sq, hn = s0 + i, more or i < n - 1
                    accs, sc, pre = step(
                        accs,
                        fx.Int32(sq),
                        sq % 4,
                        sc,
                        pre,
                        has_next=hn,
                        s_static=sq,
                        xs_skip=EARLY_DMA and sq >= 1 and not xdead(sq),
                        xs_issue=EARLY_DMA and hn and not xdead(sq + 1),
                    )
                return accs, pre

            q0 = 0
            if const_expr(ACC_AGPR and KQF > 0):
                # K quad 0 peeled: its first step's MFMAs start from the zero
                # accumulators (src2 = 0), so the loop phi starts from an MFMA result
                # (no constant seed to split / copy around the loop)
                accs, pre = static_steps(accs, pre, 0, 4, KQF > 1 or KT > 0)
                q0 = 1
            # the last full quad is peeled with static steps when its steps would issue
            # DMAs past the last K step
            q1 = KQF - 1 if (KT < 2 and KQF - 1 >= q0) else KQF
            if const_expr(q1 > q0):
                # the stage offsets as loop-carried values; f8x4 (A8W6 256x128) lost 3%
                # with it: f8x8 loops only
                CARRY_OFFS = NSTAGE == 3 and ACC_AGPR
                NRG = 3 if CARRY_OFFS else 0
                init = (
                    accs
                    + carry(pre)
                    + (
                        [fx.Int32(((q0 + c) % 3) * STAGE) for c in range(3)]
                        if CARRY_OFFS
                        else []
                    )
                )
                for qd, state in range(
                    fx.Int32(q0), fx.Int32(q1), fx.Int32(1), init=init
                ):
                    st = list(state)
                    O = None
                    if const_expr(CARRY_OFFS):
                        O = st[len(st) - NRG :]
                        st = st[: len(st) - NRG]
                    cur_acc = st[: FM * FN]
                    pre = uncarry(st)
                    qi = fx.Int32(qd)
                    sc = None
                    for j in range_constexpr(4):
                        rg = (
                            (lambda k, j=j, O=O: O[(j + k) % 3]) if CARRY_OFFS else None
                        )
                        cur_acc, sc, pre = step(
                            cur_acc,
                            qi * fx.Int32(4) + fx.Int32(j),
                            j,
                            sc,
                            pre,
                            has_next=True,
                            ring=rg,
                            xs_skip=EARLY_DMA,
                            xs_issue=EARLY_DMA,
                        )
                    # 4 steps per quad: stage offsets of steps 4q+4.. are those of
                    # 4q+1.. (rotate by one)
                    results = (
                        yield cur_acc
                        + carry(pre)
                        + ([O[1], O[2], O[0]] if CARRY_OFFS else [])
                    )
                res = list(results)
                if const_expr(CARRY_OFFS):
                    res = res[: len(res) - NRG]
                accs = res[: FM * FN]
                pre = uncarry(res)
            if const_expr(q1 < KQF):
                accs, pre = static_steps(accs, pre, q1 * 4, 4, KT > 0)
            if const_expr(KT > 0):
                accs, pre = static_steps(accs, pre, KQF * 4, KT, False)
            rocdl.wait_asyncmark(0)

            # wait_asyncmark(0) alone left LDS DMAs in flight (racy 256x128 runs)
            rocdl.s_waitcnt(vmcnt=0)
            if const_expr(STAGING is None):
                # Every wave is done with the ring and the scale slots. fx.barrier, not
                # rocdl.s_barrier: the raw barrier has no memory semantics and LLVM
                # hoisted the staging ds_writes above it. Invariant (stage 2): no warm
                # DMA may be issued until every wave has passed this barrier, because
                # the warm strip covers scale slot 1 (and more of the top of LDS), which
                # a lagging wave still reads when the last K quad has one step (with the
                # warm first, such layers gave wrong stage-2 tiles under memory load,
                # about 1 call in 10^5).
                fx.barrier()
            if const_expr(WARM):
                warm_next_group()
            if const_expr(SWI):
                for mi in range_constexpr(FM):
                    stage_h(mi, accs)
            else:
                stage_c(accs)
            fx.barrier()
            if const_expr(WARM):
                # the warm DMAs land (in the strip) before the stores and the end of the
                # wave
                rocdl.s_waitcnt(vmcnt=0)
            if const_expr(epi == "swiglu_q"):
                quant_store_h()
            else:
                store_c()

    @flyc.jit
    def launch(
        A: fx.Pointer,
        AS: fx.Pointer,
        RIDX: fx.Pointer,
        TEXP: fx.Pointer,
        B: fx.Pointer,
        BS: fx.Pointer,
        C: fx.Pointer,
        CS: fx.Pointer,
        m_tiles: fx.Int32,
        stream: fx.Stream,
    ):
        gemm(A, AS, RIDX, TEXP, B, BS, C, CS, m_tiles).launch(
            grid=(N // BN_, m_tiles * 2 if HALF_TILES else m_tiles, 1),
            block=(NT, 1, 1),
            stream=stream,
        )

    if ACC_AGPR:
        _agpr_form_mfma(launch)
    return launch


def compile_mxfp6_moe_pad_rows():
    """The copy kernel of the padded rows (compile_mxfp6_moe_gemm's PAD_A): gathered
    6-bit rows [T][KS][96 B] -> [T][KS][128 B], head
    / tail order (four 16-B heads, then the four 8-B tails), pad bytes written as zero,
    so every destination line is written whole. One 16-B output chunk per lane per pass,
    CPT (8) passes per block; a wave pass writes 1 KB contiguous, reads 768 B."""
    NTX, CPT = 256, 8

    @flyc.kernel(name="mxfp6_moe_pad_rows", known_block_size=[NTX, 1, 1])
    def pad_rows(SRC: fx.Pointer, DST: fx.Pointer, NCH: fx.Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        d4 = ptr_buf_tensor(_i64(DST), fx.Int32, unit_elems=4, unit_stride=1)
        s2 = ptr_buf_tensor(_i64(SRC), fx.Int32, unit_elems=2, unit_stride=1)
        vals = []
        for k in range_constexpr(CPT):
            c = (bid * fx.Int32(CPT) + fx.Int32(k)) * fx.Int32(NTX) + tid
            c = (c < NCH).select(c, NCH - fx.Int32(1))
            j = c % fx.Int32(8)
            # output chunk j < 4: words 3j, 3j+1 (lane g=j's head); j = 4: words 2, 5; j
            # = 5: words 8, 11 (tails)
            wa = (j < fx.Int32(4)).select(
                j * fx.Int32(3), (j == fx.Int32(4)).select(fx.Int32(2), fx.Int32(8))
            )
            wb = (j < fx.Int32(4)).select(
                j * fx.Int32(3) + fx.Int32(1),
                (j == fx.Int32(4)).select(fx.Int32(5), fx.Int32(11)),
            )
            rb = (c // fx.Int32(8)) * fx.Int32(24)
            va = fx.Vector(buf_copy_load(s2, rb + wa * fx.Int32(2), fx.Int32, 2))
            vb = fx.Vector(buf_copy_load(s2, rb + wb * fx.Int32(2), fx.Int32, 2))
            v = [va[0], va[1], vb[0], vb[1]]
            vals.append((c, j, v))
        for c, j, v in vals:
            keep = j < fx.Int32(6)
            w = fx.Vector.from_elements(
                [keep.select(fx.Int32(v[i]), fx.Int32(0)) for i in range(4)], fx.Int32
            )
            buf_copy_store(d4, c * fx.Int32(4), w, fx.Int32, 4)

    @flyc.jit
    def launch(SRC: fx.Pointer, DST: fx.Pointer, NCH: fx.Int32, stream: fx.Stream):
        pad_rows(SRC, DST, NCH).launch(
            grid=((NCH + NTX * CPT - 1) // (NTX * CPT), 1, 1),
            block=(NTX, 1, 1),
            stream=stream,
        )

    return launch
