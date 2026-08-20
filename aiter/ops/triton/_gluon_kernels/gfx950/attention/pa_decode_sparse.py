# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950) sparse decode for DeepSeek-V4 and separated-rope MLA (GLM-5).

One kernel serves two geometries, selected by the ROPE_SEPARATE constexpr:

    False (DSv4): rope lives inside the pow-2 row. One LDS plane
        kv_smem [BLOCK_K, HEAD_SIZE], one QK MFMA chain, V = the same plane.
    True (MLA):   rope is appended (kv_lora_rank latent + rope = QK width;
        V = the latent only). A second pow-2 LDS plane rope_smem holds the
        K-only rope and the QK contraction chains a second MFMA into the first.
        Plane 0 is the entire V in both geometries.

Cache formats (Fmt.KIND), per segment:

    "bf16"     full-row bf16
    "dsv4"     448 fp8 | 64 bf16 rope (576 B rows) + 8 B UE8M0-per-64 trailer
    "uniform"  whole-row fp8 + separate f32 per-64 kv_scales
    "tensor"   whole-row fp8 + a single per-tensor f32 scale (k_scale). The
               scale never touches the tile loop: K-side folds into qk_scale,
               V-side into p.
    "dsmla"    vLLM fp8_ds_mla: 512 fp8 | 4 f32 per-128 scales | 64 bf16 rope
               (656 B rows); requires ROPE_SEPARATE.

Compile-time state is bundled into aggregates instead of long parameter lists:
Cfg (geometry, layouts, behavior knobs), Fmt (per-segment cache format), Seg
(runtime pointers/bounds). Two-loop (SWA + top-k) or a single segment; 2D and
3D (split-K + reduce) share one kernel. Launchers:
aiter/ops/triton/attention/pa_decode_sparse.py (DSv4 / uniform pool) and
aiter/ops/triton/attention/sparse_mla_decode.py (separated-rope MLA).
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import PropagateNan
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.common_utils import strip_annotate

# Triton's default max ignores NaN, which on AMD costs a canonicalize per
# operand. Nothing here produces NaN (masked lanes are -inf and the all-masked
# row is guarded), so propagate instead.
_MAX_PROP_NAN: gl.constexpr = gl.constexpr(PropagateNan.ALL)


@gluon.jit
def _max2(a, b):
    return gl.maximum(a, b, propagate_nan=_MAX_PROP_NAN)


@gluon.jit
def _rmax(x, axis):
    return gl.reduce(x, axis, _max2)


@gluon.jit
def _cache_load(ptr, row, col, USE_BUFFER_LOAD: gl.constexpr, mask=None, other=None):
    """Gather rows[i] + col[j]. row is the per-token offset in ptr's element
    units; col a small compile-time arange. Keeping them apart resolves one
    pointer per token on the 64-bit path (the column offset folds into the
    load's immediate) instead of a 64-bit add per element."""
    if USE_BUFFER_LOAD:
        return gl.amd.cdna4.buffer_load(
            ptr=ptr,
            offsets=row.to(gl.int32)[:, None] + col.to(gl.int32)[None, :],
            mask=mask,
            other=other,
            cache=".cg",
        )
    row_ptr = ptr + row.to(gl.int64)
    return gl.load(
        row_ptr[:, None] + col[None, :],
        mask=mask,
        other=other,
        cache_modifier=".cg",
    )


@gluon.jit
def _fp8_to_f32(x_u8, FP8_FNUZ: gl.constexpr):
    # fnuz -> f32 has no native cvt on gfx950 (software unpack); fnuz -> bf16 is
    # cheap and exact (3 mantissa bits), so route fnuz through bf16.
    if FP8_FNUZ:
        return x_u8.to(gl.float8e4b8, bitcast=True).to(gl.bfloat16).to(gl.float32)
    return x_u8.to(gl.float8e4nv, bitcast=True).to(gl.float32)


@gluon.jit
def _fp8_to_bf16(x_u8, FP8_FNUZ: gl.constexpr):
    # Exact: fp8's 3 mantissa bits fit bf16's 8.
    if FP8_FNUZ:
        return x_u8.to(gl.float8e4b8, bitcast=True).to(gl.bfloat16)
    return x_u8.to(gl.float8e4nv, bitcast=True).to(gl.bfloat16)


@gluon.jit
def _scale_load(
    ptr,
    row,
    valid,
    USE_BUFFER_LOAD: gl.constexpr,
    gather_l: gl.constexpr,
    scl_l: gl.constexpr,
    NG: gl.constexpr,
    W_FULL: gl.constexpr,
    MASKED: gl.constexpr,
    OTHER: gl.constexpr,
):
    """Gather the NG per-group scales of each token and broadcast to W_FULL
    columns. Indexing the full row with offs // GROUP would build a
    [BLOCK_K, W_FULL] pointer tensor for NG distinct values, and only
    buffer_load's 32-bit offsets CSE that redundancy away. scl_l is chosen so
    the broadcast lands on gather_l as a register rename (assert_trivial
    proves it).
    Element type follows ptr: u8 E8M0 bytes (OTHER=127 -> 2^0) or f32
    (OTHER=0.0 -> masked lanes dequant to 0)."""
    cols = gl.arange(0, NG, layout=gl.SliceLayout(0, scl_l))
    rows = gl.convert_layout(row, gl.SliceLayout(1, scl_l))
    if MASKED:
        m = gl.convert_layout(valid, gl.SliceLayout(1, scl_l))[:, None]
        if USE_BUFFER_LOAD:
            sc = gl.amd.cdna4.buffer_load(
                ptr=ptr,
                offsets=rows.to(gl.int32)[:, None] + cols[None, :],
                mask=m,
                other=OTHER,
                cache=".cg",
            )
        else:
            sc = gl.load(
                (ptr + rows.to(gl.int64))[:, None] + cols[None, :],
                mask=m,
                other=OTHER,
                cache_modifier=".cg",
            )
    else:
        if USE_BUFFER_LOAD:
            sc = gl.amd.cdna4.buffer_load(
                ptr=ptr,
                offsets=rows.to(gl.int32)[:, None] + cols[None, :],
                cache=".cg",
            )
        else:
            sc = gl.load(
                (ptr + rows.to(gl.int64))[:, None] + cols[None, :],
                cache_modifier=".cg",
            )
    wide = gl.expand_dims(sc, 2).broadcast_to([sc.shape[0], NG, W_FULL // NG])
    return gl.convert_layout(
        wide.reshape([sc.shape[0], W_FULL]), gather_l, assert_trivial=True
    )


@gluon.jit
def _split2(x):
    """Register split along dim 1: [A, B] -> two [A, B//2] in the input's own
    layout. Free only while the column direction is a per-lane register repeat;
    assert_trivial turns anything else into a compile error."""
    layout: gl.constexpr = x.type.layout
    x_r = x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1)
    x0, x1 = gl.split(x_r)
    x0 = gl.convert_layout(x0, layout, assert_trivial=True)
    x1 = gl.convert_layout(x1, layout, assert_trivial=True)
    return x0, x1


@gluon.jit
def _split2_dim0(x):
    """Dim-0 counterpart of _split2, for the row-wide gather layout where the
    per-lane register repeats live on dim 0."""
    layout: gl.constexpr = x.type.layout
    x_r = x.reshape(2, x.shape[0] // 2, x.shape[1]).permute(1, 2, 0)
    x0, x1 = gl.split(x_r)
    x0 = gl.convert_layout(x0, layout, assert_trivial=True)
    x1 = gl.convert_layout(x1, layout, assert_trivial=True)
    return x0, x1


@gluon.jit
def _split_ax(x, AXIS: gl.constexpr):
    if AXIS == 1:
        a, b = _split2(x)
    else:
        a, b = _split2_dim0(x)
    return a, b


# fp8 (e4m3 OCP) x E8M0 -> bf16, one instruction per 2 elements. Emitted as two
# single-output asm calls, not one two-output blob: a single instruction reads
# all its sources before writing dst, so no output needs early clobber. The
# blob form does, and that forced =&v liveness spills a kernel with no headroom.
_DEQ_LO: gl.constexpr = gl.constexpr("v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2")
_DEQ_HI: gl.constexpr = gl.constexpr(
    "v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2 op_sel:[1,0,0]"
)
_DEQ_CONS: gl.constexpr = gl.constexpr("=v,v,v")


@gluon.jit
def _deq_asm(x16, e_u8, W8: gl.constexpr, out_l: gl.constexpr):
    """[BLOCK_K, W8/2] int16 (4 packed fp8) + raw E8M0 byte -> [BLOCK_K, W8] bf16.

    The hardware reads only bits [30:23] of the scale operand, i.e. bits [14:7]
    of its high half, so a 16-bit e << 7 lands the exponent where an f32
    e << 23 would and costs one register instead of two. The scale is a power of
    two, so this stays bit-identical to the unfused convert + multiply."""
    sc16 = e_u8.to(gl.uint16) << 7
    lo = gl.inline_asm_elementwise(
        _DEQ_LO, _DEQ_CONS, [x16, sc16], dtype=gl.bfloat16, is_pure=True, pack=2
    )
    hi = gl.inline_asm_elementwise(
        _DEQ_HI, _DEQ_CONS, [x16, sc16], dtype=gl.bfloat16, is_pure=True, pack=2
    )
    # Flat order is 4i + 2*lohi + s; a lane's int16s are the same 16-byte run
    # the gather used, so the interleave stays lane-local (assert_trivial).
    W16: gl.constexpr = W8 // 2
    lo3 = lo.reshape(lo.shape[0], W16 // 2, 2)
    hi3 = hi.reshape(hi.shape[0], W16 // 2, 2)
    both = gl.join(lo3, hi3).permute(0, 1, 3, 2).reshape(lo.shape[0], W8)
    # Back to the gather layout, or the kv_smem store lowers to narrow ds_writes.
    return gl.convert_layout(both, out_l, assert_trivial=True)


# ---------------------------------------------------------------------------
# Aggregates: Cfg (shared compile-time), Fmt (per-segment compile-time),
# Seg (per-segment runtime). Built once at the kernel entry.
# ---------------------------------------------------------------------------


@aggregate
@strip_annotate
class Cfg:
    """Compile-time geometry, layouts, and behavior knobs shared by both segments."""

    # geometry
    BLOCK_M: gl.constexpr
    BLOCK_K: gl.constexpr
    KV_DIM: gl.constexpr        # plane-0 width = LDS tile width = V width (pow-2)
    ROPE_DIM: gl.constexpr      # plane-1 width when ROPE_SEPARATE, else the bf16
                                # tail width inside plane 0 (DSv4 packed)
    ROPE_SEPARATE: gl.constexpr # False: rope inside plane 0. True: K-only second
                                # plane; QK contracts over KV_DIM + ROPE_DIM.
    QK_DIM: gl.constexpr        # q row width: KV_DIM (+ ROPE_DIM if separate)
    MFMA_K: gl.constexpr
    NUM_WARPS: gl.constexpr
    GATHER_TW1: gl.constexpr
    LDS_PAD: gl.constexpr
    # behavior knobs
    UNI_TILE: gl.constexpr
    HAS_INVALID: gl.constexpr
    HEAD_ALIGNED: gl.constexpr
    IDX_BUFFER_LOAD: gl.constexpr
    FP8_MFMA: gl.constexpr    # "tensor" only: feed the matrix core the cache's
                              # own fp8 instead of dequantizing to bf16
    # operator layouts
    qk_layout: gl.constexpr
    pv_layout: gl.constexpr
    q_layout: gl.constexpr
    k_layout: gl.constexpr
    p_layout: gl.constexpr
    v_layout: gl.constexpr
    # memory layouts
    gather_l: gl.constexpr
    gather_rope_l: gl.constexpr
    gather16_l: gl.constexpr
    slot_l: gl.constexpr
    blocked_q: gl.constexpr
    kv_shared: gl.constexpr
    rope_shared: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        BLOCK_M,
        BLOCK_K,
        KV_DIM,
        ROPE_DIM,
        ROPE_SEPARATE,
        MFMA_K,
        NUM_WARPS,
        GATHER_TW1,
        LDS_PAD,
        UNI_TILE,
        HAS_INVALID,
        HEAD_ALIGNED,
        IDX_BUFFER_LOAD,
        FP8_MFMA=False,
    ):
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.KV_DIM = gl.constexpr(KV_DIM)
        self.ROPE_DIM = gl.constexpr(ROPE_DIM)
        self.ROPE_SEPARATE = gl.constexpr(ROPE_SEPARATE)
        self.QK_DIM = gl.constexpr(KV_DIM + (ROPE_DIM if ROPE_SEPARATE else 0))
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.GATHER_TW1 = gl.constexpr(GATHER_TW1)
        self.LDS_PAD = gl.constexpr(LDS_PAD)
        self.UNI_TILE = gl.constexpr(UNI_TILE)
        self.HAS_INVALID = gl.constexpr(HAS_INVALID)
        self.HEAD_ALIGNED = gl.constexpr(HEAD_ALIGNED)
        self.IDX_BUFFER_LOAD = gl.constexpr(IDX_BUFFER_LOAD)
        self.FP8_MFMA = gl.constexpr(FP8_MFMA)
        # K=32 is the only shape the backend offers for plain fp8 operands, and
        # it divides KV_DIM, ROPE_DIM and BLOCK_K alike, so one layout still
        # covers both dots.
        if FP8_MFMA:
            MFMA_K = 32
        self.MFMA_K = gl.constexpr(MFMA_K)

        self.qk_layout = gl.constexpr(
            gl.amd.AMDMFMALayout(
                version=4,
                instr_shape=[16, 16, MFMA_K],
                transposed=True,
                warps_per_cta=[1, NUM_WARPS],
            )
        )
        self.pv_layout = gl.constexpr(
            gl.amd.AMDMFMALayout(
                version=4,
                instr_shape=[16, 16, MFMA_K],
                transposed=True,
                warps_per_cta=[1, NUM_WARPS],
            )
        )
        KW = MFMA_K // 2
        self.q_layout = gl.constexpr(gl.DotOperandLayout(0, self.qk_layout, KW))
        self.k_layout = gl.constexpr(gl.DotOperandLayout(1, self.qk_layout, KW))
        self.p_layout = gl.constexpr(gl.DotOperandLayout(0, self.pv_layout, KW))
        self.v_layout = gl.constexpr(gl.DotOperandLayout(1, self.pv_layout, KW))

        # 16 uint8 per thread = 128-bit gather loads. Warps tile dim 0 so the
        # per-lane slot vector stays short and the column direction keeps
        # register repeats (what the chunked dequant splits for free).
        # GATHER_TW1 = threads spent on the head dim: 32 requests a whole
        # 512 B token row per instruction, at the cost of a longer slot vector.
        GSPT = 16
        self.gather_l = gl.constexpr(
            gl.BlockedLayout(
                size_per_thread=[1, GSPT],
                threads_per_warp=[64 // GATHER_TW1, GATHER_TW1],
                warps_per_cta=[NUM_WARPS, 1],
                order=[1, 0],
            )
        )
        # One warp already covers all rope columns; warps on dim 1 would
        # re-gather the same tile NUM_WARPS times.
        self.gather_rope_l = gl.constexpr(
            gl.BlockedLayout(
                size_per_thread=[1, 8],
                threads_per_warp=[8, 8],
                warps_per_cta=[NUM_WARPS, 1],
                order=[1, 0],
            )
        )
        # gather_l with 2-byte elements: same 16-byte per-lane run, half the
        # columns (for the fused-dequant int16 view).
        self.gather16_l = gl.constexpr(
            gl.BlockedLayout(
                size_per_thread=[1, GSPT // 2],
                threads_per_warp=[64 // GATHER_TW1, GATHER_TW1],
                warps_per_cta=[NUM_WARPS, 1],
                order=[1, 0],
            )
        )
        self.slot_l = gl.constexpr(gl.SliceLayout(1, self.gather_l.value))
        self.blocked_q = gl.constexpr(
            gl.BlockedLayout(
                size_per_thread=[1, 8],
                threads_per_warp=[8, 8],
                warps_per_cta=[1, NUM_WARPS],
                order=[1, 0],
            )
        )
        # Row pitch (KV_DIM + LDS_PAD) decides which banks the transposed K
        # read (walks down a column) lands on.
        self.kv_shared = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for(
                [[KV_DIM, LDS_PAD]], [BLOCK_K, KV_DIM], [1, 0]
            )
        )
        # Plane 1 (K-only rope) when ROPE_SEPARATE; dead constexpr otherwise.
        self.rope_shared = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for(
                [[ROPE_DIM, LDS_PAD]], [BLOCK_K, ROPE_DIM], [1, 0]
            )
        )


@aggregate
@strip_annotate
class Fmt:
    """Compile-time description of one segment's cache format."""

    KIND: gl.constexpr            # "bf16" | "dsv4" | "uniform" | "tensor" | "dsmla"
    IS_FP8: gl.constexpr          # pipeline select: prefetched fp8 loop vs bf16 loop
    BLOCK_SIZE: gl.constexpr
    USE_BUFFER_LOAD: gl.constexpr
    FP8_FNUZ: gl.constexpr
    ASM_DEQ: gl.constexpr
    NOPE_CHUNK: gl.constexpr
    CHUNK_AXIS: gl.constexpr
    NOPE_DIM: gl.constexpr        # fp8 payload width (448 dsv4; KV_DIM elsewhere)
    GROUP: gl.constexpr           # scale group width (64 dsv4/uniform, 128 dsmla)
    NG: gl.constexpr              # scale groups per row (KV_DIM // GROUP)
    NARROW_SCALE: gl.constexpr
    scl_l: gl.constexpr
    # packed-row addressing constants (element units named in the suffix)
    TOK_U8: gl.constexpr          # bytes per token row inside a block (576 / 656)
    TOK_U16: gl.constexpr         # the same row in bf16-view units
    ROPE_U16_OFF: gl.constexpr    # bf16-view offset of the rope tail in a row
    SCL_TRAILER_U8: gl.constexpr  # dsv4: scale bytes per token in the block trailer
    TOK_F32: gl.constexpr         # dsmla: f32-view stride per token row
    SCL_F32_OFF: gl.constexpr     # dsmla: f32-view offset of the group scales
    TOK_EL: gl.constexpr          # flat formats: cache elements per token row

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        KIND,
        BLOCK_SIZE,
        USE_BUFFER_LOAD,
        FP8_FNUZ,
        ASM_DEQ,
        NOPE_DIM,
        NOPE_CHUNK,
        CHUNK_AXIS,
    ):
        self.KIND = gl.constexpr(KIND)
        self.IS_FP8 = gl.constexpr(KIND != "bf16")
        self.BLOCK_SIZE = gl.constexpr(BLOCK_SIZE)
        self.USE_BUFFER_LOAD = gl.constexpr(USE_BUFFER_LOAD)
        self.FP8_FNUZ = gl.constexpr(FP8_FNUZ)
        self.ASM_DEQ = gl.constexpr(ASM_DEQ)
        self.NOPE_CHUNK = gl.constexpr(NOPE_CHUNK)
        self.CHUNK_AXIS = gl.constexpr(CHUNK_AXIS)
        self.NOPE_DIM = gl.constexpr(NOPE_DIM)

        KV_DIM = cfg.KV_DIM.value
        ROPE_DIM = cfg.ROPE_DIM.value
        GROUP = 128 if KIND == "dsmla" else 64
        NG = KV_DIM // GROUP
        self.GROUP = gl.constexpr(GROUP)
        self.NG = gl.constexpr(NG)
        # 3-D companion of gather_l for _scale_load: dim 1 carries the NG scale
        # groups, dim 2 the columns inside a group, so the [BLOCK_K, NG, GROUP]
        # -> 2-D reshape reproduces gather_l exactly. Legal only when a group's
        # columns fill whole threads.
        GSPT = 16
        TW1 = cfg.GATHER_TW1.value
        NARROW_SCALE = TW1 % NG == 0 and GSPT * (TW1 // NG) == GROUP
        self.NARROW_SCALE = gl.constexpr(NARROW_SCALE)
        scl_l3 = gl.BlockedLayout(
            size_per_thread=[1, 1, GSPT],
            threads_per_warp=[
                64 // TW1,
                NG if NARROW_SCALE else 1,
                (TW1 // NG) if NARROW_SCALE else TW1,
            ],
            warps_per_cta=[cfg.NUM_WARPS.value, 1, 1],
            order=[2, 1, 0],
        )
        self.scl_l = gl.constexpr(gl.SliceLayout(2, scl_l3))

        # dsv4 row: [NOPE_DIM fp8 | ROPE_DIM bf16] + 8 B UE8M0 per token after
        # the block; dsmla row: [KV_DIM fp8 | NG f32 | ROPE_DIM bf16] inline.
        if KIND == "dsv4":
            TOK_U8 = NOPE_DIM + 2 * ROPE_DIM            # 448 + 128 = 576
        elif KIND == "dsmla":
            TOK_U8 = KV_DIM + 4 * NG + 2 * ROPE_DIM     # 512 + 16 + 128 = 656
        else:
            TOK_U8 = 0
        self.TOK_U8 = gl.constexpr(TOK_U8)
        self.TOK_U16 = gl.constexpr(TOK_U8 // 2)
        self.ROPE_U16_OFF = gl.constexpr(
            (NOPE_DIM // 2) if KIND == "dsv4" else ((KV_DIM + 4 * NG) // 2)
        )
        self.SCL_TRAILER_U8 = gl.constexpr(8)
        self.TOK_F32 = gl.constexpr(TOK_U8 // 4)
        self.SCL_F32_OFF = gl.constexpr(KV_DIM // 4)
        # Flat formats gather in cache elements: KV_DIM wide, plus the appended
        # rope when the geometry separates it.
        self.TOK_EL = gl.constexpr(
            KV_DIM + (ROPE_DIM if cfg.ROPE_SEPARATE.value and KIND != "uniform" else 0)
        )


@aggregate
@strip_annotate
class Seg:
    """Runtime context of one segment. Pointer roles by format ("--" = unused
    duplicate):

        KIND      cache_ptr        alt_ptr                  scl_ptr
        bf16      --               bf16 cache               --
        dsv4      u8 cache         bf16 view of the cache   --
        uniform   u8 cache         f32 per-64 kv_scales     --
        tensor    u8 cache         --                       f32 scalar k_scale
        dsmla     u8 cache         bf16 view (rope tail)    f32 view (scales)
    """

    fmt: Fmt
    cache_ptr: gl.tensor
    alt_ptr: gl.tensor
    scl_ptr: gl.tensor
    indices_ptr: gl.tensor
    seg_start: gl.tensor
    cs0: gl.tensor
    num_rows: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self, fmt, cache_ptr, alt_ptr, scl_ptr, indices_ptr, seg_start, cs0, num_rows
    ):
        self.fmt = fmt
        self.cache_ptr = cache_ptr
        self.alt_ptr = alt_ptr
        self.scl_ptr = scl_ptr
        self.indices_ptr = indices_ptr
        self.seg_start = seg_start
        self.cs0 = cs0
        self.num_rows = num_rows


# ---------------------------------------------------------------------------
# Dequant-and-store into the LDS plane(s)
# ---------------------------------------------------------------------------


@gluon.jit
def _deq_store(x_u8, sc, kv_smem, off, cfg, fmt, AXIS: gl.constexpr):
    """Dequant one fp8 slab into kv_smem[:, off:off+W]. Dequant stays in f32
    (gfx950 has no bf16 multiply). sc: raw UE8M0 byte (dsv4), f32 scale
    (uniform/dsmla), or unused ("tensor": bare fp8 -> bf16 convert)."""
    if fmt.ASM_DEQ:
        # x_u8 is the int16 view here (see _gather_full), so its column count is W8/2.
        W8: gl.constexpr = x_u8.shape[1] * 2
        # Adjacent fp8 columns share a scale (groups are even), so dropping
        # every other broadcast column is exact.
        s_even, _ = gl.split(sc.reshape(sc.shape[0], sc.shape[1] // 2, 2))
        s_even = gl.convert_layout(s_even, x_u8.type.layout)
        val = _deq_asm(x_u8.to(gl.int16, bitcast=True), s_even, W8, cfg.gather_l)
        if AXIS == 1:
            kv_smem.slice(off, W8, dim=1).store(val)
        else:
            kv_smem.slice(off, x_u8.shape[0], dim=0).store(val)
    else:
        if fmt.KIND == "tensor":
            val = _fp8_to_bf16(x_u8, fmt.FP8_FNUZ)
        else:
            if fmt.KIND == "uniform" or fmt.KIND == "dsmla":
                scale = sc
            else:
                scale = gl.exp2(sc.to(gl.float32) - 127.0)
            val = (_fp8_to_f32(x_u8, fmt.FP8_FNUZ) * scale).to(gl.bfloat16)
        if AXIS == 1:
            kv_smem.slice(off, x_u8.shape[1], dim=1).store(val)
        else:
            kv_smem.slice(off, x_u8.shape[0], dim=0).store(val)


@gluon.jit
def _deq_store_tile(x_u8, sc, kv_smem, cfg, fmt):
    """Dequant a gathered fp8 tile into kv_smem in NOPE_CHUNK-sized pieces
    along CHUNK_AXIS (0 = rows, 1 = columns). The f32 expansion is 4x the fp8
    tile; chunking keeps only 1/pieces of it live (each piece's converts feed
    its ds_writes and die). The splits are register renames (see _split2)."""
    AXIS: gl.constexpr = fmt.CHUNK_AXIS
    NOPE_CHUNK: gl.constexpr = fmt.NOPE_CHUNK
    W: gl.constexpr = x_u8.shape[1] if AXIS == 1 else x_u8.shape[0]
    if NOPE_CHUNK >= W:
        _deq_store(x_u8, sc, kv_smem, 0, cfg, fmt, AXIS)
    else:
        x0, x1 = _split_ax(x_u8, AXIS)
        s0, s1 = _split_ax(sc, AXIS)
        W2: gl.constexpr = W // 2
        if NOPE_CHUNK >= W2:
            _deq_store(x0, s0, kv_smem, 0, cfg, fmt, AXIS)
            _deq_store(x1, s1, kv_smem, W2, cfg, fmt, AXIS)
        else:
            x00, x01 = _split_ax(x0, AXIS)
            s00, s01 = _split_ax(s0, AXIS)
            x10, x11 = _split_ax(x1, AXIS)
            s10, s11 = _split_ax(s1, AXIS)
            W4: gl.constexpr = W // 4
            if NOPE_CHUNK >= W4:
                _deq_store(x00, s00, kv_smem, 0, cfg, fmt, AXIS)
                _deq_store(x01, s01, kv_smem, W4, cfg, fmt, AXIS)
                _deq_store(x10, s10, kv_smem, 2 * W4, cfg, fmt, AXIS)
                _deq_store(x11, s11, kv_smem, 3 * W4, cfg, fmt, AXIS)
            else:
                W8: gl.constexpr = W // 8
                y0, y1 = _split_ax(x00, AXIS)
                t0, t1 = _split_ax(s00, AXIS)
                _deq_store(y0, t0, kv_smem, 0, cfg, fmt, AXIS)
                _deq_store(y1, t1, kv_smem, W8, cfg, fmt, AXIS)
                y0, y1 = _split_ax(x01, AXIS)
                t0, t1 = _split_ax(s01, AXIS)
                _deq_store(y0, t0, kv_smem, 2 * W8, cfg, fmt, AXIS)
                _deq_store(y1, t1, kv_smem, 3 * W8, cfg, fmt, AXIS)
                y0, y1 = _split_ax(x10, AXIS)
                t0, t1 = _split_ax(s10, AXIS)
                _deq_store(y0, t0, kv_smem, 4 * W8, cfg, fmt, AXIS)
                _deq_store(y1, t1, kv_smem, 5 * W8, cfg, fmt, AXIS)
                y0, y1 = _split_ax(x11, AXIS)
                t0, t1 = _split_ax(s11, AXIS)
                _deq_store(y0, t0, kv_smem, 6 * W8, cfg, fmt, AXIS)
                _deq_store(y1, t1, kv_smem, 7 * W8, cfg, fmt, AXIS)


@gluon.jit
def _slots(
    cfg,
    seg,
    k_pos,
    hi,
    num_rows,
    MASKED: gl.constexpr,
    UNI_TILE: gl.constexpr = False,
):
    """Index-list read -> (block, pos, valid), in whatever layout k_pos carries.
    A masked gl.load predicates on exec while a masked buffer_load folds the
    mask into the offset, so the unmasked paths clamp the read in-range and
    mask the score instead (UNI_TILE); -1 sentinels are handled the same way."""
    indices_ptr = seg.indices_ptr
    seg_start = seg.seg_start
    BLOCK_SIZE: gl.constexpr = seg.fmt.BLOCK_SIZE
    HAS_INVALID: gl.constexpr = cfg.HAS_INVALID
    IDX_BUFFER_LOAD: gl.constexpr = cfg.IDX_BUFFER_LOAD
    if MASKED:
        in_range = k_pos < hi
        if IDX_BUFFER_LOAD:
            slot = gl.amd.cdna4.buffer_load(
                ptr=indices_ptr + seg_start, offsets=k_pos, mask=in_range, other=-1
            )
        else:
            slot = gl.load(indices_ptr + seg_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < num_rows)
        slot = gl.where(valid, slot, 0)
    else:
        # hi >= 1 whenever UNI_TILE runs (guarded by n_full > 0).
        off = gl.minimum(k_pos, hi - 1) if UNI_TILE else k_pos
        if IDX_BUFFER_LOAD:
            slot = gl.amd.cdna4.buffer_load(ptr=indices_ptr + seg_start, offsets=off)
        else:
            slot = gl.load(indices_ptr + seg_start + off)
        valid = (k_pos < hi) if UNI_TILE else (slot >= 0)
        if HAS_INVALID:
            if UNI_TILE:
                valid = valid & (slot >= 0)
            slot = gl.where(valid, slot, 0)  # -1 sentinels: clamp, mask score below
    return (slot // BLOCK_SIZE).to(gl.int32), (slot % BLOCK_SIZE).to(gl.int32), valid


@gluon.jit
def _qk_scores(cfg, q_dot, q_rope_dot, kv_smem, rope_smem):
    """QK scores for one tile; ROPE_SEPARATE chains a second MFMA over the rope
    plane (MFMA accumulates natively, so KV_DIM + ROPE_DIM is two dots)."""
    k = kv_smem.permute([1, 0]).load(cfg.k_layout)  # [KV_DIM, BLOCK_K]
    S = gl.amd.cdna4.mfma(
        q_dot,
        k,
        gl.zeros([cfg.BLOCK_M, cfg.BLOCK_K], gl.float32, layout=cfg.qk_layout),
    )
    if cfg.ROPE_SEPARATE:
        k_rope = rope_smem.permute([1, 0]).load(cfg.k_layout)  # [ROPE_DIM, BLOCK_K]
        S = gl.amd.cdna4.mfma(q_rope_dot, k_rope, S)
    return S


# ---------------------------------------------------------------------------
# Prefetched fp8 pipeline: _gather_full issues tile N+1's loads while _qkpv
# stages and dots tile N.
# ---------------------------------------------------------------------------


@gluon.jit
def _gather_full(
    cfg,
    seg,
    k_start,
    seg_hi,
    offs_full,
    offs_full16,
    offs_rope,
    k_rng_slot,
    k_rng_rope,
):
    """Gather one full fp8 tile, split from the LDS-write/MFMA so it issues an
    iteration early. The prefetch stays in raw fp8: dequantizing here would
    double the loop-carried registers, so the consumer dequants in chunks.
    Returns (x, sc, k_rope, valid); unused slots carry a duplicate DCE removes."""
    fmt = seg.fmt
    cs0 = seg.cs0
    if not fmt.USE_BUFFER_LOAD:
        cs0 = cs0.to(gl.int64)  # >2 GB cache: 64-bit gather offsets
    bg, pg, valid = _slots(
        cfg,
        seg,
        k_start + k_rng_slot,
        seg_hi,  # hi: unused unless UNI_TILE
        0,
        False,
        cfg.UNI_TILE,
    )
    if fmt.KIND == "uniform":
        NGRP: gl.constexpr = cfg.KV_DIM // 64
        x_u8 = _cache_load(
            seg.cache_ptr, bg * cs0 + pg * cfg.KV_DIM, offs_full, fmt.USE_BUFFER_LOAD
        )
        sc = _cache_load(
            seg.alt_ptr, bg * NGRP, offs_full // 64, fmt.USE_BUFFER_LOAD
        )
        k_rope = x_u8  # no rope side-channel -> DCE'd
    elif fmt.KIND == "tensor":
        # Per-tensor scale is folded outside the loop (qk_scale / p), so this
        # is a bare gather; the K-only rope tail follows when separated.
        x_u8 = _cache_load(
            seg.cache_ptr, bg * cs0 + pg * fmt.TOK_EL, offs_full, fmt.USE_BUFFER_LOAD
        )
        sc = x_u8  # no scale vector -> DCE'd
        if cfg.ROPE_SEPARATE:
            bgr, pgr, _ = _slots(
                cfg,
                seg,
                k_start + k_rng_rope,
                seg_hi,
                0,
                False,
                cfg.UNI_TILE,
            )
            k_rope = _cache_load(
                seg.cache_ptr,
                bgr * cs0 + pgr * fmt.TOK_EL + cfg.KV_DIM,
                offs_rope,
                fmt.USE_BUFFER_LOAD,
            )
        else:
            k_rope = x_u8  # rope lives inside plane 0 -> DCE'd
    elif fmt.KIND == "dsmla":
        nope_row = bg * cs0 + pg * fmt.TOK_U8
        scl_row = bg * (cs0 // 4) + pg * fmt.TOK_F32 + fmt.SCL_F32_OFF
        # Scales issue before the bulk fp8: vmcnt is one in-order FIFO, so
        # issued last they would stall the first dequant piece behind every
        # data load as well.
        if fmt.NARROW_SCALE and not fmt.USE_BUFFER_LOAD:
            sc = _scale_load(
                seg.scl_ptr, scl_row, scl_row, fmt.USE_BUFFER_LOAD, cfg.gather_l,
                fmt.scl_l, fmt.NG, cfg.KV_DIM, False, 0.0,
            )
        else:
            sc = _cache_load(
                seg.scl_ptr, scl_row, offs_full // fmt.GROUP, fmt.USE_BUFFER_LOAD
            )
        x_u8 = _cache_load(seg.cache_ptr, nope_row, offs_full, fmt.USE_BUFFER_LOAD)
        bgr, pgr, _ = _slots(
            cfg,
            seg,
            k_start + k_rng_rope,
            seg_hi,
            0,
            False,
            cfg.UNI_TILE,
        )
        k_rope = _cache_load(
            seg.alt_ptr,
            bgr * (cs0 // 2) + pgr * fmt.TOK_U16 + fmt.ROPE_U16_OFF,
            offs_rope,
            fmt.USE_BUFFER_LOAD,
        )
    else:  # "dsv4"
        nope_row = bg * cs0 + pg * fmt.TOK_U8
        scl_row = bg * cs0 + fmt.BLOCK_SIZE * fmt.TOK_U8 + pg * fmt.SCL_TRAILER_U8
        # Scales first (vmcnt FIFO; see the dsmla branch).
        if fmt.NARROW_SCALE and not fmt.USE_BUFFER_LOAD:
            sc = _scale_load(
                seg.cache_ptr, scl_row, scl_row, fmt.USE_BUFFER_LOAD, cfg.gather_l,
                fmt.scl_l, fmt.NG, cfg.KV_DIM, False, 127,
            )
        else:
            sc = _cache_load(
                seg.cache_ptr, scl_row, offs_full // 64, fmt.USE_BUFFER_LOAD
            )
        if fmt.ASM_DEQ:
            # 2-byte elements: <2 x i16> = 4 packed fp8 per VGPR out of one
            # dword load. Same byte as nope_row (both even), addressed through
            # the bf16 view; the layout convert is a rename (shared dim-0 tiling).
            row16 = gl.convert_layout(
                nope_row >> 1, gl.SliceLayout(1, cfg.gather16_l)
            )
            x_u8 = _cache_load(
                seg.alt_ptr, row16, offs_full16, fmt.USE_BUFFER_LOAD
            )
        else:
            x_u8 = _cache_load(seg.cache_ptr, nope_row, offs_full, fmt.USE_BUFFER_LOAD)
        bgr, pgr, _ = _slots(
            cfg,
            seg,
            k_start + k_rng_rope,
            seg_hi,
            0,
            False,
            cfg.UNI_TILE,
        )
        k_rope = _cache_load(
            seg.alt_ptr,
            bgr * (cs0 // 2) + pgr * fmt.TOK_U16 + fmt.ROPE_U16_OFF,
            offs_rope,
            fmt.USE_BUFFER_LOAD,
        )
    return x_u8, sc, k_rope, valid


@gluon.jit
def _stage(cfg, seg, x_u8, sc, k_rope, kv_smem, rope_smem):
    """Write one prefetched tile into the LDS plane(s). Plane 0 is always the
    full KV_DIM-wide dequant: "dsv4" gathers KV_DIM bytes too, the last 64 being
    bf16 rope read as garbage fp8 and overwritten by the slice-store below, so
    the gather stays pow-2 wide."""
    fmt = seg.fmt
    if cfg.FP8_MFMA:
        # No dequant: the scale is folded outside the loop (qk_scale on the K
        # side, the accumulator on the V side), so what lands in LDS is exactly
        # what the gather returned.
        kv_smem.store(x_u8.to(gl.float8e4nv, bitcast=True))
        if cfg.ROPE_SEPARATE:
            rope_smem.store(k_rope.to(gl.float8e4nv, bitcast=True))
    else:
        _deq_store_tile(x_u8, sc, kv_smem, cfg, fmt)
        if fmt.KIND == "dsv4":
            kv_smem.slice(fmt.NOPE_DIM, cfg.ROPE_DIM, dim=1).store(k_rope)
        elif fmt.KIND == "dsmla":
            rope_smem.store(k_rope)
        elif fmt.KIND == "tensor":
            if cfg.ROPE_SEPARATE:
                rope_smem.store(_fp8_to_bf16(k_rope, fmt.FP8_FNUZ))
        # "uniform": the whole head is one fp8 tile; nothing else to store.


@gluon.jit
def _qkpv(
    cfg,
    seg,
    x_u8,
    sc,
    k_rope,
    valid,
    q_dot,
    q_rope_dot,
    m_i,
    l_i,
    acc,
    head_mask,
    qk_scale,
    v_scale,
    kv_smem,
    rope_smem,
    k_start=0,
    seg_hi=0,
):
    """Stage a prefetched fp8 tile into LDS, then QK -> softmax -> PV."""
    neg_inf = float("-inf")
    _stage(cfg, seg, x_u8, sc, k_rope, kv_smem, rope_smem)
    S = _qk_scores(cfg, q_dot, q_rope_dot, kv_smem, rope_smem)
    # UNI_TILE folds the tile's range test into the score mask, which is what
    # makes the partial last tile correct without a peeled masked body.
    COL_MASK: gl.constexpr = cfg.HAS_INVALID or cfg.UNI_TILE
    NEED_MASK: gl.constexpr = COL_MASK or (not cfg.HEAD_ALIGNED)
    if NEED_MASK:
        if COL_MASK:
            if cfg.UNI_TILE and not cfg.HAS_INVALID:
                # Build the range mask directly in the MFMA layout; converting
                # the slot-layout valid vector costs a cross-lane convert per
                # tile.
                col_mask = (
                    k_start
                    + gl.arange(0, cfg.BLOCK_K, layout=gl.SliceLayout(0, cfg.qk_layout))
                    < seg_hi
                )[None, :]
            else:
                col_mask = gl.convert_layout(
                    valid, gl.SliceLayout(0, cfg.qk_layout)
                )[None, :]
            if not cfg.HEAD_ALIGNED:
                col_mask = (
                    gl.convert_layout(head_mask, gl.SliceLayout(1, cfg.qk_layout))[
                        :, None
                    ]
                    & col_mask
                )
        else:
            col_mask = gl.convert_layout(head_mask, gl.SliceLayout(1, cfg.qk_layout))[
                :, None
            ]
        S = gl.where(col_mask, S, neg_inf)
    # Online softmax in the base-2 exponent domain; m_i carries qk_scale
    # (= scale * log2e [* k_scale for "tensor"]). max commutes with a positive
    # scale, so scale the row max instead of every element of S; what is left,
    # S * qk_scale - m_new, lowers to one FMA, and -inf columns stay -inf.
    m_block = _rmax(S, 1) * qk_scale
    m_new = _max2(m_i, m_block)
    m_new = gl.where(m_new > neg_inf, m_new, 0.0)
    p = gl.exp2(S * qk_scale - m_new[:, None])
    alpha = gl.exp2(m_i - m_new)
    l_new = l_i * alpha + gl.sum(p, axis=1)
    v = kv_smem.load(cfg.v_layout)
    # "tensor": V was staged as raw fp8 code points; apply the per-tensor scale
    # on the small side (p) and leave l scale-free: out = sum(p*s*V)/l exactly.
    if seg.fmt.KIND == "tensor" and not cfg.FP8_MFMA:
        p = p * v_scale
    if cfg.FP8_MFMA:
        # p goes to e4m3 as it is. Pre-scaling it by a power of two to lift the
        # small end out of e4m3's subnormals is worth nothing: the error is set
        # by the p values near 1, and 1 / 16 / 256 agree to three digits.
        p_dot = gl.convert_layout(p.to(gl.float8e4nv), cfg.p_layout)
    else:
        p_dot = gl.convert_layout(p.to(gl.bfloat16), cfg.p_layout)
    alpha_pv = gl.convert_layout(alpha, gl.SliceLayout(1, cfg.pv_layout))
    acc = acc * alpha_pv[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v, acc)
    return m_new, l_new, acc


# ---------------------------------------------------------------------------
# Non-prefetched path: bf16 segments, and the peeled masked tail when UNI_TILE
# is off (dsv4/uniform only; tensor/dsmla require UNI_TILE).
# ---------------------------------------------------------------------------


@gluon.jit
def _decode_tile(
    cfg,
    seg,
    q_dot,
    q_rope_dot,
    k_start,
    hi,
    m_i,
    l_i,
    acc,
    head_mask,
    qk_scale,
    v_scale,
    kv_smem,
    rope_smem,
    offs_full,
    offs_full16,
    offs_rope,
    k_rng_slot,
    k_rng_rope,
    MASKED: gl.constexpr,
):
    """One KV tile -> online-softmax update. MASKED=True is the peeled tail
    (fully predicated); full tiles clamp -1 sentinels and mask scores when
    HAS_INVALID."""
    neg_inf = float("-inf")
    fmt = seg.fmt
    cs0 = seg.cs0
    if not fmt.USE_BUFFER_LOAD:
        cs0 = cs0.to(gl.int64)  # >2 GB cache: 64-bit gather offsets
    block_idx, pos, valid1d = _slots(
        cfg,
        seg,
        k_start + k_rng_slot,
        hi,
        seg.num_rows,
        MASKED,
    )
    block_idx_g = gl.convert_layout(block_idx, gl.SliceLayout(1, cfg.gather_l))
    pos_g = gl.convert_layout(pos, gl.SliceLayout(1, cfg.gather_l))
    if MASKED:
        valid_g = gl.convert_layout(valid1d, gl.SliceLayout(1, cfg.gather_l))

    if fmt.KIND == "uniform":
        NGRP: gl.constexpr = cfg.KV_DIM // 64
        kv_row = block_idx_g * cs0 + pos_g * cfg.KV_DIM
        scl_row = block_idx_g * NGRP
        scl_col = offs_full // 64
        if MASKED:
            x_u8 = _cache_load(
                seg.cache_ptr, kv_row, offs_full, fmt.USE_BUFFER_LOAD,
                mask=valid_g[:, None], other=0,
            )
            sc = _cache_load(
                seg.alt_ptr,
                scl_row,
                scl_col,
                fmt.USE_BUFFER_LOAD,
                mask=valid_g[:, None],
                other=0.0,
            )
        else:
            x_u8 = _cache_load(seg.cache_ptr, kv_row, offs_full, fmt.USE_BUFFER_LOAD)
            sc = _cache_load(seg.alt_ptr, scl_row, scl_col, fmt.USE_BUFFER_LOAD)
        _deq_store_tile(x_u8, sc, kv_smem, cfg, fmt)
    elif fmt.KIND == "dsv4":
        nope_row = block_idx_g * cs0 + pos_g * fmt.TOK_U8
        scl_row = (
            block_idx_g * cs0 + fmt.BLOCK_SIZE * fmt.TOK_U8 + pos_g * fmt.SCL_TRAILER_U8
        )
        scl_col = offs_full // 64
        if MASKED:  # scales first: see _gather_full
            if fmt.NARROW_SCALE and not fmt.USE_BUFFER_LOAD:
                exps = _scale_load(
                    seg.cache_ptr, scl_row, valid1d, fmt.USE_BUFFER_LOAD,
                    cfg.gather_l, fmt.scl_l, fmt.NG, cfg.KV_DIM, True, 127,
                )
            else:
                exps = _cache_load(
                    seg.cache_ptr, scl_row, scl_col, fmt.USE_BUFFER_LOAD,
                    mask=valid_g[:, None], other=127,
                )
            if fmt.ASM_DEQ:
                row16 = gl.convert_layout(
                    nope_row >> 1, gl.SliceLayout(1, cfg.gather16_l)
                )
                x_u8 = _cache_load(
                    seg.alt_ptr, row16, offs_full16, fmt.USE_BUFFER_LOAD,
                    mask=gl.convert_layout(
                        valid1d, gl.SliceLayout(1, cfg.gather16_l)
                    )[:, None],
                    other=0.0,
                )
            else:
                x_u8 = _cache_load(
                    seg.cache_ptr, nope_row, offs_full, fmt.USE_BUFFER_LOAD,
                    mask=valid_g[:, None], other=0,
                )
        else:
            if fmt.NARROW_SCALE and not fmt.USE_BUFFER_LOAD:
                exps = _scale_load(
                    seg.cache_ptr, scl_row, scl_row, fmt.USE_BUFFER_LOAD,
                    cfg.gather_l, fmt.scl_l, fmt.NG, cfg.KV_DIM, False, 127,
                )
            else:
                exps = _cache_load(seg.cache_ptr, scl_row, scl_col, fmt.USE_BUFFER_LOAD)
            if fmt.ASM_DEQ:
                row16 = gl.convert_layout(
                    nope_row >> 1, gl.SliceLayout(1, cfg.gather16_l)
                )
                x_u8 = _cache_load(
                    seg.alt_ptr, row16, offs_full16, fmt.USE_BUFFER_LOAD
                )
            else:
                x_u8 = _cache_load(
                    seg.cache_ptr, nope_row, offs_full, fmt.USE_BUFFER_LOAD
                )
        _deq_store_tile(x_u8, exps, kv_smem, cfg, fmt)
        block_idx_gr, pos_gr, valid_gr = _slots(
            cfg,
            seg,
            k_start + k_rng_rope,
            hi,
            seg.num_rows,
            MASKED,
        )
        rope_row = block_idx_gr * (cs0 // 2) + pos_gr * fmt.TOK_U16 + fmt.ROPE_U16_OFF
        if MASKED:
            k_rope = _cache_load(
                seg.alt_ptr,
                rope_row,
                offs_rope,
                fmt.USE_BUFFER_LOAD,
                mask=valid_gr[:, None],
                other=0.0,
            )
        else:
            k_rope = _cache_load(seg.alt_ptr, rope_row, offs_rope, fmt.USE_BUFFER_LOAD)
        kv_smem.slice(fmt.NOPE_DIM, cfg.ROPE_DIM, dim=1).store(k_rope)
    else:  # "bf16" (tensor/dsmla require UNI_TILE, so they never come here)
        kv_row2 = block_idx_g * cs0 + pos_g * fmt.TOK_EL
        if MASKED:
            kv = _cache_load(
                seg.alt_ptr, kv_row2, offs_full, fmt.USE_BUFFER_LOAD,
                mask=valid_g[:, None], other=0.0,
            )
        else:
            kv = _cache_load(seg.alt_ptr, kv_row2, offs_full, fmt.USE_BUFFER_LOAD)
        kv_smem.store(kv)
        if cfg.ROPE_SEPARATE:
            block_idx_gr, pos_gr, valid_gr = _slots(
                cfg,
                seg,
                k_start + k_rng_rope,
                hi,
                seg.num_rows,
                MASKED,
            )
            rope_row = block_idx_gr * cs0 + pos_gr * fmt.TOK_EL + cfg.KV_DIM
            if MASKED:
                k_rope = _cache_load(
                    seg.alt_ptr, rope_row, offs_rope, fmt.USE_BUFFER_LOAD,
                    mask=valid_gr[:, None], other=0.0,
                )
            else:
                k_rope = _cache_load(
                    seg.alt_ptr, rope_row, offs_rope, fmt.USE_BUFFER_LOAD
                )
            rope_smem.store(k_rope)

    S = _qk_scores(cfg, q_dot, q_rope_dot, kv_smem, rope_smem)
    COL_VALID: gl.constexpr = MASKED or cfg.HAS_INVALID
    NEED_MASK: gl.constexpr = COL_VALID or (not cfg.HEAD_ALIGNED)
    if NEED_MASK:
        if COL_VALID:
            col_mask = gl.convert_layout(
                valid1d, gl.SliceLayout(0, cfg.qk_layout)
            )[None, :]
            if not cfg.HEAD_ALIGNED:
                col_mask = (
                    gl.convert_layout(head_mask, gl.SliceLayout(1, cfg.qk_layout))[
                        :, None
                    ]
                    & col_mask
                )
        else:
            col_mask = gl.convert_layout(head_mask, gl.SliceLayout(1, cfg.qk_layout))[
                :, None
            ]
        S = gl.where(col_mask, S, neg_inf)

    # exp2 softmax with qk_scale folded in; masked cols (-inf) give exp2 = 0.
    S = S * qk_scale
    m_block = _rmax(S, 1)
    m_new = _max2(m_i, m_block)
    m_new = gl.where(m_new > neg_inf, m_new, 0.0)  # guard all-masked rows
    p = gl.exp2(S - m_new[:, None])
    alpha = gl.exp2(m_i - m_new)
    l_new = l_i * alpha + gl.sum(p, axis=1)

    v = kv_smem.load(cfg.v_layout)  # [BLOCK_K, KV_DIM]
    if seg.fmt.KIND == "tensor":
        p = p * v_scale  # per-tensor V scale on the small side (see _qkpv)
    p_dot = gl.convert_layout(p.to(gl.bfloat16), cfg.p_layout)
    alpha_pv = gl.convert_layout(alpha, gl.SliceLayout(1, cfg.pv_layout))
    acc = acc * alpha_pv[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v, acc)
    return m_new, l_new, acc


@gluon.jit
def _process_segment(
    cfg,
    seg,
    q_dot,
    q_rope_dot,
    lo,
    hi,
    m_i,
    l_i,
    acc,
    head_mask,
    qk_scale,
    v_scale,
    kv_smem,
    rope_smem,
):
    offs_full = gl.arange(0, cfg.KV_DIM, layout=gl.SliceLayout(0, cfg.gather_l))
    offs_full16 = gl.arange(
        0, cfg.KV_DIM // 2, layout=gl.SliceLayout(0, cfg.gather16_l)
    )
    offs_rope = gl.arange(0, cfg.ROPE_DIM, layout=gl.SliceLayout(0, cfg.gather_rope_l))
    k_rng_slot = gl.arange(0, cfg.BLOCK_K, layout=cfg.slot_l)
    k_rng_rope = gl.arange(0, cfg.BLOCK_K, layout=gl.SliceLayout(1, cfg.gather_rope_l))

    # [lo, hi_full) are full mask-free tiles; only the peeled tail is masked.
    hi_full = lo + ((hi - lo) // cfg.BLOCK_K) * cfg.BLOCK_K

    if seg.fmt.IS_FP8:
        # UNI_TILE: the partial tile is just the last iteration (no peeled
        # masked copy of the body).
        if cfg.UNI_TILE:
            n_full = (hi - lo + cfg.BLOCK_K - 1) // cfg.BLOCK_K
        else:
            n_full = (hi_full - lo) // cfg.BLOCK_K
        if n_full > 0:
            kn, ks, kr, vld = _gather_full(
                cfg,
                seg,
                lo,
                hi,
                offs_full,
                offs_full16,
                offs_rope,
                k_rng_slot,
                k_rng_rope,
            )
            for i in range(1, n_full):
                kn2, ks2, kr2, vld2 = _gather_full(
                    cfg,
                    seg,
                    lo + i * cfg.BLOCK_K,
                    hi,
                    offs_full,
                    offs_full16,
                    offs_rope,
                    k_rng_slot,
                    k_rng_rope,
                )
                m_i, l_i, acc = _qkpv(
                    cfg,
                    seg,
                    kn,
                    ks,
                    kr,
                    vld,
                    q_dot,
                    q_rope_dot,
                    m_i,
                    l_i,
                    acc,
                    head_mask,
                    qk_scale,
                    v_scale,
                    kv_smem,
                    rope_smem,
                    lo + (i - 1) * cfg.BLOCK_K,
                    hi,
                )
                kn, ks, kr, vld = kn2, ks2, kr2, vld2
            m_i, l_i, acc = _qkpv(
                cfg,
                seg,
                kn,
                ks,
                kr,
                vld,
                q_dot,
                q_rope_dot,
                m_i,
                l_i,
                acc,
                head_mask,
                qk_scale,
                v_scale,
                kv_smem,
                rope_smem,
                lo + (n_full - 1) * cfg.BLOCK_K,
                hi,
            )
    else:
        for k_start in range(lo, hi_full, cfg.BLOCK_K):
            m_i, l_i, acc = _decode_tile(
                cfg,
                seg,
                q_dot,
                q_rope_dot,
                k_start,
                hi,
                m_i,
                l_i,
                acc,
                head_mask,
                qk_scale,
                v_scale,
                kv_smem,
                rope_smem,
                offs_full,
                offs_full16,
                offs_rope,
                k_rng_slot,
                k_rng_rope,
                False,
            )

    if (not cfg.UNI_TILE) or (not seg.fmt.IS_FP8):
      if hi_full < hi:
        m_i, l_i, acc = _decode_tile(
            cfg,
            seg,
            q_dot,
            q_rope_dot,
            hi_full,
            hi,
            m_i,
            l_i,
            acc,
            head_mask,
            qk_scale,
            v_scale,
            kv_smem,
            rope_smem,
            offs_full,
            offs_full16,
            offs_rope,
            k_rng_slot,
            k_rng_rope,
            True,
        )
    return m_i, l_i, acc


_pa_decode_sparse_repr = make_kernel_repr(
    "_pa_decode_sparse",
    ["BLOCK_M", "BLOCK_K", "HEAD_SIZE", "NUM_SPLITS", "MAIN_FMT", "ROPE_SEPARATE"],
)


@gluon.jit(repr=_pa_decode_sparse_repr)
def _pa_decode_sparse(
    # Shapes below: C = queries, H = num_heads, S = HEAD_SIZE (the V width),
    # R = ROPE_DIM, nnz = total gathered tokens in a segment's index list.
    q_ptr,                # [C, H, S (+R when ROPE_SEPARATE)] bf16
    # One segment = a cache plus its index list. The cache is paged
    # [num_blocks, BLOCK_SIZE, row] (BLOCK_SIZE = 1 for a flat pool), and
    # indices[indptr[t]:indptr[t + 1]] are the rows query t attends to. The two
    # cache pointers are the same allocation under different element types;
    # which of them is live depends on the format (see Seg).
    main_cache_ptr,       # main (SWA) cache, u8 view
    main_cache_bf16_ptr,  # bf16 view of it, or the f32 scale pool ("uniform")
    main_indices_ptr,     # [nnz_main] int32 row ids
    main_indptr_ptr,      # [C + 1] int32
    extra_cache_ptr,      # top-k segment; aliases main when HAS_EXTRA=False
    extra_cache_bf16_ptr,
    extra_indices_ptr,    # [nnz_extra] int32
    extra_indptr_ptr,     # [C + 1] int32
    attn_sink_ptr,        # [H] f32, HAS_SINK only
    out_ptr,              # [C, H, S] bf16, written when NUM_SPLITS == 1
    # Split-K partials, written instead of out_ptr when NUM_SPLITS > 1 (unused
    # placeholders otherwise).
    part_m_ptr,           # [C, NUM_SPLITS, H] f32 row max, base-2 domain
    part_l_ptr,           # [C, NUM_SPLITS, H] f32 row sum
    part_acc_ptr,         # [C, NUM_SPLITS, H, S] bf16 or f32, un-normalized
    # f32 side-channel per segment: scalar k_scale ("tensor") or f32 cache view
    # ("dsmla"). None elides the argument, keeping other formats' kernarg
    # layouts unchanged.
    main_scl_ptr,
    extra_scl_ptr,
    scale: gl.constexpr,
    q_stride0: gl.constexpr,
    q_stride1: gl.constexpr,
    out_stride0: gl.constexpr,
    out_stride1: gl.constexpr,
    main_cs0,
    extra_cs0,
    main_num_rows,
    extra_num_rows,
    pm_stride0: gl.constexpr,
    pm_stride_s: gl.constexpr,
    pa_stride0: gl.constexpr,
    pa_stride_s: gl.constexpr,
    pa_stride_h: gl.constexpr,
    num_heads: gl.constexpr,
    HAS_EXTRA: gl.constexpr,
    HAS_SINK: gl.constexpr,
    MAIN_FMT: gl.constexpr,
    EXTRA_FMT: gl.constexpr,
    MAIN_BLOCK_SIZE: gl.constexpr,
    EXTRA_BLOCK_SIZE: gl.constexpr,
    CS0_ALIGN: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    ROPE_SEPARATE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    HEAD_ALIGNED: gl.constexpr,
    MFMA_K: gl.constexpr,
    GATHER_TW1: gl.constexpr,
    LDS_PAD: gl.constexpr,
    # NOPE_CHUNK: extent of one dequant piece along CHUNK_AXIS (0 = rows,
    # 1 = columns); >= the tile's extent means one shot.
    NOPE_CHUNK: gl.constexpr,
    CHUNK_AXIS: gl.constexpr,
    PART_STORE_CACHE: gl.constexpr,
    UNI_TILE: gl.constexpr,
    GRID_ORDER: gl.constexpr,
    # MAIN_SPLITS <= NUM_SPLITS: splitting the SWA window past its tile count
    # only manufactures masked partial tiles, so main stops early and extra
    # keeps all programs (surplus ones get an empty main range).
    MAIN_SPLITS: gl.constexpr,
    # ADAPTIVE_SPLITS: re-decide the useful split count per query at runtime.
    ADAPTIVE_SPLITS: gl.constexpr,
    ASM_DEQ: gl.constexpr,
    # Per-cache buffer/global gate: buffer_load carries a 32-bit offset (2 GB),
    # and the two caches are sized independently.
    MAIN_USE_BUFFER_LOAD: gl.constexpr,
    EXTRA_USE_BUFFER_LOAD: gl.constexpr,
    IDX_BUFFER_LOAD: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    FP8_FNUZ: gl.constexpr,
    FP8_MFMA: gl.constexpr = False,
    # q already quantized to e4m3 by the caller, plus the scalar f32 scale it
    # was quantized with. This is the calling convention aiter's asm
    # mla_decode_fwd uses, where vLLM passes layer._q_scale.
    q_scl_ptr=None,
    Q_FP8: gl.constexpr = False,
):
    """One program = (query, split, head-block). Two-loop: main (SWA) then
    extra (top-k). NUM_SPLITS==1 writes the output directly; otherwise stores
    un-normalized partials for the reduce kernel."""
    NUM_WARPS: gl.constexpr = gl.num_warps()
    gl.static_assert(
        UNI_TILE or (MAIN_FMT != "tensor" and MAIN_FMT != "dsmla"),
        "tensor/dsmla formats require UNI_TILE=1",
    )
    gl.static_assert(
        UNI_TILE or (EXTRA_FMT != "tensor" and EXTRA_FMT != "dsmla"),
        "tensor/dsmla formats require UNI_TILE=1",
    )
    gl.static_assert(
        (not ROPE_SEPARATE) or (MAIN_FMT != "dsv4" and MAIN_FMT != "uniform"),
        "dsv4/uniform formats carry rope inside plane 0 (ROPE_SEPARATE=False)",
    )
    gl.static_assert(
        MAIN_FMT != "dsmla" or ROPE_SEPARATE,
        "fp8_ds_mla is a separated-rope (MLA) format",
    )
    gl.static_assert((not ASM_DEQ) or MAIN_FMT == "dsv4" or EXTRA_FMT == "dsv4",
                     "ASM_DEQ is the dsv4 E8M0 dequant")
    # The fp8 path needs one positive scalar scale per cache, since that is what
    # folds outside the loop, and OCP e4m3 code points, which is what the matrix
    # core reads.
    gl.static_assert(
        (not FP8_MFMA) or (MAIN_FMT == "tensor" and (not HAS_EXTRA or EXTRA_FMT == "tensor")),
        "FP8_MFMA requires the per-tensor fp8 format on every segment",
    )
    gl.static_assert(not (FP8_MFMA and FP8_FNUZ), "FP8_MFMA is OCP e4m3 only")
    gl.static_assert(
        not (FP8_MFMA and HAS_EXTRA),
        "FP8_MFMA defers the V-side scale to the epilogue, so it needs one segment",
    )
    # Row bases are block*cs0 + pos*TOK with runtime block/pos, so divisibility
    # analysis sees 1-byte alignment unless the driver vouches for cs0.
    if CS0_ALIGN > 1:
        main_cs0 = gl.multiple_of(main_cs0, CS0_ALIGN)
        extra_cs0 = gl.multiple_of(extra_cs0, CS0_ALIGN)
    # GRID_ORDER names the launch axes in grid-dim order; dim 0 varies fastest,
    # which decides XCD/L2 sharing.
    query_idx = gl.program_id(GRID_ORDER.index("q"))
    split_id = gl.program_id(GRID_ORDER.index("s"))
    pid_h = gl.program_id(GRID_ORDER.index("h"))

    cfg = Cfg(
        BLOCK_M,
        BLOCK_K,
        HEAD_SIZE,
        ROPE_DIM,
        ROPE_SEPARATE,
        MFMA_K,
        NUM_WARPS,
        GATHER_TW1,
        LDS_PAD,
        UNI_TILE,
        HAS_INVALID,
        HEAD_ALIGNED,
        IDX_BUFFER_LOAD,
        FP8_MFMA,
    )
    main_fmt = Fmt(
        cfg,
        MAIN_FMT,
        MAIN_BLOCK_SIZE,
        MAIN_USE_BUFFER_LOAD,
        FP8_FNUZ,
        ASM_DEQ and MAIN_FMT == "dsv4",
        NOPE_DIM,
        NOPE_CHUNK,
        CHUNK_AXIS,
    )
    extra_fmt = Fmt(
        cfg,
        EXTRA_FMT,
        EXTRA_BLOCK_SIZE,
        EXTRA_USE_BUFFER_LOAD,
        FP8_FNUZ,
        ASM_DEQ and EXTRA_FMT == "dsv4",
        NOPE_DIM,
        NOPE_CHUNK,
        CHUNK_AXIS,
    )

    h_off = pid_h * BLOCK_M

    # Segment lengths issue first: they gate the three-deep memory chain
    # (indptr -> indices -> cache), and Q is independent of all of it.
    main_start = gl.load(main_indptr_ptr + query_idx)
    main_end = gl.load(main_indptr_ptr + query_idx + 1)
    if HAS_EXTRA:
        extra_start = gl.load(extra_indptr_ptr + query_idx)
        extra_end = gl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start
    else:
        extra_start = 0
        extra_len = 0
    main_len = main_end - main_start

    # exp2 softmax: fold scale*log2(e) into the loop exponent; keep raw scale
    # for the sink (a scaled-score-space logit). For "tensor" the cache's
    # k_scale joins the per-segment fold (max/exp2 commute with a positive
    # scale) and the V-side scale hits p in the loop. Both are exact.
    RCP_LN2: gl.constexpr = 1.4426950408889634
    qk_scale = scale * RCP_LN2
    main_qk_scale = qk_scale
    main_v_scale = 1.0
    if MAIN_FMT == "tensor":
        main_k_scale = gl.load(main_scl_ptr)
        main_qk_scale = qk_scale * main_k_scale
        main_v_scale = main_k_scale
    extra_qk_scale = qk_scale
    extra_v_scale = 1.0
    if HAS_EXTRA and EXTRA_FMT == "tensor":
        extra_k_scale = gl.load(extra_scl_ptr)
        extra_qk_scale = qk_scale * extra_k_scale
        extra_v_scale = extra_k_scale

    # Load Q. ROPE_SEPARATE loads the two pieces separately (the combined
    # width is not a pow-2 arange), each converted to its dot layout.
    offs_m_q = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.blocked_q))
    offs_d_q = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, cfg.blocked_q))
    h_q = h_off + offs_m_q
    h_mask_q = h_q < num_heads
    q_off = (query_idx * q_stride0 + h_q[:, None] * q_stride1 + offs_d_q[None, :]).to(
        gl.int32
    )
    q = gl.amd.cdna4.buffer_load(
        ptr=q_ptr, offsets=q_off, mask=h_mask_q[:, None], other=0.0
    )
    if FP8_MFMA and not Q_FP8:
        # bf16 q: quantize here, one e4m3 scale for this program's whole Q tile
        # (nope and rope), so the fold below is one extra factor on qk_scale.
        # m/l stay in the true-score domain, so split-K programs stay comparable
        # in the reduce.
        E4M3_MAX: gl.constexpr = 448.0
        q_amax = gl.max(gl.max(gl.abs(q).to(gl.float32), axis=1), axis=0)
    q_dot = gl.convert_layout(q, cfg.q_layout)
    if ROPE_SEPARATE:
        offs_d_qr = gl.arange(0, ROPE_DIM, layout=gl.SliceLayout(0, cfg.blocked_q))
        qr_off = (
            query_idx * q_stride0
            + h_q[:, None] * q_stride1
            + HEAD_SIZE
            + offs_d_qr[None, :]
        ).to(gl.int32)
        q_rope = gl.amd.cdna4.buffer_load(
            ptr=q_ptr, offsets=qr_off, mask=h_mask_q[:, None], other=0.0
        )
        q_rope_dot = gl.convert_layout(q_rope, cfg.q_layout)
    else:
        q_rope_dot = q_dot  # unused (single-plane QK) -> DCE'd

    if Q_FP8:
        # Nothing to quantize; fold the caller's scale the way k_scale is
        # folded.
        q_scale = gl.load(q_scl_ptr)
        if not FP8_MFMA:
            # fp8 -> bf16 is exact (3 mantissa bits into 8), so a quantized q
            # costs nothing extra on the bf16 dots.
            q_dot = gl.convert_layout(q.to(gl.bfloat16), cfg.q_layout)
            if ROPE_SEPARATE:
                q_rope_dot = gl.convert_layout(q_rope.to(gl.bfloat16), cfg.q_layout)
            else:
                q_rope_dot = q_dot
        main_qk_scale = main_qk_scale * q_scale
        extra_qk_scale = extra_qk_scale * q_scale
    elif FP8_MFMA:
        if ROPE_SEPARATE:
            q_amax = gl.maximum(
                q_amax, gl.max(gl.max(gl.abs(q_rope).to(gl.float32), axis=1), axis=0)
            )
        q_amax = gl.maximum(q_amax, 1e-30)
        q_rcp = E4M3_MAX / q_amax
        q_dot = gl.convert_layout(
            (q.to(gl.float32) * q_rcp).to(gl.float8e4nv), cfg.q_layout
        )
        if ROPE_SEPARATE:
            q_rope_dot = gl.convert_layout(
                (q_rope.to(gl.float32) * q_rcp).to(gl.float8e4nv), cfg.q_layout
            )
        else:
            q_rope_dot = q_dot
        q_scale = q_amax / E4M3_MAX
        main_qk_scale = main_qk_scale * q_scale
        extra_qk_scale = extra_qk_scale * q_scale

    # head mask in pv-slice layout (for output / partial masking)
    offs_m_pv = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout))
    h_pv = h_off + offs_m_pv
    head_mask_pv = h_pv < num_heads

    # online-softmax state
    m_i = gl.full(
        [BLOCK_M], float("-inf"), gl.float32, layout=gl.SliceLayout(1, cfg.qk_layout)
    )
    l_i = gl.zeros([BLOCK_M], gl.float32, layout=gl.SliceLayout(1, cfg.qk_layout))
    acc = gl.zeros([BLOCK_M, HEAD_SIZE], gl.float32, layout=cfg.pv_layout)

    # An fp8 plane is half the bytes of the bf16 staging it replaces.
    SMEM_DT: gl.constexpr = gl.float8e4nv if FP8_MFMA else gl.bfloat16
    kv_smem = gl.allocate_shared_memory(
        SMEM_DT, [BLOCK_K, HEAD_SIZE], cfg.kv_shared
    )
    if ROPE_SEPARATE:
        rope_smem = gl.allocate_shared_memory(
            SMEM_DT, [BLOCK_K, ROPE_DIM], cfg.rope_shared
        )
    else:
        rope_smem = kv_smem  # never read as plane 1 in this geometry

    # ADAPTIVE_SPLITS: the host split count is sized for batch averages; in a
    # ragged batch the surplus programs would each gather a mostly-masked tile
    # and write a full partial. Recompute from this query's own lengths and let
    # those programs write a neutral partial (m = -inf) and leave. The reduce
    # skips them, so their part_acc never has to be written.
    if ADAPTIVE_SPLITS:
        m_tiles = (main_len + BLOCK_K - 1) // BLOCK_K
        e_tiles = (extra_len + BLOCK_K - 1) // BLOCK_K
        work_splits = gl.minimum(
            gl.maximum(gl.maximum(m_tiles, e_tiles), 1), NUM_SPLITS
        )
        main_splits = gl.minimum(gl.maximum(m_tiles, 1), work_splits)
        if split_id >= work_splits:
            pm_base = query_idx * pm_stride0 + split_id * pm_stride_s
            gl.amd.cdna4.buffer_store(
                gl.full(
                    [BLOCK_M],
                    float("-inf"),
                    gl.float32,
                    layout=gl.SliceLayout(1, cfg.pv_layout),
                ),
                ptr=part_m_ptr + pm_base,
                offsets=h_pv.to(gl.int32),
                mask=head_mask_pv,
            )
            gl.amd.cdna4.buffer_store(
                gl.zeros([BLOCK_M], gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout)),
                ptr=part_l_ptr + pm_base,
                offsets=h_pv.to(gl.int32),
                mask=head_mask_pv,
            )
            return
    else:
        work_splits = NUM_SPLITS
        main_splits = MAIN_SPLITS

    # main (SWA) segment
    main_seg = Seg(
        main_fmt,
        main_cache_ptr,
        main_cache_bf16_ptr,
        main_scl_ptr if (MAIN_FMT == "dsmla") else main_cache_ptr,
        main_indices_ptr,
        main_start,
        main_cs0,
        main_num_rows,
    )
    main_chunk = (main_len + main_splits - 1) // main_splits
    main_lo = gl.minimum(split_id * main_chunk, main_len)
    main_hi = gl.minimum(main_lo + main_chunk, main_len)
    m_i, l_i, acc = _process_segment(
        cfg,
        main_seg,
        q_dot,
        q_rope_dot,
        main_lo,
        main_hi,
        m_i,
        l_i,
        acc,
        head_mask_pv,
        main_qk_scale,
        main_v_scale,
        kv_smem,
        rope_smem,
    )

    if HAS_EXTRA:
        extra_seg = Seg(
            extra_fmt,
            extra_cache_ptr,
            extra_cache_bf16_ptr,
            extra_scl_ptr if (EXTRA_FMT == "dsmla") else extra_cache_ptr,
            extra_indices_ptr,
            extra_start,
            extra_cs0,
            extra_num_rows,
        )
        extra_chunk = (extra_len + work_splits - 1) // work_splits
        extra_lo = split_id * extra_chunk
        extra_hi = gl.minimum(extra_lo + extra_chunk, extra_len)
        m_i, l_i, acc = _process_segment(
            cfg,
            extra_seg,
            q_dot,
            q_rope_dot,
            extra_lo,
            extra_hi,
            m_i,
            l_i,
            acc,
            head_mask_pv,
            extra_qk_scale,
            extra_v_scale,
            kv_smem,
            rope_smem,
        )

    if FP8_MFMA:
        # The fp8 PV dot ran on raw code points, so the V-side scale comes off
        # here, once per program instead of once per tile. l is untouched, so
        # out = acc*s/l is what the bf16 path computes.
        acc = acc * main_v_scale

    # Move the row reductions into pv-slice space for output/partials.
    m_pv = gl.convert_layout(m_i, gl.SliceLayout(1, cfg.pv_layout))
    l_pv = gl.convert_layout(l_i, gl.SliceLayout(1, cfg.pv_layout))

    if NUM_SPLITS == 1:
        if HAS_SINK:
            # m_pv is in the base-2 exponent domain; lift the sink into it.
            sink = gl.amd.cdna4.buffer_load(
                ptr=attn_sink_ptr, offsets=h_pv, mask=head_mask_pv, other=float("-inf")
            ).to(gl.float32) * RCP_LN2
            m_final = _max2(m_pv, sink)
            alpha = gl.exp2(m_pv - m_final)
            l_final = l_pv * alpha + gl.exp2(sink - m_final)
            acc = acc * alpha[:, None]
        else:
            l_final = l_pv
        one_over_l = 1.0 / l_final
        out = acc * one_over_l[:, None]
        offs_d_o = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, cfg.pv_layout))
        o_off = (
            query_idx * out_stride0 + h_pv[:, None] * out_stride1 + offs_d_o[None, :]
        ).to(gl.int32)
        gl.amd.cdna4.buffer_store(
            out.to(out_ptr.dtype.element_ty),
            ptr=out_ptr,
            offsets=o_off,
            mask=head_mask_pv[:, None],
        )
    else:
        # Un-normalized partials for the reduce kernel; m stays in the base-2
        # exponent domain (the triton reduce's convention too).
        pm_base = query_idx * pm_stride0 + split_id * pm_stride_s
        gl.amd.cdna4.buffer_store(
            m_pv,
            ptr=part_m_ptr + pm_base,
            offsets=h_pv.to(gl.int32),
            mask=head_mask_pv,
            cache=PART_STORE_CACHE,
        )
        gl.amd.cdna4.buffer_store(
            l_pv,
            ptr=part_l_ptr + pm_base,
            offsets=h_pv.to(gl.int32),
            mask=head_mask_pv,
            cache=PART_STORE_CACHE,
        )
        offs_d_a = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, cfg.pv_layout))
        a_base = query_idx * pa_stride0 + split_id * pa_stride_s
        a_off = (a_base + h_pv[:, None] * pa_stride_h + offs_d_a[None, :]).to(gl.int32)
        # Follow part_acc's own dtype: bf16 halves the partial HBM traffic.
        gl.amd.cdna4.buffer_store(
            acc.to(part_acc_ptr.dtype.element_ty),
            ptr=part_acc_ptr,
            offsets=a_off,
            mask=head_mask_pv[:, None],
            cache=PART_STORE_CACHE,
        )


_pa_decode_sparse_reduce_repr = make_kernel_repr(
    "_pa_decode_sparse_reduce",
    ["BLOCK_M", "HEAD_SIZE", "NUM_SPLITS"],
)


@gluon.jit(repr=_pa_decode_sparse_reduce_repr)
def _pa_decode_sparse_reduce(
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    attn_sink_ptr,
    out_ptr,
    out_stride0: gl.constexpr,
    out_stride1: gl.constexpr,
    pm_stride0: gl.constexpr,
    pm_stride_s: gl.constexpr,
    pa_stride0: gl.constexpr,
    pa_stride_s: gl.constexpr,
    pa_stride_h: gl.constexpr,
    num_heads: gl.constexpr,
    HAS_SINK: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    HEAD_ALIGNED: gl.constexpr,
    ADAPTIVE_SPLITS: gl.constexpr,
):
    """Split-KV combine: merge per-split partials, fold the sink, write the
    output. Identical for both geometries (partials are HEAD_SIZE = V wide).
    Grid: (num_queries, heads_blocks); the combine is pure bandwidth, so
    BLOCK_M is sized for workgroup count, not for the attention kernel's tile."""
    NUM_WARPS: gl.constexpr = gl.num_warps()
    RCP_LN2: gl.constexpr = 1.4426950408889634
    query_idx = gl.program_id(0)
    pid_h = gl.program_id(1)

    # Lay the 64 lanes out so a small BLOCK_M spends them on the head dim.
    TPW0: gl.constexpr = BLOCK_M if BLOCK_M < 8 else 8
    TPW1: gl.constexpr = 64 // TPW0
    BLK: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[TPW0, TPW1],
        warps_per_cta=[1, NUM_WARPS],
        order=[1, 0],
    )
    row_l: gl.constexpr = gl.SliceLayout(1, BLK)  # [BLOCK_M]

    h_off = pid_h * BLOCK_M
    offs_m = gl.arange(0, BLOCK_M, layout=row_l)
    h = h_off + offs_m
    head_mask = h < num_heads
    offs_d = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, BLK))

    neg_inf = float("-inf")
    # Deliberately NOT bounded by the per-query split count: that would make
    # these dynamic loops and lose the static unroll. A split that bowed out
    # wrote m = -inf, which is what actually has to be handled (below).
    m_final = gl.full([BLOCK_M], neg_inf, gl.float32, layout=row_l)
    # pass 1: global max over splits
    for s in range(NUM_SPLITS):
        base = query_idx * pm_stride0 + s * pm_stride_s
        m_s = gl.amd.cdna4.buffer_load(
            ptr=part_m_ptr + base, offsets=h, mask=head_mask, other=neg_inf,
        )
        m_final = _max2(m_final, m_s)  # m_s already in base-2 exponent domain
    if HAS_SINK:
        sink = gl.amd.cdna4.buffer_load(
            ptr=attn_sink_ptr, offsets=h, mask=head_mask, other=neg_inf, cache=".cg",
        ).to(gl.float32)
        scaled_sink = sink * RCP_LN2
        m_final = _max2(m_final, scaled_sink)  # lift sink to base-2

    # pass 2: weighted sums
    l_final = gl.zeros([BLOCK_M], gl.float32, layout=row_l)
    acc = gl.zeros([BLOCK_M, HEAD_SIZE], gl.float32, layout=BLK)
    for s in range(NUM_SPLITS):
        base = query_idx * pm_stride0 + s * pm_stride_s
        m_s = gl.amd.cdna4.buffer_load(
            ptr=part_m_ptr + base, offsets=h, mask=head_mask, other=neg_inf,
            cache=".cg",
        )
        l_s = gl.amd.cdna4.buffer_load(
            ptr=part_l_ptr + base, offsets=h, mask=head_mask, other=0.0,
            cache=".cg",
        )
        w = gl.exp2(m_s - m_final)
        l_final = l_final + w * l_s
        a_base = query_idx * pa_stride0 + s * pa_stride_s
        a_off = (a_base + h[:, None] * pa_stride_h + offs_d[None, :]).to(gl.int32)
        # A bowed-out split's part_acc is uninitialized, so mask the load
        # rather than relying on w == 0 (0 * NaN is NaN).
        if ADAPTIVE_SPLITS:
            acc_mask = head_mask[:, None] & (m_s > neg_inf)[:, None]
        else:
            acc_mask = head_mask[:, None]
        acc_s = gl.amd.cdna4.buffer_load(
            ptr=part_acc_ptr, offsets=a_off, mask=acc_mask, other=0.0,
            cache=".cg",
        )
        acc = acc + w[:, None] * acc_s.to(gl.float32)

    if HAS_SINK:
        l_final = l_final + gl.exp2(scaled_sink - m_final)

    # One reciprocal per row instead of a per-element f32 divide.
    one_over_l = 1.0 / l_final
    out = acc * one_over_l[:, None]
    o_off = (query_idx * out_stride0 + h[:, None] * out_stride1 + offs_d[None, :]).to(
        gl.int32
    )
    gl.amd.cdna4.buffer_store(
        out.to(out_ptr.dtype.element_ty),
        ptr=out_ptr,
        offsets=o_off,
        mask=head_mask[:, None],
    )
