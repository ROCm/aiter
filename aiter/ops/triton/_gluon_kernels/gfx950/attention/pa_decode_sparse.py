# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950) sparse-MLA decode for DeepSeek-V4 *and* GLM-5 (MLA) geometries.

Restructuring of the previous DSv4-only kernel (last DSv4-only head:
``645a30b0e``); for the existing DSv4/uniform configurations it compiles to
byte-identical .amdgcn (gate: ``compile_check.py`` in the port workspace,
``/app/scripts/glm_gluon_port_8_19``, which keeps the frozen reference).
Two changes:

1. **Aggregate config objects** (the ``unified_attention_2d.py`` pattern): the
   ~40-entry constexpr parameter lists threaded through every helper are replaced by
   three aggregates —

       Cfg  compile-time geometry, layouts, and behavior knobs (one per launch)
       Fmt  compile-time cache-format description        (one per segment)
       Seg  runtime segment context: pointers + bounds    (one per segment)

   Helpers take ``(cfg, seg, ...)``; the flat constexpr list exists only at the
   kernel entry, where the aggregates are built.

2. **Two geometries under one constexpr, ``ROPE_SEPARATE``**:

       ROPE_SEPARATE=False (DSv4): rope lives *inside* the 512-wide row. One LDS
           plane ``kv_smem [BLOCK_K, KV_DIM]``, one QK MFMA chain over KV_DIM,
           V = the same plane. Bit-for-bit today's kernel (compile_check.py
           diffs the .amdgcn against the aiter reference to prove it).
       ROPE_SEPARATE=True (GLM-5): rope is *appended* (512 latent + 64 rope = 576
           QK; V = the 512 latent only). Two LDS planes, both pow-2:
           ``kv_smem [BLOCK_K, KV_DIM]`` + ``rope_smem [BLOCK_K, ROPE_DIM]``.
           QK = mfma(q_rope, k_rope, mfma(q_nope, k_nope, 0)) — MFMA accumulates
           natively so the 576-wide contraction is two chained dots. The PV dot
           and the accumulator are unchanged: V is always exactly plane 0.

Cache formats (``Fmt.KIND``), all sharing one gather/stage/dot pipeline:

    KIND      row layout (per token)                        scales        rope
    "bf16"    [KV_DIM (+ROPE_DIM if separate)] bf16         —             in-row
    "dsv4"    448 fp8 | 64 bf16 rope (576 B), 8 B E8M0/64   E8M0, block   in-plane0
              trailer per 64-token block (aiter fp8_ds_mla-packed DSv4)
    "uniform" [KV_DIM] fp8 + separate [tokens, NG] f32      f32 per-64    in-row
    "tensor"  [KV_DIM (+ROPE_DIM if separate)] fp8, one     scalar        fp8 tail
              per-tensor f32 scale (GLM-5's production asm format; also the
              DSv4 per-tensor experiment)
    "dsmla"   512 fp8 | 4 f32 | 64 bf16 rope (656 B)        f32 per-128   bf16 tail
              (vLLM CacheDType "fp8_ds_mla"; requires ROPE_SEPARATE)

Per-tensor scale handling ("tensor") never touches the tile loop: the K-side scale
folds into the segment's qk_scale (max/exp2 commute with a positive scale) and the
V-side scale multiplies ``p`` — 4 f32 per lane — right before the PV dot, so the
staged tile is a raw fp8→bf16 convert with no multiplies at all.

Two-loop (SWA + top-k) or a single segment; 2D and 3D (split-K + reduce) share one
kernel. Launchers: ``aiter/ops/triton/attention/pa_decode_sparse.py`` (DSv4 /
uniform-pool, geometry unchanged) and
``aiter/ops/triton/attention/sparse_mla_decode.py`` (GLM-5 / separated-rope MLA).
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import PropagateNan
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.common_utils import strip_annotate

# Triton's default max ignores NaN, which on AMD costs a v_max_f32 x, x, x
# canonicalize per operand before the real compare -- 60 of the 96 v_max in this
# kernel were those no-ops. Nothing here produces NaN (masked lanes are -inf and
# the all-masked row is guarded explicitly), so propagate instead.
_MAX_PROP_NAN: gl.constexpr = gl.constexpr(PropagateNan.ALL)


@gluon.jit
def _max2(a, b):
    return gl.maximum(a, b, propagate_nan=_MAX_PROP_NAN)


@gluon.jit
def _rmax(x, axis):
    return gl.reduce(x, axis, _max2)


@gluon.jit
def _cache_load(ptr, row, col, USE_BUFFER_LOAD: gl.constexpr, mask=None, other=None):
    """Gather rows[i] + col[j] out of a cache.

    ``row`` is the per-token offset in ptr's element units ([BLOCK_K]); ``col`` is
    the in-row offset ([W]), always a small compile-time arange. Keeping them apart
    is what makes the >2 GB path affordable: buffer_load carries a 32-bit offset
    (2 GB cap), so a bigger cache has to gather through 64-bit addresses -- and
    adding the pointer to a fully materialized [BLOCK_K, W] offset tensor makes
    that a 64-bit add *per element*. Resolving one pointer per token instead costs
    BLOCK_K of them, and the leftover column offset is a constant the load can fold
    into its immediate field.
    """
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
    # gfx950's native e4m3 cvt is OCP (float8e4nv). fnuz (float8e4b8) -> f32 has no
    # native cvt and lowers to a ~5x software unpack that spills; but fnuz -> bf16 is
    # cheap and fp8 -> bf16 is exact (3 mantissa bits), so route fnuz through bf16.
    if FP8_FNUZ:
        return x_u8.to(gl.float8e4b8, bitcast=True).to(gl.bfloat16).to(gl.float32)
    return x_u8.to(gl.float8e4nv, bitcast=True).to(gl.float32)


@gluon.jit
def _fp8_to_bf16(x_u8, FP8_FNUZ: gl.constexpr):
    # fp8 -> bf16 is exact (3 mantissa bits into 8), so the per-tensor mode can
    # stage the raw code points without any scale arithmetic at all.
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
    """Gather the NG per-group scale entries of each token and broadcast them out
    to W_FULL columns in registers.

    The obvious spelling -- index the full row with ``offs_full // GROUP`` -- builds
    a [BLOCK_K, W_FULL] pointer tensor for a value that has only NG distinct entries
    per row. On the buffer_load path the identical 32-bit offsets CSE away; on the
    64-bit path they do not, and the wide form costs 6x the loads. Gather NG wide
    instead and broadcast: ``scl_l`` is the dim-2 slice of a 3-D layout picked so
    that reshaping [BLOCK_K, NG, GROUP] back to [BLOCK_K, W_FULL] lands exactly on
    ``gather_l`` -- so the broadcast is a register rename (assert_trivial proves it
    at compile time; GLM5_PORT_PLAN flagged re-verifying this for NG != 8).

    Element type follows ``ptr``: u8 E8M0 bytes for the DSv4 packed cache
    (OTHER=127 -> 2^0), f32 group scales for fp8_ds_mla (OTHER=0.0 -> masked
    lanes dequant to 0).
    """
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
    """Contiguous register split along dim 1: [A, B] -> two [A, B//2].

    x0 takes columns [0, B//2), x1 takes [B//2, B). Both halves come back in the
    input's own layout, so with warps tiling dim 0 (see ``Cfg.gather_l``) the column
    direction is a pure per-lane register repeat and the split is a rename -- no
    cross-lane traffic. ``assert_trivial`` makes a non-free split a compile error
    rather than a silent LDS round-trip.
    """
    layout: gl.constexpr = x.type.layout
    x_r = x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1)
    x0, x1 = gl.split(x_r)
    x0 = gl.convert_layout(x0, layout, assert_trivial=True)
    x1 = gl.convert_layout(x1, layout, assert_trivial=True)
    return x0, x1


@gluon.jit
def _split2_dim0(x):
    """Contiguous register split along dim 0: [A, B] -> two [A//2, B].

    The dim-0 counterpart of _split2, for the row-wide gather layout where dim 1
    is spent entirely on threads (one instruction = whole token rows) and the
    per-lane register repeats live on dim 0 instead.
    """
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
# separate single-output asm calls, NOT as one two-output blob: a single-instruction
# asm reads all its sources before writing dst, so no output can clobber an input and
# neither needs early clobber. The blob form does -- without `=&v` on its first output
# the allocator may give it the scale's register and the second convert then reads a
# clobbered scale, wrong in exactly the elements it produces -- and that forced
# liveness costs spills: 21 st/28 ld against 11/12 here, on a kernel with zero
# headroom at the 256-VGPR cap. The split form costs +184 instructions (each call
# re-reads the sources) and is worth it: 0.970-0.977x against the blob's 0.983-1.017x
# across C=16/64/128 at short top-k, and it removes the reduce knock-on that the
# blob's scratch traffic caused by evicting the split-K partials from L2.
_DEQ_LO: gl.constexpr = gl.constexpr("v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2")
_DEQ_HI: gl.constexpr = gl.constexpr(
    "v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2 op_sel:[1,0,0]"
)
_DEQ_CONS: gl.constexpr = gl.constexpr("=v,v,v")


@gluon.jit
def _deq_asm(x16, e_u8, W8: gl.constexpr, out_l: gl.constexpr):
    """[BLOCK_K, W8/2] int16 (4 packed fp8 per 32 bits) + raw E8M0 byte -> [BLOCK_K, W8] bf16.

    The scale operand is read as E8M0 -- the hardware takes bits [30:23] and nothing
    else. Those are bits [14:7] of the operand's high half, so a 16-bit `e << 7` in
    both halves of the register puts the exponent exactly where an f32 `e << 23`
    would, and an i16 operand costs ONE register at pack=2 where f32 cost two.
    Converting to f32 with exp2 also works but is a longer chain for nothing. Because the scale is a power of two this is bit-identical to the f32
    chain: fp8 -> bf16 is exact (3 mantissa bits into 8) and scaling by 2^k stays exact.
    """
    sc16 = e_u8.to(gl.uint16) << 7
    lo = gl.inline_asm_elementwise(
        _DEQ_LO, _DEQ_CONS, [x16, sc16], dtype=gl.bfloat16, is_pure=True, pack=2
    )
    hi = gl.inline_asm_elementwise(
        _DEQ_HI, _DEQ_CONS, [x16, sc16], dtype=gl.bfloat16, is_pure=True, pack=2
    )
    # lo carries fp8 elements 4i+{0,1} and hi 4i+{2,3}, so the flat order is
    # 4i + 2*lohi + s. A lane's 8 int16 are the same 16-byte run the u8 gather used,
    # so this whole interleave stays inside one lane -- assert_trivial proves it.
    W16: gl.constexpr = W8 // 2
    lo3 = lo.reshape(lo.shape[0], W16 // 2, 2)
    hi3 = hi.reshape(hi.shape[0], W16 // 2, 2)
    both = gl.join(lo3, hi3).permute(0, 1, 3, 2).reshape(lo.shape[0], W8)
    # Back to the byte-gather layout, or the kv_smem store sees the join/permute layout
    # and lowers to narrow ds_writes (112 ds_write_b128 -> 28, and ~1.5x slower).
    return gl.convert_layout(both, out_l, assert_trivial=True)


# ---------------------------------------------------------------------------
# Aggregates. Compile-time state lives in Cfg (shared) and Fmt (per segment);
# runtime pointers/bounds live in Seg. Built once at the kernel entry, passed
# in place of the former ~40-parameter lists.
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
    ROPE_SEPARATE: gl.constexpr # False: rope inside plane 0 (DSv4). True: rope is
                                # a K-only second plane (GLM-5); QK contracts over
                                # KV_DIM + ROPE_DIM via two chained MFMAs.
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
    ):
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.KV_DIM = gl.constexpr(KV_DIM)
        self.ROPE_DIM = gl.constexpr(ROPE_DIM)
        self.ROPE_SEPARATE = gl.constexpr(ROPE_SEPARATE)
        self.QK_DIM = gl.constexpr(KV_DIM + (ROPE_DIM if ROPE_SEPARATE else 0))
        self.MFMA_K = gl.constexpr(MFMA_K)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.GATHER_TW1 = gl.constexpr(GATHER_TW1)
        self.LDS_PAD = gl.constexpr(LDS_PAD)
        self.UNI_TILE = gl.constexpr(UNI_TILE)
        self.HAS_INVALID = gl.constexpr(HAS_INVALID)
        self.HEAD_ALIGNED = gl.constexpr(HEAD_ALIGNED)
        self.IDX_BUFFER_LOAD = gl.constexpr(IDX_BUFFER_LOAD)

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

        # 16 uint8 = 128-bit fp8 gather loads
        GSPT = 16
        # Warps on dim 1 (columns) vs dim 0 (rows). Coalescing is identical either way
        # -- a row's 128 contiguous bytes always come from 8 threads -- but the row
        # index vector lives in SliceLayout(1, gather_l), so warps-on-dim-1 replicates
        # it across all NUM_WARPS and every lane carries BLOCK_K/8 slots, while
        # warps-on-dim-0 carries BLOCK_K/(8*NUM_WARPS). Dim 0 also makes the column
        # direction a pure per-lane register repeat, so a chunked dequant can split it
        # with a trivial (register-renaming) layout convert.
        # GATHER_TW1 = threads spent on the head dim. Each thread loads 16 B, so
        # TW1=8 requests 128 B of a token row per instruction and TW1=32 requests the
        # whole 512 B row -- for a scattered gather that is one request instead of
        # four quarters. The cost is that the row-index vector lives in
        # SliceLayout(1, gather_l), so a wider TW1 leaves fewer dim-0 thread slots and
        # every lane carries more slots (BLOCK_K*TW1/64 of them). Whichever dim keeps
        # per-lane register repeats is the one the chunked dequant can split
        # (CHUNK_AXIS).
        self.gather_l = gl.constexpr(
            gl.BlockedLayout(
                size_per_thread=[1, GSPT],
                threads_per_warp=[64 // GATHER_TW1, GATHER_TW1],
                warps_per_cta=[NUM_WARPS, 1],
                order=[1, 0],
            )
        )
        # Warps tile dim 0 here, one warp already covers all 64 RoPE columns, so putting
        # them on dim 1 (as the NoPE gather does) overshoots 4x and every warp re-gathers
        # the same tile. Worth ~10% on packed fp8.
        self.gather_rope_l = gl.constexpr(
            gl.BlockedLayout(
                size_per_thread=[1, 8],
                threads_per_warp=[8, 8],
                warps_per_cta=[NUM_WARPS, 1],
                order=[1, 0],
            )
        )
        # Same tiling as gather_l but 2-byte elements: half the per-thread run, so a lane
        # still covers the same 16 bytes and the lo/hi interleave stays lane-local.
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
        # LDS pad after every row. Row pitch (512 + LDS_PAD) bf16 sets which banks a
        # transposed K read (ds_read_b64_tr_b16 walks down a column) lands on:
        # bank = (row * pitch_dwords) mod 32, so pitch_dwords mod 32 == 4 at PAD=8 means
        # 32 lanes share 8 banks. Kept as a knob because the conflict share is high
        # (46%) but a high sub-metric ratio is not proof of a bottleneck.
        self.kv_shared = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for(
                [[KV_DIM, LDS_PAD]], [BLOCK_K, KV_DIM], [1, 0]
            )
        )
        # Plane 1 (K-only rope) when ROPE_SEPARATE; same padding rule as plane 0
        # (its K read is the same transposed ds_read pattern). Built regardless --
        # a dead constexpr field costs nothing when the plane is never allocated.
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
    SCL_F32_OFF: gl.constexpr     # dsmla: f32-view offset of the 4 group scales
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
        # 3-D companion of gather_l used by _scale_load: dim 1 carries the NG scale
        # groups (one thread each) and dim 2 the GROUP columns inside a group, so
        # that reshaping [BLOCK_K, NG, GROUP] back to 2-D reproduces gather_l
        # exactly. Legal only when the group's columns fill whole threads, i.e.
        # GSPT * (TW1 // NG) is the group width; otherwise fall back to the wide
        # (redundant) gather.
        GSPT = 16
        TW1 = cfg.GATHER_TW1.value
        NARROW_SCALE = TW1 % NG == 0 and GSPT * (TW1 // NG) == GROUP
        self.NARROW_SCALE = gl.constexpr(NARROW_SCALE)
        # ...and only worth it on the 64-bit path. With buffer_load the wide gather's
        # identical 32-bit offsets CSE to the same few loads, so the narrow form
        # only adds a layout convert per tile (+3.4% at extra=1024). Without it they
        # do not CSE and the wide form costs 6x the loads.
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

        # Packed-row constants. dsv4 row: [NOPE_DIM fp8 | ROPE_DIM bf16] with an
        # 8-byte-per-token E8M0 trailer after the block; dsmla row:
        # [KV_DIM fp8 | NG f32 | ROPE_DIM bf16] all inline.
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
        # Flat formats gather in cache elements (bf16 or fp8-byte): the row is
        # KV_DIM wide, plus the appended rope when the geometry separates it.
        self.TOK_EL = gl.constexpr(
            KV_DIM + (ROPE_DIM if cfg.ROPE_SEPARATE.value and KIND != "uniform" else 0)
        )


@aggregate
@strip_annotate
class Seg:
    """Runtime context of one segment: cache pointers, index list, bounds.

    Pointer roles by format ("--" = unused, carries a harmless duplicate):

        KIND      cache_ptr        alt_ptr                  scl_ptr
        bf16      --               bf16 cache               --
        dsv4      u8 cache         bf16 view of the cache   --
        uniform   u8 cache         f32 per-64 kv_scales     --
        tensor    u8 cache         --                       f32 scalar k_scale
        dsmla     u8 cache         bf16 view (rope tail)    f32 view (group scales)
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
    """Dequant one fp8 slab and write it straight to kv_smem[:, off:off+W].

    Keep the dequant in f32: gfx950 has no bf16 multiply, so a bf16 path lowers to
    an emulated mul and doubles the loop. ``sc`` is the raw per-group exponent
    byte (dsv4 packed, UE8M0), an f32 scale (uniform pool / fp8_ds_mla), or unused
    ("tensor": the per-tensor scale is folded outside the tile loop, so staging is
    a bare fp8 -> bf16 convert).
    """
    if fmt.ASM_DEQ:
        # x_u8 is the int16 view here (see _gather_full), so its column count is W8/2.
        W8: gl.constexpr = x_u8.shape[1] * 2
        # Adjacent fp8 columns share a scale (groups are 64 wide, so even), which makes
        # dropping every other broadcast column exact.
        s_even, _ = gl.split(sc.reshape(sc.shape[0], sc.shape[1] // 2, 2))
        # split leaves a SliceLayout of a 3-D parent; the asm needs both operands in
        # the int16 gather's own layout.
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
    """Dequant a whole gathered fp8 tile into kv_smem in NOPE_CHUNK-sized pieces
    along CHUNK_AXIS (0 = rows, 1 = columns).

    The f32 expansion is 4x the fp8 tile (a [64, 512] tile is 128 f32 VGPRs/lane),
    so materializing it in one shot is what pins the kernel at 1 wave/SIMD.
    Splitting first makes the dependence chain explicit -- piece c's converts feed
    piece c's ds_writes and die -- so only a 1/pieces fraction of it is ever
    live. The splits themselves are register renames (see ``_split2``).
    """
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
    # Returns in whatever layout k_pos carries. Called once per gather layout: NoPE
    # and RoPE tile their warps differently, and re-loading this tiny broadcast
    # vector is cheaper than a cross-lane convert between the two.
    # The index list is one int32 per gathered token -- orders of magnitude under
    # buffer_load's 2 GB offset limit even for a full batch -- so it can keep the
    # fast path however big the KV caches are. It is also the largest single source
    # of exec-mask branching in the kernel (20 of 55 s_and_saveexec_b64 on the
    # buffer path, all from the masked tail), because a masked gl.load predicates
    # while a masked buffer_load folds the mask into the offset.
    indices_ptr = seg.indices_ptr
    seg_start = seg.seg_start
    BLOCK_SIZE: gl.constexpr = seg.fmt.BLOCK_SIZE
    HAS_INVALID: gl.constexpr = cfg.HAS_INVALID
    IDX_BUFFER_LOAD: gl.constexpr = cfg.IDX_BUFFER_LOAD
    if MASKED:
        # Legacy peeled tail (UNI_TILE=0, and the bf16 path). Predicating the read is
        # what makes this expensive: a masked gl.load branches on exec.
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
        # UNI_TILE lets the partial tile ride this same body: clamp the index-list read
        # into the segment rather than predicating it, so every lane reads a genuine
        # in-segment index, and add the range test to `valid` -- which becomes a -inf
        # score mask, exactly how -1 sentinels are already handled here. hi >= 1
        # whenever this runs (_gather_full is guarded by n_full > 0, so hi > lo >= 0).
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
    """QK scores for one tile. ROPE_SEPARATE=False is a single MFMA chain over
    plane 0; ROPE_SEPARATE=True chains a second MFMA over the rope plane (MFMA
    accumulates natively, so the KV_DIM + ROPE_DIM contraction is two dots)."""
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
# The prefetched fp8 pipeline: _gather_full issues tile N+1's loads while
# _qkpv stages + dots tile N.
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
    """Gather one full fp8 tile. Split from the LDS-write/MFMA so the gather issues
    an iteration early.

    The prefetch is carried across the MFMA in **raw fp8**, not dequantized bf16:
    the loop keeps two tiles in flight, and a [BLOCK_K, KV_DIM] tile is 32
    VGPRs/lane as u8 but 64 as bf16, so dequantizing here would burn an extra 64
    VGPRs on loop-carried state alone. The consumer dequants (chunked) instead.

    Returns (x, sc, k_rope, valid); unused slots carry a harmless duplicate that
    DCE removes ("tensor" has no scale vector, "uniform" no rope side-channel).
    """
    fmt = seg.fmt
    cs0 = seg.cs0
    if not fmt.USE_BUFFER_LOAD:
        cs0 = cs0.to(gl.int64)  # >2 GB cache: 64-bit gather offsets (see _cache_load)
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
        k_rope = x_u8  # unused for "uniform" (rope slice-store skipped) -> DCE'd
    elif fmt.KIND == "tensor":
        # Flat fp8 rows, one per-tensor scale for the whole cache. No scale work
        # here at all: the K-side scale is folded into this segment's qk_scale and
        # the V-side scale into p (see _qkpv), so staging is a bare convert.
        x_u8 = _cache_load(
            seg.cache_ptr, bg * cs0 + pg * fmt.TOK_EL, offs_full, fmt.USE_BUFFER_LOAD
        )
        sc = x_u8  # no scale vector -> DCE'd
        if cfg.ROPE_SEPARATE:
            # K-only rope tail: fp8 columns [KV_DIM, KV_DIM + ROPE_DIM) of the row.
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
        # vLLM fp8_ds_mla row: [KV_DIM fp8 | NG f32 scales | ROPE_DIM bf16].
        nope_row = bg * cs0 + pg * fmt.TOK_U8
        scl_row = bg * (cs0 // 4) + pg * fmt.TOK_F32 + fmt.SCL_F32_OFF
        # f32 group scales first, bulk fp8 after (same vmcnt-FIFO argument as dsv4).
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
        # UE8M0 scales first, bulk fp8 after. vmcnt is one in-order FIFO, so a
        # wait can only name "at most N outstanding", never a specific load:
        # issuing the scales last would make the first dequant piece wait behind
        # every data load as well. Issued first, the scales are covered by the
        # wait that the first piece's own data needs anyway.
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
            # 2-byte elements so the asm gets <2 x i16> = 4 packed fp8 in one VGPR out
            # of a single dword load. Same bytes, same 16-B per-lane run, half the
            # columns; alt_ptr is already the 2-byte view of this cache and the
            # RoPE gather below uses the same cs0//2 / TOK_U16 addressing.
            # The row vector comes from _slots in SliceLayout(1, gather_l); the i16
            # columns live in gather16_l, and a broadcast needs one parent layout. The
            # two layouts share their dim-0 tiling, so this convert is a rename.
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
    """Write one prefetched tile into the LDS plane(s). Plane 0 is always the full
    KV_DIM-wide dequant ("dsv4" gathers KV_DIM bytes too: the last 64 are the bf16
    rope read as garbage fp8, dequanted, and then overwritten by the real rope
    slice-store below -- that keeps the gather pow-2 wide)."""
    fmt = seg.fmt
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
    """Dequant a prefetched fp8 tile into LDS, then QK -> softmax -> PV. When
    HAS_INVALID, mask the columns of -1-sentinel slots (``valid`` from
    _gather_full) to -inf."""
    neg_inf = float("-inf")
    _stage(cfg, seg, x_u8, sc, k_rope, kv_smem, rope_smem)
    S = _qk_scores(cfg, q_dot, q_rope_dot, kv_smem, rope_smem)
    # UNI_TILE folds the tile's range test into `valid`, so the score mask is no
    # longer only about -1 sentinels: it is what makes the partial last tile correct.
    COL_MASK: gl.constexpr = cfg.HAS_INVALID or cfg.UNI_TILE
    NEED_MASK: gl.constexpr = COL_MASK or (not cfg.HEAD_ALIGNED)
    if NEED_MASK:
        if COL_MASK:
            if cfg.UNI_TILE and not cfg.HAS_INVALID:
                # Build the range mask directly in the MFMA layout. Converting the
                # slot-layout `valid` vector instead costs a cross-lane layout change
                # per tile, which at 16 tiles ate the whole saving.
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
    # m_i is carried in base-2 exponent space, hence qk_scale. max commutes with a
    # positive scale (qk_scale = scale * log2(e) [* k_scale for "tensor"]; any real
    # attention scale is > 0), so scale the row max -- one value per lane -- rather
    # than every element of S. That leaves `S * qk_scale - m_new` as the only use of
    # the scaled S, which lowers to v_fma_f32 where the split form needed a separate
    # multiply over the tile. -inf * positive = -inf, so masked columns stay masked,
    # and the row max is the same product bit for bit; only the fused subtract
    # differs, one rounding instead of two -- measured <= 0.23 of a bf16 output ulp.
    m_block = _rmax(S, 1) * qk_scale
    m_new = _max2(m_i, m_block)
    m_new = gl.where(m_new > neg_inf, m_new, 0.0)
    p = gl.exp2(S * qk_scale - m_new[:, None])
    alpha = gl.exp2(m_i - m_new)
    l_new = l_i * alpha + gl.sum(p, axis=1)
    v = kv_smem.load(cfg.v_layout)
    # "tensor": V was staged as raw fp8 code points, so apply the per-tensor scale
    # on the small side -- p is [BLOCK_M, BLOCK_K], 4 f32 per lane, vs 32 for the
    # accumulator. l_new stays scale-free (it normalizes p, not p*V), so
    # out = sum(p * v_scale * V_raw) / l is exactly sum(p * (v_scale * V_raw)) / l.
    if seg.fmt.KIND == "tensor":
        p = p * v_scale
    p_dot = gl.convert_layout(p.to(gl.bfloat16), cfg.p_layout)
    alpha_pv = gl.convert_layout(alpha, gl.SliceLayout(1, cfg.pv_layout))
    acc = acc * alpha_pv[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v, acc)
    return m_new, l_new, acc


# ---------------------------------------------------------------------------
# The non-prefetched path: bf16 segments, and the peeled masked tail when
# UNI_TILE is off (dsv4/uniform only; the new fp8 formats require UNI_TILE).
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
    """One KV tile -> online-softmax update. MASKED=False (peeled full tiles) drops
    the in-range / gather / score masking; MASKED=True (the tail) keeps it. When
    HAS_INVALID, full tiles also clamp -1 sentinels in-bounds for the gather and
    mask their scores to -inf (matching the tail's slot-validity handling)."""
    neg_inf = float("-inf")
    fmt = seg.fmt
    cs0 = seg.cs0
    if not fmt.USE_BUFFER_LOAD:
        cs0 = cs0.to(gl.int64)  # >2 GB cache: 64-bit gather offsets (see _cache_load)
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
        # uniform pool: one fp8 gather over the whole head + separate fp32 scales.
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
            )  # uniform pool: fp32 scales, left wide (see _scale_load)
        else:
            x_u8 = _cache_load(seg.cache_ptr, kv_row, offs_full, fmt.USE_BUFFER_LOAD)
            sc = _cache_load(seg.alt_ptr, scl_row, scl_col, fmt.USE_BUFFER_LOAD)
        _deq_store_tile(x_u8, sc, kv_smem, cfg, fmt)
    elif fmt.KIND == "dsv4":
        # DSv4 packed fp8_ds_mla: NoPE fp8 + embedded UE8M0 + separate RoPE-bf16.
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
                # Same byte as nope_row, addressed in 2-byte units: one shift on a
                # live vector, and block_idx_g * cs0 stays common with the scale
                # gather.  Exact -- cs0 and TOK_U8 are both even.
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
                # Same byte as nope_row, addressed in 2-byte units: one shift on a
                # live vector, and block_idx_g * cs0 stays common with the scale
                # gather.  Exact -- cs0 and TOK_U8 are both even.
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
    else:  # "bf16" (the new fp8 formats require UNI_TILE, so they never come here)
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
            # Appended bf16 rope columns [KV_DIM, KV_DIM + ROPE_DIM) -> plane 1.
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
    # exp2 softmax: qk_scale folds in log2(e) so we hit the HW exp2 directly.
    # Running max stays in raw-score space; masked cols (-inf) give exp2=0.
    COL_VALID: gl.constexpr = MASKED or cfg.HAS_INVALID  # valid1d defined in both
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
        else:  # not HEAD_ALIGNED and no col invalidity -> head mask only
            col_mask = gl.convert_layout(head_mask, gl.SliceLayout(1, cfg.qk_layout))[
                :, None
            ]
        S = gl.where(col_mask, S, neg_inf)

    # Online softmax in the base-2 exponent domain: fold qk_scale*log2(e) into S
    # once, right out of the MFMA, and carry m_i already scaled. The alternative
    # (raw running max, scale at use) needs qk_scale on both the S tile and the two
    # m vectors every iteration; here the per-element work in the loop is one
    # subtract feeding exp2, and the whole epilogue -- sink combine and partial
    # store, both of which want base-2 m -- stops converting.
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
    # int16 column offsets for the asm dequant's 2-byte gather (half the columns).
    offs_full16 = gl.arange(
        0, cfg.KV_DIM // 2, layout=gl.SliceLayout(0, cfg.gather16_l)
    )
    offs_rope = gl.arange(0, cfg.ROPE_DIM, layout=gl.SliceLayout(0, cfg.gather_rope_l))
    k_rng_slot = gl.arange(0, cfg.BLOCK_K, layout=cfg.slot_l)
    k_rng_rope = gl.arange(0, cfg.BLOCK_K, layout=gl.SliceLayout(1, cfg.gather_rope_l))

    # Peel the (possibly partial) last tile: [lo, hi_full) are full BLOCK_K tiles
    # whose slots are all valid -> mask-free. Only the peeled tail carries masking.
    hi_full = lo + ((hi - lo) // cfg.BLOCK_K) * cfg.BLOCK_K

    if seg.fmt.IS_FP8:
        # UNI_TILE: the partial tile is just the last iteration -- no peeled masked
        # copy of the body (RESULTS.md 19).
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
    q_ptr,
    main_cache_ptr,
    main_cache_bf16_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_cache_bf16_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    # f32 side-channel per segment: the scalar k_scale ("tensor") or the f32 view
    # of the cache ("dsmla"). None -> specialized out of the binary entirely, so
    # DSv4 launches keep their exact kernarg layout.
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
    # NOPE_CHUNK: extent of one dequant piece along CHUNK_AXIS (0 = rows, 1 =
    # columns); >= the tile's extent on that axis means one shot.
    NOPE_CHUNK: gl.constexpr,
    CHUNK_AXIS: gl.constexpr,
    PART_STORE_CACHE: gl.constexpr,
    UNI_TILE: gl.constexpr,
    GRID_ORDER: gl.constexpr,
    # How many of the NUM_SPLITS programs share the main (SWA) segment. The two
    # segments have very different shapes -- main is a contiguous window whose
    # length is fixed by the sliding window, extra is the top-k list -- so one
    # split count has to be wrong for one of them. Splitting main past
    # main_len/BLOCK_K turns every full tile into a half-empty masked one, while
    # extra still wants the CTAs. MAIN_SPLITS <= NUM_SPLITS lets main stop at whole
    # tiles and extra keep going; programs with split_id >= MAIN_SPLITS get an
    # empty main range and contribute an extra-only partial.
    MAIN_SPLITS: gl.constexpr,
    # ADAPTIVE_SPLITS: re-decide the useful split count per query at runtime.
    ADAPTIVE_SPLITS: gl.constexpr,
    ASM_DEQ: gl.constexpr,
    # Per-cache buffer/global gate. buffer_load carries a 32-bit offset, so a cache
    # whose span exceeds that must gather via 64-bit gl.load -- but the two caches
    # are sized independently (SWA window vs full compressed history), so gating
    # them together would drop the fast path on a small cache just because its
    # partner is large. At tiny top-k the main/SWA gather is ~94% of the tokens.
    MAIN_USE_BUFFER_LOAD: gl.constexpr,
    EXTRA_USE_BUFFER_LOAD: gl.constexpr,
    IDX_BUFFER_LOAD: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    FP8_FNUZ: gl.constexpr,
):
    """One program = (query t, split, head-block). Two-loop: main (SWA) then
    extra (top-k). NUM_SPLITS==1 writes the output directly; NUM_SPLITS>1 stores
    un-normalized partials for the reduce kernel. HAS_INVALID gates -1-sentinel
    handling (clamp + score mask) on the full-tile fast paths."""
    NUM_WARPS: gl.constexpr = gl.num_warps()
    # The new fp8 formats never trace the peeled masked tail (_decode_tile);
    # requiring UNI_TILE keeps it that way instead of silently miscompiling.
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
        "fp8_ds_mla is a separated-rope (GLM) format",
    )
    gl.static_assert((not ASM_DEQ) or MAIN_FMT == "dsv4" or EXTRA_FMT == "dsv4",
                     "ASM_DEQ is the dsv4 E8M0 dequant")
    # Page stride alignment hint. The row base of every cache gather is
    # ``block_idx*cs0 + pos*TOK`` with block_idx/pos gathered at runtime, so the
    # divisibility analysis has to assume 1-byte alignment and a contiguous
    # 512-byte row gather lowers to hundreds of global_load_ubyte instead of
    # global_load_dwordx4. TOK/HEAD_SIZE are literals the compiler can already
    # reason about; cs0 is the one opaque term. The driver sets CS0_ALIGN>1 only
    # after checking the strides on the host.
    if CS0_ALIGN > 1:
        main_cs0 = gl.multiple_of(main_cs0, CS0_ALIGN)
        extra_cs0 = gl.multiple_of(extra_cs0, CS0_ALIGN)
    # GRID_ORDER names the launch axes in grid-dim order (see the driver): dim 0
    # varies fastest, which is what decides XCD/L2 sharing.
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

    # ---- segment lengths, issued first ----
    # These gate the whole KV chain: indptr -> segment range -> indices gather ->
    # cache addresses -> cache gather, three dependent memory round trips deep.
    # Q is independent of all of it, so issue the indptr loads before Q rather
    # than after: the scalar readfirstlane they feed needs s_waitcnt vmcnt(0), and
    # behind Q's load plus its through-LDS layout conversion (two barriers) that
    # wait lands at the end of a long serial prologue instead of overlapping it.
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

    # ---- per-segment softmax/V scales ----
    # exp2 softmax: fold scale*log2(e) into the loop exponent; keep raw `scale`
    # for the sink/normalization (sink is a scaled-score-space logit). For the
    # per-tensor fp8 format the cache's k_scale joins the fold (max/exp2 commute
    # with a positive scale) and the V-side scale is applied to p in the loop --
    # both are exact, so m/l keep the dequantized-score-domain convention the
    # reduce kernel expects.
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

    # ---- load Q [BLOCK_M, QK_DIM] ----
    # ROPE_SEPARATE loads the two q pieces separately (576 is not a pow-2 arange):
    # nope columns [0, KV_DIM) and rope columns [KV_DIM, KV_DIM + ROPE_DIM), each
    # converted to the dot-operand layout of its own MFMA in the QK chain.
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

    # head mask in pv-slice layout (for output / partial masking)
    offs_m_pv = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout))
    h_pv = h_off + offs_m_pv
    head_mask_pv = h_pv < num_heads

    # ---- online-softmax state ----
    m_i = gl.full(
        [BLOCK_M], float("-inf"), gl.float32, layout=gl.SliceLayout(1, cfg.qk_layout)
    )
    l_i = gl.zeros([BLOCK_M], gl.float32, layout=gl.SliceLayout(1, cfg.qk_layout))
    acc = gl.zeros([BLOCK_M, HEAD_SIZE], gl.float32, layout=cfg.pv_layout)

    kv_smem = gl.allocate_shared_memory(
        gl.bfloat16, [BLOCK_K, HEAD_SIZE], cfg.kv_shared
    )
    if ROPE_SEPARATE:
        rope_smem = gl.allocate_shared_memory(
            gl.bfloat16, [BLOCK_K, ROPE_DIM], cfg.rope_shared
        )
    else:
        rope_smem = kv_smem  # never sliced/read as plane 1 in this geometry

    # ---- how many of the launched splits this query actually wants ----
    # NUM_SPLITS / MAIN_SPLITS come from the host, which only knows the batch's
    # AVERAGE segment lengths. In a ragged batch (the SWA window saturates at
    # sliding_window while top-k keeps growing with context) the per-query lengths
    # differ a lot, and a split count sized for the average over-splits the short
    # queries -- each surplus program still gathers a mostly-masked tile and writes
    # a full [BLOCK_M, HEAD_SIZE] f32 partial. Recompute the useful count from this
    # query's own lengths and let the surplus programs write a neutral partial and
    # leave. The reduce skips a split whose m is -inf, so their part_acc is never
    # read and does not have to be written.
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

    # ---- main (SWA) segment ----
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

    # m_i/l_i are in SliceLayout(1, qk_layout); acc in pv_layout. Move the row
    # reductions into pv-slice space for output/partials.
    m_pv = gl.convert_layout(m_i, gl.SliceLayout(1, cfg.pv_layout))
    l_pv = gl.convert_layout(l_i, gl.SliceLayout(1, cfg.pv_layout))

    if NUM_SPLITS == 1:
        if HAS_SINK:
            # m_pv is already the row-max in the base-2 exponent domain
            # (row_max * softmax_scale * log2e); lift the sink -- a scaled-score
            # logit -- into the same domain and combine there.
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
        # store un-normalized partials for the reduce kernel. m is already in the
        # base-2 exponent domain (row-max * softmax_scale * log2e), which is the
        # reduce/skip_reduce convention the triton kernel also uses.
        pm_base = query_idx * pm_stride0 + split_id * pm_stride_s
        # The reduce kernel reads these back immediately, so they want to stay
        # resident rather than stream to memory. ATT puts 8.5% of all stall cycles
        # on the 68 buffer_store_dwordx4 of the accumulator alone (~154 cycles
        # each), and the partials are ~31% of the kernel's HBM traffic.
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
        # Follow part_acc's own dtype: at bf16 this halves both the partial traffic
        # (~31% of the kernel's HBM bytes) and the store instruction count.
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
    PART_LOAD_CACHE: gl.constexpr,
):
    """Split-KV combine: merge the per-split partials, fold the attn sink, and write
    the final output. Partials store m in the base-2 exponent domain (row-max *
    softmax_scale * log2e), matching the triton reduce. Grid: (num_queries, heads_blocks).

    Identical for both geometries: partials are [*, HEAD_SIZE] where HEAD_SIZE is
    the V width (512 for DSv4 and GLM-5 alike) -- the rope plane never reaches the
    output. BLOCK_M is the reduce's own head tile and is deliberately decoupled
    from the attention kernel's: the combine is pure bandwidth over
    [num_queries, num_heads, HEAD_SIZE] f32, so the only thing that matters is
    having enough workgroups to cover the machine. At BLOCK_M = num_heads there is
    one workgroup per query -- 64 of them on a 256-CU part, i.e. 3/4 of the GPU
    idle and one wave to hide every partial-load latency behind.
    """
    NUM_WARPS: gl.constexpr = gl.num_warps()
    RCP_LN2: gl.constexpr = 1.4426950408889634
    query_idx = gl.program_id(0)
    pid_h = gl.program_id(1)

    # One warp covers [BLOCK_M, 64//BLOCK_M * 8] -- lay the 64 lanes out so a
    # small BLOCK_M spends them on the head dim instead of idling them on rows.
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
    # NOT made adaptive. Bounding these loops by the per-query split count instead
    # of NUM_SPLITS turns them into dynamic loops, and losing the static unroll
    # costs 5-7% on a uniform batch -- more than the ~0.9 us it saves when the
    # launch over-splits. The acc load is masked on m > -inf instead, which is
    # what actually has to be right (a split that bowed out wrote no accumulator).
    m_final = gl.full([BLOCK_M], neg_inf, gl.float32, layout=row_l)
    # pass 1: global max over splits
    for s in range(NUM_SPLITS):
        base = query_idx * pm_stride0 + s * pm_stride_s
        m_s = gl.amd.cdna4.buffer_load(
            ptr=part_m_ptr + base, offsets=h, mask=head_mask, other=neg_inf,
            cache=PART_LOAD_CACHE,
        )
        m_final = _max2(m_final, m_s)  # m_s already in base-2 exponent domain
    if HAS_SINK:
        sink = gl.amd.cdna4.buffer_load(
            ptr=attn_sink_ptr, offsets=h, mask=head_mask, other=neg_inf
        ).to(gl.float32)
        m_final = _max2(m_final, sink * RCP_LN2)  # lift sink to base-2

    # pass 2: weighted sums
    l_final = gl.zeros([BLOCK_M], gl.float32, layout=row_l)
    acc = gl.zeros([BLOCK_M, HEAD_SIZE], gl.float32, layout=BLK)
    for s in range(NUM_SPLITS):
        base = query_idx * pm_stride0 + s * pm_stride_s
        m_s = gl.amd.cdna4.buffer_load(
            ptr=part_m_ptr + base, offsets=h, mask=head_mask, other=neg_inf,
            cache=PART_LOAD_CACHE,
        )
        l_s = gl.amd.cdna4.buffer_load(
            ptr=part_l_ptr + base, offsets=h, mask=head_mask, other=0.0,
            cache=PART_LOAD_CACHE,
        )
        w = gl.exp2(m_s - m_final)
        l_final = l_final + w * l_s
        a_base = query_idx * pa_stride0 + s * pa_stride_s
        a_off = (a_base + h[:, None] * pa_stride_h + offs_d[None, :]).to(gl.int32)
        # An adaptive-split program that bowed out wrote m = -inf and no
        # accumulator, so its part_acc slot is uninitialized -- mask the load
        # rather than relying on w == 0 (0 * NaN is NaN).
        if ADAPTIVE_SPLITS:
            acc_mask = head_mask[:, None] & (m_s > neg_inf)[:, None]
        else:
            acc_mask = head_mask[:, None]
        acc_s = gl.amd.cdna4.buffer_load(
            ptr=part_acc_ptr, offsets=a_off, mask=acc_mask, other=0.0,
            cache=PART_LOAD_CACHE,
        )
        acc = acc + w[:, None] * acc_s.to(gl.float32)

    if HAS_SINK:
        sink = gl.amd.cdna4.buffer_load(
            ptr=attn_sink_ptr, offsets=h, mask=head_mask, other=neg_inf
        ).to(gl.float32)
        l_final = l_final + gl.exp2(sink * RCP_LN2 - m_final)

    # One reciprocal per row, then a broadcast multiply: an f32 divide lowers to a
    # ~10-instruction sequence, so dividing the whole [BLOCK_M, HEAD_SIZE] tile costs
    # that per element.
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
